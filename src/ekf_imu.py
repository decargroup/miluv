import numpy as np
import matplotlib.pyplot as plt
import tqdm
from pyuwbcalib.utils import (
    set_plotting_env, 
)
from utils.misc import (
    GaussianResult,
    GaussianResultList,
    plot_error,
)
from utils.models import (
    CompositeProcessModel,
    AltitudeById,
)
from utils.imu import ( 
    IMU,
    IMUState,
    IMUKinematics,
)
from utils.inputs import (
    CompositeInput,
)
from utils.states import (
    CompositeState, 
    StateWithCovariance,
)
from utils.meas import (
    Measurement,
    RangeData,
)
from miluv.data import DataLoader
from src.filters import ExtendedKalmanFilter
import time
from tqdm import tqdm
import os
import sys
import pickle
import argparse
import csv

""" 
All Matrix Lie groups are perturbed in the right direction.
"""

# Set the plotting environment
set_plotting_env()
plt.rcParams.update({'font.size': 10})  


# plots
ekf = True
imu_plot = False
trajectory_plot = False
error_plot = True
save_fig = False
save_results = False


# """ Get data """
# parser = argparse.ArgumentParser()
# parser.add_argument('--exp', required=True)
# args = parser.parse_args()
# exp = args.exp
exp = "1c"
folder = "/media/syedshabbir/Seagate B/data"
miluv = DataLoader(exp, exp_dir = folder, barometer = False)
robots = list(miluv.data.keys())
input_sensor = "imu_px4"
input_freq = 190

# Get the time range
start_time, end_time = miluv.get_timerange(
                            sensors = input_sensor)
end_time = end_time - 5
query_stamps = np.arange(start_time, end_time, 1/input_freq)

# Get input data
imus = miluv.by_timestamps(query_stamps, sensors=input_sensor)

mag = miluv.by_timerange(start_time, end_time, sensors=["mag"])
mag = [mag.data[robot]["mag"] for robot in robots]

height = miluv.by_timerange(start_time, end_time, sensors=["height"])
height = [height.data[robot]["height"] for robot in robots]
min_height = [h['range'].min() for h in height]

# Get pose data
pose = [miluv.data[robot]["mocap"].extended_pose_matrix(
                            query_stamps) for robot in robots]
position = [miluv.data[robot]["mocap"].position(
                            query_stamps) for robot in robots]
gyro = [miluv.data[robot]["mocap"].angular_velocity(
    query_stamps) for robot in robots]
accel = [miluv.data[robot]["mocap"].accelerometer(
    query_stamps) for robot in robots]


# Get range data
range_data = RangeData(miluv)
range_data = range_data.filter_by_bias( max_bias=0.3)
range_data = range_data.by_timerange(start_time, 
                                     end_time, 
                                     sensors=["uwb_range"])
meas_data = range_data.to_measurements(
    reference_id = 'world')

R = [3*0.1, 3*0.1, 3*0.1]
# bias = [-0.0924, -0.0088, -0.1207]
bias = [height[n]['range'].mean() - position[n][:,2].mean() for n in range(len(robots))]
for n, robot in enumerate(robots):
    for i in range(len(height[n])):
        y = Measurement(value = height[n].iloc[i]['range'],
                            stamp = height[n].iloc[i]['timestamp'],
                            model = AltitudeById(R = R[n], 
                            state_id = robot,
                            minimum=min_height[n],
                            bias = bias[n]))
        meas_data.append(y)

# sort the measurements
meas_data = sorted(meas_data, key=lambda x: x.stamp)

""" Create ground truth data """
ground_truth = []
bias_gyro = [0.0, 0.0, 0.0]
bias_accel = [0.0, 0.0, 0.0]
for i in range(len(query_stamps)):
    x = [IMUState(  SE23_state = pose[n][i], 
                    bias_gyro = bias_gyro,
                    bias_accel = bias_accel,
                    state_id = robot,
                    stamp = query_stamps[i],
                    direction='right') 
        for n, robot in enumerate(robots)
        ]
    ground_truth.append(CompositeState(x,
                        stamp = query_stamps[i]))
    
""" State and input covariance, process model, and initial state"""
# Initial state
x0 = ground_truth[0].copy()

# State and input covariance
P0 = np.diag([0.1, 0.1, 0.1, 
             0.1, 0.1, 0.1,
             0.01, 0.01, 0.1,
             0.0001, 0.0001,0.0001,
             0.0001, 0.0001,0.0001])

Q  = {robot: [] for robot in robots}
for n, robot in enumerate(robots):
    Q_c = np.eye(12)
    w_gyro = [(np.pi/180*w)**2 for w in 
              imus.setup["imu_px4_calib"][robot]["bias_gyro"]]
    w_accel = [w**2 for w in 
               imus.setup["imu_px4_calib"][robot]["bias_accel"]]
    w_gyro_bias = [(np.pi/180*w)**2 for w in 
                   imus.setup["imu_px4_calib"][robot]["bias_gyro_walk"]]
    w_accel_bias = [w**2 for w in 
                    imus.setup["imu_px4_calib"][robot]["bias_accel_walk"]]
    Q_c[0:3,0:3] = 2e4*np.diag(w_gyro)
    Q_c[3:6,3:6] = 2e4*np.diag(w_accel)
    Q_c[6:9,6:9] = np.diag(w_gyro_bias)
    Q_c[9:12,9:12] = np.diag(w_accel_bias)
    dt = 1/input_freq
    Q[robot] = Q_c / dt

process_model = CompositeProcessModel(
    [IMUKinematics(Q[robot]) for robot in robots])

# Composite covariance
n_states = len(x0.value)
P0 = np.kron(np.eye(n_states), P0)

""" Create input data """
input_data = []
for i in range(len(query_stamps)):
    u = [IMU(
        gyro = imus.data[robot][input_sensor].iloc[i][
            ['angular_velocity.x', 
             'angular_velocity.y', 
             'angular_velocity.z']].values, 
        # gyro = gyro[n][i],
        accel= imus.data[robot][input_sensor].iloc[i][
            ['linear_acceleration.x', 
             'linear_acceleration.y', 
             'linear_acceleration.z']].values,
        # accel = accel[n][i],
        stamp = query_stamps[i], 
        state_id = robot)
        for n, robot in enumerate(robots)
        ]
    
    input_data.append(CompositeInput(u))

if ekf:
    """ Run the filter """
    x = StateWithCovariance(x0, P0)

    # Try an EKF or an IterEKF
    ekf = ExtendedKalmanFilter(process_model, 
                            reject_outliers=True)

    meas_idx = 0
    y = meas_data[meas_idx]
    init_stamp = time.time()
    results_list = []
    for k in tqdm(range(len(input_data) - 1)):

        u = input_data[k]
                
        # Fuse any measurements that have occurred.
        while y.stamp < input_data[k + 1].stamp and meas_idx < len(meas_data):

            x = ekf.correct(x, y, u)

            # Load the next measurement
            meas_idx += 1
            if meas_idx < len(meas_data):
                y = meas_data[meas_idx]

        dt = input_data[k + 1].stamp - x.stamp
        x = ekf.predict(x, u, dt)

        results_list.append(GaussianResult(x, ground_truth[k]))

    print("Average filter computation frequency (Hz):")
    print(1 / ((time.time() - init_stamp) / len(input_data)))

    results = GaussianResultList(results_list)

""" Print position RMSE """
if ekf:
    pos_rmse = {robot: {} for robot in robots}
    dof = 3
    for i, robot in enumerate(robots):
        pos = np.array([r.get_state_by_id(robot).value[0].value[0:3, -1]
                        for r in results.state])
        true_pos = np.array([r.get_state_by_id(robot).value[0].value[0:3, -1]
                             for r in results.state_true])
        error = pos - true_pos
        pos_rmse[robot] = np.sqrt(np.mean([e.T @ e / dof for e in error]))
    for robot in robots:
        print(f"Position RMSE for Experiment: {exp} and robot {robot}: {pos_rmse[robot]} m")
        filename = 'results_imu.csv'
        if not os.path.isfile(filename):
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f"{exp}",f"{robot}",f"{pos_rmse[robot]}"])

if ekf and save_results:
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    folder = os.path.join(script_dir, f'results')
    os.umask(0)
    os.makedirs(folder, exist_ok=True)
    filename = f'results_imu_{exp}.pkl'
    file_path = os.path.join(folder, filename)
    with open(file_path, 'wb') as file:
        pickle.dump((results, pos_rmse), file)

if ekf and error_plot:
    separate_figs = True
    figs = plot_error(results, 
                      separate_figs=separate_figs)
    titles = ["Att. Error", 
              "Vel. Error", 
              "Pos. Error",
              "Gyro Bias Error",
              "Accel Bias Error"]
    
    for fig, axs in figs:
        for ax in axs:
            for a in ax:
                if a in axs[-1,:]:
                    a.set_xlabel("Time (s)")
        j = 0
        for a in axs[0,:]:
            a.set_title(titles[j])
            j += 1
            if j == len(titles):
                j = 0
    
    # Have one legend for each figure
    for i, (fig, axs) in enumerate(figs):
        for ax in axs:
            for a in ax:
                if a == axs[0,-1]:
                    a.legend([robots[i]], handlelength=0)
        if save_fig:
            plt.savefig(f'./figures/error_' + robots[i] + '_' + init_stamp + '.png')

plt.show()
