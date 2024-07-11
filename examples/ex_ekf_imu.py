import numpy as np
import matplotlib.pyplot as plt
import tqdm
from pyuwbcalib.utils import (
    set_plotting_env, 
)
from utils.misc import (
    ResultList,
    plot_error,
)
from utils.process_models import ( 
    Input,
    IMUKinematics,
    CompositeProcessModel,
)
from utils.states import (
    IMUState,
    CompositeState, 
    StateWithCovariance,
)
from utils.misc import (
    RangeData,
    height_measurements,
)
from miluv.data import DataLoader
from src.filters import ExtendedKalmanFilter
import time
from tqdm import tqdm
import os
import sys
import pickle
import argparse


""" 
All Matrix Lie groups are perturbed in the right direction.
All jacobians for the EKF are computed using finite difference.
"""

# Set the plotting environment
set_plotting_env()
plt.rcParams.update({'font.size': 10})  


# plots
ekf = True
error_plot = True
save_fig = False
save_results = True


""" Load data """
parser = argparse.ArgumentParser()
parser.add_argument('--exp', required=True)
args = parser.parse_args()
exp = args.exp
folder = "/media/syedshabbir/Seagate B/data" # Change this to the path of your data
miluv = DataLoader(exp, exp_dir = folder, barometer = False)

""" Preliminaries """
robots = list(miluv.data.keys())
input_sensor = "imu_px4"
input_freq = 190
start_time, end_time = miluv.get_timerange(sensors = input_sensor)
end_time = end_time - 5
query_stamps = np.arange(start_time, end_time, 1/input_freq)

""" Get Data """
imus = miluv.by_timestamps(query_stamps, sensors=input_sensor)
pose = [miluv.data[robot]['mocap'].SE23_state(query_stamps) for robot in robots]
range_data = miluv.by_timerange(start_time, end_time, sensors=['uwb_range'])

""" Create measurements """
range_data = RangeData(range_data)
range_data = range_data.filter_by_bias( max_bias=0.3)
meas_data = range_data.to_measurements(reference_id = 'world')
R = [3*0.1, 3*0.1, 3*0.1]
bias = [-0.0924, -0.0088, -0.1207]
height = height_measurements(miluv, start_time, end_time,  R, bias)
meas_data = sorted(meas_data, key=lambda x: x.stamp)

""" Create ground truth data """
ground_truth = []
for i in range(len(query_stamps)):
    x = [IMUState(  SE23_state = pose[n][i], 
                    bias_gyro = [0,0,0],
                    bias_accel =[0,0,0],
                    state_id = robot,
                    stamp = query_stamps[i],) 
        for n, robot in enumerate(robots) ]
    ground_truth.append(CompositeState(x, stamp = query_stamps[i]))
    
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

process_model = CompositeProcessModel([IMUKinematics(Q[robot]) for robot in robots])

# Composite covariance
n_states = len(x0.value)
P0 = np.kron(np.eye(n_states), P0)

""" Create input data """

input_data = []
for i in range(len(query_stamps)):
    u = [Input(
        gyro = imus.data[robot][input_sensor].iloc[i][
            ['angular_velocity.x', 
             'angular_velocity.y', 
             'angular_velocity.z']].values, 
        accel= imus.data[robot][input_sensor].iloc[i][
            ['linear_acceleration.x', 
             'linear_acceleration.y', 
             'linear_acceleration.z']].values,
        stamp = query_stamps[i], state_id = robot) 
        for robot in robots]
    input_data.append(u)

""" Implement your filter here """

if ekf:
    """ Run the filter """
    x = StateWithCovariance(x0, P0)
    ekf = ExtendedKalmanFilter(process_model, reject_outliers=True)

    meas_idx = 0
    y = meas_data[meas_idx]
    init_stamp = time.time()
    results_list = []
    for k in tqdm(range(len(input_data) - 1)):
        u = input_data[k]
                
        # Fuse any measurements that have occurred.
        while y.stamp < input_data[k + 1][0].stamp and meas_idx < len(meas_data):
            x = ekf.correct(x, y, u)
            # Load the next measurement
            meas_idx += 1
            if meas_idx < len(meas_data):
                y = meas_data[meas_idx]
        dt = input_data[k + 1][0].stamp - x.stamp
        x = ekf.predict(x, u, dt)

        results_list.append([x, ground_truth[k]])
    results = ResultList(results_list)

script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
if ekf:
    """ Print position RMSE """
    pos_rmse = {robot: {} for robot in robots}
    for i, robot in enumerate(robots):
        pos = np.array([r.get_state_by_id(robot).value[0].value[0:3, -1]
                        for r in results.state])
        true_pos = np.array([r.get_state_by_id(robot).value[0].value[0:3, -1]
                             for r in results.state_true])
        error = (pos - true_pos).ravel()
        pos_rmse[robot] = np.sqrt(error.T @ error / len(error))

    for robot in robots:
        print(f"Position RMSE for Experiment: {exp} and robot {robot}: {pos_rmse[robot]} m")

if ekf and error_plot:
    """ Plot error """
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
        figs_folder = os.path.join(script_dir, 'figures')
        os.umask(0)
        os.makedirs(figs_folder, exist_ok=True)      
        for i, (fig, axs) in enumerate(figs):
            plt.savefig(os.path.join(figs_folder, f'error_imu_{exp}_{robots[i]}.png'))

plt.show()
