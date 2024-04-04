import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
from pyuwbcalib.utils import (
    set_plotting_env,
)
from utils.meas import (
    Measurement,
    RangeData,
)

from utils.inputs import (
    VectorInput,
    CompositeInput,
)
from utils.states import (
    CompositeState,
    MatrixLieGroupState,
    StateWithCovariance,
)
from utils.models import (
    BodyFrameVelocity,
    CompositeProcessModel,
    AltitudeById,)
from utils.misc import (
    GaussianResult,
    GaussianResultList,
    plot_error,
)
from miluv.data import DataLoader
from src.filters import ExtendedKalmanFilter
import time
from tqdm import tqdm
import os
import sys
import pickle
import argparse
import yaml

def read_vio_yaml(exp_name:str, exp_dir:str, robot:str) -> pd.DataFrame:
    """Read a yaml file for a given robot and topic."""
    path = "vio/" + exp_name + "/" + robot + "_alignment_pose.yaml"
    path = os.path.join(exp_dir, path)
    return yaml.safe_load(open(path, 'r'))

# Set the plotting environment
set_plotting_env()
plt.rcParams.update({'font.size': 10})  

# plots
ekf = True
trajectory_plot = False
error_plot = True
vio_plot = False
imu_plot = False
save_fig = False
body_frame_vel = False
height_plot = False
save_results = False


""" Get data """
# parser = argparse.ArgumentParser()
# parser.add_argument('--exp', required=True)
# args = parser.parse_args()
# exp = args.exp
exp = "1c"
folder = "/media/syedshabbir/Seagate B/data"
miluv = DataLoader(exp, exp_dir = folder, barometer = False)
robots = list(miluv.data.keys())
input_sensor = "vio"
start_time, end_time = miluv.get_timerange(
                            sensors = input_sensor,)
end_time = end_time - 50

input_freq = 30
query_stamps = np.arange(start_time, end_time, 1/input_freq)

# Get input data
vio = miluv.by_timestamps(query_stamps, sensors=input_sensor)
imus = miluv.by_timestamps(query_stamps, sensors=["imu_px4"])

mag = miluv.by_timerange(start_time, end_time, sensors=["mag"])
mag = [mag.data[robot]["mag"] for robot in robots]

accel = miluv.by_timerange(start_time, end_time, sensors=["imu_px4"])

height = miluv.by_timerange(start_time, end_time, sensors=["height"])
height = [height.data[robot]["height"] for robot in robots]
min_height = [h['range'].min() for h in height]

# Get pose data
pose = [miluv.data[robot]["mocap"].pose_matrix(
                            query_stamps) for robot in robots]
position = [miluv.data[robot]["mocap"].position(
                            query_stamps) for robot in robots]
angular_velocity = [miluv.data[robot]["mocap"].angular_velocity(
                            query_stamps) for robot in robots]
velocity = [miluv.data[robot]["mocap"].velocity(
                            query_stamps) for robot in robots]
    

# Get range data
range_data = RangeData(miluv)
range_data = range_data.filter_by_bias( max_bias=0.3)
range_data = range_data.by_timerange(start_time, end_time,
                                     sensors=["uwb_range"])
meas_data = range_data.to_measurements(
    reference_id = 'world')

R = [3*0.1, 3*0.1, 10*0.1]
bias = [height[n]['range'].mean() - position[n][:,2].mean() for n in range(len(robots))]
# bias = [-0.0924, -0.0088, -0.1207]
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
for i in range(len(query_stamps)):
    x = [MatrixLieGroupState(  value = pose[n][i], 
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
P0 = np.diag([0.02**2, 0.02**2, 0.02**2, 
                2*0.1, 2*0.1, 2*0.1])
Q = 2*np.diag([ 5*0.1**2, 5*0.1**2, 10 * 0.15**2, 
               2*0.1,2*0.1, 2*0.01])

# Process Model
process_model = CompositeProcessModel(
    [BodyFrameVelocity(Q) for r in robots])

# Composite covariance
n_states = len(x0.value)
P0 = np.kron(np.eye(n_states), P0)
Q = np.kron(np.eye(n_states), Q)

""" Create input data """
input_data = []
for i in range(len(query_stamps)):
    u = [VectorInput(
        value = np.hstack((
        # angular_velocity[n][i],
        imus.data[robot]["imu_px4"].iloc[i][
        ['angular_velocity.x', 
        'angular_velocity.y', 
        'angular_velocity.z']].values, 

        (pose[n][i][:3,:3].T @ 
         (vio.data[robot]['vio'].iloc[i][
        ['twist.linear.x', 'twist.linear.y', 'twist.linear.z']].values).
        reshape(-1,1)).ravel(),
    #    velocity[n][:,i],
                            )),
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
        pos = np.array([r.get_state_by_id(robot).value[0:3,-1] 
                        for r in results.state])
        true_pos = np.array([r.get_state_by_id(robot).value[0:3,-1] 
                             for r in results.state_true])
        error = pos - true_pos
        pos_rmse[robot] = np.sqrt(np.mean([e.T @ e / dof for e in error]))

        error = error.ravel()
        # pos_rmse[robot] = np.sqrt(error.T @ error / len(error))
    for robot in robots:
        print(f"Position RMSE for Experiment: {exp} and robot {robot}: {pos_rmse[robot]} m")

if ekf and save_results:
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    folder = os.path.join(script_dir, f'results')
    os.umask(0)
    os.makedirs(folder, exist_ok=True)
    filename = f'results_vel_{exp}.pkl'
    file_path = os.path.join(folder, filename)
    with open(file_path, 'wb') as file:
        pickle.dump((results, pos_rmse), file)

if ekf and error_plot:
    separate_figs = True
    figs = plot_error(results, 
                      separate_figs=separate_figs)
    titles = ["Att. Error", 
              "Pos. Error",]
    
    y_labels = ["roll (rad)", "pitch (rad)", "yaw (rad)",
                "x (m)", "y (m)", "z (m)"]
    for fig, axs in figs:
        index = 0
        for j in range(2):
            for i in range(3):
                axs[i, j].set_ylabel(y_labels[index])
                index += 1
        for  ax in axs:
            for i, a in enumerate(ax):
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
    

if body_frame_vel:
    for n, robot in enumerate(robots):
        fig, ax = plt.subplots(3, 2)
        ax[0,0].plot(query_stamps, angular_velocity[n][:,0], label="roll")
        ax[1,0].plot(query_stamps, angular_velocity[n][:,1], label="pitch")
        ax[2,0].plot(query_stamps, angular_velocity[n][:,2], label="yaw")
        ax[0,1].plot(query_stamps, velocity[n][0,:], label="x")
        ax[1,1].plot(query_stamps, velocity[n][1,:], label="y")
        ax[2,1].plot(query_stamps, velocity[n][2,:], label="z")

if imu_plot:
    for n, robot in enumerate(robots):
        fig, ax = plt.subplots(3)
        ax[0].plot(query_stamps, angular_velocity[n][:,0], label="roll")
        ax[0].plot(query_stamps, imus.data[robot]['imu_px4']['angular_velocity.x'].values, label="imu")
        ax[1].plot(query_stamps, angular_velocity[n][:,1], label="pitch")
        ax[1].plot(query_stamps, imus.data[robot]['imu_px4']['angular_velocity.y'].values, label="imu")
        ax[2].plot(query_stamps, angular_velocity[n][:,2], label="yaw")
        ax[2].plot(query_stamps, imus.data[robot]['imu_px4']['angular_velocity.z'].values, label="imu")

if vio_plot:
    for n, robot in enumerate(robots):

        fig, ax = plt.subplots(3)
        vins_velocity = []
        vio_data = vio.data[robot]['vio'][['twist.linear.x', 'twist.linear.y', 'twist.linear.z']].values
        # for v in vio_data:
        #     vins_velocity.append((offset[robot]['C_vm'].T @ v.reshape(-1,1)).ravel())
        for v in vio_data:
            vins_velocity.append(v.ravel())
        vins_velocity = np.array(vins_velocity)
        ax[0].plot(query_stamps, velocity[n][:,0], label="true")
        # ax[0].plot(query_stamps, velocity[n][0,:], label="true")
        # ax[0].plot(query_stamps, vio.data[robot]['vio']['velocity.x'].values, label="vio")
        ax[0].plot(query_stamps, vins_velocity[:,0], label="vio")
        ax[0].set_ylabel("x (m/s)")
        ax[1].plot(query_stamps, velocity[n][:,1])
        # ax[1].plot(query_stamps, velocity[n][1,:], label="true")
        # ax[1].plot(query_stamps, vio.data[robot]['vio']['velocity.y'].values)
        ax[1].plot(query_stamps, vins_velocity[:,1])
        ax[1].set_ylabel("y (m/s)")
        ax[2].plot(query_stamps, velocity[n][:,2])
        # ax[2].plot(query_stamps, velocity[n][2,:], label="true")
        # ax[2].plot(query_stamps, vio.data[robot]['vio']['velocity.z'].values)
        ax[2].plot(query_stamps, vins_velocity[:,2])
        ax[2].set_ylabel("z (m/s)")
        for a in ax:
            a.legend(loc = "upper right")
            a.set_xlabel("Time (s)")
if height_plot:
    for n, robot in enumerate(robots):
        fig, ax = plt.subplots(1)
        ax.plot(query_stamps, position[n][:,2], label="true")
        ax.plot(height[n]['timestamp'], height[n]['range'], label="range")
        ax.set_ylabel("z (m)")
        ax.legend(loc = "upper right")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(ymin = 0)


plt.show()
