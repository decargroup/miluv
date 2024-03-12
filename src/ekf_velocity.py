import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import tqdm
import datetime
from typing import List
import pandas as pd

from navlie.lib.models import (
    BodyFrameVelocity,
)

from pyuwbcalib.utils import (
    set_plotting_env,
)

from utils.measurement import (
    Measurement,
    RangeData,
)

from utils.misc import (
    GaussianResult,
    GaussianResultList,
    plot_error,
)

from utils.states import (
    SE3State,
    CompositeState,
    StateWithCovariance,
)

from utils.inputs import (
    VectorInput,
    CompositeInput,
)

from utils.models import (
    CompositeProcessModel,
    AltitudeById,
    MagnetometerById,
)

from miluv.data import DataLoader
from src.filters import ExtendedKalmanFilter
import time
from tqdm import tqdm
from scipy.spatial.transform import Rotation

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


""" Get data """
miluv = DataLoader("1c", barometer=False)
robots = list(miluv.data.keys())
input_sensor = "vio"
start_time, end_time = miluv.get_timerange(
                            sensors = input_sensor,)

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
velocity = [miluv.data[robot]["mocap"].body_velocity(
                            query_stamps) for robot in robots]

init_attitude = [p[0][:3,:3] for p in pose]

# Get range data
range_data = RangeData(miluv, miluv)
# range_data = range_data.filter_by_bias( max_bias=0.01)
range_data = range_data.by_timerange(start_time, end_time,
                                     sensors=["uwb_range"])
meas_data = range_data.to_measurements(
    reference_id = 'world')

R = [0.1, 3*0.1, 10*0.1]
# bias = [height[n]['range'].mean() - position[n][:,2].mean() for n in range(len(robots))]
bias = [-0.0924, -0.0088, -0.1207]
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
    x = [SE3State(  value = pose[n][i], 
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
                0.1, 0.1, 0.1])
Q = np.diag([ 5*0.1**2, 5*0.1**2, 10 * 0.15**2, 
               0.1,0.1, 0.001])

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
                           (pose[n][i][:3,:3].T @ init_attitude[n] @ (vio.data[robot]['vio'].iloc[i][
                           ['velocity.x', 'velocity.y', 'velocity.z']].values).reshape(-1,1)).ravel(),
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
    # legend Robot 1, 2, 3
    legend = ["Robot 1", "Robot 2", "Robot 3"]
    
    # Have one legend for each figure
    for i, (fig, axs) in enumerate(figs):
        for ax in axs:
            for a in ax:
                if a == axs[0,-1]:
                    a.legend([legend[i]], handlelength=0)
    

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
        vio_data = vio.data[robot]['vio'][['velocity.x', 'velocity.y', 'velocity.z']].values
        for v in vio_data:
            vins_velocity.append((init_attitude[n] @ v.reshape(-1,1)).ravel())
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
