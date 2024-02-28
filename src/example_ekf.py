import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import tqdm
import datetime
from typing import List
import pandas as pd

from pyuwbcalib.utils import (
    set_plotting_env, 
    read_anchor_positions
)

from utils.measurement import (
    Measurement,
    RangeData,
)

from utils.misc import (
    GaussianResult,
    GaussianResultList,
    merge_range,
    plot_error,
    plot_uav3d,
    plot_range3d,
    plot_trajectory3d,
)

from utils.states import (
    SE23State,
    CompositeState,
    StateWithCovariance,
)

from utils.inputs import (
    IMU,
    IMUState,
    CompositeInput,
)

from utils.models import (
    BodyFrameIMU,
    CompositeProcessModel,
)

from miluv.data import DataLoader
from src.filters import ExtendedKalmanFilter
import time
from tqdm import tqdm

# Set the plotting environment
set_plotting_env()
plt.rcParams.update({'font.size': 10})  

# plots
trajectory_plot = False
error_plot = True
save_fig = False


""" Get data """
miluv = DataLoader("1c", baro=False)
robots = list(miluv.data.keys())
input_sensor = ["imu_px4"]
start_time, end_time = miluv.get_timerange(
                            sensors = input_sensor,
                            seconds=False)

query_stamps = np.arange(start_time, end_time, 0.01*1e9)
imus = miluv.by_timestamps(query_stamps, sensors=input_sensor)

range_data = RangeData(miluv)
range_data = range_data.filter_by_bias( max_bias=0.3)
range_data = range_data.by_timerange(start_time, 
                                     end_time, 
                                     sensors=["uwb_range"])
range_data = range_data.merge_range(sensors=["uwb_range"],
                                    seconds=True)


pose = [miluv.data[robot]["mocap"].extended_pose_matrix(
                            query_stamps) for robot in robots]
query_stamps = np.array(query_stamps)/1e9


""" Create ground truth data """
bias_gyro = np.array([0, 0, 0])
bias_accel = np.array([0, 0, 0])
ground_truth = []
for i in range(len(query_stamps)):
    x = [IMUState(  nav_state = pose[n][i], 
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
              1, 1, 1,
              1, 1, 1,
              0.0001, 0.0001,0.0001,
              0.0001, 0.0001,0.0001])

Q = np.diag([0.0025**2, 0.0025**2,0.0025**2, 
             0.025**2, 0.025**2,0.025**2,
             0.0001**2, 0.0001**2,0.0001**2,
             0.0001**2, 0.0001**2,0.0001**2,])

# Process Model
process_model = CompositeProcessModel(
    [BodyFrameIMU(Q) for i in range(len(robots))])

# Composite covariance
n_states = len(x0.value)
P0 = np.kron(np.eye(n_states), P0)
Q = np.kron(np.eye(n_states), Q)

""" Create input data """
input_data = []
for i in range(len(query_stamps)):
    u = [IMU(
            gyro = imus[robot][input_sensor].iloc[i][
                ['angular_velocity.x', 
                 'angular_velocity.y', 
                 'angular_velocity.z']
                 ].values, 

            accel= imus[robot][input_sensor].iloc[i][
                ['linear_acceleration.x', 
                 'linear_acceleration.y', 
                 'linear_acceleration.z']].values,

            stamp = query_stamps[i], 
            state_id = robot)
            for n, robot in enumerate(robots)
        ]
    
    input_data.append(CompositeInput(u))


""" Run the filter """
x = StateWithCovariance(x0, P0)

# Try an EKF or an IterEKF
ekf = ExtendedKalmanFilter(process_model, 
                        reject_outliers=True)


results_list = []
init_stamp = time.time()
for k in tqdm(range(len(input_data) - 1)):

    u = input_data[k]

    dt = input_data[k + 1].stamp - x.stamp
    x = ekf.predict(x, u, dt)

    results_list.append(GaussianResult(x, ground_truth[k]))

print("Average filter computation frequency (Hz):")
print(1 / ((time.time() - init_stamp) / len(input_data)))

results = GaussianResultList(results_list)


if error_plot:
    fig, axs = plot_error(results)
    titles = ["Attitude Error", 
            "Velocity Error", 
            "Position Error",]
    y_labels = ["x (-)", "y (-)", "z (-)"]
    titles.append("Gyro Bias Error")
    titles.append("Accel Bias Error")

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
    
    for i, a in enumerate(axs[:,0]):
        a.set_ylabel(y_labels[i])
    if save_fig:
        plt.savefig(f'./figures/' + init_stamp + '.png')
    plt.show()
