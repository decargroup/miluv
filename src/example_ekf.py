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
    IMUKinematics,
    MagnetometerById,
    CompositeProcessModel,
    AltitudeById,
)

from miluv.data import DataLoader
from src.filters import ExtendedKalmanFilter
import time
from tqdm import tqdm

# Set the plotting environment
set_plotting_env()
plt.rcParams.update({'font.size': 10})  

# plots
ekf = True
imu_plot = False
trajectory_plot = False
error_plot = True
save_fig = False


""" Get data """
miluv = DataLoader("1c", baro=False)
robots = list(miluv.data.keys())
input_sensor = "imu_px4"
input_freq = 190

# Get the time range
start_time, end_time = miluv.get_timerange(
                            sensors = input_sensor)
query_stamps = np.arange(start_time, end_time, 1/input_freq)


# Get input data
imus = miluv.by_timestamps(query_stamps, sensors=input_sensor)

mag = miluv.by_timerange(start_time, end_time, sensors=["mag"])
mag = [mag.data[robot]["mag"] for robot in robots]

height = miluv.by_timerange(start_time, end_time, sensors=["height"])
height = [height.data[robot]["height"] for robot in robots]

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
# range_data = range_data.filter_by_bias( max_bias=0.03)
range_data = range_data.by_timerange(start_time, 
                                     end_time, 
                                     sensors=["uwb_range"])
meas_data = range_data.to_measurements(
    reference_id = 'world')

# Add the altitude measurements
R = 0.1**2
for i in range(len(query_stamps)):
    y = [Measurement(  value = position[n][i][-1],
                       stamp = query_stamps[i],
                       model = AltitudeById(R = R, 
                       nb_state_id = robot))
        for n, robot in enumerate(robots)
        ]
    meas_data.extend(y)


# Add the magnetic field measurements
R = 0.01**2
for n, robot in enumerate(robots):
    for i in range(len(mag[n])):
        y = Measurement(value = mag[n].iloc[i][
                                ['magnetic_field.x',
                                 'magnetic_field.y',
                                 'magnetic_field.z']].values,
                            stamp = mag[n].iloc[i]['timestamp'],
                            model = MagnetometerById(R = R, 
                            nb_state_id = robot))
        meas_data.append(y)

# sort the measurements
meas_data = sorted(meas_data, key=lambda x: x.stamp)


""" Create ground truth data """
ground_truth = []
bias_gyro = np.array([0.0, 0.0, 0.01])
bias_accel = np.array([0.01, 0.01, 0.0])
for i in range(len(query_stamps)):
    x = [IMUState(  nav_state = pose[n][i], 
                    bias_gyro = imus.setup["imu_px4_calib"
                                           ][robot]["bias_gyro"], 
                    bias_accel = imus.setup["imu_px4_calib"
                                            ][robot]["bias_accel"],
                    # bias_gyro = bias_gyro,
                    # bias_accel = bias_accel,
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

Q =  np.diag([0.0025**2, 0.0025**2,0.0025**2, 
             0.025**2, 0.025**2,0.025**2,
             0.0001**2, 0.0001**2,0.0001**2,
             0.0001**2, 0.0001**2,0.0001**2,])

# # Process Model
# process_model = CompositeProcessModel(
#     [BodyFrameIMU(Q) for i in range(len(robots))])
process_model = CompositeProcessModel(
    [IMUKinematics(Q) for i in range(len(robots))])

# Composite covariance
n_states = len(x0.value)
P0 = np.kron(np.eye(n_states), P0)
Q = np.kron(np.eye(n_states), Q)

""" Create input data """
input_data = []
for i in range(len(query_stamps)):
    u = [IMU(
        # gyro = imus.data[robot][input_sensor].iloc[i][
        #     ['angular_velocity.x', 
        #      'angular_velocity.y', 
        #      'angular_velocity.z']].values, 
        gyro = gyro[n][i],
        # accel= imus.data[robot][input_sensor].iloc[i][
        #     ['linear_acceleration.x', 
        #      'linear_acceleration.y', 
        #      'linear_acceleration.z']].values,
        accel = accel[n][i],
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

    legend = list(miluv.data.keys())
    
    # Have one legend for each figure
    for i, (fig, axs) in enumerate(figs):
        for ax in axs:
            for a in ax:
                if a == axs[0,-1]:
                    a.legend([legend[i]], handlelength=0)
        if save_fig:
            plt.savefig(f'./figures/error_' + legend[i] + '_' + init_stamp + '.png')

if imu_plot:


    for n, robot in enumerate(robots):
        fig, ax = plt.subplots(3, 2)
        ax[0,0].plot(query_stamps, imus.data[robot][input_sensor]['angular_velocity.x'].values, label = 'gyro x')
        ax[0,0].plot(query_stamps, gyro[n][:,0], label = 'gt gyro x')
        ax[1,0].plot(query_stamps, imus.data[robot][input_sensor]['angular_velocity.y'].values, label = 'gyro y')
        ax[1,0].plot(query_stamps, gyro[n][:,1], label = 'gt gyro y')
        ax[2,0].plot(query_stamps, imus.data[robot][input_sensor]['angular_velocity.z'].values, label = 'gyro z')
        ax[2,0].plot(query_stamps, gyro[n][:,2], label = 'gt gyro z')
        ax[0,1].plot(query_stamps, imus.data[robot][input_sensor]['linear_acceleration.x'].values, label = 'accel x')
        ax[0,1].plot(query_stamps, accel[n][:,0], label = 'gt accel x')
        ax[1,1].plot(query_stamps, imus.data[robot][input_sensor]['linear_acceleration.y'].values, label = 'accel y')
        ax[1,1].plot(query_stamps, accel[n][:,1], label = 'gt accel y')
        ax[2,1].plot(query_stamps, imus.data[robot][input_sensor]['linear_acceleration.z'].values, label = 'accel z')
        ax[2,1].plot(query_stamps, accel[n][:,2], label = 'gt accel z')
        for a in ax:
            for b in a:
                b.legend()


plt.show()
