import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
from pyuwbcalib.utils import (
    set_plotting_env,
)
from utils.states import (
    CompositeState,
    MatrixLieGroupState,
    StateWithCovariance,
)
from utils.misc import (
    RangeData,
    GaussianResult,
    GaussianResultList,
    height_measurements,
    plot_error,
)
from utils.process_models import (
    Input,
    BodyFrameVelocity,
    CompositeProcessModel,
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
"""

# Set the plotting environment
set_plotting_env()
plt.rcParams.update({'font.size': 10})  

# plots
ekf = True
error_plot = True
save_fig = True
save_results = False


""" Load data """
# parser = argparse.ArgumentParser()
# parser.add_argument('--exp', required=True)
# args = parser.parse_args()
# exp = args.exp
exp = "1c"
folder = "/media/syedshabbir/Seagate B/data"
miluv = DataLoader(exp, exp_dir = folder, barometer = False)

""" Preliminaries """
robots = list(miluv.data.keys())
input_sensors = ['vio', 'imu_px4']
input_freq = 30
start_time, end_time = miluv.get_timerange(sensors = input_sensors)
end_time = end_time - 5
query_stamps = np.arange(start_time, end_time, 1/input_freq)

""" Get Data """
input = miluv.by_timestamps(query_stamps, sensors= input_sensors)
pose = [miluv.data[robot]['mocap'].SE3_state(query_stamps) for robot in robots]
range_data = miluv.by_timerange(start_time, end_time,sensors=['uwb_range'])

""" Create measurements """
range_data = RangeData(range_data)
range_data = range_data.filter_by_bias( max_bias=0.3)
meas_data = range_data.to_measurements(reference_id = 'world')
R = [3*0.1, 3*0.1, 3*0.1]
bias = [-0.0924, -0.0088, -0.1207]
height = height_measurements(miluv, start_time, end_time, R, bias)
meas_data.extend(height)
meas_data = sorted(meas_data, key=lambda x: x.stamp)
    
""" Create ground truth data """
ground_truth = []
for i in range(len(query_stamps)):
    x = [MatrixLieGroupState(  value = pose[n][i], 
            state_id = robot, stamp = query_stamps[i],) 
        for n, robot in enumerate(robots) ]
    ground_truth.append(CompositeState(x, stamp = query_stamps[i]))
    
""" State and input covariance, process model, and initial state """
# Initial state
x0 = ground_truth[0].copy()

# Covariance matrices and process model
P0 = np.diag([0.02**2, 0.02**2, 0.02**2, 2*0.1, 2*0.1, 2*0.1])
Q = 2*np.diag([ 5*0.1**2, 5*0.1**2, 10 * 0.15**2, 2*0.1,2*0.1, 2*0.01])
process_model = CompositeProcessModel([BodyFrameVelocity(Q) for r in robots])

# Composite covariance
n_states = len(x0.value)
P0 = np.kron(np.eye(n_states), P0)

""" Create input data """

input_data = []
for i in range(len(query_stamps)):
    u = [Input(
        gyro = input.data[robot]['imu_px4'].iloc[i][
        ['angular_velocity.x', 
        'angular_velocity.y', 
        'angular_velocity.z']].values, 

        vio = (pose[n][i][:3,:3].T @ # Rotate to body frame
         (input.data[robot]['vio'].iloc[i][
        ['twist.linear.x', 
         'twist.linear.y', 
         'twist.linear.z']].values)),
        stamp = query_stamps[i], 
        state_id = robot)
        for n, robot in enumerate(robots)]
    
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
        results_list.append(GaussianResult(x, ground_truth[k]))
    results = GaussianResultList(results_list)

script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
if ekf:
    """ Print position RMSE """
    pos_rmse = {robot: {} for robot in robots}
    for i, robot in enumerate(robots):
        pos = np.array([r.get_state_by_id(robot).value[0:3,-1] 
                        for r in results.state])
        true_pos = np.array([r.get_state_by_id(robot).value[0:3,-1] 
                             for r in results.state_true])
        error = (pos - true_pos).ravel()
        pos_rmse[robot] = np.sqrt(error.T @ error / len(error))
    for robot in robots:
        print(f"Position RMSE for Experiment: {exp} and robot {robot}: {pos_rmse[robot]} m")

if ekf and save_results:
    """ Save results to a pkl file """
    folder = os.path.join(script_dir, f'results')
    os.umask(0)
    os.makedirs(folder, exist_ok=True)
    filename = f'results_vio_{exp}.pkl'
    file_path = os.path.join(folder, filename)
    with open(file_path, 'wb') as file:
        pickle.dump((results, pos_rmse), file)

if ekf and error_plot:
    """ Plot error """
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
                    
    if save_fig:      
        figs_folder = os.path.join(script_dir, 'figures')
        os.umask(0)
        os.makedirs(figs_folder, exist_ok=True)      
        for i, (fig, axs) in enumerate(figs):
            plt.savefig(os.path.join(figs_folder, f'error_vio_{exp}_{robots[i]}.png'))

plt.show()