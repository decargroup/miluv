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


""" Load data """
exp = '33'
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
velocity = [miluv.data[robot]['mocap'].velocity(query_stamps) for robot in robots]
max_speed = np.max([np.linalg.norm(v, axis=1) for v in velocity])
print(max_speed)