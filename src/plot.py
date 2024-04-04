import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils.misc import plot_error
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
folder = os.path.join(script_dir, f'results')
# robots = ["ifo001", "ifo002", "ifo003"]
robots = ["ifo001"]

error_plot = False

exp = "35"
filename = f'results_vel_{exp}.pkl'
file_path = os.path.join(folder, filename)
with open(file_path, 'rb') as file:
   results, pos_rmse = pickle.load(file)

pos_rmse = {robot: {} for robot in robots}
dof = 3
for i, robot in enumerate(robots):
    pos = np.array([r.get_state_by_id(robot).position 
                    for r in results.state])
    true_pos = np.array([r.get_state_by_id(robot).position 
                            for r in results.state_true])
    error = pos - true_pos
    pos_rmse[robot] = np.sqrt(np.mean([e.T @ e / dof for e in error]))
for robot in robots:
    print(f"Position RMSE for Experiment: {exp} and robot {robot}: {pos_rmse[robot]} m")

if error_plot:
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

plt.show()

   
   