import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils.misc import plot_error
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
folder = os.path.join(script_dir, f'results')
robots = ["ifo001", "ifo002", "ifo003"]

error_plot = False

exp = "1c"
filename = f'results_{exp}.pkl'
file_path = os.path.join(folder, filename)
with open(file_path, 'rb') as file:
   results, pos_rmse = pickle.load(file)

# individual rmses imu
(att_dof, pos_dof, vel_dof, 
gyro_bias_dof, accel_bias_dof) = (3, 3, 3, 3, 3)
dof = att_dof + pos_dof + vel_dof + gyro_bias_dof + accel_bias_dof
n_states = int(results.dof[0] / dof)
robot_ids = ["ifo001", "ifo002", "ifo003"]
pos_e = {id: {} for id in robot_ids}
pos_rmse = {id: {} for id in robot_ids}
for i, id in enumerate(robot_ids):
    pos_e[id] = np.array([(e.reshape(-1, dof)[i]
                [att_dof + vel_dof:att_dof + vel_dof + pos_dof]).ravel() 
                for e in results.error]).ravel()
    pos_rmse[id] = np.sqrt(pos_e[id].T @ pos_e[id] / len(pos_e[id]))

for id in robot_ids:
    print(f"Position RMSE for Experiment: {exp} and robot {id}: {pos_rmse[id]} m")


# overall rmse
(att_dof, pos_dof, vel_dof, 
gyro_bias_dof, accel_bias_dof) = (3, 3, 3, 3, 3)
dof = att_dof + pos_dof + vel_dof + gyro_bias_dof + accel_bias_dof
n_states = int(results.dof[0] / dof)
pos_e = np.array([(e.reshape(-1, dof)
            [:,att_dof + vel_dof:att_dof + vel_dof + pos_dof]).ravel() 
            for e in results.error]).ravel()
pos_rmse = np.sqrt(pos_e.T @ pos_e / len(pos_e))
print(f"Position RMSE for Experiment: {exp}: {pos_rmse} m")

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

   
   