# %%
import pandas as pd
import subprocess

from joblib import Parallel, delayed

def call_vins_ekf(exp_name, num_robots):
    print("Running VINS EKF for experiment", exp_name)
    
    if num_robots == 1:
        subprocess.run(["python", "examples/ekf_vins_one_robot.py", exp_name])
    elif num_robots == 3:
        subprocess.run(["python", "examples/ekf_vins_three_robots.py", exp_name])
        
def call_imu_ekf(exp_name, num_robots):
    print("Running IMU EKF for experiment", exp_name)
    
    if num_robots == 1:
        subprocess.run(["python", "examples/ekf_imu_one_robot.py", exp_name])
    elif num_robots == 3:
        subprocess.run(["python", "examples/ekf_imu_three_robots.py", exp_name])
    

if __name__ == "__main__":
    data = pd.read_csv("config/experiments.csv")

    tasks = []
    for i in range(len(data)):
        num_robots = data["num_robots"].iloc[i]
        exp_name = str(data["experiment"].iloc[i])
        tasks.append(delayed(call_vins_ekf)(exp_name, num_robots))
        tasks.append(delayed(call_imu_ekf)(exp_name, num_robots))

    # Run tasks in parallel
    Parallel(n_jobs=-1)(tasks)
 
# %%
