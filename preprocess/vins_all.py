import subprocess
import pandas as pd
from examples.evaluate_vins import evaluate_vins

def call_vins(exp_name, robot_id):
    print("Running VINS for experiment", exp_name, "robot", robot_id)
    subprocess.run(["./preprocess/run_vins.sh", exp_name, robot_id])
    
    return evaluate_vins(exp_name, robot_id, False)
    

if __name__ == "__main__":
    data = pd.read_csv("config/experiments.csv")
    rmse_df = pd.DataFrame(columns=["experiment", "robot", "rmse_loop", "rmse_no_loop"])
    for i in range(len(data)):
        num_robots = data["num_robots"].iloc[i]
        for j in range(num_robots):
            exp_name = str(data["experiment"].iloc[i])
            rmse = call_vins(exp_name, f"ifo00{j+1}")
            new_df = pd.DataFrame({
                "experiment": [exp_name],
                "robot": [f"ifo00{j+1}"],
                "rmse_loop": [rmse["rmse_loop"]],
                "rmse_no_loop": [rmse["rmse_no_loop"]]
            })
            rmse_df = pd.concat([rmse_df, new_df], ignore_index=True)
            
            rmse_df.to_csv("data/vins/vins_rmse.csv", index=False)
            
    # rmse_df.to_csv("data/vins/vins_rmse.csv", index=False)
            