# %%
import numpy as np
import pandas as pd

import sys

from miluv.data import DataLoader
import miluv.utils as utils
import examples.ekfutils.imu_one_robot_models as model
import examples.ekfutils.common as common

def run_ekf_imu_one_robot(exp_name: str):
    
    #################### LOAD SENSOR DATA ####################
    miluv = DataLoader(exp_name, imu = "px4", cam = None, mag = False)
    data = miluv.data["ifo001"]

    #################### ALIGN SENSOR DATA TIMESTAMPS ####################
    # Set the query timestamps to be all the timestamps where UWB range or height data is available
    query_timestamps = np.append(
        data["uwb_range"]["timestamp"].to_numpy(), data["height"]["timestamp"].to_numpy()
    )
    query_timestamps = np.sort(np.unique(query_timestamps))

    imu_at_query_timestamps = miluv.query_by_timestamps(query_timestamps, robots="ifo001", sensors="imu_px4")["ifo001"]
    accel: pd.DataFrame = imu_at_query_timestamps["imu_px4"][["timestamp", "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"]]
    gyro: pd.DataFrame = imu_at_query_timestamps["imu_px4"][["timestamp", "angular_velocity.x", "angular_velocity.y", "angular_velocity.z"]]

    #################### LOAD GROUND TRUTH DATA ####################
    gt_se23 = utils.get_se23_poses(
        data["mocap_quat"](query_timestamps), data["mocap_pos"].derivative(nu=1)(query_timestamps), data["mocap_pos"](query_timestamps)
    )
    gt_bias = imu_at_query_timestamps["imu_px4"][[
        "gyro_bias.x", "gyro_bias.y", "gyro_bias.z", 
        "accel_bias.x", "accel_bias.y", "accel_bias.z"
    ]].to_numpy()

    #################### EKF ####################
    # Initialize a variable to store the EKF state and covariance at each query timestamp for postprocessing
    ekf_history = {
        "pose": common.MatrixStateHistory(state_dim=5, covariance_dim=9),
        "bias": common.VectorStateHistory(state_dim=6)
    }

    # Initialize the EKF with the first ground truth pose, the anchor positions, and UWB tag moment arms
    ekf = model.EKF(gt_se23[0], miluv.anchors, miluv.tag_moment_arms)

    # Iterate through the query timestamps
    for i in range(0, len(query_timestamps)):
        # Get the gyro and vins data at this query timestamp for the EKF input
        input = np.array([
            gyro.iloc[i]["angular_velocity.x"], gyro.iloc[i]["angular_velocity.y"], 
            gyro.iloc[i]["angular_velocity.z"], accel.iloc[i]["linear_acceleration.x"], 
            accel.iloc[i]["linear_acceleration.y"], accel.iloc[i]["linear_acceleration.z"]
        ])
        
        # Do an EKF prediction using the gyro and vins data
        dt = (query_timestamps[i] - query_timestamps[i - 1]) if i > 0 else 0
        ekf.predict(input, dt)
        
        # Check if range data is available at this query timestamp, and do an EKF correction
        range_idx = np.where(data["uwb_range"]["timestamp"] == query_timestamps[i])[0]
        if len(range_idx) > 0:
            range_data = data["uwb_range"].iloc[range_idx]
            ekf.correct({
                "range": float(range_data["range"].iloc[0]),
                "to_id": int(range_data["to_id"].iloc[0]),
                "from_id": int(range_data["from_id"].iloc[0])
            })
            
        # Check if height data is available at this query timestamp, and do an EKF correction
        height_idx = np.where(data["height"]["timestamp"] == query_timestamps[i])[0]
        if len(height_idx) > 0:
            height_data = data["height"].iloc[height_idx]
            ekf.correct({"height": float(height_data["range"].iloc[0])})
            
        # Store the EKF state and covariance at this query timestamp
        ekf_history["pose"].add(query_timestamps[i], ekf.pose, ekf.pose_covariance)
        ekf_history["bias"].add(query_timestamps[i], ekf.bias, ekf.bias_covariance)

    #################### POSTPROCESS ####################
    analysis = model.EvaluateEKF(gt_se23, gt_bias, ekf_history, exp_name)

    analysis.plot_error()
    analysis.plot_poses()
    analysis.plot_bias_error()
    analysis.save_results()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        exp_name = "default_1_random3_0"
    else:
        exp_name = sys.argv[1]
    
    run_ekf_imu_one_robot(exp_name)

# %%
