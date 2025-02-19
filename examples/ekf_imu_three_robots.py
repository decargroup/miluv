# %%
import numpy as np
import pandas as pd

import sys

from miluv.data import DataLoader
import miluv.utils as utils
import examples.ekfutils.imu_three_robots_models as model
import examples.ekfutils.common as common

def run_ekf_imu_three_robots(exp_name: str):
    #################### LOAD SENSOR DATA ####################
    miluv = DataLoader(exp_name, imu = "px4", cam = None, mag = False)
    data = miluv.data

    # Merge the UWB range and height data from all robots into a single dataframe
    uwb_range = pd.concat([data[robot]["uwb_range"] for robot in data.keys()])
    height = pd.concat([data[robot]["height"].assign(robot=robot) for robot in data.keys()])

    #################### ALIGN SENSOR DATA TIMESTAMPS ####################
    # Set the query timestamps to be all the timestamps where UWB range or height data is available
    query_timestamps = np.append(uwb_range["timestamp"].to_numpy(), height["timestamp"].to_numpy())
    query_timestamps = np.sort(np.unique(query_timestamps))

    imu_at_query_timestamps = {
        robot: miluv.query_by_timestamps(query_timestamps, robots=robot, sensors="imu_px4")[robot]
        for robot in data.keys()
    }
    gyro: pd.DataFrame = {
        robot: imu_at_query_timestamps[robot]["imu_px4"][["timestamp", "angular_velocity.x", "angular_velocity.y", "angular_velocity.z"]]
        for robot in data.keys()
    }
    accel: pd.DataFrame = {
        robot: imu_at_query_timestamps[robot]["imu_px4"][["timestamp", "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"]]
        for robot in data.keys()
    }

    #################### LOAD GROUND TRUTH DATA ####################
    gt_se23 = {
        robot: utils.get_se23_poses(
            data[robot]["mocap_quat"](query_timestamps), data[robot]["mocap_pos"].derivative(nu=1)(query_timestamps), data[robot]["mocap_pos"](query_timestamps)
        )
        for robot in data.keys()
    }
    gt_bias = {
        robot: imu_at_query_timestamps[robot]["imu_px4"][[
            "gyro_bias.x", "gyro_bias.y", "gyro_bias.z", 
            "accel_bias.x", "accel_bias.y", "accel_bias.z"
        ]].to_numpy()
        for robot in data.keys()
    }

    #################### EKF ####################
    # Initialize a variable to store the EKF state and covariance at each query timestamp for postprocessing
    ekf_history = {
        robot: {
            "pose": common.MatrixStateHistory(state_dim=5, covariance_dim=9),
            "bias": common.VectorStateHistory(state_dim=6)
        }
        for robot in data.keys()
    }

    # Initialize the EKF with the first ground truth pose, the anchor positions, and UWB tag moment arms
    ekf = model.EKF(
        {robot: gt_se23[robot][0] for robot in data.keys()}, 
        miluv.anchors, 
        miluv.tag_moment_arms
    )

    # Iterate through the query timestamps
    for i in range(0, len(query_timestamps)):
        # Get the gyro and vins data at this query timestamp for the EKF input
        input = {
            robot: np.array([
                gyro[robot].iloc[i]["angular_velocity.x"], gyro[robot].iloc[i]["angular_velocity.y"], 
                gyro[robot].iloc[i]["angular_velocity.z"], accel[robot].iloc[i]["linear_acceleration.x"], 
                accel[robot].iloc[i]["linear_acceleration.y"], accel[robot].iloc[i]["linear_acceleration.z"]
            ])
            for robot in data.keys()
        }
        
        # Do an EKF prediction using the gyro and vins data
        dt = (query_timestamps[i] - query_timestamps[i - 1]) if i > 0 else 0
        ekf.predict(input, dt)
        
        # Check if range data is available at this query timestamp, and do an EKF correction
        range_idx = np.where(uwb_range["timestamp"] == query_timestamps[i])[0]
        if len(range_idx) > 0:
            range_data = uwb_range.iloc[range_idx]
            ekf.correct({
                "range": float(range_data["range"].iloc[0]),
                "to_id": int(range_data["to_id"].iloc[0]),
                "from_id": int(range_data["from_id"].iloc[0])
            })
            
        # Check if height data is available at this query timestamp, and do an EKF correction
        height_idx = np.where(height["timestamp"] == query_timestamps[i])[0]
        if len(height_idx) > 0:
            height_data = height.iloc[height_idx]
            ekf.correct({
                "height": float(height_data["range"].iloc[0]),
                "robot": height_data["robot"].iloc[0]
            })
            
        # Store the EKF state and covariance at this query timestamp
        for robot in data.keys():
            ekf_history[robot]["pose"].add(query_timestamps[i], ekf.pose[robot], ekf.pose_covariance[robot])
            ekf_history[robot]["bias"].add(query_timestamps[i], ekf.bias[robot], ekf.bias_covariance[robot])

    #################### POSTPROCESS ####################
    analysis = model.EvaluateEKF(gt_se23, gt_bias, ekf_history, exp_name)

    analysis.plot_error()
    analysis.plot_poses()
    analysis.plot_bias_error()
    analysis.save_results()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        exp_name = "default_3_random_0"
    else:
        exp_name = sys.argv[1]
    
    run_ekf_imu_three_robots(exp_name)

# %%
