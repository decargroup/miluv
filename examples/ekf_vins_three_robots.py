# %%
from miluv.data import DataLoader
import utils.liegroups as liegroups
import miluv.utils as utils
import examples.ekfutils.vins_three_robots as vins_three_robots
import examples.ekfutils.common as common

import numpy as np
import pandas as pd
from pymlg import SE3

#################### EXPERIMENT DETAILS ####################
exp_name = "1c"

#################### LOAD SENSOR DATA ####################
miluv = DataLoader(exp_name, imu = "px4", cam = None, mag = False)
data = miluv.data
vins = {robot: utils.load_vins(exp_name, robot, loop = False, postprocessed = True) for robot in data.keys()}

# Merge the UWB range and height data from all robots into a single dataframe
uwb_range = pd.concat([data[robot]["uwb_range"] for robot in data.keys()])
height = pd.concat([data[robot]["height"] for robot in data.keys()])

#################### ALIGN SENSOR DATA TIMESTAMPS ####################
# Set the query timestamps to be all the timestamps where UWB range or height data is available
# and within the time range of the VINS data
query_timestamps = np.append(uwb_range["timestamp"].to_numpy(), height["timestamp"].to_numpy())
query_timestamps = query_timestamps[
    (query_timestamps > vins["ifo001"]["timestamp"].iloc[0]) & (query_timestamps < vins["ifo001"]["timestamp"].iloc[-1]) &
    (query_timestamps > vins["ifo002"]["timestamp"].iloc[0]) & (query_timestamps < vins["ifo002"]["timestamp"].iloc[-1]) &
    (query_timestamps > vins["ifo003"]["timestamp"].iloc[0]) & (query_timestamps < vins["ifo003"]["timestamp"].iloc[-1])
]
query_timestamps = np.sort(np.unique(query_timestamps))

imu_at_query_timestamps = {
    robot: miluv.query_by_timestamps(query_timestamps, robots=robot, sensors="imu_px4")[robot]
    for robot in data.keys()
}
gyro: pd.DataFrame = {
    robot: imu_at_query_timestamps[robot]["imu_px4"][["timestamp", "angular_velocity.x", "angular_velocity.y", "angular_velocity.z"]]
    for robot in data.keys()
}
vins_at_query_timestamps = {
    robot: utils.zero_order_hold(query_timestamps, vins[robot]) for robot in data.keys()
}

#################### LOAD GROUND TRUTH DATA ####################
gt_se3 = {
    robot: liegroups.get_se3_poses(data[robot]["mocap_pos"](query_timestamps), data[robot]["mocap_quat"](query_timestamps)) 
    for robot in data.keys()
}

# Use ground truth data to convert VINS data from the absolute (mocap) frame to the robot's body frame
vins_body_frame = {
    robot: common.convert_vins_velocity_to_body_frame(vins_at_query_timestamps[robot], gt_se3[robot]) 
    for robot in data.keys()
}

#################### EKF ####################
# Initialize a variable to store the EKF state and covariance at each query timestamp for postprocessing
ekf_history = {
    robot: common.MatrixStateHistory(state_dim=4, covariance_dim=6) for robot in data.keys()
}

# Initialize the EKF with the first ground truth pose, the anchor postions, and UWB tag moment arms
ekf = vins_three_robots.EKF(
    {robot: gt_se3[robot][0] for robot in data.keys()}, 
    miluv.anchors, 
    miluv.tag_moment_arms
)

# Iterate through the query timestamps
for i in range(0, len(query_timestamps)):
    # Get the gyro and vins data at this query timestamp for the EKF input
    input = {
        robot: np.array([
            gyro[robot].iloc[i]["angular_velocity.x"], gyro[robot].iloc[i]["angular_velocity.y"], 
            gyro[robot].iloc[i]["angular_velocity.z"], vins_body_frame[robot].iloc[i]["twist.linear.x"],
            vins_body_frame[robot].iloc[i]["twist.linear.y"], vins_body_frame[robot].iloc[i]["twist.linear.z"],
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
        ekf.correct({"height": float(height_data["range"].iloc[0])})
        
    # Store the EKF state and covariance at this query timestamp
    for robot in data.keys():
        ekf_history[robot].add(query_timestamps[i], ekf.x[robot], ekf.get_covariance(robot))

#################### POSTPROCESS ####################
analysis = vins_three_robots.EvaluateEKF(gt_se3, ekf_history, exp_name)

analysis.plot_error()
analysis.plot_poses()
analysis.save_results()

# %%
