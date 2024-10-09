# %%
from miluv.data import DataLoader
import utils.liegroups as liegroups
import miluv.utils as utils
import examples.ekfutils.vins_one as vins_one
import examples.ekfutils.postprocessing as postprocessing
import examples.ekfutils.common as common

import numpy as np
import pandas as pd

#################### EXPERIMENT DETAILS ####################
exp_name = "13"

#################### LOAD SENSOR DATA ####################
miluv = DataLoader(exp_name, imu = "px4", cam = None, mag = False)
data = miluv.data["ifo001"]
vins = utils.load_vins(exp_name, "ifo001", loop = False, postprocessed = True)

#################### ALIGN SENSOR DATA TIMESTAMPS ####################
# TODO DOCUMENTATION: Timestamps where range measurements or height are available, assume vins measurements are constant between these timestamps
# We also get rid of timestamps before the first vins measurement and after the last vins measurement
query_timestamps = np.append(
    data["uwb_range"]["timestamp"].to_numpy(), data["height"]["timestamp"].to_numpy()
)
query_timestamps = query_timestamps[query_timestamps > vins["timestamp"].iloc[0]]
query_timestamps = query_timestamps[query_timestamps < vins["timestamp"].iloc[-1]]
query_timestamps = np.sort(np.unique(query_timestamps))

imu_at_query_timestamps = miluv.query_by_timestamps(query_timestamps, robots="ifo001", sensors="imu_px4")["ifo001"]
gyro: pd.DataFrame = imu_at_query_timestamps["imu_px4"][["timestamp", "angular_velocity.x", "angular_velocity.y", "angular_velocity.z"]]
vins = utils.zero_order_hold(query_timestamps, vins)

#################### LOAD GROUND TRUTH DATA ####################
gt_se3 = liegroups.get_se3_poses(
    data["mocap_pos"](query_timestamps), data["mocap_quat"](query_timestamps)
)

# Use ground truth data to convert VINS data from the absolute (mocap) frame to the robot's body frame
vins = common.convert_vins_velocity_to_body_frame(vins, gt_se3)

#################### EKF ####################
# Initialize a variable to store the EKF state and covariance at each query timestamp for postprocessing
ekf_history = postprocessing.History()

# Initialize the EKF with the first ground truth pose, the anchor postions, and UWB tag moment arms
ekf = vins_one.EKF(gt_se3[0], miluv.anchors, miluv.tag_moment_arms)

# Iterate through the query timestamps
for i in range(0, len(query_timestamps)):
    # Get the gyro and vins data at this query timestamp for the EKF input
    input = np.array([
        gyro.iloc[i]["angular_velocity.x"], gyro.iloc[i]["angular_velocity.y"], 
        gyro.iloc[i]["angular_velocity.z"], vins.iloc[i]["twist.linear.x"],
        vins.iloc[i]["twist.linear.y"], vins.iloc[i]["twist.linear.z"],
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
        
    ekf_history.add(query_timestamps[i], ekf.x, ekf.P)

#################### POSTPROCESS ####################
analysis = postprocessing.Evaluate(gt_se3, ekf_history, exp_name)

# TODO: Mention in the documentation that we need to create the results/plots directory
analysis.plot_error()
analysis.plot_poses()
analysis.save_results()

# %%
