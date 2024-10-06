# %%
from miluv.data import DataLoader
import utils.liegroups as liegroups
import miluv.utils as utils
import examples.ekfutils.vins_one as vins_one

import numpy as np
import pandas as pd

#################### LOAD SENSOR DATA ####################
miluv = DataLoader("13", imu = "px4", cam = None, mag = False)
data = miluv.data["ifo001"]
vins = utils.load_vins("13", "ifo001", loop = False, postprocessed = True)

#################### GET UWB TRANSCEIVER POSITIONS ####################
# anchor_positions = utils.get_anchors()
# tag_moment_arms = utils.get_tag_moment_arms()
anchor_positions = {
    0: np.array([3.273827392578125, 3.46404736328125, 1.8093309326171875]).reshape(3, 1),
    1: np.array([3.186386962890625, 0.27394485473632812, 1.5884853515625]).reshape(3, 1),
    2: np.array([2.850500244140625, -2.923056884765625, 1.89742041015625]).reshape(3, 1),
    3: np.array([-2.497634521484375, -3.5018203125, 1.7730911865234375]).reshape(3, 1),
    4: np.array([-2.95793310546875, 0.6128419189453125, 1.65714208984375]).reshape(3, 1),
    5: np.array([-2.734676513671875, 3.65854248046875, 1.890254638671875]).reshape(3, 1),
}
tag_moment_arms = {
    10: np.array([0.13189,-0.17245,-0.05249]).reshape(3, 1),
    11: np.array([-0.17542,0.15712,-0.05307]).reshape(3, 1),
    20: np.array([0.16544,-0.15085,-0.03456]).reshape(3, 1),
    21: np.array([-0.15467,0.16972,-0.01680]).reshape(3, 1),
    30: np.array([0.16685,-0.18113,-0.05576]).reshape(3, 1),
    31: np.array([-0.13485,0.15468,-0.05164]).reshape(3, 1),
}

#################### ALIGN SENSOR DATA TIMESTAMPS ####################
# TODO DOCUMENTATION: Timestamps where range measurements or height are available, assume vins measurements are constant between these timestamps
# We also get rid of timestamps before the first vins measurement and after the last vins measurement
query_timestamps = np.append(
    data["uwb_range"]["timestamp"].to_numpy(),
    data["height"]["timestamp"].to_numpy()
)
query_timestamps = query_timestamps[query_timestamps > vins["timestamp"].iloc[0]]
query_timestamps = query_timestamps[query_timestamps < vins["timestamp"].iloc[-1]]
query_timestamps = np.sort(np.unique(query_timestamps))

imu_at_query_timestamps = miluv.query_by_timestamps(query_timestamps, robots="ifo001", sensors="imu_px4")["ifo001"]
gyro: pd.DataFrame = imu_at_query_timestamps["imu_px4"][["timestamp", "angular_velocity.x", "angular_velocity.y", "angular_velocity.z"]]
vins = utils.zero_order_hold(query_timestamps, vins) # add this and use it in by_timestamp inside the dataloader

#################### LOAD GROUND TRUTH DATA ####################
gt_se3 = liegroups.get_se3_poses(
    data["mocap_pos"](query_timestamps), 
    data["mocap_quat"](query_timestamps)
)

#################### EKF ####################
ekf = vins_one.EKF(gt_se3[0], anchor_positions, tag_moment_arms)
for i in range(0, len(query_timestamps)):
    input = np.array([
        gyro.iloc[i]["angular_velocity.x"], 
        gyro.iloc[i]["angular_velocity.y"], 
        gyro.iloc[i]["angular_velocity.z"],
        vins.iloc[i]["twist.linear.x"],
        vins.iloc[i]["twist.linear.y"],
        vins.iloc[i]["twist.linear.z"],
    ])
    
    dt = (query_timestamps[i] - query_timestamps[i - 1]) if i > 0 else 0
    ekf.predict(input, dt)
    
    range_idx = np.where(data["uwb_range"]["timestamp"] == query_timestamps[i])[0]
    if len(range_idx) > 0:
        range_data = data["uwb_range"].iloc[range_idx]
        ekf.correct({
            "range": float(range_data["range"].iloc[0]),
            "to_id": int(range_data["to_id"].iloc[0]),
            "from_id": int(range_data["from_id"].iloc[0])
        })
        
    height_idx = np.where(data["height"]["timestamp"] == query_timestamps[i])[0]
    if len(height_idx) > 0:
        height_data = data["height"].iloc[height_idx]
        ekf.correct({
            "height": float(height_data["range"].iloc[0])
        })
        
    print(ekf.x)

#################### POSTPROCESS ####################

# %%
