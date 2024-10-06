# %%
from miluv.data import DataLoader
# from miluv.utils import load_vins, zero_order_hold
import utils.liegroups as liegroups
import miluv.utils as utils

import numpy as np

#################### LOAD SENSOR DATA ####################
miluv = DataLoader("13", imu = "px4", cam = None, mag = False)
data = miluv.data["ifo001"]
vins = utils.load_vins("13", "ifo001", loop = False, postprocessed = True)

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
gyro = imu_at_query_timestamps["imu_px4"][["timestamp", "angular_velocity.x", "angular_velocity.y", "angular_velocity.z"]]
vins = utils.zero_order_hold(query_timestamps, vins) # add this and use it in by_timestamp inside the dataloader

#################### LOAD GROUND TRUTH DATA ####################
gt_se3 = liegroups.get_se3_poses(
    data["mocap_pos"](query_timestamps), 
    data["mocap_quat"](query_timestamps)
)

#################### EKF ####################
# ekf = EKF("vins_one")
# ekf.initialize(state)
# for sensor data
#     ekf.predict()
#     ekf.correct()

#################### POSTPROCESS ####################

# %%
