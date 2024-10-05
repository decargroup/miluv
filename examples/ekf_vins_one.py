# %%
from miluv.data import DataLoader
from miluv.utils import load_vins, zero_order_hold

import numpy as np

#################### LOAD SENSOR DATA ####################
miluv = DataLoader("13", imu = "px4", cam = None, mag = False)
data = miluv.data["ifo001"]
# vins = load_vins("13", "ifo001", postprocessed = True) # TODO: load postprocessed vins data

#################### ALIGN SENSOR DATA TIMESTAMPS ####################
# TODO DOCUMENTATION: Timestamps where range measurements or height are available, assume vins measurements are constant between these timestamps
query_timestamps = np.append(
    data["uwb_range"]["timestamp"].to_numpy(),
    data["height"]["timestamp"].to_numpy()
)
query_timestamps = np.sort(np.unique(query_timestamps))

data_at_query_timestamps = miluv.by_timestamps(query_timestamps)["ifo001"]
gyro = data_at_query_timestamps["imu_px4"][["angular_velocity.x", "angular_velocity.y", "angular_velocity.z"]]
vins = zero_order_hold(query_timestamps, vins) # add this and use it in by_timestamp inside the dataloader

#################### LOAD GROUND TRUTH DATA ####################
gt_pos = data["mocap_pos"](vins["timestamp"])
gt_quat = data["mocap_quat"](vins["timestamp"])
# gt_SE3 = ??

#################### EKF ####################
# ekf = EKF("vins_one")
# ekf.initialize(state)
# for sensor data
#     ekf.predict()
#     ekf.correct()

#################### POSTPROCESS ####################

# %%
