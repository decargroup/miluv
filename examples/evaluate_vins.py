# %%
from miluv.data import DataLoader
from miluv.utils import (
    get_mocap_splines, load_vins, align_frames
)
import pandas as pd
import matplotlib.pyplot as plt

exp_name = "1c"
robot_id = "ifo001"

# Read sensor and mocap data
mv = DataLoader(exp_name, barometer=False, cir=False)

# Read vins data
vins = load_vins(exp_name, robot_id)

# Get mocap data at vins timestamps
pos = mv.data["ifo001"]["mocap_pos"](vins["timestamp"])
quat = mv.data["ifo001"]["mocap_quat"](vins["timestamp"])

# Align frame
df_mocap = pd.DataFrame({
    "timestamp": vins["timestamp"],
    "pose.position.x": pos[0],
    "pose.position.y": pos[1],
    "pose.position.z": pos[2],
    "pose.orientation.x": quat[0],
    "pose.orientation.y": quat[1],
    "pose.orientation.z": quat[2],
    "pose.orientation.w": quat[3],
})
# vins_old = vins.copy()
vins = align_frames(vins, df_mocap)

# Compare vins and mocap data
plt.figure()
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot3D(vins["pose.position.x"], vins["pose.position.y"], vins["pose.position.z"], label="vins")
ax.plot3D(pos[0], pos[1], pos[2], label="mocap")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
# plt.plot(vins["pose.position.x"], vins["pose.position.y"], label="vins")
# plt.plot(pos[0], pos[1], label="mocap")

fig, axs = plt.subplots(3, 1)
axs[0].plot(vins["timestamp"], vins["pose.position.x"], label="vins")
axs[0].plot(vins["timestamp"], pos[0], label="mocap")
axs[0].set_title("x")
axs[0].legend()
axs[1].plot(vins["timestamp"], vins["pose.position.y"], label="vins")
axs[1].plot(vins["timestamp"], pos[1], label="mocap")
axs[1].set_title("y")
axs[1].legend()
axs[2].plot(vins["timestamp"], vins["pose.position.z"], label="vins")
axs[2].plot(vins["timestamp"], pos[2], label="mocap")
axs[2].set_title("z")
axs[2].legend()
plt.legend()

plt.show(block=True)

# %%
