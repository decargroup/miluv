from miluv.data import DataLoader
import miluv.utils as utils

import matplotlib.pyplot as plt

plt.rcParams['axes.grid'] = True

#################### EXPERIMENT DETAILS ####################
exp_name = "1c"
robot = "ifo001"

#################### LOAD SENSOR DATA ####################
miluv = DataLoader(exp_name, cam = None, mag = False)
data = miluv.data[robot]

imu_px4 = data["imu_px4"]
time = imu_px4["timestamp"]
pos = data["mocap_pos"](time)
quat = data["mocap_quat"](time)

#################### GROUND TRUTH IMU ####################
gt_gyro = utils.get_angular_velocity_splines(time, data["mocap_quat"])(time)
gt_accelerometer = utils.get_accelerometer_splines(time, data["mocap_pos"], data["mocap_quat"])(time)

#################### VISUALIZE GYROSCOPE ####################
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
fig.suptitle("Gyroscope")

axs[0].plot(time, imu_px4["angular_velocity.x"], label="IMU PX4 Measurement")
axs[1].plot(time, imu_px4["angular_velocity.y"], label="IMU PX4 Measurement")
axs[2].plot(time, imu_px4["angular_velocity.z"], label="IMU PX4 Measurement")

axs[0].plot(time, gt_gyro[0, :], label="Ground Truth")
axs[1].plot(time, gt_gyro[1, :], label="Ground Truth")
axs[2].plot(time, gt_gyro[2, :], label="Ground Truth")

axs[0].set_ylabel("Gyro X (rad/s)")
axs[1].set_ylabel("Gyro Y (rad/s)")
axs[2].set_ylabel("Gyro Z (rad/s)")

axs[0].set_ylim([-1, 1])
axs[1].set_ylim([-1, 1])
axs[2].set_ylim([-1, 1])

axs[0].legend()

#################### VISUALIZE GYROSCOPE ERROR AND BIAS ####################
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
fig.suptitle("Gyroscope Error and Bias")

axs[0].plot(time, gt_gyro[0, :] - imu_px4["angular_velocity.x"], label="Measurement Error")
axs[1].plot(time, gt_gyro[1, :] - imu_px4["angular_velocity.y"], label="Measurement Error")
axs[2].plot(time, gt_gyro[2, :] - imu_px4["angular_velocity.z"], label="Measurement Error")

axs[0].plot(time, imu_px4["gyro_bias.x"], label="IMU Bias")
axs[1].plot(time, imu_px4["gyro_bias.y"], label="IMU Bias")
axs[2].plot(time, imu_px4["gyro_bias.z"], label="IMU Bias")

axs[0].set_ylabel("Gyro X (rad/s)")
axs[1].set_ylabel("Gyro Y (rad/s)")
axs[2].set_ylabel("Gyro Z (rad/s)")

axs[0].set_ylim([-0.5, 0.5])
axs[1].set_ylim([-0.5, 0.5])
axs[2].set_ylim([-0.5, 0.5])

axs[0].legend()

#################### VISUALIZE ACCELEROMETER ####################
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
fig.suptitle("Accelerometer")

axs[0].plot(time, imu_px4["linear_acceleration.x"], label="IMU PX4 Measurement")
axs[1].plot(time, imu_px4["linear_acceleration.y"], label="IMU PX4 Measurement")
axs[2].plot(time, imu_px4["linear_acceleration.z"], label="IMU PX4 Measurement")

axs[0].plot(time, gt_accelerometer[0, :], label="Ground Truth")
axs[1].plot(time, gt_accelerometer[1, :], label="Ground Truth")
axs[2].plot(time, gt_accelerometer[2, :], label="Ground Truth")

axs[0].set_ylabel("Accel X (m/s^2)")
axs[1].set_ylabel("Accel Y (m/s^2)")
axs[2].set_ylabel("Accel Z (m/s^2)")

axs[0].set_ylim([-5, 5])
axs[1].set_ylim([-5, 5])
axs[2].set_ylim([5, 15])

axs[0].legend()

#################### VISUALIZE ACCELEROMETER ERROR AND BIAS ####################
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
fig.suptitle("Accelerometer Error and Bias")

axs[0].plot(time, gt_accelerometer[0, :] - imu_px4["linear_acceleration.x"], label="Measurement Error")
axs[1].plot(time, gt_accelerometer[1, :] - imu_px4["linear_acceleration.y"], label="Measurement Error")
axs[2].plot(time, gt_accelerometer[2, :] - imu_px4["linear_acceleration.z"], label="Measurement Error")

axs[0].plot(time, imu_px4["accel_bias.x"], label="IMU Bias")
axs[1].plot(time, imu_px4["accel_bias.y"], label="IMU Bias")
axs[2].plot(time, imu_px4["accel_bias.z"], label="IMU Bias")

axs[0].set_ylabel("Accel X (m/s^2)")
axs[1].set_ylabel("Accel Y (m/s^2)")
axs[2].set_ylabel("Accel Z (m/s^2)")

axs[0].set_ylim([-3, 3])
axs[1].set_ylim([-3, 3])
axs[2].set_ylim([-3, 3])

axs[0].legend()

plt.show(block=True)

