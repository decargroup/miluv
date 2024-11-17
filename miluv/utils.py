import numpy as np
from csaps import csaps
from csaps import ISmoothingSpline
import pandas as pd
from scipy.spatial.transform import Rotation
import scipy as sp
import yaml

from pymlg import SO3

def get_anchors() -> dict[str, dict[int, np.ndarray]]:
    """
    Get anchor positions.
    
    Returns: 
    dict
    - Anchor positions.
    """
    
    with open(f"config/uwb/anchors.yaml", "r") as file:
        anchor_positions = yaml.safe_load(file)
    
    for constellation in anchor_positions:
        anchors = list(anchor_positions[constellation].keys())
        for anchor in anchors:
            pos = np.array([eval(anchor_positions[constellation][anchor])]).reshape(3, 1)
            anchor_positions[constellation][int(anchor)] = pos
            anchor_positions[constellation].pop(anchor)
    
    return anchor_positions

def get_tag_moment_arms() -> dict[str, dict[int, np.ndarray]]:
    """
    Get tag moment arms in the robot's own body frame.
    
    Args:
    - exp_name: Experiment name.
    
    Returns:
    dict
    - Tag moment arms in the robot's own body frame.
    """
    
    with open(f"config/uwb/tags.yaml", "r") as file:
        tag_moment_arms = yaml.safe_load(file)
    
    for robot in tag_moment_arms:
        tags = list(tag_moment_arms[robot].keys())
        for tag in tags:
            pos = np.array([eval(tag_moment_arms[robot][tag])]).reshape(3, 1)
            tag_moment_arms[robot][int(tag)] = pos
            tag_moment_arms[robot].pop(tag)
    
    return tag_moment_arms

def zero_order_hold(query_timestamps, data: pd.DataFrame) -> pd.DataFrame:
    """
    Zero-order hold interpolation for data.
    
    Args:
    - query_timestamps: Query timestamps.
    - data: Data to perform zero-order hold interpolation on.
    
    Returns:
    pd.DataFrame
    - New data with zero-order hold interpolation.
    """
    new_data = pd.DataFrame()
    
    # Ensure that query timestamps and data timestamps are sorted in ascending order
    query_timestamps = np.sort(query_timestamps)
    data.sort_values("timestamp", inplace=True)

    # Find the indices associated with the query timestamps using a zero-order hold
    idx_to_keep = []
    most_recent_idx = 0
    
    new_data["timestamp"] = query_timestamps
    for timestamp in query_timestamps:
        while most_recent_idx < len(data) and data["timestamp"].iloc[most_recent_idx] <= timestamp:
            most_recent_idx += 1
        idx_to_keep.append(most_recent_idx - 1)
        
    # Add the columns at the indices associated with the query timestamps
    for col in data.columns:
        if col == "timestamp":
            continue
        new_data[col] = data.iloc[idx_to_keep][col].values

    return new_data

def get_mocap_splines(mocap: pd.DataFrame) -> list[ISmoothingSpline, ISmoothingSpline]:
    """
    Get spline interpolations for mocap data.
    
    Args:
    - mocap: DataFrame containing mocap data.
    
    Returns:
    - pos_splines: Spline interpolation for position.
    - quat_splines: Spline interpolation for orientation.
    """

    # Get mocap data
    time = mocap['timestamp'].values
    pos = mocap[["pose.position.x", "pose.position.y",
                 "pose.position.z"]].values
    quat = mocap[[
        "pose.orientation.x", "pose.orientation.y", "pose.orientation.z",
        "pose.orientation.w"
    ]].values

    # Remove mocap gaps
    pos_gaps = np.linalg.norm(pos, axis=1) < 1e-6
    quat_gaps = np.linalg.norm(quat, axis=1) < 1e-6
    gaps = pos_gaps | quat_gaps

    time = time[~gaps]
    pos = pos[~gaps]
    quat = quat[~gaps]

    # Remove mocap outliers
    outliers = np.zeros(len(time), dtype=bool)
    last_good_R = Rotation.from_quat(quat[0]).as_matrix()
    for i in range(1, len(quat)):
        R_now = Rotation.from_quat(quat[i]).as_matrix()
        
        if (Rotation.from_matrix(last_good_R.T @ R_now).magnitude() > 1):
            outliers[i-1] = True
            outliers[i] = True
        else:
            last_good_R = R_now
            
    time = time[~outliers]
    pos = pos[~outliers]
    quat = quat[~outliers]

    # Normalize quaternion
    quat /= np.linalg.norm(quat, axis=1)[:, None]

    # Resolve quaternion discontinuities
    for i in range(1, len(quat)):
        if np.dot(quat[i], quat[i - 1]) < 0:
            quat[i] *= -1

    # Fit splines
    pos_splines = csaps(time, pos.T, smooth=0.9999).spline
    quat_splines = csaps(time, quat.T, smooth=0.9999).spline

    return pos_splines, quat_splines

def add_imu_bias(
    imu_data: pd.DataFrame,
    pos_spline: ISmoothingSpline, 
    quat_spline: ISmoothingSpline
) -> None:
    """
    Get IMU biases.
    
    Args:
    - imu_data: IMU data with the following columns:
        - timestamp
        - angular_velocity.x
        - angular_velocity.y
        - angular_velocity.z
        - linear_acceleration.x
        - linear_acceleration.y
        - linear_acceleration.z
    - pos_spline: Spline interpolation for position.
    - quat_spline: Spline interpolation for orientation.
    
    Returns:
    tuple
    - gyro_bias: Gyroscope bias at the query timestamps.
    - accel_bias: Accelerometer bias at the query timestamps.
    """
    time = imu_data["timestamp"].values
    gyro = np.array([
        imu_data["angular_velocity.x"],
        imu_data["angular_velocity.y"],
        imu_data["angular_velocity.z"],
    ])
    accel = np.array([
        imu_data["linear_acceleration.x"],
        imu_data["linear_acceleration.y"],
        imu_data["linear_acceleration.z"],
    ])
    
    gt_gyro = get_angular_velocity_splines(time, quat_spline)(time)
    gt_accel = get_accelerometer_splines(time, pos_spline, quat_spline)(time)
    
    gyro_bias = np.array([
        csaps(time, gyro[0, :] - gt_gyro[0, :], time, smooth=1e-4), 
        csaps(time, gyro[1, :] - gt_gyro[1, :], time, smooth=1e-4),
        csaps(time, gyro[2, :] - gt_gyro[2, :], time, smooth=1e-4)
    ])
    
    accel_bias = np.array([
        csaps(time, accel[0, :] - gt_accel[0, :], time, smooth=1e-3), 
        csaps(time, accel[1, :] - gt_accel[1, :], time, smooth=1e-3),
        csaps(time, accel[2, :] - gt_accel[2, :], time, smooth=1e-3)
    ])
    
    imu_data["gyro_bias.x"] = gyro_bias[0, :]
    imu_data["gyro_bias.y"] = gyro_bias[1, :]
    imu_data["gyro_bias.z"] = gyro_bias[2, :]
    
    imu_data["accel_bias.x"] = accel_bias[0, :]
    imu_data["accel_bias.y"] = accel_bias[1, :]
    imu_data["accel_bias.z"] = accel_bias[2, :]    

def get_angular_velocity_splines(time: np.ndarray, quat_splines: ISmoothingSpline) -> ISmoothingSpline:
    """
    Get spline interpolations for angular velocity in the robot's own body frame.
    
    Args:
    - time: Timestamps.
    - quat_splines: Spline interpolations for orientation.
    
    Returns:
    - gyro_splines: Spline interpolations for angular velocity.
    """
    q = quat_splines(time)
    q: np.ndarray = q / np.linalg.norm(q, axis=0)
    N = q.shape[1]
    q_dot = np.atleast_2d(quat_splines.derivative(nu=1)(time)).T
    eta = q[3]
    eps = q[:3]

    S = np.zeros((N, 3, 4))
    for i in range(N):
        e = eps[:, i].reshape((-1, 1))
        S[i, :, :] = np.hstack((2 * (eta[i] * np.eye(3) - SO3.wedge(e)), -2 * e))
                
    omega = (S @ np.expand_dims(q_dot, 2)).squeeze()
    return csaps(time, omega.T, smooth=0.9).spline

def get_accelerometer_splines(time: np.ndarray, pos_splines: ISmoothingSpline, quat_splines: ISmoothingSpline) -> ISmoothingSpline:
    """
    Get spline interpolations for accelerometer.
    
    Args:
    - time: Timestamps.
    - pos_splines: Spline interpolations for position.
    - quat_splines: Spline interpolations for orientation.
    
    Returns:
    - accel_splines: Spline interpolations for accelerometer.
    """
    gravity = np.array([0, 0, -9.80665])
    
    q = quat_splines(time)
    acceleration = np.atleast_2d(pos_splines.derivative(nu=2)(time)).T
    
    accelerometer = np.zeros((len(time), 3))
    for i in range(len(time)):
        R = Rotation.from_quat(q[:, i]).as_matrix()
        accelerometer[i] = R.T @ (acceleration[i] - gravity)
        
    return csaps(time, accelerometer.T, smooth=0.99).spline
        
def get_timeshift(exp_name):
    """
    Get timeshift.
    
    Args:
    - exp_name: Experiment name.
    
    Returns:
    - timeshift: Timeshift in seconds.
    """

    with open(f"data/{exp_name}/timeshift.yaml", "r") as file:
        timeshift = yaml.safe_load(file)
    timeshift_s = timeshift["timeshift_s"]
    timeshift_ns = timeshift["timeshift_ns"]

    return timeshift_s + timeshift_ns / 1e9

def get_imu_noise_params(robot_name, sensor_name):
    """
    Get IMU noise parameters that were generated using allan_variance_ros, available at
    https://github.com/ori-drs/allan_variance_ros. The noise parameters are stored in 
    the config/imu directory.
    
    Args:
    - robot_name: Robot name, e.g., "ifo001".
    - sensor_name: Sensor name, options are "px4" and "cam".
    
    Returns:
    dict
    - gyro: Gyroscope noise parameters.
    - accel: Accelerometer noise parameters.
    - gyro_bias: Gyroscope bias noise parameters.
    - accel_bias: Accelerometer bias noise parameters.
    """
    
    with open(f"config/imu/{robot_name}/{sensor_name}_output.log", "r") as file:
        imu_params = yaml.safe_load(file)
    
    gyro = np.array([
        eval(imu_params["X Angle Random Walk"].split(" ")[0]),
        eval(imu_params["Y Angle Random Walk"].split(" ")[0]),
        eval(imu_params["Z Angle Random Walk"].split(" ")[0])
    ]) * np.pi / 180
    accel = np.array([
        eval(imu_params["X Velocity Random Walk"].split(" ")[0]),
        eval(imu_params["Y Velocity Random Walk"].split(" ")[0]),
        eval(imu_params["Z Velocity Random Walk"].split(" ")[0])
    ])
    
    gyro_bias = np.array([
        eval(imu_params["X Rate Random Walk"].split(" ")[0]),
        eval(imu_params["Y Rate Random Walk"].split(" ")[0]),
        eval(imu_params["Z Rate Random Walk"].split(" ")[0])
    ]) * np.pi / 180
    accel_bias = np.array([
        eval(imu_params["X Accel Random Walk"].split(" ")[0]),
        eval(imu_params["Y Accel Random Walk"].split(" ")[0]),
        eval(imu_params["Z Accel Random Walk"].split(" ")[0])
    ])
    
    return {"gyro": gyro, "accel": accel, "gyro_bias": gyro_bias, "accel_bias": accel_bias}
    

def load_vins(exp_name, robot_id, loop = True, postprocessed: bool = False) -> pd.DataFrame:
    """
    Load VINS data.
    
    Args:
    - exp_name: Experiment name.
    - robot_id: Robot ID.
    - loop: Whether to load VINS data with loop closure or not.
    - postprocessed: Whether to load postprocessed (aligned and shifted) VINS data or not.
    
    Returns:
    - vins: VINS data.
    """
    
    if postprocessed:
        suffix = "_aligned_and_shifted"
    else:
        suffix = ""

    if loop:
        file = f"data/vins_results/{exp_name}/{robot_id}_vio_loop{suffix}.csv"
    else:
        file = f"data/vins_results/{exp_name}/{robot_id}_vio{suffix}.csv"

    data = pd.read_csv(
        file,
        names=[
            "timestamp",
            "pose.position.x",
            "pose.position.y",
            "pose.position.z",
            "pose.orientation.x",
            "pose.orientation.y",
            "pose.orientation.z",
            "pose.orientation.w",
            "twist.linear.x",
            "twist.linear.y",
            "twist.linear.z",
        ],
        index_col=False,
        header = (0 if postprocessed else None)
    )

    timeshift = get_timeshift(exp_name)
    if not postprocessed:
        data["timestamp"] = data["timestamp"] / 1e9 - timeshift

    return data


def save_vins(data: pd.DataFrame,
              exp_name: str,
              robot_id: str,
              loop: bool = True,
              postprocessed: bool = False):
    """
    Save VINS data.
    
    Args:
    - data: VINS data.
    - exp_name: Experiment name.
    - robot_id: Robot ID.
    - loop: Whether loop closure was enabled or not, only affects csv file name.
    - postprocessed: Whether the data is postprocessed or not, only affects csv file name.
    """
    
    if postprocessed:
        suffix = "_aligned_and_shifted"
    else:
        suffix = ""
    
    if loop:
        data.to_csv(f"data/vins_results/{exp_name}/{robot_id}_vio_loop{suffix}.csv",
                    index=False)
    else:
        data.to_csv(f"data/vins_results/{exp_name}/{robot_id}_vio{suffix}.csv",
                    index=False)


def align_frames(df1, df2):
    """
    Align inertial reference frames for two dataframes consisting of body-frame data. 
    The data in the first dataframe is resolved to the inertial reference frame of 
    the second dataframe. The data in the second dataframe is not modified. 
    
    Both dataframes must have measurements at the same timestamps and have the 
    following columns:
    - timestamp
    - pose.position.x
    - pose.position.y
    - pose.position.z
    - pose.orientation.x
    - pose.orientation.y
    - pose.orientation.z
    - pose.orientation.w
    
    Args:
    - df1: First dataframe.
    - df2: Second dataframe.
    
    Returns: dict
    - data: First dataframe with aligned data.
    - C: Rotation matrix from mocap frame to VINS frame.
    - r: Translation vector from mocap frame to VINS frame, resolved in the mocap frame.
    """
    pos1 = df1[[
        "pose.position.x",
        "pose.position.y",
        "pose.position.z",
    ]].values

    pos2 = df2[[
        "pose.position.x",
        "pose.position.y",
        "pose.position.z",
    ]].values

    y = pos1

    C_hat = np.eye(3)
    r_hat = np.zeros(3)

    # Levenberg-Marquardt optimization
    def error(y, C, r):
        return (y - (C @ (pos2 - r).T).T).flatten()

    def jacobian(C, r):
        J = np.empty((0, 6))
        for pos in pos2:
            J_iter = np.zeros((3, 6))
            J_iter[:, :3] = -C @ so3_wedge_matrix(pos - r)
            J_iter[:, 3:] = -C
            J = np.vstack((J, J_iter))
        return J

    del_x = np.ones(6)
    iter = 0
    e = error(y, C_hat, r_hat)
    while np.linalg.norm(del_x) > 1e-12 and iter < 100:
        J = jacobian(C_hat, r_hat)
        K = np.linalg.inv(J.T @ J + 1e-6 * np.eye(6)) @ J.T
        del_x = K @ e
        r_hat = r_hat + del_x[3:]
        C_hat = C_hat @ sp.linalg.expm(so3_wedge_matrix(del_x[:3]))
        iter += 1

        e = error(y, C_hat, r_hat)
        print("Iteration: ", iter)
        print("Error: ", e)
        print("Error norm: ", np.linalg.norm(e))
        print("Delta x: ", del_x)
        print("Delta x norm: ", np.linalg.norm(del_x))
        print("C_hat: ", C_hat)
        print("r_hat: ", r_hat)
        print("-------------------")

    # Apply transformation to df1
    df1 = apply_transformation(df1, C_hat, r_hat)

    return {"data": df1, "C": C_hat, "r": r_hat}


def apply_transformation(df, C, r):
    """
    Apply a transformation to a dataframe consisting of body-frame data.
    
    Args:
    - df: Dataframe.
    - C: Rotation matrix.
    - r: Translation vector.
    
    Returns:
    - df: Dataframe with transformed data.
    """

    pose = df[[
        "pose.position.x", "pose.position.y", "pose.position.z",
        "pose.orientation.x", "pose.orientation.y", "pose.orientation.z",
        "pose.orientation.w"
    ]].values

    df_r = pose[:, :3]
    df_quat = pose[:, 3:]
    df_C = np.array([
        Rotation.from_quat(df_quat[i]).as_matrix() for i in range(len(df_quat))
    ])

    pose = np.array([C.T @ df_r[i] + r for i in range(len(df_r))])
    df[[
        "pose.position.x",
        "pose.position.y",
        "pose.position.z",
    ]] = np.array([pose[i] for i in range(len(pose))])
    df[[
        "pose.orientation.x", "pose.orientation.y", "pose.orientation.z",
        "pose.orientation.w"
    ]] = np.array([
        Rotation.from_matrix(df_C[i].T @ C).as_quat() for i in range(len(df_C))
    ])

    if "twist.linear.x" in df.columns:
        df_vel = df[["twist.linear.x", "twist.linear.y",
                     "twist.linear.z"]].values
        vel = np.array([C.T @ df_vel[i] for i in range(len(df_vel))])
        df[["twist.linear.x", "twist.linear.y",
            "twist.linear.z"]] = np.array([vel[i] for i in range(len(vel))])

    return df


def so3_wedge_matrix(omega):
    """
    Create a 3x3 SO(3) wedge matrix from a 3x1 vector.
    
    Args:
    - omega: 3x1 vector.
    
    Returns:
    - omega_hat: 3x3 SO(3) cross matrix.
    """

    omega_hat = np.zeros((3, 3))
    omega_hat[0, 1] = -omega[2]
    omega_hat[0, 2] = omega[1]
    omega_hat[1, 0] = omega[2]
    omega_hat[1, 2] = -omega[0]
    omega_hat[2, 0] = -omega[1]
    omega_hat[2, 1] = omega[0]

    return omega_hat


def compute_position_rmse(df1, df2):
    """
    Compute the root mean squared error (RMSE) between two dataframes consisting of 
    position data.
    
    Args:
    - df1: First dataframe.
    - df2: Second dataframe.
    
    Returns:
    - rmse: RMSE.
    """

    pos1 = df1[["pose.position.x", "pose.position.y",
                "pose.position.z"]].values

    pos2 = df2[["pose.position.x", "pose.position.y",
                "pose.position.z"]].values

    return np.sqrt(np.mean(np.linalg.norm(pos1 - pos2, axis=1)**2))
