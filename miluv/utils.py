import numpy as np
from csaps import csaps
import pandas as pd
from scipy.spatial.transform import Rotation
import scipy as sp
import yaml
import matplotlib.pyplot as plt

def zero_order_hold(query_timestamps, data):
    """
    Zero-order hold interpolation for data.
    
    Args:
    - query_timestamps: Query timestamps.
    - data: Data to interpolate.
    
    Returns:
    - data: Interpolated data.
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

    return data

def get_mocap_splines(mocap: pd.DataFrame) -> 'tuple[callable, callable]':
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

    # Normalize quaternion
    quat /= np.linalg.norm(quat, axis=1)[:, None]

    # Resolve quaternion discontinuities
    for i in range(1, len(quat)):
        if np.dot(quat[i], quat[i - 1]) < 0:
            quat[i] *= -1

    # Fit splines
    pos_splines = csaps(time, pos.T, smooth=0.9999)
    quat_splines = csaps(time, quat.T, smooth=0.9999)

    return pos_splines, quat_splines


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
