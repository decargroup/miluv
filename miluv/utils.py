import numpy as np
from csaps import csaps
import pandas as pd
from scipy.spatial.transform import Rotation
import scipy as sp
import yaml
import matplotlib.pyplot as plt

def get_mocap_splines(mocap: pd.DataFrame) -> tuple[callable, callable]:
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
    pos = mocap[
        ["pose.position.x", "pose.position.y", "pose.position.z"]
    ].values
    quat = mocap[
        ["pose.orientation.x", "pose.orientation.y", "pose.orientation.z", "pose.orientation.w"]
    ].values
    
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
        if np.dot(quat[i], quat[i-1]) < 0:
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

def load_vins(exp_name, robot_id):
    """
    Load VINS data.
    
    Args:
    - exp_name: Experiment name.
    - robot_id: Robot ID.
    
    Returns:
    - vins: VINS data.
    """
    
    data = pd.read_csv(
        f"data/vins/{exp_name}/{robot_id}_vio_loop.csv", 
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
        index_col=False
    )
    
    timeshift = get_timeshift(exp_name)
    data["timestamp"] = data["timestamp"]/1e9 - timeshift
    
    return data
    
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
    
    Returns:
    - df1: First dataframe resolved to the reference frame of the second dataframe.
    """
    pos1 = df1[[
        "pose.position.x", "pose.position.y", "pose.position.z", 
    ]].values
    
    pos2 = df2[[
        "pose.position.x", "pose.position.y", "pose.position.z",
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
    pose1 = df1[[
        "pose.position.x", "pose.position.y", "pose.position.z",
        "pose.orientation.x", "pose.orientation.y", "pose.orientation.z", "pose.orientation.w"
    ]].values

    vins_r = pose1[:, :3]
    vins_quat = pose1[:, 3:]
    vins_C = np.array([Rotation.from_quat(vins_quat[i]).as_matrix() for i in range(len(vins_quat))])

    pose1 = np.array([C_hat.T @ vins_r[i] + r_hat for i in range(len(vins_r))])
    df1[[
        "pose.position.x", "pose.position.y", "pose.position.z",
    ]] = np.array([pose1[i] for i in range(len(pose1))])    
    df1[[
        "pose.orientation.x", "pose.orientation.y", "pose.orientation.z", "pose.orientation.w"
    ]] = np.array([Rotation.from_matrix(vins_C[i] @ C_hat).as_quat() for i in range(len(vins_C))])
    
    return df1

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