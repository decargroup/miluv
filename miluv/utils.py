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
        ["pose.orientation.w", "pose.orientation.x", "pose.orientation.y", "pose.orientation.z"]
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
    
    # Generate transformation matrices
    pose1 = df1[[
        "pose.position.x", "pose.position.y", "pose.position.z", 
        "pose.orientation.x", "pose.orientation.y", "pose.orientation.z", "pose.orientation.w"
    ]].values
    T1 = np.array([se3_pose(pose1[i, :3], pose1[i, 3:]) for i in range(len(pose1))])
    
    pose2 = df2[[
        "pose.position.x", "pose.position.y", "pose.position.z",
        "pose.orientation.x", "pose.orientation.y", "pose.orientation.z", "pose.orientation.w"
    ]].values
    T2 = np.array([se3_pose(pose2[i, :3], pose2[i, 3:]) for i in range(len(pose2))])
    
    Y = T1 @ np.linalg.inv(T2)
    T_hat = Y[0]
    
    num_meas = np.shape(Y)[0]
    
    # Levenberg-Marquardt optimization
    def error(Y, T):
        y_wedge = np.array(
            [sp.linalg.logm(T @ np.linalg.inv(Y_i)) for Y_i in Y]
        )
        return np.array([se3_vee(y_wedge[i]) for i in range(len(y_wedge))]).flatten()
    
    def jacobian():
        x = np.eye(6)
        return np.array(list(x)*num_meas)
    
    del_x = np.ones(6)*1e6
    iter = 0
    J = jacobian()
    print(J)
    K = np.linalg.inv(J.T @ J + 0* 1e-6 * np.eye(6)) @ J.T
    print(K)
    e = error(Y, T_hat)
    while np.linalg.norm(del_x) > 1e-6 and iter < 100:
        del_x = K @ e
        del_x_wedge = se3_wedge_matrix(del_x)
        T_hat =  T_hat @ sp.linalg.expm(del_x_wedge)
        iter += 1
        
        e = error(Y, T_hat)
        print("Iteration: ", iter)
        print("Error: ", e)
        print("Error norm: ", np.linalg.norm(e))
        print("Delta x: ", del_x)
        print("Delta x norm: ", np.linalg.norm(del_x))
        print("T_hat: ", T_hat)
        print("-------------------")
            
    # Apply transformation to df1
    pose1 = np.array([se3_pose(pose1[i, :3], pose1[i, 3:]) for i in range(len(pose1))])
    pose1 = np.array([np.linalg.inv(T_hat) @ pose1[i] for i in range(len(pose1))])
    df1[[
        "pose.position.x", "pose.position.y", "pose.position.z",
    ]] = np.array([pose1[i][:3, 3] for i in range(len(pose1))])
    df1[[
        "pose.orientation.x", "pose.orientation.y", "pose.orientation.z", "pose.orientation.w"
    ]] = np.array([Rotation.from_matrix(pose1[i][:3, :3]).as_quat() for i in range(len(pose1))])
    
    # # Plot measurements Y over time
    # fig, axs = plt.subplots(3, 1)
    # y = np.array([se3_vee(sp.linalg.logm(np.linalg.inv(T1[i]) @ T2[i])) for i in range(len(T1))])
    # axs[0].plot(df1["timestamp"], y[:, 0], label="x")
    # axs[0].plot(df1["timestamp"], y[:, 1], label="y")
    # axs[0].plot(df1["timestamp"], y[:, 2], label="z")
    # axs[0].set_title("Orientation error")
    # axs[0].legend()
    # axs[1].plot(df1["timestamp"], y[:, 3], label="x")
    # axs[1].plot(df1["timestamp"], y[:, 4], label="y")
    # axs[1].plot(df1["timestamp"], y[:, 5], label="z")
    # axs[1].set_title("Position error")
    # axs[1].legend()
    # axs[2].plot(df1["timestamp"], np.linalg.norm(y, axis=1))
    # axs[2].set_title("Error norm")
    # plt.show()
    
    return df1
    
    
def se3_pose(pos, quat):
    """
    Create a 4x4 SE(3) pose matrix from position and quaternion.
    
    Args:
    - pos: Position.
    - quat: Quaternion, [x, y, z, w].
    
    Returns:
    - pose: SE(3) pose matrix.
    """
    
    pose = np.eye(4)
    pose[:3, 3] = pos
    pose[:3, :3] = Rotation.from_quat(quat).as_matrix()
    
    return pose
    
def se3_wedge_matrix(omega):
    """
    Create a 4x4 SE(3) wedge matrix from a 6x1 twist vector.
    
    Args:
    - omega: 6x1 twist vector.
    
    Returns:
    - omega_hat: 4x4 SE(3) wedge matrix.
    """
    
    omega_hat = np.zeros((4, 4))
    omega_hat[:3, :3] = so3_wedge_matrix(omega[:3])
    omega_hat[:3, 3] = omega[3:]
    
    return omega_hat

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

def se3_vee(omega_hat):
    """
    Create a 6x1 twist vector from a 4x4 SE(3) wedge matrix.
    
    Args:
    - omega_hat: 4x4 SE(3) wedge matrix.
    
    Returns:
    - omega: 6x1 twist vector.
    """
    
    omega = np.zeros(6)
    omega[:3] = so3_vee(omega_hat[:3, :3])
    omega[3:] = omega_hat[:3, 3]
    
    return omega

def so3_vee(omega_hat):
    """
    Create a 3x1 vector from a 3x3 SO(3) wedge matrix.
    
    Args:
    - omega_hat: 3x3 SO(3) cross matrix.
    
    Returns:
    - omega: 3x1 vector.
    """
    
    omega = np.zeros(3)
    omega[0] = omega_hat[2, 1]
    omega[1] = omega_hat[0, 2]
    omega[2] = omega_hat[1, 0]
    
    return omega