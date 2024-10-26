import numpy as np
from pymlg import SO3, SE3, SE23

def get_se3_poses(quat: np.ndarray, pos: np.ndarray) -> list[SE3]:
    """
    Get SE3 poses from position and quaternion data.
    
    Args:
    - quat: Quaternion data.
    - pos: Position data.
    
    Returns:
    - SE3 poses.
    """
    
    poses = []
    for i in range(pos.shape[1]):
        R = SO3.from_quat(quat[:, i], "xyzw")
        poses.append(SE3.from_components(R, pos[:, i]))
    return poses

def get_se23_poses(quat: np.ndarray, vel: np.ndarray, pos: np.ndarray) -> list[SE23]:
    """
    Get SE23 poses from position, velocity, and quaternion data.
    
    Args:
    - quat: Quaternion data.
    - vel: Velocity data.
    - pos: Position data.
    
    Returns:
    - SE23 poses.
    """
    
    poses = []
    for i in range(pos.shape[1]):
        R = SO3.from_quat(quat[:, i], "xyzw")
        poses.append(SE23.from_components(R, vel[:, i], pos[:, i]))
    return poses
