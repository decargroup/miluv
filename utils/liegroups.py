import numpy as np
from pymlg import SO3, SE3

def get_se3_poses(pos: np.ndarray, quat: np.ndarray) -> list[SE3]:
    """
    Get SE3 poses from position and quaternion data.
    
    Args:
    - pos: Position data.
    - quat: Quaternion data.
    
    Returns:
    - SE3 poses.
    """
    
    se3 = []
    for i in range(pos.shape[1]):
        R = SO3.from_quat(quat[:, i], "xyzw")
        se3.append(SE3.from_components(R, pos[:, i]))
    return se3
