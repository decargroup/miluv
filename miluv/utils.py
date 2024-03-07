import numpy as np
from csaps import csaps
import pandas as pd

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
    