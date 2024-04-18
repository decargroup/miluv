from typing import Any
import numpy as np
from csaps import csaps
from pymlg import SO3
import pandas as pd
from scipy.spatial.transform import Rotation

""" 
This module contains data wrappers for:
- MocapTrajectory
"""
class MocapTrajectory:
    """
    This class provides a spline representation 
    of a trajectory from mocap data.
    """

    def __init__(
        self,
        mocap: pd.DataFrame,
        frame_id: Any = None,
    ):
        self.stamps = np.array(mocap["timestamp"]).ravel()
        self.raw_position = mocap[["pose.position.x", 
                                "pose.position.y", 
                                "pose.position.z"]].to_numpy()
        self.raw_quaternion = mocap[["pose.orientation.w", 
                                    "pose.orientation.x", 
                                    "pose.orientation.y", 
                                    "pose.orientation.z"]].to_numpy()
        self.frame_id = frame_id
        self._fit_position_spline(self.stamps, self.raw_position)
        self._fit_quaternion_spline(self.stamps, self.raw_quaternion)

    def _fit_position_spline(self, stamps, pos):
        # Filter out positions with zero norm
        is_valid = np.linalg.norm(pos, axis=1) > 1e-6
        stamps = stamps[is_valid]
        pos = pos[is_valid]
        self._pos_spline = csaps(stamps, pos.T, smooth=0.9999)

    def _fit_quaternion_spline(self, stamps, quat):
        # Filter out invalid quaternions with zero norm
        is_valid = np.linalg.norm(quat, axis=1) > 1e-6
        stamps = stamps[is_valid]
        quat = quat[is_valid]
        quat /= np.linalg.norm(quat, axis=1)[:, None]
        
        # Resolve quaternion ambiguities so that quaternion trajectories look
        # smooth.
        for idx, q in enumerate(quat[1:]):
            q_old = quat[idx]
            if np.linalg.norm((-q - q_old)) < np.linalg.norm((q - q_old)):
                q *= -1
        self._quat_spline = csaps(stamps, quat.T, smooth=0.99999)

    def _rot_matrix(self, stamps):
        quat = self._quat_spline(stamps, 0).T
        quat = quat / np.linalg.norm(quat, axis=1)[:, None]
        return np.array([Rotation.from_quat(
                        [q[1],q[2],q[3],q[0]]).as_matrix() 
                        for q in quat])

    def SE3_state(self, stamps):
        # Get the SE(3) pose matrix at one or more query times.
        r = self._pos_spline(stamps, 0).T
        C = self._rot_matrix(stamps)
        T = np.zeros((C.shape[0], 4, 4))
        T[:, :3, :3] = C
        T[:, :3, 3] = r
        T[:, 3, 3] = 1
        return T

    def SE23_state(self, stamps):
        # Get the SE_2(3) extended pose matrix at one or more query times.
        r = self._pos_spline(stamps, 0).T
        C = self._rot_matrix(stamps)
        v = self._pos_spline(stamps, 1).T
        T = np.zeros((C.shape[0], 5, 5))
        T[:, :3, :3] = C
        T[:, :3, 3] = v
        T[:, :3, 4] = r
        T[:, 3, 3] = 1
        T[:, 4, 4] = 1
        return T