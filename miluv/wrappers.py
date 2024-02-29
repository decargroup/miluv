from typing import List, Any
import numpy as np
from dataclasses import dataclass
from abc import ABC
from csaps import csaps
from pymlg import SO3
import pandas as pd


""" 
This module contains data wrappers for:
- MocapTrajectory
"""

def tags_to_df(anchors: Any = None, 
               moment_arms: Any = None, 
               april_tags: Any = None) -> pd.DataFrame:

    uwb_tags = None
    april_tags = None

    if anchors is not None:
        tags = []
        for key, value in anchors.items():
            # parse value
            value = value.strip('[]').split(',')
            tags.append({
                'tag_id': key,
                'parent_id': 'world',
                'position.x': value[0],
                'position.y': value[1],
                'position.z': value[2],
            })
        # Convert the data dictionary to a DataFrame
        uwb_tags = pd.DataFrame(tags)
    
    if moment_arms is not None:
        tags = []
        for robot, arms in moment_arms.items():
            for tag, value in arms.items():
                value = value.strip('[]').split(',')
                tags.append({
                    'tag_id': tag,
                    'parent_id': robot,
                    'position.x': value[0],
                    'position.y': value[1],
                    'position.z': value[2],
                })
        tags = pd.DataFrame(tags)
    if uwb_tags is not None:
        uwb_tags = pd.concat([uwb_tags, tags], 
                                    ignore_index=True)
    else:
        uwb_tags = tags
    
    if april_tags is not None:
        tags = []
        for key, value in april_tags.items():
            # parse value
            value = value.strip('[]').split(',')
            tags.append({
                'tag_id': key,
                'parent_id': 'world',
                'position.x': value[0],
                'position.y': value[1],
                'position.z': value[2],
            })
        # Convert the data dictionary to a DataFrame
        april_tags = pd.DataFrame(tags)
    return uwb_tags, april_tags

def beye(dim: int, num: int) -> np.ndarray:
    """
    Batch identity matrix
    """
    return np.tile(np.eye(dim), (num, 1, 1))

def bwedge_so3(phi: np.ndarray) -> np.ndarray:
    """
    Batch wedge for SO(3). phi is provided as a [N x 3] ndarray
    """

    if phi.shape[1] != 3:
        raise ValueError("phi must have shape ({},) or (N,{})".format(3, 3))

    Xi = np.zeros((phi.shape[0], 3, 3))
    Xi[:, 0, 1] = -phi[:, 2]
    Xi[:, 0, 2] = phi[:, 1]
    Xi[:, 1, 0] = phi[:, 2]
    Xi[:, 1, 2] = -phi[:, 0]
    Xi[:, 2, 0] = -phi[:, 1]
    Xi[:, 2, 1] = phi[:, 0]

    return Xi

def bquat_to_so3(quat: np.ndarray, ordering="wxyz"):
    """
    Form a rotation matrix from a unit length quaternion.
    Valid orderings are 'xyzw' and 'wxyz'.
    """

    if not np.allclose(np.linalg.norm(quat, axis=1), 1.0):
        raise ValueError("Quaternions must be unit length")

    if ordering == "wxyz":
        eta = quat[:, 0]
        eps = quat[:, 1:]
    elif ordering == "xyzw":
        eta = quat[:, 3]
        eps = quat[:, 0:3]
    else:
        raise ValueError("order must be 'wxyz' or 'xyzw'. ")
    eta = eta.reshape((-1, 1, 1))
    eps = eps.reshape((-1, 3, 1))
    eps_T = np.transpose(eps, [0, 2, 1])
    return (
        (1 - 2 * eps_T @ eps) * beye(3, quat.shape[0])
        + 2 * eps @ eps_T
        + 2 * eta * bwedge_so3(eps.squeeze())
    )

class MocapTrajectory:
    """
    This class holds a mocap dataset and provides several convient getters.
    A smoothing spline is fit through the position and attitude data, giving access
    via its derivatives to estimated velocity and acceleration data. Furthermore,
    the spline can be queried at any point so that ground truth becomes available
    at any time.
    """

    def __init__(
        self,
        mocap: pd.DataFrame,
        frame_id: Any = None,
    ):
        """
        Parameters
        ----------
        stamps : np.ndarray with shape (N,)
            Timestamps of the data
        position_data : np.ndarray with shape (N, 3)
            Position data where each row is a 3D position
        quaternion_data : np.ndarray with shape (N, 4)
            Attitude data where each row is a quaternion with `wxyz` ordering
        frame_id : Any
            Optional frame ID to assign to this data. Will be used as the state
            ID when converting to ``pynav`` states.
        """

        self.stamps = np.array(mocap["timestamp"]).ravel()
        self.raw_position = np.array(mocap[["pose.position.x", 
                                            "pose.position.y", 
                                            "pose.position.z"]])
        self.raw_quaternion = np.array(mocap[["pose.orientation.w", 
                                              "pose.orientation.x", 
                                              "pose.orientation.y", 
                                              "pose.orientation.z"]])
        self.frame_id = frame_id

        self._fit_position_spline(self.stamps, self.raw_position)
        self._fit_quaternion_spline(self.stamps, self.raw_quaternion)
    
    def __repr__(self):
        return f"MocapTrajectory, frame_id: {self.frame_id}"


    def _fit_position_spline(self, stamps, pos):

        # First, filter out positions with zero norm
        # We assume that this is gaps in the mocap data
        is_valid = np.linalg.norm(pos, axis=1) > 1e-6
        stamps = stamps[is_valid]
        pos = pos[is_valid]

        # Fit spline
        self._pos_spline = csaps(stamps, pos.T, smooth=0.9999)

    def _fit_quaternion_spline(self, stamps, quat):
        # First, filter out invalid quaternions with zero norm
        # We assume that this is gaps in the mocap data
        is_valid = np.linalg.norm(quat, axis=1) > 1e-6
        stamps = stamps[is_valid]
        quat = quat[is_valid]


        # Normalize quaternion
        quat /= np.linalg.norm(quat, axis=1)[:, None]

        # Resolve quaternion ambiguities so that quaternion trajectories look
        # smooth. This is recursive so cannot be vectorized.
        for idx, q in enumerate(quat[1:]):
            q_old = quat[idx]
            if np.linalg.norm((-q - q_old)) < np.linalg.norm((q - q_old)):
                q *= -1

        self._quat_spline = csaps(stamps, quat.T, smooth=0.99999)

    def position(self, stamps: np.ndarray) -> np.ndarray:
        """
        Get the position at one or more query times.

        Parameters
        ----------
        stamps : np.ndarray
            Query times

        Returns
        -------
        np.ndarray with shape (N,3)
            Position data
        """
        return self._pos_spline(stamps, 0).T

    def velocity(self, stamps: np.ndarray) -> np.ndarray:
        """
        Get the velocity at one or more query times.

        Parameters
        ----------
        stamps : np.ndarray
            Query times

        Returns
        -------
        np.ndarray with shape (N,3)
            velocity data
        """
        return self._pos_spline(stamps, 1).T

    def acceleration(self, stamps: np.ndarray) -> np.ndarray:
        """
        Get the acceleration at one or more query times.

        Parameters
        ----------
        stamps : np.ndarray
            Query times

        Returns
        -------
        np.ndarray with shape (N,3)
            acceleration data
        """
        return self._pos_spline(stamps, 2).T

    def accelerometer(self, stamps: np.ndarray, g_a=None) -> np.ndarray:
        """
        Get simuluated accelerometer readings

        Parameters
        ----------
        stamps : float or np.ndarray
            query times
        g_a : List[float], optional
            gravity vector, by default [0, 0, -9.80665]

        Returns
        -------
        ndarray with shape `(len(stamps),3)`
            Accelerometer readings
        """
        if g_a is None:
            g_a = [0, 0, -9.80665]

        a_zwa_a = self._pos_spline(stamps, 2).T
        C_ab = self.rot_matrix(stamps)
        C_ba = np.transpose(C_ab, axes=[0, 2, 1])
        g_a = np.array(g_a).reshape((-1, 1))
        return (C_ba @ (np.expand_dims(a_zwa_a, 2) - g_a)).squeeze()

    def quaternion(self, stamps):
        """
        Get the quaternion at one or more query times.

        Parameters
        ----------
        stamps : np.ndarray
            Query times

        Returns
        -------
        np.ndarray with shape (N,4)
            quaternion data
        """
        q = self._quat_spline(stamps, 0).T
        return q / np.linalg.norm(q, axis=1)[:, None]

    def rot_matrix(self, stamps):
        """
        Get the DCM/rotation matrix at one or more query times.

        Parameters
        ----------
        stamps : np.ndarray
            Query times

        Returns
        -------
        np.ndarray with shape (N,3,3)
            DCM/rotation matrix data
        """

        quat = self.quaternion(stamps)
        return bquat_to_so3(quat)

    def pose_matrix(self, stamps):
        """
        Get the SE(3) pose matrix at one or more query times.

        Parameters
        ----------
        stamps : np.ndarray
            Query times

        Returns
        -------
        np.ndarray with shape (N,4,4)
            pose data
        """
        r = self.position(stamps)
        C = self.rot_matrix(stamps)
        T = np.zeros((C.shape[0], 4, 4))
        T[:, :3, :3] = C
        T[:, :3, 3] = r
        T[:, 3, 3] = 1
        return T

    def extended_pose_matrix(self, stamps):
        """
        Get the SE_2(3) extended pose matrix at one or more query times.

        Parameters
        ----------
        stamps : np.ndarray
            Query times

        Returns
        -------
        np.ndarray with shape (N,5,5)
            extended pose data
        """
        r = self.position(stamps)
        C = self.rot_matrix(stamps)
        v = self.velocity(stamps)
        T = np.zeros((C.shape[0], 5, 5))
        T[:, :3, :3] = C
        T[:, :3, 3] = v
        T[:, :3, 4] = r
        T[:, 3, 3] = 1
        T[:, 4, 4] = 1
        return T

    def angular_velocity(self, stamps: np.ndarray) -> np.ndarray:
        """
        Get the angular velocity at one or more query times.

        Parameters
        ----------
        stamps : np.ndarray
            Query times

        Returns
        -------
        np.ndarray with shape (N,3)
            angular velocity data
        """
        q = self._quat_spline(stamps, 0)
        q: np.ndarray = q / np.linalg.norm(q, axis=0)
        N = q.shape[1]
        q_dot = np.atleast_2d(self._quat_spline(stamps, 1)).T
        eta = q[0]
        eps = q[1:]

        # TODO: this loop can be vectorized.
        S = np.zeros((N, 3, 4))
        for i in range(N):
            e = eps[:, i].reshape((-1, 1))
            S[i, :, :] = np.hstack((-2 * e, 2 * (eta[i] * np.eye(3) - SO3.wedge(e))))

        omega = (S @ np.expand_dims(q_dot, 2)).squeeze()
        return omega

    def body_velocity(self, stamps: np.ndarray) -> np.ndarray:
        """
        Get the body-frame-resolved translational velocity
        at one or more query times.

        Parameters
        ----------
        stamps : np.ndarray
            Query times

        Returns
        -------
        np.ndarray with shape (N,3)
            body-frame-resolved velocity data
        """
        v_zw_a = self.velocity(stamps)
        v_zw_a = np.expand_dims(v_zw_a, 2)
        C_ab = self.rot_matrix(stamps)
        C_ba = np.transpose(C_ab, axes=[0, 2, 1])
        return (C_ba @ v_zw_a).squeeze().T