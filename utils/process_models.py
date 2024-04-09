import numpy as np
from typing import List
from utils.states import (
    IMUState,
    CompositeState)
from scipy.linalg import block_diag
import copy
from utils.misc import (
    get_unbiased_imu, 
    G_matrix, U_matrix, L_matrix )

class Input:
    """
    Data container for an Input.
    """
    def __init__(self, stamp: float, gyro: np.ndarray = None, 
        accel: np.ndarray = None, vio: np.ndarray = None, state_id = None, 
        covariance: np.ndarray = None,):
        
        self.stamp = stamp
        self.state_id = state_id
        self.covariance = covariance
        self.gyro = np.array(gyro).ravel()
        self.accel = np.array(accel).ravel()
        self.vio = np.array(vio).ravel()

    def copy(self):
        return copy.deepcopy(self)

class ProcessModel:
    
    def evaluate(self, x: CompositeState, u, dt):
        pass

    def jacobian(self, x: CompositeState, u, dt, step_size=1e-6):
        """
        Calculates the model jacobian with finite difference.
        """
        Y_bar = self.evaluate(x.copy(), u, dt)
        jac_fd = np.zeros((x.dof, x.dof))
        for i in range(x.dof):
            dx = np.zeros((x.dof))
            dx[i] = step_size
            x_pert = x.plus(dx)
            Y = self.evaluate(x_pert, u, dt)
            jac_fd[:, i] = Y.minus(Y_bar).flatten() / step_size
        return jac_fd
    
    def evaluate_with_jacobian(self, x: CompositeState, u, dt):
        return self.evaluate(x, u, dt), self.jacobian(x, u, dt)

class BodyFrameVelocity(ProcessModel):
    """
    The body-frame velocity process model assumes that the input contains
    both translational and angular velocity measurements, both relative to
    a local reference frame, but resolved in the robot body frame.
    """
    def __init__(self, Q: np.ndarray):
        self._Q = Q

    def evaluate(self, x, u: Input, dt: float):
        u = u.copy()
        u = np.hstack((u.gyro, u.vio)).ravel()
        x = x.copy()
        x.value = x.value @ x.group.Exp(u * dt)
        return x
        
    def covariance(self, x, u: Input, dt: float) -> np.ndarray:
        u = u.copy()
        u = np.hstack((u.gyro, u.vio)).ravel()
        L = dt * x.group.left_jacobian(-u * dt)
        return L @ self._Q @ L.T

class IMUKinematics(ProcessModel):

    def __init__(self, Q: np.ndarray):
        self._Q = Q
        self._gravity = np.array([0, 0, -9.80665]).ravel()

    def evaluate(self, x: IMUState, u, dt: float) -> IMUState:
        x = x.copy()
        u_no_bias = get_unbiased_imu(x, u) # Get unbiased inputs
        G = G_matrix(self._gravity, dt)
        U = U_matrix(u_no_bias.gyro, u_no_bias.accel, dt)
        x.value[0].value = G @ x.value[0].value @ U
        return x

    def covariance(self, x: IMUState, u, dt: float) -> np.ndarray:
        u_no_bias = get_unbiased_imu(x, u)
        L_pn = L_matrix(u_no_bias.gyro, u_no_bias.accel, dt)  # Jacobian of pose wrt to noise
        L_bn = np.zeros((6, 6)) # Jacobian of bias random walk wrt to noise
        L_pw = np.zeros((9, 6)) # Jacobian of pose wrt to bias random walk
        L_bw = dt * np.identity(6) # Jacobian of bias wrt to bias random walk
        L = np.block([[L_pn, L_pw], [L_bn, L_bw]])
        return L @ self._Q @ L.T

class CompositeProcessModel(ProcessModel):

    def __init__(self, model_list: List,):
        self._model_list = model_list

    def evaluate(self, x: CompositeState, u: List, dt: float,):
        x = x.copy()
        x.value = [self._model_list[i].evaluate(x_sub, 
                    u[i], dt) for i, x_sub in enumerate(x.value)]
        return x

    def covariance(self, x: CompositeState, u: List, dt: float):
        cov = [self._model_list[i].covariance(x_sub, u[i], dt)
               for i, x_sub in enumerate(x.value)]
        return block_diag(*cov)