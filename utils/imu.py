from pymlg import SO3, SE23
import numpy as np
from typing import List
from utils.states import (
    IMUState,
    CompositeState)
from scipy.linalg import block_diag
import copy

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

def get_unbiased_imu(x: IMUState, u):
    u = u.copy()
    u.gyro = u.gyro.ravel() - x.value[1].value.ravel()
    u.accel = u.accel.ravel() - x.value[2].value.ravel()
    return u

def U_matrix(omega, accel, dt):
    phi = omega * dt
    a = accel.reshape((-1, 1))
    U = np.identity(5)
    U[:3, :3] = SO3.Exp(phi)
    U[:3, 3] = np.ravel(dt * SO3.left_jacobian(phi) @ a)
    U[:3, 4] = np.ravel(dt**2 / 2 * SO3.wedge(a) @ a)
    U[3, 4] = dt
    return U

def G_matrix(gravity, dt):
    G = np.identity(5)
    G[:3, 3] = dt * gravity
    G[:3, 4] = -0.5 * dt**2 * gravity
    G[3, 4] = -dt
    return G

def L_matrix(unbiased_gyro, unbiased_accel, dt):
    """
    Computes the jacobian of the SE23_state with respect to the input.
    """
    a = unbiased_accel
    om = unbiased_gyro
    omdt = om * dt
    J_att_inv_times_N = SO3.left_jacobian_inv(omdt) @ SO3.wedge(omdt)
    xi = np.zeros((9,))
    xi[:3] = dt * om
    xi[3:6] = dt * a
    xi[6:9] = (dt**2 / 2) * J_att_inv_times_N @ a
    J = SE23.left_jacobian(-xi)
    Om = SO3.wedge(omdt)
    OmOm = Om @ Om
    A = SO3.wedge(a)
    Up = dt * np.eye(9, 6)
    Up[6:9, 0:3] = (-0.5 * (dt**2 / 2) * ((1 / 360) * (dt**3)
            * (OmOm @ A + Om @ (SO3.wedge(Om @ a)) + SO3.wedge(OmOm @ a))
            - (1 / 6) * dt * A))
    Up[6:9, 3:6] = (dt**2 / 2) * J_att_inv_times_N
    L = J @ Up
    return L

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