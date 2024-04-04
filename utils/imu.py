from pymlg import SO3, SE23
import numpy as np
from typing import Any
from utils.states import (
    MatrixLieGroupState, 
    VectorState, 
    CompositeState)
import copy

class IMU:
    """
    Data container for an IMU reading.
    """
    def __init__(
        self, gyro: np.ndarray, accel: np.ndarray, stamp: float, 
        state_id: Any = None, covariance: np.ndarray = None, ):
        self.dof = 12
        self.stamp = stamp
        self.state_id = state_id
        self.covariance = covariance
        self.gyro = np.array(gyro).ravel()
        self.accel = np.array(accel).ravel()

    def plus(self, w: np.ndarray):
        new = self.copy()
        w = w.ravel()
        new.gyro = new.gyro + w[0:3]
        new.accel = new.accel + w[3:6]
        new.bias_gyro_walk = new.bias_gyro_walk + w[6:9]
        new.bias_accel_walk = new.bias_accel_walk + w[9:12]
        return new

    def copy(self):
        return copy.deepcopy(self)

class IMUState(CompositeState):
    def __init__(self, SE23_state: np.ndarray, bias_gyro: np.ndarray,
        bias_accel: np.ndarray, stamp: float = None, state_id = None,
        direction="right"):

        SE23_state = MatrixLieGroupState(SE23_state, stamp, "pose", direction)
        bias_gyro = VectorState(bias_gyro, stamp, "gyro_bias")
        bias_accel = VectorState(bias_accel, stamp, "accel_bias")
        state_list = [SE23_state, bias_gyro, bias_accel]
        super().__init__(state_list, stamp, state_id)
        self.direction = direction

    def copy(self):
        return copy.deepcopy(self)

    def jacobian_from_blocks( self, attitude = None, position = None, 
        velocity = None, bias_gyro = None, bias_accel = None,):

        for jac in [attitude, position, velocity, bias_gyro, bias_accel]:
            if jac is not None:
                dim = jac.shape[0]
                break
        SE23_state_jac = self.value[0].jacobian_from_blocks(
            attitude=attitude, position=position, velocity=velocity)
        if bias_gyro is None:
            bias_gyro = np.zeros((dim, 3))
        if bias_accel is None:
            bias_accel = np.zeros((dim, 3))

        return np.hstack([SE23_state_jac, bias_gyro, bias_accel])

def get_unbiased_imu(x: IMUState, u: IMU) -> IMU:
    u = u.copy()
    u.gyro = u.gyro.ravel() - x.value[1].value.ravel()
    u.accel = u.accel.ravel() - x.value[2].value.ravel()
    return u

def N_matrix(phi_vec):
    if np.linalg.norm(phi_vec) < SO3._small_angle_tol:
        return np.identity(3)
    else:
        phi = np.linalg.norm(phi_vec)
        a = phi_vec / phi
        a = a.reshape((-1, 1))
        a_wedge = SO3.wedge(a)
        c = (1 - np.cos(phi)) / phi**2
        s = (phi - np.sin(phi)) / phi**2
        N = 2 * c * np.identity(3) + (1 - 2 * c) * (a @ a.T) + 2 * s * a_wedge
        return N

def adjoint_IE3(X):
    """
    Adjoint matrix of the "Incremental Euclidean Group".
    """
    R = X[:3, :3]
    c = X[3, 4]
    a = X[:3, 3].reshape((-1, 1))
    b = X[:3, 4].reshape((-1, 1))
    Ad = np.zeros((9, 9))
    Ad[:3, :3] = R
    Ad[3:6, :3] = SO3.wedge(a) @ R
    Ad[3:6, 3:6] = R
    Ad[6:9, :3] = -SO3.wedge(c * a - b) @ R
    Ad[6:9, 3:6] = -c * R
    Ad[6:9, 6:9] = R
    return Ad

def inverse_IE3(X):
    """
    Inverse matrix on the "Incremental Euclidean Group".
    """
    R = X[:3, :3]
    c = X[3, 4]
    a = X[:3, 3].reshape((-1, 1))
    b = X[:3, 4].reshape((-1, 1))
    X_inv = np.identity(5)
    X_inv[:3, :3] = R.T
    X_inv[:3, 3] = np.ravel(-R.T @ a)
    X_inv[:3, 4] = np.ravel(R.T @ (c * a - b))
    X_inv[3, 4] = np.ravel(-c)
    return X_inv

def U_matrix(omega, accel, dt):
    phi = omega * dt
    O = SO3.Exp(phi)
    J = SO3.left_jacobian(phi)
    a = accel.reshape((-1, 1))
    V = N_matrix(phi)
    U = np.identity(5)
    U[:3, :3] = O
    U[:3, 3] = np.ravel(dt * J @ a)
    U[:3, 4] = np.ravel(dt**2 / 2 * V @ a)
    U[3, 4] = dt
    return U

def U_matrix_inv(omega, accel, dt):
    return inverse_IE3(U_matrix(omega, accel, dt))

def G_matrix(gravity, dt):
    G = np.identity(5)
    G[:3, 3] = dt * gravity
    G[:3, 4] = -0.5 * dt**2 * gravity
    G[3, 4] = -dt
    return G

def L_matrix(unbiased_gyro, unbiased_accel, dt) -> np.ndarray:
    """
    Computes the jacobian of the SE23_state with respect to the input.
    """
    a = unbiased_accel
    om = unbiased_gyro
    omdt = om * dt
    J_att_inv_times_N = SO3.left_jacobian_inv(omdt) @ N_matrix(omdt)
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

class IMUKinematics:

    def __init__(self, Q: np.ndarray):
        self._Q = Q
        self._gravity = np.array([0, 0, -9.80665]).ravel()

    def evaluate(self, x: IMUState, u: IMU, dt: float) -> IMUState:
        x = x.copy()
        u_no_bias = get_unbiased_imu(x, u) # Get unbiased inputs
        G = G_matrix(self._gravity, dt)
        U = U_matrix(u_no_bias.gyro, u_no_bias.accel, dt)
        x.value[0].value = G @ x.value[0].value @ U
        return x

    def jacobian(self, x: IMUState, u: IMU, dt: float) -> np.ndarray:

        u_no_bias = get_unbiased_imu(x, u) # Get unbiased inputs
        G = G_matrix(self._gravity, dt)
        U_inv = U_matrix_inv(u_no_bias.gyro, u_no_bias.accel, dt)

        # Jacobian of process model wrt to pose
        if x.direction == "right":
            jac_pose = adjoint_IE3(U_inv)
        jac_kwargs = {}
        jac_bias = -self._get_input_jacobian(x, u, dt) # Jacobian of pose wrt to bias

        # Jacobian of bias random walk wrt to pose
        jac_pose = np.vstack([jac_pose, np.zeros((6, jac_pose.shape[1]))])

        # Jacobian of bias random walk wrt to biases
        jac_bias = np.vstack([jac_bias, np.identity(6)])
        jac_gyro = jac_bias[:, :3]
        jac_accel = jac_bias[:, 3:6]
        jac_kwargs["bias_gyro"] = jac_gyro
        jac_kwargs["bias_accel"] = jac_accel
        jac_kwargs["attitude"] = jac_pose[:, :3]
        jac_kwargs["velocity"] = jac_pose[:, 3:6]
        jac_kwargs["position"] = jac_pose[:, 6:9]

        return x.jacobian_from_blocks(**jac_kwargs)

    def covariance(self, x: IMUState, u: IMU, dt: float) -> np.ndarray:
        L_pn = self._get_input_jacobian(x, u, dt)  # Jacobian of pose wrt to noise
        L_bn = np.zeros((6, 6)) # Jacobian of bias random walk wrt to noise
        L_pw = np.zeros((9, 6)) # Jacobian of pose wrt to bias random walk
        L_bw = dt * np.identity(6) # Jacobian of bias wrt to bias random walk
        L = np.block([[L_pn, L_pw], [L_bn, L_bw]])
        return L @ self._Q @ L.T

    def _get_input_jacobian(self, x: IMUState, u: IMU, dt: float) -> np.ndarray:
        """
        Computes the jacobian of the nav state with respect to the input.
        """
        u_no_bias = get_unbiased_imu(x, u) # Get unbiased inputs
        G = G_matrix(self._gravity, dt)
        U = U_matrix(u_no_bias.gyro, u_no_bias.accel, dt)
        L = L_matrix(u_no_bias.gyro, u_no_bias.accel, dt)
        if x.direction == "right":
            jac = L
        return jac