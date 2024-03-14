from pymlg import SO3, SE23
import numpy as np
from typing import Any, List, Tuple
from navlie.lib.states import CompositeState, VectorState, SE23State
from math import factorial
from utils.states import CompositeState
import copy

class IMU:
    """
    Data container for an IMU reading.
    """
    def __init__(
        self,
        gyro: np.ndarray,
        accel: np.ndarray,
        stamp: float,
        state_id: Any = None,
        covariance: np.ndarray = None,
    ):
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
    def __init__(
        self,
        SE23_state: np.ndarray,
        bias_gyro: np.ndarray,
        bias_accel: np.ndarray,
        stamp: float = None,
        state_id = None,
        direction="right",
    ):
        SE23_state = SE23State(SE23_state, stamp, "pose", direction)
        bias_gyro = VectorState(bias_gyro, stamp, "gyro_bias")
        bias_accel = VectorState(bias_accel, stamp, "accel_bias")
        state_list = [SE23_state, bias_gyro, bias_accel]
        super().__init__(state_list, stamp, state_id)
        self.direction = direction

    @property
    def attitude(self) -> np.ndarray:
        return self.value[0].attitude

    @attitude.setter
    def attitude(self, C: np.ndarray):
        self.value[0].attitude = C

    @property
    def velocity(self) -> np.ndarray:
        return self.value[0].velocity

    @velocity.setter
    def velocity(self, v: np.ndarray):
        self.value[0].velocity = v

    @property
    def position(self) -> np.ndarray:
        return self.value[0].position

    @position.setter
    def position(self, r: np.ndarray):
        self.value[0].position = r

    @property
    def bias(self) -> np.ndarray:
        """[gyro_bias, accel_bias]"""
        return np.concatenate(
            [self.value[1].value.ravel(), self.value[2].value.ravel()]
        )

    @bias.setter
    def bias(self, new_bias: np.ndarray) -> np.ndarray:
        bias_gyro = new_bias[0:3]
        bias_accel = new_bias[3:6]
        self.value[1].value = bias_gyro
        self.value[2].value = bias_accel

    @property
    def bias_gyro(self) -> np.ndarray:
        return self.value[1].value

    @bias_gyro.setter
    def bias_gyro(self, gyro_bias: np.ndarray):
        self.value[1].value = gyro_bias.ravel()

    @property
    def bias_accel(self) -> np.ndarray:
        return self.value[2].value

    @bias_accel.setter
    def bias_accel(self, accel_bias: np.ndarray):
        self.value[2].value = accel_bias.ravel()

    @property
    def nav_state(self) -> np.ndarray:
        return self.value[0].value

    @property
    def pose(self) -> np.ndarray:
        return self.value[0].pose

    @pose.setter
    def pose(self, pose):
        self.value[0].pose = pose

    def copy(self):
        return copy.deepcopy(self)

    def jacobian_from_blocks(
        self,
        attitude: np.ndarray = None,
        position: np.ndarray = None,
        velocity: np.ndarray = None,
        bias_gyro: np.ndarray = None,
        bias_accel: np.ndarray = None,
    ):
        for jac in [attitude, position, velocity, bias_gyro, bias_accel]:
            if jac is not None:
                dim = jac.shape[0]
                break

        nav_jacobian = self.value[0].jacobian_from_blocks(
            attitude=attitude, position=position, velocity=velocity)
        if bias_gyro is None:
            bias_gyro = np.zeros((dim, 3))
        if bias_accel is None:
            bias_accel = np.zeros((dim, 3))

        return np.hstack([nav_jacobian, bias_gyro, bias_accel])

def get_unbiased_imu(x: IMUState, u: IMU) -> IMU:
    u = u.copy()
    if hasattr(x, "bias_gyro"):
        u.gyro = u.gyro.ravel() - x.bias_gyro.ravel()
    if hasattr(x, "bias_accel"):
        u.accel = u.accel.ravel() - x.bias_accel.ravel()

    return u

def N_matrix(phi_vec: np.ndarray):
    """
    The N matrix from Barfoot 2nd edition, equation 9.211
    """
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

def U_matrix(omega, accel, dt: float):
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

def U_matrix_inv(omega, accel, dt: float):
    return inverse_IE3(U_matrix(omega, accel, dt))

def G_matrix(gravity, dt):
    G = np.identity(5)
    G[:3, 3] = dt * gravity
    G[:3, 4] = -0.5 * dt**2 * gravity
    G[3, 4] = -dt
    return G

def L_matrix(unbiased_gyro, unbiased_accel, dt: float) -> np.ndarray:
    """
    Computes the jacobian of the nav state with respect to the input.
    Since the noise and bias are both additive to the input, they have the
    same jacobians.
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
    # See Barfoot 2nd edition, equation 9.247
    Up = dt * np.eye(9, 6)
    Up[6:9, 0:3] = (
        -0.5 * (dt**2 / 2) * (
            (1 / 360)
            * (dt**3)
            * (OmOm @ A + Om @ (SO3.wedge(Om @ a)) + SO3.wedge(OmOm @ a))
            - (1 / 6) * dt * A
        )
    )
    Up[6:9, 3:6] = (dt**2 / 2) * J_att_inv_times_N

    L = J @ Up
    return L

class IMUKinematics:

    def __init__(self, Q: np.ndarray):
        self._Q = Q
        self._gravity = np.array([0, 0, -9.80665]).ravel()

    def evaluate(self, x: IMUState, u: IMU, dt: float) -> IMUState:
        x = x.copy()
        # Get unbiased inputs
        u_no_bias = get_unbiased_imu(x, u)
        G = G_matrix(self._gravity, dt)
        U = U_matrix(u_no_bias.gyro, u_no_bias.accel, dt)
        x.pose = G @ x.pose @ U

        return x

    def jacobian(self, x: IMUState, u: IMU, dt: float) -> np.ndarray:

        # Get unbiased inputs
        u_no_bias = get_unbiased_imu(x, u)
        G = G_matrix(self._gravity, dt)
        U_inv = U_matrix_inv(u_no_bias.gyro, u_no_bias.accel, dt)

        # Jacobian of process model wrt to pose
        if x.direction == "right":
            jac_pose = adjoint_IE3(U_inv)
        elif x.direction == "left":
            raise NotImplementedError("Left-incremental not implemented yet")

        jac_kwargs = {}
        if hasattr(x, "bias_gyro"):
            # Jacobian of pose wrt to bias
            jac_bias = -self._get_input_jacobian(x, u, dt)

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
        # Jacobian of pose wrt to noise
        L_pn = self._get_input_jacobian(x, u, dt)

        # Jacobian of bias random walk wrt to noise
        L_bn = np.zeros((6, 6))

        if hasattr(x, "bias_gyro"):
            # Jacobian of pose wrt to bias random walk
            L_pw = np.zeros((9, 6))

            # Jacobian of bias wrt to bias random walk
            L_bw = dt * np.identity(6)
            L = np.block([[L_pn, L_pw], [L_bn, L_bw]])
            
        else:
            L = np.hstack([[L_pn, L_bn]])

        return L @ self._Q @ L.T

    def _get_input_jacobian(self, x: IMUState, u: IMU, dt: float) -> np.ndarray:
        """
        Computes the jacobian of the nav state with respect to the input.

        Since the noise and bias are both additive to the input, they have the
        same jacobians.
        """
        # Get unbiased inputs
        u_no_bias = get_unbiased_imu(x, u)

        G = G_matrix(self._gravity, dt)
        U = U_matrix(u_no_bias.gyro, u_no_bias.accel, dt)
        L = L_matrix(u_no_bias.gyro, u_no_bias.accel, dt)

        if x.direction == "right":
            jac = L
        elif x.direction == "left":
            raise NotImplementedError("Left-incremental not implemented yet")
        return jac