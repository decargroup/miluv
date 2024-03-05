import numpy as np
from utils.states import (
    State,
    CompositeState,
)

from typing import Any, List, Union
from pymlg import SO2, SO3, SE23
from utils.states import (
    State,
    CompositeState,
    MatrixLieGroupState,
    )
from abc import ABC, abstractmethod
from scipy.linalg import block_diag
from typing import Tuple
from scipy.linalg import expm
from utils.inputs import (
    Input, 
    IMU,
    IMUState,
    CompositeInput
)
from math import factorial

"""
Module containing:
- functions required for process models
- ProcessModels
- MeasurementModels
"""

def van_loans(
    A_c: np.ndarray,
    L_c: np.ndarray,
    Q_c: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Van Loan's method for computing the discrete-time A and Q matrices.

    Given a continuous-time system of the form

    .. math::
        \dot{\mathbf{x}} = \mathbf{A}_c \mathbf{x} + \mathbf{L}_c \mathbf{w}, \hspace{5mm}
        \mathbf{w} \sim \mathcal{N} (\mathbf{0}, \mathbf{Q}_c ),

    where :math:``\mathbf{Q}_c`` is a power spectral density,
    Van Loan's method can be used to find its equivalent discrete-time representation,

    .. math::
        \mathbf{x}_k = \mathbf{A}_{d} \mathbf{x}_{k-1} + \mathbf{w}_{k-1}, \hspace{5mm}
        \mathbf{w} \sim \mathcal{N} (\mathbf{0}, \mathbf{Q}_d ).

    These are computed using the matrix exponential, with a sampling period :math:``\Delta t``.

    Parameters
    ----------
    A_c : np.ndarray
        Continuous-time A matrix.
    L_c : np.ndarray
        Continuous-time L matrix.
    Q_c : np.ndarray
        Continuous-time noise matrix
    dt : float
        Discretization timestep.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A_d and Q_d, discrete-time matrices.
    """
    N = A_c.shape[0]

    A_c = np.atleast_2d(A_c)
    L_c = np.atleast_2d(L_c)
    Q_c = np.atleast_2d(Q_c)

    # Form Xi matrix and compute Upsilon using matrix exponential
    Xi = block_diag(A_c, -A_c.T, A_c, np.zeros((N, N)))
    Xi[:N, N : 2 * N] = L_c @ Q_c @ L_c.T
    Upsilon = expm(Xi * dt)

    # Extract relevant parts of Upsilon
    A_d = Upsilon[:N, :N]
    Q_d = Upsilon[:N, N : 2 * N] @ A_d.T
    
    return A_d, Q_d

class ProcessModel(ABC):
    """
    Abstract process model base class for process models of the form

    .. math::
        \mathbf{x}_k = \mathbf{f}(\mathbf{x}_{k-1}, \mathbf{u}, \Delta t) + \mathbf{w}_{k}

    where :math:`\mathbf{u}` is the input, :math:`\Delta t` is the time
    period between the two states, and :math:`\mathbf{w}_{k} \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)`
    is additive Gaussian noise.
    """

    @abstractmethod
    def evaluate(self, x: State, u: Input, dt: float) -> State:
        """
        Implementation of :math:`\mathbf{f}(\mathbf{x}_{k-1}, \mathbf{u}, \Delta t)`.

        Parameters
        ----------
        x : State
            State at time :math:`k-1`.
        u : Input
            The input value :math:`\mathbf{u}` provided as a Input object.
            The actual numerical value is accessed via `u.value`.
        dt : float
            The time interval :math:`\Delta t` between the two states.

        Returns
        -------
        State
            State at time :math:`k`.
        """
        pass

    def covariance(self, x: State, u: Input, dt: float) -> np.ndarray:
        pass

    def jacobian(self, x: State, u: Input, dt: float) -> np.ndarray:
        pass

    def evaluate_with_jacobian(
        self, x: State, u: Input, dt: float
    ) -> Union[State, np.ndarray]:
        """
        Evaluates the process model and simultaneously returns the Jacobian as
        its second output argument. This is useful to override for
        performance reasons when the model evaluation and Jacobian have a lot of
        common calculations, and it is more efficient to calculate them in the
        same function call.
        """
        return self.evaluate(x, u, dt), self.jacobian(x, u, dt)

    def __repr__(self):
        return f"{self.__class__.__name__} at {hex(id(self))}"


def unbiased_imu(x: IMUState,
                 u: IMU,
                ):

    if hasattr(x, "bias_gyro") and hasattr(x, "bias_accel"):
        u_gyro = u.gyro + x.bias_gyro
        u_accel = u.accel + x.bias_accel
        bias = True
    else:
        u_gyro = u.gyro
        u_accel = u.accel
        bias = False
    
    return u_gyro, u_accel, bias


class BodyFrameIMU(ProcessModel):

    def __init__(self, Q: np.ndarray,
                 ):
        self._Q = Q

    def evaluate(
        self, x: IMUState, u: IMU, dt: float
    ) -> IMUState:
        x = x.copy()

        u_gyro, u_accel, bias = unbiased_imu(x, u)

        g_a = np.array([0, 0, -9.81])

        x_out = x.copy()
        x_out.attitude = x.attitude @ SO3.Exp(u_gyro * dt)
        x_out.velocity = x.velocity + dt * x.attitude @ u_accel + dt * g_a
        x_out.position = x.position + dt * x.velocity

        return x_out
    
    def continuous_time_matrices(
            self, x: IMUState, u: IMU
            ) -> np.ndarray:
        
        x = x.copy()
        
        u_gyro, u_accel, bias = unbiased_imu(x, u)

        A = np.zeros((x.dof, x.dof))
        A[0:3, 0:3] = - SO3.wedge(u_gyro)
        A[3:6, 0:3] = - x.attitude @ SO3.wedge(u_accel)
        A[6:9, 3:6] = np.eye(len(x.velocity))

        if bias:
            A[0:3, 9:12] = np.eye(len(u_gyro))
            A[3:6, 12:15] = x.attitude

        L = np.zeros((len(A), len(self._Q)))
        L[0:3, 0:3] = - np.eye(len(u_gyro))
        L[3:6, 3:6] = - x.attitude
        
        if bias:
            L[9:12, 6:9] = -np.eye(len(u_gyro))
            L[12:15, 9:12] = -np.eye(len(u_accel))

        return A, L
    
    def jacobian(
        self, x: IMUState, u: IMU, dt: float
    ) -> np.ndarray:
        
        x = x.copy()
        A, L = self.continuous_time_matrices(x, u)
        A_d, Q_d = van_loans(A, L, self._Q, dt)
        return A_d

    def covariance(
        self, x: IMUState, u: IMU, dt: float
    ) -> np.ndarray:

        x = x.copy()
        A, L = self.continuous_time_matrices(x, u)
        A_d, Q_d = van_loans(A, L, self._Q, dt)

        return Q_d
    
    def evaluate_with_jacobian(
        self, x: State, u: Input, dt: float
    ):
        
        return self.evaluate(x, u, dt), self.jacobian(x, u, dt)
    
    def __repr__(self):
        return f"{self.__class__.__name__} at {hex(id(self))}"


def get_unbiased_imu(x: IMUState, u: IMU) -> IMU:
    """
    Removes bias from the measurement.

    Parameters
    ----------
    x : IMUState
        Contains the biases
    u : IMU
        IMU data correupted by bias

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        unbiased gyro and accelerometer measurements
    """

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


def M_matrix(phi_vec):
    phi_mat = SO3.wedge(phi_vec)
    M = np.sum(
        [
            (2 / factorial(n + 2)) * np.linalg.matrix_power(phi_mat, n)
            for n in range(100)
        ],
        axis=0,
    )
    return M


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


def U_tilde_matrix(omega, accel, dt: float):
    phi = omega * dt
    O = SO3.Exp(phi)
    J = SO3.left_jacobian(phi)
    a = accel.reshape((-1, 1))
    V = N_matrix(phi)
    U = np.identity(5)
    U[:3, :3] = O
    U[:3, 3] = np.ravel(dt * J @ a)
    U[:3, 4] = np.ravel(dt**2 / 2 * V @ a)
    return U


def delta_matrix(dt: float):
    U = np.identity(5)
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


def G_matrix_inv(gravity, dt):
    return inverse_IE3(G_matrix(gravity, dt))


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
        -0.5
        * (dt**2 / 2)
        * (
            (1 / 360)
            * (dt**3)
            * (OmOm @ A + Om @ (SO3.wedge(Om @ a)) + SO3.wedge(OmOm @ a))
            - (1 / 6) * dt * A
        )
    )
    Up[6:9, 3:6] = (dt**2 / 2) * J_att_inv_times_N

    L = J @ Up
    return L

class IMUKinematics(ProcessModel):
    """
    The IMU Kinematics refer to the following continuous time model:

    .. math::

        \\dot{\mathbf{r}} &= \mathbf{v}

        \\dot{\mathbf{v}} &= \mathbf{C}\mathbf{a} +  \mathbf{g}

        \\dot{\mathbf{C}} &= \mathbf{C}\mathbf{\omega}^\wedge

    Using :math:`SE_2(3)` extended poses, it can be shown that the
    discrete-time IMU kinematics are given by:

    .. math::
        \mathbf{T}_{k} = \mathbf{G}_{k-1} \mathbf{T}_{k-1} \mathbf{U}_{k-1}

    where :math:`\mathbf{T}_{k}` is the pose at time :math:`k`, :math:`\mathbf{G}_{k-1}`
    is a matrix that depends on the gravity vector, and :math:`\mathbf{U}_{k-1}` is a matrix
    that depends on the IMU measurements.

    The :math:`\mathbf{G}_{k-1}` and :math:`\mathbf{U}_{k-1}` matrices are
    not quite elements of :math:`SE_2(3)`, but instead belong to a new group
    named here the "Incremental Euclidean Group" :math:`IE(3)`.

    """

    def __init__(self, Q: np.ndarray, gravity=None):
        """
        Parameters
        ----------
        Q : np.ndarray
            Discrete-time noise matrix.
        g_a : np.ndarray
            Gravity vector resolved in the inertial frame.
            If None, default value is set to [0; 0; -9.80665].
        """
        self._Q = Q

        if gravity is None:
            gravity = np.array([0, 0, -9.80665])

        self._gravity = np.array(gravity).ravel()

    def evaluate(self, x: IMUState, u: IMU, dt: float) -> IMUState:
        """
        Propagates an IMU state forward one timestep from an IMU measurement.

        The continuous-time IMU equations are discretized using the assumption
        that the IMU measurements are constant between two timesteps.

        Parameters
        ----------
        x : IMUState
            Current IMU state
        u : IMU
            IMU measurement,
        dt : float
            timestep.

        Returns
        -------
        IMUState
            Propagated IMUState.
        """
        x = x.copy()

        # Get unbiased inputs
        u_no_bias = get_unbiased_imu(x, u)

        G = G_matrix(self._gravity, dt)
        U = U_matrix(u_no_bias.gyro, u_no_bias.accel, dt)

        x.pose = G @ x.pose @ U

        # Propagate the biases forward in time using random walk
        if hasattr(u, "bias_gyro_walk") and hasattr(x, "bias_gyro"):
            x.bias_gyro = x.bias_gyro.ravel() + dt * u.bias_gyro_walk.ravel()

        if hasattr(u, "bias_accel_walk") and hasattr(x, "bias_accel"):
            x.bias_accel = x.bias_accel.ravel() + dt * u.bias_accel_walk.ravel()

        return x

    def jacobian(self, x: IMUState, u: IMU, dt: float) -> np.ndarray:
        """
        Returns the Jacobian of the IMU kinematics model with respect
        to the full state
        """

        # Get unbiased inputs
        u_no_bias = get_unbiased_imu(x, u)

        G = G_matrix(self._gravity, dt)
        U_inv = U_matrix_inv(u_no_bias.gyro, u_no_bias.accel, dt)

        # Jacobian of process model wrt to pose
        if x.direction == "right":
            jac_pose = adjoint_IE3(U_inv)
        elif x.direction == "left":
            jac_pose = adjoint_IE3(G)

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
            jac = SE23.adjoint(G @ x.pose @ U) @ L
        return jac
    

class CompositeProcessModel(ProcessModel):

    def __init__(
        self,
        model_list: List[ProcessModel],
        shared_input: bool = False,
    ):
        self._model_list = model_list
        self._shared_input = shared_input

    def evaluate(
        self,
        x: CompositeState,
        u: CompositeInput,
        dt: float,
    ) -> CompositeState:
        x = x.copy()
        for i, x_sub in enumerate(x.value):
            if self._shared_input:
                u_sub = u
            else:
                u_sub = u.input_list[i]
            x.value[i] = self._model_list[i].evaluate(x_sub, u_sub, dt)

        return x

    def jacobian(
        self,
        x: CompositeState,
        u: CompositeInput,
        dt: float,
    ) -> np.ndarray:
        jac = []
        for i, x_sub in enumerate(x.value):
            if self._shared_input:
                u_sub = u
            else:
                u_sub = u.input_list[i]
            jac.append(self._model_list[i].jacobian(x_sub, u_sub, dt))

        return block_diag(*jac)

    def covariance(
        self,
        x: CompositeState,
        u: CompositeInput,
        dt: float,
    ) -> np.ndarray:
        cov = []
        for i, x_sub in enumerate(x.value):
            if self._shared_input:
                u_sub = u
            else:
                u_sub = u.input_list[i]
            cov.append(self._model_list[i].covariance(x_sub, u_sub, dt))

        return block_diag(*cov)
    

class MeasurementModel(ABC):
    """
    Abstract measurement model base class, used to implement measurement models
    of the form

    .. math::
        \mathbf{y} = \mathbf{g}(\mathbf{x}) + \mathbf{v}

    where :math:`\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{R})`.

    """

    @abstractmethod
    def evaluate(self, x: State) -> np.ndarray:
        """
        Evaluates the measurement model :math:`\mathbf{g}(\mathbf{x})`.
        """
        pass

    @abstractmethod
    def covariance(self, x: State) -> np.ndarray:
        """
        Returns the covariance :math:`\mathbf{R}` associated with additive Gaussian noise.
        """
        pass

    def jacobian(self, x: State) -> np.ndarray:
        """
        Evaluates the measurement model Jacobian
        :math:`\mathbf{G} = \partial \mathbf{g}(\mathbf{x})/ \partial \mathbf{x}`.
        """
        return self.jacobian_fd(x)

    def evaluate_with_jacobian(self, x: State) -> Union[np.ndarray, np.ndarray]:
        """
        Evaluates the measurement model and simultaneously returns the Jacobian
        as its second output argument. This is useful to override for
        performance reasons when the model evaluation and Jacobian have a lot of
        common calculations, and it is more efficient to calculate them in the
        same function call.
        """
        return self.evaluate(x), self.jacobian(x)

    def jacobian_fd(self, x: State, step_size=1e-6):
        """
        Calculates the model jacobian with finite difference.
        """
        N = x.dof
        y = self.evaluate(x)
        m = y.size
        jac_fd = np.zeros((m, N))
        for i in range(N):
            dx = np.zeros((N,))
            dx[i] = step_size
            x_temp = x.plus(dx)
            jac_fd[:, i] = (self.evaluate(x_temp) - y).flatten() / step_size

        return jac_fd

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def sqrt_information(self, x: State):
        R = np.atleast_2d(self.covariance(x))
        return np.linalg.cholesky(np.linalg.inv(R))
    

class CompositeMeasurementModel(MeasurementModel):
    """
    Wrapper for a standard measurement model that assigns the model to a specific
    substate (referenced by `state_id`) inside a CompositeState.
    """

    def __init__(self, model: MeasurementModel, state_id):
        self.model = model
        self.state_id = state_id

    def __repr__(self):
        return f"{self.model}(of substate {self.state_id})"

    def evaluate(self, x: CompositeState) -> np.ndarray:
        return self.model.evaluate(x.get_state_by_id(self.state_id))

    def jacobian(self, x: CompositeState) -> np.ndarray:
        x_sub = x.get_state_by_id(self.state_id)
        jac_sub = self.model.jacobian(x_sub)
        jac = np.zeros((jac_sub.shape[0], x.dof))
        slc = x.get_slice_by_id(self.state_id)
        jac[:, slc] = jac_sub
        return jac

    def covariance(self, x: CompositeState) -> np.ndarray:
        x_sub = x.get_state_by_id(self.state_id)
        return self.model.covariance(x_sub)
    

class RangePoseToAnchor(MeasurementModel):
    """
    Range measurement from a pose state to an anchor.
    """

    def __init__(
        self,
        anchor_position: List[float],
        tag_body_position: List[float],
        R: float,
    ):
        self._r_cw_a = np.array(anchor_position).flatten()
        self._R = R
        self._r_tz_b = np.array(tag_body_position).flatten()

    def evaluate(self, x: MatrixLieGroupState) -> np.ndarray:
        r_zw_a = x.position
        C_ab = x.attitude

        r_tw_a = C_ab @ self._r_tz_b.reshape((-1, 1)) + r_zw_a.reshape((-1, 1))
        r_tc_a: np.ndarray = r_tw_a - self._r_cw_a.reshape((-1, 1))
        return np.linalg.norm(r_tc_a)

    def jacobian(self, x: MatrixLieGroupState) -> np.ndarray:
        r_zw_a = x.position
        C_ab = x.attitude
        if C_ab.shape == (2, 2):
            att_group = SO2
        elif C_ab.shape == (3, 3):
            att_group = SO3

        r_tw_a = C_ab @ self._r_tz_b.reshape((-1, 1)) + r_zw_a.reshape((-1, 1))
        r_tc_a: np.ndarray = r_tw_a - self._r_cw_a.reshape((-1, 1))
        rho = r_tc_a / np.linalg.norm(r_tc_a)

        if x.direction == "right":
            jac_attitude = rho.T @ C_ab @ att_group.odot(self._r_tz_b)
            jac_position = rho.T @ C_ab
        elif x.direction == "left":
            jac_attitude = rho.T @ att_group.odot(C_ab @ self._r_tz_b + r_zw_a)
            jac_position = rho.T @ np.identity(r_zw_a.size)

        jac = x.jacobian_from_blocks(
            attitude=jac_attitude,
            position=jac_position,
        )
        return jac

    def covariance(self, x: MatrixLieGroupState) -> np.ndarray:
        return self._R


class RangePoseToPose(MeasurementModel):
    """
    Range model given two absolute poses of rigid bodies, each containing a tag.
    """

    # TODO. tag_body_positions should be optional. argh but this will be
    # a breaking change since the argument order needs to be different.
    def __init__(
        self, tag_body_position1, tag_body_position2, state_id1, state_id2, R
    ):
        self.tag_body_position1 = np.array(tag_body_position1).flatten()
        self.tag_body_position2 = np.array(tag_body_position2).flatten()
        self.state_id1 = state_id1
        self.state_id2 = state_id2
        self._R = R

    def evaluate(self, x: CompositeState) -> np.ndarray:
        x1: MatrixLieGroupState = x.get_state_by_id(self.state_id1)
        x2: MatrixLieGroupState = x.get_state_by_id(self.state_id2)
        r_1w_a = x1.position.reshape((-1, 1))
        C_a1 = x1.attitude
        r_2w_a = x2.position.reshape((-1, 1))
        C_a2 = x2.attitude
        r_t1_1 = self.tag_body_position1.reshape((-1, 1))
        r_t2_2 = self.tag_body_position2.reshape((-1, 1))
        r_t1t2_a: np.ndarray = (C_a1 @ r_t1_1 + r_1w_a) - (
            C_a2 @ r_t2_2 + r_2w_a
        )
        return np.array(np.linalg.norm(r_t1t2_a.flatten()))

    def jacobian(self, x: CompositeState) -> np.ndarray:
        x1: MatrixLieGroupState = x.get_state_by_id(self.state_id1)
        x2: MatrixLieGroupState = x.get_state_by_id(self.state_id2)
        r_1w_a = x1.position.reshape((-1, 1))
        C_a1 = x1.attitude
        r_2w_a = x2.position.reshape((-1, 1))
        C_a2 = x2.attitude
        r_t1_1 = self.tag_body_position1.reshape((-1, 1))
        r_t2_2 = self.tag_body_position2.reshape((-1, 1))
        r_t1t2_a: np.ndarray = (C_a1 @ r_t1_1 + r_1w_a) - (
            C_a2 @ r_t2_2 + r_2w_a
        )

        if C_a1.shape == (2, 2):
            att_group = SO2
        elif C_a1.shape == (3, 3):
            att_group = SO3

        rho: np.ndarray = (
            r_t1t2_a / np.linalg.norm(r_t1t2_a.flatten())
        ).reshape((-1, 1))

        if x1.direction == "right":
            jac1 = x1.jacobian_from_blocks(
                attitude=rho.T @ C_a1 @ att_group.odot(r_t1_1),
                position=rho.T @ C_a1,
            )
        elif x1.direction == "left":
            jac1 = x1.jacobian_from_blocks(
                attitude=rho.T @ att_group.odot(C_a1 @ r_t1_1 + r_1w_a),
                position=rho.T @ np.identity(r_t1_1.size),
            )

        if x2.direction == "right":
            jac2 = x2.jacobian_from_blocks(
                attitude=-rho.T @ C_a2 @ att_group.odot(r_t2_2),
                position=-rho.T @ C_a2,
            )
        elif x2.direction == "left":
            jac2 = x2.jacobian_from_blocks(
                attitude=-rho.T @ att_group.odot(C_a2 @ r_t2_2 + r_2w_a),
                position=-rho.T @ np.identity(r_t2_2.size),
            )

        return x.jacobian_from_blocks(
            {self.state_id1: jac1, self.state_id2: jac2}
        )

    def covariance(self, x: CompositeState) -> np.ndarray:
        return self._R


class RangeRelativePose(CompositeMeasurementModel):
    """
    Range model given a pose of another body relative to current pose. This
    model operates on a CompositeState where it is assumed that the neighbor
    relative pose is stored as a substate somewhere inside the composite state
    with a state_id matching the `nb_state_id` supplied to this model.
    """

    def __init__(
        self,
        tag_body_position: np.ndarray,
        nb_tag_body_position: np.ndarray,
        nb_state_id: Any,
        R: np.ndarray,
    ):
        """

        Parameters
        ----------
        tag_body_position : numpy.ndarray
            Position of tag with respect to Frame 1.
        nb_tag_body_position : numpy.ndarray
            Position of 2nd tag with respect to Frame 2, in Robot 2.
        nb_state_id : Any
            State ID of Robot 2.
        R : float or numpy.ndarray
            covariance associated with range measurement
        """

        model = RangePoseToAnchor(tag_body_position, nb_tag_body_position, R)
        super(RangeRelativePose, self).__init__(model, nb_state_id)

    def __repr__(self):
        return f"RangeRelativePose (of substate {self.state_id})"

class Altitude(MeasurementModel):
    """
    A model that returns that z component of a position vector.
    """

    def __init__(self, R: np.ndarray, minimum=None, bias=0.0):
        """

        Parameters
        ----------
        R : np.ndarray
            variance associated with the measurement
        minimum : float, optional
            Minimal height for the measurement to be valid, by default None
        bias : float, optional
            Fixed sensor bias, by default 0.0. This bias will be added to the
            z component of position to create the modelled measurement.
        """
        self.R = R
        if minimum is None:
            minimum = -np.inf
        self.minimum = minimum
        self.bias = bias

    def evaluate(self, x: MatrixLieGroupState):
        h = x.position[2] + self.bias
        return h if h > self.minimum else None

    def jacobian(self, x: MatrixLieGroupState):
        if x.direction == "right":
            return x.jacobian_from_blocks(
                position=x.attitude[2, :].reshape((1, -1))
            )
        elif x.direction == "left":
            return x.jacobian_from_blocks(
                attitude=SO3.odot(x.position)[2, :].reshape((1, -1)),
                position=np.array(([[0, 0, 1]])),
            )

    def covariance(self, x: MatrixLieGroupState) -> np.ndarray:
        return self.R

class AltitudeById(CompositeMeasurementModel):

    def __init__(
        self, 
        R: np.ndarray, 
        nb_state_id: Any,
        minimum=None, 
        bias=0.0,
    ):
        """

        Parameters
        ----------
        nb_state_id : Any
            State ID of Robot 2.
        R : float or numpy.ndarray
            covariance associated with range measurement
        """

        model = Altitude(R, minimum, bias)
        super(AltitudeById, self).__init__(model, nb_state_id)

    def __repr__(self):
        return f"RangeAltitude (of substate {self.state_id})"


class Magnetometer(MeasurementModel):
    """
    Magnetometer model of the form

    .. math::

        \mathbf{y} = \mathbf{C}_{ab}^T \mathbf{m}_a + \mathbf{v}

    where :math:`\mathbf{m}_a` is the magnetic field vector in a world frame `a`.
    """

    def __init__(self, R: np.ndarray, magnetic_vector: List[float] = None):
        """

        Parameters
        ----------
        R : np.ndarray
            Covariance associated with :math:`\mathbf{v}`
        magnetic_vector : list[float] or numpy.ndarray, optional
            local magnetic field vector, by default [1, 0, 0]
        """
        if magnetic_vector is None:
            magnetic_vector = [1, 0, 0]

        self.R = R
        self._m_a = np.array(magnetic_vector).reshape((-1, 1))

    def evaluate(self, x: MatrixLieGroupState):
        return x.attitude.T @ self._m_a

    def jacobian(self, x: MatrixLieGroupState):
        if x.direction == "right":
            return x.jacobian_from_blocks(
                attitude=-SO3.odot(x.attitude.T @ self._m_a)
            )
        elif x.direction == "left":
            return x.jacobian_from_blocks(
                attitude=-x.attitude.T @ SO3.odot(self._m_a)
            )

    def covariance(self, x: MatrixLieGroupState) -> np.ndarray:
        if np.isscalar(self.R):
            return self.R * np.identity(x.position.size)
        else:
            return self.R
        


class MagnetometerById(CompositeMeasurementModel):

    def __init__(
        self, 
        R: np.ndarray, 
        nb_state_id: Any,
        magnetic_vector: List[float] = None
    ):
        """

        Parameters
        ----------
        nb_state_id : Any
            State ID of Robot 2.
        R : float or numpy.ndarray
            covariance associated with range measurement
        """

        model = Magnetometer(R, magnetic_vector= magnetic_vector)
        super(MagnetometerById, self).__init__(model, nb_state_id)

    def __repr__(self):
        return f"RangeAltitude (of substate {self.state_id})"