import numpy as np
from utils.states import (
    State,
    CompositeState,
)

from typing import Any, List, Union
from pymlg import SO2, SO3
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

        attitude = x.attitude
        velocity = x.velocity
        position = x.position

        x.attitude = attitude @ SO3.Exp(u_gyro * dt)
        x.velocity = velocity + dt * attitude @ u_accel + dt * g_a
        x.position = position + dt * velocity

        return x
    
    def continuous_time_matrices(
            self, x: IMUState, u: IMU
            ) -> np.ndarray:
        
        x = x.copy()
        
        u_gyro, u_accel, bias = unbiased_imu(x, u)

        A = np.zeros((x.dof, x.dof))
        A[0:3, 0:3] = - SO3.wedge(u_gyro)
        A[3:6, 0:3] = - x.attitude @ SO3.wedge(u_accel)
        A[6:9, 3:6] = np.eye(len(x.velocity))

        # if bias:
        A[0:3, 9:12] = np.eye(len(u_gyro))
        A[3:6, 12:15] = x.attitude

        L = np.zeros((len(A), len(self._Q)))
        L[0:3, 0:3] = - np.eye(len(u_gyro))
        L[3:6, 3:6] = - x.attitude
        
        # if bias:
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
    ) -> Union[State, np.ndarray]:
        
        return self.evaluate(x, u, dt), self.jacobian(x, u, dt)
    
    def __repr__(self):
        return f"{self.__class__.__name__} at {hex(id(self))}"
    

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
            Position of tag in body frame of Robot 1.
        nb_tag_body_position : numpy.ndarray
            Position of 2nd tag in body frame of Robot 2.
        nb_state_id : Any
            State ID of Robot 2.
        R : float or numpy.ndarray
            covariance associated with range measurement
        """

        model = RangePoseToAnchor(tag_body_position, nb_tag_body_position, R)
        super(RangeRelativePose, self).__init__(model, nb_state_id)

    def __repr__(self):
        return f"RangeRelativePose (of substate {self.state_id})"