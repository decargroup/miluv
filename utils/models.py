import numpy as np
from utils.states import (
    CompositeState,
)
from typing import Any, List, Union
from pymlg import SO2, SO3
from utils.states import (
    CompositeState,
    )
from utils.inputs import (
    VectorInput,
    )
from scipy.linalg import block_diag
from utils.inputs import (
    CompositeInput
)

"""
Module containing:
- functions required for process models
- ProcessModels
- MeasurementModels
"""

def jacobian_fd(self, x, step_size=1e-6):
    """
    Calculates the model jacobian with finite difference.
    """
    N = x.dof
    y = self.evaluate(x)
    m = y.size
    jac_fd = np.zeros((m, N))
    for i in range(N):
        dx = np.zeros((N, 1))
        dx[i, 0] = step_size
        x_temp = x.plus(dx)
        jac_fd[:, i] = (self.evaluate(x_temp) - y).flatten() / step_size

    return jac_fd

class CompositeProcessModel:

    def __init__(
        self,
        model_list: List,
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

    def evaluate_with_jacobian(
        self, x, u: VectorInput, dt: float
    ):
        return self.evaluate(x, u, dt), self.jacobian(x, u, dt)

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

class BodyFrameVelocity:
    """
    The body-frame velocity process model assumes that the input contains
    both translational and angular velocity measurements, both relative to
    a local reference frame, but resolved in the robot body frame.
    """

    def __init__(self, Q: np.ndarray):
        self._Q = Q

    def evaluate(
        self, x, u: VectorInput, dt: float):
        x = x.copy()
        x.value = x.value @ x.group.Exp(u.value * dt)
        return x

    def jacobian(
        self, x, u: VectorInput, dt: float
    ) -> np.ndarray:
        if x.direction == "right":
            return x.group.adjoint(x.group.Exp(-u.value * dt))
        elif x.direction == "left":
            return np.identity(x.dof)

    def covariance(
        self, x, u: VectorInput, dt: float
    ) -> np.ndarray:
        if x.direction == "right":
            L = dt * x.group.left_jacobian(-u.value * dt)
        elif x.direction == "left":
            Ad = x.group.adjoint(x.value @ x.group.Exp(u.value * dt))
            L = dt * Ad @ x.group.left_jacobian(-u.value * dt)

        return L @ self._Q @ L.T

class CompositeMeasurementModel:
    """
    Wrapper for a standard measurement model that assigns the model to a specific
    substate (referenced by `state_id`) inside a CompositeState.
    """

    def __init__(self, model, state_id):
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
    
    def evaluate_with_jacobian(self, x) -> Union[np.ndarray, np.ndarray]:
        return self.evaluate(x), self.jacobian(x)

    def covariance(self, x: CompositeState) -> np.ndarray:
        x_sub = x.get_state_by_id(self.state_id)
        return self.model.covariance(x_sub)
    

class RangePoseToAnchor:
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

    def evaluate(self, x) -> np.ndarray:
        r_zw_a = x.position
        C_ab = x.attitude

        r_tw_a = C_ab @ self._r_tz_b.reshape((-1, 1)) + r_zw_a.reshape((-1, 1))
        r_tc_a: np.ndarray = r_tw_a - self._r_cw_a.reshape((-1, 1))
        return np.linalg.norm(r_tc_a)

    def jacobian(self, x) -> np.ndarray:
        r_zw_a = x.position
        C_ab = x.attitude
        att_group = SO3

        r_tw_a = C_ab @ self._r_tz_b.reshape((-1, 1)) + r_zw_a.reshape((-1, 1))
        r_tc_a: np.ndarray = r_tw_a - self._r_cw_a.reshape((-1, 1))
        rho = r_tc_a / np.linalg.norm(r_tc_a)

        if x.direction == "right":
            jac_attitude = rho.T @ C_ab @ att_group.odot(self._r_tz_b)
            jac_position = rho.T @ C_ab
        elif x.direction == "left":
            raise NotImplementedError("Not implemented.")

        jac = x.jacobian_from_blocks(
            attitude=jac_attitude,
            position=jac_position,
        )
        return jac
    
    def evaluate_with_jacobian(self, x) -> Union[np.ndarray, np.ndarray]:
        return self.evaluate(x), self.jacobian(x)

    def covariance(self, x) -> np.ndarray:
        return self._R


class RangePoseToPose:
    """
    Range model given two absolute poses of rigid bodies, each containing a tag.
    """

    def __init__(
        self, tag_body_position1, tag_body_position2, state_id1, state_id2, R
    ):
        self.tag_body_position1 = np.array(tag_body_position1).flatten()
        self.tag_body_position2 = np.array(tag_body_position2).flatten()
        self.state_id1 = state_id1
        self.state_id2 = state_id2
        self._R = R

    def evaluate(self, x: CompositeState) -> np.ndarray:
        x1 = x.get_state_by_id(self.state_id1)
        x2 = x.get_state_by_id(self.state_id2)
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
        x1 = x.get_state_by_id(self.state_id1)
        x2 = x.get_state_by_id(self.state_id2)
        r_1w_a = x1.position.reshape((-1, 1))
        C_a1 = x1.attitude
        r_2w_a = x2.position.reshape((-1, 1))
        C_a2 = x2.attitude
        r_t1_1 = self.tag_body_position1.reshape((-1, 1))
        r_t2_2 = self.tag_body_position2.reshape((-1, 1))
        r_t1t2_a: np.ndarray = (C_a1 @ r_t1_1 + r_1w_a) - (
            C_a2 @ r_t2_2 + r_2w_a
        )
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
            raise NotImplementedError("Not implemented.")

        if x2.direction == "right":
            jac2 = x2.jacobian_from_blocks(
                attitude=-rho.T @ C_a2 @ att_group.odot(r_t2_2),
                position=-rho.T @ C_a2,
            )
        elif x2.direction == "left":
            raise NotImplementedError("Not implemented.")

        return x.jacobian_from_blocks(
            {self.state_id1: jac1, self.state_id2: jac2}
        )
    
    def evaluate_with_jacobian(self, x ) -> Union[np.ndarray, np.ndarray]:
        return self.evaluate(x), self.jacobian(x)

    def covariance(self, x: CompositeState) -> np.ndarray:
        return self._R


class RangePoseToAnchorById(CompositeMeasurementModel):
    """
    Range model given a pose of another body relative to current pose.
    """

    def __init__(
        self,
        anchor_position: np.ndarray,
        tag_body_position: np.ndarray,
        state_id: Any,
        R: np.ndarray,
    ):

        model = RangePoseToAnchor(anchor_position, tag_body_position, R)
        super(RangePoseToAnchorById, self).__init__(model, state_id)

class Altitude:
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
            Fixed sensor bias, by default 0.0.
        """
        self.R = R
        if minimum is None:
            minimum = -np.inf
        self.minimum = minimum
        self.bias = bias

    def evaluate(self, x):
        h = x.position[2] + self.bias
        return h if h > self.minimum else None

    def jacobian(self, x):
        if x.direction == "right":
            return x.jacobian_from_blocks(
                position=x.attitude[2, :].reshape((1, -1))
            )
        elif x.direction == "left":
            NotImplementedError("Not implemented.")
        
    def evaluate_with_jacobian(self, x) -> Union[np.ndarray, np.ndarray]:
        return self.evaluate(x), self.jacobian(x)

    def covariance(self, x) -> np.ndarray:
        return self.R

class AltitudeById(CompositeMeasurementModel):

    def __init__(
        self, 
        R: np.ndarray, 
        state_id: Any,
        minimum=None, 
        bias=0.0,
    ):

        model = Altitude(R, minimum, bias)
        super(AltitudeById, self).__init__(model, state_id)


class Magnetometer:
    """
    Magnetometer model
    """

    def __init__(self, R: np.ndarray, magnetic_vector: List[float] = None):
        """

        Parameters
        ----------
        R : np.ndarray
            Covariance associated with :math:`\mathbf{v}`
        magnetic_vector : list[float], by default [1, 0, 0]
        """
        if magnetic_vector is None:
            magnetic_vector = [1, 0, 0]

        self.R = R
        self._m_a = np.array(magnetic_vector).reshape((-1, 1))

    def evaluate(self, x):
        return x.attitude.T @ self._m_a

    def jacobian(self, x):
        if x.direction == "right":
            return x.jacobian_from_blocks(
                attitude=-SO3.odot(x.attitude.T @ self._m_a)
            )
        elif x.direction == "left":
            NotImplementedError("Not implemented.")

    def evaluate_with_jacobian(self, x) -> Union[np.ndarray, np.ndarray]:
        return self.evaluate(x), self.jacobian(x)
    
    def covariance(self, x) -> np.ndarray:
        if np.isscalar(self.R):
            return self.R * np.identity(x.position.size)
        else:
            return self.R

class MagnetometerById(CompositeMeasurementModel):

    def __init__(
        self, 
        R: np.ndarray, 
        state_id: Any,
        magnetic_vector: List[float] = None
    ):
        
        model = Magnetometer(R, magnetic_vector= magnetic_vector)
        super(MagnetometerById, self).__init__(model, state_id)