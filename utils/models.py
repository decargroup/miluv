import numpy as np
from typing import List, Any, Union
from utils.inputs import VectorInput, CompositeInput
from utils.states import CompositeState
from utils.imu import IMUState
from pymlg import SO3
from scipy.linalg import block_diag

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
        self, x, u: VectorInput, dt: float) -> np.ndarray:
        if x.direction == "right":
            return x.group.adjoint(x.group.Exp(-u.value * dt))
        
    def covariance(
        self, x, u: VectorInput, dt: float) -> np.ndarray:
        if x.direction == "right":
            L = dt * x.group.left_jacobian(-u.value * dt)

        return L @ self._Q @ L.T
    

class CompositeProcessModel:

    def __init__(
        self,
        model_list: List,
        shared_input: bool = False,
    ):
        self._model_list = model_list
        self._shared_input = shared_input

    def evaluate(
        self, x: CompositeState, u: CompositeInput, dt: float,
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
        self, x: CompositeState, u: CompositeInput, dt: float,
    ) -> np.ndarray:
        jac = []
        for i, x_sub in enumerate(x.value):
            if self._shared_input:
                u_sub = u
            else:
                u_sub = u.input_list[i]
            jac.append(self._model_list[i].jacobian(x_sub, u_sub, dt))

        return block_diag(*jac)

    def evaluate_with_jacobian(self, x, u: VectorInput, dt: float):
        return self.evaluate(x, u, dt), self.jacobian(x, u, dt)

    def covariance(self, x: CompositeState, u: CompositeInput, dt: float):
        cov = []
        for i, x_sub in enumerate(x.value):
            if self._shared_input:
                u_sub = u
            else:
                u_sub = u.input_list[i]
            cov.append(self._model_list[i].covariance(x_sub, u_sub, dt))

        return block_diag(*cov)

class CompositeMeasurementModel:
    """
    Wrapper for a standard measurement model that assigns the model to a specific
    substate (referenced by `state_id`) inside a CompositeState.
    """
    def __init__(self, model, state_id):
        self.model = model
        self.state_id = state_id

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
    # Range measurement from a pose state to an anchor.
    def __init__(self, anchor_position: List[float],
        tag_body_position: List[float], R: float, ):
        self._r_cw_a = np.array(anchor_position).flatten()
        self._R = R
        self._r_tz_b = np.array(tag_body_position).flatten()

    def evaluate(self, x) -> np.ndarray:
        if isinstance(x, IMUState):
            _x = x.value[0]
        else:
            _x = x
        r_zw_a = _x.value[0:3, -1]
        C_ab = _x.value[0:3, 0:3]
        r_tw_a = C_ab @ self._r_tz_b.reshape((-1, 1)) + r_zw_a.reshape((-1, 1))
        r_tc_a: np.ndarray = r_tw_a - self._r_cw_a.reshape((-1, 1))
        return np.linalg.norm(r_tc_a)

    def jacobian(self, x) -> np.ndarray:
        if isinstance(x, IMUState):
            _x = x.value[0]
        else:
            _x = x
        r_zw_a = _x.value[0:3, -1]
        C_ab = _x.value[0:3, 0:3]
        r_tw_a = C_ab @ self._r_tz_b.reshape((-1, 1)) + r_zw_a.reshape((-1, 1))
        r_tc_a = r_tw_a - self._r_cw_a.reshape((-1, 1))
        rho = r_tc_a / np.linalg.norm(r_tc_a)
        if x.direction == "right":
            jac_attitude = rho.T @ C_ab @ SO3.odot(self._r_tz_b)
            jac_position = rho.T @ C_ab
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
    # Range model given two absolute poses of rigid bodies, each containing a tag.
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

        if isinstance(x1, IMUState) and isinstance(x2, IMUState):
            _x1, _x2 = x1.value[0], x2.value[0]
        else:
            _x1, _x2 = x1, x2
        r_1w_a = _x1.value[0:3, -1].reshape((-1, 1))
        C_a1 = _x1.value[0:3, 0:3]
        r_2w_a = _x2.value[0:3, -1].reshape((-1, 1))
        C_a2 = _x2.value[0:3, 0:3]
        r_t1_1 = self.tag_body_position1.reshape((-1, 1))
        r_t2_2 = self.tag_body_position2.reshape((-1, 1))
        r_t1t2_a = (C_a1 @ r_t1_1 + r_1w_a) - (
            C_a2 @ r_t2_2 + r_2w_a)
        return np.array(np.linalg.norm(r_t1t2_a.flatten()))

    def jacobian(self, x: CompositeState) -> np.ndarray:
        x1 = x.get_state_by_id(self.state_id1)
        x2 = x.get_state_by_id(self.state_id2)

        if isinstance(x1, IMUState) and isinstance(x2, IMUState):
            _x1, _x2 = x1.value[0], x2.value[0]
        else:
            _x1, _x2 = x1, x2
        r_1w_a = _x1.value[0:3, -1].reshape((-1, 1))
        C_a1 = _x1.value[0:3, 0:3]
        r_2w_a = _x2.value[0:3, -1].reshape((-1, 1))
        C_a2 = _x2.value[0:3, 0:3]
        r_t1_1 = self.tag_body_position1.reshape((-1, 1))
        r_t2_2 = self.tag_body_position2.reshape((-1, 1))
        r_t1t2_a = (C_a1 @ r_t1_1 + r_1w_a) - (
            C_a2 @ r_t2_2 + r_2w_a)
        rho = (r_t1t2_a / np.linalg.norm(r_t1t2_a.flatten())
            ).reshape((-1, 1))
        
        if x1.direction == "right":
            jac1 = x1.jacobian_from_blocks(
                attitude=rho.T @ C_a1 @ SO3.odot(r_t1_1),
                position=rho.T @ C_a1,)

        if x2.direction == "right":
            jac2 = x2.jacobian_from_blocks(
                attitude=-rho.T @ C_a2 @ SO3.odot(r_t2_2),
                position=-rho.T @ C_a2,)

        return x.jacobian_from_blocks(
            {self.state_id1: jac1, self.state_id2: jac2})
    
    def evaluate_with_jacobian(self, x ) -> Union[np.ndarray, np.ndarray]:
        return self.evaluate(x), self.jacobian(x)

    def covariance(self, x: CompositeState) -> np.ndarray:
        return self._R


class RangePoseToAnchorById(CompositeMeasurementModel):
    """
    Range model given a pose of another body relative to current pose.
    """
    def __init__(
        self, anchor_position: np.ndarray, 
        tag_body_position: np.ndarray,state_id: Any, R: np.ndarray,):

        model = RangePoseToAnchor(anchor_position, tag_body_position, R)
        super(RangePoseToAnchorById, self).__init__(model, state_id)

class Altitude:
    """
    A model that returns that z component of a position vector.
    """
    def __init__(self, R: np.ndarray, minimum=None, bias=0.0):
        self.R = R # variance
        if minimum is None:
            minimum = -np.inf
        self.minimum = minimum # minimal height for the measurement to be valid
        self.bias = bias # fixed sensor bias

    def evaluate(self, x):

        if isinstance(x, IMUState):
            _x = x.value[0]
        else:
            _x = x
        position = _x.value[0:3, -1]
        h = position[2] + self.bias
        return h if h > self.minimum else None

    def jacobian(self, x):

        if isinstance(x, IMUState):
            _x = x.value[0]
        else:
            _x = x
        attitude = _x.value[0:3, 0:3]
        if x.direction == "right":
            return x.jacobian_from_blocks(
            position=attitude[2, :].reshape((1, -1)))
        
    def evaluate_with_jacobian(self, x) -> Union[np.ndarray, np.ndarray]:
        return self.evaluate(x), self.jacobian(x)

    def covariance(self, x) -> np.ndarray:
        return self.R

class AltitudeById(CompositeMeasurementModel):

    def __init__(
        self, R: np.ndarray, state_id: Any, minimum=None, bias=0.0,):
        model = Altitude(R, minimum, bias)
        super(AltitudeById, self).__init__(model, state_id)