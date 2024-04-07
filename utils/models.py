import numpy as np
from typing import List, Any
from utils.states import CompositeState
from utils.imu import IMUState
    
class MeasurementModel:

    def jacobian(self, x: np.ndarray, step_size = 1e-6) -> np.ndarray:
        """
        Calculates the model jacobian with finite difference.
        """
        N = x.dof
        y = self.evaluate(x)
        if y is None:
            return None
        m = y.size
        jac_fd = np.zeros((m, N))
        for i in range(N):
            dx = np.zeros((N, 1))
            dx[i, 0] = step_size
            x_temp = x.plus(dx)
            jac_fd[:, i] = (self.evaluate(x_temp) - y).flatten() / step_size
        return jac_fd
    
    def evaluate_with_jacobian(self, x: np.ndarray):
        return self.evaluate(x), self.jacobian(x)

class RangePoseToAnchor(MeasurementModel):
    """ 
    Range measurement from a pose state to an anchor 
    """
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

    def covariance(self, x) -> np.ndarray:
        return self._R

class RangePoseToPose(MeasurementModel):
    """ 
    Range model given two absolute poses of rigid bodies, each containing a tag 
    """
    def __init__(
        self, tag_body_position1, tag_body_position2, state_id1, state_id2, R
    ):
        self.r_t1_1 = np.array(tag_body_position1).reshape((-1, 1))
        self.r_t2_2 = np.array(tag_body_position2).reshape((-1, 1))
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
        r_t1t2_a = (C_a1 @ self.r_t1_1 + r_1w_a) - (
            C_a2 @ self.r_t2_2 + r_2w_a)
        return np.array(np.linalg.norm(r_t1t2_a.flatten()))

    def covariance(self, x: CompositeState) -> np.ndarray:
        return self._R

class RangePoseToAnchorById(MeasurementModel):
    """
    Range model given a pose of another body relative to current pose.
    """
    def __init__(
        self, anchor_position: np.ndarray, 
        tag_body_position: np.ndarray,state_id: Any, R: np.ndarray,):
        self.model = RangePoseToAnchor(anchor_position, 
                                       tag_body_position, R)
        self.state_id = state_id
    
    def evaluate(self, x: CompositeState) -> np.ndarray:
        return self.model.evaluate(x.get_state_by_id(self.state_id))

    def covariance(self, x: CompositeState) -> np.ndarray:
        return self.model.covariance(x.get_state_by_id(self.state_id))

class Altitude(MeasurementModel):
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
            h = x.value[0].value[3, -1] + self.bias
        else:
            h = x.value[3, -1] + self.bias
        return h if h > self.minimum else None

    def covariance(self, x) -> np.ndarray:
        return self.R

class AltitudeById(MeasurementModel):

    def __init__(
        self, R: np.ndarray, state_id: Any, minimum=None, bias=0.0,):
        self.model = Altitude(R, minimum, bias)
        self.state_id = state_id

    def evaluate(self, x: CompositeState) -> np.ndarray:
        return self.model.evaluate(x.get_state_by_id(self.state_id))
    
    def covariance(self, x: CompositeState) -> np.ndarray:
        return self.model.covariance(x.get_state_by_id(self.state_id))