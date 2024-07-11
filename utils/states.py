from typing import List
import numpy as np
import copy
from pymlg import SE3, SE23

class VectorState:
    """
    A standard vector-based state, with value represented by a 1D numpy array.
    """
    def __init__(self, value: np.ndarray, stamp: float = None, state_id=None):
        value = np.array(value, dtype=np.float64).ravel()
        self.value: np.ndarray = value
        self.dof = value.size
        self.stamp = stamp
        self.state_id = state_id

    def plus(self, dx: np.ndarray) -> "VectorState":
        new = self.copy()
        new.value = new.value.ravel() + dx.ravel()
        return new

    def minus(self, x: "VectorState") -> np.ndarray:
        return (self.value.ravel() - x.value.ravel()).reshape(self.value.shape)

    def copy(self) -> "VectorState":
        return copy.deepcopy(self)

class MatrixLieGroupState:
    
    def __init__(self, value: np.ndarray,
        stamp: float = None, state_id=None):
        self.value: np.ndarray = value
        self.stamp = stamp
        self.state_id = state_id
        if value.shape == (4, 4):
            self.group = SE3
        elif value.shape == (5, 5):
            self.group = SE23
        self.dof = self.group.dof
    
    def copy(self):
        return copy.deepcopy(self)
    
    def plus(self, dx: np.ndarray):
        new = self.copy()
        new.value = self.value @ self.group.Exp(dx)
        return new

    def minus(self, x) -> np.ndarray:
        return self.group.Log(self.group.inverse(x.value) @ self.value).ravel()

class CompositeState:
    """
    A "composite" state object intended to hold a list of poses
    as a single state.
    """
    def __init__(
        self, state_list: List, stamp: float = None, state_id=None):
        self.value = state_list
        self.stamp = stamp
        self.state_id = state_id
        self.dof = sum([x.dof for x in self.value])

    def get_slices(self) -> List[slice]:
        slices = []
        counter = 0
        for state in self.value:
            slices.append(slice(counter, counter + state.dof))
            counter += state.dof
        return slices
    
    def get_slice_by_id(self, state_id, slices=None):
        # Get slice of a particular state_id in the list of states.
        if slices is None:
            slices = self.get_slices()
        idx = [x.state_id for x in self.value].index(state_id)
        return slices[idx]

    def get_state_by_id(self, state_id):
        idx = [x.state_id for x in self.value].index(state_id)
        return self.value[idx]

    def copy(self) -> "CompositeState":
        return copy.deepcopy(self)

    def plus(self, dx) -> "CompositeState":
        new = self.copy()
        new.value = [x.plus(dx[s]) for x, s in zip(self.value, self.get_slices())]
        return new

    def minus(self, x: "CompositeState") -> np.ndarray:
        dx = [self.value[i].minus(x.value[i]) for i in range(len(self.value))]
        return np.concatenate(dx).reshape((-1, 1))

class IMUState(CompositeState):
    def __init__(self, SE23_state: np.ndarray, bias_gyro: np.ndarray,
        bias_accel: np.ndarray, stamp: float = None, state_id = None,):
        SE23_state = MatrixLieGroupState(SE23_state, stamp, "pose")
        bias_gyro = VectorState(bias_gyro, stamp, "gyro_bias")
        bias_accel = VectorState(bias_accel, stamp, "accel_bias")
        state_list = [SE23_state, bias_gyro, bias_accel]
        super().__init__(state_list, stamp, state_id)

class StateWithCovariance:
    # A data container containing a State object and a covariance array.
    def __init__(self, state, covariance: np.ndarray):
        self.state = state
        self.covariance = covariance
        self.stamp = state.stamp

    def symmetrize(self):
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def copy(self) -> "StateWithCovariance":
        return StateWithCovariance(self.state.copy(), self.covariance.copy())