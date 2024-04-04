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
        if dx.size == self.dof:
            new.value = new.value.ravel() + dx.ravel()
            return new
        else:
            raise ValueError("Array of mismatched size added to VectorState.")

    def minus(self, x: "VectorState") -> np.ndarray:
        og_shape = self.value.shape
        return (self.value.ravel() - x.value.ravel()).reshape(og_shape)

    def copy(self) -> "VectorState":
        return copy.deepcopy(self)

class MatrixLieGroupState:
    
    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        state_id=None,
        direction="right",
    ):
        
        self.direction = direction
        self.value: np.ndarray = value
        self.stamp = stamp
        self.state_id = (state_id) # Any

        if value.shape == (4, 4):
            self.group = SE3
        elif value.shape == (5, 5):
            self.group = SE23
        
        self.dof = self.group.dof
    
    def copy(self):
        return copy.deepcopy(self)
    
    def plus(self, dx: np.ndarray):
        new = self.copy()
        if self.direction == "right":
            new.value = self.value @ self.group.Exp(dx)
        return new

    def minus(self, x) -> np.ndarray:
        if self.direction == "right":
            diff = self.group.Log(self.group.inverse(x.value) @ self.value)
        return diff.ravel()

    def jacobian_from_blocks(
        self, attitude: np.ndarray = None,
        position: np.ndarray = None, 
        velocity: np.ndarray = None,
    ):
        for jac in [attitude, position, velocity]:
            if jac is not None:
                dim = jac.shape[0]
            
        if attitude is None:
            attitude = np.zeros((dim, 3))
        if position is None:
            position = np.zeros((dim, 3))

        if self.group == SE3:
            return np.block([attitude, position])
        
        elif self.group == SE23:
            if velocity is None:
                velocity = np.zeros((dim, 3))
            return np.block([attitude, velocity, position])

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

    def plus(self, dx, new_stamp: float = None) -> "CompositeState":
        # Updates the value of each sub-state given a dx.
        new = self.copy()
        for i, state in enumerate(new.value):
            new.value[i] = state.plus(dx[: state.dof])
            dx = dx[state.dof :]
        if new_stamp is not None:
            new.set_stamp_for_all(new_stamp)
        return new

    def minus(self, x: "CompositeState") -> np.ndarray:
        dx = []
        for i, v in enumerate(x.value):
            dx.append(
            self.value[i].minus(x.value[i]).reshape((self.value[i].dof,)))
        return np.concatenate(dx).reshape((-1, 1))

    def jacobian_from_blocks(self, block_dict: dict):
        """
        Returns the jacobian of the entire composite state given jacobians
        associated with some of the substates.
        """
        block: np.ndarray = list(block_dict.values())[0]
        m = block.shape[0]  # Dimension of "y" value
        jac = np.zeros((m, self.dof))
        slices = self.get_slices()
        for state_id, block in block_dict.items():
            slc = self.get_slice_by_id(state_id, slices)
            jac[:, slc] = block
        return jac

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