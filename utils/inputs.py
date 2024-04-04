from typing import List, Any
import numpy as np
import copy

"""This module contains classes for the following:
- Inputs
    - VectorInput
    - CompositeInput
"""

class VectorInput:
    # Generic data container for timestamped information.
    def __init__(self, value: np.ndarray, stamp: float = None,
        state_id: Any = None, covariance=None,):
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)
        self.value = value
        self.dof = value.size
        self.stamp = stamp
        self.state_id = state_id
        self.covariance = covariance

    def plus(self, w: np.ndarray) -> "VectorInput":
        new = self.copy()
        og_shape = new.value.shape
        new.value = new.value.ravel() + w.ravel()
        new.value = new.value.reshape(og_shape)
        return new

    def copy(self) -> "VectorInput":
        return copy.deepcopy(self)
    
class CompositeInput:
    def __init__(self, input_list: List):
        self.input_list = input_list
        self.stamp = input_list[0].stamp
        self.dof = sum([input.dof for input in input_list])

    def get_input_by_id(self, state_id):
        idx = [x.state_id for x in self.input_list].index(state_id)
        return self.input_list[idx]

    def copy(self) -> "CompositeInput":
        return CompositeInput([input.copy() for input in self.input_list])

    def plus(self, w: np.ndarray):
        new = self.copy()
        temp = w
        for i, input in enumerate(self.input_list):
            new.input_list[i] = input.plus(temp[: input.dof])
            temp = temp[input.dof :]
        return new