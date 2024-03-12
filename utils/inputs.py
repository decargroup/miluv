from typing import List, Any
import numpy as np
import copy

class VectorInput:
    """
    Generic data container for timestamped information.
    """
    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        state_id: Any = None,
        covariance=None,
    ):
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)
        self.value = value
        self.dof = value.size
        self.stamp = stamp
        self.state_id = state_id
        self.covariance = covariance

    def plus(self, w: np.ndarray) -> "VectorInput":
        """
        Generic addition operation.
        w : np.ndarray
            to be added to the instance's .value
        """
        new = self.copy()
        og_shape = new.value.shape
        new.value = new.value.ravel() + w.ravel()
        new.value = new.value.reshape(og_shape)
        return new

    def copy(self) -> "VectorInput":
        return copy.deepcopy(self)

class IMU:
    """
    Data container for an IMU reading.
    """

    def __init__(
        self,
        gyro: np.ndarray,
        accel: np.ndarray,
        stamp: float,
        bias_gyro_walk: np.ndarray = [0, 0, 0],
        bias_accel_walk: np.ndarray = [0, 0, 0],
        state_id: Any = None,
        covariance: np.ndarray = None,
    ):
        self.dof = 12
        self.stamp = stamp
        self.state_id = state_id
        self.covariance = covariance

        self.gyro = np.array(gyro).ravel()
        self.accel = np.array(
            accel
        ).ravel()  #:np.ndarray: Accelerometer reading

        if bias_accel_walk is None:
            bias_accel_walk = np.zeros((3, 1))
        else:
            self.bias_gyro_walk = np.array(bias_gyro_walk).ravel()

        if bias_gyro_walk is None:
            bias_gyro_walk = np.zeros((3, 1))
        else:
            self.bias_accel_walk = np.array(bias_accel_walk).ravel()

        self.state_id = state_id

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
    
class CompositeInput:
    def __init__(self, input_list: List) -> None:
        self.input_list = input_list

    @property
    def dof(self) -> int:
        return sum([input.dof for input in self.input_list])

    @property
    def stamp(self) -> float:
        return self.input_list[0].stamp

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