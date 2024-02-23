from typing import List, Any
import numpy as np
from abc import ABC
from utils.states import (
    VectorState,
    SE23State,
    CompositeState,
)
from abc import ABC, abstractmethod

class Input(ABC):
    __slots__ = ["stamp", "dof", "covariance", "state_id"]
    """
    An abstract data container that holds a process model input value.
    """

    def __init__(
        self,
        dof: int,
        stamp: float = None,
        state_id: Any = None,
        covariance: np.ndarray = None,
    ):
        self.stamp = stamp  #:float: Timestamp
        self.dof = dof  #:int: Degrees of freedom of the object

        #:Any: Arbitrary optional identifier, possible to "assign" to a state.
        self.state_id = state_id

        #:np.ndarray: Covariance matrix of the object. Has shape (dof, dof)
        self.covariance = covariance

    @abstractmethod
    def plus(self, w: np.ndarray) -> "Input":
        """
        Generic addition operation to modify the internal value of the input,
        and return a new modified object.
        """
        pass

    @abstractmethod
    def copy(self) -> "Input":
        """
        Creates a deep copy of the object.
        """
        pass

class IMU(Input):
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
        super().__init__(dof=12, stamp=stamp, covariance=covariance)
        self.gyro = np.array(gyro).ravel()  #:np.ndarray: Gyro reading
        self.accel = np.array(
            accel
        ).ravel()  #:np.ndarray: Accelerometer reading

        if bias_accel_walk is None:
            bias_accel_walk = np.zeros((3, 1))
        else:
            #:np.ndarray: driving input for gyro bias random walk
            self.bias_gyro_walk = np.array(bias_gyro_walk).ravel()

        if bias_gyro_walk is None:
            bias_gyro_walk = np.zeros((3, 1))
        else:
            #:np.ndarray: driving input for accel bias random walk
            self.bias_accel_walk = np.array(bias_accel_walk).ravel()

        self.state_id = state_id  #:Any: State ID associated with the reading

    def plus(self, w: np.ndarray):
        """
        Modifies the IMU data. This is used to add noise to the IMU data.

        Parameters
        ----------
        w : np.ndarray with size 12
            w[0:3] is the gyro noise, w[3:6] is the accel noise,
            w[6:9] is the gyro bias walk noise, w[9:12] is the accel bias walk
            noise
        """
        new = self.copy()
        w = w.ravel()
        new.gyro = new.gyro + w[0:3]
        new.accel = new.accel + w[3:6]
        new.bias_gyro_walk = new.bias_gyro_walk + w[6:9]
        new.bias_accel_walk = new.bias_accel_walk + w[9:12]
        return new

    def copy(self):
        if self.covariance is None:
            cov_copy = None
        else:
            cov_copy = self.covariance.copy()
        return IMU(
            self.gyro.copy(),
            self.accel.copy(),
            self.stamp,
            self.bias_gyro_walk.copy(),
            self.bias_accel_walk.copy(),
            self.state_id,
            cov_copy,
        )

    def __repr__(self):
        s = [
            f"IMU(stamp={self.stamp}, state_id={self.state_id})",
            f"    gyro: {self.gyro.ravel()}",
            f"    accel: {self.accel.ravel()}",
        ]

        if np.any(self.bias_accel_walk) or np.any(self.bias_gyro_walk):
            s.extend(
                [
                    f"    gyro_bias_walk: {self.bias_gyro_walk.ravel()}",
                    f"    accel_bias_walk: {self.bias_accel_walk.ravel()}",
                ]
            )

        return "\n".join(s)


class IMUState(CompositeState):
    def __init__(
        self,
        nav_state: np.ndarray,
        bias_gyro: np.ndarray,
        bias_accel: np.ndarray,
        stamp: float = None,
        state_id: Any = None,
        direction="right",
    ):
        """
        Instantiate and IMUState object.

        Parameters
        ----------
        nav_state : np.ndarray with shape (5, 5)
            The navigation state stored as an element of SE_2(3).
            Contains orientation, velocity, and position.
        bias_gyro : np.ndarray with size 3
            Gyroscope bias
        bias_accel : np.ndarray with size 3
            Accelerometer bias
        stamp : float, optional
            Timestamp, by default None
        state_id : Any, optional
            Unique identifier, by default None
        direction : str, optional
            Direction of the perturbation for the nav state, by default "right"
        """
        nav_state = SE23State(nav_state, stamp, "pose", direction)
        bias_gyro = VectorState(bias_gyro, stamp, "gyro_bias")
        bias_accel = VectorState(bias_accel, stamp, "accel_bias")

        state_list = [nav_state, bias_gyro, bias_accel]
        super().__init__(state_list, stamp, state_id)

        # Just for type hinting
        self.value: List[SE23State, VectorState, VectorState] = self.value

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
        """Bias vector with in order [gyro_bias, accel_bias]"""
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

    @property
    def direction(self) -> str:
        return self.value[0].direction

    @direction.setter
    def direction(self, direction: str) -> None:
        self.value[0].direction = direction

    def copy(self):
        """
        Returns a new composite state object where the state values have also
        been copied.
        """
        return IMUState(
            self.nav_state.copy(),
            self.bias_gyro.copy(),
            self.bias_accel.copy(),
            self.stamp,
            self.state_id,
            self.direction,
        )

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
            attitude=attitude, position=position, velocity=velocity
        )
        if bias_gyro is None:
            bias_gyro = np.zeros((dim, 3))
        if bias_accel is None:
            bias_accel = np.zeros((dim, 3))

        return np.hstack([nav_jacobian, bias_gyro, bias_accel])

class UnBiasedIMU(Input):
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
        super().__init__(dof=12, stamp=stamp, covariance=covariance)
        self.gyro = np.array(gyro).ravel()  #:np.ndarray: Gyro reading
        self.accel = np.array(
            accel
        ).ravel()  #:np.ndarray: Accelerometer reading

        self.state_id = state_id  #:Any: State ID associated with the reading

    def plus(self, w: np.ndarray):
        """
        Modifies the IMU data. This is used to add noise to the IMU data.

        Parameters
        ----------
        w : np.ndarray with size 12
            w[0:3] is the gyro noise, w[3:6] is the accel noise,
        """
        new = self.copy()
        w = w.ravel()
        new.gyro = new.gyro + w[0:3]
        new.accel = new.accel + w[3:6]
        return new

    def copy(self):
        if self.covariance is None:
            cov_copy = None
        else:
            cov_copy = self.covariance.copy()
        return UnBiasedIMU(
            self.gyro.copy(),
            self.accel.copy(),
            self.stamp,
            self.state_id,
            cov_copy,
        )

    def __repr__(self):
        s = [
            f"UnBiasedIMU(stamp={self.stamp}, state_id={self.state_id})",
            f"    gyro: {self.gyro.ravel()}",
            f"    accel: {self.accel.ravel()}",
        ]

        return "\n".join(s)

    @staticmethod
    def random():
        return UnBiasedIMU(
            np.random.normal(size=3),
            np.random.normal(size=3),
            0.0,
        )
    

class CompositeInput(Input):
    # TODO: add tests to new methods
    def __init__(self, input_list: List[Input]) -> None:
        self.input_list = input_list

    @property
    def dof(self) -> int:
        return sum([input.dof for input in self.input_list])

    @property
    def stamp(self) -> float:
        return self.input_list[0].stamp

    def get_index_by_id(self, state_id):
        """
        Get index of a particular state_id in the list of inputs.
        """
        return [x.state_id for x in self.input_list].index(state_id)

    def add_input(self, input: Input):
        """
        Adds an input to the composite input.
        """
        self.input_list.append(input)

    def remove_input_by_id(self, state_id):
        """
        Removes a given input by ID.
        """
        idx = self.get_index_by_id(state_id)
        self.input_list.pop(idx)

    def get_input_by_id(self, state_id) -> Input:
        """
        Get input object by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.input_list[idx]

    def get_dof_by_id(self, state_id) -> int:
        """
        Get degrees of freedom of sub-input by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.input_list[idx].dof

    def get_stamp_by_id(self, state_id) -> float:
        """
        Get timestamp of sub-input by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.input_list[idx].stamp

    def set_stamp_by_id(self, stamp: float, state_id):
        """
        Set the timestamp of a sub-input by id.
        """
        idx = self.get_index_by_id(state_id)
        self.input_list[idx].stamp = stamp

    def set_input_by_id(self, input: Input, state_id):
        """
        Set the whole sub-input by id.
        """
        idx = self.get_index_by_id(state_id)
        self.input_list[idx] = input

    def set_stamp_for_all(self, stamp: float):
        """
        Set the timestamp of all subinputs.
        """
        for input in self.input_list:
            input.stamp = stamp

    def to_list(self):
        """
        Converts the CompositeInput object back into a list of inputs.
        """
        return self.input_list

    def copy(self) -> "CompositeInput":
        return CompositeInput([input.copy() for input in self.input_list])

    def plus(self, w: np.ndarray):
        new = self.copy()
        temp = w
        for i, input in enumerate(self.input_list):
            new.input_list[i] = input.plus(temp[: input.dof])
            temp = temp[input.dof :]

        return new