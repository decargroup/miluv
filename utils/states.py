import numpy as np
from pymlg import SE3, SE23
from typing import List
import copy

class SE3State:
    group = SE3
    
    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        state_id=None,
        direction="right",
    ):
        if isinstance(value, list):
            value = np.array(value)

        if value.size == self.group.dof:
            value = self.group.Exp(value)
        elif value.shape[0] != value.shape[1]:
            raise ValueError(
                f"value must either be a {self.group.dof}-length vector of exponential"
                "coordinates or a matrix direct element of the group."
            )
        
        self.direction = direction
        self.value: np.ndarray = value
        self.dof = self.group.dof
        self.stamp = stamp
        self.state_id = (state_id) # Any

    @property
    def attitude(self) -> np.ndarray:
        return self.value[0:3, 0:3]

    @attitude.setter
    def attitude(self, C):
        self.value[0:3, 0:3] = C

    @property
    def position(self) -> np.ndarray:
        return self.value[0:3, 3]

    @position.setter
    def position(self, r):
        self.value[0:3, 3] = r
    
    def copy(self):
        return copy.deepcopy(self)
    
    def plus(self, dx: np.ndarray):
        new = self.copy()
        if self.direction == "right":
            new.value = self.value @ self.group.Exp(dx)
        elif self.direction == "left":
            raise NotImplementedError("Left perturbation not implemented.")
        return new

    def minus(self, x) -> np.ndarray:
        if self.direction == "right":
            diff = self.group.Log(self.group.inverse(x.value) @ self.value)
        elif self.direction == "left":
            raise NotImplementedError("Left perturbation not implemented.")
        return diff.ravel()
    
    @staticmethod
    def jacobian_from_blocks(
        attitude: np.ndarray = None, position: np.ndarray = None
    ):
        for jac in [attitude, position]:
            if jac is not None:
                dim = jac.shape[0]

        if attitude is None:
            attitude = np.zeros((dim, 3))
        if position is None:
            position = np.zeros((dim, 3))

        return np.block([attitude, position])

    def dot(self, other):
        new = self.copy()
        new.value = self.value @ other.value
        return new

class SE23State:
    group = SE23

    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        state_id=None,
        direction="right",
    ):
        if isinstance(value, list):
            value = np.array(value)

        if value.size == self.group.dof:
            value = self.group.Exp(value)
        elif value.shape[0] != value.shape[1]:
            raise ValueError(
                f"value must either be a {self.group.dof}-length vector of exponential"
                "coordinates or a matrix direct element of the group."
            )
    
        self.direction = direction
        self.value: np.ndarray = value
        self.dof = self.group.dof
        self.stamp = stamp
        self.state_id = (state_id) # Any

    @property
    def pose(self) -> np.ndarray:
        return self.value[0:5, 0:5]

    @pose.setter
    def pose(self, T):
        self.value[0:5, 0:5] = T

    @property
    def attitude(self) -> np.ndarray:
        return self.value[0:3, 0:3]

    @attitude.setter
    def attitude(self, C):
        self.value[0:3, 0:3] = C

    @property
    def position(self) -> np.ndarray:
        return self.value[0:3, 4]

    @position.setter
    def position(self, r):
        self.value[0:3, 4] = r.ravel()

    @property
    def velocity(self) -> np.ndarray:
        return self.value[0:3, 3]

    @velocity.setter
    def velocity(self, v) -> np.ndarray:
        self.value[0:3, 3] = v
    
    def copy(self):
        return copy.deepcopy(self)
    
    def plus(self, dx: np.ndarray):
        new = self.copy()
        if self.direction == "right":
            new.value = self.value @ self.group.Exp(dx)
        elif self.direction == "left":
            raise NotImplementedError("Left perturbation not implemented.")
        return new

    def minus(self, x) -> np.ndarray:
        if self.direction == "right":
            diff = self.group.Log(self.group.inverse(x.value) @ self.value)
        elif self.direction == "left":
            raise NotImplementedError("Left perturbation not implemented.")
        return diff.ravel()

    @staticmethod
    def jacobian_from_blocks(
        attitude: np.ndarray = None,
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
        if velocity is None:
            velocity = np.zeros((dim, 3))

        return np.block([attitude, velocity, position])
    
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
        return VectorState(self.value.copy(), self.stamp, self.state_id)

class CompositeState:
    """
    A "composite" state object intended to hold a list of poses
    as a single state at a specific time.
    """

    def __init__(
        self, state_list: List, stamp: float = None, state_id=None
    ):
        #:List[State]: The substates are the CompositeState's value.
        self.value = state_list
        self.stamp = stamp
        self.state_id = state_id

    @property
    def dof(self):
        return sum([x.dof for x in self.value])

    def get_slices(self) -> List[slice]:
        """
        Get slices for each state in the list of states.
        """
        slices = []
        counter = 0
        for state in self.value:
            slices.append(slice(counter, counter + state.dof))
            counter += state.dof

        return slices

    def get_slice_by_id(self, state_id, slices=None):
        """
        Get slice of a particular state_id in the list of states.
        """

        if slices is None:
            slices = self.get_slices()

        idx = [x.state_id for x in self.value].index(state_id)
        return slices[idx]

    def get_state_by_id(self, state_id):
        idx = [x.state_id for x in self.value].index(state_id)
        return self.value[idx]

    def copy(self) -> "CompositeState":
        return self.__class__(
            [state.copy() for state in self.value], self.stamp, self.state_id
        )

    def plus(self, dx, new_stamp: float = None) -> "CompositeState":
        """
        Updates the value of each sub-state given a dx. Interally parses
        the dx vector.
        """
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
                self.value[i].minus(x.value[i]).reshape((self.value[i].dof,))
            )

        return np.concatenate(dx).reshape((-1, 1))

    def jacobian_from_blocks(self, block_dict: dict):
        """
        Returns the jacobian of the entire composite state given jacobians
        associated with some of the substates. These are provided as a dictionary
        with the the keys being the substate IDs.
        """
        block: np.ndarray = list(block_dict.values())[0]
        m = block.shape[0]  # Dimension of "y" value
        jac = np.zeros((m, self.dof))
        slices = self.get_slices()
        for state_id, block in block_dict.items():
            slc = self.get_slice_by_id(state_id, slices)
            jac[:, slc] = block

        return jac


class IMUState(CompositeState):
    def __init__(
        self,
        nav_state: np.ndarray,
        bias_gyro: np.ndarray,
        bias_accel: np.ndarray,
        stamp: float = None,
        state_id = None,
        direction="right",
    ):
        nav_state = SE23State(nav_state, stamp, "pose", direction)
        bias_gyro = VectorState(bias_gyro, stamp, "gyro_bias")
        bias_accel = VectorState(bias_accel, stamp, "accel_bias")

        state_list = [nav_state, bias_gyro, bias_accel]
        super().__init__(state_list, stamp, state_id)

        # Just for type hinting
        self.value: List[SE23State, VectorState, VectorState] = self.value
        self.attitude = self.value[0].attitude
        self.position = self.value[0].position
        self.velocity = self.value[0].velocity
        self.bias_gyro = self.value[1].value

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
        return copy.deepcopy(self)

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

class StateWithCovariance:
    """
    A data container containing a State object and a covariance array.
    """

    __slots__ = ["state", "covariance"]

    def __init__(self, state, covariance: np.ndarray):
        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be an n x n array.")
        
        if covariance.shape[0] != state.dof:
            raise ValueError(
                "Covariance matrix does not correspond with state DOF."
            )

        self.state = state
        self.covariance = covariance

    @property
    def stamp(self):
        return self.state.stamp

    @stamp.setter
    def stamp(self, stamp):
        self.state.stamp = stamp

    def symmetrize(self):
        # Enforces symmetry of the covariance matrix.
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def copy(self) -> "StateWithCovariance":
        return StateWithCovariance(self.state.copy(), self.covariance.copy())