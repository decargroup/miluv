import numpy as np
from pymlg import SE2, SE23
from pymlg import MatrixLieGroup
from abc import ABC, abstractmethod
from typing import Any, List

class State(ABC):
    """
    An abstract state :math:`\mathbf{x}` is an object containing the following attributes:

        - a value of some sort;
        - a certain number of degrees of freedom (dof);
        - an update rule that modified the state value given an update vector
          ``dx`` containing ``dof`` elements.

    Optionally, it is often useful to assign a timestamp (``stamp``) and a label
    (``state_id``) to differentiate state instances from others.
    """

    __slots__ = ["value", "dof", "stamp", "state_id"]

    def __init__(
        self, value: Any, dof: int, stamp: float = None, state_id=None
    ):
        self.value = value  #:Any: State value
        self.dof = dof  #:int: Degree of freedom of the state
        self.stamp = stamp  #:float: Timestamp
        self.state_id = (
            state_id  #:Any: Some identifier associated with the state
        )

    @abstractmethod
    def plus(self, dx: np.ndarray) -> "State":
        """
        A generic "addition" operation given a ``dx`` numpy array with as many
        elements as the ``dof`` of this state.
        """
        pass

    @abstractmethod
    def minus(self, x: "State") -> np.ndarray:
        """
        A generic "subtraction" operation given another State object of the same
        type, always returning a numpy array.
        """
        pass

    @abstractmethod
    def copy(self) -> "State":
        """
        Returns a copy of this State instance.
        """
        pass

    def __repr__(self):
        value_str = str(self.value).split("\n")
        value_str = "\n".join(["    " + s for s in value_str])
        s = [
            f"{self.__class__.__name__}(stamp={self.stamp}, dof={self.dof}, state_id={self.state_id})",
            f"{value_str}",
        ]
        return "\n".join(s)


class MatrixLieGroupState(State):
    """
    The MatrixLieGroupState class. Although this class can be used directly,
    it is recommended to use one of the subclasses, such as SE2State or SO3State.
    """

    # TODO. Add identity() and random() functions to this
    __slots__ = ["group", "direction"]

    def __init__(
        self,
        value: np.ndarray,
        group: MatrixLieGroup,
        stamp: float = None,
        state_id: Any = None,
        direction="right",
    ):
        """
        Parameters
        ----------
        value : np.ndarray
            Value of of the state. If the value has as many elements as the
            DOF of the group, then it is assumed to be a vector of exponential
            coordinates. Otherwise, the value must be a 2D numpy array representing
            a direct element of the group in matrix form.
        group : MatrixLieGroup
            A `pymlg.MatrixLieGroup` class, such as `pymlg.SE2` or `pymlg.SO3`.
        stamp : float, optional
            timestamp, by default None
        state_id : Any, optional
            optional state ID, by default None
        direction : str, optional
            either "left" or "right", by default "right". Defines the perturbation
            :math:`\\delta \mathbf{x}` as either

            .. math::
                \mathbf{X} = \mathbf{X} \exp(\delta \mathbf{x}^\wedge) \text{ (right)}

                \mathbf{X} = \exp(\delta \mathbf{x}^\wedge) \mathbf{X} \text{ (left)}
        """
        if isinstance(value, list):
            value = np.array(value)

        if value.size == group.dof:
            value = group.Exp(value)
        elif value.shape[0] != value.shape[1]:
            raise ValueError(
                f"value must either be a {group.dof}-length vector of exponential"
                "coordinates or a matrix direct element of the group."
            )

        self.direction = direction
        self.group = group
        super(MatrixLieGroupState, self).__init__(
            value, self.group.dof, stamp, state_id
        )
        self.value: np.ndarray = self.value  # just for type hinting

    def plus(self, dx: np.ndarray) -> "MatrixLieGroupState":
        new = self.copy()
        if self.direction == "right":
            new.value = self.value @ self.group.Exp(dx)
        elif self.direction == "left":
            new.value = self.group.Exp(dx) @ self.value
        else:
            raise ValueError("direction must either be 'left' or 'right'.")
        return new

    def minus(self, x: "MatrixLieGroupState") -> np.ndarray:
        if self.direction == "right":
            diff = self.group.Log(self.group.inverse(x.value) @ self.value)
        elif self.direction == "left":
            diff = self.group.Log(self.value @ self.group.inverse(x.value))
        else:
            raise ValueError("direction must either be 'left' or 'right'.")
        return diff.ravel()

    def copy(self) -> "MatrixLieGroupState":
        ## Check if instance of this class as opposed to a child class
        if type(self) == MatrixLieGroupState:
            return MatrixLieGroupState(
                self.value.copy(),
                self.group,
                self.stamp,
                self.state_id,
                self.direction,
            )
        else:
            return self.__class__(
                self.value.copy(),
                self.stamp,
                self.state_id,
                self.direction,
            )

    def plus_jacobian(self, dx: np.ndarray) -> np.ndarray:
        if self.direction == "right":
            jac = self.group.right_jacobian(dx)
        elif self.direction == "left":
            jac = self.group.left_jacobian(dx)
        else:
            raise ValueError("direction must either be 'left' or 'right'.")
        return jac

    def minus_jacobian(self, x: "MatrixLieGroupState") -> np.ndarray:
        dx = self.minus(x)
        if self.direction == "right":
            jac = self.group.right_jacobian_inv(dx)
        elif self.direction == "left":
            jac = self.group.left_jacobian_inv(dx)
        else:
            raise ValueError("direction must either be 'left' or 'right'.")
        return jac

    def __repr__(self):
        value_str = str(self.value).split("\n")
        value_str = "\n".join(["    " + s for s in value_str])
        s = [
            f"{self.__class__.__name__}(stamp={self.stamp},"
            + f" state_id={self.state_id}, direction={self.direction})",
            f"{value_str}",
        ]
        return "\n".join(s)

    def dot(self, other: "MatrixLieGroupState") -> "MatrixLieGroupState":
        new = self.copy()
        new.value = self.value @ other.value
        return new

    @property
    def attitude(self) -> np.ndarray:
        raise NotImplementedError(
            "{0} does not have attitude property".format(
                self.__class__.__name__
            )
        )

    @property
    def position(self) -> np.ndarray:
        raise NotImplementedError(
            "{0} does not have position property".format(
                self.__class__.__name__
            )
        )

    @property
    def velocity(self) -> np.ndarray:
        raise NotImplementedError(
            "{0} does not have velocity property".format(
                self.__class__.__name__
            )
        )

    def jacobian_from_blocks(self, **kwargs) -> np.ndarray:
        raise NotImplementedError()
    
class VectorState(State):
    """
    A standard vector-based state, with value represented by a 1D numpy array.
    """

    def __init__(self, value: np.ndarray, stamp: float = None, state_id=None):
        value = np.array(value, dtype=np.float64).ravel()
        super(VectorState, self).__init__(
            value=value,
            dof=value.size,
            stamp=stamp,
            state_id=state_id,
        )
        self.value: np.ndarray = self.value

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

    def plus_jacobian(self, dx: np.ndarray) -> np.ndarray:
        return np.identity(self.dof)

    def minus_jacobian(self, x: State) -> np.ndarray:
        return np.identity(self.dof)

    def copy(self) -> "VectorState":
        return VectorState(self.value.copy(), self.stamp, self.state_id)
        

class SE23State(MatrixLieGroupState):
    group = SE23

    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        state_id=None,
        direction="right",
    ):
        super().__init__(value, self.group, stamp, state_id, direction)

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

    @staticmethod
    def random(stamp: float = None, state_id=None, direction="right"):
        return SE23State(
            SE23.random(), stamp=stamp, state_id=state_id, direction=direction
        )

class CompositeState(State):
    """
    A "composite" state object intended to hold a list of State objects as a
    single conceptual "state". The intended use is to hold a list of poses
    as a single state at a specific time.

    Parameters
    ----------
    state_list: List[State]
        List of State that forms this composite state


    Each state in the provided list has an index (the index in the list), as
    well as a state_id, which is found as an attribute in the corresponding State
    object.

    It is possible to access sub-states in the composite states both by index
    and by ID.
    """

    def __init__(
        self, state_list: List[State], stamp: float = None, state_id=None
    ):
        #:List[State]: The substates are the CompositeState's value.
        self.value = state_list

        self.stamp = stamp
        self.state_id = state_id

    def __getstate__(self):
        """
        Get the state of the object for pickling.
        """
        return (
            None,
            {
                "value": self.value,
                "stamp": self.stamp,
                "state_id": self.state_id,
            },
        )

    def __setstate__(self, attributes):
        """
        Set the state of the object for unpickling.
        """

        attributes = attributes[1]
        self.value = attributes["value"]
        self.stamp = attributes["stamp"]
        self.state_id = attributes["state_id"]

    @property
    def dof(self):
        return sum([x.dof for x in self.value])

    def get_index_by_id(self, state_id):
        """
        Get index of a particular state_id in the list of states.
        """
        return [x.state_id for x in self.value].index(state_id)

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

    def add_state(self, state: State, stamp: float = None, state_id=None):
        """Adds a state and it's corresponding slice to the composite state."""
        self.value.append(state)

    def remove_state_by_id(self, state_id):
        """Removes a given state by ID."""
        idx = self.get_index_by_id(state_id)
        self.value.pop(idx)

    def get_slice_by_id(self, state_id, slices=None):
        """
        Get slice of a particular state_id in the list of states.
        """

        if slices is None:
            slices = self.get_slices()

        idx = self.get_index_by_id(state_id)
        return slices[idx]

    def get_matrix_block_by_ids(
        self, mat: np.ndarray, state_id_1: Any, state_id_2: Any = None
    ) -> np.ndarray:
        """Gets the portion of a matrix corresponding to two states.

        This function is useful when extract specific blocks of a covariance
        matrix, for example.

        Parameters
        ----------
        mat : np.ndarray
            N x N matrix
        state_id_1 : Any
            State ID of state 1.
        state_id_2 : Any, optional
            State ID of state 2. If None, state_id_2 is set to state_id_1.

        Returns
        -------
        np.ndarray
            Subblock of mat corrsponding to
            slices of state_id_1 and state_id_2.
        """

        if state_id_2 is None:
            state_id_2 = state_id_1

        slice_1 = self.get_slice_by_id(state_id_1)
        slice_2 = self.get_slice_by_id(state_id_2)

        return mat[slice_1, slice_2]

    def set_matrix_block_by_ids(
        self,
        new_mat_block: np.ndarray,
        mat: np.ndarray,
        state_id_1: Any,
        state_id_2: Any = None,
    ) -> np.ndarray:
        """Sets the portion of the covariance block corresponding to two states.

        Parameters
        ----------
        new_mat_block : np.ndarray
            A subblock to be entered into mat.
        mat : np.ndarray
            Full matrix.
        state_id_1 : Any
            State ID of state 1.
        state_id_2 : Any, optional
            State ID of state 2. If None, state_id_2 is set to state_id_1.

        Returns
        -------
        np.ndarray
            mat with updated subblock.
        """

        if state_id_2 is None:
            state_id_2 = state_id_1

        slice_1 = self.get_slice_by_id(state_id_1)
        slice_2 = self.get_slice_by_id(state_id_2)

        mat[slice_1, slice_2] = new_mat_block
        return mat

    def get_value_by_id(self, state_id) -> Any:
        """
        Get state value by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.value[idx].value

    def get_state_by_id(self, state_id) -> State:
        """
        Get state object by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.value[idx]

    def get_dof_by_id(self, state_id) -> int:
        """
        Get degrees of freedom of sub-state by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.value[idx].dof

    def get_stamp_by_id(self, state_id) -> float:
        """
        Get timestamp of sub-state by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.value[idx].stamp

    def set_stamp_by_id(self, stamp: float, state_id):
        """
        Set the timestamp of a sub-state by id.
        """
        idx = self.get_index_by_id(state_id)
        self.value[idx].stamp = stamp

    def set_state_by_id(self, state: State, state_id):
        """
        Set the whole sub-state by id.
        """
        idx = self.get_index_by_id(state_id)
        self.value[idx] = state

    def set_value_by_id(self, value: Any, state_id: Any):
        """
        Set the value of a sub-state by id.
        """
        idx = self.get_index_by_id(state_id)
        self.value[idx].value = value

    def set_stamp_for_all(self, stamp: float):
        """
        Set the timestamp of all substates.
        """
        for state in self.value:
            state.stamp = stamp

    def to_list(self):
        """
        Converts the CompositeState object back into a list of states.
        """
        return self.value

    def copy(self) -> "CompositeState":
        """
        Returns a new composite state object where the state values have also
        been copied.
        """
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

    def plus_by_id(
        self, dx, state_id: int, new_stamp: float = None
    ) -> "CompositeState":
        """
        Updates a specific sub-state.
        """
        new = self.copy()
        idx = new.get_index_by_id(state_id)
        new.value[idx].plus(dx)
        if new_stamp is not None:
            new.set_stamp_by_id(new_stamp, state_id)

        return new

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

    def plus_jacobian(self, dx: np.ndarray) -> np.ndarray:
        dof = self.dof
        jac = np.zeros((dof, dof))
        counter = 0
        for state in self.value:
            jac[
                counter : counter + state.dof,
                counter : counter + state.dof,
            ] = state.plus_jacobian(dx[: state.dof])
            dx = dx[state.dof :]
            counter += state.dof

        return jac

    def minus_jacobian(self, x: "CompositeState") -> np.ndarray:
        dof = self.dof
        jac = np.zeros((dof, dof))
        counter = 0
        for i, state in enumerate(self.value):
            jac[
                counter : counter + state.dof,
                counter : counter + state.dof,
            ] = state.minus_jacobian(x.value[i])
            counter += state.dof

        return jac

    def __repr__(self):
        substate_line_list = []
        for v in self.value:
            substate_line_list.extend(v.__repr__().split("\n"))
        substates_str = "\n".join(["    " + s for s in substate_line_list])
        s = [
            f"{self.__class__.__name__}(stamp={self.stamp}, state_id={self.state_id}) with substates:",
            substates_str,
        ]
        return "\n".join(s)


class StateWithCovariance:
    """
    A data container containing a State object and a covariance array.
    """

    __slots__ = ["state", "covariance"]

    def __init__(self, state: State, covariance: np.ndarray):
        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be an n x n array.")

        if covariance.shape[0] != state.dof:
            raise ValueError(
                "Covariance matrix does not correspond with state DOF."
            )

        #:navlie.types.State: state object
        self.state = state

        #:numpy.ndarray: covariance associated with state
        self.covariance = covariance

    @property
    def stamp(self):
        return self.state.stamp

    @stamp.setter
    def stamp(self, stamp):
        self.state.stamp = stamp

    def symmetrize(self):
        """
        Enforces symmetry of the covariance matrix.
        """
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def copy(self) -> "StateWithCovariance":
        return StateWithCovariance(self.state.copy(), self.covariance.copy())

    def __repr__(self):
        return f"StateWithCovariance(stamp={self.stamp})"