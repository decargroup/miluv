import numpy as np
from typing import Any, List
from utils.models import (
    MeasurementModel,
    RangeRelativePose,
    RangePoseToPose,
)

class Measurement:
    """
    A data container containing a generic measurement's value, timestamp,
    and corresponding model.
    """

    __slots__ = ["value", "stamp", "model", "state_id"]

    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        model: MeasurementModel = None,
        state_id: Any = None,
    ):
        """
        Parameters
        ----------
        value : np.ndarray
            the value of the measurement reading
        stamp : float, optional
            timestamp, by default None
        model : MeasurementModel, optional
            model for this measurement, by default None
        state_id : Any, optional
            optional state ID, by default None
        """
        #:numpy.ndarray: Container for the measurement value
        self.value = np.array(value) if np.isscalar(value) else value
        #:float: Timestamp
        self.stamp = stamp
        # MeasurementModel: measurement model associated with this measurement.
        self.model = model
        #:Any: Optional, ID of the state this measurement is associated.
        self.state_id = state_id

    def minus(self, y_check: np.ndarray) -> np.ndarray:
        """Evaluates the difference between the current measurement
        and a predicted measurement.

        By default, assumes that the measurement is a column vector,
        and thus, the ``minus`` operator is simply vector subtraction.
        """

        return self.value.reshape((-1, 1)) - y_check.reshape((-1, 1))