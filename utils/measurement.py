import numpy as np
from typing import Any, List
from utils.models import (
    MeasurementModel,
    RangeRelativePose,
    RangePoseToPose,
    TagHolder,
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

class RangeData:
    """
    Class for storing UWB range data.
    """
    def __init__(
        self,
        stamps,
        range,
        from_id,
        to_id,
        covariance,
        std,
        bias,
    ):
        self.stamps = np.array(stamps).ravel()
        self.range = np.array(range).ravel()
        self.from_id = np.array(from_id).ravel()
        self.to_id = np.array(to_id).ravel()
        self.covariance = np.array(covariance).ravel()
        self.std = np.array(std).ravel()
        self.bias = np.array(bias).ravel()
    
    def copy(self):
        return RangeData(self.stamps.copy(),
                                   self.range.copy(),
                                   self.from_id.copy(),
                                   self.to_id.copy(),
                                   self.covariance.copy(),
                                   self.std.copy(),
                                   self.bias.copy())
    
    def filter_by_bias(self, 
                       max_bias:float,
                       min_bias :float = None) -> "RangeData":

        if min_bias is None:
            min_bias = - np.inf
        match_mask = np.logical_and(
            abs(self.bias) > min_bias, abs(self.bias) < max_bias
        )
        out = self.copy()
        out.stamps = out.stamps[match_mask]
        out.range = out.range[match_mask]
        out.from_id = out.from_id[match_mask]
        out.to_id = out.to_id[match_mask]
        out.covariance = out.covariance[match_mask]
        out.std = out.std[match_mask]
        out.bias = out.bias[match_mask]
        return out

    def by_pair(self, from_id :int, to_id:int) -> "RangeData":
        """
        Get a RangeData object containing only the measurements between the
        specified pair of tags.

        Parameters
        ----------
        from_id : int
            The ID of the intiating tag.
        to_id : int
            The ID of the receiving tag.

        Returns
        -------
        RangeData
            RangeData object
        """
        match_mask = np.logical_and(
            self.from_id == from_id, self.to_id == to_id
        )

        out = self
        out.stamps = out.stamps[match_mask]
        out.range = out.range[match_mask]
        out.from_id = out.from_id[match_mask]
        out.to_id = out.to_id[match_mask]
        out.covariance = out.covariance[match_mask]
        out.std = out.std[match_mask]
        out.bias = out.bias[match_mask]
        return out
    
    def by_tags(self, tags :List[TagHolder]) -> "RangeData":
        """
        Get a RangeData object containing only the measurements between the
        specified pair of tags.
        """
        tag_ids = [tag.id for tag in tags]
        match_mask = np.logical_and(
            np.isin(self.from_id, tag_ids), np.isin(self.to_id, tag_ids)
        )

        out = self.copy()
        out.stamps = out.stamps[match_mask]
        out.range = out.range[match_mask]
        out.from_id = out.from_id[match_mask]
        out.to_id = out.to_id[match_mask]
        out.covariance = out.covariance[match_mask]
        out.std = out.std[match_mask]
        out.bias = out.bias[match_mask]
        return out

    def by_timestamps(self, start :float, end:float = None) -> "RawIMUData":
        """
        Get a RangeData object containing only the measurements between the
        specified pair of tags.

        Parameters
        ----------
        from_id : int
            The ID of the intiating tag.
        to_id : int
            The ID of the receiving tag.

        Returns
        -------
        RangeData
            RangeData object
        """
        if end is None:
            end = np.inf
        match_mask = np.logical_and(
            self.stamps > start, self.stamps < end
        )

        out = self.copy()
        out.stamps = out.stamps[match_mask]
        out.range = out.range[match_mask]
        out.from_id = out.from_id[match_mask]
        out.to_id = out.to_id[match_mask]
        out.covariance = out.covariance[match_mask]
        out.std = out.std[match_mask]
        out.bias = out.bias[match_mask]
        return out

    def to_measurements(self, tags: List[TagHolder], 
                  reference_id: Any = None,
                  variance: float = None, 
                  state_id: Any = None) -> List[Measurement]:
        """
        Convert to a list of measurements.

        Parameters
        ----------
        tags : List[Tag]
            Information about the tags
        variance : float, optional
            If specified, overrides the variance for all measurements. Otherwise
            the variance from the bag file or the calibration is used.
        state_id : Any, optional
            optional identifier to add to the measurement, by default None

        Returns
        -------
        List[Measurement]
            List of measurements.
        """

        tag_dict = {t.id: t for t in tags}
        measurements = []

        for i, stamp in enumerate(self.stamps):
            from_tag = tag_dict[self.from_id[i]]
            to_tag = tag_dict[self.to_id[i]]

            if variance is not None:
                v = variance
            else:
                v = self.covariance[i]

            if from_tag.parent_id == reference_id:
                model = RangeRelativePose(
                    from_tag.position,
                    to_tag.position,
                    to_tag.parent_id,
                    v,
                )
            elif to_tag.parent_id == reference_id:
                model = RangeRelativePose(
                    to_tag.position,
                    from_tag.position,
                    from_tag.parent_id,
                    v,
                )
            else:
                model = RangePoseToPose(
                    from_tag.position,
                    to_tag.position,
                    from_tag.parent_id,
                    to_tag.parent_id,
                    v,
                )

            measurements.append(
                Measurement(self.range[i], stamp, model, state_id=state_id)
            )

        return measurements