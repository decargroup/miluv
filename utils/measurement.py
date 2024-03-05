import numpy as np
from typing import Any, List, Dict
from utils.models import (
    MeasurementModel,
    RangeRelativePose,
    RangePoseToPose,
)
from miluv.data import DataLoader
import pandas as pd
import copy

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

class RangeData(DataLoader):
    """
    Class for storing UWB range data.
    """
    def __init__(self, 
                 dataloader: DataLoader):
        
        # remove all data which are not "uwb_range"
        data = dataloader.data
        self.setup = dataloader.setup

        self._sensors = ["uwb_range"]
        self.data = {id: {sensor: data[id][sensor] 
                          for sensor in data[id] if 
                          sensor in self._sensors} for id in data}


    def copy(self):
        """
        Create a copy of the RangeData object.
        """
        return copy.deepcopy(self)
                
    def by_timestamps(self, 
                      stamps, 
                      robot_id: List = None, 
                      sensors: List = None):


        if robot_id is None:
            robot_id = list(self.data.keys())


        if sensors is not None and not all(
            sensor in self._sensors for sensor in sensors):
            raise ValueError(f"Invalid sensor type. Must be one of {self._sensors}")
        else:
            sensors = self._sensors
        
        return super().by_timestamps(stamps, 
                                     robot_id, 
                                     sensors)

    def by_timerange(self, 
                     start_time: float, 
                     end_time: float, 
                     robot_id: List = None, 
                     sensors: List = None):


        if robot_id is None:
            robot_id = list(self.data.keys())

        if sensors is not None and not all(
            sensor in self._sensors for sensor in sensors):
            raise ValueError(f"Invalid sensor type. Must be one of {self._sensors}")
        else:
            sensors = self._sensors
        
        return super().by_timerange(start_time, 
                                    end_time, 
                                    robot_id, 
                                    sensors
                                    )

    def filter_by_bias(self, 
                  
                   max_bias:float,
                   min_bias :float = None) -> "RangeData":


        out = self.copy()

        for id in out.data:
            for sensor in out.data[id]:
                condition = (abs(out.data[id][sensor]['bias']) <= max_bias)
                if min_bias is not None:
                    condition &= (abs(out.data[id][sensor]['bias']) >= min_bias)
                out.data[id][sensor] = out.data[id][sensor][condition]

        return out

    def by_pair(self, 
                from_id :int, 
                to_id:int) -> "RangeData":
        """
        Get a RangeData object containing only the measurements between the
        specified pair of tags.

        Parameters
        ----------
        from_id : int
            The ID of the initiating tag.
        to_id : int
            The ID of the receiving tag.

        Returns
        -------
        RangeData
            RangeData object
        """
        out = self.copy()

        for id in out.data:
            for sensor in out.data[id]:
                match_mask = np.logical_and(
                    out.data[id][sensor]['from_id'] == from_id, 
                    out.data[id][sensor]['to_id'] == to_id
                )
                out.data[id][sensor] = out.data[id][sensor][match_mask]

        return out
    
    def by_tags(self, tag_ids :List) -> "RangeData":
        """
        Get a RangeData object containing only the measurements between the
        specified pair of tags.
        """

        out = self.copy()

        for id in out.data:
            for sensor in out.data[id]:
                match_mask = np.logical_and(
                    np.isin(out.data[id][sensor]['from_id'], tag_ids), 
                    np.isin(out.data[id][sensor]['to_id'], tag_ids)
                )
                out.data[id][sensor] = out.data[id][sensor][match_mask]

        return out

    # TODO: Need to check
    def merge_range(self,
                robot_id:List=None, 
                sensors:List=None,
                ) -> pd.DataFrame:


        if robot_id is None:
            robot_id = list(self.data.keys())

        if sensors is not None and not all(
            sensor in self._sensors for sensor in sensors):
                
                raise ValueError(f"Invalid sensor type. Must be one of {self._sensors}")
        else:
            sensors = self._sensors
        sensors = [sensors] if type(sensors) is str else sensors


        # create a pd dataframe with all the range data
        out = {sensor: [] for sensor in sensors}

        for id in robot_id:
            for sensor in sensors:
                if sensor in self.data[id]:
                    out[sensor].append(self.data[id][sensor])

        out = {sensor: pd.concat(out[sensor]) 
               for sensor in sensors}
        

        # sort the data by timestamp
        for sensor in sensors:
            out[sensor] = out[sensor].sort_values(by="timestamp")
            # remove duplicates
            out[sensor] = out[sensor].drop_duplicates(
                subset=["from_id", "to_id", "timestamp"], 
                keep="last")

        return out

    # TODO: Need to check
    def to_measurements(self, 
                  reference_id: Any = 'world',) -> List[Measurement]:
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

        range_data = self.merge_range()
        # TODO: remove this line
        # save the range data in a csv file
        range_data['uwb_range'].to_csv('range_data.csv', index=False)
        
        measurements = []
        for i, data in range_data['uwb_range'].iterrows():


            from_tag = self.setup['uwb_tags'].loc[
                       self.setup['uwb_tags']['tag_id'] == data.from_id].iloc[0]
            
            to_tag = self.setup['uwb_tags'].loc[
                     self.setup['uwb_tags']['tag_id'] == data.to_id].iloc[0]
            
            from_tag_pos = from_tag[['position.x',
                                     'position.y',
                                     'position.z']].tolist()
            to_tag_pos = to_tag[['position.x',    
                                 'position.y',
                                 'position.z']].tolist()
            
            variance = 5 * data["std"]**2
            if from_tag.parent_id == reference_id:
                model = RangeRelativePose(
                    from_tag_pos,
                    to_tag_pos,
                    to_tag.parent_id,
                    variance,
                )
            elif to_tag.parent_id == reference_id:
                model = RangeRelativePose(
                    to_tag_pos,
                    from_tag_pos,
                    from_tag.parent_id,
                    variance,
                )
            else:
                model = RangePoseToPose(
                    from_tag_pos,
                    to_tag_pos,
                    from_tag.parent_id,
                    to_tag.parent_id,
                    variance,
                )
            measurements.append(
                Measurement(data.range, data.timestamp, model)
            )
        return measurements