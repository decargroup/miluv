import numpy as np
from typing import Any, List
from utils.models import (
    RangePoseToAnchorById,
    RangePoseToPose,
)
from miluv.data import DataLoader
import pandas as pd
import copy

class Measurement:
    """
    A data container containing a measurement's value, timestamp,
    and corresponding model.
    """
    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        model = None,
        state_id: Any = None,
    ):
        self.value = np.array(value) if np.isscalar(value) else value
        self.stamp = stamp
        self.model = model
        self.state_id = state_id

    def minus(self, y_check: np.ndarray) -> np.ndarray:
        return self.value.reshape((-1, 1)) - y_check.reshape((-1, 1))

class RangeData(DataLoader):
    """
    Class for storing UWB range data.
    """
    def __init__(self, 
                 dataloader: DataLoader, miluv):
        
        # remove all data which are not "uwb_range"
        data = dataloader.data
        self.setup = dataloader.setup

        self._sensors = ["uwb_range"]
        self.data = {id: {sensor: data[id][sensor] 
                          for sensor in data[id] if 
                          sensor in self._sensors} for id in data}
        self.miluv = miluv
    def copy(self):
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
        Get a RangeData object containing only the measurements for 
        a specified set of tags.
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

    def to_measurements(self, 
                  reference_id: Any = 'world',) -> List[Measurement]:
        """
        Convert to a list of Measurement objects.
        """

        range_data = self.merge_range()
        measurements = []
        for i, data in range_data['uwb_range'].iterrows():
            from_tag = self.setup['uwb_tags'].loc[
                       self.setup['uwb_tags']['tag_id'] == data.from_id].iloc[0]
            to_tag = self.setup['uwb_tags'].loc[
                     self.setup['uwb_tags']['tag_id'] == data.to_id].iloc[0]
            
            variance = data["std"]**2
            if to_tag.parent_id == reference_id:
                model = RangePoseToAnchorById(
                    to_tag[['position.x','position.y','position.z']].tolist(),
                    from_tag[['position.x','position.y','position.z']].tolist(),
                    from_tag.parent_id,
                    variance,
                )
            else:
                model = RangePoseToPose(
                    from_tag[['position.x','position.y','position.z']].tolist(),
                    to_tag[['position.x','position.y','position.z']].tolist(),
                    from_tag.parent_id,
                    to_tag.parent_id,
                    variance,
                    )
                
            measurements.append(
                Measurement(data.range, data.timestamp, model)
            )
        return measurements