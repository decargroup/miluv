
import pandas as pd
import os
from miluv.wrappers import (
    MocapTrajectory,
)
import numpy as np
from typing import List

# TODO: look into dataclasses
class DataLoader:
    def __init__(
        self,
        exp_name: str,
        exp_dir: str = "./data",
        imu: str = "both",
        cam: str = "both",
        uwb: bool = True,
        height: bool = True,
        mag: bool = True,
        baro: bool = True,
        # calib_uwb: bool = True,
    ):
        
        # TODO: Add checks for valid exp dir and name
        self.exp_name = exp_name
        self.exp_dir = exp_dir
        
        # TODO: read robots from configs
        robot_ids = ["ifo001", "ifo002", "ifo003"]
        self.data = {id: {} for id in robot_ids}

        for id in robot_ids:    
            if imu == "both" or imu == "px4":
                self.data[id].update({"imu_px4": []})
                self.data[id]["imu_px4"] = self.read_csv("imu_px4", id)
            
            if imu == "both" or imu == "cam":
                self.data[id].update({"imu_cam": []})
                self.data[id]["imu_cam"] = self.read_csv("imu_cam", id)
            
            # TODO: UWB topics to be read depend on configs, for now range only
            if uwb:
                self.data[id].update({"uwb_range": []})
                self.data[id]["uwb_range"] = self.read_csv("uwb_range", id)
            
            if height:
                self.data[id].update({"height": []})
                self.data[id]["height"] = self.read_csv("height", id)
            
            if mag:
                self.data[id].update({"mag": []})
                self.data[id]["mag"] = self.read_csv("mag", id)
            
            if baro:
                self.data[id].update({"baro": []})
                self.data[id]["baro"] = self.read_csv("baro", id)

            self.data[id].update({"mocap": MocapTrajectory})
            self.data[id]["mocap"] = MocapTrajectory(
                                     self.read_csv("mocap", id), 
                                     frame_id=id)

        # TODO: Load timestamp-to-image mapping?
        # if cam is "both" or cam is "bottom":
        #     self.load_imgs("bottom")
        # if cam is "both" or cam is "front":
        #     self.load_imgs("front")
                
    def read_csv(self, topic: str, robot_id) -> pd.DataFrame:
        path = os.path.join(
            self.exp_dir, 
            self.exp_name, 
            robot_id, 
            topic + ".csv"
        )
        return pd.read_csv(path)
    

    def by_stamps(self, stamps, 
                  robot_id:List = None, 
                  sensors:List = None):
        """
        Get the data at one or more query times.

        Parameters
        ----------
        stamps : np.ndarray
            Query times

        Returns
        -------
        Data
            data at the closest time to the query time
            data is at the lower bound of the time window
            data is not interpolated
        """
        stamps = np.array(stamps) * 1e9

        if robot_id is None:
            robot_id = self.data.keys()


        robot_id = [robot_id] if type(robot_id) is str else robot_id
        sensors = [sensors] if type(sensors) is str else sensors

        out = {}
        for id in robot_id:
            if sensors is None:
                sensors = list(self.data[id].keys() - ["mocap"])

            indices_dict = {}
            data = self.data[id].copy()

            for sensor in sensors:
                indices = []
                for s in stamps:
                    index = self._get_index(s, id, sensor)
                    indices.append(index)
                indices_dict[sensor] = indices
                data[sensor] = data[sensor].loc[indices_dict[sensor]]
            out[id] = data
        return out

    def _get_index(self, stamp: float, robot_id: str, topic: str, ) -> int:
        """
        Get the index of the closest earlier time to the query time.

        Parameters
        ----------
        topic : str
            The topic to query
        stamp : float
            Query time

        Returns
        -------
        int
            index of the closest earlier time to the query time
        """
        mask = self.data[robot_id][topic]["timestamp"] <= stamp
        last_index = mask[::-1].idxmax() if mask.any() else None
        return last_index
    