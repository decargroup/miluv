import pandas as pd
import cv2
import os
from miluv.mocap_trajectory import (
    MocapTrajectory
)
import numpy as np
from typing import List
import copy
from miluv.utils import (
    get_experiment_info,
    get_anchors, 
    get_tags,
    tags_to_df,
)
import yaml


# TODO: look into dataclasses
class DataLoader:
    def __init__(
        self,
        exp_name: str,
        exp_dir: str = "./data",
        imu: str = "both",
        cam: list = [
            "color",
            "bottom",
            "infra1",
            "infra2",
        ],
        uwb: bool = True,
        cir: bool = False,
        vio: bool = True,
        vio_loop: bool = False,
        height: bool = True,
        mag: bool = True,
        barometer: bool = True,
    ):

        # TODO: Add checks for valid exp dir and name
        self.exp_name = exp_name
        self.exp_dir = exp_dir
        self.cam = cam
        self.setup = {'uwb_tags': None,  
                      'april_tags': None,
                      'imu_px4_calib': None,
                      'imu_cam_calib': None,}

        # TODO: read robots from configs
        exp_path = os.path.join(self.exp_dir, self.exp_name)
        exp_info = get_experiment_info(exp_path)
        
        if  exp_info["num_robots"] == 1:
            robot_ids = ["ifo001"]
        else:
            robot_ids = [
                "ifo001",
                "ifo002",
                "ifo003",
            ]

        self.data = {id: {} for id in robot_ids}

        for id in robot_ids:    
            if imu == "both" or imu == "px4":
                self.data[id].update({"imu_px4": []})
                self.data[id]["imu_px4"] = self.read_csv("imu_px4", id)

            if imu == "both" or imu == "cam":
                self.data[id].update({"imu_cam": []})
                self.data[id]["imu_cam"] = self.read_csv("imu_cam", id)

            if uwb:
                self.data[id].update({"uwb_range": []})
                self.data[id]["uwb_range"] = self.read_csv("uwb_range", id)

                self.data[id].update({"uwb_passive": []})
                self.data[id]["uwb_passive"] = self.read_csv("uwb_passive", id)

            if vio:
                self.data[id].update({"vio": []})
                self.data[id]["vio"] = self.read_csv("vio", id)

            if vio_loop:
                self.data[id].update({"vio_loop": []})
                self.data[id]["vio_loop"] = self.read_csv("vio_loop", id)
                
            if cir:
                self.data[id].update({"uwb_cir": []})
                self.data[id]["uwb_cir"] = self.read_csv("uwb_cir", id)

            if height:
                self.data[id].update({"height": []})
                self.data[id]["height"] = self.read_csv("height", id)

            if mag:
                self.data[id].update({"mag": []})
                self.data[id]["mag"] = self.read_csv("mag", id)

            if barometer:
                self.data[id].update({"barometer": []})
                self.data[id]["barometer"] = self.read_csv("barometer", id)

            self.data[id].update({"mocap": MocapTrajectory})
            self.data[id]["mocap"] = MocapTrajectory(
                                     self.read_csv("mocap", id), 
                                     frame_id=id)
            
        # Load anchors and moment arms from experiment info
        anchors = get_anchors(exp_info["anchor_constellation"])
        tags = get_tags()
        (self.setup["uwb_tags"], 
        self.setup["april_tags"]) = tags_to_df(anchors, tags)
        self.setup["imu_px4_calib"] = self.read_yaml("imu", "imu_px4_calib")

        
        # TODO: Add april tags to the data
        # TODO: Load cam imu calibration
        # self.setup['imu_cam_calib'] = self.read_yaml("imu", "imu_cam_calib")

        # TODO: Load timestamp-to-image mapping?
        # if cam == "both" or cam == "bottom":
        #     self.load_imgs("bottom")
        # if cam == "both" or cam == "front":
        #     self.load_imgs("front")

    def read_csv(self, topic: str, robot_id) -> pd.DataFrame:
        """Read a csv file for a given robot and topic."""
        path = os.path.join(self.exp_dir, self.exp_name, robot_id,
                            topic + ".csv")
        return pd.read_csv(path)
    
    def read_yaml(self, sensor:str, topic: str) -> pd.DataFrame:

        """Read a yaml file for a given robot and topic."""
        path = os.path.join("config/" + sensor + "/" + topic + ".yaml")
        return yaml.safe_load(open(path, 'r'))
    

    def copy(self):
        return copy.deepcopy(self)
    

    def by_timestamps(self, stamps, 
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
        stamps = np.array(stamps)

        if robot_id is None:
            robot_id = self.data.keys()


        robot_id = [robot_id] if type(robot_id) is str else robot_id
        sensors = [sensors] if type(sensors) is str else sensors

        out = self.copy()
        out.data = {id: {} for id in robot_id}
        for id in robot_id:
            if sensors is None:
                sensors = list(self.data[id].keys() - ["mocap"])

            indices_dict = {}
            data = {sensor: self.data[id][sensor].copy() 
                    for sensor in sensors}


            for sensor in sensors:
                indices = []
                for s in stamps:
                    index = self._get_index(s, id, sensor)
                    indices.append(index)
                indices_dict[sensor] = indices
                data[sensor] = data[sensor].loc[indices_dict[sensor]]

            out.data[id] = data

        return out
    
    def by_timerange(self, 
                    start_time: float, 
                    end_time: float,
                    robot_id:List = None, 
                    sensors:List = None):
            """
            Get the data within a time range.
    
            Parameters
            ----------
            start_time : float
                Start time of the range
            end_time : float
                End time of the range
    
            Returns
            -------
            Data
                data within the time range
            """
            start_time = start_time
            end_time = end_time
    
            if robot_id is None:
                robot_id = self.data.keys()
    
            robot_id = [robot_id] if type(robot_id) is str else robot_id
            sensors = [sensors] if type(sensors) is str else sensors
    
            out = self.copy()
            out.data = {id: {} for id in robot_id}
            for id in robot_id:
                if sensors is None:
                    sensors = list(self.data[id].keys() - ["mocap"])
    
                data = {sensor: self.data[id][sensor].copy() 
                        for sensor in sensors}
    
                for sensor in sensors:
                    mask = (data[sensor]["timestamp"] >= start_time
                            ) & (data[sensor]["timestamp"] <= end_time)
                    
                    data[sensor] = data[sensor].loc[mask]
    
                out.data[id] = data
    
            return out

    def get_timerange(self, 
                   robot_id:List = None, 
                   sensors: List = None) -> float:
        """
        Get the start time of the data for a robot.

        Parameters
        ----------
        robot_id : str
            The robot ID

        Returns
        -------
        float
            Start time of the data
        """
        start_times = []
        end_times = []

        if robot_id is None:
            robot_id = self.data.keys()

        robot_id = [robot_id] if type(robot_id) is str else robot_id
        sensors = [sensors] if type(sensors) is str else sensors


        for id in robot_id:
            if sensors is None:
                sensors = list(self.data[id].keys() - ["mocap"])
            
            for sensor in sensors:
                start_times.append(self.data[id][sensor]["timestamp"].iloc[0])
                end_times.append(self.data[id][sensor]["timestamp"].iloc[-1])

        start_time = max(start_times)
        end_time = min(end_times)

        return (start_time, end_time)
        

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

    def imgs_from_timestamps(self,
                             timestamps: list,
                             robot_ids=None,
                             cams=None) -> dict:
        """Return all images from a given timestamp."""

        def imgs_from_timestamp_robot(robot_id: str, cams: list,
                                      timestamps: list) -> dict:
            """Return all images from a given timestamp for a given robot."""
            img_by_robot = {}
            for cam in cams:
                valid_ts = []
                imgs = []
                for timestamp in timestamps:
                    if cam:
                        img_ts = self.closest_past_timestamp(
                            robot_id, cam, timestamp)
                        if img_ts is None:
                            # print("No", cam, "image found for timestamp",
                            #       timestamp, "for robot_id", robot_id)    # Debugging msg
                            continue
                        img_path = os.path.join(self.exp_dir, self.exp_name,
                                                robot_id, cam,
                                                str(img_ts) + ".jpeg")
                        imgs.append(cv2.imread(img_path))
                        valid_ts.append(img_ts)
                img_by_robot[cam] = pd.DataFrame({
                    "timestamp": valid_ts,
                    "image": imgs
                })
            return img_by_robot

        if robot_ids is None:
            robot_ids = self.data.keys()
        if cams is None:
            cams = self.cam

        img_by_timestamp = {}
        for robot_id in robot_ids:
            img_by_timestamp[robot_id] = imgs_from_timestamp_robot(
                robot_id, cams, timestamps)
        return img_by_timestamp


if __name__ == "__main__":
    mv = DataLoader(
        "1c",
        barometer=False,
        height=False,
    )

    print("done!")
