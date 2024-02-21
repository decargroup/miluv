
import pandas as pd
import os

# TODO: look into dataclasses
class Miluv:
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

            # TODO: replace this with adding gt to each robot's data by fitting a spline
            self.data[id].update({"mocap": []})
            self.data[id]["mocap"] = self.read_csv("mocap", id)

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
        