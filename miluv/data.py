import pandas as pd
from PIL import Image
import os


# TODO: look into dataclasses
class DataLoader:

    def __init__(
        self,
        exp_name: str,
        exp_dir: str = "./data",
        imu: str = "both",
        cam: dict = {
            "color": True,
            "bottom": True,
            "infra1": True,
            "infra2": True
        },
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
            # self.data[id].update({"mocap": []})
            # self.data[id]["mocap"] = self.read_csv("mocap", id)

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

    def closest_past_timestamp(self, robot_id: str, sensor: str,
                               timestamp: float) -> int:
        """Return the closest timestamp in the past for a given sensor."""
        not_over = None
        if sensor != "bottom" and sensor != "color" and sensor != "infra1" and sensor != "infra2":
            not_over = [
                ts for ts in self.data[robot_id][sensor]["timestamp"]
                if ts <= timestamp
            ]
        else:
            all_imgs = os.listdir(
                os.path.join(self.exp_dir, self.exp_name, robot_id, sensor))
            all_imgs = [int(img.split(".")[0]) for img in all_imgs]
            not_over = [ts for ts in all_imgs if ts <= timestamp]

        if not_over == []:
            return None
        return max(not_over)

    def data_from_timestamp(
        self,
        timestamps: list,
        robot_ids=None,
        sensors=None,
    ) -> dict:
        """Return all data from a given timestamp."""
        if robot_ids is None:
            robot_ids = self.data.keys()
        if sensors is None:
            sensors = self.data['ifo001'].keys()

        def data_from_timestamp_robot(robot_id: str,
                                      timestamps: float) -> dict:
            """Return all data from a given timestamp for a given robot."""
            data_by_robot = {}
            for sensors in self.data[robot_id]:
                data_by_robot[sensors] = data_from_timestamp_sensor(
                    robot_id, sensors, timestamps)

            return data_by_robot

        def data_from_timestamp_sensor(robot_id: str, sensor: str,
                                       timestamps: float) -> dict:
            """Return all data from a given timestamp for a given sensor for a given robot."""
            col_names = self.data[robot_id][sensor].columns
            df = pd.DataFrame(columns=col_names)
            for timestamp in timestamps:
                if timestamp in self.data[robot_id][sensor][
                        "timestamp"].values:
                    df = pd.concat([
                        df if not df.empty else None, self.data[robot_id]
                        [sensor].loc[self.data[robot_id][sensor]["timestamp"]
                                     == timestamp]
                    ])
                else:
                    df = pd.concat([
                        df if not df.empty else None, self.data[robot_id]
                        [sensor].loc[self.data[robot_id][sensor]["timestamp"]
                                     == self.closest_past_timestamp(
                                         robot_id, sensor, timestamp)]
                    ])
            return df

        data_by_timestamp = {}
        for robot_id in robot_ids:
            data_by_timestamp[robot_id] = data_from_timestamp_robot(
                robot_id, timestamps)

        return data_by_timestamp

    def imgs_from_timestamp(self, timestamp: int) -> dict:
        """Return all images from a given timestamp."""
        img_by_timestamp = {}
        for robot_id in self.data:
            img_by_timestamp[robot_id] = self.imgs_from_timestamp_robot(
                robot_id, timestamp)
        return img_by_timestamp

    def imgs_from_timestamp_robot(self, robot_id: str, timestamp: int) -> dict:
        """Return all images from a given timestamp for a given robot."""
        img_by_robot = {}
        for cam in self.cam:
            if cam:
                img_ts = self.closest_past_timestamp(robot_id, cam, timestamp)
                if img_ts is None:
                    print("No", cam, "image found for timestamp", timestamp)
                    continue
                img_path = os.path.join(self.exp_dir, self.exp_name, robot_id,
                                        cam,
                                        str(img_ts) + ".jpeg")
                img = Image.open(img_path)
                img_by_robot[cam] = img
        return img_by_robot


if __name__ == "__main__":
    mv = DataLoader(
        "1c",
        baro=False,
        height=False,
    )

    print("done!")
