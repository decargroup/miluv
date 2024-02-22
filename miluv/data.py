import pandas as pd
from PIL import Image
import os


# TODO: look into dataclasses
class Miluv:

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

        self.cam = cam

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
        path = os.path.join(self.exp_dir, self.exp_name, robot_id,
                            topic + ".csv")
        return pd.read_csv(path)

    def closest_past_timestamp(self, robot_id: str, sensor: str,
                               timestamp: float):
        if sensor != "bottom" and sensor != "color" and sensor != "infra1" and sensor != "infra2":
            return min(self.data[robot_id][sensor]["timestamp"],
                       key=lambda x: abs(x - timestamp))
        else:
            all_imgs = os.listdir(
                os.path.join(self.exp_dir, self.exp_name, robot_id, sensor))
            all_imgs = [int(img.split(".")[0]) for img in all_imgs]
            return min(all_imgs, key=lambda x: abs(x - timestamp))

    def data_from_timestamp(self, timestamp: int):

        def data_from_timestamp_robot(robot_id: str, timestamp: float):

            def data_from_timestamp_sensor(sensor: str, timestamp: float):
                if timestamp in self.data[robot_id][sensor][
                        "timestamp"].values:
                    return self.data[robot_id][sensor].loc[
                        self.data[robot_id][sensor]["timestamp"] == timestamp]
                else:
                    return self.data[robot_id][sensor].loc[
                        self.data[robot_id][sensor]["timestamp"] == self.
                        closest_past_timestamp(robot_id, sensor, timestamp)]

            data_by_robot = {}
            for sensor in self.data[robot_id]:
                data_by_robot[sensor] = data_from_timestamp_sensor(
                    sensor, timestamp)

            return data_by_robot

        data_by_timestamp = {}
        for robot_id in self.data:
            data_by_timestamp[robot_id] = data_from_timestamp_robot(
                robot_id, timestamp)

        return data_by_timestamp

    def imgs_from_timestamp(self, robot_id: str, timestamp: int):
        for cam in self.cam:
            if cam:
                img_path = os.path.join(
                    self.exp_dir, self.exp_name, robot_id, cam,
                    str(self.closest_past_timestamp(robot_id, cam, timestamp))
                    + ".jpeg")
                img = Image.open(img_path)
                img.show()
                print(img_path)


if __name__ == "__main__":
    mv = Miluv(
        "1c",
        baro=False,
        height=False,
    )

    mv.imgs_from_timestamp("ifo001", 0)
