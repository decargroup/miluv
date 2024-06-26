from .utils import get_mocap_splines
import pandas as pd
import cv2
import os


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
        cir: bool = True,
        height: bool = True,
        mag: bool = True,
        barometer: bool = True,
        # calib_uwb: bool = True,
    ):

        # TODO: Add checks for valid exp dir and name
        self.exp_name = exp_name
        self.exp_dir = exp_dir
        self.cam = cam

        # TODO: read robots from configs
        exp_data = pd.read_csv("config/experiments.csv")
        exp_data = exp_data[exp_data["experiment"].astype(str) == exp_name]
        robot_ids = [
            f"ifo00{i}" for i in range(1, exp_data["num_robots"].iloc[0] + 1)
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

            # self.data[id].update({"mocap": []})
            mocap_df = self.read_csv("mocap", id)
            self.data[id]["mocap_pos"], self.data[id]["mocap_quat"] \
                = get_mocap_splines(mocap_df)

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
            all_imgs = [
                float(img.split(".")[0].replace(r"_", r"."))
                for img in all_imgs
            ]
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

        def data_from_timestamp_robot(robot_id: str, timestamps: list) -> dict:
            """Return all data from a given timestamp for a given robot."""
            data_by_robot = {}
            for sensor in sensors:
                data_by_robot[sensor] = data_from_timestamp_sensor(
                    robot_id, sensor, timestamps)

            return data_by_robot

        def data_from_timestamp_sensor(robot_id: str, sensor: str,
                                       timestamps: list) -> dict:
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

        if robot_ids is None:
            robot_ids = self.data.keys()
        if sensors is None:
            sensors = self.data['ifo001'].keys()

        data_by_timestamp = {}
        for robot_id in robot_ids:
            data_by_timestamp[robot_id] = data_from_timestamp_robot(
                robot_id, timestamps)

        return data_by_timestamp

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
                        img_path = os.path.join(
                            self.exp_dir, self.exp_name, robot_id, cam,
                            str(img_ts).replace(r".", r"_", 1) + ".jpeg")
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
        "3a",
        barometer=False,
        height=False,
        cir=False,
    )

    print("done!")
