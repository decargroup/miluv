import unittest
from miluv.data import DataLoader
import numpy as np
import os

with open("tests/imgs.csv", "r") as f:
    img_names = [line[:-1] for line in f.readlines()]


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.loader = DataLoader(
            "3a",
            barometer=False,
            height=False,
            cir=False,
        )

    def test_read_csv(self):
        robot_ids = [
            "ifo001",
            "ifo002",
            "ifo003",
        ]

        topics = [
            "imu_cam",
            "imu_px4",
            "uwb_range",
            "mag",
            "uwb_passive",
            "mocap",
        ]

        for id in robot_ids:
            for t in topics:
                data = self.loader.read_csv(t, id)
                self.assertTrue(data is not None)
                self.assertTrue(len(data) > 0)
        pass

    def test_closest_past_timestamp(self):
        timestamps = np.arange(0, 20, 1)

        robot_ids = [
            "ifo001",
            "ifo002",
            "ifo003",
        ]

        sensor = [
            "imu_cam",
            "imu_px4",
            "uwb_range",
            "mag",
            # "bottom",  # Uncomment when images available
            # "color",  # Uncomment when images available
            # "infra1",  # Uncomment when images available
            # "infra2",  # Uncomment when images available
        ]

        for ts in timestamps:
            for id in robot_ids:
                for s in sensor:
                    t_closest = self.loader.closest_past_timestamp(id, s, ts)
                    self.assertTrue(t_closest is None or t_closest <= ts)
                    if t_closest is not None:
                        if s not in ["bottom", "color", "infra1", "infra2"]:
                            idx = np.where(self.loader.data[id][s]["timestamp"] <= ts)[0][-1]
                            self.assertTrue(t_closest == self.loader.data[id][s]["timestamp"][idx])
                        else:
                            all_imgs = os.listdir(os.path.join(self.loader.exp_dir, self.loader.exp_name, id, s))
                            all_imgs = [int(img.split(".")[0]) for img in all_imgs]
                            not_over = [t for t in all_imgs if t <= ts]
                            t = max(not_over)
                            self.assertTrue(t_closest == t)
        pass

    def test_data_from_timestamp(self):
        timestamps = np.arange(0, 20, 1)

        robot_ids = [
            "ifo001",
            "ifo002",
            "ifo003",
        ]
        sensors = [
            "imu_cam",
            "imu_px4",
            "uwb_range",
            "mag",
        ]

        data = self.loader.data_from_timestamp(
            timestamps,
            robot_ids,
            sensors,
        )

        self.assertTrue(data is not None)
        self.assertTrue(len(data) > 0)
        self.assertTrue(all([id in data for id in robot_ids]))
        self.assertTrue(all([s in data[id] for s in sensors for id in robot_ids]))
        self.assertTrue(all([len(data[id][s]) <= len(timestamps) for s in sensors for id in robot_ids]))
        self.assertTrue(all([len(data[id][s]) > 0 for s in sensors for id in robot_ids]))
        self.assertTrue(all([len(data[id][s].columns) > 0 for s in sensors for id in robot_ids]))
        self.assertAlmostEqual(data["ifo001"]["imu_px4"].iloc[0]["timestamp"], 0.996361971)
        self.assertAlmostEqual(data["ifo001"]["imu_px4"].iloc[0]["angular_velocity.x"], -0.002761804)
        self.assertAlmostEqual(data["ifo001"]["imu_px4"].iloc[0]["angular_velocity.y"], 0.00182065)
        self.assertAlmostEqual(data["ifo001"]["imu_px4"].iloc[0]["angular_velocity.z"], 0.001275649)
        self.assertAlmostEqual(data["ifo001"]["imu_px4"].iloc[0]["linear_acceleration.x"], -0.449486375)
        self.assertAlmostEqual(data["ifo001"]["imu_px4"].iloc[0]["linear_acceleration.y"], 0.049452316)
        self.assertAlmostEqual(data["ifo001"]["imu_px4"].iloc[0]["linear_acceleration.z"], 9.807592392)

        pass

    def test_images_present(self):
        robot_ids = [
            "ifo001",
            "ifo002",
            "ifo003",
        ]

        cams = [
            "bottom",
            "color",
            "infra1",
            "infra2",
        ]

        for name in img_names:
            self.assertTrue(os.path.exists(name))

        pass

    def test_imgs_from_timestamp(self):
        timestamps = np.arange(0, 20, 1)

        robot_ids = [
            "ifo001",
            "ifo002",
            "ifo003",
        ]

        cams = [
            "bottom",
            "color",
            "infra1",
            "infra2",
        ]

        imgs = self.loader.imgs_from_timestamps(timestamps, robot_ids, cams)

        self.assertTrue(imgs is not None)
        self.assertTrue(imgs['ifo001']['bottom'].iloc[0]["timestamp"] == 0.9653124809265137)
        self.assertTrue(imgs['ifo001']['color'].iloc[0]["timestamp"] == 0.9723629951477051)
        self.assertTrue(imgs['ifo001']['infra1'].iloc[0]["timestamp"] == 0.9723784923553467)
        self.assertTrue(imgs['ifo001']['infra2'].iloc[0]["timestamp"] == 0.9723784923553467)

        pass


if __name__ == '__main__':
    unittest.main()
