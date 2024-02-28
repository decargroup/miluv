import unittest
from data import DataLoader
import numpy as np


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.loader = DataLoader("1c", baro=False, height=False)

    def test_read_csv(self):
        # TODO: Write test cases for the read_csv method
        pass

    def test_closest_past_timestamp(self):
        # TODO: Write test cases for the closest_past_timestamp method
        timestamps = np.arange(0, 20E9, 1E9)
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
            "bottom",
            "color",
            "infra1",
            "infra2",
        ]

        for ts in timestamps:
            for id in robot_ids:
                for s in sensor:
                    t_closest = self.loader.closest_past_timestamp(id, s, ts)
                    self.assertTrue(t_closest is None or t_closest <= ts)
        pass

    def test_data_from_timestamp(self):
        # TODO: Write test cases for the data_from_timestamp method
        pass

    def test_data_from_timestamp_robot(self):
        # TODO: Write test cases for the data_from_timestamp_robot method
        pass

    def test_data_from_timestamp_sensor(self):
        # TODO: Write test cases for the data_from_timestamp_sensor method
        pass

    def test_imgs_from_timestamp(self):
        # TODO: Write test cases for the imgs_from_timestamp method
        pass

    def test_imgs_from_timestamp_robot(self):
        # TODO: Write test cases for the imgs_from_timestamp_robot method
        pass


if __name__ == '__main__':
    unittest.main()
