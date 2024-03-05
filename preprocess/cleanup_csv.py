import sys
from os import listdir, remove, walk, rename
from os.path import join
import pandas as pd

# TODO: barometer and cir
# headers to keep for every file
imu = [
    "timestamp",
    "angular_velocity.x",
    "angular_velocity.y",
    "angular_velocity.z",
    "linear_acceleration.x",
    "linear_acceleration.y",
    "linear_acceleration.z",
]
mag = [
    "timestamp",
    "magnetic_field.x",
    "magnetic_field.y",
    "magnetic_field.z",
]
height = ["timestamp", "range"]
mocap = [
    "timestamp",
    "pose.position.x",
    "pose.position.y",
    "pose.position.z",
    "pose.orientation.x",
    "pose.orientation.y",
    "pose.orientation.z",
    "pose.orientation.w",
]
barometer = ["timestamp", "fluid_pressure"]

def cleanup_csvs(dir):
    # Find all csv files
    files = [f for f in listdir(dir) if f.endswith('.csv')]

    for file in files:
        if "imu" in file and "camera" in file:
            process_csv(dir, file, imu, "imu_cam")
        elif "mag" in file and file != "mag.csv":
            process_csv(dir, file, mag, "mag")
        elif "hrlv" in file:
            process_csv(dir, file, height, "height")
        elif "vrpn" in file:
            process_csv(dir, file, mocap, "mocap")
        elif "static_pressure" in file:
            process_csv(dir, file, barometer, "barometer")
        elif "imu" in file and "mavros" in file and "raw" in file:
            process_csv(dir, file, imu, "imu_px4")


def process_csv(dir, file, headers, name):
    df = pd.read_csv(join(dir, file))
    df = merge_time(df)
    df = df[headers]
    df.to_csv(join(dir, name + ".csv"), index=False)
    remove(join(dir, file))


def merge_time(df):
    sec = df["header.stamp.secs"]
    nsec = df["header.stamp.nsecs"]
    df["timestamp"] = sec + nsec/1e9
    return df


def find_min_timestamp(all_files):
    """Find the minimum timestamp in all csv files."""
    min_timestamp = float('inf')
    for file in all_files:
        df = pd.read_csv(file)
        if df["timestamp"].min() < min_timestamp:
            min_timestamp = df["timestamp"].min()
    return min_timestamp


def shift_timestamps(path):
    """Shift all timestamps by the minimum timestamp."""
    all_csvs = []
    all_jpegs = []
    for subdir, dirs, files in walk(path):
        for file in files:
            if file.endswith('.csv'):
                all_csvs.append(join(subdir, file))
            elif file.endswith('.jpeg'):
                all_jpegs.append(join(subdir, file))

    min_timestamp = find_min_timestamp(all_csvs)
    for file in all_csvs:
        df = pd.read_csv(file)
        df["timestamp"] = df["timestamp"] - min_timestamp
        df.to_csv(file, index=False)
    for file in all_jpegs:
        img_timestamp = int(file.split(".")[0].split("/")[-1]) - min_timestamp
        rename(
            file, "/".join(file.split("/")[:-1]) + "/" + str(img_timestamp) +
            ".jpeg")


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print(
            "Not enough arguments. Usage: python cleanup_csv.py path_to_csvs")
        sys.exit(1)

    path = sys.argv[1]
    files = [f for f in listdir(path) if f.endswith('.bag')]

    for file in files:
        cleanup_csvs(join(path, file.split(".")[0]))

    shift_timestamps(path)