import sys
from os import listdir, remove, walk, rename
from os.path import join, isfile
import pandas as pd
import yaml

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
height = [
    "timestamp",
    "range",
]

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
barometer = [
    "timestamp",
    "fluid_pressure",
]

cir = [
    "timestamp",
    "my_id",
    "from_id",
    "to_id",
    "idx",
    "cir",
]


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
        elif "uwb" in file and "cir" in file:
            process_cir(dir, file, cir)


def process_csv(dir, file, headers, name):
    df = pd.read_csv(join(dir, file))
    df = merge_time(df)
    df = df[headers]
    df.to_csv(join(dir, name + ".csv"), index=False)
    remove(join(dir, file))


def process_cir(dir, file, headers):
    df = pd.read_csv(join(dir, file))
    df = merge_time(df)
    cir_headers = [f"cir_{int(i)}" for i in range(1016)]
    df["cir"] = df[cir_headers].values.tolist()
    df = df[headers]
    df.to_csv(join(dir, "uwb_cir.csv"), index=False)
    remove(join(dir, file))


def merge_time(df):
    sec = df["header.stamp.secs"]
    nsec = df["header.stamp.nsecs"]
    df["timestamp"] = sec + nsec / 1e9
    return df


def find_min_timestamp(all_files):
    """Find the minimum timestamp in all csv files."""
    min_timestamp = float('inf')
    for file in all_files:
        if file.endswith('.csv'):
            df = pd.read_csv(file)
            if df["timestamp"].min() < min_timestamp:
                min_timestamp = df["timestamp"].min()
        elif file.endswith('.jpeg'):
            img_timestamp = int(file.split(".")[0].split("/")[-1]) / 1e9
            if img_timestamp < min_timestamp:
                min_timestamp = img_timestamp

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
        if "timestamp_n" in df.columns:
            df["timestamp_n"] = df["timestamp_n"] - min_timestamp
        df.to_csv(file, index=False)
    for file in all_jpegs:
        img_timestamp = int(
            file.split(".")[0].split("/")[-1]) / 1e9 - min_timestamp
        if img_timestamp < 0:
            remove(file)
        else:
            rename(
                file, "/".join(file.split("/")[:-1]) + "/" +
                str(img_timestamp) + ".jpeg")

    # Save timeshift to yaml file
    if not isfile(path + "/timeshift.yaml"):
        seconds = int(min_timestamp)
        nanoseconds = int((min_timestamp - seconds) * 1e9)
        with open(path + "/timeshift.yaml", 'w') as file:
            yaml.dump({
                'timeshift_s': seconds,
                'timeshift_ns': nanoseconds
            },
                      file,
                      default_flow_style=False)


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print(
            "Not enough arguments. Usage: python cleanup_csv.py path_to_csvs")
        sys.exit(1)

    path = sys.argv[1]
    if path.endswith('/'):
        path = path[:-1]

    files = [f for f in listdir(path) if f.endswith('.bag')]

    for file in files:
        cleanup_csvs(join(path, file.split(".")[0]))

    shift_timestamps(path)
