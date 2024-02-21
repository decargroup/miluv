import sys
from os import listdir, remove
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
height = [
    "timestamp",
    "range"
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
uwb_range = [
    "timestamp",
    "range",
    "from_id",
    "to_id",
    "tx1",
    "rx1",
    "tx2",
    "rx2",
    "tx3",
    "rx3",
    "fpp1",
    "fpp2",
    "skew1",
    "skew2"
]
uwb_passive= [
    "timestamp",
    "my_id",
    "from_id",
    "to_id",
    "rx1",
    "rx2",
    "rx3",
    "tx1_n",
    "rx1_n",
    "tx2_n",
    "rx2_n",
    "tx3_n",
    "rx3_n",
    "fpp1",
    "fpp2",
    "fpp3",
    "skew1",
    "skew2",
    "skew3",
    "fpp1_n",
    "fpp2_n",
    "skew1_n",
    "skew2_n"
]

def cleanup_csvs(dir):
    # Find all csv files
    files = [f for f in listdir(dir) if f.endswith('.csv')]

    for file in files:
        if "imu" in file and "camera" in file:
            process_csv(dir, file, imu, "imu_cam")
        elif "imu" in file and "mavros" in file:
            process_csv(dir, file, imu, "imu_px4")
        elif "mag" in file:
            process_csv(dir, file, mag, "mag")
        elif "hrlv" in file:
            process_csv(dir, file, height, "height")
        elif "vrpn" in file:
            process_csv(dir, file, mocap, "mocap")
        elif "range" in file:
            process_csv(dir, file, uwb_range, "uwb_range")
        elif "passive" in file:
            process_csv(dir, file, uwb_passive, "uwb_passive")

def process_csv(dir, file, headers, name):
    df = pd.read_csv(join(dir, file))
    merge_time(df)
    df = df[headers]
    df.to_csv(join(dir, name + ".csv"), index=False)
    remove(join(dir, file))

def merge_time(df):
    sec = df["header.stamp.secs"]
    nsec = df["header.stamp.nsecs"]
    df["timestamp"] = sec * 1e9 + nsec
    df["timestamp"] = df["timestamp"].astype(int)
    return df

if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Not enough arguments. Usage: python cleanup_csv.py path_to_csvs")
        sys.exit(1)
    
    path = sys.argv[1]
    files = [f for f in listdir(path) if f.endswith('.bag')]
    
    for file in files:
        cleanup_csvs(join(path, file.split(".")[0]))
