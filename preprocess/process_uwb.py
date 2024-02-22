# %%
from pyuwbcalib.machine import RosMachine
from pyuwbcalib.postprocess import PostProcess
from pyuwbcalib.utils import load, set_plotting_env, read_anchor_positions
from pyuwbcalib.uwbcalibrate import ApplyCalibration
from configparser import ConfigParser, ExtendedInterpolation
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sys
from os.path import join
import os

# Set the plotting environment
set_plotting_env()

def process_uwb(path):
    # The configuration files
    # TODO: must dynamically load the appropriate config file based on # of robots + if has anchors 
    config = join(path, "uwb_config.config")

    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read(config)

    # Read anchor positions
    anchor_positions = read_anchor_positions(parser)

    # Create a RosMachine object for every machine
    machines = {}
    for i,machine in enumerate(parser['MACHINES']):
        machine_id = parser['MACHINES'][machine]
        machines[machine_id] = RosMachine(parser, i)

    # Process and merge the data from all the machines
    data = PostProcess(machines, anchor_positions)

    # Load the UWB calibration results
    calib_results = load(
        "config/uwb_calib.pickle",
    )

    # Apply the calibration
    df = data.df
    df["range_raw"] = df["range"]
    df["bias_raw"] = df["bias"]
    df["tx1_raw"] = df["tx1"]
    df["tx2_raw"] = df["tx2"]
    df["tx3_raw"] = df["tx3"]
    df["rx1_raw"] = df["rx1"]
    df["rx2_raw"] = df["rx2"]
    df["rx3_raw"] = df["rx3"]
    df = ApplyCalibration.antenna_delays(
        df, 
        calib_results["delays"], 
        max_value=1e9 * (1.0 / 499.2e6 / 128.0) * 2.0**32
    )
    df = ApplyCalibration.power(
        df, 
        calib_results["bias_spl"], 
        calib_results["std_spl"], 
        max_value=1e9 * (1.0 / 499.2e6 / 128.0) * 2.0**32
    )

    df_passive = data.df_passive
    df_passive["rx1_raw"] = df_passive["rx1"]
    df_passive["rx2_raw"] = df_passive["rx2"]
    df_passive["rx3_raw"] = df_passive["rx3"]
    df_passive = ApplyCalibration.antenna_delays_passive(
        df_passive, 
        calib_results["delays"]
    )
    df_passive = ApplyCalibration.power_passive(
        df_passive, 
        calib_results["bias_spl"], 
        calib_results["std_spl"]
    )

    # Convert timestamps from seconds to nanoseconds
    df["timestamp"] = (df["time"]*1e9).astype(int)
    df.drop(columns=["time"], inplace=True)
    df_passive["timestamp"] = (df_passive["time"]*1e9).astype(int)
    df_passive.drop(columns=["time"], inplace=True)

    # Add back important info to df_passive
    df_iter = df.iloc[df_passive["idx"]]
    to_copy = ["tx1", "rx1", "tx2", "rx2", "tx3", "rx3", "range", "bias",
            "tx1_raw", "rx1_raw", "tx2_raw", "rx2_raw", "tx3_raw", "rx3_raw", "range_raw", "bias_raw",
            "gt_range", "timestamp"]
    for col in to_copy:
        df_passive[col + "_n"] = df_iter[col].values

    # Drop unnecessary columns
    df.drop(
        columns=[
            "header.seq", "header.frame_id", "covariance",
            "tof1", "tof2", "tof3", 
            "sum_t1", "sum_t2"
        ], 
        inplace=True
    )
    df_passive.drop(
        columns=[
            "header.seq", "header.frame_id", "covariance", "idx"
        ],
        inplace=True
    )

    # Separate for each robot and save csvs
    for robot in data.tag_ids:
        tags = data.tag_ids[robot]
        robot_init_bool = df["from_id"].isin(tags)
        robot_targ_bool = df["to_id"].isin(tags)
        df_robot = df[robot_init_bool | robot_targ_bool]
        df_robot.to_csv(
            join(path, f"{robot}/uwb_range.csv"),
            index=False
        )

        robot_init_bool = df_passive["from_id"].isin(tags)
        robot_targ_bool = df_passive["to_id"].isin(tags)
        df_robot = df_passive[robot_init_bool | robot_targ_bool]
        df_robot.to_csv(
            join(path, f"{robot}/uwb_range_passive.csv"),
            index=False
        )

if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Not enough arguments. Usage: python cleanup_csv.py path_to_csvs")
        sys.exit(1)
    
    path = sys.argv[1]
    
    process_uwb(path)

    robots = [f for f in os.listdir(path) if f.endswith('.bag')]

    for robot in robots:
        robot_id = robot.split('.')[0]
        robot_folder = os.path.join(path, robot_id)
        for file in os.listdir(robot_folder):
            file_path = os.path.join(robot_folder, file)
            if robot_id in file and "uwb" in file:
                os.remove(file_path)
    


# %%
