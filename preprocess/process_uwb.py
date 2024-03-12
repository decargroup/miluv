# %%
from pyuwbcalib.machine import RosMachine
from pyuwbcalib.postprocess import PostProcess
from pyuwbcalib.utils import load, read_anchor_positions
from pyuwbcalib.uwbcalibrate import ApplyCalibration
import sys
from os.path import join
import os
import pandas as pd
import yaml
from miluv.utils import (
    get_experiment_info, 
    get_anchors, 
    get_tags
)

    
def generate_config(exp_info):
    params = {
        "max_ts_value": "2**32",
        "ts_to_ns": "1e9 * (1.0 / 499.2e6 / 128.0)",
        "ds_twr": "True",
        "passive_listening": "True",
        "fpp_exists": "True",
        "rxp_exists": "False",
        "std_exists": "False",
    }
    
    pose_path = {
        "directory": f"data/{exp_info['experiment']}/",
    }
    for i in range(exp_info["num_robots"]):
        pose_path.update({
            f"{i}": f"ifo00{i+1}.bag"
        })
        
    uwb_path = pose_path.copy()
    anchors = get_anchors(str(exp_info["anchor_constellation"]))
    machines = {}
    for i in range(exp_info["num_robots"]):
        machines.update({f"{i}": f"ifo00{i+1}"})
    
    tags = {}
    for i in range(exp_info["num_robots"]):
        if exp_info["num_tags_per_robot"] == 2:
            tags.update({f"{i}": f"[{(i+1)*10}, {(i+1)*10 + 1}]"})
        elif exp_info["num_tags_per_robot"] == 1:
            tags.update({f"{i}": f"[{(i+1)*10}]"})
        
    moment_arms = get_tags(flatten=True)
    
    pose_topic = {}
    for i in range(exp_info["num_robots"]):
        pose_topic.update({
            f"{i}": f"/ifo00{i+1}/vrpn_client_node/ifo00{i+1}/pose"
        })
        
    uwb_topic = {}
    for i in range(exp_info["num_robots"]):
        uwb_topic.update({
            f"{i}": f"/ifo00{i+1}/uwb/range"
        })
        
    listening_topic = {}
    for i in range(exp_info["num_robots"]):
        listening_topic.update({
            f"{i}": f"/ifo00{i+1}/uwb/passive"
        })
    
    uwb_message = {
        "from_id": "from_id",
        "to_id": "to_id",
        "tx1": "tx1",
        "rx1": "rx1",
        "tx2": "tx2",
        "rx2": "rx2",
        "tx3": "tx3",
        "rx3": "rx3",
        "fpp1": "fpp1",
        "fpp2": "fpp2",
    }
    
    listening_message = {
        "my_id": "my_id",
        "from_id": "from_id",
        "to_id": "to_id",
        "covariance": "covariance",
        "rx1": "rx1",
        "rx2": "rx2",
        "rx3": "rx3",
        "tx1_n": "tx1_n",
        "rx1_n": "rx1_n",
        "tx2_n": "tx2_n",
        "rx2_n": "rx2_n",
        "tx3_n": "tx3_n",
        "rx3_n": "rx3_n",
        "fpp1": "pr1",
        "fpp2": "pr2",
        "fpp3": "pr3",
        "fpp1_n": "pr1_n",
        "fpp2_n": "pr2_n ",
    }
    
    return {
        "PARAMS": params,
        "POSE_PATH": pose_path,
        "UWB_PATH": uwb_path,
        "ANCHORS": anchors,
        "MACHINES": machines,
        "TAGS": tags,
        "MOMENT_ARMS": moment_arms,
        "POSE_TOPIC": pose_topic,
        "UWB_TOPIC": uwb_topic,
        "LISTENING_TOPIC": listening_topic,
        "UWB_MESSAGE": uwb_message,
        "LISTENING_MESSAGE": listening_message
    }
    
def process_uwb(path):
    # The configuration files
    # TODO: must dynamically load the appropriate config file based on # of robots + if has anchors 
    # config = join(path, "uwb_config.config")

    # parser = ConfigParser(interpolation=ExtendedInterpolation())
    # parser.read(config)
    exp_info = get_experiment_info(path)
    uwb_config = generate_config(exp_info)

    # Read anchor positions
    anchor_positions = read_anchor_positions(uwb_config)

    # Create a RosMachine object for every machine
    machines = {}
    for i,machine in enumerate(uwb_config['MACHINES']):
        machine_id = uwb_config['MACHINES'][machine]
        machines[machine_id] = RosMachine(uwb_config, i)

    # Process and merge the data from all the machines
    data = PostProcess(machines, anchor_positions)

    # Load the UWB calibration results
    calib_results = load(
        "config/uwb/uwb_calib.pickle",
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
    df["timestamp"] = df["time"]
    df.drop(columns=["time"], inplace=True)
    df_passive["timestamp"] = df_passive["time"]
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
            join(path, f"{robot}/uwb_passive.csv"),
            index=False
        )

if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Not enough arguments. Usage: python cleanup_csv.py path_to_csvs")
        sys.exit(1)
    path = sys.argv[1]
    
    process_uwb(path)

    # Remove the bagreader-generated UWB csv files
    robots = [f for f in os.listdir(path) if f.endswith('.bag')]
    for robot in robots:
        robot_id = robot.split('.')[0]
        robot_folder = os.path.join(path, robot_id)
        for file in os.listdir(robot_folder):
            file_path = os.path.join(robot_folder, file)
            if robot_id in file and "uwb" in file and "cir" not in file:
                os.remove(file_path)
    


# %%
