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

# Set the plotting environment
set_plotting_env()

# The configuration files
# TODO: must dynamically load the appropriate config file based on # of robots + if has anchors 
config = "data/1c/uwb_config.config"

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
    "config/calib_results.pickle",
)

# Apply the calibration
df = data.df
df_passive = data.df_passive
df = ApplyCalibration.antenna_delays(
    df, 
    calib_results["delays"], 
    max_value=1e9 * (1.0 / 499.2e6 / 128.0) * 2.0**32
)
df_passive = ApplyCalibration.antenna_delays_passive(
    df_passive, 
    calib_results["delays"]
)
df = ApplyCalibration.power(
    df, 
    calib_results["bias_spl"], 
    calib_results["std_spl"], 
    max_value=1e9 * (1.0 / 499.2e6 / 128.0) * 2.0**32
)
df_passive = ApplyCalibration.power_passive(
    df_passive, 
    calib_results["bias_spl"], 
    calib_results["std_spl"]
)

df["timestamp"] = (df["time"]*1e9).astype(int)
df.drop(columns=["time"], inplace=True)
df_passive["timestamp"] = (df_passive["time"]*1e9).astype(int)
df_passive.drop(columns=["time"], inplace=True)

# TODO: from the indices add back the info to df_passive
# TODO: drop large ranges
# TODO: separate for each robot and save csvs

# %%
