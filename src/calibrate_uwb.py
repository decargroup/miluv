# %%
from pyuwbcalib.machine import RosMachine
from pyuwbcalib.postprocess import PostProcess
from pyuwbcalib.utils import save, set_plotting_env
from pyuwbcalib.uwbcalibrate import UwbCalibrate
from configparser import ConfigParser, ExtendedInterpolation
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Set the plotting environment
set_plotting_env()

# The configuration files
config_files = [
    'data/bias_calibration_anchors0/config.config',
    'data/bias_calibration_anchors1/config.config',
    'data/bias_calibration_tags0/config.config',
    'data/bias_calibration_tags1/config.config',
]

for config in config_files:
    # Parse through the configuration file
    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read(config)

    # Create a RosMachine object for every machine
    machines = {}
    for i,machine in enumerate(parser['MACHINES']):
        machine_id = parser['MACHINES'][machine]
        machines[machine_id] = RosMachine(parser, i)

    # Process and merge the data from all the machines
    data = PostProcess(machines)

    # Instantiate a UwbCalibrate object, and remove static extremes
    calib = UwbCalibrate(data, rm_static=True)

    # Compute the raw bias measurements
    bias_raw = np.array(calib.df['bias'])

    # Correct antenna delays
    calib.calibrate_antennas(inplace=True, loss='huber')

    # Compute the antenna-delay-corrected measurements
    bias_antenna_delay = np.array(calib.df['bias'])

    # Correct power-correlated bias
    calib.fit_power_model(
        inplace = True,
    )

    # Compute the fully-calibrated measurements
    bias_fully_calib = np.array(calib.df['bias'])

    # Save the calibration results
    calib_results = {
        'delays': calib.delays,
        'bias_spl': calib.bias_spl,
        'std_spl': calib.std_spl,
    }
    save(
        calib_results, 
        config.split('/config')[0] + '/calib_results.pickle'
    )

    

# %%
