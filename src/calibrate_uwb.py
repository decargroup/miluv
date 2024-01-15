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

# The configuration file
config_file = 'data/bias_calibration_tags1/config.config'

# Parse through the configuration file
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read(config_file)

# Create a RosMachine object for every machine
machines = {}
for i,machine in enumerate(parser['MACHINES']):
    machine_id = parser['MACHINES'][machine]
    machines[machine_id] = RosMachine(parser, i)
# %%
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
    visualize = True
)

# Compute the fully-calibrated measurements
bias_fully_calib = np.array(calib.df['bias'])
# %%
# Plot the measurements pre- and post-correction.
plt.figure()
bins = np.linspace(-0.5,1,100)
plt.hist(bias_raw,bins=bins, alpha=0.5, density=True)
plt.hist(bias_antenna_delay, bins=bins, alpha=0.5, density=True)
plt.hist(bias_fully_calib, bins=bins, alpha=0.5, density=True)

plt.show()

print(calib.delays)

# %%
