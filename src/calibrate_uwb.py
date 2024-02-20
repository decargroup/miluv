# %%
from pyuwbcalib.machine import RosMachine
from pyuwbcalib.postprocess import PostProcess
from pyuwbcalib.utils import save, set_plotting_env, merge_calib_results
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

bias_raw = np.empty(0)
bias_antenna_delay = np.empty(0)
bias_fully_calib = np.empty(0)
calib_results_list = []

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
    bias_raw = np.append(bias_raw, np.array(calib.df['bias']))

    # Correct antenna delays
    calib.calibrate_antennas(inplace=True, loss='huber')

    # Compute the antenna-delay-corrected measurements
    bias_antenna_delay = np.append(bias_antenna_delay, np.array(calib.df['bias']))

    # Correct power-correlated bias
    calib.fit_power_model(
        inplace = True,
    )

    # Compute the fully-calibrated measurements
    bias_fully_calib = np.append(bias_fully_calib, np.array(calib.df['bias']))

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
    
    calib_results_list.append(calib_results)
    
calib_results = merge_calib_results(calib_results_list)

plt.rc('legend', fontsize=40)
print(calib_results['delays'])

fig, axs = plt.subplots(2, 1, sharex=True)
x = np.linspace(0, 1.5)
axs[0].plot(x, calib.bias_spl(x)*100, label='Bias')
axs[1].plot(x, calib.std_spl(x)*100, label='Standard deviation')
axs[0].set_ylabel('Bias [cm]')
axs[0].set_yticks([-10, -5, 0, 5, 10])
axs[1].set_ylabel('Bias Std. [cm]')
axs[1].set_xlabel("Lifted signal strength")
axs[1].set_yticks([0, 10, 20])
axs[1].set_xticks(np.arange(0, 1.6, 0.2))

bins = 200
fig2 = plt.figure()
fig2.hist(bias_raw, density=True, bins=bins, alpha=0.5, label='Raw')
fig2.hist(bias_antenna_delay, density=True, bins=bins, alpha=0.5, label='Antenna-delay calibrated')
fig2.hist(bias_fully_calib, density=True, bins=bins, alpha=0.5, label='Fully calibrated')
fig2.xticks(np.arange(-0.4, 1, 0.2))
fig2.xlabel('Bias [m]')
fig2.xlim([-0.5, 1])
fig2.legend()

fig.savefig('figs/calib_results.pdf')
fig2.savefig('figs/bias_histogram.pdf')

plt.show()
    

# %%
