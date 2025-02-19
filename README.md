## <img src="https://github.com/decargroup/miluv/blob/gh-pages/assets/decar_logo.png?raw=true" alt="DECAR Logo" width="14"/> DECAR &mdash; MILUV devkit
Welcome to the MILUV devkit page. This Python devkit provides useful functions and examples to accompany the MILUV dataset. To begin using this devkit, clone or download and extract the repository.
![](https://github.com/decargroup/miluv/blob/gh-pages/assets/banner_image.jpg?raw=true)

## Table of Contents
- [Changelog](#changelog)
- [Devkit setup and installation](#devkit-setup-and-installation)
- [Getting started with MILUV](#getting-started-with-MILUV)
    - [Setting up the dataset](#setting-up-the-dataset)
    - [Examples](#examples)
- [Wiki](#wiki)
- [License](#license)

## Changelog
03-07-2024: MILUV devkit v1.0.0 released.

## Devkit setup and installation
The devkit requires Python 3.8 or greater. To install the devkit and its dependencies, run
```
pip3 install .
``` 
inside the devkit's root directory (~/path/to/project/MILUV). 

Alternatively, run
```
pip3 install -e .
```
inside the devkit's root directory, which installs the package in-place, allowing you make changes to the code without having to reinstall every time. 

For a list of all dependencies, refer to ``requirements.txt`` in the repository's root directory.

To ensure installation was completed without any errors, test the code by running
```
pytest
```    
in the root directory.

## Getting started with MILUV
### Setting up the dataset
To get started, download the MILUV dataset. By default, the devkit expects the data for each experiment is present in **/miluv/data/EXP_NUMBER**, where EXP_NUMBER is the number of the experiment.

If you wish to change the default data location, be sure to modify the data directory in the devkit code.

### Examples
Using the MILUV devkit, retrieving sensor data by timestamp from experiment ``1c`` can be implemented as:
```py
from miluv.data import DataLoader
import numpy as np

mv = DataLoader(
    "default_3_random_0",
    height=False,
)

timestamps = np.arange(0, 10, 1)  # Time in s

data_at_timestamps = mv.data_from_timestamps(timestamps)
```

This example can be made elaborate by selecting specific robots and sensors to fetch from at the given timestamps.
```py
from miluv.data import DataLoader
import numpy as np

mv = DataLoader(
    "default_3_random_0",
    height=False,
)

timestamps = np.arange(0, 10, 1)  # Time in s

robots = ["ifo001", "ifo002"]  # We are leaving out ifo003
sensors = ["imu_px4", "imu_cam"]  # Fetching just the imu data

data_at_timestamps = mv.data_from_timestamps(
    timestamps,
    robots,
    sensors,
)
```

## Wiki
For more information regarding the MILUV development kit, please refer to the [documentation](https://decargroup.github.io/miluv/).

## License
This development kit is distributed under the MIT License.
