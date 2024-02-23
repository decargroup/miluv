TODO:
- Add UWB ROS messages with instructions 
- UWB calibration in rosbags? Maybe provide a node to do calibration when Rosbag is playing
- Mention lack of camera images outside of rosbags
- Fit spline to mocap and have it as an attribute of Miluv to be called by the user
- Load the CIR files. Is it possible to match each to range/passive measurements?

## Installation
Python 3.8 or greater is required. Inside this repo's directory, you may run

    pip3 install .
or

    pip3 install -e .

which installs the package in-place, allowing you make changes to the code without having to reinstall every time. 