---
title: Home
nav_order: 1
usemathjax: true
permalink: /
layout: default
---

# MILUV Dataset

Introducing MILUV, the first multi-robot dataset with ultra-wideband (UWB) measurements and vision data. This dataset features
* Pose data
  * On-board inertial measurement unit (IMU) data
  * Camera IMU data from Intel RealSense d435i
  * Height measurements from an onboard downward-facing laser rangefinder
  * Ground truth pose data from motion capture system
* UWB data from Decawave DWM1000
  * Range measurements
  * Passive listening measurements
  * Channel impulse response (CIR) data
  * Low-level data (raw ROS timestamps, received signal power)
* Vision data with AprilTag as features
  * Downward-facing camera measurements
  * Monocular and colour forward-facing camera measurements from onboard camera
  * Stereo and infrared forward-facing camera measurements from RealSense d435i
* Magnetometer data

# MILUV Devkit
MILUV is accompanied by a Python devkit to facilitate the development of applications that make use of the dataset. This devkit features
* Parsing and data loading scripts
* Three localization benchmarks with which users can compare their localization algorithms
  * $SE(3)$ extended Kalman filter
  * $SE_{2}(3)$ extended Kalman filter
  * Visual inertial odometry using VINS-Fusion
* Examples of alternative uses for the dataset
  * Line-of-sight/non line-of-sight classification from CIR data
  * AprilTag detection from camera measurements

The repository is available [here](https://github.com/decargroup/miluv), and the directory structure of the repository is as follows:
```
miluv
├── config
│   ├── apriltags    (AprilTag positions)
│   ├── height       (Height bias)
│   ├── imu          (Camera IMU and PX4 IMU calibration files from the allan_variance_ros package)
│   ├── realsense    (Camera calibration files from Kalibr)
│   ├── setup        (The files provided by Vicon mocap system for apriltags and UWB anchors)
│   ├── uwb          (Anchor and tag position files and UWB calibration results)
│   ├── vins         (VINS-Fusion configuration files)
├── data             (this is where the dataset should be downloaded)
├── examples         (Example scripts on how to use the devkit)
├── launch           (ROS launch files)
├── miluv            (The devkit with the data loading and utility scripts)
├── paper            (Scripts used to produce the results in the paper)
├── preprocess       (Scripts used to preprocess the raw data into the dataset format)
├── tests            (Unit tests)
├── uwb_ros          (ROS package with msg files for UWB data)
```

To get started with the MILUV dataset, follow the instructions in the [Getting Started](https://decargroup.github.io/miluv/docs/gettingstarted.html) page.
