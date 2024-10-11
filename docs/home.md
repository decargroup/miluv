---
title: Home
nav_order: 1
usemathjax: true
---

# MILUV Dataset

Introducing MILUV, the first multi-robot dataset with ultra-wide band (UWB) measurements. This dataset features
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
