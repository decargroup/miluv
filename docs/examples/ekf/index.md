---
title: Extended Kalman Filter
parent: Examples
nav_order: 4
---

# Extended Kalman Filter (EKF) Examples

The following examples demonstrate the usage of the MILUV dataset with the Extended Kalman Filter (EKF) for state estimation. We show EKF examples that use IMU for state propagation and UWB for state correction in single- and multi-robot scenarios. We also show EKF examples that use Visual-Inertial Navigation System (VINS) output for state propagation in a loosely-coupled manner, alongside UWB for state correction in single- and multi-robot scenarios.

The goal of these examples is to demonstrate how to use the MILUV dataset, and to provide a starting point for users to develop their own localization algorithms. The focus is not on the EKF implementation itself, and as such we split the examples into two parts: a main script for every example that loads the data and runs the EKF, and a corresponding utility script that contains the EKF model and other utilities. The files are organized as follows:

```
├── ekf_imu_one_robot.py
├── ekf_imu_three_robots.py
├── ekf_vins_one_robot.py
├── ekf_vins_three_robots.py
├── ekfutils
│   ├── common.py
│   ├── imu_models.py
│   ├── imu_one_robot_models.py
│   ├── imu_three_robots_models.py
│   ├── vins_one_robot_models.py
│   └── vins_three_robots_models.py
```

As you can see, the main scripts are named according to the scenario they demonstrate, and are in the form `ekf_<model>_<num_robots>.py`. The utility scripts are named according to the model they contain, and are in the form `<model>_<num_robots>_models.py`. The `common.py` script contains utilities that are common to all EKF models, such as an outlier rejection algorithm and classes to store the EKF state and covariance history. Meanwhile, the `imu_models.py` is IMU-EKF-specific and contains necessary functions for the IMU EKF models. We focus on explaining how to load and use data from MILUV in these examples, but also do provide a brief explanation of the EKF models used for the interested reader.
