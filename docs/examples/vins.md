---
title: Visual-Inertial Navigation
parent: Examples
nav_order: 3
---

# Visual-Inertial Navigation System (VINS) Example

<div align="center">
    <img src="https://decargroup.github.io/miluv/assets/vins.gif" alt="VINS Visualization">
</div>

## Overview
This example shows how we can use the MILUV dataset to test a visual-inertial localization solution. For this example, we use [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion). Given that VINS-Fusion requires the installation of certain dependencies such as ROS and Ceres Solver, we provide a Dockerfile that contains all the necessary dependencies to run VINS-Fusion.

## Setting up the Workspace
To run this example, you must have Docker installed on your machine, which can be installed by following the instructions [here](https://docs.docker.com/get-docker/). Additionally, we assume that you the host machine has a NVIDIA GPU, and that you have installed the NVIDIA Container Toolkit, which can be installed by following the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Another prerequisite is that you have the MILUV dataset downloaded and placed in a directory called `data` in the root directory of the repository, as indicated in the [Getting Started](https://decargroup.github.io/miluv/docs/gettingstarted.html) page. You do not need the entire dataset downloaded, but at least the experiments that you would like to test this example on. Additionally, for this example, you should add a `vins` subdirectory to the `data` directory by running
```
mkdir -p data/vins
```
from the root directory of the repository. This is where the VINS-Fusion output will be saved.

## Building the Docker Image
The Dockerfile is located in the `examples` directory, and is called `vins.Dockerfile`. To build the Docker image, run
```
docker build -f examples/vins.Dockerfile -t miluv_vins .
```
from the root directory of the repository, which will create a Docker image called `miluv_vins`. This took 11 minutes on my machine and with my internet connection. This image contains all the necessary dependencies to run VINS-Fusion and the MILUV devkit.

## Running the Docker Container 
Now that the the workspace is set up and the Docker image is built, we will now run the Docker container. To simplify the process, we provide a Docker compose file that will mount the necessary directories and run the container. To run the container, run
```
docker compose -f examples/vins_docker_compose.yaml up
```
from the root directory of the repository, which will start the container. If you do not have a NVIDIA GPU, you can remove the `runtime: nvidia` line from the `vins_docker_compose.yaml` file, but you might not be able to see the RViz visualization. To open a shell inside the container, run in a separate terminal
```
docker exec -it examples-miluv-1 bash
```

## Running the Example
Once inside the container, you can run the VINS-Fusion example by running
```
./examples/run_vins.sh <exp_name> <robot>
```
where `<exp_name>` is the name of the experiment you would like to run VINS-Fusion on, and `<robot>` is the robot ID you would like to run VINS-Fusion on. For example, to run VINS-Fusion on experiment `default_3_random_0` with robot ID `ifo001`, you would run
```
./examples/run_vins.sh default_3_random_0 ifo001
```
Assuming everything has been set up correctly, you should see the RViz visualization showing the video feed, the robot's estimated trajectory, and a map of the keypoints detected by VINS-Fusion as shown at the top of this page. 

The output of VINS-Fusion will be saved in the `data/vins` directory in a directory with the same name as the experiment. The file `{exp_name}/{robot}_vio.csv` contains the estimated pose of the trajectory in the absence of loop closures, while the file `{exp_name}/{robot}_vio_loop.csv` contains the estimated pose of the trajectory with loop closures. The `alignment_pose.yaml` provides the estimated transformation between the VINS frame and the Mocap frame, and the filed appended with `aligned_and_shifted` contain the estimated pose of the trajectory after aligning and shifting the estimated trajectory to the ground truth trajectory.
