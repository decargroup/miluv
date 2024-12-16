---
title: Getting Started
nav_order: 2
usemathjax: true
---

# Getting Started

This page provides a guide on how to set up the MILUV devkit and dataset. The MILUV devkit is a Python package that provides tools to work with the MILUV dataset. Details on the dataset can be found [here](https://decargroup.github.io/miluv/docs/data). The devkit provides tools to load data from the dataset, visualize data, and run experiments on the dataset. The devkit can be installed locally or can be run virtually using Docker, as outlined below.

## Devkit installation 

### Local installation
To install the MILUV devkit, clone the repository by running

```
git clone https://github.com/decargroup/miluv.git
```

in the directory where you would like to store the repository. To intall the devkit locally, run

```
pip3 install .
```

in the root directory of the repository. This will install the devkit and its dependencies. For a list of all dependencies, refer to `requirements.txt` in the repository's root directory.

### Docker installation
To install the MILUV devkit using Docker, you must have Docker installed on your machine, which can be installed by following the instructions [here](https://docs.docker.com/get-docker/).

Start by cloning the repository by running

```
git clone https://github.com/decargroup/miluv.git
```

in the directory where you would like to store the repository. To build the Docker image, run

```
docker build -t miluv .
```

in the root directory of the repository. This will create a Docker image called `miluv` that contains the devkit and its dependencies. This took 2 minutes on my machine and with my internet connection.

## Dataset setup
The devkit is used alongside the MILUV dataset, and by default the devkit expects that the data for each experiment is available in `/path/to/miluv/data/{exp_number}`, where `{exp_number}` is the number of the experiment outlined [here](https://decargroup.github.io/miluv/docs/data#summary-of-experiments). If you wish to change the default data location, be sure to specify the data directory when creating an instance of the MILUV Dataloader, but it is recommended to keep the data in the default location. You do not have to download the entire dataset, but at least the experiments that you would like to work with.

## Using the devkit

### Local usage

After installing the devkit, you should be able to use the devkit by importing it in your Python scripts. For example, to use the `DataLoader`, you can add

```python
from miluv.data import DataLoader
```

to the top of your script. 

If you want to use the devkit as a ROS package, you will have to create a symlink to the devkit and the `uwb_ros` subdirectory in the `src` directory of your ROS workspace. To do this, run

```
ln -s /path/to/miluv /path/to/your/ros/workspace/src/miluv
ln -s /path/to/miluv/uwb_ros /path/to/your/ros/workspace/src/uwb_ros
```

and then build your ROS workspace as you normally would.

### Docker usage

Alternatively, you can run the devkit using Docker, which sets up the environment for you. To run the Docker container, we provide a Docker compose file that will mount the necessary directories and run the container. To run the container, run

```
docker compose up
```

from the root directory of the repository, which will start the container. To open a shell inside the container, run in a separate terminal

```
docker exec -it miluv-miluv-1 bash
``` 


## Troubleshooting

If you encounter any issues during installation or usage of the MILUV devkit, you can open an issue on the [GitHub repository](https://github.com/decargroup/miluv/issues) for further assistance.

You are now ready to start using the MILUV devkit! For examples on how to use the devkit, refer to the [Examples](https://decargroup.github.io/miluv/docs/examples). 
