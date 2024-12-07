# Use the official ROS Melodic image as the base image
FROM ros:melodic

# Set environment variables
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Set working directory
WORKDIR /workspace

# Install ROS Melodic and dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    gnupg2 \
    lsb-release \
    wget \
    unzip \
    build-essential \
    git \
    cmake \
    libopencv-dev \
    libeigen3-dev \
    libboost-all-dev \
    libgflags-dev \
    libgtest-dev \
    libyaml-cpp-dev \
    libsuitesparse-dev \
    libgoogle-glog-dev \
    python3-pip \
    python3-colcon-common-extensions \
    ros-melodic-vision-opencv \
    ros-melodic-pcl-ros \
    ros-melodic-tf2 \
    ros-melodic-tf2-ros \
    ros-melodic-std-msgs \
    ros-melodic-geometry-msgs \
    ros-melodic-sensor-msgs \
    ros-melodic-image-transport \
    ros-melodic-compressed-image-transport \
    ros-melodic-rviz \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 and dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libnss3-dev \
    libsqlite3-dev \
    libreadline-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Download and build Python 3.10
RUN wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz \
    && tar -xvf Python-3.10.12.tgz \
    && cd Python-3.10.12 \
    && ./configure --enable-optimizations \
    && make \
    && make install \
    && cd .. \
    && rm -rf Python-3.10.12.tgz Python-3.10.12

# Update alternatives to make python3 point to python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1
RUN update-alternatives --set python3 /usr/local/bin/python3.10
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install numpy pandas csaps scipy matplotlib

# Install Ceres Solver and additional dependencies
RUN sudo ln -s /usr/share/pyshared/lsb_release.py /usr/local/lib/python3.10/site-packages/lsb_release.py
RUN apt-get update && apt-get install -y \
    libceres-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a workspace and clone the VINS-Fusion repository and the MILUV repository
RUN mkdir -p /workspace/src
WORKDIR /workspace/src
RUN git clone https://github.com/decargroup/miluv.git
RUN git clone https://github.com/HKUST-Aerial-Robotics/VINS-Fusion.git

# Make symlink to build uwb_ros with UWB messages
RUN ln -s miluv/uwb_ros .

# Install MILUV 
WORKDIR /workspace/src/miluv
RUN /bin/bash -c "pip3 install ."

# Build the workspace
WORKDIR /workspace
RUN /bin/bash -c "source /opt/ros/melodic/setup.bash && catkin_make -DCMAKE_SUPPRESS_DEVELOPER_WARNINGS=ON"

# Source ROS setup
RUN echo "source /workspace/devel/setup.bash" >> ~/.bashrc

# Expose the necessary ports
EXPOSE 11311

# Install x11-apps for testing GUI applications
RUN apt update
RUN apt install -y x11-apps

# Set entrypoint to bash so you can run commands interactively
WORKDIR /workspace/src/miluv
CMD ["/bin/bash"]