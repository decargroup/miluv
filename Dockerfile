# Use the official Ubuntu 24.04 as the base image
FROM ubuntu:24.04

# Set environment variables
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Set working directory
WORKDIR /workspace

# Install dependencies
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
    python3-pip \
    python3-venv \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Make a virtual environment and activate it
RUN python3 -m venv /virtualenv/miluv
RUN echo "source /virtualenv/miluv/bin/activate" >> ~/.bashrc

# Clone the MILUV repository
RUN git clone https://github.com/decargroup/miluv.git

# Make symlink to build uwb_ros with UWB messages
RUN ln -s miluv/uwb_ros .

# Install MILUV 
WORKDIR /workspace/miluv
RUN /bin/bash -c "source /virtualenv/miluv/bin/activate && pip3 install csaps && pip3 install ."

# Install some dependencies for remote visualization
RUN apt update
RUN /bin/bash -c "source /virtualenv/miluv/bin/activate && pip install PyQt5"
RUN apt-get install -y libxcb-xinerama0 libxcb1 libx11-xcb1 libxrender1 libxi6 libxext6
RUN /bin/bash -c "source /virtualenv/miluv/bin/activate && pip install opencv-python-headless"
RUN apt-get install -y qt5-qmake qtbase5-dev qtchooser qt5-qmake-bin libqt5core5a libqt5gui5
ENV XDG_RUNTIME_DIR=/tmp/runtime-dir

# Expose the necessary ports
EXPOSE 11311

# Set entrypoint to bash so you can run commands interactively
CMD ["/bin/bash", "-c", "source /virtualenv/miluv/bin/activate && exec /bin/bash"]