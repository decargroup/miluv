#!/bin/bash

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <exp_name> <robot>"
    exit 1
fi

# Assign the arguments to variables
exp_name=$1
robot=$2

# Run VINS with a 4 minute timeout
bag=$(rospack find miluv)/data/${exp_name}/${robot}.bag
timeout 240 roslaunch miluv vins.launch robot:=$robot bag:=$bag > /dev/null 2>&1

# Rename the output file 
cd $(rospack find miluv)/data/vins
mkdir -p $exp_name
mv vio.csv $exp_name/${robot}_vio.csv
mv vio_loop.csv $exp_name/${robot}_vio_loop.csv
