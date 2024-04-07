#!/bin/bash

# Assuming that we have already run 'sh install_torch_scatter.sh'
# Install robotarium
pip install git+https://github.com/robotarium/robotarium_python_simulator.git@6bb184e#egg=robotarium_python_simulator

# Get MARBLER repo
cd ./../
if [ ! -d "$(pwd)/3rdparty" ]; then
    mkdir 3rdparty
fi
cd 3rdparty
git clone https://github.com/GT-STAR-Lab/MARBLER.git
cd MARBLER/

# Install MARBLER
pip install -e .
