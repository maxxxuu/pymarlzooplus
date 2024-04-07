#!/bin/bash
# Install SC2 and add the custom maps

# Get PyTorch version and CUDA version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" | cut -d '+' -f1)
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" | tr -d '.')

# Construct the installation command with CUDA support
pip install torch_scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html

cd ./../
if [ ! -d "$(pwd)/3rdparty" ]; then
    mkdir 3rdparty
fi
cd 3rdparty

export SC2PATH=`pwd`'/StarCraftII'
echo 'SC2PATH is set to '$SC2PATH

if [ ! -d $SC2PATH ]; then
        echo 'StarCraftII is not installed. Installing now ...';
        wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.6.2.69232.zip
        unzip -P iagreetotheeula SC2.4.6.2.69232.zip
        rm -rf SC2.4.6.2.69232.zip
else
        echo 'StarCraftII is already installed.'
fi

echo 'Adding SMAC maps.'
MAP_DIR="$SC2PATH/Maps/"
echo 'MAP_DIR is set to '$MAP_DIR

if [ ! -d $MAP_DIR ]; then
        mkdir -p $MAP_DIR
fi

cd ..
wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
unzip SMAC_Maps.zip
mv SMAC_Maps $MAP_DIR
rm -rf SMAC_Maps.zip
rm -rf __MACOSX

echo 'StarCraft II and SMAC are installed.'

