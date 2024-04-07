#!/bin/bash

pip install protobuf==3.20.*

# Get PyTorch version and CUDA version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" | cut -d '+' -f1)
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")

# Construct the installation command with CUDA support
pip install torch_scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html