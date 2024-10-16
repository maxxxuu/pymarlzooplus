#!/bin/bash

# Get PyTorch version and CUDA version
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" | cut -d '+' -f1)
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")

# Check if CUDA_VERSION is None or empty
if [ "$CUDA_VERSION" = "None" ] || [ -z "$CUDA_VERSION" ]; then
    # If CUDA version is None or empty, use the CPU wheel
    python3 -m pip install torch_scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html
else
    # If CUDA version is available, use the CUDA wheel
    python3 -m pip install torch_scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html
fi
