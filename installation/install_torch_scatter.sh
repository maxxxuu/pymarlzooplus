#!/bin/bash

# Get PyTorch version and CUDA version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" | cut -d '+' -f1)
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")

# Check if CUDA_VERSION is None or empty
if [ "$CUDA_VERSION" = "None" ] || [ -z "$CUDA_VERSION" ]; then
    # If CUDA version is None or empty, use the CPU wheel
    pip install torch_scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html
else
    # If CUDA version is available, use the CUDA wheel
    pip install torch_scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html
fi
