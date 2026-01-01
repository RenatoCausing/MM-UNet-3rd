#!/bin/bash
# ==========================================
# MM-UNet FIVEs Setup for vastai/base-image:cuda-11.8.0-cudnn8-devel-ubuntu22.04-py310
# This Docker image already has CUDA 11.8 and Python 3.10
# Just install Python packages from requirements.txt
# ==========================================

echo "=========================================="
echo "MM-UNet FIVEs Dataset Setup"
echo "Using Docker image: cuda-11.8.0-cudnn8-devel-ubuntu22.04-py310"
echo "=========================================="

# Verify environment
echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version | grep release)"

# Install system dependencies for OpenCV
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 unzip git

# Upgrade pip
pip install --upgrade pip

echo "Installing packages from requirements.txt..."

# Install PyTorch 2.0.0 with CUDA 11.8 FIRST
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch
python -c "import torch; print(f'✓ PyTorch {torch.__version__} CUDA {torch.version.cuda}')"

# Install compatible versions BEFORE requirements.txt to prevent upgrades
echo "Installing compatible package versions for PyTorch 2.0..."
pip install transformers==4.30.2 mmengine==0.7.4 timm==0.4.12 accelerate==0.18.0 monai==1.1.0

# Install all packages from requirements.txt
pip install -r requirements/requirements.txt

# Verify PyTorch didn't upgrade
python -c "import torch; v = torch.__version__; assert v.startswith('2.0.0'), f'ERROR: PyTorch upgraded to {v}'; print(f'✓ PyTorch still {v}')"

# Compile CUSTOM Mamba from source (with bimamba_type and nslices parameters)
echo ""
echo "Compiling custom Mamba from source..."
cd requirements/Mamba/causal-conv1d
pip install . --no-build-isolation
cd ../mamba
pip install . --no-build-isolation
cd ../../..

# Verify custom Mamba has the required parameters
echo "Verifying custom Mamba..."
python -c "
from mamba_ssm import Mamba
import inspect
sig = str(inspect.signature(Mamba.__init__))
assert 'bimamba_type' in sig and 'nslices' in sig, f'Missing custom parameters! Got: {sig}'
print('✓ Custom Mamba with bimamba_type and nslices verified')
"

# Install additional packages for training
pip install easydict gdown

# Final verification
echo ""
echo "Verifying installation..."
python -c "
import torch
import mamba_ssm
import monai
import timm
print('✓ PyTorch:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
print('✓ mamba-ssm imported successfully')
print('✓ MONAI imported successfully')
print('✓ timm imported successfully')
"

# Download FIVEs dataset
echo ""
echo "Downloading FIVEs dataset..."
gdown --id 1VTFhKLxdzQAZv3Jj4mZgixI70RzfF68p
unzip -q fives_preprocessed.zip
rm fives_preprocessed.zip

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "To start training:"
echo "  python train_fives.py"
echo ""
echo "To test after training:"
echo "  python test_fives.py --checkpoint ./checkpoints_fives/best_model.pth --data_root ./fives_preprocessed --save_predictions"
echo "=========================================="
