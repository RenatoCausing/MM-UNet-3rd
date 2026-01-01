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

echo "Installing PyTorch 2.0.0 with CUDA 11.8 (LOCKED)..."
# Install PyTorch 2.0.0 with CUDA 11.8 FIRST and LOCK IT
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch
python -c "import torch; print(f'✓ PyTorch {torch.__version__} CUDA {torch.version.cuda}')"

echo "Installing all required packages with specific versions..."
# Install ALL packages manually to prevent PyTorch upgrade (DO NOT use requirements.txt)
pip install numpy==1.24.3 pillow opencv-python scipy scikit-learn pandas matplotlib
pip install transformers==4.30.2 mmengine==0.7.4 timm==0.4.12 accelerate==0.18.0 monai==1.1.0
pip install tensorboard yacs easydict pyyaml objprint
pip install SimpleITK nibabel openpyxl gdown
pip install einops

# Verify PyTorch STILL 2.0.0
python -c "import torch; v = torch.__version__; assert v.startswith('2.0.0'), f'ERROR: PyTorch upgraded to {v}!'; print(f'✓ PyTorch still {v}')"

# Install standard mamba-ssm (pre-compiled wheel)
echo ""
echo "Installing standard mamba-ssm..."
pip install mamba-ssm causal-conv1d

echo "✓ Installation complete"

# Install additional packages for training
pip install gdown

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
