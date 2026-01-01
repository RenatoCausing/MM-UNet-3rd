#!/bin/bash
# ==========================================
# COMPLETE SETUP - Python 3.10 Required
# Forces Python 3.10 venv for compatibility
# Run this from within the cloned MM-UNet-3rd directory
# ==========================================

echo "=========================================="
echo "MM-UNet FIVEs Dataset Setup - Python 3.10"
echo "=========================================="

# 1. System dependencies + Python 3.10
echo "Installing system dependencies and Python 3.10..."
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 unzip git python3.10 python3.10-venv python3.10-dev

# 2. Create Python 3.10 virtual environment
echo "Creating Python 3.10 virtual environment..."
python3.10 -m venv venv
source venv/bin/activate

# 4. Verify Python version (MUST be 3.10.x)
echo "Python version:"
python --version

# 5. Upgrade pip
pip install --upgrade pip

# 6. Install build dependencies
echo "Installing build dependencies..."
pip install wheel setuptools packaging ninja

# 7. Install exact PyTorch with CUDA 11.8
echo "Installing PyTorch 2.0.0 with CUDA 11.8..."
pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 7. Install NumPy (force exact version)
echo "Installing NumPy 1.24.3..."
pip install --force-reinstall numpy==1.24.3

# 8. Install core dependencies
echo "Installing core dependencies..."
pip install timm==0.4.12 objprint==0.2.3 accelerate==0.18.0 monai==1.1.0 mmengine==0.7.4
pip install tensorboard easydict SimpleITK nibabel pyyaml opencv-python openpyxl scikit-learn pandas matplotlib Pillow gdown yacs

# 9. Try compiling Mamba from source (author's intended method)
echo "Attempting to compile Mamba from source..."
cd requirements/Mamba/causal-conv1d
pip install . --no-build-isolation 2>&1 | tee /tmp/causal_install.log
CAUSAL_STATUS=$?
cd ../../..

cd requirements/Mamba/mamba
pip install . --no-build-isolation 2>&1 | tee /tmp/mamba_install.log
MAMBA_STATUS=$?
cd ../../..

# 10. Fallback to prebuilt if compilation failed
echo "Checking Mamba installation..."
python -c "import mamba_ssm; print('Mamba installed successfully')" 2>/dev/null
if [ $? -ne 0 ]; then
  echo "Source compilation failed, installing prebuilt wheels..."
  # Use direct wheel URLs for cu118
  pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.1/causal_conv1d-1.1.1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
  pip install https://github.com/state-spaces/mamba/releases/download/v1.1.1/mamba_ssm-1.1.1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
fi

# 11. Download dataset
echo "Downloading FIVEs dataset..."
gdown --id 1VTFhKLxdzQAZv3Jj4mZgixI70RzfF68p
unzip fives_preprocessed.zip
rm fives_preprocessed.zip

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "To start training:"
echo "  source venv/bin/activate  # If not already activated"
echo "  python train_fives.py"
echo ""
echo "To test after training:"
echo "  python test_fives.py --checkpoint ./checkpoints_fives/best_model.pth --data_root ./fives_preprocessed --save_predictions"
echo "=========================================="
