#!/bin/bash
# ==========================================
# COMPLETE SETUP - Python 3.10 + CUDA 11.8 Required
# Forces Python 3.10 venv and CUDA 11.8 for custom Mamba compilation
# Run this from within the cloned MM-UNet-3rd directory
# ==========================================

echo "=========================================="
echo "MM-UNet FIVEs Dataset Setup - Python 3.10 + CUDA 11.8"
echo "=========================================="

# 1. System dependencies + Python 3.10
echo "Installing system dependencies and Python 3.10..."
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 unzip git python3.10 python3.10-venv python3.10-dev wget

# 2. Install CUDA 11.8 toolkit
echo "Installing CUDA 11.8 toolkit..."
# Remove existing CUDA to avoid conflicts
apt-get --purge remove cuda* nvidia-cuda-toolkit -y 2>/dev/null || true

# Download and install CUDA 11.8
wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent --override
rm cuda_11.8.0_520.61.05_linux.run

# Set CUDA environment
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Verify CUDA installation
echo "CUDA version:"
nvcc --version

# 3. Create Python 3.10 virtual environment
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

# 7. Install NumPy FIRST with no dependencies to prevent upgrades
echo "Installing NumPy 1.24.3 (CRITICAL - blocks NumPy 2.x)..."
pip install numpy==1.24.3 --no-deps --force-reinstall

# 8. Install exact versions from original requirements
echo "Installing EXACT versions from original requirements..."
pip install timm==0.4.12 --no-deps
pip install objprint==0.2.3 --no-deps
pip install accelerate==0.18.0
pip install monai==1.1.0
pip install mmengine==0.7.4
pip install transformers==4.30.2
pip install tensorboard easydict pyyaml yacs
pip install SimpleITK nibabel opencv-python openpyxl scikit-learn pandas matplotlib Pillow gdown

# 9. Force NumPy back to 1.24.3 if anything upgraded it
echo "Re-forcing NumPy 1.24.3..."
pip install numpy==1.24.3 --force-reinstall --no-deps

# 10. Try compiling CUSTOM Mamba from source (REQUIRED for MM-UNet)
echo "Compiling CUSTOM Mamba from source (this will work with CUDA 11.8)..."
cd requirements/Mamba/causal-conv1d
pip install . --no-build-isolation 2>&1 | tee /tmp/causal_install.log
CAUSAL_STATUS=$?
cd ../../..

cd requirements/Mamba/mamba
pip install . --no-build-isolation 2>&1 | tee /tmp/mamba_install.log
MAMBA_STATUS=$?
cd ../../..

# 11. Verify custom Mamba installation (MUST succeed)
echo "Verifying custom Mamba installation..."
python -c "from mamba_ssm import Mamba; print('Custom Mamba successfully installed!')" 2>/dev/null
if [ $? -ne 0 ]; then
  echo "ERROR: Custom Mamba compilation failed!"
  echo "Check logs: /tmp/causal_install.log and /tmp/mamba_install.log"
  exit 1
fi

# 12. FINAL NumPy check - force 1.24.3 one last time
echo "Final NumPy version lock..."
pip install numpy==1.24.3 --force-reinstall --no-deps

# 13. Download dataset
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
