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

# 7. Install exact PyTorch with CUDA 11.8 - FORCE and LOCK
echo "Installing PyTorch 2.0.0 with CUDA 11.8 (LOCKED)..."
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118 --force-reinstall --no-cache-dir

# 8. Verify PyTorch version immediately
echo "Verifying PyTorch installation..."
python -c "import torch; v = torch.__version__; c = torch.version.cuda; assert v.startswith('2.0.0'), f'Wrong PyTorch {v}'; assert c == '11.8', f'Wrong CUDA {c}'; print(f'✓ PyTorch {v} CUDA {c}')"

# 9. Install NumPy FIRST with no dependencies to prevent upgrades
echo "Installing NumPy 1.24.3 (CRITICAL - blocks NumPy 2.x)..."
pip install numpy==1.24.3 --no-deps --force-reinstall

# 10. Install exact versions from original requirements (all with --no-deps to prevent PyTorch upgrade)
echo "Installing EXACT versions from original requirements..."
pip install timm==0.4.12 --no-deps
pip install objprint==0.2.3 --no-deps
pip install accelerate==0.18.0 --no-deps
pip install monai==1.1.0 --no-deps
pip install mmengine==0.7.4 --no-deps
pip install transformers==4.30.2 --no-deps
pip install tensorboard easydict pyyaml yacs
pip install SimpleITK nibabel opencv-python openpyxl scikit-learn pandas matplotlib Pillow gdown

# 11. FINAL verification - PyTorch must still be 2.0.0
echo "Final PyTorch verification..."
python -c "import torch; v = torch.__version__; assert v.startswith('2.0.0'), f'ERROR: PyTorch upgraded to {v}'; print(f'✓ PyTorch locked at {v}')"

# 11b. Install missing dependencies (without upgrading PyTorch)
echo "Installing missing dependencies..."
pip install psutil addict rich termcolor yapf --no-deps
pip install huggingface-hub regex safetensors tokenizers tqdm --no-deps

# Verify PyTorch still locked
python -c "import torch; v = torch.__version__; assert v.startswith('2.0.0'), f'ERROR: PyTorch upgraded to {v}'"

# 12. Try compiling CUSTOM Mamba from source (REQUIRED for MM-UNet)
echo "Compiling CUSTOM Mamba from source (this will work with CUDA 11.8)..."

# CRITICAL: Uninstall any existing mamba-ssm first
echo "Removing any existing mamba-ssm installation..."
pip uninstall -y mamba-ssm causal-conv1d 2>/dev/null || true

# Clear pip cache to prevent using old wheels
echo "Clearing pip cache for mamba packages..."
pip cache remove mamba-ssm 2>/dev/null || true
pip cache remove causal-conv1d 2>/dev/null || true

cd requirements/Mamba/causal-conv1d
rm -rf build/ dist/ *.egg-info 2>/dev/null || true
pip install . --no-build-isolation --force-reinstall --no-cache-dir 2>&1 | tee /tmp/causal_install.log
CAUSAL_STATUS=$?
cd ../../..

cd requirements/Mamba/mamba
rm -rf build/ dist/ *.egg-info 2>/dev/null || true
pip install . --no-build-isolation --force-reinstall --no-cache-dir 2>&1 | tee /tmp/mamba_install.log
MAMBA_STATUS=$?
cd ../../..

# 13. Verify custom Mamba has bimamba_type and nslices parameters (CRITICAL!)
echo "Verifying custom Mamba parameters..."
python -c "
from mamba_ssm import Mamba
import inspect
sig = str(inspect.signature(Mamba.__init__))
if 'bimamba_type' not in sig or 'nslices' not in sig:
    print('ERROR: Custom Mamba missing required parameters!')
    print('Expected: bimamba_type and nslices')
    print('Got signature:', sig)
    print('This means the standard mamba-ssm was installed instead of the custom fork.')
    exit(1)
print('✓ Custom Mamba with bimamba_type and nslices verified!')
print('Signature includes:', [p for p in sig.split(',') if 'bimamba' in p or 'nslices' in p])
" 2>/dev/null
if [ $? -ne 0 ]; then
  echo "ERROR: Custom Mamba verification failed!"
  echo "Check logs: /tmp/causal_install.log and /tmp/mamba_install.log"
  exit 1
fi

# 14. FINAL NumPy check - force 1.24.3 one last time
echo "Final NumPy version lock..."
pip install numpy==1.24.3 --force-reinstall --no-deps

# 15. Download dataset
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
