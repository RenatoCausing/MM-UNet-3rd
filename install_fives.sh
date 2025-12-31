#!/bin/bash
# ==========================================
# MM-UNet Installation Script for FIVEs Dataset
# Designed for Python 3.10.x in cloud environments
# FOR VAST.AI PYTORCH TEMPLATE - Forces reinstall to avoid conflicts
# ==========================================

echo "=========================================="
echo "MM-UNet Installation for FIVEs Dataset"
echo "Python 3.10.x Required"
echo "FORCING REINSTALL for Vast.ai compatibility"
echo "=========================================="

# Step 1: Create and activate virtual environment (optional but recommended)
echo ""
echo "Step 1: Create virtual environment (optional)"
echo "Run these commands if you want a virtual environment:"
echo "  python3.10 -m venv mmunet_env"
echo "  source mmunet_env/bin/activate"
echo ""

# Step 2: Upgrade pip
echo "Step 2: Upgrading pip..."
pip install --upgrade pip

# Step 3: Install PyTorch with CUDA support (FORCE REINSTALL)
# Adjust CUDA version as needed (cu118 for CUDA 11.8, cu121 for CUDA 12.1)
echo ""
echo "Step 3: Installing PyTorch 2.0.0 with CUDA 11.8 (FORCE)..."
pip install --force-reinstall torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# Step 4: Install core dependencies (FORCE REINSTALL)
echo ""
echo "Step 4: Installing core dependencies (FORCE)..."
pip install --force-reinstall --no-deps timm==0.4.12
pip install --force-reinstall --no-deps objprint==0.2.3
pip install --force-reinstall --no-deps accelerate==0.18.0
pip install --force-reinstall tensorboard easydict SimpleITK monai nibabel pyyaml opencv-python openpyxl scikit-learn pandas matplotlib Pillow

# Step 5: Install Mamba dependencies (causal-conv1d and mamba-ssm)
echo ""
echo "Step 5: Installing Mamba components..."
echo "NOTE: This step may require compilation. Ensure you have CUDA toolkit installed."

# Install causal-conv1d
cd requirements/Mamba/causal-conv1d
pip install . --force-reinstall --no-build-isolation
cd ../../..

# Install mamba-ssm
cd requirements/Mamba/mamba
pip install . --force-reinstall --no-build-isolation
cd ../../..

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "To train on FIVEs dataset, run:"
echo "  python train_fives.py --data_root ./fives_preprocessed --lr 0.001 --batch_size 4 --epochs 100"
echo ""
echo "To test:"
echo "  python test_fives.py --checkpoint ./checkpoints_fives/best_model.pth --data_root ./fives_preprocessed"
echo ""
