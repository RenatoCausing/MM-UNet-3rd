#!/usr/bin/env python3
"""
Inference test script for MM-UNet on APTOS 2019 dataset
Tests on 21 retinal fundus images with epochs 10, 20, 30, 40
"""

import os
import sys
import json
import random
import warnings
warnings.filterwarnings('ignore')

# Apply causal_conv1d patch BEFORE any other imports
try:
    from causal_conv1d_patch import patch_causal_conv1d
    patch_causal_conv1d()
except ImportError:
    import types
    if 'causal_conv1d' not in sys.modules:
        causal_conv1d = types.ModuleType('causal_conv1d')
        sys.modules['causal_conv1d'] = causal_conv1d

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    os.system("pip install huggingface-hub -q")
    from huggingface_hub import hf_hub_download

try:
    from torch.utils.data import DataLoader, Dataset
    import yaml
    from easydict import EasyDict
except ImportError as e:
    print(f"Warning: Missing package {e}, installing...")
    os.system(f"pip install {str(e).split()[-1]} -q")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Normalization config
NORM_CONFIG = {
    'train_mean': [0.485, 0.456, 0.406],
    'train_std': [0.229, 0.224, 0.225]
}

print("\nNormalization config loaded:")
for key, val in NORM_CONFIG.items():
    print(f"  {key}: {val}")

# Output paths
OUTPUT_BASE = Path("inference_results_aptos")
SINGLE_IMAGE_DIR = OUTPUT_BASE / "single_image_epochs_10_20_30_40"
MULTI_IMAGE_DIR = OUTPUT_BASE / "21_aptos_images_epoch_40"

# HuggingFace config
HF_REPO = "23LebronJames23/MM-UNet"
HF_CHECKPOINTS = {
    10: "checkpoint_epoch_0010.pth",
    20: "checkpoint_epoch_0020.pth",
    30: "checkpoint_epoch_0030.pth",
    40: "checkpoint_epoch_0040.pth",
}

def preprocess_image(image, norm_mean=None, norm_std=None, target_size=1024):
    """Preprocess a single image for inference"""
    if norm_mean is None:
        norm_mean = NORM_CONFIG['train_mean']
    if norm_std is None:
        norm_std = NORM_CONFIG['train_std']
    
    # Read image if path is given
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_COLOR)
        if image is None:
            return None, None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    original_h, original_w = image.shape[:2]
    
    # Resize to target size
    image_resized = cv2.resize(image, (target_size, target_size))
    
    # Normalize to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Apply normalization
    image_normalized = (image_normalized - np.array(norm_mean)) / np.array(norm_std)
    
    # Convert to torch tensor and add batch dimension
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).float()
    
    return image_tensor, (original_h, original_w)

def save_comparison(original_img, segmented_img, output_path, epoch=None):
    """Save original and segmented images side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Original image - ensure it's uint8 in [0, 255] range
    if isinstance(original_img, np.ndarray):
        original_display = original_img.copy().astype(np.uint8)
        if original_display.max() <= 1.0:
            original_display = (original_display * 255).astype(np.uint8)
    else:
        original_display = original_img
    
    axes[0].imshow(original_display)
    axes[0].set_title(f"Original APTOS Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Segmented image
    segmented_img_display = segmented_img.squeeze()
    if segmented_img_display.max() <= 1.0:
        axes[1].imshow(segmented_img_display, cmap='gray', vmin=0, vmax=1)
    else:
        axes[1].imshow(segmented_img_display, cmap='gray', vmin=0, vmax=255)
    
    if epoch is not None:
        axes[1].set_title(f"Segmentation (Epoch {epoch})", fontsize=12, fontweight='bold')
    else:
        axes[1].set_title(f"Segmentation", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comparison: {output_path}")

def download_checkpoints(epochs=[10, 20, 30, 40]):
    """Download checkpoints from local cache or HuggingFace"""
    checkpoints = {}
    
    for epoch in epochs:
        checkpoint_name = f"checkpoint_epoch_{epoch:04d}.pth"
        cache_path = f"./hf_cache/{checkpoint_name}"
        
        # Check if file already exists locally
        if os.path.exists(cache_path):
            print(f"Found cached checkpoint: {cache_path}")
            checkpoints[epoch] = cache_path
            continue
        
        # Try to download if not cached
        try:
            print(f"Downloading {checkpoint_name} from {HF_REPO}...")
            path = hf_hub_download(
                repo_id=HF_REPO,
                filename=checkpoint_name,
                cache_dir="./hf_cache"
            )
            checkpoints[epoch] = path
            print(f"  Downloaded to: {path}")
        except Exception as e:
            print(f"  ⚠ Error downloading {checkpoint_name}: {e}")
            print(f"  Skipping epoch {epoch}...")
    
    return checkpoints

def load_model_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint"""
    try:
        from safe_model_loading import safe_load_checkpoint
        return safe_load_checkpoint(model, checkpoint_path, device)
    except ImportError:
        print(f"Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            return True
        except Exception as e:
            print(f"Warning: Checkpoint loading failed: {e}")
            return False

def run_inference_single_image():
    """Test single APTOS image with multiple epochs"""
    print("\n" + "="*80)
    print("TESTING SINGLE APTOS IMAGE WITH EPOCHS 10, 20, 30, 40")
    print("="*80)
    
    # Load config
    try:
        with open('config.yml', 'r') as f:
            config_dict = yaml.safe_load(f)
        config = EasyDict(config_dict)
    except Exception as e:
        print(f"Error loading config.yml: {e}")
        config = EasyDict({'finetune': {'model_choose': 'MM_Net'}, 'models': {}})
    
    # Import model
    try:
        from safe_model_loading import load_model_safe
        model = load_model_safe(config, device)
        if model is None:
            print("❌ Error: Could not load model")
            return
    except Exception as e:
        print(f"❌ Error: Could not import model: {e}")
        return
    
    # Download checkpoints
    checkpoints = download_checkpoints([10, 20, 30, 40])
    
    if not checkpoints:
        print("Error: No checkpoints downloaded!")
        return
    
    # Find first test image
    test_dir = Path("aptos_preprocessed/Original")
    if not test_dir.exists():
        print(f"Error: {test_dir} not found!")
        return
    
    image_files = sorted(list(test_dir.glob("*.png")))
    if not image_files:
        print(f"Error: No images found in {test_dir}")
        return
    
    test_image_path = str(image_files[0])
    print(f"\nTesting with image: {test_image_path}")
    
    # Read and preprocess
    original_image = cv2.imread(test_image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        print(f"Error: Could not load image {test_image_path}")
        return
    
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_tensor, orig_size = preprocess_image(original_image)
    
    if image_tensor is None:
        print(f"Error preprocessing image")
        return
    
    image_tensor = image_tensor.to(device)
    
    # Run inference with each checkpoint
    for epoch, checkpoint_path in sorted(checkpoints.items()):
        print(f"\nRunning inference with epoch {epoch}...")
        
        try:
            from safe_model_loading import load_model_safe
            model = load_model_safe(config, device)
            if model is None:
                print(f"⚠ Could not load model for epoch {epoch}, skipping...")
                continue
        except Exception as e:
            print(f"⚠ Error loading model for epoch {epoch}: {e}")
            continue
        
        if not load_model_checkpoint(model, checkpoint_path, device):
            print(f"⚠ Could not load checkpoint for epoch {epoch}, skipping...")
            continue
        
        model.eval()
        
        with torch.no_grad():
            output = model(image_tensor).cpu().squeeze().numpy()
        
        # Threshold at 0.5
        output_binary = (output > 0.5).astype(np.float32)
        
        # Resize back to original size if needed
        if orig_size != (1024, 1024):
            output_binary = cv2.resize(output_binary, (orig_size[1], orig_size[0]))
        
        # Save comparison
        output_dir = SINGLE_IMAGE_DIR / f"epoch_{epoch:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_comparison(
            original_image,
            output_binary,
            output_dir / "comparison.png",
            epoch=epoch
        )
        
        # Save just the segmentation mask
        cv2.imwrite(str(output_dir / "segmentation_mask.png"), (output_binary * 255).astype(np.uint8))
        print(f"  Results saved to: {output_dir}")

def run_inference_21_images():
    """Test 21 APTOS images with epoch 40"""
    print("\n" + "="*80)
    print("TESTING 21 APTOS IMAGES WITH EPOCH 40")
    print("="*80)
    
    # Load config
    try:
        with open('config.yml', 'r') as f:
            config_dict = yaml.safe_load(f)
        config = EasyDict(config_dict)
    except Exception as e:
        print(f"Error loading config.yml: {e}")
        config = EasyDict({'finetune': {'model_choose': 'MM_Net'}, 'models': {}})
    
    # Import model
    try:
        from safe_model_loading import load_model_safe
        model = load_model_safe(config, device)
        if model is None:
            print("❌ Error: Could not load model")
            return
    except Exception as e:
        print(f"❌ Error importing model: {e}")
        return
    
    # Download checkpoint for epoch 40
    checkpoints = download_checkpoints([40])
    
    if 40 not in checkpoints:
        print("Error: Could not download epoch 40 checkpoint!")
        return
    
    checkpoint_path = checkpoints[40]
    
    # Find test images
    test_dir = Path("aptos_preprocessed/Original")
    if not test_dir.exists():
        print(f"Error: {test_dir} not found!")
        return
    
    image_files = sorted(list(test_dir.glob("*.png")))[:21]
    
    if not image_files:
        print(f"Error: No images found in {test_dir}")
        return
    
    print(f"Found {len(image_files)} images to process\n")
    
    if not load_model_checkpoint(model, checkpoint_path, device):
        print("Error: Could not load checkpoint!")
        return
    
    model.eval()
    
    # Process each image
    for idx, image_path in enumerate(image_files):
        print(f"\nProcessing image {idx+1}/{len(image_files)}: {image_path.name}")
        
        # Read and preprocess
        original_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if original_image is None:
            print(f"  Error: Could not load image")
            continue
        
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        image_tensor, orig_size = preprocess_image(original_image)
        
        if image_tensor is None:
            print(f"  Error preprocessing image")
            continue
        
        image_tensor = image_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(image_tensor).cpu().squeeze().numpy()
        
        # Threshold at 0.5
        output_binary = (output > 0.5).astype(np.float32)
        
        # Resize back to original size if needed
        if orig_size != (1024, 1024):
            output_binary = cv2.resize(output_binary, (orig_size[1], orig_size[0]))
        
        # Create output directory for this image
        img_name = Path(image_path).stem
        output_dir = MULTI_IMAGE_DIR / f"image_{idx+1:02d}_{img_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comparison
        save_comparison(
            original_image,
            output_binary,
            output_dir / "comparison.png",
            epoch=40
        )
        
        # Save segmentation mask
        cv2.imwrite(str(output_dir / "segmentation_mask.png"), (output_binary * 255).astype(np.uint8))
        
        # Save original image
        cv2.imwrite(str(output_dir / "original_image.png"), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
        
        print(f"  Results saved to: {output_dir}")

if __name__ == "__main__":
    print("MM-UNet Inference Test Script - APTOS 2019")
    print(f"Device: {device}")
    print(f"Output base directory: {OUTPUT_BASE}")
    
    try:
        # Test single image with multiple epochs
        run_inference_single_image()
        
        # Test 21 images with epoch 40
        run_inference_21_images()
        
        print("\n" + "="*80)
        print("✓ Testing Complete!")
        print(f"Results saved to: {OUTPUT_BASE}")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
