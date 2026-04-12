#!/usr/bin/env python3
"""
Inference test script for MM-UNet model
Tests 1 image with multiple epochs (10, 20, 30, 40) and 20 images with epoch 40
"""

import os
import sys
import json
import random
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Suppress selective_scan_cuda import warnings
os.environ['MAMBA_SKIP_CUDA_BUILD'] = '1'

try:
    from huggingface_hub import hf_hub_download, login
except ImportError:
    print("Warning: huggingface_hub not found, will try to install...")
    os.system("pip install huggingface-hub -q")
    from huggingface_hub import hf_hub_download, login

try:
    from torch.utils.data import DataLoader, Dataset
    import yaml
    from easydict import EasyDict
except ImportError as e:
    print(f"Warning: Missing package {e}, installing...")
    os.system("pip install pyyaml easydict -q")
    from torch.utils.data import DataLoader, Dataset
    import yaml
    from easydict import EasyDict

# Set device
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except Exception as e:
    print(f"Warning: Error checking CUDA availability: {e}")
    device = torch.device('cpu')

print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")

# HuggingFace credentials
HF_TOKEN = "0b489b4559640c39a305a24313c85587"
HF_REPO = "23LebronJames23/MM-UNet"

# Normalization stats from user
NORM_CONFIG = {
    "n_train": 2926,
    "n_test": 154,
    "train_mean": [0.485, 0.456, 0.406],
    "train_std": [0.229, 0.224, 0.225],
    "test_mean": [0.485, 0.456, 0.406],
    "test_std": [0.229, 0.224, 0.225],
    "image_size": 1024
}

# Output directories
OUTPUT_BASE = Path("./inference_results")
OUTPUT_BASE.mkdir(exist_ok=True)

SINGLE_IMAGE_DIR = OUTPUT_BASE / "single_image_epochs_10_20_30_40"
MULTI_IMAGE_DIR = OUTPUT_BASE / "20_images_epoch_40"

SINGLE_IMAGE_DIR.mkdir(exist_ok=True, parents=True)
MULTI_IMAGE_DIR.mkdir(exist_ok=True, parents=True)

print(f"Normalization config loaded:")
print(f"  Mean: {NORM_CONFIG['train_mean']}")
print(f"  Std: {NORM_CONFIG['train_std']}")
print(f"  Image size: {NORM_CONFIG['image_size']}")


class SimpleImageDataset(Dataset):
    """Simple dataset for loading images from a directory"""
    def __init__(self, image_paths):
        self.image_paths = image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        return self.image_paths[idx]


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
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title(f"Original Image")
    axes[0].axis('off')
    
    # Segmented image
    segmented_img_display = segmented_img.squeeze()
    axes[1].imshow(segmented_img_display, cmap='gray')
    if epoch is not None:
        axes[1].set_title(f"Segmentation (Epoch {epoch})")
    else:
        axes[1].set_title(f"Segmentation")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comparison: {output_path}")


def load_model_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint"""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return True
    else:
        print(f"Warning: Checkpoint not found: {checkpoint_path}")
        return False


def download_checkpoints(epochs=[10, 20, 30, 40]):
    """Download model checkpoints from HuggingFace"""
    login(token=HF_TOKEN)
    
    checkpoints = {}
    
    for epoch in epochs:
        checkpoint_name = f"checkpoint_epoch_{epoch:04d}.pth"
        try:
            print(f"Downloading {checkpoint_name} from {HF_REPO}...")
            path = hf_hub_download(
                repo_id=HF_REPO,
                filename=checkpoint_name,
                token=HF_TOKEN,
                cache_dir="./hf_cache"
            )
            checkpoints[epoch] = path
            print(f"  Downloaded to: {path}")
        except Exception as e:
            print(f"  Error downloading {checkpoint_name}: {e}")
    
    return checkpoints


def run_inference_single_image():
    """Test single image with multiple epochs"""
    print("\n" + "="*80)
    print("TESTING SINGLE IMAGE WITH EPOCHS 10, 20, 30, 40")
    print("="*80)
    
    # Load config
    with open('config.yml', 'r') as f:
        config_dict = yaml.safe_load(f)
    config = EasyDict(config_dict)
    
    # Import model
    from src.models import give_model
    model = give_model(config).to(device)
    
    # Download checkpoints
    checkpoints = download_checkpoints([10, 20, 30, 40])
    
    if not checkpoints:
        print("Error: No checkpoints downloaded!")
        return
    
    # For demo purposes, use synthetic image or first available test image
    # You may need to modify this to load actual test images from your dataset
    test_image_path = None
    
    # Try to find test images in common locations
    search_paths = [
        './fives_preprocessed/test',
        './data/test',
        './test_images',
        './datasets/test'
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            image_files = [f for f in os.listdir(search_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            if image_files:
                test_image_path = os.path.join(search_path, image_files[0])
                break
    
    if test_image_path is None:
        # Create a dummy test image for demonstration
        print("No test images found. Creating a synthetic test image for demonstration...")
        dummy_image = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        test_image_path = "dummy_test_image.png"
        cv2.imwrite(test_image_path, dummy_image)
    
    print(f"Testing with image: {test_image_path}")
    
    # Read and preprocess the image
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
        
        model = give_model(config).to(device)
        
        if not load_model_checkpoint(model, checkpoint_path, device):
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
        output_dir.mkdir(exist_ok=True)
        
        save_comparison(
            original_image,
            output_binary,
            output_dir / "comparison.png",
            epoch=epoch
        )
        
        # Save just the segmentation mask
        cv2.imwrite(str(output_dir / "segmentation_mask.png"), (output_binary * 255).astype(np.uint8))
        print(f"  Results saved to: {output_dir}")


def run_inference_20_images():
    """Test 20 random images with epoch 40"""
    print("\n" + "="*80)
    print("TESTING 20 RANDOM IMAGES WITH EPOCH 40")
    print("="*80)
    
    # Load config
    with open('config.yml', 'r') as f:
        config_dict = yaml.safe_load(f)
    config = EasyDict(config_dict)
    
    # Import model
    from src.models import give_model
    model = give_model(config).to(device)
    
    # Download checkpoint for epoch 40
    checkpoints = download_checkpoints([40])
    
    if 40 not in checkpoints:
        print("Error: Could not download epoch 40 checkpoint!")
        return
    
    checkpoint_path = checkpoints[40]
    
    # Find test images
    search_paths = [
        './fives_preprocessed/test',
        './data/test',
        './test_images',
        './datasets/test'
    ]
    
    image_files = []
    for search_path in search_paths:
        if os.path.exists(search_path):
            found_files = [
                os.path.join(search_path, f) 
                for f in os.listdir(search_path) 
                if f.endswith(('.jpg', '.png', '.jpeg'))
            ]
            image_files.extend(found_files)
            break
    
    if not image_files:
        print("No test images found. Creating synthetic images for demonstration...")
        os.makedirs("demo_images", exist_ok=True)
        for i in range(21):
            dummy_image = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
            path = f"demo_images/image_{i:03d}.png"
            cv2.imwrite(path, dummy_image)
            image_files.append(path)
    
    # Select 20 random images (or use first 20)
    if len(image_files) > 20:
        selected_images = random.sample(image_files, 20)
    else:
        selected_images = image_files[:20] if len(image_files) >= 20 else image_files
    
    print(f"Selected {len(selected_images)} images for testing")
    
    # Load model checkpoint
    if not load_model_checkpoint(model, checkpoint_path, device):
        print("Error loading checkpoint!")
        return
    
    model.eval()
    
    # Process each image
    for idx, image_path in enumerate(selected_images):
        print(f"\nProcessing image {idx+1}/{len(selected_images)}: {os.path.basename(image_path)}")
        
        # Read and preprocess the image
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
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
        output_dir.mkdir(exist_ok=True)
        
        # Save comparison
        save_comparison(
            original_image,
            output_binary,
            output_dir / "comparison.png",
            epoch=40
        )
        
        # Save just the segmentation mask
        cv2.imwrite(str(output_dir / "segmentation_mask.png"), (output_binary * 255).astype(np.uint8))
        
        # Save the original image as well
        cv2.imwrite(str(output_dir / "original_image.png"), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
        
        print(f"  Results saved to: {output_dir}")


def save_config_info():
    """Save configuration info to JSON"""
    config_info = {
        "normalization": NORM_CONFIG,
        "model": "MM_Net",
        "epochs_tested": {
            "single_image": [10, 20, 30, 40],
            "multi_images": 40
        },
        "num_images_single": 1,
        "num_images_multi": 20,
        "output_directories": {
            "single_image": str(SINGLE_IMAGE_DIR),
            "multi_images": str(MULTI_IMAGE_DIR)
        }
    }
    
    config_path = OUTPUT_BASE / "config_info.json"
    with open(config_path, 'w') as f:
        json.dump(config_info, f, indent=2)
    
    print(f"\nConfiguration saved to: {config_path}")


if __name__ == "__main__":
    print("MM-UNet Inference Test Script")
    print(f"Device: {device}")
    print(f"Output base directory: {OUTPUT_BASE}")
    
    try:
        # Test single image with multiple epochs
        run_inference_single_image()
        
        # Test 20 images with epoch 40
        run_inference_20_images()
        
        # Save configuration info
        save_config_info()
        
        print("\n" + "="*80)
        print("INFERENCE TESTING COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nResults saved to:")
        print(f"  Single image (epochs 10, 20, 30, 40): {SINGLE_IMAGE_DIR}")
        print(f"  20 images (epoch 40): {MULTI_IMAGE_DIR}")
        
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
