#!/usr/bin/env python3
"""
Preprocess APTOS 2019 dataset for testing
Creates Original folder with 1024x1024 images
"""

import os
import cv2
import numpy as np
from pathlib import Path
import sys

def preprocess_aptos(raw_dir, output_dir, num_images=21):
    """
    Preprocess APTOS dataset
    
    Args:
        raw_dir: Path to APTOS dataset directory
        output_dir: Path to output directory
        num_images: Number of images to preprocess
    """
    
    # Create output directory
    original_dir = Path(output_dir) / "Original"
    original_dir.mkdir(parents=True, exist_ok=True)
    
    raw_path = Path(raw_dir)
    
    # APTOS structure: contains train_images, test_images, val_images folders
    image_sources = [
        raw_path / "train_images",
        raw_path / "test_images",
        raw_path / "val_images",
    ]
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    image_files = []
    
    print(f"Searching for images in:")
    for source in image_sources:
        print(f"  - {source}")
        if source.exists():
            # Try multiple glob patterns
            for ext in image_extensions:
                image_files.extend(source.glob(f'*{ext}'))
                image_files.extend(source.glob(f'*{ext.upper()}'))
            
            # Also try recursive search
            for ext in image_extensions:
                image_files.extend(source.rglob(f'*{ext}'))
                image_files.extend(source.rglob(f'*{ext.upper()}'))
        else:
            print(f"    (not found)")
    
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"\n❌ No images found in {raw_dir}")
        print(f"\nDirectory contents:")
        for source in image_sources:
            if source.exists():
                print(f"\n  {source}:")
                try:
                    for item in source.iterdir():
                        print(f"    {item.name}")
                except:
                    pass
        return False
    
    print(f"\n✓ Found {len(image_files)} total images")
    print(f"Processing first {num_images} images...\n")
    
    processed = 0
    
    for img_path in image_files:
        if processed >= num_images:
            break
        
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠ Could not read: {img_path.name}")
                continue
            
            # Resize to 1024x1024
            img_resized = cv2.resize(img, (1024, 1024))
            
            # Save as original
            filename = f"image_{processed+1:04d}.png"
            original_path = original_dir / filename
            cv2.imwrite(str(original_path), img_resized)
            
            print(f"[{processed+1:2d}/{num_images}] ✓ {img_path.name} -> {filename}")
            processed += 1
            
        except Exception as e:
            print(f"⚠ Error processing {img_path.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Preprocessing Complete")
    print(f"{'='*60}")
    print(f"Processed: {processed} images")
    print(f"Output: {original_dir}")
    print(f"{'='*60}\n")
    
    return processed > 0

if __name__ == "__main__":
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "aptos2019_raw"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "aptos_preprocessed"
    num_images = int(sys.argv[3]) if len(sys.argv) > 3 else 21
    
    success = preprocess_aptos(raw_dir, output_dir, num_images)
    sys.exit(0 if success else 1)
