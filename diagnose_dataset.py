#!/usr/bin/env python3
"""
Diagnose raw dataset images
Check if images are valid and displayable
"""

import cv2
import numpy as np
from pathlib import Path
import sys

def diagnose_images(raw_dir, num_samples=5):
    """Check if images are valid"""
    
    raw_path = Path(raw_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    
    # Find images
    image_files = []
    for ext in image_extensions:
        image_files.extend(raw_path.rglob(f'*{ext}'))
        image_files.extend(raw_path.rglob(f'*{ext.upper()}'))
    
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"❌ No images found in {raw_dir}")
        return False
    
    print(f"Found {len(image_files)} images\n")
    print("Checking first", min(num_samples, len(image_files)), "images...\n")
    
    valid_count = 0
    
    for i, img_path in enumerate(image_files[:num_samples]):
        print(f"[{i+1}] {img_path.name}")
        print(f"    Size: {img_path.stat().st_size / 1024:.1f} KB")
        
        # Try to read
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"    ❌ FAILED: Could not read image")
            continue
        
        print(f"    ✓ Shape: {img.shape}")
        print(f"    ✓ Min pixel: {img.min()}, Max pixel: {img.max()}, Mean: {img.mean():.1f}")
        
        # Check if it looks like actual image data
        if img.max() > img.min():
            print(f"    ✓ Contains valid image data")
            valid_count += 1
        else:
            print(f"    ⚠ WARNING: Image appears to be all same color (possible corruption)")
        
        print()
    
    print(f"{'='*60}")
    print(f"Summary: {valid_count}/{min(num_samples, len(image_files))} images are valid")
    print(f"{'='*60}\n")
    
    if valid_count == 0:
        print("❌ All images failed to load!")
        print("The dataset may be corrupted or in an unsupported format.")
        return False
    
    return True

if __name__ == "__main__":
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "fives_raw"
    diagnose_images(raw_dir, num_samples=5)
