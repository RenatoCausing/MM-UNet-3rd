"""
FIVEs Dataset Loader for MM-UNet
- Supports 1024x1024 retinal vessel images
- 95:5 train/test split with random selection
- Independent normalization for train and test sets
"""

import os
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import numpy as np
from PIL import Image
import random
from easydict import EasyDict
from typing import Callable, Tuple, List, Dict, Optional
import cv2


class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    LIGHT_RED = "\033[1;31m"
    END = "\033[0m"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_normalization_stats(image_paths: List[str]) -> Tuple[List[float], List[float]]:
    """
    Compute mean and std for a list of images independently.
    Returns mean and std as lists [R, G, B] in range [0, 1].
    """
    print(f"{Colors.BLUE}Computing normalization statistics for {len(image_paths)} images...{Colors.END}")
    
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    
    for idx, img_path in enumerate(image_paths):
        if idx % 500 == 0:
            print(f"  Processing image {idx + 1}/{len(image_paths)}")
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"{Colors.YELLOW}Warning: Could not read image {img_path}{Colors.END}")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float64) / 255.0
        
        pixel_sum += img.sum(axis=(0, 1))
        pixel_sq_sum += (img ** 2).sum(axis=(0, 1))
        pixel_count += img.shape[0] * img.shape[1]
    
    if pixel_count == 0:
        print(f"{Colors.RED}Error: No valid images found for normalization!{Colors.END}")
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)
    
    # Ensure std is not zero to avoid division by zero
    std = np.maximum(std, 1e-6)
    
    mean_list = mean.tolist()
    std_list = std.tolist()
    
    print(f"{Colors.GREEN}Computed mean: {mean_list}{Colors.END}")
    print(f"{Colors.GREEN}Computed std: {std_list}{Colors.END}")
    
    return mean_list, std_list


def center_padding(img, target_size: List[int], pad_digit: int = 0):
    """Center pad image to target size."""
    is_pil = isinstance(img, Image.Image)
    if is_pil:
        original_mode = img.mode
        img_tensor = TF.to_tensor(img)
    else:
        img_tensor = img

    if img_tensor.ndim == 2:
        img_tensor = img_tensor.unsqueeze(0)
    
    in_h, in_w = img_tensor.shape[-2], img_tensor.shape[-1]
    target_h, target_w = target_size[0], target_size[1]

    if in_h >= target_h and in_w >= target_w:
        return img

    pad_left = max(0, (target_w - in_w) // 2)
    pad_right = max(0, target_w - in_w - pad_left)
    pad_top = max(0, (target_h - in_h) // 2)
    pad_bot = max(0, target_h - in_h - pad_top)

    import torch.nn.functional as F
    tensor_padded = F.pad(img_tensor, [pad_left, pad_right, pad_top, pad_bot], 'constant', pad_digit)

    if is_pil:
        pil_image_padded = TF.to_pil_image(
            tensor_padded.squeeze(0) if tensor_padded.ndim == 4 and tensor_padded.shape[0] == 1 else tensor_padded,
            mode=original_mode if original_mode in ['L', 'RGB'] else None)
        return pil_image_padded
    else:
        return tensor_padded


class FIVEsDataset(Dataset):
    """
    FIVEs Dataset for retinal vessel segmentation.
    Structure expected:
    - fives_preprocessed/
        - Original/     (contains 0001.png, 0002.png, ..., XXXX.png)
        - Segmented/    (contains 0001_segment.png, 0002_segment.png, ...)
    """

    def __init__(
        self,
        samples: List[Dict[str, str]],
        mode: str,
        image_size: int = 1024,
        image_mean: List[float] = None,
        image_std: List[float] = None,
        loader: Callable = Image.open,
    ):
        super().__init__()
        self.samples = samples
        self.mode = mode
        self.image_size = [image_size, image_size] if isinstance(image_size, int) else image_size
        self.image_mean = image_mean if image_mean is not None else [0.485, 0.456, 0.406]
        self.image_std = image_std if image_std is not None else [0.229, 0.224, 0.225]
        self.loader = loader

        self.img_paths_x = [s["image"] for s in self.samples]
        self.img_paths_y = [s["label"] for s in self.samples]

        print(f'{Colors.LIGHT_RED}Loading FIVEs dataset: Mode={self.mode}, Samples={len(self.samples)}{Colors.END}')
        print(f'{Colors.BLUE}Using normalization - Mean: {self.image_mean}, Std: {self.image_std}{Colors.END}')
        
        # Load all images into memory for faster training
        self.images_pil_x = []
        self.images_pil_y = []
        
        for idx in range(len(self.img_paths_x)):
            if idx % 500 == 0:
                print(f"  Loading image {idx + 1}/{len(self.img_paths_x)}")
            try:
                img_x = self.loader(self.img_paths_x[idx]).convert('RGB')
                img_y = self.loader(self.img_paths_y[idx]).convert('L')
                self.images_pil_x.append(img_x)
                self.images_pil_y.append(img_y)
            except Exception as e:
                print(f"{Colors.RED}Error loading: {self.img_paths_x[idx]} - {e}{Colors.END}")
                raise

        if not self.images_pil_x:
            raise ValueError(f"No images loaded for mode {self.mode}")
        
        print(f"{Colors.GREEN}Successfully loaded {len(self.images_pil_x)} images for {self.mode}{Colors.END}")

    def __len__(self) -> int:
        return len(self.images_pil_x)

    def _transform(self, image: Image.Image, target: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image_p = image.copy()
        target_p = target.copy()
        target_h, target_w = self.image_size[0], self.image_size[1]

        # For test mode, just resize without augmentation
        if self.mode == 'test':
            img_w_orig, img_h_orig = image_p.size
            if img_h_orig < target_h or img_w_orig < target_w:
                image_p = center_padding(image_p, [target_h, target_w], pad_digit=0)
                target_p = center_padding(target_p, [target_h, target_w], pad_digit=0)

        # Training augmentations
        if self.mode == 'train':
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                image_p = TF.hflip(image_p)
                target_p = TF.hflip(target_p)
            
            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                image_p = TF.vflip(image_p)
                target_p = TF.vflip(target_p)
            
            # Random rotation
            if torch.rand(1).item() > 0.5:
                angle = random.uniform(-30, 30)
                image_p = TF.rotate(image_p, angle, interpolation=InterpolationMode.BILINEAR)
                target_p = TF.rotate(target_p, angle, interpolation=InterpolationMode.NEAREST)

        # Image transforms with normalization
        img_transform_list = [
            transforms.Resize([target_h, target_w], antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.image_mean, std=self.image_std),
        ]
        
        # Additional augmentation for training
        if self.mode == 'train':
            # Color jitter (before ToTensor)
            if torch.rand(1).item() > 0.3:
                jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                image_p = jitter(image_p)

        image_tensor = transforms.Compose(img_transform_list)(image_p)

        # Label processing - same as original VesselLoader
        if target_p.mode != 'L':
            target_p = target_p.convert('L')
        lbl_tensor_raw = TF.to_tensor(target_p)
        lbl_tensor_binary = (lbl_tensor_raw > 0.5).float()
        target_tensor = TF.resize(lbl_tensor_binary, [target_h, target_w],
                                  interpolation=InterpolationMode.NEAREST, antialias=False)
        
        return image_tensor, target_tensor

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_x_pil = self.images_pil_x[index]
        img_y_pil = self.images_pil_y[index]
        img_x_tensor, img_y_tensor = self._transform(img_x_pil, img_y_pil)
        return img_x_tensor, img_y_tensor

def generate_fives_dataset_list(
    data_root: str,
    original_subdir: str = "Original",
    segmented_subdir: str = "Segmented",
) -> List[Dict[str, str]]:
    """
    Generate dataset list for FIVEs dataset.
    Expects:
    - data_root/Original/0001.png, 0002.png, ...
    - data_root/Segmented/0001_segment.png, 0002_segment.png, ...
    """
    dataset_list = []
    original_folder = os.path.join(data_root, original_subdir)
    segmented_folder = os.path.join(data_root, segmented_subdir)

    if not os.path.isdir(original_folder):
        raise ValueError(f"Original folder not found: {original_folder}")
    if not os.path.isdir(segmented_folder):
        raise ValueError(f"Segmented folder not found: {segmented_folder}")

    image_files = sorted([f for f in os.listdir(original_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    
    print(f"{Colors.BLUE}Found {len(image_files)} images in {original_folder}{Colors.END}")

    for img_filename in image_files:
        image_base_name, ext = os.path.splitext(img_filename)
        # Expected segmented filename: XXXX_segment.png
        expected_label_filename = f"{image_base_name}_segment{ext}"
        
        full_image_path = os.path.join(original_folder, img_filename)
        full_label_path = os.path.join(segmented_folder, expected_label_filename)

        if os.path.exists(full_label_path):
            dataset_list.append({"image": full_image_path, "label": full_label_path})
        else:
            # Try without _segment suffix (in case naming is different)
            alt_label_path = os.path.join(segmented_folder, img_filename)
            if os.path.exists(alt_label_path):
                dataset_list.append({"image": full_image_path, "label": alt_label_path})
            else:
                print(f"{Colors.YELLOW}Warning: No label found for {img_filename}{Colors.END}")

    print(f"{Colors.GREEN}Generated {len(dataset_list)} valid image-label pairs{Colors.END}")
    return dataset_list


def get_fives_dataloader(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = 1024,
    test_ratio: float = 0.05,
    random_seed: int = 42,
    compute_norm_stats: bool = True,
) -> Tuple[TorchDataLoader, TorchDataLoader, Dict]:
    """
    Get train and test dataloaders for FIVEs dataset.
    
    Args:
        data_root: Path to fives_preprocessed folder
        batch_size: Batch size for training and testing
        num_workers: Number of data loading workers
        image_size: Target image size (1024 for FIVEs)
        test_ratio: Ratio of test samples (0.05 = 5%)
        random_seed: Random seed for reproducibility
        compute_norm_stats: Whether to compute normalization stats from data
    
    Returns:
        train_loader, test_loader, stats_dict
    """
    print(f"{Colors.GREEN}=" * 60 + f"{Colors.END}")
    print(f"{Colors.GREEN}Loading FIVEs Dataset from: {data_root}{Colors.END}")
    print(f"{Colors.GREEN}=" * 60 + f"{Colors.END}")
    
    # Generate full dataset list
    all_samples = generate_fives_dataset_list(data_root)
    
    if len(all_samples) == 0:
        raise ValueError(f"No valid samples found in {data_root}")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Shuffle and split into train/test (95:5)
    shuffled_samples = all_samples.copy()
    random.shuffle(shuffled_samples)
    
    n_test = max(1, int(len(shuffled_samples) * test_ratio))
    n_train = len(shuffled_samples) - n_test
    
    test_samples = shuffled_samples[:n_test]
    train_samples = shuffled_samples[n_test:]
    
    print(f"{Colors.GREEN}Dataset split: {n_train} training, {n_test} testing (ratio: {1-test_ratio:.0%}:{test_ratio:.0%}){Colors.END}")
    
    # Compute normalization statistics independently
    train_image_paths = [s["image"] for s in train_samples]
    test_image_paths = [s["image"] for s in test_samples]
    
    if compute_norm_stats:
        print(f"\n{Colors.BLUE}Computing TRAINING set normalization...{Colors.END}")
        train_mean, train_std = compute_normalization_stats(train_image_paths)
        
        print(f"\n{Colors.BLUE}Computing TEST set normalization...{Colors.END}")
        test_mean, test_std = compute_normalization_stats(test_image_paths)
    else:
        # Use ImageNet defaults
        train_mean = test_mean = [0.485, 0.456, 0.406]
        train_std = test_std = [0.229, 0.224, 0.225]
        print(f"{Colors.YELLOW}Using default ImageNet normalization stats{Colors.END}")
    
    # Create datasets with their respective normalization
    train_dataset = FIVEsDataset(
        samples=train_samples,
        mode='train',
        image_size=image_size,
        image_mean=train_mean,
        image_std=train_std,
    )
    
    test_dataset = FIVEsDataset(
        samples=test_samples,
        mode='test',
        image_size=image_size,
        image_mean=test_mean,
        image_std=test_std,
    )
    
    # Create dataloaders
    g_train = torch.Generator()
    g_train.manual_seed(random_seed)
    
    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g_train,
        pin_memory=True,
        drop_last=True,
    )
    
    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    stats_dict = {
        'n_train': n_train,
        'n_test': n_test,
        'train_mean': train_mean,
        'train_std': train_std,
        'test_mean': test_mean,
        'test_std': test_std,
        'image_size': image_size,
        'test_samples': test_samples,
        'train_samples': train_samples,
    }
    
    print(f"\n{Colors.GREEN}Dataloaders created successfully!{Colors.END}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, test_loader, stats_dict


if __name__ == "__main__":
    # Test the loader
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./fives_preprocessed')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    train_loader, test_loader, stats = get_fives_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
    )
    
    # Test loading a batch
    for images, masks in train_loader:
        print(f"Train batch - Images: {images.shape}, Masks: {masks.shape}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Mask unique values: {torch.unique(masks)}")
        break
