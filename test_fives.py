"""
Test script for FIVEs dataset with MM-UNet
Reports: Accuracy, Precision, Recall, F1, AUC, and Dice scores
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import argparse
import json
import random
from typing import Dict

import monai
import torch
import yaml
import numpy as np
from accelerate import Accelerator
from easydict import EasyDict
from sklearn.metrics import roc_auc_score
import warnings
from huggingface_hub import hf_hub_download

from src.models import give_model
from src.FIVEsLoader import FIVEsDataset, generate_fives_dataset_list

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Test MM-UNet on FIVEs dataset')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to local model checkpoint (.pth file)')
    parser.add_argument('--hf_repo', type=str, default='',
                        help='Hugging Face repo id to pull checkpoint from (e.g. user/repo)')
    parser.add_argument('--hf_filename', type=str, default='best_model.pth',
                        help='Checkpoint filename in Hugging Face repo')
    parser.add_argument('--hf_token', type=str, default='',
                        help='Hugging Face token (or set HF_TOKEN env var)')
    parser.add_argument('--hf_cache_dir', type=str, default='./hf_cache',
                        help='Cache directory for Hugging Face downloads')
    parser.add_argument('--data_root', type=str, default='./fives_preprocessed',
                        help='Path to FIVEs dataset')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for testing')
    parser.add_argument('--image_size', type=int, default=1024,
                        help='Image size (default 1024 for FIVEs)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--test_ratio', type=float, default=0.05,
                        help='Test set ratio (default 0.05 = 5%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (must match training seed for same split)')
    parser.add_argument('--split_file', type=str, default='./test_results/fives_split_seed42_ratio005.json',
                        help='Split manifest file. If exists, reuse exact split; otherwise create it.')
    parser.add_argument('--norm_mean', type=float, nargs=3, default=[0.485, 0.456, 0.406],
                        help='Normalization mean as 3 floats')
    parser.add_argument('--norm_std', type=float, nargs=3, default=[0.229, 0.224, 0.225],
                        help='Normalization std as 3 floats')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='Directory to save test results')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction images')
    return parser.parse_args()


def resolve_checkpoint(args, accelerator):
    """Resolve checkpoint path from local file or Hugging Face."""
    if args.checkpoint and os.path.exists(args.checkpoint):
        return args.checkpoint

    if not args.hf_repo:
        raise ValueError(
            "Checkpoint not found locally and --hf_repo is empty. "
            "Provide --checkpoint or --hf_repo."
        )

    token = args.hf_token or os.environ.get('HF_TOKEN', '')
    accelerator.print(f"Downloading checkpoint from Hugging Face: {args.hf_repo}/{args.hf_filename}")
    ckpt_path = hf_hub_download(
        repo_id=args.hf_repo,
        filename=args.hf_filename,
        token=token if token else None,
        cache_dir=args.hf_cache_dir,
    )
    accelerator.print(f"Checkpoint downloaded to: {ckpt_path}")
    return ckpt_path


def build_fives_test_loader_with_fixed_split(
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    test_ratio: float,
    seed: int,
    norm_mean,
    norm_std,
    split_file: str,
):
    """
    Build deterministic 95/5 split with seed. Reuse split_file if present.
    This keeps preprocessing on-the-fly without writing precomputed tensors.
    """
    # Resolve data_root to absolute path
    data_root = os.path.abspath(data_root)
    
    # Verify dataset folders exist
    original_folder = os.path.join(data_root, 'Original')
    segmented_folder = os.path.join(data_root, 'Segmented')
    
    if not os.path.isdir(original_folder):
        raise ValueError(
            f"Original folder not found: {original_folder}\n"
            f"Expected structure:\n"
            f"  {data_root}/\n"
            f"  ├── Original/\n"
            f"  └── Segmented/\n"
            f"Run: bash setup_fives_dataset.sh"
        )
    
    if not os.path.isdir(segmented_folder):
        raise ValueError(f"Segmented folder not found: {segmented_folder}")
    
    all_samples = generate_fives_dataset_list(data_root)
    if len(all_samples) == 0:
        raise ValueError(f"No valid samples found under {data_root}")

    os.makedirs(os.path.dirname(split_file) or '.', exist_ok=True)

    if os.path.exists(split_file):
        with open(split_file, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        test_samples = split_data.get('test_samples', [])
        train_samples = split_data.get('train_samples', [])
    else:
        rng = random.Random(seed)
        shuffled = list(all_samples)
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_ratio))
        test_samples = shuffled[:n_test]
        train_samples = shuffled[n_test:]

        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'seed': seed,
                    'test_ratio': test_ratio,
                    'n_total': len(shuffled),
                    'n_train': len(train_samples),
                    'n_test': len(test_samples),
                    'train_samples': train_samples,
                    'test_samples': test_samples,
                },
                f,
                indent=2,
            )

    test_dataset = FIVEsDataset(
        samples=test_samples,
        mode='test',
        image_size=image_size,
        image_mean=list(norm_mean),
        image_std=list(norm_std),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    data_stats = {
        'n_train': len(train_samples),
        'n_test': len(test_samples),
        'train_mean': list(norm_mean),
        'train_std': list(norm_std),
        'test_mean': list(norm_mean),
        'test_std': list(norm_std),
        'image_size': image_size,
        'split_file': split_file,
    }

    return test_loader, data_stats


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute AUC score from flattened arrays."""
    try:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Check if we have both classes
        if len(np.unique(y_true_flat)) < 2:
            print("Warning: Only one class present in y_true, AUC is undefined")
            return 0.0
        
        return roc_auc_score(y_true_flat, y_pred_flat)
    except Exception as e:
        print(f"Error computing AUC: {e}")
        return 0.0


@torch.no_grad()
def test_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    inference: monai.inferers.Inferer,
    accelerator: Accelerator,
    save_predictions: bool = False,
    output_dir: str = './test_results',
):
    """
    Test the model and compute comprehensive metrics.
    Returns: dict with Accuracy, Precision, Recall, F1, AUC, Dice
    """
    model.eval()
    
    # Accumulators for metrics
    all_preds_prob = []  # For AUC (probabilities)
    all_preds_binary = []  # For other metrics (binary)
    all_targets = []
    
    # MONAI metrics
    dice_metric = monai.metrics.DiceMetric(
        include_background=True,
        reduction=monai.utils.MetricReduction.MEAN_BATCH,
        get_not_nans=True
    )
    confusion_metrics = {
        'accuracy': monai.metrics.ConfusionMatrixMetric(
            include_background=True, metric_name="accuracy"
        ),
        'precision': monai.metrics.ConfusionMatrixMetric(
            include_background=True, metric_name="precision"
        ),
        'recall': monai.metrics.ConfusionMatrixMetric(
            include_background=True, metric_name="recall"
        ),
        'f1': monai.metrics.ConfusionMatrixMetric(
            include_background=True, metric_name="f1 score"
        ),
    }
    
    # Post-processing
    sigmoid_transform = monai.transforms.Activations(sigmoid=True)
    threshold_transform = monai.transforms.AsDiscrete(threshold=0.5)
    
    if save_predictions:
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    
    sample_idx = 0
    accelerator.print("Starting evaluation...")
    
    for batch_idx, (images, masks) in enumerate(test_loader):
        accelerator.print(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
        
        # Move images to device (GPU)
        images = images.to(accelerator.device)
        masks = masks.to(accelerator.device)
        
        # Forward pass with sliding window inference
        logits = inference(images, model)
        
        # Get probabilities (for AUC)
        probs = sigmoid_transform(logits)
        
        # Get binary predictions (for other metrics)
        preds_binary = threshold_transform(probs)
        
        # Update MONAI metrics
        dice_metric(y_pred=preds_binary, y=masks)
        for metric in confusion_metrics.values():
            metric(y_pred=preds_binary, y=masks)
        
        # Store for AUC computation
        all_preds_prob.append(probs.cpu().numpy())
        all_preds_binary.append(preds_binary.cpu().numpy())
        all_targets.append(masks.cpu().numpy())
        
        # Save predictions if requested
        if save_predictions:
            batch_size = images.shape[0]
            for i in range(batch_size):
                pred_img = (preds_binary[i, 0].cpu().numpy() * 255).astype(np.uint8)
                mask_img = (masks[i, 0].cpu().numpy() * 255).astype(np.uint8)
                
                import cv2
                cv2.imwrite(
                    os.path.join(output_dir, 'predictions', f'pred_{sample_idx:04d}.png'),
                    pred_img
                )
                cv2.imwrite(
                    os.path.join(output_dir, 'predictions', f'mask_{sample_idx:04d}.png'),
                    mask_img
                )
                sample_idx += 1
    
    # Aggregate MONAI metrics
    dice_score = float(dice_metric.aggregate()[0].mean())
    dice_metric.reset()
    
    results = {'dice': dice_score}
    
    for name, metric in confusion_metrics.items():
        score = float(metric.aggregate()[0].mean())
        metric.reset()
        results[name] = score
    
    # Compute AUC
    all_preds_prob = np.concatenate(all_preds_prob, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    auc_score = compute_auc(all_targets, all_preds_prob)
    results['auc'] = auc_score
    
    return results


def main():
    args = parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Load config
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    
    # Initialize accelerator
    accelerator = Accelerator(cpu=False)
    
    accelerator.print("=" * 60)
    accelerator.print("FIVEs Dataset Testing")
    accelerator.print("=" * 60)
    accelerator.print(f"Checkpoint (local): {args.checkpoint if args.checkpoint else 'N/A'}")
    accelerator.print(f"Checkpoint (HF): {args.hf_repo}/{args.hf_filename}" if args.hf_repo else "Checkpoint (HF): N/A")
    accelerator.print(f"Data Root: {args.data_root}")
    accelerator.print(f"Image Size: {args.image_size}")
    accelerator.print(f"Batch Size: {args.batch_size}")
    accelerator.print(f"Split Ratio: {1-args.test_ratio:.2f}/{args.test_ratio:.2f}, Seed: {args.seed}")
    accelerator.print(f"Norm Mean: {args.norm_mean}")
    accelerator.print(f"Norm Std: {args.norm_std}")
    accelerator.print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    accelerator.print('Loading Model...')
    model = give_model(config)
    
    # Load checkpoint
    checkpoint_path = resolve_checkpoint(args, accelerator)
    accelerator.print(f'Loading checkpoint from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        accelerator.print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown') + 1}")
    else:
        # Try loading as direct state dict
        model.load_state_dict(checkpoint)
    
    # Load test data
    accelerator.print('Loading Test Dataset...')
    test_loader, data_stats = build_fives_test_loader_with_fixed_split(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_ratio=args.test_ratio,
        seed=args.seed,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
        split_file=args.split_file,
    )
    
    # Setup inference - use direct forward pass (images are already 1024x1024)
    # No sliding window needed for this resolution
    class DirectInference:
        def __call__(self, images, model):
            return model(images)
    
    inference = DirectInference()
    
    # Prepare model for evaluation
    model = accelerator.prepare(model)
    
    # Run evaluation
    accelerator.print("\n" + "=" * 60)
    accelerator.print("Running Evaluation...")
    accelerator.print("=" * 60 + "\n")
    
    results = test_model(
        model=model,
        test_loader=test_loader,
        inference=inference,
        accelerator=accelerator,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir,
    )
    
    # Print results
    accelerator.print("\n" + "=" * 60)
    accelerator.print("TEST RESULTS")
    accelerator.print("=" * 60)
    accelerator.print(f"{'Metric':<15} {'Score':<15}")
    accelerator.print("-" * 30)
    accelerator.print(f"{'Accuracy':<15} {results['accuracy']:.5f}")
    accelerator.print(f"{'Precision':<15} {results['precision']:.5f}")
    accelerator.print(f"{'Recall':<15} {results['recall']:.5f}")
    accelerator.print(f"{'F1 Score':<15} {results['f1']:.5f}")
    accelerator.print(f"{'AUC':<15} {results['auc']:.5f}")
    accelerator.print(f"{'Dice':<15} {results['dice']:.5f}")
    accelerator.print("=" * 60)
    
    # Save results to JSON
    results_path = os.path.join(args.output_dir, 'test_results.json')
    results['checkpoint'] = checkpoint_path
    results['data_root'] = args.data_root
    results['n_test_samples'] = data_stats['n_test']
    results['image_size'] = args.image_size
    results['split_file'] = data_stats['split_file']
    results['seed'] = args.seed
    results['test_ratio'] = args.test_ratio
    results['norm_mean'] = args.norm_mean
    results['norm_std'] = args.norm_std
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    accelerator.print(f"\nResults saved to: {results_path}")
    
    if args.save_predictions:
        accelerator.print(f"Predictions saved to: {os.path.join(args.output_dir, 'predictions')}")


if __name__ == '__main__':
    main()
