"""
Test script for FIVEs dataset with MM-UNet
Reports: Accuracy, Precision, Recall, F1, AUC, and Dice scores
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import argparse
import json
from typing import Dict

import monai
import torch
import yaml
import numpy as np
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from sklearn.metrics import roc_auc_score
import warnings

from src.models import give_model
from src.FIVEsLoader import get_fives_dataloader, FIVEsDataset, generate_fives_dataset_list

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Test MM-UNet on FIVEs dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--data_root', type=str, default='./fives_preprocessed',
                        help='Path to FIVEs dataset')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for testing')
    parser.add_argument('--image_size', type=int, default=1024,
                        help='Image size (default 1024 for FIVEs)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--stats_file', type=str, default=None,
                        help='Path to data_stats.json for normalization params')
    parser.add_argument('--test_ratio', type=float, default=0.05,
                        help='Test set ratio (default 0.05 = 5%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (must match training seed for same split)')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='Directory to save test results')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction images')
    return parser.parse_args()


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
    accelerator.print(f"Checkpoint: {args.checkpoint}")
    accelerator.print(f"Data Root: {args.data_root}")
    accelerator.print(f"Image Size: {args.image_size}")
    accelerator.print(f"Batch Size: {args.batch_size}")
    accelerator.print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    accelerator.print('Loading Model...')
    model = give_model(config)
    
    # Load checkpoint
    accelerator.print(f'Loading checkpoint from {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        accelerator.print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown') + 1}")
    else:
        # Try loading as direct state dict
        model.load_state_dict(checkpoint)
    
    # Load normalization stats if available
    test_mean = None
    test_std = None
    
    if args.stats_file and os.path.exists(args.stats_file):
        with open(args.stats_file, 'r') as f:
            stats = json.load(f)
        test_mean = stats.get('test_mean')
        test_std = stats.get('test_std')
        accelerator.print(f"Loaded normalization stats from {args.stats_file}")
        accelerator.print(f"  Test Mean: {test_mean}")
        accelerator.print(f"  Test Std: {test_std}")
    
    # Load test data
    accelerator.print('Loading Test Dataset...')
    
    # Use the same split as training (with same seed)
    train_loader, test_loader, data_stats = get_fives_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        compute_norm_stats=True,  # Compute stats for test set
    )
    
    # Setup inference
    inference = monai.inferers.SlidingWindowInferer(
        roi_size=ensure_tuple_rep(args.image_size, 2),
        overlap=0.5,
        sw_device=accelerator.device,
        device=accelerator.device
    )
    
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
    results['checkpoint'] = args.checkpoint
    results['data_root'] = args.data_root
    results['n_test_samples'] = data_stats['n_test']
    results['image_size'] = args.image_size
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    accelerator.print(f"\nResults saved to: {results_path}")
    
    if args.save_predictions:
        accelerator.print(f"Predictions saved to: {os.path.join(args.output_dir, 'predictions')}")


if __name__ == '__main__':
    main()
