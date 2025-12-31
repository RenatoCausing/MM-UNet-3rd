"""
Training script for FIVEs dataset with MM-UNet
Supports CLI arguments for learning rate, batch size, and epochs
Saves weights at each epoch
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import argparse
from datetime import datetime
from typing import Dict

import monai
import torch
import yaml
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory

from src import utils
from src.models import give_model
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger
from src.FIVEsLoader import get_fives_dataloader
import warnings
import torch.nn as nn

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Train MM-UNet on FIVEs dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--data_root', type=str, default='./fives_preprocessed', 
                        help='Path to FIVEs dataset')
    parser.add_argument('--image_size', type=int, default=1024, 
                        help='Image size (default 1024 for FIVEs)')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of data loading workers')
    parser.add_argument('--test_ratio', type=float, default=0.05, 
                        help='Test set ratio (default 0.05 = 5%)')
    parser.add_argument('--warmup', type=int, default=5, 
                        help='Warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.05, 
                        help='Weight decay')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_fives', 
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--compute_norm', action='store_true', default=True,
                        help='Compute normalization stats from data')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID')
    return parser.parse_args()


def train_one_epoch(
    model: torch.nn.Module, 
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    post_trans: monai.transforms.Compose, 
    accelerator: Accelerator, 
    epoch: int, 
    step: int, 
    loss_weights: Dict[str, float],
    num_epochs: int,
):
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    for i, (images, masks) in enumerate(train_loader):
        # Check for NaN in inputs
        if torch.isnan(images).any() or torch.isnan(masks).any():
            accelerator.print(f"Warning: NaN detected in input batch {i}, skipping...")
            continue
            
        logits = model(images)
        
        # Check for NaN in outputs
        if torch.isnan(logits).any():
            accelerator.print(f"Warning: NaN detected in model output batch {i}, skipping...")
            continue
        
        total_loss = 0
        log = ''
        for name, loss_fn in loss_functions.items():
            current_loss = loss_fn(logits, masks)
            
            # Skip if loss is NaN
            if torch.isnan(current_loss):
                accelerator.print(f"Warning: NaN loss detected for {name} in batch {i}")
                continue
                
            weighted_loss = loss_weights[name] * current_loss
            accelerator.log({'Train/' + name: float(current_loss)}, step=step)
            total_loss += weighted_loss
            log += f'{name}: {current_loss:.4f} '
        
        # Skip if total loss is NaN
        if torch.isnan(total_loss):
            accelerator.print(f"Warning: Total loss is NaN in batch {i}, skipping...")
            continue
        
        # Compute metrics
        val_outputs = post_trans(logits)
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=masks)

        accelerator.backward(total_loss)
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += float(total_loss)
        num_batches += 1
        
        accelerator.log({'Train/Total Loss': float(total_loss)}, step=step)
        accelerator.print(
            f'Epoch [{epoch + 1}/{num_epochs}] Training [{i + 1}/{len(train_loader)}] '
            f'Loss: {total_loss:.5f} {log}',
            flush=True
        )
        step += 1
    
    scheduler.step(epoch)
    
    # Aggregate metrics
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metrics[metric_name].reset()
        metric[f'Train/mean {metric_name}'] = float(batch_acc.mean())
    
    avg_loss = epoch_loss / max(num_batches, 1)
    metric['Train/avg_loss'] = avg_loss
    
    accelerator.print(f'Epoch [{epoch + 1}/{num_epochs}] Training metrics: {metric}')
    accelerator.log(metric, step=epoch)
    
    return step, avg_loss


@torch.no_grad()
def test_one_epoch(
    model: torch.nn.Module, 
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    inference: monai.inferers.Inferer, 
    test_loader: torch.utils.data.DataLoader,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric], 
    step: int,
    post_trans: monai.transforms.Compose, 
    accelerator: Accelerator, 
    epoch: int,
    num_epochs: int,
):
    model.eval()
    epoch_loss = 0.0
    num_batches = 0
    
    for i, (images, masks) in enumerate(test_loader):
        logits = inference(images, model)
        
        total_loss = 0
        log = ''
        for name in loss_functions:
            loss = loss_functions[name](logits, masks)
            if not torch.isnan(loss):
                accelerator.log({'Test/' + name: float(loss)}, step=step)
                log += f' {name} {float(loss):.5f} '
                total_loss += loss
        
        val_outputs = post_trans(logits)
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=masks)
        
        epoch_loss += float(total_loss)
        num_batches += 1
        
        accelerator.log({'Test/Total Loss': float(total_loss)}, step=step)
        accelerator.print(
            f'Epoch [{epoch + 1}/{num_epochs}] Testing [{i + 1}/{len(test_loader)}] '
            f'Loss: {total_loss:.5f} {log}',
            flush=True
        )
        step += 1
    
    # Aggregate metrics
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metrics[metric_name].reset()
        metric[f'Test/mean {metric_name}'] = float(batch_acc.mean())
    
    avg_loss = epoch_loss / max(num_batches, 1)
    metric['Test/avg_loss'] = avg_loss
    
    accelerator.print(f'Epoch [{epoch + 1}/{num_epochs}] Test metrics: {metric}')
    accelerator.log(metric, step=epoch)
    
    # Return dice score as main metric for best model selection
    dice_score = metric.get('Test/mean dice_metric', 0.0)
    return torch.tensor([dice_score]).to(accelerator.device), metric, step


def main():
    args = parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Load config
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    
    # Override config with CLI args
    config.trainer.lr = args.lr
    config.trainer.num_epochs = args.epochs
    config.trainer.warmup = args.warmup
    config.trainer.weight_decay = args.weight_decay
    
    # Set seed
    utils.same_seeds(args.seed)
    
    # Setup logging directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging_dir = os.path.join(
        os.getcwd(), 'logs', 
        f'FIVEs_{config.finetune.checkpoint}_{timestamp}'
    )
    
    # Initialize accelerator
    accelerator = Accelerator(cpu=False)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers("FIVEs_training")
    
    accelerator.print("=" * 60)
    accelerator.print("FIVEs Dataset Training Configuration")
    accelerator.print("=" * 60)
    accelerator.print(f"Learning Rate: {args.lr}")
    accelerator.print(f"Batch Size: {args.batch_size}")
    accelerator.print(f"Epochs: {args.epochs}")
    accelerator.print(f"Image Size: {args.image_size}")
    accelerator.print(f"Data Root: {args.data_root}")
    accelerator.print(f"Test Ratio: {args.test_ratio}")
    accelerator.print(f"Checkpoint Dir: {args.checkpoint_dir}")
    accelerator.print("=" * 60)
    accelerator.print(objstr(config))
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load model
    accelerator.print('Loading Model...')
    model = give_model(config)
    
    # Load dataloader
    accelerator.print('Loading FIVEs Dataset...')
    train_loader, test_loader, data_stats = get_fives_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        compute_norm_stats=args.compute_norm,
    )
    
    # Save data stats for later use
    import json
    stats_path = os.path.join(args.checkpoint_dir, 'data_stats.json')
    stats_to_save = {
        'n_train': data_stats['n_train'],
        'n_test': data_stats['n_test'],
        'train_mean': data_stats['train_mean'],
        'train_std': data_stats['train_std'],
        'test_mean': data_stats['test_mean'],
        'test_std': data_stats['test_std'],
        'image_size': data_stats['image_size'],
    }
    with open(stats_path, 'w') as f:
        json.dump(stats_to_save, f, indent=2)
    accelerator.print(f"Data stats saved to {stats_path}")
    
    # Setup inference
    image_size = args.image_size
    inference = monai.inferers.SlidingWindowInferer(
        roi_size=ensure_tuple_rep(image_size, 2), 
        overlap=0.5,
        sw_device=accelerator.device, 
        device=accelerator.device
    )
    
    # Setup metrics
    include_background = True
    metrics = {
        'dice_metric': monai.metrics.DiceMetric(
            include_background=include_background,
            reduction=monai.utils.MetricReduction.MEAN_BATCH, 
            get_not_nans=True
        ),
        'miou_metric': monai.metrics.MeanIoU(
            include_background=include_background, 
            reduction="mean_channel"
        ),
        'f1': monai.metrics.ConfusionMatrixMetric(
            include_background=include_background, 
            metric_name='f1 score'
        ),
        'precision': monai.metrics.ConfusionMatrixMetric(
            include_background=include_background,
            metric_name="precision"
        ),
        'recall': monai.metrics.ConfusionMatrixMetric(
            include_background=include_background, 
            metric_name="recall"
        ),
        'ACC': monai.metrics.ConfusionMatrixMetric(
            include_background=include_background, 
            metric_name="accuracy"
        ),
    }
    
    # Post-processing transforms
    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True), 
        monai.transforms.AsDiscrete(threshold=0.5)
    ])
    
    # Optimizer
    optimizer = optim_factory.create_optimizer_v2(
        model, 
        config.trainer.optimizer,
        weight_decay=args.weight_decay,
        lr=args.lr, 
        betas=(0.9, 0.95)
    )
    
    # Scheduler
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, 
        warmup_epochs=args.warmup,
        max_epochs=args.epochs
    )
    
    # Loss functions - DiceFocalLoss is robust against class imbalance
    loss_functions = {
        'dice_focal_loss': monai.losses.DiceFocalLoss(
            smooth_nr=1e-5, 
            smooth_dr=1e-5, 
            to_onehot_y=False, 
            sigmoid=True
        ),
    }
    
    loss_weights = {
        'dice_focal_loss': 1.0
    }
    
    # Training state
    step = 0
    test_step = 0
    starting_epoch = 0
    best_dice = torch.tensor(0.0)
    best_epoch = -1
    
    # Prepare for distributed training
    model, optimizer, scheduler, train_loader, test_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, test_loader
    )
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        accelerator.print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=accelerator.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', torch.tensor(0.0))
        best_epoch = checkpoint.get('best_epoch', -1)
        accelerator.print(f"Resumed from epoch {starting_epoch}")
    
    best_dice = best_dice.to(accelerator.device)
    
    # Start training
    accelerator.print("\n" + "=" * 60)
    accelerator.print("Starting Training!")
    accelerator.print("=" * 60 + "\n")
    
    for epoch in range(starting_epoch, args.epochs):
        accelerator.print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        
        # Train
        step, train_loss = train_one_epoch(
            model, loss_functions, train_loader,
            optimizer, scheduler, metrics,
            post_trans, accelerator, epoch, step, loss_weights,
            args.epochs
        )
        
        # Test
        dice_score, test_metrics, test_step = test_one_epoch(
            model, loss_functions, inference, test_loader,
            metrics, test_step, post_trans, accelerator, epoch,
            args.epochs
        )
        
        # Save checkpoint for this epoch
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            epoch_checkpoint_path = os.path.join(
                args.checkpoint_dir, 
                f'checkpoint_epoch_{epoch + 1:04d}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'test_metrics': test_metrics,
                'best_dice': best_dice,
                'best_epoch': best_epoch,
            }, epoch_checkpoint_path)
            accelerator.print(f"Saved epoch checkpoint: {epoch_checkpoint_path}")
        
        # Save best model
        if dice_score > best_dice:
            best_dice = dice_score
            best_epoch = epoch
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_dice': best_dice,
                    'test_metrics': test_metrics,
                }, best_checkpoint_path)
                accelerator.print(f"New best model saved! Dice: {best_dice:.5f}")
        
        accelerator.print(
            f'Epoch [{epoch + 1}/{args.epochs}] '
            f'Best Dice: {best_dice:.5f} (Epoch {best_epoch + 1}), '
            f'Current Dice: {dice_score:.5f}'
        )
    
    # Final summary
    accelerator.print("\n" + "=" * 60)
    accelerator.print("Training Complete!")
    accelerator.print("=" * 60)
    accelerator.print(f"Best Dice Score: {best_dice:.5f}")
    accelerator.print(f"Best Epoch: {best_epoch + 1}")
    accelerator.print(f"Checkpoints saved in: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
