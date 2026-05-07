"""
Temporal GAT Training Script.
Bismillah.

Usage:
  python training/train_gat.py                          # Train main model
  python training/train_gat.py --model gat_m            # GAT-M ablation
  python training/train_gat.py --model gcn              # GCN ablation
"""
import argparse
import json
import os
import sys
import time
import csv
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, Batch
from scipy.stats import spearmanr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.gat.temporal_gat import get_model


# ============================================================
# CONFIG
# ============================================================
DATASET_DIR = Path('F:/footagent/data/processed/gat_dataset')
CHECKPOINT_DIR = Path('F:/footagent/checkpoints/gat')
LOG_DIR = Path('F:/footagent/logs/gat')


def load_split(split_name):
    """Load preprocessed graph dicts from disk, convert to PyG Data objects."""
    path = DATASET_DIR / f'{split_name}.pt'
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw = torch.load(path, weights_only=False)
    data_list = []
    for g in raw:
        data = Data(
            x=g['x'],
            edge_index=g['edge_index'],
            y=g['y'],
        )
        data_list.append(data)

    return data_list


def create_batches(data_list, batch_size, shuffle=True):
    """Create mini-batches from Data list using PyG Batch."""
    if shuffle:
        indices = torch.randperm(len(data_list)).tolist()
    else:
        indices = list(range(len(data_list)))

    batches = []
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_data = [data_list[j] for j in batch_indices]
        batch = Batch.from_data_list(batch_data)
        batches.append(batch)

    return batches


def train_epoch(model, data_list, optimizer, criterion, device, batch_size):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    batches = create_batches(data_list, batch_size, shuffle=True)

    for batch in batches:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(pred, batch.y.squeeze())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, data_list, criterion, device, batch_size):
    """Evaluate on a dataset split."""
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds = []
    all_labels = []

    batches = create_batches(data_list, batch_size, shuffle=False)

    for batch in batches:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(pred, batch.y.squeeze())

        total_loss += loss.item()
        n_batches += 1

        all_preds.extend(pred.cpu().numpy().tolist())
        all_labels.extend(batch.y.squeeze().cpu().numpy().tolist())

    avg_loss = total_loss / max(n_batches, 1)

    # Metrics
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    mae = np.mean(np.abs(preds - labels))
    mse = np.mean((preds - labels) ** 2)
    spearman_r, spearman_p = spearmanr(preds, labels)

    metrics = {
        'loss': avg_loss,
        'mae': mae,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'pred_mean': float(np.mean(preds)),
        'pred_std': float(np.std(preds)),
        'label_mean': float(np.mean(labels)),
    }

    return metrics


def train(args):
    """Main training loop."""
    print("=" * 60)
    print(f"TRAINING: {args.model.upper()}")
    print(f"Device: {args.device}")
    print("=" * 60)

    device = torch.device(args.device)

    # Load data
    print("\nLoading data...")
    train_data = load_split('train')
    val_data = load_split('val')
    test_data = load_split('test')
    print(f"  Train: {len(train_data)} graphs")
    print(f"  Val:   {len(val_data)} graphs")
    print(f"  Test:  {len(test_data)} graphs")

    # Create model
    model = get_model(args.model, dropout=args.dropout)
    model = model.to(device)
    print(f"\nModel: {args.model}")
    print(f"  Parameters: {model.count_parameters():,}")

    # Loss, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Logging setup
    run_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = CHECKPOINT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = LOG_DIR / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config['n_params'] = model.count_parameters()
    config['n_train'] = len(train_data)
    config['n_val'] = len(val_data)
    config['n_test'] = len(test_data)
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # CSV logger
    csv_path = log_dir / 'training_log.csv'
    csv_fields = ['epoch', 'train_loss', 'val_loss', 'val_mae', 'val_rmse',
                  'val_spearman_r', 'lr', 'time_sec']
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"{'Epoch':>5} {'Train Loss':>11} {'Val Loss':>10} {'Val MAE':>9} "
          f"{'Val RMSE':>10} {'Spearman':>10} {'LR':>10}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_epoch(model, train_data, optimizer, criterion,
                                 device, args.batch_size)

        # Validate
        val_metrics = evaluate(model, val_data, criterion, device, args.batch_size)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - t0

        # Log
        print(f"{epoch:5d} {train_loss:11.6f} {val_metrics['loss']:10.6f} "
              f"{val_metrics['mae']:9.5f} {val_metrics['rmse']:10.5f} "
              f"{val_metrics['spearman_r']:10.4f} {current_lr:10.2e}")

        csv_writer.writerow({
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'val_loss': round(val_metrics['loss'], 6),
            'val_mae': round(val_metrics['mae'], 5),
            'val_rmse': round(val_metrics['rmse'], 5),
            'val_spearman_r': round(val_metrics['spearman_r'], 4),
            'lr': current_lr,
            'time_sec': round(elapsed, 1),
        })
        csv_file.flush()

        # Checkpoint best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'config': config,
            }, run_dir / 'best_model.pt')
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
            break

    csv_file.close()

    # Final evaluation on test set with best model
    print(f"\n{'=' * 60}")
    print(f"FINAL EVALUATION (best model from epoch {best_epoch})")
    print(f"{'=' * 60}")

    checkpoint = torch.load(run_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_data, criterion, device, args.batch_size)

    print(f"  Test Loss:    {test_metrics['loss']:.6f}")
    print(f"  Test MAE:     {test_metrics['mae']:.5f}")
    print(f"  Test RMSE:    {test_metrics['rmse']:.5f}")
    print(f"  Spearman r:   {test_metrics['spearman_r']:.4f}")
    print(f"  Spearman p:   {test_metrics['spearman_p']:.2e}")

    # Save final results
    results = {
        'model': args.model,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'config': config,
    }
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {run_dir}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Train Temporal GAT')
    parser.add_argument('--model', type=str, default='temporal_gat',
                        choices=['temporal_gat', 'gat_m', 'gcn'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
