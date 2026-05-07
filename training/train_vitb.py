"""
ViT-B Foul Classifier Training Script.
Bismillah.

Usage:
  python training/train_vitb.py                     # Full fine-tune
  python training/train_vitb.py --freeze_backbone   # Frozen backbone
"""
import argparse
import json
import sys
import time
import csv
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.vit_foul_classifier import ViTFoulClassifier


# ============================================================
# CONFIG
# ============================================================
DATASET_DIR = Path('F:/footagent/data/processed/mvfouls_dataset')
CHECKPOINT_DIR = Path('F:/footagent/checkpoints/vitb')
LOG_DIR = Path('F:/footagent/logs/vitb')


class FoulDataset(Dataset):
    """Load MVFouls samples on the fly from mp4 files."""

    def __init__(self, split_name):
        # Load clip index and metadata
        from preprocessing.mvfouls_preprocess import run_full_pipeline, TARGET_FRAMES, TARGET_SIZE, IMAGENET_MEAN, IMAGENET_STD
        import cv2

        self.target_frames = TARGET_FRAMES
        self.target_size = TARGET_SIZE
        self.imagenet_mean = IMAGENET_MEAN
        self.imagenet_std = IMAGENET_STD
        
        # We run the pipeline quickly to get the index
        print(f"Loading {split_name} clip index...")
        clip_index, weights, class_to_idx, _, _ = run_full_pipeline('F:/footagent/data', validate=False)
        self.split_data = clip_index[clip_index['split'] == split_name].reset_index(drop=True)
        self.weights = weights
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.split_data)
        
    def _extract_frames(self, clip_path):
        import cv2
        import numpy as np
        
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.target_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, self.target_frames, dtype=int).tolist()

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame.shape[:2]
                min_dim = min(h, w)
                top = (h - min_dim) // 2
                left = (w - min_dim) // 2
                frame = frame[top:top+min_dim, left:left+min_dim]
                frame = cv2.resize(frame, self.target_size)
                frame = frame.astype(np.float32) / 255.0
                for c in range(3):
                    frame[:, :, c] = (frame[:, :, c] - self.imagenet_mean[c]) / self.imagenet_std[c]
                frame = np.transpose(frame, (2, 0, 1))
                frames.append(frame)

        cap.release()

        if len(frames) == 0:
            return None

        while len(frames) < self.target_frames:
            frames.append(frames[-1])

        return torch.tensor(np.stack(frames[:self.target_frames]), dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.split_data.iloc[idx]
        clip_paths = row['clip_paths']
        label = row['label']
        cls_name = row['action_class']
        weight = self.weights.get(cls_name, 1.0)
        
        frames_tensor = None
        for cp in clip_paths:
            if Path(cp).exists():
                frames_tensor = self._extract_frames(cp)
                if frames_tensor is not None:
                    break
                    
        if frames_tensor is None:
            # Fallback to zero tensor if video is missing or broken
            frames_tensor = torch.zeros((self.target_frames, 3, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            
        return frames_tensor, torch.tensor(label, dtype=torch.long), torch.tensor(weight, dtype=torch.float32)


def train_epoch(model, loader, optimizer, criterion_action, criterion_severity,
                device, action_weight=1.0, severity_weight=0.3):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for frames, labels, weights in loader:
        frames = frames.to(device)
        labels = labels.to(device)
        weights = weights.to(device)

        optimizer.zero_grad()

        action_logits, severity_logits = model(frames)

        # Weighted cross-entropy for action classification
        loss_action = criterion_action(action_logits, labels)
        # Weight by sample weight (class imbalance correction)
        loss_action = (loss_action * weights).mean()

        # Total loss (severity not used for now — no severity labels in dataset)
        loss = action_weight * loss_action

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = action_logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion_action, device):
    """Evaluate on a dataset split."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for frames, labels, weights in loader:
        frames = frames.to(device)
        labels = labels.to(device)

        action_logits, _ = model(frames)
        loss = criterion_action(action_logits, labels).mean()

        total_loss += loss.item() * labels.size(0)
        preds = action_logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    n = max(len(all_labels), 1)
    avg_loss = total_loss / n
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return {
        'loss': avg_loss,
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'preds': all_preds,
        'labels': all_labels,
    }


def train(args):
    """Main training loop."""
    print("=" * 60)
    print("TRAINING: ViT-B FOUL CLASSIFIER")
    print(f"Device: {args.device}")
    print(f"Backbone frozen: {args.freeze_backbone}")
    print("=" * 60)

    device = torch.device(args.device)

    # Load metadata
    meta_path = DATASET_DIR / 'meta.json'
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    class_to_idx = meta['class_to_idx']
    idx_to_class = {int(k): v for k, v in meta['idx_to_class'].items()}
    n_classes = len(class_to_idx)
    print(f"\n{n_classes} classes: {list(class_to_idx.keys())}")

    # Load data
    print("\nLoading data...")
    train_ds = FoulDataset('train')
    val_ds = FoulDataset('valid')
    test_ds = FoulDataset('test')
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    # Create model
    model = ViTFoulClassifier(n_classes=n_classes,
                               freeze_backbone=args.freeze_backbone,
                               dropout=args.dropout)
    model = model.to(device)
    total_p, trainable_p = model.count_parameters()
    print(f"\nModel: ViT-B/16")
    print(f"  Total params: {total_p:,}")
    print(f"  Trainable params: {trainable_p:,}")

    # Loss and optimizer
    criterion_action = nn.CrossEntropyLoss(reduction='none')
    criterion_severity = nn.CrossEntropyLoss()

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Logging
    run_name = f"vitb_{'frozen' if args.freeze_backbone else 'full'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = CHECKPOINT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = LOG_DIR / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config['n_classes'] = n_classes
    config['total_params'] = total_p
    config['trainable_params'] = trainable_p
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    csv_path = log_dir / 'training_log.csv'
    csv_fields = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
                  'val_f1_macro', 'lr', 'time_sec']
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # Training loop
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"{'Epoch':>5} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>9} "
          f"{'Val Acc':>8} {'Val F1':>8} {'LR':>10}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion_action,
            criterion_severity, device
        )

        val_metrics = evaluate(model, val_loader, criterion_action, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"{epoch:5d} {train_loss:11.4f} {train_acc:10.4f} "
              f"{val_metrics['loss']:9.4f} {val_metrics['accuracy']:8.4f} "
              f"{val_metrics['f1_macro']:8.4f} {current_lr:10.2e}")

        csv_writer.writerow({
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 4),
            'val_loss': round(val_metrics['loss'], 4),
            'val_acc': round(val_metrics['accuracy'], 4),
            'val_f1_macro': round(val_metrics['f1_macro'], 4),
            'lr': current_lr,
            'time_sec': round(elapsed, 1),
        })
        csv_file.flush()

        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'config': config,
            }, run_dir / 'best_model.pt')
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    csv_file.close()

    # Final test evaluation
    print(f"\n{'=' * 60}")
    print(f"FINAL EVALUATION (best model from epoch {best_epoch})")
    print(f"{'=' * 60}")

    checkpoint = torch.load(run_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, criterion_action, device)

    print(f"\n  Test Accuracy:   {test_metrics['accuracy']:.4f}")
    print(f"  Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"  Test F1 (wtd):   {test_metrics['f1_weighted']:.4f}")

    # Per-class report
    report = classification_report(
        test_metrics['labels'], test_metrics['preds'],
        target_names=[idx_to_class.get(i, f'Class {i}') for i in range(n_classes)],
        zero_division=0,
    )
    print(f"\nClassification Report:\n{report}")

    with open(run_dir / 'test_report.txt', 'w') as f:
        f.write(report)

    results = {
        'model': 'vitb',
        'frozen': args.freeze_backbone,
        'best_epoch': best_epoch,
        'test_accuracy': test_metrics['accuracy'],
        'test_f1_macro': test_metrics['f1_macro'],
        'test_f1_weighted': test_metrics['f1_weighted'],
    }
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {run_dir}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Train ViT-B Foul Classifier')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
