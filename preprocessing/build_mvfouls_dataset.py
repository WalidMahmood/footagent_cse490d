"""
Build ViT-B dataset from MVFouls video clips.
Bismillah.

Produces: data/processed/mvfouls_dataset/ with train/val/test .pt files
Each sample: {frames: [16, 3, 224, 224], label: int, weight: float}
"""
import json
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.mvfouls_preprocess import (
    load_annotations, compute_class_weights, build_clip_index,
    TARGET_FRAMES, TARGET_SIZE, IMAGENET_MEAN, IMAGENET_STD,
)

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = Path('F:/footagent/data')
OUTPUT_DIR = Path('F:/footagent/data/processed/mvfouls_dataset')


def extract_frames_from_clip(clip_path, n_frames=TARGET_FRAMES,
                              target_size=TARGET_SIZE):
    """
    Extract n_frames uniformly sampled from a video clip.
    Returns tensor [n_frames, 3, H, W] normalized with ImageNet stats.
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < n_frames:
        # If clip is too short, sample all frames and pad
        indices = list(range(total_frames))
    else:
        # Uniform sampling
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Center crop to square, then resize
            h, w = frame.shape[:2]
            min_dim = min(h, w)
            top = (h - min_dim) // 2
            left = (w - min_dim) // 2
            frame = frame[top:top+min_dim, left:left+min_dim]
            frame = cv2.resize(frame, target_size)
            # Normalize to [0, 1] then ImageNet normalize
            frame = frame.astype(np.float32) / 255.0
            for c in range(3):
                frame[:, :, c] = (frame[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
            # HWC -> CHW
            frame = np.transpose(frame, (2, 0, 1))
            frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    # Pad if needed
    while len(frames) < n_frames:
        frames.append(frames[-1])

    return torch.tensor(np.stack(frames[:n_frames]), dtype=torch.float32)


def build_and_save_dataset():
    """Main entry point: build MVFouls dataset and save splits to disk."""
    print("=" * 60)
    print("BUILDING MVFOULS DATASET FOR VIT-B CLASSIFIER")
    print("=" * 60)

    # Step 1: Load annotations
    print("\nStep 1: Loading annotations...")
    actions_df, split_stats = load_annotations(DATA_ROOT)
    print(f"  {len(actions_df)} actions: {split_stats}")

    # Step 2: Build clip index and class weights
    print("\nStep 2: Building clip index and class weights...")
    clip_index, class_to_idx = build_clip_index(actions_df)
    weights, class_counts = compute_class_weights(actions_df)
    print(f"  {len(class_to_idx)} classes: {class_to_idx}")
    print(f"  Weights: { {k: round(v, 2) for k, v in weights.items()} }")

    # Step 3: Process clips per split
    print("\nStep 3: Processing clips into frame tensors...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save class mapping
    meta = {
        'class_to_idx': class_to_idx,
        'idx_to_class': {v: k for k, v in class_to_idx.items()},
        'class_weights': {k: round(v, 4) for k, v in weights.items()},
        'n_frames': TARGET_FRAMES,
        'frame_size': list(TARGET_SIZE),
        'imagenet_mean': IMAGENET_MEAN,
        'imagenet_std': IMAGENET_STD,
    }

    manifest = {}
    for split in ['train', 'valid', 'test']:
        split_data = clip_index[clip_index['split'] == split]
        if len(split_data) == 0:
            print(f"  {split}: 0 actions, skipping")
            continue

        samples = []
        skipped = 0

        for i, (_, row) in enumerate(split_data.iterrows()):
            clip_paths = row['clip_paths']
            label = row['label']
            cls_name = row['action_class']
            weight = weights.get(cls_name, 1.0)

            # Use first available clip
            frames_tensor = None
            for cp in clip_paths:
                if Path(cp).exists():
                    frames_tensor = extract_frames_from_clip(cp)
                    if frames_tensor is not None:
                        break

            if frames_tensor is None:
                skipped += 1
                continue

            samples.append({
                'frames': frames_tensor,
                'label': torch.tensor(label, dtype=torch.long),
                'weight': torch.tensor(weight, dtype=torch.float32),
                'action_id': row['action_id'],
            })

            if (i + 1) % 100 == 0:
                print(f"    {split}: {i+1}/{len(split_data)}, "
                      f"{len(samples)} saved, {skipped} skipped")

        if samples:
            save_path = OUTPUT_DIR / f'{split}.pt'
            torch.save(samples, save_path)
            size_mb = save_path.stat().st_size / (1024 * 1024)
            manifest[split] = {
                'n_samples': len(samples),
                'skipped': skipped,
                'file': str(save_path),
                'size_mb': round(size_mb, 2),
            }
            print(f"  {split}: {len(samples)} samples, {skipped} skipped, "
                  f"{size_mb:.1f} MB")

    # Save metadata
    meta['manifest'] = manifest
    meta_path = OUTPUT_DIR / 'meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")

    total = sum(m['n_samples'] for m in manifest.values())
    print(f"\nDONE: {total} total samples across {len(manifest)} splits")
    print("Dataset ready for ViT-B training.")

    return manifest


if __name__ == '__main__':
    build_and_save_dataset()
