"""
SoccerNet-Tracking Preprocessing Utilities — For Phase 1 Pipeline Evaluation
Bismillah
"""
import configparser
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# FEATURE SELECTION — What we keep and why
# ============================================================
KEPT_FEATURES = {
    'frame': 'Frame index — temporal ordering for tracking',
    'track_id': 'Object identity — evaluated via ID-switch metrics',
    'bb_left,bb_top,bb_width,bb_height': 'Bounding box — YOLO detection evaluation',
    'image_frames': 'Raw images — YOLO+ByteTrack pipeline input',
    'fps': 'ByteTrack temporal config parameter',
    'resolution': 'YOLO input size configuration',
}

DROPPED_FEATURES = {
    'gt_confidence': 'Always 1.0 in GT — not informative',
    'gt_class_3_ball': 'Handled separately by TrackNetV2',
    'gt_class_4_referee': 'Excluded from player tracking eval',
    'gt_visibility': 'Optional — could filter heavily occluded players',
    'game_info_teams': 'Not used by tracker',
    'det_baseline': 'We run our own YOLO — det.txt is comparison only',
}

# SoccerNet-Tracking class semantics (from EDA)
GT_CLASSES = {1: 'Player', 2: 'Goalkeeper', 3: 'Ball', 4: 'Referee'}
EVAL_CLASSES = [1, 2]  # Only players + GK for MOT eval


def resolve_tracking_dir(base_dir, split_name):
    """Handle nested directory structure (e.g., train/train/)."""
    split_dir = base_dir / split_name
    nested = split_dir / split_name
    if nested.exists() and nested.is_dir():
        return nested
    return split_dir


def parse_seqinfo(seq_dir):
    """Parse seqinfo.ini for sequence metadata."""
    ini_path = seq_dir / 'seqinfo.ini'
    if not ini_path.exists():
        return {}
    
    config = configparser.ConfigParser()
    config.read(str(ini_path))
    
    if 'Sequence' in config:
        return dict(config['Sequence'])
    return {}


def load_gt(seq_dir, filter_classes=EVAL_CLASSES, min_visibility=0.3):
    """Load GT annotations in MOT format, filtered to evaluation classes and visibility.
    Note: SoccerNet-Tracking uses -1 for class and visibility (unlabeled).
    When -1, all annotations are treated as valid players."""
    gt_path = seq_dir / 'gt' / 'gt.txt'
    if not gt_path.exists():
        return pd.DataFrame(), 0
    
    gt = pd.read_csv(gt_path, header=None,
                     names=['frame', 'track_id', 'bb_left', 'bb_top',
                            'bb_width', 'bb_height', 'conf', 'class', 'visibility'])
    
    total_before = len(gt)
    
    # SoccerNet uses -1 for class (all are players) — skip filtering if so
    if filter_classes and not (gt['class'] == -1).all():
        gt = gt[gt['class'].isin(filter_classes)].copy()
    
    # SoccerNet uses -1 for visibility (not annotated) — skip filtering if so
    if min_visibility and 'visibility' in gt.columns and not (gt['visibility'] == -1).all():
        gt = gt[gt['visibility'] >= min_visibility].copy()
    
    return gt, total_before


def build_sequence_inventory(data_root):
    """Build inventory of all sequences across all splits."""
    tracking_dir = data_root / 'soccernet' / 'tracking'
    
    inventory = []
    for split in ['test', 'train', 'challenge']:
        split_dir = resolve_tracking_dir(tracking_dir, split)
        if not split_dir.exists():
            continue
        
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir() or not seq_dir.name.startswith('SNMOT'):
                continue
            
            meta = parse_seqinfo(seq_dir)
            has_gt = (seq_dir / 'gt' / 'gt.txt').exists()
            has_det = (seq_dir / 'det' / 'det.txt').exists()
            
            inventory.append({
                'name': seq_dir.name,
                'split': split,
                'path': str(seq_dir),
                'seq_length': int(meta.get('seqlength', 0)),
                'img_width': int(meta.get('imwidth', 0)),
                'img_height': int(meta.get('imheight', 0)),
                'fps': int(meta.get('framerate', 0)),
                'has_gt': has_gt,
                'has_det': has_det,
            })
    
    return pd.DataFrame(inventory)


def run_full_pipeline(data_root):
    """Run the tracking preprocessing pipeline."""
    data_root = Path(data_root)
    
    print("Building sequence inventory...")
    inventory = build_sequence_inventory(data_root)
    print(f"  {len(inventory)} sequences across {inventory['split'].nunique()} splits")
    
    # Load GT for test sequences only (evaluation)
    test_seqs = inventory[inventory['split'] == 'test']
    print(f"Loading GT for {len(test_seqs)} test sequences...")
    
    all_gt = []
    filter_stats = {'before': 0, 'after': 0}
    
    for _, row in test_seqs.iterrows():
        seq_dir = Path(row['path'])
        gt, n_before = load_gt(seq_dir)
        if len(gt) > 0:
            gt['sequence'] = row['name']
            all_gt.append(gt)
            filter_stats['before'] += n_before
            filter_stats['after'] += len(gt)
    
    gt_df = pd.concat(all_gt, ignore_index=True) if all_gt else pd.DataFrame()
    
    pct = 100 * filter_stats['after'] / max(filter_stats['before'], 1)
    print(f"  GT: {filter_stats['before']:,} → {filter_stats['after']:,} annotations ({pct:.1f}% kept after class filter)")
    
    return inventory, gt_df, filter_stats
