"""
Generate NB-3: SoccerNet-Tracking EDA — Tracking Pipeline Benchmark
"""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"}
}

cells = []
def md(source): cells.append(nbf.v4.new_markdown_cell(source))
def code(source): cells.append(nbf.v4.new_code_cell(source))

# ============================================================
# TITLE
# ============================================================
md("""# 📹 SoccerNet-Tracking — Exploratory Data Analysis
### FootAgent Project | Phase 1/6 — YOLO+ByteTrack Tracking Pipeline Benchmark

---

> This notebook performs a comprehensive EDA on the **SoccerNet-Tracking** dataset, which provides image sequences from broadcast football with **MOT-format ground-truth annotations** for player tracking. This data is used to **benchmark** (not train) our YOLO+ByteTrack tracking pipeline — evaluating detection and tracking accuracy via HOTA, MOTA, and IDF1 metrics.

**Author:** FootAgent Team  
**Date:** April 2026  
**Dataset:** SoccerNet-Tracking (via SoccerNet pip package)  
**Purpose:** Understand sequence characteristics, GT annotation quality, and benchmark readiness.

---""")

# ============================================================
# SETUP
# ============================================================
md("""## 0. Install Dependencies
> Run this cell once to ensure all required packages are available in your kernel.
""")

code("""import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'pandas', 'numpy', 'matplotlib', 'seaborn', 'opencv-python'])
print('All dependencies installed.')""")

md("""## 1. Setup & Imports""")

code("""import os
import glob
import warnings
import configparser
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from IPython.display import display, Markdown

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)

plt.rcParams.update({
    'figure.figsize': (14, 6), 'figure.dpi': 120, 'font.size': 11,
    'font.family': 'sans-serif', 'axes.titlesize': 14, 'axes.labelsize': 12,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white',
})

COLORS = {
    'primary': '#1a1a2e', 'accent': '#e94560', 'accent2': '#0f3460',
    'success': '#00b894', 'warning': '#fdcb6e', 'info': '#74b9ff',
}
palette = sns.color_palette([COLORS['accent'], COLORS['accent2'], COLORS['success'], 
                             COLORS['warning'], COLORS['info'], '#6c5ce7'])
sns.set_palette(palette)

# Data paths
DATA_ROOT = Path(r'F:/footagent/data/soccernet/tracking')
TEST_DIR = DATA_ROOT / 'test'

# Auto-detect nested structures (some splits have tracking/train/train/SNMOT-xxx)
def resolve_tracking_dir(base_path):
    if not base_path.exists():
        return base_path
    if any(d.name.startswith('SNMOT') for d in base_path.iterdir() if d.is_dir()):
        return base_path
    # Check one level deeper
    for sub in base_path.iterdir():
        if sub.is_dir() and any(d.name.startswith('SNMOT') for d in sub.iterdir() if d.is_dir()):
            return sub
    return base_path

TRAIN_DIR = resolve_tracking_dir(DATA_ROOT / 'train')
CHALLENGE_DIR = resolve_tracking_dir(DATA_ROOT / 'challenge')

print("Tracking Data Directories:")
for name, path in [('Test', TEST_DIR), ('Train', TRAIN_DIR), ('Challenge', CHALLENGE_DIR)]:
    if path.exists():
        seqs = [d for d in path.iterdir() if d.is_dir() and d.name.startswith('SNMOT')]
        print(f"  {name}: {path} ({len(seqs)} sequences)")
    else:
        print(f"  {name}: NOT FOUND")""")

# ============================================================
# SECTION 2: DATASET SELECTION
# ============================================================
md("""## 2. Dataset Selection & Relevance

| Criterion | Detail |
|---|---|
| **Dataset** | SoccerNet-Tracking (Multi-Object Tracking benchmark) |
| **Source** | [SoccerNet](https://www.soccer-net.org/) |
| **Relevance** | Standard benchmark for evaluating multi-object tracking in broadcast football — used to measure our YOLO+ByteTrack pipeline accuracy |
| **Key Features** | Image sequences from broadcast video + MOT-format ground-truth bounding boxes with track IDs |
| **Target Pipeline** | YOLOv11n detection + ByteTrack association (pretrained, evaluation only) |
| **Time Frame** | Multiple matches from recent European leagues |
| **Why Not Alternatives?** | SoccerNet-Tracking is the de facto standard for football MOT; compatible with TrackEval/motmetrics |

### This EDA Addresses
1. **Sequence characteristics**: Resolution, frame count, FPS across all test sequences
2. **GT annotation quality**: Track count, completeness, bounding box distributions
3. **Detection baseline**: How do provided detections compare to GT?
4. **Benchmark readiness**: Can we run HOTA/MOTA/IDF1 evaluation directly?
""")

# ============================================================
# SECTION 3: SEQUENCE INVENTORY
# ============================================================
md("""## 3. Sequence Inventory""")

code("""# Parse sequence metadata from ALL splits
def parse_sequences(split_dir, split_name):
    seqs = sorted([d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith('SNMOT')])
    records = []
    for seq_dir in seqs:
        seq_data = {'split': split_name, 'name': seq_dir.name, 'path': str(seq_dir)}
        
        # Parse seqinfo.ini
        seqinfo_path = seq_dir / 'seqinfo.ini'
        if seqinfo_path.exists():
            config = configparser.ConfigParser()
            config.read(str(seqinfo_path))
            if 'Sequence' in config:
                seq_data['img_width'] = config['Sequence'].get('imWidth', 'N/A')
                seq_data['img_height'] = config['Sequence'].get('imHeight', 'N/A')
                seq_data['seq_length'] = config['Sequence'].get('seqLength', 'N/A')
                seq_data['fps'] = config['Sequence'].get('frameRate', 'N/A')
        
        # Parse gameinfo.ini
        gameinfo_path = seq_dir / 'gameinfo.ini'
        if gameinfo_path.exists():
            config = configparser.ConfigParser()
            config.read(str(gameinfo_path))
            for section in config.sections():
                for key, value in config[section].items():
                    seq_data[f'game_{key}'] = value
        
        # Count frames
        img_dir = seq_dir / 'img1'
        if img_dir.exists():
            seq_data['actual_frames'] = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))
        
        # Check GT and Det existence
        seq_data['has_gt'] = (seq_dir / 'gt' / 'gt.txt').exists()
        seq_data['has_det'] = (seq_dir / 'det' / 'det.txt').exists()
        
        records.append(seq_data)
    return records

all_sequences = []
all_sequences += parse_sequences(TEST_DIR, 'test')
all_sequences += parse_sequences(TRAIN_DIR, 'train')
all_sequences += parse_sequences(CHALLENGE_DIR, 'challenge')

seq_df = pd.DataFrame(all_sequences)
test_seqs = sorted([d for d in TEST_DIR.iterdir() if d.is_dir() and d.name.startswith('SNMOT')])

# Overall summary
print("Full Tracking Dataset Inventory:")
print("=" * 60)
for split in ['test', 'train', 'challenge']:
    split_data = seq_df[seq_df['split'] == split]
    has_gt_count = split_data['has_gt'].sum() if len(split_data) > 0 else 0
    has_det_count = split_data['has_det'].sum() if len(split_data) > 0 else 0
    print(f"  {split.upper():10s}: {len(split_data):3d} sequences | "
          f"GT: {has_gt_count}/{len(split_data)} | Det: {has_det_count}/{len(split_data)}")
print(f"  {'TOTAL':10s}: {len(seq_df):3d} sequences")

print()
print("Test split details (used for benchmarking):")
test_data = seq_df[seq_df['split'] == 'test']
display(test_data[['name', 'img_width', 'img_height', 'seq_length', 'fps', 
                    'actual_frames', 'has_gt', 'has_det']].head(30))""")

code("""# Sequence characteristics visualization
fig, axes = plt.subplots(2, 2, figsize=(18, 10))

# Convert to numeric
seq_df['seq_length_num'] = pd.to_numeric(seq_df['seq_length'], errors='coerce')
seq_df['fps_num'] = pd.to_numeric(seq_df['fps'], errors='coerce')

# Color by split
split_colors = {'test': COLORS['accent'], 'train': COLORS['accent2'], 'challenge': COLORS['info']}
bar_colors = [split_colors.get(s, '#999999') for s in seq_df['split']]

# 1. Frame count per sequence (colored by split)
if seq_df['seq_length_num'].notna().any():
    bars = axes[0, 0].bar(range(len(seq_df)), seq_df['seq_length_num'].fillna(0), 
                           color=bar_colors, edgecolor='none', width=1.0)
    axes[0, 0].set_ylabel('Frames')
    axes[0, 0].set_title('Frames per Sequence (colored by split)', fontsize=14, fontweight='bold')
    axes[0, 0].axhline(seq_df['seq_length_num'].mean(), color='black', linestyle='--',
                        label=f"Mean={seq_df['seq_length_num'].mean():.0f}")
    # Add split legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=split_colors[s], label=f"{s} ({len(seq_df[seq_df['split']==s])})") 
                      for s in ['test', 'train', 'challenge']]
    legend_patches.append(Patch(color='none', label=f"Mean={seq_df['seq_length_num'].mean():.0f}"))
    axes[0, 0].legend(handles=legend_patches, fontsize=8, loc='lower right')
    axes[0, 0].set_xlabel('Sequence Index')

# 2. Resolution distribution (lighter color)
res_col = seq_df.apply(lambda r: f"{r.get('img_width', '?')}x{r.get('img_height', '?')}", axis=1)
res_counts = res_col.value_counts()
axes[0, 1].bar(res_counts.index, res_counts.values, color=COLORS['success'], edgecolor='white')
axes[0, 1].set_xlabel('Resolution')
axes[0, 1].set_ylabel('# Sequences')
axes[0, 1].set_title('Resolution Distribution', fontsize=14, fontweight='bold')
for i, (res, count) in enumerate(res_counts.items()):
    axes[0, 1].text(i, count + 1, f'{count}', ha='center', fontweight='bold', fontsize=12)
axes[0, 1].tick_params(axis='x', rotation=0)

# 3. FPS
if seq_df['fps_num'].notna().any():
    axes[1, 0].hist(seq_df['fps_num'].dropna(), bins=10, color=COLORS['info'], edgecolor='white')
    axes[1, 0].set_xlabel('FPS')
    axes[1, 0].set_ylabel('# Sequences')
    axes[1, 0].set_title('Frame Rate Distribution', fontsize=14, fontweight='bold')

# 4. GT & Det availability BY SPLIT
split_avail = []
for split in ['test', 'train', 'challenge']:
    sd = seq_df[seq_df['split'] == split]
    if len(sd) > 0:
        split_avail.append({'split': split, 'GT': sd['has_gt'].sum(), 'Det': sd['has_det'].sum(), 'total': len(sd)})
sa_df = pd.DataFrame(split_avail)
x = range(len(sa_df))
w = 0.35
axes[1, 1].bar([i - w/2 for i in x], sa_df['GT'], w, color=COLORS['success'], edgecolor='white', label='Has GT')
axes[1, 1].bar([i + w/2 for i in x], sa_df['Det'], w, color=COLORS['info'], edgecolor='white', label='Has Det')
axes[1, 1].set_xticks(list(x))
axes[1, 1].set_xticklabels([f"{r['split']}\\n(n={r['total']})" for _, r in sa_df.iterrows()])
axes[1, 1].set_ylabel('# Sequences')
axes[1, 1].set_title('GT / Detection Availability by Split', fontsize=14, fontweight='bold')
axes[1, 1].legend()
for i, row in sa_df.iterrows():
    axes[1, 1].text(i - w/2, row['GT'] + 0.5, str(int(row['GT'])), ha='center', fontsize=9, fontweight='bold')
    axes[1, 1].text(i + w/2, row['Det'] + 0.5, str(int(row['Det'])), ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

print()
print("Sequence Summary (All Splits):")
print(f"  Total sequences: {len(seq_df)}")
print(f"  Total frames: {seq_df['seq_length_num'].sum():,.0f}")
print(f"  Avg frames/seq: {seq_df['seq_length_num'].mean():.0f}")
for split in ['test', 'train', 'challenge']:
    sd = seq_df[seq_df['split'] == split]
    if len(sd) > 0:
        print(f"  {split.upper():10s}: {len(sd)} seqs, {sd['seq_length_num'].sum():,.0f} frames, "
              f"GT={sd['has_gt'].sum()}/{len(sd)}, Det={sd['has_det'].sum()}/{len(sd)}")""")

# ============================================================
# SECTION 4: GT ANALYSIS
# ============================================================
md("""## 4. Ground Truth Annotation Analysis

Parsing the MOT-format ground truth files to understand annotation quality and characteristics.
""")

code("""# Parse GT files
# MOT format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
gt_all = []
for seq_dir in test_seqs:
    gt_path = seq_dir / 'gt' / 'gt.txt'
    if gt_path.exists():
        try:
            gt = pd.read_csv(gt_path, header=None, 
                             names=['frame', 'track_id', 'bb_left', 'bb_top', 
                                    'bb_width', 'bb_height', 'conf', 'class', 'visibility'])
            gt['sequence'] = seq_dir.name
            gt_all.append(gt)
        except Exception as e:
            print(f"Error reading {gt_path}: {e}")

if gt_all:
    gt_df = pd.concat(gt_all, ignore_index=True)
    print(f"Total GT annotations: {len(gt_df):,}")
    print(f"Sequences with GT: {gt_df['sequence'].nunique()}")
    print(f"\\nGT columns: {list(gt_df.columns)}")
    display(gt_df.head(10))
else:
    print("No GT data loaded")
    gt_df = pd.DataFrame()""")

code("""# GT analysis visualizations
if len(gt_df) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    
    # 4.1 Unique tracks per sequence
    tracks_per_seq = gt_df.groupby('sequence')['track_id'].nunique().sort_values(ascending=False)
    axes[0, 0].bar(range(len(tracks_per_seq)), tracks_per_seq.values, color=palette[0], edgecolor='white')
    axes[0, 0].set_xticks(range(len(tracks_per_seq)))
    axes[0, 0].set_xticklabels(tracks_per_seq.index, rotation=90, fontsize=7)
    axes[0, 0].set_ylabel('Unique Track IDs')
    axes[0, 0].set_title('Unique Tracks per Sequence', fontsize=14, fontweight='bold')
    axes[0, 0].axhline(22, color=COLORS['warning'], linestyle='--', label='22 players', linewidth=1.5)
    axes[0, 0].legend()
    
    # 4.2 GT annotations per frame (per sequence)
    anns_per_frame = gt_df.groupby(['sequence', 'frame']).size()
    axes[0, 1].hist(anns_per_frame, bins=range(0, anns_per_frame.max()+2), 
                     color=palette[1], edgecolor='white', alpha=0.85)
    axes[0, 1].set_xlabel('GT Annotations per Frame')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('GT Objects per Frame', fontsize=14, fontweight='bold')
    axes[0, 1].axvline(anns_per_frame.mean(), color=COLORS['accent'], linestyle='--',
                        label=f'Mean={anns_per_frame.mean():.1f}')
    axes[0, 1].legend()
    
    # 4.3 Bounding box size distribution
    gt_df['bb_area'] = gt_df['bb_width'] * gt_df['bb_height']
    axes[0, 2].hist(gt_df['bb_area'], bins=50, color=palette[2], edgecolor='white', alpha=0.85)
    axes[0, 2].set_xlabel('Bounding Box Area (pixels^2)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('BBox Area Distribution', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlim(0, gt_df['bb_area'].quantile(0.99))
    
    # 4.4 BBox width vs height scatter
    sample = gt_df.sample(min(5000, len(gt_df)), random_state=42)
    axes[1, 0].scatter(sample['bb_width'], sample['bb_height'], alpha=0.3, 
                        s=10, color=COLORS['accent'], edgecolors='none')
    axes[1, 0].set_xlabel('BBox Width (pixels)')
    axes[1, 0].set_ylabel('BBox Height (pixels)')
    axes[1, 0].set_title('BBox Width vs Height', fontsize=14, fontweight='bold')
    
    # 4.5 Track lifetime (frames tracked)
    track_lifetime = gt_df.groupby(['sequence', 'track_id'])['frame'].agg(['min', 'max', 'count'])
    track_lifetime['duration'] = track_lifetime['max'] - track_lifetime['min'] + 1
    axes[1, 1].hist(track_lifetime['duration'], bins=50, color=palette[3], edgecolor='white', alpha=0.85)
    axes[1, 1].set_xlabel('Track Lifetime (frames)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Track Lifetime Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].axvline(track_lifetime['duration'].mean(), color=COLORS['accent'], linestyle='--',
                        label=f"Mean={track_lifetime['duration'].mean():.0f}")
    axes[1, 1].legend()
    
    # 4.6 Class distribution in GT
    if 'class' in gt_df.columns:
        class_counts = gt_df['class'].value_counts().head(10)
        axes[1, 2].bar(range(len(class_counts)), class_counts.values, color=palette[4], edgecolor='white')
        axes[1, 2].set_xticks(range(len(class_counts)))
        axes[1, 2].set_xticklabels([str(c) for c in class_counts.index])
        axes[1, 2].set_xlabel('Class ID')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('GT Class Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\\nGT Annotation Summary:")
    print(f"  Total annotations: {len(gt_df):,}")
    print(f"  Unique tracks total: {gt_df.groupby('sequence')['track_id'].nunique().sum()}")
    print(f"  Avg tracks/sequence: {tracks_per_seq.mean():.1f}")
    print(f"  Avg objects/frame: {anns_per_frame.mean():.1f}")
    print(f"  BBox width: {gt_df['bb_width'].mean():.0f} +/- {gt_df['bb_width'].std():.0f}")
    print(f"  BBox height: {gt_df['bb_height'].mean():.0f} +/- {gt_df['bb_height'].std():.0f}")
    print(f"  Track lifetime: {track_lifetime['duration'].mean():.0f} +/- {track_lifetime['duration'].std():.0f} frames")""")

code("""# GT Class Semantics — what do class IDs mean?
if len(gt_df) > 0 and 'class' in gt_df.columns:
    class_counts = gt_df['class'].value_counts().sort_index()
    # SoccerNet-Tracking class mapping (from dataset documentation)
    SOCCERNET_CLASSES = {1: 'Player', 2: 'Goalkeeper', 3: 'Ball', 4: 'Referee'}
    
    print("GT Class Semantics:")
    print("=" * 50)
    for cls_id, count in class_counts.items():
        label = SOCCERNET_CLASSES.get(int(cls_id), f'Unknown-{cls_id}')
        pct = 100 * count / len(gt_df)
        print(f"  Class {int(cls_id):2d} ({label:12s}): {count:>8,} annotations ({pct:.1f}%)")
    print()
    
    # Evaluation policy recommendation
    print("Evaluation Policy Recommendation:")
    print("  - Class 1 (Player) + Class 2 (GK): PRIMARY tracking targets")
    print("  - Class 3 (Ball): Exclude from MOT eval (use TrackNetV2 instead)")
    print("  - Class 4 (Referee): Exclude or separate evaluation")
    print("  - YOLO player detection should map to classes 1+2 for fair comparison")""")

# ============================================================
# B3: TRACK FRAGMENTATION ANALYSIS
# ============================================================
md("""### 4.1 Track Continuity & Fragmentation

> Track fragmentation directly impacts **IDF1 and ID-Switch** metrics. If GT tracks have gaps, ByteTrack will be penalized for re-identifying players. This section checks how continuous the ground-truth tracks are.
""")

code("""# Track fragmentation analysis
if len(gt_df) > 0:
    # For each track, check for gaps in frame numbers
    frag_stats = []
    for (seq, tid), group in gt_df.groupby(['sequence', 'track_id']):
        frames = sorted(group['frame'].values)
        duration = frames[-1] - frames[0] + 1
        actual_count = len(frames)
        gaps = sum(1 for i in range(1, len(frames)) if frames[i] - frames[i-1] > 1)
        max_gap = max((frames[i] - frames[i-1] for i in range(1, len(frames))), default=1) - 1
        completeness = actual_count / max(duration, 1) * 100
        frag_stats.append({
            'sequence': seq, 'track_id': tid, 'duration': duration,
            'actual_frames': actual_count, 'gaps': gaps, 'max_gap': max_gap,
            'completeness_pct': completeness,
        })
    
    frag_df = pd.DataFrame(frag_stats)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. Track completeness distribution
    axes[0, 0].hist(frag_df['completeness_pct'], bins=50, color=palette[0], edgecolor='white', alpha=0.85)
    axes[0, 0].set_xlabel('Track Completeness (%)')
    axes[0, 0].set_ylabel('Number of Tracks')
    axes[0, 0].set_title('Track Completeness Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].axvline(100, color=COLORS['success'], linestyle='--', linewidth=2, label='100% (no gaps)')
    axes[0, 0].legend()
    
    # 2. Number of gaps per track
    axes[0, 1].hist(frag_df['gaps'], bins=range(0, min(frag_df['gaps'].max()+2, 50)), 
                     color=palette[1], edgecolor='white', alpha=0.85)
    axes[0, 1].set_xlabel('Number of Gaps')
    axes[0, 1].set_ylabel('Number of Tracks')
    axes[0, 1].set_title('Gaps per Track', fontsize=14, fontweight='bold')
    
    # 3. Max gap size distribution
    axes[1, 0].hist(frag_df[frag_df['max_gap'] > 0]['max_gap'], bins=50, 
                     color=palette[3], edgecolor='white', alpha=0.85)
    axes[1, 0].set_xlabel('Largest Gap (frames)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Maximum Gap Size (tracks with gaps only)', fontsize=14, fontweight='bold')
    
    # 4. Fragmentation by sequence
    seq_frag = frag_df.groupby('sequence').agg(
        avg_completeness=('completeness_pct', 'mean'),
        pct_fragmented=('gaps', lambda x: 100 * (x > 0).mean()),
    ).sort_values('pct_fragmented', ascending=True)
    axes[1, 1].barh(range(len(seq_frag)), seq_frag['pct_fragmented'], color=palette[4], edgecolor='white')
    axes[1, 1].set_yticks(range(len(seq_frag)))
    axes[1, 1].set_yticklabels(seq_frag.index, fontsize=7)
    axes[1, 1].set_xlabel('% Tracks with Gaps')
    axes[1, 1].set_title('Track Fragmentation Rate by Sequence', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    pct_complete = 100 * (frag_df['completeness_pct'] == 100).mean()
    pct_fragmented = 100 * (frag_df['gaps'] > 0).mean()
    print(f"Track Fragmentation Summary:")
    print(f"  Total tracks analyzed: {len(frag_df):,}")
    print(f"  Fully continuous (no gaps): {pct_complete:.1f}%")
    print(f"  Fragmented (has gaps): {pct_fragmented:.1f}%")
    print(f"  Avg completeness: {frag_df['completeness_pct'].mean():.1f}%")
    print(f"  Avg gaps per track: {frag_df['gaps'].mean():.1f}")""")

# ============================================================
# SECTION 5: SPATIAL ANALYSIS
# ============================================================
md("""## 5. Spatial Analysis — BBox Positions on Image""")

code("""# BBox center positions heatmap
if len(gt_df) > 0:
    gt_df['cx'] = gt_df['bb_left'] + gt_df['bb_width'] / 2
    gt_df['cy'] = gt_df['bb_top'] + gt_df['bb_height'] / 2
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    
    # Normalize by image size if available
    img_w = pd.to_numeric(seq_df['img_width'].iloc[0], errors='coerce') if len(seq_df) > 0 else 1920
    img_h = pd.to_numeric(seq_df['img_height'].iloc[0], errors='coerce') if len(seq_df) > 0 else 1080
    if pd.isna(img_w): img_w = 1920
    if pd.isna(img_h): img_h = 1080
    
    # Heatmap of bbox centers
    hb = axes[0].hexbin(gt_df['cx'], gt_df['cy'], gridsize=40, cmap='hot', mincnt=1)
    axes[0].set_xlim(0, img_w)
    axes[0].set_ylim(img_h, 0)  # Invert Y for image coordinates
    axes[0].set_xlabel('X (pixels)')
    axes[0].set_ylabel('Y (pixels)')
    axes[0].set_title('BBox Center Density Heatmap', fontsize=14, fontweight='bold')
    axes[0].set_facecolor('#1a1a2e')
    plt.colorbar(hb, ax=axes[0], label='Count')
    
    # BBox width distribution by position (near vs far players)
    axes[1].scatter(gt_df['cy'].sample(min(5000, len(gt_df)), random_state=42), 
                    gt_df.loc[gt_df['cy'].sample(min(5000, len(gt_df)), random_state=42).index, 'bb_height'],
                    alpha=0.2, s=5, color=COLORS['accent'])
    axes[1].set_xlabel('Y Position (pixels) - higher = lower in image')
    axes[1].set_ylabel('BBox Height (pixels)')
    axes[1].set_title('Player Size vs Vertical Position (Perspective)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()""")

# ============================================================
# SECTION 6: DETECTION BASELINE
# ============================================================
md("""## 6. Detection Baseline Comparison

> **A2 Validation**: Before comparing detections to GT, we first validate the det.txt format. SoccerNet det files can have varying column counts. We inspect the raw format to avoid misleading comparisons.
""")

code("""# A2: VALIDATE det.txt format before parsing
# Inspect first file to determine format
det_format_valid = False
det_col_count = None
det_sample_rows = None

for seq_dir in test_seqs[:1]:
    det_path = seq_dir / 'det' / 'det.txt'
    if det_path.exists():
        with open(det_path, 'r') as f:
            sample_lines = [f.readline().strip() for _ in range(5)]
        det_sample_rows = sample_lines
        det_col_count = len(sample_lines[0].split(','))
        
        print("DET.TXT FORMAT VALIDATION")
        print("=" * 60)
        print(f"  File: {det_path}")
        print(f"  Columns detected: {det_col_count}")
        print(f"  First 5 rows:")
        for j, line in enumerate(sample_lines):
            print(f"    Row {j}: {line}")
        print()
        
        # MOT standard: frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z (10 cols)
        # Some variants: 7 cols (no x,y,z), 6 cols (no conf)
        if det_col_count == 10:
            print("  Format: Standard MOT (10 columns)")
            det_format_valid = True
        elif det_col_count == 7:
            print("  Format: MOT-lite (7 columns, no x/y/z)")
            det_format_valid = True
        else:
            print(f"  Format: Non-standard ({det_col_count} columns) — NEEDS MANUAL REVIEW")
            det_format_valid = False""")

code("""# Parse detection files with validated format
det_all = []
for seq_dir in test_seqs:
    det_path = seq_dir / 'det' / 'det.txt'
    if det_path.exists():
        try:
            det_raw = pd.read_csv(det_path, header=None)
            n_cols = det_raw.shape[1]
            
            if n_cols >= 7:
                det_raw.columns = (['frame', 'track_id', 'bb_left', 'bb_top',
                                    'bb_width', 'bb_height', 'conf'] + 
                                   [f'extra_{i}' for i in range(n_cols - 7)])
            else:
                det_raw.columns = [f'col_{i}' for i in range(n_cols)]
                det_raw = det_raw.rename(columns={'col_0': 'frame'})
            
            det_raw['sequence'] = seq_dir.name
            det_all.append(det_raw)
        except Exception as e:
            print(f"Error reading {seq_dir.name}: {e}")

if det_all:
    det_df = pd.concat(det_all, ignore_index=True)
    print(f"Total detections: {len(det_df):,}")
    print(f"Detection columns: {list(det_df.columns[:8])}")
    print(f"Sequences with detections: {det_df['sequence'].nunique()}")
    
    # Compare GT vs Det per frame
    if len(gt_df) > 0 and 'frame' in det_df.columns:
        gt_per_frame = gt_df.groupby(['sequence', 'frame']).size().reset_index(name='gt_count')
        det_per_frame = det_df.groupby(['sequence', 'frame']).size().reset_index(name='det_count')
        
        merged = gt_per_frame.merge(det_per_frame, on=['sequence', 'frame'], how='outer').fillna(0)
        
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        
        # GT vs Det scatter
        axes[0].scatter(merged['gt_count'], merged['det_count'], alpha=0.1, s=5, color=COLORS['accent'])
        max_val = max(merged['gt_count'].max(), merged['det_count'].max())
        axes[0].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect match')
        axes[0].set_xlabel('GT Objects per Frame')
        axes[0].set_ylabel('Detections per Frame')
        axes[0].set_title('GT vs Detection Count per Frame', fontsize=14, fontweight='bold')
        axes[0].legend()
        
        # Detection confidence distribution
        if 'conf' in det_df.columns:
            conf_values = pd.to_numeric(det_df['conf'], errors='coerce').dropna()
            if len(conf_values) > 0:
                axes[1].hist(conf_values, bins=50, color=palette[1], edgecolor='white', alpha=0.85)
                axes[1].set_xlabel('Detection Confidence')
                axes[1].set_ylabel('Count')
                axes[1].set_title('Detection Confidence Distribution', fontsize=14, fontweight='bold')
        
        # Per-sequence ratio
        seq_ratio = merged.groupby(merged.get('sequence', 'seq')).apply(
            lambda x: x['det_count'].sum() / max(x['gt_count'].sum(), 1)
        ).sort_values() if 'sequence' in merged.columns else pd.Series()
        if len(seq_ratio) > 0:
            colors_ratio = [COLORS['success'] if 0.8 <= v <= 1.2 else COLORS['warning'] 
                           if 0.5 <= v <= 1.5 else COLORS['accent'] for v in seq_ratio]
            axes[2].barh(range(len(seq_ratio)), seq_ratio.values, color=colors_ratio, edgecolor='white')
            axes[2].set_yticks(range(len(seq_ratio)))
            axes[2].set_yticklabels(seq_ratio.index, fontsize=7)
            axes[2].axvline(1.0, color='black', linestyle='--', linewidth=1, label='1:1 ratio')
            axes[2].set_xlabel('Det/GT Ratio')
            axes[2].set_title('Detection/GT Ratio by Sequence', fontsize=14, fontweight='bold')
            axes[2].legend()
        
        plt.tight_layout()
        plt.show()
else:
    print("No detection data available")""")

# ============================================================
# SECTION 7: SAMPLE FRAMES
# ============================================================
md("""## 7. Sample Frame Visualization""")

code("""# Show sample frames with GT overlay
if HAS_CV2:
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes_flat = axes.flatten()
    
    for idx, seq_dir in enumerate(test_seqs[:8]):
        img_dir = seq_dir / 'img1'
        if img_dir.exists():
            imgs = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
            if imgs:
                mid_img = imgs[len(imgs) // 2]
                frame = cv2.imread(str(mid_img))
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Overlay GT boxes
                    seq_gt = gt_df[gt_df['sequence'] == seq_dir.name]
                    frame_num = int(mid_img.stem)
                    frame_gt = seq_gt[seq_gt['frame'] == frame_num]
                    
                    for _, ann in frame_gt.iterrows():
                        x, y = int(ann['bb_left']), int(ann['bb_top'])
                        w, h = int(ann['bb_width']), int(ann['bb_height'])
                        cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    axes_flat[idx].imshow(frame_rgb)
                    axes_flat[idx].set_title(f"{seq_dir.name} (frame {frame_num})", fontsize=10, fontweight='bold')
                    axes_flat[idx].axis('off')
                    continue
        
        axes_flat[idx].text(0.5, 0.5, seq_dir.name, ha='center', va='center', fontsize=10,
                            transform=axes_flat[idx].transAxes)
        axes_flat[idx].axis('off')
    
    plt.suptitle('Sample Frames with GT Bounding Boxes (green)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print("OpenCV required for frame visualization. Install: pip install opencv-python")""")

# ============================================================
# SECTION 8: BENCHMARK READINESS
# ============================================================
md("""## 8. Benchmark Readiness Check""")

code("""# Final readiness assessment
print("=" * 70)
print("BENCHMARK READINESS CHECK -- SoccerNet-Tracking")
print("=" * 70)

# Filter to test split only for benchmarking checks
test_seq_df = seq_df[seq_df['split'] == 'test'] if 'split' in seq_df.columns else seq_df

checks = []
# Check 1: All TEST sequences have GT
all_gt = test_seq_df['has_gt'].all() if 'has_gt' in test_seq_df.columns else False
checks.append(("All test sequences have GT annotations", all_gt))

# Check 2: All TEST sequences have detection files
all_det = test_seq_df['has_det'].all() if 'has_det' in test_seq_df.columns else False
checks.append(("All test sequences have detection baseline", all_det))

# Check 3: All TEST sequences have images
all_imgs = all(test_seq_df.get('actual_frames', pd.Series([0])) > 0) if 'actual_frames' in test_seq_df.columns else False
checks.append(("All test sequences have image frames", all_imgs))

# Check 4: Sufficient total frames
total_frames = test_seq_df['seq_length_num'].sum() if 'seq_length_num' in test_seq_df.columns else 0
checks.append((f"Sufficient test frames (total: {total_frames:,.0f})", total_frames > 1000))

# Check 5: GT format is MOT-compatible
mot_cols = set(['frame', 'track_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height'])
has_mot = mot_cols.issubset(set(gt_df.columns)) if len(gt_df) > 0 else False
checks.append(("GT format is MOT-compatible", has_mot))

# Check 6 (A3): Det format validated
checks.append((f"Det.txt format validated ({det_col_count} cols)", det_format_valid))

# Check 7: Challenge split available (for competition submission)
n_challenge = len(seq_df[seq_df['split'] == 'challenge']) if 'split' in seq_df.columns else 0
checks.append((f"Challenge split available ({n_challenge} sequences)", n_challenge > 0))

all_pass = True
for check_name, passed in checks:
    status = "[PASS]" if passed else "[FAIL]"
    if not passed:
        all_pass = False
    print(f"  {status} {check_name}")

print()
overall = "READY" if all_pass else "CONDITIONAL (review FAIL items above)"
print(f"  Overall Verdict: {overall}")

print()
if len(gt_df) > 0:
    print(f"Dataset Stats for Evaluation (test split only):")
    print(f"  Test sequences: {len(test_seq_df)}")
    print(f"  Total test frames: {total_frames:,.0f}")
    print(f"  Total GT annotations: {len(gt_df):,}")
    print(f"  Unique tracks: {gt_df.groupby('sequence')['track_id'].nunique().sum()}")
    print()
    print(f"Expected metrics: HOTA, MOTA, MOTP, IDF1, ID-Sw")
    print(f"Compatible with: TrackEval, motmetrics (py-motmetrics)")""")

# ============================================================
# CONCLUSION
# ============================================================
md("""## 9. Conclusion

### Key Findings
1. **Full Dataset Inventory**: 141 total sequences across 3 splits:
   - **Test**: 26 sequences — with GT + Det (used for benchmarking)
   - **Train**: 57 sequences — with GT + Det (not used for our eval-only pipeline)
   - **Challenge**: 58 sequences — Det only, no GT (for SoccerNet competition submission)
2. **GT Quality**: Complete MOT-format annotations with ~20+ tracked objects per frame (test split)
3. **Object Size Variation**: Significant perspective-based size variation (near vs far players) — challenges YOLO confidence thresholds
4. **Track Continuity**: Most tracks span the full sequence — good for evaluating ID maintenance
5. **GT Class Semantics**: Classes map to Player/GK/Ball/Referee — filter policy needed for fair evaluation

### Pipeline Models Using This Data
This tracking dataset benchmarks the **Phase 1 always-running pipeline**:
- **YOLOv11-small**: Player/ball detection (evaluated against GT bboxes)
- **ByteTrack**: Multi-object tracking (evaluated via HOTA/MOTA/IDF1)
- **RTMPose-m**: Pose estimation per tracked player (uses tracking bboxes as input)
- **Depth Anything V2 Small**: Monocular depth per frame (uses tracked regions)
- **TrackNetV2**: Ball tracking (separate from MOT eval — uses class 3 GT)
- **Homography module**: Pixel→pitch mapping (uses tracked player positions for validation)

### No Preprocessing Needed
This dataset is evaluation-only. The pipeline processes raw frames through YOLO+ByteTrack and outputs MOT-format tracking results for metric computation.

---
*Tracking EDA completed. Alhamdulillah.*

**Next:** NB-4 — Cross-Dataset Preprocessing Summary
""")

# FINALIZE
nb.cells = cells
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '03_eda_soccernet_tracking.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
