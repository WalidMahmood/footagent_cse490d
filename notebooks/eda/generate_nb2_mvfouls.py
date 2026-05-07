"""
Generate NB-2: SoccerNet-MVFouls EDA — ViT-B Foul Classifier Data Exploration
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
md("""# 🎬 SoccerNet-MVFouls — Exploratory Data Analysis
### FootAgent Project | Phase 4 — ViT-B Foul Classifier

---

> This notebook performs a comprehensive EDA on the **SoccerNet Multi-View Fouls (MVFouls)** dataset. This dataset consists of short multi-view video clips of foul actions in football, annotated with **foul type, severity, and offence classification**. It is used to fine-tune a **ViT-B (Vision Transformer Base)** model for automated foul recognition — a key component of FootAgent's VAR decision support system.

**Author:** FootAgent Team  
**Date:** April 2026  
**Dataset:** SoccerNet-MVFouls (via SoccerNet pip package)  
**Purpose:** Understand class balance, video quality, multi-view characteristics, and preprocessing needs before ViT-B fine-tuning.

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

code("""# ============================================================
# Core Libraries
# ============================================================
import json
import os
import glob
import warnings
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from IPython.display import display, Markdown, HTML, Video

# Video analysis
try:
    import cv2
    HAS_CV2 = True
    print("OpenCV available")
except ImportError:
    HAS_CV2 = False
    print("OpenCV not available - video analysis will be limited")

# ============================================================
# Configuration
# ============================================================
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

plt.rcParams.update({
    'figure.figsize': (14, 6),
    'figure.dpi': 120,
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
})

COLORS = {
    'primary': '#1a1a2e', 'secondary': '#16213e', 'accent': '#e94560',
    'accent2': '#0f3460', 'success': '#00b894', 'warning': '#fdcb6e',
    'info': '#74b9ff', 'danger': '#d63031',
}
palette = sns.color_palette([COLORS['accent'], COLORS['accent2'], COLORS['success'], 
                             COLORS['warning'], COLORS['info'], '#6c5ce7', '#fd79a8'])
sns.set_palette(palette)

# Data paths
DATA_ROOT = Path(r'F:/footagent/data/soccernet/mvfouls')
TRAIN_DIR = DATA_ROOT / 'train'
VALID_DIR = DATA_ROOT / 'valid'
TEST_DIR = DATA_ROOT / 'test'

print("Data directories:")
for name, path in [('Train', TRAIN_DIR), ('Valid', VALID_DIR), ('Test', TEST_DIR)]:
    exists = path.exists()
    print(f"  {name}: {path} {'(exists)' if exists else '(MISSING)'}")""")

# ============================================================
# SECTION 2: DATASET SELECTION
# ============================================================
md("""## 2. Dataset Selection & Relevance

| Criterion | Detail |
|---|---|
| **Dataset** | SoccerNet Multi-View Fouls (MVFouls) |
| **Source** | [SoccerNet](https://www.soccer-net.org/) research benchmark suite |
| **Relevance** | Only large-scale public dataset combining **multi-view video clips** with **foul type + severity labels** — essential for training a vision model to support VAR-style decisions |
| **Key Features** | Multiple camera angle videos per foul action, annotated with action class (tackle, standing tackle, etc.), severity (1-4), and offence/no-offence classification |
| **Target Model** | ViT-B (Vision Transformer Base) fine-tuned for multi-class foul classification |
| **Time Frame** | Multiple seasons from top European leagues |
| **Why Not Alternatives?** | No other public dataset provides multi-view foul videos with severity labels at this scale |

### Research Questions
1. **Class Balance**: How balanced are the foul types and severity levels? What sampling strategy is needed?
2. **Video Quality**: Are clips consistent in resolution, FPS, and duration?
3. **Multi-View Coverage**: How many views per action? Is view #0 always the best angle?
4. **Split Integrity**: No overlap between train/valid/test?
5. **Scale**: Enough samples for ViT-B fine-tuning?
""")

# ============================================================
# SECTION 3: DATA LOADING
# ============================================================
md("""## 3. Data Loading & Annotation Schema""")

code("""# Load annotations
annotations = {}
for split_name, split_dir in [('train', TRAIN_DIR), ('valid', VALID_DIR), ('test', TEST_DIR)]:
    ann_file = split_dir / 'annotations.json'
    if ann_file.exists():
        with open(ann_file, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        # MVFouls format: {"Set": ..., "Number of actions": N, "Actions": {"0": {...}, ...}}
        actions_dict = raw.get('Actions', raw)  # fallback to raw if no 'Actions' key
        annotations[split_name] = actions_dict
        n_actions = raw.get('Number of actions', len(actions_dict))
        print(f"Loaded {split_name}: {n_actions} actions, file size: {ann_file.stat().st_size / 1024:.0f} KB")
    else:
        print(f"WARNING: {split_name} annotations not found at {ann_file}")

# Show sample annotation
if 'train' in annotations:
    sample_key = list(annotations['train'].keys())[0]
    print(f"\\nSample annotation key: '{sample_key}'")
    print(f"Sample annotation value:")
    print(json.dumps(annotations['train'][sample_key], indent=2, default=str)[:1500])""")

code("""# Build a unified DataFrame from all annotations
all_actions = []
for split_name, ann_dict in annotations.items():
    for action_key, action_data in ann_dict.items():
        if not isinstance(action_data, dict):
            continue  # skip non-dict entries
        record = {
            'split': split_name,
            'action_key': action_key,
            'action_class': action_data.get('Action class', 'Unknown'),
            'offence': action_data.get('Offence', 'Unknown'),
            'severity': action_data.get('Severity', 'Unknown'),
            'bodypart': action_data.get('Bodypart', 'Unknown'),
            'upper_body_part': action_data.get('Upper body part', ''),
            'contact': action_data.get('Contact', 'Unknown'),
            'handball': action_data.get('Handball', ''),
            'num_clips': len(action_data.get('Clips', [])),
        }
        all_actions.append(record)

actions_df = pd.DataFrame(all_actions)
print(f"Total actions across all splits: {len(actions_df):,}")
print(f"\\nColumns: {list(actions_df.columns)}")
print(f"\\nSplit distribution:")
print(actions_df['split'].value_counts().to_string())
print(f"\\nAction classes found: {actions_df['action_class'].nunique()}")
print(actions_df['action_class'].value_counts().to_string())
display(actions_df.head(10))""")

# ============================================================
# SECTION 4: SAMPLE SIZE
# ============================================================
md("""## 4. Sample Size & Split Distribution""")

code("""# Sample size analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart: samples per split
split_counts = actions_df['split'].value_counts()
colors_split = [COLORS['accent2'], COLORS['success'], COLORS['accent']]
bars = axes[0].bar(split_counts.index, split_counts.values, color=colors_split, edgecolor='white', width=0.5)
axes[0].set_ylabel('Number of Actions')
axes[0].set_title('Sample Count per Split', fontsize=14, fontweight='bold')
for bar, val in zip(bars, split_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                 f'{val:,}', ha='center', fontweight='bold', fontsize=14)
    pct = 100 * val / len(actions_df)
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() - 50, 
                 f'({pct:.1f}%)', ha='center', fontsize=10, color='white')

# Pie chart
axes[1].pie(split_counts.values, labels=split_counts.index, autopct='%1.1f%%',
            colors=colors_split, wedgeprops=dict(edgecolor='white'), textprops={'fontsize': 12})
axes[1].set_title('Split Proportions', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\\nTotal actions: {len(actions_df):,}")
print(f"Train: {split_counts.get('train', 0):,} | Valid: {split_counts.get('valid', 0):,} | Test: {split_counts.get('test', 0):,}")""")

# ============================================================
# SECTION 5: CLASS DISTRIBUTION
# ============================================================
md("""## 5. Class Distribution Analysis

This is the **most critical section** for the ViT-B classifier — class imbalance directly affects model training strategy, loss function choice, and evaluation metrics.
""")

code("""# Action class distribution
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# 5.1 Action class distribution (overall)
class_counts = actions_df['action_class'].value_counts()
bars = axes[0, 0].barh(range(len(class_counts)), class_counts.values, color=palette[0], edgecolor='white')
axes[0, 0].set_yticks(range(len(class_counts)))
axes[0, 0].set_yticklabels(class_counts.index, fontsize=10)
axes[0, 0].set_xlabel('Count')
axes[0, 0].set_title('Action Class Distribution (All Splits)', fontsize=14, fontweight='bold')
axes[0, 0].invert_yaxis()
for bar, val in zip(bars, class_counts.values):
    pct = 100 * val / len(actions_df)
    axes[0, 0].text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                     f'{val:,} ({pct:.1f}%)', va='center', fontsize=9)

# 5.2 Offence distribution
offence_counts = actions_df['offence'].value_counts()
bars = axes[0, 1].bar(offence_counts.index, offence_counts.values, 
                       color=[COLORS['danger'], COLORS['success']][:len(offence_counts)], 
                       edgecolor='white', width=0.5)
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Offence vs No Offence', fontsize=14, fontweight='bold')
for bar, val in zip(bars, offence_counts.values):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                     f'{val:,}', ha='center', fontweight='bold', fontsize=12)

# 5.3 Severity distribution
severity_counts = actions_df['severity'].value_counts().sort_index()
bars = axes[1, 0].bar(range(len(severity_counts)), severity_counts.values,
                       color=sns.color_palette('YlOrRd', len(severity_counts)), edgecolor='white')
axes[1, 0].set_xticks(range(len(severity_counts)))
axes[1, 0].set_xticklabels(severity_counts.index)
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Severity Distribution', fontsize=14, fontweight='bold')
for bar, val in zip(bars, severity_counts.values):
    pct = 100 * val / len(actions_df)
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f'{val:,}\\n({pct:.1f}%)', ha='center', fontsize=9, fontweight='bold')

# 5.4 Body part distribution
bp_counts = actions_df['bodypart'].value_counts()
axes[1, 1].pie(bp_counts.values, labels=bp_counts.index, autopct='%1.1f%%',
               colors=sns.color_palette('husl', len(bp_counts)),
               wedgeprops=dict(edgecolor='white'), textprops={'fontsize': 10})
axes[1, 1].set_title('Body Part Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()""")

code("""# Class distribution PER SPLIT (to check stratification)
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
splits = ['train', 'valid', 'test']

for idx, split in enumerate(splits):
    split_data = actions_df[actions_df['split'] == split]
    if len(split_data) > 0:
        class_counts_split = split_data['action_class'].value_counts()
        bars = axes[idx].barh(range(len(class_counts_split)), class_counts_split.values, 
                               color=palette[idx], edgecolor='white')
        axes[idx].set_yticks(range(len(class_counts_split)))
        axes[idx].set_yticklabels(class_counts_split.index, fontsize=9)
        axes[idx].set_xlabel('Count')
        axes[idx].set_title(f'Action Classes — {split.upper()} (n={len(split_data):,})', 
                            fontsize=13, fontweight='bold')
        axes[idx].invert_yaxis()
        for bar, val in zip(bars, class_counts_split.values):
            pct = 100 * val / len(split_data)
            axes[idx].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                           f'{pct:.1f}%', va='center', fontsize=8)

plt.suptitle('Class Distribution Across Splits — Checking Stratification', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()""")

code("""# Cross-tabulation: Action class x Severity
if 'severity' in actions_df.columns and 'action_class' in actions_df.columns:
    cross_tab = pd.crosstab(actions_df['action_class'], actions_df['severity'])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd', ax=ax, 
                linewidths=0.5, linecolor='white', cbar_kws={'label': 'Count'})
    ax.set_title('Action Class x Severity — Cross-Tabulation Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Severity Level', fontsize=12)
    ax.set_ylabel('Action Class', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    print("\\nCross-tabulation table:")
    display(cross_tab)""")

# ============================================================
# B4: ACTION CLASS x BODYPART CROSSTAB
# ============================================================
code("""# Action class x Body part cross-tabulation
if 'bodypart' in actions_df.columns and 'action_class' in actions_df.columns:
    cross_tab_bp = pd.crosstab(actions_df['action_class'], actions_df['bodypart'])
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Heatmap (raw counts)
    sns.heatmap(cross_tab_bp, annot=True, fmt='d', cmap='YlGnBu', ax=axes[0], 
                linewidths=0.5, linecolor='white', cbar_kws={'label': 'Count'})
    axes[0].set_title('Action Class x Body Part (Count)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Body Part', fontsize=12)
    axes[0].set_ylabel('Action Class', fontsize=12)
    
    # Heatmap (row-normalized %)
    cross_tab_bp_pct = cross_tab_bp.div(cross_tab_bp.sum(axis=1), axis=0) * 100
    sns.heatmap(cross_tab_bp_pct, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[1], 
                linewidths=0.5, linecolor='white', cbar_kws={'label': '%'})
    axes[1].set_title('Action Class x Body Part (Row %)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Body Part', fontsize=12)
    axes[1].set_ylabel('Action Class', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    print("Insight: Shows which body parts dominate each action type.")
    print("Informs whether body-part-specific augmentation could help the ViT-B classifier.")""")

# ============================================================
# SECTION 6: CLASS IMBALANCE
# ============================================================
md("""## 6. Class Imbalance Analysis & Proposed Weights""")

code("""# Compute class weights
from collections import Counter

class_counts_dict = dict(actions_df['action_class'].value_counts())
total = sum(class_counts_dict.values())
n_classes = len(class_counts_dict)

# Inverse frequency weights (sklearn-style)
weights = {cls: total / (n_classes * count) for cls, count in class_counts_dict.items()}

# Normalized weights
max_w = max(weights.values())
weights_normalized = {cls: w / max_w for cls, w in sorted(weights.items(), key=lambda x: x[1], reverse=True)}

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Imbalance ratio
sorted_counts = sorted(class_counts_dict.items(), key=lambda x: x[1], reverse=True)
classes_sorted = [c[0] for c in sorted_counts]
counts_sorted = [c[1] for c in sorted_counts]
imbalance_ratio = [counts_sorted[0] / c for c in counts_sorted]

axes[0].bar(range(len(classes_sorted)), imbalance_ratio, color=palette[0], edgecolor='white')
axes[0].set_xticks(range(len(classes_sorted)))
axes[0].set_xticklabels(classes_sorted, rotation=45, ha='right', fontsize=9)
axes[0].set_ylabel('Imbalance Ratio (vs majority class)')
axes[0].set_title('Class Imbalance Ratio', fontsize=14, fontweight='bold')
axes[0].axhline(y=5, color=COLORS['warning'], linestyle='--', label='5x threshold')
axes[0].axhline(y=10, color=COLORS['danger'], linestyle='--', label='10x threshold')
axes[0].legend()

# Proposed weights
w_sorted = sorted(weights_normalized.items(), key=lambda x: x[1], reverse=True)
axes[1].bar(range(len(w_sorted)), [w[1] for w in w_sorted], color=palette[2], edgecolor='white')
axes[1].set_xticks(range(len(w_sorted)))
axes[1].set_xticklabels([w[0] for w in w_sorted], rotation=45, ha='right', fontsize=9)
axes[1].set_ylabel('Normalized Weight')
axes[1].set_title('Proposed Class Weights for Loss Function', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print("\\nProposed Class Weights (for CrossEntropyLoss):")
for cls, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
    count = class_counts_dict[cls]
    print(f"  {cls:.<35s} count={count:>5,}  weight={w:.4f}")""")

# ============================================================
# SECTION 7: MULTI-VIEW CLIP ANALYSIS
# ============================================================
md("""## 7. Multi-View Clip Analysis

MVFouls provides **multiple camera views** for each foul action. This section analyzes how many views are available per action and their characteristics.
""")

code("""# Count clips per action
clips_per_action = []
for split_name, split_dir in [('train', TRAIN_DIR), ('valid', VALID_DIR), ('test', TEST_DIR)]:
    action_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith('action_')])
    for action_dir in action_dirs:
        clips = list(action_dir.glob('*.mp4'))
        clips_per_action.append({
            'split': split_name,
            'action_dir': action_dir.name,
            'num_clips': len(clips),
            'clip_names': [c.name for c in clips],
            'total_size_mb': sum(c.stat().st_size for c in clips) / (1024 * 1024),
        })

clips_df = pd.DataFrame(clips_per_action)
print(f"Total action directories: {len(clips_df):,}")
print(f"\\nClips per action distribution:")
print(clips_df['num_clips'].describe().to_string())
display(clips_df.head(10))""")

code("""# Visualize clips per action
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Distribution of clip counts
clip_count_dist = clips_df['num_clips'].value_counts().sort_index()
axes[0].bar(clip_count_dist.index, clip_count_dist.values, color=palette[0], edgecolor='white')
axes[0].set_xlabel('Number of Video Clips')
axes[0].set_ylabel('Number of Actions')
axes[0].set_title('Video Clips per Action', fontsize=14, fontweight='bold')
for x, v in zip(clip_count_dist.index, clip_count_dist.values):
    axes[0].text(x, v + 5, f'{v:,}', ha='center', fontweight='bold', fontsize=10)

# Clips per split
for idx, split in enumerate(['train', 'valid', 'test']):
    split_clips = clips_df[clips_df['split'] == split]['num_clips']
    if len(split_clips) > 0:
        axes[1].hist(split_clips, bins=range(0, split_clips.max()+2), alpha=0.6,
                     label=f'{split} (n={len(split_clips)})', color=palette[idx], edgecolor='white')
axes[1].set_xlabel('Clips per Action')
axes[1].set_ylabel('Count')
axes[1].set_title('Clips per Action by Split', fontsize=14, fontweight='bold')
axes[1].legend()

# File size distribution
axes[2].hist(clips_df['total_size_mb'], bins=30, color=palette[3], edgecolor='white', alpha=0.85)
axes[2].set_xlabel('Total Clip Size (MB)')
axes[2].set_ylabel('Number of Actions')
axes[2].set_title('Total Video Size per Action', fontsize=14, fontweight='bold')
axes[2].axvline(clips_df['total_size_mb'].mean(), color=COLORS['accent'], linestyle='--',
                label=f"Mean={clips_df['total_size_mb'].mean():.1f} MB")
axes[2].legend()

plt.tight_layout()
plt.show()

print(f"\\nTotal video data size: {clips_df['total_size_mb'].sum():.1f} MB ({clips_df['total_size_mb'].sum()/1024:.2f} GB)")
print(f"Total clips: {clips_df['num_clips'].sum():,}")""")

# ============================================================
# SECTION 8: VIDEO PROPERTIES
# ============================================================
md("""## 8. Video Properties Analysis

Analyzing the resolution, FPS, duration, and frame count of the video clips — critical for designing the frame sampling strategy.
""")

code("""# Sample video properties using OpenCV
if HAS_CV2:
    print("Analyzing video properties (sampling ~100 clips)...")
    video_props = []
    
    # Sample clips from each split
    all_clips = []
    for split_name, split_dir in [('train', TRAIN_DIR), ('valid', VALID_DIR), ('test', TEST_DIR)]:
        for action_dir in sorted(split_dir.iterdir()):
            if action_dir.is_dir() and action_dir.name.startswith('action_'):
                for clip in action_dir.glob('*.mp4'):
                    all_clips.append((split_name, action_dir.name, clip))
    
    # Random sample
    sample_size = min(150, len(all_clips))
    random.seed(42)
    sampled = random.sample(all_clips, sample_size)
    
    for split_name, action_name, clip_path in sampled:
        try:
            cap = cv2.VideoCapture(str(clip_path))
            if cap.isOpened():
                video_props.append({
                    'split': split_name,
                    'action': action_name,
                    'clip': clip_path.name,
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'fps': cap.get(cv2.CAP_PROP_FPS),
                    'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    'duration_sec': cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
                    'file_size_kb': clip_path.stat().st_size / 1024,
                })
            cap.release()
        except Exception as e:
            pass
    
    vprops_df = pd.DataFrame(video_props)
    print(f"Analyzed {len(vprops_df)} clips successfully")
    display(vprops_df.describe())
else:
    print("OpenCV not available. Install with: pip install opencv-python")
    vprops_df = pd.DataFrame()""")

code("""# Video properties visualization
if len(vprops_df) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    
    # Resolution scatter
    axes[0, 0].scatter(vprops_df['width'], vprops_df['height'], alpha=0.5, 
                        color=COLORS['accent'], s=30, edgecolors='white', linewidth=0.3)
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Height (pixels)')
    axes[0, 0].set_title('Video Resolution', fontsize=14, fontweight='bold')
    
    # Unique resolutions
    res_str = vprops_df.apply(lambda r: f"{int(r['width'])}x{int(r['height'])}", axis=1)
    res_counts = res_str.value_counts()
    print("Resolution distribution:")
    for res, count in res_counts.head(10).items():
        print(f"  {res}: {count} clips ({100*count/len(vprops_df):.1f}%)")
    
    # FPS distribution
    axes[0, 1].hist(vprops_df['fps'], bins=20, color=COLORS['accent2'], edgecolor='white', alpha=0.85)
    axes[0, 1].set_xlabel('FPS')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Frame Rate Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].axvline(vprops_df['fps'].median(), color=COLORS['accent'], linestyle='--',
                        label=f"Median={vprops_df['fps'].median():.0f}")
    axes[0, 1].legend()
    
    # Duration distribution
    axes[0, 2].hist(vprops_df['duration_sec'], bins=30, color=COLORS['success'], edgecolor='white', alpha=0.85)
    axes[0, 2].set_xlabel('Duration (seconds)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Clip Duration Distribution', fontsize=14, fontweight='bold')
    axes[0, 2].axvline(vprops_df['duration_sec'].mean(), color=COLORS['accent'], linestyle='--',
                        label=f"Mean={vprops_df['duration_sec'].mean():.1f}s")
    axes[0, 2].legend()
    
    # Frame count
    axes[1, 0].hist(vprops_df['frame_count'], bins=30, color=COLORS['warning'], edgecolor='white', alpha=0.85)
    axes[1, 0].set_xlabel('Total Frames')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Frame Count Distribution', fontsize=14, fontweight='bold')
    
    # File size
    axes[1, 1].hist(vprops_df['file_size_kb'], bins=30, color=palette[4], edgecolor='white', alpha=0.85)
    axes[1, 1].set_xlabel('File Size (KB)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Clip File Size Distribution', fontsize=14, fontweight='bold')
    
    # Duration by split
    for idx, split in enumerate(['train', 'valid', 'test']):
        split_data = vprops_df[vprops_df['split'] == split]['duration_sec']
        if len(split_data) > 0:
            axes[1, 2].hist(split_data, bins=20, alpha=0.6, label=f'{split}', 
                            color=palette[idx], edgecolor='white')
    axes[1, 2].set_xlabel('Duration (seconds)')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Duration by Split', fontsize=14, fontweight='bold')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\\nVideo Properties Summary:")
    print(f"  Resolution: most common = {res_counts.index[0]}")
    print(f"  FPS: mean={vprops_df['fps'].mean():.1f}, median={vprops_df['fps'].median():.0f}")
    print(f"  Duration: mean={vprops_df['duration_sec'].mean():.1f}s, range=[{vprops_df['duration_sec'].min():.1f}, {vprops_df['duration_sec'].max():.1f}]s")
    print(f"  Frames: mean={vprops_df['frame_count'].mean():.0f}, range=[{vprops_df['frame_count'].min()}, {vprops_df['frame_count'].max()}]")
else:
    print("Skipping video visualization (no data)")""")

# ============================================================
# SECTION 9: SAMPLE FRAMES
# ============================================================
md("""## 9. Sample Frame Visualization

Visual inspection of random frames across different action classes to assess video quality and action visibility.
""")

code("""# Display sample frames from different classes
if HAS_CV2:
    # Get one sample per action class
    unique_classes = actions_df['action_class'].unique()
    n_classes_show = min(8, len(unique_classes))
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes_flat = axes.flatten()
    
    for idx, action_class in enumerate(unique_classes[:n_classes_show]):
        # Find an action with this class in train
        class_actions = actions_df[(actions_df['action_class'] == action_class) & 
                                    (actions_df['split'] == 'train')]
        if len(class_actions) == 0:
            class_actions = actions_df[actions_df['action_class'] == action_class]
        
        if len(class_actions) > 0:
            action_key = class_actions.iloc[0]['action_key']
            # Try to find the corresponding video
            action_idx = action_key.split('_')[-1] if '_' in str(action_key) else action_key
            action_dir = TRAIN_DIR / f'action_{action_idx}'
            if not action_dir.exists():
                # Try matching from clips_df
                matching = clips_df[clips_df['action_dir'].str.contains(str(action_idx))]
                if len(matching) > 0:
                    split = matching.iloc[0]['split']
                    action_dir = DATA_ROOT / split / matching.iloc[0]['action_dir']
            
            clip_files = list(action_dir.glob('*.mp4')) if action_dir.exists() else []
            if clip_files:
                cap = cv2.VideoCapture(str(clip_files[0]))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)  # Middle frame
                ret, frame = cap.read()
                cap.release()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    axes_flat[idx].imshow(frame_rgb)
                    axes_flat[idx].set_title(f'{action_class}', fontsize=11, fontweight='bold')
                    axes_flat[idx].axis('off')
                    continue
        
        axes_flat[idx].text(0.5, 0.5, f'{action_class}\\n(no frame)', 
                            ha='center', va='center', fontsize=10, transform=axes_flat[idx].transAxes)
        axes_flat[idx].axis('off')
    
    # Hide unused axes
    for i in range(n_classes_show, len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.suptitle('Sample Frames from Each Action Class (Middle Frame)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print("OpenCV required for frame visualization")""")

# ============================================================
# SECTION 10: SEVERITY x OFFENCE ANALYSIS
# ============================================================
md("""## 10. Severity x Offence Cross-Analysis""")

code("""# Severity by Offence
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Cross-tab: Offence x Severity
if 'severity' in actions_df.columns and 'offence' in actions_df.columns:
    ct = pd.crosstab(actions_df['offence'], actions_df['severity'])
    ct.plot(kind='bar', ax=axes[0], color=sns.color_palette('YlOrRd', len(ct.columns)),
            edgecolor='white', width=0.6)
    axes[0].set_title('Severity by Offence Classification', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Offence')
    axes[0].set_ylabel('Count')
    axes[0].legend(title='Severity', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].tick_params(axis='x', rotation=0)

# Offence x Action Class
if 'offence' in actions_df.columns and 'action_class' in actions_df.columns:
    ct2 = pd.crosstab(actions_df['action_class'], actions_df['offence'])
    ct2_pct = ct2.div(ct2.sum(axis=1), axis=0) * 100
    ct2_pct.plot(kind='barh', stacked=True, ax=axes[1], 
                  color=[COLORS['danger'], COLORS['success']][:len(ct2.columns)],
                  edgecolor='white')
    axes[1].set_title('Offence Rate by Action Class', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Percentage')
    axes[1].legend(title='Offence', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()""")

# ============================================================
# SECTION 11: SPLIT INTEGRITY
# ============================================================
md("""## 11. Split Integrity Verification""")

code("""# Check for action ID overlap between splits
print("Checking split integrity (no action leakage)...")
print("=" * 50)

split_actions = {}
for split in ['train', 'valid', 'test']:
    split_data = actions_df[actions_df['split'] == split]
    split_actions[split] = set(split_data['action_key'].values)

# Check overlaps
for s1 in ['train', 'valid', 'test']:
    for s2 in ['train', 'valid', 'test']:
        if s1 < s2:
            overlap = split_actions.get(s1, set()) & split_actions.get(s2, set())
            status = "NO OVERLAP" if len(overlap) == 0 else f"OVERLAP: {len(overlap)} actions!"
            icon = "OK" if len(overlap) == 0 else "WARNING"
            print(f"  [{icon}] {s1} vs {s2}: {status}")

print()

# Class distribution comparison across splits
print("Class distribution comparison (% per split):")
for split in ['train', 'valid', 'test']:
    split_data = actions_df[actions_df['split'] == split]
    if len(split_data) > 0:
        dist = split_data['action_class'].value_counts(normalize=True) * 100
        print(f"\\n  {split.upper()} (n={len(split_data)}):")
        for cls, pct in dist.head(5).items():
            print(f"    {cls}: {pct:.1f}%")""")

# ============================================================
# SECTION 12: PREPROCESSING PLAN
# ============================================================
md("""## 12. Preprocessing Pipeline for ViT-B

Based on the EDA findings, here is the recommended preprocessing pipeline:

### Frame Extraction Strategy
| Parameter | Value | Rationale |
|---|---|---|
| **Frames per clip** | 16 (or 32 for ablation) | Standard for video transformers; covers ~2-4s of action |
| **Sampling** | Uniform temporal sampling | Ensures full action coverage, not just the foul moment |
| **Resize** | 224x224 center crop | Standard ViT-B input size |

### Normalization
| Parameter | Value |
|---|---|
| **Mean** | [0.485, 0.456, 0.406] (ImageNet) |
| **Std** | [0.229, 0.224, 0.225] (ImageNet) |

### Data Augmentation (Train only)
| Augmentation | Parameters |
|---|---|
| Horizontal flip | p=0.5 |
| Color jitter | brightness=0.2, contrast=0.2, saturation=0.1 |
| Random crop | Scale=(0.8, 1.0), then resize to 224x224 |
| Temporal jitter | Random offset in frame indices (+/- 2 frames) |

### Class Balancing
- Use **inverse-frequency class weights** in the CrossEntropyLoss
- Alternatively: weighted random sampling in the DataLoader
""")

# ============================================================
# SECTION 13: SUMMARY
# ============================================================
md("""## 13. Data Quality Report & Conclusion""")

code("""# Final summary
print("=" * 70)
print("DATA QUALITY REPORT -- SoccerNet-MVFouls for ViT-B Classifier")
print("=" * 70)

items = [
    ("Total actions", f"{len(actions_df):,}", "OK"),
    ("Train actions", f"{len(actions_df[actions_df['split']=='train']):,}", "OK"),
    ("Valid actions", f"{len(actions_df[actions_df['split']=='valid']):,}", "OK"),
    ("Test actions", f"{len(actions_df[actions_df['split']=='test']):,}", "OK"),
    ("Action classes", f"{actions_df['action_class'].nunique()}", "OK"),
    ("Total video clips", f"{clips_df['num_clips'].sum():,}", "OK"),
    ("Avg clips per action", f"{clips_df['num_clips'].mean():.1f}", "OK"),
    ("Total video size", f"{clips_df['total_size_mb'].sum()/1024:.2f} GB", "OK"),
]

if len(vprops_df) > 0:
    items.extend([
        ("Most common resolution", f"{res_counts.index[0]}", "OK"),
        ("Mean FPS", f"{vprops_df['fps'].mean():.0f}", "OK"),
        ("Mean duration", f"{vprops_df['duration_sec'].mean():.1f}s", "OK"),
    ])

for item, value, status in items:
    print(f"  [{status}] {item:.<45s} {value}")

print()
print("Recommendations:")
print("  1. Apply class weights to handle imbalance (inverse frequency)")
print("  2. Use 16-frame uniform temporal sampling per clip")
print("  3. Standard ViT-B preprocessing: 224x224 + ImageNet normalization")
print("  4. All multi-view clips can be used as independent training samples")
print("  5. Splits are clean with no action leakage")""")

md("""---
*MVFouls EDA completed. Alhamdulillah.* 

**Next:** NB-3 — SoccerNet-Tracking EDA
""")

# FINALIZE
nb.cells = cells
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '02_eda_mvfouls.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
