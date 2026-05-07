"""
MVFouls Preprocessing — Feature Extraction for ViT-B Classifier
Bismillah
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter


# ============================================================
# FEATURE SELECTION — What we keep and why
# ============================================================
KEPT_FEATURES = {
    'action_class': 'Classification target (Tackling, Standing, etc.)',
    'severity': 'Multi-task auxiliary label (1-4 scale)',
    'offence': 'Binary subtask label (Offence / No offence)',
    'video_frames': 'ViT-B input — 16 uniformly sampled frames at 224x224',
    'class_weights': 'Loss function balancing from inverse class frequency',
}

DROPPED_FEATURES = {
    'bodypart': 'Analyzed in EDA but not used as model input — low predictive signal',
    'contact': 'Too noisy for reliable classification',
    'multiple_offence': 'Very rare flag, adds noise',
    'handball': 'Extremely rare, not enough samples',
    'url_local': 'Match identifier only, not a feature',
    'camera_type': 'All views treated equally per MVFouls methodology',
    'replay_speed': 'Handled by uniform frame sampling strategy',
}

# Constants from EDA findings
TARGET_FRAMES = 16       # Uniform temporal sampling
TARGET_SIZE = (224, 224)  # ViT-B input size
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_annotations(data_root: Path):
    """Load and parse MVFouls annotations using fixed Actions key parsing."""
    mvfouls_dir = data_root / 'soccernet' / 'mvfouls'
    
    all_actions = []
    split_stats = {}
    
    for split_name in ['train', 'valid', 'test']:
        split_dir = mvfouls_dir / split_name
        ann_path = split_dir / 'annotations.json'
        
        if not ann_path.exists():
            continue
        
        with open(ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Fixed parsing: iterate over Actions dict (not top-level keys)
        actions_dict = data.get('Actions', {})
        n_actions = 0
        
        for action_id, action_data in actions_dict.items():
            action = {
                'action_id': action_id,
                'split': split_name,
                'action_class': action_data.get('Action class', 'Unknown'),
                'severity': action_data.get('Severity', ''),
                'offence': action_data.get('Offence', ''),
                'bodypart': action_data.get('Bodypart', ''),
                'n_clips': len(action_data.get('Clips', [])),
            }
            
            # Collect clip paths for video loading
            # Annotation URLs are like "Dataset/Train/action_0/clip_0"
            # Real paths are like split_dir/action_0/clip_0.mp4
            clips = action_data.get('Clips', [])
            clip_paths = []
            for clip in clips:
                url = clip.get('Url', '')
                if url:
                    # Extract action_X/clip_X from the URL
                    parts = url.replace('\\', '/').split('/')
                    # Find action_X and clip_X parts
                    action_part = None
                    clip_part = None
                    for p in parts:
                        if p.startswith('action_'):
                            action_part = p
                        if p.startswith('clip_'):
                            clip_part = p
                    if action_part and clip_part:
                        # Try with .mp4 extension
                        clip_file = clip_part if clip_part.endswith('.mp4') else clip_part + '.mp4'
                        clip_path = split_dir / action_part / clip_file
                        clip_paths.append(str(clip_path))
            
            action['clip_paths'] = clip_paths
            all_actions.append(action)
            n_actions += 1
        
        split_stats[split_name] = n_actions
    
    actions_df = pd.DataFrame(all_actions)
    return actions_df, split_stats


def compute_class_weights(actions_df):
    """Compute inverse-frequency class weights from EDA findings."""
    class_counts = actions_df['action_class'].value_counts()
    total = len(actions_df)
    n_classes = len(class_counts)
    
    weights = {}
    for cls, count in class_counts.items():
        weights[cls] = total / (n_classes * count)
    
    return weights, class_counts


def build_clip_index(actions_df):
    """Build the final clip index DataFrame with only relevant features."""
    kept_cols = ['action_id', 'split', 'action_class', 'severity', 'offence', 
                 'n_clips', 'clip_paths']
    
    clip_index = actions_df[kept_cols].copy()
    
    # Encode action_class as integer labels
    classes = sorted(clip_index['action_class'].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    clip_index['label'] = clip_index['action_class'].map(class_to_idx)
    
    return clip_index, class_to_idx


def validate_clips(clip_index, sample_n=50):
    """Validate that video files exist."""
    total = 0
    missing = 0
    sampled = clip_index.head(sample_n) if len(clip_index) > sample_n else clip_index
    
    for _, row in sampled.iterrows():
        for cp in row['clip_paths']:
            total += 1
            if not Path(cp).exists():
                missing += 1
    
    return {'total_checked': total, 'missing': missing, 'valid_pct': 100 * (1 - missing/max(total,1))}


def run_full_pipeline(data_root, validate=True):
    """Run the complete MVFouls preprocessing pipeline."""
    data_root = Path(data_root)
    
    print("Loading annotations...")
    actions_df, split_stats = load_annotations(data_root)
    print(f"  Loaded {len(actions_df)} actions: {split_stats}")
    
    print("Computing class weights...")
    weights, class_counts = compute_class_weights(actions_df)
    print(f"  {len(weights)} classes, max weight: {max(weights.values()):.2f}")
    
    print("Building clip index...")
    clip_index, class_to_idx = build_clip_index(actions_df)
    print(f"  {len(clip_index)} clips indexed, {len(class_to_idx)} classes")
    
    validation = None
    if validate:
        print("Validating clip files...")
        validation = validate_clips(clip_index)
        print(f"  {validation['valid_pct']:.1f}% valid ({validation['total_checked']} checked)")
    
    return clip_index, weights, class_to_idx, class_counts, validation
