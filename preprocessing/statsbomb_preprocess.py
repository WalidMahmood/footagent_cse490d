"""
StatsBomb 360 Preprocessing — Feature Extraction for Temporal GAT
Bismillah
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist


# ============================================================
# FEATURE SELECTION — What we keep and why
# ============================================================
KEPT_FEATURES = {
    'player_x': 'Graph node X position (normalized to [0,1])',
    'player_y': 'Graph node Y position (normalized to [0,1])',
    'teammate': 'Team encoding for edge types (same-team vs cross-team)',
    'actor': 'Flag for the event performer (attention anchor)',
    'keeper': 'Goalkeeper flag (special node role)',
    'xg': 'Regression label — expected goals value',
    'event_type': 'Window anchor — we build windows around shots',
    'timestamp': 'Sequence ordering within temporal window',
    'match_id': 'Split key — split by competition to prevent leakage',
}

DROPPED_FEATURES = {
    'team_names': 'Not a model feature — only for human readability',
    'match_score': 'Would leak outcome information',
    'competition_name': 'Used only for split strategy, not model input',
    'player_name': 'Not a model feature',
    'jersey_number': 'Not reliably available in 360 data',
    'manager': 'Irrelevant to shot prediction',
    'stadium': 'Irrelevant to shot prediction',
    'referee': 'Irrelevant to shot prediction',
}

# Constants from EDA findings
PITCH_X_MAX = 120.0
PITCH_Y_MAX = 80.0
EDGE_THRESHOLD_M = 17.5  # From EDA: connects ~40-50% of pairs
WINDOW_SECONDS = 10.0    # From EDA: 10s captures 3-8 events


def load_usable_matches(data_root: Path):
    """Filter to matches with BOTH events AND 360 data (EDA: ~323 matches)."""
    events_dir = data_root / 'statsbomb' / 'open-data' / 'data' / 'events'
    threesixty_dir = data_root / 'statsbomb' / 'open-data' / 'data' / 'three-sixty'
    
    event_ids = {int(f.stem) for f in events_dir.glob('*.json')}
    threesixty_ids = {int(f.stem) for f in threesixty_dir.glob('*.json')}
    usable_ids = sorted(event_ids & threesixty_ids)
    
    return usable_ids, events_dir, threesixty_dir


def extract_shots_with_windows(match_id, events_dir, window_sec=WINDOW_SECONDS):
    """Extract shot events and their temporal windows from a match."""
    events_path = events_dir / f'{match_id}.json'
    with open(events_path, 'r', encoding='utf-8') as f:
        events = json.load(f)
    
    shots = []
    for evt in events:
        if evt.get('type', {}).get('name') == 'Shot':
            shot_info = evt.get('shot', {})
            xg = shot_info.get('statsbomb_xg')
            if xg is None:
                continue
            
            shot_ts = _parse_timestamp(evt.get('timestamp', '00:00:00.000'))
            window_start = max(0, shot_ts - window_sec)
            
            # Collect events in window
            window_events = []
            for we in events:
                we_ts = _parse_timestamp(we.get('timestamp', '00:00:00.000'))
                if window_start <= we_ts <= shot_ts:
                    loc = we.get('location', [None, None])
                    window_events.append({
                        'event_type': we.get('type', {}).get('name', 'Unknown'),
                        'timestamp': we_ts,
                        'event_x': loc[0] if loc else None,
                        'event_y': loc[1] if loc else None,
                    })
            
            shots.append({
                'match_id': match_id,
                'event_id': evt.get('id'),
                'xg': xg,
                'outcome': shot_info.get('outcome', {}).get('name', 'Unknown'),
                'shot_timestamp': shot_ts,
                'window_events': window_events,
                'n_window_events': len(window_events),
            })
    
    return shots


def extract_freeze_frames(match_id, threesixty_dir):
    """Extract freeze-frame player positions for a match."""
    ff_path = threesixty_dir / f'{match_id}.json'
    with open(ff_path, 'r', encoding='utf-8') as f:
        frames = json.load(f)
    
    result = []
    for frame in frames:
        event_id = frame.get('event_uuid')
        ff = frame.get('freeze_frame', [])
        players = []
        for p in ff:
            loc = p.get('location', [None, None])
            if loc and len(loc) == 2 and loc[0] is not None:
                players.append({
                    'event_id': event_id,
                    'x_raw': loc[0],
                    'y_raw': loc[1],
                    'teammate': p.get('teammate', False),
                    'actor': p.get('actor', False),
                    'keeper': p.get('keeper', False),
                })
        result.extend(players)
    
    return result


def normalize_positions(df):
    """Clip out-of-range coords and normalize to [0,1]. EDA found 26K outliers."""
    df = df.copy()
    # Before stats
    before = {
        'x_min': df['x_raw'].min(), 'x_max': df['x_raw'].max(),
        'y_min': df['y_raw'].min(), 'y_max': df['y_raw'].max(),
        'n_outliers': ((df['x_raw'] < 0) | (df['x_raw'] > PITCH_X_MAX) | 
                       (df['y_raw'] < 0) | (df['y_raw'] > PITCH_Y_MAX)).sum(),
    }
    
    # Clip and normalize
    df['x'] = df['x_raw'].clip(0, PITCH_X_MAX) / PITCH_X_MAX
    df['y'] = df['y_raw'].clip(0, PITCH_Y_MAX) / PITCH_Y_MAX
    
    # After stats
    after = {
        'x_min': df['x'].min(), 'x_max': df['x'].max(),
        'y_min': df['y'].min(), 'y_max': df['y'].max(),
        'n_outliers': 0,
    }
    
    return df, before, after


def build_edges(positions, threshold=EDGE_THRESHOLD_M):
    """Build graph edges based on distance threshold. EDA recommended 15-20m."""
    if len(positions) < 2:
        return np.array([]).reshape(0, 2)
    
    pts = positions[['x_raw', 'y_raw']].values
    dists = cdist(pts, pts)
    edges = np.argwhere((dists > 0) & (dists <= threshold))
    
    return edges


def _parse_timestamp(ts_str):
    """Parse StatsBomb timestamp string to seconds."""
    parts = ts_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def run_full_pipeline(data_root, max_matches=None):
    """Run the complete StatsBomb preprocessing pipeline."""
    data_root = Path(data_root)
    usable_ids, events_dir, threesixty_dir = load_usable_matches(data_root)
    
    if max_matches:
        usable_ids = usable_ids[:max_matches]
    
    print(f"Processing {len(usable_ids)} usable matches...")
    
    all_shots = []
    all_positions = []
    
    for i, mid in enumerate(usable_ids):
        try:
            shots = extract_shots_with_windows(mid, events_dir)
            positions = extract_freeze_frames(mid, threesixty_dir)
            all_shots.extend(shots)
            all_positions.extend(positions)
        except Exception as e:
            pass
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(usable_ids)} matches...")
    
    shots_df = pd.DataFrame(all_shots)
    positions_df = pd.DataFrame(all_positions)
    
    # Normalize positions
    if len(positions_df) > 0:
        positions_df, before_stats, after_stats = normalize_positions(positions_df)
    else:
        before_stats, after_stats = {}, {}
    
    print(f"Done: {len(shots_df)} shots, {len(positions_df)} player positions")
    
    return shots_df, positions_df, before_stats, after_stats
