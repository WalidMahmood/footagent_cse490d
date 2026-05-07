"""
Build PyG-compatible GAT dataset from StatsBomb 360 data.
Bismillah.

Produces: data/processed/gat_dataset/ with train/val/test .pt files
Each sample is a graph snapshot (one freeze-frame around a shot):
  - x: [N, 5] node features (x_norm, y_norm, teammate, actor, keeper)
  - edge_index: [2, E] edges from distance threshold
  - y: scalar xG label
"""
import json
import sys
import hashlib
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.statsbomb_preprocess import (
    load_usable_matches, _parse_timestamp,
    PITCH_X_MAX, PITCH_Y_MAX, EDGE_THRESHOLD_M, WINDOW_SECONDS,
)

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = Path('F:/footagent/data')
OUTPUT_DIR = Path('F:/footagent/data/processed/gat_dataset')
MATCHES_DIR = DATA_ROOT / 'statsbomb' / 'open-data' / 'data' / 'matches'


def get_competition_for_match(match_id: int) -> str:
    """Resolve competition name for a match ID (used for split policy)."""
    for comp_dir in MATCHES_DIR.iterdir():
        if not comp_dir.is_dir():
            continue
        for season_file in comp_dir.glob('*.json'):
            try:
                with open(season_file, 'r', encoding='utf-8') as f:
                    matches = json.load(f)
                for m in matches:
                    if m.get('match_id') == match_id:
                        return m.get('competition', {}).get('competition_name', 'Unknown')
            except:
                pass
    return 'Unknown'


def build_competition_map(usable_ids):
    """Build match_id -> competition mapping for all usable matches."""
    print("Building competition map for split policy...")
    comp_map = {}
    all_matches = {}

    for comp_dir in MATCHES_DIR.iterdir():
        if not comp_dir.is_dir():
            continue
        for season_file in comp_dir.glob('*.json'):
            try:
                with open(season_file, 'r', encoding='utf-8') as f:
                    matches = json.load(f)
                for m in matches:
                    mid = m.get('match_id')
                    comp = m.get('competition', {}).get('competition_name', 'Unknown')
                    all_matches[mid] = comp
            except:
                pass

    for mid in usable_ids:
        comp_map[mid] = all_matches.get(mid, 'Unknown')

    comps = {}
    for mid, c in comp_map.items():
        comps[c] = comps.get(c, 0) + 1
    print(f"  {len(comp_map)} matches mapped to {len(comps)} competitions:")
    for c, n in sorted(comps.items(), key=lambda x: -x[1]):
        print(f"    {c}: {n} matches")

    return comp_map


def assign_splits(comp_map):
    """
    Competition-based split to prevent data leakage.
    Strategy: hold out 1-2 smaller competitions for test, 1 for val, rest for train.
    Uses deterministic hash for reproducibility.
    """
    comps = {}
    for mid, c in comp_map.items():
        comps.setdefault(c, []).append(mid)

    # Sort competitions by size (ascending) for deterministic assignment
    sorted_comps = sorted(comps.items(), key=lambda x: len(x[1]))

    # Assign: smallest competition(s) to test, next to val, rest to train
    # But ensure test has enough samples (at least ~50 shots)
    test_comps, val_comps, train_comps = [], [], []
    test_count, val_count = 0, 0

    for comp_name, match_ids in sorted_comps:
        if test_count < 30:
            test_comps.append(comp_name)
            test_count += len(match_ids)
        elif val_count < 30:
            val_comps.append(comp_name)
            val_count += len(match_ids)
        else:
            train_comps.append(comp_name)

    splits = {}
    for comp_name, match_ids in comps.items():
        if comp_name in test_comps:
            split = 'test'
        elif comp_name in val_comps:
            split = 'val'
        else:
            split = 'train'
        for mid in match_ids:
            splits[mid] = split

    # Report
    for s in ['train', 'val', 'test']:
        n = sum(1 for v in splits.values() if v == s)
        print(f"  {s}: {n} matches")

    return splits


def build_graph_from_freeze_frame(freeze_frame, xg_value):
    """
    Convert a single freeze-frame into a PyG-compatible graph dict.

    Node features: [x_norm, y_norm, teammate, actor, keeper]
    Edges: distance-based threshold (17.5m on raw coords)
    Label: xG
    """
    players = []
    for p in freeze_frame:
        loc = p.get('location', [None, None])
        if loc and len(loc) == 2 and loc[0] is not None:
            players.append({
                'x_raw': loc[0],
                'y_raw': loc[1],
                'x_norm': max(0, min(loc[0], PITCH_X_MAX)) / PITCH_X_MAX,
                'y_norm': max(0, min(loc[1], PITCH_Y_MAX)) / PITCH_Y_MAX,
                'teammate': float(p.get('teammate', False)),
                'actor': float(p.get('actor', False)),
                'keeper': float(p.get('keeper', False)),
            })

    if len(players) < 2:
        return None

    # Node features: [x, y, teammate, actor, keeper]
    node_feats = torch.tensor(
        [[p['x_norm'], p['y_norm'], p['teammate'], p['actor'], p['keeper']]
         for p in players],
        dtype=torch.float32
    )

    # Edge construction: distance-based on RAW coordinates
    raw_pts = np.array([[p['x_raw'], p['y_raw']] for p in players])
    dists = cdist(raw_pts, raw_pts)
    src, dst = np.where((dists > 0) & (dists <= EDGE_THRESHOLD_M))
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    # Label
    y = torch.tensor([xg_value], dtype=torch.float32)

    return {'x': node_feats, 'edge_index': edge_index, 'y': y,
            'num_nodes': len(players)}


def process_match(match_id, events_dir, threesixty_dir):
    """Process one match: extract shot freeze-frames as graphs."""
    # Load events
    events_path = events_dir / f'{match_id}.json'
    with open(events_path, 'r', encoding='utf-8') as f:
        events = json.load(f)

    # Build event_id -> shot xG mapping
    shot_xg = {}
    for evt in events:
        if evt.get('type', {}).get('name') == 'Shot':
            xg = evt.get('shot', {}).get('statsbomb_xg')
            eid = evt.get('id')
            if xg is not None and eid:
                shot_xg[eid] = xg

    if not shot_xg:
        return []

    # Load 360 data
    ff_path = threesixty_dir / f'{match_id}.json'
    with open(ff_path, 'r', encoding='utf-8') as f:
        frames_360 = json.load(f)

    # Match freeze-frames to shots
    graphs = []
    for frame in frames_360:
        event_uuid = frame.get('event_uuid')
        if event_uuid in shot_xg:
            ff = frame.get('freeze_frame', [])
            graph = build_graph_from_freeze_frame(ff, shot_xg[event_uuid])
            if graph is not None:
                graph['match_id'] = match_id
                graph['event_id'] = event_uuid
                graphs.append(graph)

    return graphs


def build_and_save_dataset():
    """Main entry point: build full dataset and save splits to disk."""
    print("=" * 60)
    print("BUILDING GAT DATASET FROM STATSBOMB 360")
    print("=" * 60)

    # Step 1: Find usable matches
    usable_ids, events_dir, threesixty_dir = load_usable_matches(DATA_ROOT)
    print(f"\nStep 1: {len(usable_ids)} usable matches found")

    # Step 2: Build competition map for splitting
    comp_map = build_competition_map(usable_ids)

    # Step 3: Assign splits
    print("\nStep 3: Assigning train/val/test splits (by competition)...")
    splits = assign_splits(comp_map)

    # Step 4: Process all matches into graphs
    print("\nStep 4: Processing matches into graphs...")
    all_graphs = {'train': [], 'val': [], 'test': []}
    failed = 0

    for i, mid in enumerate(usable_ids):
        try:
            graphs = process_match(mid, events_dir, threesixty_dir)
            split = splits.get(mid, 'train')
            all_graphs[split].extend(graphs)
        except Exception as e:
            failed += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(usable_ids):
            total = sum(len(v) for v in all_graphs.values())
            print(f"  {i+1}/{len(usable_ids)} matches, {total} graphs, {failed} failed")

    # Step 5: Save to disk
    print(f"\nStep 5: Saving to {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for split_name, graphs in all_graphs.items():
        if not graphs:
            print(f"  WARNING: {split_name} has 0 graphs!")
            continue

        save_path = OUTPUT_DIR / f'{split_name}.pt'
        torch.save(graphs, save_path)
        size_mb = save_path.stat().st_size / (1024 * 1024)

        # Stats
        n_nodes = [g['num_nodes'] for g in graphs]
        n_edges = [g['edge_index'].shape[1] for g in graphs]
        xg_vals = [g['y'].item() for g in graphs]

        manifest[split_name] = {
            'n_graphs': len(graphs),
            'file': str(save_path),
            'size_mb': round(size_mb, 2),
            'avg_nodes': round(np.mean(n_nodes), 1),
            'avg_edges': round(np.mean(n_edges), 1),
            'xg_mean': round(np.mean(xg_vals), 4),
            'xg_std': round(np.std(xg_vals), 4),
        }
        print(f"  {split_name}: {len(graphs)} graphs, {size_mb:.1f} MB, "
              f"avg {np.mean(n_nodes):.0f} nodes, {np.mean(n_edges):.0f} edges")

    # Save manifest
    manifest_path = OUTPUT_DIR / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

    # Final summary
    total = sum(m['n_graphs'] for m in manifest.values())
    print(f"\nDONE: {total} total graphs across {len(manifest)} splits")
    print("Dataset ready for Temporal GAT training.")

    return manifest


if __name__ == '__main__':
    build_and_save_dataset()
