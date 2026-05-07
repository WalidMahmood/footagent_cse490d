# FootAgent Detailed Implementation Plan

## Vision Lock
Build a real-time-capable football intelligence system from single broadcast video, with strict under-8GB VRAM operation and publishable research outputs.

## Architecture Lock
- Always-running tracking pipeline: detection, tracking, pose, depth, homography, ball tracking.
- Event-driven specialist agents: off-ball and VAR only on trigger.
- Unified multimodal model strategy: Qwen2.5-VL-7B for orchestrator and VAR explanation.
- Match memory as JSON during prototype phase.

## Dataset Lock
- StatsBomb 360 Open Data: Temporal GAT training and label construction.
- SoccerNet-MVFouls: ViT-B foul and severity training.
- SoccerNet-Tracking (primary): objective tracking metrics.
- SoccerNet-Tracking-2023 (optional): additional benchmark depth, not required for first paper milestone.
- SoccerNet-v3 clips or raw SoccerNet-Tracking videos: optional for extended end-to-end demonstration only.
- Optional: Metrica sample and Roboflow football data.

## Storage Policy
- Keep model training datasets local under a configurable data root.
- Use external drive relocation for very large archives and raw video bundles.
- Required for initial milestones: StatsBomb open-data, MVFouls, SoccerNet tracking test split.
- Defer large raw video pulls until data root migration is complete and free space is validated.

## Model Lock
### Direct-use models
- YOLOv11-small
- ByteTrack
- RTMPose-m
- Depth Anything V2 Small
- TrackNetV2
- Qwen2.5-VL-7B-Instruct

### Trainable models
- Temporal GAT (main contribution)
- ViT-B/16 foul classifier

## Delivery Gates
### Gate A (end of Phase 0)
- CUDA-enabled torch verified
- stable dependency set installed
- dataset access path validated
- data root migration plan validated (if local disk is below target capacity)

### Gate B (end of Phase 1)
- 30s clip fully processed with stable IDs
- top-down pitch visualization generated
- frame-wise world-state JSON output validated

### Gate C (end of Phase 2)
- FastAPI WebSocket streaming world state at target frame rate
- Three.js digital twin renders players, skeletons, ball

### Gate D (end of Phase 3)
- Temporal GAT trained with ablations
- Off-ball metrics reported with confidence intervals

### Gate E (end of Phase 4)
- ViT-B foul classifier trained and evaluated
- VAR explanation pipeline operational with Qwen2.5-VL

### Gate F (end of Phase 5)
- LangGraph routing works for normal, incident, and chance triggers
- Match memory and report generation integrated

### Gate G (end of Phase 6)
- Full benchmark pack completed
- Reproducible experiment logs and paper materials prepared

## Risks and Controls
- VRAM overflow risk: enforce model load/unload scheduler and profile every phase.
- Label noise risk in off-ball data: use deterministic label builder and analyst sanity sampling.
- Trigger instability risk: use temporal smoothing and threshold calibration runs.
- Scope creep risk: no phase jump until current gate criteria are passed.

## Execution Rhythm
- Daily: short build log and blockers.
- Per component: checklist completion before merge.
- Per phase: gate review with metrics, artifacts, and demo capture.

## Ownership Defaults
- Owner: Walid
- Reviewer: Copilot session review
- ETA fields in component files should be set to date or sprint code.
