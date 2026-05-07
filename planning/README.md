# FootAgent Planning Hub

This folder is the execution source of truth for build, training, integration, and paper delivery.

## Planning Format
- Option: phase-0 through phase-6
- Structure: one phase file plus sub-checklists per component
- Task format: checkbox + owner + ETA + status notes

## Current Reality Snapshot
- [x] Repo skeleton created
- [x] Initial tracking prototype file exists
- [x] Basic requirements file exists
- [x] Test video exists at data/matches/test_clip.mp4
- [x] FFmpeg binaries exist
- [ ] CUDA-enabled PyTorch confirmed in environment
- [ ] Tracking pipeline validated end-to-end
- [x] StatsBomb data available locally (open-data clone)
- [ ] SoccerNet tracking fully downloaded (test split only confirmed)
- [x] SoccerNet MVFouls available locally (train/valid/test zips)
- [ ] Data root relocation completed (current disk capacity is insufficient for full raw video bundles)

## Phases
- [Phase 0: Environment and Data Foundations](phase-0/phase-0.md)
- [Phase 1: Tracking Foundation](phase-1/phase-1.md)
- [Phase 2: 3D Digital Twin UI](phase-2/phase-2.md)
- [Phase 3: Temporal GAT Off-Ball Credit](phase-3/phase-3.md)
- [Phase 4: VAR Agent](phase-4/phase-4.md)
- [Phase 5: Orchestration and Integration](phase-5/phase-5.md)
- [Phase 6: Evaluation and Paper](phase-6/phase-6.md)

## Master Execution Document
- [Implementation Plan](implementation-plan.md)
