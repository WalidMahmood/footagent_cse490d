# Phase 1 - Tracking Foundation

## Goal
Deliver a robust tracking pipeline that generates consistent world-state output from a match clip.

## Exit Criteria
- [ ] Tracking agent rewritten and bug-fixed
- [ ] Stable IDs for players over 30-second sequence
- [ ] Team assignment is temporally stable
- [ ] Pose and depth integrated per player
- [ ] Ball tracking integrated
- [ ] Top-down pitch visualization generated
- [ ] Output schema validated for downstream phases

## Component Checklists
- [Tracking Core](components/tracking-core.md)
- [Pose and Depth](components/pose-depth.md)
- [Homography and Field Mapping](components/homography-field-mapping.md)
- [Ball and Shot Context](components/ball-shot-context.md)
- [Tracking Validation](components/tracking-validation.md)

## Notes
No orchestration or agent triggers in this phase beyond local diagnostics.
