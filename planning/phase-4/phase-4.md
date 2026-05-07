# Phase 4 - VAR Agent

## Goal
Build and validate single-camera incident analysis with foul classification and explainable VAR output.

## Exit Criteria
- [ ] Incident trigger pipeline operational
- [ ] ViT-B classifier trained and evaluated on MVFouls
- [ ] Qwen2.5-VL decision explanation integrated
- [ ] End-to-end VAR inference tested on clips
- [ ] VRAM profile under budget during incident workflow

## Component Checklists
- [MVFouls Data Pipeline](components/mvfouls-data-pipeline.md)
- [ViT-B Training](components/vitb-training.md)
- [VAR Reasoning with Qwen2.5-VL](components/var-reasoning-qwen.md)
- [Incident Pipeline Integration](components/incident-pipeline-integration.md)
- [VAR Validation](components/var-validation.md)

## Notes
Keep the ViT classifier and LLM roles separated for controlled error analysis.
