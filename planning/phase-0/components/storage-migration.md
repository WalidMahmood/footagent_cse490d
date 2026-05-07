# Phase 0 Component - Storage Migration

| Task | Owner | ETA | Status notes |
|---|---|---|---|
| [ ] Select new data root with at least 200GB free space | Walid | 2026-04-22 | Required before pulling full raw video bundles |
| [ ] Define target path convention for datasets on new drive | Walid | 2026-04-22 | Example: D:/footagent-data |
| [ ] Move existing large folders to new data root | Walid | 2026-04-22 | Start with data/statsbomb and data/soccernet |
| [ ] Create filesystem link from project data folder to new location | Walid | 2026-04-22 | Use junction/symlink strategy on Windows |
| [ ] Re-run sanity checks after migration | Walid | 2026-04-22 | Verify statsbomb and soccernet paths resolve |
| [ ] Resume deferred large downloads only after migration validation | Walid | 2026-04-22 | Includes raw video bundles and optional tracking-2023 expansion |
