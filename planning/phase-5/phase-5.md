# Phase 5 - Orchestration and Integration

## Goal
Connect all agents through LangGraph routing with stable event memory and report generation.

## Exit Criteria
- [ ] Trigger detector routes correctly by event type
- [ ] Agent loading and unloading policy works under VRAM limits
- [ ] Match memory logging stable and queryable
- [ ] Report agent produces coherent half-time and full-time summaries
- [ ] Full pipeline runs on one complete match

## Component Checklists
- [LangGraph Orchestration](components/langgraph-orchestration.md)
- [VRAM Scheduler](components/vram-scheduler.md)
- [Match Memory and Logging](components/match-memory-logging.md)
- [Report Agent](components/report-agent.md)
- [Integration Validation](components/integration-validation.md)

## Notes
Use deterministic routing tests before subjective quality review.
