# Agent 5 · Resource optimization

Owner: Member 5. DSO3 / BO4. Recommend bounded resource changes with confidence, observation
windows, SLO constraints and rollback conditions through `resource.recommendation.created`.

Start with sustained-load rules, then evaluate forecasting. Keep policy bounds authoritative and
show when scaling is temporary containment. Implement the planner port; no scaling backend or
autonomous execution is active in `service.py`.
