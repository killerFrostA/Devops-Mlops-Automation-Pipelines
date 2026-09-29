# Agent 2 · Diagnosis and root cause analysis

Owner: Member 2. DSO2 / BO1, BO2. Consume bounded signals, evidence and recent changes; query
allowlisted domain tools and curated Qdrant knowledge; return ranked hypotheses, limitations and
candidate actions through `diagnosis.completed`.

Start with deterministic signatures. Add constrained reasoning after evidence quality and
evaluation exist. No infrastructure execution or full Context reads. Use the analyzer/retriever
ports and append evidence/results. `service.py` is currently NOT_IMPLEMENTED.
