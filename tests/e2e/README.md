# Deployed acceptance scenarios

No end-to-end test is claimed while workers, RPC servers and action adapters are unimplemented.
Use `experiments/scenarios.yaml` when the sandbox stack exists. Cover an incident, release gate,
drift/retraining path, approved/rejected/expired HITL, stale intent, duplicate event and dependency
outage. Verify authoritative context, audit records and independent recovery, not just HTTP status.
