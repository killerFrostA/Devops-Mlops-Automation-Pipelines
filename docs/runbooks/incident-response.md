# Incident response implementation outline

This is an operational design template. No real incident handlers or tools are connected.

Record correlation ID, affected scope, environment, initial signals and immutable evidence.
Check platform health independently from target-service health. The orchestrator reloads context,
requests bounded diagnosis and records the candidate action plus policy assessment.

For HITL, inspect the exact target, action version, parameters, evidence, risk, expiry and rollback
plan. Approve or reject through the orchestrator UI only. Expired or changed actions need a fresh
decision. Capture pre-state, execute idempotently, then let Monitoring independently verify.

Do not mark recovery on execution success alone. Record failed/inconclusive checks and escalate
according to the future approved on-call policy. Preserve action/result/evidence linkage for review.

Fill in team contacts, incident severity targets, communication channels and service-specific
rollback/verification profiles before operational use.
