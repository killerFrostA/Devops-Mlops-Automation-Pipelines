# ADR 0003: Kafka workflow, bounded RPC and centralized HITL

Status: accepted for scaffolding

Use Kafka for durable asynchronous work/results and gRPC for short typed queries/state operations.
The v5 approval path is Dashboard → Orchestrator → HITL, followed by an orchestrator-published
continuation. The dashboard never calls execution tools.

Use at-least-once delivery with durable idempotent action effects. Approval binds exact action
intent, scope, version and expiry. The skeleton blocks all actions until these controls exist.
