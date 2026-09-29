# Operator dashboard skeleton

`index.html` is a static layout outline: open locally to inspect navigation and screen structure.
It contains no simulated incidents, approvals, metrics or live API connection. Select the frontend
framework and identity integration during the dashboard implementation milestone.

Required screens: incident timeline/evidence, six-agent status, exact action/risk preview, pending
approvals with expiry, approved/rejected/expired decisions, execution results and verification.

The future UI sends human decisions to the orchestrator-facing API. Authentication determines actor
identity. It never calls Kubernetes, agents or Kafka directly. The scaffold reserves
`POST /api/v1/approvals/{approval_id}/resolve` and returns HTTP 501; incidents and streams are pending.
Document request/response schemas and operator RBAC before adding approval controls.
