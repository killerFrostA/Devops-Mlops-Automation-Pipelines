-- Proposed initial model. Review and migrate; no application currently executes this file.
CREATE TABLE incidents (
    incident_id text PRIMARY KEY,
    service_id text NOT NULL,
    environment text NOT NULL,
    status text NOT NULL,
    priority text NOT NULL CHECK (priority IN ('P0', 'P1', 'P2', 'P3')),
    severity text NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    context_version bigint NOT NULL DEFAULT 1 CHECK (context_version > 0),
    opened_at timestamptz NOT NULL DEFAULT now(),
    closed_at timestamptz
);

CREATE TABLE agent_results (
    result_id text PRIMARY KEY,
    incident_id text NOT NULL REFERENCES incidents(incident_id),
    task_id text NOT NULL,
    agent_id text NOT NULL,
    result_type text NOT NULL,
    confidence double precision CHECK (confidence BETWEEN 0 AND 1),
    payload_json jsonb NOT NULL,
    idempotency_key text NOT NULL UNIQUE,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX agent_results_incident_idx ON agent_results (incident_id, created_at);

CREATE TABLE evidence_metadata (
    evidence_id text PRIMARY KEY,
    incident_id text NOT NULL REFERENCES incidents(incident_id),
    source text NOT NULL,
    object_uri text NOT NULL,
    checksum_sha256 text NOT NULL CHECK (checksum_sha256 ~ '^[0-9a-f]{64}$'),
    observed_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (incident_id, source, checksum_sha256)
);

CREATE TABLE actions (
    action_id text NOT NULL,
    action_version bigint NOT NULL CHECK (action_version > 0),
    incident_id text NOT NULL REFERENCES incidents(incident_id),
    intent_json jsonb NOT NULL,
    fingerprint text NOT NULL,
    policy_result text NOT NULL CHECK (policy_result IN ('AUTO', 'APPROVAL_REQUIRED', 'BLOCK')),
    execution_status text NOT NULL DEFAULT 'PROPOSED',
    idempotency_key text NOT NULL UNIQUE,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (action_id, action_version)
);

CREATE TABLE approvals (
    approval_id text PRIMARY KEY,
    action_id text NOT NULL,
    action_version bigint NOT NULL,
    fingerprint text NOT NULL,
    state text NOT NULL CHECK (state IN ('WAITING', 'APPROVED', 'REJECTED', 'EXPIRED')),
    approver_id text,
    reason text NOT NULL,
    token_hash text,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL,
    resolved_at timestamptz,
    FOREIGN KEY (action_id, action_version) REFERENCES actions(action_id, action_version),
    CHECK (expires_at > created_at)
);

CREATE TABLE audit_records (
    audit_id text PRIMARY KEY,
    incident_id text NOT NULL REFERENCES incidents(incident_id),
    actor_id text NOT NULL,
    record_type text NOT NULL,
    evidence_json jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE event_outbox (
    message_id text PRIMARY KEY,
    correlation_id text NOT NULL,
    topic text NOT NULL,
    envelope_json jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    published_at timestamptz
);
CREATE INDEX event_outbox_pending_idx ON event_outbox (created_at) WHERE published_at IS NULL;
