# Security reporting

Do not put vulnerabilities, secrets or sensitive incident data into a public issue. The project
owner must configure a private reporting channel before external distribution. No email address
or organization policy is assumed in this scaffold.

The HTTP API is local development software. It provides no authentication provider, JWT validation,
mTLS, Kafka ACL enforcement or production approval service. The runtime rejects staging/production
configuration; policy always returns BLOCK and approval resolution always returns HTTP 501.

Use synthetic fixtures. Keep credentials in local ignored environment files or a secret manager.
Scan source, dependencies and images before release. Treat LLM outputs, telemetry, retrieved
documents and event payloads as untrusted input.
