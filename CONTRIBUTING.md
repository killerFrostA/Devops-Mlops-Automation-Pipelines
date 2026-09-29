# Contributing

Work in the package matching your ownership area. Implement one behavior per pull request and
update its contracts, documentation and meaningful tests together. Changes to shared contracts,
policy, HITL or action execution need review from the shared platform owner and affected agent owner.

1. Create `feat/<component>-<intent>` or `fix/<component>-<intent>` from the integration branch.
2. Keep infrastructure dependencies behind ports; never import another agent's implementation.
3. Use bounded task inputs, typed outputs and immutable evidence references.
4. Run `python scripts/check.py` in the project virtual environment.
5. Explain the resulting behavior, acceptance evidence and remaining implementation gaps in the PR.

Use conventional commits (`feat:`, `fix:`, `docs:`, `chore:`). Commit no production data, credentials,
binary model artifacts or generated gRPC code. A schema change must update examples and exported
schemas; incompatible changes require a new major contract directory and an ADR.

Before enabling action tools, implement authenticated identity, policy, exact-action approval,
expiry, stale-context handling, durable idempotency and independent verification together.
