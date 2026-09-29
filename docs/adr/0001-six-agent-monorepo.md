# ADR 0001: Six agents in an interface-first monorepo

Status: accepted for scaffolding

The supplied architecture has six specialist owners and shared platform integration. Use one
Python `src` package initially, with separate agent packages and shared contract/port modules.
This supports independent development while keeping interface review and CI consistent.

The orchestrator, Context, HITL, dashboard and communication backbone are not extra agents.
Begin with one reusable development image; split deployment artifacts when dependency/runtime
needs are established. Production process isolation and identities still need implementation.
