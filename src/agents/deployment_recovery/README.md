# Agent 4 · Deployment and recovery

Owner: Member 4. BO1, BO2, BO3. Consume exact orchestrator action requests/continuations. Before
writes, verify identity, schema, current context, policy, approval scope/signature/expiry and durable
idempotency. Capture before/after state and emit `deployment.action.completed`.

Use allowlisted Kubernetes/Helm/API operations. Request independent Monitoring verification;
execution success is not recovery. The backend port is unimplemented and skeleton policy blocks
all actions.
