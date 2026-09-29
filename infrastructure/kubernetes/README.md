# Kubernetes template

Build/load `devops-mlops-automation-pipelines:local` into an isolated local cluster, then review the rendered manifest:

```bash
kubectl kustomize infrastructure/kubernetes/overlays/local
kubectl apply -k infrastructure/kubernetes/overlays/local
```

The base deploys only the skeleton API, without ingress or cluster-write permissions. Its service
account token is not mounted. Health probes mean skeleton HTTP readiness only.
The NetworkPolicy requires a cluster CNI that enforces policies and allows only same-namespace
ingress. Configure operator access deliberately; access behavior can vary with CNI/port-forwarding.

Add real six-agent workers, orchestrator and Context/HITL servers only after their entry points exist.
Before staging/production: image digests, registry access, secret manager, identities/mTLS, topic
ACLs, namespaced tool RBAC, resource limits, disruption/backup plans and real dependency readiness
must be defined. No staging/production overlay is provided while the runtime is a skeleton.
