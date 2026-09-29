# Platform observability

Implement correlated spans for tasks, Kafka publication/consumption, bounded RPCs and tool calls.
Track task latency/failure, consumer lag, Context latency, approval wait/expiry, policy distribution,
action verification and LLM tokens/latency. Use service/component labels with bounded cardinality;
incident IDs belong in traces/logs rather than unbounded metric labels.

Agent 1 observes target services. This folder covers the health of the agent platform itself.
Compose provisions Prometheus/Grafana and a datasource; business dashboards and instrumented
metrics remain implementation work.
