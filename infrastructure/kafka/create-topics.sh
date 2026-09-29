#!/usr/bin/env bash
set -euo pipefail

topics=(
  monitoring.anomaly.detected monitoring.recovery.verified diagnosis.completed
  quality.release.assessed deployment.action.requested deployment.action.completed
  resource.recommendation.created mlops.model.health.assessed approval.requested
  approval.resolved workflow.continuation
)
for topic in "${topics[@]}"; do
  for suffix in '' '.retry' '.dlq'; do
    /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:19092 \
      --create --if-not-exists --topic "${topic}${suffix}" \
      --partitions 3 --replication-factor 1
  done
done
