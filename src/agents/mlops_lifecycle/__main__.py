"""Assess a local JSON window: python -m src.agents.mlops_lifecycle --input FILE."""

import argparse
import asyncio
from pathlib import Path

from src.agents.mlops_lifecycle.health import RuleBasedModelHealthAssessor
from src.agents.mlops_lifecycle.service import MLOpsLifecycleAgent
from src.contracts.mlops import ModelHealthInput, ModelHealthPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 6: local model-health assessment")
    parser.add_argument("--input", required=True, type=Path, help="ModelHealthInput JSON file")
    parser.add_argument("--policy", type=Path, help="Optional ModelHealthPolicy JSON file")
    args = parser.parse_args()
    try:
        request = ModelHealthInput.model_validate_json(args.input.read_text(encoding="utf-8"))
        policy = (
            ModelHealthPolicy.model_validate_json(args.policy.read_text(encoding="utf-8"))
            if args.policy
            else ModelHealthPolicy()
        )
        agent = MLOpsLifecycleAgent(RuleBasedModelHealthAssessor(policy))
        report = asyncio.run(agent.assess_model_health(request))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
