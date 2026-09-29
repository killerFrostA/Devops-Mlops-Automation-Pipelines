"""Assess prepared or observation-based local Agent 6 windows."""

import argparse
import asyncio
from pathlib import Path

from src.agents.mlops_lifecycle.health import RuleBasedModelHealthAssessor
from src.agents.mlops_lifecycle.service import MLOpsLifecycleAgent
from src.contracts.mlops import ModelHealthInput, ModelHealthPolicy
from src.contracts.mlops_observations import WindowBuildRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 6: local model-health assessment")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Prepared ModelHealthInput JSON file")
    source.add_argument("--records", type=Path, help="WindowBuildRequest JSON file")
    parser.add_argument("--policy", type=Path, help="Optional ModelHealthPolicy JSON file")
    args = parser.parse_args()
    try:
        policy = (
            ModelHealthPolicy.model_validate_json(args.policy.read_text(encoding="utf-8"))
            if args.policy
            else ModelHealthPolicy()
        )
        agent = MLOpsLifecycleAgent(RuleBasedModelHealthAssessor(policy))
        if args.input:
            request = ModelHealthInput.model_validate_json(args.input.read_text(encoding="utf-8"))
            report = asyncio.run(agent.assess_model_health(request))
        else:
            window = WindowBuildRequest.model_validate_json(
                args.records.read_text(encoding="utf-8")
            )
            report = asyncio.run(agent.assess_observation_window(window))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
