"""Evaluate a registered project/model against a local evidence batch."""

import argparse
import asyncio
from pathlib import Path

from src.agents.mlops_lifecycle.evaluation.engine import ProjectEvaluationService
from src.contracts.mlops_projects import EvaluationBatch, ProjectManifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--provider", choices=("openai", "groq"))
    parser.add_argument("--model", help="Provider model supporting strict structured outputs")
    args = parser.parse_args()
    manifest = ProjectManifest.model_validate_json(args.manifest.read_text(encoding="utf-8-sig"))
    batch = EvaluationBatch.model_validate_json(args.batch.read_text(encoding="utf-8-sig"))
    judge = None
    if manifest.task_type == "text_generation":
        from src.agents.mlops_lifecycle.evaluation.llm_judge import (
            LLMJudgeSettings,
            ResponsesAPIJudge,
        )

        settings = LLMJudgeSettings()
        provider = args.provider or settings.agent6_llm_provider
        model = args.model or settings.agent6_llm_model
        if not provider or not model.strip():
            parser.error(
                "Text evaluation requires AGENT6_LLM_PROVIDER and AGENT6_LLM_MODEL "
                "in the Agent 6 .env, or --provider and --model"
            )
        judge = ResponsesAPIJudge.from_env(provider, model, settings=settings)
    elif args.provider or args.model:
        parser.error("--provider and --model apply only to text evaluation")
    service = ProjectEvaluationService(text_judge=judge)
    service.register_project(manifest)
    report = asyncio.run(service.evaluate(batch))
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
