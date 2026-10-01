"""Render versioned Jinja prompts from validated advisor context."""

from jinja2 import Environment, PackageLoader, StrictUndefined

from src.platform.orchestrator.decision.contracts import AdvisorRequest

PROMPT_VERSION = "candidate-ranking-v1"
_ENVIRONMENT = Environment(
    loader=PackageLoader("src.platform.orchestrator.decision.prompts"),
    undefined=StrictUndefined,
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_candidate_ranking_prompt(request: AdvisorRequest) -> tuple[str, str]:
    """Keep provider prompts separate from HTTP transport and workflow decisions."""
    context = {
        "environment": request.environment.value,
        "evidence_count": request.evidence_count,
        "hypotheses": [
            {"code": item.code, "probability": item.probability} for item in request.hypotheses
        ],
        "candidates": [
            {
                "action_id": item.action_id,
                "action_type": item.action_type.value,
                "action_fingerprint": item.action_fingerprint,
            }
            for item in request.candidates
        ],
    }
    system = _ENVIRONMENT.get_template("candidate_ranking_system_v1.j2").render(
        prompt_version=PROMPT_VERSION
    )
    user = _ENVIRONMENT.get_template("candidate_ranking_user_v1.j2").render(context=context)
    return system.strip(), user.strip()
