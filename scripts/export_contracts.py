"""Export reproducible schemas or fail if committed contracts have drifted."""

import argparse
import json
from pathlib import Path

from src.contracts.actions import ActionIntent, ActionRequest
from src.contracts.base import Contract
from src.contracts.context import EvidenceReference, IncidentContext
from src.contracts.events import EventEnvelope
from src.contracts.mlops import ModelHealthInput, ModelHealthPolicy, ModelHealthReport
from src.contracts.mlops_observations import BaselineProfile, WindowBuildRequest
from src.contracts.tasks import AgentResult, AgentTask
from src.contracts.topics import TOPICS

ROOT = Path(__file__).resolve().parents[1]
MODELS: tuple[type[Contract], ...] = (
    EventEnvelope,
    AgentTask,
    AgentResult,
    ActionIntent,
    ActionRequest,
    EvidenceReference,
    IncidentContext,
    ModelHealthInput,
    ModelHealthPolicy,
    ModelHealthReport,
    BaselineProfile,
    WindowBuildRequest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = ROOT / "contracts/jsonschema/v1"
    if not args.check:
        target.mkdir(parents=True, exist_ok=True)
    models = {model.__name__: model for model in MODELS}
    models.update({model.__name__: model for _, model in TOPICS.values()})
    for name, model in sorted(models.items()):
        schema = model.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        if model is EventEnvelope:
            # JSON Schema must enforce the same topic/type/payload coupling as Pydantic.
            definitions = schema.setdefault("$defs", {})
            conditions = []
            schema["properties"]["topic"]["enum"] = sorted(TOPICS)
            for topic, (message_type, payload_model) in sorted(TOPICS.items()):
                payload_schema = payload_model.model_json_schema()
                definitions.update(payload_schema.pop("$defs", {}))
                definitions[payload_model.__name__] = payload_schema
                conditions.append(
                    {
                        "if": {"properties": {"topic": {"const": topic}}, "required": ["topic"]},
                        "then": {
                            "properties": {
                                "message_type": {"const": message_type},
                                "payload": {"$ref": f"#/$defs/{payload_model.__name__}"},
                            }
                        },
                    }
                )
            schema["allOf"] = conditions
        serialized = json.dumps(schema, indent=2, sort_keys=True) + "\n"
        path = target / f"{name}.schema.json"
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != serialized:
                raise SystemExit(f"Contract drift: regenerate {path.relative_to(ROOT)}")
        else:
            path.write_text(serialized, encoding="utf-8")
    print(f"{'Checked' if args.check else 'Exported'} {len(models)} JSON Schemas")


if __name__ == "__main__":
    main()
