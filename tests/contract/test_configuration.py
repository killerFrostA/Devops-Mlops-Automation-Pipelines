from pathlib import Path

import yaml

from src.agents.registry import AGENTS
from src.contracts.topics import TOPICS

ROOT = Path(__file__).resolve().parents[2]


def test_topic_configuration_matches_contracts_and_bootstrap() -> None:
    configuration = yaml.safe_load((ROOT / "configs/topics.yaml").read_text())
    assert set(configuration["topics"]) == set(TOPICS)
    bootstrap = (ROOT / "infrastructure/kafka/create-topics.sh").read_text()
    assert all(topic in bootstrap for topic in TOPICS)


def test_compose_agent_processes_have_exactly_six_known_identities() -> None:
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text())
    identities = {
        service["environment"]["PIDS_SERVICE_NAME"]
        for service in compose["services"].values()
        if "agents" in service.get("profiles", [])
    }
    assert identities == {identity.value for identity in AGENTS}


def test_referenced_local_compose_files_exist() -> None:
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text())
    for service in compose["services"].values():
        for mount in service.get("volumes", []):
            source = mount.split(":", maxsplit=1)[0]
            if source.startswith("./"):
                assert (ROOT / source).exists()
