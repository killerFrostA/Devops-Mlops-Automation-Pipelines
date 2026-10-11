"""Run Agent 1 autonomously: python -m src.agents.monitoring [--config FILE] [--cycles N]
[--verbose] [--graph FILE].

Ctrl+C stops the agent cleanly after the current cycle.
"""

import argparse
import asyncio
import contextlib
from pathlib import Path

from src.agents.monitoring.collectors.http import SourceUnavailableError
from src.agents.monitoring.collectors.loki import LokiClient
from src.agents.monitoring.collectors.prometheus import PrometheusClient, PrometheusCollector
from src.agents.monitoring.config import load_config
from src.agents.monitoring.decision import DecisionMaker, SnapshotReader
from src.agents.monitoring.detectors.logs import LogDetector, LogPatternDetector
from src.agents.monitoring.detectors.slo import SLODetector
from src.agents.monitoring.detectors.statistical import StatisticalDetector
from src.agents.monitoring.grouping import InMemoryIncidentStore, SymptomGrouper
from src.agents.monitoring.labelers.gemini import GeminiLabeler
from src.agents.monitoring.loop import ConsoleReporter, MonitoringLoop
from src.agents.monitoring.memory import AlertMemory
from src.agents.monitoring.reasoning import Reasoner, load_reasoning_config
from src.config import Settings
from src.platform.resilience import CircuitOpenError


async def main(config_path: Path, graph_path: Path, cycles: int | None, verbose: bool) -> None:
    config = load_config(config_path)
    reasoner = Reasoner(load_reasoning_config(graph_path)) if graph_path.exists() else None
    settings = Settings()
    labeler = (
        GeminiLabeler(settings.gemini_api_key, settings.gemini_model)
        if settings.gemini_api_key
        else None
    )
    memory = AlertMemory(config.memory.path, config.memory) if config.memory.enabled else None
    async with (
        PrometheusClient(settings.prometheus_url) as prometheus,
        LokiClient(settings.loki_url, containers=config.containers()) as loki,
    ):
        collector = PrometheusCollector(
            prometheus, profiles={s.service_id: s for s in config.services}
        )

        async def prometheus_up() -> bool:
            try:
                return await prometheus.query_value("vector(1)") == 1.0
            except (SourceUnavailableError, CircuitOpenError):
                return False

        loop = MonitoringLoop(
            config=config,
            slo=SLODetector(collector, config.slo),
            statistical=StatisticalDetector(collector, config.statistical),
            logs=LogDetector(loki, LogPatternDetector(labeler, config.logs)),
            grouper=SymptomGrouper(InMemoryIncidentStore(), config.incident_cooldown_seconds),
            reporter=ConsoleReporter(verbose=verbose),
            health_checks={"prometheus": prometheus_up},
            reasoner=reasoner,
            decider=DecisionMaker(
                config.decision, SnapshotReader(collector.collect), memory=memory
            ),
        )
        names = ", ".join(f"{s.service_id} ({s.mode})" for s in config.services)
        print(
            f"Agent 1 demarre : {names}, un cycle toutes les {config.interval_seconds:.0f}s, "
            f"logs {'actifs' if config.logs_enabled else 'desactives'}, "
            f"Gemini {'actif' if labeler else 'non configure'}, "
            f"raisonnement {'actif' if reasoner else 'inactif'}, "
            f"memoire {config.memory.path if memory else 'inactive'} (Ctrl+C pour arreter)"
        )
        stop = asyncio.Event()
        try:
            await loop.run_forever(stop, max_cycles=cycles)
        finally:
            if labeler:
                await labeler.aclose()
            if memory:
                memory.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/monitoring.yaml"))
    parser.add_argument("--graph", type=Path, default=Path("configs/dependencies.yaml"))
    parser.add_argument("--cycles", type=int, default=None, help="stop after N cycles")
    parser.add_argument(
        "--verbose", action="store_true", help="show every detector decision each cycle"
    )
    args = parser.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main(args.config, args.graph, args.cycles, args.verbose))
    print("Agent 1 arrete.")
