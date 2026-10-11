"""Wiring of Agent 1: builds every component once from the configuration.

Used by the autonomous loop (`python -m src.agents.monitoring`) and by the task/event inbox
(`python -m src.agents.monitoring.inbox`), so both run exactly the same detectors.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from src.agents.monitoring.collectors.http import SourceUnavailableError
from src.agents.monitoring.collectors.loki import LokiClient
from src.agents.monitoring.collectors.prometheus import PrometheusClient, PrometheusCollector
from src.agents.monitoring.config import MonitoringConfig
from src.agents.monitoring.decision import DecisionMaker, SnapshotReader
from src.agents.monitoring.detectors.forecast import ChronosForecaster, ForecastDetector
from src.agents.monitoring.detectors.logs import LogDetector, LogPatternDetector
from src.agents.monitoring.detectors.slo import SLODetector
from src.agents.monitoring.detectors.statistical import StatisticalDetector
from src.agents.monitoring.events import AnomalyEmitter, JsonlEventPublisher
from src.agents.monitoring.grouping import InMemoryIncidentStore, SymptomGrouper
from src.agents.monitoring.labelers.gemini import GeminiLabeler
from src.agents.monitoring.loop import ConsoleReporter, MonitoringLoop
from src.agents.monitoring.memory import AlertMemory
from src.agents.monitoring.reasoning import Reasoner, load_reasoning_config
from src.agents.monitoring.recovery import RecoveryWatcher, Verifier
from src.agents.monitoring.service import MonitoringAgent, OneShotDetector
from src.config import Settings
from src.platform.resilience import CircuitOpenError


@dataclass(frozen=True, slots=True)
class Runtime:
    config: MonitoringConfig
    loop: MonitoringLoop
    agent: MonitoringAgent  # handles orchestrator tasks
    watcher: RecoveryWatcher  # handles deployment.action.completed events
    description: str


@asynccontextmanager
async def build_runtime(
    config: MonitoringConfig,
    graph_path: Path,
    verbose: bool = False,
    use_forecast: bool = True,
) -> AsyncIterator[Runtime]:
    reasoner = Reasoner(load_reasoning_config(graph_path)) if graph_path.exists() else None
    settings = Settings()
    labeler = (
        GeminiLabeler(settings.gemini_api_key, settings.gemini_model)
        if settings.gemini_api_key
        else None
    )
    forecaster = None
    if config.forecast.enabled and use_forecast:
        try:
            import chronos  # noqa: F401 - only checks that the optional extra is installed
        except ImportError:
            print('Chronos non installe (pip install -e ".[forecast]") : detecteur desactive')
        else:
            forecaster = ChronosForecaster(config.forecast.model, config.forecast.threads)
            print(f"Chargement de Chronos ({config.forecast.model})...")
            await asyncio.to_thread(forecaster.warm_up)
    memory = AlertMemory(config.memory.path, config.memory) if config.memory.enabled else None
    publisher = JsonlEventPublisher(config.events.path) if config.events.enabled else None
    versions = {s.service_id: s.version for s in config.services}
    try:
        async with (
            PrometheusClient(settings.prometheus_url) as prometheus,
            LokiClient(settings.loki_url, containers=config.containers()) as loki,
        ):
            collector = PrometheusCollector(
                prometheus, profiles={s.service_id: s for s in config.services}
            )
            reader = SnapshotReader(collector.collect)

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
                forecast=(
                    ForecastDetector(collector, forecaster, config.forecast) if forecaster else None
                ),
                emitter=(AnomalyEmitter(publisher, config.events, versions) if publisher else None),
                decider=DecisionMaker(config.decision, reader, memory=memory),
            )
            verifier = Verifier(reader, config.recovery)
            agent = MonitoringAgent(OneShotDetector(loop), verifier, publisher, versions)
            sink = publisher or _NoPublisher()
            names = ", ".join(f"{s.service_id} ({s.mode})" for s in config.services)
            description = (
                f"{names}, un cycle toutes les {config.interval_seconds:.0f}s, "
                f"logs {'actifs' if config.logs_enabled else 'desactives'}, "
                f"Gemini {'actif' if labeler else 'non configure'}, "
                f"raisonnement {'actif' if reasoner else 'inactif'}, "
                f"Chronos {config.forecast.model if forecaster else 'inactif'}, "
                f"memoire {config.memory.path if memory else 'inactive'}, "
                f"evenements {config.events.path if publisher else 'inactifs'}"
            )
            yield Runtime(config, loop, agent, RecoveryWatcher(verifier, sink), description)
    finally:
        if labeler:
            await labeler.aclose()
        if memory:
            memory.close()


class _NoPublisher:
    async def publish(self, event: object) -> None:
        return None
