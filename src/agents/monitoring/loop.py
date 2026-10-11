"""Autonomous loop of Agent 1: perceive -> detect -> group -> report, every interval.

One failing source or detector never stops the loop: its error is recorded in the cycle report
and the other detectors and services carry on. Each service is bounded by a timeout, and every
cycle measures its own duration and memory so the cost stays visible on a small machine.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from src.agents.monitoring.collectors.profile import Mode
from src.agents.monitoring.config import MonitoringConfig, WatchedService
from src.agents.monitoring.decision import Action, DecisionMaker, DecisionRecord
from src.agents.monitoring.detectors.logs import LogFinding
from src.agents.monitoring.detectors.slo import SLOEvaluation
from src.agents.monitoring.detectors.statistical import StatisticalFinding
from src.agents.monitoring.events import AnomalyEmitter
from src.agents.monitoring.grouping import (
    Decision,
    Symptom,
    SymptomGrouper,
    from_logs,
    from_slo,
    from_statistical,
)
from src.agents.monitoring.reasoning import Hypothesis, Reasoner
from src.contracts.events import EventEnvelope

try:  # optional: precise process memory on any OS
    import psutil
except ImportError:  # pragma: no cover - measured as unavailable
    psutil = None


class ServiceDetector[T](Protocol):
    async def evaluate(self, service_id: str) -> list[T]: ...


@dataclass(frozen=True, slots=True)
class ServiceReport:
    service_id: str
    namespace: str
    symptoms: int
    errors: dict[str, str]  # detector name -> error, for detectors that failed this cycle
    # Raw detector results, normal ones included, for the detailed (verbose) report.
    slo: tuple[SLOEvaluation, ...] = ()
    statistical: tuple[StatisticalFinding, ...] = ()
    logs: tuple[LogFinding, ...] = ()
    logs_checked: bool = False
    forecast: tuple[StatisticalFinding, ...] = ()  # Chronos, when enabled


@dataclass(frozen=True, slots=True)
class CycleReport:
    started_at: float
    duration_seconds: float
    memory_mb: float | None
    sources_down: tuple[str, ...]
    services: tuple[ServiceReport, ...]
    decisions: tuple[Decision, ...]
    hypotheses: tuple[Hypothesis, ...] = ()  # suspect services, most probable first
    decision: DecisionRecord | None = None
    event: EventEnvelope | None = None  # alert sent to the other agents this cycle
    event_error: str | None = None  # why it could not be sent

    @property
    def notifications(self) -> tuple[Decision, ...]:
        return tuple(d for d in self.decisions if d.notify)


class Reporter(Protocol):
    def report(self, cycle: CycleReport) -> None: ...


def process_memory_mb() -> float | None:
    if psutil is None:
        return None
    rss: int = psutil.Process().memory_info().rss
    return rss / (1024 * 1024)


@dataclass
class MonitoringLoop:
    config: MonitoringConfig
    slo: ServiceDetector[SLOEvaluation]
    statistical: ServiceDetector[StatisticalFinding]
    logs: ServiceDetector[LogFinding] | None
    grouper: SymptomGrouper
    reporter: Reporter
    # Cheap probes run at the start of each cycle, e.g. {"prometheus": ...}.
    health_checks: dict[str, Callable[[], Awaitable[bool]]] = field(default_factory=dict)
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    reasoner: Reasoner | None = None
    decider: DecisionMaker | None = None
    forecast: ServiceDetector[StatisticalFinding] | None = None
    emitter: AnomalyEmitter | None = None

    async def run_cycle(self) -> CycleReport:
        started = time.perf_counter()
        started_at = self.clock()
        # Probes and services run concurrently, all bounded by the same timeout.
        probes = asyncio.ensure_future(self._probe_sources())
        results = await asyncio.gather(*(self.watch(s) for s in self.config.services))
        sources_down = await probes
        symptoms = [symptom for _, found in results for symptom in found]
        decisions = self.grouper.ingest(symptoms)
        hypotheses: list[Hypothesis] = []
        if self.reasoner is not None:
            self.reasoner.observe([d.incident for d in decisions])
            hypotheses = self.reasoner.rank()
        decision = None
        if self.decider is not None:
            decision = await self.decider.decide(hypotheses, [d.incident for d in decisions])
        event, event_error = None, None
        if self.emitter is not None:
            event, event_error = await self.emitter.emit(
                decision, [d.incident for d in decisions], hypotheses
            )
        cycle = CycleReport(
            started_at=started_at,
            duration_seconds=time.perf_counter() - started,
            memory_mb=process_memory_mb(),
            sources_down=sources_down,
            services=tuple(report for report, _ in results),
            decisions=tuple(decisions),
            hypotheses=tuple(hypotheses),
            decision=decision,
            event=event,
            event_error=event_error,
        )
        self.reporter.report(cycle)
        return cycle

    async def run_forever(self, stop: asyncio.Event, max_cycles: int | None = None) -> None:
        """Run cycles every `interval_seconds` until `stop` is set (or max_cycles, for tests)."""
        done = 0
        while not stop.is_set() and (max_cycles is None or done < max_cycles):
            cycle = await self.run_cycle()
            done += 1
            remaining = max(0.0, self.config.interval_seconds - cycle.duration_seconds)
            sleeper = asyncio.ensure_future(self.sleep(remaining))
            stopper = asyncio.ensure_future(stop.wait())
            await asyncio.wait({sleeper, stopper}, return_when=asyncio.FIRST_COMPLETED)
            for task in (sleeper, stopper):
                task.cancel()

    async def _probe_sources(self) -> tuple[str, ...]:
        names = list(self.health_checks)
        timeout = self.config.service_timeout_seconds
        answers = await asyncio.gather(
            *(asyncio.wait_for(self.health_checks[n](), timeout) for n in names),
            return_exceptions=True,
        )
        return tuple(n for n, ok in zip(names, answers, strict=True) if ok is not True)

    async def watch(self, service: WatchedService) -> tuple[ServiceReport, list[Symptom]]:
        errors: dict[str, str] = {}
        symptoms: list[Symptom] = []
        sid, ns = service.service_id, service.namespace
        slo_sli = {s.name: s.sli for s in self.config.slo.slos}

        slo: list[SLOEvaluation] = []
        statistical: list[StatisticalFinding] = []
        forecast: list[StatisticalFinding] = []
        logs: list[LogFinding] = []

        async def run[T](
            name: str,
            detector: ServiceDetector[T],
            to_symptoms: Callable[[list[T]], list[Symptom]],
            sink: list[T],
        ) -> None:
            try:
                results = await detector.evaluate(sid)
            except Exception as error:  # noqa: BLE001 - one detector must not stop the agent
                errors[name] = f"{type(error).__name__}: {error}"
                return
            sink.extend(results)
            symptoms.extend(to_symptoms(results))

        jobs = [
            run(
                "statistical",
                self.statistical,
                lambda r: from_statistical(sid, ns, r),
                statistical,
            ),
        ]
        if self.forecast is not None:
            jobs.append(
                run("forecast", self.forecast, lambda r: from_statistical(sid, ns, r), forecast)
            )
        # A container watched from outside has no request counter: there is no SLO to check.
        if service.mode is Mode.APP:
            jobs.append(run("slo", self.slo, lambda r: from_slo(sid, ns, slo_sli, r), slo))
        logs_checked = self.logs is not None and self.config.logs_enabled
        if self.logs is not None and logs_checked:
            jobs.append(run("logs", self.logs, lambda r: from_logs(sid, ns, r), logs))
        try:
            await asyncio.wait_for(asyncio.gather(*jobs), self.config.service_timeout_seconds)
        except TimeoutError:
            errors["timeout"] = f"over {self.config.service_timeout_seconds:.0f}s"
        report = ServiceReport(
            sid,
            ns,
            len(symptoms),
            errors,
            tuple(slo),
            tuple(statistical),
            tuple(logs),
            logs_checked,
            tuple(forecast),
        )
        return report, symptoms


def _num(value: float | None, unit: str = "") -> str:
    return "-" if value is None else f"{value:.2f}{unit}"


class ConsoleReporter:
    """Human-readable report: one summary line per cycle, plus each incident that changed.

    With `verbose`, also every detector's decision for every service, normal ones included,
    to show that all layers run each cycle.
    """

    def __init__(self, write: Callable[[str], None] = print, verbose: bool = False) -> None:
        self._write = write
        self._verbose = verbose

    def report(self, cycle: CycleReport) -> None:
        stamp = time.strftime("%H:%M:%S", time.localtime(cycle.started_at))
        memory = "-" if cycle.memory_mb is None else f"{cycle.memory_mb:.0f} Mo"
        seen = len({d.incident.incident_id for d in cycle.decisions})
        self._write(
            f"[{stamp}] cycle {cycle.duration_seconds:.2f}s, memoire de l'agent {memory}, "
            f"{len(cycle.services)} service(s), {seen} incident(s) observe(s) ce cycle"
        )
        for source in cycle.sources_down:
            self._write(f"   ! source indisponible : {source}")
        for service in cycle.services:
            if self._verbose:
                self._details(service)
            for detector, error in service.errors.items():
                self._write(f"   ! {service.service_id} / {detector} : {error}")
        for decision in cycle.notifications:
            incident = decision.incident
            verb = "OUVERT" if decision.action.value == "OPENED" else "MIS A JOUR"
            self._write(
                f"   >> INCIDENT {verb} {incident.incident_id} {incident.service_id} "
                f"{incident.category} gravite={incident.severity} "
                f"sources={'+'.join(sorted(incident.sources))} vu {incident.occurrences} fois"
            )
        # Hypotheses change only when incidents change: shown then, or every cycle in verbose.
        if cycle.hypotheses and (self._verbose or cycle.notifications):
            self._write("   ?? HYPOTHESES (service a l'origine le plus probable) :")
            for rank, h in enumerate(cycle.hypotheses[:3], start=1):
                proofs = "; ".join(
                    f"{e.category} sur {e.observed_on}"
                    + ("" if e.kind == "own_symptom" else f" ({e.detail})")
                    for e in h.evidence
                )
                self._write(
                    f"      {rank}. {h.service_id:15} {h.probability:.2f}  preuves : {proofs}"
                )

        record = cycle.decision
        if record and (
            self._verbose
            or record.notify
            or record.action is Action.COLLECT_MORE
            or record.investigations
        ):
            for inv in record.investigations:
                values = ", ".join(f"{k}={_num(v)}" for k, v in inv.observed.items())
                self._write(f"   .. collecte sur {inv.service_id} : {inv.outcome} ({values})")
            label = {"ALERT": "ALERTER", "COLLECT_MORE": "COLLECTER", "WAIT": "ATTENDRE"}
            self._write(
                f"   !! DECISION : {label[record.action.value]} {record.target} "
                f"(probabilite {record.probability:.2f}) - {record.reason}"
            )
            if cycle.event is not None:
                e = cycle.event
                self._write(
                    f"   => EVENEMENT {e.topic} -> {e.target} : {e.service.service_id} "
                    f"priorite={e.priority} gravite={e.severity} confiance={e.confidence} "
                    f"(cle {e.idempotency_key})"
                )
            if cycle.event_error is not None:
                self._write(f"   ! envoi de l'alerte impossible : {cycle.event_error}")
            if record.alert_id is not None:
                self._write(
                    f"   ## alerte n.{record.alert_id} enregistree. Ton avis : "
                    f"python -m src.agents.monitoring.feedback vraie|fausse {record.alert_id}"
                )

    def _details(self, service: ServiceReport) -> None:
        write = self._write
        write(f"   {service.service_id} ({service.namespace})")
        for r in service.slo:
            write(
                f"     SLO          {r.slo:15} {r.rule:10} burn long={_num(r.burn_rate_long)} "
                f"court={_num(r.burn_rate_short)} seuil={r.threshold:g} -> {r.status}"
            )
        for f in service.statistical:
            if f.method == "growth":
                detail = f"pente={_num(f.slope_per_minute, '/min')} r2={_num(f.r2)}"
            else:
                detail = (
                    f"actuel={_num(f.current)} habituel={_num(f.baseline_median)} "
                    f"z={_num(f.robust_z)}"
                )
            write(f"     Statistique  {f.signal:15} {detail:48} -> {f.status}")
        for f in service.forecast:
            detail = (
                f"actuel={_num(f.current)} prevu={_num(f.baseline_median)} "
                f"haut={_num(f.expected_high)} dehors={_num(f.outside_share)}"
            )
            write(f"     Chronos      {f.signal:15} {detail:48} -> {f.status}")
        if not service.logs_checked or "logs" in service.errors:
            write("     Logs         non verifies ce cycle")
        elif not service.logs:
            write("     Logs         aucun modele nouveau ou en hausse" + " " * 33 + "-> NORMAL")
        for lf in service.logs:
            write(
                f"     Logs         {lf.kind:11} {lf.label:20} recent={lf.count_recent} "
                f"habituel={lf.count_baseline} -> ANOMALY"
            )
