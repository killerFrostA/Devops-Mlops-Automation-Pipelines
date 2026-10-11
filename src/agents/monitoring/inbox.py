"""Hand Agent 1 a task or an event, as the orchestrator / Kafka will (until they are wired).

    python -m src.agents.monitoring.inbox detect payment-service
    python -m src.agents.monitoring.inbox verify payment-service --action ACT-1 [--window 120]
    python -m src.agents.monitoring.inbox event experiments/checks/action_completed.json

`detect` and `verify` build an AgentTask and go through MonitoringAgent.handle, exactly like an
orchestrator task; `event` gives a contract event (deployment.action.completed) to the
recovery watcher. Results are printed; contract events go to the configured events file.
"""

import argparse
import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.agents.monitoring.config import load_config
from src.agents.monitoring.events import new_id
from src.agents.monitoring.runtime import build_runtime
from src.agents.monitoring.service import DETECT, VERIFY
from src.contracts.enums import AgentId
from src.contracts.events import EventEnvelope
from src.contracts.tasks import AgentTask, TaskContext


def make_task(
    objective: str,
    service_id: str,
    namespace: str,
    environment: str,
    signals: dict[str, str],
    seconds: float,
) -> AgentTask:
    return AgentTask(
        task_id=new_id("TASK"),
        incident_id=new_id("INC"),
        agent_id=AgentId.MONITORING,
        objective=objective,
        deadline=datetime.now(UTC) + timedelta(seconds=seconds),
        task_context=TaskContext(
            service_id=service_id,
            namespace=namespace,
            environment=environment,  # type: ignore[arg-type]
            signals=dict(signals),
        ),
    )


async def main(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.window is not None:
        recovery = config.recovery.model_copy(update={"window_seconds": args.window})
        config = config.model_copy(update={"recovery": recovery})
    async with build_runtime(config, args.graph, use_forecast=not args.no_forecast) as runtime:
        if args.command == "event":
            document = json.loads(Path(args.target).read_text(encoding="utf-8"))
            event = EventEnvelope.model_validate(document)
            print(f"evenement recu : {event.topic} pour {event.service.service_id}")
            reply = await runtime.watcher.on_event(event)
            if reply is None:
                print("rien a verifier (action non reussie, verification non demandee)")
            else:
                print(json.dumps(reply.model_dump(mode="json"), indent=2, ensure_ascii=False))
            return
        objective = DETECT if args.command == "detect" else VERIFY
        signals = {"action_id": args.action} if args.action else {}
        budget = config.recovery.settle_seconds + config.recovery.window_seconds + 120
        task = make_task(
            objective,
            args.target,
            args.namespace,
            config.events.environment.value,
            signals,
            budget,
        )
        print(f"tache {task.objective} recue pour {args.target} ({task.task_id})")
        result = await runtime.agent.handle(task)
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=["detect", "verify", "event"])
    parser.add_argument("target", help="service_id (detect/verify) or event JSON file (event)")
    parser.add_argument("--action", help="action_id to verify (verify)")
    parser.add_argument("--namespace", default="local")
    parser.add_argument("--window", type=float, help="observation window in seconds (verify)")
    parser.add_argument("--no-forecast", action="store_true", help="skip loading Chronos")
    parser.add_argument("--config", type=Path, default=Path("configs/monitoring.yaml"))
    parser.add_argument("--graph", type=Path, default=Path("configs/dependencies.yaml"))
    asyncio.run(main(parser.parse_args()))
