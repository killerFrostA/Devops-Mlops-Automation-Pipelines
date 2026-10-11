"""Operator feedback on Agent 1 alerts, stored in the agent's memory.

    python -m src.agents.monitoring.feedback list                 # recent alerts and verdicts
    python -m src.agents.monitoring.feedback fausse 3 --note "test de charge prevu"
    python -m src.agents.monitoring.feedback vraie 4
    python -m src.agents.monitoring.feedback stats                # what the agent has learned

The running agent reads the memory at each decision: no restart is needed.
"""

import argparse
import time
from pathlib import Path

from src.agents.monitoring.config import load_config
from src.agents.monitoring.memory import AlertMemory, Verdict


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=["list", "stats", *[v.value for v in Verdict]])
    parser.add_argument("alert_id", type=int, nargs="?")
    parser.add_argument("--note", default="")
    parser.add_argument("--config", type=Path, default=Path("configs/monitoring.yaml"))
    args = parser.parse_args(argv)

    config = load_config(args.config)
    memory = AlertMemory(config.memory.path, config.memory)
    try:
        if args.command == "list":
            for alert in memory.alerts():
                stamp = time.strftime("%d/%m %H:%M:%S", time.localtime(alert.at))
                verdict = alert.verdict.value if alert.verdict else "sans avis"
                print(
                    f"n.{alert.alert_id:<4} {stamp}  {alert.service_id:18} "
                    f"{','.join(alert.categories):28} p={alert.probability:.2f}  [{verdict}]"
                    + (f"  note : {alert.note}" if alert.note else "")
                )
        elif args.command == "stats":
            for r in memory.pairs():
                print(
                    f"{r.service_id:18} {r.category:18} {r.true_alerts} vraie(s) "
                    f"{r.false_alerts} fausse(s)  precision={r.precision:.2f}  "
                    f"facteur={r.odds_factor:.2f}"
                )
        else:
            if args.alert_id is None:
                parser.error("give the alert number, e.g. fausse 3")
            memory.add_feedback(args.alert_id, Verdict(args.command), args.note)
            print(f"alerte n.{args.alert_id} marquee comme {args.command}")
    except KeyError as error:
        parser.error(str(error))
    finally:
        memory.close()


if __name__ == "__main__":
    main()
