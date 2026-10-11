"""Run Agent 1 autonomously: python -m src.agents.monitoring [--config FILE] [--cycles N]
[--verbose] [--graph FILE].

Ctrl+C stops the agent cleanly after the current cycle.
"""

import argparse
import asyncio
import contextlib
from pathlib import Path

from src.agents.monitoring.config import load_config
from src.agents.monitoring.runtime import build_runtime


async def main(config_path: Path, graph_path: Path, cycles: int | None, verbose: bool) -> None:
    async with build_runtime(load_config(config_path), graph_path, verbose) as runtime:
        print(f"Agent 1 demarre : {runtime.description} (Ctrl+C pour arreter)")
        await runtime.loop.run_forever(asyncio.Event(), max_cycles=cycles)


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
