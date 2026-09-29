import argparse
import json

import uvicorn

from src import PROJECT_NAME
from src.agents.registry import AGENTS
from src.config import Settings
from src.platform.observability.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{PROJECT_NAME} repository skeleton tooling")
    subcommands = parser.add_subparsers(dest="command", required=True)
    serve = subcommands.add_parser("serve", help="Run the local health/inventory API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    subcommands.add_parser("agents", help="List the six specialist ownership areas")
    args = parser.parse_args()
    if args.command == "agents":
        print(
            json.dumps({key.value: value.responsibility for key, value in AGENTS.items()}, indent=2)
        )
        return
    config = Settings()
    configure_logging(config.log_level)
    uvicorn.run(
        "src.api.app:create_app", factory=True, host=args.host, port=args.port, reload=args.reload
    )


if __name__ == "__main__":
    main()
