"""Capture installed core/dev versions. Use a clean, reviewed environment when refreshing."""

import importlib.metadata
import tomllib
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def closure(roots: list[str]) -> list[str]:
    pending = [Requirement(root) for root in roots]
    visited: set[tuple[str, tuple[str, ...]]] = set()
    pins: dict[str, str] = {}
    while pending:
        requirement = pending.pop()
        name = canonicalize_name(requirement.name)
        key = (name, tuple(sorted(requirement.extras)))
        if key in visited:
            continue
        visited.add(key)
        distribution = importlib.metadata.distribution(name)
        pins[name] = distribution.version
        for text in distribution.requires or ():
            dependency = Requirement(text)
            environments = [
                dict(default_environment(), extra=extra) for extra in ("", *requirement.extras)
            ]
            if dependency.marker is None or any(
                dependency.marker.evaluate(environment) for environment in environments
            ):
                pending.append(dependency)
    return [f"{name}=={version}" for name, version in sorted(pins.items())]


def main() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    target = ROOT / "requirements"
    target.mkdir(exist_ok=True)
    runtime = project["dependencies"]
    dev = runtime + project["optional-dependencies"]["dev"]
    build = ["build", "setuptools"]
    groups = (
        ("runtime.lock", runtime),
        ("dev.lock", dev),
        ("build.lock", build),
        ("rpc.lock", ["grpcio-tools"]),
    )
    for filename, roots in groups:
        header = "# Exact installed version pins; no hashes. Integration extras are separate.\n"
        (target / filename).write_text(header + "\n".join(closure(roots)) + "\n", encoding="utf-8")
        print(f"Captured {filename}")


if __name__ == "__main__":
    main()
