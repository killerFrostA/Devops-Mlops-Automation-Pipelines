"""Generate namespaced Protobuf clients after installing the messaging extra."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import grpc_tools

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compile in a temporary directory")
    args = parser.parse_args()
    source = ROOT / "contracts/protobuf"
    well_known = Path(grpc_tools.__file__).parent / "_proto"
    files = sorted(source.rglob("*.proto"))
    with tempfile.TemporaryDirectory(prefix="devops-protobuf-") as scratch:
        output = Path(scratch) if args.check else ROOT
        subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"-I{source}",
                f"-I{well_known}",
                f"--python_out={output}",
                f"--grpc_python_out={output}",
                *[str(path) for path in files],
            ],
            check=True,
        )
    print(f"{'Checked' if args.check else 'Generated'} {len(files)} Protobuf contracts")


if __name__ == "__main__":
    main()
