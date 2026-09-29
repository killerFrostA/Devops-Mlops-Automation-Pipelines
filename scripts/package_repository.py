"""Create a clean shareable source archive from Git-visible project files."""

import subprocess
import zipfile
from pathlib import Path

from src import PROJECT_NAME

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    paths = sorted(set(result.stdout.decode("utf-8").split("\0")) - {""})
    target = ROOT / f"dist/{PROJECT_NAME}-skeleton.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in paths:
            path = ROOT / name
            if path.is_file():
                archive.write(path, arcname=f"{PROJECT_NAME}/{name}")
    print(f"Packaged {len(paths)} source files: {target}")


if __name__ == "__main__":
    main()
