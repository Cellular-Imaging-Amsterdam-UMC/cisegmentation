from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cisegmentation.roundtrip import (  # noqa: E402
    docker_build_state_matches,
    docker_source_id,
    read_docker_build_state,
    write_docker_build_state,
)


def image_id(image: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
    )
    return str(json.loads(result.stdout)[0]["Id"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record or validate a local workflow Docker build")
    parser.add_argument("action", choices=("check", "record"))
    parser.add_argument("--image", default="w_cisegmentation:latest")
    args = parser.parse_args(argv)

    source_id = docker_source_id(ROOT)
    try:
        current_image_id = image_id(args.image)
    except (subprocess.CalledProcessError, OSError, ValueError, KeyError):
        return 1
    if args.action == "check":
        return 0 if docker_build_state_matches(
            read_docker_build_state(ROOT), source_id, current_image_id
        ) else 1
    write_docker_build_state(ROOT, source_id, current_image_id, args.image)
    print(f"Recorded Docker build state: {source_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
