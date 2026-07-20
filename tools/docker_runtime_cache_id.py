"""Print a stable identity for the heavy Docker runtime-cache image."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_INPUTS = (
    ROOT / "Dockerfile.runtime",
    ROOT / "requirements.txt",
    ROOT / "tools" / "download_models.py",
)


def runtime_cache_id(model_cache_id: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"model-cache-id\0")
    digest.update(model_cache_id.strip().encode("ascii"))
    for path in RUNTIME_INPUTS:
        digest.update(b"\0path\0")
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0content\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cache-id", required=True)
    args = parser.parse_args()
    print(runtime_cache_id(args.model_cache_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
