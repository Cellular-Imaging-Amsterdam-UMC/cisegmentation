from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cisegmentation.roundtrip import DEFAULT_BIOMERO_ROOT, RoundtripRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local OMERO-BIOMERO-Slurm roundtrip")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--parameters-json", required=True)
    parser.add_argument("--gpu", choices=["true", "false"], default="true")
    parser.add_argument("--biomero-root", type=Path, default=DEFAULT_BIOMERO_ROOT)
    parser.add_argument("--timeout", type=int, default=6 * 60 * 60)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parameters = json.loads(args.parameters_json)
    if not isinstance(parameters, dict):
        raise SystemExit("--parameters-json must contain a JSON object")
    runner = RoundtripRunner(
        ROOT,
        args.input_dir,
        args.output_dir,
        parameters,
        gpu=args.gpu == "true",
        biomero_root=args.biomero_root,
        timeout=args.timeout,
        emit=lambda line: print(line, flush=True),
    )
    return runner.execute()


if __name__ == "__main__":
    raise SystemExit(main())
