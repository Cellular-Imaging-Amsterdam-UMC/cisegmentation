from __future__ import annotations

import argparse
import json
import sys
import time

from cisegmentation.engine import run_workflow
from cisegmentation.settings import SegmentationSettings


def _bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bilayers OME-Zarr segmentation workflow"
    )
    parser.add_argument("--input-dir", "--infolder", default="/data/in")
    parser.add_argument("--output-dir", "--outfolder", default="/data/out")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--parameters", help="JSON object of workflow parameters")
    for field in SegmentationSettings.__dataclass_fields__.values():
        flag = "--" + field.name.replace("_", "-")
        aliases = [flag]
        underscored = "--" + field.name
        if underscored != flag:
            aliases.append(underscored)
        if field.type in {int, "int"}:
            kind = int
        elif field.type in {float, "float"}:
            kind = float
        elif field.type in {bool, "bool"}:
            kind = _bool
        else:
            kind = str
        parser.add_argument(
            *aliases,
            dest=field.name,
            type=kind,
            default=argparse.SUPPRESS,
            nargs="?" if kind is _bool else None,
            const=True if kind is _bool else None,
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    args, unknown = build_parser().parse_known_args(argv)
    values = {}
    if args.parameters:
        values.update(json.loads(args.parameters))
    for name in SegmentationSettings.__dataclass_fields__:
        if hasattr(args, name):
            values[name] = getattr(args, name)
    settings = SegmentationSettings(**values)
    print(
        f"CI segmentation: model={settings.model}, target={settings.target}, benchmark={settings.benchmark}"
    )
    try:
        outputs = run_workflow(args.input_dir, args.output_dir, settings)
    except Exception as exc:
        print(f"CI segmentation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            f"CI segmentation failed after {time.perf_counter() - started:.2f} seconds.",
            file=sys.stderr,
        )
        return 1
    for output in outputs:
        print(f"Output: {output}")
    print(f"CI segmentation completed in {time.perf_counter() - started:.2f} seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
