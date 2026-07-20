from __future__ import annotations

import argparse
import json
import sys
import time

PROCESS_STARTED = time.perf_counter()

from cisegmentation.engine import run_workflow
from cisegmentation.settings import SegmentationSettings, normalize_legacy_workflow_values


_TIMING_LABELS = (
    ("startup_seconds", "startup"),
    ("zarr_read_seconds", "read"),
    ("import_seconds", "imports"),
    ("device_setup_seconds", "device"),
    ("model_load_seconds", "model-load"),
    ("inference_seconds", "inference"),
    ("zarr_write_seconds", "write"),
    ("total_seconds", "total"),
)


def output_timing_line(output) -> str | None:
    """Format timing provenance without importing Zarr again."""
    attrs_path = output / ".zattrs"
    try:
        metadata = json.loads(attrs_path.read_text(encoding="utf-8"))[
            "cisegmentation"
        ]
        timings = metadata["timings"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    phases = " | ".join(
        f"{label}={float(timings.get(key, 0.0)):.2f}s"
        for key, label in _TIMING_LABELS
    )
    hits = metadata.get("model_cache_hits")
    misses = metadata.get("model_cache_misses")
    cache = (
        f" | cache-hits={int(hits)} | cache-misses={int(misses)}"
        if hits is not None and misses is not None
        else ""
    )
    return f"Timing: {output.name} | {phases}{cache}"


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
    legacy_types = {
        "cell_step": _bool,
        "cell_method": str,
        "cell_nuclei_model": str,
        "cell_expansion_nucleus_model": str,
        "nucleus_step": _bool,
        **{f"foci_step_{slot}": _bool for slot in range(1, 5)},
    }
    for name, kind in legacy_types.items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            "--" + name,
            dest=name,
            type=kind,
            default=argparse.SUPPRESS,
            nargs="?" if kind is _bool else None,
            const=True if kind is _bool else None,
            help=argparse.SUPPRESS,
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
    for name in (
        "cell_step",
        "cell_method",
        "cell_nuclei_model",
        "cell_expansion_nucleus_model",
        "nucleus_step",
        *(f"foci_step_{slot}" for slot in range(1, 5)),
    ):
        if hasattr(args, name):
            values[name] = getattr(args, name)
    settings = SegmentationSettings(**normalize_legacy_workflow_values(values))
    print(f"CI segmentation: optional steps, benchmark={settings.benchmark}", flush=True)
    try:
        outputs = run_workflow(
            args.input_dir,
            args.output_dir,
            settings,
            startup_seconds=time.perf_counter() - PROCESS_STARTED,
            log=lambda line: print(line, flush=True),
        )
    except Exception as exc:
        print(f"CI segmentation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            f"CI segmentation failed after {time.perf_counter() - started:.2f} seconds.",
            file=sys.stderr,
        )
        return 1
    for output in outputs:
        print(f"Output: {output}")
        timing_line = output_timing_line(output)
        if timing_line:
            print(timing_line)
    print(f"CI segmentation completed in {time.perf_counter() - started:.2f} seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
