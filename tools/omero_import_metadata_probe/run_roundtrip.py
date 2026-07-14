from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROBE = Path(
    r"C:\rahoebe\Python\cideconvolve\tools\omero_import_metadata_probe"
)
DEFAULT_REPORTS = Path.home() / "Downloads" / "cisegmentation_omero_roundtrips"


def run_logged(command: list[str], cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    log.write_text(
        f"COMMAND: {subprocess.list2cmdline(command)}\nEXIT: {process.returncode}\n\nSTDOUT\n{process.stdout}\n\nSTDERR\n{process.stderr}",
        encoding="utf-8",
    )
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}); see {log}")


def build_wrapper_command(
    python: Path,
    input_dir: Path,
    output_dir: Path,
    model: str,
    target_type: str,
    device: str,
) -> list[str]:
    return [
        str(python),
        str(ROOT / "wrapper.py"),
        "--infolder",
        str(input_dir),
        "--outfolder",
        str(output_dir),
        "--model",
        model,
        "--target",
        target_type,
        "--device",
        device,
        "--local",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run CI Segmentation through the local OMERO/BIOMERO metadata probe"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--existing-image", type=int)
    parser.add_argument(
        "--target",
        required=True,
        help="Dataset:ID for images or Screen:ID for HCS plates",
    )
    parser.add_argument("--model", default="cellpose3:nuclei")
    parser.add_argument(
        "--target-type", choices=["nuclei", "cells", "foci", "spots"], default="nuclei"
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--user", default="root")
    parser.add_argument("--group", default="system")
    parser.add_argument(
        "--cleanup", choices=["always", "success", "never"], default="never"
    )
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args(argv)
    probe_run = args.probe_dir / "run.cmd"
    if not probe_run.exists():
        raise FileNotFoundError(
            f"Missing omero_import_metadata_probe runner: {probe_run}"
        )
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report = args.report_root / f"roundtrip_{run_id}"
    report.mkdir(parents=True)
    logs = report / "logs"
    if args.existing_image is not None:
        export_report = report / "01_source_export"
        run_logged(
            [
                "cmd.exe",
                "/d",
                "/c",
                str(probe_run),
                "--slurm-input-image",
                str(args.existing_image),
                "--user",
                args.user,
                "--group",
                args.group,
                "--out",
                str(export_report),
            ],
            args.probe_dir,
            logs / "source_export.log",
        )
        stores = sorted(export_report.rglob("*.ome.zarr"))
        if not stores:
            raise RuntimeError(f"Probe exported no OME-Zarr below {export_report}")
        source_store = stores[0]
    else:
        source_store = args.input.resolve()
    input_dir = report / "02_wrapper" / "input"
    output_dir = report / "02_wrapper" / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    staged = input_dir / source_store.name
    if source_store.resolve() != staged.resolve():
        import shutil

        shutil.copytree(source_store, staged)
    wrapper_command = build_wrapper_command(
        Path(sys.executable),
        input_dir,
        output_dir,
        args.model,
        args.target_type,
        args.device,
    )
    run_logged(wrapper_command, ROOT, logs / "wrapper.log")
    outputs = sorted(output_dir.glob("*.ome.zarr"))
    if len(outputs) != 1:
        raise RuntimeError(
            f"Expected one wrapper OME-Zarr output, found {len(outputs)}"
        )
    import_report = report / "03_output_import"
    probe_command = [
        "cmd.exe",
        "/d",
        "/c",
        str(probe_run),
        "--input",
        str(outputs[0]),
        "--target",
        args.target,
        "--user",
        args.user,
        "--group",
        args.group,
        "--mode",
        "both",
        "--cleanup",
        args.cleanup,
        "--out",
        str(import_report),
    ]
    run_logged(probe_command, args.probe_dir, logs / "output_import.log")
    summary = {
        "source": str(source_store),
        "staged_input": str(staged),
        "segmentation_output": str(outputs[0]),
        "model": args.model,
        "target_type": args.target_type,
        "omero_target": args.target,
        "probe_report": str(import_report),
        "cleanup": args.cleanup,
    }
    (report / "roundtrip_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (report / "roundtrip_summary.md").write_text(
        "# CI Segmentation OMERO round trip\n\n"
        + "\n".join(f"- **{key}:** `{value}`" for key, value in summary.items())
        + "\n",
        encoding="utf-8",
    )
    print(f"Round-trip report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
