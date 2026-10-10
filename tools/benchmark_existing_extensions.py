"""Re-run only optional measurements on a copy of a completed result.

No inference and no source modifications. Fixtures: input.zarr, labels.ome.zarr,
original.duckdb, and optionally original-geometry.duckdb in --fixture.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cisegmentation.measurement_extensions import (
    _SCHEMA,
    connect_database,
    write_extensions,
)
from cisegmentation.ome_zarr_io import enumerate_resources
from cisegmentation.settings import SegmentationSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    stage = args.fixture / args.name
    stage.mkdir(exist_ok=False)
    database = stage / "measurements.duckdb"
    shutil.copy2(args.fixture / "original.duckdb", database)
    db = connect_database(database, "duckdb")
    settings_data = json.loads(
        db.execute(
            "SELECT settings_json FROM measurement_extensions LIMIT 1"
        ).fetchone()[0]
    )
    for statement in _SCHEMA.split(";"):
        if statement.strip():
            db.execute("DROP TABLE IF EXISTS " + statement.split()[2])
    db.close()
    settings_data["max_measurement_workers"] = args.workers
    settings = SegmentationSettings(**settings_data)
    profiler = cProfile.Profile()
    if args.profile:
        profiler.enable()
    summary, _ = write_extensions(
        database,
        "duckdb",
        enumerate_resources(args.fixture / "input.zarr"),
        args.fixture / "labels.ome.zarr",
        settings,
        stage_dir=stage,
        log=print,
    )
    if args.profile:
        profiler.disable()
        profiler.dump_stats(str(stage / "profile.pstats"))
    (stage / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
