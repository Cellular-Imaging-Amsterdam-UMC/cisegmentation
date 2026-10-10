"""Check all stored values and geometry WKB between two extension benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def compare(left: Path, right: Path):
    checked = {}
    for filename in ["measurements.duckdb", "geometry.duckdb"]:
        a, b = left / filename, right / filename
        db = duckdb.connect(":memory:")
        try:
            db.execute(
                "ATTACH '" + str(a).replace("'", "''") + "' AS previous (READ_ONLY)"
            )
            db.execute(
                "ATTACH '" + str(b).replace("'", "''") + "' AS current (READ_ONLY)"
            )
            tables = [
                row[0]
                for row in db.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_catalog='previous' AND table_type='BASE TABLE' ORDER BY table_name"
                ).fetchall()
            ]
            counts = {}
            for table in tables:
                quoted = '"' + table.replace('"', '""') + '"'
                # Worker setting is intentionally different; compare version/name.
                selection = "name,version" if table == "measurement_extensions" else "*"
                query = f"SELECT {selection} FROM previous.main.{quoted} EXCEPT ALL SELECT {selection} FROM current.main.{quoted}"
                reverse = f"SELECT {selection} FROM current.main.{quoted} EXCEPT ALL SELECT {selection} FROM previous.main.{quoted}"
                assert (
                    db.execute(f"SELECT count(*) FROM ({query})").fetchone()[0] == 0
                ), (filename, table, "removed/changed")
                assert (
                    db.execute(f"SELECT count(*) FROM ({reverse})").fetchone()[0] == 0
                ), (filename, table, "added/changed")
                counts[table] = db.execute(
                    f"SELECT count(*) FROM previous.main.{quoted}"
                ).fetchone()[0]
            checked[filename] = counts
        finally:
            db.close()
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.left, args.right), indent=2))
