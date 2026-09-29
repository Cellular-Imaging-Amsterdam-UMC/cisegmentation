"""Disk-backed native spot coordinates; raster labels keep their existing IDs."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np


def native_records(points, labels):
    """Associate native coordinates with retained (including displaced) pixels."""
    points = np.asarray(points, dtype=np.float64)
    retained = set(map(int, np.unique(labels)))
    return np.asarray(
        [(index, *point) for index, point in enumerate(points, 1) if index in retained],
        dtype=np.float64,
    ).reshape(-1, 4)


def append_raw(path, t, records, mapping=None, origin=(0, 0, 0)):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=120)) as db, db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS points (t INTEGER, label INTEGER, z DOUBLE, y DOUBLE, x DOUBLE, PRIMARY KEY(t,label))"
        )
        rows = []
        for label, z, y, x in records:
            label = int(label)
            if mapping is not None:
                label = mapping.get(label, 0)
            if label:
                rows.append(
                    (
                        int(t),
                        label,
                        float(z) + origin[0],
                        float(y) + origin[1],
                        float(x) + origin[2],
                    )
                )
            if len(rows) == 1024:
                db.executemany("INSERT OR IGNORE INTO points VALUES (?,?,?,?,?)", rows)
                rows.clear()
        db.executemany("INSERT OR IGNORE INTO points VALUES (?,?,?,?,?)", rows)


def finalize_points(raw_path, destination, label_name, t, mapping):
    if not Path(raw_path).exists():
        return
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        closing(sqlite3.connect(raw_path)) as source,
        closing(sqlite3.connect(destination, timeout=120)) as target,
        target,
    ):
        target.execute(
            "CREATE TABLE IF NOT EXISTS points (label_name TEXT,t INTEGER,label INTEGER,z DOUBLE,y DOUBLE,x DOUBLE,PRIMARY KEY(label_name,t,label))"
        )
        cursor = source.execute(
            "SELECT label,z,y,x FROM points WHERE t=? ORDER BY label", (t,)
        )
        while rows := cursor.fetchmany(1024):
            target.executemany(
                "INSERT OR IGNORE INTO points VALUES (?,?,?,?,?,?)",
                [
                    (label_name, t, mapping[int(label)], z, y, x)
                    for label, z, y, x in rows
                    if int(label) in mapping
                ],
            )
