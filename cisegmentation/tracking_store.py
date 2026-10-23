"""Disk-backed tracking: keep only the two frames being matched in RAM.

The observation/lineage graph and results live in a disposable SQLite spool.
Ordering and the two-stage LAP policy match ``tracking.link_observations``.
"""

from __future__ import annotations

import math
import sqlite3
from contextlib import suppress
from itertools import groupby

import numpy as np

from .resources import snapshot
from .tracking import sparse_assignment


class TrackingStore:
    def __init__(self, path, *, create=True):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA cache_size=-8192")
        self.db.execute("PRAGMA temp_store=FILE")
        if create:
            self.db.executescript("""
            CREATE TABLE observations(id INTEGER PRIMARY KEY,t INTEGER,z REAL,y REAL,x REAL,size INTEGER,source TEXT);
            CREATE INDEX observation_frame ON observations(t,id);
            CREATE TABLE links(source INTEGER,target INTEGER UNIQUE,kind TEXT);
            CREATE INDEX link_source ON links(source);
            CREATE TABLE assignments(object_id INTEGER PRIMARY KEY,track_id INTEGER);
            CREATE INDEX assignment_track ON assignments(track_id,object_id);
            CREATE TABLE track_info(track_id INTEGER PRIMARY KEY,lineage_id INTEGER,parent_track_id INTEGER);
        """)
        self.pending = []
        self.count = (
            0
            if create
            else self.db.execute("SELECT count(*) FROM observations").fetchone()[0]
        )
        self.track_count = 0

    def close(self):
        self.db.close()

    def add(self, observation):
        p = observation
        self.pending.append(
            (p["id"], p["t"], *map(float, p["position"]), p["size"], p["source"])
        )
        self.count += 1
        if len(self.pending) >= 512:
            self.flush()

    def flush(self):
        if self.pending:
            self.db.executemany(
                "INSERT INTO observations VALUES (?,?,?,?,?,?,?)", self.pending
            )
            self.pending.clear()
            self.db.commit()

    def frames(self):
        # The number of timepoints is small relative to the number of objects.
        return [
            r[0]
            for r in self.db.execute("SELECT DISTINCT t FROM observations ORDER BY t")
        ]

    def frame(self, t, mode="all"):
        where = {
            "all": "",
            "unmatched_source": "AND NOT EXISTS (SELECT 1 FROM links WHERE source=o.id)",
            "unmatched_target": "AND NOT EXISTS (SELECT 1 FROM links WHERE target=o.id)",
            "division": """AND (SELECT count(*) FROM links WHERE source=o.id)=1
                AND EXISTS (SELECT 1 FROM links l JOIN observations q ON q.id=l.target
                            WHERE l.source=o.id AND q.t=o.t+1)""",
        }[mode]
        count = self.db.execute(
            f"SELECT count(*) FROM observations o WHERE o.t=? {where}", (t,)
        ).fetchone()[0]
        # Two frame dictionaries, KD tree and temporary arrays must fit. This
        # depends on the actual Slurm allocation/cgroup, never a fixed 16 GB.
        if count * 2048 > snapshot(include_gpu=False).ram_available // 4:
            raise MemoryError("Tracking frame observations exceed live RAM budget")
        result = []
        for oid, frame, z, y, x, size in self.db.execute(
            f"SELECT o.id,o.t,o.z,o.y,o.x,o.size FROM observations o WHERE o.t=? {where} ORDER BY o.id",
            (t,),
        ):
            result.append(
                dict(id=oid, t=frame, position=np.array([z, y, x]), size=size)
            )
        return result

    def connect(self, pairs, kind):
        self.db.executemany(
            "INSERT INTO links VALUES (?,?,?)",
            ((a["id"], b["id"], kind) for a, b in pairs),
        )
        self.db.commit()

    def build(self, *, radius=20.0, max_gap=2, divisions=False, object_type="spots"):
        self.flush()
        frames = self.frames()
        for t in frames:
            left = self.frame(t)
            # Only the previous coordinates are needed for velocity prediction.
            previous = {
                oid: (pt, np.array([z, y, x]))
                for oid, pt, z, y, x in self.db.execute(
                    """SELECT o.id,p.t,p.z,p.y,p.x FROM observations o JOIN links l ON l.target=o.id
                   JOIN observations p ON p.id=l.source WHERE o.t=?""",
                    (t,),
                )
            }
            for p in left:
                if p["id"] in previous:
                    pt, position = previous[p["id"]]
                    p["predicted_position"] = p["position"] + (
                        p["position"] - position
                    ) / (t - pt)
            del previous
            right = self.frame(t + 1)
            self.connect(sparse_assignment(left, right, radius), "link")
            del left, right

        if divisions and object_type in {"cells", "nuclei"}:
            for t in frames:
                left = self.frame(t, "division")
                first = {
                    oid: size
                    for oid, size in self.db.execute(
                        """SELECT l.source,q.size FROM links l JOIN observations q ON q.id=l.target
                       JOIN observations p ON p.id=l.source WHERE p.t=?""",
                        (t,),
                    )
                }

                def plausible(parent, daughter):
                    size = parent["size"]
                    first_size = first[parent["id"]]
                    return (
                        size > 0
                        and 0.5 <= (first_size + daughter["size"]) / size <= 1.8
                        and max(first_size, daughter["size"]) <= 1.25 * size
                    )

                right = self.frame(t + 1, "unmatched_target")
                pairs = sparse_assignment(left, right, radius, plausible)
                self.db.executemany(
                    "UPDATE links SET kind='division' WHERE source=?",
                    ((a["id"],) for a, _ in pairs),
                )
                self.connect(pairs, "division")
                del left, right, first, pairs

        for gap in range(2, max_gap + 2):
            for t in frames:
                left = self.frame(t, "unmatched_source")
                right = self.frame(t + gap, "unmatched_target")
                self.connect(sparse_assignment(left, right, radius), "gap")
                del left, right

        for t in frames:
            rows = self.db.execute(
                """
                SELECT o.id,a.track_id,tr.lineage_id,
                    (SELECT count(*) FROM links WHERE source=l.source)
                FROM observations o LEFT JOIN links l ON l.target=o.id
                LEFT JOIN assignments a ON a.object_id=l.source
                LEFT JOIN track_info tr ON tr.track_id=a.track_id
                WHERE o.t=? ORDER BY o.id
            """,
                (t,),
            )
            for oid, parent_track, lineage, outgoing_count in rows:
                if parent_track is not None and outgoing_count == 1:
                    track = parent_track
                else:
                    self.track_count += 1
                    track = self.track_count
                    self.db.execute(
                        "INSERT INTO track_info VALUES (?,?,?)",
                        (track, lineage if parent_track else track, parent_track),
                    )
                self.db.execute("INSERT INTO assignments VALUES (?,?)", (oid, track))
            self.db.commit()

    def observations(self):
        return self.db.execute("""SELECT o.id,a.track_id,o.z,o.y,o.x,o.source
            FROM observations o JOIN assignments a ON a.object_id=o.id ORDER BY o.t,o.id""")

    def links(self, seconds_per_frame=None):
        for a, b, kind, frames, z, y, x in self.db.execute("""
            SELECT l.source,l.target,l.kind,q.t-p.t,q.z-p.z,q.y-p.y,q.x-p.x
            FROM links l JOIN observations p ON p.id=l.source JOIN observations q ON q.id=l.target
            ORDER BY l.source,l.target
        """):
            distance = math.sqrt(z * z + y * y + x * x)
            elapsed = frames * (seconds_per_frame or 1.0)
            yield a, b, kind, frames, elapsed, distance, distance / elapsed

    def divisions(self):
        return self.db.execute("""SELECT l.source,min(l.target),max(l.target),a.track_id
            FROM links l JOIN assignments a ON a.object_id=l.source WHERE l.kind='division'
            GROUP BY l.source,a.track_id HAVING count(*)=2 ORDER BY l.source""")

    def tracks(self, seconds_per_frame=None):
        rows = self.db.execute("""SELECT a.track_id,tr.lineage_id,tr.parent_track_id,o.t,o.z,o.y,o.x
            FROM assignments a JOIN observations o ON o.id=a.object_id
            JOIN track_info tr ON tr.track_id=a.track_id ORDER BY a.track_id,o.t,o.id""")
        for track, points in groupby(rows, lambda r: r[0]):
            first = previous = next(points)
            count, path, maximum = 1, 0.0, None
            for point in points:
                distance = math.dist(previous[4:], point[4:])
                elapsed = (point[3] - previous[3]) * (seconds_per_frame or 1.0)
                path += distance
                maximum = max(maximum or 0.0, distance / elapsed)
                previous = point
                count += 1
            duration = (previous[3] - first[3]) * (seconds_per_frame or 1.0)
            displacement = math.dist(first[4:], previous[4:])
            yield (
                track,
                first[1],
                first[2],
                first[3],
                previous[3],
                count,
                duration,
                path,
                displacement,
                path / duration if duration else None,
                maximum,
                displacement / path if path else None,
            )


class TrackingRun:
    """One optional label-set result; failure cannot undo base measurements."""

    def __init__(
        self, path, image_id, label_set_id, name, object_type, settings, seconds, log
    ):
        self.image_id, self.label_set_id, self.name = image_id, label_set_id, name
        self.path, self.object_type = path, object_type
        self.settings, self.seconds, self.log = settings, seconds, log
        self.store = None
        self.result = dict(label_set_id=label_set_id, label_name=name, status="pending")
        try:
            self.store = TrackingStore(path)
        except Exception as exc:
            self.fail(exc)

    def fail(self, exc):
        memory = (
            isinstance(exc, MemoryError)
            or "outofmemory" in type(exc).__name__.lower()
            or "out of memory" in str(exc).lower()
        )
        self.result.update(
            status="failed",
            error_type=type(exc).__name__,
            reason=str(exc),
            failure_kind="memory" if memory else "tracking_error",
        )
        if self.log:
            self.log(
                f"WARNING: Tracking FAILED {'due to memory' if memory else 'due to a tracking error'} for {self.name} "
                f"(label_set_id={self.label_set_id}): {type(exc).__name__}: {exc}. "
                "Segmentation and measurements are retained; no partial tracking results are published for this label set."
            )

    def add(self, p):
        if self.result["status"] == "pending":
            try:
                self.store.add(p)
            except Exception as exc:
                self.fail(exc)
                self.close()

    def close(self):
        if self.store is not None:
            try:
                self.store.close()
            except Exception as exc:
                if self.log:
                    self.log(
                        f"WARNING: Cannot close tracking spool for {self.name}: {exc}"
                    )
            finally:
                self.store = None

    def seal(self):
        if self.result["status"] == "pending":
            try:
                self.store.flush()
                self.result["observation_count"] = self.store.count
            except Exception as exc:
                self.fail(exc)
        self.close()

    def finish(self, db, database_format, offset):
        from .measurement_extensions import Batches

        counts, track_count, transaction = {}, 0, False
        try:
            if self.result["status"] == "pending":
                if self.store is None:
                    self.store = TrackingStore(self.path, create=False)
                self.result["observation_count"] = self.store.count
                resources = snapshot(include_gpu=False)
                self.result.update(
                    ram_limit_bytes=resources.ram_limit,
                    ram_available_bytes=resources.ram_available,
                )
                self.store.build(
                    radius=self.settings.tracking_distance_um,
                    max_gap=self.settings.tracking_max_gap,
                    divisions=self.settings.tracking_divisions,
                    object_type=self.object_type,
                )
                # All ordinary measurements and geometry are already committed.
                # Each label set's tracking rows commit together or roll back.
                db.execute("BEGIN TRANSACTION")
                transaction = True
                sink = Batches(db, database_format)
                divide = self.settings.tracking_divisions and self.object_type in {
                    "cells",
                    "nuclei",
                }
                sink.add(
                    "tracking_parameters",
                    (
                        self.label_set_id,
                        self.settings.tracking_distance_um,
                        self.settings.tracking_max_gap,
                        divide,
                        "sparse_two_stage_lap_v1_disk",
                    ),
                )
                time_unit = "second" if self.seconds else "frame"
                for track, lineage, parent, *values in self.store.tracks(self.seconds):
                    sink.add(
                        "tracks",
                        (
                            track + offset,
                            self.image_id,
                            self.label_set_id,
                            lineage + offset,
                            parent + offset if parent else None,
                            *values,
                            time_unit,
                            f"micrometer_per_{time_unit}",
                        ),
                    )
                for oid, track, *values in self.store.observations():
                    sink.add("track_observations", (oid, track + offset, *values))
                for row in self.store.links(self.seconds):
                    sink.add("temporal_links", (*row, time_unit))
                for parent, a, b, track in self.store.divisions():
                    sink.add("cell_divisions", (parent, a, b, track + offset))
                sink.flush()
                db.execute("COMMIT")
                transaction = False
                counts = dict(sink.counts)
                track_count = self.store.track_count
                self.result.update(status="complete", track_count=track_count)
        except Exception as exc:
            if transaction:
                with suppress(Exception):
                    db.execute("ROLLBACK")
            self.fail(exc)
        finally:
            self.close()
        # Persist explicit completeness alongside the surviving measurements.
        # Metadata/log still record the failure if the DB cannot store status.
        try:
            import json

            db.execute("BEGIN TRANSACTION")
            db.execute(
                "INSERT INTO tracking_status VALUES (?,?,?,?)",
                (
                    self.label_set_id,
                    self.result["status"],
                    self.result.get("error_type"),
                    json.dumps(self.result, sort_keys=True),
                ),
            )
            db.execute("COMMIT")
            counts["tracking_status"] = 1
        except Exception as exc:
            with suppress(Exception):
                db.execute("ROLLBACK")
            if self.log:
                self.log(
                    f"WARNING: Cannot write tracking status to measurement database: {exc}; see output metadata."
                )
        return self.result, counts, track_count
