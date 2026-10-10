"""Exact pixel-edge polygons and portable WKB, without spatial extensions.

Pixels have integer centre coordinates; their edges lie at half integers.
Boundary segments stay in bounded RAM and spill to SQLite when necessary.
A single object's outlines are assembled
only after checking their size against live RAM. Masks remain authoritative.
"""

from __future__ import annotations

import sqlite3
import struct
from collections import defaultdict
from itertools import pairwise

import numpy as np

from .resources import ResourceMonitor


def point_wkb(x, y, z=None):
    return struct.pack("<BI", 1, 1 if z is None else 1001) + struct.pack(
        "<dd" if z is None else "<ddd", *([x, y] if z is None else [x, y, z])
    )


def _area(ring):
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in pairwise(ring)) / 2


def _inside(point, ring):
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in pairwise(ring):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def polygon_wkb(rings):
    outers = [(r, []) for r in rings if _area(r) > 0]
    holes = [r for r in rings if _area(r) < 0]
    for hole in holes:
        a, b = hole[:2]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = abs(dx) + abs(dy)
        interior = (
            (a[0] + b[0]) / 2 + dy / length * 0.125,
            (a[1] + b[1]) / 2 - dx / length * 0.125,
        )
        containers = [
            (abs(_area(r)), i)
            for i, (r, _) in enumerate(outers)
            if _inside(interior, r)
        ]
        if not containers:
            raise ValueError("Polygon hole has no containing outer ring")
        outers[min(containers)[1]][1].append(hole)
    if not outers:
        raise ValueError("No polygon outer boundary")

    def polygon(outer, inner):
        result = bytearray(struct.pack("<BII", 1, 3, len(inner) + 1))
        for ring in [outer, *inner]:
            result.extend(struct.pack("<I", len(ring)))
            for x, y in ring:
                result.extend(struct.pack("<dd", x, y))
        return bytes(result)

    polygons = [polygon(r, h) for r, h in outers]
    if len(polygons) == 1:
        return polygons[0], "POLYGON"
    return struct.pack("<BII", 1, 6, len(polygons)) + b"".join(polygons), "MULTIPOLYGON"


def trace_rings(edges):
    successors = defaultdict(list)
    for x1, y1, x2, y2 in edges:
        successors[(x1, y1)].append((x2, y2))
    rings = []
    while successors:
        first = next(iter(successors))
        current, previous = first, None
        ring = [first]
        while True:
            candidates = successors.get(current)
            if not candidates:
                raise ValueError("Open mask boundary")
            if previous is None or len(candidates) == 1:
                nxt = candidates[0]
            else:
                dx, dy = current[0] - previous[0], current[1] - previous[1]
                # Right-hand pixel boundary rule separates diagonal components.
                nxt = max(
                    candidates,
                    key=lambda p: dx * (p[1] - current[1]) - dy * (p[0] - current[0]),
                )
            candidates.remove(nxt)
            if not candidates:
                del successors[current]
            previous, current = current, nxt
            ring.append(current)
            if current == first:
                break
        # A component can touch itself at a pixel corner. Split repeated
        # vertices into simple cycles rather than exporting a bow-tie ring.
        path, indexes, cycles = [], {}, []
        for vertex in ring:
            if vertex in indexes:
                start = indexes[vertex]
                cycles.append(path[start:] + [vertex])
                for discarded in path[start + 1 :]:
                    indexes.pop(discarded)
                path = path[: start + 1]
            else:
                indexes[vertex] = len(path)
                path.append(vertex)
        for cycle in cycles:
            compact = []
            for i, p in enumerate(cycle[:-1]):
                a, b = cycle[i - 1] if i else cycle[-2], cycle[i + 1]
                if (p[0] - a[0]) * (b[1] - p[1]) != (p[1] - a[1]) * (b[0] - p[0]):
                    compact.append((p[0] / 2, p[1] / 2))
            compact.append(compact[0])
            rings.append(compact)
    return rings


def mask_polygons(
    array,
    label,
    z,
    bbox,
    spool_path,
    block_size=512,
    *,
    return_bounds=False,
    monitor=None,
    edge_memory_bytes=None,
):
    """Extract one final label plane using bounded crops plus a one-pixel halo."""
    y0, x0, y1, x1 = map(int, bbox)
    monitor = monitor or ResourceMonitor()
    budget = min(16 * 1024**2, monitor.get().ram_available // 64)
    if edge_memory_bytes is not None:
        budget = min(budget, edge_memory_bytes)
    edges, spool, count = [], None, 0
    try:
        for y in range(y0, y1, block_size):
            for x in range(x0, x1, block_size):
                if monitor.get().ram_available < 16 * 1024**2:
                    raise MemoryError("Insufficient live RAM for polygon extraction")
                end_y, end_x = min(y1, y + block_size), min(x1, x + block_size)
                sy, sx = max(0, y - 1), max(0, x - 1)
                ey, ex = (
                    min(array.shape[-2], end_y + 1),
                    min(array.shape[-1], end_x + 1),
                )
                raw = np.asarray(array[z, sy:ey, sx:ex]) == label
                padded = np.pad(
                    raw,
                    (
                        (int(sy == 0), int(ey == array.shape[-2])),
                        (int(sx == 0), int(ex == array.shape[-1])),
                    ),
                )
                cy, cx = y - sy + int(sy == 0), x - sx + int(sx == 0)
                h, w = end_y - y, end_x - x
                centre = padded[cy : cy + h, cx : cx + w]
                neighbors = [
                    padded[cy - 1 : cy + h - 1, cx : cx + w],
                    padded[cy : cy + h, cx + 1 : cx + w + 1],
                    padded[cy + 1 : cy + h + 1, cx : cx + w],
                    padded[cy : cy + h, cx - 1 : cx + w - 1],
                ]
                for side, neighbor in enumerate(neighbors):
                    yy, xx = np.nonzero(centre & ~neighbor)
                    yy, xx = 2 * (yy + y), 2 * (xx + x)
                    if side == 0:
                        rows = zip(xx - 1, yy - 1, xx + 1, yy - 1)
                    elif side == 1:
                        rows = zip(xx + 1, yy - 1, xx + 1, yy + 1)
                    elif side == 2:
                        rows = zip(xx + 1, yy + 1, xx - 1, yy + 1)
                    else:
                        rows = zip(xx - 1, yy + 1, xx - 1, yy - 1)
                    rows = [tuple(map(int, r)) for r in rows]
                    count += len(rows)
                    if spool is None and count * 320 > budget:
                        spool = sqlite3.connect(spool_path)
                        # Scratch only; scientific results live in the parent DB.
                        spool.execute("PRAGMA journal_mode=OFF")
                        spool.execute("PRAGMA synchronous=OFF")
                        spool.execute("DROP TABLE IF EXISTS edges")
                        spool.execute(
                            "CREATE TABLE edges (x1 INTEGER,y1 INTEGER,x2 INTEGER,y2 INTEGER)"
                        )
                        spool.executemany("INSERT INTO edges VALUES (?,?,?,?)", edges)
                        edges.clear()
                    if spool is None:
                        edges.extend(rows)
                    else:
                        spool.executemany("INSERT INTO edges VALUES (?,?,?,?)", rows)
        if not count:
            return None
        if count * 320 > monitor.get().ram_available // 4:
            raise MemoryError(
                "One object's polygon vertices exceed live RAM budget; request more RAM (raster labels are preserved)"
            )
        rings = trace_rings(spool.execute("SELECT * FROM edges") if spool else edges)
        data, kind = polygon_wkb(rings)
        result = (data, kind, sum(_area(r) for r in rings))
        if return_bounds:
            result += (
                (
                    min(p[0] for r in rings for p in r),
                    min(p[1] for r in rings for p in r),
                    max(p[0] for r in rings for p in r),
                    max(p[1] for r in rings for p in r),
                ),
            )
        return result
    finally:
        if spool is not None:
            spool.close()
