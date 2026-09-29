"""Sparse two-stage LAP linking of immutable, frame-local object observations.

Frame-to-frame links are followed by segment gap closing and division links.
No dense all-object cost matrix, mask relabeling, or merging is performed.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree

from .resources import snapshot


def sparse_assignment(left, right, radius, accept=None):
    """Minimize gated squared distance, with one private unmatched node per row."""
    if not left or not right:
        return []
    tree = cKDTree(np.asarray([p["position"] for p in right]))
    rows, cols, costs = [], [], []
    budget = max(1, snapshot().ram_available // 8)
    for i, source in enumerate(left):
        neighbors = tree.query_ball_point(
            source["position"], radius, return_sorted=True
        )
        if (len(rows) + len(neighbors) + len(left)) * 160 > budget:
            raise MemoryError(
                "Tracking candidate graph exceeds live RAM budget; reduce tracking distance or request more RAM"
            )
        for j in neighbors:
            target = right[j]
            if accept is not None and not accept(source, target):
                continue
            distance = float(
                np.linalg.norm(
                    np.asarray(source.get("predicted_position", source["position"]))
                    - target["position"]
                )
            )
            rows.append(i)
            cols.append(j)
            # Sparse matching removes literal zeros. Positive epsilon also
            # gives stable tie preference in the sorted observation order.
            costs.append(
                (distance / radius) ** 2 + 1e-10 * (1 + ((i + 1) * (j + 1)) % 997)
            )
        rows.append(i)
        cols.append(len(right) + i)
        costs.append(1.05)
        if i % 256 == 0 and len(rows) * 160 > snapshot().ram_available // 8:
            raise MemoryError("Tracking candidate graph no longer fits available RAM")
    matrix = coo_matrix(
        (costs, (rows, cols)), shape=(len(left), len(right) + len(left))
    ).tocsr()
    ii, jj = min_weight_full_bipartite_matching(matrix)
    return [(left[int(i)], right[int(j)]) for i, j in zip(ii, jj) if j < len(right)]


def link_observations(
    observations,
    *,
    radius=20.0,
    max_gap=2,
    divisions=False,
    object_type="spots",
    seconds_per_frame=None,
):
    """Return track segments, observations, temporal links and division events.

    Division hypotheses are adjacent-frame, binary splits of cells/nuclei. Both
        daughters must lie inside the configured distance gate and their combined
        raster size must be 0.5--1.8 times the parent's. Spots never branch.
    """
    ordered = sorted(observations, key=lambda p: (p["t"], p["id"]))
    if len(ordered) * 1024 > snapshot().ram_available // 4:
        raise MemoryError("Tracking observations exceed live RAM budget")
    by_frame = defaultdict(list)
    by_id = {p["id"]: p for p in ordered}
    for p in ordered:
        by_frame[p["t"]].append(p)
    incoming, outgoing, kinds = {}, defaultdict(list), {}

    def connect(a, b, kind):
        if b["id"] in incoming:
            raise RuntimeError("Tracker attempted an unsupported merge")
        incoming[b["id"]] = a["id"]
        outgoing[a["id"]].append(b["id"])
        kinds[(a["id"], b["id"])] = kind

    # Stage 1: adjacent-frame assignment with births/deaths as alternatives.
    for t in sorted(by_frame):
        candidates = []
        for p in by_frame[t]:
            candidate = dict(p)
            previous = by_id.get(incoming.get(p["id"]))
            if previous is not None:
                candidate["predicted_position"] = np.asarray(p["position"]) + (
                    np.asarray(p["position"]) - previous["position"]
                ) / (t - previous["t"])
            candidates.append(candidate)
        for a, b in sparse_assignment(candidates, by_frame.get(t + 1, []), radius):
            connect(a, b, "link")

    # Stage 2: adjacent division hypotheses precede missed-frame recovery,
    # so a historical unmatched end cannot steal a plausible daughter.
    if divisions and object_type in {"cells", "nuclei"}:
        for t in sorted(by_frame):
            left = [
                p
                for p in by_frame[t]
                if len(outgoing[p["id"]]) == 1
                and by_id[outgoing[p["id"]][0]]["t"] == t + 1
            ]
            right = [p for p in by_frame.get(t + 1, []) if p["id"] not in incoming]

            def plausible(parent, daughter):
                first = by_id[outgoing[parent["id"]][0]]
                size = parent.get("size", 0)
                if size <= 0:
                    return False
                combined = (first.get("size", 0) + daughter.get("size", 0)) / size
                return (
                    0.5 <= combined <= 1.8
                    and max(first.get("size", 0), daughter.get("size", 0))
                    <= 1.25 * size
                )

            for a, b in sparse_assignment(left, right, radius, plausible):
                connect(a, b, "division")
                kinds[(a["id"], outgoing[a["id"]][0])] = "division"

    for gap in range(2, max_gap + 2):
        for t in sorted(by_frame):
            left = [p for p in by_frame[t] if not outgoing[p["id"]]]
            right = [p for p in by_frame.get(t + gap, []) if p["id"] not in incoming]
            for a, b in sparse_assignment(left, right, radius):
                connect(a, b, "gap")

    assignments, segments, parents, lineages = {}, defaultdict(list), {}, {}
    next_track = 0
    for p in ordered:
        predecessor = incoming.get(p["id"])
        if predecessor is not None and len(outgoing[predecessor]) == 1:
            track = assignments[predecessor]
        else:
            next_track += 1
            track = next_track
            parents[track] = (
                assignments[predecessor] if predecessor is not None else None
            )
            lineages[track] = lineages[parents[track]] if parents[track] else track
        assignments[p["id"]] = track
        segments[track].append(p)

    temporal_links = []
    for (a, b), kind in sorted(kinds.items()):
        p, q = by_id[a], by_id[b]
        distance = float(np.linalg.norm(np.asarray(p["position"]) - q["position"]))
        frames = q["t"] - p["t"]
        elapsed = frames * seconds_per_frame if seconds_per_frame else float(frames)
        temporal_links.append(
            (a, b, kind, frames, elapsed, distance, distance / elapsed)
        )

    summaries = []
    for track, points in sorted(segments.items()):
        distances = [
            float(np.linalg.norm(np.asarray(a["position"]) - b["position"]))
            for a, b in pairwise(points)
        ]
        durations = [
            (b["t"] - a["t"]) * (seconds_per_frame or 1.0) for a, b in pairwise(points)
        ]
        path = sum(distances)
        displacement = float(
            np.linalg.norm(np.asarray(points[-1]["position"]) - points[0]["position"])
        )
        duration = (points[-1]["t"] - points[0]["t"]) * (seconds_per_frame or 1.0)
        speeds = [d / dt for d, dt in zip(distances, durations)]
        summaries.append(
            (
                track,
                lineages[track],
                parents[track],
                points[0]["t"],
                points[-1]["t"],
                len(points),
                duration,
                path,
                displacement,
                path / duration if duration else None,
                max(speeds) if speeds else None,
                displacement / path if path else None,
            )
        )
    return summaries, [(p["id"], assignments[p["id"]]) for p in ordered], temporal_links
