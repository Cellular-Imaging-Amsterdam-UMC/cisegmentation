"""Repeatable validation using an existing runtime; never builds an image.

Run pytest for unit/compatibility coverage. This tool separately validates real
GPU Spotiflow, public CTC tracking, or extension export over completed 40K labels.
Source images/databases are read-only; all generated files go under --output.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import psutil
from scipy.ndimage import center_of_mass, sum_labels

from cisegmentation.measurement_extensions import (
    colocalization_values,
    connect_database,
    thresholds,
    write_extensions,
)
from cisegmentation.ome_zarr_io import ImageResource, enumerate_resources, read_image
from cisegmentation.resources import snapshot
from cisegmentation.settings import SegmentationSettings
from cisegmentation.tracking import link_observations


def public_colocalization(folder, output):
    """Compare bounded object statistics with independent whole-ROI calculations."""
    results = []
    for source in sorted(Path(folder).glob("CBS*RGM.ome.zarr")):
        image = read_image(ImageResource(source), lazy=True)
        height, width = image.data.shape[-2:]
        limits = thresholds(image, 0, [0, 1, 2])
        # The benchmark ROI is one complete small frame, including background;
        # this checks coefficients and does not assert cell segmentation accuracy.
        labels = np.ones((1, height, width), dtype="u1")
        values = colocalization_values(
            image, labels, 0, 1, [0, 0, 0, 1, height, width], (0, 1), limits
        )
        red = np.asarray(image.data[0, 0, 0], dtype=np.float64)
        green = np.asarray(image.data[0, 1, 0], dtype=np.float64)
        independent = [
            float(np.corrcoef(red.ravel(), green.ravel())[0, 1]),
            float(red[green > limits[1][0]].sum() / red.sum()),
            float(green[red > limits[0][0]].sum() / green.sum()),
        ]
        np.testing.assert_allclose(values[:3], independent, rtol=1e-12, atol=1e-12)
        assert values[3] == height * width and values[4] is None
        empty_blue = colocalization_values(
            image, labels, 0, 1, [0, 0, 0, 1, height, width], (0, 2), limits
        )
        assert empty_blue[0] is None and empty_blue[2] is None
        assert "constant_channel" in empty_blue[4] and "zero_signal" in empty_blue[4]
        results.append(
            {
                "dataset": source.name,
                "nominal_percent": image.attrs["publicdata"][
                    "nominal_colocalization_percent"
                ],
                "pearson_r": values[0],
                "manders_red_in_green": values[1],
                "manders_green_in_red": values[2],
                "thresholds": [limits[0][0], limits[1][0]],
                "pixels": values[3],
                "independent_calculation_agrees": True,
                "empty_blue_null_handled": True,
            }
        )
        print(f"Colocalization {source.name}: Pearson {values[0]:.4f}", flush=True)
    assert len(results) == 10, "Expected ten CBS benchmark images"
    from scipy.stats import spearmanr

    rank = float(
        spearmanr(
            [r["nominal_percent"] for r in results], [r["pearson_r"] for r in results]
        ).statistic
    )
    assert rank > 0.95, f"Unexpected colocalization trend {rank}"
    record = {
        "source": "https://colocalization-benchmark.com/downloads/",
        "datasets": results,
        "nominal_vs_pearson_spearman": rank,
        "caveat": "Whole-frame benchmark ROIs including background. Nominal overlap is not expected to equal thresholded directional Manders.",
    }
    (output / "public-colocalization.json").write_text(json.dumps(record, indent=2))
    return record


def public_spots(archive, output):
    """Assess linking on independent ISBI reference positions, in pixels/frames.

    This isolates assignment from image detection. Physical calibration is not
    provided in this archive: radius=20 means benchmark pixels, never micrometres.
    Author-generated _Tracks.xml files are deliberately excluded from truth.
    """
    results = []
    with zipfile.ZipFile(archive) as data:
        names = sorted(
            n
            for n in data.namelist()
            if n.endswith(".xml") and not n.endswith("_Tracks.xml")
        )
        for name in names:
            root = ET.fromstring(data.read(name))
            truth = root.find("TrackContestISBI2012")
            if truth is None:
                continue
            observations, expected, truth_identity = [], set(), {}
            for reference_id, particle in enumerate(truth.findall("particle"), 1):
                previous = None
                for detection in particle.findall("detection"):
                    value = detection.attrib
                    oid = len(observations) + 1
                    observations.append(
                        {
                            "id": oid,
                            "t": int(value["t"]),
                            "size": 1,
                            "position": np.array(
                                [
                                    float(value["z"]),
                                    float(value["y"]),
                                    float(value["x"]),
                                ]
                            ),
                        }
                    )
                    truth_identity[oid] = reference_id
                    if previous is not None:
                        expected.add((previous, oid))
                    previous = oid
            started = time.perf_counter()
            tracks, assignments, links = link_observations(
                observations,
                radius=20,
                max_gap=2,
                divisions=True,
                object_type="spots",
            )
            predicted = {(a, b) for a, b, *_ in links}
            correct = predicted & expected
            assert all(kind != "division" for _, _, kind, *_ in links)
            assert len({a for a, b, *_ in links}) == len(links), "Spot split"
            assert len({b for a, b, *_ in links}) == len(links), "Spot merge"
            assert len(assignments) == len(observations)
            results.append(
                {
                    "dataset": name,
                    "scenario": truth.attrib["scenario"],
                    "density": truth.attrib["density"],
                    "observations": len(observations),
                    "reference_tracks": len(truth.findall("particle")),
                    "predicted_tracks": len(tracks),
                    "reference_links": len(expected),
                    "predicted_links": len(predicted),
                    "correct_adjacent_links": len(correct),
                    "edge_precision": len(correct) / max(1, len(predicted)),
                    "edge_recall": len(correct) / max(1, len(expected)),
                    "runtime_seconds": time.perf_counter() - started,
                    "tracking_only": True,
                    "spot_splits": 0,
                    "spot_merges": 0,
                    "settings": {"radius_pixels": 20, "max_gap": 2},
                }
            )
            print(
                f"Spot benchmark {truth.attrib['scenario']}/{truth.attrib['density']}: precision {results[-1]['edge_precision']:.3f}, recall {results[-1]['edge_recall']:.3f}",
                flush=True,
            )
    record = {
        "source": "https://zenodo.org/records/14043236/",
        "archive": str(archive),
        "datasets": results,
        "caveat": "Linking reference coordinates only; no image-detection accuracy assessment. Pixel/frame units, not physical calibration.",
    }
    assert len(results) == 6, "Expected six ISBI simulation reference datasets"
    (output / "public-spots.json").write_text(json.dumps(record, indent=2))
    return record


def public_lineage(archive, output):
    """Track reference marker positions without using truth IDs for assignment."""
    from PIL import Image

    with zipfile.ZipFile(archive) as data:
        tracks_name = next(
            n for n in data.namelist() if n.endswith("01_GT/TRA/man_track.txt")
        )
        records = [
            tuple(map(int, line.split()))
            for line in data.read(tracks_name).decode().splitlines()
            if line.strip()
        ]
        truth = {r[0]: r for r in records}
        masks = sorted(
            n
            for n in data.namelist()
            if "01_GT/TRA/man_track" in n and n.endswith(".tif")
        )
        observations, truth_label = [], {}
        for t, name in enumerate(masks):
            with Image.open(io.BytesIO(data.read(name))) as frame:
                mask = np.asarray(frame)
            labels = np.unique(mask)
            labels = labels[labels > 0]
            positions = center_of_mass(np.ones(mask.shape, dtype="u1"), mask, labels)
            sizes = sum_labels(np.ones(mask.shape, dtype="u1"), mask, labels)
            for label, (y, x), size in zip(labels, positions, sizes):
                oid = len(observations) + 1
                observations.append(
                    {
                        "id": oid,
                        "t": t,
                        "position": np.array([0, y * 0.645, x * 0.645]),
                        "size": int(size),
                    }
                )
                truth_label[oid] = (int(label), t)
        tracks, _assignments, links = link_observations(
            observations,
            radius=20,
            max_gap=2,
            divisions=True,
            object_type="nuclei",
            seconds_per_frame=1800,
        )

        def is_true(a, b, kind):
            la, ta = truth_label[a]
            lb, tb = truth_label[b]
            return (
                truth[lb][3] == la and truth[la][2] == ta and truth[lb][1] == tb
                if kind == "division"
                else la == lb
            )

        predicted = Counter(l[2] for l in links)
        correct = Counter(l[2] for l in links if is_true(l[0], l[1], l[2]))
        daughters = {}
        for label, start, end, parent in records:
            if parent:
                daughters.setdefault(parent, []).append(label)
        recovered = set()
        predicted_daughters = {}
        for a, b, kind, *_ in links:
            if kind == "division":
                predicted_daughters.setdefault(a, []).append(b)
        for parent, children in predicted_daughters.items():
            p = truth_label[parent][0]
            if sorted(truth_label[c][0] for c in children) == sorted(
                daughters.get(p, [])
            ):
                recovered.add(p)
        expected_edges = sum(max(0, end - start) for _, start, end, _ in records)
        record = {
            "source": "https://celltrackingchallenge.net/2d-datasets/",
            "dataset": "Fluo-N2DL-HeLa",
            "sequence": "01",
            "archive_sha256": None,
            "input": "reference tracking marker centroids; no image segmentation accuracy assessment",
            "frames": len(masks),
            "observations": len(observations),
            "tracks": len(tracks),
            "predicted_links": dict(predicted),
            "correct_links": dict(correct),
            "link_precision": sum(correct.values()) / max(1, len(links)),
            "ordinary_edge_recall": (correct["link"] + correct["gap"])
            / max(1, expected_edges),
            "reference_divisions": len(daughters),
            "recovered_divisions": len(recovered),
            "division_precision": len(recovered) / max(1, len(predicted_daughters)),
            "settings": {
                "distance_um": 20,
                "max_missed_frames": 2,
                "pixel_um": 0.645,
                "seconds_per_frame": 1800,
            },
            "caveat": "Distance/size heuristics cannot guarantee correct biological lineages; marker sizes may differ from true mask sizes.",
        }
        with Path(archive).open("rb") as handle:
            record["archive_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        assert len({l[1] for l in links}) == len(links), "Unsupported merge"
        assert predicted.get("division", 0) % 2 == 0, "Incomplete binary division"
        (output / "public-lineage.json").write_text(json.dumps(record, indent=2))
        return record


def large_extensions(args, output):
    source = Path(args.large_input)
    labels = Path(args.large_labels)
    original = Path(args.large_measurements)
    staged = output / "large-stage"
    staged.mkdir(exist_ok=True)
    calibration_note = "source calibration"
    if args.synthetic_pixel_um is not None:
        view = staged / "synthetic-calibration.ome.zarr"
        view.mkdir(exist_ok=True)
        attrs = json.loads((source / ".zattrs").read_text())
        multiscale = attrs["multiscales"][0]
        axes = multiscale["axes"]
        for i, axis in enumerate(axes):
            if axis["name"] in "zyx":
                axis["unit"] = "micrometer"
                for dataset in multiscale["datasets"]:
                    for transform in dataset.get("coordinateTransformations", []):
                        if transform["type"] == "scale" and axis["name"] in "yx":
                            transform["scale"][i] *= args.synthetic_pixel_um
        attrs["cisegmentation_validation"] = {
            "synthetic_pixel_um": args.synthetic_pixel_um,
            "original_source": str(source),
            "biological_calibration": False,
        }
        for child in source.iterdir():
            if child.name != ".zattrs" and not (view / child.name).exists():
                os.symlink(child, view / child.name, target_is_directory=child.is_dir())
        (view / ".zattrs").write_text(json.dumps(attrs))
        source = view
        calibration_note = (
            f"synthetic {args.synthetic_pixel_um} um/pixel; resource test only"
        )
    database = staged / original.name
    shutil.copy2(original, database)
    settings = SegmentationSettings(
        cell_model="skip",
        nucleus_model="stardist:SD_Nuclei_Versatile",
        spatial_measurements=True,
    )
    summary, _ = write_extensions(
        database,
        "duckdb",
        enumerate_resources(source),
        labels,
        settings,
        stage_dir=staged,
        log=print,
    )
    db = connect_database(database, "duckdb", read_only=True)
    count = db.execute("SELECT count(*) FROM objects").fetchone()[0]
    assert (
        db.execute("SELECT count(*) FROM spatial_measurements").fetchone()[0] == count
    )
    assert db.execute(
        "SELECT value FROM schema_info WHERE key='schema_version'"
    ).fetchone() == ("5",)
    sample = db.execute(
        "SELECT min(nearest_distance_um),max(neighbor_count) FROM spatial_measurements"
    ).fetchone()
    db.close()
    record = {
        "reuses_completed_segmentation": True,
        "new_model_inference": False,
        "objects": count,
        "source_shape": list(read_image(ImageResource(source), lazy=True).data.shape),
        "new_features": ["spatial"],
        "geometry_scope": "tested separately; full 40K geometry not exported by this check",
        "spatial_range": sample,
        "calibration": calibration_note,
        "summary": summary,
        "database": str(database),
    }
    (output / "large.json").write_text(json.dumps(record, indent=2))
    return record


def gpu_spots(fixture, output):
    import torch
    import zarr

    from cisegmentation.engine import run_workflow
    from cisegmentation.resources import apply_gpu_limit

    assert torch.cuda.is_available()
    apply_gpu_limit(torch)
    image = read_image(ImageResource(Path(fixture)), lazy=True)
    raw = np.asarray(image.data[0, 1:2, :1, :256, :256])
    result = []
    for model, z_size in [("spotiflow:general", 1), ("spotiflow:smfish_3d", 24)]:
        source = output / (model.split(":")[-1] + ".ome.zarr")
        root = zarr.open_group(str(source), mode="w")
        shape = (3, 1, z_size, *raw.shape[-2:])
        array = root.create_dataset(
            "0", shape=shape, chunks=(1, 1, 1, 64, 64), dtype="u2"
        )
        # Smooth axial intensity supplies a positive volumetric spot fixture.
        axial = (
            np.exp(-(((np.arange(z_size) - (z_size - 1) / 2) / 2.5) ** 2))
            if z_size > 1
            else np.ones(1)
        )
        for t in range(3):
            array[t, 0] = np.roll(
                np.broadcast_to(raw[0], (z_size, *raw.shape[-2:]))
                * axial[:, None, None],
                t,
                axis=2,
            ).astype("u2")
        root.attrs["multiscales"] = [
            {
                "version": "0.4",
                "axes": [
                    {
                        "name": a,
                        "type": "time"
                        if a == "t"
                        else "channel"
                        if a == "c"
                        else "space",
                        **(
                            {"unit": "micrometer"}
                            if a in "zyx"
                            else {"unit": "second"}
                            if a == "t"
                            else {}
                        ),
                    }
                    for a in "tczyx"
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [0.5, 1, 1, 0.5, 0.5]}
                        ],
                    }
                ],
            }
        ]
        settings = SegmentationSettings(
            cell_model="skip",
            nucleus_model="skip",
            foci_model_1=model,
            device="cuda",
            tracking=True,
            export_geometry=True,
            spatial_measurements=True,
            max_inference_workers=1,
            max_measurement_workers=1,
        )
        outputs = run_workflow(
            source, output / model.split(":")[-1], settings, log=print
        )
        db = connect_database(
            next(p for p in outputs if "_measurements." in p.name),
            "duckdb",
            read_only=True,
        )
        counts = db.execute("SELECT count(*) FROM objects").fetchone()[0]
        assert counts > 0, f"No positive detections: {model}"
        assert (
            db.execute("SELECT count(*) FROM point_localizations").fetchone()[0]
            == counts
        )
        assert db.execute("SELECT count(*) FROM cell_divisions").fetchone() == (0,)
        assert (
            db.execute("SELECT count(*) FROM track_observations").fetchone()[0]
            == counts
        )
        fractional = db.execute(
            "SELECT count(*) FROM point_localizations WHERE abs(x_px-round(x_px))>0.000001 OR abs(y_px-round(y_px))>0.000001"
        ).fetchone()[0]
        assert fractional > 0, "Native subpixel positions were rounded away"
        db.close()
        result.append(
            {
                "model": model,
                "objects": counts,
                "subpixel_points": fractional,
                "gpu": torch.cuda.get_device_name(),
                "outputs": list(map(str, outputs)),
            }
        )
    (output / "gpu-spots.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--public-lineage-zip")
    parser.add_argument("--public-spots-zip")
    parser.add_argument("--public-colocalization")
    parser.add_argument("--large-input")
    parser.add_argument("--large-labels")
    parser.add_argument("--large-measurements")
    parser.add_argument("--gpu-fixture")
    parser.add_argument(
        "--synthetic-pixel-um",
        type=float,
        help="Linux-only metadata view for uncalibrated large synthetic resource tests; never changes the original",
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "resources_start": snapshot().to_dict(),
        "started": time.time(),
        "status": "failed",
    }
    peak = [0]
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.2):
            process = psutil.Process()
            peak[0] = max(
                peak[0],
                process.memory_info().rss
                + sum(
                    p.memory_info().rss
                    for p in process.children(recursive=True)
                    if p.is_running()
                ),
            )

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        if args.public_lineage_zip:
            record["public_lineage"] = public_lineage(args.public_lineage_zip, output)
        if args.public_spots_zip:
            record["public_spots"] = public_spots(args.public_spots_zip, output)
        if args.public_colocalization:
            record["public_colocalization"] = public_colocalization(
                args.public_colocalization, output
            )
        if args.gpu_fixture:
            record["gpu"] = gpu_spots(args.gpu_fixture, output)
        if args.large_input:
            if not args.large_labels or not args.large_measurements:
                parser.error(
                    "--large-input requires --large-labels and --large-measurements"
                )
            record["large"] = large_extensions(args, output)
        record["status"] = "passed"
    finally:
        stop.set()
        watcher.join()
        record["peak_process_tree_rss_bytes"] = peak[0]
        record["runtime_seconds"] = time.time() - record["started"]
        (output / "validation.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
