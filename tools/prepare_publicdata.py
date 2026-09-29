"""Prepare retained public microscopy benchmarks as pyramidal NGFF 0.4/Zarr v2.

Run from the repository root: python -m tools.prepare_publicdata --publicdata publicdata
Downloads are deliberately separate: originals, checksums and licence information
must be retained. Existing complete outputs are validated, never overwritten.
The full-resolution pixels retain their source dtype and values. XY intensity
pyramids use 2x2 area means; reference masks use nearest-neighbour reduction.
Unspecified microscopy calibration stays unspecified, rather than using TIFF DPI.
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import re
import shutil
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import tifffile
import zarr
from numcodecs import Blosc
from PIL import Image


def decode_tiff(payload):
    """Pillow supplies LZW support without an optional imagecodecs installation."""
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image).copy()


def mean_downsample(block):
    """Preserve odd-edge coverage and source dtype with rounded area means."""
    padding = ((0, block.shape[0] % 2), (0, block.shape[1] % 2))
    if any(end for _, end in padding):
        block = np.pad(block, padding, mode="edge")
    means = (
        block.astype(np.float64)
        .reshape(block.shape[0] // 2, 2, block.shape[1] // 2, 2)
        .mean(axis=(1, 3))
    )
    if np.issubdtype(block.dtype, np.integer):
        means = np.rint(means)
    return means.astype(block.dtype)


def xml_metadata(name, shape, dtype, channels, pixel_um=None, seconds=None):
    """Only declare physical units supported by the source documentation."""
    namespace = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
    ET.register_namespace("", namespace)
    root = ET.Element(f"{{{namespace}}}OME")
    image = ET.SubElement(root, "Image", ID="Image:0", Name=name)
    t, c, z, y, x = shape
    attributes = {
        "ID": "Pixels:0",
        "DimensionOrder": "XYZCT",
        "Type": np.dtype(dtype).name,
        "SizeX": str(x),
        "SizeY": str(y),
        "SizeZ": str(z),
        "SizeC": str(c),
        "SizeT": str(t),
    }
    if pixel_um is not None:
        attributes.update(
            PhysicalSizeX=str(pixel_um),
            PhysicalSizeY=str(pixel_um),
            PhysicalSizeXUnit="µm",
            PhysicalSizeYUnit="µm",
        )
    if seconds is not None:
        attributes.update(TimeIncrement=str(seconds), TimeIncrementUnit="s")
    pixels = ET.SubElement(image, "Pixels", **attributes)
    for i, (label, color) in enumerate(channels):
        rgba = (int(color, 16) << 8) | 255
        if rgba >= 2**31:
            rgba -= 2**32
        ET.SubElement(
            pixels,
            "Channel",
            ID=f"Channel:0:{i}",
            Name=label,
            SamplesPerPixel="1",
            Color=str(rgba),
        )
    ET.SubElement(pixels, "MetadataOnly")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def create_store(
    path,
    shape,
    dtype,
    channels,
    provenance,
    read_plane,
    *,
    pixel_um=None,
    seconds=None,
    is_label=False,
):
    """Bounded plane/chunk conversion followed by exact full-resolution checks."""
    path = Path(path)
    if path.exists():
        return validate_store(path, read_plane=read_plane)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise FileExistsError(
            f"Review interrupted conversion before retrying: {partial}"
        )
    partial.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(partial), mode="w")
    axes = [
        {"name": "t", "type": "time"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]
    if pixel_um is not None:
        for axis in axes[-2:]:
            axis["unit"] = "micrometer"
    if seconds is not None:
        axes[0]["unit"] = "second"
    arrays, datasets = [], []
    current_shape, factor = tuple(shape), 1
    while True:
        array = root.create_dataset(
            str(len(arrays)),
            shape=current_shape,
            dtype=dtype,
            chunks=(1, 1, 1, min(256, current_shape[-2]), min(256, current_shape[-1])),
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
        )
        array.attrs["_ARRAY_DIMENSIONS"] = [a["name"] for a in axes]
        arrays.append(array)
        datasets.append(
            {
                "path": str(len(arrays) - 1),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [
                            seconds or 1,
                            1,
                            1,
                            (pixel_um or 1) * factor,
                            (pixel_um or 1) * factor,
                        ],
                    }
                ],
            }
        )
        if max(current_shape[-2:]) <= 128:
            break
        current_shape = (
            *shape[:-2],
            (current_shape[-2] + 1) // 2,
            (current_shape[-1] + 1) // 2,
        )
        factor *= 2
    low = np.full(shape[1], np.inf)
    high = np.full(shape[1], -np.inf)
    for t in range(shape[0]):
        for c in range(shape[1]):
            for z in range(shape[2]):
                plane = np.asarray(read_plane(t, c, z))
                if plane.shape != shape[-2:] or plane.dtype != np.dtype(dtype):
                    raise ValueError(
                        f"Unexpected source plane {plane.shape}/{plane.dtype}"
                    )
                arrays[0][t, c, z] = plane
                low[c] = min(low[c], float(plane.min()))
                high[c] = max(high[c], float(plane.max()))
    for source, target in itertools.pairwise(arrays):
        for t in range(shape[0]):
            for c in range(shape[1]):
                for z in range(shape[2]):
                    for y in range(0, target.shape[-2], 256):
                        for x in range(0, target.shape[-1], 256):
                            block = source[
                                t, c, z, 2 * y : 2 * (y + 256), 2 * x : 2 * (x + 256)
                            ]
                            small = (
                                block[::2, ::2] if is_label else mean_downsample(block)
                            )
                            target[
                                t, c, z, y : y + small.shape[0], x : x + small.shape[1]
                            ] = small
    root.attrs.update(
        {
            "multiscales": [
                {
                    "version": "0.4",
                    "name": path.name.removesuffix(".ome.zarr"),
                    "axes": axes,
                    "datasets": datasets,
                    "type": "nearest" if is_label else "mean",
                }
            ],
            "omero": {
                "version": "0.4",
                "name": path.name,
                "channels": [
                    {
                        "label": label,
                        "color": color,
                        "active": True,
                        "window": {
                            "min": float(low[i]),
                            "max": float(high[i]),
                            "start": float(low[i]),
                            "end": float(high[i])
                            if high[i] > low[i]
                            else float(low[i] + 1),
                        },
                    }
                    for i, (label, color) in enumerate(channels)
                ],
            },
            "publicdata": dict(
                **provenance,
                calibration={
                    "pixel_size_um": pixel_um,
                    "seconds_per_frame": seconds,
                    "missing_units": "pixels and frame index; no assumed physical calibration",
                },
                conversion={
                    "format": "NGFF 0.4 / Zarr v2",
                    "axes": "TCZYX",
                    "full_resolution": "unchanged values and dtype",
                    "xy_pyramid": "nearest"
                    if is_label
                    else "rounded 2x2 area mean, edge replication",
                },
            ),
        }
    )
    (partial / "OME").mkdir()
    (partial / "OME" / "METADATA.ome.xml").write_bytes(
        xml_metadata(
            path.name,
            shape,
            dtype,
            channels,
            pixel_um,
            seconds,
        )
    )
    root.store.close()
    record = validate_store(partial, read_plane=read_plane)
    partial.rename(path)
    record["path"] = str(path)
    return record


def validate_store(path, *, read_plane=None):
    """Check every original plane, metadata/axes, and deterministic pyramid windows."""
    root = zarr.open_group(str(path), mode="r")
    try:
        if json.loads((Path(path) / ".zgroup").read_text())["zarr_format"] != 2:
            raise ValueError("OMERO input requires Zarr v2")
        scale = root.attrs["multiscales"][0]
        if scale["version"] != "0.4":
            raise ValueError("Expected NGFF 0.4")
        if [a["name"] for a in scale["axes"]] != list("tczyx"):
            raise ValueError("Unexpected axes")
        arrays = [root[d["path"]] for d in scale["datasets"]]
        count = 0
        for t in range(arrays[0].shape[0]):
            for c in range(arrays[0].shape[1]):
                for z in range(arrays[0].shape[2]):
                    if read_plane is not None:
                        np.testing.assert_array_equal(
                            arrays[0][t, c, z], read_plane(t, c, z)
                        )
                    count += 1
        is_label = scale["type"] == "nearest"
        for level, (source, target) in enumerate(itertools.pairwise(arrays), 1):
            for y, x in [
                (0, 0),
                (max(0, target.shape[-2] - 19), max(0, target.shape[-1] - 23)),
            ]:
                block = source[0, 0, 0, 2 * y : 2 * (y + 19), 2 * x : 2 * (x + 23)]
                expected = block[::2, ::2] if is_label else mean_downsample(block)
                np.testing.assert_array_equal(
                    target[
                        0, 0, 0, y : y + expected.shape[0], x : x + expected.shape[1]
                    ],
                    expected,
                    err_msg=f"Pyramid mismatch {path}, level {level}",
                )
        return {
            "path": str(path),
            "shape": list(arrays[0].shape),
            "dtype": str(arrays[0].dtype),
            "pyramid_shapes": [list(a.shape) for a in arrays],
            "original_planes_compared": count if read_plane is not None else 0,
            "channels": [c["label"] for c in root.attrs["omero"]["channels"]],
            "calibration": root.attrs["publicdata"]["calibration"],
            "validation": "passed",
        }
    finally:
        root.store.close()


def truth_csv(payload, path):
    tree = ET.fromstring(payload)
    truth = tree.find("TrackContestISBI2012")
    if truth is None:
        truth = tree
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reference_track_id", "timepoint", "z_px", "y_px", "x_px"])
        for track, particle in enumerate(truth.findall("particle"), 1):
            for detection in particle.findall("detection"):
                writer.writerow(
                    [
                        track,
                        detection.attrib["t"],
                        detection.attrib.get("z", 0),
                        detection.attrib["y"],
                        detection.attrib["x"],
                    ]
                )


def prepare(root):
    output = root / "ome-zarr"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    source_url = "https://zenodo.org/records/14043236/"
    for archive_name, selected in [
        ("SIMULATED-test.zip", "/test/"),
        ("REAL.zip", "/test/"),
    ]:
        archive_path = root / "ISBI-SubVFI" / archive_name
        with zipfile.ZipFile(archive_path) as archive:
            groups = {}
            for name in archive.namelist():
                if name.lower().endswith(".tif") and (
                    name.startswith("test/")
                    if archive_name.startswith("SIM")
                    else selected in name
                ):
                    groups.setdefault(str(PurePosixPath(name).parent), []).append(name)
            for folder, names in sorted(groups.items()):
                names.sort()
                name = (
                    ("ISBI-" + folder.split("/")[-1].replace(" snr 7 density ", "-"))
                    if archive_name.startswith("SIM")
                    else "REAL-" + folder.split("/")[1]
                )
                plane = decode_tiff(archive.read(names[0]))
                path = output / f"{name}.ome.zarr"
                record = create_store(
                    path,
                    (len(names), 1, 1, *plane.shape),
                    plane.dtype,
                    [(name, "FFFFFF")],
                    {
                        "source": source_url,
                        "original_archive": archive_name,
                        "original_members": names,
                        "purpose": "spot-only tracking",
                    },
                    lambda t, c, z, names=names: decode_tiff(archive.read(names[t])),
                )
                xml_name = (
                    f"{folder}/{PurePosixPath(folder).name}.xml"
                    if archive_name.startswith("SIM")
                    else f"{folder}/_Tracks.xml"
                )
                if xml_name in archive.namelist():
                    sidecar = output / "references" / name
                    sidecar.mkdir(parents=True, exist_ok=True)
                    payload = archive.read(xml_name)
                    (sidecar / "tracks.xml").write_bytes(payload)
                    truth_csv(payload, sidecar / "tracks.csv")
                    record["reference_tracks"] = str(sidecar / "tracks.csv")
                    record["reference_type"] = (
                        "ISBI simulation ground truth"
                        if archive_name.startswith("SIM")
                        else "authors' TrackMate trajectories, not independent simulation ground truth"
                    )
                records.append(record)
                print(f"Validated {name}: {record['shape']}", flush=True)
    with zipfile.ZipFile(root / "CBS-red-green" / "CBS001RGM-CBS010RGM.zip") as archive:
        for name in sorted(
            n for n in archive.namelist() if n.endswith(".tiff") and "__MACOSX" not in n
        ):
            plane = decode_tiff(archive.read(name))
            title = Path(name).stem
            record = create_store(
                output / f"{title}.ome.zarr",
                (1, 3, 1, *plane.shape[:2]),
                plane.dtype,
                [
                    ("Red", "FF0000"),
                    ("Green", "00FF00"),
                    ("Blue (source RGB)", "0000FF"),
                ],
                {
                    "source": "https://colocalization-benchmark.com/downloads/",
                    "original_member": name,
                    "licence": "CC BY-NC-SA 4.0",
                    "purpose": "Pearson/Manders edge cases and colocalization trend",
                    "nominal_colocalization_percent": (
                        int(re.search(r"\d+", title)[0]) - 1
                    )
                    * 10,
                    "caution": "Nominal percent is not identical to measured directional thresholded Manders.",
                },
                lambda t, c, z, plane=plane: plane[:, :, c],
            )
            records.append(record)
            print(f"Validated {title}", flush=True)
    with zipfile.ZipFile(root / "Fluo-N2DL-HeLa.zip") as archive:
        names = sorted(
            n for n in archive.namelist() if re.search(r"(?:^|/)01/t\d+\.tif$", n)
        )
        plane = decode_tiff(archive.read(names[0]))
        records.append(
            create_store(
                output / "CTC-Fluo-N2DL-HeLa-01.ome.zarr",
                (len(names), 1, 1, *plane.shape),
                plane.dtype,
                [("Nuclei", "00FFFF")],
                {
                    "source": "https://celltrackingchallenge.net/2d-datasets/",
                    "original_archive": "Fluo-N2DL-HeLa.zip",
                    "purpose": "nuclear tracking with divisions",
                },
                lambda t, c, z: decode_tiff(archive.read(names[t])),
                pixel_um=0.645,
                seconds=1800,
            )
        )
        sidecar = output / "references" / "CTC-Fluo-N2DL-HeLa-01"
        sidecar.mkdir(parents=True, exist_ok=True)
        tracks = next(
            n for n in archive.namelist() if n.endswith("01_GT/TRA/man_track.txt")
        )
        (sidecar / "man_track.txt").write_bytes(archive.read(tracks))
        records[-1]["reference_tracks"] = str(sidecar / "man_track.txt")
        records[-1]["reference_masks"] = (
            "Retained in original ZIP; TRA markers are not full nuclear masks."
        )
        print("Validated CTC HeLa sequence 01", flush=True)
    visium = root / "VisiumFluo" / "visium_fluo_image_crop.tiff"
    image = tifffile.memmap(visium).reshape(3, 7272, 7272)
    records.append(
        create_store(
            output / "Visium-mouse-brain-DAPI-NeuN-GFAP.ome.zarr",
            (1, 3, 1, 7272, 7272),
            image.dtype,
            [("DAPI", "0000FF"), ("anti-NeuN", "00FF00"), ("anti-GFAP", "FF0000")],
            {
                "source": "https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_visium_fluo.html",
                "original_file": visium.name,
                "purpose": "spatial tissue neighborhoods, compartment-specific colocalization",
                "caution": "Distinct neuronal/glial markers need not biologically colocalize; TIFF omits physical calibration.",
            },
            lambda t, c, z: image[c],
        )
    )
    print("Validated Visium mouse brain", flush=True)
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stores": records,
        "root": str(root),
        "format": "NGFF 0.4 / Zarr v2",
        "raw_data_retained": True,
    }
    (root / "ome-zarr-catalog.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publicdata", type=Path, default=Path("publicdata"))
    parser.add_argument(
        "--copy-to", type=Path, help="Optional directory visible to the local importer"
    )
    args = parser.parse_args()
    manifest = prepare(args.publicdata.resolve())
    if args.copy_to:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        for record in manifest["stores"]:
            path = Path(record["path"])
            shutil.copytree(path, args.copy_to / path.name, dirs_exist_ok=True)
        shutil.copytree(
            args.publicdata / "ome-zarr" / "references",
            args.copy_to / "references",
            dirs_exist_ok=True,
        )
        (args.copy_to / "ome-zarr-catalog.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    print(
        f"Prepared {len(manifest['stores'])} verified pyramidal OME-Zarr stores",
        flush=True,
    )


if __name__ == "__main__":
    main()
