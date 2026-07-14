from pathlib import Path

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from cisegmentation.ome_zarr_io import (  # noqa: E402
    LabelResult,
    discover_ome_zarrs,
    enumerate_resources,
    read_image,
    write_label_image,
)


DATA = Path(__file__).parent / "data"


def test_discover_and_read_committed_ome_zarrs():
    stores = discover_ome_zarrs(DATA)
    assert [path.name for path in stores] == [
        "nuclei-large.ome.zarr",
        "nuclei-medium.ome.zarr",
        "nuclei-small.ome.zarr",
    ]
    image = read_image(enumerate_resources(stores[-1])[0])
    assert image.data.shape == (1, 1, 1, 520, 520)
    assert image.scales["x"] == 0.5


def test_write_standalone_uint32_label_zarr(tmp_path):
    source = read_image(enumerate_resources(DATA / "nuclei-small.ome.zarr")[0])
    labels = np.zeros((1, 1, 1, 64, 64), dtype=np.uint32)
    labels[0, 0, 0, 10:20, 10:20] = 1
    result = LabelResult(labels, source, "stardist:SD_Nuclei_Versatile", "nuclei")
    output = write_label_image(result, tmp_path / "labels.ome.zarr")
    root = zarr.open_group(str(output), mode="r")
    assert root["0"].dtype == np.dtype("uint32")
    assert root["0"].shape == labels.shape
    assert root.attrs["multiscales"][0]["version"] == "0.4"
    assert (output / "OME" / "METADATA.ome.xml").exists()


def test_write_multichannel_benchmark_gallery(tmp_path):
    source = read_image(enumerate_resources(DATA / "nuclei-small.ome.zarr")[0])
    labels = np.zeros((1, 2, 1, 32, 32), dtype=np.uint32)
    result = LabelResult(
        labels,
        source,
        "benchmark-gallery",
        "nuclei",
        channel_labels=["model:a", "model:b"],
    )
    output = write_label_image(result, tmp_path / "benchmark.ome.zarr")
    root = zarr.open_group(str(output), mode="r")
    assert root["0"].shape[1] == 2
    assert [channel["label"] for channel in root.attrs["omero"]["channels"]] == [
        "model:a",
        "model:b",
    ]


def test_hcs_resource_enumeration(tmp_path):
    store = tmp_path / "plate.ome.zarr"
    root = zarr.open_group(str(store), mode="w")
    root.attrs["plate"] = {"wells": [{"path": "A/1"}], "version": "0.4"}
    well = root.require_group("A/1")
    well.attrs["well"] = {"images": [{"path": "0"}, {"path": "1"}], "version": "0.4"}
    for field in ("0", "1"):
        group = root.require_group(f"A/1/{field}")
        group.create_dataset("0", shape=(8, 8), data=np.zeros((8, 8), dtype=np.uint8))
        group.attrs["multiscales"] = [
            {"axes": ["y", "x"], "datasets": [{"path": "0"}], "version": "0.4"}
        ]
    resources = enumerate_resources(store)
    assert [resource.plate_path for resource in resources] == [
        ("A", "1", "0"),
        ("A", "1", "1"),
    ]
