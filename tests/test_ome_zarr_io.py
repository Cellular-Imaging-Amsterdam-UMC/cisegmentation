import json

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from cisegmentation.ome_zarr_io import (  # noqa: E402
    LabelResult,
    discover_ome_zarrs,
    enumerate_resources,
    read_image,
    write_hcs_plate,
    write_label_image,
)


def test_discover_and_read_staged_ome_zarrs(inputfolder):
    stores = discover_ome_zarrs(inputfolder)
    names = {path.name for path in stores}
    assert {
        "nuclei-large.ome.zarr",
        "nuclei-medium.ome.zarr",
        "nuclei-small.ome.zarr",
    } <= names
    image = read_image(enumerate_resources(inputfolder / "nuclei-small.ome.zarr")[0])
    assert image.data.shape == (1, 1, 1, 520, 520)
    assert image.scales["x"] == 0.5


def test_read_converts_big_endian_pixels_to_native_order(inputfolder):
    image = read_image(
        enumerate_resources(inputfolder / "nuclei-spots-cytoplasm.ome.zarr")[0]
    )
    assert image.source_dtype == ">u2"
    assert image.data.dtype == np.dtype("uint16")
    assert image.data.dtype.isnative


def test_discover_accepts_biomero_zarr_name(tmp_path):
    store = tmp_path / "renamed-by-biomero.zarr"
    store.mkdir()
    (store / ".zattrs").write_text(
        '{"multiscales":[{"datasets":[{"path":"0"}]}]}', encoding="utf-8"
    )
    (store / ".zgroup").write_text('{"zarr_format":2}', encoding="utf-8")
    assert discover_ome_zarrs(tmp_path) == [store]


def test_write_standalone_int32_label_zarr(inputfolder, outputfolder):
    source = read_image(enumerate_resources(inputfolder / "nuclei-small.ome.zarr")[0])
    labels = np.zeros(
        (source.data.shape[0], 1, *source.data.shape[2:]), dtype=np.uint32
    )
    labels[0, 0, 0, 10:20, 10:20] = 7
    result = LabelResult(labels, source, "stardist:SD_Nuclei_Versatile", "nuclei")
    output = write_label_image(result, outputfolder / "labels.ome.zarr")
    root = zarr.open_group(str(output), mode="r")
    assert root["0"].dtype == np.dtype("int32")
    assert root["0"].shape == labels.shape
    assert json.loads((output / "0" / ".zarray").read_text(encoding="utf-8"))[
        "dimension_separator"
    ] == "/"
    np.testing.assert_array_equal(np.asarray(root["0"]), labels)
    assert root.attrs["multiscales"][0]["version"] == "0.4"
    channel = root.attrs["omero"]["channels"][0]
    assert channel["color"] == "0000FF"
    assert channel["lookupTable"] == "glasbey_inverted.lut"
    assert channel["window"] == {"start": 0.0, "end": 7.0, "min": 0.0, "max": 7.0}
    assert root.attrs["cisegmentation"]["label_rendering"] == {
        "lookup_table": "glasbey_inverted.lut",
        "rendering_only": True,
        "pixel_values_transformed": False,
    }
    assert root.attrs["cisegmentation"]["storage_dtype"] == "int32"
    assert 'Type="int32"' in (output / "OME" / "METADATA.ome.xml").read_text(
        encoding="utf-8"
    )
    timings = root.attrs["cisegmentation"]["timings"]
    assert timings["zarr_write_seconds"] > 0
    assert timings["total_seconds"] >= timings["zarr_write_seconds"]
    assert set(timings) >= {
        "startup_seconds",
        "zarr_read_seconds",
        "import_seconds",
        "model_load_seconds",
        "inference_seconds",
        "zarr_write_seconds",
        "total_seconds",
    }
    assert (output / "OME" / "METADATA.ome.xml").exists()


def test_hcs_root_aggregates_result_reuse_counts(inputfolder, outputfolder):
    source = read_image(enumerate_resources(inputfolder / "nuclei-small.ome.zarr")[0])
    source.resource.plate_path = ("A", "1", "0")
    labels = np.zeros(
        (source.data.shape[0], 1, *source.data.shape[2:]), dtype=np.uint32
    )
    result = LabelResult(
        labels,
        source,
        "multi-step",
        "multi-step",
        provenance={
            "model_cache_hits": 1,
            "model_cache_misses": 1,
            "result_cache_hits": 2,
            "timings": {
                "inference_seconds": 1.0,
                "spot_detection_seconds": 0.25,
                "local_refinement_seconds": 0.5,
            },
        },
    )

    output = write_hcs_plate([result], outputfolder / "plate-result.ome.zarr")
    root = zarr.open_group(str(output), mode="r")
    assert root.attrs["cisegmentation"]["result_cache_hits"] == 2
    assert root.attrs["cisegmentation"]["timings"]["local_refinement_seconds"] == 0.5


def test_label_writer_rejects_ids_outside_int32_range(inputfolder, outputfolder):
    source = read_image(enumerate_resources(inputfolder / "nuclei-small.ome.zarr")[0])
    labels = np.array([[[[[np.iinfo(np.int32).max + 1]]]]], dtype=np.uint64)
    result = LabelResult(labels, source, "test", "nuclei")
    with pytest.raises(OverflowError, match="exceeds QuPath-compatible int32"):
        write_label_image(result, outputfolder / "overflow.ome.zarr")


def test_writer_can_prepend_original_channels_without_changing_labels(
    inputfolder, outputfolder
):
    source = read_image(enumerate_resources(inputfolder / "nuclei-small.ome.zarr")[0])
    labels = np.zeros(
        (source.data.shape[0], 1, *source.data.shape[2:]), dtype=np.uint32
    )
    labels[0, 0, 0, 10:20, 10:20] = 7
    result = LabelResult(
        labels,
        source,
        "test",
        "nuclei",
        channel_labels=["nuclei"],
        include_original_channels=True,
    )
    output = write_label_image(result, outputfolder / "original-and-labels.ome.zarr")
    root = zarr.open_group(str(output), mode="r")
    assert root["0"].dtype == np.dtype("int32")
    assert root["0"].shape[1] == source.data.shape[1] + 1
    np.testing.assert_array_equal(root["0"][:, : source.data.shape[1]], source.data)
    np.testing.assert_array_equal(root["0"][:, -1:], labels.astype(np.int32))
    channels = root.attrs["omero"]["channels"]
    assert "lookupTable" not in channels[0]
    assert channels[-1]["lookupTable"] == "glasbey_inverted.lut"


def test_writer_can_store_native_ome_zarr_04_label_images(inputfolder, outputfolder):
    source = read_image(
        enumerate_resources(inputfolder / "nuclei-spots-cytoplasm.ome.zarr")[0]
    )
    labels = np.zeros(
        (source.data.shape[0], 2, *source.data.shape[2:]), dtype=np.uint32
    )
    labels[0, 0, 0, 10:20, 10:20] = 7
    labels[0, 1, 0, 30:35, 30:35] = 2
    result = LabelResult(
        labels,
        source,
        "multi-step",
        "multi-step",
        channel_labels=["labels_spots_channel_2", "labels_spots_channel_2"],
        include_original_channels=False,
        write_ome_zarr_labels=True,
    )

    output = write_label_image(result, outputfolder / "native-labels.ome.zarr")
    root = zarr.open_group(str(output), mode="r")

    assert root["0"].dtype == source.data.dtype
    np.testing.assert_array_equal(np.asarray(root["0"]), source.data)
    assert root["0"].shape[1] == source.data.shape[1]
    assert root["labels"].attrs["labels"] == [
        "labels_spots_channel_2",
        "labels_spots_channel_2_2",
    ]
    first = root["labels/labels_spots_channel_2"]
    second = root["labels/labels_spots_channel_2_2"]
    np.testing.assert_array_equal(np.asarray(first["0"]), labels[:, 0:1])
    np.testing.assert_array_equal(np.asarray(second["0"]), labels[:, 1:2])
    assert first.attrs["image-label"] == {
        "version": "0.4",
        "source": {"image": "../../"},
    }
    assert first.attrs["multiscales"][0]["version"] == "0.4"
    assert root.attrs["cisegmentation"]["output_layout"] == (
        "ome-zarr-0.4-labels"
    )
    assert json.loads((output / "0" / ".zarray").read_text(encoding="utf-8"))[
        "dimension_separator"
    ] == "/"
    for group_name in root["labels"].attrs["labels"]:
        assert json.loads(
            (output / "labels" / group_name / "0" / ".zarray").read_text(
                encoding="utf-8"
            )
        )["dimension_separator"] == "/"
    assert 'Type="uint16"' in (
        output / "OME" / "METADATA.ome.xml"
    ).read_text(encoding="utf-8")


def test_included_float_channels_are_rounded_to_int32(inputfolder, outputfolder):
    source = read_image(enumerate_resources(inputfolder / "nuclei-small.ome.zarr")[0])
    source.data = source.data.astype(np.float32) + np.float32(0.6)
    source.source_dtype = "float32"
    labels = np.zeros(
        (source.data.shape[0], 1, *source.data.shape[2:]), dtype=np.uint32
    )
    result = LabelResult(
        labels, source, "test", "nuclei", include_original_channels=True
    )
    output = write_label_image(result, outputfolder / "float-original.ome.zarr")
    root = zarr.open_group(str(output), mode="r")
    np.testing.assert_array_equal(
        root["0"][:, : source.data.shape[1]], np.rint(source.data).astype(np.int32)
    )
    metadata = root.attrs["cisegmentation"]
    assert metadata["original_source_dtype"] == "float32"
    assert metadata["original_channels_conversion"] == "round-to-nearest-int32"


def test_write_multichannel_benchmark_gallery(inputfolder, outputfolder):
    source = read_image(enumerate_resources(inputfolder / "nuclei-small.ome.zarr")[0])
    labels = np.zeros((1, 2, 1, 32, 32), dtype=np.uint32)
    result = LabelResult(
        labels,
        source,
        "benchmark-gallery",
        "nuclei",
        channel_labels=["model:a", "model:b"],
    )
    output = write_label_image(result, outputfolder / "benchmark.ome.zarr")
    root = zarr.open_group(str(output), mode="r")
    assert root["0"].shape[1] == 2
    assert [channel["label"] for channel in root.attrs["omero"]["channels"]] == [
        "model:a",
        "model:b",
    ]
    assert all(
        channel["lookupTable"] == "glasbey_inverted.lut"
        for channel in root.attrs["omero"]["channels"]
    )


def test_hcs_resource_enumeration(outputfolder):
    store = outputfolder / "plate.ome.zarr"
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
