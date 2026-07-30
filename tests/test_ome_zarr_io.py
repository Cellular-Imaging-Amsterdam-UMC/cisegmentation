import json

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from cisegmentation.ome_zarr_io import (  # noqa: E402
    HCSPlateWriter,
    ImageData,
    ImageResource,
    LabelResult,
    discover_ome_zarrs,
    enumerate_resources,
    read_image,
    set_measurement_timing,
    write_hcs_plate,
    write_label_image,
    write_native_label_groups,
    write_rgb_gallery,
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


def test_write_standalone_native_label_zarr(inputfolder, outputfolder):
    source = read_image(enumerate_resources(inputfolder / "nuclei-small.ome.zarr")[0])
    labels = np.zeros(
        (source.data.shape[0], 1, *source.data.shape[2:]), dtype=np.uint32
    )
    labels[0, 0, 0, 10:20, 10:20] = 7
    result = LabelResult(labels, source, "stardist:SD_Nuclei_Versatile", "nuclei")
    output = write_label_image(result, outputfolder / "labels.ome.zarr")
    root = zarr.open_group(str(output), mode="r")
    assert root["0"].dtype == source.data.dtype
    assert root["0"].shape == source.data.shape
    assert json.loads((output / "0" / ".zarray").read_text(encoding="utf-8"))[
        "dimension_separator"
    ] == "/"
    np.testing.assert_array_equal(np.asarray(root["0"]), source.data)
    np.testing.assert_array_equal(
        np.asarray(root["labels/labels_nuclei/0"]), labels
    )
    assert root.attrs["multiscales"][0]["version"] == "0.4"
    assert root["labels"].attrs["labels"] == ["labels_nuclei"]
    assert root.attrs["cisegmentation"]["storage_dtype"] == str(source.data.dtype)
    assert root.attrs["cisegmentation"]["label_storage_dtype"] == "uint32"
    assert 'Type="uint8"' in (output / "OME" / "METADATA.ome.xml").read_text(
        encoding="utf-8"
    )
    timings = root.attrs["cisegmentation"]["timings"]
    assert timings["zarr_write_seconds"] > 0
    assert timings["measurement_seconds"] == 0
    assert timings["total_seconds"] >= timings["zarr_write_seconds"]
    assert set(timings) >= {
        "startup_seconds",
        "zarr_read_seconds",
        "import_seconds",
        "model_load_seconds",
        "inference_seconds",
        "zarr_write_seconds",
        "measurement_seconds",
        "total_seconds",
    }
    assert (output / "OME" / "METADATA.ome.xml").exists()


def test_measurement_timing_updates_total(inputfolder, outputfolder):
    source = read_image(enumerate_resources(inputfolder / "nuclei-small.ome.zarr")[0])
    labels = np.zeros(
        (source.data.shape[0], 1, *source.data.shape[2:]), dtype=np.uint32
    )
    output = write_label_image(
        LabelResult(labels, source, "multi-step", "multi-step"),
        outputfolder / "measured.ome.zarr",
    )
    root = zarr.open_group(str(output), mode="r")
    before = dict(root.attrs["cisegmentation"]["timings"])
    root.store.close()

    set_measurement_timing(output, 2.5)

    root = zarr.open_group(str(output), mode="r")
    after = dict(root.attrs["cisegmentation"]["timings"])
    root.store.close()
    assert after["measurement_seconds"] == 2.5
    assert after["total_seconds"] == pytest.approx(before["total_seconds"] + 2.5)


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
            "runtime_seconds": 3.0,
            "segmentation_count": 4,
            "timings": {
                "inference_seconds": 1.0,
                "spot_detection_seconds": 0.25,
                "local_refinement_seconds": 0.5,
            },
        },
    )

    output = write_hcs_plate(
        [result],
        outputfolder / "plate-result.ome.zarr",
        output_store_uuid="e976fd49-41df-45ca-b5bb-ec186facf26f",
    )
    root = zarr.open_group(str(output), mode="r")
    assert root.attrs["cisegmentation"]["output_store_uuid"] == (
        "e976fd49-41df-45ca-b5bb-ec186facf26f"
    )
    assert root.attrs["cisegmentation"]["result_cache_hits"] == 2
    assert root.attrs["cisegmentation"]["runtime_seconds"] == 3.0
    assert root.attrs["cisegmentation"]["segmentation_count"] == 4
    assert root.attrs["cisegmentation"]["timings"]["local_refinement_seconds"] == 0.5


def test_hcs_plate_writer_persists_each_field_before_finalizing(
    inputfolder, outputfolder
):
    results = []
    plate_attrs = {
        "plate": {"version": "0.4", "wells": [{"path": "A/1"}]}
    }
    for field in ("0", "1"):
        source = read_image(
            enumerate_resources(inputfolder / "nuclei-small.ome.zarr")[0]
        )
        source.resource.plate_path = ("A", "1", field)
        source.resource.plate_attrs = plate_attrs
        source.resource.well_attrs = {
            "well": {
                "images": [{"path": field, "acquisition": 0}],
                "version": "0.4",
            }
        }
        labels = np.zeros(
            (source.data.shape[0], 1, *source.data.shape[2:]), dtype=np.uint32
        )
        results.append(LabelResult(labels, source, "multi-step", "multi-step"))

    output = outputfolder / "streamed-plate.ome.zarr"
    partial = output.with_name(output.name + ".partial")
    writer = HCSPlateWriter(
        [result.source.resource for result in results], output
    )
    writer.append(results[0])

    assert (partial / "A" / "1" / "0" / "0" / ".zarray").exists()
    assert not output.exists()

    writer.append(results[1])
    writer.finalize()

    assert not partial.exists()
    assert (output / "A" / "1" / "0" / "0" / ".zarray").exists()
    assert (output / "A" / "1" / "1" / "0" / ".zarray").exists()
    root = zarr.open_group(str(output), mode="r")
    assert root.attrs["cisegmentation"]["field_count"] == 2
    assert root["A/1"].attrs["well"]["images"] == [
        {"path": "0"},
        {"path": "1"},
    ]


def test_hcs_plate_writer_preserves_declared_acquisition(inputfolder, outputfolder):
    source = read_image(enumerate_resources(inputfolder / "nuclei-small.ome.zarr")[0])
    source.resource.plate_path = ("A", "1", "0")
    source.resource.plate_attrs = {
        "plate": {
            "version": "0.4",
            "wells": [{"path": "A/1"}],
            "acquisitions": [{"id": 7, "name": "round-1"}],
        }
    }
    source.resource.well_attrs = {
        "well": {
            "images": [{"path": "0", "acquisition": 7}],
            "version": "0.4",
        }
    }
    labels = np.zeros(
        (source.data.shape[0], 1, *source.data.shape[2:]), dtype=np.uint32
    )
    result = LabelResult(labels, source, "multi-step", "multi-step")

    output = write_hcs_plate([result], outputfolder / "acquisition-plate.ome.zarr")
    root = zarr.open_group(str(output), mode="r")

    assert root["A/1"].attrs["well"]["images"] == [{"path": "0", "acquisition": 7}]


def test_native_label_writer_preserves_ids_outside_int32_range(
    inputfolder, outputfolder
):
    source = read_image(enumerate_resources(inputfolder / "nuclei-small.ome.zarr")[0])
    labels = np.array([[[[[np.iinfo(np.int32).max + 1]]]]], dtype=np.uint64)
    result = LabelResult(labels, source, "test", "nuclei")
    output = write_label_image(result, outputfolder / "large-label.ome.zarr")
    root = zarr.open_group(str(output), mode="r")
    assert int(root["labels/labels_nuclei/0"][0, 0, 0, 0, 0]) == (
        np.iinfo(np.int32).max + 1
    )


def test_writer_keeps_original_pixels_and_writes_only_native_labels(
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
    )
    output = write_label_image(result, outputfolder / "original-and-labels.ome.zarr")
    root = zarr.open_group(str(output), mode="r")
    assert root["0"].dtype == source.data.dtype
    assert root["0"].shape[1] == source.data.shape[1]
    np.testing.assert_array_equal(root["0"], source.data)
    np.testing.assert_array_equal(root["labels/nuclei/0"], labels)
    assert len(root.attrs["omero"]["channels"]) == source.data.shape[1]


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


def test_native_overlay_labels_keep_source_axis_dimensionality(outputfolder):
    output = outputfolder / "two-dimensional-overlay.ome.zarr"
    root = zarr.open_group(str(output), mode="w", zarr_version=2)
    root.store.close()
    source = ImageData(
        np.zeros((1, 1, 1, 8, 8), dtype=np.uint16),
        ("y", "x"),
        {"y": 0.5, "x": 0.5},
        {},
        ImageResource(output),
        "uint16",
    )
    result = LabelResult(
        np.ones((1, 1, 1, 8, 8), dtype=np.uint32),
        source,
        "test",
        "cells",
        channel_labels=["labels_cells"],
    )

    write_native_label_groups(
        output, "", result, ["labels_cells"], ["labels_cells"]
    )

    root = zarr.open_group(str(output), mode="r")
    label = root["labels/labels_cells/0"]
    assert label.shape == (8, 8)
    assert label.attrs["_ARRAY_DIMENSIONS"] == ["y", "x"]
    assert [
        axis["name"]
        for axis in root["labels/labels_cells"].attrs["multiscales"][0][
            "axes"
        ]
    ] == ["y", "x"]
    root.store.close()


def test_float_source_pixels_keep_their_native_values(inputfolder, outputfolder):
    source = read_image(enumerate_resources(inputfolder / "nuclei-small.ome.zarr")[0])
    source.data = source.data.astype(np.float32) + np.float32(0.6)
    source.source_dtype = "float32"
    labels = np.zeros(
        (source.data.shape[0], 1, *source.data.shape[2:]), dtype=np.uint32
    )
    result = LabelResult(labels, source, "test", "nuclei")
    output = write_label_image(result, outputfolder / "float-original.ome.zarr")
    root = zarr.open_group(str(output), mode="r")
    np.testing.assert_array_equal(
        root["0"], source.data
    )
    metadata = root.attrs["cisegmentation"]
    assert metadata["storage_dtype"] == "float32"


def test_write_multichannel_benchmark_gallery(inputfolder, outputfolder):
    source = read_image(enumerate_resources(inputfolder / "nuclei-small.ome.zarr")[0])
    rgb = np.zeros((3, 32, 32), dtype=np.uint8)
    output = write_rgb_gallery(
        rgb,
        source,
        {"models": ["model:a", "model:b"]},
        outputfolder / "benchmark.ome.zarr",
    )
    root = zarr.open_group(str(output), mode="r")
    assert root["0"].shape[1] == 3
    assert [channel["label"] for channel in root.attrs["omero"]["channels"]] == [
        "Red",
        "Green",
        "Blue",
    ]
    assert root.attrs["cisegmentation"]["models"] == ["model:a", "model:b"]


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
