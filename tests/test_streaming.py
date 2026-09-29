import numpy as np
import pytest
import zarr
from scipy.ndimage import label

from cisegmentation.engine import run_workflow
from cisegmentation.ome_zarr_io import ImageResource, read_image
from cisegmentation.registry import MODEL_REGISTRY, get_model_spec
from cisegmentation.resources import GIB, ResourceSnapshot
from cisegmentation.settings import SegmentationSettings
from cisegmentation.streaming import ArrayView, infer_streamed, iter_regions, raw_array


def _source(path, shape=(1, 1, 1, 192, 256)):
    group = zarr.open_group(str(path), mode="w")
    array = group.create_dataset("0", shape=shape, dtype="u2", chunks=(1, 1, 1, 64, 64))
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {
                    "name": axis,
                    "type": "channel"
                    if axis == "c"
                    else "time"
                    if axis == "t"
                    else "space",
                    **({"unit": "micrometer"} if axis in "zyx" else {}),
                }
                for axis in "tczyx"
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1, 1, 1, 0.5, 0.5]}
                    ],
                }
            ],
        }
    ]
    if shape[-2:] == (192, 256):
        yy, xx = np.ogrid[:192, :256]
        mask = (
            ((yy - 64) ** 2 + (xx - 64) ** 2 < 20**2)
            | ((yy - 95) ** 2 + (xx - 125) ** 2 < 16**2)
            | ((yy - 130) ** 2 + (xx - 190) ** 2 < 18**2)
        )
        array[:] = np.broadcast_to(mask * 100, shape)
    return ImageResource(path)


def _segment(data, spec, settings, scales, **kwargs):
    labels, count = label(data[0] > 0)
    return labels.astype("u4"), {
        "device": "cpu",
        "runtime_seconds": 0.01,
        "object_count": count,
        "timings": {"inference_seconds": 0.01},
        "model_cache_hits": 0,
        "model_cache_misses": 1,
        "effective_parameters": {},
    }


def test_array_view_reads_only_requested_regions_and_transposes_axes(tmp_path):
    group = zarr.open_group(str(tmp_path / "array.zarr"), mode="w")
    data = np.arange(2 * 3 * 7 * 9, dtype=">u2").reshape(7, 9, 3, 2)
    array = group.create_dataset("0", data=data)
    view = ArrayView(array, "yxct", tuple("tczyx"))
    actual = np.asarray(view[1, 2, 0, 1:6:2, 2:8:2])
    np.testing.assert_array_equal(actual, data[1:6:2, 2:8:2, 2, 1])
    assert actual.dtype.isnative
    assert view[1, 2, 0, 4, 5] == data[4, 5, 2, 1]


def test_large_source_is_metadata_only_until_region_read(tmp_path):
    resource = _source(tmp_path / "large.zarr", (1, 1, 1, 40000, 40000))
    image = read_image(resource, lazy=True)
    assert image.data.shape == (1, 1, 1, 40000, 40000)
    with pytest.raises(MemoryError, match="bounded regions"):
        np.asarray(image.data)
    assert np.asarray(image.data[0, 0, 0, :32, :32]).shape == (32, 32)


@pytest.mark.parametrize("model_id", list(MODEL_REGISTRY))
def test_every_model_uses_shared_overlap_and_id_stitching(tmp_path, model_id):
    resource = _source(tmp_path / "source.zarr")
    image = read_image(resource, lazy=True)
    spec = get_model_spec(model_id)
    settings = SegmentationSettings(
        model=model_id,
        target=spec.targets[0],
        streaming_mode="on",
        tile_size=64,
        tile_overlap=24,
        device="cpu",
    )
    path = tmp_path / "raw.zarr"
    infos, _ = infer_streamed(image, spec, settings, path, _segment)
    labels = np.asarray(raw_array(path))
    expected = np.asarray(image.data)[0, 0] > 0
    np.testing.assert_array_equal(labels[0] > 0, expected)
    assert len(np.unique(labels)) == 4
    assert infos[0]["streaming"]["tiles"] == 12
    assert labels[0, 0, 64, 50] == labels[0, 0, 64, 75] != 0


def test_oom_splits_current_tile_without_repeating_completed_cores(tmp_path):
    image = read_image(_source(tmp_path / "source.zarr"), lazy=True)
    calls = []

    def segment(data, *args, **kwargs):
        calls.append(data.shape)
        if max(data.shape[-2:]) > 128:
            raise RuntimeError("CUDA out of memory")
        return _segment(data, *args, **kwargs)

    settings = SegmentationSettings(tile_size=256, tile_overlap=16, device="cpu")
    path = tmp_path / "raw.zarr"
    infos, _ = infer_streamed(
        image, get_model_spec("stardist:SD_Nuclei_Versatile"), settings, path, segment
    )
    labels = np.asarray(raw_array(path))
    np.testing.assert_array_equal(labels[0] > 0, np.asarray(image.data)[0, 0] > 0)
    assert len(np.unique(labels)) == 4
    assert infos[0]["streaming"]["oom_retries"] > 0
    assert infos[0]["streaming"]["smallest_core_xy"] < 128


def test_live_memory_pressure_reduces_pending_tiles(tmp_path, monkeypatch):
    from cisegmentation import streaming

    image = read_image(_source(tmp_path / "source.zarr"), lazy=True)
    state = {"calls": 0}

    def resources():
        state["calls"] += 1
        return ResourceSnapshot(
            4, 16 * GIB, 16 * GIB if state["calls"] < 3 else 3 * 1024**2
        )

    monkeypatch.setattr(streaming, "snapshot", resources)
    settings = SegmentationSettings(tile_size=128, tile_overlap=8, device="cpu")
    infos, _ = infer_streamed(
        image,
        get_model_spec("stardist:SD_Nuclei_Versatile"),
        settings,
        tmp_path / "raw.zarr",
        _segment,
    )
    assert infos[0]["streaming"]["pressure_splits"] > 0


def test_operation_size_retry_keeps_completed_cores_and_stitched_detections(tmp_path):
    image = read_image(_source(tmp_path / "source.zarr"), lazy=True)
    calls = []

    def segment(data, *args, **kwargs):
        # A later tile hits an operation limit after the first core is saved.
        calls.append(data.shape)
        if len(calls) == 2:
            raise RuntimeError("quantile() input tensor is too large")
        return _segment(data, *args, **kwargs)

    path = tmp_path / "raw.zarr"
    infos, _ = infer_streamed(
        image,
        get_model_spec("stardist:SD_Nuclei_Versatile"),
        SegmentationSettings(tile_size=128, tile_overlap=16, device="cpu"),
        path,
        segment,
    )
    actual = np.asarray(raw_array(path))
    np.testing.assert_array_equal(actual[0] > 0, np.asarray(image.data)[0, 0] > 0)
    assert len(np.unique(actual)) == 4
    stats = infos[0]["streaming"]
    assert stats["size_limit_retries"] == 1
    assert stats["oom_retries"] == 0
    assert stats["tiles"] == 5
    assert len(calls) == 6  # Four original cores, one replaced by two children.


def test_streamed_regions_keep_global_coordinates_and_full_instance(tmp_path):
    array = zarr.open(
        str(tmp_path / "labels.zarr"),
        mode="w",
        shape=(1, 1100, 1100),
        dtype="u4",
        chunks=(1, 128, 128),
    )
    array[:, 490:530, 500:540] = 90000
    regions = list(iter_regions(ArrayView(array, "zyx")))
    assert len(regions) == 1
    assert regions[0].label == 90000
    assert regions[0].bbox == (0, 490, 500, 1, 530, 540)
    assert regions[0].centroid == (0, 509.5, 519.5)
    assert regions[0].area == 1600


def test_full_streamed_workflow_and_measurements_match_eager(tmp_path, monkeypatch):
    import duckdb

    import cisegmentation.parallel_pipeline as pipeline

    monkeypatch.setenv("CISEGMENTATION_INLINE_WORKERS", "1")
    monkeypatch.setattr(pipeline, "segment_czyx", _segment)
    resource = _source(tmp_path / "source.zarr")
    for mode in ("auto", "on"):
        outputs = run_workflow(
            resource.store_path,
            tmp_path / mode,
            SegmentationSettings(
                cell_model="cellpose3:cyto3",
                nucleus_model="cellpose3:nuclei",
                remove_border_cells=False,
                streaming_mode=mode,
                tile_size=64,
                tile_overlap=24,
                max_inference_workers=1,
                max_measurement_workers=1,
            ),
        )
        group = zarr.open_group(str(outputs[0]), mode="r")
        assert (
            len(group["labels/labels_nuclei"].attrs["multiscales"][0]["datasets"]) == 1
        )
        with duckdb.connect(str(outputs[1]), read_only=True) as connection:
            rows = connection.execute(
                "SELECT object_type, voxel_count, centroid_y_px, centroid_x_px FROM objects ORDER BY object_type, centroid_y_px"
            ).fetchall()
        if mode == "auto":
            expected = rows
            expected_arrays = {
                name: np.asarray(group[f"labels/{name}/0"])
                for name in group["labels"].attrs["labels"]
            }
        else:
            assert rows == expected
            for name, array in expected_arrays.items():
                np.testing.assert_array_equal(group[f"labels/{name}/0"][:], array)


def test_decoded_chunk_cache_is_bounded_and_shared_by_crops(tmp_path):
    from cisegmentation.streaming import ChunkCache

    array = zarr.open(
        str(tmp_path / "cache.zarr"),
        mode="w",
        shape=(2, 3, 33, 41),
        dtype="u2",
        chunks=(1, 1, 8, 8),
    )
    expected = np.arange(array.size, dtype="u2").reshape(array.shape)
    array[:] = expected
    cache = ChunkCache(256)
    view = ArrayView(array, "tcyx", cache=cache)
    for key in (
        (1, 2, slice(4, 28), slice(7, 39, 3)),
        (0, 1, slice(None, None, 5), slice(None, None, 7)),
    ):
        np.testing.assert_array_equal(np.asarray(view[key]), expected[key])
        assert cache.bytes <= cache.limit
    assert view[1, 2, 16, 17] == expected[1, 2, 16, 17]


def test_nonstreamed_operation_limit_falls_back_to_streaming(tmp_path, monkeypatch):
    import cisegmentation.parallel_pipeline as pipeline

    calls = []

    def segment(data, spec, settings, scales, **kwargs):
        calls.append(bool(kwargs.get("bounded")))
        if not kwargs.get("bounded"):
            raise RuntimeError("quantile() input tensor is too large")
        return _segment(data, spec, settings, scales, **kwargs)

    resource = _source(tmp_path / "source.zarr")
    monkeypatch.setenv("CISEGMENTATION_INLINE_WORKERS", "1")
    monkeypatch.setattr(pipeline, "segment_czyx", segment)
    outputs = run_workflow(
        resource.store_path,
        tmp_path / "output",
        SegmentationSettings(
            cell_model="skip",
            nucleus_model="stardist:SD_Nuclei_Versatile",
            streaming_mode="auto",
            tile_size=128,
            tile_overlap=24,
            remove_border_cells=False,
            max_inference_workers=1,
            max_measurement_workers=1,
            device="cpu",
        ),
    )
    assert calls[0] is False and all(calls[1:])
    group = zarr.open_group(str(outputs[0]), mode="r")
    actual = np.asarray(group["labels/labels_nuclei/0"])
    expected = np.asarray(read_image(resource).data)
    np.testing.assert_array_equal(actual > 0, expected > 0)


def test_native_3d_tiles_match_ids_across_z_and_xy_boundaries(tmp_path):
    resource = _source(tmp_path / "volume.zarr", (1, 1, 9, 192, 256))
    image = read_image(resource, lazy=True)
    settings = SegmentationSettings(
        tile_size=64, tile_overlap=24, tile_depth=3, tile_overlap_z=2, device="cpu"
    )
    path = tmp_path / "raw.zarr"
    infos, _ = infer_streamed(
        image, get_model_spec("cellpose3:nuclei"), settings, path, _segment
    )
    labels = np.asarray(raw_array(path))
    np.testing.assert_array_equal(labels[0] > 0, np.asarray(image.data)[0, 0] > 0)
    assert len(np.unique(labels)) == 4
    assert infos[0]["streaming"]["tiles"] == 36


def test_native_spotiflow_keeps_volume_context_when_core_depth_is_small(tmp_path):
    image = read_image(
        _source(tmp_path / "volume.zarr", (1, 1, 8, 192, 256)), lazy=True
    )
    settings = SegmentationSettings(
        tile_size=64, tile_overlap=24, tile_depth=2, tile_overlap_z=1, device="cpu"
    )

    def contextual_segment(data, *args, **kwargs):
        # A positive detector requiring the original volume context must not
        # become empty just because its written cores are two slices deep.
        assert data.shape[1:] == (8, 192, 256)
        return _segment(data, *args, **kwargs)

    path = tmp_path / "raw.zarr"
    infer_streamed(
        image, get_model_spec("spotiflow:smfish_3d"), settings, path, contextual_segment
    )
    actual = np.asarray(raw_array(path))
    np.testing.assert_array_equal(actual[0] > 0, np.asarray(image.data)[0, 0] > 0)
    assert len(np.unique(actual)) == 4


def test_large_measurement_tables_are_copied_in_bounded_batches():
    import sqlite3

    from cisegmentation.parallel_measurements import _copy_large_table

    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE objects (id INTEGER, image INTEGER)")
    connection.executemany(
        "INSERT INTO objects VALUES (?, 1)", [(i,) for i in range(3000)]
    )

    class Writer:
        def __init__(self):
            self.rows = []
            self.sizes = []

        def insert(self, table, columns, rows):
            self.rows.extend(rows)
            self.sizes.append(len(rows))

    writer = Writer()
    assert _copy_large_table(connection, writer, "objects", {0: 20, 1: 10}) == 3000
    assert max(writer.sizes) == 1024
    assert writer.rows[0] == (20, 11)
    assert writer.rows[-1] == (3019, 11)


def test_large_benchmark_crops_before_reading_source_pixels(tmp_path, monkeypatch):
    from cisegmentation import engine
    from cisegmentation.benchmark import center_crop

    resource = _source(tmp_path / "large.ome.zarr", (1, 1, 1, 40000, 40000))

    def benchmark(image, settings, output_dir, **kwargs):
        assert isinstance(image.data, ArrayView)
        crop, coordinates = center_crop(image.data[0])
        assert np.asarray(crop).shape == (1, 1, 1024, 1024)
        assert coordinates["x"] == coordinates["y"] == 19488
        return tmp_path / "gallery.ome.zarr", False

    monkeypatch.setattr(engine, "run_benchmark", benchmark)
    outputs = run_workflow(
        resource.store_path, tmp_path / "output", SegmentationSettings(benchmark=True)
    )
    assert outputs == [tmp_path / "gallery.ome.zarr"]


@pytest.mark.parametrize("ids", [[1, 2, 500000], [1, 2, 4000000000]])
def test_global_remapping_handles_dense_and_sparse_ids_without_large_allocations(ids):
    from cisegmentation.streaming import LabelRemap

    mapping = LabelRemap(dict(zip(ids, (7, 9, 11))))
    source = np.array([0, *ids, 3, 4000000001], dtype="u4")
    np.testing.assert_array_equal(mapping(source), [0, 7, 9, 11, 0, 0])
    assert mapping.lookup is None
    with pytest.raises(OverflowError, match="uint32"):
        LabelRemap({1: 2**32})


def test_dense_global_remapping_is_prepared_once_for_many_chunks():
    from cisegmentation.streaming import LabelRemap

    mapping = LabelRemap({value: value + 10 for value in range(1, 500001)})
    source = np.array([0, 1, 250000, 500000, 500001], dtype="u4")
    for _ in range(3):
        np.testing.assert_array_equal(mapping(source), [0, 11, 250010, 500010, 0])
    assert mapping.lookup.nbytes == 500001 * 4


def test_streamed_expansion_matches_eager_across_storage_chunk_boundaries(tmp_path):
    from cisegmentation.engine import _expand_nuclei_to_cells
    from cisegmentation.streaming import _expand_streamed

    source = zarr.open(
        str(tmp_path / "nuclei.zarr"),
        mode="w",
        shape=(1, 1100, 1100),
        dtype="u4",
        chunks=(1, 128, 128),
    )
    source[0, 490:515, 500:540] = 27
    source[0, 520:540, 550:570] = 39
    destination = zarr.open(
        str(tmp_path / "cells.zarr"),
        mode="w",
        shape=source.shape,
        dtype="u4",
        chunks=source.chunks,
    )
    scales = {"x": 0.5, "y": 0.7, "z": 1}
    _expand_streamed(ArrayView(source, "zyx"), destination, 10, scales)
    np.testing.assert_array_equal(
        destination[:], _expand_nuclei_to_cells(source[:], 10, scales)
    )


def test_streamed_label_pyramids_preserve_permuted_axes_and_odd_dimensions(tmp_path):
    from cisegmentation.ome_zarr_io import LabelResult, write_native_label_groups

    path = tmp_path / "source.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.create_dataset(
        "0", shape=(1031, 1103, 1, 1), dtype="u2", chunks=(128, 128, 1, 1)
    )
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {
                    "name": axis,
                    "type": "channel"
                    if axis == "c"
                    else "time"
                    if axis == "t"
                    else "space",
                }
                for axis in "yxct"
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [0.5, 0.7, 1, 1]}
                    ],
                }
            ],
        }
    ]
    image = read_image(ImageResource(path), lazy=True)
    labels = zarr.open(
        str(tmp_path / "labels.zarr"),
        mode="w",
        shape=image.data.shape,
        dtype="u4",
        chunks=(1, 1, 1, 128, 128),
    )
    labels[0, 0, 0, 400:900, 490:800] = 123456
    result = LabelResult(
        ArrayView(labels, "tczyx"),
        image,
        "test",
        "nuclei",
        channel_labels=["labels_nuclei"],
    )
    output = tmp_path / "overlay.zarr"
    write_native_label_groups(output, "", result, ["labels_nuclei"], ["labels_nuclei"])
    group = zarr.open_group(str(output), mode="r")["labels/labels_nuclei"]
    assert [list(group[str(level)].shape) for level in range(3)] == [
        [1031, 1103, 1, 1],
        [516, 552, 1, 1],
        [258, 276, 1, 1],
    ]
    for level in range(3):
        np.testing.assert_array_equal(
            group[str(level)][:, :, 0, 0], labels[0, 0, 0, :: 2**level, :: 2**level]
        )
    assert [axis["name"] for axis in group.attrs["multiscales"][0]["axes"]] == list(
        "yxct"
    )
