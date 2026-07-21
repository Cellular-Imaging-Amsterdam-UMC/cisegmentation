from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np
import pytest

import cisegmentation.engine as engine
from cisegmentation.measurements import (
    measurement_database_path,
    write_measurements_database,
)
from cisegmentation.ome_zarr_io import ImageData, ImageResource, LabelResult
from cisegmentation.settings import SegmentationSettings


def _measurement_result(tmp_path: Path) -> LabelResult:
    raw = np.zeros((1, 2, 1, 6, 6), dtype=np.uint16)
    raw[0, 0, 0] = np.arange(36, dtype=np.uint16).reshape(6, 6)
    raw[0, 1, 0] = 100
    labels = np.zeros((1, 4, 1, 6, 6), dtype=np.uint32)
    labels[0, 0, 0, 1:5, 1:5] = 1
    labels[0, 1, 0, 2:4, 2:4] = 1
    labels[0, 2] = np.where(labels[0, 1] > 0, 0, labels[0, 0])
    labels[0, 3, 0, 2, 2] = 10
    labels[0, 3, 0, 1, 1] = 11
    source = ImageData(
        raw,
        ("t", "c", "z", "y", "x"),
        {"y": 0.5, "x": 0.25},
        {
            "omero": {
                "channels": [
                    {"label": "Signal", "color": "00FF00"},
                    {"label": "Reference", "color": "FF00FF"},
                ]
            }
        },
        ImageResource(
            tmp_path / "screen.ome.zarr",
            image_path="A/1/0",
            plate_path=("A", "1", "0"),
        ),
        "uint16",
    )
    return LabelResult(
        labels,
        source,
        "multi-step",
        "multi-step",
        provenance={
            "parameters": {"spotiflow_local_refinement": False},
            "output_statistics": [
                {"locations_only": False},
                {"locations_only": False},
                {"locations_only": False},
                {"locations_only": True},
            ],
        },
        channel_labels=[
            "labels_cells",
            "labels_nuclei",
            "labels_cytoplasm",
            "labels_spots_channel_1",
        ],
    )


def _query(path: Path, database_format: str, sql: str):
    if database_format == "sqlite":
        with sqlite3.connect(path) as connection:
            return connection.execute(sql).fetchall()
    import duckdb

    with duckdb.connect(str(path), read_only=True) as connection:
        return connection.execute(sql).fetchall()


@pytest.mark.parametrize("database_format", ["duckdb", "sqlite"])
def test_measurements_database_contains_shapes_intensities_and_relationships(
    tmp_path, database_format
):
    result = _measurement_result(tmp_path)
    output = tmp_path / f"measurements.{database_format}"
    summary = write_measurements_database(
        [result],
        output,
        database_format,
        output_ome_zarr=tmp_path / "result.ome.zarr",
    )

    assert output.is_file()
    assert summary["objects"] == 5
    assert summary["intensities"] == 10
    assert summary["relationships"] > 0
    assert _query(output, database_format, "SELECT COUNT(*) FROM images") == [(1,)]
    assert _query(output, database_format, "SELECT COUNT(*) FROM intensity_features") == [(10,)]

    cell = _query(
        output,
        database_format,
        """
        SELECT area_px2, area_um2, centroid_y_um, centroid_x_um,
               bbox_min_y_px, bbox_max_y_px, solidity, circularity
        FROM object_features WHERE object_type='cells'
        """,
    )[0]
    assert cell[:6] == pytest.approx((16, 2.0, 1.25, 0.625, 1, 5))
    assert cell[6] == pytest.approx(1.0)
    assert cell[7] > 0

    point_rows = _query(
        output,
        database_format,
        "SELECT is_point, bbox_min_x_px, centroid_x_um FROM object_features WHERE object_type='spots' ORDER BY label_value",
    )
    expected_true = True if database_format == "duckdb" else 1
    assert point_rows == [(expected_true, None, 0.5), (expected_true, None, 0.25)]

    constant_channel = _query(
        output,
        database_format,
        """
        SELECT intensity_mean, intensity_stddev, intensity_median
        FROM intensity_features
        WHERE object_type='cells' AND channel_name='Reference'
        """,
    )[0]
    assert constant_channel == pytest.approx((100, 0, 100))

    assignments = _query(
        output,
        database_format,
        """
        SELECT source_object_type, target_object_type, relation,
               source_overlap_fraction, source_centroid_in_target
        FROM foci_assignments
        ORDER BY target_object_type, source_object_id
        """,
    )
    assert len(assignments) == 4
    assert {row[1] for row in assignments} == {"cells", "cytoplasm", "nuclei"}
    assert all(row[2] == "inside" and row[3] == 1 for row in assignments)


def test_measurement_database_path_is_one_file_per_input_store(tmp_path):
    assert measurement_database_path(tmp_path, "plate", "duckdb") == (
        tmp_path / "plate_multistep_measurements.duckdb"
    )
    assert measurement_database_path(tmp_path, "plate", "sqlite") == (
        tmp_path / "plate_multistep_measurements.sqlite"
    )


def test_output_label_paths_match_native_groups_and_appended_channels(tmp_path):
    native = _measurement_result(tmp_path)
    native.channel_labels = ["labels duplicate"] * 4
    native.write_ome_zarr_labels = True
    native_path = tmp_path / "native.sqlite"
    write_measurements_database(
        [native], native_path, "sqlite", output_ome_zarr="output.ome.zarr"
    )
    assert _query(
        native_path,
        "sqlite",
        "SELECT output_label_path FROM label_sets ORDER BY label_set_index",
    ) == [
        ("A/1/0/labels/labels_duplicate",),
        ("A/1/0/labels/labels_duplicate_2",),
        ("A/1/0/labels/labels_duplicate_3",),
        ("A/1/0/labels/labels_duplicate_4",),
    ]

    channels = _measurement_result(tmp_path)
    channels.include_original_channels = True
    channels_path = tmp_path / "channels.sqlite"
    write_measurements_database(
        [channels], channels_path, "sqlite", output_ome_zarr="output.ome.zarr"
    )
    assert _query(
        channels_path,
        "sqlite",
        "SELECT output_label_path FROM label_sets ORDER BY label_set_index",
    ) == [
        ("A/1/0:channel:3",),
        ("A/1/0:channel:4",),
        ("A/1/0:channel:5",),
        ("A/1/0:channel:6",),
    ]


def test_workflow_writes_database_next_to_output_and_skip_omits_it(
    tmp_path, monkeypatch
):
    result = _measurement_result(tmp_path)
    store = tmp_path / "sample.ome.zarr"
    result.source.resource.store_path = store
    result.source.resource.image_path = ""
    result.source.resource.plate_path = None
    monkeypatch.setattr(engine, "discover_ome_zarrs", lambda _path: [store])
    monkeypatch.setattr(
        engine, "enumerate_resources", lambda _store: [result.source.resource]
    )
    monkeypatch.setattr(engine, "read_image", lambda _resource: result.source)
    monkeypatch.setattr(
        engine,
        "_segment_multistep_image",
        lambda *_args, **_kwargs: result,
    )

    def fake_write(_result, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)

    monkeypatch.setattr(engine, "write_label_image", fake_write)

    output_dir = tmp_path / "output"
    outputs = engine.run_workflow(
        tmp_path / "input",
        output_dir,
        SegmentationSettings(measurements_database="sqlite"),
    )
    assert outputs == [
        output_dir / "sample_multistep.ome.zarr",
        output_dir / "sample_multistep_measurements.sqlite",
    ]
    assert outputs[1].is_file()

    skip_outputs = engine.run_workflow(
        tmp_path / "input",
        output_dir,
        SegmentationSettings(measurements_database="skip"),
    )
    assert skip_outputs == [output_dir / "sample_multistep.ome.zarr"]
