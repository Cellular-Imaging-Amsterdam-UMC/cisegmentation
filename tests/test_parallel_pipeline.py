from __future__ import annotations

import hashlib
import errno
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

import cisegmentation.parallel_pipeline as parallel_pipeline  # noqa: E402
from cisegmentation.engine import run_workflow  # noqa: E402
from cisegmentation.ome_zarr_io import (  # noqa: E402
    ImageResource,
    enumerate_resources,
    existing_label_names,
)
from cisegmentation.parallel_pipeline import (  # noqa: E402
    build_model_passes,
    calculate_gpu_workers,
    publish_consumed_store,
    recover_label_commit,
    resolve_label_policy,
)
from cisegmentation.settings import SegmentationSettings  # noqa: E402


def _fake_segment(czyx, _spec, _settings, _scales):
    labels = np.zeros(czyx.shape[1:], dtype=np.uint32)
    labels[..., 10:20, 10:20] = 1
    return labels, {
        "device": "cpu",
        "dimension_mode": "slice-2d",
        "runtime_seconds": 0.01,
        "model_cache_hit": False,
        "model_cache_hits": 0,
        "model_cache_misses": 1,
        "timings": {"inference_seconds": 0.01},
        "effective_parameters": {},
    }


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _run_small_store(
    tmp_path: Path,
    fixture: Path,
    monkeypatch,
    *,
    include_original_data: bool,
) -> tuple[Path, Path]:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True)
    source = input_dir / "sample.ome.zarr"
    shutil.copytree(fixture, source)
    monkeypatch.setenv("CISEGMENTATION_INLINE_WORKERS", "1")
    monkeypatch.setattr(parallel_pipeline, "segment_czyx", _fake_segment)
    outputs = run_workflow(
        input_dir,
        output_dir,
        SegmentationSettings(
            cell_model="cellpose3:cyto3",
            nucleus_model="skip",
            remove_border_cells=False,
            include_original_data=include_original_data,
            measurements_database="skip",
            max_inference_workers=1,
            max_measurement_workers=1,
        ),
    )
    return source, outputs[0]


def test_model_passes_follow_first_appearance_and_deduplicate_exact_requests():
    settings = SegmentationSettings(
        cell_model="cellpose3:cyto3",
        nucleus_model="stardist:SD_Nuclei_Versatile",
        foci_model_1="stardist:SD_Foci_Finn",
        foci_channel_1=2,
        foci_model_2="stardist:SD_Foci_Finn",
        foci_channel_2=2,
        foci_model_3="spotiflow:general",
        foci_channel_3=3,
    )

    passes = build_model_passes(settings)

    assert [item.model_id for item in passes] == [
        "cellpose3:cyto3",
        "stardist:SD_Nuclei_Versatile",
        "stardist:SD_Foci_Finn",
        "spotiflow:general",
    ]
    repeated = passes[2]
    assert len(repeated.requests) == 1
    assert [consumer.step for consumer in repeated.requests[0].consumers] == [
        "Step 3a foci",
        "Step 3b foci",
    ]


def test_gpu_worker_capacity_reserves_memory_applies_margin_and_cap():
    assert calculate_gpu_workers(
        total_mb=10_000,
        free_mb=9_000,
        peak_worker_mb=1_000,
        task_count=10,
        cap=5,
    ) == 4
    assert calculate_gpu_workers(
        total_mb=8_000,
        free_mb=1_100,
        peak_worker_mb=2_000,
        task_count=20,
    ) == 1
    assert calculate_gpu_workers(
        total_mb=8_000,
        free_mb=7_000,
        peak_worker_mb=500,
        task_count=2,
    ) == 2


def test_gpu_probe_uses_complete_nvidia_process_memory(monkeypatch):
    monkeypatch.setattr(
        parallel_pipeline.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "1234, [N/A]\n"
                f"{os.getpid()}, 8192\n"
                "5678, 1024\n"
            ),
        ),
    )

    assert parallel_pipeline._process_gpu_memory_mb() == 8192.0


def test_parent_progress_reports_percentage_elapsed_rate_and_eta(monkeypatch):
    clock = iter((100.0, 110.0))
    monkeypatch.setattr(
        parallel_pipeline.time, "perf_counter", lambda: next(clock)
    )
    messages: list[str] = []
    progress = parallel_pipeline.PhaseProgress(
        "Label finalization", 100, messages.append
    )

    progress.set_completed(50)

    assert messages[0] == (
        "Label finalization: 0/100 (0.0%) | elapsed=00:00:00 "
        "| rate=0.00 fields/s | ETA=calculating"
    )
    assert messages[1] == (
        "Label finalization: 50/100 (50.0%) | elapsed=00:00:10 "
        "| rate=5.00 fields/s | ETA=00:00:10"
    )


@pytest.mark.parametrize(
    ("policy", "expected_generated", "expected_final"),
    [
        ("remove", ["labels_cells"], ["labels_cells"]),
        (
            "overwrite",
            ["labels_cells"],
            ["labels_cells_2", "unrelated", "labels_cells"],
        ),
        (
            "append",
            ["labels_cells_3"],
            ["labels_cells", "labels_cells_2", "unrelated", "labels_cells_3"],
        ),
    ],
)
def test_existing_label_policy_is_resolved_plate_wide(
    monkeypatch, policy, expected_generated, expected_final
):
    resources = [
        ImageResource(Path("plate.ome.zarr"), "A/1/0"),
        ImageResource(Path("plate.ome.zarr"), "B/1/0"),
    ]
    existing = {
        "A/1/0": ["labels_cells", "labels_cells_2", "unrelated"],
        "B/1/0": ["labels_cells"],
    }
    monkeypatch.setattr(
        parallel_pipeline,
        "existing_label_names",
        lambda resource: list(existing[resource.image_path]),
    )
    settings = SegmentationSettings(
        cell_model="cellpose3:cyto3",
        nucleus_model="skip",
        existing_labels=policy,
    )

    _, generated, final = resolve_label_policy(resources, settings)

    assert generated == expected_generated
    if policy == "append":
        assert final["A/1/0"] == expected_final
        assert final["B/1/0"] == ["labels_cells", "labels_cells_3"]
    elif policy == "overwrite":
        assert final["A/1/0"] == expected_final
        assert final["B/1/0"] == ["labels_cells"]
    else:
        assert final["A/1/0"] == expected_final
        assert final["B/1/0"] == expected_final


def test_invalid_declared_existing_label_is_rejected(tmp_path, inputfolder):
    source = tmp_path / "invalid.ome.zarr"
    shutil.copytree(inputfolder / "nuclei-small.ome.zarr", source)
    root = zarr.open_group(str(source), mode="a")
    root.require_group("labels").attrs["labels"] = ["missing"]
    root.store.close()

    with pytest.raises(ValueError, match="missing"):
        existing_label_names(enumerate_resources(source)[0])


def test_labels_only_output_is_sparse_and_source_is_byte_unchanged(
    tmp_path, inputfolder, monkeypatch
):
    source_fixture = inputfolder / "nuclei-small.ome.zarr"
    source, output = _run_small_store(
        tmp_path,
        source_fixture,
        monkeypatch,
        include_original_data=False,
    )
    before = _file_hashes(source_fixture)

    assert source.is_dir()
    assert _file_hashes(source) == before
    assert not (output / "0").exists()
    assert (output / "labels" / "labels_cells" / "0" / ".zarray").is_file()
    manifest = json.loads(
        (output / "cisegmentation-label-overlay.json").read_text(encoding="utf-8")
    )
    assert manifest["source_store"] == "sample.ome.zarr"
    assert manifest["existing_labels_policy"] == "overwrite"
    assert manifest["generated_label_mapping"] == {
        "labels_cells": "labels_cells"
    }


def test_staging_is_outside_source_when_input_and_output_folder_are_equal(
    tmp_path, inputfolder, monkeypatch
):
    working = tmp_path / "same-folder"
    working.mkdir()
    source = working / "sample.ome.zarr"
    shutil.copytree(inputfolder / "nuclei-small.ome.zarr", source)
    monkeypatch.setenv("CISEGMENTATION_INLINE_WORKERS", "1")
    monkeypatch.setattr(parallel_pipeline, "segment_czyx", _fake_segment)

    outputs = run_workflow(
        working,
        working,
        SegmentationSettings(
            cell_model="cellpose3:cyto3",
            nucleus_model="skip",
            remove_border_cells=False,
            include_original_data=False,
            measurements_database="skip",
            max_inference_workers=1,
            max_measurement_workers=1,
        ),
    )

    assert source.is_dir()
    assert outputs == [working / "sample__cisegmentation.ome.zarr"]
    assert not list(source.glob(".cisegmentation-staging-*"))
    assert not list(working.glob(".sample.cisegmentation-staging-*"))


def test_full_data_output_consumes_source_and_keeps_pixels(
    tmp_path, inputfolder, monkeypatch
):
    source, output = _run_small_store(
        tmp_path,
        inputfolder / "nuclei-small.ome.zarr",
        monkeypatch,
        include_original_data=True,
    )

    assert not source.exists()
    assert (output / "0" / ".zarray").is_file()
    assert (output / "labels" / "labels_cells" / "0" / ".zarray").is_file()
    assert not (
        output / ".cisegmentation-label-journal.json"
    ).exists()
    assert not any(
        path.name.startswith(".cisegmentation-label-backup-")
        for path in output.iterdir()
    )
    root = zarr.open_group(str(output), mode="r")
    assert root.attrs["cisegmentation"]["output_mode"] == "full-data"
    assert root.attrs["cisegmentation"]["timings"]["total_seconds"] > 0
    root.store.close()


def test_overlay_manifest_merge_matches_full_data_labels(
    tmp_path, inputfolder, monkeypatch
):
    overlay_source, overlay = _run_small_store(
        tmp_path / "overlay-run",
        inputfolder / "nuclei-small.ome.zarr",
        monkeypatch,
        include_original_data=False,
    )
    _consumed_source, full = _run_small_store(
        tmp_path / "full-run",
        inputfolder / "nuclei-small.ome.zarr",
        monkeypatch,
        include_original_data=True,
    )
    manifest = json.loads(
        (overlay / "cisegmentation-label-overlay.json").read_text(
            encoding="utf-8"
        )
    )
    for field in manifest["fields"]:
        for relative in field["replace_paths"]:
            destination = overlay_source / relative
            if destination.exists():
                shutil.rmtree(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(overlay / relative, destination)
        label_prefix = (
            overlay_source / field["resource_path"] / "labels"
            if field["resource_path"]
            else overlay_source / "labels"
        )
        overlay_prefix = (
            overlay / field["resource_path"] / "labels"
            if field["resource_path"]
            else overlay / "labels"
        )
        shutil.copy2(overlay_prefix / ".zgroup", label_prefix / ".zgroup")
        shutil.copy2(overlay_prefix / ".zattrs", label_prefix / ".zattrs")

    merged = zarr.open_group(str(overlay_source), mode="r")
    expected = zarr.open_group(str(full), mode="r")
    np.testing.assert_array_equal(merged["0"], expected["0"])
    np.testing.assert_array_equal(
        merged["labels/labels_cells/0"],
        expected["labels/labels_cells/0"],
    )
    assert merged["labels"].attrs["labels"] == expected["labels"].attrs["labels"]
    merged.store.close()
    expected.store.close()


def test_failed_full_data_publication_restores_source_byte_for_byte(
    tmp_path, inputfolder, monkeypatch
):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    source = input_dir / "sample.ome.zarr"
    shutil.copytree(inputfolder / "nuclei-small.ome.zarr", source)
    before = _file_hashes(source)
    monkeypatch.setenv("CISEGMENTATION_INLINE_WORKERS", "1")
    monkeypatch.setattr(parallel_pipeline, "segment_czyx", _fake_segment)
    monkeypatch.setattr(
        parallel_pipeline,
        "publish_consumed_store",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("publication failed")
        ),
    )

    with pytest.raises(OSError, match="publication failed"):
        run_workflow(
            input_dir,
            output_dir,
            SegmentationSettings(
                cell_model="cellpose3:cyto3",
                nucleus_model="skip",
                remove_border_cells=False,
                include_original_data=True,
                measurements_database="skip",
                max_inference_workers=1,
                max_measurement_workers=1,
            ),
        )

    assert source.is_dir()
    assert _file_hashes(source) == before
    assert not (
        output_dir / "sample__cisegmentation.ome.zarr"
    ).exists()


def test_cuda_oom_retry_halves_inference_pool(
    tmp_path, inputfolder, monkeypatch
):
    resources = [
        ImageResource(inputfolder / "nuclei-small.ome.zarr", "field1"),
        ImageResource(inputfolder / "nuclei-medium.ome.zarr", "field2"),
    ]
    calls: list[int] = []

    def fake_execute(payloads, workers, **_kwargs):
        calls.append(workers)
        if len(calls) == 1:
            resource = payloads[0]["resource"]
            return [
                {
                    "ok": True,
                    "resource_path": resource.image_path,
                    "records": {"request_01": [{}]},
                    "runtime_seconds": 0.1,
                    "zarr_read_seconds": 0.01,
                    "peak_cuda_mb": 100,
                    "rss_mb": 100,
                    "device": "cpu",
                }
            ]
        if len(calls) == 2:
            return [
                {
                    "ok": False,
                    "resource_path": payloads[0]["resource"].image_path,
                    "error": "CUDA out of memory",
                    "traceback": "CUDA out of memory",
                    "cuda_oom": True,
                }
            ]
        resource = payloads[0]["resource"]
        return [
            {
                "ok": True,
                "resource_path": resource.image_path,
                "records": {"request_01": [{}]},
                "runtime_seconds": 0.1,
                "zarr_read_seconds": 0.01,
                "peak_cuda_mb": 100,
                "rss_mb": 100,
                "device": "cpu",
            }
        ]

    monkeypatch.setattr(parallel_pipeline, "_execute_tasks", fake_execute)
    monkeypatch.setattr(
        parallel_pipeline, "_largest_resource", lambda _resources: resources[0]
    )
    monkeypatch.setattr(
        parallel_pipeline, "calculate_cpu_workers", lambda *_args, **_kwargs: 4
    )
    settings = SegmentationSettings(
        cell_model="cellpose3:cyto3",
        nucleus_model="skip",
    )

    _, _, provenance = parallel_pipeline.run_inference_passes(
        resources, settings, tmp_path / "stage"
    )

    assert calls == [1, 4, 2]
    assert provenance["inference_passes"][0]["workers"] == 2
    assert provenance["inference_passes"][0]["retries"] == 1


def test_cross_filesystem_publication_copies_verifies_and_replaces_destination(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.ome.zarr"
    destination = tmp_path / "result.ome.zarr"
    source.mkdir()
    (source / ".zgroup").write_text("new", encoding="utf-8")
    destination.mkdir()
    (destination / ".zgroup").write_text("old", encoding="utf-8")
    original_replace = parallel_pipeline._replace_destination
    attempts = 0

    def force_cross_filesystem(origin, target):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(errno.EXDEV, "cross-device link")
        return original_replace(origin, target)

    monkeypatch.setattr(
        parallel_pipeline, "_replace_destination", force_cross_filesystem
    )

    assert publish_consumed_store(source, destination) == "copy"
    assert not source.exists()
    assert (destination / ".zgroup").read_text(encoding="utf-8") == "new"
    assert not destination.with_name(destination.name + ".previous").exists()


def test_cross_filesystem_label_install_uses_verified_copy(
    tmp_path, monkeypatch
):
    source = tmp_path / "overlay-labels"
    destination = tmp_path / "source-field" / "labels"
    source.mkdir()
    (source / ".zgroup").write_text("metadata", encoding="utf-8")
    original_rename = parallel_pipeline._rename_with_retry
    first = True

    def force_cross_filesystem(origin, target):
        nonlocal first
        if first:
            first = False
            raise OSError(errno.EXDEV, "cross-device link")
        return original_rename(origin, target)

    monkeypatch.setattr(
        parallel_pipeline, "_rename_with_retry", force_cross_filesystem
    )

    assert (
        parallel_pipeline._move_or_copy_tree(source, destination) == "copy"
    )
    assert not source.exists()
    assert (destination / ".zgroup").read_text(encoding="utf-8") == "metadata"


def test_interrupted_label_commit_journal_restores_original_tree(tmp_path):
    store = tmp_path / "plate.ome.zarr"
    labels = store / "A" / "1" / "0" / "labels"
    backup = store / ".label-backup" / "field" / "labels"
    (labels / "labels_cells").mkdir(parents=True)
    (labels / "labels_cells" / "value").write_text("new", encoding="utf-8")
    (labels / "unrelated").mkdir()
    (labels / "unrelated" / "value").write_text("preserved", encoding="utf-8")
    (backup / "labels_cells").mkdir(parents=True)
    (backup / "labels_cells" / "value").write_text("old", encoding="utf-8")
    (store / ".cisegmentation-label-journal.json").write_text(
        json.dumps(
            {
                "version": 1,
                "backup_root": ".label-backup",
                "fields": [
                    {
                        "labels_path": "A/1/0/labels",
                        "backup_path": ".label-backup/field/labels",
                        "had_labels": True,
                        "preserve": ["unrelated"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert recover_label_commit(store) is True
    assert (labels / "labels_cells" / "value").read_text(
        encoding="utf-8"
    ) == "old"
    assert (labels / "unrelated" / "value").read_text(
        encoding="utf-8"
    ) == "preserved"
    assert not (store / ".label-backup").exists()
    assert not (
        store / ".cisegmentation-label-journal.json"
    ).exists()
