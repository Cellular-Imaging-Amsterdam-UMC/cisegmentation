from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import multiprocessing
import os
from pathlib import Path
import sqlite3
import time
import traceback
from typing import Any, Callable

from . import __version__
from .measurements import (
    FIELD_QUALITY_COLUMNS,
    IMAGE_QUALITY_COLUMNS,
    SCHEMA_VERSION,
    _DatabaseWriter,
    _field_quality_rows,
    _quality_rows,
    write_measurements_database,
)
from .parallel_pipeline import (
    PhaseProgress,
    _iter_future_results,
    _worker_environment,
    available_cpu_workers,
    resolved_label_result,
)
from .settings import SegmentationSettings


def _measurement_task(payload: dict[str, Any]) -> dict[str, Any]:
    _worker_environment()
    started = time.perf_counter()
    shard = Path(payload["shard"])
    try:
        result = resolved_label_result(
            payload["resource"],
            Path(payload["overlay_path"]),
            payload["final_names"],
            payload["generated_names"],
            payload["provenance"],
        )
        summary = write_measurements_database(
            [result],
            shard,
            payload["database_format"],
            output_ome_zarr=payload["output_ome_zarr"],
            output_store_uuid=payload["output_store_uuid"],
        )
        return {
            "ok": True,
            "resource_path": payload["resource"].image_path,
            "shard": str(shard),
            "summary": summary,
            "runtime_seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        shard.unlink(missing_ok=True)
        return {
            "ok": False,
            "resource_path": payload["resource"].image_path,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _open_database(path: Path, database_format: str):
    if database_format == "sqlite":
        return sqlite3.connect(path)
    import duckdb

    return duckdb.connect(str(path), read_only=True)


def _table_rows(connection, table: str) -> tuple[tuple[str, ...], list[tuple]]:
    info = connection.execute(f"PRAGMA table_info('{table}')").fetchall()
    columns = tuple(str(row[1]) for row in info)
    rows = [tuple(row) for row in connection.execute(f"SELECT * FROM {table}").fetchall()]
    return columns, rows


def _remap(row: tuple, offsets: dict[int, int]) -> tuple:
    values = list(row)
    for index, offset in offsets.items():
        values[index] = int(values[index]) + offset
    return tuple(values)


def merge_measurement_shards(
    shards: list[Path],
    output_path: Path,
    database_format: str,
    *,
    source_store: str,
    output_ome_zarr: str | Path,
    output_store_uuid: str,
    settings: SegmentationSettings,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Merge independent field databases through one parent writer."""
    temporary = output_path.with_name(output_path.name + ".partial")
    temporary.unlink(missing_ok=True)
    writer = _DatabaseWriter(temporary, database_format)
    started = time.perf_counter()
    counts = {
        "images": 0,
        "label_sets": 0,
        "objects": 0,
        "intensities": 0,
        "image_quality": 0,
        "field_quality": 0,
        "relationships": 0,
    }
    writer.insert(
        "schema_info",
        ("key", "value"),
        [
            ("format", "CI Segmentation measurements"),
            ("schema_version", str(SCHEMA_VERSION)),
            ("coordinate_unit", "micrometer"),
            ("bbox_maximum", "exclusive"),
        ],
    )
    writer.insert(
        "measurement_runs",
        (
            "run_id",
            "created_utc",
            "software_version",
            "schema_version",
            "database_format",
            "source_store",
            "output_ome_zarr",
            "output_store_uuid",
            "settings_json",
        ),
        [
            (
                1,
                datetime.now(timezone.utc).isoformat(),
                __version__,
                SCHEMA_VERSION,
                database_format,
                source_store,
                str(output_ome_zarr),
                output_store_uuid,
                __import__("json").dumps(settings.to_dict(), sort_keys=True),
            )
        ],
    )
    offsets = {
        "image": 0,
        "channel": 0,
        "label": 0,
        "object": 0,
        "relationship": 0,
    }
    image_quality_records: list[dict[str, Any]] = []
    field_quality_records: list[dict[str, Any]] = []
    progress = PhaseProgress(
        "Measurement database merge",
        len(shards),
        log,
    )
    finished = False
    try:
        for shard in shards:
            connection = _open_database(shard, database_format)
            columns, images = _table_rows(connection, "images")
            mapped_images = [
                _remap(row, {0: offsets["image"]})[:1]
                + (1,)
                + _remap(row, {0: offsets["image"]})[2:]
                for row in images
            ]
            writer.insert("images", columns, mapped_images)

            channel_columns, channels = _table_rows(connection, "channels")
            channel_index_by_id = {
                int(row[0]): int(row[2]) for row in channels
            }
            writer.insert(
                "channels",
                channel_columns,
                [
                    _remap(
                        row,
                        {0: offsets["channel"], 1: offsets["image"]},
                    )
                    for row in channels
                ],
            )
            label_columns, labels = _table_rows(connection, "label_sets")
            writer.insert(
                "label_sets",
                label_columns,
                [
                    _remap(row, {0: offsets["label"], 1: offsets["image"]})
                    for row in labels
                ],
            )
            source_columns, sources = _table_rows(
                connection, "label_set_sources"
            )
            writer.insert(
                "label_set_sources",
                source_columns,
                [
                    _remap(
                        row,
                        {0: offsets["label"], 1: offsets["channel"]},
                    )
                    for row in sources
                ],
            )
            object_columns, objects = _table_rows(connection, "objects")
            writer.insert(
                "objects",
                object_columns,
                [
                    _remap(
                        row,
                        {
                            0: offsets["object"],
                            1: offsets["image"],
                            2: offsets["label"],
                        },
                    )
                    for row in objects
                ],
            )
            intensity_columns, intensities = _table_rows(
                connection, "intensity_measurements"
            )
            writer.insert(
                "intensity_measurements",
                intensity_columns,
                [
                    _remap(
                        row,
                        {0: offsets["object"], 1: offsets["channel"]},
                    )
                    for row in intensities
                ],
            )
            relationship_columns, relationships = _table_rows(
                connection, "relationships"
            )
            writer.insert(
                "relationships",
                relationship_columns,
                [
                    _remap(
                        row,
                        {
                            0: offsets["relationship"],
                            1: offsets["image"],
                            3: offsets["object"],
                            4: offsets["object"],
                            5: offsets["label"],
                            6: offsets["label"],
                        },
                    )
                    for row in relationships
                ],
            )
            quality_columns, quality_rows = _table_rows(
                connection, "image_quality_measurements"
            )
            for row in quality_rows:
                record = dict(zip(quality_columns, row))
                record["channel_index"] = channel_index_by_id[
                    int(record["channel_id"])
                ]
                record["image_id"] += offsets["image"]
                record["channel_id"] += offsets["channel"]
                image_quality_records.append(record)
            field_columns, field_rows = _table_rows(
                connection, "field_quality_measurements"
            )
            for row in field_rows:
                record = dict(zip(field_columns, row))
                field_quality_records.append(
                    {
                        "image_id": record["image_id"] + offsets["image"],
                        "timepoint": record["timepoint"],
                        "cell_count": record["cell_count"],
                        "nucleus_count": record["nucleus_count"],
                        "foci_count": record["foci_count"],
                        "total_label_count": record["total_label_count"],
                    }
                )
            connection.close()

            counts["images"] += len(images)
            counts["label_sets"] += len(labels)
            counts["objects"] += len(objects)
            counts["intensities"] += len(intensities)
            counts["relationships"] += len(relationships)
            offsets["image"] += len(images)
            offsets["channel"] += len(channels)
            offsets["label"] += len(labels)
            offsets["object"] += len(objects)
            offsets["relationship"] += len(relationships)
            progress.advance()

        if log:
            log(
                "Measurement database merge: calculating plate-wide quality "
                "scores and finalizing indexes/views"
            )
        for index, record in enumerate(image_quality_records, start=1):
            record["image_quality_id"] = index
        quality_rows = _quality_rows(image_quality_records)
        writer.insert(
            "image_quality_measurements",
            IMAGE_QUALITY_COLUMNS,
            quality_rows,
        )
        field_rows = _field_quality_rows(
            field_quality_records, image_quality_records
        )
        writer.insert(
            "field_quality_measurements",
            FIELD_QUALITY_COLUMNS,
            field_rows,
        )
        counts["image_quality"] = len(quality_rows)
        counts["field_quality"] = len(field_rows)
        writer.finish()
        finished = True
        for attempt in range(50):
            try:
                os.replace(temporary, output_path)
                break
            except PermissionError:
                if attempt == 49:
                    raise
                time.sleep(0.1)
    except Exception:
        if not finished:
            writer.abort()
        temporary.unlink(missing_ok=True)
        raise
    counts.update(
        {
            "runtime_seconds": time.perf_counter() - started,
            "path": str(output_path),
            "format": database_format,
        }
    )
    return counts


def write_parallel_measurements(
    *,
    resources,
    settings: SegmentationSettings,
    overlay_path: Path,
    final_by_resource: dict[str, list[str]],
    generated_names: list[str],
    finalization: dict[str, dict],
    shard_dir: Path,
    output_path: Path,
    output_ome_zarr: Path,
    output_store_uuid: str,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    phase_started = time.perf_counter()
    shard_dir.mkdir(parents=True, exist_ok=True)
    workers = available_cpu_workers(
        len(resources), settings.max_measurement_workers
    )
    progress = PhaseProgress(
        "Measurements",
        len(resources),
        log,
    )
    pending = list(resources)
    completed: dict[str, Path] = {}
    retries = 0
    context = multiprocessing.get_context("spawn")
    while pending:
        payloads = []
        for resource in pending:
            suffix = resource.image_path.replace("/", "_") or "root"
            payloads.append(
                {
                    "resource": resource,
                    "overlay_path": str(overlay_path),
                    "final_names": final_by_resource[resource.image_path],
                    "generated_names": generated_names,
                    "provenance": finalization[resource.image_path][
                        "provenance"
                    ],
                    "shard": str(
                        shard_dir
                        / f"{suffix}.{settings.measurements_database}"
                    ),
                    "database_format": settings.measurements_database,
                    "output_ome_zarr": str(output_ome_zarr),
                    "output_store_uuid": output_store_uuid,
                }
            )
        results = []
        if workers == 1 and os.environ.get("CISEGMENTATION_INLINE_WORKERS") == "1":
            for payload in payloads:
                result = _measurement_task(payload)
                results.append(result)
                if result["ok"]:
                    progress.advance()
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_worker_environment,
            ) as executor:
                futures = {
                    executor.submit(_measurement_task, payload): payload[
                        "resource"
                    ].image_path
                    for payload in payloads
                }
                for result in _iter_future_results(futures, progress):
                    results.append(result)
                    if result["ok"]:
                        progress.advance()
        failures = {
            result["resource_path"] for result in results if not result["ok"]
        }
        for result in results:
            if result["ok"]:
                completed[result["resource_path"]] = Path(result["shard"])
        if not failures:
            break
        retries += 1
        if retries > 2:
            first = next(result for result in results if not result["ok"])
            raise RuntimeError(
                f"Measurements failed after two retries: {first['error']}\n"
                f"{first['traceback']}"
            )
        pending = [
            resource for resource in pending if resource.image_path in failures
        ]
        if log:
            log(f"Measurement retry {retries}/2 for {len(pending)} field(s)")

    ordered_shards = [completed[resource.image_path] for resource in resources]
    summary = merge_measurement_shards(
        ordered_shards,
        output_path,
        settings.measurements_database,
        source_store=resources[0].store_path.name,
        output_ome_zarr=output_ome_zarr,
        output_store_uuid=output_store_uuid,
        settings=settings,
        log=log,
    )
    summary["runtime_seconds"] = time.perf_counter() - phase_started
    return summary, {"workers": workers, "retries": retries}
