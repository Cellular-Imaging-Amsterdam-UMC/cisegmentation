---
name: analyze-cisegmentation-measurements
description: Analyze, query, and explain CI Segmentation measurement databases in DuckDB or SQLite. Use in JupyterLite AI for object morphology, per-channel intensity statistics, label sets, image or HCS plate metadata, mask relationships, focus assignments, SQL, pandas, and interpretation of CI Segmentation measurement results.
metadata:
  version: "4"
  biomero-purpose: "attachment-analysis"
  biomero-consumers: "omero-analysis,omero-jupyterlite"
  biomero-auto-activate: "true"
  biomero-file-extensions: ".duckdb,.sqlite"
  biomero-filename-globs: "*__cisegmentation_measurements.duckdb,*__cisegmentation_measurements.sqlite"
  biomero-required-tables: "schema_info,measurement_runs"
  biomero-required-resources: "references/REFERENCE.md"
  biomero-required-capabilities: "sql-readonly,zarr-render-v2,zarr-gallery-v1"
---

# Instructions

Help the user inspect and analyze an attached CI Segmentation measurements
database in OMERO Analysis or JupyterLite.

## Required reference

The consumer must load this reference automatically when the skill activates:

```text
references/REFERENCE.md
```

Use that resource as the authoritative reference for the database schema,
measurement semantics, convenience views, and query examples.

## Workflow

1. Identify the attached or uploaded `.duckdb` or `.sqlite` file.
2. Open the database read-only.
3. Read `schema_info` once. Inspect tables or columns only when the schema
   version is unknown or the documented query fails. Reuse verified schema
   facts for every follow-up while the file hash is unchanged.
4. Read `schema_info` and `measurement_runs` when schema version, provenance,
   workflow settings, output store identity, or source and output paths matter.
5. Prefer the documented convenience views when they contain the needed
   context.
6. Filter and aggregate in SQL before returning data to pandas.
7. Explain the queried columns, filters, grouping, units, relationship
   direction, and relevant calibration or missing-value caveats.
8. Close the database connection.

For object-to-render requests, use the canonical combined query in the
reference so the object, its related objects, navigation rows, and render
coordinates are resolved in one execution. Reuse that successful result for
rendering; do not rediscover files or schema. Treat image-QC findings as
**review candidates**, never definitive bad images.

For questions whose answer should be a PNG, load
`references/PNG_QUESTIONS.md` only when needed. Prefer a single bounded gallery
request over one render request per object.

For a gallery, the analysis result is also the render contract. Assign
`result = {"store_uuid": store_uuid, "render_panels": panels}` and include
every panel's exact field, ROI, source channels, overlay paths and values,
T/Z, title, and caption. Do not return only a shortened display table or rely
on another Python variable: consumers receive only `result`. Copy
`render_panels` unchanged into the gallery tool so label values cannot become
detached from their field and bounding box.

## JupyterLite constraints

Apply these constraints only when working in JupyterLite:

- Use paths in the JupyterLite browser filesystem.
- Do not use host-computer paths, shell commands, Docker, or Conda.
- Expect browser execution to be single-threaded and memory-limited.
- Avoid loading whole multi-million-row tables into pandas.
- Install DuckDB with `%pip install duckdb` in a notebook cell if importing it
  fails.
- Use Python's standard-library `sqlite3` module for SQLite.
- Never overwrite the uploaded database.

## Database connections

Open DuckDB read-only:

```python
import duckdb

db_path = "screen__cisegmentation_measurements.duckdb"
db = duckdb.connect(db_path, read_only=True)
```

Open SQLite read-only:

```python
from pathlib import Path
import sqlite3

db_path = Path("screen__cisegmentation_measurements.sqlite").resolve()
db = sqlite3.connect(
    f"file:{db_path.as_posix()}?mode=ro",
    uri=True,
)
```

If the database path is unknown, locate candidates with:

```python
from pathlib import Path

database_files = sorted(
    list(Path.cwd().rglob("*.duckdb"))
    + list(Path.cwd().rglob("*.sqlite"))
)
database_files
```

## Analysis rules

- Inspect relation columns before adapting a query from the reference.
- Use parameters for user-supplied values when the database API supports them.
- Treat pixel coordinates and timepoints as zero-based.
- Treat `channel_index` as one-based.
- Treat bounding-box minima as inclusive and maxima as exclusive.
- Use `object_navigation` for viewer/ROI coordinates when schema version 3 or
  newer provides it. Never invent an OMERO Image or Plate ID from a portable
  database.
- Treat `output_store_uuid` as the cross-check between a database and an
  active output OME-Zarr. Refuse navigation when both UUIDs exist and differ.
- Do not substitute pixel units for unavailable physical units.
- Interpret relationships directionally; do not interchange source and target.
- Distinguish 2D masks, true 3D masks, and point-only objects.
- Do not claim that intensities were normalized or background-corrected.
- Separate database observations from biological interpretation.

## Response

Provide executable notebook cells when the user needs code. Report:

- the database engine and file;
- the tables or views queried;
- the SQL or analysis logic;
- filters, grouping keys, and relationship direction;
- measurement units;
- calibration, anisotropy, point-only, and missing-value caveats.

If a query fails because the actual schema differs from the reference, inspect
the schema, revise the query, and explain the difference.
