---
name: analyze-cisegmentation-measurements
description: Analyze, query, and explain CI Segmentation measurement databases in DuckDB or SQLite. Use in JupyterLite AI for object morphology, per-channel intensity statistics, label sets, image or HCS plate metadata, mask relationships, focus assignments, SQL, pandas, and interpretation of CI Segmentation measurement results.
---

# Instructions

Help the user inspect and analyze a CI Segmentation measurements database in
JupyterLite.

## Load the reference

Before writing a nontrivial query or interpreting measurement fields, load:

```text
references/REFERENCE.md
```

Call `load_skill` with:

```json
{
  "name": "analyze-cisegmentation-measurements",
  "resource": "references/REFERENCE.md"
}
```

Use that resource as the authoritative reference for the database schema,
measurement semantics, convenience views, and query examples.

## Workflow

1. Identify the uploaded `.duckdb` or `.sqlite` file.
2. Open the database read-only.
3. Inspect its actual tables, views, and columns before drafting a substantive
   query. Do not assume every database uses the latest schema.
4. Read `schema_info` and `measurement_runs` when schema version, provenance,
   workflow settings, or source and output paths matter.
5. Prefer the documented convenience views when they contain the needed
   context.
6. Filter and aggregate in SQL before returning data to pandas.
7. Explain the queried columns, filters, grouping, units, relationship
   direction, and relevant calibration or missing-value caveats.
8. Close the database connection.

## JupyterLite constraints

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

db_path = "screen_multistep_measurements.duckdb"
db = duckdb.connect(db_path, read_only=True)
```

Open SQLite read-only:

```python
from pathlib import Path
import sqlite3

db_path = Path("screen_multistep_measurements.sqlite").resolve()
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
