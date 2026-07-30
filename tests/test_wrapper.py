from pathlib import Path
import json

import wrapper


def test_output_timing_line_reads_root_provenance(tmp_path):
    output = tmp_path / "labels.ome.zarr"
    output.mkdir()
    timings = {key: index / 10 for index, (key, _label) in enumerate(wrapper._TIMING_LABELS)}
    (output / ".zattrs").write_text(
        json.dumps(
            {
                "cisegmentation": {
                    "timings": timings,
                    "runtime_seconds": 2.0,
                    "segmentation_count": 4,
                    "model_cache_hits": 3,
                    "model_cache_misses": 1,
                    "result_cache_hits": 2,
                }
            }
        ),
        encoding="utf-8",
    )

    line = wrapper.output_timing_line(output)
    assert line is not None
    assert "startup=0.00s" in line
    assert "inference=0.50s" in line
    assert "segmentations-total=2.00s" in line
    assert "segmentation-count=4" in line
    assert "per-segmentation=0.50s" in line
    assert "ome-zarr-read=0.10s" in line
    assert "ome-zarr-write=0.60s" in line
    assert "measurements=0.70s" in line
    assert "total=0.80s" in line
    assert "cache-hits=3 | cache-misses=1" in line
    assert "result-reuses=2" in line


def test_success_message_is_last_and_contains_elapsed_time(monkeypatch, capsys):
    monkeypatch.setattr(
        wrapper,
        "run_workflow",
        lambda *_args, **_kwargs: [Path("result.ome.zarr")],
    )

    assert wrapper.main([]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-2] == "Output: result.ome.zarr"
    assert lines[-1].startswith("CI segmentation completed in ")
    assert lines[-1].endswith(" seconds.")
