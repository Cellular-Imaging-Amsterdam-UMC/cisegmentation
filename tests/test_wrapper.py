from pathlib import Path

import wrapper


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
