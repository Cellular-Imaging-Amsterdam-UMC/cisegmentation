import json
import sys

import numpy as np
import pytest
import zarr

from cisegmentation.registry import MODEL_REGISTRY
from tools import test_model_matrix as matrix


def test_plan_covers_every_checkpoint_target_and_native_mode():
    plan = matrix.make_plan()
    assert {case["model"] for case in plan} == set(MODEL_REGISTRY)
    assert len({case["id"] for case in plan}) == len(plan)
    for spec in MODEL_REGISTRY.values():
        cases = [case for case in plan if case["model"] == spec.id]
        assert {case["target"] for case in cases} == set(spec.targets)
        assert {case["stage"] for case in cases} == {"small", "large"}
        if spec.dimensions == "3d":
            assert any(
                case["stage"] == "small" and case["mode"] == "native-3d"
                for case in cases
            )


def test_subset_quick_plan_and_unknown_checkpoint():
    plan = matrix.make_plan("stardist:SD_Nuclei_Versatile", quick_only=True)
    assert len(plan) == 1 and plan[0]["stage"] == "small"
    with pytest.raises(ValueError, match="Unknown model"):
        matrix.make_plan(["missing-model"])


def test_empty_reference_is_inconclusive_and_lost_positive_reference_fails():
    empty = np.zeros((1, 8, 8), dtype="u4")
    assert (
        matrix.compare_labels(empty, empty)["quality_status"]
        == "inconclusive_no_reference_detections"
    )
    positive = empty.copy()
    positive[0, 3, 3] = 1
    with pytest.raises(AssertionError, match="lost every") as caught:
        matrix.compare_labels(positive, empty)
    assert caught.value.comparison["quality_status"] == "lost_reference_detections"
    assert caught.value.comparison["untiled_objects"] == 1
    assert caught.value.comparison["tiled_objects"] == 0
    shifted = empty.copy()
    shifted[0, 3, 4] = 1
    assert (
        matrix.compare_labels(positive, shifted, points=True)["point_recall_within_2px"]
        == 1
    )


def test_synthetic_channels_and_z_are_generated_only_for_requested_region(tmp_path):
    path = tmp_path / "source.ome.zarr"
    root = zarr.open_group(str(path), mode="w")
    source = np.arange(16 * 16, dtype="u2").reshape(1, 1, 1, 16, 16)
    root.create_dataset("0", data=source)
    root.attrs["multiscales"] = [
        {
            "axes": list("tczyx"),
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1, 1, 1, 1, 1]}
                    ],
                }
            ],
        }
    ]
    case = {
        "stage": "small",
        "model": "instanseg:brightfield_nuclei",
        "xy": 512,
        "z": 8,
    }
    image, info = matrix.prepare_image(case, {"fixture": str(path)})
    assert image.data.shape == (1, 3, 8, 16, 16)
    assert info["repeated_channels"] and info["synthetic_z_volume"]
    region = image.data[0, :, 3:5, 2:4, 5:8]
    assert region.shape == (3, 2, 2, 3)
    np.testing.assert_array_equal(region[0, 1], source[0, 0, 0, 2:4, 5:8])
    np.testing.assert_array_equal(region[0], region[2])


def test_timeout_returns_control(tmp_path):
    exit_code, timeout = matrix.run_isolated(
        [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "log.txt", 0.2
    )
    assert timeout and exit_code != 0


def test_failure_does_not_stop_later_cases_and_resume_preserves_attempts(
    tmp_path, monkeypatch
):
    calls = []

    def run(command, log_path, timeout):
        output = matrix.Path(command[command.index("--worker-output") + 1])
        case = json.loads((output / "case.json").read_text())
        calls.append(case["id"])
        failed = case["stage"] == "small"
        matrix.save_json(
            output / "report.json",
            {**case, "execution_status": "failed" if failed else "passed"},
        )
        return int(failed), False

    monkeypatch.setattr(matrix, "run_isolated", run)
    config = {"output": str(tmp_path), "models": ["stardist:SD_Nuclei_Versatile"]}
    assert matrix.coordinate(config, tmp_path / "config.json") == 1
    assert len(calls) == 2
    assert (tmp_path / "evidence.tar.gz").exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["completed_cases"] == 2
    assert summary["execution_counts"] == {"failed": 1, "passed": 1}
    matrix.coordinate(config, tmp_path / "config.json")
    assert len(calls) == 2
    matrix.coordinate(config, tmp_path / "config.json", retry_failures=True)
    assert len(calls) == 3
    assert len(list((tmp_path / "cases").glob("*/attempt-*"))) == 3
