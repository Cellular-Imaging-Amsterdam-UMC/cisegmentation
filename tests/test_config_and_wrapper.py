from pathlib import Path

from bilayers_cli import generate_cli_command, load_config, validate_config
from cisegmentation.settings import (
    CELL_MODELS,
    EXPANSION_PREFIX,
    FOCI_MODELS,
    SKIP,
    STEP1_NUCLEUS_MODELS,
    STEP2_NUCLEUS_MODELS,
    normalize_legacy_workflow_values,
)
from wrapper import build_parser


def test_bilayers_config_is_structurally_valid():
    config = load_config()
    assert validate_config(config) == []
    parameters = {item["name"]: item for item in config["parameters"]}
    assert "instanseg_pixel_size_um" not in parameters
    assert "input_channels" not in parameters
    assert parameters["diameter"]["mode"] == "advanced"
    assert parameters["diameter"]["minimum"] == -1.0
    assert parameters["spotiflow_min_distance"]["type"] == "float"
    assert "multi_step" not in parameters
    removed = {
        "cell_step",
        "cell_method",
        "cell_nuclei_model",
        "cell_expansion_nucleus_model",
        "nucleus_step",
        *(f"foci_step_{slot}" for slot in range(1, 5)),
    }
    assert removed.isdisjoint(parameters)
    assert parameters["cell_model"]["default"] == "cellpose3:cyto3"
    assert parameters["nucleus_model"]["default"] == SKIP
    assert all(parameters[f"foci_model_{slot}"]["default"] == SKIP for slot in range(1, 5))
    assert parameters["cell_channel"]["default"] == 1
    assert parameters["cell_channel"]["label"] == "Step 1 Cyto Channel"
    assert parameters["cell_nuclei_channel"]["default"] == 0
    assert parameters["nucleus_channel"]["default"] == 1
    assert all(parameters[f"foci_channel_{slot}"]["default"] == 1 for slot in range(1, 5))
    assert parameters["include_original_channels"]["default"] is False
    assert parameters["write_ome_zarr_labels"]["default"] is False
    assert parameters["write_ome_zarr_labels"]["mode"] == "advanced"
    assert parameters["write_ome_zarr_labels"]["section_id"] == "advanced"
    assert parameters["smooth_stardist_labels"]["default"] is True
    assert parameters["smooth_stardist_labels"]["mode"] == "advanced"
    assert parameters["remove_border_cells"]["default"] is True
    assert parameters["remove_border_cells"]["mode"] == "beginner"
    parameter_names = [item["name"] for item in config["parameters"]]
    beginner_names = [
        item["name"]
        for item in config["parameters"]
        if item.get("mode") == "beginner"
    ]
    assert beginner_names.index("cell_channel") + 1 == beginner_names.index(
        "cell_nuclei_channel"
    )
    assert beginner_names.index("cell_nuclei_channel") + 1 == beginner_names.index(
        "cell_expansion_distance"
    )
    assert beginner_names[-2:] == ["remove_border_cells", "include_original_channels"]
    assert parameters["include_original_channels"]["mode"] == "beginner"
    assert parameters["include_original_channels"]["section_id"] == "essential"
    assert parameters["benchmark"]["mode"] == "advanced"
    advanced_names = [
        item["name"]
        for item in config["parameters"]
        if item.get("mode") == "advanced"
    ]
    assert all(
        parameters[f"foci_model_{slot}"]["section_id"] == "essential"
        and parameters[f"foci_model_{slot}"]["mode"] == "beginner"
        for slot in range(1, 5)
    )
    spot_models = {option["value"] for option in parameters["foci_model_1"]["options"]}
    assert {
        "stardist:SD_Foci_Aggregates",
        "stardist:SD_Foci_Finn",
        "cellpose3:bact_phase_cp3",
        "cellpose3:bact_fluor_cp3",
    } <= spot_models
    assert "benchmark_models" not in parameters
    assert tuple(option["value"] for option in parameters["cell_model"]["options"]) == (
        SKIP,
        *CELL_MODELS,
        *(f"{EXPANSION_PREFIX}{model}" for model in STEP1_NUCLEUS_MODELS),
    )
    assert tuple(
        option["value"] for option in parameters["nucleus_model"]["options"]
    ) == (SKIP, *STEP2_NUCLEUS_MODELS)
    assert tuple(
        option["value"] for option in parameters["foci_model_1"]["options"]
    ) == (SKIP, *FOCI_MODELS)


def test_wrapper_accepts_hyphenated_bilayers_parameters():
    args = build_parser().parse_args(
        [
            "--infolder",
            "in",
            "--outfolder",
            "out",
            "--model",
            "stardist:SD_Foci_Finn",
            "--target",
            "foci",
            "--primary-channel",
            "2",
            "--benchmark",
            "true",
        ]
    )
    assert args.model == "stardist:SD_Foci_Finn"
    assert args.primary_channel == 2
    assert args.benchmark is True


def test_bilayers_serializes_skip_selectors():
    command = generate_cli_command(
        load_config(), {"cell_model": SKIP, "nucleus_model": "cellpose3:nuclei"}
    )
    assert "--cell-model skip" in command
    assert "--nucleus-model cellpose3:nuclei" in command


def test_bilayers_serializes_native_label_output_option():
    command = generate_cli_command(
        load_config(), {"write_ome_zarr_labels": True}
    )
    assert "--write-ome-zarr-labels True" in command
    args = build_parser().parse_args(["--write-ome-zarr-labels", "true"])
    assert args.write_ome_zarr_labels is True


def test_bilayers_serializes_stardist_smoothing_option():
    command = generate_cli_command(
        load_config(), {"smooth_stardist_labels": False}
    )
    assert "--smooth-stardist-labels False" in command
    args = build_parser().parse_args(["--smooth-stardist-labels", "false"])
    assert args.smooth_stardist_labels is False


def test_legacy_workflow_values_translate_to_selectors():
    values = normalize_legacy_workflow_values(
        {
            "cell_step": True,
            "cell_method": "deep-learning",
            "cell_model": "cellpose3:cyto3",
            "cell_nuclei_channel": 2,
            "cell_nuclei_model": "stardist:SD_Nuclei_Versatile",
            "nucleus_step": False,
            "foci_step_1": False,
            "foci_model_1": "spotiflow:general",
        }
    )
    assert values["cell_model"] == "cellpose3:cyto3"
    assert values["nucleus_model"] == "stardist:SD_Nuclei_Versatile"
    assert values["nucleus_channel"] == 2
    assert values["cell_nuclei_channel"] == 2
    assert values["foci_model_1"] == SKIP


def test_wrapper_accepts_legacy_step_flags_for_normalization():
    args = build_parser().parse_args(
        [
            "--cell-step",
            "false",
            "--nucleus-step",
            "true",
            "--foci-step-2",
            "true",
        ]
    )
    values = normalize_legacy_workflow_values(
        {
            name: getattr(args, name)
            for name in ("cell_step", "nucleus_step", "foci_step_2")
        }
    )
    assert values["cell_model"] == SKIP
    assert values["nucleus_model"] == "cellpose3:nuclei"
    assert values["foci_model_2"] == "spotiflow:general"


def test_environment_bootstrap_installs_launcher_dependencies():
    bootstrap = Path("create_env.cmd").read_text(encoding="utf-8")
    assert "requirements_launcher.txt" in bootstrap
