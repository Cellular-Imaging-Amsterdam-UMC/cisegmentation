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


ROOT = Path(__file__).resolve().parents[1]


def test_bilayers_config_is_structurally_valid():
    config = load_config()
    assert validate_config(config) == []
    assert config["docker_image"]["tag"] == (ROOT / "version.txt").read_text(
        encoding="utf-8"
    ).strip()
    parameters = {item["name"]: item for item in config["parameters"]}
    assert "instanseg_pixel_size_um" not in parameters
    assert "input_channels" not in parameters
    assert parameters["diameter"]["mode"] == "advanced"
    assert parameters["diameter"]["minimum"] == -1.0
    assert parameters["spotiflow_min_distance"]["type"] == "float"
    assert parameters["spotiflow_local_refinement"] == {
        "name": "spotiflow_local_refinement",
        "type": "checkbox",
        "label": "Spotiflow Local Mask Refinement",
        "description": "Grow each Spotiflow point into a bounded mask using local signal and background. Supports 2D and forced slice-wise 2D.",
        "default": False,
        "cli_tag": "--spotiflow-local-refinement",
        "cli_order": 28,
        "optional": True,
        "append_value": True,
        "section_id": "advanced",
        "mode": "advanced",
    }
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
    assert parameters["include_original_data"]["default"] is True
    assert parameters["existing_labels"]["default"] == "overwrite"
    assert [
        option["value"] for option in parameters["existing_labels"]["options"]
    ] == ["remove", "overwrite", "append"]
    assert "include_original_channels" not in parameters
    assert "write_ome_zarr_labels" not in parameters
    assert parameters["labels_log_info"]["default"] is False
    assert parameters["labels_log_info"]["mode"] == "advanced"
    assert parameters["labels_log_info"]["section_id"] == "advanced"
    assert parameters["smooth_stardist_labels"]["default"] is True
    assert parameters["smooth_stardist_labels"]["mode"] == "advanced"
    assert parameters["remove_border_cells"]["default"] is True
    assert parameters["remove_border_cells"]["mode"] == "beginner"
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
    assert beginner_names[-4:] == [
        "remove_border_cells",
        "include_original_data",
        "existing_labels",
        "measurements_database",
    ]
    assert parameters["include_original_data"]["mode"] == "beginner"
    assert parameters["include_original_data"]["section_id"] == "essential"
    assert parameters["benchmark"]["mode"] == "advanced"
    assert parameters["measurements_database"]["default"] == "duckdb"
    assert parameters["measurements_database"]["mode"] == "beginner"
    assert parameters["measurements_database"]["section_id"] == "essential"
    assert [
        option["value"] for option in parameters["measurements_database"]["options"]
    ] == ["duckdb", "sqlite", "skip"]
    advanced_names = [
        item["name"]
        for item in config["parameters"]
        if item.get("mode") == "advanced"
    ]
    assert parameters["foci_model_1"]["section_id"] == "essential"
    assert parameters["foci_model_1"]["mode"] == "beginner"
    assert advanced_names[:6] == [
        "foci_model_2",
        "foci_channel_2",
        "foci_model_3",
        "foci_channel_3",
        "foci_model_4",
        "foci_channel_4",
    ]
    assert parameters["max_inference_workers"]["default"] == 0
    assert parameters["max_measurement_workers"]["default"] == 0
    assert all(
        parameters[f"foci_model_{slot}"]["section_id"] == "advanced"
        and parameters[f"foci_model_{slot}"]["mode"] == "advanced"
        and parameters[f"foci_channel_{slot}"]["section_id"] == "advanced"
        and parameters[f"foci_channel_{slot}"]["mode"] == "advanced"
        for slot in range(2, 5)
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
    assert parameters["cell_model"]["options"][2] == {
        "label": "Cellpose-SAM v2",
        "value": "cellpose-sam:cpsam_v2",
    }
    assert parameters["cell_model"]["options"][3] == {
        "label": "Cellpose-SAM original",
        "value": "cellpose-sam:cpsam",
    }


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


def test_wrapper_accepts_spotiflow_local_refinement_boolean_and_legacy_alias():
    args = build_parser().parse_args(["--spotiflow-local-refinement", "true"])
    assert args.spotiflow_local_refinement is True
    legacy = build_parser().parse_args(["--spotiflow-microsam-refinement", "true"])
    assert legacy.spotiflow_microsam_refinement is True
    assert normalize_legacy_workflow_values(vars(legacy))[
        "spotiflow_local_refinement"
    ] is True

    command = generate_cli_command(
        load_config(), {"spotiflow_local_refinement": True}
    )
    assert "--spotiflow-local-refinement True" in command


def test_bilayers_serializes_skip_selectors():
    command = generate_cli_command(
        load_config(), {"cell_model": SKIP, "nucleus_model": "cellpose3:nuclei"}
    )
    assert "--cell-model skip" in command
    assert "--nucleus-model cellpose3:nuclei" in command


def test_bilayers_serializes_cellpose_sam_v2_and_original_selectors():
    v2 = generate_cli_command(
        load_config(),
        {"cell_model": "cellpose-sam:cpsam_v2", "nucleus_model": "skip"},
    )
    original = generate_cli_command(
        load_config(),
        {"cell_model": "cellpose-sam:cpsam", "nucleus_model": "skip"},
    )
    assert "--cell-model cellpose-sam:cpsam_v2" in v2
    assert "--cell-model cellpose-sam:cpsam" in original


def test_bilayers_serializes_output_and_existing_label_options():
    command = generate_cli_command(
        load_config(),
        {"include_original_data": False, "existing_labels": "append"},
    )
    assert "--include-original-data False" in command
    assert "--existing-labels append" in command
    args = build_parser().parse_args(
        ["--include-original-data", "false", "--existing-labels", "remove"]
    )
    assert args.include_original_data is False
    assert args.existing_labels == "remove"


def test_wrapper_accepts_legacy_output_options_for_one_compatibility_period():
    args = build_parser().parse_args(
        ["--include-original-channels", "false", "--write-ome-zarr-labels", "false"]
    )
    assert args.include_original_channels is False
    assert args.write_ome_zarr_labels is False


def test_bilayers_serializes_measurements_database_option():
    command = generate_cli_command(
        load_config(), {"measurements_database": "sqlite"}
    )
    assert "--measurements-database sqlite" in command
    args = build_parser().parse_args(["--measurements-database", "skip"])
    assert args.measurements_database == "skip"


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
