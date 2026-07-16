from __future__ import annotations

from dataclasses import asdict, dataclass
import json


CELL_MODELS = (
    "cellpose3:cyto3",
    "cellpose-sam:cpsam",
    "instanseg:fluorescence_nuclei_and_cells",
)
STEP1_NUCLEUS_MODELS = (
    "cellpose3:nuclei",
    "cellpose-sam:cpsam",
    "stardist:SD_Nuclei_Versatile",
    "instanseg:single_channel_nuclei",
)
STEP2_NUCLEUS_MODELS = (
    *STEP1_NUCLEUS_MODELS,
    "instanseg:fluorescence_nuclei_and_cells",
)
FOCI_MODELS = (
    "spotiflow:general",
    "spotiflow:hybiss",
    "spotiflow:synth_complex",
    "spotiflow:synth_3d",
    "spotiflow:smfish_3d",
    "spotiflow:fluo_live",
    "stardist:SD_Foci_Aggregates",
    "stardist:SD_Foci_Finn",
    "cellpose3:bact_phase_cp3",
    "cellpose3:bact_fluor_cp3",
)


@dataclass
class SegmentationSettings:
    model: str = "cellpose3:nuclei"
    target: str = "nuclei"
    primary_channel: int = 1
    nuclei_channel: int = 0
    device: str = "auto"
    dimension_mode: str = "auto"
    diameter: float = 0.0
    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.4
    stardist_prob_threshold: float = -1.0
    stardist_nms_threshold: float = -1.0
    spotiflow_prob_threshold: float = -1.0
    spotiflow_min_distance: float = 1.0
    benchmark: bool = False
    cell_step: bool = True
    cell_method: str = "deep-learning"
    cell_model: str = "cellpose3:cyto3"
    cell_channel: int = 3
    cell_nuclei_channel: int = 1
    cell_nuclei_model: str = "cellpose3:nuclei"
    cell_expansion_nucleus_model: str = "cellpose3:nuclei"
    cell_expansion_distance: float = 10.0
    nucleus_step: bool = False
    nucleus_model: str = "cellpose3:nuclei"
    nucleus_channel: int = 1
    foci_step_1: bool = False
    foci_model_1: str = "spotiflow:general"
    foci_channel_1: int = 2
    foci_step_2: bool = False
    foci_model_2: str = "spotiflow:general"
    foci_channel_2: int = 2
    foci_step_3: bool = False
    foci_model_3: str = "spotiflow:general"
    foci_channel_3: int = 2
    foci_step_4: bool = False
    foci_model_4: str = "spotiflow:general"
    foci_channel_4: int = 2
    include_original_channels: bool = False
    remove_border_cells: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def selected_channels(self, channel_count: int) -> list[int]:
        channels = [self.primary_channel - 1]
        if self.nuclei_channel > 0 and self.nuclei_channel != self.primary_channel:
            channels.append(self.nuclei_channel - 1)
        if not channels or any(
            index < 0 or index >= channel_count for index in channels
        ):
            raise ValueError(
                f"Selected one-based channels are outside input channel count {channel_count}"
            )
        return channels

    def enabled_foci_steps(self) -> list[tuple[int, str, int]]:
        """Return enabled Step 3 slots as ``(slot, model, one-based channel)``."""
        return [
            (slot, getattr(self, f"foci_model_{slot}"), getattr(self, f"foci_channel_{slot}"))
            for slot in range(1, 5)
            if getattr(self, f"foci_step_{slot}")
        ]

    def validate_steps(self) -> None:
        if not (self.cell_step or self.nucleus_step or self.enabled_foci_steps()):
            raise ValueError(
                "Select at least one segmentation step: Cell Detection, "
                "Nuclei Detection, or a Foci Detection slot"
            )
        if self.cell_method not in {"deep-learning", "cell-expansion"}:
            raise ValueError(f"Unknown cell detection method: {self.cell_method}")
        if self.cell_expansion_distance < 0:
            raise ValueError("Cell expansion distance must be zero or greater")


def parse_model_selection(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            loaded = json.loads(text)
            if isinstance(loaded, list):
                return [str(item).strip() for item in loaded if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]
