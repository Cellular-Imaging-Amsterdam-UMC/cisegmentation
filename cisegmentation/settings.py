from __future__ import annotations

from dataclasses import asdict, dataclass
import json


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
    benchmark_models: str = "all"
    multi_step: bool = False
    cell_step: bool = True
    cell_model: str = "cellpose3:cyto3"
    cell_channel: int = 3
    cell_nuclei_channel: int = 1
    nucleus_step: bool = True
    nucleus_model: str = "cellpose3:nuclei"
    nucleus_channel: int = 1
    spot_step: bool = True
    spot_model: str = "spotiflow:general"
    spot_channels: str | list[int] = "2"
    derive_cytoplasm: bool = True
    remove_border_cells: bool = False

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

    def selected_spot_channels(self, channel_count: int) -> list[int]:
        """Return the requested one-based spot channels as zero-based indices.

        Duplicates are intentionally retained: ``2,2`` produces two independent
        spot-label output channels.
        """
        if isinstance(self.spot_channels, (list, tuple)):
            values = [str(item).strip() for item in self.spot_channels]
        else:
            values = [
                item.strip()
                for item in str(self.spot_channels or "")
                .replace(";", ",")
                .split(",")
                if item.strip()
            ]
        if not values:
            raise ValueError("At least one spot channel must be selected")
        try:
            channels = [int(value) - 1 for value in values]
        except ValueError as exc:
            raise ValueError("Spot channels must be comma-separated channel numbers") from exc
        if any(index < 0 or index >= channel_count for index in channels):
            raise ValueError(
                f"Selected one-based spot channels are outside input channel count {channel_count}"
            )
        return channels


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
