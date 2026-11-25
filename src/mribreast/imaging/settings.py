"""Typed view of the resolved configuration keys used by the imaging pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from mribreast.imaging.volume import input_channels


def _path(value: Any) -> str:
    return str(value) if value not in (None, "") else ""


@dataclass(frozen=True)
class ImagingSettings:
    image_dim: str = "3d"                 # 3d | 2d
    image_source: str = "nifti"           # nifti | manifest_2d
    image_root: str = ""
    prepared_root: str = ""
    subtraction_root: str = ""
    image_manifest_file: str = ""
    annotation_boxes_file: str = ""
    phase_count: int = 5
    subtraction_mode: str = "none"
    target_size: tuple[int, ...] = (256, 256, 64)
    target_size_2d: tuple[int, int] = (224, 224)
    slice_axis: int = 2
    slice_selection: str = "center"
    slices_per_study: int = 1
    aggregation: str = "mean_prob"
    use_annotation_boxes: bool = False
    annotation_erase_attempts: int = 8
    mask_mode: str = "none"
    mask_kind: str = "manual"
    mask_roi_classes: tuple[int, ...] = (1,)
    roi_margin: float = 0.1
    mask_missing_policy: str = "error"
    mask_empty_policy: str = "full_image"
    mask_invalid_policy: str = "error"   # no config field yet (requested)
    mask_provenance: str = "unknown"
    positive_class: int = 1
    # Duke annotation boxes as a classifier input (3D NIfTI; used at training AND inference)
    box_mode: str = "none"                # none | roi_crop | box_channel | dual_stream
    box_margin: float = 0.1
    box_local_size: tuple[int, ...] = (64, 64, 32)
    dual_stream_embed_dim: int = 128
    box_missing_policy: str = "error"     # error | full_image | drop

    @property
    def spatial_dims(self) -> int:
        return 3 if self.image_dim == "3d" else 2

    @property
    def uses_masks(self) -> bool:
        return self.mask_mode != "none" or self.slice_selection in ("mask_max_area", "mask_all")

    @property
    def uses_boxes(self) -> bool:
        """Annotation boxes are a model input (training, selection and inference)."""
        return self.box_mode != "none"

    def base_channels(self) -> int:
        """Channels from NIfTI phases (manifest_2d channels come from the images)."""
        return input_channels(self.subtraction_mode, self.phase_count)

    def to_json(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


def settings_from_cfg(cfg: Mapping[str, Any]) -> ImagingSettings:
    """Build settings from the dict returned by ``mribreast.config.schema.resolve``."""
    return ImagingSettings(
        image_dim=str(cfg.get("image_dim", "3d")),
        image_source=str(cfg.get("image_source", "nifti")),
        image_root=_path(cfg.get("image_root")),
        prepared_root=_path(cfg.get("prepared_root")),
        subtraction_root=_path(cfg.get("subtraction_root")),
        image_manifest_file=_path(cfg.get("image_manifest_file")),
        annotation_boxes_file=_path(cfg.get("annotation_boxes_file")),
        phase_count=int(cfg.get("phase_count", 5)),
        subtraction_mode=str(cfg.get("subtraction_mode", "none")),
        target_size=tuple(int(v) for v in cfg.get("target_size", (256, 256, 64))),
        target_size_2d=tuple(int(v) for v in cfg.get("target_size_2d", (224, 224))),  # type: ignore[arg-type]
        slice_axis=int(cfg.get("slice_axis", 2)),
        slice_selection=str(cfg.get("slice_selection", "center")),
        slices_per_study=int(cfg.get("slices_per_study", 1)),
        aggregation=str(cfg.get("aggregation", "mean_prob")),
        use_annotation_boxes=bool(cfg.get("use_annotation_boxes", False)),
        annotation_erase_attempts=int(cfg.get("annotation_erase_attempts", 8)),
        mask_mode=str(cfg.get("mask_mode", "none")),
        mask_kind=str(cfg.get("mask_kind", "manual")),
        mask_roi_classes=tuple(int(c) for c in cfg.get("mask_roi_classes", ("1",))),
        roi_margin=float(cfg.get("roi_margin", 0.1)),
        mask_missing_policy=str(cfg.get("mask_missing_policy", "error")),
        mask_empty_policy=str(cfg.get("mask_empty_policy", "full_image")),
        mask_invalid_policy=str(cfg.get("mask_invalid_policy", "error")),
        mask_provenance=str(cfg.get("mask_provenance", "unknown")),
        positive_class=int(cfg.get("positive_class", 1)),
        box_mode=str(cfg.get("box_mode", "none")),
        box_margin=float(cfg.get("box_margin", 0.1)),
        box_local_size=tuple(int(v) for v in cfg.get("box_local_size", (64, 64, 32))),
        dual_stream_embed_dim=int(cfg.get("dual_stream_embed_dim", 128)),
        box_missing_policy=str(cfg.get("box_missing_policy", "error")),
    )
