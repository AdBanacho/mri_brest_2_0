"""Item tables and torch datasets for the three image paths (one measurement path for 1..n images/study).

* 3D NIfTI: one item per study -> ``[C, *target_size]``.
* 2D from NIfTI: ``slices_per_study`` (or mask-selected) slices per study -> ``[C, H, W]``.
* 2D manifest: every manifest image of the study -> ``[C, H, W]``.

Items carry ``study_index`` (row position in the study frame) so item
predictions are aggregated back to study level (``aggregate_study_probs``).
Masks are resolved once while building items (policies + audit) and fetched
again lazily by the dataset (providers are cheap and picklable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mribreast.contracts import (LABEL, PATIENT_ID, SERIES_ID, SERIES_UIDS, SLICE_ID, STUDY_ID, Geometry, ImageKey,
                                 MaskProvider)
from mribreast.imaging.annotations import Box, erase_background_box, resize_box
from mribreast.imaging.geometry import (geometry_from_affine, parse_series_uids, read_header_geometry, series_path,
                                        slice_affine)
from mribreast.imaging.loader import StudyVolumeLoader
from mribreast.imaging.manifest import (CHANNEL_COLUMN, load_manifest_image, read_image_file, read_image_manifest,
                                        zscore_channels)
from mribreast.imaging.masks.policies import (MaskDecision, MaskPolicyError, apply_mask_mode, check_mask,
                                              resample_roi, resolve_mask, roi_array)
from mribreast.imaging.settings import ImagingSettings
from mribreast.imaging.slices import extract_slice, roi_area_per_slice, select_slices

ITEM_COLUMNS = ("study_index", PATIENT_ID, STUDY_ID, LABEL, SERIES_ID, SLICE_ID, "mask_action")


@dataclass
class ItemTable:
    items: pd.DataFrame
    decisions: list[MaskDecision] = field(default_factory=list)
    notes: list[dict[str, Any]] = field(default_factory=list)   # study-level notes (no images, dropped, ...)
    n_studies: int = 0

    @property
    def studies_with_items(self) -> int:
        return int(self.items["study_index"].nunique()) if len(self.items) else 0


def image_shape_2d(path: str) -> tuple[int, int]:
    suffix = Path(path).suffix.lower()
    if suffix == ".png":
        from PIL import Image

        with Image.open(path) as image:
            return int(image.size[1]), int(image.size[0])
    return tuple(read_image_file(path).shape[1:])  # type: ignore[return-value]


def _label(row: Any) -> int:
    value = getattr(row, LABEL, -1)
    return int(value) if value is not None and not pd.isna(value) else -1


class ItemBuilder:
    """Builds ``ItemTable`` objects for a study frame (resolving mask policies)."""

    def __init__(self, settings: ImagingSettings, provider: MaskProvider | None, manifest: pd.DataFrame | None,
                 *, raise_errors: bool = True):
        self.s = settings
        self.provider = provider
        self.manifest = manifest
        # raise_errors=False (cohort audits): policy "error" is reported as action "error" and the item excluded
        self.raise_errors = bool(raise_errors)
        if settings.uses_masks and provider is None:
            raise ValueError("mask_mode / mask-based slice selection require a mask provider")

    def _resolve(self, key: ImageKey, fold: int | None, expected: Geometry | None, purpose: str) -> MaskDecision:
        assert self.provider is not None
        return resolve_mask(self.provider, key, fold=fold, expected=expected, roi_classes=self.s.mask_roi_classes,
                            missing_policy=self.s.mask_missing_policy, empty_policy=self.s.mask_empty_policy,
                            invalid_policy=self.s.mask_invalid_policy, mask_kind=self.s.mask_kind, purpose=purpose,
                            raise_errors=self.raise_errors)

    def build(self, frame: pd.DataFrame, fold: int | None) -> ItemTable:
        table = ItemTable(pd.DataFrame(columns=list(ITEM_COLUMNS)), n_studies=len(frame))
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(frame.itertuples(index=False)):
            if self.s.image_source == "manifest_2d":
                rows.extend(self._manifest_items(index, row, fold, table))
            elif self.s.image_dim == "3d":
                rows.extend(self._volume_items(index, row, fold, table))
            else:
                rows.extend(self._slice_items(index, row, fold, table))
        if rows:
            table.items = pd.DataFrame(rows)
        return table

    # --------------------------------------------------------------- 3D
    def _volume_items(self, index: int, row: Any, fold: int | None, table: ItemTable) -> list[dict[str, Any]]:
        uids = parse_series_uids(getattr(row, SERIES_UIDS))
        if not uids:
            raise ValueError(f"study {getattr(row, STUDY_ID)} has no series UIDs")
        key = ImageKey(str(getattr(row, PATIENT_ID)), str(getattr(row, STUDY_ID)), uids[0], "")
        action = "none"
        if self.s.mask_mode != "none":
            expected = read_header_geometry(series_path(self.s.image_root, uids[0]))
            decision = self._resolve(key, fold, expected, "mask_mode")
            table.decisions.append(decision)
            if decision.action in ("drop", "error"):
                table.notes.append({"study_index": index, STUDY_ID: key.study_id, "note": "dropped_by_mask_policy"})
                return []
            action = decision.action
        return [{"study_index": index, PATIENT_ID: key.patient_id, STUDY_ID: key.study_id, LABEL: _label(row),
                 SERIES_ID: key.series_id, SLICE_ID: "", "mask_action": action, SERIES_UIDS: "|".join(uids)}]

    # --------------------------------------------------------------- 2D from NIfTI
    def _slice_items(self, index: int, row: Any, fold: int | None, table: ItemTable) -> list[dict[str, Any]]:
        s = self.s
        uids = parse_series_uids(getattr(row, SERIES_UIDS))
        if not uids:
            raise ValueError(f"study {getattr(row, STUDY_ID)} has no series UIDs")
        patient, study = str(getattr(row, PATIENT_ID)), str(getattr(row, STUDY_ID))
        geometry = read_header_geometry(series_path(s.image_root, uids[0]))
        n_slices = int(geometry.shape[s.slice_axis])
        volume_key = ImageKey(patient, study, uids[0], "")
        volume_decision: MaskDecision | None = None
        if s.slice_selection in ("mask_max_area", "mask_all"):
            volume_decision = self._resolve(volume_key, fold, geometry, "slice_selection")
            table.decisions.append(volume_decision)
            if volume_decision.action in ("drop", "error"):
                table.notes.append({"study_index": index, STUDY_ID: study, "note": "dropped_by_mask_policy"})
                return []
            if volume_decision.action == "use":
                area = roi_area_per_slice(volume_decision.record.array, s.slice_axis, s.mask_roi_classes)
                indices = select_slices(n_slices, s.slice_selection, s.slices_per_study, area)
            else:  # full_image fallback -> deterministic centre selection
                indices = select_slices(n_slices, "center", s.slices_per_study)
                table.notes.append({"study_index": index, STUDY_ID: study, "note": "slice_selection_fallback_center"})
        else:
            indices = select_slices(n_slices, s.slice_selection, s.slices_per_study)
        items = []
        for slice_index in indices:
            action = "none"
            if s.mask_mode != "none":
                key = ImageKey(patient, study, uids[0], str(slice_index))
                if volume_decision is None:
                    volume_decision = self._resolve(volume_key, fold, geometry, "mask_mode")
                if volume_decision.action == "use":
                    mask_slice = extract_slice(volume_decision.record.array, s.slice_axis, slice_index,
                                               channel_first=False)
                    if np.isin(mask_slice, np.asarray(s.mask_roi_classes)).any():
                        decision = MaskDecision(key, "ok", "use", volume_decision.record, "", "mask_mode")
                    else:
                        decision = MaskDecision(key, "empty", "", volume_decision.record, "empty ROI on slice",
                                                "mask_mode")
                        if s.mask_empty_policy == "error" and self.raise_errors:
                            raise MaskPolicyError(f"mask empty for {key.as_tuple()}")
                        decision.action = s.mask_empty_policy
                else:
                    decision = MaskDecision(key, volume_decision.status, volume_decision.action,
                                            volume_decision.record, volume_decision.detail, "mask_mode")
                table.decisions.append(decision)
                if decision.action in ("drop", "error"):
                    continue
                action = decision.action
            items.append({"study_index": index, PATIENT_ID: patient, STUDY_ID: study, LABEL: _label(row),
                          SERIES_ID: uids[0], SLICE_ID: str(slice_index), "mask_action": action,
                          SERIES_UIDS: "|".join(uids)})
        if not items:
            table.notes.append({"study_index": index, STUDY_ID: study, "note": "no_items_after_mask_policy"})
        return items

    # --------------------------------------------------------------- 2D manifest
    def _manifest_items(self, index: int, row: Any, fold: int | None, table: ItemTable) -> list[dict[str, Any]]:
        assert self.manifest is not None
        patient, study = str(getattr(row, PATIENT_ID)), str(getattr(row, STUDY_ID))
        images = self.manifest[self.manifest[STUDY_ID] == study]
        if len(images) and (images[PATIENT_ID] != patient).any():
            raise ValueError(f"manifest maps study {study} to another patient than the cohort ({patient})")
        keys = images.drop_duplicates([SERIES_ID, SLICE_ID])[[SERIES_ID, SLICE_ID, "path"]]
        if not len(keys):
            table.notes.append({"study_index": index, STUDY_ID: study, "note": "no_images_in_manifest"})
            return []
        items = []
        for series_id, slice_id, path in keys.itertuples(index=False, name=None):
            key = ImageKey(patient, study, series_id, slice_id)
            action = "none"
            if self.s.mask_mode != "none":
                decision = self._resolve(key, fold, Geometry(shape=image_shape_2d(path)), "mask_mode")
                table.decisions.append(decision)
                if decision.action in ("drop", "error"):
                    continue
                action = decision.action
            items.append({"study_index": index, PATIENT_ID: patient, STUDY_ID: study, LABEL: _label(row),
                          SERIES_ID: series_id, SLICE_ID: slice_id, "mask_action": action, SERIES_UIDS: ""})
        if not items:
            table.notes.append({"study_index": index, STUDY_ID: study, "note": "no_items_after_mask_policy"})
        return items


def expected_channels(settings: ImagingSettings, manifest: pd.DataFrame | None = None) -> int:
    """Input channels of the network: phases/differences (or image channels) + 1 for ``mask_channel``."""
    if settings.image_source == "manifest_2d":
        if manifest is None or not len(manifest):
            raise ValueError("manifest_2d needs a non-empty image manifest to derive channels")
        first = manifest.iloc[0]
        rows = manifest[(manifest[STUDY_ID] == first[STUDY_ID]) & (manifest[SERIES_ID] == first[SERIES_ID])
                        & (manifest[SLICE_ID] == first[SLICE_ID])]
        base = int(sum(read_image_file(p).shape[0] for p in rows["path"]))
    else:
        base = settings.base_channels()
    return base + (1 if settings.mask_mode == "mask_channel" else 0)


class ImageDataset(Dataset):
    """``(x [C, *S] float32, y long, study_index long)``; annotation erasing only when ``training``."""

    def __init__(self, items: pd.DataFrame, settings: ImagingSettings, *, channels: int,
                 provider: MaskProvider | None = None, fold: int | None = None, training: bool = False,
                 annotation_boxes: Mapping[str, tuple[Box, tuple[int, ...]]] | None = None, seed: int = 0,
                 manifest: pd.DataFrame | None = None):
        self.items = items.reset_index(drop=True)
        self.s = settings
        self.channels = int(channels)
        self.provider = provider
        self.fold = fold
        self.training = bool(training)
        self.annotation_boxes = dict(annotation_boxes or {}) if training else {}
        self.seed = int(seed)
        self.epoch = 0
        self.manifest = manifest
        self.erase_log: list[tuple[str, tuple[int, ...] | None]] = []
        base = settings.base_channels() if settings.image_source == "nifti" else None
        if settings.image_source == "nifti":
            target = settings.target_size if settings.image_dim == "3d" else None
            self.loader: StudyVolumeLoader | None = StudyVolumeLoader(
                settings.image_root, subtraction_mode=settings.subtraction_mode, channels=int(base),
                target_size=target, prepared_root=settings.prepared_root or None,
                subtraction_root=settings.subtraction_root or None)
        else:
            self.loader = None
        self._last: tuple[str, Any] | None = None   # one-study cache (items of a study are consecutive)
        self._native_loader: StudyVolumeLoader | None = None
        if settings.image_source == "nifti" and settings.image_dim == "3d" and settings.mask_mode == "roi_crop":
            if settings.use_annotation_boxes:
                raise ValueError("use_annotation_boxes cannot be combined with 3D mask_mode=roi_crop "
                                 "(the erased cuboid is defined on the full target grid)")
            self._native_loader = StudyVolumeLoader(
                settings.image_root, subtraction_mode=settings.subtraction_mode, channels=int(base),
                target_size=None, prepared_root=settings.prepared_root or None,
                subtraction_root=settings.subtraction_root or None)

    def __len__(self) -> int:
        return len(self.items)

    def _study_volume(self, uids: Sequence[str]) -> Any:
        token = "|".join(uids)
        if self._last is None or self._last[0] != token:
            assert self.loader is not None
            self._last = (token, self.loader.load(uids))
        return self._last[1]

    def _native_volume(self, uids: Sequence[str]) -> Any:
        assert self._native_loader is not None
        return self._native_loader.load(uids)

    def _mask(self, key: ImageKey, action: str) -> Any:
        if action != "use" or self.provider is None:
            return None
        record = self.provider.get(key, self.fold)
        if record is None:
            raise MaskPolicyError(f"mask disappeared for {key.as_tuple()}")
        return record

    def load_item(self, index: int, log: list[dict[str, Any]] | None = None) -> np.ndarray:
        item = self.items.iloc[index]
        s = self.s
        action = str(item["mask_action"])
        if s.image_source == "manifest_2d":
            assert self.manifest is not None
            rows = self.manifest[(self.manifest[STUDY_ID] == item[STUDY_ID])
                                 & (self.manifest[SERIES_ID] == item[SERIES_ID])
                                 & (self.manifest[SLICE_ID] == item[SLICE_ID])]
            image = zscore_channels(load_manifest_image(rows))
            key = ImageKey(item[PATIENT_ID], item[STUDY_ID], item[SERIES_ID], item[SLICE_ID])
            record = self._mask(key, action)
            if record is not None:
                status, detail = check_mask(record, key, Geometry(shape=tuple(image.shape[1:])), None)
                if status != "ok":
                    raise MaskPolicyError(f"{status}: {detail}")
            out = apply_mask_mode(image, None if record is None else record.array, s.mask_mode,
                                  roi_classes=s.mask_roi_classes, margin=s.roi_margin, out_size=s.target_size_2d,
                                  log=log)
        elif s.image_dim == "3d":
            uids = parse_series_uids(item[SERIES_UIDS])
            volume = self._study_volume(uids)
            data = volume.data
            study = str(item[STUDY_ID])
            if self.training and study in self.annotation_boxes:
                box, source_shape = self.annotation_boxes[study]
                lesion = resize_box(box, source_shape, data.shape[-3:])
                rng = np.random.default_rng([self.seed, self.epoch, int(index)])
                data, origin = erase_background_box(data, lesion, rng, s.annotation_erase_attempts)
                self.erase_log.append((study, origin))
                if log is not None:
                    log.append({"op": "annotation_erase", "lesion": [list(lesion[0]), list(lesion[1])],
                                "origin": list(origin) if origin else None})
            key = ImageKey(item[PATIENT_ID], study, item[SERIES_ID], "")
            record = self._mask(key, action)
            if s.mask_mode == "none":
                out = data
            elif record is None:      # full_image fallback
                out = apply_mask_mode(data, None, s.mask_mode, roi_classes=s.mask_roi_classes, margin=s.roi_margin,
                                      out_size=s.target_size, log=log)
            elif s.mask_mode == "roi_crop":
                # ROI bbox on the native mask grid; crop image + mask at native resolution, then resize
                native = self._native_volume(uids)
                out = apply_mask_mode(native.data, record.array, "roi_crop", roi_classes=s.mask_roi_classes,
                                      margin=s.roi_margin, out_size=s.target_size, log=log)
            else:
                # ROI on the native grid -> target grid by any-overlap resampling (never loses a lesion voxel)
                roi, rule = resample_roi(roi_array(record.array, s.mask_roi_classes), s.target_size)
                if log is not None:
                    log.append({"op": "roi_resample", "rule": rule, "from": list(record.array.shape),
                                "to": list(s.target_size)})
                out = apply_mask_mode(data, roi.astype(np.int64), s.mask_mode, roi_classes=(1,),
                                      margin=s.roi_margin, out_size=s.target_size, log=log)
        else:
            uids = parse_series_uids(item[SERIES_UIDS])
            volume = self._study_volume(uids)
            slice_index = int(item[SLICE_ID])
            image = extract_slice(volume.data, s.slice_axis, slice_index)
            mask = None
            if action == "use":
                record = self._mask(ImageKey(item[PATIENT_ID], item[STUDY_ID], item[SERIES_ID], ""), action)
                # same reference grid + tolerance as ItemBuilder: header geometry of the first series
                expected = read_header_geometry(series_path(s.image_root, uids[0]))
                status, detail = check_mask(record, record.key, expected, None)
                if status != "ok":
                    raise MaskPolicyError(f"{status}: {detail}")
                mask = extract_slice(record.array, s.slice_axis, slice_index, channel_first=False)
            if log is not None:
                log.append({"op": "slice", "axis": s.slice_axis, "index": slice_index,
                            "slice_affine": slice_affine(volume.affine, s.slice_axis, slice_index).tolist()})
            out = apply_mask_mode(image, mask, s.mask_mode, roi_classes=s.mask_roi_classes, margin=s.roi_margin,
                                  out_size=s.target_size_2d, log=log)
        if out.shape[0] != self.channels:
            raise ValueError(f"item {index} has {out.shape[0]} channels, expected {self.channels}")
        return np.ascontiguousarray(out, dtype=np.float32)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.load_item(index))
        y = torch.tensor(int(self.items.iloc[index][LABEL]), dtype=torch.long)
        return x, y, torch.tensor(int(self.items.iloc[index]["study_index"]), dtype=torch.long)


def aggregate_study_probs(item_probs: np.ndarray, study_index: np.ndarray, n_studies: int, strategy: str,
                          positive_class: int = 1) -> np.ndarray:
    """Study-level probabilities ``[n_studies, C]``; studies without items get NaN rows.

    * ``mean_prob``: mean of item probability vectors (default).
    * ``max_prob``: binary -> ``p_pos = max_i p_i[pos]``, other class ``1 - p_pos``;
      multiclass -> per-class max over items, renormalised to sum 1.
    """
    item_probs = np.asarray(item_probs, dtype=np.float64)
    study_index = np.asarray(study_index, dtype=np.int64)
    n_classes = item_probs.shape[1] if item_probs.ndim == 2 else 0
    out = np.full((n_studies, max(n_classes, 1)), np.nan)
    for study in range(n_studies):
        selected = item_probs[study_index == study]
        if not len(selected):
            continue
        if strategy == "mean_prob":
            out[study] = selected.mean(axis=0)
        elif strategy == "max_prob":
            if n_classes == 2:
                p = selected[:, positive_class].max()
                out[study, positive_class] = p
                out[study, 1 - positive_class] = 1.0 - p
            else:
                peak = selected.max(axis=0)
                out[study] = peak / peak.sum()
        else:
            raise ValueError(f"unknown aggregation {strategy!r}")
    return out


def load_manifest_for(settings: ImagingSettings) -> pd.DataFrame | None:
    if settings.image_source != "manifest_2d":
        return None
    if not settings.image_manifest_file:
        raise ValueError("image_source=manifest_2d requires image_manifest_file")
    return read_image_manifest(settings.image_manifest_file)


__all__ = ["ItemBuilder", "ItemTable", "ImageDataset", "aggregate_study_probs", "expected_channels",
           "load_manifest_for", "image_shape_2d", "CHANNEL_COLUMN"]
