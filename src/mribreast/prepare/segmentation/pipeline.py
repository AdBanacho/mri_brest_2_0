"""Box-prompted lesion segmentation → mask manifest for ``mask_provider=local_files``.

docs/segmentation-mri-core.md §4. The segmenter is injected (``SliceSegmenter``), so this module needs no
torch and is tested with a fake. Two inputs:

* NIfTI studies (``cohort.csv`` + Duke annotation boxes): one uint8 NIfTI per study on the grid of its
  first series (``slice_id=""``). It serves both the 3D path and 2D slicing of NIfTI.
* Ready 2D images (image manifest with a ``lesion_box`` column ``[h0, h1, w0, w1]``): one PNG per image key.

Outputs are published atomically per study. A rerun skips studies whose sidecar carries the same
``settings_fingerprint``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from mribreast.artifacts.core import (COMPLETE_FLAG, code_version, environment_versions, mark_complete, read_json,
                                      write_json_atomic, write_text_atomic)
from mribreast.contracts import IMAGE_KEY_COLUMNS, PATIENT_ID, SERIES_UIDS, STUDY_ID, ImageKey
from mribreast.imaging.annotations import Box, annotation_coverage
from mribreast.imaging.geometry import load_series, parse_series_uids, same_grid, series_path
from mribreast.imaging.slices import extract_slice
from mribreast.prepare.segmentation import geometry as g

PIPELINE_VERSION = "segment-box-prompt-v1"
INPUT_KINDS = ("phase", "post_minus_pre")
CLASS_CODES_JSON = json.dumps({"0": "background", "1": "lesion"})
PROMPT_SOURCE = "annotation_box"
DEFAULT_KNOWN_OVERLAP = ("unknown: MRI-CORE self-supervised pretraining data may include Duke breast MRI "
                         "(no labels used)")
QC_COLUMNS = ("patient_id", "study_id", "series_id", "slice_id", "status", "detail", "n_slices_prompted",
              "n_slices_nonempty", "mask_voxels", "box_voxels", "mask_box_fraction", "frac_removed_by_clip",
              "mean_pred_iou", "empty_mask", "reference_voxels", "dice", "iou", "seconds", "reused")
OWNED = ("masks", "previews", "mask_manifest.csv", "segmentation_qc.csv", "annotation_coverage.csv",
         "segmentation_cohort.csv", "segmentation_manifest.json", COMPLETE_FLAG)


@dataclass(frozen=True)
class SegmentSettings:
    input: str = "post_minus_pre:1"          # phase:<k> | post_minus_pre:<k> (NIfTI); ignored for ready 2D images
    input_channel: int = 0                   # ready 2D images: channel to segment
    slice_axis: int = 2
    slice_margin: int = 0
    context: float = 3.0
    min_window: int = 64
    clip_percentiles: tuple[float, float] = (0.5, 99.5)
    threshold: float = 0.0
    clip_to_box: bool = True
    box_margin: float = 0.1
    training_provenance: str = "frozen_external"
    known_overlap: str = DEFAULT_KNOWN_OVERLAP

    def __post_init__(self) -> None:
        parse_input(self.input)
        g.plane_axes(self.slice_axis)
        lo, hi = self.clip_percentiles
        if not 0 <= lo < hi <= 100:
            raise ValueError("clip_percentiles must satisfy 0 <= low < high <= 100")
        if self.slice_margin < 0 or self.min_window < 1 or self.context < 1 or not 0 <= self.box_margin <= 2:
            raise ValueError("slice_margin >= 0, min_window >= 1, context >= 1 and box_margin in [0, 2] required")


def parse_input(spec: str) -> tuple[str, int]:
    kind, _, index = str(spec).partition(":")
    if kind not in INPUT_KINDS or not index.isdigit():
        raise ValueError(f"input must be phase:<k> or post_minus_pre:<k>, got {spec!r}")
    k = int(index)
    if kind == "post_minus_pre" and k < 1:
        raise ValueError("post_minus_pre needs a post-contrast phase index >= 1")
    return kind, k


def settings_fingerprint(settings: SegmentSettings, segmenter_info: Mapping[str, Any]) -> str:
    """Hash of everything that changes mask voxels or manifest rows (not device, batch size or paths)."""
    ignored = {"device", "torch", "repo", "sam_checkpoint", "mri_core_checkpoint", "checkpoint"}
    payload = {"pipeline": PIPELINE_VERSION, "settings": asdict(settings),
               "segmenter": {k: v for k, v in segmenter_info.items() if k not in ignored}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(text))


# ----------------------------------------------------------------------------------------- per-plane core
@dataclass
class PlaneOutcome:
    mask: np.ndarray            # bool [H, W]
    pred_iou: float
    removed: int                # voxels removed by clip_to_box


def window_unit(window: np.ndarray) -> np.ndarray:
    """Min-max of one window to [0, 1] (constant window → zeros)."""
    lo, hi = float(window.min()), float(window.max())
    return np.zeros_like(window, dtype=np.float32) if hi <= lo else ((window - lo) / (hi - lo)).astype(np.float32)


def segment_planes(planes: Sequence[np.ndarray], box2d: g.Box2D, settings: SegmentSettings, segmenter,
                   fill: float) -> list[PlaneOutcome]:
    """Segment the lesion in each plane with the same box prompt and context window."""
    shape = planes[0].shape
    window = g.context_window(box2d, shape, settings.context, settings.min_window)
    prompt = g.box_xyxy(window.box_in_window(box2d))
    windows = [window_unit(g.extract_window(p, window, fill)) for p in planes]
    results = segmenter.segment(windows, [prompt] * len(windows))
    keep_box = g.expand_box2d(box2d, shape, settings.box_margin)
    out = []
    for result in results:
        plane_mask = np.zeros(shape, dtype=bool)
        g.paste_window(plane_mask, result.logits > settings.threshold, window)
        removed = 0
        if settings.clip_to_box:
            plane_mask, removed = g.clip_to_box(plane_mask, keep_box)
        out.append(PlaneOutcome(plane_mask, float(result.pred_iou), removed))
    return out


def _qc(outcomes: Sequence[PlaneOutcome], box_voxels: int) -> dict[str, Any]:
    voxels = int(sum(int(o.mask.sum()) for o in outcomes))
    removed = int(sum(o.removed for o in outcomes))
    return {"n_slices_prompted": len(outcomes), "n_slices_nonempty": int(sum(bool(o.mask.any()) for o in outcomes)),
            "mask_voxels": voxels, "box_voxels": int(box_voxels),
            "mask_box_fraction": voxels / box_voxels if box_voxels else float("nan"),
            "frac_removed_by_clip": removed / (removed + voxels) if removed + voxels else 0.0,
            "mean_pred_iou": float(np.mean([o.pred_iou for o in outcomes])) if outcomes else float("nan"),
            "empty_mask": voxels == 0}


# --------------------------------------------------------------------------------------------- NIfTI input
def load_input_volume(uids: Sequence[str], image_root: str | os.PathLike, spec: str
                      ) -> tuple[np.ndarray, tuple[int, ...], np.ndarray]:
    """Volume to segment on the grid of the first series; returns ``(volume, shape, affine)`` of that grid."""
    kind, k = parse_input(spec)
    if k >= len(uids):
        raise ValueError(f"input {spec} needs phase {k}, the study has {len(uids)} series")
    ref, ref_affine = load_series(series_path(image_root, uids[0]))
    target, target_affine = (ref, ref_affine) if k == 0 else load_series(series_path(image_root, uids[k]))
    if not same_grid(ref.shape, ref_affine, target.shape, target_affine):
        raise ValueError(f"phase {k} ({uids[k]}) is not on the grid of the first series ({uids[0]})")
    volume = target - ref if kind == "post_minus_pre" else target
    return np.ascontiguousarray(volume, dtype=np.float32), tuple(ref.shape), ref_affine


def clip_volume(volume: np.ndarray, percentiles: tuple[float, float]) -> tuple[np.ndarray, float]:
    lo, hi = np.percentile(volume, percentiles)
    return np.clip(volume, lo, hi).astype(np.float32), float(lo)


def segment_volume(volume: np.ndarray, box: Box, settings: SegmentSettings, segmenter
                   ) -> tuple[np.ndarray, dict[str, Any], int]:
    """Return ``(uint8 mask on the volume grid, qc, index of the slice with the largest mask)``."""
    clipped, fill = clip_volume(volume, settings.clip_percentiles)
    axis = settings.slice_axis
    slices = list(g.slice_range(box, axis, volume.shape[axis], settings.slice_margin))
    planes = [extract_slice(clipped, axis, i, channel_first=False) for i in slices]
    box2d = g.box_in_plane(box, axis)
    outcomes = segment_planes(planes, box2d, settings, segmenter, fill)
    mask = np.zeros(volume.shape, dtype=np.uint8)
    for index, outcome in zip(slices, outcomes):
        where = [slice(None)] * 3
        where[axis] = index
        mask[tuple(where)] = outcome.mask
    h0, h1, w0, w1 = box2d
    box_voxels = (h1 - h0) * (w1 - w0) * (box[1][axis] - box[0][axis])
    best = slices[int(np.argmax([o.mask.sum() for o in outcomes]))]
    return mask, _qc(outcomes, box_voxels), best


# ------------------------------------------------------------------------------------------------ writing
def _save_nifti_atomic(mask: np.ndarray, affine: np.ndarray, path: Path) -> None:
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name[:-len('.nii.gz')]}.tmp-{os.getpid()}.nii.gz")
    image = nib.Nifti1Image(mask.astype(np.uint8), affine)
    image.header.set_data_dtype(np.uint8)
    nib.save(image, str(tmp))
    os.replace(tmp, path)


def _save_png_atomic(mask: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp-{os.getpid()}.png")
    Image.fromarray(mask.astype(np.uint8)).save(tmp)
    os.replace(tmp, path)


def _reference_scores(reference, key: ImageKey, mask: np.ndarray) -> dict[str, Any]:
    if reference is None:
        return {}
    record = reference.get(key)
    if record is None:
        return {"reference_voxels": float("nan")}
    ref = np.asarray(record.array) > 0
    if ref.shape != mask.shape:
        return {"reference_voxels": int(ref.sum()), "detail": f"reference grid {ref.shape} != mask {mask.shape}"}
    pred = mask > 0
    inter, total, union = int((pred & ref).sum()), int(pred.sum() + ref.sum()), int((pred | ref).sum())
    return {"reference_voxels": int(ref.sum()), "dice": 2 * inter / total if total else 1.0,
            "iou": inter / union if union else 1.0}


@dataclass
class _Run:
    output_dir: Path
    settings: SegmentSettings
    segmenter: Any
    fingerprint: str
    previews: int
    reference: Any
    log: Callable[[str], None]
    rows: list[dict[str, Any]] = field(default_factory=list)
    qc: list[dict[str, Any]] = field(default_factory=list)
    n_previews: int = 0

    def manifest_row(self, key: ImageKey, rel: str, grid_source: str) -> dict[str, Any]:
        return {PATIENT_ID: key.patient_id, STUDY_ID: key.study_id, "series_id": key.series_id,
                "slice_id": key.slice_id, "path": rel, "class_codes": CLASS_CODES_JSON, "source": "predicted",
                "model_id": self.segmenter.model_id, "model_version": self.segmenter.model_version,
                "grid_source": grid_source, "training_provenance": self.settings.training_provenance,
                "known_overlap": self.settings.known_overlap, "fold": "", "prompt_source": PROMPT_SOURCE}

    def reuse(self, sidecar: Path, mask_path: Path) -> dict[str, Any] | None:
        if not (sidecar.is_file() and mask_path.is_file()):
            return None
        try:
            data = read_json(sidecar)
        except (OSError, ValueError):
            return None
        return data if data.get("settings_fingerprint") == self.fingerprint else None

    def preview(self, name: str, image: np.ndarray, mask: np.ndarray, key: ImageKey, meta: Mapping[str, Any]) -> None:
        if self.n_previews >= self.previews:
            return
        from mribreast.imaging.masks.preview import write_mask_preview

        # write_mask_preview applies with_suffix(): give it a suffix to replace, ids like "2.25" keep their dots
        write_mask_preview(image, mask, self.output_dir / "previews" / f"{name}.preview", key=key.as_tuple(),
                           metadata=dict(meta))
        self.n_previews += 1


def _prepare_output_dir(output_dir: Path, fingerprint: str, overwrite: bool) -> None:
    manifest = output_dir / "segmentation_manifest.json"
    if manifest.is_file():
        previous = read_json(manifest).get("settings_fingerprint")
        if previous != fingerprint:
            if not overwrite:
                raise FileExistsError(f"{output_dir} holds masks made with other settings ({str(previous)[:12]}); "
                                      "use another --output_dir or --overwrite")
            for name in OWNED:                       # only what this tool writes
                target = output_dir / name
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
    (output_dir / COMPLETE_FLAG).unlink(missing_ok=True)       # complete again only after this run succeeds
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)


def _finish(run: _Run, *, inputs: Mapping[str, Any], coverage: pd.DataFrame | None,
            studies: pd.DataFrame | None = None) -> dict[str, Any]:
    from mribreast.imaging.masks.local import mask_manifest_frame

    out = run.output_dir
    manifest_path = out / "mask_manifest.csv"
    frame = mask_manifest_frame(run.rows) if run.rows else pd.DataFrame(
        columns=[*IMAGE_KEY_COLUMNS, "path", "prompt_source"])
    write_text_atomic(manifest_path, frame.to_csv(index=False, lineterminator="\n"))
    qc = pd.DataFrame(run.qc).reindex(columns=list(QC_COLUMNS))
    write_text_atomic(out / "segmentation_qc.csv", qc.to_csv(index=False, lineterminator="\n"))
    if coverage is not None:
        write_text_atomic(out / "annotation_coverage.csv", coverage.to_csv(index=False, lineterminator="\n"))
    if studies is not None:                   # the studies that were considered (traceability of the cohort)
        write_text_atomic(out / "segmentation_cohort.csv",
                          studies.loc[:, [PATIENT_ID, STUDY_ID, SERIES_UIDS]].to_csv(index=False, lineterminator="\n"))
    done = qc[qc["status"] == "segmented"]
    errors = qc[qc["status"] == "error"]

    def q(column: str) -> dict[str, float] | None:
        values = pd.to_numeric(done[column], errors="coerce").dropna()
        return None if values.empty else {p: float(values.quantile(v)) for p, v in
                                          (("p10", .1), ("median", .5), ("p90", .9))}

    summary = {"n_items": int(len(qc)), "status_counts": {k: int(v) for k, v in qc["status"].value_counts().items()},
               "n_segmented": int(len(done)), "n_errors": int(len(errors)),
               "empty_mask_rate": float(done["empty_mask"].astype(bool).mean()) if len(done) else None,
               "mask_box_fraction": q("mask_box_fraction"), "frac_removed_by_clip": q("frac_removed_by_clip"),
               "mean_pred_iou": q("mean_pred_iou"), "dice": q("dice"), "iou": q("iou")}
    limitations = ["masks exist only where a Duke annotation box exists: a box is needed for every study at "
                   "inference (research setting, as box_mode)",
                   "annotation-box axis convention (column, row, slice) is unverified against real overlays "
                   "(docs/imaging.md §4); check previews"]
    if run.settings.known_overlap.startswith("unknown"):
        limitations.append(f"segmenter pretraining overlap with this cohort: {run.settings.known_overlap}")
    if summary["dice"] is None:
        limitations.append("no reference masks: segmentation quality is unmeasured (QC proxies only)")
    payload = {"schema_version": 1, "pipeline": PIPELINE_VERSION, "settings_fingerprint": run.fingerprint,
               "settings": asdict(run.settings), "segmenter": run.segmenter.describe(), "inputs": dict(inputs),
               "outputs": {"mask_manifest": manifest_path.name, "qc": "segmentation_qc.csv"},
               "summary": summary, "limitations": limitations, "code_version": code_version(),
               "environment": environment_versions(), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json_atomic(out / "segmentation_manifest.json", payload)
    if errors.empty:
        mark_complete(out, {"settings_fingerprint": run.fingerprint, "n_masks": int(len(frame))})
    else:
        run.log(f"{len(errors)} item(s) failed; see segmentation_qc.csv. Rerun to retry them (others are reused).")
    return payload


def run_nifti(studies: pd.DataFrame, boxes: Mapping[str, Box], image_root: str | os.PathLike, segmenter,
              output_dir: str | os.PathLike, settings: SegmentSettings = SegmentSettings(), *, previews: int = 0,
              reference=None, overwrite: bool = False, inputs: Mapping[str, Any] | None = None,
              log: Callable[[str], None] = print) -> dict[str, Any]:
    """Segment every eligible study of ``studies`` (columns patient_id, study_id, series_uids)."""
    output_dir = Path(output_dir)
    fingerprint = settings_fingerprint(settings, segmenter.describe())
    _prepare_output_dir(output_dir, fingerprint, overwrite)
    names = studies[STUDY_ID].astype(str).map(safe_name)
    if names.duplicated().any():
        raise ValueError(f"study ids collide after making them file-safe: {list(names[names.duplicated()])[:3]}")
    matched, coverage = annotation_coverage(studies, boxes, image_root)
    status = dict(zip(coverage[STUDY_ID], coverage["status"]))
    run = _Run(output_dir, settings, segmenter, fingerprint, previews, reference, log)
    for row, name in zip(studies.itertuples(index=False), names):
        patient, study = str(getattr(row, PATIENT_ID)), str(getattr(row, STUDY_ID))
        uids = parse_series_uids(getattr(row, SERIES_UIDS))
        key = ImageKey(patient, study, uids[0] if uids else "", "")
        base = {PATIENT_ID: patient, STUDY_ID: study, "series_id": key.series_id, "slice_id": ""}
        if study not in matched:
            run.qc.append({**base, "status": status.get(study, "missing_box"), "detail": "no eligible annotation box"})
            continue
        box, _ = matched[study]
        rel = f"masks/{name}.nii.gz"
        mask_path, sidecar = output_dir / rel, output_dir / "masks" / f"{name}.json"
        started = time.time()
        reused = run.reuse(sidecar, mask_path)
        if reused is not None:
            run.qc.append({**base, **reused["qc"], "reused": True})
            run.rows.append(run.manifest_row(key, rel, key.series_id))
            continue
        try:
            volume, shape, affine = load_input_volume(uids, image_root, settings.input)
            mask, qc, best = segment_volume(volume, box, settings, segmenter)
            qc.update(_reference_scores(reference, key, mask))
            _save_nifti_atomic(mask, affine, mask_path)
        except (OSError, ValueError) as error:
            run.qc.append({**base, "status": "error", "detail": f"{type(error).__name__}: {error}"})
            log(f"[error] {study}: {error}")
            continue
        qc = {"status": "segmented", "detail": qc.pop("detail", ""), **qc, "seconds": round(time.time() - started, 3)}
        write_json_atomic(sidecar, {"settings_fingerprint": fingerprint, "study_id": study, "box": box,
                                    "grid_shape": shape, "qc": qc})
        run.qc.append({**base, **qc, "reused": False})
        run.rows.append(run.manifest_row(key, rel, key.series_id))
        clipped, _ = clip_volume(volume, settings.clip_percentiles)
        run.preview(name, extract_slice(clipped, settings.slice_axis, best, channel_first=False),
                    extract_slice(mask, settings.slice_axis, best, channel_first=False), key,
                    {"slice_index": best, "slice_axis": settings.slice_axis, "box": box, "input": settings.input})
        log(f"[ok] {study}: {qc['mask_voxels']} voxels on {qc['n_slices_nonempty']}/{qc['n_slices_prompted']} slices")
    return _finish(run, inputs={"image_root": str(image_root), "n_studies": int(len(studies)), **(inputs or {})},
                   coverage=coverage, studies=studies)


def parse_lesion_box(text: str, shape: Sequence[int]) -> g.Box2D | None:
    """``lesion_box`` JSON ``[h0, h1, w0, w1]`` (half-open), or ``None`` when empty."""
    if not str(text).strip():
        return None
    values = json.loads(text)
    if not (isinstance(values, list) and len(values) == 4 and all(isinstance(v, int) for v in values)):
        raise ValueError(f"lesion_box must be a JSON list of 4 integers [h0, h1, w0, w1], got {text!r}")
    h0, h1, w0, w1 = values
    if not (0 <= h0 < h1 <= shape[0] and 0 <= w0 < w1 <= shape[1]):
        raise ValueError(f"lesion_box {values} is empty or outside the {shape[0]}x{shape[1]} image")
    return h0, h1, w0, w1


def run_images_2d(image_manifest: str | os.PathLike, segmenter, output_dir: str | os.PathLike,
                  settings: SegmentSettings = SegmentSettings(), *, previews: int = 0, reference=None,
                  overwrite: bool = False, inputs: Mapping[str, Any] | None = None,
                  log: Callable[[str], None] = print) -> dict[str, Any]:
    """Segment ready 2D images whose manifest carries a ``lesion_box`` column."""
    from mribreast.imaging.manifest import CHANNEL_COLUMN, load_manifest_image, read_image_manifest

    output_dir = Path(output_dir)
    frame = read_image_manifest(image_manifest)
    if "lesion_box" not in frame.columns:
        raise ValueError(f"{image_manifest}: ready 2D images need a lesion_box column [h0, h1, w0, w1] for the prompt")
    fingerprint = settings_fingerprint(settings, segmenter.describe())
    _prepare_output_dir(output_dir, fingerprint, overwrite)
    run = _Run(output_dir, settings, segmenter, fingerprint, previews, reference, log)
    for key_values, rows in frame.groupby(list(IMAGE_KEY_COLUMNS), sort=False):
        key = ImageKey(*(str(v) for v in key_values))
        base = dict(zip(IMAGE_KEY_COLUMNS, key.as_tuple()))
        name = safe_name("__".join(key.as_tuple()))
        rel = f"masks/{name}.png"
        mask_path, sidecar = output_dir / rel, output_dir / "masks" / f"{name}.json"
        started = time.time()
        reused = run.reuse(sidecar, mask_path)
        if reused is not None:
            run.qc.append({**base, **reused["qc"], "reused": True})
            run.rows.append(run.manifest_row(key, rel, key.series_id))
            continue
        try:
            image = load_manifest_image(rows.sort_values(CHANNEL_COLUMN))
            if not 0 <= settings.input_channel < image.shape[0]:
                raise ValueError(f"input_channel {settings.input_channel} but the image has {image.shape[0]} channels")
            plane = image[settings.input_channel].astype(np.float32)
            boxes = {b for b in rows["lesion_box"] if str(b).strip()}
            if len(boxes) > 1:
                raise ValueError("channels of one image key carry different lesion_box values")
            box2d = parse_lesion_box(next(iter(boxes), ""), plane.shape)
            if box2d is None:
                run.qc.append({**base, "status": "missing_box", "detail": "empty lesion_box"})
                continue
            clipped, fill = clip_volume(plane, settings.clip_percentiles)
            (outcome,) = segment_planes([clipped], box2d, settings, segmenter, fill)
            mask = outcome.mask.astype(np.uint8)
            h0, h1, w0, w1 = box2d
            qc = _qc([outcome], (h1 - h0) * (w1 - w0))
            qc.update(_reference_scores(reference, key, mask))
            _save_png_atomic(mask, mask_path)
        except (OSError, ValueError) as error:
            run.qc.append({**base, "status": "error", "detail": f"{type(error).__name__}: {error}"})
            log(f"[error] {'/'.join(key.as_tuple())}: {error}")
            continue
        qc = {"status": "segmented", "detail": qc.pop("detail", ""), **qc, "seconds": round(time.time() - started, 3)}
        write_json_atomic(sidecar, {"settings_fingerprint": fingerprint, "key": key.as_tuple(), "box": box2d, "qc": qc})
        run.qc.append({**base, **qc, "reused": False})
        run.rows.append(run.manifest_row(key, rel, key.series_id))
        run.preview(name, clipped, mask, key, {"box": box2d})
    return _finish(run, inputs={"image_manifest": str(image_manifest), "n_items": int(len(run.qc)), **(inputs or {})},
                   coverage=None)


__all__ = ["DEFAULT_KNOWN_OVERLAP", "PIPELINE_VERSION", "PROMPT_SOURCE", "QC_COLUMNS", "SegmentSettings",
           "load_input_volume", "parse_input", "parse_lesion_box", "run_images_2d", "run_nifti", "segment_planes",
           "segment_volume", "settings_fingerprint"]
