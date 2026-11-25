"""MRI-CORE (``~/mri_foundation``) frozen-encoder embeddings for the Duke cohort.

Job ``python -m mribreast.prepare mricore-embed`` (GPU recommended): for every cohort
study it

1. takes one DCE phase of the study (``--phase``, default 1 = first post-contrast;
   optionally minus ``--subtract_phase``, e.g. 0 = pre-contrast);
2. selects the axial slices inside the patient's Duke annotation box (evenly
   thinned to ``--max_slices``); studies without a box are skipped for
   ``--pool roi`` and use the volume's central slices for ``--pool global``;
3. per volume clips intensities to the [0.5, 99.5] percentiles and scales to [0, 1],
   resizes each slice to ``--image_size``, replicates it to 3 channels and applies
   ImageNet normalisation (as MRI-CORE's ``normalize_type=ours`` dataset does);
4. runs the frozen ViT-B image encoder (no adapters; weights loaded strictly, see
   :func:`map_backbone_state`), takes the 768-d patch tokens after the backbone's final
   LayerNorm (``--layer blocks``; the released checkpoint has no trained SAM neck) and
   average-pools the feature map over the box (``roi``, box widened by ``--roi_margin``) and/or the whole slice
   (``global``), then averages over slices.

Output (written atomically, never next to or inside ``image_root``):

* ``<output>`` CSV - ``patient_id, study_id, series_uid, phase, n_slices`` +
  ``mricore_<pool>_<k>`` columns (768 per pool); read by feature group ``mricore``
  (``mricore_features_file``);
* ``<output stem>_audit.csv`` - one row per cohort study with status/detail;
* ``<output stem>_provenance.json`` - checkpoint sha256, MRI-CORE git commit, arguments,
  weight-loading check, counts.

Box convention: :mod:`mribreast.imaging.annotations` (array axes ``(column, row, slice)``).
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from mribreast.contracts import PATIENT_ID, SERIES_UIDS, STUDY_ID

POOLS = ("roi", "global", "both")
LAYERS = ("neck", "blocks")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 16


# ----------------------------------------------------------------------------- pure helpers
def normalise_volume(volume: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    """Clip to volume percentiles and scale to [0, 1] (constant volumes -> zeros)."""
    volume = np.asarray(volume, dtype=np.float32)
    lo, hi = np.percentile(volume, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(volume)
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def select_slices(start: int, end: int, max_slices: int) -> list[int]:
    """Evenly spaced slice indices in the half-open range ``[start, end)``."""
    count = end - start
    if count <= 0:
        return []
    if max_slices <= 0 or count <= max_slices:
        return list(range(start, end))
    return sorted({int(round(v)) for v in np.linspace(start, end - 1, max_slices)})


def widen_box(box, shape: Sequence[int], margin: float):
    """Widen the in-plane extent of ``((c0, r0, s0), (c1, r1, s1))`` by ``margin`` per side, clipped."""
    (c0, r0, s0), (c1, r1, s1) = box
    dc = int(round((c1 - c0) * margin))
    dr = int(round((r1 - r0) * margin))
    return ((max(0, c0 - dc), max(0, r0 - dr), max(0, s0)),
            (min(shape[0], c1 + dc), min(shape[1], r1 + dr), min(shape[2], s1)))


def feature_window(start: int, end: int, size: int, grid: int) -> tuple[int, int]:
    """Map a half-open pixel range on an axis of ``size`` to the encoder grid (>= 1 cell)."""
    lo = int(math.floor(start * grid / size))
    hi = int(math.ceil(end * grid / size))
    lo = min(max(lo, 0), grid - 1)
    return lo, max(hi, lo + 1)


def feature_columns(pool: str, dim: int) -> list[str]:
    pools = ("roi", "global") if pool == "both" else (pool,)
    return [f"mricore_{p}_{k:03d}" for p in pools for k in range(dim)]


# ----------------------------------------------------------------------------- MRI-CORE model
def _git_commit(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _model_args(image_size: int) -> types.SimpleNamespace:
    # Feature extraction without fine-tuning: adapters/LoRA off (README "Step 2").
    return types.SimpleNamespace(
        arch="vit_b", image_size=image_size, num_cls=1, if_encoder_adapter=False, encoder_adapter_depths=[],
        if_mask_decoder_adapter=False, decoder_adapt_depth=2, if_encoder_lora_layer=False,
        if_decoder_lora_layer=False, encoder_lora_layer=[], thd=False, depth=1, encoder_depth_layer=[])


def map_backbone_state(state: Mapping[str, Any], grid: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """MRI-CORE (DINOv2-style ``backbone.*``) checkpoint -> SAM ``ImageEncoderViT`` keys.

    Same mapping as MRI-CORE's ``build_sam._build_sam`` (strip ``backbone.``, drop the
    FSDP chunk index ``blocks.<chunk>.<i>`` -> ``blocks.<i>``, ``fc`` -> ``lin``, drop the
    cls token from ``pos_embed`` and resize it to the token grid) with one fix:
    upstream stores the resized ``pos_embed`` under its *unmapped* key, so with
    ``strict=False`` it is silently dropped and the encoder runs with zero
    positional embeddings.  Returns ``(encoder_state, final_norm)``; the DINO final
    ``norm`` has no counterpart in the SAM encoder and is returned separately.
    """
    import torch
    import torch.nn.functional as F

    for wrapper in ("model", "teacher", "student"):
        if wrapper in state and isinstance(state[wrapper], Mapping):
            state = state[wrapper]
    encoder: dict[str, Any] = {}
    norm: dict[str, Any] = {}
    for key, value in state.items():
        if not key.startswith("backbone.") or not torch.is_tensor(value):
            continue
        key = key[len("backbone."):]
        if key in ("cls_token", "mask_token") or key.startswith(("_fsdp", "head")):
            continue
        if key.startswith("norm."):
            norm[key[len("norm."):]] = value
            continue
        parts = key.split(".")
        if parts[0] == "blocks" and len(parts) > 3 and parts[1].isdigit() and parts[2].isdigit():
            parts = [parts[0], *parts[2:]]
        key = ".".join(parts).replace("fc", "lin")
        if key == "pos_embed":
            tokens = value[:, 1:, :]
            side = int(round(math.sqrt(tokens.shape[1])))
            if side * side != tokens.shape[1]:
                raise ValueError(f"pos_embed with {tokens.shape[1]} patch tokens is not square")
            grid_embed = tokens.reshape(1, side, side, -1).permute(0, 3, 1, 2).float()
            if side != grid:
                grid_embed = F.interpolate(grid_embed, size=(grid, grid), mode="bilinear", align_corners=False)
            value = grid_embed.permute(0, 2, 3, 1).contiguous()
        encoder[key] = value
    return encoder, norm


def load_encoder(repo: str | Path, checkpoint: str | Path, image_size: int, device: str, layer: str):
    """Build MRI-CORE ViT-B, load the checkpoint strictly and return ``(encoder, final_norm, weight_check)``."""
    import torch

    repo = Path(repo).resolve()
    if not (repo / "models" / "sam" / "build_sam.py").is_file():
        raise SystemExit(f"--mricore_repo {repo} has no models/sam/build_sam.py")
    # Register MRI-CORE's ``models`` / ``models.sam`` as bare packages: their __init__ pulls in the
    # predictor/mask generator (torchvision), which the encoder does not need and the Helios
    # ML-bundle does not ship.  build_sam -> modeling needs only torch + einops.
    for name, directory in (("models", repo / "models"), ("models.sam", repo / "models" / "sam")):
        existing = sys.modules.get(name)
        if existing is not None and list(getattr(existing, "__path__", [])) != [str(directory)]:
            raise SystemExit(f"a different '{name}' package is already imported: {existing}")
        if existing is None:
            package = types.ModuleType(name)
            package.__path__ = [str(directory)]
            sys.modules[name] = package
    from models.sam.build_sam import sam_model_registry  # noqa: PLC0415 - MRI-CORE repository

    encoder = sam_model_registry["vit_b"](_model_args(image_size), checkpoint=None, num_classes=1,
                                          image_size=image_size, pretrained_sam=False).image_encoder
    try:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - older pickles with non-tensor entries
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    mapped, final_norm = map_backbone_state(state, image_size // PATCH_SIZE)
    result = encoder.load_state_dict(mapped, strict=False)
    missing = sorted(result.missing_keys)
    check = {"loaded_tensors": len(mapped), "missing": missing, "unexpected": sorted(result.unexpected_keys),
             "final_norm": sorted(final_norm), "neck_loaded": not any(k.startswith("neck.") for k in missing)}
    if result.unexpected_keys or any(not k.startswith("neck.") for k in missing):
        raise SystemExit(f"MRI-CORE checkpoint does not match the ViT-B encoder: missing={missing[:8]} "
                         f"unexpected={sorted(result.unexpected_keys)[:8]}")
    if layer == "neck" and not check["neck_loaded"]:
        raise SystemExit("checkpoint has no trained 'neck' (DINOv2-style backbone); use --layer blocks")
    if layer == "blocks" and set(final_norm) != {"weight", "bias"}:
        raise SystemExit(f"checkpoint lacks the backbone final norm (found {sorted(final_norm)})")
    encoder = encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    final_norm = {k: v.float().to(device) for k, v in final_norm.items()}
    return encoder, final_norm, check


def encode(encoder, batch, layer: str, final_norm: Mapping[str, Any] | None = None):
    """``[B, 3, S, S]`` -> feature map ``[B, C, S/16, S/16]``.

    ``neck``: SAM neck output, C=256. ``blocks``: ViT patch tokens after the backbone's
    final LayerNorm (DINOv2 feature convention), C=768.
    """
    import torch.nn.functional as F

    if layer == "neck":
        return encoder(batch)
    x = encoder.patch_embed(batch)
    if encoder.pos_embed is not None:
        x = x + encoder.pos_embed
    for block in encoder.blocks:
        x = block(x)
    if final_norm:
        x = F.layer_norm(x, (x.shape[-1],), final_norm["weight"], final_norm["bias"], eps=1e-6)
    return x.permute(0, 3, 1, 2)


# ----------------------------------------------------------------------------- per study
def study_embedding(volume: np.ndarray, box, *, encoder, layer: str, pool: str, image_size: int,
                    max_slices: int, roi_margin: float, batch_size: int, device: str,
                    final_norm: Mapping[str, Any] | None = None) -> tuple[np.ndarray, int]:
    """Embedding vector + number of slices used, for one normalised ``(column, row, slice)`` volume."""
    import torch
    import torch.nn.functional as F

    cols, rows, depth = volume.shape
    if box is not None:
        (c0, r0, s0), (c1, r1, s1) = widen_box(box, volume.shape, roi_margin)
        slices = select_slices(s0, s1, max_slices)
    else:
        c0, r0, c1, r1 = 0, 0, cols, rows
        centre = depth // 2
        half = max(1, max_slices) // 2
        slices = select_slices(max(0, centre - half), min(depth, centre - half + max(1, max_slices)), max_slices)
    if not slices:
        raise ValueError(f"annotation box selects no slices (volume depth {depth})")
    grid = image_size // PATCH_SIZE
    y0, y1 = feature_window(r0, r1, rows, grid)
    x0, x1 = feature_window(c0, c1, cols, grid)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    sums: dict[str, Any] = {}
    for i in range(0, len(slices), batch_size):
        chunk = slices[i:i + batch_size]
        # (column, row) -> (row, column) image; rows map to feature y, columns to x.
        images = torch.from_numpy(np.stack([volume[:, :, s].T for s in chunk])).float().unsqueeze(1).to(device)
        images = F.interpolate(images, size=(image_size, image_size), mode="bilinear", align_corners=False)
        images = (images.repeat(1, 3, 1, 1) - mean) / std
        with torch.no_grad():
            fmap = encode(encoder, images, layer, final_norm).float()
        parts = {}
        if pool in ("roi", "both"):
            parts["roi"] = fmap[:, :, y0:y1, x0:x1].mean(dim=(2, 3))
        if pool in ("global", "both"):
            parts["global"] = fmap.mean(dim=(2, 3))
        for name, value in parts.items():
            sums[name] = sums.get(name, 0) + value.sum(dim=0)
    order = ("roi", "global") if pool == "both" else (pool,)
    vector = torch.cat([sums[name] / len(slices) for name in order]).cpu().numpy().astype(np.float64)
    return vector, len(slices)


# ----------------------------------------------------------------------------- job
def cohort_studies(clinical_features_file: str, metadata_file: str, profile: str = "general",
                   phase_count: int = 4) -> pd.DataFrame:
    """Cohort studies (``patient_id, study_id, series_uids``) exactly as training builds them.

    ``general`` orders the series by parsed phase (0 = pre, 1 = first post-contrast, ...), so ``--phase``
    means the same acquisition in every study. ``duke_legacy_3d`` keeps metadata-file row order, which on the
    real Duke digest mixes in non-DCE series (``ax t1``) for most studies; use it only to reproduce legacy runs.
    """
    from mribreast.data.duke import DukeCohortSource

    cfg = {"clinical_features_file": clinical_features_file, "metadata_file": metadata_file,
           "profile": profile, "phase_count": phase_count,
           "phase_order": "explicit" if profile == "general" else "metadata", "feature_groups": ("clinical",)}
    frame = DukeCohortSource(cfg).load(cfg).frame
    return frame[[PATIENT_ID, STUDY_ID, SERIES_UIDS]].reset_index(drop=True)


def run(args) -> int:
    import torch

    from mribreast.artifacts import write_json_atomic, write_text_atomic
    from mribreast.imaging.annotations import annotation_coverage, load_annotation_boxes
    from mribreast.imaging.geometry import SeriesNotFoundError, load_series, parse_series_uids, series_path

    output = Path(args.output)
    image_root = Path(args.image_root).resolve()
    if output.resolve().parent == image_root or image_root in output.resolve().parents:
        raise SystemExit(f"--output must not be inside image_root ({image_root})")
    if args.pool == "roi" and not args.annotation_boxes_file:
        raise SystemExit("--pool roi needs --annotation_boxes_file")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    studies = cohort_studies(args.clinical_features_file, args.metadata_file, args.profile, args.phase_count)
    if args.limit:
        studies = studies.head(args.limit)
    # Same box eligibility as training (out_of_volume_box, ambiguous studies, geometry mismatch ...).
    matched: dict[str, Any] = {}
    box_status: dict[str, tuple[str, str]] = {}
    if args.annotation_boxes_file:
        matched, coverage = annotation_coverage(studies, load_annotation_boxes(args.annotation_boxes_file), image_root)
        box_status = {r[STUDY_ID]: (r["status"], r["detail"]) for r in coverage.to_dict("records") if r[STUDY_ID]}
    encoder, final_norm, weight_check = load_encoder(args.mricore_repo, args.checkpoint, args.image_size,
                                                     device, args.layer)
    dim = 256 if args.layer == "neck" else 768
    columns = feature_columns(args.pool, dim)

    rows, audit = [], []
    for n, study in enumerate(studies.itertuples(index=False), start=1):
        patient, sid = getattr(study, PATIENT_ID), getattr(study, STUDY_ID)
        uids = parse_series_uids(getattr(study, SERIES_UIDS))
        record = {PATIENT_ID: patient, STUDY_ID: sid, "status": "ok", "detail": ""}
        try:
            if args.phase >= len(uids):
                raise ValueError(f"study has {len(uids)} phases, --phase {args.phase} requested")
            box = matched[sid][0] if sid in matched else None
            if box is None and args.pool != "global":
                status, detail = box_status.get(sid, ("missing_box", ""))
                raise LookupError(f"box {status}: {detail}".rstrip(": "))
            volume, _ = load_series(series_path(image_root, uids[args.phase]))
            volume = np.asarray(volume, dtype=np.float32)
            if args.subtract_phase is not None:
                base, _ = load_series(series_path(image_root, uids[args.subtract_phase]))
                if base.shape != volume.shape:
                    raise ValueError(f"phase shapes differ: {volume.shape} vs {base.shape}")
                volume = volume - np.asarray(base, dtype=np.float32)
            vector, used = study_embedding(
                normalise_volume(volume), box, encoder=encoder, layer=args.layer, pool=args.pool,
                image_size=args.image_size, max_slices=args.max_slices, roi_margin=args.roi_margin,
                batch_size=args.batch_size, device=device, final_norm=final_norm)
            rows.append({PATIENT_ID: patient, STUDY_ID: sid, "series_uid": uids[args.phase], "phase": args.phase,
                         "n_slices": used, **dict(zip(columns, vector))})
            record["detail"] = f"slices={used}"
        except (SeriesNotFoundError, LookupError, ValueError) as exc:
            record.update(status="skipped", detail=str(exc))
        audit.append(record)
        print(f"[{n}/{len(studies)}] {patient} {record['status']} {record['detail']}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.with_suffix("")
    table = pd.DataFrame(rows, columns=[PATIENT_ID, STUDY_ID, "series_uid", "phase", "n_slices", *columns])
    write_text_atomic(output, table.to_csv(index=False, float_format="%.7g"))
    write_text_atomic(Path(f"{stem}_audit.csv"), pd.DataFrame(audit).to_csv(index=False))
    ckpt = Path(args.checkpoint)
    write_json_atomic(Path(f"{stem}_provenance.json"), {
        "mricore_repo": str(Path(args.mricore_repo).resolve()), "mricore_commit": _git_commit(Path(args.mricore_repo)),
        "checkpoint": str(ckpt.resolve()), "checkpoint_sha256": _sha256(ckpt.resolve()),
        "weight_check": weight_check, "device": device, "torch": torch.__version__,
        "arguments": {k: v for k, v in vars(args).items() if k != "func"},
        "studies": len(studies), "embedded": len(rows), "skipped": len(audit) - len(rows), "features": len(columns)})
    print(f"wrote {output} ({len(rows)}/{len(studies)} studies, {len(columns)} features)")
    return 0 if rows else 1


def add_parser(sub) -> None:
    p = sub.add_parser("mricore-embed", help="frozen MRI-CORE encoder embeddings per study (feature group mricore)")
    p.add_argument("--clinical_features_file", required=True)
    p.add_argument("--metadata_file", required=True)
    p.add_argument("--image_root", required=True, help="directory of <SeriesInstanceUID>.nii.gz (read only)")
    p.add_argument("--annotation_boxes_file", default=None, help="Duke Annotation_Boxes.xlsx (needed for --pool roi/both)")
    p.add_argument("--output", required=True, help="embedding CSV (pipeline mricore_features_file)")
    p.add_argument("--mricore_repo", required=True, help="checkout of mazurowski-lab/mri_foundation")
    p.add_argument("--checkpoint", required=True, help="MRI-CORE ViT-B checkpoint (pretrained_weights/MRI_CORE_vitb.pth)")
    p.add_argument("--profile", default="general", choices=("general", "duke_legacy_3d"),
                   help="cohort profile; must match the training config (general = series ordered by parsed phase)")
    p.add_argument("--phase_count", type=int, default=4, help="phases per study, as TRAIN[phase_count] (pre + 3 posts)")
    p.add_argument("--phase", type=int, default=1, help="index into the study's ordered series (0 = pre, 1 = post1)")
    p.add_argument("--subtract_phase", type=int, default=None, help="optional phase subtracted from --phase (e.g. 0)")
    p.add_argument("--pool", choices=POOLS, default="roi")
    p.add_argument("--layer", choices=LAYERS, default="blocks",
                   help="blocks = 768-d ViT tokens after the final norm; neck = 256-d SAM neck (needs a checkpoint with a trained neck)")
    p.add_argument("--image_size", type=int, default=1024)
    p.add_argument("--max_slices", type=int, default=16)
    p.add_argument("--roi_margin", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    p.add_argument("--limit", type=int, default=0, help="first N studies only (smoke runs)")
    p.set_defaults(func=run)
