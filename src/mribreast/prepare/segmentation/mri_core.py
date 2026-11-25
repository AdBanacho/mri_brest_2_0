"""MRI-CORE / SAM ViT-B box-prompted slice segmenter (docs/segmentation-mri-core.md §2).

The external repository (``mri_foundation``, read-only) supplies only the model classes. Weights are loaded
here, not through its ``sam_model_registry`` path, because that path hard-codes ``map_location='cuda:0'``,
reads SAM weights relative to the working directory, may prompt interactively for a download, and with
``num_cls=1`` leaves the decoder's mask tokens random.

Weights (``--weights``):

* ``mri_core``: SAM ViT-B everywhere, then MRI-CORE's DINOv2 teacher backbone (``patch_embed`` + all 12
  blocks) over the image encoder. ``pos_embed`` stays SAM's, which is what the authors' loader ends up
  doing (their interpolated MRI-CORE ``pos_embed`` is stored under a key that ``strict=False`` drops).
* ``sam``: plain SAM ViT-B (baseline).
* ``state_dict``: a full fine-tuned state dict. The decoder size and adapters are inferred from its keys.

Needs the optional ``.[segment]`` dependencies (``einops``, ``torchvision``).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

WEIGHTS = ("mri_core", "sam", "state_dict")
MASK_OUTPUTS = ("best_iou", "0", "1", "2")
FRAME = 1024                    # SAM ViT-B input size used by MRI-CORE
SAM_NUM_MASK_OUTPUTS = 3        # SAM's multimask decoder; keeps every decoder weight pretrained
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PREPROCESSING_VERSION = "window-minmax-uint8-bilinear1024-imagenet-v1"
VIT_B = dict(encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12, encoder_global_attn_indexes=[2, 5, 8, 11])
_PACKAGE = "_mribreast_mri_foundation_sam"


@dataclass
class SliceResult:
    logits: np.ndarray   # float32 [side, side], window frame; mask = logits > threshold
    pred_iou: float      # decoder's predicted IoU of the chosen mask


class SliceSegmenter(Protocol):
    """Box-prompted 2D segmenter. ``windows`` are float32 ``[side, side]`` in [0, 1]; boxes are window-pixel
    ``(x0, y0, x1, y1)``. Implementations must be deterministic for identical inputs."""

    model_id: str
    model_version: str

    def segment(self, windows: Sequence[np.ndarray], boxes: Sequence[tuple[float, float, float, float]]
                ) -> list[SliceResult]: ...

    def describe(self) -> dict[str, Any]: ...


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_revision(repo: str | Path) -> dict[str, Any]:
    """Git SHA and dirty flag of the external repository ("unknown" without git)."""
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True, timeout=30).stdout.strip()
        return {"sha": sha, "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"sha": "unknown", "dirty": None}


def import_sam_package(repo: str | Path):
    """Import ``<repo>/models/sam`` under a private name (no top-level ``models``, no bytecode in the repo)."""
    if _PACKAGE in sys.modules:
        loaded = Path(sys.modules[_PACKAGE].__file__).resolve().parents[2]
        if loaded != Path(repo).resolve():
            raise RuntimeError(f"MRI-CORE model code already imported from {loaded}, not {repo}")
        return sys.modules[_PACKAGE]
    init = Path(repo) / "models" / "sam" / "__init__.py"
    if not init.is_file():
        raise FileNotFoundError(f"{init} not found: --mri_foundation_repo must point at the mri_foundation checkout")
    spec = importlib.util.spec_from_file_location(_PACKAGE, init, submodule_search_locations=[str(init.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except ImportError as error:
        del sys.modules[_PACKAGE]
        raise ImportError(f"{error}. Install the segmentation extras: pip install -e '.[segment]'") from error
    finally:
        sys.dont_write_bytecode = previous
    return module


def _model_args(encoder_adapter_depths: Sequence[int] = (), decoder_adapter: bool = False,
                decoder_adapt_depth: int = 2) -> argparse.Namespace:
    return argparse.Namespace(arch="vit_b", if_encoder_adapter=bool(encoder_adapter_depths),
                              encoder_adapter_depths=list(encoder_adapter_depths),
                              if_mask_decoder_adapter=decoder_adapter, decoder_adapt_depth=decoder_adapt_depth)


def dino_backbone_to_sam(state: Mapping[str, Any]) -> dict[str, Any]:
    """Map a DINOv2 teacher backbone onto SAM image-encoder keys (blocks + patch_embed only)."""
    teacher = state.get("teacher", state)
    out: dict[str, Any] = {}
    for key, value in teacher.items():
        block = re.fullmatch(r"backbone\.blocks\.(?:\d+\.)?(\d+)\.(.+)", key)
        if block:
            out[f"image_encoder.blocks.{block.group(1)}.{block.group(2).replace('fc', 'lin')}"] = value
        elif key.startswith("backbone.patch_embed."):
            out["image_encoder." + key[len("backbone."):]] = value
    return out


def build_model(repo: str | Path, weights: str, *, sam_checkpoint: str | Path | None = None,
                mri_core_checkpoint: str | Path | None = None, checkpoint: str | Path | None = None):
    """Return ``(model, info)``. ``info`` describes where every weight group came from."""
    import torch

    if weights not in WEIGHTS:
        raise ValueError(f"weights must be one of {WEIGHTS}")
    package = import_sam_package(repo)
    build_sam = sys.modules[f"{_PACKAGE}.build_sam"]
    info: dict[str, Any] = {"weights": weights}

    if weights == "state_dict":
        if not checkpoint:
            raise ValueError("--weights state_dict needs --checkpoint")
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        depths = sorted({int(m.group(1)) for k in state
                         if (m := re.match(r"image_encoder\.blocks\.(\d+)\..*Adapter", k))})
        decoder_adapter = any("mask_decoder" in k and "Adapter" in k for k in state)
        num_outputs = int(state["mask_decoder.mask_tokens.weight"].shape[0]) - 1
        model = build_sam._build_sam(_model_args(depths, decoder_adapter), **VIT_B, num_classes=num_outputs,
                                     checkpoint=None, image_size=FRAME)
        model.load_state_dict(state, strict=True)
        info.update(checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint), num_mask_outputs=num_outputs,
                    encoder_adapter_depths=depths, decoder_adapter=decoder_adapter, sources={"all": "checkpoint"})
        return model.eval(), info

    if not sam_checkpoint:
        raise ValueError(f"--weights {weights} needs --sam_checkpoint (sam_vit_b_01ec64.pth)")
    model = build_sam._build_sam(_model_args(), **VIT_B, num_classes=SAM_NUM_MASK_OUTPUTS, checkpoint=None,
                                 image_size=FRAME)
    sam_state = torch.load(str(sam_checkpoint), map_location="cpu", weights_only=True)
    result = model.load_state_dict(sam_state, strict=False)
    unexpected = [k for k in result.unexpected_keys if "rel_pos" not in k]   # use_rel_pos=False in this repo
    if result.missing_keys or unexpected:
        raise ValueError(f"{sam_checkpoint} is not SAM ViT-B: missing {result.missing_keys[:5]}, "
                         f"unexpected {unexpected[:5]}")
    info.update(sam_checkpoint=str(sam_checkpoint), sam_checkpoint_sha256=sha256_file(sam_checkpoint),
                num_mask_outputs=SAM_NUM_MASK_OUTPUTS,
                sources={"image_encoder": "sam", "prompt_encoder": "sam", "mask_decoder": "sam"})
    if weights == "mri_core":
        if not mri_core_checkpoint:
            raise ValueError("--weights mri_core needs --mri_core_checkpoint (mri_foundation.pth)")
        core = dino_backbone_to_sam(torch.load(str(mri_core_checkpoint), map_location="cpu", weights_only=True))
        target = model.state_dict()
        expected = [k for k in target if k.startswith(("image_encoder.blocks.", "image_encoder.patch_embed."))]
        missing = sorted(set(expected) - set(core))
        bad = [k for k in core if k not in target or tuple(core[k].shape) != tuple(target[k].shape)]
        if missing or bad:
            raise ValueError(f"{mri_core_checkpoint}: MRI-CORE backbone does not cover the SAM ViT-B encoder "
                             f"(missing {missing[:5]}, mismatched {bad[:5]})")
        model.load_state_dict(core, strict=False)
        info.update(mri_core_checkpoint=str(mri_core_checkpoint),
                    mri_core_checkpoint_sha256=sha256_file(mri_core_checkpoint), n_mri_core_tensors=len(core),
                    sources={"image_encoder.blocks+patch_embed": "mri_core", "image_encoder.pos_embed": "sam",
                             "image_encoder.neck": "sam", "prompt_encoder": "sam", "mask_decoder": "sam"},
                    pos_embed_source="sam")
    return model.eval(), info


class MriCoreSegmenter:
    """``SliceSegmenter`` over the MRI-CORE / SAM model (batched over slices, ``torch.no_grad``)."""

    def __init__(self, repo: str | Path, weights: str = "mri_core", *, sam_checkpoint: str | Path | None = None,
                 mri_core_checkpoint: str | Path | None = None, checkpoint: str | Path | None = None,
                 device: str = "cpu", batch_size: int = 4, mask_output: str = "best_iou"):
        import torch

        if mask_output not in MASK_OUTPUTS:
            raise ValueError(f"mask_output must be one of {MASK_OUTPUTS}")
        self.repo = Path(repo)
        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))
        self.mask_output = mask_output
        self.model, self.info = build_model(repo, weights, sam_checkpoint=sam_checkpoint,
                                            mri_core_checkpoint=mri_core_checkpoint, checkpoint=checkpoint)
        self.model.to(self.device)
        n_out = self.info["num_mask_outputs"]
        if mask_output != "best_iou" and int(mask_output) >= n_out:
            raise ValueError(f"mask_output {mask_output} but the decoder has {n_out} outputs")
        self.revision = repo_revision(repo)
        digest = {"mri_core": self.info.get("mri_core_checkpoint_sha256"), "sam": self.info.get("sam_checkpoint_sha256"),
                  "state_dict": self.info.get("checkpoint_sha256")}[weights] or ""
        self.model_id = {"mri_core": "mri_core_sam_vitb", "sam": "sam_vitb",
                         "state_dict": f"state_dict:{digest[:12]}"}[weights]
        self.model_version = f"{digest[:12]}+mf{str(self.revision['sha'])[:7]}"
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def describe(self) -> dict[str, Any]:
        import torch

        return {"model_id": self.model_id, "model_version": self.model_version, "repo": str(self.repo),
                "repo_revision": self.revision, "frame": FRAME, "mask_output": self.mask_output,
                "preprocessing": PREPROCESSING_VERSION, "device": str(self.device), "torch": torch.__version__,
                **self.info}

    def _prepare(self, windows: Sequence[np.ndarray]):
        import torch
        import torch.nn.functional as F

        batch = []
        for window in windows:
            t = torch.as_tensor(np.asarray(window, dtype=np.float32), device=self.device)[None, None]
            t = F.interpolate(t, (FRAME, FRAME), mode="bilinear", align_corners=False).clamp_(0, 1)
            t = torch.round(t * 255.0) / 255.0                  # uint8 quantisation, as the repo's PIL path
            batch.append(t.expand(1, 3, FRAME, FRAME))
        return (torch.cat(batch) - self._mean) / self._std

    def segment(self, windows: Sequence[np.ndarray], boxes: Sequence[tuple[float, float, float, float]]
                ) -> list[SliceResult]:
        import torch
        import torch.nn.functional as F

        if len(windows) != len(boxes):
            raise ValueError("one box per window required")
        results: list[SliceResult] = []
        model = self.model
        with torch.no_grad():
            for start in range(0, len(windows), self.batch_size):
                chunk = list(windows[start:start + self.batch_size])
                sides = [int(w.shape[0]) for w in chunk]
                prompt = torch.tensor([[c * FRAME / side for c in box] for box, side in
                                       zip(boxes[start:start + self.batch_size], sides)],
                                      dtype=torch.float32, device=self.device)
                embeddings = model.image_encoder(self._prepare(chunk))
                sparse, dense = model.prompt_encoder(points=None, boxes=prompt, masks=None)
                logits, iou = model.mask_decoder(image_embeddings=embeddings, image_pe=model.prompt_encoder.get_dense_pe(),
                                                 sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                                 multimask_output=True)
                if self.mask_output == "best_iou":
                    pick = iou.argmax(dim=1)
                else:
                    pick = torch.full((len(chunk),), int(self.mask_output), device=self.device, dtype=torch.long)
                for i, side in enumerate(sides):
                    j = int(pick[i])
                    low = logits[i:i + 1, j:j + 1].float()
                    up = F.interpolate(low, (side, side), mode="bilinear", align_corners=False)[0, 0]
                    results.append(SliceResult(up.cpu().numpy().astype(np.float32), float(iou[i, j])))
        return results


__all__ = ["FRAME", "MASK_OUTPUTS", "MriCoreSegmenter", "PREPROCESSING_VERSION", "SliceResult", "SliceSegmenter",
           "WEIGHTS", "build_model", "dino_backbone_to_sam", "import_sam_package", "repo_revision", "sha256_file"]
