"""``ImageBranch`` (``contracts.BranchTrainer``) and its ``FittedPredictor`` for 3D / 2D images.

fit(): train on ``data.fit`` (annotation-box erasing only here), select the
checkpoint by study-level ROC-AUC on ``data.selection`` (``ModelCheckpoint`` on
``val_auc_roc`` max, ``save_top_k=1`` + ``last``) and write into
``data.output_dir``::

    best.ckpt  last.ckpt  image_branch.json  history.csv  [annotation_coverage.csv] [mask_audit_fit.csv]

predict_proba(frame) returns STUDY-level probabilities ``[len(frame), C]``
row-aligned with ``frame``; studies whose images were all dropped by a mask
policy get NaN rows (see ``last_audit``).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from mribreast import MANIFEST_SCHEMA_VERSION
from mribreast.artifacts.core import read_json, write_json_atomic
from mribreast.contracts import FoldData, MaskProvider
from mribreast.imaging.annotations import annotation_coverage, load_annotation_boxes
from mribreast.imaging.cache import CACHE_SCHEMA_VERSION
from mribreast.imaging.datasets import (ImageDataset, ItemBuilder, ItemTable, aggregate_study_probs,
                                        expected_channels, load_manifest_for)
from mribreast.imaging.masks.policies import audit_frame, summarize_mask_audit
from mribreast.imaging.settings import ImagingSettings, settings_from_cfg
from mribreast.imaging.volume import NORMALISATION_DIFFERENCE, NORMALISATION_PHASE, RESIZE_BACKEND
from mribreast.models.classifier import ImageClassifier, compute_balanced_class_weights, trainer_options
from mribreast.models.networks import make_network

BRANCH_FILE = "image_branch.json"


def _bind_provider(provider: MaskProvider | None, settings: ImagingSettings) -> MaskProvider | None:
    binder = getattr(provider, "with_image_sources", None)
    if provider is None or binder is None:
        return provider
    return binder(image_root=settings.image_root or None, image_manifest_file=settings.image_manifest_file or None,
                  slice_axis=settings.slice_axis)


def _channel_description(settings: ImagingSettings, channels: int) -> list[str]:
    if settings.image_source == "manifest_2d":
        names = [f"image_channel_{i}" for i in range(channels - (settings.mask_mode == "mask_channel"))]
    elif settings.subtraction_mode == "none":
        names = [f"phase_{i}" for i in range(settings.phase_count)]
    elif settings.subtraction_mode == "post_minus_pre":
        names = [f"phase_{i}-phase_0" for i in range(1, settings.phase_count)]
    else:
        names = [f"phase_{i}-phase_{i - 1}" for i in range(1, settings.phase_count)]
    if settings.mask_mode == "mask_channel":
        names.append("roi_mask")
    return names


class _EpochSetter(pl.Callback):
    """Feeds the epoch number to the training dataset (seeds the annotation erasing RNG)."""

    def __init__(self, dataset: ImageDataset):
        self.dataset = dataset

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.dataset.epoch = int(trainer.current_epoch)


class ImagePredictor:
    """Fitted image branch for one fold (``contracts.FittedPredictor``)."""

    branch = "image"

    def __init__(self, network: torch.nn.Module, settings: ImagingSettings, *, num_classes: int, channels: int,
                 provider: MaskProvider | None, fold: int | None, meta: Mapping[str, Any], directory: Path,
                 batch_size: int = 4, num_workers: int = 0):
        self.network = network.eval()
        self.settings = settings
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.provider = provider
        self.fold = fold
        self.meta = dict(meta)
        self.directory = Path(directory)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.last_audit: pd.DataFrame = pd.DataFrame()

    def item_probabilities(self, frame: pd.DataFrame) -> tuple[ItemTable, np.ndarray]:
        manifest = load_manifest_for(self.settings)
        table = ItemBuilder(self.settings, self.provider, manifest).build(frame.reset_index(drop=True), self.fold)
        self.last_audit = audit_frame(table.decisions)
        if not len(table.items):
            return table, np.zeros((0, self.num_classes))
        dataset = ImageDataset(table.items, self.settings, channels=self.channels, provider=self.provider,
                               fold=self.fold, training=False, manifest=manifest)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        outputs = []
        with torch.no_grad():
            for x, _, _ in loader:
                outputs.append(torch.softmax(self.network(x.float()), dim=1).cpu().numpy())
        return table, np.concatenate(outputs, axis=0)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        table, probs = self.item_probabilities(frame)
        index = table.items["study_index"].to_numpy() if len(table.items) else np.zeros(0, dtype=int)
        out = aggregate_study_probs(probs.reshape(-1, self.num_classes), index, len(frame), self.settings.aggregation,
                                    self.settings.positive_class)
        return out[:, : self.num_classes]

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        if directory.resolve() == self.directory.resolve():
            return
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("best.ckpt", "last.ckpt", BRANCH_FILE, "history.csv"):
            source = self.directory / name
            if source.exists():
                shutil.copyfile(source, directory / name)


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - our own Lightning checkpoint; fall back to full unpickling
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    return {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}


class ImageBranch:
    """``BranchTrainer`` for the image branch (3D NIfTI, 2D slices of NIfTI, or ready 2D images)."""

    branch = "image"

    def __init__(self, cfg: Mapping[str, Any], *, num_classes: int, mask_provider: MaskProvider | None = None,
                 logs_dir: Path | None = None):
        self.cfg = dict(cfg)
        self.settings = settings_from_cfg(cfg)
        self.num_classes = int(num_classes)
        self.mask_provider = _bind_provider(mask_provider, self.settings)
        self.logs_dir = Path(logs_dir) if logs_dir else None
        if self.settings.uses_masks and self.mask_provider is None:
            raise ValueError("mask_mode / mask-based slice selection need a mask provider (also at inference)")
        self.spatial_dims = self.settings.spatial_dims

    # ------------------------------------------------------------------ audits
    def _boxes(self) -> dict[str, Any]:
        if not self.settings.annotation_boxes_file:
            raise ValueError("use_annotation_boxes requires annotation_boxes_file")
        return load_annotation_boxes(self.settings.annotation_boxes_file)

    def annotation_map(self, frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
        return annotation_coverage(frame, self._boxes(), self.settings.image_root)

    def audit(self, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Cohort-level audits for the training loop: ``annotation_coverage`` / ``mask_audit`` (+ summary)."""
        out: dict[str, pd.DataFrame] = {}
        frame = frame.reset_index(drop=True)
        if self.settings.use_annotation_boxes:
            out["annotation_coverage"] = self.annotation_map(frame)[1]
        if self.settings.uses_masks:
            manifest = load_manifest_for(self.settings)
            builder = ItemBuilder(self.settings, self.mask_provider, manifest)
            table = builder.build(frame, None)
            audit = audit_frame(table.decisions)
            out["mask_audit"] = audit
            images_in = len(table.items) + int((audit["action"] == "drop").sum()) if len(audit) else len(table.items)
            out["mask_audit_summary"] = summarize_mask_audit(audit, len(frame), table.studies_with_items,
                                                             images_in, len(table.items))
        return out

    # ------------------------------------------------------------------ fit
    def _network(self, channels: int) -> torch.nn.Module:
        return make_network(str(self.cfg.get("mri_model", "densenet121")), spatial_dims=self.spatial_dims,
                            in_channels=channels, num_classes=self.num_classes,
                            extra=self.cfg.get("mri_extra_json") or {})

    def fit(self, data: FoldData) -> ImagePredictor:
        cfg, s = self.cfg, self.settings
        pl.seed_everything(int(data.seed), workers=True)
        output_dir = Path(data.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = load_manifest_for(s)
        channels = expected_channels(s, manifest)
        builder = ItemBuilder(s, self.mask_provider, manifest)
        fit_frame = data.fit.reset_index(drop=True)
        selection_frame = data.selection.reset_index(drop=True)
        fit_table = builder.build(fit_frame, data.fold)
        selection_table = builder.build(selection_frame, data.fold)
        if not len(fit_table.items) or not len(selection_table.items):
            raise ValueError(f"fold {data.fold}: no image items for fit ({len(fit_table.items)}) or selection "
                             f"({len(selection_table.items)})")
        matched: dict[str, Any] = {}
        coverage_summary: dict[str, int] = {}
        if s.use_annotation_boxes:
            matched, report = self.annotation_map(fit_frame)
            report.to_csv(output_dir / "annotation_coverage.csv", index=False)
            coverage_summary = {str(k): int(v) for k, v in report["status"].value_counts().items()}
            if not matched:
                raise ValueError("no eligible MRI studies for annotation-guided training; disable "
                                 "use_annotation_boxes or supply annotations matching the source grids")
        if fit_table.decisions or selection_table.decisions:
            audit = pd.concat([audit_frame(fit_table.decisions).assign(split="fit"),
                               audit_frame(selection_table.decisions).assign(split="selection")], ignore_index=True)
            audit.to_csv(output_dir / "mask_audit_fit.csv", index=False)

        train_ds = ImageDataset(fit_table.items, s, channels=channels, provider=self.mask_provider, fold=data.fold,
                                training=True, annotation_boxes=matched, seed=int(data.seed), manifest=manifest)
        val_ds = ImageDataset(selection_table.items, s, channels=channels, provider=self.mask_provider,
                              fold=data.fold, training=False, manifest=manifest)
        batch_size = int(cfg.get("batch_size", 4))
        workers = int(cfg.get("num_workers", 0))
        pin = torch.cuda.is_available()
        # Legacy NiftiDataModule does not shuffle the training loader.
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                                  pin_memory=pin)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)

        class_weights = compute_balanced_class_weights(fit_table.items["label"].to_numpy(dtype=int), self.num_classes,
                                                       float(cfg.get("positive_boost", 1.0)), s.positive_class)
        module = ImageClassifier(
            self._network(channels), self.num_classes, lr=float(cfg.get("lr", 1e-4)), class_weights=class_weights,
            sensitivity_lambda=float(cfg.get("sensitivity_lambda", 0.05)),
            loss=str(cfg.get("loss", "ce_soft_sensitivity")), optimizer=str(cfg.get("optimizer", "adam")),
            weight_decay=float(cfg.get("weight_decay", 0.0)), scheduler=str(cfg.get("scheduler", "none")),
            positive_class=s.positive_class, aggregation=s.aggregation)

        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_lightning.loggers import CSVLogger

        checkpoint = ModelCheckpoint(dirpath=str(output_dir), filename="best", monitor="val_auc_roc", mode="max",
                                     save_top_k=1, save_last=True, auto_insert_metric_name=False)
        callbacks: list[Any] = [checkpoint, _EpochSetter(train_ds)]
        if bool(cfg.get("early_stopping", False)):
            callbacks.append(EarlyStopping(monitor="val_auc_roc", mode="max",
                                           patience=int(cfg.get("early_stopping_patience", 8)),
                                           min_delta=float(cfg.get("early_stopping_min_delta", 1e-4))))
        logs_root = self.logs_dir or (output_dir / "logs")
        run_name = output_dir.parent.parent.name if self.logs_dir else "image"
        logger = CSVLogger(save_dir=str(logs_root), name=run_name, version=f"fold_{data.fold}")
        trainer = pl.Trainer(**trainer_options(cfg, logger=logger, callbacks=callbacks,
                                               default_root_dir=str(logs_root)))
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best = output_dir / "best.ckpt"
        best_path = Path(checkpoint.best_model_path) if checkpoint.best_model_path else None
        if best_path is not None and best_path.exists() and best_path.resolve() != best.resolve():
            shutil.copyfile(best_path, best)
        if not best.exists():
            last = output_dir / "last.ckpt"
            if not last.exists():
                raise RuntimeError(f"fold {data.fold} did not produce a checkpoint")
            shutil.copyfile(last, best)
        history = pd.DataFrame(module.history_rows())
        history.to_csv(output_dir / "history.csv", index=False)
        score = checkpoint.best_model_score
        best_epoch = None
        if len(history) and "val_auc_roc" in history:
            finite = history.dropna(subset=["val_auc_roc"])
            if len(finite):
                best_epoch = int(finite.loc[finite["val_auc_roc"].idxmax(), "epoch"])
        meta = {
            "schema_version": MANIFEST_SCHEMA_VERSION, "branch": "image", "fold": int(data.fold),
            "num_classes": self.num_classes,
            "network": {"name": str(cfg.get("mri_model", "densenet121")), "spatial_dims": self.spatial_dims,
                        "in_channels": channels, "out_channels": self.num_classes,
                        "extra": dict(cfg.get("mri_extra_json") or {})},
            "imaging": s.to_json(),
            "channels": _channel_description(s, channels),
            "geometry": {
                "axis_order": "nibabel voxel axes (i, j, k); no reorientation" if s.image_source == "nifti"
                else "image rows, columns",
                "input_shape": [channels] + (list(s.target_size) if s.image_dim == "3d" else list(s.target_size_2d)),
                "slice_axis": s.slice_axis if s.image_dim == "2d" and s.image_source == "nifti" else None,
                "spacing_units": "mm (from NIfTI affine)" if s.image_source == "nifti" else "pixels",
            },
            "cache": {"cache_schema_version": CACHE_SCHEMA_VERSION, "resize_backend": RESIZE_BACKEND,
                      "align_corners": False,
                      "normalisation": NORMALISATION_PHASE if s.subtraction_mode == "none"
                      else NORMALISATION_DIFFERENCE},
            "mask": {"mode": s.mask_mode, "kind": s.mask_kind, "roi_classes": list(s.mask_roi_classes),
                     "roi_margin": s.roi_margin, "missing_policy": s.mask_missing_policy,
                     "empty_policy": s.mask_empty_policy, "invalid_policy": s.mask_invalid_policy,
                     "provenance": s.mask_provenance,
                     "provider": self.mask_provider.describe() if self.mask_provider is not None else None},
            "annotation_boxes": {"enabled": s.use_annotation_boxes, "eligible_fit_studies": len(matched),
                                 "coverage": coverage_summary, "applies_to": "fit rows only"},
            "training": {"epochs": int(cfg.get("epoch", 30)), "batch_size": batch_size,
                         "lr": float(cfg.get("lr", 1e-4)), "optimizer": module.optimizer_name,
                         "weight_decay": module.weight_decay, "scheduler": module.scheduler_name,
                         "loss": module.loss_name, "sensitivity_lambda": module.sensitivity_lambda,
                         "positive_boost": float(cfg.get("positive_boost", 1.0)),
                         "class_weights": [float(w) for w in class_weights],
                         "class_weight_unit": "image items" if s.image_dim == "2d" else "studies",
                         "early_stopping": bool(cfg.get("early_stopping", False)),
                         "trainer_extra": dict(cfg.get("trainer_extra_json") or {}), "shuffle": False},
            "selection": {"monitor": "val_auc_roc", "mode": "max", "unit": "study",
                          "aggregation": s.aggregation,
                          "best_score": float(score) if score is not None else None, "best_epoch": best_epoch,
                          "epochs_run": int(trainer.current_epoch)},
            "counts": {"fit_studies": len(fit_frame), "fit_items": len(fit_table.items),
                       "fit_studies_with_items": fit_table.studies_with_items,
                       "selection_studies": len(selection_frame), "selection_items": len(selection_table.items),
                       "selection_studies_with_items": selection_table.studies_with_items},
            "files": {"best": "best.ckpt", "last": "last.ckpt", "history": "history.csv"},
        }
        write_json_atomic(output_dir / BRANCH_FILE, meta)
        return self.load(output_dir, data.fold)

    # ------------------------------------------------------------------ load
    def load(self, directory: Path, fold: int) -> ImagePredictor:
        directory = Path(directory)
        meta = read_json(directory / BRANCH_FILE)
        if int(meta.get("schema_version", 0)) != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"{directory / BRANCH_FILE}: unsupported schema_version {meta.get('schema_version')}")
        network_meta = meta["network"]
        if int(network_meta["out_channels"]) != self.num_classes:
            raise ValueError(f"saved image branch has {network_meta['out_channels']} classes, expected "
                             f"{self.num_classes}")
        saved = meta["imaging"]
        current = self.settings.to_json()
        for key in ("image_dim", "image_source", "subtraction_mode", "phase_count", "target_size", "target_size_2d",
                    "slice_axis", "slice_selection", "slices_per_study", "mask_mode", "mask_roi_classes"):
            if saved.get(key) != current.get(key):
                raise ValueError(f"saved image branch was trained with {key}={saved.get(key)!r}, current "
                                 f"configuration has {current.get(key)!r}")
        network = make_network(network_meta["name"], spatial_dims=int(network_meta["spatial_dims"]),
                               in_channels=int(network_meta["in_channels"]), num_classes=self.num_classes,
                               extra=network_meta.get("extra") or {})
        network.load_state_dict(_load_state_dict(directory / "best.ckpt"))
        workers = int(self.cfg.get("validate_num_workers", self.cfg.get("num_workers", 0)))
        return ImagePredictor(network, self.settings, num_classes=self.num_classes,
                              channels=int(network_meta["in_channels"]), provider=self.mask_provider, fold=fold,
                              meta=meta, directory=directory, batch_size=int(self.cfg.get("batch_size", 4)),
                              num_workers=workers)


def build_image_branch(cfg: Mapping[str, Any], *, num_classes: int, mask_provider: MaskProvider | None = None,
                       logs_dir: Path | None = None) -> ImageBranch:
    """Composition helper used by the CLI (``cfg`` = ``mribreast.config.schema.resolve`` output)."""
    return ImageBranch(cfg, num_classes=num_classes, mask_provider=mask_provider, logs_dir=logs_dir)


__all__ = ["ImageBranch", "ImagePredictor", "build_image_branch", "BRANCH_FILE"]
