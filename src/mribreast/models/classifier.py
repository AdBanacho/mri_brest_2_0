"""Lightning image classifier reproducing legacy ``NiftiClassifier`` (+ AdamW / cosine options).

Loss (legacy ``sensitivity_aware_cross_entropy``, logits ``[B, C]``):

* binary: ``CE(weight=class_weights) + lambda * (1 - soft_sensitivity)`` with
  ``soft_sensitivity = sum(y_pos * p_pos) / (n_pos + 1e-6)`` (1.0 if the batch has
  no positives);
* multiclass: CE + ``lambda * (1 - mean over present classes of soft recall)``.
* ``loss=ce`` drops the sensitivity term (it is still logged).

Validation metrics are computed at STUDY level after aggregating item
probabilities (identical to legacy for 3D, one item per study):
``val_auc_roc`` (macro OvR AUROC, torchmetrics), ``val_sensitivity`` (recall of
the positive class), ``val_balanced_accuracy`` (mean per-class recall),
``val_specificity`` (binary).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional.classification import multiclass_auroc, multiclass_recall

PROTECTED_TRAINER_KEYS = frozenset({"callbacks", "logger", "default_root_dir", "enable_checkpointing"})
HISTORY_KEYS = ("train_loss", "train_ce_loss", "train_sensitivity_loss", "train_soft_sensitivity", "train_acc",
                "val_loss", "val_ce_loss", "val_sensitivity_loss", "val_soft_sensitivity", "val_acc",
                "val_auc_roc", "val_balanced_accuracy", "val_sensitivity", "val_specificity")


def compute_balanced_class_weights(labels: Sequence[int], num_classes: int | None = None,
                                   positive_boost: float = 1.0, positive_class: int = 1) -> torch.Tensor:
    """Legacy ``_compute_balanced_class_weights``: ``total / (K * max(count, 1))``; binary positive x boost.

    Difference: ``num_classes`` fixes the vector length (legacy ``bincount`` shrank
    when the highest class was absent from the fit rows, breaking CE).
    """
    label_tensor = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    counts = torch.bincount(label_tensor, minlength=num_classes or 0)
    total = counts.sum().float()
    k = counts.numel()
    weights = total / (k * counts.float().clamp_min(1.0))
    if k == 2:
        weights[positive_class] *= positive_boost
    return weights


def soft_sensitivity_loss(logits: torch.Tensor, y: torch.Tensor, *, num_classes: int,
                          class_weights: torch.Tensor | None, sensitivity_lambda: float, positive_class: int = 1,
                          use_sensitivity: bool = True, eps: float = 1e-6
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(total, ce, sensitivity_loss, soft_sensitivity)``."""
    ce = F.cross_entropy(logits, y.long(), weight=class_weights)
    probs = torch.softmax(logits, dim=1)
    if num_classes == 2:
        positives = (y == positive_class).float()
        support = positives.sum()
        if support > 0:
            soft = (positives * probs[:, positive_class]).sum() / (support + eps)
        else:
            soft = torch.tensor(1.0, device=logits.device, dtype=logits.dtype)
    else:
        one_hot = F.one_hot(y.long(), num_classes=num_classes).float()
        tp = (one_hot * probs).sum(dim=0)
        support = one_hot.sum(dim=0)
        present = support > 0
        soft = (tp / (support + eps))[present].mean()
    sens_loss = 1.0 - soft
    total = ce + sensitivity_lambda * sens_loss if use_sensitivity else ce
    return total, ce, sens_loss, soft


class ImageClassifier(pl.LightningModule):
    def __init__(self, model: nn.Module, num_classes: int, *, lr: float = 1e-4, class_weights: Any = None,
                 sensitivity_lambda: float = 0.05, loss: str = "ce_soft_sensitivity", optimizer: str = "adam",
                 weight_decay: float = 0.0, scheduler: str = "none", positive_class: int = 1,
                 aggregation: str = "mean_prob"):
        super().__init__()
        if loss not in ("ce_soft_sensitivity", "ce"):
            raise ValueError(f"unknown loss {loss!r}")
        if optimizer not in ("adam", "adamw"):
            raise ValueError(f"unknown optimizer {optimizer!r}")
        if scheduler not in ("none", "cosine"):
            raise ValueError(f"unknown scheduler {scheduler!r}")
        self.model = model
        self.num_classes = int(num_classes)
        self.lr = float(lr)
        self.sensitivity_lambda = float(sensitivity_lambda)
        self.loss_name = loss
        self.optimizer_name = optimizer
        self.weight_decay = float(weight_decay)
        self.scheduler_name = scheduler
        self.positive_class = int(positive_class)
        self.aggregation = aggregation
        weights = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", weights)
        self.history: dict[int, dict[str, float]] = {}
        self._val_outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._epoch_sums: dict[str, list[float]] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: Any, stage: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch[0], batch[1]
        logits = self(x)
        total, ce, sens_loss, soft = soft_sensitivity_loss(
            logits, y, num_classes=self.num_classes, class_weights=self.class_weights,
            sensitivity_lambda=self.sensitivity_lambda, positive_class=self.positive_class,
            use_sensitivity=self.loss_name == "ce_soft_sensitivity")
        acc = (logits.argmax(dim=1) == y).float().mean()
        batch_size = int(y.shape[0])
        for name, value in (("loss", total), ("ce_loss", ce), ("sensitivity_loss", sens_loss),
                            ("soft_sensitivity", soft), ("acc", acc)):
            self.log(f"{stage}_{name}", value, prog_bar=name in ("loss", "acc"), on_step=False, on_epoch=True,
                     batch_size=batch_size)
            if not self.trainer.sanity_checking:
                self._epoch_sums.setdefault(f"{stage}_{name}", []).append(float(value.detach().cpu()))
        return total, logits, y

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, _, _ = self._step(batch, "train")
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, logits, y = self._step(batch, "val")
        study = batch[2] if len(batch) > 2 else torch.arange(len(y), device=y.device) + batch_idx * len(y)
        self._val_outputs.append((torch.softmax(logits, dim=1).detach().cpu(), y.detach().cpu(),
                                  study.detach().cpu()))
        return loss

    def _study_metrics(self) -> dict[str, float]:
        from mribreast.imaging.datasets import aggregate_study_probs

        probs = torch.cat([o[0] for o in self._val_outputs]).numpy()
        labels = torch.cat([o[1] for o in self._val_outputs]).numpy()
        studies = torch.cat([o[2] for o in self._val_outputs]).numpy()
        unique, inverse = np.unique(studies, return_inverse=True)
        study_probs = aggregate_study_probs(probs, inverse, len(unique), self.aggregation, self.positive_class)
        study_labels = np.array([labels[inverse == i][0] for i in range(len(unique))])
        p = torch.as_tensor(study_probs, dtype=torch.float32)
        t = torch.as_tensor(study_labels, dtype=torch.long)
        auc = float(multiclass_auroc(p, t, num_classes=self.num_classes, average="macro"))
        recall = multiclass_recall(p.argmax(dim=1), t, num_classes=self.num_classes, average=None)
        metrics = {"val_auc_roc": auc, "val_sensitivity": float(recall[self.positive_class]),
                   "val_balanced_accuracy": float(recall.mean())}
        if self.num_classes == 2:
            metrics["val_specificity"] = float(recall[1 - self.positive_class])
        return metrics

    def on_validation_epoch_end(self) -> None:
        if not self._val_outputs:
            return
        metrics = self._study_metrics()
        self._val_outputs.clear()
        for name, value in metrics.items():
            self.log(name, value, prog_bar=name != "val_specificity", on_step=False, on_epoch=True)
        if not self.trainer.sanity_checking:
            row = self.history.setdefault(int(self.current_epoch), {"epoch": int(self.current_epoch)})
            row.update(metrics)
            for name in [k for k in self._epoch_sums if k.startswith("val_")]:
                row[name] = float(np.mean(self._epoch_sums.pop(name)))

    def on_train_epoch_end(self) -> None:
        row = self.history.setdefault(int(self.current_epoch), {"epoch": int(self.current_epoch)})
        for name in [k for k in self._epoch_sums if k.startswith("train_")]:
            row[name] = float(np.mean(self._epoch_sums.pop(name)))

    def configure_optimizers(self) -> Any:
        optimizer_cls = torch.optim.Adam if self.optimizer_name == "adam" else torch.optim.AdamW
        kwargs: dict[str, Any] = {"lr": self.lr}
        if self.optimizer_name == "adamw" or self.weight_decay:
            kwargs["weight_decay"] = self.weight_decay
        optimizer = optimizer_cls(self.parameters(), **kwargs)
        if self.scheduler_name == "none":
            return optimizer
        t_max = max(1, int(self.trainer.max_epochs or 1)) if self._trainer is not None else 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def history_rows(self) -> list[dict[str, float]]:
        return [self.history[k] for k in sorted(self.history)]


def trainer_options(cfg: Mapping[str, Any], *, logger: Any, callbacks: list[Any], default_root_dir: str,
                    accelerator: str | None = None) -> dict[str, Any]:
    """Legacy trainer options + ``trainer_extra_json``; protected keys cannot be overridden."""
    extra = dict(cfg.get("trainer_extra_json") or {})
    clash = PROTECTED_TRAINER_KEYS.intersection(extra)
    if clash:
        raise ValueError(f"trainer_extra_json cannot override {sorted(PROTECTED_TRAINER_KEYS)} (got {sorted(clash)})")
    options: dict[str, Any] = dict(
        max_epochs=int(cfg.get("epoch", 30)),
        accelerator=accelerator or ("gpu" if torch.cuda.is_available() else "cpu"),
        devices=1, logger=logger, default_root_dir=default_root_dir, callbacks=callbacks,
        enable_checkpointing=True, log_every_n_steps=1, enable_progress_bar=True,
    )
    options.update(extra)
    return options
