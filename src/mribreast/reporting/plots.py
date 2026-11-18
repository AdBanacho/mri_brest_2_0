"""Static validation charts (matplotlib, Agg): pooled ROC and confusion matrices per branch."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(alpha=0.2)


def save_roc(path: Path, title: str, fold_curves: Sequence[tuple[str, np.ndarray, np.ndarray]],
             pooled_labels: np.ndarray, pooled_probs: np.ndarray, num_classes: int, positive_class: int = 1) -> bool:
    """Pooled ROC (binary: positive class; multiclass: one-vs-rest per class) + thin per-fold curves (binary)."""
    from sklearn.metrics import auc, roc_curve

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    plotted = False
    labels = np.asarray(pooled_labels, dtype=int)
    if num_classes == 2:
        for name, y, p in fold_curves:
            yb = (np.asarray(y) == positive_class).astype(int)
            if len(np.unique(yb)) < 2:
                continue
            fpr, tpr, _ = roc_curve(yb, np.asarray(p)[:, positive_class])
            ax.plot(fpr, tpr, color=MUTED, alpha=0.35, linewidth=1)
        yb = (labels == positive_class).astype(int)
        if len(np.unique(yb)) == 2:
            fpr, tpr, _ = roc_curve(yb, pooled_probs[:, positive_class])
            ax.plot(fpr, tpr, color=SERIES[0], linewidth=2, label=f"pooled, class {positive_class} (AUC={auc(fpr, tpr):.3f})")
            plotted = True
    else:
        for k in range(num_classes):
            yk = (labels == k).astype(int)
            if len(np.unique(yk)) < 2:
                continue
            fpr, tpr, _ = roc_curve(yk, pooled_probs[:, k])
            ax.plot(fpr, tpr, color=SERIES[k % len(SERIES)], linewidth=2, label=f"class {k} vs rest (AUC={auc(fpr, tpr):.3f})")
            plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1)
    ax.set_xlabel("False positive rate", color=INK)
    ax.set_ylabel("True positive rate", color=INK)
    ax.set_title(title, color=INK)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def save_confusion(path: Path, title: str, matrix: np.ndarray, class_names: Sequence[str]) -> None:
    plt = _plt()
    matrix = np.asarray(matrix)
    n = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(n), labels=list(class_names), rotation=30, ha="right")
    ax.set_yticks(range(n), labels=list(class_names))
    ax.set_xlabel("Predicted", color=INK)
    ax.set_ylabel("True", color=INK)
    ax.set_title(title, color=INK)
    cut = matrix.max() * 0.6 if matrix.size else 0
    for r in range(n):
        for c in range(n):
            ax.text(c, r, str(int(matrix[r, c])), ha="center", va="center",
                    color="white" if matrix[r, c] > cut else INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_bars(path: Path, title: str, labels: Sequence[str], series: Mapping[str, Sequence[float]], xlabel: str) -> None:
    """Grouped horizontal bars (summary charts); ``series`` keeps a fixed colour order."""
    plt = _plt()
    names = list(series)
    if not names or not labels:
        return
    y = np.arange(len(labels))
    height = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(12, max(4.0, 0.5 * len(labels) + 2)))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for index, name in enumerate(names):
        offset = (index - (len(names) - 1) / 2) * height
        ax.barh(y + offset, np.asarray(series[name], dtype=float), height=height * 0.9, label=name,
                color=SERIES[index % len(SERIES)])
    ax.set_yticks(y, labels=list(labels), fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_title(title, color=INK)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
