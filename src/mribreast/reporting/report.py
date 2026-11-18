"""Markdown reports (per evaluation and for the cross-configuration summary)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from mribreast.reporting.tables import fmt

GENERALISATION_NOTE = (
    "Generalisation claims for a configuration that was *selected* by comparing outer-fold results "
    "(e.g. the best of a grid, or the best `fusion_alpha`) require an independent test set or an extra "
    "level of nesting; the outer-fold metrics of the selected configuration are optimistically biased."
)

_SELECTION_TEXT = {
    "shared": ("Each outer-training fold was split once more by patient (StratifiedGroupKFold, seed+fold). "
               "The inner **calibration** patients selected the image checkpoint (ROC-AUC) and tuned the decision "
               "thresholds; models and preprocessing were fitted on the inner **fit** patients only."),
    "separate": ("Each outer-training fold was split by patient into **fit** and inner held-out patients, which were "
                 "split again (seed+fold+1000) into **selection** (checkpoint / early stopping) and **threshold** "
                 "(decision threshold) patients. Selection and threshold tuning therefore use disjoint data."),
    "outer": ("**Historical image-only protocol**: the model was fitted on the whole outer-training fold and the "
              "**outer evaluation fold itself selected the checkpoint and the threshold**. These metrics are "
              "optimistic and are not an unbiased estimate of performance."),
}


def _cell(value: Any, integer: bool) -> str:
    if isinstance(value, (bool, np.bool_, str)):
        return str(value)
    if integer and value is not None and not pd.isna(value):
        return str(int(value))
    return fmt(value)


def _md_table(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    columns = [c for c in columns if c in frame.columns]
    integer = {c: frame[c].dtype.kind in "iu" for c in columns}
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for index in range(len(frame)):
        lines.append("| " + " | ".join(_cell(frame[c].iloc[index], integer[c]) for c in columns) + " |")
    return lines


def validation_report(*, manifest: Mapping[str, Any], cfg: Mapping[str, Any], metrics: pd.DataFrame,
                      summary: pd.DataFrame, comparison: pd.DataFrame, thresholds: Mapping[str, Any],
                      class_names: Sequence[str]) -> str:
    mode = manifest["selection_split"]
    lines = [
        f"# Validation report `{manifest['run_id']}`",
        "",
        f"- train_id: `{manifest['train_id']}`; eval_id: `{manifest['eval_id']}`; split_hash: `{manifest['split_hash']}`",
        f"- label scheme: `{manifest['label_definition']['scheme']}` classes {list(class_names)}; "
        f"positive class: {manifest['label_definition']['positive_class']}",
        f"- branches evaluated: {', '.join(manifest['branches_evaluated'])}; primary branch: **{manifest['primary_branch']}**",
        f"- fusion: `p_fusion = {manifest['fusion_alpha']:g} * p_image + {1 - manifest['fusion_alpha']:g} * p_tabular`"
        if "fusion" in manifest["branches_evaluated"] else "- fusion: not evaluated (single branch)",
        f"- decision rule: {manifest['decision_rule']} (threshold strategy `{manifest['threshold_strategy']}`), "
        f"report unit: `{manifest['report_unit']}`",
        f"- profile: `{cfg.get('profile')}`, folds: {manifest['num_folds']}",
        "",
        "## How the configuration was selected and which data were used",
        "",
        _SELECTION_TEXT[mode],
        "",
        f"- checkpoint selection used roles: {', '.join(manifest['roles']['selection'])}",
        f"- threshold tuning used roles: {', '.join(manifest['roles']['threshold'])} "
        "(one independent threshold per branch; the fusion threshold is tuned on fused calibration probabilities)",
        f"- final evaluation used roles: {', '.join(manifest['roles']['evaluation'])} (every study evaluated exactly once)",
        f"- selection_uses_outer_eval: **{str(manifest['selection_uses_outer_eval']).lower()}**; "
        f"threshold_uses_outer_eval: **{str(manifest['threshold_uses_outer_eval']).lower()}**",
        "- pooled metrics concatenate per-fold decisions, each made with the threshold of its own fold.",
        "",
        GENERALISATION_NOTE,
        "",
        "## Branch comparison (mean over folds, population std, pooled)",
        "",
    ]
    cols = ["branch"] + [f"{m}_{s}" for m in ("auc_roc", "balanced_accuracy", "sensitivity", "specificity", "accuracy")
                         for s in ("mean", "pooled")]
    lines += _md_table(comparison, cols)
    lines += ["", "## Per-fold metrics", ""]
    fold_cols = ["fold", "inner_fit_rows", "inner_calibration_rows", "outer_validation_rows"] + [
        f"{b}_{m}" for b in manifest["branches_evaluated"] for m in ("auc_roc", "balanced_accuracy", "decision_threshold")]
    lines += _md_table(metrics, fold_cols)
    lines += ["", "## Decision thresholds per fold", ""]
    for fold, values in thresholds.items():
        lines.append(f"- fold {fold}: " + ", ".join(f"{b}={fmt(v)}" for b, v in values.items()))
    lines.append("")
    return "\n".join(lines)


def summary_report(*, configurations: pd.DataFrame, ranked: pd.DataFrame, warnings: Sequence[str],
                   ranking_columns: Sequence[str], include_incomplete: bool, top_n: int) -> str:
    completed = int(configurations["is_complete"].sum()) if len(configurations) else 0
    lines = [
        "# Configurable fusion validation report",
        "",
        f"- Discovered configurations: {len(configurations)}",
        f"- Complete configurations: {completed}",
        f"- Incomplete configurations: {len(configurations) - completed}",
    ]
    if not ranked.empty:
        best = ranked.iloc[0]
        lines += [f"- Best configuration: `{best['run_id']}`",
                  f"- Best mean {best.get('primary_branch', 'fusion')} AUC ROC: "
                  f"{fmt(best.get('primary_auc_roc_mean', best.get('fusion_auc_roc_mean')))}"]
    lines += [
        "",
        "## How the best configuration was selected",
        "",
        f"Configurations were ranked by mean outer-fold {', '.join(ranking_columns)} (descending, NaN last, "
        "stable order of discovery for ties). "
        + ("Incomplete runs were included in the ranking. " if include_incomplete
           else "Incomplete runs (missing folds or COMPLETE flag) were excluded from the ranking. ")
        + f"The outer folds that produced these metrics were also used to choose among {len(ranked)} "
          "ranked configurations.",
        "",
        GENERALISATION_NOTE,
        "",
    ]
    if "selection_uses_outer_eval" in configurations.columns and any(v is True for v in configurations["selection_uses_outer_eval"].tolist()):
        lines += ["Some runs used the historical `selection_split=outer` protocol (checkpoint and threshold chosen on "
                  "the evaluated fold); their metrics are optimistic.", ""]
    if not ranked.empty:
        lines += [f"## Top {min(top_n, len(ranked))} configurations", ""]
        cols = ["rank", "run_id", "primary_branch"] + list(ranking_columns)
        lines += _md_table(ranked.head(top_n), cols)
        lines.append("")
    if warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in warnings] + [""]
    return "\n".join(lines)
