"""Legacy experiment / run naming, reproduced exactly and parsed back.

* Fusion workflow (``configurable_imaging_features_fusion_workflow.build_experiment_name``)::

    ImagingFeaturesFusion_<mri_model>_<subtraction_mode>_<groups joined by '-'>_<feature_model>
      [_lasso-cv<k>-cs<n>-min<m>][_clinical-safe-v2]_auc-cal<k>_lr<lr:.0e>_sens<s:g>_boost<b:g>_bs<n>
      [_annboxes][_<experiment_id>]

  The Slurm controller passes ``experiment_id = cfg<sha256[:12]>`` of the sorted
  training row + input/cache paths.  Validation writes to
  ``<output_dir>/<experiment_name>/fusion_alpha_<alpha:g>`` and the summarizer's
  ``run_id`` is that path relative to the validation root.

* Historical image-only workflow (``trainingWorkflow.py`` / ``validate_best_checkpoints.build_model_name``)::

    <FCN|DenseNet|ResNet18|ResNet10>_lr_<lr:.0e>_sens_<str(s)>_posboost_<str(b)>[_wd_<wd:.0e>]_bs_<n>
      [_sub_<mode>][_annboxes]

Known legacy collisions (documented in docs/migration.md): ``lr`` is rendered
with one significant digit (``1.6e-4`` and ``2e-4`` share ``lr2e-04``), and
options outside the name (seed, target_size, epoch, xgb_*/mlp_* settings,
include_sensitive, input paths) only differ through ``experiment_id``.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

FUSION_PREFIX = "ImagingFeaturesFusion"
CLINICAL_SAFE_SUFFIX = "_clinical-safe-v2"
HISTORICAL_MODELS = ("FCN", "DenseNet", "ResNet18", "ResNet10")  # --model 0..3
HISTORICAL_MODEL_TO_NEW = {"FCN": "fcn", "DenseNet": "densenet121", "ResNet18": "resnet18", "ResNet10": "resnet10"}

_MRI = "fcn|densenet121|resnet10|resnet18"
_SUB = "none|post_minus_pre|consecutive"
FUSION_NAME_PATTERN = re.compile(
    rf"^{FUSION_PREFIX}_(?P<mri_model>{_MRI})_(?P<subtraction_mode>{_SUB})_"
    r"(?P<feature_groups>[a-z]+(?:-[a-z]+)*)_(?P<feature_model>xgboost|mlp)"
    r"(?:_lasso-cv(?P<lasso_cv_folds>\d+)-cs(?P<lasso_cs>\d+)-min(?P<lasso_min_features>\d+))?"
    r"(?P<clinical_safe>_clinical-safe-v2)?"
    r"_auc-cal(?P<threshold_calibration_folds>\d+)"
    r"_lr(?P<lr>[0-9.eE+-]+?)_sens(?P<sensitivity_lambda>[0-9.eE+-]+?)_boost(?P<positive_boost>[0-9.eE+-]+?)"
    r"_bs(?P<batch_size>\d+)(?P<annboxes>_annboxes)?(?:_(?P<experiment_id>.+))?$"
)
#: Older fusion names without inner calibration (pre ``auc-cal``): anything else starting with the prefix.
HISTORICAL_IMAGE_PATTERN = re.compile(
    r"^(?P<model>FCN|DenseNet|ResNet18|ResNet10)_lr_(?P<lr>[0-9.eE+-]+?)_sens_(?P<sensitivity_lambda>[0-9.eE+-]+?)"
    r"_posboost_(?P<positive_boost>[0-9.eE+-]+?)(?:_wd_(?P<weight_decay>[0-9.eE+-]+?))?_bs_(?P<batch_size>\d+)"
    r"(?:_sub_(?P<subtraction_mode>post_minus_pre|consecutive))?(?P<annboxes>_annboxes)?$"
)
ALPHA_DIR_PATTERN = re.compile(r"^fusion_alpha_(?P<alpha>[0-9.eE+-]+)$")


def _groups(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [v for v in re.split(r"[,\s]+", value) if v]
    return tuple(dict.fromkeys(str(v) for v in value))


def build_experiment_name(cfg: Mapping[str, Any]) -> str:
    """Byte-identical port of legacy ``build_experiment_name(args, selected_groups)``.

    ``cfg`` uses the new field names (identical to the legacy CLI names for every
    key involved).  ``experiment_id`` defaults to ``""`` (no suffix).
    """
    groups = _groups(cfg["feature_groups"])
    selector_suffix = ""
    if cfg.get("feature_selector", "none") == "lasso":
        selector_suffix = (f"_lasso-cv{int(cfg.get('lasso_cv_folds', 5))}-cs{int(cfg.get('lasso_cs', 20))}-"
                           f"min{int(cfg.get('lasso_min_features', 1))}")
    clinical_suffix = CLINICAL_SAFE_SUFFIX if "clinical" in groups else ""
    experiment_id = str(cfg.get("experiment_id", "") or "")
    return (
        f"{FUSION_PREFIX}_{cfg['mri_model']}_{cfg['subtraction_mode']}_"
        f"{'-'.join(groups)}_{cfg['feature_model']}{selector_suffix}{clinical_suffix}_"
        f"auc-cal{int(cfg['threshold_calibration_folds'])}_lr{float(cfg['lr']):.0e}_"
        f"sens{float(cfg['sensitivity_lambda']):g}_boost{float(cfg['positive_boost']):g}_"
        f"bs{int(cfg['batch_size'])}"
        f"{'_annboxes' if _truthy(cfg.get('use_annotation_boxes', False)) else ''}"
        f"{'_' + experiment_id if experiment_id else ''}"
    )


def alpha_dir(alpha: float) -> str:
    """``fusion_alpha_<alpha:g>`` (legacy validation output sub-directory)."""
    return f"fusion_alpha_{float(alpha):g}"


def legacy_run_id(cfg: Mapping[str, Any], alpha: float) -> str:
    return f"{build_experiment_name(cfg)}/{alpha_dir(alpha)}"


def build_historical_image_name(model: str | int, lr: float, sensitivity_lambda: float, positive_boost: float,
                                batch_size: int, subtraction_mode: str = "none", use_annotation_boxes: bool = False,
                                weight_decay: float | None = None) -> str:
    """Name used by ``trainingWorkflow.py`` (and matched by ``validate_best_checkpoints.build_model_name``).

    ``sensitivity_lambda``/``positive_boost`` are rendered with ``str(float)`` as in legacy f-strings.
    ``weight_decay`` only exists on the validation side (training never wrote it).
    """
    base = HISTORICAL_MODELS[model] if isinstance(model, int) else model
    parts = [f"{base}_lr_{float(lr):.0e}", f"sens_{float(sensitivity_lambda)}", f"posboost_{float(positive_boost)}"]
    if weight_decay is not None:
        parts.append(f"wd_{float(weight_decay):.0e}")
    parts.append(f"bs_{int(batch_size)}")
    if subtraction_mode != "none":
        parts.append(f"sub_{subtraction_mode}")
    return "_".join(parts) + ("_annboxes" if use_annotation_boxes else "")


def parse_experiment_name(name: str) -> dict[str, Any]:
    """Parse a legacy experiment (checkpoint/log directory) name.

    Returns a dict with ``workflow`` in ``fusion | historical_image | unknown``,
    the decoded options (new field names) and era flags:

    * ``clinical_safe_v2`` - True/False for fusion runs that include ``clinical``;
      None when the clinical group is absent.
    * ``inner_calibration`` - True for ``auc-cal<k>`` fusion names, False for the
      historical image-only workflow (outer fold used for checkpoint selection).
    """
    name = name.strip().strip("/")
    match = FUSION_NAME_PATTERN.match(name)
    if match:
        g = match.groupdict()
        groups = tuple(g["feature_groups"].split("-"))
        out: dict[str, Any] = {
            "workflow": "fusion",
            "mri_model": g["mri_model"],
            "subtraction_mode": g["subtraction_mode"],
            "feature_groups": ",".join(groups),
            "feature_model": g["feature_model"],
            "feature_selector": "lasso" if g["lasso_cv_folds"] else "none",
            "threshold_calibration_folds": int(g["threshold_calibration_folds"]),
            "lr": float(g["lr"]),
            "sensitivity_lambda": float(g["sensitivity_lambda"]),
            "positive_boost": float(g["positive_boost"]),
            "batch_size": int(g["batch_size"]),
            "use_annotation_boxes": bool(g["annboxes"]),
            "experiment_id": g["experiment_id"] or "",
            "clinical_safe_v2": (bool(g["clinical_safe"]) if "clinical" in groups else None),
            "inner_calibration": True,
            "branches": "image,tabular",
            "label_scheme": "oncotype_binary_18",
        }
        if g["lasso_cv_folds"]:
            out.update(lasso_cv_folds=int(g["lasso_cv_folds"]), lasso_cs=int(g["lasso_cs"]),
                       lasso_min_features=int(g["lasso_min_features"]))
        return out
    match = HISTORICAL_IMAGE_PATTERN.match(name)
    if match:
        g = match.groupdict()
        return {
            "workflow": "historical_image",
            "mri_model": HISTORICAL_MODEL_TO_NEW[g["model"]],
            "subtraction_mode": g["subtraction_mode"] or "none",
            "lr": float(g["lr"]),
            "sensitivity_lambda": float(g["sensitivity_lambda"]),
            "positive_boost": float(g["positive_boost"]),
            "weight_decay": float(g["weight_decay"]) if g["weight_decay"] else None,
            "batch_size": int(g["batch_size"]),
            "use_annotation_boxes": bool(g["annboxes"]),
            "experiment_id": "",
            "clinical_safe_v2": None,
            "inner_calibration": False,
            "branches": "image",
            # the class count is not part of the name (``--is_binary_classification`` was not encoded)
            "label_scheme": "unknown",
        }
    return {"workflow": "unknown", "inner_calibration": None, "clinical_safe_v2": None}


def split_run_id(run_id: str) -> tuple[str, float | None]:
    """``<experiment_name>/fusion_alpha_<a>`` -> (experiment_name, alpha); alpha None if absent."""
    parts = [p for p in str(run_id).replace("\\", "/").split("/") if p]
    if parts and ALPHA_DIR_PATTERN.match(parts[-1]):
        return "/".join(parts[:-1]), float(ALPHA_DIR_PATTERN.match(parts[-1]).group("alpha"))
    return "/".join(parts), None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)
