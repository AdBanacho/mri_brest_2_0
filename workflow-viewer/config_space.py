"""Enumerate the configuration space of mribreast using the project's OWN validation code.

Imported by sync_data.py. Nothing here re-implements a rule: validity comes from
``mribreast.config.schema.validate_combinations`` and required input files from
``mribreast.cli.required_inputs``, so the viewer cannot drift from the code.

What is enumerated
------------------
* Structural dimensions (fields that change which stages run or that interact through
  validation rules) are enumerated exhaustively. A dimension that has no effect in a
  combination is recorded as "n/a" instead of multiplying identical pipelines
  (e.g. image settings when branches=tabular).
* Independent categorical options (network type, optimizer, feature groups, ...) are not
  enumerated row by row; for each structural row we count how many valid choices they
  add (validated with the real code where a rule applies) -> ``variants``.
* Free numeric / text fields (epochs, learning rate, sizes, paths, JSON) have unbounded
  ranges and are listed but not counted.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.dont_write_bytecode = True

from mribreast.config import schema  # noqa: E402  (stdlib-only module)
from mribreast import cli  # noqa: E402  (imports schema + stdlib at module level only)

PLACEHOLDER_PATH = "<placeholder>"

# ----------------------------------------------------------------------------- dimensions
# (key, label, plain-language explanation)
DIMS = [
    ("branches", "Branches", "Which models are trained: image only, tabular only, or both (then fused)."),
    ("profile", "Profile", "duke_legacy_3d reproduces the old pipeline exactly; general enables 2D, masks and verified phase selection."),
    ("label_scheme", "Label scheme", "Binary (<=18 / >18) or three classes (<=18 / 18-31 / >31)."),
    ("selection_split", "Selection split", "Which held-aside patients choose the checkpoint and the decision threshold."),
    ("image_dim", "Image dimension", "3D volumes or 2D slices/images."),
    ("image_source", "Image source", "NIfTI volumes, or ready 2D images listed in a manifest."),
    ("subtraction_mode", "Subtraction", "Use raw contrast phases or differences between phases."),
    ("mask_provider", "Mask provider", "Where lesion masks come from (none, mask files, or synthetic masks for tests)."),
    ("mask_mode", "Mask mode", "How a mask is used: crop to the lesion, extra channel, or zero the background."),
    ("slice_selection", "2D slice selection", "Which slices are cut from a volume in 2D mode."),
    ("box_mode", "Box mode", "Use the Duke annotation box as a model input (3D only)."),
    ("use_annotation_boxes", "Box erasing augmentation", "Randomly blank background (not the lesion box) during 3D training."),
    ("feature_selector", "Feature selector", "Optional LASSO feature selection in the tabular branch."),
]
DIM_KEYS = [d[0] for d in DIMS]

BRANCH_CHOICES = ["image", "tabular", "image,tabular"]


def _choices(key: str) -> list[str]:
    if key == "branches":
        return BRANCH_CHOICES
    if schema.FIELDS[key].type == "bool":
        return ["false", "true"]
    return list(schema.FIELDS[key].choices)


def relevant(key: str, v: dict[str, str]) -> bool:
    """Is dimension ``key`` meaningful given the other (canonical string) values?"""
    if "branches" not in v:
        return True
    image ="image" in v["branches"].split(",")
    tabular = "tabular" in v["branches"].split(",")
    if key in ("image_dim", "image_source", "subtraction_mode", "mask_provider", "box_mode", "use_annotation_boxes"):
        return image
    if key == "slice_selection":
        return image and v.get("image_dim") == "2d" and v.get("image_source") == "nifti"
    if key == "mask_mode":
        return image and v.get("mask_provider", "none") != "none"
    if key == "feature_selector":
        return tabular
    return True


def _base_cfg() -> dict[str, Any]:
    cfg = schema.resolve("validate", {})
    cfg["annotation_boxes_file"] = PLACEHOLDER_PATH  # box_mode rule needs a configured file
    return cfg


def _apply(base: dict[str, Any], values: dict[str, str]) -> dict[str, Any]:
    cfg = dict(base)
    for k, raw in values.items():
        cfg[k] = schema.parse_value(schema.FIELDS[k], raw)
    # validate-time choices are counted as variants; use the always-valid one here
    cfg["threshold_strategy"] = "fixed_0.5" if cfg["label_scheme"] == "oncotype_3class_18_31" else "balanced_accuracy"
    cfg["primary_branch"] = "auto"
    return cfg


def _check(cfg: dict[str, Any]) -> str | None:
    try:
        schema.validate_combinations(cfg, "validate")
        return None
    except schema.ConfigError as exc:
        return str(exc)


def _enumerate_structural():
    """Yield dicts of canonical values; irrelevant dimensions are omitted (n/a)."""
    def rec(i: int, values: dict[str, str]):
        if i == len(DIM_KEYS):
            yield dict(values)
            return
        key = DIM_KEYS[i]
        if not relevant(key, values):
            yield from rec(i + 1, values)
            return
        for choice in _choices(key):
            values[key] = choice
            yield from rec(i + 1, values)
            del values[key]
    yield from rec(0, {})


def _defaults() -> dict[str, str]:
    return {k: schema.format_value(schema.FIELDS[k], schema.parse_value(schema.FIELDS[k], schema.FIELDS[k].default))
            for k in schema.FIELDS}


def _full_values(values: dict[str, str]) -> dict[str, str]:
    d = _defaults()
    out = {k: d[k] for k in DIM_KEYS}
    out.update(values)
    return out


def _uses_masks(v: dict[str, str]) -> bool:
    return v.get("mask_mode", "none") != "none" or v.get("slice_selection") in ("mask_max_area", "mask_all")


def _variants(base: dict[str, Any], v: dict[str, str]) -> tuple[int, list[dict[str, Any]]]:
    """Count valid choices of the independent categorical options for one structural row."""
    branches = v["branches"].split(",")
    image, tabular = "image" in branches, "tabular" in branches
    cfg = _apply(base, _full_values(v))
    parts: list[dict[str, Any]] = []

    def part(label: str, n: int, note: str = "") -> None:
        parts.append({"label": label, "n": n, "note": note})

    # validate-time choices, checked with the real validator
    n_thr = sum(_check(dict(cfg, threshold_strategy=t)) is None for t in schema.FIELDS["threshold_strategy"].choices)
    part("threshold_strategy", n_thr)
    n_pri = sum(_check(dict(cfg, primary_branch=p)) is None for p in schema.FIELDS["primary_branch"].choices)
    part("primary_branch", n_pri)
    part("report_unit", len(schema.FIELDS["report_unit"].choices))
    if image:
        for key in ("mri_model", "loss", "optimizer", "scheduler"):
            part(key, len(schema.FIELDS[key].choices))
        part("early_stopping", 2)
        if v.get("image_dim") == "2d":
            part("aggregation", len(schema.FIELDS["aggregation"].choices), "several images per study")
        if _uses_masks(v):
            n_mask = sum(_check(dict(cfg, mask_kind=k, mask_provenance=p)) is None
                         for k in schema.FIELDS["mask_kind"].choices for p in schema.FIELDS["mask_provenance"].choices)
            part("mask_kind x mask_provenance", n_mask, "fold_trained needs predicted masks")
            n_pol = len(schema.FIELDS["mask_missing_policy"].choices) * len(schema.FIELDS["mask_empty_policy"].choices) \
                * len(schema.FIELDS["mask_invalid_policy"].choices)
            part("mask missing/empty/invalid policies", n_pol)
        if v.get("box_mode", "none") != "none":
            part("box_missing_policy", len(schema.FIELDS["box_missing_policy"].choices))
    if tabular:
        groups = list(schema.FIELDS["feature_groups"].choices)
        n_groups = 0
        for r in range(1, len(groups) + 1):
            for combo in itertools.combinations(groups, r):
                n_groups += 2 if "clinical" in combo else 1  # include_sensitive only matters with clinical
        part("feature_groups x include_sensitive", n_groups, "non-empty subsets of 5 groups; include_sensitive only with clinical")
        n_model = sum(1 if m == "xgboost" else len(schema.FIELDS["mlp_early_stopping"].choices)
                      for m in schema.FIELDS["feature_model"].choices)
        part("feature_model x mlp_early_stopping", n_model, "mlp_early_stopping only for mlp")
        if v.get("feature_selector") == "lasso":
            part("selector_preprocessing", len(schema.FIELDS["selector_preprocessing"].choices))
    total = 1
    for p in parts:
        total *= max(1, p["n"])
    return total, parts


# ----------------------------------------------------------------------------- stage relevance
def stage_notes(v: dict[str, str]) -> dict[str, str]:
    """Stages that are skipped (value starts with 'off:') or behave differently for this configuration."""
    branches = v["branches"].split(",")
    image, tabular = "image" in branches, "tabular" in branches
    notes: dict[str, str] = {}
    if not image:
        for s in ("image_preprocess", "image_branch_fit"):
            notes[s] = "off: no image branch"
        notes["segment_mri_core"] = "off: masks need the image branch"
        notes["nbia_download"] = "off: no images are read without the image branch"
        notes["dicom_to_nifti"] = "off: no images are read without the image branch"
    if not tabular:
        for s in ("tabular_features", "tabular_branch_fit"):
            notes[s] = "off: no tabular branch"
        notes["mricore_embed"] = "off: MRI-CORE embeddings feed the tabular branch"
    else:
        notes["mricore_embed"] = "on: only if feature_groups includes mricore (reads NIfTI volumes even without the image branch)"
    if not (image and tabular):
        notes["fusion"] = "off: fusion needs both branches"
    if image and v.get("image_source") == "manifest_2d":
        notes["nbia_download"] = "off: ready 2D images are used instead of NIfTI volumes"
        notes["dicom_to_nifti"] = "off: ready 2D images are used instead of NIfTI volumes"
    if image and v.get("mask_provider") != "local_files":
        notes.setdefault("segment_mri_core", "off: no mask files are read (mask_provider != local_files)")
    elif image:
        notes["segment_mri_core"] = "on: one way to produce the mask manifest (manual masks also work)"
    if v["label_scheme"] == "oncotype_3class_18_31":
        notes["thresholds"] = "changed: three classes use argmax, no threshold is tuned"
    if v["selection_split"] == "outer":
        notes["split_plan"] = "changed: no inner split; the outer fold selects the checkpoint (optimistic)"
    elif v["selection_split"] == "separate":
        notes["split_plan"] = "changed: calibration patients split again into selection and threshold"
    if image and v.get("box_mode", "none") != "none":
        notes["image_preprocess"] = f"changed: box_mode={v['box_mode']} - the Duke box is needed at inference too"
    elif image and _uses_masks(v):
        notes["image_preprocess"] = "changed: lesion masks drive cropping / channel / slice selection"
    elif image and v.get("mask_provider", "none") != "none":
        notes["image_preprocess"] = "changed: masks are loaded and audited but not used as input (mask_mode=none)"
    return notes


# ----------------------------------------------------------------------------- field catalogue
GROUPS = [
    ("Pipeline shape", "resolve_config", lambda k: k in ("branches", "profile", "label_scheme", "positive_class")),
    ("Cohort & data", "build_cohort", lambda k: k in ("phase_count", "phase_order", "seed", "experiment_id")),
    ("Splits", "split_plan", lambda k: k in ("num_folds", "threshold_calibration_folds", "selection_split")),
    ("Imaging", "image_preprocess", lambda k: k in ("image_dim", "image_source", "subtraction_mode", "target_size",
                                                     "target_size_2d", "slice_axis", "slice_selection",
                                                     "slices_per_study", "aggregation", "loss_unit", "resize_backend",
                                                     "use_annotation_boxes", "annotation_erase_attempts")),
    ("Masks", "image_preprocess", lambda k: k.startswith("mask_") or k == "roi_margin"),
    ("Boxes as model input", "image_preprocess", lambda k: k.startswith("box_") or k == "dual_stream_embed_dim"),
    ("Image model training", "image_branch_fit", lambda k: k in ("mri_model", "epoch", "batch_size", "num_workers",
                                                                 "positive_boost", "sensitivity_lambda", "loss",
                                                                 "optimizer", "lr", "weight_decay", "scheduler",
                                                                 "mri_extra_json", "trainer_extra_json")
     or k.startswith("early_stopping")),
    ("Tabular", "tabular_branch_fit", lambda k: k.startswith(("feature_", "lasso_", "xgb_", "mlp_")) or k in (
        "imaging_patient_id_column", "allow_missing_imaging_features", "include_sensitive", "prediction_time",
        "selector_preprocessing")),
]


def field_catalogue() -> list[dict[str, Any]]:
    out = []
    for f in schema.FIELDS.values():
        if f.section == "path":
            group, stage = "Paths (input files & output roots)", "controller"
        elif f.section == "validate":
            group, stage = "Validation", "thresholds"
        elif f.section == "summarize":
            group, stage = "Summarize", "summarize"
        else:
            group, stage = next(((g, s) for g, s, test in GROUPS if test(f.key)), ("Other", "resolve_config"))
        out.append({"key": f.key, "section": f.section, "type": f.type, "default": f.default,
                    "choices": list(f.choices), "identity": bool(f.identity), "help": f.help,
                    "group": group, "stage": stage, "structural": f.key in DIM_KEYS})
    return out


# ----------------------------------------------------------------------------- shipped configs
_ASSOC = re.compile(r"declare -A (\w+)=\((.*?)\)\s*(?:#.*)?$", re.S | re.M)
_ENTRY = re.compile(r"\[(\w+)\]=('(?:[^']*)'|\"(?:[^\"]*)\"|[^\s#)]*)")


def _resolved_text(path: Path) -> str:
    """Config text; a file that sources another one (configs/campaign/) is resolved by Bash itself."""
    text = path.read_text(encoding="utf-8")
    if not re.search(r"^\s*(source|\.)\s", text, re.M):
        return text
    import subprocess

    script = 'source "$1" >/dev/null; declare -p TRAIN TRAIN_GRID VALIDATE VALIDATE_GRID PATHS 2>/dev/null; true'
    return subprocess.run(["bash", "-c", script, "bash", str(path)], cwd=path.parent, capture_output=True,
                          text=True, check=True, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")}).stdout


def parse_conf(path: Path) -> dict[str, dict[str, str]]:
    text = _resolved_text(path)
    blocks: dict[str, dict[str, str]] = {}
    for match in re.finditer(r"declare -A (\w+)=\(", text):
        name, i, depth = match.group(1), match.end(), 1
        j = i
        while j < len(text) and depth:  # find the matching ')' outside quotes / comments
            c = text[j]
            if c == "'":
                j = text.index("'", j + 1)
            elif c == "#":
                j = text.index("\n", j)
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            j += 1
        body = re.sub(r"#[^\n]*", "", text[i:j - 1])
        blocks[name] = {k: v.strip("'\"") for k, v in _ENTRY.findall(body)}
    return blocks


def structural_key(values: dict[str, str]) -> tuple:
    return tuple(values.get(k, "n/a") if relevant(k, values) else "n/a" for k in DIM_KEYS)


def shipped_configs(index_of: dict[tuple, int]) -> list[dict[str, Any]]:
    out = []
    files = [PROJECT / "experiment.conf", *sorted((PROJECT / "configs").glob("*.conf")),
             *sorted(f for f in (PROJECT / "configs" / "campaign").glob("*.conf") if f.name != "base.conf")]
    for path in files:
        b = parse_conf(path)
        train, grid = b.get("TRAIN", {}), b.get("TRAIN_GRID", {})
        vgrid = b.get("VALIDATE_GRID", {})
        validate_fixed = {k: v for k, v in b.get("VALIDATE", {}).items() if k in schema.FIELDS}
        paths = {k: v for k, v in b.get("PATHS", {}).items() if k in schema.FIELDS and v}
        axes = sorted(grid)
        choices = [grid[k].split("|") for k in axes]
        rows = []
        for combo in itertools.product(*choices):  # last sorted key varies fastest (controller semantics)
            row = dict(zip(axes, combo))
            raw = {**paths, **train, **validate_fixed, **row}
            raw = {k: v for k, v in raw.items() if k in schema.FIELDS}
            raw["fusion_alpha"] = vgrid.get("fusion_alpha", "0.5").split("|")[0]
            error, sidx = None, None
            try:
                cfg = schema.resolve("validate", raw)
                canon = {k: schema.format_value(schema.FIELDS[k], cfg[k]) for k in DIM_KEYS}
                sidx = index_of.get(structural_key(canon))
            except schema.ConfigError as exc:
                error = str(exc)
            rows.append({"values": row, "structural": sidx, "error": error})
        out.append({
            "file": path.relative_to(PROJECT).as_posix(),
            "fixed": {k: v for k, v in train.items() if k in DIM_KEYS or k in ("mri_model", "feature_groups", "feature_model")},
            "axes": axes,
            "validateGrid": {k: v.split("|") for k, v in vgrid.items()},
            "rows": rows,
            "trainTasks": len(rows),
            "validateTasks": len(rows) * max(1, len(vgrid.get("fusion_alpha", "x").split("|"))),
        })
    return out


# ----------------------------------------------------------------------------- main
def build() -> dict[str, Any]:
    base = _base_cfg()
    dims = [{"key": k, "label": label, "help": help_, "values": _choices(k),
             "default": _defaults()[k] if k != "branches" else "image,tabular"} for k, label, help_ in DIMS]
    value_index = {d["key"]: {v: i for i, v in enumerate(d["values"])} for d in dims}
    path_keys = [f.key for f in schema.FIELDS.values() if f.section == "path"]

    rows, rejected, seen = [], Counter(), 0
    shared: list[Any] = []          # de-duplicated row details; rows refer to them by index
    shared_index: dict[str, int] = {}

    def intern(obj: Any) -> int:
        key = json.dumps(obj, sort_keys=True)
        if key not in shared_index:
            shared_index[key] = len(shared)
            shared.append(obj)
        return shared_index[key]

    index_of: dict[tuple, int] = {}
    total = 0
    for values in _enumerate_structural():
        seen += 1
        error = _check(_apply(base, _full_values(values)))
        if error:
            rejected[error] += 1
            continue
        n_var, parts = _variants(base, values)
        total += n_var
        cfg = _apply(base, _full_values(values))
        cfg["feature_groups"] = ("clinical",)
        needed = cli.required_inputs(cfg)
        index_of[structural_key(values)] = len(rows)
        rows.append([
            [value_index[k][values[k]] if k in values else -1 for k in DIM_KEYS],
            n_var,
            intern(parts),
            intern([path_keys.index(k) for k in needed if k in path_keys]),
            intern(stage_notes(values)),
        ])
    numeric = [f.key for f in schema.FIELDS.values()
               if f.section in ("train", "validate", "summarize") and not f.choices and f.type != "bool"
               and f.key not in DIM_KEYS]
    return {
        "source": ["src/mribreast/config/fields.tsv", "src/mribreast/config/schema.py:validate_combinations",
                   "src/mribreast/cli.py:required_inputs", "experiment.conf", "configs/*.conf", "configs/campaign/*.conf"],
        "dims": dims,
        "pathKeys": path_keys,
        "rowFormat": ["dimension value indexes (-1 = n/a)", "categorical variants", "shared[] index: variant parts",
                      "shared[] index: required input path indexes", "shared[] index: stage notes"],
        "rows": rows,
        "shared": shared,
        "enumerated": seen,
        "valid": len(rows),
        "totalVariants": str(total),
        "rejected": [{"message": m, "count": c} for m, c in rejected.most_common()],
        "numericFields": numeric,
        "fields": field_catalogue(),
        "shipped": shipped_configs(index_of),
        "notes": [
            "Validity of every combination is decided by schema.validate_combinations (the code the CLI runs).",
            "A dimension shown as 'n/a' has no effect in that configuration and is not multiplied.",
            "Required inputs are computed by cli.required_inputs with feature_groups=clinical; kinetic/morphology/"
            "heterogeneity add imaging_features_file, density adds density_file and mricore adds "
            "mricore_features_file.",
            "annotation_boxes_file is assumed to be configured (box_mode needs it).",
        ],
    }


if __name__ == "__main__":
    data = build()
    print(data["enumerated"], "enumerated,", data["valid"], "valid,", data["totalVariants"], "categorical variants")
    for s in data["shipped"]:
        print(s["file"], s["trainTasks"], "rows", [(r["structural"], r["error"]) for r in s["rows"]][:3])
