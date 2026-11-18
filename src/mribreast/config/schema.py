"""Single source of truth for experiment fields.

``fields.tsv`` (next to this module) is read by BOTH this module and the Bash
Slurm controller (``run_experiment.slurm``), so the controller never needs
Python on the login node.  Columns:

    key  section  type  default  choices  identity  legacy  help

* ``section``: ``path`` | ``train`` | ``validate`` | ``summarize``.
* ``type``: ``int`` ``float`` ``bool`` ``str`` ``choice`` ``csv`` (comma list,
  each item validated against ``choices`` when given) ``ints`` (space-separated
  ints) ``json`` (JSON object) ``path``.
* ``identity``: ``1`` when the value is part of the training identity
  (``train_id``; path + train sections) or evaluation identity (``eval_id``;
  validate section).

CLI flags are always ``--<key> <value>``; booleans are passed explicitly as
``--<key> true|false`` (the controller emits exactly this form).  Legacy
``store_true`` style (bare ``--flag``) is accepted as ``true``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from mribreast import CONFIG_SCHEMA_VERSION

SECTIONS = ("path", "train", "validate", "summarize")
STAGE_SECTIONS = {
    "prepare": ("path", "train"),
    "train": ("path", "train"),
    "validate": ("path", "train", "validate"),
    "summarize": ("path", "summarize"),
}


@dataclass(frozen=True)
class Field:
    key: str
    section: str
    type: str
    default: str
    choices: tuple[str, ...]
    identity: bool
    legacy: str
    help: str


def fields_tsv_path() -> Path:
    return Path(str(resources.files("mribreast.config").joinpath("fields.tsv")))


def load_fields() -> dict[str, Field]:
    out: dict[str, Field] = {}
    with open(fields_tsv_path(), encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            field = Field(
                key=row["key"],
                section=row["section"],
                type=row["type"],
                default=row["default"],
                choices=tuple(c for c in row["choices"].split("|") if c),
                identity=row["identity"] == "1",
                legacy=row["legacy"],
                help=row["help"],
            )
            if field.section not in SECTIONS:
                raise ValueError(f"fields.tsv: bad section for {field.key}")
            if field.key in out:
                raise ValueError(f"fields.tsv: duplicate key {field.key}")
            out[field.key] = field
    return out


FIELDS = load_fields()


class ConfigError(ValueError):
    """Invalid configuration value or forbidden combination."""


def _parse_bool(text: str) -> bool:
    # Strict, identical to the Bash controller (and legacy): only true/false.
    lowered = str(text).strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ConfigError(f"expected true/false, got {text!r}")


def parse_value(field: Field, raw: Any) -> Any:
    """Parse a raw (string) value into its typed form."""
    if not isinstance(raw, str):
        raw = format_value(field, raw)
    t = field.type
    try:
        if t == "int":
            return int(raw)
        if t == "float":
            value = float(raw)
            if not math.isfinite(value):
                raise ConfigError(f"{field.key} must be finite")
            return value
        if t == "bool":
            return _parse_bool(raw)
        if t in ("str", "path"):
            return raw
        if t == "choice":
            if raw not in field.choices:
                raise ConfigError(f"{field.key}={raw!r} not in {field.choices}")
            return raw
        if t == "csv":
            items = [s.strip() for s in raw.split(",") if s.strip()]
            if not items and field.default:
                raise ConfigError(f"{field.key} needs at least one item")
            if field.choices:
                bad = [i for i in items if i not in field.choices]
                if bad:
                    raise ConfigError(f"{field.key}: unknown items {bad}; allowed {field.choices}")
            # order-preserving de-duplication (legacy dict.fromkeys)
            return tuple(dict.fromkeys(items))
        if t == "ints":
            parts = raw.replace(",", " ").split()
            values = tuple(int(p) for p in parts)
            if any(v <= 0 for v in values):
                raise ConfigError(f"{field.key} needs positive integers")
            return values
        if t == "json":
            obj = json.loads(raw) if raw.strip() else {}
            if not isinstance(obj, dict):
                raise ConfigError(f"{field.key} must be a JSON object")
            return obj
    except ConfigError:
        raise
    except (ValueError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{field.key}: cannot parse {raw!r}: {exc}") from exc
    raise ConfigError(f"unknown field type {t}")


def format_value(field: Field, value: Any) -> str:
    """Inverse of :func:`parse_value` (canonical CLI/string form)."""
    t = field.type
    if t == "bool":
        return "true" if value else "false"
    if t == "csv":
        return ",".join(value) if not isinstance(value, str) else value
    if t == "ints":
        return " ".join(str(v) for v in value) if not isinstance(value, str) else value
    if t == "json":
        return json.dumps(value, sort_keys=True, separators=(",", ":")) if not isinstance(value, str) else value
    if t == "float":
        return repr(float(value))
    return str(value)


def fields_for_stage(stage: str) -> list[Field]:
    sections = STAGE_SECTIONS[stage]
    return [f for f in FIELDS.values() if f.section in sections]


def build_parser(stage: str, parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(prog=f"mribreast {stage}")
    for field in fields_for_stage(stage):
        kwargs: dict[str, Any] = dict(dest=field.key, default=None, help=f"{field.help} [default: {field.default!r}]")
        if field.type == "bool":
            kwargs["nargs"] = "?"
            kwargs["const"] = "true"
        elif field.type == "ints":
            kwargs["nargs"] = "+"
        elif field.type == "csv":
            kwargs["nargs"] = "+"  # accepts legacy "--feature_groups a b" and "a,b"
        aliases = [f"--{field.key}"]
        if "-" not in field.key and "_" in field.key:
            aliases.append(f"--{field.key.replace('_', '-')}")
        parser.add_argument(*aliases, **kwargs)
    parser.add_argument("--config_json", default=None, help="Optional JSON file with field values (CLI wins).")
    return parser


def resolve(stage: str, cli_values: Mapping[str, Any] | None = None, base: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the fully resolved, typed configuration for ``stage``.

    Precedence: CLI values > ``base`` mapping > fields.tsv defaults.  Raw CLI
    lists (nargs='+') are joined appropriately before parsing.
    """
    resolved: dict[str, Any] = {}
    cli_values = dict(cli_values or {})
    base = dict(base or {})
    unknown = set(base).difference(FIELDS)
    if unknown:
        raise ConfigError(f"unknown configuration keys: {sorted(unknown)}")
    for field in fields_for_stage(stage):
        raw = cli_values.get(field.key)
        if raw is None:
            raw = base.get(field.key)
        if raw is None:
            raw = field.default
        if isinstance(raw, list):
            raw = ",".join(raw) if field.type == "csv" else " ".join(raw)
        resolved[field.key] = parse_value(field, raw)
    validate_combinations(resolved, stage)
    return resolved


def validate_combinations(cfg: Mapping[str, Any], stage: str) -> None:
    """Reject unsupported or meaningless field combinations with clear errors."""
    if stage == "summarize":
        if cfg["expected_folds"] < 1 or cfg["top_n"] < 1:
            raise ConfigError("expected_folds and top_n must be >= 1")
        return

    def bad(msg: str) -> None:
        raise ConfigError(msg)

    branches = set(cfg["branches"])
    if not branches or not branches <= {"image", "tabular"}:
        bad("branches must be a non-empty subset of image,tabular")
    if cfg["num_folds"] < 2:
        bad("num_folds must be >= 2")
    if cfg["threshold_calibration_folds"] < 2:
        bad("threshold_calibration_folds must be >= 2")
    if cfg["phase_count"] < 1:
        bad("phase_count must be >= 1")
    three_class = cfg["label_scheme"] == "oncotype_3class_18_31"
    n_classes = 3 if three_class else 2
    if not 0 <= cfg["positive_class"] < n_classes:
        bad("positive_class outside label range")
    if cfg["profile"] == "duke_legacy_3d":
        if cfg["image_dim"] != "3d" or cfg["image_source"] != "nifti":
            bad("profile duke_legacy_3d requires image_dim=3d and image_source=nifti (use profile=general for 2D)")
        if cfg["mask_provider"] != "none":
            bad("profile duke_legacy_3d does not use external masks (use profile=general)")
    if "tabular" in branches and not cfg["feature_groups"]:
        bad("tabular branch needs feature_groups")
    if "tabular" not in branches and cfg["feature_selector"] != "none":
        bad("feature_selector requires the tabular branch")
    if "image" in branches:
        if cfg["image_dim"] == "3d":
            if len(cfg["target_size"]) != 3:
                bad("target_size needs three integers for 3D")
            if cfg["image_source"] != "nifti":
                bad("image_dim=3d requires image_source=nifti")
        else:
            if len(cfg["target_size_2d"]) != 2:
                bad("target_size_2d needs two integers for 2D")
            if cfg["slices_per_study"] < 1:
                bad("slices_per_study must be >= 1")
            if cfg["slice_axis"] not in (0, 1, 2):
                bad("slice_axis must be 0, 1 or 2")
            if cfg["image_source"] == "manifest_2d" and cfg["subtraction_mode"] != "none":
                bad("subtraction_mode applies to NIfTI phases only; ready 2D images need subtraction_mode=none")
            if cfg["image_source"] == "manifest_2d" and cfg["slice_selection"] != "center":
                bad("slice_selection is for slicing NIfTI volumes; leave it at center for manifest_2d")
            if cfg["use_annotation_boxes"]:
                bad("use_annotation_boxes is a 3D training augmentation; use masks for 2D")
        if cfg["slice_selection"] in ("mask_max_area", "mask_all") and cfg["mask_provider"] == "none":
            bad("mask-based slice selection requires a mask_provider (masks must also exist at inference)")
        if cfg["mask_mode"] != "none" and cfg["mask_provider"] == "none":
            bad("mask_mode requires a mask_provider")
        if cfg["mask_provider"] != "none":
            if not cfg["mask_roi_classes"]:
                bad("mask_provider needs explicit mask_roi_classes")
            try:
                [int(c) for c in cfg["mask_roi_classes"]]
            except ValueError:
                bad("mask_roi_classes must be integer class codes")
            if cfg["mask_provenance"] == "fold_trained" and cfg["mask_kind"] != "predicted":
                bad("mask_provenance=fold_trained only applies to mask_kind=predicted")
        if not 0.0 <= cfg["roi_margin"] <= 2.0:
            bad("roi_margin must be in [0, 2]")
    else:
        if cfg["use_annotation_boxes"] or cfg["mask_mode"] != "none":
            bad("annotation boxes / masks need the image branch")
    if cfg["selection_split"] == "outer" and "tabular" in branches and cfg["profile"] != "duke_legacy_3d":
        bad("selection_split=outer (historical image-only) is only available in the legacy profile")
    if cfg["selection_split"] == "outer" and branches != {"image"}:
        bad("selection_split=outer reproduces the historical image-only protocol; use branches=image")
    for key in ("trainer_extra_json",):
        protected = {"callbacks", "logger", "default_root_dir", "enable_checkpointing"}
        if protected.intersection(cfg[key]):
            bad(f"{key} cannot override {sorted(protected)}")
    if stage == "validate":
        if not 0.0 <= cfg["fusion_alpha"] <= 1.0:
            bad("fusion_alpha must be within [0, 1]")
        if cfg["primary_branch"] == "fusion" and branches != {"image", "tabular"}:
            bad("primary_branch=fusion needs both branches")
        if cfg["primary_branch"] in ("image", "tabular") and cfg["primary_branch"] not in branches:
            bad(f"primary_branch={cfg['primary_branch']} was not trained")
        if three_class and cfg["threshold_strategy"] != "fixed_0.5":
            # multiclass predictions use argmax; a binary threshold is meaningless
            raise ConfigError("threshold_strategy=balanced_accuracy is binary-only; use fixed_0.5 (argmax) for 3 classes")


def num_classes(cfg: Mapping[str, Any]) -> int:
    return 3 if cfg["label_scheme"] == "oncotype_3class_18_31" else 2


def primary_branch(cfg: Mapping[str, Any]) -> str:
    if cfg.get("primary_branch", "auto") != "auto":
        return cfg["primary_branch"]
    branches = set(cfg["branches"])
    return "fusion" if branches == {"image", "tabular"} else next(iter(branches))


def to_jsonable(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}


def _digest(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def train_identity(cfg: Mapping[str, Any], input_fingerprints: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Values that define a training run.  Paths enter via content fingerprints when given."""
    ident: dict[str, Any] = {"config_schema_version": CONFIG_SCHEMA_VERSION}
    for f in FIELDS.values():
        if f.section == "train" and f.identity:
            ident[f.key] = format_value(f, cfg[f.key])
        elif f.section == "path" and f.identity:
            ident[f.key] = (input_fingerprints or {}).get(f.key, str(cfg.get(f.key, "")))
    return ident


def train_id(cfg: Mapping[str, Any], input_fingerprints: Mapping[str, str] | None = None) -> str:
    return "tr-" + _digest(train_identity(cfg, input_fingerprints))[:12]


def eval_identity(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f.key: format_value(f, cfg[f.key])
        for f in FIELDS.values()
        if f.section == "validate" and f.identity
    }


def eval_id(cfg: Mapping[str, Any]) -> str:
    """Readable + hashed evaluation id, e.g. ``ev-a0.5-1a2b3c4d``."""
    return f"ev-a{cfg['fusion_alpha']:g}-" + _digest(eval_identity(cfg))[:8]
