"""Composition root: ``python -m mribreast <command> --<key> <value> ...``.

This is the only place where concrete infrastructure (cohort source, mask
provider, branch trainers, filesystem roots) is assembled.  Protocol code only
sees the contracts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from mribreast.config import schema

log = logging.getLogger("mribreast")

STAGES = ("prepare", "train", "validate", "summarize")
INPUT_PATH_KEYS = tuple(f.key for f in schema.FIELDS.values() if f.section == "path" and f.identity)


# ----------------------------------------------------------------------------- config
def _resolve_cfg(stage: str, argv: Sequence[str]) -> dict[str, Any]:
    parser = schema.build_parser(stage)
    ns = parser.parse_args(list(argv))
    values = {k: v for k, v in vars(ns).items() if k != "config_json"}
    base: dict[str, Any] = {}
    if ns.config_json:
        base = json.loads(Path(ns.config_json).read_text(encoding="utf-8"))
        base = {k: v for k, v in base.items() if k in schema.FIELDS}
    cfg = schema.resolve(stage, values, base)
    for key, field in schema.FIELDS.items():
        if key in cfg and field.type == "path" and cfg[key]:
            cfg[key] = str(Path(cfg[key]).expanduser().resolve())
    return cfg


def _require_paths(cfg: Mapping[str, Any], keys: Sequence[str], stage: str) -> None:
    missing = [k for k in keys if not cfg.get(k)]
    if missing:
        raise schema.ConfigError(f"{stage}: missing required path(s): {', '.join('--' + k for k in missing)}")
    absent = [f"{k}={cfg[k]}" for k in keys if not Path(cfg[k]).exists()]
    if absent:
        raise schema.ConfigError(f"{stage}: input not found: {'; '.join(absent)}")


def required_inputs(cfg: Mapping[str, Any]) -> list[str]:
    """Inputs needed for this configuration (diagnostics per mode)."""
    keys: list[str] = []
    branches = set(cfg["branches"])
    if cfg["image_source"] == "nifti" or "tabular" in branches or cfg["profile"] == "duke_legacy_3d":
        keys += ["clinical_features_file", "metadata_file"]
    if "image" in branches:
        keys.append("image_root" if cfg["image_source"] == "nifti" else "image_manifest_file")
        if cfg["use_annotation_boxes"]:
            keys.append("annotation_boxes_file")
        if cfg["mask_provider"] == "local_files":
            keys.append("mask_manifest_file")
    if "tabular" in branches:
        groups = set(cfg["feature_groups"])
        if groups & {"kinetic", "morphology", "heterogeneity"}:
            keys.append("imaging_features_file")
        if "density" in groups:
            keys.append("density_file")
    return list(dict.fromkeys(keys))


def _input_fingerprints(cfg: Mapping[str, Any]) -> dict[str, str]:
    from mribreast.artifacts import paths_fingerprint

    used = set(required_inputs(cfg))
    # Large image trees: fingerprint by listing (names + sizes); caches keyed separately.
    return paths_fingerprint({k: (cfg.get(k) if k in used else None) for k in INPUT_PATH_KEYS})


# ----------------------------------------------------------------------------- assembly
def build_cohort(cfg: Mapping[str, Any]):
    from mribreast.data.duke import DukeCohortSource

    return DukeCohortSource(cfg).load(cfg)


def build_mask_provider(cfg: Mapping[str, Any]):
    if cfg["mask_provider"] == "none":
        return None
    if cfg["mask_provider"] == "local_files":
        from mribreast.imaging.masks.local import LocalFileMaskProvider

        return LocalFileMaskProvider(cfg["mask_manifest_file"])
    from mribreast.imaging.masks.synthetic import SyntheticMaskProvider

    return SyntheticMaskProvider(seed=cfg["seed"])


def build_trainers(cfg: Mapping[str, Any], cohort) -> dict[str, Any]:
    n_classes = schema.num_classes(cfg)
    trainers: dict[str, Any] = {}
    if "tabular" in cfg["branches"]:
        from mribreast.tabular.branch import build_tabular_branch

        trainers["tabular"] = build_tabular_branch(
            cfg, continuous_columns=cohort.continuous_columns,
            categorical_columns=cohort.categorical_columns, num_classes=n_classes)
    if "image" in cfg["branches"]:
        from mribreast.models.image_branch import build_image_branch

        logs_dir = Path(cfg["logs_root"]) if cfg.get("logs_root") else None
        trainers["image"] = build_image_branch(cfg, num_classes=n_classes,
                                               mask_provider=build_mask_provider(cfg), logs_dir=logs_dir)
    return trainers


def _image_audits(trainers: Mapping[str, Any], cohort) -> dict[str, Any]:
    image = trainers.get("image")
    if image is None or not hasattr(image, "audit"):
        return {}
    return dict(image.audit(cohort.frame) or {})


# ----------------------------------------------------------------------------- commands
def cmd_prepare(argv: Sequence[str]) -> int:
    """Validate inputs, build the cohort + audits, warm deterministic caches. No training."""
    from mribreast.artifacts import write_json_atomic

    cfg = _resolve_cfg("prepare", argv)
    _require_paths(cfg, required_inputs(cfg), "prepare")
    cohort = build_cohort(cfg)
    fingerprints = _input_fingerprints(cfg)
    tid = schema.train_id(cfg, fingerprints)
    out = Path(cfg["checkpoint_root"]) / "_prepare" / tid
    out.mkdir(parents=True, exist_ok=True)
    cohort.frame.drop(columns=["oncotype_score"], errors="ignore").to_csv(out / "cohort.csv", index=False)
    cohort.audit.to_csv(out / "cohort_audit.csv", index=False)
    trainers = build_trainers(cfg, cohort)
    for name, frame in _image_audits(trainers, cohort).items():
        frame.to_csv(out / f"{name}.csv", index=False)
    image = trainers.get("image")
    if image is not None and hasattr(image, "warm_cache"):
        image.warm_cache(cohort.frame)
    write_json_atomic(out / "prepare_manifest.json", {"train_id": tid, "input_fingerprints": fingerprints,
                                                     "studies": len(cohort.frame), "config": schema.to_jsonable(cfg)})
    print(f"prepare: {len(cohort.frame)} studies, train_id={tid}, outputs in {out}")
    return 0


def cmd_train(argv: Sequence[str]) -> int:
    from mribreast.protocol.train import run_training

    cfg = _resolve_cfg("train", argv)
    _require_paths(cfg, required_inputs(cfg), "train")
    import pytorch_lightning as pl

    pl.seed_everything(cfg["seed"], workers=True)
    cohort = build_cohort(cfg)
    trainers = build_trainers(cfg, cohort)
    train_dir = run_training(cfg, cohort, trainers, checkpoint_root=Path(cfg["checkpoint_root"]),
                             input_fingerprints=_input_fingerprints(cfg),
                             extra_audits=_image_audits(trainers, cohort))
    print(f"train: complete -> {train_dir}")
    return 0


def cmd_validate(argv: Sequence[str]) -> int:
    from mribreast.protocol.validate import run_validation

    cfg = _resolve_cfg("validate", argv)
    _require_paths(cfg, required_inputs(cfg), "validate")
    cohort = build_cohort(cfg)
    trainers = build_trainers(cfg, cohort)
    out = run_validation(cfg, cohort, trainers, checkpoint_root=Path(cfg["checkpoint_root"]),
                         validation_root=Path(cfg["validation_root"]),
                         input_fingerprints=_input_fingerprints(cfg))
    print(f"validate: complete -> {out}")
    return 0


def cmd_summarize(argv: Sequence[str]) -> int:
    from mribreast.reporting.summarize import summarize

    cfg = _resolve_cfg("summarize", argv)
    if not cfg.get("validation_root") or not cfg.get("summary_root"):
        raise schema.ConfigError("summarize needs --validation_root and --summary_root")
    result = summarize(Path(cfg["validation_root"]), Path(cfg["summary_root"]), expected_folds=cfg["expected_folds"],
                       top_n=cfg["top_n"], include_incomplete=cfg["include_incomplete"])
    print(f"summarize: {json.dumps(result, default=str)[:2000]}")
    return 0


def cmd_make_synthetic(argv: Sequence[str]) -> int:
    from mribreast.synthetic.dataset import make_synthetic_dataset

    p = argparse.ArgumentParser(prog="mribreast make-synthetic")
    p.add_argument("--out", required=True)
    p.add_argument("--n_patients", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shape", type=int, nargs=3, default=(24, 24, 12))
    p.add_argument("--size_2d", type=int, nargs=2, default=(32, 32))
    p.add_argument("--images_per_study", type=int, default=3)
    a = p.parse_args(list(argv))
    info = make_synthetic_dataset(Path(a.out), n_patients=a.n_patients, seed=a.seed, shape=tuple(a.shape),
                                  size_2d=tuple(a.size_2d), images_per_study=a.images_per_study)
    print(json.dumps({k: str(v) for k, v in info.items() if k != "studies"}, indent=2))
    return 0


def cmd_import_legacy(argv: Sequence[str]) -> int:
    from mribreast.legacy import importer

    return importer.main(list(argv))


def cmd_fields(argv: Sequence[str]) -> int:
    """Print the field registry (for docs / controller debugging)."""
    p = argparse.ArgumentParser(prog="mribreast fields")
    p.add_argument("--stage", choices=STAGES)
    a = p.parse_args(list(argv))
    fields = schema.fields_for_stage(a.stage) if a.stage else list(schema.FIELDS.values())
    for f in fields:
        print(f"{f.key}\t{f.section}\t{f.type}\t{f.default}\t{'|'.join(f.choices)}")
    return 0


COMMANDS = {
    "prepare": cmd_prepare,
    "train": cmd_train,
    "validate": cmd_validate,
    "summarize": cmd_summarize,
    "make-synthetic": cmd_make_synthetic,
    "import-legacy": cmd_import_legacy,
    "fields": cmd_fields,
}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python -m mribreast {" + ",".join(COMMANDS) + "} [--key value ...]\n"
              "       python -m mribreast <command> --help")
        return 0 if argv else 2
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"unknown command {command!r}; choose from {', '.join(COMMANDS)}", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return COMMANDS[command](rest)
    except schema.ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
