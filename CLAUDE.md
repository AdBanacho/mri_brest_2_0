# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`mribreast` 2.0 rewrites the legacy `mri_brest` research pipeline (reference repo `C:\code\mri_brest`, SHA `127a260`). It predicts Oncotype DX risk on the Duke breast-MRI cohort. Inputs are 3D or 2D MRI, external masks or Duke annotation boxes, and tabular branches, combined by late fusion `p = α·p_image + (1−α)·p_tabular`. Evaluation uses patient-grouped nested CV. The original brief for the rewrite (in Polish) is `prompt-przepisanie-pipeline-mri.md`. `docs/legacy-parity.md` maps every legacy option to its new key and the test that checks it. Check it before you change behaviour that has a legacy counterpart.

## Commands

The platform is Windows. Bash scripts run under Git Bash.

```bash
python -m pip install -e ".[test]"                 # compute stages; ".[prepare]" adds NBIA/DICOM/MAMA-MIA deps
python -m pytest -q                                # full suite (unit, parity, bash controller, e2e)
python -m pytest tests/unit/test_box_modes.py -q   # one file
python -m pytest tests/unit/test_box_modes.py::test_name -q
python -m pytest -m "not slow" -q                  # skip e2e mode smoke runs (minutes on CPU)
bash tests/slurm/test_grid.sh                      # one Bash controller suite directly (fake sbatch/python/module)
```

Pytest markers: `slow` (e2e), `parity` and `bash`. Parity tests compare against the reference repo. They look for it in `MRI_BREAST_LEGACY_REPO` and fall back to `C:\code\mri_brest`. When neither exists they skip. They must never write to that repo: a fixture checks its tree before and after the run, and `tests/conftest.py` turns off bytecode writing for this reason. `tests/unit/test_controller_bash.py` runs every `tests/slurm/test_*.sh` under pytest. It also checks that `run_experiment.sh` stays at 5 lines or fewer, that the launcher files use LF line endings, and that no cluster paths appear in the package.

Running the pipeline:

```bash
python -m mribreast make-synthetic --out runs/synthetic/data
bash run_experiment.sh --local --train     --config configs/synthetic_cpu_3d.conf   # then --validate, --summarize
bash run_experiment.sh --train --dry-run --show-config                              # inspect grid/argv without running
python -m mribreast fields                                                          # list all config keys
```

`python -m mribreast {prepare,train,validate,summarize,make-synthetic,import-legacy,prepare-data,fields}` takes `--<key> <value>` for every key in `fields.tsv`. With `--local`, the controller runs each task on this machine using exactly the argv a Slurm array task would receive.

## Architecture (see `docs/architecture.md`, which is a binding contract)

- **Single field registry:** `src/mribreast/config/fields.tsv` has the columns `key, section, type, default, choices, identity, legacy, help` and sections `path/train/validate/summarize`. Bash (`run_experiment.slurm`, which builds argv from `experiment.conf` and the grid) and Python (`config/schema.py`, which builds argparse and the typed resolve) both read it. To add a config option, add a row here and then add any combination rules in `schema.validate_combinations`. `tests/unit/test_controller_cli_parity.py` checks that every argv the controller builds for the shipped configs is accepted by the Python CLI.
- **Identity:** `train_id = "tr-" + sha256(train identity)[:12]` is computed from `identity=1` train fields plus content fingerprints of identity paths. `eval_id = "ev-a<alpha>-<hash8>"` is computed from validate fields. Parallelism settings, output and cache roots, and `experiment_id` are excluded. Python is authoritative, and the Bash controller never computes ids. If you add a field to an existing feature, keep old `train_id`s stable. For example, `BOX_IDENTITY_FIELDS` only enter the identity when `box_mode != none`.
- **Composition root:** only `cli.py` constructs infrastructure (sources, mask providers, branch trainers). `protocol/` (splits, thresholds, fusion, resumable train, validate) works only through the Protocols in `contracts.py` (`CohortSource`, `MaskProvider`, `BranchTrainer`, `FittedPredictor`) and never imports a concrete model. No module touches the network or reads data at import time. `prepare/` is a separate stage with its own optional dependencies, and train/validate never import it.
- **Artifacts:** training writes `<checkpoint_root>/<train_id>/fold_<k>/…` and validation writes `<validation_root>/<train_id>/<eval_id>/…`. Directories are published atomically with `COMPLETE` flags. Training resumes by skipping complete folds whose manifest matches the current `train_id` and split hash. Validation raises `ArtifactMismatch` when the train manifest differs from what it recomputes. Manifests carry `schema_version` and `config_schema_version`. Use the helpers in `artifacts/core.py` for atomic writes and fingerprints.
- **Splits:** outer `StratifiedGroupKFold` by patient. Roles are `fit/calibration/selection/threshold/outer_eval`, and `selection_split` (`shared|separate|outer`) controls how checkpoint selection and threshold tuning share patients. Leakage checks live in `tests/review/`.
- **Imaging:** the deterministic cache uses file locks and `os.replace` publishing, and augmentations are never cached. Lesion location comes from exactly one of three sources, which are mutually exclusive: external masks (`mask_mode`: `roi_crop|mask_channel|foreground_only`, in `imaging/masks/`), Duke annotation boxes as model input (`box_mode`: `roi_crop|box_channel|dual_stream`, in `imaging/boxes/`, 3D only), or mask-based 2D slice selection. Crops are computed on the **native** grid before resizing. Geometry conventions and the order of operations are in `docs/imaging.md`.
- **Lesion segmentation (`prepare/segmentation`):** `prepare-data segment-mri-core` prompts MRI-CORE/SAM (external, read-only `C:\code\mri_foundation`, override with `MRI_FOUNDATION_REPO`) with the Duke boxes and writes a mask manifest for `mask_provider=local_files`. Our own loader replaces the repo's builder. Never use the repo's `num_cls=1` setup zero-shot, because its mask tokens are random. Real-weight tests (`test_segmentation_mri_core.py`) need the `.[segment]` extras and skip otherwise. See `docs/segmentation-mri-core.md`.
- **Reporting:** metric columns follow the legacy pattern `<branch>_<metric>`. `legacy/` imports and exports historical reports and maps legacy checkpoint names and run ids (see `docs/migration.md`).

## Conventions

- The coordinator-owned shared contracts are `config/`, `contracts.py`, `artifacts/`, `reporting/metrics.py` and `cli.py`. If you change them, update `docs/architecture.md` and the matching area doc (`docs/protocol.md`, `docs/data-and-tabular.md`, `docs/imaging.md`, `docs/slurm.md`).
- Record what was actually run and verified, and the open limits, in `docs/acceptance.md`. Do not claim that cluster or real-data runs were verified when they weren't.
- Raw images (`*.nii*`, `*.dcm`), `*.xlsx` workbooks (except test fixtures), checkpoints and `runs/`/`data/` are git-ignored and must never be committed.
- Labels: binary Oncotype `≤18 → 0, >18 → 1`, with an optional 3-class variant `≤18 / (18,31] / >31`. `oncotype_score` is never a predictor. The allowed-predictor policy is in `data/allowed_predictors_v1.json`.
