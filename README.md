# mribreast 2.0: Duke breast-MRI Oncotype pipeline

This is a rewrite of the `mri_brest` research pipeline (reference SHA `127a260`). It covers:

- 3D MRI and **real 2D** classification (2D slices cut from NIfTI, or ready PNG/TIFF images).
- **External segmentation masks** (`roi_crop`, `mask_channel`, `foreground_only`).
- Tabular branches (clinical plus the Duke imaging-feature groups, and optional radiologist density).
- **Late fusion**: `p = α·p_image + (1−α)·p_tabular`.
- Patient-grouped nested cross-validation, with audited and resumable artifacts.

It runs on a laptop CPU and on Slurm (Helios GH200) through one Bash-readable configuration.

> Labels: Oncotype DX recurrence score `≤18 → 0, >18 → 1` (binary). The historical 3-class variant is `≤18 / (18,31] / >31`. These cut-offs are definitions of this research project, not clinical guidance.

## Install

```bash
python -m pip install -e ".[test]"            # compute stage (train/validate/summarize)
python -m pip install -e ".[prepare]"         # optional: NBIA download, DICOM→NIfTI, MAMA-MIA
```

Python ≥3.10. The verified versions are listed in `docs/reference-test-results.md`. On Helios, NumPy and PyTorch come from the `ML-bundle` module (see `docs/slurm.md`).

## Run

All runs go through the same controller. With `--local`, the controller runs every task on this machine with exactly the argv a Slurm array task would receive.

```bash
bash run_experiment.sh --train     [--config experiment.conf] [--dry-run] [--dependency afterok:ID] [--local] [--show-config]
bash run_experiment.sh --validate  ...     # aliases: --training, --validation
bash run_experiment.sh --summarize ...
bash run_experiment.sh --prepare   ...     # cohort audit + cache warm-up, no training
```

The Python entry point is `python -m mribreast {prepare,train,validate,summarize,make-synthetic,import-legacy,prepare-data,fields}`. Each stage accepts `--<key> <value>` for every key in `src/mribreast/config/fields.tsv`. That file is the single field registry, shared by Bash and Python.

### CPU acceptance runs on synthetic data

```bash
python -m mribreast make-synthetic --out runs/synthetic/data
# 3D image + tabular + fusion (legacy profile, annotation boxes, LASSO; xgboost|mlp × α∈{0,0.5,1})
bash run_experiment.sh --local --train     --config configs/synthetic_cpu_3d.conf
bash run_experiment.sh --local --validate  --config configs/synthetic_cpu_3d.conf
bash run_experiment.sh --local --summarize --config configs/synthetic_cpu_3d.conf
# 2D slices selected by mask + external masks (roi_crop|mask_channel) + tabular + fusion
bash run_experiment.sh --local --train     --config configs/synthetic_cpu_2d.conf
bash run_experiment.sh --local --validate  --config configs/synthetic_cpu_2d.conf
bash run_experiment.sh --local --summarize --config configs/synthetic_cpu_2d.conf
python -m pytest -q                           # unit, parity (set MRI_BREAST_LEGACY_REPO), bash controller tests
```

### Duke on Helios

1. Edit `PATHS` in `experiment.conf`. You need the clinical workbook, the NBIA digest (not in Git), `Imaging_Features.xlsx`, `Annotation_Boxes.xlsx`, and a NIfTI directory named `<SeriesInstanceUID>.nii.gz`.
2. Run `bash run_experiment.sh --train --dry-run` and then `--train`.
3. Run `--validate --dependency afterok:<train id>` and `--summarize --dependency afterok:<validate id>`.

The commands for a short smoke run are in `docs/slurm.md`.

## Layout

| Path | Contents |
| --- | --- |
| `src/mribreast/config` | `fields.tsv` registry, typed parsing, combination rules, `train_id` / `eval_id` |
| `src/mribreast/contracts.py` | typed columns, label definition, geometry, mask record, Protocols (`CohortSource`, `MaskProvider`, `BranchTrainer`, `FittedPredictor`) |
| `src/mribreast/data` | Duke readers, cohort profiles, imaging features, density adapter, allowed-predictor policy |
| `src/mribreast/tabular` | preprocessing, LASSO, XGBoost/MLP, `TabularBranch` |
| `src/mribreast/imaging` | geometry, NIfTI loading, subtraction, locked atomic cache, annotation boxes, 2D slicing, masks |
| `src/mribreast/models` | 3D/2D networks, Lightning classifier, `ImageBranch` |
| `src/mribreast/protocol` | split plan, thresholds, fusion, resumable training, validation |
| `src/mribreast/reporting` | metrics, validation tables/plots, summary ranking |
| `src/mribreast/legacy` | historical report importer/exporter, legacy checkpoint names |
| `src/mribreast/prepare` | explicit data preparation (NBIA/TCIA, DICOM→NIfTI, MAMA-MIA) |
| `run_experiment.{sh,slurm}`, `experiment.conf`, `configs/` | Bash controller and example configurations |

## Documentation

- `docs/architecture.md`: module boundaries, lifecycle, identity rules, schemas, artifact versioning.
- `docs/legacy-parity.md`: every legacy function/option mapped to its new key, its semantics and the test that checks it.
- `docs/migration.md`: mapping of old commands, paths, run ids, artifacts and checkpoints, and which runs need retraining.
- `docs/protocol.md`, `docs/data-and-tabular.md`, `docs/imaging.md`, `docs/slurm.md`: reference for each area.
- `docs/reference-test-results.md`: results from running the reference repository's own tests.
- `docs/acceptance.md`: what was actually run, the results, and the open verification limits.

Raw images, identifying metadata, checkpoints and run outputs are git-ignored and must never be committed.
