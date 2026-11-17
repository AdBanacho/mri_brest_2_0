# Architecture — `mribreast` 2.0

Status: **contract document**. Modules, column names, ids and artifact formats
defined here are binding for every implementation area. Changes go through the
coordinator (owner of `config/`, `contracts.py`, `artifacts/`, `reporting/metrics.py`, `cli.py`).

## 1. Module boundaries

| Package | Responsibility | Depends on |
| --- | --- | --- |
| `mribreast.config` | `fields.tsv` (single field registry shared with Bash), typed parsing, combination rules, `train_id`/`eval_id` | — |
| `mribreast.contracts` | Typed columns, `LabelDefinition`, `Geometry`, `ImageKey`, `MaskRecord`, Protocols: `CohortSource`, `MaskProvider`, `BranchTrainer`, `FittedPredictor` | numpy, pandas |
| `mribreast.artifacts` | Fingerprints, code/env versions, atomic files/directories, `COMPLETE` flags | stdlib |
| `mribreast.data` | Duke sources (clinical workbook, NBIA metadata digest, imaging features, density), cohort building (`duke_legacy_3d`, `general`), label encoding, allowed-predictor policy, 2D image manifest reader | contracts |
| `mribreast.tabular` | Preprocessor (median+standardise / mode+one-hot ignore-unknown), LASSO selector, XGBoost / sklearn MLP factories, `TabularBranch` (`BranchTrainer`) | data, contracts |
| `mribreast.imaging` | Geometry, NIfTI loading, resize/pad, intensity normalisation, subtraction, deterministic cache (locks + atomic publish), annotation boxes, 2D slicing, joint image–mask transforms, datasets | contracts, artifacts |
| `mribreast.imaging.masks` | `MaskProvider` implementations (local files 2D/NIfTI, synthetic), mask policies (`none/roi_crop/mask_channel/foreground_only`), audits, preview | contracts |
| `mribreast.models` | 3D/2D networks (`fcn`, `densenet121`, `resnet10`, `resnet18`), Lightning classifier (Adam/AdamW, weighted CE + soft sensitivity), `ImageBranch` (`BranchTrainer`) with study-level aggregation | imaging, contracts |
| `mribreast.protocol` | Split plan (outer `StratifiedGroupKFold`, inner calibration split, roles), threshold tuning, fusion, training loop over folds (resume), validation (load artifacts, verify, predict, thresholds, metrics) | contracts, artifacts, reporting.metrics — **never** imports a concrete model |
| `mribreast.reporting` | Metric definitions, validation tables/plots, summarize + ranking, `validation_report.md` | contracts |
| `mribreast.legacy` | Importer/exporter for historical CSV/JSON, legacy checkpoint filename parsing, run-id mapping | reporting |
| `mribreast.prepare` | Explicit data-preparation stage: NBIA/TCIA download, DICOM→NIfTI (UID + geometry verified), MAMA-MIA mapping import. Own optional deps (`.[prepare]`), local fakes in tests | contracts |
| `mribreast.cli` | Composition root: builds sources, providers, branch trainers from the resolved config and calls protocol functions. `python -m mribreast {prepare,train,validate,summarize,import-legacy,make-synthetic,fields}` | everything |
| `run_experiment.sh/.slurm` | Bash-only controller: reads `experiment.conf` + `fields.tsv`, expands grid, submits arrays, frozen snapshots, compute-node env | `fields.tsv` |

No module performs network access or reads data at import time. Infrastructure
(paths, providers, models) is constructed only in `cli.py`.

## 2. Experiment lifecycle

```
experiment.conf ──(controller: grid row i)──► python -m mribreast <stage> --key value ...
                                               │
 prepare : cohort audit, cache warm-up (deterministic), mask audit  → <checkpoint_root>/_prepare/<train_id>/
 train   : resolve cfg → train_id → cohort → split plan (frozen) → per fold: fit branches → publish fold_k/COMPLETE
 validate: same train args + validate args → load train_manifest & verify → per fold: predict selection/threshold/
           outer_eval, thresholds per branch on threshold split, fusion(alpha) → metrics → publish eval dir
 summarize: scan <validation_root>/*/*/COMPLETE → aggregate, rank, report
```

* **Resume:** `train` skips folds whose `fold_k/COMPLETE` exists *and* whose
  `fold_manifest.json` matches the current `train_id` and split hash; an
  incomplete `fold_k` is rebuilt (atomic directory publish).
* **Rejection:** `validate` raises `ArtifactMismatch` when `train_manifest.json`
  differs from the recomputed `train_id`, split hash, label definition or data
  fingerprint.

## 3. Identity rules

* `train_id = "tr-" + sha256(canonical JSON(train_identity))[:12]` where
  `train_identity` = config schema version + every `train` field with
  `identity=1` (canonical string form) + content fingerprints of every `path`
  field with `identity=1`. Parallelism (`num_workers`, `*_n_jobs`), output
  roots, cache roots and `experiment_id` are **excluded**.
* `eval_id = "ev-a<alpha>-" + sha256(validate identity fields)[:8]`; one training
  run has many evaluations (e.g. per `fusion_alpha`) without retraining.
* `run_id` used in summaries = `<train_id>/<eval_id>` (legacy:
  `<experiment_name>/fusion_alpha_<a>`; mapping in `docs/migration.md`).
* The Slurm controller does **not** compute ids; Python is authoritative.

## 4. Data schemas

Cohort (`cohort.csv`, one row per study): `patient_id, study_id, label,
series_uids` (`|`-joined, ordered phases) + tabular columns. `oncotype_score`
is used only for label and eligibility and is never written as a predictor.

Split manifest (`split_manifest.csv`): `fold, patient_id, study_id, label,
role` with `role ∈ {fit, calibration, selection, threshold, outer_eval}` and
metadata columns `requested_inner_folds, effective_inner_folds, outer_seed,
inner_seed`. Its sha256 (`split_hash`) is stored in manifests.

* `selection_split=shared` (legacy fusion): `fit` + `calibration`; calibration
  selects the checkpoint (ROC-AUC) and tunes thresholds.
* `separate`: calibration patients are split once more (grouped, stratified,
  `seed+fold+1000`) into `selection` and `threshold`.
* `outer` (historical image-only): `fit` = whole outer-train, the outer fold
  selects the checkpoint **and** is evaluated (optimistic; documented in
  migration).

Image key: `(patient_id, study_id, series_id, slice_id)`.

Predictions (`predictions_<split>.csv`): `fold, patient_id, study_id, label,
branch, split, prob_0..prob_{C-1}, decision`.

## 5. Artifact layout and versioning

```
<checkpoint_root>/<train_id>/
  train_manifest.json   {schema_version, train_id, code_version, environment, resolved_config,
                         label_definition, input_fingerprints, cohort_fingerprint, split_hash,
                         branches, num_classes, created_at}
  cohort.csv  cohort_audit.csv  split_manifest.csv  annotation_coverage.csv (if boxes)  mask_audit.csv (if masks)
  fold_<k>/fold_manifest.json   {schema_version, train_id, fold, split_hash, roles counts, branch artifacts}
  fold_<k>/image/{best.ckpt, last.ckpt, image_branch.json, history.csv}
  fold_<k>/tabular/{preprocessor.joblib, selector.joblib, model.joblib, feature_names.txt,
                    selected_feature_names.txt, coefficients.csv, lasso_feature_selection.csv, *.png}
  fold_<k>/COMPLETE
  lasso_feature_stability.csv / .png
  COMPLETE
<validation_root>/<train_id>/<eval_id>/
  evaluation_manifest.json  validation_config.json  validation_metrics.csv  validation_summary.csv
  decision_thresholds.json  threshold_calibration_metrics.csv
  predictions_calibration.csv  predictions_validation.csv  branch_comparison.csv
  confusion_matrices.json  roc_<branch>.png  confusion_<branch>.png  validation_report.md  COMPLETE
<summary_root>/ all_validation_metrics.csv configuration_summary.csv ranked_configurations.csv
                best_configuration.json validation_report.md
```

Every manifest carries `schema_version` (`MANIFEST_SCHEMA_VERSION`) and
`config_schema_version`. Readers reject unknown major versions.

Metric column names keep the legacy pattern `<branch>_<metric>` with branch ∈
`image, tabular, fusion` and metric ∈ `accuracy, balanced_accuracy,
sensitivity, specificity, auc_roc, decision_threshold` (legacy additionally
duplicated the tabular branch under `xgboost_*`/`mlp_*`; the exporter restores
this).

## 6. Caches

Key = sha256 of `{cache_schema_version, source series UIDs + file fingerprints
(size+mtime+sha of header), target_size, subtraction_mode, normalisation,
resize backend}`. Entries are written to a temp file on the same filesystem and
published with `os.replace`, under a shared/exclusive file lock (readers shared,
writers exclusive). A corrupt entry (load failure / checksum mismatch) is rebuilt
from the read-only sources. Augmentations are never cached.
