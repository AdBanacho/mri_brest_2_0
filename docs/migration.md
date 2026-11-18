# Migration from `mri_brest` (legacy, SHA 127a2604) to `mribreast` 2.0

The parity table is `docs/legacy-parity.md`. The protocol details are in `docs/protocol.md` §8 and
`docs/data-and-tabular.md` §7.

## 1. Commands

| Legacy | New |
|---|---|
| `bash run_experiment.sh --train [--config F] [--dry-run] [--dependency afterok:ID] [--task-index N]` | **unchanged** (aliases `--training` and `--validation` kept). The Python command becomes `python -m mribreast train --<key> <value> …` |
| `bash run_experiment.sh --validate …` | unchanged → `python -m mribreast validate …` |
| `bash run_experiment.sh --summarize …` | unchanged → `python -m mribreast summarize …` |
| `bash ConfigurableImagingFeaturesFusionBinary.sh`, `submit_configurable_imaging_features_fusion.sh` / `validate_…` / `summarize_….sh` | `bash run_experiment.sh --train / --validate / --summarize` |
| `python -m mriBreastDuke.configurable_imaging_features_fusion_workflow --mri_model … --feature_groups clinical kinetic` | `python -m mribreast train --mri_model … --feature_groups clinical,kinetic` (the space-separated form is still accepted) |
| `python -m mriBreastDuke.validate_configurable_imaging_features_fusion … --fusion_alpha 0.5 --output_dir D` | `python -m mribreast validate <same train keys> --fusion_alpha 0.5 --validation_root D` |
| `python -m mriBreastDuke.summarize_configurable_imaging_features_fusion --input-dir D --output-dir S --expected-folds 5 --top-n 20 [--include-incomplete]` | `python -m mribreast summarize --validation_root D --summary_root S --expected_folds 5 --top_n 20 --include_incomplete false` |
| `sbatch denseBinary.slurm` (historical image-only grid, `trainingWorkflow --model 1 --is_binary_classification True`) | `experiment.conf` with `TRAIN[branches]=image`, `TRAIN[selection_split]=outer` (reproduces the optimistic historical selection; flagged in the report) **or**, recommended, `selection_split=shared`. Grid: `TRAIN_GRID[lr]='1e-3\|1e-4\|1e-5' [sensitivity_lambda]='0.3\|0.7\|1.0' [positive_boost]='1.0\|2.0\|3.0' [batch_size]=8`, `epoch=30` |
| `sbatch validate_dense_binary.slurm` (`validate_best_checkpoints`, top-3 checkpoints per fold) | `bash run_experiment.sh --validate`. One checkpoint per fold; the top-k variant is dropped |
| `trainingWorkflow` with no `--is_binary_classification` (3 classes) | `--label_scheme oncotype_3class_18_31 --threshold_strategy fixed_0.5` |
| `mriBreastDuke/fusion.slurm`, `run_xgboost_fusion.slurm` (modules absent at this SHA) | `TRAIN_GRID[mri_model]=resnet18 [feature_model]=xgboost`, `TRAIN[epoch]=50`, `VALIDATE_GRID[fusion_alpha]=0.5` |
| `sbatch downloadAll.slurm` (`downloadDicom.download_all`) | `python -m mribreast.prepare nbia-download --metadata_file … --cohort all --dest <dicom_root>` |
| `downloadDicom.download_oncotype()` | `… nbia-download --cohort outcome --clinical_features_file …` |
| `transfer_dcm_to_nifti()` | `python -m mribreast.prepare dicom-to-nifti --download_root <dicom_root> --image_root <image_root> [--backend sitk]` |
| `read_nifti_metadata_images()` | `python -m mribreast.prepare nifti-info --image_root <image_root> [--output info.csv]` |
| `mama_mia/downloadMamamia.py` | `SYNAPSE=<token> python -m mribreast.prepare mama-mia-download --dest <dir>` (entity `syn64871114`) |
| `mama_mia/readFromXlmx.py` | `python -m mribreast.prepare mama-mia-map --table <41597_2025_4707_MOESM2_ESM.xlsx> --metadata_file … --clinical_features_file … --output_dir <dir>` |
| (none) | `python -m mribreast import-legacy --summary_dir … [--validation_root …] [--checkpoint_root …] --output_dir …` |

Install the preparation dependencies separately with `pip install .[prepare]` (pydicom, SimpleITK,
requests; `synapseclient` is installed on demand). Train, validate and summarize never import
them.

## 2. Run identities

| Legacy | New |
|---|---|
| Controller `experiment_id = cfg<sha256(sorted TRAIN row + 7 input/cache path strings)[:12]>` | `train_id = tr-<sha256(config schema version + identity train fields + content fingerprints of input files)[:12]>`. Cache roots, output roots, workers and `experiment_id` are excluded |
| `experiment_name = ImagingFeaturesFusion_<mri>_<sub>_<groups>_<model>[_lasso-cv…][_clinical-safe-v2]_auc-cal<k>_lr<lr:.0e>_sens<s>_boost<b>_bs<n>[_annboxes][_cfg<id>]` | directory `<train_id>`. `experiment_id` is a free label. `mribreast.legacy.names.build_experiment_name(cfg)` rebuilds the legacy name |
| validation `run_id = <experiment_name>/fusion_alpha_<alpha:g>` | `run_id = <train_id>/<eval_id>`, `eval_id = ev-a<alpha>-<sha8(validate identity fields)>` |
| imported legacy run | `train_id = lg-<sha256(experiment_name)[:12]>`, `eval_id = ev-a<alpha>-legacy`; `legacy_run_id` is kept (`run_id_map.csv`) |
| historical image-only name `<Model>_lr_<lr>_sens_<s>_posboost_<b>_bs_<n>[_sub_<m>][_annboxes]` | parsed by `parse_experiment_name` (workflow `historical_image`) |

The legacy names have known collisions. `lr` is formatted with one significant digit, so `1.6e-4`
and `2e-4` both give `lr2e-04`. `seed`, `target_size`, `epoch`, `xgb_*`/`mlp_*` and
`include_sensitive` change the name only through `cfg<id>`, and runs from before the controller have
no `cfg<id>` at all. In the new pipeline, every identity field changes `train_id`.

## 3. Paths

| Legacy | New |
|---|---|
| `constants.py` defaults under `/net/storage/...` and `/net/scratch/...` | no package defaults. `experiment.conf` `PATHS[...]`, resolved relative to the config file |
| `features/*.xlsx` next to the code | any location (`clinical_features_file`, `metadata_file`, `imaging_features_file`, `annotation_boxes_file`, `density_file`) |
| `prepared_root/<sha(abspath(image_root))[:12]>/size_DxHxW/<uid>.nii.gz` | `prepared_root/<cache key>/…` (`imaging.cache`). The keys are incompatible, so **legacy caches are not reused**. Point `prepared_root` at a new directory; the old caches can be deleted |
| `subtraction_root/<src>/<mode>/size_…/<study>/<NN>_<a>_minus_<b>_<hash>.nii.gz` | `subtraction_root/<cache key>/…`, not reused |
| `logs_root/<experiment_name>/fold_k` (TensorBoard), `…/cross_validation_summary` | `logs_root/<train_id>/fold_k` |
| `checkpoint_root/<experiment_name>/fold_k/checkpoints/` | `checkpoint_root/<train_id>/fold_k/{image,tabular}/` |
| `validation_charts/<experiment_name>/fusion_alpha_<a>/` | `validation_root/<train_id>/<eval_id>/` |
| `validation_charts/summary/` | `summary_root/` |
| `features/<split>_volume_sizes.csv` (`SIZE_CACHE_PATH`, unused) | — |

## 4. Artifacts

### Training (per fold)

| Legacy (`<experiment>/fold_k/checkpoints/`) | New (`<train_id>/fold_k/`) |
|---|---|
| `best-epoch=EE-val_auc_roc=A-val_balanced_accuracy=B-val_sensitivity=S[-vN].ckpt` | `image/best.ckpt` (metrics in `image/image_branch.json` and `history.csv`) |
| `last.ckpt` | `image/last.ckpt` |
| `inner_calibration_split.csv` (`studyId, patientId, label, split∈{fit,calibration}, requested_inner_folds, effective_inner_folds`; train rows only) | `<train_id>/split_manifest.csv` (all folds; `fold, patient_id, study_id, label, role∈{fit,calibration\|selection,threshold,outer_eval}, requested_inner_folds, effective_inner_folds, outer_seed, inner_seed`) + `split_hash` in every manifest |
| `tabular_preprocessor.joblib` | `tabular/preprocessor.joblib` |
| `tabular_feature_selector.joblib` | `tabular/selector.joblib` |
| `xgboost_model.joblib` / `mlp_model.joblib` | `tabular/model.joblib` (model type in the fold manifest) |
| `tabular_feature_names.txt`, `tabular_selected_feature_names.txt` | `tabular/feature_names.txt`, `tabular/selected_feature_names.txt` |
| `lasso_feature_selection.csv`, `lasso_feature_importance.png` | `tabular/lasso_feature_selection.csv`, `tabular/*.png` (+ `coefficients.csv`) |
| `annotation_coverage.csv` (copied into every fold) | `<train_id>/annotation_coverage.csv` (once) |
| `<experiment>/lasso_feature_selection_by_fold.csv`, `lasso_feature_stability.csv/.png` | `<train_id>/lasso_feature_stability.csv/.png` |
| — | `train_manifest.json`, `fold_k/fold_manifest.json`, `COMPLETE` flags, `cohort.csv`, `cohort_audit.csv` |

### Validation

| Legacy (`<experiment>/fusion_alpha_<a>/`) | New (`<train_id>/<eval_id>/`) |
|---|---|
| `validation_metrics.csv` (`image_*`, **`xgboost_*`/`mlp_*`**, `fusion_*`) | `validation_metrics.csv` (same head, `tabular_*`, extra trailing columns) |
| `validation_summary.csv`, `validation_config.json` (vars(args)+experiment_name+device) | same names (the config is the resolved new configuration) + `evaluation_manifest.json` |
| `threshold_calibration_metrics.csv` (root) and `fold_k/threshold_calibration_metrics.csv` | `threshold_calibration_metrics.csv` (a `fold` column) |
| `fold_k/decision_thresholds.json` `{"fusion","image","xgboost"}` | `decision_thresholds.json` `{"folds": {"k": {"image","tabular","fusion"}}, …}` |
| `fold_k/threshold_calibration_predictions.csv`, `fold_k/fusion_validation_predictions.csv` (wide) | `predictions_calibration.csv`, `predictions_validation.csv` (long: `fold, patient_id, study_id, label, branch, split, prob_k, decision`) |
| `fold_k/<branch>/confusion_matrix*.png, roc_curve.png`, `aggregate/<branch>/…` | `roc_<branch>.png`, `confusion_<branch>.png`, `confusion_matrices.json`, `branch_comparison.csv`, `validation_report.md` |

`mribreast.legacy.exporter.export_evaluation_dir(<eval_dir>, <legacy_root>)` writes the legacy layout
from a new evaluation: column order, `xgboost_*`/`mlp_*` names, per-fold JSON/CSV, wide prediction
files, and `<experiment_name>_<train_id>/fusion_alpha_<a>`. `summarize_legacy` reproduces the legacy
summary on top of that layout. Legacy run names are about 150 characters, so on Windows the
exporter uses extended-length paths (`\\?\`).

### Summary

All five files keep their names and legacy column prefix (`all_validation_metrics.csv`,
`configuration_summary.csv`, `ranked_configurations.csv`, `best_configuration.json`,
`validation_report.md`). The new summarizer also requires `COMPLETE` and ranks on the primary branch.
For fusion runs this is the same ranking as legacy.

## 5. Old checkpoints: supported formats and migration conditions

Supported filename formats. Ranking is identical to legacy `checkpoint_selection.checkpoint_score`
(`mribreast.legacy.checkpoints`):

| Era | Pattern | Selected by | Importable? |
|---|---|---|---|
| AUC (current legacy fusion) | `best-epoch=EE-val_auc_roc=…-val_balanced_accuracy=…-val_sensitivity=…[-vN].ckpt` | ROC-AUC on the inner **calibration** split | yes, if the conditions below hold |
| balanced-accuracy era | `best-epoch=EE-val_balanced_accuracy=…-val_sensitivity=…[-val_auc_roc=…][-vN].ckpt` | balanced accuracy, on the **outer** fold (pre-inner-calibration code) | no, retrain |
| sensitivity era | `best-epoch=EE-val_sensitivity=…[-val_auc_roc=…][-vN].ckpt` | sensitivity, on the **outer** fold (pre-inner-calibration code) | no, retrain |
| other `best-*.ckpt` | — | unknown | no |

The weights of an AUC-era checkpoint load into the new network with
`mribreast.legacy.checkpoints.load_legacy_network_state(path)`. The Lightning `state_dict` keys are
`model.<param>`, the same prefix that `models.image_branch` strips, and the `class_weights` buffer is
dropped. Legacy files need `torch.load(weights_only=False)`, so load them only from a trusted root.

A legacy fold can be re-used **without retraining** only if every condition holds. Otherwise retrain:

1. The run is a fusion run whose name contains `auc-cal<k>` and whose best checkpoint is AUC-era
   (`inventory_fold(...).checkpoint_format == "auc"`).
2. `inner_calibration_split.csv` exists in every fold and has `studyId, patientId, split,
   requested_inner_folds, effective_inner_folds`. Without it, run membership cannot be verified, so
   the run gets **`split_metadata_missing` → retrain**.
3. The cohort can be rebuilt from the *same* clinical workbook and NBIA digest (the digest is not
   tracked). Rebuilding the new `split_manifest.csv` needs the outer folds, which come from
   `StratifiedGroupKFold(seed)` over the cohort row order. They are bit-identical when the inputs
   and `seed` are unchanged (`tests/parity/test_parity_protocol.py`).
4. The clinical predictor list matches the current policy, so the name has `_clinical-safe-v2`
   whenever `clinical` is used.
5. Configuration values that are not in the name (`seed`, `target_size`, `epoch`, …) are known. The
   `validation_config.json` files of the stored era lack `seed` and `target_size`, and the importer
   assumes the CLI defaults (42; 256 256 64) and records that in `notes`.
6. The tabular joblib files pickle legacy classes (`mriBreastDuke.dataLoaders.ClinicalFeaturePreprocessor`,
   `LassoFeatureSelector`). The new branch cannot load them without the legacy package on
   `sys.path`. **Refit the tabular branch** (seconds on CPU) or keep the legacy pipeline to re-score it.
7. The caches are rebuilt (§3). Legacy prepared volumes are not trusted as inputs.

In practice the image weights of a qualifying fold can be re-used for re-evaluation. Every new
headline result must come from a full new `train`.

## 6. Which historical runs must be retrained

Run `python -m mribreast import-legacy …`. `run_id_map.csv` and `import_report.md` list every run
with `requires_retraining` and its reasons:

| Reason flag | Applies to | Why |
|---|---|---|
| `pre_clinical_safe_v2_predictors` | **all 16 runs in the tracked `mriBreastDuke/validationSummary`** (and any fusion run with `clinical` but no `_clinical-safe-v2`) | the clinical branch used the old predictor list (ER/PR/HER2/subtype/stage/grade were removed in v2) |
| `no_inner_calibration` | fusion runs from before `auc-cal<k>` | the outer fold selected checkpoints and thresholds, as `AUC_THRESHOLD_TUNING.md` says ("This change requires retraining") |
| `image_only_outer_selection` | every historical image-only run (`denseBinary`, `trainingWorkflow`) | checkpoint chosen on the reported outer fold; `StratifiedKFold` without patient groups; no split file written (`split_metadata=none_written`) |
| `split_metadata_missing` | any run whose `fold_k/checkpoints/inner_calibration_split.csv` is missing (checked with `--checkpoint_root`) | membership cannot be verified or converted |
| `checkpoint_format_balanced_accuracy` / `_sensitivity` / `_unknown` | runs whose best checkpoints are from older eras | selection metric and data differ |

The stored `validationSummary` (16 configurations, resnet18 / densenet121 × clinical /
clinical+imaging features × xgboost × LASSO × sens 0.05 / 0.3 × boost 1 / 2, alpha 0.5) also predates
the Slurm controller (no `cfg<id>`). It does not match the current `experiment.conf` grid. It is
context only: its numbers must not be reported as results of the new pipeline.

## 7. Protocol differences: legacy profile (`profile=duke_legacy_3d`) vs general

| # | Legacy / `duke_legacy_3d` | `general` | Test |
|---|---|---|---|
| 1 | study = first 5 metadata rows (file order), ≥2 unique series | explicit phase order parsed from metadata; missing pre/post interpretation reported | tests/unit/test_data_duke.py (`test_general_profile_*`), tests/unit/test_data_legacy_parity.py |
| 2 | one study per patient (Duke) | all studies, slices and variants grouped by `patient_id` | tests/unit/test_protocol_splits.py::test_plan_is_deterministic_and_patient_disjoint_with_multi_study_patients |
| 3 | calibration both selects the checkpoint and tunes thresholds (`shared`) | `separate` selection and threshold patients available | tests/unit/test_protocol_train_validate.py::test_selection_split_modes_feed_branches_and_flags |
| 4 | historical image-only: outer fold selects the checkpoint (`outer`) | not allowed outside the legacy profile | tests/unit/test_config_schema.py |
| 5 | LASSO CV on features preprocessed once (`selector_preprocessing=outer`) | `fold_local` refits preprocessing in each selector subfold | tests/unit/test_tabular_branch.py::test_fold_local_preprocessing_differs_from_outer |
| 6 | MLP early stopping on random 15% rows (`mlp_early_stopping=internal`) | `grouped` (patient-grouped holdout) or `off` | tests/unit/test_tabular_branch.py::test_mlp_grouped_early_stopping_respects_groups |
| 7 | split re-derived at validation and compared with the CSV | frozen `split_manifest.csv` + hash; resume; `ArtifactMismatch` | tests/unit/test_protocol_train_validate.py |
| 8 | training loader not shuffled; EarlyStopping commented out | **same in both profiles**: training order is the fit-row order (not shuffled) and `early_stopping=false` by default (opt-in). Only change: a trailing training batch of exactly one sample is dropped, because BatchNorm cannot train on it (legacy crashed there) | docs/legacy-parity.md T36/T40, tests/e2e/test_modes_smoke.py |
| 9 | caches keyed by image-root path + size | content and parameter keyed, with schema version | tests/unit/test_imaging_cache.py |
| 10 | `fcn` crashes | working C-channel FCN | tests/unit/test_models_networks.py |
| 11 | study-level metrics only; 3-class without specificity | `report_unit=patient`; macro specificity | tests/unit/test_protocol_train_validate.py |
| 12 | claims about generalisation of the selected configuration | the report states how the configuration was chosen and that outer folds were reused for ranking. A generalisation claim needs an independent set or nested CV | docs/protocol.md |
| 13 | every study must yield a prediction | studies a branch cannot score (mask policy `drop`) are excluded from **all** branches and fusion; listed in `excluded_studies.csv`; denominators in `validation_metrics.csv` | tests/unit/test_protocol_train_validate.py::test_studies_without_prediction_are_excluded_from_all_branches_and_audited |
| 14 | fusion renormalises at every alpha | alpha 0/1 return the single branch unchanged (differs by ≤1e-16 from legacy) so the extremes reproduce single-branch ROC-AUC exactly | tests/unit/test_protocol_fusion.py |

## 8. Assumptions that still need data or a cluster

* The NBIA digest (`Duke-Breast-Cancer-MRI_v2_20220609-nbia-digest.xlsx`) is not tracked. Its row
  order defines the legacy phase selection.
* The annotation-box convention (0-based inclusive, `(column, row, slice)` on SimpleITK axes) has
  never been checked against DICOM overlays.
* DICOM→NIfTI has been validated only on synthetic series (including the SimpleITK cross-check).
  Real Duke series (multi-frame or enhanced MR, if any) have not been tried.
* TCIA and Synapse downloads have been exercised only against local fakes.
* Helios (GH200, `ML-bundle`, `flock`) has not been exercised. The legacy controller test passed
  on Linux bash only with an `rg` substitute.
