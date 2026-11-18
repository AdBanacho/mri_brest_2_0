# Legacy parity table

The reference is `C:\code\mri_brest` at `127a26046db20e5e506f44ce6d4c1946d34c6416`. Test runs and
environment are recorded in `docs/reference-test-results.md`.

New keys come from `src/mribreast/config/fields.tsv`. When the new key is the same as the legacy
key, the table writes "=".

**Status**

* **verified**: a test runs the new code, either against the legacy function itself (parity tests,
  `tests/parity/`) or against a documented legacy rule.
* **historical**: the legacy feature is obsolete, broken or dead. The table records how it is mapped
  or why it is dropped.
* **extended**: the behaviour is kept, and the new pipeline adds options or checks.
* **needs-data/cluster**: correctness can only be confirmed with real Duke data, real DICOM or
  NIfTI files, or Helios.

"Test" names the test file (and test name, where useful) that accepts the row. Names that do not
exist yet are marked *(planned)*; the coordinator reconciles them.

## 1. Call graph traced from `run_experiment.sh` and `run_experiment.slurm`

```
run_experiment.sh  ──exec bash──► run_experiment.slurm  (login node: source experiment.conf, validate, plan grid,
   ConfigurableImagingFeaturesFusionBinary.sh / submit_*.sh  = --train      cleanup, frozen snapshot, sbatch itself)
   validate_configurable_*.sh = --validate, summarize_*.sh = --summarize
        │ compute node: module load, PYTHONNOUSERSITE=1, optional venv bootstrap (flock), exec:
        ├─ --train     python -m mriBreastDuke.configurable_imaging_features_fusion_workflow
        │                 prepare_studies_and_features → dataLoaders.read_from_xlmx.get_oncotype_clinical_predictors_as_study_df
        │                   (+ precomputed_imaging_features.merge_precomputed_imaging_features)
        │                 build_experiment_name → n_fold_cv_run.run_5fold_cv
        │                   StratifiedGroupKFold → threshold_tuning.make_inner_calibration_split(seed+fold)
        │                   ClinicalFeaturePreprocessor → LassoFeatureSelector → XGBClassifier|MLPClassifier (joblib)
        │                   NiftiDataModule/NiftiDataset(nifti_cache, subtraction, annotation_boxes) + pad_collate
        │                   classificators.NiftiClassifier(MONAI net | Simple3DFCN) + Lightning ModelCheckpoint(val_auc_roc)
        ├─ --validate  python -m mriBreastDuke.validate_configurable_imaging_features_fusion
        │                 same cohort/folds → checkpoint_selection.rank_checkpoints → threshold_tuning.select_balanced_accuracy_threshold
        │                 classificators.xgboost_fusion.{aligned_predict_proba,fuse_probabilities,probability_metrics,save_fusion_predictions}
        └─ --summarize python -m mriBreastDuke.summarize_configurable_imaging_features_fusion
Historical/stand-alone (not reachable from the controller):
   denseBinary.sh/.slurm → mriBreastDuke.trainingWorkflow (image-only, n_fold_cv_run without inner split)
   validate_dense_binary.sh/.slurm → mriBreastDuke.validate_best_checkpoints
   downloadAll.sh/.slurm → mriBreastDuke.dataDownloaders.downloadDicom (download_all)
   mriBreastDuke/fusion.slurm → mriBreastDuke.fusionTrainingWorkflow            ← module ABSENT at 127a2604 (last in history: 7941ebd/6495d99)
   mriBreastDuke/run_xgboost_fusion.slurm → mriBreastDuke.xgboost_fusion_training_workflow ← module ABSENT
   No caller: dataLoaders/transfer_dcm_to_nifti.py, dataLoaders/read_from_dicom.py (runs at import), load_or_compute_sizes.py,
              BucketBySizeSampler.py, BucketByExactSizeBatchSampler.py, mama_mia/*.py (run at import)
```

## 2. Controller (`run_experiment.slurm`, `experiment.conf`)

| # | Legacy file:symbol | Legacy key (default) | New key | Semantic difference | Test | Status |
|---|---|---|---|---|---|---|
| C1 | run_experiment.slurm argv | `--train\|--training` | = | — | tests/slurm/test_argv.sh | verified |
| C2 | " | `--validate\|--validation` | = | — | tests/slurm/test_argv.sh | verified |
| C3 | " | `--summarize` | = | — | tests/slurm/test_argv.sh | verified |
| C4 | " | `--config FILE` (`experiment.conf` next to the script) | = | — | tests/slurm/test_argv.sh | verified |
| C5 | " | `--dry-run` | = | no side effects in either version | tests/slurm/test_submit.sh | verified |
| C6 | " | `--dependency SPEC` | = | — | tests/slurm/test_submit.sh | verified |
| C7 | " | `--task-index N` | = | — | tests/slurm/test_grid.sh | verified |
| C8 | " | `--snapshot`, `--project-root` (internal) | = | snapshot must be named `experiment-<sha256>.conf` | tests/slurm/test_submit.sh | verified |
| C9 | " | `--help/-h` | = | — | tests/slurm/test_argv.sh | verified |
| C10 | experiment.conf | `PYTHON_BIN=python3` | = | — | tests/slurm/test_submit.sh | verified |
| C11 | " | `PYTHON_VENV_ROOT=.venvs`, `AUTO_INSTALL_PYTHON_PACKAGES=true`, `PYTHON_PACKAGES=(…)` | = | per-architecture venv, `flock`, pip constraints from the module's NumPy/PyTorch | tests/slurm/test_submit.sh | needs-data/cluster (Helios GH200) |
| C12 | " | `MODULES=(ML-bundle)` | = | loaded only on compute nodes; `PYTHONNOUSERSITE=1` before and after loading | tests/slurm/test_submit.sh | needs-data/cluster |
| C13 | " | `MAX_JOBS=0` | = | caps training rows before validation expansion | tests/slurm/test_grid.sh | verified |
| C14 | " | `MAX_CONCURRENT=4` | = | `--array=0-N%4` | tests/slurm/test_grid.sh | verified |
| C15 | " | `CLEAN_BEFORE_TRAIN/VALIDATE=false` | = | needs the explicit flag; refuses broad or overlapping targets | tests/slurm/test_submit.sh | verified |
| C16 | " | `PATHS[clinical_features_file … slurm_logs]` (12 keys) | = (fields.tsv `section=path`) + new `density_file, allowed_predictors_file, image_manifest_file, mask_manifest_file` | relative paths resolve against the config directory | tests/slurm/test_validation.sh, tests/unit/test_config_schema.py | extended |
| C17 | " | `JOB_TRAIN/VALIDATE/SUMMARIZE[job_name cpus gpus memory time account partition]` | = | — | tests/slurm/test_submit.sh | verified |
| C18 | " | `EXTRA_SBATCH_*` (only qos, constraint, reservation, nice, mail-*) | = | — | tests/slurm/test_submit.sh | verified |
| C19 | " | `TRAIN`, `TRAIN_GRID` (`\|` between choices, `,` inside feature_groups) | = | the allowed keys come from fields.tsv instead of a hard-coded `TRAIN_KEYS` | tests/slurm/test_grid.sh | extended |
| C20 | " | `VALIDATE[num_workers]=4` | `validate_num_workers` (fields.tsv `legacy=renamed:num_workers`) | rename only | tests/unit/test_config_schema.py | verified |
| C21 | " | `VALIDATE_GRID[fusion_alpha]=0.5` | `fusion_alpha` | 24/24/72 job counts kept | tests/slurm/test_grid.sh | verified |
| C22 | " | `SUMMARIZE[expected_folds=5 top_n=20 include_incomplete=false]` | = | — | tests/slurm/test_grid.sh | verified |
| C23 | run_experiment.slurm `build_command` | `experiment_id=cfg<sha256(sorted row + 7 input/cache paths)[:12]>` | `train_id = tr-<sha256(identity fields + input fingerprints)[:12]>`, `eval_id` | Python computes the identity; paths enter as content fingerprints, cache roots are excluded; `experiment_id` is now only a label | tests/unit/test_config_schema.py | extended |
| C24 | " | grid order: keys sorted, last key varies fastest; validation index = train×alphas | = | — | tests/slurm/test_grid.sh | verified |
| C25 | " | `VALIDATION_TRAIN_KEYS` (validation re-sends the training identity keys) | validate CLI accepts every `path`+`train` field | the validation run checks `train_manifest.json` (ArtifactMismatch) instead of rebuilding the name | tests/unit/test_protocol_train_validate.py::test_rejects_artifacts_from_another_configuration | extended |
| C26 | " | booleans: `true` → bare flag, `false` → omitted | `--key true\|false` (bare flag still accepted) | — | tests/unit/test_config_schema.py | verified |

## 3. Training CLI (`configurable_imaging_features_fusion_workflow.py:parse_args`, then `n_fold_cv_run.run_5fold_cv`)

| # | Legacy symbol | Legacy key (default) | New key | Semantic difference | Test | Status |
|---|---|---|---|---|---|---|
| T1 | parse_args | `--mri_model` (densenet121; fcn\|densenet121\|resnet10\|resnet18) | = | **legacy `fcn` crashes** (`Simple3DFCN` has no `in_channels`); the new FCN takes C input channels | tests/unit/test_models_networks.py | extended |
| T2 | " | `--subtraction_mode` (none) | = | channels 5/4 kept | tests/unit/test_imaging_volume.py | verified |
| T3 | " | `--feature_groups` (clinical; nargs+) | = (csv) | adds the opt-in `density` group | tests/unit/test_data_duke.py | extended |
| T4 | " | `--feature_model` (xgboost) | = | — | tests/unit/test_tabular_branch.py | verified |
| T5 | " | `--feature_selector` (none) | = | — | tests/unit/test_tabular_branch.py | verified |
| T6 | " | `--imaging_features_file` (constants path) | = (path) | no package-relative default | tests/unit/test_data_duke.py | verified |
| T7 | " | `--imaging_patient_id_column` (`Patient ID`) | = | — | tests/unit/test_data_duke.py | verified |
| T8 | " | `--allow_missing_imaging_features` (false) | = | — | tests/unit/test_data_duke.py | verified |
| T9 | " | `--include_sensitive` (false) | = | goes through the versioned allowed-predictor policy | tests/unit/test_data_duke.py (`test_policy_*`) | extended |
| T10 | " | `--use_annotation_boxes` (false; experiment.conf true) | = (+ `annotation_erase_attempts=8`) | — | tests/unit/test_imaging_annotations.py | verified (needs-data/cluster for the overlay convention) |
| T11 | " | `--annotation_boxes_file` | = | — | tests/unit/test_imaging_annotations.py | verified |
| T12 | " | `--clinical_features_file`, `--metadata_file`, `--image_root`, `--prepared_root`, `--subtraction_root`, `--logs_root`, `--checkpoint_root` | = | defaults are no longer `/net/...` constants | tests/unit/test_config_schema.py | verified |
| T13 | " | `--target_size` (256 256 64) | = | — | tests/unit/test_imaging_volume.py | verified |
| T14 | " | `--seed` (42) | = | — | tests/parity/test_parity_protocol.py | verified |
| T15 | " | `--experiment_id` ("") | = | label only, outside the identity | tests/unit/test_config_schema.py | extended |
| T16 | " | `--epoch` (30) | = | — | tests/unit/test_protocol_train_validate.py | verified |
| T17 | " | `--num_folds` (5) | = | — | tests/parity/test_parity_protocol.py::test_split_plan_matches_legacy_fold_loop | verified |
| T18 | " | `--threshold_calibration_folds` (5) | = | effective folds recorded | tests/parity/test_parity_protocol.py::test_inner_calibration_split_identical | verified |
| T19 | " | `--batch_size` (4), `--num_workers` (4) | = | `num_workers` is not part of the identity | tests/unit/test_config_schema.py | verified |
| T20 | " | `--positive_boost` (1.0) | = | tied to `positive_class` (default 1) | tests/unit/test_tabular_branch.py (`test_sample_weights_*`), tests/unit/test_models_networks.py | verified |
| T21 | " | `--sensitivity_lambda` (0.05) | = (+ `loss=ce_soft_sensitivity\|ce`) | — | tests/unit/test_models_networks.py | verified |
| T22 | " | `--lr` (1e-4) | = (+ `optimizer`, `weight_decay`, `scheduler`) | legacy uses Adam only | tests/unit/test_models_networks.py | extended |
| T23 | " | `--xgb_n_estimators 300 --xgb_max_depth 3 --xgb_learning_rate 0.03 --xgb_subsample 0.8 --xgb_colsample_bytree 0.8 --xgb_n_jobs 8` | = | `tree_method=hist`, objective and eval_metric as in legacy | tests/unit/test_tabular_branch.py | verified |
| T24 | " | `--lasso_cv_folds 5 --lasso_cs 20 --lasso_max_iter 5000 --lasso_tolerance 1e-4 --lasso_min_features 1 --lasso_n_jobs 8 --lasso_plot_top_n 30` | = (+ `selector_preprocessing=outer\|fold_local`) | legacy preprocesses before the selector's CV (`outer`) | tests/unit/test_tabular_branch.py::test_fold_local_preprocessing_differs_from_outer | extended |
| T25 | " | `--mlp_hidden_layers 128,64 --mlp_alpha 1e-4 --mlp_learning_rate 1e-3 --mlp_max_iter 500` | = | — | tests/unit/test_tabular_branch.py | verified |
| T26 | make_feature_model | hard-coded MLP `early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, relu, adam` | `mlp_early_stopping=internal\|grouped\|off` | legacy holds out random rows, not patients | tests/unit/test_tabular_branch.py::test_mlp_grouped_early_stopping_respects_groups | extended |
| T27 | " | `--xgb_extra_json/--mlp_extra_json/--mri_extra_json` ({}) | = | — | tests/unit/test_config_schema.py | verified |
| T28 | run_5fold_cv | `--trainer_extra_json` ({}; protected callbacks, logger, default_root_dir) | = (also protects `enable_checkpointing`) | — | tests/unit/test_config_schema.py | extended |
| T29 | prepare_studies_and_features | always `isBinary=True` | `label_scheme=oncotype_binary_18` | 3-class becomes selectable | tests/unit/test_data_duke.py, tests/unit/test_protocol_train_validate.py::test_three_class_flow_uses_argmax | extended |
| T30 | build_experiment_name | `ImagingFeaturesFusion_…` checkpoint/log directory name | `<train_id>`; legacy name rebuilt by `mribreast.legacy.names.build_experiment_name` | see migration §2 | tests/unit/test_legacy_names.py | verified |
| T31 | run_5fold_cv | `StratifiedGroupKFold(n, shuffle, seed)` + inner `make_inner_calibration_split(seed+fold)` | `protocol.splits.make_split_plan` (`selection_split=shared`) | bit-identical | tests/parity/test_parity_protocol.py | verified |
| T32 | " | `_validate_cv_inputs` / `_validate_fold_separation` (labels 0..C-1, ≥ num_folds groups per class, no leakage) | `splits._check_cohort`, `validate_split_plan` | — | tests/unit/test_protocol_splits.py | verified |
| T33 | " | tabular `compute_sample_weight('balanced')` × positive_boost for label 1 | `tabular.models.balanced_sample_weights` | — | tests/unit/test_tabular_branch.py | verified |
| T34 | `_compute_balanced_class_weights` | `total/(C·count)` with `weights[1]*=boost` | `models.classifier.compute_balanced_class_weights` | — | tests/unit/test_models_networks.py + tests/parity/test_parity_inputs_weights.py | verified |
| T35 | run_5fold_cv | `ModelCheckpoint(monitor=val_auc_roc, max, save_top_k=1, save_last)`, filename with auc/bal/sens | `fold_k/image/best.ckpt`, `last.ckpt` | same monitor and data (calibration split); fixed filename; metrics go to `image_branch.json`/`history.csv` | tests/unit/test_protocol_train_validate.py | extended |
| T36 | " | `EarlyStopping(patience 8, min_delta 1e-4)` declared but **commented out** | `early_stopping=false`, `early_stopping_patience=8`, `early_stopping_min_delta=1e-4` | off by default, as in legacy | tests/unit/test_models_networks.py *(planned)* | verified |
| T37 | " | `DebugBatchShapeCallback`, `_BestValidationMetrics`, TensorBoard logger `logs_root/<name>/fold_k`, CV plots in `cross_validation_summary` | logger and callbacks owned by the runner | debug shape printing dropped | — | historical |
| T38 | " | `annotation_coverage.csv` per fold | `<train_id>/annotation_coverage.csv` | written once | tests/unit/test_imaging_annotations.py | verified |
| T39 | " | `lasso_feature_selection_by_fold.csv`, `lasso_feature_stability.csv/.png` | `<train_id>/lasso_feature_stability.csv/.png` | — | tests/unit/test_tabular_branch.py | verified |
| T40 | NiftiDataModule.train_dataloader | **no `shuffle`**: training batches are in fixed cohort order | image datasets | the new loader shuffles (seeded); a legacy-order option is not provided | tests/unit/test_imaging_2d.py *(planned check)* | historical (difference recorded) |

## 4. Validation CLI (`validate_configurable_imaging_features_fusion.py`)

| # | Legacy symbol | Legacy key (default) | New key | Semantic difference | Test | Status |
|---|---|---|---|---|---|---|
| V1 | parse_args | the same train keys as T1–T14, T17–T22, T24 (no epoch, xgb_*, mlp_*, extra_json except `mri_extra_json`) | validate CLI = path+train+validate sections | the training configuration is checked against `train_manifest.json` | tests/unit/test_protocol_train_validate.py | extended |
| V2 | " | `--fusion_alpha` (0.5) | = | part of `eval_id` | tests/unit/test_protocol_fusion.py | verified |
| V3 | " | `--checkpoint_root` | = | — | — | verified |
| V4 | " | `--output_dir` (validation_charts) | `validation_root` | layout `<train_id>/<eval_id>` | tests/unit/test_protocol_train_validate.py | extended |
| V5 | " | `--num_workers` (4) | `validate_num_workers` | rename | tests/unit/test_config_schema.py | verified |
| V6 | " | `--use_annotation_boxes` (audit only at validation) | = (audit) | inference always uses the full image | tests/unit/test_imaging_annotations.py | verified |
| V7 | find_best_checkpoint / checkpoint_selection | three filename eras, `checkpoint_score` 7-tuple | `mribreast.legacy.checkpoints` (legacy import); new runs use fixed `best.ckpt` | identical ranking | tests/parity/test_parity_checkpoints.py, tests/unit/test_legacy_checkpoints.py | verified |
| V8 | _validate_calibration_manifest | rebuilds the split and compares it with `inner_calibration_split.csv` | frozen `split_manifest.csv` + `split_hash` | — | tests/unit/test_protocol_train_validate.py::test_rejects_modified_split_or_fold_artifacts | extended |
| V9 | threshold_tuning.select_balanced_accuracy_threshold | balanced accuracy → sensitivity → closeness to 0.5 → first (lowest) candidate | `protocol.thresholds` (`threshold_strategy=balanced_accuracy`; new `fixed_0.5`) | identical | tests/parity/test_parity_protocol.py::test_threshold_*, tests/unit/test_protocol_thresholds.py | verified |
| V10 | _tune_branch_thresholds | separate thresholds for image, `<model>` and fusion on calibration | per branch on the `threshold` role | — | tests/unit/test_protocol_train_validate.py | verified |
| V11 | xgboost_fusion.fuse_probabilities | `α·p_img + (1-α)·p_tab`, renormalised | `protocol.fusion.fuse_probabilities` | identical (atol 1e-15) | tests/parity/test_parity_protocol.py::test_fusion_and_metrics | verified |
| V12 | xgboost_fusion.aligned_predict_proba | maps `classes_` to 0..C-1 | `tabular.branch.aligned_predict_proba` | — | tests/unit/test_tabular_branch.py | verified |
| V13 | xgboost_fusion.probability_metrics | accuracy, balanced_accuracy, sensitivity, decision_threshold, specificity, auc_roc; 3-class macro, no specificity | `reporting.metrics` | 3-class adds macro specificity | tests/parity/test_parity_protocol.py::test_fusion_and_metrics (atol 1e-12) | extended |
| V14 | run_validation | per-fold dirs `fold_k/{decision_thresholds.json, threshold_calibration_metrics.csv, threshold_calibration_predictions.csv, fusion_validation_predictions.csv, <branch>/confusion_matrix*.png, roc_curve.png}`, `aggregate/` | flat files with a `fold` column (architecture §5); legacy layout via `mribreast.legacy.exporter.write_legacy_run` / `export_evaluation_dir` | — | tests/unit/test_legacy_exporter.py | verified |
| V15 | " | `validation_metrics.csv` (legacy column order), `validation_summary.csv` (np.std ddof=0), `validation_config.json` (vars(args)+experiment_name+device) | same names; `tabular_*` columns instead of `<model>_*` | the exporter restores the legacy names | tests/unit/test_legacy_exporter.py | verified |
| V16 | " | pooled confusion from per-fold decisions | = | — | tests/unit/test_protocol_train_validate.py::test_pooled_confusion_uses_per_fold_thresholds | verified |
| V17 | " | `num_classes != 2` → error | 3-class validation with argmax (`fixed_0.5`) | — | tests/unit/test_protocol_train_validate.py::test_three_class_flow_uses_argmax | extended |

## 5. Summary CLI (`summarize_configurable_imaging_features_fusion.py`)

| # | Legacy symbol | Legacy key (default) | New key | Semantic difference | Test | Status |
|---|---|---|---|---|---|---|
| S1 | parse_args | `--input-dir` (validation_charts) | `validation_root` | — | tests/unit/test_reporting_summarize.py | verified |
| S2 | " | `--output-dir` (`<input>/summary`) | `summary_root` | — | " | verified |
| S3 | " | `--expected-folds` (5; overridden by `num_folds` in the config) | `expected_folds` | — | " | verified |
| S4 | " | `--top-n` (20) | `top_n` | — | " | verified |
| S5 | " | `--include-incomplete` (false) | `include_incomplete` | — | tests/unit/test_reporting_summarize.py::test_include_incomplete | verified |
| S6 | collect_results / rank_configurations | rank by `fusion_{auc_roc,balanced_accuracy,sensitivity,specificity}_mean` desc, NaN last; complete = folds 1..N | `reporting.summarize` ranks on the primary branch and also needs `COMPLETE`; legacy port `mribreast.legacy.exporter.summarize_legacy` | the port's output is identical on synthetic trees; it reproduces the stored ranked order | tests/parity/test_parity_summary.py, tests/unit/test_reporting_summarize.py::test_ranking_matches_legacy_implementation | verified |
| S7 | _canonicalize_tabular_metrics | adds `tabular_*` copies of `<model>_*` | importer normalises to `tabular_*`; exporter writes `<model>_*` | — | tests/unit/test_legacy_importer.py | verified |
| S8 | outputs | `all_validation_metrics.csv, configuration_summary.csv, ranked_configurations.csv, best_configuration.json, validation_report.md, fusion_metric_comparison.png, branch_auc_comparison.png` | = (architecture §5) | — | tests/parity/test_parity_summary.py::test_stored_summary_headers_reproduced | verified |

## 6. Data, imaging, models, helpers

| # | Legacy file:symbol | Legacy key (default) | New key / API | Semantic difference | Test | Status |
|---|---|---|---|---|---|---|
| D1 | read_from_xlmx.read_useful_clinical_predictors | sheet Data, skip 3 rows, cols A,T,U,V,AB,AK,AW–BA; missing tokens; age = -days/365.25 | `data.duke.read_clinical_workbook` | — | tests/unit/test_data_legacy_parity.py | verified |
| D2 | _categorize_oncotype_score | ≤18 / >18; ≤18 / (18,31] / >31 | `contracts.label_definition` (`label_scheme`) | — | tests/unit/test_data_duke.py | verified |
| D3 | get_oncotype_score_for_series_as_studyId_and_label_df | ≥2 unique series per study; first `MAX_SERIES_PER_STUDY=5` rows in metadata order | `profile=duke_legacy_3d`, `phase_count=5`, `phase_order=metadata` | general profile: explicit phase order and checks | tests/unit/test_data_legacy_parity.py | verified (needs-data: the real NBIA digest is not tracked) |
| D4 | get_oncotype_clinical_predictors_as_study_df | inner merge, many_to_one | `DukeCohortSource.load` (+ audit) | — | tests/unit/test_data_legacy_parity.py | verified |
| D5 | get_unique_studies / get_unique_study_instance_for_oncotype_score_as_not_na | study sets for download | `prepare.nbia.select_study_uids(cohort=all\|outcome)` | metadata order kept | tests/unit/test_prepare_nbia.py | verified |
| D6 | get_oncotype_score_for_series_as_serie_and_label_df | series-level sampling (num_of_samples, max_per_class) | — | dead (no caller) | — | historical |
| D7 | precomputed_imaging_features | groups kinetic/morphology/heterogeneity; ID column; complete-match check | `data.imaging_features` | — | tests/unit/test_data_duke.py | verified |
| D8 | (not read) Breast_Radiologist_Density_Assessments.xlsx | — | `density_file`, group `density` (off) | new | tests/unit/test_data_duke.py (`test_density_*`) | extended |
| D9 | clinical_preprocessing.ClinicalFeaturePreprocessor | median+scale / mode+one-hot(ignore) | `tabular.preprocessing.TabularPreprocessor` | — | tests/unit/test_tabular_branch.py | verified |
| D10 | lasso_feature_selection.LassoFeatureSelector | L1-logistic CV over Cs, sample weights, groups, min features | `tabular.lasso.LassoFeatureSelector` | — | tests/unit/test_tabular_branch.py | verified |
| D11 | NiftiDataset._monai_preprocess_and_save | NormalizeIntensity(channel_wise) → Resize trilinear (align_corners=False) | `imaging.volume` | — | tests/unit/test_imaging_volume.py | verified |
| D12 | NiftiDataset subtraction | resize the raw phases, then difference, then z-score (population std + 1e-5); shape and affine check | `imaging.volume` | — | tests/unit/test_imaging_volume.py | verified |
| D13 | NiftiDataset cache key | `sha(abspath(image_root))[:12]/size_…/<uid>.nii.gz`, subtraction by pair hash | `imaging.cache` (key includes schema version, source fingerprints and every transform parameter) | legacy key ignores normalisation and backend, and changes when the root path changes | tests/unit/test_imaging_cache.py | extended |
| D14 | nifti_cache.load_or_create_nifti_cache / atomic_save_nifti | exclusive lock for readers too, temp file + `os.replace`, corrupt rebuild | `imaging.cache`, `imaging.locking` (shared readers, exclusive writers) | — | tests/unit/test_imaging_cache.py, tests/unit/test_imaging_cache_workers.py | extended |
| D15 | annotation_boxes.load_annotation_boxes / annotation_coverage / resize_box + erasing in `__getitem__` | 0-based inclusive → half-open; one study per patient; identical grids; ≤8 draws of `max(1,dim//4)` | `imaging.annotations` | — | tests/unit/test_imaging_annotations.py | verified (needs-data: overlay convention) |
| D16 | pad_collate | pads series to `max_series`, spatial high-side zero padding | imaging datasets (fixed target size) | — | tests/unit/test_imaging_volume.py *(planned)* | verified |
| D17 | BucketBySizeSampler, BucketByExactSizeBatchSampler | size-bucket samplers | — | dead: depends on `dataset.sizes`, which is commented out | — | historical |
| D18 | load_or_compute_sizes + constants.SIZE_CACHE_PATH | shape cache `features/<split>_volume_sizes.csv` | — | dead (call commented out; `SIZE_CACHE_PATH` computed but unused) | — | historical |
| D19 | subtraction.get_input_channels / build_subtraction_pairs | 5 / 4 channels; (post_i − pre), (s_{i+1} − s_i) | `imaging.volume` | — | tests/unit/test_imaging_volume.py | verified |
| M1 | NiftiClassifier | weighted CE + λ·soft sensitivity (binary: class-1 recall; multiclass: macro), Adam(lr), torchmetrics val AUC (macro), class-1 recall, macro recall | `models.classifier.ImageClassifier` | — | tests/unit/test_models_networks.py | verified |
| M2 | Simple3DFCN | 1 input channel, no `in_channels` → broken in every workflow | `models.networks.SimpleFCN` (C channels, 2D/3D) | fixed | tests/unit/test_models_networks.py | historical (legacy) / extended (new) |
| M3 | MONAI DenseNet121 / resnet10 / resnet18 (spatial_dims=3) | — | `models.networks` (2D/3D) | state-dict prefix `model.` matches legacy (see migration §5) | tests/unit/test_models_networks.py | verified |

## 7. Historical entry points and preparation scripts

| # | Legacy file:symbol | Legacy key (default) | New key / command | Semantic difference | Test | Status |
|---|---|---|---|---|---|---|
| H1 | trainingWorkflow.py | `--model` (0; 0 FCN, 1 DenseNet, 2 ResNet18, 3 ResNet10) | `mri_model` | `--model 0` crashes (M2) | — | historical |
| H2 | " | `--epoch` (50) | `epoch` | — | — | historical |
| H3 | " | `--is_binary_classification` (False, **`type=bool`**: any value → True) | `label_scheme` | omitting the flag gives 3 classes, any value gives binary | tests/unit/test_protocol_train_validate.py::test_three_class_flow_uses_argmax | historical |
| H4 | " | `--lr 1e-3 --batch_size 8 --num_workers 2 --sensitivity_lambda 0.3 --positive_boost 1.0` | = | other defaults | — | historical |
| H5 | " | `--use_annotation_boxes`, `--annotation_boxes_file`, `--subtraction_mode` | = | — | — | historical |
| H6 | " → run_5fold_cv(group_column=None, inner_calibration_folds=None, num_folds=5) | **StratifiedKFold without patient groups**; checkpoint chosen on the **outer** fold; argmax | `branches=image, selection_split=outer, profile=duke_legacy_3d` | the new outer mode groups by patient and is flagged as optimistic | tests/unit/test_protocol_train_validate.py::test_selection_split_modes_feed_branches_and_flags[outer] | extended |
| H7 | trainingWorkflow name | `<Model>_lr_<lr:.0e>_sens_<s>_posboost_<b>_bs_<n>[_sub_<m>][_annboxes]` | parsed by `legacy.names.parse_experiment_name` | — | tests/unit/test_legacy_names.py | verified |
| H8 | validate_best_checkpoints.py | `--model`, `--lr/--sensitivity_lambda/--positive_boost/--weight_decay` (None; name only), `--top_k_checkpoints 3`, `--num_folds 5`, `--batch_size 8`, `--num_workers 2`, `--is_binary_classification`, `--subtraction_mode`, `--use_annotation_boxes`, `--annotation_boxes_file`, `--checkpoint_root`, `--charts_dir` (validation_charts) | validate CLI | top-k validation of several checkpoints per fold is dropped (one best per fold); `target_size`/`image_root` were hard-coded; `weight_decay` appeared in the name only | tests/unit/test_legacy_checkpoints.py (`resolve_historical_fold_dir`) | historical |
| H9 | denseBinary.slurm / validate_dense_binary.slurm | grid LR {1e-3,1e-4,1e-5} × sens {0.3,0.7,1.0} × boost {1,2,3} × bs {8} = 27, but **`--array=0-27` gives 28 tasks** (task 27 repeats task 0) | expressed as `TRAIN_GRID` with `branches=image, selection_split=outer` | the off-by-one is not reproduced | — | historical |
| H10 | mriBreastDuke/fusion.slurm | `python -m mriBreastDuke.fusionTrainingWorkflow --backbone resnet18 --epoch 50 --lr 1e-4 --batch_size 4 --num_workers 8` | `mri_model=resnet18 epoch=50 …` | **module absent at 127a2604** (exists only in history, commits 7941ebd/6495d99) | `ls mriBreastDuke` (audit) | historical |
| H11 | mriBreastDuke/run_xgboost_fusion.slurm | `python -m mriBreastDuke.xgboost_fusion_training_workflow --backbone resnet18 --epoch 50 --lr 1e-4 --fusion_alpha 0.5 --xgb_* … --batch_size 4 --num_workers 8` | standard train+validate with `feature_model=xgboost` | **module absent at 127a2604** | audit | historical |
| P1 | dataDownloaders/downloadDicom.download_all / download_oncotype (tcia_utils) | collection `Duke-Breast-Cancer-MRI`, dest `DCM_PATH` | `python -m mribreast.prepare nbia-download --cohort all\|outcome --dest …` | injected client, atomic per-series publish, resume marker, manifest, zip-slip guard | tests/unit/test_prepare_nbia.py (fake client + local HTTP server) | extended (needs-data/cluster for real TCIA) |
| P2 | downloadAll.slurm | runs `download_all` on a GPU node (16 CPU, 120G) | the same command on any node | — | — | historical |
| P3 | transfer_dcm_to_nifti.transfer_dcm_to_nifti (SimpleITK) | folder name taken as the UID; output `NIFTI_PATH+uid+'.nii.gz'` (**missing path separator**); `zip(folder_list, series)` misaligns when non-directories exist | `python -m mribreast.prepare dicom-to-nifti --backend nibabel\|sitk` | UID read from every header must equal the expected UID; one study; matrix, spacing, orientation, uniform slice spacing and no tilt are checked; atomic write; sidecar JSON | tests/unit/test_prepare_dicom_to_nifti.py (incl. `test_matches_legacy_simpleitk_conversion`: identical voxels, affine within 1e-4 on an oblique series) | verified (needs-data for real Duke DICOM) |
| P4 | transfer_dcm_to_nifti.read_nifti_metadata_images | prints size, spacing, origin, direction, stats | `python -m mribreast.prepare nifti-info` | CSV listing | — | extended |
| P5 | dataLoaders/read_from_dicom.py | module-level exploratory script (counts series of the first study at import) | `prepare.dicom_to_nifti.inspect_series` | — | tests/unit/test_prepare_dicom_to_nifti.py | historical |
| P6 | mama_mia/downloadMamamia.py | `syn.get("syn64871114")` at import, token `$SYNAPSE` via dotenv | `python -m mribreast.prepare mama-mia-download` (`--entity_id`, `--token_env`) | client injected, no import-time login | tests/unit/test_prepare_mama_mia.py (fake client) | extended (needs-data: Synapse access) |
| P7 | mama_mia/readFromXlmx.read_series_uids / get_mamamia_images_for_oncotype | `features/41597_2025_4707_MOESM2_ESM.xlsx`, sheet `dataset_info`, column `tcia_series_uid`; prints the intersection count with outcome-patient series | `python -m mribreast.prepare mama-mia-map` → `mama_mia_duke_mapping.csv`, `mama_mia_summary.json` (`legacy_intersection`) | per-series mapping and coverage | tests/unit/test_prepare_mama_mia.py::test_mapping_reproduces_legacy_intersection | extended (needs-data: table not tracked) |
| P8 | urls.txt | MAMA-MIA paper and a resolution reference | documented in `mribreast.prepare.mama_mia` | — | — | historical |

## 8. `constants.py` paths

| Constant | Legacy value | New key | Status |
|---|---|---|---|
| `PROJECT_ROOT`, `DUKE_PATH`, `FEATURES_PATH` | repository / `mriBreastDuke` / `mriBreastDuke/features` | none; every path is explicit and relative to the config file | historical |
| `STORAGE_PATH` | `/net/storage/pr3/plgrid/plggsiwomipipan/aBanacho_duke_oncotype` | `experiment.conf` only (never in the package) | historical |
| `IMAGES_HELIOS_PATH` | `/net/scratch/hscra/plgrid/plgabanacho` | `experiment.conf` only | historical |
| `SEED=42` | — | `seed` | verified |
| `MAX_SERIES_PER_STUDY=5` | — | `phase_count` | verified |
| `IMAGES_PATH` (`…/images`) | unused | — | historical |
| `DCM_PATH` (`…/tciaDownload`) | download destination | `prepare nbia-download --dest` | extended |
| `NIFTI_PATH` (`…/tciaNifti`) | — | `image_root` | verified |
| `PREPARED_TO_TRAIN_PATH`, `SUBTRACTION_PATH` | scratch caches | `prepared_root`, `subtraction_root` | verified |
| `DUKE_CANCER_MRI_COLLECTION` | `Duke-Breast-Cancer-MRI` | `prepare.nbia.DUKE_COLLECTION` (`--collection`) | verified |
| `IMAGES_METADATA` | `features/Duke-Breast-Cancer-MRI_v2_20220609-nbia-digest.xlsx` (not tracked) | `metadata_file` | needs-data/cluster |
| `TARGETS_FILE_NAME`, `IMAGING_FEATURES_FILE_NAME`, `ANNOTATION_BOXES_FILE_NAME` | `features/*.xlsx` | `clinical_features_file`, `imaging_features_file`, `annotation_boxes_file` | verified |
| `SIZE_CACHE_PATH` | lambda; unused | — | historical |
| `LIGHTING_LOGS`, `CHECKPOINTS_PATH`, `VALIDATION_CHART_PATH` | `lightning_logs`, scratch `check_points`, `validation_charts` | `logs_root`, `checkpoint_root`, `validation_root` (+ `summary_root`) | verified |

## 9. Discrepancies between the legacy docs and the code (at 127a2604)

1. `AUC_THRESHOLD_TUNING.md` tells users to run `THRESHOLD_CALIBRATION_FOLDS=5 sbatch ConfigurableImagingFeaturesFusionBinary.sh`. The launcher now runs `bash run_experiment.slurm --train` and ignores that variable, and `EXPERIMENT_CONFIG.md` says to use `bash`, not `sbatch`.
2. `ANNOTATION_BOXES.md` mentions the environment variables `USE_ANNOTATION_BOXES`/`ANNOTATION_BOXES_FILE` for the Slurm scripts. The controller uses `TRAIN[use_annotation_boxes]` and `PATHS[annotation_boxes_file]`.
3. `NIFTI_CACHE.md` describes locking "for both reads and creation". The implementation uses one exclusive `FileLock`, so readers serialise; there are no shared readers.
4. `EXPERIMENT_CONFIG.md` says the shipped grid gives 24 jobs, which is true for `experiment.conf`. The tracked `validationSummary` was produced by a different 16-configuration grid, older code (no `_clinical-safe-v2`, no `cfg<id>`) and an older `validation_config.json` schema.
5. `tests/slurm_controller_test.slurm` needs `rg` and `flock`, but the docs list only `realpath`, `sha256sum`, `sort` and `mktemp`.
6. `test_clinical_predictors.py` cannot be collected (wrong module path), and the test file `test_nifti_cache.py` is POSIX-only.
7. The EarlyStopping callback is configured but commented out (T36). Training batches are not shuffled (T40).
8. `pyproject.toml` declares no dependencies. `requirementsCuda.txt` lacks `monai` and `xgboost`, `requirementsNoCuda.txt` lacks `xgboost` and is UTF-16.

## 10. Parity status (Etap 4, comparison with the reference repo)

| Area | Status | Where | Tolerance |
|---|---|---|---|
| Checkpoint ranking (three eras, `-vN`, mtime tie) | verified, identical | tests/parity/test_parity_checkpoints.py | exact |
| Summary tables, ranking order, best JSON, report text (complete, incomplete, NaN, ties, mlp/xgboost mix, empty or invalid files) | verified | tests/parity/test_parity_summary.py | rtol 1e-12, `run_id` separator normalised |
| Stored summary headers and stored ranking order | verified (headers and order only; no values reused) | tests/parity/test_parity_summary.py | exact |
| Inner calibration split, outer fold plan | verified, identical | tests/parity/test_parity_protocol.py | exact |
| Threshold selection (incl. ties) | verified, identical | tests/parity/test_parity_protocol.py | exact |
| Fusion, binary metrics | verified | tests/parity/test_parity_protocol.py | 1e-15 / 1e-12 |
| Cohort and labels | verified by the data agent | tests/unit/test_data_legacy_parity.py | exact |
| DICOM→NIfTI against the legacy SimpleITK path | verified | tests/unit/test_prepare_dicom_to_nifti.py | voxels exact, affine 1e-4 mm |
| Image inputs (normalisation, resize, subtraction: none/post_minus_pre/consecutive) against legacy `NiftiDataset` (MONAI path) | verified | tests/parity/test_parity_inputs_weights.py | float32 atol 1e-4, rtol 1e-4 |
| Class and sample weights against legacy `_compute_balanced_class_weights` / sample weights | verified | tests/parity/test_parity_inputs_weights.py | 1e-6 (torch float32) / 1e-12 |
| Report columns of a real new evaluation exported to the legacy layout | *(planned: e2e)* `export_evaluation_dir` is unit-tested on a synthetic eval dir that has the `protocol.validate` file shapes | coordinator | — |
