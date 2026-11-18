# Protocol: splits, selection, thresholds, evaluation, reporting

Code: `src/mribreast/protocol/` (`splits`, `thresholds`, `fusion`, `train`, `validate`, `manifests`) and
`src/mribreast/reporting/` (`tables`, `plots`, `report`, `summarize`; metric definitions in `metrics.py`).
The protocol depends only on `contracts.BranchTrainer` / `FittedPredictor`; branches are injected by `cli.py`.
Test doubles for every area: `tests/fakes.py` (`FakeTrainer`, `make_fake_cohort`, `make_cfg`, `make_trainers`,
`import_legacy`).

## 1. Split plan (`split_manifest.csv`)

* Outer: `StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)` over the cohort rows **in
  cohort order**, grouped by `patient_id` (all studies, phases, slices and variants of one person stay together).
* Inner (fold *k*): legacy `make_inner_calibration_split` on the outer-train rows (cohort order) with
  `random_state=seed+k`, `requested_folds=threshold_calibration_folds`; the first inner fold is held out.
  `effective_inner_folds = min(requested, smallest number of patients per class)` is recorded per row.
* `selection_split`:
  * `shared` (legacy fusion): roles `fit`, `calibration`, `outer_eval`.
  * `separate`: the calibration patients are split once more (grouped, stratified, 2 folds,
    `random_state=seed+k+1000`) into `selection` and `threshold`.
  * `outer` (historical image-only; legacy profile, `branches=image` only): `fit` = whole outer-train, no inner
    split (`requested/effective_inner_folds=0`, `inner_seed=-1`).
* Columns: `fold, patient_id, study_id, label, role, requested_inner_folds, effective_inner_folds, outer_seed,
  inner_seed`, rows ordered by fold, role, then cohort order. `split_hash = "sha256:" + sha256(canonical CSV)`
  and equals `artifacts.file_fingerprint(split_manifest.csv)`.
* Validators (`validate_split_plan`): each (fold, study) once; patients disjoint across roles inside a fold; each
  study in `outer_eval` exactly once; every class present in `fit` and in every inner held-out role; one label per
  patient; plan covers exactly the cohort studies with the same patient ids.
* **Frozen**: once `split_manifest.csv` exists in the train directory it is reused and verified (ids, roles vs
  `selection_split`, fold count, class presence with the *current* labels), never regenerated. Labels that differ
  from the stored ones are counted in `train_manifest.json::split_label_changes`.

## 2. What data each decision uses

| mode | fit (weights, preprocessing, selector, tabular model) | checkpoint selection / early stopping (`FoldData.selection`) | threshold tuning | evaluation |
| --- | --- | --- | --- | --- |
| shared | fit | calibration | calibration | outer_eval |
| separate | fit | selection | threshold | outer_eval |
| outer | fit (= outer-train) | **outer_eval** | **outer_eval** | outer_eval |

`evaluation_manifest.json` records these roles plus `selection_uses_outer_eval` / `threshold_uses_outer_eval`.
Labels used for metrics always come from the current cohort; the split file only fixes the assignment.

Comparing many configurations (grid rows, `fusion_alpha`) on the same outer folds and reporting the best one is a
selection step on the evaluation data: both `validation_report.md` files state that a generalisation claim for the
selected configuration needs an independent test set or an extra level of nesting.

## 3. Thresholds and decisions

* `balanced_accuracy` (legacy `select_balanced_accuracy_threshold`, ported verbatim): candidates
  `np.unique({0, 0.5, 1} ∪ scores)` ascending; key `(balanced_accuracy, sensitivity, -|t-0.5|)`; Python `max`
  keeps the **first (lowest)** candidate on a full tie. A full tie between two distinct candidates is in fact
  impossible (equal sensitivity forces all rows between them to be negatives, which changes specificity), so the
  effective rules are BA → sensitivity → float64 `|t-0.5|` (note `|0.7-0.5| < |0.3-0.5|` in float64, which favours
  the upper candidate). Locked by `test_protocol_thresholds.py::test_reference_against_legacy_on_tie_heavy_inputs`
  (400 tie-heavy random cases vs the legacy function) and `::test_documented_tie_breaks`.
* Decision: `p[:, positive_class] >= threshold` → `positive_class`, else the other class
  (`thresholds.decide`, `metrics.binary_decisions`).
* `fixed_0.5`: threshold 0.5; the calibration table still reports calibration metrics at 0.5.
* Three classes: no threshold, `argmax` for predictions, metrics and confusion matrices; macro sensitivity /
  specificity / OvR ROC-AUC (`reporting.metrics`). `decision_thresholds.json` stores `null`.
* Thresholds are independent per branch; the fusion threshold is tuned on fused calibration probabilities.
* Pooled metrics and the pooled confusion matrix concatenate per-fold decisions, each made with that fold's
  threshold; pooled ROC-AUC uses concatenated probabilities.

## 4. Fusion

`p_fusion = alpha * p_image + (1 - alpha) * p_tabular`, then the legacy row renormalisation. Branch outputs are
aligned by `study_id` (`fusion.align_to_reference`), rejecting missing / duplicate / unexpected studies, class
columns other than `prob_0..prob_{C-1}` and rows that are not distributions. `alpha` is evaluation identity only
(`eval_id`), so every alpha reuses the same training run.

## 5. Training loop and resume (`protocol.train.run_training`)

1. `train_id = schema.train_id(cfg, input_fingerprints)`; directory `<checkpoint_root>/<train_id>/`.
2. New directory: split plan (or frozen one), `cohort.csv` (no `oncotype_score`, `series_uids` `|`-joined),
   `cohort_audit.csv`, extra audits (`<name>.csv`), `train_manifest.json` (schema versions, ids, train identity,
   code + environment versions, resolved config, label definition, input fingerprints, cohort fingerprint
   (= sha256 of `cohort.csv`), split hash, branches, classes, folds, counts).
   Existing directory: manifest verified → `ArtifactMismatch` on a different train_id, label definition, input
   fingerprints, cohort fingerprint or split hash.
3. Per fold: skip when `fold_k/COMPLETE` exists and `fold_manifest.json` matches train_id + split_hash; otherwise
   build in a staging directory: per branch `FoldData(fit, selection, seed, output_dir=staging/<branch>)`,
   `trainer.fit`, then `predictor.save(output_dir)` (**must be idempotent**; artifacts must reference each other
   with relative paths because the directory is renamed). `fold_manifest.json` stores role counts, inner-fold
   metadata and sha256 of every branch artifact. Published with `atomic_dir` + `COMPLETE`.
4. After all folds: optional `trainer.after_all_folds(train_dir)` hooks; if the tabular branch left
   `fold_k/tabular/lasso_feature_selection.csv` files and no hook wrote `lasso_feature_stability.csv`, the legacy
   aggregation (`n_fold_cv_run.py` 676-708) is applied. Top-level `COMPLETE`.

## 6. Validation outputs (`<validation_root>/<train_id>/<eval_id>/`, atomic, `COMPLETE`)

| file | content |
| --- | --- |
| `evaluation_manifest.json` | ids, eval identity, code/env versions, split hash, cohort fingerprint, label definition, roles, flags, primary branch, decision rule, report unit, pooled metrics |
| `validation_config.json` | resolved config + `train_id, eval_id, run_id, primary_branch_resolved, num_classes, threshold_strategy_effective` |
| `validation_metrics.csv` | one row per fold: legacy head `fold, checkpoint_path, inner_fit_rows, inner_calibration_rows, effective_inner_calibration_folds, outer_validation_rows`, then `<branch>_{accuracy, balanced_accuracy, sensitivity, decision_threshold, specificity, auc_roc}` for image, tabular, fusion (legacy order; no `decision_threshold` for 3 classes), then `requested_inner_folds, selection_rows, threshold_rows, evaluation_units, report_unit, selection_split, run_id` |
| `validation_summary.csv` | `metric, mean, std (ddof=0), pooled` |
| `decision_thresholds.json` | `{threshold_strategy, positive_class, threshold_split, report_unit, folds: {"k": {branch: t}}}` |
| `threshold_calibration_metrics.csv` | `fold, branch, threshold, balanced_accuracy, sensitivity, specificity, strategy, calibration_rows, threshold_split, report_unit` |
| `predictions_calibration.csv`, `predictions_validation.csv` | `contracts.predictions_frame` + `decision`, long format over branches |
| `branch_comparison.csv` | per branch `<metric>_{mean,std,pooled}`, `is_primary` |
| `confusion_matrices.json` | per branch pooled + per fold (rows true, columns predicted) |
| `roc_<branch>.png`, `confusion_<branch>.png`, `validation_report.md` | pooled charts; report with data-usage statement |

`report_unit=patient`: study probabilities are averaged per patient before thresholds, decisions and metrics;
prediction rows are then per patient with `study_id` = `|`-joined study ids. `checkpoint_path` is the image
predictor's `checkpoint_path` attribute when present, else the first branch directory.
An evaluation that is already complete is returned unchanged (identity checked).

## 7. Summary (`reporting.summarize.summarize`)

Scans `<validation_root>/*/*/validation_metrics.csv` (hidden staging dirs skipped). Per run: mean and population
std of every numeric column except `fold`; `is_complete` = folds `1..num_folds` (from `validation_config.json`,
fallback `expected_folds`) **and** a `COMPLETE` flag. Ranking: the legacy `sort_values` call (descending,
`na_position="last"`, ties in discovery order) on `primary_{auc_roc, balanced_accuracy, sensitivity,
specificity}_mean` — copies of the primary branch columns, equal to the legacy `fusion_*` keys for fusion runs
(`test_reporting_summarize.py::test_ranking_matches_legacy_implementation`). Incomplete runs are listed and
warned about, ranked only with `include_incomplete`. Outputs: `all_validation_metrics.csv`,
`configuration_summary.csv`, `ranked_configurations.csv` (legacy column prefix), `best_configuration.json`,
`validation_report.md`, `fusion_metric_comparison.png`, `branch_auc_comparison.png`.

## 8. Compatibility profile vs general protocol

| # | Legacy / compatibility behaviour | New / general protocol | Test |
| --- | --- | --- | --- |
| 1 | Outer SGKF(seed) + inner split seed+fold, effective folds recorded | identical (bit-for-bit) | `test_protocol_splits.py::test_plan_reproduces_legacy_fold_loop`, `::test_inner_split_matches_legacy_bit_for_bit`, `::test_effective_inner_folds_recorded_when_requested_exceeds_groups` |
| 2 | One study per patient (Duke) | several studies/slices per patient grouped by `patient_id` | `::test_plan_is_deterministic_and_patient_disjoint_with_multi_study_patients` |
| 3 | Validation re-derived the split and compared it with `inner_calibration_split.csv` | split frozen in `split_manifest.csv`, hash in every manifest, reused on resume | `test_protocol_train_validate.py::test_existing_split_manifest_is_frozen_and_reused`, `::test_rejects_modified_split_or_fold_artifacts` |
| 4 | Calibration set both selects the checkpoint and tunes thresholds (`shared`) | optional `separate` selection/threshold patients | `::test_selection_split_modes_feed_branches_and_flags[separate]` |
| 5 | Historical image-only: outer fold selects checkpoint, ungrouped `StratifiedKFold`, argmax | `outer` mode keeps the optimistic selection but groups by patient and flags it | `::test_selection_split_modes_feed_branches_and_flags[outer]` |
| 6 | Legacy 3-class metrics: no specificity | argmax + macro specificity from `reporting.metrics` | `::test_three_class_flow_uses_argmax` |
| 7 | Study-level metrics only | `report_unit=patient` (mean probability per patient) | `::test_patient_report_unit_aggregates_studies` |
| 8 | Outer-eval labels never touched training/thresholds (claimed) | proven on artifacts with a frozen split | `::test_frozen_split_outer_eval_labels_only_change_evaluation` |
| 9 | No resume / no identity checks | resume per fold, `ArtifactMismatch` on foreign artifacts | `::test_resume_after_interrupted_fold`, `::test_rejects_artifacts_from_another_configuration` |
| 10 | Per-fold sub-directories `fold_k/{decision_thresholds.json,...}`, tabular columns named `xgboost_*`/`mlp_*` | flat files with a `fold` column, `tabular_*` (legacy exporter restores model-named copies; summarize canonicalises `xgboost_*` → `tabular_*`) | `test_reporting_summarize.py::test_ranking_order_and_completeness` |
| 11 | Summary: any `validation_metrics.csv` below the root; complete = all folds | two-level layout, also requires `COMPLETE`; ranking on primary branch | `test_reporting_summarize.py::test_ranking_order_and_completeness`, `::test_include_incomplete` |
| 12 | Pooled confusion from per-fold thresholds; decision `p >= t` | identical | `test_protocol_train_validate.py::test_pooled_confusion_uses_per_fold_thresholds`, `test_protocol_thresholds.py::test_decision_is_greater_or_equal` |
| 13 | Fusion with renormalisation; alpha=0/1 equals one branch | identical | `test_protocol_fusion.py::test_alpha_extremes_equal_single_branch`, `test_protocol_train_validate.py::test_alpha_extremes_match_single_branch` |

Preprocessing inside selector CV (`selector_preprocessing`) and grouped MLP early stopping (`mlp_early_stopping`)
are tabular-branch options and are documented/tested by that branch.
