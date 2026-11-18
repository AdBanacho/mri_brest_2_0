# Data sources, cohort and tabular branch

Scope: `mribreast.data` (Duke sources, cohort, labels, allowed-predictor policy,
density), `mribreast.tabular` (preprocessing, LASSO, XGBoost/MLP, `TabularBranch`)
and `mribreast.synthetic.duke` (synthetic workbooks). Legacy references are to
`C:\code\mri_brest` (`mriBreastDuke/...`).

## 1. Data model

| Level | Id | Source | Notes |
| --- | --- | --- | --- |
| patient | `patient_id` (`Breast_MRI_NNN`) | clinical workbook col A, metadata `Patient ID` | label, clinical predictors; **every split groups by patient** |
| study | `study_id` (StudyInstanceUID) | metadata `Study Instance UID` | one cohort row; patients may have several studies |
| series / phase | SeriesInstanceUID | metadata `Series Instance UID` | image file `<image_root>/<uid>.nii.gz`; `series_uids` = ordered phase tuple |
| slice | `slice_id` | loaded volume | not in metadata (optional `Number of Images` only) |

`mribreast.data.duke.SeriesRecord` / `StudyRecord` and `study_records(frame, metadata)`
expose this explicitly (row index in the metadata file, description, parsed phase
index, slice count).

## 2. Inputs

### Clinical workbook (`clinical_features_file`)
`Clinical_and_Other_Features.xlsx`, sheet `Data`, **3 header rows** (group, name,
coding), data from row 4, 922 patients. Read columns (legacy `read_from_xlmx.py:16-29`):

| Letter | Name | Use |
| --- | --- | --- |
| A | `patient_id` | key |
| T | `date_of_birth_days` | `age_at_diagnosis_years = -T / 365.25` |
| U | `menopause` | clinical (categorical) |
| V | `race_ethnicity` | sensitive, only with `include_sensitive=true` |
| AB | `oncotype_score` | **label / eligibility only, never a predictor** |
| AK | `tumor_laterality` (L/R) | clinical (categorical, string) |
| AW..BA | multicentric, contralateral, suspicious nodes, skin/nipple, pectoral/chest | clinical (categorical) |

Tokens `"", NA, N/A, NC, NP, X, NAN, NONE` → missing; every column except id and
laterality is coerced numeric (legacy-identical). Duplicate patient ids raise.
Verified on the real workbook: 922 patients, 261 numeric scores, binary 160/101,
3-class 160/79/22 (= legacy `test_clinical_predictors.py`).

### NBIA metadata digest (`metadata_file`) — ASSUMED columns
The file (`Duke-Breast-Cancer-MRI_v2_20220609-nbia-digest.xlsx`) is not in the
legacy repo. From legacy `read_from_xlmx.py:161-177` it is an xlsx with sheet
`Metadata`, header row 0, and at least `Patient ID`, `Study Instance UID`,
`Series Instance UID`. Optional columns used when present: `Series Description`
(phase parsing), `Phase Index` (explicit phase, wins over parsing), `Series Number`,
`Number of Images`, `Modality`, `Study Date`. CSV with the same header is accepted.
**Row order matters** for `duke_legacy_3d` (see 3.1).

### Imaging features (`imaging_features_file`)
`Imaging_Features.xlsx`, sheet `Imaging Features`, `Patient ID` + 529 numeric
columns (or a CSV export). Family classification is the legacy one
(`precomputed_imaging_features.py:87-100`): morphology exact names → heterogeneity
markers → kinetic markers; real file: kinetic 161, morphology 15, heterogeneity 353.
Columns are renamed `<family>_<source name>`. Id column = `imaging_patient_id_column`
(default `Patient ID`). Missing/duplicate ids and non-numeric columns raise. Left
many-to-one merge by patient; zero matches always raise; incomplete matches raise
unless `allow_missing_imaging_features=true` (then features are imputed fold-locally
and the audit column `imaging_features_matched` records it).

### Density (`density_file`, group `density`, default OFF) — `mribreast.data.density`
Real schema of `Breast_Radiologist_Density_Assessments.xlsx`: sheet `Assessments`,
**50 rows**, `Subject_ID` (= patient id, unique), `Radiologist A/B/C` with BI-RADS
letters `a` almost entirely fat, `b` scattered, `c` heterogeneous, `d` extreme;
a 5th empty column whose header is the legend. Raters disagree on 26/50. Coverage:
**18 of the 261 scored patients** (≈7 %). Predictors: `density_consensus`
(categorical; majority vote, else lower-median category) and `density_ordinal_mean`
(continuous; mean of a=1..d=4). Cohort patients without an assessment get NaN; the
cohort notes report coverage and warn below 50 %. Prediction-time eligibility:
allowed at `prediction_time=mri` (policy v1) because the reads were made on the
same pre-treatment MRI; it is a research reader study, not a routinely available
report field, and its availability is non-random (selection bias) — use only as an
explicit sensitivity analysis.

### Annotation boxes
Read by the imaging package; the synthetic writer produces the legacy 7-column
`Sheet1` layout (see 6).

## 3. Cohort (`DukeCohortSource(cfg).load(cfg) -> CohortTable`)

cfg keys used: `clinical_features_file, metadata_file, imaging_features_file,
density_file, allowed_predictors_file, profile, phase_count, phase_order,
label_scheme, positive_class, branches, feature_groups, imaging_patient_id_column,
allow_missing_imaging_features, include_sensitive, prediction_time`.

`frame`: `patient_id, study_id, label (int64), series_uids (tuple)`, tabular
columns, plus informational `series_descriptions` (tuple). `oncotype_score` is
dropped (kept only in `audit`). Use `cohort_to_csv_frame` / `cohort_from_csv_frame`
for the `|`-joined CSV form. Tabular groups are attached only when `tabular` is in
`branches`.

### 3.1 Profile `duke_legacy_3d` (requires `phase_order=metadata`)
Exact port of `get_oncotype_score_for_series_as_studyId_and_label_df`
(`read_from_xlmx.py:226-247`) + `get_oncotype_clinical_predictors_as_study_df`
(`:250-267`):
1. patients with a numeric score; label from the score;
2. inner join of metadata rows with those patients (file row order preserved);
3. keep studies with **≥ 2 unique** series UIDs;
4. `series_uids = first phase_count (5) rows of the study in metadata-file order`
   — duplicate rows are kept and studies with 2–4 series keep 2–4 entries (audit
   notes `duplicate_series_rows_selected`, `only_k_series`, `truncated_n_to_5`);
5. one row per study, sorted by study UID (`groupby` order); `patient_id` = first row.

**Ordering assumption:** the digest lists a study's series in acquisition order
(pre, post1, …). Legacy never checks this; the note is written into cohort notes.
Verified exact parity with legacy on synthetic data (`test_data_legacy_parity.py`).

### 3.2 Profile `general`
Phase index per series = `Phase Index` column, else parsed from `Series Description`
(`parse_phase_index`: `pre` → 0, `Nth pass` / `postN` / `phN` → N, else not a DCE
phase). Missing both columns → `CohortError` (cannot interpret pre/post). Metadata
verification: missing ids raise; a series UID in several studies raises; per study
(excluded with reason): `study_spans_patients`, `duplicate_series_uid`,
`phase_unparseable`, `ambiguous_phase_index`, `missing_pre_contrast`,
`missing_post_contrast`, `insufficient_phases` (fewer than `phase_count`),
`non_contiguous_or_unordered_phases`. `phase_order=explicit` sorts by phase index;
`phase_order=metadata` keeps row order but rejects studies whose order is not
0..phase_count-1. Non-phase series are ignored and noted.

### 3.3 Audit (`CohortTable.audit`, one row per study, plus scored patients without metadata)
`patient_id, study_id, oncotype_score, label, n_series_rows, n_unique_series,
n_selected, included, reason, notes[, imaging_features_matched]`; reasons for
excluded patients: `missing_oncotype_score`, `no_clinical_record`, `no_metadata_series`,
`fewer_than_2_unique_series` (legacy) and the general reasons above.

### 3.4 Labels
`contracts.label_definition`: binary `score <= 18 → 0, > 18 → 1`; 3-class `<=18 / (18,31] / >31`
— identical to legacy `_categorize_oncotype_score` (`read_from_xlmx.py:70-79`).
Legacy fusion always used the binary scheme (`configurable_imaging_features_fusion_workflow.py:409`);
3 classes were historical image-only.

## 4. Allowed-predictor policy (`mribreast.data.predictor_policy`)
Packaged `src/mribreast/data/allowed_predictors_v1.json` (`policy_id
duke_oncotype_allowed_predictors`, version `1.0`); `allowed_predictors_file`
overrides it (major version must be 1). Per `prediction_time` (`mri`), allowed
groups: clinical (8 legacy columns), clinical_sensitive (`race_ethnicity`, needs
`include_sensitive`), `kinetic_`/`morphology_`/`heterogeneity_` prefixes, density
(2 columns). Forbidden: exact names (`oncotype_score`, `label`, ids), tokens (ER,
PR, HER2, subtype, stage/staging, grade/Nottingham/histologic, metastatic,
surgery/therapy/chemo/endocrine/radiation, response, recurrence, death …) and
substrings (`oncotype`, `label`, …). `check()` rejects every column that is
forbidden or not explicitly allowed; `check_label_leakage()` additionally rejects
any predictor whose values equal the label or the score. Both run in
`DukeCohortSource.load` and in `TabularBranch.__init__`. The policy id, version,
source and sha256 are written into cohort notes and `tabular_branch.json`.
All 529 real imaging columns and the 8 clinical columns pass v1.

## 5. Tabular branch (`mribreast.tabular`)

`build_tabular_branch(cfg, *, continuous_columns, categorical_columns, num_classes) -> TabularBranch`
(`contracts.BranchTrainer`, `branch="tabular"`). `fit(FoldData)` uses **`data.fit`
only** (`data.selection` ignored, as legacy), requires every class present, and writes
into `data.output_dir`:

`preprocessor.joblib, selector.joblib (lasso), model.joblib, feature_names.txt,
selected_feature_names.txt, coefficients.csv (lasso: feature, group, selected,
coefficient_class_*), lasso_feature_selection.csv (+fold column),
lasso_feature_importance.png, model_feature_importance.csv/.png (XGBoost),
tabular_branch.json` (fold, seeds, columns, policy, class counts, sample-weight per
class, dims, model params, best C, effective inner folds, MLP epochs, library
versions, schema_version). `FittedTabular.predict_proba(frame)` returns `[n, C]`
aligned to classes 0..C-1 (legacy `aligned_predict_proba`); `load(directory, fold)`
rejects other folds / class counts / column lists.

| Step | Implementation | Legacy |
| --- | --- | --- |
| Preprocess | `TabularPreprocessor`: median+StandardScaler / most_frequent+OneHot(ignore), `keep_empty_features`, names `continuous__x`, `categorical__x_level` | `clinical_preprocessing.py` (bit-identical output) |
| Sample weights | `balanced_sample_weights`: `compute_sample_weight('balanced')`, binary × `positive_boost` on `positive_class` | `n_fold_cv_run.py:360-366` (label 1) |
| LASSO | `LassoFeatureSelector`: `LogisticRegressionCV(penalty=l1, solver=saga, Cs=lasso_cs, scoring roc_auc / roc_auc_ovr_weighted, refit)`, patient-grouped `StratifiedGroupKFold(shuffle, random_state=seed)`, folds capped at groups of rarest class, `min_features` fallback, threshold 1e-8 | `lasso_feature_selection.py` (identical coefficients) |
| XGBoost | `hist`, legacy defaults, binary:logistic / multi:softprob, `xgb_extra_json` last | `make_feature_model` |
| MLP | relu/adam, `hidden_layers`, alpha, lr, max_iter, `mlp_extra_json` last | idem |
| Seed | `duke_legacy_3d`: global `cfg.seed` for every fold; `general`: `FoldData.seed` | `args.seed` |

`selector_preprocessing`:
* `outer` (legacy): preprocess the fit set once, then LASSO CV on that matrix — inner
  validation rows influence medians/scales/levels.
* `fold_local`: each inner subfold fits `Pipeline([TabularPreprocessor, LogisticRegression(l1)])`
  on its own training rows only (same splits, C grid `logspace(-4,4,cs)`, unweighted
  ROC-AUC, first max of summed scores = LogisticRegressionCV rule). Final coefficients:
  `LogisticRegression(C=best)` on the outer-preprocessed fit set (LogisticRegressionCV
  warm-starts its refit from mean fold coefficients — small numerical difference).
  `cv_details_` records each subfold preprocessor (not persisted).

`mlp_early_stopping`:
* `internal` (legacy): sklearn `early_stopping=True, validation_fraction=0.15,
  n_iter_no_change=20` — random **row** holdout, a patient's studies can straddle it.
* `grouped`: `GroupedEarlyStoppingMLP` holds out ≈15 % of **patients** (first fold of
  shuffled `StratifiedGroupKFold(round(1/0.15)=7)`), trains with `partial_fit` one epoch
  at a time (adam state kept), scores weighted accuracy on the holdout with sklearn's
  rule (improve > tol, stop after `n_iter_no_change`), restores the best epoch.
* `off`: no early stopping, `max_iter` epochs on all fit rows.
`mlp_extra_json.early_stopping=true` with `grouped`/`off` is rejected.

**Sample weights in MLP:** `MLPClassifier.fit(sample_weight=…)` exists from
scikit-learn 1.7 (installed 1.7.2 = legacy pin in `requirementsCuda.txt`); older
versions make the legacy call fail. `require_mlp_sample_weight()` raises an explicit
error when absent. Compatible range: `scikit-learn>=1.7,<1.8` (pyproject pins `<1.8`;
1.8 changes the L1 spelling — handled — but is unverified).

Stability: `lasso_stability(reports)` (legacy `n_fold_cv_run.py:676-707`: per
feature `feature_occurrence_count, selection_count, mean/max_lasso_importance,
mean_lasso_rank, selection_frequency = count / n_folds`, sorted by frequency,
mean importance desc, name asc); `save_lasso_stability(reports, dir, top_n)` writes
`lasso_feature_selection_by_fold.csv`, `lasso_feature_stability.csv/.png`;
`save_lasso_feature_importance_chart` = legacy chart (+ density colour).

## 6. Synthetic tables (`mribreast.synthetic.duke.write_duke_tables`)
`write_duke_tables(root, n_patients=60, seed=0, shape=(32,32,16), studies_per_patient_extra=0,
label_scheme_balance=(0.5,0.3,0.2))` → dict with paths keyed by config field
(`clinical_features_file`, `metadata_file` (`nbia_metadata.xlsx`), `imaging_features_file`,
`annotation_boxes_file`, `density_file`), `root`, `studies` (legacy-eligible:
`patient_id, study_id, label` (binary), `label_3class, oncotype_score, series_uids`),
`series`, `patients`, `boxes`. Contents: ~10 % unscored patients, two patients whose only
study has 1 series, 2 (+extra) patients with two studies, ~30 % studies with a 6th
non-DCE series after the five DCE phases, two studies with DCE rows out of order
(same 5 UIDs in both profiles), one study with blank descriptions (`general`
excludes it), exact scores 18/19/31, `NC`/`NP` tokens, real imaging column names with
synthetic values, density in the real schema, boxes zero-based inclusive inside
`shape` as (columns, rows, slices) = NIfTI axes (0,1,2). Predictors correlate weakly
with the score. UIDs are `2.25.<random>`; no real patient data.

## 7. Differences from legacy (summary)
| Item | Legacy | New | Test |
| --- | --- | --- | --- |
| Phase order | metadata row order only | `general` parses + verifies phases | `test_general_profile_*` |
| Metadata checks | none | `general`: spans/duplicates/phase reasons, audit | idem |
| Per-study audit | none (prints) | `CohortTable.audit` | `test_legacy_eligibility_*` |
| Predictor policy | hard-coded list + `_clinical-safe-v2` name suffix | versioned JSON + checks | `test_policy_*` |
| Density | not read | optional group | `test_density_*` |
| Selector preprocessing | outer | `outer` default, `fold_local` option | `test_fold_local_preprocessing_differs_from_outer` |
| MLP early stopping | random 15 % rows | `internal` default, `grouped`, `off` | `test_mlp_grouped_early_stopping_respects_groups` |
| Positive boost class | label 1 | `positive_class` (default 1) | `test_sample_weights_*` |
| Artifact names | `tabular_preprocessor.joblib`, `<model>_model.joblib`, `tabular_*names.txt` | architecture names (§5) | `test_tabular_branch_round_trip` |
| Metadata patient id | `astype(str)` | same in legacy profile; stripped in `general` | — |
| Image-only runs | tabular always loaded | tabular groups only when `tabular` in branches | `test_image_only_branch_skips_tabular` |
