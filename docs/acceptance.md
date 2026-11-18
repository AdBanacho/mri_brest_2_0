# Acceptance record

This file records what was **actually run**, on which data, and what remains unverified. All runs
used **synthetic data** on a Windows 11 laptop CPU: Python 3.11.5, torch 2.9.1 (CPU), monai 1.5.1,
pytorch-lightning 2.5.6, scikit-learn 1.7.2, numpy 1.26.0, pandas 2.2.3. The metrics below only
show that the pipeline works mechanically. They say nothing about the research question.
Reference repository: `C:\code\mri_brest` at `127a26046db20e5e506f44ce6d4c1946d34c6416`.

## 1. Test suite (executed 2026-11-18)

```bash
MRI_BREAST_LEGACY_REPO=C:/code/mri_brest python -m pytest tests -q -p no:cacheprovider
```

Result: **342 passed, 0 failed, 0 skipped** (6 min 04 s). After the run, `git status --ignored` in
the reference repository was empty: it was not modified.

| Area | Files |
| --- | --- |
| Config schema, ids, artifacts | `tests/unit/test_config_schema.py`, `test_artifacts_core.py` |
| Cohort, labels, tabular, LASSO, MLP | `tests/unit/test_data_*.py`, `test_tabular_branch.py` |
| Imaging 3D/2D, cache concurrency, boxes, masks, models | `tests/unit/test_imaging_*.py`, `test_masks_providers.py`, `test_models_*.py` |
| Splits, thresholds, fusion, train/validate (resume, rejection, frozen split, exclusions), summary | `tests/unit/test_protocol_*.py`, `test_reporting_summarize.py` |
| Legacy importer/exporter/checkpoints, prepare stage (fake NBIA/Synapse, synthetic DICOM) | `tests/unit/test_legacy_*.py`, `test_prepare_*.py` |
| Bash controller (fake `sbatch`/`python`/`module`) and controller↔CLI parity | `tests/unit/test_controller_*.py` → `tests/slurm/*.sh` |
| Parity with the reference repo | `tests/parity/*.py`, `tests/unit/test_data_legacy_parity.py` |
| Independent review (splits/leakage, mask geometry) | `tests/review/*.py` |
| Remaining modes, end to end through the CLI | `tests/e2e/test_modes_smoke.py` |

## 2. End-to-end acceptance runs (Bash controller, `--local`, CPU)

```bash
python -m mribreast make-synthetic --out runs/synthetic/data     # 48 patients, 50 studies
bash run_experiment.sh --local --train     --config configs/synthetic_cpu_3d.conf
bash run_experiment.sh --local --validate  --config configs/synthetic_cpu_3d.conf
bash run_experiment.sh --local --summarize --config configs/synthetic_cpu_3d.conf
bash run_experiment.sh --local --train     --config configs/synthetic_cpu_2d.conf
bash run_experiment.sh --local --validate  --config configs/synthetic_cpu_2d.conf
bash run_experiment.sh --local --summarize --config configs/synthetic_cpu_2d.conf
```

All six commands exited with code 0. The cohort audit included 44 studies and excluded 6: 4
`missing_oncotype_score` and 2 `fewer_than_2_unique_series`.

**3D run: image + tabular + late fusion.** Profile `duke_legacy_3d`, `resnet10`, 5 phases, training-only
annotation-box erasing, clinical + kinetic + morphology + heterogeneity features, LASSO, a
`xgboost|mlp` grid × α ∈ {0, 0.5, 1}. This gives 2 trainings × 3 folds and 6 evaluations.
Annotation coverage: 40 studies eligible, 6 unmatched, 4 ambiguous.

| rank | feature model | α | image AUC | tabular AUC | fusion AUC |
| --- | --- | --- | --- | --- | --- |
| 1 | xgboost | 0.5 | 0.715 | 0.750 | 0.796 |
| 2 | xgboost | 0 | 0.715 | 0.750 | 0.750 |
| 3 | mlp | 1 | 0.715 | 0.519 | 0.715 |

At α=1 fusion equals the image branch exactly, and at α=0 it equals the tabular branch
(checked per fold, tolerance 1e-12). The image branch is identical across the two trainings, as
expected: same seed, same data, and the image branch does not depend on the tabular model.

**2D run: mask-selected slices + external masks + tabular + fusion.** Profile `general`, `resnet18` 2D,
`slice_selection=mask_max_area`, `mask_mode` grid `roi_crop|mask_channel`, predicted masks from a
frozen external (synthetic) segmenter, `selector_preprocessing=fold_local`,
`mlp_early_stopping=grouped`, α ∈ {0.5, 1}.

The masks were really used. `mask_audit_summary.csv` shows 78 `ok`, 2 `missing` and 2 `empty`
masks; all four missing or empty ones fell back to the full image (`action_full_image`); 0 studies
were dropped. Results: image AUC 0.933 (`mask_channel`) and 0.846 (`roi_crop`).

**Other modes** (`tests/e2e/test_modes_smoke.py`, through `python -m mribreast`):
- historical image-only 3-class run with outer-fold checkpoint selection, flagged in the manifest;
- tabular-only baseline (MLP + LASSO);
- `post_minus_pre` subtraction + `fcn` + `selection_split=separate`;
- ready PNG/TIFF images with 3 images per study, `max_prob` aggregation, `foreground_only`, and
  `mask_missing_policy=drop`, with exclusions audited in `excluded_studies.csv`.

**Controller.** The 24 / 24 / 72 task counts of the legacy profile, the index mapping (identical to
the legacy controller on all 24 rows), frozen snapshots, the dry run having no side effects, and
`PYTHONNOUSERSITE` handling are covered by `tests/slurm/*.sh`, using fakes.

## 3. Comparison with the reference repository (same synthetic data or probabilities)

| Item | Result | Tolerance |
| --- | --- | --- |
| Cohort, row order, series tuples, binary and 3-class labels | identical | exact |
| Tabular preprocessing output, LASSO coefficients and report | identical | exact |
| Outer fold plan + inner calibration split (Duke-style ids) | identical | exact |
| Threshold selection, including ties | identical | exact |
| Fusion at 0 < α < 1 | identical | 1e-15 |
| Fusion at α ∈ {0, 1} | ≤ 1.1e-16 difference | 1e-15 (decision 14 in `migration.md`) |
| Binary metrics | identical | 1e-12 |
| MRI network inputs (`none`, `post_minus_pre`, `consecutive`) vs legacy `NiftiDataset` | match | atol 1e-4, rtol 1e-4 (float32) |
| Class weights / tabular sample weights | match | 1e-6 / 1e-12 |
| Checkpoint ranking, summary tables, ranking order, best JSON | identical | rtol 1e-12 |
| New evaluations exported to the legacy layout (`mribreast.legacy.exporter`) and summarised by the ported legacy summarizer | every historical column present (extras: `mlp_*` duplicates, paths); same ranking order as the new summarizer | — |
| DICOM→NIfTI vs the legacy SimpleITK path (synthetic oblique series) | voxels exact | affine 1e-4 mm |

Every intended difference is listed in `docs/migration.md` §7, each with its test.

## 4. Independent review (`tests/review/`)

The reviewer confirmed 5 findings:

1. **3D mask modes lost small lesions** (mask downsampled before ROI). Fixed: ROI on the native
   grid; `roi_crop` crops at native resolution.
2. **Pre-training audit ignored fold-trained masks.** Fixed: the audit runs per fold.
3. **Split validation checked patient ids in fold 1 only.** Fixed: every fold is checked.
4. **2D subtraction used two reference grids and two tolerances.** Fixed: one grid, one tolerance.
5. **Data-based label-leak check catches only exact copies.** Documented limitation; the
   name-based policy remains the primary control (`docs/data-and-tabular.md`).

All review tests pass now.

## 5. Not verified: requires data or the cluster

* **Real Duke data.** No training or evaluation has been run on real Duke images. The NBIA
  digest (`Duke-Breast-Cancer-MRI_v2_20220609-nbia-digest.xlsx`) is not in either repository. Its
  column names are inferred from legacy code, and its row order defines legacy phase selection.
* **Annotation-box convention vs overlays.** The 0-based inclusive convention and the
  (column, row, slice) → array-axis mapping are reproduced from legacy code and tests only. They
  have not been checked against DICOM overlays.
* **Helios GH200.** Not exercised: no `sbatch`, `module load ML-bundle`, real `flock`, or ARM venv
  bootstrap. Commands for a short Helios smoke run and a Duke run are in `docs/slurm.md`.
* **DICOM→NIfTI on real series.** Tested only on synthetic series. Enhanced or multi-frame MR,
  if present in Duke, has not been tried.
* **TCIA / Synapse downloads.** Exercised only against local fakes.
* **Generalisation claims.** Ranking reuses the outer folds. Choosing a configuration and claiming
  it generalises requires an independent test set or an extra nesting level (stated in every
  `validation_report.md`).
* **Legacy controller test on Windows.** `tests/slurm_controller_test.slurm` passes up to the
  auto-venv block on Git Bash, then needs `flock`/`rg`. It passes on WSL bash with an `rg` shim
  (`docs/reference-test-results.md`).
