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

## 4b. Lesion segmentation with MRI-CORE (executed 2026-11-24/25)

`docs/segmentation-mri-core.md`. External repository `C:\code\mri_foundation` at
`0f4f452614013873f8bb58a1fd613dbe36b58aba`, weights `mri_foundation.pth` (sha256 prefix `0ca90aeba095`) and
`sam_vit_b_01ec64.pth`. Runs used a scratch venv = the environment above + `einops 0.8.2` +
`torchvision 0.24.1` (CPU), because the main environment has no `torchvision`. The checkout was unchanged
after every run (`git status --porcelain --untracked-files=all` empty, no `__pycache__`).

* Weight analysis: MRI-CORE = DINOv2 ViT-B/16 backbone only. Building it as the repo does takes the neck,
  prompt encoder, decoder and `pos_embed` from SAM. With the repo's forced `num_cls=1`, the decoder's mask
  tokens are random. Our loader reproduces the repo's `num_cls=3` build tensor for tensor
  (`test_loader_matches_the_repos_own_builder`).
* Zero-shot box prompts on **toy** 2D lesions (not MRI): MRI-CORE with SAM's decoder reached Dice 0.92–0.97
  (large lesions) and 0.69–0.86 (8–24 px lesions on 512²). SAM ViT-B scored similarly. MRI-CORE with
  `num_cls=1` scored 0.00–0.52.
* `tests/unit/test_segmentation_mri_core.py` in the scratch venv: **3 passed** (29 s). In the main
  environment these tests skip (no `torchvision`).
* Real-model CLI smoke: `prepare-data segment-mri-core --weights mri_core --device cpu --limit 2` on
  synthetic NIfTI (48×40×10, cohort built from the tables with `profile=general`). 2/2 segmented, exit 0.
  Masks cover 82–99% of each box (synthetic lesion = box) and 0% lies outside the clip region.
* Pipeline, CLI and manifest/audit integration with a fake segmenter: `tests/unit/test_segmentation.py`.
  End to end (segment → train → validate → summarize, 3D `roi_crop` and 2D `mask_max_area` +
  `mask_channel`): `tests/e2e/test_segmentation_e2e.py`.
* Full suite after these changes (2026-11-25, main environment,
  `MRI_BREAST_LEGACY_REPO=C:/code/mri_brest python -m pytest tests -q -p no:cacheprovider`):
  **428 passed, 0 failed, 3 skipped** (12 min 50 s). The 3 skips are the real-weight tests (no `torchvision`
  in the main environment), which passed in the scratch venv as listed above. The reference repository was
  unchanged afterwards (`git status --porcelain --ignored` empty).

Not verified: any real Duke slice (quality, axis convention of the prompts), GPU/Helios runs, Dice
against expert masks (MAMA-MIA masks are not imported yet), and MRI-CORE's pretraining overlap with Duke.

Helios job script (2026-11-25): `segment_mri_core.slurm` replaces the earlier `sbatch --wrap` recipe. That
recipe called `.venvs/<arch>/bin/python`, which does not exist when `AUTO_INSTALL_PYTHON_PACKAGES=true`
(the controller's env is `runtime-<hash>/`), and its `-o` directory was never created. Checked:
`tests/slurm/test_segment.sh` with fake sbatch/module/python (preflight, submission, env built once with
the module constraints, argv), and `bash segment_mri_core.slurm --local --config
configs/synthetic_cpu_3d.conf --output-dir <scratch> -- --device cpu --limit 2` with the real weights on
CPU (2/2 segmented). The real-weight tests also passed in the main environment after installing
`einops 0.8.2` + `torchvision 0.24.1` (3 passed). Not run on Helios: sbatch, `module load ML-bundle`, a
torchvision wheel that matches the module torch on aarch64, CUDA.

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
