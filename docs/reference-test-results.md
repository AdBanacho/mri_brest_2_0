# Reference repository – test results and environment

Recorded by the audit agent on 2026-11-17. The reference checkout was only read. Every run used a
copy made with `git -c core.autocrlf=false archive 127a2604… | tar -x` in a scratch directory, so the
copy has the LF line endings stored in the index. No file was written into `C:\code\mri_brest`, and
its `git status` stayed clean.

## 1. Reference revision

| Item | Value |
| --- | --- |
| Repository | `C:\code\mri_brest` (GitHub `AdBanacho/mri_brest`) |
| SHA | `127a26046db20e5e506f44ce6d4c1946d34c6416` |
| Commit | 2026-10-13 23:38:56 +0200, "Merge pull request #25 from AdBanacho/fix/atomic-nifti-cache" (93 commits) |
| Working tree | clean. With `core.autocrlf=true` the working tree has CRLF files and the index has LF (`git ls-files --eol`: `i/lf w/crlf`) |
| Untracked data required by `experiment.conf` | `mriBreastDuke/features/Duke-Breast-Cancer-MRI_v2_20220609-nbia-digest.xlsx` (git-ignored `/mriBreastDuke/features/Duke*`) and all NIfTI and DICOM data |

## 2. Dependency declarations in the reference repository

| File | Content (relevant pins) |
| --- | --- |
| `pyproject.toml` | `mri-breast-duke 0.1.0`, `requires-python >=3.10`, **`dependencies = []`**, packages `mriBreastDuke*`, `mama_mia*` |
| `requirements.txt` | `brotli==1.2.0 brotlicffi==1.2.0.0 matplotlib==3.10.8 monai==1.5.1 urllib3_secure_extra==0.1.0` (5 lines only) |
| `requirementsCuda.txt` (UTF-8) | `torch==2.5.1+cu124.post3`, `numpy==2.3.4`, `pandas==2.3.3`, `scikit-learn==1.7.2`, `scipy==1.16.3`, `lightning==2.5.6`, `pytorch-lightning==2.5.6`, `torchmetrics==1.8.2`, `nibabel==5.3.2`, `pydicom==3.0.1`, `simpleitk==2.5.2`, `tcia_utils==3.2.1`, `filelock==3.20.0`, `openpyxl==3.1.5`. **No `monai`, no `xgboost`.** |
| `requirementsNoCuda.txt` (**UTF-16 LE, CRLF**) | `torch==2.9.1`, `numpy==1.26.0`, `pandas==2.2.3`, `scikit-learn==1.7.2`, `monai==1.5.1`, `lightning==2.5.6`, `pytorch-lightning==2.5.6`, `torchmetrics==1.8.2`, `nibabel==5.3.2`, `pydicom==3.0.1`, `simpleitk==2.5.3`, `tcia_utils==3.2.1`. **No `xgboost`.** |
| `experiment.conf` `PYTHON_PACKAGES` | `monai==1.5.1 pytorch_lightning==2.5.6 lightning==2.5.6 torchmetrics==1.8.2` plus unpinned `requests pandas scikit-learn xgboost nibabel openpyxl tensorboard einops matplotlib scipy joblib filelock tqdm pillow`. NumPy and PyTorch come from the Helios `ML-bundle` module and are passed as pip constraints. |
| Not declared anywhere | `synapseclient`, `python-dotenv` (`mama_mia/downloadMamamia.py`); `tcia_utils` (`downloadDicom.py`) appears only in the requirements files |

## 3. Local execution environment (where the tests ran)

Windows 11 Pro 10.0.26200, Git Bash 5.2.15 (MSYS), Python 3.11.5 (CPU). WSL Ubuntu: bash 5.1.16,
`flock` (util-linux) and `/usr/bin/python3` 3.10.12. WSL has no pip/venv and no `rg`.

`pip list` subset: numpy 1.26.0, pandas 2.2.3, scipy 1.16.3, scikit-learn 1.7.2, xgboost 3.2.0,
torch 2.9.1 (CPU), monai 1.5.1, lightning 2.5.6, pytorch-lightning 2.5.6, torchmetrics 1.8.2,
nibabel 5.3.2, openpyxl 3.1.5, joblib 1.5.2, matplotlib 3.10.7, filelock 3.20.0, Pillow 10.0.1,
tifffile 2023.9.26, tensorboard 2.20.0, tqdm 4.67.1, requests 2.31.0, pydicom 3.0.1, simpleitk 2.5.3,
tcia_utils 3.2.1, python-dotenv 1.0.0, pytest 9.2.0. Not installed: dicom2nifti and synapseclient.

The local versions match `requirementsNoCuda.txt` for every package that file pins.

## 4. Results

Commands, run from the root of the LF snapshot:

```bash
# Slurm controller: Git Bash, then WSL
bash tests/slurm_controller_test.slurm
# Python tests (Windows, Python 3.11.5): nothing is written to the snapshot
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q -rA <file>
```

| Test | Environment | Result | Detail |
| --- | --- | --- | --- |
| `tests/slurm_controller_test.slurm` | Git Bash on Windows | **FAIL (exit 2)** after about 40 s | Every assertion up to the "Automatic setup" block passed: dry runs, task-index mapping, identical `--experiment_id` for train and validate, 24-row grid end, identity stability, frozen snapshot, tamper rejection and venv selection. It then stopped with `Error: Automatic environment setup requires flock`, because MSYS has no `flock`. |
| same | WSL Ubuntu, bash 5.1.16 | **FAIL (exit 1)** | `line 186: rg: command not found`. The test calls ripgrep (`rg -c '^install '`), and ripgrep is not part of the declared login-node requirements. |
| same | WSL with a 3-line `rg` shim (`rg -c P F` → `grep -c -E P F`) in a temporary PATH directory | **PASS** in about 3 s | Prints `PASS: Slurm controller integration tests`. The shim is in the harness script, not in the reference repository. |
| `tests/test_annotation_boxes.py` | Windows | **6 passed** (+5 subtests) | The earlier report blamed a missing `nibabel` for this failure. With nibabel 5.3.2 installed, the test now passes. |
| `tests/test_nifti_cache.py` | Windows | **6 passed, 3 failed** | The 3 failures come from the test code on Windows, not from the cache logic: `test_concurrent_processes_create_only_one_complete_cache` raises `ValueError: cannot find context for 'fork'` (Windows has no fork start method). `test_corrupt_source_is_not_hidden_or_published` and `test_permission_errors_do_not_trigger_regeneration` raise `re.error: incomplete escape \U`, because `assertRaisesRegex(..., str(path))` puts an unescaped Windows path into a regex. These 3 tests could not run on Linux here (WSL has no pip/venv for nibabel), so they are **unverified on Linux**. The other 6 tests pass, including truncated-gzip rebuild, partial-write protection and the lock wait. |
| `mriBreastDuke/test_checkpoint_selection.py` | Windows | **5 passed** | |
| `mriBreastDuke/test_threshold_tuning.py` | Windows | **4 passed** | Together with the 5 above, these are the "9 tests" named in the task statement. |
| `mriBreastDuke/dataLoaders/test_clinical_predictors.py` | Windows | **collection ERROR** | `FileNotFoundError: …/mriBreastDuke/mriBreastDuke/dataLoaders/read_from_xlmx.py`: `Path(__file__).parents[1]` already points to `mriBreastDuke`, so the module path is wrong. The test cannot run on any OS. If the path were right, the test would still skip without `DUKE_CLINICAL_FEATURES_TEST_FILE` (it expects 922 patients in the real workbook). |
| `mriBreastDuke/dataLoaders/test_lasso_feature_selection.py` | Windows | **5 passed** (8 sklearn warnings) | |
| `mriBreastDuke/dataLoaders/test_pad_collate.py` | Windows | **2 passed** | |
| `mriBreastDuke/classificators/test_xgboost_fusion.py` | Windows | **4 passed** | |

Totals for the Python tests: **42 passed, 3 failed (Windows-only test artifacts), 1 collection error (broken test path)**.
Slurm controller: **passes on Linux bash with flock and an `rg` substitute**. On Git Bash it fails because `flock` is missing.

## 5. Ad-hoc checks on the reference code (scratch scripts, read-only)

| Check | Result |
| --- | --- |
| `Simple3DFCN(num_classes=2, in_channels=5)`, as called by `configurable_imaging_features_fusion_workflow.make_mri_network("fcn")` and `validate_best_checkpoints` | `TypeError: unexpected keyword argument 'in_channels'`, so **`--mri_model fcn` cannot run in the current workflows** |
| `Simple3DFCN(num_classes=2)` on a 5-channel batch (historical `trainingWorkflow --model 0`) | `RuntimeError: expected input … to have 1 channels, but got 5`, so the **historical FCN was broken too** |
| `argparse type=bool` (`--is_binary_classification False`) | parses to `True`. The historical image-only scripts get the 3-class task only when the flag is omitted. |
| `f"{lr:.0e}"` in experiment names | `1.6e-4` and `2e-4` both give `lr2e-04`, a name collision. `1.5e-4` gives `1e-04`. |
| Stored `mriBreastDuke/validationSummary` | 16 configurations × 5 folds, all complete. The grid is `densenet121\|resnet18` × `clinical\|clinical,kinetic,morphology,heterogeneity` × `xgboost` × lasso × `sens 0.05\|0.3` × `boost 1\|2`, alpha 0.5. It does **not** match the current `experiment.conf` grid (`resnet10`, `mlp`, `sens 0.2` and `boost 1` only are missing or different). The run names lack `_clinical-safe-v2` and `_cfg<id>`, and `validation_config` lacks `seed/target_size/use_annotation_boxes/experiment_id/...`. The stored results therefore come from an older code state, before the clinical-safe-v2 predictor list and before the Slurm controller. Mean row counts are 166.8 fit / 42 calibration / 52.2 outer, so the cohort has 261 studies. |
| Re-ranking the stored `configuration_summary.csv` with the legacy `rank_configurations` and with the port | Both reproduce the stored `ranked_configurations.csv` order exactly (`tests/parity/test_parity_summary.py::test_stored_ranking_order_is_reproduced`) |

## 6. New-repository parity results (this agent)

`PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/parity` gives **40 passed**
on the Windows environment above, with the reference repo found at the default path. The session
guard (`tests/parity/conftest.py`) confirmed that the reference file set did not change. Details are
in `docs/legacy-parity.md` §7.

Unit tests owned by this agent (`tests/unit/test_legacy_*.py`, `tests/unit/test_prepare_*.py`):
**26 + 18 passed**. This includes a SimpleITK cross-check of the DICOM→NIfTI conversion on an
oblique synthetic series (identical voxels, affine within 1e-4 mm).
