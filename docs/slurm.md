# Running experiments: `run_experiment.sh` / `run_experiment.slurm`

`run_experiment.sh` is a one-line launcher. `run_experiment.slurm` is both the
**controller** (login node: Bash only, no Python, no TOML) and the **Slurm batch
task** (compute node: prepares Python and runs `python -m mribreast <stage>`).
Requirements on the submitting host: Bash >= 4.3 (4.4+ recommended), GNU coreutils
(`realpath`, `sha256sum`, `mktemp`, `sort`), `awk`, `sed`; `sbatch` for submission.

## Usage

```bash
bash run_experiment.sh --show-config [--validate] [--config FILE]   # readable preview + grid table
bash run_experiment.sh --train --dry-run                            # plan + example task; changes nothing
bash run_experiment.sh --train                                      # submit array (prints Slurm ID)
bash run_experiment.sh --validate --dependency afterok:<train id>
bash run_experiment.sh --summarize --dependency afterok:<validate id>
bash run_experiment.sh --prepare                                    # optional: cohort/cache/mask audit per training row
bash run_experiment.sh --validate --task-index 23 --dry-run         # print the command of one task
bash run_experiment.sh --local --train --config configs/synthetic_cpu_3d.conf   # run here, sequentially
```

| Option | Meaning |
| --- | --- |
| `--prepare` / `--train` (`--training`) / `--validate` (`--validation`) / `--summarize` | Stage (exactly one) |
| `--config FILE` | Configuration (default `<project>/experiment.conf`) |
| `--dry-run` | Prints array size, sbatch command, cleanup targets, packages and task command(s). Creates no files or directories, calls no sbatch/module/Python |
| `--show-config` | Resolved paths (with "does not exist" markers), resources, fixed values (`*` = fields.tsv default), grid axes and the full row table |
| `--local` | Writes the same frozen snapshot as a submission, then runs every task (or `--task-index N`) sequentially through the **frozen controller's task path**: identical argv to Slurm. Logs: `<slurm_logs>/run-*/local-<stage>-<i>.log`. Stops at the first failing task |
| `--dependency SPEC` | Passed as `--dependency=SPEC` (not allowed with `--local`) |
| `--task-index N` | With `--dry-run`: print task N. Without: run task N directly from the (unfrozen) config (debugging) |
| `--snapshot`, `--project-root` | Internal, used by queued tasks |

## Configuration file

Trusted Bash (sourced). Scalars: `PYTHON_BIN`, `PYTHON_VENV_ROOT`,
`AUTO_INSTALL_PYTHON_PACKAGES`, `MAX_JOBS`, `MAX_CONCURRENT`,
`CLEAN_BEFORE_TRAIN`, `CLEAN_BEFORE_VALIDATE`. Arrays: `PYTHON_PACKAGES`,
`MODULES`, `EXTRA_SBATCH_{TRAIN,VALIDATE,SUMMARIZE,PREPARE}`. Associative arrays
(`declare -A`): `PATHS`, `TRAIN`, `TRAIN_GRID`, `VALIDATE`, `VALIDATE_GRID`,
`SUMMARIZE`, `JOB_TRAIN`, `JOB_VALIDATE`, `JOB_SUMMARIZE`, optional `JOB_PREPARE`
(defaults to a copy of `JOB_TRAIN` named `mri_prepare`).

**Field registry.** Keys, sections, types, defaults and allowed choices are read
by Bash (awk) from `src/mribreast/config/fields.tsv`, the same file
`mribreast.config.schema` uses. `PATHS` takes `path` fields, `TRAIN`/`TRAIN_GRID`
`train` fields, `VALIDATE`/`VALIDATE_GRID` `validate` fields, `SUMMARIZE`
`summarize` fields. Unknown keys fail, with a hint when the key belongs to another
section. Legacy alias: `VALIDATE[num_workers]` → `validate_num_workers` (from the
`renamed:` entry of the legacy column). Unset keys take the fields.tsv default,
so the snapshot and the argv are fully explicit.

**Checks done in Bash before anything is submitted**: types (`int`, `float`,
`bool` = `true|false` only, `choice`, `csv` items against choices, `ints` positive
words, `json` must be `{...}`), numeric ranges (a small table in the controller,
e.g. `num_folds>=2`, `fusion_alpha` in [0,1], `xgb_subsample` in (0,1]), resources
(`cpus>0`, `gpus>=0`, memory like `48G`, time like `24:00:00`), `EXTRA_SBATCH_*`
whitelist (`--qos --constraint --reservation --nice --mail-type --mail-user`,
`--name=value` form), and cheap combination rules per row (legacy profile needs
3D NIfTI without masks; masks need a provider and a manifest; `density` needs
`PATHS[density_file]`; 2D/3D target sizes; `selection_split=outer` is image-only;
3-class labels need `threshold_strategy=fixed_0.5`; protected
`trainer_extra_json` keys). Grid axes that cannot affect a row (e.g. `xgb_*` when
`feature_model=mlp`, `mri_model` without the image branch) fail when inert in
every row and warn when inert in some. **Python is authoritative**: on the
compute node `schema.resolve` re-validates everything and the task fails within
seconds of starting; the login node never runs Python.

**Paths.** Relative `PATHS` resolve against the configuration file's directory.
The 12 legacy keys are required; `density_file`, `allowed_predictors_file`,
`image_manifest_file`, `mask_manifest_file` may be empty (passed as `''`).
Which inputs a mode actually needs is checked by Python (`cli.required_inputs`).

## CLI contract (controller → Python)

```
$PYTHON_BIN -m mribreast <stage> --<key> <value> ...
```

* every field of the stage's sections, in fields.tsv order:
  prepare/train = path+train, validate = path+train+validate, summarize = path+summarize;
* booleans explicit (`--use_annotation_boxes true`), csv as one comma-joined
  argument (`--feature_groups clinical,kinetic`), `ints` as words
  (`--target_size 256 256 64`), json/str/paths as one argument (`--imaging_patient_id_column 'Patient ID'`);
* the controller computes no ids; Python derives `train_id`/`eval_id`.

`tests/unit/test_controller_cli_parity.py` parses every task argv of every
shipped configuration with `schema.build_parser(stage)` + `schema.resolve` and
checks exact key coverage and value round-trips.

## Grid semantics

* `|` separates choices; commas stay inside a choice (csv keys); spaces stay inside
  a choice (`TRAIN_GRID[target_size]='256 256 64|128 128 32'`, `'Patient ID|Subject Id'`).
* Empty choices (`a|`, `|a`, `a||b`), duplicates and >1,000,000 rows are rejected.
* A key is either fixed (`TRAIN`) or an axis (`TRAIN_GRID`), never both.
* Keys are sorted (C locale); **the last sorted key varies fastest** (legacy
  `select_grid`). For the legacy profile the varying axes are
  `feature_groups(2) feature_model(2) mri_model(3) sensitivity_lambda(2)`:
  row 1 = lambda 0.2, row 2 = resnet10, row 6 = mlp, row 12 = all feature groups,
  row 23 = all last choices. Verified identical to the reference controller for
  all 24 rows (`MRI_BREAST_LEGACY_REPO=... bash tests/slurm/test_grid.sh`).
* `MAX_JOBS>0` keeps the first N training rows **before** validation expansion.
* Validation task `i` = training row `i / |VALIDATE_GRID|`, validation row
  `i % |VALIDATE_GRID|`. Legacy profile: 24 train, 24 validate (1 alpha), 72 (3 alphas).
* `MAX_CONCURRENT` is the array throttle (`--array=0-N%MAX_CONCURRENT`).

## Snapshot model

On submission (and `--local`) the controller creates
`<slurm_logs>/run-<stage>-<timestamp>.XXXXXX/` containing, all read-only:

| File | Content |
| --- | --- |
| `experiment-<sha256>.conf` | `declare -p` of the fully resolved configuration (absolute paths, defaults filled). Verified by its file name before it is sourced |
| `run_experiment.slurm` | frozen controller copy; this is what `sbatch` runs |
| `fields.tsv` | frozen field registry; its sha256 is stored in the snapshot and verified |
| `code_version.txt` | `git:<sha>[-dirty]` (or `package:<version>` without git), package version, controller/fields/snapshot sha256, user, host, time |
| `slurm_job_id` / `local-*.log` | job id, or local logs |

Editing `experiment.conf` (or fields.tsv) after submission does not change queued
tasks; a modified snapshot or registry stops the task before anything runs. The
Python code itself is **not** frozen: tasks run `src/` of `--project-root` as it
is when the task starts (`code_version.txt` records the submit-time version and
Python records its own version in every manifest). Do not edit the checkout while
jobs are queued, or use a separate worktree per campaign.

## Compute-node runtime (Helios GH200)

Per task: `cd <project root>`; `export PYTHONNOUSERSITE=1`; `module load` each
`MODULES` entry; `export PYTHONNOUSERSITE=1` again (modules may reset it);
`PYTHONPATH=<project>/src:$PYTHONPATH` (the package is **not installed**; it runs
from the checkout, so no build step and no stale installed copy). Then:

* `AUTO_INSTALL_PYTHON_PACKAGES=true`: environment
  `$PYTHON_VENV_ROOT/$(uname -m)/runtime-<hash>/` (`venv --system-site-packages`)
  where the hash covers the base interpreter, its version, module NumPy/PyTorch
  versions, `MODULES` and `PYTHON_PACKAGES`. The module's `numpy==`/`torch==` are
  pip constraints; after installation `pip check`, the NumPy/PyTorch versions,
  `import numpy, torch` and `python -m mribreast --help` are verified, then
  `.ready` is written. The install is serialised with `flock`, or with an atomic
  `mkdir` lock when `flock` is missing (timeout `ENV_LOCK_TIMEOUT`, default 7200 s;
  a lock left by a killed job is the directory `runtime-<hash>.lock.d`, remove it).
  Failed setups leave no `.ready` and are retried by the next task. Requires
  package-index access from compute nodes on first use.
* `AUTO_INSTALL_PYTHON_PACKAGES=false` with `PYTHON_VENV_ROOT` set: use a prepared
  venv — either `PYTHON_VENV_ROOT` itself (if it contains `pyvenv.cfg`) or
  `PYTHON_VENV_ROOT/$(uname -m)`. Missing → clear error.
* `PYTHON_VENV_ROOT=''`: `PYTHON_BIN` as is (module or local Python).

Prepare a venv by hand on a GH200 node (interactive job):

```bash
srun -A <account> -p plgrid-gpu-gh200 --gres=gpu:1 -c 4 --mem=16G -t 1:00:00 --pty bash -l
export PYTHONNOUSERSITE=1; module load ML-bundle; export PYTHONNOUSERSITE=1
python3 -m venv --system-site-packages .venvs/$(uname -m)
python3 -m pip show numpy torch | awk '$1=="Name:"{n=tolower($2)} $1=="Version:"{print n"=="$2}' > /tmp/c.txt
.venvs/$(uname -m)/bin/python -m pip install -c /tmp/c.txt pandas scikit-learn xgboost monai==1.5.1 \
  pytorch-lightning==2.5.6 torchmetrics==1.8.2 nibabel openpyxl joblib matplotlib pillow tifffile filelock scipy
# experiment.conf: AUTO_INSTALL_PYTHON_PACKAGES=false, PYTHON_VENV_ROOT=.venvs
```

No `/net/...` path appears in package code; Helios locations appear only as
comments in `experiment.conf`.

## Cleanup

`CLEAN_BEFORE_TRAIN=true` removes `checkpoint_root` and `logs_root`;
`CLEAN_BEFORE_VALIDATE=true` removes `validation_root` (and `summary_root` unless
nested in it) — once, in the submitting controller (or `--local` run), after the
snapshot is written, never in array tasks. Refused when a target is `/`, `$HOME`,
an ancestor of `$HOME` or of the project, has fewer than three path components,
is not a directory, or overlaps any input/cache path, `slurm_logs` (snapshots) or
another output root. Checks use POSIX paths (also on Git Bash). `--dry-run` prints
`Would remove:` lines. Do not clean while older jobs still write there.

## Short Helios run (smoke test; marked: not executed by the author)

```bash
git clone <repo> mri_brest_2_0 && cd mri_brest_2_0
mkdir -p data/duke && ln -s /path/to/Clinical_and_Other_Features.xlsx /path/to/Imaging_Features.xlsx \
  /path/to/Annotation_Boxes.xlsx /path/to/Duke-Breast-Cancer-MRI_v2_20220609-nbia-digest.xlsx data/duke/
ln -s /net/storage/.../tciaNifti data/duke/nifti        # <SeriesInstanceUID>.nii.gz files
cp experiment.conf smoke.conf
cat >> smoke.conf <<'EOF'
MAX_JOBS=1
TRAIN[epoch]=1
TRAIN[num_folds]=2
TRAIN[threshold_calibration_folds]=2
JOB_TRAIN[time]=02:00:00
JOB_VALIDATE[time]=01:00:00
PATHS[slurm_logs]=runs/smoke/slurm
PATHS[checkpoint_root]=runs/smoke/checkpoints
PATHS[logs_root]=runs/smoke/lightning_logs
PATHS[validation_root]=runs/smoke/validation
PATHS[summary_root]=runs/smoke/validation/summary
EOF
bash run_experiment.sh --show-config --config smoke.conf
bash run_experiment.sh --train --config smoke.conf --dry-run
T=$(bash run_experiment.sh --train --config smoke.conf | sed -n 's/.*Slurm ID \([0-9_]*\).*/\1/p')
V=$(bash run_experiment.sh --validate --config smoke.conf --dependency afterok:$T | sed -n 's/.*Slurm ID \([0-9_]*\).*/\1/p')
bash run_experiment.sh --summarize --config smoke.conf --dependency afterok:$V
```

## Duke run (full legacy profile)

Inputs (paths in `PATHS`): `Clinical_and_Other_Features.xlsx` (Oncotype score,
patient id), the NBIA digest `Duke-Breast-Cancer-MRI_v2_20220609-nbia-digest.xlsx`
(series metadata; **row order defines the phase order** in `duke_legacy_3d`; not
tracked in the reference repository — download it from TCIA), `Imaging_Features.xlsx`
(`Patient ID` column), `Annotation_Boxes.xlsx`, and `image_root` with one
`<SeriesInstanceUID>.nii.gz` per series (produced by `python -m mribreast prepare`
download/convert commands or the historical `tciaNifti` tree). Caches
(`prepared_root`, `subtraction_root`) and `checkpoint_root` belong on scratch.

```bash
bash run_experiment.sh --show-config                 # expect 24 training rows
T=$(bash run_experiment.sh --train | sed -n 's/.*Slurm ID \([0-9_]*\).*/\1/p')
V=$(bash run_experiment.sh --validate --dependency afterok:$T | sed -n 's/.*Slurm ID \([0-9_]*\).*/\1/p')
bash run_experiment.sh --summarize --dependency afterok:$V
```

Use `afterany` instead of `afterok` if a single failed array task should not
block validation of the others. Three alphas: `VALIDATE_GRID[fusion_alpha]='0.25|0.5|0.75'`
→ 72 validation tasks, no retraining.

## Local CPU runs (synthetic data)

```bash
python -m mribreast make-synthetic --out runs/synthetic/data
for s in train validate summarize; do bash run_experiment.sh --local --$s --config configs/synthetic_cpu_3d.conf; done
for s in train validate summarize; do bash run_experiment.sh --local --$s --config configs/synthetic_cpu_2d.conf; done
```

`synthetic_cpu_3d.conf`: legacy profile, ResNet10 at 32×32×16, image+tabular with
LASSO, `feature_model` xgboost|mlp (2 rows), alpha 0|0.5|1 (6 validations).
`synthetic_cpu_2d.conf`: general profile, 2D slices with the largest mask area,
`mask_mode` roi_crop|mask_channel (2 rows), alpha 0.5|1. On Windows use Git Bash;
paths are handed to native Python as `C:/...` (via `cygpath -m`).

## Tests

```bash
for t in tests/slurm/test_*.sh; do bash "$t"; done      # fake sbatch/python/module
MRI_BREAST_LEGACY_REPO=/path/to/mri_brest bash tests/slurm/test_grid.sh
python -m pytest tests/unit/test_controller_bash.py tests/unit/test_controller_cli_parity.py
```

## Differences from the reference controller

* Python entry point `python -m mribreast <stage>` with **all** stage fields and
  explicit booleans (legacy: per-module CLIs, presence flags, `--feature_groups a b`,
  hyphenated summary flags, `--experiment_id cfg<hash>` computed in Bash).
* `--prepare`, `--local`, `--show-config`; `JOB_PREPARE`.
* Keys/types/choices from fields.tsv instead of hard-coded key lists; type,
  range and combination checks before submission (legacy checked only booleans,
  `target_size` and resources).
* A key in both `TRAIN` and `TRAIN_GRID` is an error (legacy: grid silently won).
* Snapshot also freezes `fields.tsv` and records the code version; run dirs are
  `run-<stage>-<time>.*` (legacy `run.*`).
* `flock` fallback (mkdir lock); environment check also verifies NumPy/PyTorch
  versions and `python -m mribreast --help`; `PYTHON_VENV_ROOT` may be a venv.
* Summary log names use `%x-%j` (no array).
