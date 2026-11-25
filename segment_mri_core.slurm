#!/bin/bash -l
# segment_mri_core.slurm -- MRI-CORE/SAM box-prompted lesion masks as one Helios GPU job.
#
# Login node:  bash segment_mri_core.slurm [options] [-- extra segment-mri-core arguments]
#   e.g.       bash segment_mri_core.slurm -- --limit 5 --previews 5      (smoke run first)
# Checks inputs, then submits ITSELF with sbatch. On the GPU node it loads MODULES, builds or reuses
# $PYTHON_VENV_ROOT/<arch>/segment-<hash>/ (the config's PYTHON_PACKAGES + einops + torchvision, with
# the module's NumPy/PyTorch as pip constraints) and runs
#   python -m mribreast prepare-data segment-mri-core ...
# Paths, MODULES, PYTHON_* and the account/partition come from the experiment config (default
# configs/duke_2d_mri_core_mask.conf); the output directory is dirname(PATHS[mask_manifest_file]), so
# the training config finds the manifest. Documentation: docs/slurm.md "Lesion segmentation".
set -euo pipefail
export LC_ALL=C
fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }
((BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 3))) || fail 'Bash >= 4.3 is required'

usage() {
  cat <<'HELP'
Usage: bash segment_mri_core.slurm [options] [-- extra segment-mri-core arguments]
  --config FILE          experiment config (default <project>/configs/duke_2d_mri_core_mask.conf)
  --mri-foundation DIR   mri_foundation checkout with pretrained_weights/ (default $MRI_FOUNDATION_REPO,
                         else <project>/../mri_foundation)
  --output-dir DIR       default: directory of PATHS[mask_manifest_file]
  --time HH:MM:SS        Slurm time limit (default 12:00:00)
  --dry-run              check inputs and print the sbatch and python commands; submit nothing
  --local                run here instead of submitting (same environment setup)
Extra arguments after -- go to segment-mri-core and override the defaults
(--weights mri_core --device cuda --batch_size 16 --previews 50), e.g. --limit 5 --overwrite.
HELP
}

SELF=$(realpath -- "${BASH_SOURCE[0]}")
PROJECT_ROOT=$(dirname -- "$SELF")
config= repo=${MRI_FOUNDATION_REPO:-} out_dir= time_limit=12:00:00 dry_run=false local_mode=false job_mode=false
extra=()
while (($#)); do
  case "$1" in
    --config|--mri-foundation|--output-dir|--time|--project-root)
      (($# >= 2)) || fail "$1 needs a value"
      case "$1" in
        --config) config=$2;; --mri-foundation) repo=$2;; --output-dir) out_dir=$2;; --time) time_limit=$2;;
        --project-root) PROJECT_ROOT=$(realpath -e -- "$2") || fail "Project root not found: $2";;
      esac
      shift 2;;
    --dry-run) dry_run=true; shift;;
    --local) local_mode=true; shift;;
    --job) job_mode=true; shift;;   # internal: running inside the Slurm job
    --help|-h) usage; exit 0;;
    --) shift; extra=("$@"); break;;
    *) fail "Unknown argument: $1 (see --help)";;
  esac
done
[[ "$time_limit" =~ ^([0-9]+-)?[0-9]+(:[0-9]{1,2}){0,2}$ ]] || fail "Invalid --time: $time_limit"

# ---------------------------------------------------------------- configuration
config=$(realpath -e -- "${config:-$PROJECT_ROOT/configs/duke_2d_mri_core_mask.conf}") || fail "Configuration not found: ${config:-<default>}"
config_dir=$(dirname -- "$config")
unset PYTHON_BIN PYTHON_VENV_ROOT AUTO_INSTALL_PYTHON_PACKAGES PYTHON_PACKAGES MODULES PATHS TRAIN JOB_TRAIN
declare -a PYTHON_PACKAGES=() MODULES=()
declare -A PATHS=() TRAIN=() JOB_TRAIN=()
# shellcheck disable=SC1090
source "$config"
PYTHON_BIN=${PYTHON_BIN:-python3}
PYTHON_VENV_ROOT=${PYTHON_VENV_ROOT:-}
AUTO_INSTALL_PYTHON_PACKAGES=${AUTO_INSTALL_PYTHON_PACKAGES:-false}
SEGMENT_PACKAGES=('einops>=0.7' torchvision)

resolve() { # path relative to $2 (default config dir) -> absolute
  local p=$1 base=${2:-$config_dir}
  [[ "$p" == /* || "$p" =~ ^[A-Za-z]:[/\\] ]] || p=$base/$p
  realpath -m -- "$p"
}
for key in clinical_features_file metadata_file annotation_boxes_file image_root; do
  [[ -n "${PATHS[$key]:-}" ]] || fail "PATHS[$key] is not set in $config"
  PATHS[$key]=$(resolve "${PATHS[$key]}")
done
[[ -n "$repo" ]] || repo=$PROJECT_ROOT/../mri_foundation
repo=$(resolve "$repo" "$PWD")
if [[ -n "$out_dir" ]]; then
  out_dir=$(resolve "$out_dir" "$PWD")
else
  [[ -n "${PATHS[mask_manifest_file]:-}" ]] || fail "PATHS[mask_manifest_file] is not set in $config (or pass --output-dir)"
  out_dir=$(dirname -- "$(resolve "${PATHS[mask_manifest_file]}")")
fi
[[ -z "$PYTHON_VENV_ROOT" ]] || PYTHON_VENV_ROOT=$(resolve "$PYTHON_VENV_ROOT")

# Every input must be readable before a job is queued (and again on the node).
preflight() {
  local key missing=()
  for key in clinical_features_file metadata_file annotation_boxes_file; do
    [[ -f "${PATHS[$key]}" ]] || missing+=("PATHS[$key]=${PATHS[$key]}")
  done
  [[ -d "${PATHS[image_root]}" ]] || missing+=("PATHS[image_root]=${PATHS[image_root]} (directory)")
  [[ -f "$repo/models/sam/__init__.py" ]] || missing+=("mri_foundation checkout: $repo/models/sam/__init__.py")
  for key in mri_foundation.pth sam_vit_b_01ec64.pth; do
    [[ -f "$repo/pretrained_weights/$key" ]] || missing+=("weights (git-ignored, copy them): $repo/pretrained_weights/$key")
  done
  ((${#missing[@]} == 0)) || fail "missing inputs:"$'\n'"$(printf '  %s\n' "${missing[@]}")"
}
preflight

# Cohort-defining values of the training config, so the masks cover the same studies.
COMMAND=(-m mribreast prepare-data segment-mri-core
  --clinical_features_file "${PATHS[clinical_features_file]}" --metadata_file "${PATHS[metadata_file]}")
for key in profile label_scheme phase_order phase_count; do
  [[ -z "${TRAIN[$key]:-}" ]] || COMMAND+=(--cohort_option "$key=${TRAIN[$key]}")
done
COMMAND+=(--annotation_boxes_file "${PATHS[annotation_boxes_file]}" --image_root "${PATHS[image_root]}"
  --mri_foundation_repo "$repo" --weights mri_core --device cuda --batch_size 16
  --output_dir "$out_dir" --previews 50 "${extra[@]}")

# ---------------------------------------------------------------- compute node
prepare_environment() { # -> PY
  local base constraints signature env_dir lock_fd installed
  base=$(command -v -- "$PYTHON_BIN") || fail "Python is unavailable: $PYTHON_BIN"
  if [[ "$AUTO_INSTALL_PYTHON_PACKAGES" != true ]]; then
    if [[ -z "$PYTHON_VENV_ROOT" ]]; then PY=$base
    elif [[ -f "$PYTHON_VENV_ROOT/pyvenv.cfg" ]]; then PY=$PYTHON_VENV_ROOT/bin/python
    else PY=$PYTHON_VENV_ROOT/$(uname -m)/bin/python; fi
    [[ -x "$PY" ]] || fail "Missing prepared environment: $PY (see docs/slurm.md)"
    "$PY" -c 'import torch, torchvision, einops' || fail "The prepared environment lacks einops/torchvision: $PY"
    return
  fi
  [[ -n "$PYTHON_VENV_ROOT" ]] || fail 'AUTO_INSTALL_PYTHON_PACKAGES=true requires PYTHON_VENV_ROOT'
  "$base" -c 'import numpy, torch' || fail 'Compute module NumPy/PyTorch cannot be imported with user-site disabled'
  constraints=$("$base" -m pip show numpy torch | awk '$1 == "Name:" {name = tolower($2)} $1 == "Version:" {print name "==" $2}')
  [[ "$constraints" == *numpy==* && "$constraints" == *torch==* ]] || fail 'Could not constrain module NumPy/PyTorch'
  signature=$({
    printf '%s\0' segment-v1 "$(readlink -f -- "$base")" "$("$base" --version 2>&1)" "$constraints"
    printf '%s\0' "${MODULES[@]}" -- "${PYTHON_PACKAGES[@]}" -- "${SEGMENT_PACKAGES[@]}"
  } | sha256sum); signature=${signature%% *}
  env_dir=$PYTHON_VENV_ROOT/$(uname -m)/segment-${signature:0:16}
  mkdir -p -- "$(dirname -- "$env_dir")"
  exec {lock_fd}>"$env_dir.lock"
  if command -v flock >/dev/null 2>&1; then flock -x "$lock_fd" || fail "Cannot lock $env_dir.lock"; fi
  if [[ ! -f "$env_dir/.ready" ]]; then
    printf 'Preparing segmentation environment: %s\n' "$env_dir"
    "$base" -m venv --system-site-packages --clear "$env_dir" || fail "Cannot create venv $env_dir"
    PY=$env_dir/bin/python
    printf '%s\n' "$constraints" > "$env_dir/module-constraints.txt"
    # torchvision is resolved to the release that matches the module's torch (pinned by the constraint).
    PIP_CONFIG_FILE=/dev/null PIP_USER=false PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 "$PY" -m pip install \
      --constraint "$env_dir/module-constraints.txt" "${PYTHON_PACKAGES[@]}" "${SEGMENT_PACKAGES[@]}" ||
      fail 'Dependency installation failed. If no torchvision wheel matches the module torch, check `module spider torchvision` and add that module to MODULES'
    "$PY" -m pip check || fail 'pip check failed in the segmentation environment'
    installed=$("$PY" -m pip show numpy torch | awk '$1 == "Name:" {name = tolower($2)} $1 == "Version:" {print name "==" $2}')
    [[ "$installed" == "$constraints" ]] || fail "Module NumPy/PyTorch were replaced: $installed"
    "$PY" -c 'import torch, torchvision, einops' || fail 'torch/torchvision/einops do not import together'
    "$PY" -m mribreast --help >/dev/null || fail 'python -m mribreast --help failed in the segmentation environment'
    touch -- "$env_dir/.ready"
  fi
  exec {lock_fd}>&-
  PY=$env_dir/bin/python
}

run_job() {
  local module_name
  cd -- "$PROJECT_ROOT"
  export PYTHONNOUSERSITE=1   # ~/.local is shared by x86 login and ARM GH200 nodes
  if ((${#MODULES[@]})); then
    declare -F module >/dev/null 2>&1 || command -v module >/dev/null 2>&1 || fail 'MODULES is set but the module command is unavailable'
    for module_name in "${MODULES[@]}"; do module load "$module_name" || fail "module load $module_name failed"; done
  fi
  export PYTHONNOUSERSITE=1
  if [[ "${OSTYPE:-}" == msys* || "${OSTYPE:-}" == cygwin* ]] && command -v cygpath >/dev/null 2>&1; then
    export PYTHONPATH="$(cygpath -m -- "$PROJECT_ROOT/src")${PYTHONPATH:+;$PYTHONPATH}"   # Git Bash --local only
  else
    export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  fi
  prepare_environment
  if [[ " ${extra[*]} " != *" --device "* ]]; then
    "$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' ||
      fail 'No CUDA device visible (request --gres=gpu:1, or pass -- --device cpu)'
  fi
  mkdir -p -- "$out_dir"
  printf 'MRI-CORE segmentation: repo %s (%s)\n' "$repo" "$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo 'no git')"
  printf '%q ' "$PY" "${COMMAND[@]}"; printf '\n'
  exec "$PY" "${COMMAND[@]}"
}

if [[ "$job_mode" == true || "$local_mode" == true ]]; then
  [[ "$dry_run" != true ]] || { printf '%q ' python "${COMMAND[@]}"; printf '\n'; exit 0; }
  run_job
fi

# ---------------------------------------------------------------- submit
SBATCH=(sbatch --parsable --job-name=mricore_seg --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
  "--time=$time_limit" --gres=gpu:1 "--chdir=$PROJECT_ROOT"
  "--output=$out_dir/slurm-%j.out" "--error=$out_dir/slurm-%j.err")
for key in account partition; do [[ -z "${JOB_TRAIN[$key]:-}" ]] || SBATCH+=("--$key=${JOB_TRAIN[$key]}"); done
ARGS=("$SELF" --job --project-root "$PROJECT_ROOT" --config "$config" --mri-foundation "$repo" --output-dir "$out_dir"
  --time "$time_limit" -- "${extra[@]}")
if [[ "$dry_run" == true ]]; then
  printf '%q ' "${SBATCH[@]}" "${ARGS[@]}"; printf '\n'
  printf 'Task: '; printf '%q ' '<segment env python>' "${COMMAND[@]}"; printf '\n'
  exit 0
fi
command -v sbatch >/dev/null 2>&1 || fail 'sbatch is unavailable (use --dry-run to inspect or --local to run here)'
mkdir -p -- "$out_dir"   # Slurm does not create the --output directory
job_id=$("${SBATCH[@]}" "${ARGS[@]}")
printf 'Submitted MRI-CORE segmentation: Slurm ID %s\nLog: %s/slurm-%s.out\nMask manifest: %s/mask_manifest.csv\n' \
  "$job_id" "$out_dir" "$job_id" "$out_dir"
