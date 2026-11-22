#!/bin/bash -l
# run_experiment.slurm -- Bash-only experiment controller AND Slurm batch task.
#
# Login node (no Python): reads experiment.conf + src/mribreast/config/fields.tsv,
# validates every value, expands the grid, writes a frozen run snapshot and
# submits ITSELF (the frozen copy) as a Slurm array.  Compute node: the frozen
# copy re-reads the verified snapshot, prepares the Python runtime and execs
#     $PYTHON_BIN -m mribreast <stage> --<key> <value> ...
# The same task path is used by --local, so local and cluster argv are identical.
# Documentation: docs/slurm.md.  Requires Bash >= 4.3 and GNU coreutils.
set -euo pipefail
export LC_ALL=C

fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }
warn() { printf 'Warning: %s\n' "$*" >&2; }
((BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 3))) || fail 'Bash >= 4.3 is required'

usage() {
  cat <<'HELP'
Usage: bash run_experiment.sh --prepare|--train|--validate|--summarize [options]
       bash run_experiment.sh --show-config [--train|--validate|...] [--config FILE]

Stages (exactly one):
  --prepare            Optional cohort/cache/mask audit stage (one task per training row)
  --train, --training  Training array (one task per training grid row)
  --validate, --validation
                       Validation array (training rows x validation grid)
  --summarize          Single summary/ranking job
Options:
  --config FILE        Shell configuration (default: <project>/experiment.conf)
  --dry-run            Print the plan and example task; no files, dirs, modules or Python
  --show-config        Readable preview of the resolved configuration and grid table
  --local              Run the tasks sequentially on this machine (no sbatch); same argv
                       as the Slurm tasks. With --task-index only that task runs.
  --dependency SPEC    Slurm dependency, e.g. afterok:12345
  --task-index N       Build (and with --dry-run only print) the command of task N
  --snapshot FILE      Internal: frozen experiment-<sha256>.conf used by queued tasks
  --project-root DIR   Project checkout used by tasks (default: controller directory)
  --help, -h           Show this help
HELP
}

SELF=$(realpath -- "${BASH_SOURCE[0]}")
PROJECT_ROOT=$(dirname -- "$SELF")
config= stage= dependency= task_index= snapshot= dry_run=false local_mode=false show_config=false
while (($#)); do
  case "$1" in
    --train|--training|--validate|--validation|--summarize|--prepare)
      [[ -z "$stage" ]] || fail 'Select exactly one stage'
      case "$1" in --train*) stage=train;; --valid*) stage=validate;; --prepare) stage=prepare;; *) stage=summarize;; esac
      shift;;
    --config|--dependency|--task-index|--snapshot|--project-root)
      (($# >= 2)) || fail "$1 needs a value"
      case "$1" in
        --config) config=$2;; --dependency) dependency=$2;;
        --task-index) task_index=$2;; --snapshot) snapshot=$2;;
        --project-root) PROJECT_ROOT=$(realpath -e -- "$2") || fail "Project root not found: $2";;
      esac
      shift 2;;
    --dry-run) dry_run=true; shift;;
    --local) local_mode=true; shift;;
    --show-config) show_config=true; shift;;
    --help|-h) usage; exit 0;;
    *) fail "Unknown argument: $1 (see --help)";;
  esac
done
if [[ -z "$stage" ]]; then
  [[ "$show_config" == true ]] || { usage >&2; exit 2; }
  stage=train
fi
[[ -z "$config" || -z "$snapshot" ]] || fail 'Use --config or --snapshot, not both'
[[ "$local_mode" != true || -z "$snapshot" ]] || fail '--local cannot be combined with --snapshot'
[[ "$local_mode" != true || -z "$dependency" ]] || fail '--dependency is a Slurm option; it has no meaning with --local'
[[ -z "$dependency" || "$dependency" =~ ^[A-Za-z0-9_:,?+@.-]+$ ]] || fail "Invalid --dependency: $dependency"
if [[ -n "$snapshot" && -z "$task_index" ]]; then
  [[ -n "${SLURM_JOB_ID:-}" ]] || fail 'A snapshot is only used by a Slurm task or with --task-index'
  task_index=${SLURM_ARRAY_TASK_ID:-0}
fi
[[ -z "$task_index" || "$task_index" =~ ^(0|[1-9][0-9]*)$ ]] || fail "Invalid task index: $task_index"

IS_WINDOWS=false
case "${OSTYPE:-}" in msys*|cygwin*) command -v cygpath >/dev/null 2>&1 && IS_WINDOWS=true;; esac
# Windows (Git Bash) only: hand native Python "C:/..." paths; a no-op on Linux.
native_path() { if [[ "$IS_WINDOWS" == true ]]; then cygpath -m -- "$1"; else printf '%s' "$1"; fi; }
is_abs() { [[ "$1" == /* || "$1" =~ ^[A-Za-z]:[/\\] ]]; }
sha256_of() { local s; s=$(sha256sum -- "$1") || fail "Cannot hash $1"; printf '%s' "${s%% *}"; }

# ---------------------------------------------------------------- configuration
run_dir=
if [[ -n "$snapshot" ]]; then
  config=$(realpath -e -- "$snapshot") || fail "Snapshot not found: $snapshot"
  actual=$(sha256_of "$config")
  [[ "${config##*/}" == "experiment-$actual.conf" ]] || fail "Configuration snapshot was changed (sha256 mismatch): $config"
  run_dir=$(dirname -- "$config")
else
  config=$(realpath -e -- "${config:-$PROJECT_ROOT/experiment.conf}") || fail "Configuration not found: ${config:-$PROJECT_ROOT/experiment.conf}"
fi
config_dir=$(dirname -- "$config")
CONFIG_VARS='PYTHON_BIN PYTHON_VENV_ROOT AUTO_INSTALL_PYTHON_PACKAGES PYTHON_PACKAGES MODULES MAX_JOBS MAX_CONCURRENT CLEAN_BEFORE_TRAIN CLEAN_BEFORE_VALIDATE PATHS JOB_PREPARE JOB_TRAIN JOB_VALIDATE JOB_SUMMARIZE EXTRA_SBATCH_PREPARE EXTRA_SBATCH_TRAIN EXTRA_SBATCH_VALIDATE EXTRA_SBATCH_SUMMARIZE TRAIN TRAIN_GRID VALIDATE VALIDATE_GRID SUMMARIZE FIELDS_SHA256 CODE_VERSION SNAPSHOT_SCHEMA'
# The environment must not leak values into the experiment definition.
# shellcheck disable=SC2086
unset $CONFIG_VARS
# Configurations are trusted shell code (sourced at global scope for declare -A).
# shellcheck disable=SC1090
source "$config"

PYTHON_BIN=${PYTHON_BIN:-python3}
PYTHON_VENV_ROOT=${PYTHON_VENV_ROOT:-}
AUTO_INSTALL_PYTHON_PACKAGES=${AUTO_INSTALL_PYTHON_PACKAGES:-false}
MAX_JOBS=${MAX_JOBS:-0}
MAX_CONCURRENT=${MAX_CONCURRENT:-4}
CLEAN_BEFORE_TRAIN=${CLEAN_BEFORE_TRAIN:-false}
CLEAN_BEFORE_VALIDATE=${CLEAN_BEFORE_VALIDATE:-false}
# One subshell lists the declarations (forks are slow on Git Bash).
DECLS=$'
'$(declare -p PYTHON_PACKAGES MODULES EXTRA_SBATCH_PREPARE EXTRA_SBATCH_TRAIN EXTRA_SBATCH_VALIDATE EXTRA_SBATCH_SUMMARIZE   PATHS TRAIN TRAIN_GRID VALIDATE VALIDATE_GRID SUMMARIZE JOB_TRAIN JOB_VALIDATE JOB_SUMMARIZE JOB_PREPARE 2>/dev/null || true)
array_kind() { # name -> declare flags (e.g. A, a) or none
  local re=$'
''declare -([A-Za-z-]+) '"$1"'(=|$|'$'
'')'
  if [[ "$DECLS" =~ $re ]]; then kind=${BASH_REMATCH[1]}; else kind=none; fi
}
for name in PYTHON_PACKAGES MODULES EXTRA_SBATCH_PREPARE EXTRA_SBATCH_TRAIN EXTRA_SBATCH_VALIDATE EXTRA_SBATCH_SUMMARIZE; do
  array_kind "$name"
  case "$kind" in none) declare -ga "$name=()";; *a*) ;; *) fail "$name must be a Bash array: $name=(...)";; esac
done
for name in PATHS TRAIN TRAIN_GRID VALIDATE VALIDATE_GRID SUMMARIZE JOB_TRAIN JOB_VALIDATE JOB_SUMMARIZE JOB_PREPARE; do
  array_kind "$name"
  case "$kind" in
    *A*) ;;
    none)
      case "$name" in JOB_TRAIN|JOB_VALIDATE|JOB_SUMMARIZE|PATHS) fail "Missing declare -A $name=(...)";; esac
      declare -gA "$name=()";;
    *) fail "$name must be an associative array: declare -A $name=(...)";;
  esac
done
if ((${#JOB_PREPARE[@]} == 0)); then
  # Optional prepare stage defaults to the training resources.
  for key in "${!JOB_TRAIN[@]}"; do JOB_PREPARE[$key]=${JOB_TRAIN[$key]}; done
  JOB_PREPARE[job_name]=mri_prepare
fi

[[ "$AUTO_INSTALL_PYTHON_PACKAGES" == true || "$AUTO_INSTALL_PYTHON_PACKAGES" == false ]] || fail 'AUTO_INSTALL_PYTHON_PACKAGES must be true or false'
if [[ "$AUTO_INSTALL_PYTHON_PACKAGES" == true ]]; then
  [[ -n "$PYTHON_VENV_ROOT" ]] || fail 'Automatic package installation requires PYTHON_VENV_ROOT'
  ((${#PYTHON_PACKAGES[@]} > 0)) || fail 'Automatic package installation requires PYTHON_PACKAGES'
fi
if [[ -n "$PYTHON_VENV_ROOT" ]]; then
  is_abs "$PYTHON_VENV_ROOT" || PYTHON_VENV_ROOT=$config_dir/$PYTHON_VENV_ROOT
  PYTHON_VENV_ROOT=$(realpath -m -- "$PYTHON_VENV_ROOT")
fi
[[ -n "$PYTHON_BIN" ]] || fail 'PYTHON_BIN is empty'
[[ "$MAX_JOBS" =~ ^(0|[1-9][0-9]*)$ ]] || fail 'MAX_JOBS must be an integer >= 0'
[[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]] || fail 'MAX_CONCURRENT must be an integer >= 1'
for boolean in "$CLEAN_BEFORE_TRAIN" "$CLEAN_BEFORE_VALIDATE"; do
  [[ "$boolean" == true || "$boolean" == false ]] || fail 'CLEAN_BEFORE_TRAIN/CLEAN_BEFORE_VALIDATE must be true or false'
done

# ---------------------------------------------------------------- field registry
# fields.tsv is the single field registry shared with mribreast.config.schema.
if [[ -n "$run_dir" ]]; then
  FIELDS_FILE=$run_dir/fields.tsv
  [[ -f "$FIELDS_FILE" && -n "${FIELDS_SHA256:-}" ]] || fail "Snapshot run directory lacks its frozen fields.tsv: $run_dir"
  [[ "$(sha256_of "$FIELDS_FILE")" == "$FIELDS_SHA256" ]] || fail "Frozen fields.tsv was changed: $FIELDS_FILE"
else
  FIELDS_FILE=$PROJECT_ROOT/src/mribreast/config/fields.tsv
  [[ -f "$FIELDS_FILE" ]] || fail "Field registry not found: $FIELDS_FILE (use --project-root)"
fi
IFS= read -r header < "$FIELDS_FILE" || fail "Empty field registry: $FIELDS_FILE"
[[ "${header%$'\r'}" == $'key\tsection\ttype\tdefault\tchoices\tidentity\tlegacy\thelp' ]] || fail "Unexpected fields.tsv header in $FIELDS_FILE"
declare -A F_SECTION=() F_TYPE=() F_DEFAULT=() F_CHOICES=() F_LEGACY=() F_ALIAS=()
FIELD_ORDER=()
# awk keeps empty columns (Bash read would merge consecutive tabs).
while IFS=$'\x1f' read -r key section type default choices identity legacy _help; do
  [[ -n "$key" ]] || continue
  [[ -z "${F_SECTION[$key]:-}" ]] || fail "fields.tsv: duplicate key $key"
  case "$section" in path|train|validate|summarize) ;; *) fail "fields.tsv: bad section '$section' for $key";; esac
  case "$type" in int|float|bool|str|choice|csv|ints|json|path) ;; *) fail "fields.tsv: unsupported type '$type' for $key";; esac
  F_SECTION[$key]=$section F_TYPE[$key]=$type F_DEFAULT[$key]=$default F_CHOICES[$key]=$choices F_LEGACY[$key]=$legacy
  [[ "$legacy" != renamed:* ]] || F_ALIAS[$section:${legacy#renamed:}]=$key
  FIELD_ORDER+=("$key")
done < <(awk -F'\t' 'NR > 1 { sub(/\r$/, ""); printf "%s\037%s\037%s\037%s\037%s\037%s\037%s\037%s\n", $1, $2, $3, $4, $5, $6, $7, $8 }' "$FIELDS_FILE")
((${#FIELD_ORDER[@]} > 0)) || fail "No fields in $FIELDS_FILE"
declare -A STAGE_SECTIONS=([prepare]='path train' [train]='path train' [validate]='path train validate' [summarize]='path summarize')
contains() { [[ " $1 " == *" $2 "* ]]; }
# Output roots are what cleanup may remove; every other path is protected.
OUTPUT_ROOT_KEYS='checkpoint_root logs_root validation_root summary_root'
for key in $OUTPUT_ROOT_KEYS slurm_logs; do [[ "${F_SECTION[$key]:-}" == path ]] || fail "fields.tsv lacks path field $key"; done

# Numeric ranges (not part of fields.tsv). [ ] inclusive, ( ) exclusive, empty = unbounded.
declare -A RANGE=(
  [seed]='[0,' [positive_class]='[0,' [phase_count]='[1,' [slice_axis]='[0,2]' [slices_per_study]='[1,'
  [annotation_erase_attempts]='[0,' [roi_margin]='[0,2]' [box_margin]='[0,2]' [dual_stream_embed_dim]='[1,' [lasso_cv_folds]='[2,' [lasso_cs]='[1,'
  [lasso_max_iter]='[1,' [lasso_tolerance]='(0,' [lasso_min_features]='[1,' [lasso_plot_top_n]='[0,'
  [xgb_n_estimators]='[1,' [xgb_max_depth]='[1,' [xgb_learning_rate]='(0,' [xgb_subsample]='(0,1]'
  [xgb_colsample_bytree]='(0,1]' [mlp_alpha]='[0,' [mlp_learning_rate]='(0,' [mlp_max_iter]='[1,'
  [num_folds]='[2,' [threshold_calibration_folds]='[2,' [epoch]='[1,' [batch_size]='[1,' [num_workers]='[0,'
  [positive_boost]='(0,' [sensitivity_lambda]='[0,' [lr]='(0,' [weight_decay]='[0,'
  [early_stopping_patience]='[0,' [early_stopping_min_delta]='[0,' [fusion_alpha]='[0,1]'
  [validate_num_workers]='[0,' [expected_folds]='[1,' [top_n]='[1,'
)
for key in "${!RANGE[@]}"; do
  [[ -n "${F_SECTION[$key]:-}" ]] || fail "Controller range table names a key missing from fields.tsv: $key"
done

# ---------------------------------------------------------------- value checks
RANGE_CHECKS=
check_value() { # key value context
  local key=$1 value=$2 ctx=$3 type=${F_TYPE[$1]} item word
  local -a items=() words=()
  [[ "$value" != *$'\n'* && "$value" != *$'\r'* ]] || fail "$ctx: value must be on one line"
  case "$type" in
    int) [[ "$value" =~ ^[+-]?[0-9]+$ ]] || fail "$ctx: '$value' is not an integer";;
    float) [[ "$value" =~ ^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$ ]] || fail "$ctx: '$value' is not a number";;
    bool) [[ "$value" == true || "$value" == false ]] || fail "$ctx: '$value' must be true or false";;
    choice) contains "${F_CHOICES[$key]//|/ }" "$value" && [[ -n "$value" ]] || fail "$ctx: '$value' not in {${F_CHOICES[$key]//|/, }}";;
    csv)
      [[ -n "$value" || -z "${F_DEFAULT[$key]}" ]] || fail "$ctx: needs at least one item"
      IFS=',' read -r -a items <<< "$value"
      for item in "${items[@]}"; do
        item=${item// /}
        [[ -n "$item" ]] || fail "$ctx: empty item in '$value'"
        [[ -z "${F_CHOICES[$key]}" ]] || contains "${F_CHOICES[$key]//|/ }" "$item" || fail "$ctx: item '$item' not in {${F_CHOICES[$key]//|/, }}"
      done;;
    ints)
      read -r -a words <<< "$value"
      ((${#words[@]} > 0)) || fail "$ctx: needs positive integers"
      for word in "${words[@]}"; do [[ "$word" =~ ^[1-9][0-9]*$ ]] || fail "$ctx: '$value' must be space-separated positive integers"; done;;
    json)
      word=${value#"${value%%[![:space:]]*}"}; word=${word%"${word##*[![:space:]]}"}
      [[ "$word" == '{'*'}' ]] || fail "$ctx: '$value' must be a JSON object {...}";;
    str|path) ;;
  esac
  [[ -z "${RANGE[$key]:-}" ]] || RANGE_CHECKS+="$ctx"$'\t'"$value"$'\t'"${RANGE[$key]}"$'\n'
}
# One awk process checks all numeric ranges collected above.
flush_range_checks() {
  local bad
  [[ -n "$RANGE_CHECKS" ]] || return 0
  bad=$(printf '%s' "$RANGE_CHECKS" | awk -F'\t' '
    { v = $2 + 0; r = $3; lo_inc = substr(r, 1, 1) == "["; hi_inc = substr(r, length(r), 1) == "]"
      split(substr(r, 2, length(r) - 2), b, ","); ok = 1
      if (b[1] != "") { if (lo_inc ? v < b[1] + 0 : v <= b[1] + 0) ok = 0 }
      if (b[2] != "") { if (hi_inc ? v > b[2] + 0 : v >= b[2] + 0) ok = 0 }
      if (!ok) { printf "%s: %s outside %s\n", $1, $2, r; exit } }')
  RANGE_CHECKS=
  [[ -z "$bad" ]] || fail "$bad"
}

# Rename legacy aliases (fields.tsv legacy column "renamed:<old>") and reject unknown keys.
check_keys() { # array section
  local -n values=$1
  local section=$2 key new other
  for key in "${!values[@]}"; do
    if [[ "${F_SECTION[$key]:-}" != "$section" && -n "${F_ALIAS[$section:$key]:-}" ]]; then
      new=${F_ALIAS[$section:$key]}
      [[ -z "${values[$new]+x}" ]] || fail "$1 sets both $key (legacy alias) and $new"
      values[$new]=${values[$key]}; unset "values[$key]"; key=$new
    fi
    other=${F_SECTION[$key]:-}
    [[ "$other" == "$section" ]] && continue
    if [[ -n "$other" ]]; then
      case "$other" in path) other=PATHS;; train) other='TRAIN or TRAIN_GRID';; validate) other='VALIDATE or VALIDATE_GRID';; summarize) other=SUMMARIZE;; esac
      fail "Unknown key in $1: $key (it belongs in $other)"
    fi
    fail "Unknown key in $1: $key (not in fields.tsv)"
  done
}
check_keys PATHS path
check_keys TRAIN train
check_keys TRAIN_GRID train
check_keys VALIDATE validate
check_keys VALIDATE_GRID validate
check_keys SUMMARIZE summarize
for key in "${!TRAIN_GRID[@]}"; do [[ -z "${TRAIN[$key]+x}" ]] || fail "TRAIN[$key] is also a TRAIN_GRID axis; keep it in one place"; done
for key in "${!VALIDATE_GRID[@]}"; do [[ -z "${VALIDATE[$key]+x}" ]] || fail "VALIDATE[$key] is also a VALIDATE_GRID axis; keep it in one place"; done

# Fill fields.tsv defaults so snapshots are fully explicit, then type-check.
fill_and_check() { # fixed grid section
  local -n fixed=$1 grid=$2
  local key
  for key in "${FIELD_ORDER[@]}"; do
    [[ "${F_SECTION[$key]}" == "$3" ]] || continue
    [[ -n "${fixed[$key]+x}" || -n "${grid[$key]+x}" ]] || fixed[$key]=${F_DEFAULT[$key]}
  done
  for key in "${!fixed[@]}"; do check_value "$key" "${fixed[$key]}" "$1[$key]"; done
}
declare -A NO_GRID=()
fill_and_check TRAIN TRAIN_GRID train
fill_and_check VALIDATE VALIDATE_GRID validate
fill_and_check SUMMARIZE NO_GRID summarize

# Paths: relative to the configuration file; legacy keys required, new keys optional.
path_keys=() path_values=()
for key in "${FIELD_ORDER[@]}"; do
  [[ "${F_SECTION[$key]}" == path ]] || continue
  value=${PATHS[$key]:-}
  if [[ -z "$value" ]]; then
    [[ "${F_LEGACY[$key]}" != legacy ]] || fail "Missing PATHS[$key]"
    PATHS[$key]=''
    continue
  fi
  check_value "$key" "$value" "PATHS[$key]"
  is_abs "$value" || value=$config_dir/$value
  path_keys+=("$key") path_values+=("$value")
done
if ((${#path_keys[@]})); then
  mapfile -t path_values < <(realpath -m -- "${path_values[@]}") || fail 'Cannot resolve PATHS'
  [[ "$IS_WINDOWS" != true ]] || mapfile -t path_values < <(cygpath -m -- "${path_values[@]}")
  ((${#path_values[@]} == ${#path_keys[@]})) || fail 'Cannot resolve PATHS'
  for i in "${!path_keys[@]}"; do PATHS[${path_keys[$i]}]=${path_values[$i]}; done
fi

for resource_stage in PREPARE TRAIN VALIDATE SUMMARIZE; do
  declare -n resources=JOB_$resource_stage
  for key in "${!resources[@]}"; do
    contains 'job_name cpus gpus memory time account partition' "$key" || fail "Unknown key in JOB_$resource_stage: $key"
  done
  [[ "${resources[cpus]:-}" =~ ^[1-9][0-9]*$ ]] || fail "JOB_$resource_stage[cpus] must be a positive integer"
  [[ "${resources[gpus]:-}" =~ ^(0|[1-9][0-9]*)$ ]] || fail "JOB_$resource_stage[gpus] must be a nonnegative integer"
  [[ "${resources[job_name]:-}" =~ ^[A-Za-z0-9_.-]+$ ]] || fail "JOB_$resource_stage[job_name] must be a simple name"
  [[ "${resources[memory]:-}" =~ ^[1-9][0-9]*[KMGT]?$ ]] || fail "JOB_$resource_stage[memory] must look like 48G"
  [[ "${resources[time]:-}" =~ ^([0-9]+-)?[0-9]+(:[0-9]{1,2}){0,2}$ ]] || fail "JOB_$resource_stage[time] must look like 24:00:00 or 1-00:00:00"
  unset -n resources
done

# ---------------------------------------------------------------- grid
# A grid value lists choices separated by |; commas stay inside a choice (csv
# keys) and spaces too (ints keys / multi-word strings). Keys are sorted and the
# LAST sorted key varies fastest (legacy select_grid order).
declare -A GN=() GC=()
parse_grid() { # NAME -> GN[NAME|key], GC[NAME|key|i], <NAME>_AXES (sorted)
  local -n grid=$1 axes=$1_AXES
  local key value choice total=1 i
  local -a choices=()
  local -A seen=()
  axes=()
  if ((${#grid[@]})); then mapfile -t axes < <(printf '%s\n' "${!grid[@]}" | sort); fi
  for key in "${axes[@]}"; do
    value=${grid[$key]}
    [[ "$value" != *$'\n'* && "$value" != *$'\r'* ]] || fail "Grid choices must be on one line: $1[$key]"
    [[ -n "$value" && "$value" != '|'* && "$value" != *'|' && "$value" != *'||'* ]] || fail "Empty grid choice: $1[$key]='$value'"
    IFS='|' read -r -a choices <<< "$value"
    seen=()
    for ((i = 0; i < ${#choices[@]}; i++)); do
      choice=${choices[$i]}
      [[ -z "${seen[$choice]:-}" ]] || fail "Duplicate grid choice: $1[$key]=$choice"
      seen[$choice]=1
      check_value "$key" "$choice" "$1[$key] choice '$choice'"
      GC[$1|$key|$i]=$choice
    done
    GN[$1|$key]=${#choices[@]}
    total=$((total * ${#choices[@]}))
    ((total <= 1000000)) || fail 'Grid exceeds 1,000,000 tasks'
  done
  GN[$1]=$total
}
TRAIN_GRID_AXES=() VALIDATE_GRID_AXES=()
parse_grid TRAIN_GRID
parse_grid VALIDATE_GRID
flush_range_checks
grid_total=${GN[TRAIN_GRID]}
train_count=$grid_total
if ((MAX_JOBS > 0 && MAX_JOBS < train_count)); then train_count=$MAX_JOBS; fi
validate_choices=${GN[VALIDATE_GRID]}
case "$stage" in
  prepare|train) count=$train_count;;
  validate) count=$((train_count * validate_choices));;
  summarize) count=1;;
esac

declare -A ROW=() VROW=()
select_row() { # NAME index fixed -> ROW/VROW
  local -n out=$4 fixed=$3
  local index=$2 key position n
  local -n axes=$1_AXES
  out=()
  for key in "${!fixed[@]}"; do out[$key]=${fixed[$key]}; done
  for ((position = ${#axes[@]} - 1; position >= 0; position--)); do
    key=${axes[$position]}; n=${GN[$1|$key]}
    out[$key]=${GC[$1|$key|$((index % n))]}
    index=$((index / n))
  done
}
has_item() { local v=${1// /}; [[ ",$v," == *",$2,"* ]]; }
nwords() { local -a w=(); read -r -a w <<< "$1"; NWORDS=${#w[@]}; }

# Cheap combination checks on one row (Python's schema.validate_combinations is
# authoritative and runs again on the compute node; it fails fast).
check_row() { # context
  local ctx=$1 branches=${ROW[branches]} image=false tabular=false n
  has_item "$branches" image && image=true
  has_item "$branches" tabular && tabular=true
  bad() { fail "$ctx: $*"; }
  if [[ "${ROW[label_scheme]}" == oncotype_3class_18_31 ]]; then n=3; else n=2; fi
  ((ROW[positive_class] < n)) || bad "positive_class=${ROW[positive_class]} outside the $n-class label range"
  if [[ "${ROW[profile]}" == duke_legacy_3d ]]; then
    [[ "${ROW[image_dim]}" == 3d && "${ROW[image_source]}" == nifti ]] || bad 'profile duke_legacy_3d requires image_dim=3d and image_source=nifti (use profile=general for 2D)'
    [[ "${ROW[mask_provider]}" == none ]] || bad 'profile duke_legacy_3d does not use external masks (use profile=general)'
  fi
  [[ "$tabular" == true || "${ROW[feature_selector]}" == none ]] || bad 'feature_selector requires the tabular branch'
  if [[ "$image" == true ]]; then
    if [[ "${ROW[image_dim]}" == 3d ]]; then
      nwords "${ROW[target_size]}"; ((NWORDS == 3)) || bad "target_size needs three integers for 3D, got '${ROW[target_size]}'"
      [[ "${ROW[image_source]}" == nifti ]] || bad 'image_dim=3d requires image_source=nifti'
    else
      nwords "${ROW[target_size_2d]}"; ((NWORDS == 2)) || bad "target_size_2d needs two integers for 2D, got '${ROW[target_size_2d]}'"
      [[ "${ROW[use_annotation_boxes]}" == false ]] || bad 'use_annotation_boxes is a 3D training augmentation; use masks for 2D'
      if [[ "${ROW[image_source]}" == manifest_2d ]]; then
        [[ "${ROW[subtraction_mode]}" == none ]] || bad 'ready 2D images (image_source=manifest_2d) need subtraction_mode=none'
        [[ "${ROW[slice_selection]}" == center ]] || bad 'slice_selection is for slicing NIfTI volumes; leave it at center for manifest_2d'
      fi
    fi
    [[ "${ROW[slice_selection]}" != mask_* || "${ROW[mask_provider]}" != none ]] || bad 'mask-based slice selection requires a mask_provider (masks must also exist at inference)'
    [[ "${ROW[mask_mode]}" == none || "${ROW[mask_provider]}" != none ]] || bad 'mask_mode requires a mask_provider'
    [[ "${ROW[mask_provenance]}" != fold_trained || "${ROW[mask_provider]}" == none || "${ROW[mask_kind]}" == predicted ]] || bad 'mask_provenance=fold_trained only applies to mask_kind=predicted'
    [[ "${ROW[mask_provider]}" != local_files || -n "${PATHS[mask_manifest_file]}" ]] || bad 'mask_provider=local_files needs PATHS[mask_manifest_file]'
    [[ "${ROW[image_source]}" != manifest_2d || -n "${PATHS[image_manifest_file]}" ]] || bad 'image_source=manifest_2d needs PATHS[image_manifest_file]'
    nwords "${ROW[box_local_size]}"; ((NWORDS == 3)) || bad "box_local_size needs three positive integers, got '${ROW[box_local_size]}'"
    if [[ "${ROW[box_mode]}" != none ]]; then
      [[ "${ROW[image_dim]}" == 3d && "${ROW[image_source]}" == nifti ]] || bad 'box_mode requires image_dim=3d and image_source=nifti'
      [[ -n "${PATHS[annotation_boxes_file]}" ]] || bad 'box_mode requires annotation_boxes_file (boxes are used at training AND inference)'
      [[ "${ROW[mask_mode]}" == none && "${ROW[slice_selection]}" != mask_* ]] || bad 'box_mode and mask_mode / mask-based slice selection are mutually exclusive (one source of lesion location)'
      [[ "${ROW[box_mode]}" != roi_crop || "${ROW[use_annotation_boxes]}" == false ]] || bad 'use_annotation_boxes cannot be combined with box_mode=roi_crop (the erased cuboid is defined on the full target grid)'
    fi
  else
    [[ "${ROW[use_annotation_boxes]}" == false && "${ROW[mask_mode]}" == none && "${ROW[box_mode]}" == none ]] || bad 'annotation boxes / masks need the image branch'
  fi
  if [[ "${ROW[selection_split]}" == outer ]]; then
    [[ "$image" == true && "$tabular" == false ]] || bad 'selection_split=outer reproduces the historical image-only protocol; use branches=image'
  fi
  [[ "$tabular" != true ]] || ! has_item "${ROW[feature_groups]}" density || [[ -n "${PATHS[density_file]}" ]] || bad 'feature group density needs PATHS[density_file]'
  [[ ! "${ROW[trainer_extra_json]}" =~ \"(callbacks|logger|default_root_dir|enable_checkpointing)\" ]] || bad 'trainer_extra_json cannot override callbacks/logger/default_root_dir/enable_checkpointing'
  if [[ "$stage" == validate ]]; then
    if [[ "${VROW[primary_branch]}" == fusion ]]; then [[ "$image" == true && "$tabular" == true ]] || bad 'primary_branch=fusion needs both branches'; fi
    if [[ "${VROW[primary_branch]}" == image || "${VROW[primary_branch]}" == tabular ]]; then
      has_item "$branches" "${VROW[primary_branch]}" || bad "primary_branch=${VROW[primary_branch]} was not trained"
    fi
    [[ $n == 2 || "${VROW[threshold_strategy]}" == fixed_0.5 ]] || bad 'threshold_strategy=balanced_accuracy is binary-only; use fixed_0.5 (argmax) for 3 classes'
  fi
  unset -f bad
}

# Grid axes that cannot influence a row (e.g. xgb_* for feature_model=mlp) only
# duplicate trainings: warn when partly inert, fail when inert in every row.
axis_active() { # key (reads ROW)
  local key=$1 b=${ROW[branches]}
  case "$key" in
    xgb_*) has_item "$b" tabular && [[ "${ROW[feature_model]}" == xgboost ]];;
    mlp_*) has_item "$b" tabular && [[ "${ROW[feature_model]}" == mlp ]];;
    lasso_*) has_item "$b" tabular && [[ "${ROW[feature_selector]}" == lasso ]];;
    feature_groups|feature_model|feature_selector|imaging_patient_id_column|allow_missing_imaging_features|include_sensitive|selector_preprocessing) has_item "$b" tabular;;
    target_size) has_item "$b" image && [[ "${ROW[image_dim]}" == 3d ]];;
    target_size_2d|slice_axis|slices_per_study|aggregation) has_item "$b" image && [[ "${ROW[image_dim]}" == 2d ]];;
    mask_kind|mask_mode|mask_roi_classes|mask_missing_policy|mask_empty_policy|mask_provenance) has_item "$b" image && [[ "${ROW[mask_provider]}" != none ]];;
    roi_margin) has_item "$b" image && [[ "${ROW[mask_mode]}" == roi_crop ]];;
    box_mode) has_item "$b" image;;
    box_missing_policy) has_item "$b" image && [[ "${ROW[box_mode]}" != none ]];;
    box_margin) has_item "$b" image && [[ "${ROW[box_mode]}" == roi_crop || "${ROW[box_mode]}" == dual_stream ]];;
    box_local_size|dual_stream_embed_dim) has_item "$b" image && [[ "${ROW[box_mode]}" == dual_stream ]];;
    early_stopping_*) has_item "$b" image && [[ "${ROW[early_stopping]}" == true ]];;
    mri_model|mri_extra_json|subtraction_mode|lr|batch_size|epoch|sensitivity_lambda|positive_boost|optimizer|weight_decay|scheduler|loss|use_annotation_boxes|trainer_extra_json|early_stopping) has_item "$b" image;;
    *) return 0;;
  esac
}
check_axes() {
  local key i inert
  local -A inert_rows=()
  for ((i = 0; i < train_count; i++)); do
    select_row TRAIN_GRID "$i" TRAIN ROW
    for key in "${TRAIN_GRID_AXES[@]}"; do
      ((GN[TRAIN_GRID|$key] > 1)) || continue
      axis_active "$key" || inert_rows[$key]=$((${inert_rows[$key]:-0} + 1))
    done
  done
  for key in "${!inert_rows[@]}"; do
    inert=${inert_rows[$key]}
    ((inert < train_count)) || fail "TRAIN_GRID[$key] has no effect in any row (model-specific field for a model/branch that is never selected); remove the axis"
    warn "TRAIN_GRID[$key] has no effect in $inert of $train_count training rows (duplicate trainings)"
  done
}

# ---------------------------------------------------------------- command
build_command() { # task index -> COMMAND
  local index=$1 key value train_index=$1 validation_index=0 section
  local -a words=()
  if [[ "$stage" == validate ]]; then
    train_index=$((index / validate_choices)); validation_index=$((index % validate_choices))
  fi
  if [[ "$stage" != summarize ]]; then
    select_row TRAIN_GRID "$train_index" TRAIN ROW
    VROW=()
    [[ "$stage" != validate ]] || select_row VALIDATE_GRID "$validation_index" VALIDATE VROW
    check_row "$stage task $index (training row $train_index)"
  fi
  COMMAND=("$PYTHON_BIN" -m mribreast "$stage")
  for key in "${FIELD_ORDER[@]}"; do
    section=${F_SECTION[$key]}
    contains "${STAGE_SECTIONS[$stage]}" "$section" || continue
    case "$section" in
      path) value=${PATHS[$key]};; train) value=${ROW[$key]};;
      validate) value=${VROW[$key]};; summarize) value=${SUMMARIZE[$key]};;
    esac
    if [[ "${F_TYPE[$key]}" == ints ]]; then
      read -r -a words <<< "$value"
      COMMAND+=("--$key" "${words[@]}")
    else
      COMMAND+=("--$key" "$value")
    fi
  done
}
print_command() { printf '%q ' "$@"; printf '\n'; }

# ---------------------------------------------------------------- compute runtime
code_version() {
  local sha dirty version
  if command -v git >/dev/null 2>&1 && sha=$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null); then
    dirty=$(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=no 2>/dev/null || true)
    printf 'git:%s%s' "$sha" "${dirty:+-dirty}"
    return
  fi
  version=$(sed -n 's/^__version__ *= *["'\'']\([^"'\'']*\)["'\''].*/\1/p' "$PROJECT_ROOT/src/mribreast/__init__.py" 2>/dev/null || true)
  printf 'package:%s' "${version:-unknown}"
}

venv_python() { # venv dir -> interpreter path (Linux bin/python, Windows Scripts/python.exe)
  if [[ -x "$1/bin/python" ]]; then printf '%s' "$1/bin/python"
  elif [[ -x "$1/Scripts/python.exe" ]]; then printf '%s' "$1/Scripts/python.exe"
  else return 1; fi
}

# Exclusive lock: flock when available, otherwise an atomic mkdir lock.
LOCK_FD= LOCK_DIR=
acquire_lock() { # lock file
  local waited=0 timeout=${ENV_LOCK_TIMEOUT:-7200}
  if command -v flock >/dev/null 2>&1; then
    exec {LOCK_FD}>"$1"
    flock -x "$LOCK_FD" || fail "Cannot lock $1"
    return
  fi
  LOCK_DIR=$1.d
  until mkdir -- "$LOCK_DIR" 2>/dev/null; do
    ((waited++ < timeout)) || fail "Timed out after ${timeout}s waiting for $LOCK_DIR (remove it if no setup is running)"
    sleep 1
  done
  printf '%s %s\n' "${HOSTNAME:-host}" "$$" > "$LOCK_DIR/owner"
  trap 'rm -rf -- "$LOCK_DIR"' EXIT
  trap 'exit 143' TERM INT
}
release_lock() {
  if [[ -n "$LOCK_FD" ]]; then exec {LOCK_FD}>&-; LOCK_FD=; fi
  if [[ -n "$LOCK_DIR" ]]; then rm -rf -- "$LOCK_DIR"; LOCK_DIR=; trap - EXIT TERM INT; fi
}

# One environment per compute architecture x runtime x package set. The module's
# NumPy/PyTorch are pip constraints and verified after installation; .ready is
# written only after all checks pass, so a failed setup is retried by later tasks.
prepare_python_environment() {
  local base_python constraints metadata runtime signature env_parent env_dir env_python installed
  base_python=$(command -v -- "$PYTHON_BIN") || fail "Python is unavailable: $PYTHON_BIN"
  "$base_python" -c 'import numpy, torch' || fail 'Compute module NumPy/PyTorch cannot be imported with user-site disabled'
  metadata=$("$base_python" -m pip show numpy torch) || fail 'Compute module must provide NumPy, PyTorch and pip'
  constraints=$(printf '%s\n' "$metadata" | awk '$1 == "Name:" {name = tolower($2)} $1 == "Version:" {print name "==" $2}')
  [[ "$constraints" == *numpy==* && "$constraints" == *torch==* ]] || fail 'Could not constrain module NumPy/PyTorch'
  runtime=$("$base_python" --version 2>&1)
  signature=$({
    printf '%s\0' bootstrap-v2 "$(readlink -f -- "$base_python")" "$runtime" "$constraints"
    printf '%s\0' "${MODULES[@]}" -- "${PYTHON_PACKAGES[@]}"
  } | sha256sum); signature=${signature%% *}
  env_parent=$PYTHON_VENV_ROOT/$(uname -m)
  env_dir=$env_parent/runtime-${signature:0:16}
  mkdir -p -- "$env_parent"
  acquire_lock "$env_parent/runtime-${signature:0:16}.lock"
  if [[ ! -f "$env_dir/.ready" ]]; then
    printf 'Preparing Python dependencies: %s\n' "$env_dir"
    "$base_python" -m venv --system-site-packages --clear "$env_dir" || fail "Cannot create venv $env_dir"
    env_python=$(venv_python "$env_dir") || fail "venv has no interpreter: $env_dir"
    printf '%s\n' "$constraints" > "$env_dir/module-constraints.txt"
    PIP_CONFIG_FILE=/dev/null PIP_USER=false PIP_DISABLE_PIP_VERSION_CHECK=1 "$env_python" -m pip install \
      --constraint "$env_dir/module-constraints.txt" "${PYTHON_PACKAGES[@]}" ||
      fail 'Dependency installation failed; module NumPy/PyTorch constraints were preserved'
    "$env_python" -m pip check || fail 'pip check failed in the new environment'
    installed=$("$env_python" -m pip show numpy torch | awk '$1 == "Name:" {name = tolower($2)} $1 == "Version:" {print name "==" $2}') ||
      fail 'Cannot query NumPy/PyTorch in the new environment'
    [[ "$installed" == "$constraints" ]] || fail "Module NumPy/PyTorch were replaced: $installed"
    "$env_python" -c 'import numpy, torch' || fail 'NumPy/PyTorch do not import in the new environment'
    # Import the real entry point without reading data.
    "$env_python" -m mribreast --help >/dev/null || fail 'python -m mribreast --help failed in the new environment'
    printf 'signature=%s\nruntime=%s\nmodules=%s\npackages=%s\n' "$signature" "$runtime" "${MODULES[*]}" "${PYTHON_PACKAGES[*]}" > "$env_dir/bootstrap.txt"
    touch -- "$env_dir/.ready"
  fi
  release_lock
  SELECTED_PYTHON=$(venv_python "$env_dir") || fail "Environment lost its interpreter: $env_dir"
}

run_task() { # executes COMMAND for the current task (never returns)
  local venv sep=':'
  cd -- "$PROJECT_ROOT"
  # ~/.local is shared by x86 login and ARM GH200 nodes: disable it before AND
  # after module loading (modules may reset the environment).
  export PYTHONNOUSERSITE=1
  if ((${#MODULES[@]})); then
    declare -F module >/dev/null 2>&1 || command -v module >/dev/null 2>&1 || fail 'MODULES is set but the module command is unavailable (use MODULES=() locally)'
    for module_name in "${MODULES[@]}"; do module load "$module_name" || fail "module load $module_name failed"; done
  fi
  export PYTHONNOUSERSITE=1
  # The package is used from the checkout (no install): src/ on PYTHONPATH.
  [[ "$IS_WINDOWS" != true ]] || sep=';'
  export PYTHONPATH="$(native_path "$PROJECT_ROOT/src")${PYTHONPATH:+$sep$PYTHONPATH}"
  if [[ "$AUTO_INSTALL_PYTHON_PACKAGES" == true ]]; then
    prepare_python_environment
    COMMAND[0]=$SELECTED_PYTHON
  elif [[ -n "$PYTHON_VENV_ROOT" ]]; then
    if [[ -f "$PYTHON_VENV_ROOT/pyvenv.cfg" ]]; then venv=$PYTHON_VENV_ROOT; else venv=$PYTHON_VENV_ROOT/$(uname -m); fi
    [[ -f "$venv/pyvenv.cfg" ]] && COMMAND[0]=$(venv_python "$venv") || fail "Missing compute-node environment: $venv (see docs/slurm.md)"
  else
    command -v -- "${COMMAND[0]}" >/dev/null 2>&1 || fail "Python is unavailable: ${COMMAND[0]}"
  fi
  printf 'Task %s/%s (%s) code=%s\n' "$1" "$count" "$stage" "${CODE_VERSION:-$(code_version)}"
  print_command "${COMMAND[@]}"
  exec "${COMMAND[@]}"
}

# ---------------------------------------------------------------- task mode
if [[ -n "$task_index" && "$local_mode" != true && "$show_config" != true ]]; then
  ((task_index < count)) || fail "Task index outside 0-$((count - 1))"
  build_command "$task_index"
  [[ "$dry_run" == true ]] || run_task "$task_index"
  print_command "${COMMAND[@]}"
  exit 0
fi

# ---------------------------------------------------------------- preview
show_configuration() {
  local key value mark i w line
  local -a cols=() widths=()
  printf 'Configuration : %s\n' "$config"
  printf 'Field registry: %s\n' "$FIELDS_FILE"
  printf 'Project root  : %s\n' "$PROJECT_ROOT"
  printf 'Runtime       : PYTHON_BIN=%s MODULES=(%s) AUTO_INSTALL=%s PYTHON_VENV_ROOT=%s\n' \
    "$PYTHON_BIN" "${MODULES[*]}" "$AUTO_INSTALL_PYTHON_PACKAGES" "${PYTHON_VENV_ROOT:-<none>}"
  printf '\nPaths (resolved; "!" = does not exist yet):\n'
  for key in "${FIELD_ORDER[@]}"; do
    [[ "${F_SECTION[$key]}" == path ]] || continue
    value=${PATHS[$key]}; mark=' '
    [[ -z "$value" || -e "$value" ]] || mark='!'
    printf '  %s %-28s %s\n' "$mark" "$key" "${value:-<unset>}"
  done
  printf '\nResources (%s):\n' "$stage"
  for key in PREPARE TRAIN VALIDATE SUMMARIZE; do
    declare -n r=JOB_$key
    printf '  %-9s name=%s cpus=%s gpus=%s mem=%s time=%s account=%s partition=%s\n' "$key" \
      "${r[job_name]}" "${r[cpus]}" "${r[gpus]}" "${r[memory]}" "${r[time]}" "${r[account]:-}" "${r[partition]:-}"
    unset -n r
  done
  for line in train:TRAIN validate:VALIDATE summarize:SUMMARIZE; do
    declare -n fixed=${line#*:}
    printf '\nFixed %s values (* = fields.tsv default):\n' "${line#*:}"
    for key in "${FIELD_ORDER[@]}"; do
      [[ "${F_SECTION[$key]}" == "${line%%:*}" && -n "${fixed[$key]+x}" ]] || continue
      mark=' '; [[ "${fixed[$key]}" != "${F_DEFAULT[$key]}" ]] || mark='*'
      printf '  %s %-30s %s\n' "$mark" "$key" "${fixed[$key]}"
    done
    unset -n fixed
  done
  printf '\nTRAIN_GRID axes (sorted; last varies fastest):\n'
  for key in "${TRAIN_GRID_AXES[@]}"; do printf '  %-30s %s  (%s)\n' "$key" "${TRAIN_GRID[$key]}" "${GN[TRAIN_GRID|$key]}"; done
  printf 'VALIDATE_GRID axes:\n'
  for key in "${VALIDATE_GRID_AXES[@]}"; do printf '  %-30s %s  (%s)\n' "$key" "${VALIDATE_GRID[$key]}" "${GN[VALIDATE_GRID|$key]}"; done
  printf '\nTraining rows: %s (grid %s, MAX_JOBS=%s); validation tasks: %s x %s = %s; MAX_CONCURRENT=%s\n' \
    "$train_count" "$grid_total" "$MAX_JOBS" "$train_count" "$validate_choices" "$((train_count * validate_choices))" "$MAX_CONCURRENT"
  cols=(row "${TRAIN_GRID_AXES[@]}")
  for ((w = 0; w < ${#cols[@]}; w++)); do widths[w]=${#cols[w]}; done
  for ((i = 0; i < train_count; i++)); do
    select_row TRAIN_GRID "$i" TRAIN ROW
    for ((w = 1; w < ${#cols[@]}; w++)); do value=${ROW[${cols[w]}]}; ((${#value} <= widths[w])) || widths[w]=${#value}; done
  done
  for ((w = 0; w < ${#cols[@]}; w++)); do printf '%-*s  ' "${widths[w]}" "${cols[w]}"; done; printf '\n'
  for ((i = 0; i < train_count; i++)); do
    select_row TRAIN_GRID "$i" TRAIN ROW
    printf '%-*s  ' "${widths[0]}" "$i"
    for ((w = 1; w < ${#cols[@]}; w++)); do printf '%-*s  ' "${widths[w]}" "${ROW[${cols[w]}]}"; done
    printf '\n'
  done
  printf '\nPython re-validates every combination on the compute node and fails fast.\n'
}

# ---------------------------------------------------------------- planning
# Preflight every row before any filesystem change or cleanup.
for ((index = 0; index < count; index++)); do build_command "$index"; done
[[ "$stage" == summarize ]] || check_axes
if [[ "$show_config" == true ]]; then show_configuration; exit 0; fi
build_command "${task_index:-0}"
upper_stage=${stage^^}
declare -n job=JOB_$upper_stage extra_sbatch=EXTRA_SBATCH_$upper_stage
# Managed options belong in JOB_* / MAX_CONCURRENT, not EXTRA_SBATCH_*.
for option in "${extra_sbatch[@]}"; do
  [[ "$option" == --*=* ]] || fail 'EXTRA_SBATCH options must use --name=value'
  case "${option%%=*}" in
    --qos|--constraint|--reservation|--nice|--mail-type|--mail-user) ;;
    *) fail "Unsupported EXTRA_SBATCH option: $option";;
  esac
done
log_pattern='%x-%A_%a'; [[ "$stage" != summarize ]] || log_pattern='%x-%j'
SBATCH=(sbatch --parsable "--job-name=${job[job_name]}" --nodes=1 --ntasks=1
  "--cpus-per-task=${job[cpus]}" "--mem=${job[memory]}" "--time=${job[time]}"
  "--output=${PATHS[slurm_logs]}/$log_pattern.out" "--error=${PATHS[slurm_logs]}/$log_pattern.err")
((job[gpus] == 0)) || SBATCH+=("--gres=gpu:${job[gpus]}")
for key in account partition; do [[ -z "${job[$key]:-}" ]] || SBATCH+=("--$key=${job[$key]}"); done
SBATCH+=("${extra_sbatch[@]}")
[[ "$stage" == summarize ]] || SBATCH+=("--array=0-$((count - 1))%$MAX_CONCURRENT")
[[ -z "$dependency" ]] || SBATCH+=("--dependency=$dependency")

# Cleanup happens once in the submitting controller, never in array tasks.
cleanup_targets=()  # PATHS keys
if [[ "$stage" == train && "$CLEAN_BEFORE_TRAIN" == true ]]; then
  cleanup_targets=(checkpoint_root logs_root)
elif [[ "$stage" == validate && "$CLEAN_BEFORE_VALIDATE" == true ]]; then
  cleanup_targets=(validation_root)
  [[ "${PATHS[summary_root]}" == "${PATHS[validation_root]}"/* ]] || cleanup_targets+=(summary_root)
fi
if ((${#cleanup_targets[@]})); then
  # All checks use POSIX forms (on Git Bash "/" becomes C:/Program Files/Git/).
  declare -A POSIX=()
  check_keys_list=() check_values=()
  for key in "${FIELD_ORDER[@]}"; do
    [[ "${F_SECTION[$key]}" == path && -n "${PATHS[$key]}" ]] || continue
    check_keys_list+=("$key") check_values+=("${PATHS[$key]}")
  done
  if [[ "$IS_WINDOWS" == true ]]; then
    mapfile -t check_values < <(cygpath -u -- "${check_values[@]}" "${HOME:-/}" "$PROJECT_ROOT")
    mapfile -t check_values < <(realpath -m -- "${check_values[@]}")
  else
    mapfile -t check_values < <(realpath -m -- "${check_values[@]}" "${HOME:-/}" "$PROJECT_ROOT")
  fi
  for i in "${!check_keys_list[@]}"; do POSIX[${check_keys_list[$i]}]=${check_values[$i]}; done
  home_dir=${check_values[${#check_keys_list[@]}]} root_dir=${check_values[${#check_keys_list[@]} + 1]}
  for target_key in "${cleanup_targets[@]}"; do
    target=${POSIX[$target_key]}
    [[ "$target" != / && "$target" != "$home_dir" && "$root_dir" != "$target" && "$root_dir" != "$target"/* && "$home_dir" != "$target"/* ]] || fail "Refusing broad cleanup: ${PATHS[$target_key]}"
    # At least three path components; no overlap with inputs, caches, Slurm
    # logs/snapshots, or other output roots.
    [[ "$target" == /*/*/* ]] || fail "Refusing broad cleanup: ${PATHS[$target_key]}"
    for key in "${check_keys_list[@]}"; do
      [[ "$key" != "$target_key" ]] || continue
      protected=${POSIX[$key]}
      if contains "$OUTPUT_ROOT_KEYS" "$key"; then
        # A summary nested inside validation_root is the only permitted overlap.
        [[ "$stage" != validate || "$key" != summary_root || "$protected" != "$target"/* ]] || continue
        [[ "$protected" == "$target" ]] && continue
      fi
      [[ "$protected" != "$target" && "$protected" != "$target"/* && "$target" != "$protected"/* ]] || fail "Cleanup overlaps $key: ${PATHS[$target_key]}"
    done
    [[ ! -e "$target" || -d "$target" ]] || fail "Cleanup target is not a directory: ${PATHS[$target_key]}"
  done
fi

if [[ "$dry_run" == true ]]; then
  printf '%s: %s job(s), max concurrent %s\n' "$stage" "$count" "$MAX_CONCURRENT"
  if [[ "$AUTO_INSTALL_PYTHON_PACKAGES" == true ]]; then
    printf 'Compute-node environment root: %s\nPackages: ' "$PYTHON_VENV_ROOT"
    print_command "${PYTHON_PACKAGES[@]}"
  fi
  for key in "${cleanup_targets[@]}"; do printf 'Would remove: %s\n' "${PATHS[$key]}"; done
  if [[ "$local_mode" == true ]]; then
    start=0 end=$count
    [[ -z "$task_index" ]] || { ((task_index < count)) || fail "Task index outside 0-$((count - 1))"; start=$task_index; end=$((task_index + 1)); }
    for ((index = start; index < end; index++)); do
      build_command "$index"
      printf 'Local task %s: ' "$index"; print_command "${COMMAND[@]}"
    done
  else
    print_command "${SBATCH[@]}" '<frozen-controller.slurm>' --"$stage" --snapshot '<resolved-experiment.conf>' --project-root "$PROJECT_ROOT"
    printf 'Example task %s: ' "${task_index:-0}"; print_command "${COMMAND[@]}"
  fi
  exit 0
fi

# ---------------------------------------------------------------- snapshot + submit
if [[ "$local_mode" != true ]]; then
  command -v sbatch >/dev/null 2>&1 || fail 'sbatch is unavailable (use --dry-run to inspect or --local to run here)'
fi
[[ -z "$task_index" || "$local_mode" == true ]] || fail 'Use --task-index with --dry-run or --local'
mkdir -p -- "${PATHS[slurm_logs]}"
run_dir=$(mktemp -d "${PATHS[slurm_logs]}/run-$stage-$(date +%Y%m%d-%H%M%S).XXXXXX")
cp -- "$FIELDS_FILE" "$run_dir/fields.tsv"
FIELDS_SHA256=$(sha256_of "$run_dir/fields.tsv")
CODE_VERSION=$(code_version)
SNAPSHOT_SCHEMA=1
# Serialize resolved shell values (absolute paths, defaults filled in).
{
  printf '# Frozen resolved configuration (%s). Verified by sha256 file name; do not edit.\n' "$stage"
  # shellcheck disable=SC2086
  declare -p $CONFIG_VARS
} > "$run_dir/resolved.conf"
digest=$(sha256_of "$run_dir/resolved.conf")
snapshot_path=$run_dir/experiment-$digest.conf
mv -- "$run_dir/resolved.conf" "$snapshot_path"
cp -- "$SELF" "$run_dir/run_experiment.slurm"
{
  printf 'code_version=%s\n' "$CODE_VERSION"
  printf 'package_version=%s\n' "$(sed -n 's/^__version__ *= *"\(.*\)".*/\1/p' "$PROJECT_ROOT/src/mribreast/__init__.py" 2>/dev/null || true)"
  printf 'project_root=%s\nconfig=%s\nstage=%s\ntasks=%s\nmode=%s\n' "$PROJECT_ROOT" "$config" "$stage" "$count" "$([[ "$local_mode" == true ]] && echo local || echo slurm)"
  printf 'controller_sha256=%s\nfields_sha256=%s\nsnapshot_sha256=%s\n' "$(sha256_of "$run_dir/run_experiment.slurm")" "$FIELDS_SHA256" "$digest"
  printf 'created=%s\nuser=%s\nhost=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${USER:-${USERNAME:-unknown}}" "${HOSTNAME:-unknown}"
} > "$run_dir/code_version.txt"
chmod a-w -- "$snapshot_path" "$run_dir/run_experiment.slurm" "$run_dir/fields.tsv" "$run_dir/code_version.txt"
for key in "${cleanup_targets[@]}"; do printf 'Removing previous output: %s\n' "${PATHS[$key]}"; rm -rf -- "${PATHS[$key]}"; done

if [[ "$local_mode" == true ]]; then
  start=0 end=$count
  if [[ -n "$task_index" ]]; then
    ((task_index < count)) || fail "Task index outside 0-$((count - 1))"
    start=$task_index end=$((task_index + 1))
  fi
  printf 'Local %s: tasks %s-%s of %s; snapshot %s\n' "$stage" "$start" "$((end - 1))" "$count" "$snapshot_path"
  for ((index = start; index < end; index++)); do
    printf '=== %s task %s ===\n' "$stage" "$index"
    bash "$run_dir/run_experiment.slurm" --"$stage" --snapshot "$snapshot_path" --task-index "$index" \
      --project-root "$PROJECT_ROOT" 2>&1 | tee "$run_dir/local-$stage-$index.log"
  done
  printf 'Local %s finished: %s task(s); logs in %s\n' "$stage" "$((end - start))" "$run_dir"
  exit 0
fi
job_id=$("${SBATCH[@]}" "$run_dir/run_experiment.slurm" --"$stage" --snapshot "$snapshot_path" --project-root "$PROJECT_ROOT")
printf '%s\n' "$job_id" > "$run_dir/slurm_job_id"
printf 'Submitted %s: Slurm ID %s (%s task(s), max concurrent %s)\nConfiguration: %s\n' "$stage" "$job_id" "$count" "$MAX_CONCURRENT" "$snapshot_path"
