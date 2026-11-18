#!/usr/bin/env bash
# Submission, frozen snapshots, task runtime, local mode and cleanup safety.
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"

# Submission passes resources, throttle and dependency; freezes config + controller.
bash "$LAUNCHER" --train --config "$CONFIG" --dependency afterok:777 > "$TMP/submit.out"
submission=$(cat "$FAKE_SBATCH_LOG")
assert_contains "$submission" '--array=0-1%4'
assert_contains "$submission" '--gres=gpu:1'
assert_contains "$submission" '--cpus-per-task=16'
assert_contains "$submission" '--mem=48G'
assert_contains "$submission" '--time=24:00:00'
assert_contains "$submission" '--account=plgvirtudrel2026-gpu-gh200'
assert_contains "$submission" '--partition=plgrid-gpu-gh200'
assert_contains "$submission" '--dependency=afterok:777'
assert_contains "$(cat "$TMP/submit.out")" 'Submitted train: Slurm ID 12345'
snapshot=$(sbatch_arg --snapshot)
controller=$(sbatch_controller)
run_dir=$(dirname -- "$snapshot")
[[ -f "$snapshot" && -f "$controller" && -f "$run_dir/fields.tsv" && -f "$run_dir/code_version.txt" ]] || die 'run dir incomplete'
[[ "$(basename "$snapshot")" == "experiment-$(sha256sum "$snapshot" | cut -d' ' -f1).conf" ]] || die 'snapshot name is not its sha256'
[[ ! -w "$snapshot" || "${OSTYPE:-}" == msys* ]] || die 'snapshot is writable'
assert_eq "$(sha256sum < "$controller" | cut -d' ' -f1)" "$(sha256sum < "$CONTROLLER" | cut -d' ' -f1)" 'frozen controller copy'
assert_contains "$(cat "$run_dir/code_version.txt")" 'code_version='
grep -Eq '^code_version=(git:[0-9a-f]{40}(-dirty)?|package:.+)$' "$run_dir/code_version.txt" || die 'bad code_version'
assert_eq "$(cat "$run_dir/slurm_job_id")" 12345 'job id recorded'
pass 'submission: resources, array throttle, dependency, frozen snapshot/controller/fields, code version'

# Queued tasks ignore later edits of experiment.conf; the task runs from the
# Slurm spool copy in another cwd, with user site disabled around module load.
printf '\nTRAIN[epoch]=99\nPATHS[clinical_features_file]=/elsewhere.xlsx\n' >> "$CONFIG"
cp "$controller" "$TMP/work/slurm_script"
reset_py
(cd "$TMP" && PYTHONNOUSERSITE=0 SLURM_JOB_ID=12345 SLURM_ARRAY_TASK_ID=1 bash "$TMP/work/slurm_script" --train --snapshot "$snapshot" --project-root "$ROOT") > "$TMP/worker.out"
args=$(last_py); env_file=${args%.lines}.env
assert_eq "$(arg_value --epoch "$args")" 30 'queued snapshot epoch'
assert_eq "$(arg_value --clinical_features_file "$args")" "$(native "$TMP/config with spaces/data/duke/Clinical_and_Other_Features.xlsx")" 'queued snapshot path'
assert_eq "$(arg_value --sensitivity_lambda "$args")" 0.2 'array task 1 row'
assert_eq "$(cat "$FAKE_MODULE_LOG")" 'load stub-module PYTHONNOUSERSITE=1' 'user site disabled BEFORE module load'
assert_contains "$(cat "$env_file")" 'PYTHONNOUSERSITE=1'   # fake module reset it to 0 -> re-exported after
assert_contains "$(cat "$env_file")" "cwd=$ROOT"
assert_contains "$(cat "$env_file")" "PYTHONPATH=$(native "$ROOT/src")"
pass 'queued snapshot is immutable after editing experiment.conf; PYTHONNOUSERSITE=1 before and after module load'

# Tampering is detected before anything runs.
chmod u+w "$snapshot"
cp "$snapshot" "$TMP/snapshot.orig"
printf '\ntouch "%s"\n' "$TMP/tampered" >> "$snapshot"
expect_fail_msg 'snapshot was changed' env SLURM_JOB_ID=1 bash "$controller" --train --snapshot "$snapshot" --project-root "$ROOT"
[[ ! -e "$TMP/tampered" ]] || die 'tampered snapshot was sourced'
cp "$TMP/snapshot.orig" "$snapshot"
chmod u+w "$run_dir/fields.tsv"
printf 'extra\ttrain\tint\t1\t\t1\tnew\tx\n' >> "$run_dir/fields.tsv"
expect_fail_msg 'Frozen fields.tsv was changed' env SLURM_JOB_ID=1 bash "$controller" --train --snapshot "$snapshot" --project-root "$ROOT"
expect_fail_msg 'only used by a Slurm task' bash "$controller" --train --snapshot "$snapshot" --project-root "$ROOT"
pass 'snapshot and frozen fields.tsv tampering are rejected before sourcing/running'

# Local mode runs the same frozen task path: argv identical to the Slurm task argv.
LOCAL=$(with_settings local 'TRAIN[epoch]=30' "PATHS[clinical_features_file]=data/duke/Clinical_and_Other_Features.xlsx")
for stage in train validate summarize prepare; do
  bash "$CONTROLLER" --"$stage" --config "$LOCAL" > /dev/null
  snap=$(sbatch_arg --snapshot); ctl=$(sbatch_controller)
  n=$(sed -n 's/^--array=0-\([0-9]*\)%.*/\1/p' "$FAKE_SBATCH_LOG"); n=$(( ${n:-0} + 1 ))
  reset_py
  for ((i = 0; i < n; i++)); do SLURM_JOB_ID=9 SLURM_ARRAY_TASK_ID=$i bash "$ctl" --"$stage" --snapshot "$snap" --project-root "$ROOT" > /dev/null; done
  mkdir -p "$TMP/slurm-argv"; rm -f "$TMP/slurm-argv/"*; cp "$FAKE_PY_LOG"/*.argv "$TMP/slurm-argv/"
  reset_py
  bash "$LAUNCHER" --local --"$stage" --config "$LOCAL" > "$TMP/local.out"
  assert_contains "$(cat "$TMP/local.out")" "Local $stage finished: $n task(s)"
  assert_eq "$(py_calls)" "$n" "$stage local task count"
  for f in "$TMP/slurm-argv/"*.argv; do
    cmp -s "$f" "$FAKE_PY_LOG/$(basename "$f")" || die "$stage: local argv differs from Slurm argv ($(basename "$f"))"
  done
done
reset_py
bash "$LAUNCHER" --local --validate --task-index 3 --config "$LOCAL" > /dev/null
assert_eq "$(py_calls)" 1 'local single task'
assert_eq "$(arg_value --fusion_alpha "$(last_py)")" 0.75 'local task 3 alpha'
FAILPY=$(with_settings failpy 'MAX_JOBS=2')
reset_py
FAKE_PY_EXIT=3 expect_fail bash "$LAUNCHER" --local --train --config "$FAILPY"
assert_eq "$(py_calls)" 1 'local mode stops at the first failing task'
pass 'local mode argv == Slurm task argv for all stages; single task; stops on failure'

# Summary: one CPU task, no array/gres, dependency kept.
bash "$LAUNCHER" --summarize --config "$LOCAL" --dependency afterok:12345 > /dev/null
submission=$(cat "$FAKE_SBATCH_LOG")
assert_not_contains "$submission" '--gres='
assert_not_contains "$submission" '--array='
assert_contains "$submission" '--dependency=afterok:12345'
assert_contains "$submission" '--cpus-per-task=2'
pass 'summary submission: single CPU job with dependency'

# Prepared environments: direct venv, per-architecture venv, or missing -> error.
venv_root="$TMP/config with spaces/envs"
mkdir -p "$venv_root/$(uname -m)/bin"; touch "$venv_root/$(uname -m)/pyvenv.cfg"
cp "$TMP/bin/python" "$venv_root/$(uname -m)/bin/python"
VENV=$(with_settings venv 'PYTHON_VENV_ROOT=envs' 'PYTHON_BIN=/missing/interpreter')
for stage in train validate summarize; do
  reset_py
  bash "$CONTROLLER" --"$stage" --config "$VENV" --task-index 0 > /dev/null
  assert_contains "$(cat "$(last_py | sed 's/lines$/env/')")" "interpreter=$venv_root/$(uname -m)/bin/python"
done
mkdir -p "$TMP/direct/bin"; touch "$TMP/direct/pyvenv.cfg"; cp "$TMP/bin/python" "$TMP/direct/bin/python"
DIRECT=$(with_settings direct "PYTHON_VENV_ROOT='$TMP/direct'")
reset_py
bash "$CONTROLLER" --train --config "$DIRECT" --task-index 0 > /dev/null
assert_contains "$(cat "$(last_py | sed 's/lines$/env/')")" "interpreter=$TMP/direct/bin/python"
rm "$venv_root/$(uname -m)/pyvenv.cfg"
expect_fail_msg 'Missing compute-node environment' bash "$CONTROLLER" --train --config "$VENV" --task-index 0
out=$(bash "$CONTROLLER" --train --config "$VENV" --task-index 0 --dry-run)
assert_contains "$out" '/missing/interpreter'
NOMOD=$(with_settings nomod 'MODULES=(x)')
(unset -f module; expect_fail_msg 'module command is unavailable' bash "$CONTROLLER" --train --config "$NOMOD" --task-index 0)
pass 'prepared env: PYTHON_VENV_ROOT/<arch> or a direct venv; missing env / module command fail clearly'

# Cleanup: explicit flag, once in the controller, never in array tasks; safety checks.
CLEAN=$(with_settings clean 'CLEAN_BEFORE_TRAIN=true')
mkdir -p "$TMP/work/checkpoints" "$TMP/work/tensorboard"; touch "$TMP/work/checkpoints/old" "$TMP/work/tensorboard/old"
bash "$LAUNCHER" --train --config "$CLEAN" > /dev/null
[[ ! -e "$TMP/work/checkpoints" && ! -e "$TMP/work/tensorboard" ]] || die 'cleanup did not run'
snap=$(sbatch_arg --snapshot); ctl=$(sbatch_controller)
mkdir -p "$TMP/work/checkpoints"; touch "$TMP/work/checkpoints/new"
SLURM_JOB_ID=1 SLURM_ARRAY_TASK_ID=0 bash "$ctl" --train --snapshot "$snap" --project-root "$ROOT" > /dev/null
[[ -f "$TMP/work/checkpoints/new" ]] || die 'array task cleaned outputs'
NOCLEAN=$(with_settings noclean 'CLEAN_BEFORE_TRAIN=false')
bash "$LAUNCHER" --train --config "$NOCLEAN" > /dev/null
[[ -f "$TMP/work/checkpoints/new" ]] || die 'cleanup without flag'
for setting in "PATHS[checkpoint_root]='$ROOT'" "PATHS[checkpoint_root]='$TMP/work/logs'" 'PATHS[checkpoint_root]=/' \
  "PATHS[checkpoint_root]='$HOME'" "PATHS[checkpoint_root]='$TMP/work'" "PATHS[logs_root]='$TMP/work/checkpoints/tb'" \
  "PATHS[checkpoint_root]='$TMP/config with spaces/data'"; do
  file=$(with_settings badclean 'CLEAN_BEFORE_TRAIN=true' "$setting")
  expect_fail bash "$CONTROLLER" --train --config "$file" --dry-run
  assert_contains "$(cat "$TMP/failure.out")" 'Error: '
  assert_contains "$(cat "$TMP/failure.out")" 'leanup'
done
file=$(with_settings badclean 'CLEAN_BEFORE_VALIDATE=true' "PATHS[summary_root]='$TMP/work'")
expect_fail_msg 'Cleanup overlaps' bash "$CONTROLLER" --validate --config "$file" --dry-run
touch "$TMP/plainfile"
file=$(with_settings badclean 'CLEAN_BEFORE_TRAIN=true' "PATHS[checkpoint_root]='$TMP/plainfile'")
expect_fail_msg 'not a directory' bash "$CONTROLLER" --train --config "$file" --dry-run
pass 'cleanup only with explicit flag, once before submission, with broad/overlap/input safety checks'
