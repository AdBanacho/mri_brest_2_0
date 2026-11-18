#!/usr/bin/env bash
# Automatic compute-node environment: per-architecture env keyed by runtime +
# package set, serialized install (flock or mkdir fallback), module NumPy/PyTorch
# constraints verified, reuse, retry after failure, frozen package lists.
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"
AUTO=$(with_settings auto 'AUTO_INSTALL_PYTHON_PACKAGES=true' "PYTHON_VENV_ROOT='$TMP/managed-envs'" "PYTHON_BIN='$TMP/bin/bootstrap-python'")
installs() { grep -c '^install ' "$FAKE_BOOTSTRAP_LOG" 2>/dev/null || true; }
ready_count() { find "$TMP/managed-envs" -name .ready 2>/dev/null | wc -l | tr -d ' '; }

out=$(bash "$CONTROLLER" --train --config "$AUTO" --dry-run)
assert_contains "$out" 'monai==1.5.1'
[[ ! -e "$TMP/managed-envs" && ! -e "$FAKE_BOOTSTRAP_LOG" ]] || die 'dry run touched the environment'
pass 'dry run shows the package list without creating environments'

# Two concurrent first-use tasks: exactly one installation, both run.
FAKE_INSTALL_SLEEP=2 bash "$CONTROLLER" --train --config "$AUTO" --task-index 0 > "$TMP/w1.out" 2>&1 &
p1=$!
FAKE_INSTALL_SLEEP=2 bash "$CONTROLLER" --validate --config "$AUTO" --task-index 0 > "$TMP/w2.out" 2>&1 &
p2=$!
wait "$p1" || die "worker 1 failed: $(cat "$TMP/w1.out")"
wait "$p2" || die "worker 2 failed: $(cat "$TMP/w2.out")"
assert_eq "$(installs)" 1 'concurrent first use installs once'
assert_eq "$(py_calls)" 2 'both tasks ran'
log=$(cat "$FAKE_BOOTSTRAP_LOG")
assert_contains "$log" 'monai==1.5.1'
assert_contains "$log" 'filelock>=3.12'
assert_contains "$log" "import mribreast PYTHONPATH=$(native "$ROOT/src")"
env_dir=$(dirname "$(find "$TMP/managed-envs" -name .ready)")
[[ "$env_dir" == "$TMP/managed-envs/$(uname -m)/runtime-"* ]] || die "unexpected env dir $env_dir"
assert_eq "$(cat "$env_dir/module-constraints.txt")" $'numpy==1.26.4\ntorch==2.6.0' 'module constraints'
assert_contains "$(cat "$TMP/w1.out")" "$env_dir/bin/python -m mribreast train"
pass "concurrent first use installs once ($(command -v flock >/dev/null && echo flock || echo mkdir-lock fallback)); per-arch env with module constraints"

# Reuse for other stages; queued snapshots keep their package list.
bash "$CONTROLLER" --summarize --config "$AUTO" --task-index 0 > /dev/null
assert_eq "$(installs)" 1 'summary reuses env'
bash "$CONTROLLER" --train --config "$AUTO" > /dev/null
snap=$(sbatch_arg --snapshot); ctl=$(sbatch_controller)
printf '\nPYTHON_PACKAGES+=(extra-package==2)\n' >> "$AUTO"
SLURM_JOB_ID=1 SLURM_ARRAY_TASK_ID=0 bash "$ctl" --train --snapshot "$snap" --project-root "$ROOT" > /dev/null
assert_eq "$(installs)" 1 'frozen package list reuses env'
pass 'environment reused across stages and by queued snapshots after config edits'

# Changed package set -> new env; failures leave no .ready and are retried.
FAKE_INSTALL_FAIL=1 expect_fail_msg 'Dependency installation failed' bash "$CONTROLLER" --train --config "$AUTO" --task-index 0
assert_eq "$(ready_count)" 1 'failed install not marked ready'
FAKE_ENTRY_FAIL=1 expect_fail_msg 'python -m mribreast --help failed' bash "$CONTROLLER" --train --config "$AUTO" --task-index 0
assert_eq "$(ready_count)" 1 'failed entry point not marked ready'
FAKE_REPLACE_TORCH=1 expect_fail_msg 'Module NumPy/PyTorch were replaced' bash "$CONTROLLER" --train --config "$AUTO" --task-index 0
assert_eq "$(ready_count)" 1 'replaced torch not marked ready'
[[ -z "$(find "$TMP/managed-envs" -name '*.lock.d')" ]] || die 'stale mkdir lock left behind'
bash "$CONTROLLER" --train --config "$AUTO" --task-index 0 > /dev/null
assert_eq "$(ready_count)" 2 'retry after failure succeeds'
FAKE_IMPORT_FAIL=1 expect_fail_msg 'cannot be imported with user-site disabled' bash "$CONTROLLER" --train --config "$AUTO" --task-index 0
pass 'new package set -> new env; failed install/entry/constraint checks are retried; locks released'
