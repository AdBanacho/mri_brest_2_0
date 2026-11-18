# Shared helpers for the controller tests (sourced, Bash). Uses fake sbatch /
# python / module from tests/slurm/fakes; never needs Slurm, Python or data.
set -euo pipefail
export LC_ALL=C
SLURM_TEST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SLURM_TEST_DIR/../.." && pwd)
CONTROLLER=$ROOT/run_experiment.slurm
LAUNCHER=$ROOT/run_experiment.sh
TMP=$(mktemp -d "${TMPDIR:-/tmp}/mribreast-slurm-test.XXXXXXXX")
cleanup_tmp() { chmod -R u+w -- "$TMP" 2>/dev/null || true; rm -rf -- "$TMP"; }
trap cleanup_tmp EXIT
mkdir -p "$TMP/bin" "$TMP/config with spaces" "$TMP/work"
cp "$SLURM_TEST_DIR/fakes/sbatch" "$SLURM_TEST_DIR/fakes/python" "$SLURM_TEST_DIR/fakes/bootstrap-python" "$TMP/bin/"
chmod +x "$TMP/bin/"*
export PATH="$TMP/bin:$PATH"
export FAKE_SBATCH_LOG="$TMP/sbatch.log" FAKE_PY_LOG="$TMP/pylog" FAKE_MODULE_LOG="$TMP/modules.log"
export FAKE_BOOTSTRAP_LOG="$TMP/bootstrap.log" FAKE_BASE_PYTHON="$TMP/bin/python"
mkdir -p "$FAKE_PY_LOG"
# shellcheck source=fakes/module.sh
source "$SLURM_TEST_DIR/fakes/module.sh"
export -f module

PASSED=0
pass() { PASSED=$((PASSED + 1)); printf 'ok %d - %s\n' "$PASSED" "$*"; }
die() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }
assert_contains() { [[ "$1" == *"$2"* ]] || die "missing '$2' in: ${1:0:400}"; }
assert_not_contains() { [[ "$1" != *"$2"* ]] || die "unexpected '$2' in: ${1:0:400}"; }
assert_eq() { [[ "$1" == "$2" ]] || die "${3:-values differ}: '$1' != '$2'"; }
expect_fail() { # command... ; stderr+stdout saved in $TMP/failure.out
  if "$@" > "$TMP/failure.out" 2>&1; then die "unexpected success: $*"; fi
}
expect_fail_msg() { local msg=$1; shift; expect_fail "$@"; assert_contains "$(cat "$TMP/failure.out")" "$msg"; }
# Value following --key in a one-arg-per-line file (args with spaces stay intact).
arg_value() { awk -v key="$1" '$0 == key {getline; print; exit}' "$2"; }
arg_values() { awk -v key="$1" -v n="$2" '$0 == key {for (i = 0; i < n; i++) {getline; printf "%s%s", (i ? " " : ""), $0}; print ""; exit}' "$3"; }
last_py() { local f; f=$(ls "$FAKE_PY_LOG"/*.lines 2>/dev/null | sort | tail -n 1); [[ -n "$f" ]] || die 'fake python was not called'; printf '%s' "$f"; }
py_calls() { ls "$FAKE_PY_LOG"/*.lines 2>/dev/null | wc -l | tr -d ' '; }
reset_py() { rm -rf -- "$FAKE_PY_LOG"; mkdir -p "$FAKE_PY_LOG"; }
sbatch_arg() { awk -v key="$1" '$0 == key {getline; print; exit}' "$FAKE_SBATCH_LOG"; }
sbatch_controller() { awk '/\/run_experiment\.slurm$/ {print; exit}' "$FAKE_SBATCH_LOG"; }
native() { if [[ "${OSTYPE:-}" == msys* || "${OSTYPE:-}" == cygwin* ]] && command -v cygpath >/dev/null; then cygpath -m -- "$1"; else printf '%s' "$1"; fi; }
tree_listing() { (cd "$1" && find . | sort); }

# Base test configuration: the shipped legacy profile with test overrides.
CONFIG="$TMP/config with spaces/experiment.conf"
cp "$ROOT/experiment.conf" "$CONFIG"
cat >> "$CONFIG" <<OVERRIDES
MAX_JOBS=2
AUTO_INSTALL_PYTHON_PACKAGES=false
PYTHON_VENV_ROOT=''
MODULES=(stub-module)
PYTHON_BIN='$TMP/bin/python'
PATHS[slurm_logs]='$TMP/work/logs'
PATHS[checkpoint_root]='$TMP/work/checkpoints'
PATHS[logs_root]='$TMP/work/tensorboard'
PATHS[validation_root]='$TMP/work/validation'
PATHS[summary_root]='$TMP/work/validation/summary'
VALIDATE_GRID[fusion_alpha]='0.25|0.75'
OVERRIDES
# derive CONFIG variants: with_settings NAME 'line' ... -> path
with_settings() {
  local name=$1 file; shift
  file="$TMP/config with spaces/$name.conf"
  cp "$CONFIG" "$file"
  printf '\n' >> "$file"
  printf '%s\n' "$@" >> "$file"
  printf '%s' "$file"
}
