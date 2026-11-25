#!/usr/bin/env bash
# segment_mri_core.slurm: input preflight before queueing, self-submission with a GPU and an
# existing log directory, segmentation env (config packages + einops/torchvision, module
# NumPy/PyTorch constraints) built once and reused, cohort options taken from the config.
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"
SEG=$ROOT/segment_mri_core.slurm
installs() { grep -c '^install ' "$FAKE_BOOTSTRAP_LOG" 2>/dev/null || true; }

DATA="$TMP/data dir"
REPO="$TMP/mri_foundation"
mkdir -p "$DATA/nifti" "$REPO/models/sam" "$REPO/pretrained_weights"
touch "$DATA/clinical.xlsx" "$DATA/digest.xlsx" "$DATA/boxes.xlsx" "$REPO/models/sam/__init__.py"
SEGCONF="$TMP/config with spaces/seg.conf"
cp "$ROOT/configs/duke_2d_mri_core_mask.conf" "$SEGCONF"
cat >> "$SEGCONF" <<OVERRIDES
PYTHON_BIN='$TMP/bin/bootstrap-python'
PYTHON_VENV_ROOT='$TMP/seg-envs'
AUTO_INSTALL_PYTHON_PACKAGES=true
MODULES=(stub-module)
PATHS[clinical_features_file]='$DATA/clinical.xlsx'
PATHS[metadata_file]='$DATA/digest.xlsx'
PATHS[annotation_boxes_file]='$DATA/boxes.xlsx'
PATHS[image_root]='$DATA/nifti'
PATHS[mask_manifest_file]='$TMP/work/seg out/mask_manifest.csv'
OVERRIDES

expect_fail_msg 'pretrained_weights/mri_foundation.pth' bash "$SEG" --config "$SEGCONF" --mri-foundation "$REPO"
assert_contains "$(cat "$TMP/failure.out")" 'sam_vit_b_01ec64.pth'
[[ ! -e "$FAKE_SBATCH_LOG" ]] || die 'submitted despite missing weights'
pass 'missing (git-ignored) weights are reported before anything is queued'

touch "$REPO/pretrained_weights/mri_foundation.pth" "$REPO/pretrained_weights/sam_vit_b_01ec64.pth"
out=$(bash "$SEG" --config "$SEGCONF" --mri-foundation "$REPO" --dry-run)
assert_contains "$out" '--gres=gpu:1'
[[ ! -e "$FAKE_SBATCH_LOG" && ! -e "$TMP/work/seg out" ]] || die 'dry run submitted or created directories'
bash "$SEG" --config "$SEGCONF" --mri-foundation "$REPO" -- --limit 5 > "$TMP/submit.out"
assert_contains "$(cat "$TMP/submit.out")" 'Slurm ID'
[[ -d "$TMP/work/seg out" ]] || die 'log directory not created before sbatch'
assert_eq "$(sbatch_arg --job)" --project-root 'job flag passed'
log=$(cat "$FAKE_SBATCH_LOG")
assert_contains "$log" '--gres=gpu:1'
assert_contains "$log" '--account=plgvirtudrel2026-gpu-gh200'
assert_contains "$log" "--output=$TMP/work/seg out/slurm-%j.out"
pass 'submits itself with a GPU, the config account and an existing log directory'

# Run the job as Slurm would (the queued argv), twice: one install, then reuse.
mapfile -t job_args < <(awk 'found {print} /segment_mri_core\.slurm$/ {found = 1; print}' "$FAKE_SBATCH_LOG")
for run in 1 2; do
  SLURM_JOB_ID=7 bash "${job_args[@]}" > "$TMP/job$run.out" 2>&1 || die "job failed: $(cat "$TMP/job$run.out")"
done
assert_eq "$(installs)" 1 'environment built once'
boot=$(cat "$FAKE_BOOTSTRAP_LOG")
assert_contains "$boot" 'einops>=0.7 torchvision'
assert_contains "$boot" 'monai==1.5.1'
assert_contains "$(cat "$FAKE_MODULE_LOG")" 'load stub-module'
env_dir=$(dirname "$(find "$TMP/seg-envs" -name .ready)")
[[ "$env_dir" == "$TMP/seg-envs/$(uname -m)/segment-"* ]] || die "unexpected env dir $env_dir"
pass 'segmentation env: config packages + einops/torchvision under module constraints, reused'

call=$(last_py)
assert_eq "$(sed -n '1,4p' "$call" | tr '\n' ' ')" '-m mribreast prepare-data segment-mri-core ' 'entry point'
assert_eq "$(arg_value --output_dir "$call")" "$TMP/work/seg out" 'output dir = mask manifest dir'
assert_eq "$(arg_value --image_root "$call")" "$DATA/nifti"
assert_eq "$(arg_value --mri_foundation_repo "$call")" "$REPO"
assert_eq "$(arg_value --limit "$call")" 5 'extra args passed through'
grep -qx 'profile=general' "$call" && grep -qx 'phase_order=explicit' "$call" || die 'cohort options not taken from TRAIN'
assert_contains "$(cat "${call%.lines}.env")" 'PYTHONNOUSERSITE=1'
pass 'runs segment-mri-core with config paths, cohort options and pass-through arguments'

printf 'All %d segmentation job checks passed\n' "$PASSED"
