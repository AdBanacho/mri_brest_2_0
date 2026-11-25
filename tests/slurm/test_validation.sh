#!/usr/bin/env bash
# Invalid options, keys, types, ranges, choices and combinations fail before
# submission, with no side effects; dry runs change nothing.
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"

before=$(tree_listing "$TMP")
# Dry runs: no Slurm, no Python, no module, no directories or files.
for stage in prepare train validate summarize; do
  bash "$LAUNCHER" --"$stage" --config "$CONFIG" --dry-run > /dev/null || die "dry run $stage failed"
done
bash "$LAUNCHER" --train --config "$CONFIG" --dry-run --local > /dev/null
bash "$LAUNCHER" --validate --config "$CONFIG" --dry-run --task-index 3 > /dev/null
bash "$LAUNCHER" --show-config --config "$CONFIG" > /dev/null
assert_eq "$(tree_listing "$TMP")" "$before" 'dry run changed the filesystem'
CLEANDRY=$(with_settings cleandry 'CLEAN_BEFORE_TRAIN=true')
mkdir -p "$TMP/work/checkpoints"; touch "$TMP/work/checkpoints/keep"
before=$(tree_listing "$TMP")
out=$(bash "$LAUNCHER" --train --config "$CLEANDRY" --dry-run)
assert_contains "$out" "Would remove: $(native "$TMP/work/checkpoints")"
after=$(tree_listing "$TMP")
assert_eq "$after" "$before" 'dry run changed the filesystem'
[[ ! -e "$FAKE_SBATCH_LOG" && ! -e "$FAKE_MODULE_LOG" && "$(py_calls)" == 0 ]] || die 'dry run called sbatch/module/python'
pass 'dry runs (all stages, --local, --task-index, --show-config, cleanup preview) have no side effects'

# Preview is readable and complete.
out=$(bash "$LAUNCHER" --show-config --validate --config "$CONFIG")
assert_contains "$out" 'TRAIN_GRID axes (sorted; last varies fastest)'
assert_contains "$out" 'Training rows: 2 (grid 24, MAX_JOBS=2); validation tasks: 2 x 2 = 4'
assert_contains "$out" 'imaging_patient_id_column      Patient ID'
assert_contains "$out" '* label_scheme'
pass 'show-config prints resolved values, defaults, grid axes and the row table'

# CLI options.
expect_fail_msg 'Unknown argument: --bogus' bash "$CONTROLLER" --train --bogus --dry-run
expect_fail_msg 'Select exactly one stage' bash "$CONTROLLER" --train --validate --dry-run
expect_fail_msg 'needs a value' bash "$CONTROLLER" --train --config
expect_fail_msg 'Task index outside 0-1' bash "$CONTROLLER" --train --config "$CONFIG" --task-index 2 --dry-run
expect_fail_msg 'Invalid task index' bash "$CONTROLLER" --train --config "$CONFIG" --task-index -1 --dry-run
expect_fail_msg '--dependency is a Slurm option' bash "$CONTROLLER" --train --config "$CONFIG" --local --dependency afterok:1
expect_fail_msg 'Invalid --dependency' bash "$CONTROLLER" --train --config "$CONFIG" --dependency 'afterok:1 --wrap=x' --dry-run
expect_fail_msg 'Use --config or --snapshot' bash "$CONTROLLER" --train --config "$CONFIG" --snapshot "$CONFIG"
expect_fail bash "$CONTROLLER" --dry-run
pass 'unknown/duplicate/missing CLI options and bad indices are rejected'

# Configuration errors: each setting must fail before any submission.
check_invalid() { # stage message setting...
  local stage=$1 msg=$2 file; shift 2
  file=$(with_settings invalid "$@")
  expect_fail_msg "$msg" bash "$CONTROLLER" --"$stage" --config "$file" --dry-run
}
check_invalid train 'Unknown key in TRAIN: typo' 'TRAIN[typo]=1'
check_invalid train 'it belongs in VALIDATE or VALIDATE_GRID' 'TRAIN[fusion_alpha]=0.5'
check_invalid train 'Unknown key in PATHS: data_dir' 'PATHS[data_dir]=x'
check_invalid train 'Unknown key in SUMMARIZE: top' 'SUMMARIZE[top]=3'
check_invalid train 'Unknown key in JOB_TRAIN: gpu' 'JOB_TRAIN[gpu]=1'
check_invalid train 'Missing PATHS[metadata_file]' "unset 'PATHS[metadata_file]'"
check_invalid train 'must be true or false' 'TRAIN[use_annotation_boxes]=yes'
check_invalid train "'vgg' not in" 'TRAIN_GRID[mri_model]="densenet121|vgg"'
check_invalid train "item 'genomics' not in" "TRAIN_GRID[feature_groups]='clinical|clinical,genomics'"
check_invalid train 'empty item' "TRAIN_GRID[feature_groups]='clinical,,kinetic'"
check_invalid train 'is not an integer' 'TRAIN[epoch]=abc'
check_invalid train 'is not a number' 'TRAIN_GRID[lr]=1e-4x'
check_invalid train 'num_folds' 'TRAIN[num_folds]=1'
check_invalid train 'outside (0,1]' 'TRAIN[xgb_subsample]=0'
check_invalid validate 'outside [0,1]' "VALIDATE_GRID[fusion_alpha]='0.5|1.5'"
check_invalid train 'positive integers' "unset 'TRAIN[target_size]'" "TRAIN_GRID[target_size]='256 0 64'"
check_invalid train 'target_size needs three integers' "TRAIN[target_size]='256 256'"
check_invalid train 'must be a JSON object' "TRAIN[xgb_extra_json]='[1]'"
check_invalid train 'trainer_extra_json cannot override' "TRAIN[trainer_extra_json]='{\"logger\": false}'"
check_invalid train 'Empty grid choice' 'TRAIN_GRID[lr]="1e-4|"'
check_invalid train 'Empty grid choice' 'TRAIN_GRID[lr]="1e-4||3e-4"'
check_invalid train 'Duplicate grid choice' 'TRAIN_GRID[lr]="1e-4|1e-4"'
check_invalid train 'also a TRAIN_GRID axis' 'TRAIN[lr]=3e-4'
check_invalid train 'MAX_CONCURRENT must be' 'MAX_CONCURRENT=0'
check_invalid train 'MAX_JOBS must be' 'MAX_JOBS=-1'
check_invalid train 'cpus] must be a positive integer' 'JOB_TRAIN[cpus]=0'
check_invalid train 'memory] must look like' 'JOB_TRAIN[memory]=lots'
check_invalid train 'Unsupported EXTRA_SBATCH option' 'EXTRA_SBATCH_TRAIN=(--array=0-99)'
check_invalid train 'CLEAN_BEFORE_TRAIN/CLEAN_BEFORE_VALIDATE must be' 'CLEAN_BEFORE_TRAIN=yes'
check_invalid train 'must be an associative array' 'unset PATHS' 'PATHS=(a b)'
check_invalid train 'AUTO_INSTALL_PYTHON_PACKAGES must be' 'AUTO_INSTALL_PYTHON_PACKAGES=1'
check_invalid train 'requires PYTHON_VENV_ROOT' 'AUTO_INSTALL_PYTHON_PACKAGES=true' "PYTHON_VENV_ROOT=''"
pass 'unknown keys, types, choices, ranges, grids, resources and runtime settings are rejected'

# Combinations that are cheap to detect in Bash (Python re-checks all of them).
check_invalid train 'duke_legacy_3d requires image_dim=3d' 'TRAIN[image_dim]=2d'
check_invalid train 'mask_mode requires a mask_provider' 'TRAIN[profile]=general' 'TRAIN[mask_mode]=roi_crop'
check_invalid train 'needs PATHS[mask_manifest_file]' 'TRAIN[profile]=general' 'TRAIN[mask_provider]=local_files'
check_invalid train 'needs PATHS[density_file]' "TRAIN_GRID[feature_groups]='clinical,density'"
check_invalid train 'feature_selector requires the tabular branch' 'TRAIN[branches]=image'
check_invalid train 'positive_class=2 outside' 'TRAIN[positive_class]=2'
check_invalid train 'use branches=image' 'TRAIN[selection_split]=outer'
check_invalid train 'box_mode=roi_crop' 'TRAIN[box_mode]=roi_crop'
check_invalid train 'mutually exclusive' 'TRAIN[profile]=general' 'TRAIN[mask_provider]=synthetic' 'TRAIN[mask_mode]=mask_channel' 'TRAIN[box_mode]=box_channel'
check_invalid train 'box_mode requires image_dim=3d' 'TRAIN[profile]=general' 'TRAIN[image_dim]=2d' 'TRAIN[use_annotation_boxes]=false' 'TRAIN[box_mode]=dual_stream'
check_invalid train 'box_local_size needs three' "TRAIN[box_local_size]='8 8'"
check_invalid train 'box_margin' 'TRAIN[box_margin]=3'
check_invalid train 'dual_stream_embed_dim' 'TRAIN[dual_stream_embed_dim]=0'
BOXES=$(with_settings boxes 'TRAIN[box_mode]=dual_stream')
bash "$CONTROLLER" --train --config "$BOXES" --dry-run > /dev/null   # erasing + dual_stream is allowed
THREE=$(with_settings three 'TRAIN[label_scheme]=oncotype_3class_18_31')
bash "$CONTROLLER" --train --config "$THREE" --dry-run > /dev/null   # training is fine
expect_fail_msg 'binary-only' bash "$CONTROLLER" --validate --config "$THREE" --dry-run
THREE_OK=$(with_settings three_ok 'TRAIN[label_scheme]=oncotype_3class_18_31' 'VALIDATE[threshold_strategy]=fixed_0.5')
bash "$CONTROLLER" --validate --config "$THREE_OK" --dry-run > /dev/null
check_invalid validate 'primary_branch=fusion needs both' "TRAIN[branches]=image" "TRAIN[feature_selector]=none" "unset 'TRAIN_GRID[feature_groups]' 'TRAIN_GRID[feature_model]'" 'VALIDATE[primary_branch]=fusion'
pass 'invalid combinations (profile/2D, masks, density, branches, 3-class threshold) are rejected'

# Model-specific fields: an axis that never affects any row is an error, a
# partly inert axis is a warning; every key is still passed to Python.
check_invalid train 'TRAIN_GRID[xgb_max_depth] has no effect in any row' "TRAIN_GRID[feature_model]=mlp" "unset 'TRAIN[xgb_max_depth]'" "TRAIN_GRID[xgb_max_depth]='3|6'"
check_invalid train 'TRAIN_GRID[mri_model] has no effect' 'TRAIN[branches]=tabular' 'TRAIN[use_annotation_boxes]=false'
PART=$(with_settings part "unset 'TRAIN[xgb_max_depth]'" "TRAIN_GRID[xgb_max_depth]='3|6'" 'MAX_JOBS=0')
out=$(bash "$CONTROLLER" --train --config "$PART" --dry-run 2>&1)
assert_contains "$out" 'Warning: TRAIN_GRID[xgb_max_depth] has no effect in 24 of 48 training rows'
mlp_row=$(bash "$CONTROLLER" --train --config "$PART" --task-index 47 --dry-run)
assert_contains "$mlp_row" '--feature_model mlp '
assert_contains "$mlp_row" '--xgb_max_depth 6 '
pass 'model-specific grid axes: inert everywhere -> error, partly inert -> warning'
[[ ! -e "$FAKE_SBATCH_LOG" ]] || die 'a failing configuration reached sbatch'
