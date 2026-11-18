#!/usr/bin/env bash
# argv contract: python -m mribreast <stage> --key value, all stage keys, value shapes.
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"
FIELDS=$ROOT/src/mribreast/config/fields.tsv

QUOTE=$(with_settings quote 'MAX_JOBS=0'   "unset 'TRAIN[imaging_patient_id_column]' 'TRAIN[target_size]'" \
  "TRAIN_GRID[imaging_patient_id_column]='Patient ID|Subject Id'" \
  "TRAIN_GRID[target_size]='256 256 64|128 128 32'" \
  "TRAIN[xgb_extra_json]='{\"reg_alpha\": 0.1, \"reg_lambda\": 2.0}'" \
  "PATHS[density_file]='inputs/density sheet.xlsx'")
# Run the real task path (fake python) for the LAST row of the enlarged grid.
last=$(( 24 * 2 * 2 - 1 ))
bash "$CONTROLLER" --train --config "$QUOTE" --task-index "$last" > "$TMP/task.out"
args=$(last_py)
assert_eq "$(sed -n 1,3p "$args" | tr '\n' ' ')" '-m mribreast train ' 'module entry point'
assert_eq "$(arg_value --imaging_patient_id_column "$args")" 'Subject Id' 'multi-word grid choice'
assert_eq "$(arg_values --target_size 3 "$args")" '128 128 32' 'ints as separate words'
assert_eq "$(arg_value --feature_groups "$args")" 'clinical,kinetic,morphology,heterogeneity' 'csv as one argument'
assert_eq "$(arg_value --mlp_hidden_layers "$args")" '128,64' 'str with comma stays one argument'
assert_eq "$(arg_value --xgb_extra_json "$args")" '{"reg_alpha": 0.1, "reg_lambda": 2.0}' 'json one argument'
assert_eq "$(arg_value --use_annotation_boxes "$args")" true 'explicit boolean true'
assert_eq "$(arg_value --include_sensitive "$args")" false 'explicit boolean false'
assert_eq "$(arg_value --density_file "$args")" "$(native "$TMP/config with spaces/inputs/density sheet.xlsx")" 'relative path resolved against config dir'
assert_eq "$(arg_value --clinical_features_file "$args")" "$(native "$TMP/config with spaces/data/duke/Clinical_and_Other_Features.xlsx")" 'legacy relative path'
assert_eq "$(arg_value --image_manifest_file "$args")" '' 'optional new path stays empty'
first=$(bash "$CONTROLLER" --train --config "$QUOTE" --task-index 0 --dry-run)
assert_contains "$first" '--imaging_patient_id_column Patient\ ID '
assert_contains "$first" '--target_size 256 256 64 '
pass 'quoting of multi-word values, | and comma splitting, ints/json/bool/path shapes'

# Every fields.tsv key of the stage's sections is emitted exactly once (and no other).
declare -A SECTIONS=([prepare]='path train' [train]='path train' [validate]='path train validate' [summarize]='path summarize')
for stage in prepare train validate summarize; do
  reset_py
  bash "$CONTROLLER" --"$stage" --config "$CONFIG" --task-index 0 > /dev/null
  args=$(last_py)
  assert_eq "$(sed -n 3p "$args")" "$stage" 'stage subcommand'
  expected=$(awk -F'\t' -v s=" ${SECTIONS[$stage]} " 'NR > 1 && index(s, " " $2 " ") {print $1}' "$FIELDS" | sort)
  emitted=$(grep -E '^--[a-z0-9_]+$' "$args" | sed 's/^--//' | sort)
  assert_eq "$emitted" "$expected" "$stage keys"
  [[ -z "$(printf '%s\n' "$emitted" | uniq -d)" ]] || die "$stage: duplicated keys"
done
pass 'each stage emits exactly the fields.tsv keys of its sections'

# Legacy alias VALIDATE[num_workers] -> validate_num_workers; train num_workers kept.
reset_py
bash "$CONTROLLER" --validate --config "$CONFIG" --task-index 1 > /dev/null
args=$(last_py)
assert_eq "$(arg_value --validate_num_workers "$args")" 4 'validate_num_workers alias'
assert_eq "$(arg_value --num_workers "$args")" 4 'train num_workers'
assert_eq "$(arg_value --fusion_alpha "$args")" 0.75 'alpha of validation task 1'
assert_eq "$(arg_value --epoch "$args")" 30 'validation receives training keys'
BOTH=$(with_settings both 'VALIDATE[validate_num_workers]=2')
expect_fail_msg 'sets both num_workers' bash "$CONTROLLER" --validate --config "$BOTH" --dry-run
pass 'legacy alias VALIDATE[num_workers] -> --validate_num_workers'

# Train and validation tasks share every training value (Python derives train_id).
reset_py
bash "$CONTROLLER" --train --config "$CONFIG" --task-index 1 > /dev/null
train_args=$(last_py)
bash "$CONTROLLER" --validate --config "$CONFIG" --task-index 3 > /dev/null
val_args=$(last_py)
for key in $(awk -F'\t' 'NR > 1 && ($2 == "train" || $2 == "path") {print $1}' "$FIELDS"); do
  [[ "$key" == target_size ]] && continue
  assert_eq "$(arg_value "--$key" "$val_args")" "$(arg_value "--$key" "$train_args")" "train/validate value of $key"
done
pass 'validation task 3 carries exactly the training values of training row 1'
