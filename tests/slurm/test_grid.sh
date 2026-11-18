#!/usr/bin/env bash
# Grid expansion, index mapping, counts, quoting and argv shape.
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"

# Legacy profile counts: 24 training rows, 24 validations (1 alpha), 72 (3 alphas).
FULL=$(with_settings full 'MAX_JOBS=0' "VALIDATE_GRID[fusion_alpha]=0.5")
out=$(bash "$LAUNCHER" --train --config "$FULL" --dry-run)
assert_contains "$out" 'train: 24 job(s), max concurrent 4'
assert_contains "$out" '--array=0-23%4'
out=$(bash "$LAUNCHER" --validate --config "$FULL" --dry-run)
assert_contains "$out" 'validate: 24 job(s)'
ALPHA3=$(with_settings alpha3 'MAX_JOBS=0' "VALIDATE_GRID[fusion_alpha]='0.25|0.5|0.75'")
out=$(bash "$LAUNCHER" --validate --config "$ALPHA3" --dry-run)
assert_contains "$out" 'validate: 72 job(s)'
assert_contains "$out" '--array=0-71%4'
out=$(bash "$LAUNCHER" --prepare --config "$FULL" --dry-run)
assert_contains "$out" 'prepare: 24 job(s)'
out=$(bash "$LAUNCHER" --summarize --config "$FULL" --dry-run)
assert_contains "$out" 'summarize: 1 job(s)'
assert_not_contains "$out" '--array='
pass '24 train / 24 validate (1 alpha) / 72 validate (3 alphas) / prepare 24 / summarize 1'

# MAX_JOBS limits training rows BEFORE validation expansion; MAX_CONCURRENT throttles.
out=$(bash "$LAUNCHER" --validate --config "$CONFIG" --dry-run)
assert_contains "$out" 'validate: 4 job(s)'
assert_contains "$out" '--array=0-3%4'
M5=$(with_settings m5 'MAX_JOBS=5' 'MAX_CONCURRENT=7' "VALIDATE_GRID[fusion_alpha]='0.25|0.5|0.75'")
out=$(bash "$LAUNCHER" --validate --config "$M5" --dry-run)
assert_contains "$out" 'validate: 15 job(s), max concurrent 7'
assert_contains "$out" '--array=0-14%7'
M99=$(with_settings m99 'MAX_JOBS=99')
out=$(bash "$LAUNCHER" --train --config "$M99" --dry-run)
assert_contains "$out" 'train: 24 job(s)'
pass 'MAX_JOBS caps training rows before validation expansion; MAX_CONCURRENT is the array throttle'

# Index -> row: sorted keys, last sorted key fastest (legacy select_grid).
# Axes with >1 choice, sorted: feature_groups(2) feature_model(2) mri_model(3) sensitivity_lambda(2)
fgs=('clinical' 'clinical,kinetic,morphology,heterogeneity'); fms=(xgboost mlp)
mms=(densenet121 resnet10 resnet18); sls=(0.05 0.2)
args_file="$TMP/row.lines"
for idx in 0 1 2 5 6 11 12 17 23; do
  bash "$CONTROLLER" --train --config "$FULL" --task-index "$idx" --dry-run > "$TMP/row.out"
  i=$idx; sl=${sls[i % 2]}; i=$((i / 2)); mm=${mms[i % 3]}; i=$((i / 3)); fm=${fms[i % 2]}; i=$((i / 2)); fg=${fgs[i % 2]}
  row=$(cat "$TMP/row.out")
  assert_contains "$row" "--mri_model $mm "
  assert_contains "$row" "--feature_model $fm "
  assert_contains "$row" "--sensitivity_lambda $sl "
  assert_contains "$row" "--feature_groups ${fg//,/\\,} "
done
pass 'task index -> grid row mapping (last sorted key varies fastest)'

# Optional cross-check against the reference controller (read-only dry runs).
LEGACY=${MRI_BREAST_LEGACY_REPO:-}
if [[ -n "$LEGACY" && -f "$LEGACY/run_experiment.slurm" && -f "$LEGACY/experiment.conf" ]]; then
  cp "$LEGACY/experiment.conf" "$TMP/legacy.conf"
  for idx in $(seq 0 23); do
    legacy_row=$(bash "$LEGACY/run_experiment.slurm" --train --config "$TMP/legacy.conf" --task-index "$idx" --dry-run)
    new_row=$(bash "$CONTROLLER" --train --config "$FULL" --task-index "$idx" --dry-run)
    for key in mri_model feature_model sensitivity_lambda; do
      lv=$(sed -n "s/.*--$key \([^ ]*\) .*/\1/p" <<< "$legacy_row"); nv=$(sed -n "s/.*--$key \([^ ]*\) .*/\1/p" <<< "$new_row")
      assert_eq "$nv" "$lv" "legacy mismatch at row $idx for $key"
    done
    lfg=$(sed -n 's/.*--feature_groups \(clinical[a-z ]*\) --.*/\1/p' <<< "$legacy_row"); nfg=$(sed -n 's/.*--feature_groups \([^ ]*\) .*/\1/p' <<< "$new_row")
    assert_eq "${nfg//\\,/ }" "$lfg" "legacy feature_groups mismatch at row $idx"
  done
  pass "index mapping identical to the reference controller for all 24 rows ($LEGACY)"
else
  printf 'skip - reference controller cross-check (set MRI_BREAST_LEGACY_REPO)\n'
fi

# Validation index: train_index = i / |VALIDATE_GRID|, alpha = i % |VALIDATE_GRID|.
row=$(bash "$CONTROLLER" --validate --config "$ALPHA3" --task-index 7 --dry-run)
assert_contains "$row" '--fusion_alpha 0.5 '
# 7 / 3 = training row 2 -> resnet10 + lambda 0.05; 7 % 3 = 1 -> alpha 0.5
assert_contains "$row" '--sensitivity_lambda 0.05 '
assert_contains "$row" '--mri_model resnet10 '
assert_contains "$row" '--validate_num_workers 4 '
row=$(bash "$CONTROLLER" --validate --config "$ALPHA3" --task-index 71 --dry-run)
assert_contains "$row" '--fusion_alpha 0.75 '
assert_contains "$row" '--mri_model resnet18 '
assert_contains "$row" '--feature_model mlp '
pass 'validation index -> (training row, alpha)'
