#!/usr/bin/env bash
# Submit the whole screening campaign (configs/campaign/README.md) with its dependencies:
#
#   MRI-CORE embeddings ──> A03 (mricore feature groups)
#   MRI-CORE segmentation ─> C02, C03 (2D mask runs)
#   every run: train ──afterok──> validate ; all validations ──afterany──> one summarize
#
# Usage (from anywhere; runs from the project root):
#   bash configs/campaign/submit.sh             # submit everything
#   bash configs/campaign/submit.sh --dry-run   # print what would be submitted, submit nothing
#   bash configs/campaign/submit.sh B01 B04     # only these run files (prefix match), prerequisites as needed
#
# Prerequisites are skipped when their output already exists. Do not edit or switch the checkout
# while jobs are queued (tasks run src/ as it is when they start): use a separate git worktree.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$ROOT"
CAMPAIGN=configs/campaign
SCRATCH=/net/scratch/hscra/plgrid/plgabanacho/mri_brest_2_0
EMBEDDINGS=data/duke/mricore/mricore_embeddings.csv
MASKS=$SCRATCH/segmentation/mri_core_zeroshot/mask_manifest.csv

dry=false; filters=()
for arg in "$@"; do
  case $arg in
    --dry-run) dry=true;;
    -h|--help) sed -n '2,15p' "$0"; exit 0;;
    *) filters+=("$arg");;
  esac
done

runs=()
for f in "$CAMPAIGN"/[A-D][0-9][0-9]_*.conf; do
  name=$(basename "$f" .conf)
  if ((${#filters[@]})); then
    keep=false; for p in "${filters[@]}"; do [[ $name == "$p"* ]] && keep=true; done
    [[ $keep == true ]] || continue
  fi
  runs+=("$f")
done
((${#runs[@]})) || { echo "No run files match: ${filters[*]}" >&2; exit 1; }

slurm_id() { sed -n 's/.*Slurm ID \([0-9_]*\).*/\1/p' | tail -1; }
needs() { grep -q "$1" "$2"; }   # does run file $2 reference prerequisite variable $1?

embed_dep= seg_dep=
if printf '%s\n' "${runs[@]}" | xargs grep -l 'MRICORE_EMBEDDINGS' >/dev/null 2>&1; then
  if [[ -f $EMBEDDINGS ]]; then echo "embeddings exist: $EMBEDDINGS"
  elif [[ $dry == true ]]; then echo "would submit: sbatch run_mricore_embed.slurm"; embed_dep=afterok:EMBED
  else
    mkdir -p "$SCRATCH/slurm"   # Slurm cannot create the --output directory itself
    id=$(sbatch --parsable run_mricore_embed.slurm); embed_dep=afterok:$id
    echo "MRI-CORE embeddings: Slurm ID $id"
  fi
fi
if printf '%s\n' "${runs[@]}" | xargs grep -l 'MRICORE_MASKS' >/dev/null 2>&1; then
  if [[ -f $MASKS ]]; then echo "masks exist: $MASKS"
  elif [[ $dry == true ]]; then bash segment_mri_core.slurm --dry-run | tail -2; seg_dep=afterok:SEG
  else
    id=$(bash segment_mri_core.slurm | slurm_id); [[ -n $id ]] || { echo "segmentation submit failed" >&2; exit 1; }
    seg_dep=afterok:$id; echo "MRI-CORE segmentation: Slurm ID $id"
  fi
fi

validations=()
for f in "${runs[@]}"; do
  name=$(basename "$f" .conf); dep=
  needs MRICORE_EMBEDDINGS "$f" && dep=$embed_dep
  needs MRICORE_MASKS "$f" && dep=$seg_dep
  train=(bash run_experiment.sh --train --config "$f"); [[ -z $dep ]] || train+=(--dependency "$dep")
  if [[ $dry == true ]]; then
    printf '%-34s %s\n' "$name" "$("${train[@]}" --dry-run | sed -n 's/^train: //p')${dep:+ after $dep}"
    continue
  fi
  t=$("${train[@]}" | slurm_id); [[ -n $t ]] || { echo "$name: train submit failed" >&2; exit 1; }
  v=$(bash run_experiment.sh --validate --config "$f" --dependency "afterok:$t" | slurm_id)
  [[ -n $v ]] || { echo "$name: validate submit failed" >&2; exit 1; }
  validations+=("$v"); printf '%-34s train %s -> validate %s\n' "$name" "$t" "$v"
done

if [[ $dry == true ]]; then echo "would submit: summarize after all validations"; exit 0; fi
s=$(bash run_experiment.sh --summarize --config "$CAMPAIGN/B01_models.conf" \
      --dependency "afterany:$(IFS=:; echo "${validations[*]}")" | slurm_id)
echo "summarize (whole campaign, runs/campaign/summary): Slurm ID $s"
