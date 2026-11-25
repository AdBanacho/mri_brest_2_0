#!/bin/bash
# MRI-CORE frozen-encoder embeddings for the Duke cohort (feature group `mricore`).
#
#   sbatch run_mricore_embed.slurm                 # full cohort
#   sbatch --export=ALL,LIMIT=3 run_mricore_embed.slurm   # smoke run (first 3 studies)
#
# Writes $OUTPUT (+ *_audit.csv, *_provenance.json). Then set
#   PATHS[mricore_features_file]=<OUTPUT>   and add `mricore` to feature_groups.
# Overridable via --export: MRICORE_REPO CHECKPOINT OUTPUT PHASE SUBTRACT_PHASE POOL LAYER MAX_SLICES LIMIT
#SBATCH --job-name=mricore_embed
#SBATCH --account=plgvirtudrel2026-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/duke/slurm/mricore_embed-%j.out
#SBATCH --error=runs/duke/slurm/mricore_embed-%j.err
set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
cd "$ROOT"
MRICORE_REPO=${MRICORE_REPO:-$HOME/mri_foundation}
CHECKPOINT=${CHECKPOINT:-$MRICORE_REPO/pretrained_weights/MRI_CORE_vitb.pth}
OUTPUT=${OUTPUT:-data/duke/mricore/mricore_embeddings.csv}
PHASE=${PHASE:-1}
POOL=${POOL:-roi}
LAYER=${LAYER:-blocks}
MAX_SLICES=${MAX_SLICES:-16}
LIMIT=${LIMIT:-0}

export PYTHONNOUSERSITE=1
module load ML-bundle
export PYTHONNOUSERSITE=1

# Own small venv on top of the module (torch/numpy from the module are pip constraints).
ENV_DIR=.venvs/$(uname -m)/mricore-embed
if [[ ! -f "$ENV_DIR/.ready" ]]; then
  python3 -m venv --system-site-packages --clear "$ENV_DIR"
  python3 -m pip show numpy torch | awk '$1 == "Name:" {n = tolower($2)} $1 == "Version:" {print n "==" $2}' > "$ENV_DIR/constraints.txt"
  PIP_CONFIG_FILE=/dev/null PIP_USER=false "$ENV_DIR/bin/python" -m pip install --constraint "$ENV_DIR/constraints.txt" \
    einops 'pandas>=2.0,<2.3' 'scikit-learn>=1.3,<1.8' 'nibabel>=5.0' 'openpyxl>=3.1' 'filelock>=3.12'
  touch "$ENV_DIR/.ready"
fi

extra=()
[[ -z "${SUBTRACT_PHASE:-}" ]] || extra+=(--subtract_phase "$SUBTRACT_PHASE")
nvidia-smi -L || true
PYTHONPATH=src "$ENV_DIR/bin/python" -m mribreast.prepare mricore-embed \
  --clinical_features_file data/duke/Clinical_and_Other_Features.xlsx \
  --metadata_file data/duke/Duke-Breast-Cancer-MRI_v2_20220609-nbia-digest.xlsx \
  --annotation_boxes_file data/duke/Annotation_Boxes.xlsx \
  --image_root data/duke/nifti \
  --mricore_repo "$MRICORE_REPO" --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" --phase "$PHASE" --pool "$POOL" --layer "$LAYER" \
  --max_slices "$MAX_SLICES" --limit "$LIMIT" "${extra[@]}"
