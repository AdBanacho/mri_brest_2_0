#!/usr/bin/env bash
# Thin launcher: all logic lives in run_experiment.slurm (see docs/slurm.md).
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_experiment.slurm" "$@"
