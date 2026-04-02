#!/bin/bash
# Submit all 8 training groups in parallel
# Usage: bash slurm/submit_all_train.sh [config] [--retrain]

CONFIG="${1:-config/config.yml}"

echo "Submitting 8 training groups with config=$CONFIG"
for i in $(seq 0 7); do
    JOB=$(sbatch slurm/train_prov_group${i}.slurm "$CONFIG" "${@:2}" 2>&1 | tail -1)
    echo "  Group $i: $JOB"
done
echo ""
echo "Check status: squeue -u $USER"
