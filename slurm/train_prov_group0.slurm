#!/bin/bash
#SBATCH --job-name=train_g0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# Train group 0: 山东 (~1.37GB)
#
# Usage:
#   sbatch slurm/train_prov_group0.slurm
#   sbatch slurm/train_prov_group0.slurm config/config.yml --retrain

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

CONFIG="${1:-config/config.yml}"
shift 2>/dev/null || true

echo "Training group 0: 山东 (~1.37GB)"
echo "Config: $CONFIG  Extra: $@"
echo "Started at: $(date)"

python -m scripts.train_embeddings --config="$CONFIG" --group=0 "$@"

echo "Completed at: $(date)"
