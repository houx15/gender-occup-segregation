#!/bin/bash
#SBATCH --job-name=train_g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# Train group 2: 河南 贵州 天津 (~1.38GB)
#
# Usage:
#   sbatch slurm/train_prov_group2.slurm
#   sbatch slurm/train_prov_group2.slurm config/config.yml --retrain

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

CONFIG="${1:-config/config.yml}"
shift 2>/dev/null || true

echo "Training group 2: 河南 贵州 天津 (~1.38GB)"
echo "Config: $CONFIG  Extra: $@"
echo "Started at: $(date)"

python -m scripts.train_embeddings --config="$CONFIG" --group=2 "$@"

echo "Completed at: $(date)"
