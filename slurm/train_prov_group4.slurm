#!/bin/bash
#SBATCH --job-name=train_g4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# Train group 4: 河北 安徽 四川 黑龙江 吉林 (~1.39GB)
#
# Usage:
#   sbatch slurm/train_prov_group4.slurm
#   sbatch slurm/train_prov_group4.slurm config/config.yml --retrain

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

CONFIG="${1:-config/config.yml}"
shift 2>/dev/null || true

echo "Training group 4: 河北 安徽 四川 黑龙江 吉林 (~1.39GB)"
echo "Config: $CONFIG  Extra: $@"
echo "Started at: $(date)"

python -m scripts.train_embeddings --config="$CONFIG" --group=4 "$@"

echo "Completed at: $(date)"
