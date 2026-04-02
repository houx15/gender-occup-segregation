#!/bin/bash
#SBATCH --job-name=train_prov_news
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# Train Word2Vec embeddings per province-year unit from provincial newspaper corpora.
#
# Usage:
#   sbatch slurm/train_provincial_newspaper.slurm
#   sbatch slurm/train_provincial_newspaper.slurm config/config.yml
#   sbatch slurm/train_provincial_newspaper.slurm config/config.yml --group=0
#   sbatch slurm/train_provincial_newspaper.slurm config/config.yml --unit=北京_2020

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

CONFIG="${1:-config/config.yml}"
shift 2>/dev/null || true

echo "Training provincial newspaper embeddings: config=$CONFIG, extra args: $@"
echo "Started at: $(date)"

python -m scripts.train_embeddings --config="$CONFIG" "$@"

echo "Completed at: $(date)"
