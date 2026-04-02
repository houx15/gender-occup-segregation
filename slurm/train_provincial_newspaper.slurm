#!/bin/bash
#SBATCH --job-name=train_prov_news
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn
#SBATCH --array=0-7

# Train Word2Vec embeddings per province-year unit from provincial newspaper corpora.
# Submits 8 balanced groups as a SLURM array job.
#
# Usage:
#   sbatch slurm/train_provincial_newspaper.slurm
#   sbatch slurm/train_provincial_newspaper.slurm config/config.yml
#
# To retrain specific group only:
#   sbatch --array=3 slurm/train_provincial_newspaper.slurm config/config.yml --retrain

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

CONFIG="${1:-config/config.yml}"
shift 2>/dev/null || true

GROUP_ID=${SLURM_ARRAY_TASK_ID}
echo "Training provincial newspaper embeddings: config=$CONFIG, group=$GROUP_ID, extra args: $@"
echo "Started at: $(date)"

python -m scripts.train_embeddings --config="$CONFIG" --group=$GROUP_ID "$@"

echo "Completed at: $(date)"
