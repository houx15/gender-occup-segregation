#!/bin/bash
#SBATCH --job-name=train_embed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu

module purge
module load anaconda3/2023.3
conda activate llm

# Usage:
#   sbatch slurm/train.slurm                                # train all units
#   sbatch slurm/train.slurm config/config.yml              # explicit config
#   sbatch slurm/train.slurm config/config.yml 1940_1949    # specific time slice
#   sbatch slurm/train.slurm config/config.yml --group=0    # specific province group

CONFIG="${1:-config/config.yml}"
shift || true

echo "Training embeddings: config=$CONFIG, extra args: $@"

python -m scripts.train_embeddings --config="$CONFIG" "$@"
