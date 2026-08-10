#!/bin/bash
#SBATCH --job-name=train_us
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_us_%j.out
#SBATCH --error=logs/train_us_%j.err

# Train one Word2Vec model per {state}_{year} unit discovered in corpora_dir.
# Reuses scripts.train_embeddings unchanged.
#   sbatch slurm/train_us.slurm config/profiles/garg_weat_american_stories.yml

module load anaconda3/2023.3
conda activate llm
set -u
CONFIG="${1:?pass a config profile}"
python -m scripts.train_embeddings --config="$CONFIG"
echo "DONE train_us"
