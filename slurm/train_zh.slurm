#!/bin/bash
#SBATCH --job-name=train_zh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu
#SBATCH --output=logs/train_zh_%j.out
#SBATCH --error=logs/train_zh_%j.err

# Princeton-llm-env training driver. Same one-liner as slurm/train.slurm
# but with the Princeton conda setup (anaconda3 module + llm env) and
# Princeton logging.
#
# Usage:
#   sbatch slurm/train_zh.slurm                                          # train all units in config/config.yml
#   sbatch slurm/train_zh.slurm config/profiles/<file>.yml               # explicit config
#   sbatch slurm/train_zh.slurm config/profiles/<file>.yml --unit=1940_1949   # one slice
#   sbatch slurm/train_zh.slurm config/profiles/<file>.yml --retrain          # force retrain
#
# Typical Chinese-arm invocations:
#   sbatch slurm/train_zh.slurm config/profiles/garg_weat_china_ngram_weighted.yml
#   sbatch slurm/train_zh.slurm config/profiles/garg_weat_renminribao.yml

module load anaconda3/2023.3
conda activate llm

CONFIG="${1:-config/config.yml}"
shift || true

echo "Training embeddings: config=$CONFIG, extra args: $@"

python -m scripts.train_embeddings --config="$CONFIG" "$@"
