#!/bin/bash
#SBATCH --job-name=full_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu

module purge
module load anaconda3/2023.3
conda activate llm

# Usage:
#   sbatch slurm/full_pipeline.slurm
#   sbatch slurm/full_pipeline.slurm config/config.yml

CONFIG="${1:-config/config.yml}"

echo "Running full pipeline with config=$CONFIG"

bash run_pipeline.sh --config "$CONFIG"
