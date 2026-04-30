#!/bin/bash
#SBATCH --job-name=full_pipeline_en
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu

module load anaconda3/2023.3
conda activate llm
# Runs the full English pipeline (ngram_en or coha) via run_pipeline.sh.
#
# Usage:
#   sbatch slurm/full_pipeline_en.slurm
#   sbatch slurm/full_pipeline_en.slurm config/profiles/coha_server.yml
#   sbatch slurm/full_pipeline_en.slurm config/profiles/ngram_en_server.yml

CONFIG="${1:-config/profiles/ngram_en_server.yml}"

echo "Running English full pipeline with config=$CONFIG"

bash run_pipeline.sh --config "$CONFIG"
