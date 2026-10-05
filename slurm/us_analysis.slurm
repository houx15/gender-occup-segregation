#!/bin/bash
#SBATCH --job-name=us_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/us_analysis_%j.out
#SBATCH --error=logs/us_analysis_%j.err

# US state-window analysis following docs/us_3dlnews_report/2026-10-05-analysis-plan.md.
#   sbatch slurm/us_analysis.slurm                      # all parts
#   sbatch slurm/us_analysis.slurm panel,1,2            # selected parts
module load anaconda3/2023.3
conda activate llm
set -eu
PARTS="${1:-panel,1,2,2b,2c,3,3s,figs}"
CONFIG="${2:-config/profiles/garg_weat_dlnews_w10.yml}"
python -m scripts.us_analysis.run --config="$CONFIG" \
    --out_dir=/scratch/network/yh6580/gender-occup/results/analysis_w10 --parts="$PARTS"
echo "DONE us_analysis $PARTS"
