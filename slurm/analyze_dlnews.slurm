#!/bin/bash
#SBATCH --job-name=analyze_dlnews
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/analyze_dlnews_%j.out
#SBATCH --error=logs/analyze_dlnews_%j.err

# Re-analyze the trained 3DLNews2 state-period models after a wordlist/config
# change (no retraining): RND + Cohen's d -> stability diagnostic -> figures
# (per-period choropleths, national trend, per-state change + CSVs).
#   sbatch slurm/analyze_dlnews.slurm                 # newspaper + TV
#   sbatch slurm/analyze_dlnews.slurm config/profiles/garg_weat_dlnews.yml
module load anaconda3/2023.3
conda activate llm
set -eu

CONFIGS=("$@")
if [ "${#CONFIGS[@]}" -eq 0 ]; then
    CONFIGS=("config/profiles/garg_weat_dlnews.yml" "config/profiles/garg_weat_dlnews_tv.yml")
fi

for CONFIG in "${CONFIGS[@]}"; do
    echo "===== $CONFIG"
    python -m scripts.analyze_category_bias --config="$CONFIG"
    python -m scripts.diagnose_unit_stability --config="$CONFIG"
    python -m scripts.visualize main --config="$CONFIG"
done
echo "DONE analyze_dlnews"
