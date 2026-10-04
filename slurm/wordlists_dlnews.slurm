#!/bin/bash
#SBATCH --job-name=wordlists_dlnews
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/wordlists_dlnews_%j.out
#SBATCH --error=logs/wordlists_dlnews_%j.err

# Coverage-driven occupation / household wordlists for the 3DLNews2
# state-period arm, then re-analysis, stability check and figures for
# newspaper + TV. No retraining — reuses the trained .kv models.
#   1. prepare_wordlists: keep candidates in vocab in >= THRESHOLD of the
#      NEWSPAPER models inside analysis.decade_range -> dlnews_cleaned_*.txt.
#      THRESHOLD=1.0 = in every analyzed unit (what the consistent set needs).
#      Per-word coverage report: <newspaper results_dir>/wordlist_precheck_report.csv
#   2. slurm/analyze_dlnews.slurm steps (analyze -> stability -> figures).
#
#   sbatch slurm/wordlists_dlnews.slurm          # threshold 1.0
#   sbatch slurm/wordlists_dlnews.slurm 0.95
module load anaconda3/2023.3
conda activate llm
set -eu

THRESHOLD="${1:-1.0}"
WL="wordlists/en/occupation_family"
NEWS="config/profiles/garg_weat_dlnews.yml"
TV="config/profiles/garg_weat_dlnews_tv.yml"

echo "== 1. prune candidates against newspaper models (threshold $THRESHOLD)"
python -m scripts.prepare_wordlists \
    --configs="$NEWS" \
    --wordlist_dir="$WL" \
    --candidates="candidates_occupation.txt,candidates_household.txt" \
    --threshold="$THRESHOLD" \
    --out_prefix="dlnews_cleaned_"
wc -l "$WL"/dlnews_cleaned_*.txt "$WL"/family_sphere.txt

for CONFIG in "$NEWS" "$TV"; do
    echo "===== $CONFIG"
    python -m scripts.analyze_category_bias --config="$CONFIG"
    python -m scripts.diagnose_unit_stability --config="$CONFIG"
    python -m scripts.visualize main --config="$CONFIG"
done
echo "DONE wordlists_dlnews"
