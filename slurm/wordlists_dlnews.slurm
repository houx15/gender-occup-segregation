#!/bin/bash
#SBATCH --job-name=wordlists_dlnews
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/wordlists_dlnews_%j.out
#SBATCH --error=logs/wordlists_dlnews_%j.err

# Coverage-driven wordlists for the 3DLNews2 state-period arm, then re-analysis
# and a stability check. No retraining — reuses the trained .kv models.
#   1. prepare_wordlists: prune the big candidates_*.txt pools to words in vocab
#      in >= THRESHOLD of the NEWSPAPER state-period models -> dlnews_cleaned_*.txt
#      (COHA's cleaned_*.txt untouched). THRESHOLD=1.0 = in every unit, which is
#      what the analyzer's consistent set requires anyway.
#   2. analyze_category_bias for newspaper and TV (both profiles read dlnews_cleaned_*).
#   3. diagnose_unit_stability: words used + adjacent-period state correlations.
#
#   sbatch slurm/wordlists_dlnews.slurm          # threshold 1.0
#   sbatch slurm/wordlists_dlnews.slurm 0.95
module load anaconda3/2023.3
conda activate llm
set -eu

THRESHOLD="${1:-1.0}"
NEWS="config/profiles/garg_weat_dlnews.yml"
TV="config/profiles/garg_weat_dlnews_tv.yml"

echo "== 1. prune candidates against newspaper models (threshold $THRESHOLD)"
python -m scripts.prepare_wordlists \
    --configs="$NEWS" \
    --wordlist_dir="wordlists/en/garg_weat" \
    --candidates="candidates_leadership.txt,candidates_family.txt,candidates_science.txt" \
    --threshold="$THRESHOLD" \
    --out_prefix="dlnews_cleaned_"
wc -l wordlists/en/garg_weat/dlnews_cleaned_*.txt

for CONFIG in "$NEWS" "$TV"; do
    echo "== 2. analyze $CONFIG"
    python -m scripts.analyze_category_bias --config="$CONFIG"
    echo "== 3. stability $CONFIG"
    python -m scripts.diagnose_unit_stability --config="$CONFIG"
done
echo "DONE wordlists_dlnews"
