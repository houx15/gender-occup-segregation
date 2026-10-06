#!/bin/bash
#SBATCH --job-name=household_wl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/household_wl_%j.out
#SBATCH --error=logs/household_wl_%j.err

# Household-work wordlist expansion (docs/research-log-wordlists.md):
#   1. coverage of every candidates_household.txt entry over the analyzed
#      state-window models; report under results_dir/wordlist_precheck_report.csv,
#      entries with coverage >= 0.5 in wordlist_dir/cleaned_household.txt
#   2. nearest neighbours of every entry that passed, for the sense screen
#   sbatch slurm/household_wordlist_dlnews.slurm [config]
module load anaconda3/2023.3
conda activate llm
set -eu
CONFIG="${1:-config/profiles/garg_weat_dlnews_w10.yml}"
WL=wordlists/en/occupation_family
echo "== 1. coverage"
python -m scripts.prepare_wordlists --configs="$CONFIG" --wordlist_dir="$WL" \
    --candidates=candidates_household.txt --threshold=0.5
echo "== 2. senses"
python -m scripts.diagnose_word_senses --config="$CONFIG" \
    --words="$(paste -sd, "$WL/cleaned_household.txt")" --n_units=4 --topn=12
echo "DONE household_wl"
