#!/bin/bash
#SBATCH --job-name=family_wl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/family_wl_%j.out
#SBATCH --error=logs/family_wl_%j.err

# Family wordlist expansion (docs/research-log-wordlists.md):
#   1. coverage of every candidates_family.txt entry over the analyzed
#      state-window models (same unit clip as the analyzer); report under
#      results_dir/wordlist_precheck_report.csv
#   2. nearest neighbours of entries with a likely non-family sense
#   sbatch slurm/family_wordlist_dlnews.slurm [config]
module load anaconda3/2023.3
conda activate llm
set -eu
CONFIG="${1:-config/profiles/garg_weat_dlnews_w10.yml}"
echo "== 1. coverage"
python -m scripts.prepare_wordlists --configs="$CONFIG" \
    --wordlist_dir=wordlists/en/occupation_family \
    --candidates=candidates_family.txt --threshold=0.5
echo "== 2. senses"
python -m scripts.diagnose_word_senses --config="$CONFIG" \
    --words=home,house,nursery,foster,engagement,domestic,loved,couple,generation,twins,kin,adoption,dinner \
    --n_units=4 --topn=12
echo "DONE family_wl"
