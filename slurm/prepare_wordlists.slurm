#!/bin/bash
#SBATCH --job-name=prepare_wordlists
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu
#SBATCH --output=logs/prepare_wordlists_%j.out
#SBATCH --error=logs/prepare_wordlists_%j.err

# Wordlist precheck for the analyze_garg_weat experiment.
#   1. per-file dedup (keep first occurrence)
#   2. OOV prune: drop words in vocab in < THRESHOLD of the pooled models
# Reads candidates_*.txt, writes cleaned_*.txt (raw lists untouched), plus a
# per-word coverage report under the first config's results_dir.
#
# Workflow: run this -> manually review cleaned_*.txt -> run the full
# garg_weat pipeline (slurm/garg_weat_all_sources.slurm), which reads the
# cleaned lists.

module load anaconda3/2023.3
conda activate llm

set -u

CONFIGS="config/profiles/garg_weat_coha_trained.yml,config/profiles/garg_weat_coha_histwords_sgns.yml,config/profiles/garg_weat_coha_histwords_svd.yml"
CANDIDATES="candidates_leadership.txt,candidates_family.txt,candidates_science.txt"
THRESHOLD="${1:-0.8}"

echo "=========================================="
echo "Wordlist precheck"
echo "  configs:    $CONFIGS"
echo "  candidates: $CANDIDATES"
echo "  threshold:  $THRESHOLD"
echo "=========================================="

python -m scripts.prepare_wordlists \
    --configs="$CONFIGS" \
    --wordlist_dir="wordlists/en/garg_weat" \
    --candidates="$CANDIDATES" \
    --threshold="$THRESHOLD"

echo ""
echo "Done. Review the cleaned_*.txt files under wordlists/en/garg_weat/,"
echo "then launch: sbatch slurm/garg_weat_all_sources.slurm"
