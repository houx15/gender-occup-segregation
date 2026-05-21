#!/bin/bash
#SBATCH --job-name=prepare_wordlists_zh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu
#SBATCH --output=logs/prepare_wordlists_zh_%j.out
#SBATCH --error=logs/prepare_wordlists_zh_%j.err

# CHINESE wordlist precheck for analyze_garg_weat (formal register):
#   1. per-file dedup (keep first occurrence)
#   2. OOV prune: drop words in vocab in < THRESHOLD of the pooled models
# Reads candidates_*.txt, writes cleaned_*.txt under wordlists/zh/garg_weat_formal
# (raw lists untouched). Coverage is pooled over the Renminribao + China-Ngram
# models (Princeton), which share the formal register.
#
# Workflow: run this -> review cleaned_*.txt -> sbatch slurm/garg_weat_zh.slurm
#
# Weibo uses the INFORMAL register (wordlists/zh/garg_weat_informal) and lives on
# the PKU server. Run the precheck there with that dir + the weibo config:
#   python -m scripts.prepare_wordlists \
#       --configs=config/profiles/garg_weat_weibo.yml \
#       --wordlist_dir=wordlists/zh/garg_weat_informal \
#       --candidates=candidates_leadership.txt,candidates_family.txt,candidates_science.txt \
#       --threshold=0.7
# (provincial newspaper uses the formal register but also runs on PKU.)

module load anaconda3/2023.3
conda activate llm

set -u

CONFIGS="config/profiles/garg_weat_renminribao.yml,config/profiles/garg_weat_china_ngram.yml"
CANDIDATES="candidates_leadership.txt,candidates_family.txt,candidates_science.txt"
# Chinese OOV threshold is 0.7 (vs 0.8 for English) — the Chinese category
# seeds include lower-frequency terms whose coverage dips below 0.8 in the
# sparser early decades. Override by passing a value as the first arg.
THRESHOLD="${1:-0.7}"

echo "=========================================="
echo "Chinese wordlist precheck (formal register)"
echo "  configs:    $CONFIGS"
echo "  candidates: $CANDIDATES"
echo "  threshold:  $THRESHOLD"
echo "=========================================="

python -m scripts.prepare_wordlists \
    --configs="$CONFIGS" \
    --wordlist_dir="wordlists/zh/garg_weat_formal" \
    --candidates="$CANDIDATES" \
    --threshold="$THRESHOLD"

echo ""
echo "Done. Review the cleaned_*.txt files under wordlists/zh/garg_weat_formal/,"
echo "then launch: sbatch slurm/garg_weat_zh.slurm"
