#!/bin/bash
#SBATCH --job-name=prep_wl_pku
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# PKU-server wordlist precheck for the provincial garg_weat arms.
# dedup + OOV-prune candidates_*.txt -> cleaned_*.txt; Chinese threshold = 0.7.
#   - Weibo     -> INFORMAL register (wordlists/zh/garg_weat_informal)
#   - Newspaper -> FORMAL   register (wordlists/zh/garg_weat_formal)
# Coverage is computed from each source's own models. Override the threshold by
# passing it as the first arg: sbatch slurm/prepare_wordlists_pku.slurm 0.6
#
# Both arms' trained models live under /lustre (read via the configs' absolute
# paths), so this runs entirely from the /lustre checkout.
#
# Workflow: run this -> review cleaned_*.txt -> sbatch slurm/garg_weat_pku.slurm

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

# Repo checkout (edit if yours is elsewhere).
REPO_DIR=/lustre/home/2401111059/gender-occup-segregation
cd "$REPO_DIR"

set -u
THRESHOLD="${1:-0.7}"
CANDIDATES="candidates_leadership.txt,candidates_family.txt,candidates_science.txt"

echo "=========================================="
echo "PKU wordlist precheck (Chinese, threshold=$THRESHOLD)"
echo "=========================================="

echo ""
echo "=== Weibo (informal register) ==="
python -m scripts.prepare_wordlists \
    --configs=config/profiles/garg_weat_weibo.yml \
    --wordlist_dir=wordlists/zh/garg_weat_informal \
    --candidates="$CANDIDATES" \
    --threshold="$THRESHOLD"

echo ""
echo "=== Provincial newspaper (formal register) ==="
python -m scripts.prepare_wordlists \
    --configs=config/profiles/garg_weat_provincial_newspaper.yml \
    --wordlist_dir=wordlists/zh/garg_weat_formal \
    --candidates="$CANDIDATES" \
    --threshold="$THRESHOLD"

echo ""
echo "Done. Review cleaned_*.txt under wordlists/zh/garg_weat_{informal,formal}/,"
echo "then: sbatch slurm/garg_weat_pku.slurm"
