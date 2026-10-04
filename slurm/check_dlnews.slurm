#!/bin/bash
#SBATCH --job-name=check_dlnews
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/check_dlnews_%j.out
#SBATCH --error=logs/check_dlnews_%j.err

# Checks on the 3DLNews2 newspaper occupation results (run after
# slurm/analyze_dlnews.slurm):
#   1. diagnose_word_senses: nearest neighbours of suspect list words in the
#      largest state-period models (surname / non-job senses).
#   2. build_occupation_shares: ACS (IPUMS) female shares by occupation word and
#      state-period survey measures (aggregation cached after the first run).
#   3. check_occupation_census: occupation RND vs ACS, nationally by occupation
#      and by state-period.
#   sbatch slurm/check_dlnews.slurm
#   sbatch slurm/check_dlnews.slurm "weaver|weavers,pilot|pilots"
module load anaconda3/2023.3
conda activate llm
set -eu

CONFIG="config/profiles/garg_weat_dlnews.yml"
WORDS="${1:-weaver|weavers,carpenter|carpenters,painter|painters,secretary|secretaries,supervisor|supervisors,operator|operators,driver|drivers,coach|coaches,judge|judges,executive|executives}"

echo "== 1. word senses"
python -m scripts.diagnose_word_senses --config="$CONFIG" --words="$WORDS" --n_units=4 --topn=12
echo "== 2. ACS shares"
python -m scripts.data_prep.build_occupation_shares --config=config/ipums_acs.yml
echo "== 3. census check"
python -m scripts.check_occupation_census --config="$CONFIG"
echo "DONE check_dlnews"
