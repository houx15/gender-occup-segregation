#!/bin/bash
#SBATCH --job-name=tagged_dlnews
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/tagged_dlnews_%j.out
#SBATCH --error=logs/tagged_dlnews_%j.err

# Approach A (separate arm): one shared word2vec model per period over all
# states, gender anchors tagged by state. Offline; reuses the downloaded raw
# 3DLNews2 data and the ACS / Project Implicit benchmark files.
#   1. build_corpora_us   pooled_{period} corpora with state-tagged anchors
#                         (per-year markers: a rerun resumes)
#   2. train_embeddings   one model per period (existing models are skipped)
#   3. analyze_state_tagged + diagnose_unit_stability
#   4. check_state_benchmarks + visualize
#   sbatch slurm/tagged_dlnews.slurm
#   sbatch slurm/tagged_dlnews.slurm config/profiles/garg_weat_dlnews_tagged.yml
module load anaconda3/2023.3
conda activate llm
set -eu

CONFIG="${1:-config/profiles/garg_weat_dlnews_tagged.yml}"

echo "== 1. build pooled, state-tagged corpora"
python -m scripts.data_prep.build_corpora_us --config="$CONFIG"
echo "== 2. train one model per period"
python -m scripts.train_embeddings --config="$CONFIG"
echo "== 3. analyze"
python -m scripts.analyze_state_tagged --config="$CONFIG"
python -m scripts.diagnose_unit_stability --config="$CONFIG"
echo "== 4. benchmarks + figures"
python -m scripts.check_state_benchmarks --config="$CONFIG"
python -m scripts.visualize main --config="$CONFIG"
echo "DONE tagged_dlnews"
