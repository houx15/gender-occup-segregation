#!/bin/bash
#SBATCH --job-name=cns_totalcounts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yh6580@princeton.edu
#SBATCH --output=logs/cns_totalcounts_%j.out
#SBATCH --error=logs/cns_totalcounts_%j.err

# One-time pass over all 105 shards to derive the CORRECT per-year denominator
# (Σ kept match_count) for per_year_capped subsampling, cached as
# raw_ngrams/totalcounts-5.derived. The shipped totalcounts-5 under-counts, so
# the cap never bit and slices blew up (e.g. 1975_1984 -> 85G / 6.7e9 lines).
#
# Run this BEFORE the per-slice sweep. After it finishes, inspect the printed
# per-year totals vs the cap in the log and confirm the cap value is sane.
#
# Usage:
#   sbatch slurm/precompute_ngram_totalcounts.slurm

module load anaconda3/2023.3
conda activate llm

set -u

CONFIG="config/profiles/garg_weat_china_ngram_subsampled.yml"

python -m scripts.data_prep.precompute_ngram_totalcounts --config="$CONFIG"
