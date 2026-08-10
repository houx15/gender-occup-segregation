#!/bin/bash
#SBATCH --job-name=check_lccn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/check_lccn_%j.out
#SBATCH --error=logs/check_lccn_%j.err

# Sanity-check lccn_from_article_id against real American Stories article ids.
#
# IMPORTANT: if no raw american_stories_{year}.jsonl exists yet, the checker
# pulls the year from HuggingFace and NEEDS INTERNET. Princeton compute nodes
# usually have no internet, so either:
#   (a) run the python one-liner directly on a LOGIN node (fastest for a
#       one-off check), or
#   (b) pre-download with download_american_stories, then submit this job, or
#   (c) submit to an internet-enabled partition if your cluster has one.
#
# Once a raw file exists, this job reads it locally (no internet needed).
#
#   sbatch slurm/check_lccn.slurm             # defaults: year=1940 n=500
#   sbatch slurm/check_lccn.slurm 1950 1000   # year n
#
# Login-node equivalent (no sbatch):
#   python -m scripts.data_prep.check_lccn_regex \
#       --config=config/profiles/garg_weat_american_stories.yml --year=1940 --n=500

module load anaconda3/2023.3
conda activate llm
set -u

YEAR="${1:-1940}"
N="${2:-500}"

python -m scripts.data_prep.check_lccn_regex \
    --config=config/profiles/garg_weat_american_stories.yml \
    --year="$YEAR" --n="$N"
