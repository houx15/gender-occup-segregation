#!/bin/bash
#SBATCH --job-name=build_corpus_coha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

# Usage:
#   sbatch slurm/build_corpus_coha.slurm
#   sbatch slurm/build_corpus_coha.slurm config/profiles/coha_server.yml

CONFIG="${1:-config/profiles/coha_server.yml}"

echo "Building COHA corpora: config=$CONFIG"

python -m scripts.data_prep.build_corpora_coha --config="$CONFIG"
