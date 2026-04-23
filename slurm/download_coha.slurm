#!/bin/bash
#SBATCH --job-name=download_coha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

# Usage:
#   sbatch slurm/download_coha.slurm
#   sbatch slurm/download_coha.slurm config/profiles/coha_server.yml

CONFIG="${1:-config/profiles/coha_server.yml}"

echo "Downloading COHA archives: config=$CONFIG"

python -m scripts.data_prep.download_coha --config="$CONFIG" --max_workers=4
