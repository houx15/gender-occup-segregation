#!/bin/bash
#SBATCH --job-name=visualize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

# Usage:
#   sbatch slurm/visualize.slurm config/profiles/ngram_weat.yml
#   sbatch slurm/visualize.slurm config/profiles/renminribao_weat.yml

CONFIG="${1:-config/config.yml}"

echo "Visualizing with config=$CONFIG"
python -m scripts.visualize main --config="$CONFIG"
