#!/bin/bash
#SBATCH --job-name=visualize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu

module purge
module load anaconda3/2023.3
conda activate llm

# Usage:
#   sbatch slurm/visualize_adroit.slurm config/profiles/ngram_weat.yml
#   sbatch slurm/visualize_adroit.slurm config/profiles/renminribao_weat.yml

CONFIG="${1:-config/config.yml}"

echo "Visualizing with config=$CONFIG"
python -m scripts.visualize main --config="$CONFIG"
