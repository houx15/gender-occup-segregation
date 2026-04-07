#!/bin/bash
#SBATCH --job-name=network
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu

module purge
module load anaconda3/2023.3
conda activate llm

python scripts/build_corpora_renminribao.py


