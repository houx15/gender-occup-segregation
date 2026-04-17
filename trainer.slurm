#!/bin/bash
#SBATCH --job-name=network
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu

module purge
module load anaconda3/2023.3
conda activate llm

python -m scripts.train_embeddings --config config/profiles/renminribao_server.yml


