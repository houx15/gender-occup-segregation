#!/bin/bash
#SBATCH --job-name=build_prov_news
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# Build province-year corpora from provincial newspaper text data.
#
# Usage:
#   sbatch slurm/build_corpus_provincial_newspaper.slurm
#   sbatch slurm/build_corpus_provincial_newspaper.slurm config/config.yml
#   sbatch slurm/build_corpus_provincial_newspaper.slurm config/config.yml --group=0
#   sbatch slurm/build_corpus_provincial_newspaper.slurm config/config.yml --province=北京

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

CONFIG="${1:-config/config.yml}"
shift 2>/dev/null || true

echo "Building provincial newspaper corpora: config=$CONFIG, extra args: $@"
echo "Started at: $(date)"

python -m scripts.data_prep.build_corpora_provincial_newspaper --config="$CONFIG" "$@"

echo "Completed at: $(date)"
