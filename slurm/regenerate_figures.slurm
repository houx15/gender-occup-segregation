#!/bin/bash
#SBATCH --job-name=revisualize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# Re-generate all figures with font fix and province labels.
# WEAT results are already computed — only visualization is re-run.
#
# Usage:
#   sbatch slurm/regenerate_figures.slurm

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

CONFIG="config/config.yml"

cd /lustre/home/2401111059/gender-occup-segregation

echo "=== Regenerating figures (font fix + province labels) ==="
echo "Config: $CONFIG"
echo "Date: $(date)"

python -m scripts.visualize main --config="$CONFIG"

echo "=== Done: $(date) ==="
