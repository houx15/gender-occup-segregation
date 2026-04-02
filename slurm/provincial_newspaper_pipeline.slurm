#!/bin/bash
#SBATCH --job-name=prov_news_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# Full pipeline for provincial newspaper gender norm analysis.
# Runs: corpus building -> embedding training (sequential) -> WEAT analysis -> visualization
#
# For PARALLEL training, run steps individually instead:
#   sbatch slurm/build_corpus_provincial_newspaper.slurm
#   sbatch slurm/train_provincial_newspaper.slurm   # <-- submits 8-group array job
#   sbatch slurm/analyze.slurm
#
# Usage:
#   sbatch slurm/provincial_newspaper_pipeline.slurm
#   sbatch slurm/provincial_newspaper_pipeline.slurm config/config.yml

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

CONFIG="${1:-config/config.yml}"

echo "=========================================="
echo "Provincial Newspaper Pipeline"
echo "Config: $CONFIG"
echo "Started at: $(date)"
echo "=========================================="

# Step 1: Build corpora
echo ""
echo "Step 1: Building corpora..."
python -m scripts.data_prep.build_corpora_provincial_newspaper --config="$CONFIG"
echo "Step 1 completed at: $(date)"

# Step 2: Train embeddings (sequential in this script; use array job for parallel)
echo ""
echo "Step 2: Training embeddings..."
python -m scripts.train_embeddings --config="$CONFIG"
echo "Step 2 completed at: $(date)"

# Step 3: WEAT analysis
echo ""
echo "Step 3: Running WEAT analysis..."
python -m scripts.analyze_weat --config="$CONFIG"
echo "Step 3 completed at: $(date)"

# Step 4: Visualization
echo ""
echo "Step 4: Creating visualizations..."
python -m scripts.visualize --config="$CONFIG"
echo "Step 4 completed at: $(date)"

echo ""
echo "=========================================="
echo "Pipeline completed at: $(date)"
echo "=========================================="
