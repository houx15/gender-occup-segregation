#!/bin/bash
#SBATCH --job-name=cnw_per_slice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=144:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yh6580@princeton.edu
#SBATCH --output=logs/cnw_per_slice_%j.out
#SBATCH --error=logs/cnw_per_slice_%j.err

# Plan-B driver for the count-weighted China Ngram retrain under disk-quota
# pressure: process slices one at a time, train each model immediately, then
# DELETE that slice's corpus_weighted/<slice>/ to reclaim disk before the
# next slice's build starts.
#
# Peak disk during this run ≈ ONE slice's corpus_weighted + cumulative
# models_weighted (small, ~5 GB total at completion). Compare to running
# slurm/build_corpus_zh.slurm + slurm/train_zh.slurm separately, which
# would hold ALL slices on disk simultaneously.
#
# Usage:
#   sbatch slurm/build_train_china_ngram_weighted_per_slice.slurm
#   sbatch slurm/build_train_china_ngram_weighted_per_slice.slurm "1990_1999 1995_2004"
#
# If args are passed, they override the default slice list (space-separated).

module load anaconda3/2023.3
conda activate llm

set -u

CONFIG="config/profiles/garg_weat_china_ngram_weighted.yml"

# Default slice list — must match the time_slices block in the profile:
#   start_year=1940, end_year=2020, window_size=10, step_size=5 → 17 slices
DEFAULT_SLICES="1940_1949 1945_1954 1950_1959 1955_1964 1960_1969 1965_1974 \
1970_1979 1975_1984 1980_1989 1985_1994 1990_1999 1995_2004 \
2000_2009 2005_2014 2010_2019 2015_2020 2020_2020"

if [ "$#" -gt 0 ]; then
    SLICES="$@"
else
    SLICES="$DEFAULT_SLICES"
fi

# Read corpora_dir from the profile so the delete step is correct even if the
# profile's path changes later.
CORPORA_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['corpora_dir'])")
echo "corpora_dir=$CORPORA_DIR"
echo "Slices to process: $SLICES"

declare -a STATUSES
i=0
for SLICE in $SLICES; do
    i=$((i + 1))
    echo ""
    echo "=========================================="
    echo "[$i] SLICE=$SLICE"
    echo "=========================================="

    # 1. Build corpus for this slice.
    if ! python -m scripts.data_prep.build_corpora_ngram --config="$CONFIG" --slice="$SLICE"; then
        echo "[$i] BUILD FAILED for $SLICE"
        STATUSES+=("$SLICE build_failed"); continue
    fi

    # 2. Sanity-check the corpus exists and is non-empty before training.
    SLICE_DIR="$CORPORA_DIR/$SLICE"
    if [ ! -d "$SLICE_DIR" ]; then
        echo "[$i] SKIP: $SLICE_DIR not created"
        STATUSES+=("$SLICE no_corpus_dir"); continue
    fi
    SLICE_SIZE=$(du -sh "$SLICE_DIR" | cut -f1)
    echo "[$i] corpus built: $SLICE_DIR ($SLICE_SIZE)"

    # 3. Train one model on this slice (writes to models_dir/, small).
    if ! python -m scripts.train_embeddings --config="$CONFIG" --unit="$SLICE"; then
        echo "[$i] TRAIN FAILED for $SLICE"
        STATUSES+=("$SLICE train_failed"); continue
    fi

    # 4. Delete this slice's corpus to reclaim disk before the next slice.
    rm -rf "$SLICE_DIR"
    echo "[$i] reclaimed: $SLICE_DIR"
    STATUSES+=("$SLICE ok")
done

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
for entry in "${STATUSES[@]}"; do
    echo "  $entry"
done
echo "=========================================="
