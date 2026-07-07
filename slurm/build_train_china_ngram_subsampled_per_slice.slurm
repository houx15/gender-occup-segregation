#!/bin/bash
#SBATCH --job-name=cns_per_slice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yh6580@princeton.edu
#SBATCH --output=logs/cns_per_slice_%j.out
#SBATCH --error=logs/cns_per_slice_%j.err

# Plan-B driver for the subsampled China Ngram retrain under disk-quota
# pressure: process slices one at a time, train each model immediately, then
# DELETE that slice's corpus_subsampled/<slice>/ to reclaim disk before the
# next slice's build starts.
#
# Peak disk during this run ≈ ONE slice's corpus_subsampled + cumulative
# models_subsampled (small, ~5 GB total at completion). Compare to running
# slurm/build_corpus_zh.slurm + slurm/train_zh.slurm separately, which
# would hold ALL slices on disk simultaneously.
#
# Corpora are stored COMPACT ("ngram<TAB>count", one line per unique ngram),
# so a slice is now type-sized (~10-16 GB even for boom decades) instead of the
# token-expanded 85 GB. train_embeddings expands the counts to sentences at read
# time (see CorpusIterator expand_counts). The cap is kept purely as a
# training-time bound now (Σcount ≈ 1e9/slice keeps the single-threaded expand
# iterator cheap), not a disk workaround.
#
# Usage:
#   sbatch slurm/build_train_china_ngram_subsampled_per_slice.slurm
#   sbatch slurm/build_train_china_ngram_subsampled_per_slice.slurm "1990 1995"
#
# Slices are identified by their START YEAR (4-digit int). The build/train
# scripts look up the canonical (start, end) pair from the profile's
# time_slices block. We pass start year — not YYYY_YYYY — because Fire's
# CLI parser coerces "1940_1949" to the int 19401949 (PEP 515: underscores
# are valid inside Python int literals).
#
# If args are passed, they override the default slice list (space-separated).
#
# 48h wall-clock cap on Princeton: the default 17-slice sweep is risky in
# one job. Split into 2-3 sbatch invocations covering disjoint start years:
#   sbatch slurm/build_train_china_ngram_subsampled_per_slice.slurm \
#     "1940 1945 1950 1955 1960 1965"
#   sbatch slurm/build_train_china_ngram_subsampled_per_slice.slurm \
#     "1970 1975 1980 1985 1990 1995"
#   sbatch slurm/build_train_china_ngram_subsampled_per_slice.slurm \
#     "2000 2005 2010 2015 2020"
# Jobs queue independently; each handles its own start-year set.
#
# IMPORTANT: per_year_capped needs the data-derived per-year denominator
# (raw_ngrams/totalcounts-5.derived). A single invocation builds it lazily on
# the first slice. If you SPLIT into concurrent invocations (as above), run the
# one-time precompute FIRST so they don't each redo the full pass and race:
#   sbatch slurm/precompute_ngram_totalcounts.slurm   # wait for it to finish
#   sbatch slurm/build_train_china_ngram_subsampled_per_slice.slurm "1970 1975 ..."

module load anaconda3/2023.3
conda activate llm

set -u

CONFIG="config/profiles/garg_weat_china_ngram_subsampled.yml"

# Default slice list — start years matching the time_slices block in the profile:
#   start_year=1940, end_year=2020, window_size=10, step_size=5 → 17 slices
DEFAULT_SLICES="1940 1945 1950 1955 1960 1965 1970 1975 1980 1985 1990 1995 2000 2005 2010 2015 2020"

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
    echo "[$i] SLICE start_year=$SLICE"
    echo "=========================================="

    # 1. Build corpus for this slice (build_corpora_ngram looks up the
    #    canonical YYYY_YYYY from the profile's time_slices block).
    if ! python -m scripts.data_prep.build_corpora_ngram --config="$CONFIG" --slice="$SLICE"; then
        echo "[$i] BUILD FAILED for $SLICE"
        STATUSES+=("$SLICE build_failed"); continue
    fi

    # 2. Find the canonical slice dir that was built (e.g. 1940 → 1940_1949,
    #    2020 → 2020_2020). Glob on "${SLICE}_*" — should match exactly one
    #    dir because start years are unique across the time_slices.
    SLICE_DIRS=( "$CORPORA_DIR/${SLICE}_"* )
    if [ ${#SLICE_DIRS[@]} -ne 1 ] || [ ! -d "${SLICE_DIRS[0]}" ]; then
        echo "[$i] SKIP: expected one dir at $CORPORA_DIR/${SLICE}_*, found ${#SLICE_DIRS[@]}"
        STATUSES+=("$SLICE no_corpus_dir"); continue
    fi
    SLICE_DIR="${SLICE_DIRS[0]}"
    SLICE_SIZE=$(du -sh "$SLICE_DIR" | cut -f1)
    echo "[$i] corpus built: $SLICE_DIR ($SLICE_SIZE)"

    # 2b. Cap guard. With a correct per-year denominator, a W-year slice emits
    #     at most W*cap TOKENS. Blowing past that means the cap didn't bite (the
    #     totalcounts under-count bug that made 1975_1984 reach 6.7e9 tokens).
    #     Corpora are stored COMPACT — one "ngram<TAB>count" line per unique
    #     ngram — so token count is Σ of the count column, NOT `wc -l` (which now
    #     counts unique ngrams). Sum column 2. Fail loudly if the cap didn't bite.
    CAP=$(python3 -c "import yaml;print(int(yaml.safe_load(open('$CONFIG'))['corpus']['per_year_token_cap']))")
    WIN=$(python3 -c "import yaml;print(int(yaml.safe_load(open('$CONFIG'))['time_slices']['window_size']))")
    TOKENS=$(find "$SLICE_DIR" -name 'corpus_*.txt' -exec cat {} + | awk -F'\t' '{s+=$2} END{printf "%.0f", s}')
    MAX=$(python3 -c "print(int($WIN * $CAP * 1.1))")
    echo "[$i] cap guard: $TOKENS tokens (max $MAX = ${WIN}y * ${CAP} * 1.1)"
    if [ "$TOKENS" -gt "$MAX" ]; then
        echo "[$i] GUARD FAILED for $SLICE: $TOKENS > $MAX — cap not biting (denominator wrong?)"
        STATUSES+=("$SLICE guard_failed"); continue
    fi

    # 3. Train one model on this slice. train_embeddings filters units by
    #    startswith(str(unit)) — passing "1940" matches the "1940_1949" unit.
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
