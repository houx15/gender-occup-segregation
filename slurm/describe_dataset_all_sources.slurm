#!/bin/bash
#SBATCH --job-name=describe_dataset_all_sources
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu
#SBATCH --output=logs/describe_dataset_all_sources_%j.out
#SBATCH --error=logs/describe_dataset_all_sources_%j.err

# ENGLISH describe_dataset: produces a methods-section Markdown summary
# (<results_dir>/dataset_summary.md) per English embedding source, with corpus
# totals (documents, tokens, raw vocab, model vocab), raw-data footprint
# (files + bytes), training hyperparameters, and a per-unit breakdown.
# The Chinese arms live in slurm/describe_dataset_zh.slurm and
# slurm/describe_dataset_pku.slurm — kept apart by server.
#
# Histwords/Google-ngram profiles use pretrained embeddings and may not have a
# raw_data_dir on this host — the walker will skip the raw-data section
# cleanly in that case (per-unit raw cells become n/a).
#
# Re-runs are near-instant once the per-unit .dataset_stats.json sidecars are
# warm. Pass --force in a one-off invocation to recompute.

module load anaconda3/2023.3
conda activate llm

set -u

DEFAULT_CONFIGS=(
    "config/profiles/garg_weat_coha_trained.yml"
    "config/profiles/garg_weat_coha_histwords_sgns.yml"
    "config/profiles/garg_weat_coha_histwords_svd.yml"
    "config/profiles/garg_weat_google_ngram_eng_all.yml"
    "config/profiles/garg_weat_google_ngram_eng_fiction_all.yml"
)

if [ "$#" -gt 0 ]; then
    CONFIGS=("$@")
else
    CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

read_config_value() {
    python3 -c "import yaml; c = yaml.safe_load(open('$1')); print($2)"
}

declare -a STATUSES
declare -a SOURCES
declare -a SUMMARY_PATHS

echo "=========================================="
echo "describe_dataset (English) — ${#CONFIGS[@]} config(s)"
for cfg in "${CONFIGS[@]}"; do echo "  - $cfg"; done
echo "=========================================="
echo ""

i=0
for CONFIG in "${CONFIGS[@]}"; do
    i=$((i + 1))
    echo "------------------------------------------"
    echo "[$i/${#CONFIGS[@]}] $CONFIG"
    echo "------------------------------------------"

    if [ ! -f "$CONFIG" ]; then
        echo "  ERROR: config file not found"
        STATUSES+=("missing_config"); SOURCES+=("?"); SUMMARY_PATHS+=("?"); continue
    fi

    SRC=$(read_config_value "$CONFIG" "c.get('embedding_source', '?')")
    CORPORA_DIR=$(read_config_value "$CONFIG" "c['paths']['corpora_dir']")
    RESULTS_DIR=$(read_config_value "$CONFIG" "c['paths']['results_dir']")
    SOURCES+=("$SRC")

    if [ ! -d "$CORPORA_DIR" ]; then
        echo "  SKIP: corpora_dir does not exist: $CORPORA_DIR"
        STATUSES+=("no_corpora_dir"); SUMMARY_PATHS+=("?"); continue
    fi

    echo "  source=$SRC  corpora_dir=$CORPORA_DIR"

    if ! python -m scripts.describe_dataset --config="$CONFIG"; then
        echo "  ERROR: describe_dataset failed"
        STATUSES+=("describe_failed"); SUMMARY_PATHS+=("?"); continue
    fi

    SUMMARY_PATH="$RESULTS_DIR/dataset_summary.md"
    if [ ! -f "$SUMMARY_PATH" ]; then
        echo "  ERROR: expected summary not written: $SUMMARY_PATH"
        STATUSES+=("no_summary_written"); SUMMARY_PATHS+=("$SUMMARY_PATH"); continue
    fi

    SUMMARY_BYTES=$(wc -c < "$SUMMARY_PATH" | tr -d ' ')
    echo "  ok: $SUMMARY_PATH (${SUMMARY_BYTES} bytes)"
    STATUSES+=("ok"); SUMMARY_PATHS+=("$SUMMARY_PATH")
done

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
for j in "${!CONFIGS[@]}"; do
    printf "  %-60s  %-22s\n" "${CONFIGS[$j]}" "${STATUSES[$j]}"
    printf "      summary: %s\n" "${SUMMARY_PATHS[$j]}"
done
echo "=========================================="
