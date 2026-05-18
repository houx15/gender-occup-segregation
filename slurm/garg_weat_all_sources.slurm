#!/bin/bash
#SBATCH --job-name=garg_weat_all_sources
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu
#SBATCH --output=logs/garg_weat_all_sources_%j.out
#SBATCH --error=logs/garg_weat_all_sources_%j.err

# Run analyze_garg_weat + visualize for every embedding source we have on
# disk, producing one per-category-trend PDF per source. Mirrors
# fig2_all_sources.slurm but for the per-category RND metric instead of
# Garg's aggregate Fig 2 line.

module load anaconda3/2023.3
conda activate llm

set -u

DEFAULT_CONFIGS=(
    "config/profiles/garg_weat_coha_trained.yml"
    "config/profiles/garg_weat_coha_histwords_sgns.yml"
    "config/profiles/garg_weat_coha_histwords_svd.yml"
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
declare -a RESULT_PATHS
declare -a FIG_PATHS

echo "=========================================="
echo "Garg-WEAT multi-source run"
echo "  Configs:   ${#CONFIGS[@]}"
for cfg in "${CONFIGS[@]}"; do
    echo "    - $cfg"
done
echo "=========================================="
echo ""

i=0
for CONFIG in "${CONFIGS[@]}"; do
    i=$((i + 1))
    echo "------------------------------------------"
    echo "[${i}/${#CONFIGS[@]}] $CONFIG"
    echo "------------------------------------------"

    if [ ! -f "$CONFIG" ]; then
        echo "  ERROR: config file not found"
        STATUSES+=("missing_config")
        SOURCES+=("?")
        RESULT_PATHS+=("?")
        FIG_PATHS+=("?")
        continue
    fi

    SRC=$(read_config_value "$CONFIG" "c.get('embedding_source', '?')")
    MODELS_DIR=$(read_config_value "$CONFIG" "c['paths']['models_dir']")
    RESULTS_DIR=$(read_config_value "$CONFIG" "c['paths']['results_dir']")
    FIG_DIR=$(read_config_value "$CONFIG" "c['paths'].get('figures_dir', c['paths']['results_dir'] + '/figures')")
    SOURCES+=("$SRC")

    if [ ! -d "$MODELS_DIR" ]; then
        echo "  SKIP: models_dir does not exist: $MODELS_DIR"
        STATUSES+=("no_models_dir")
        RESULT_PATHS+=("?")
        FIG_PATHS+=("?")
        continue
    fi
    NHIT=$(find "$MODELS_DIR" -maxdepth 2 \( -name '*.model' -o -name '*.kv' -o -name '*-w.npy' \) 2>/dev/null | head -1 | wc -l | tr -d ' ')
    if [ "$NHIT" -eq 0 ]; then
        echo "  SKIP: no .model / .kv / *-w.npy files under $MODELS_DIR"
        STATUSES+=("empty_models_dir")
        RESULT_PATHS+=("?")
        FIG_PATHS+=("?")
        continue
    fi

    echo "  source=$SRC  models_dir=$MODELS_DIR"

    if ! python -m scripts.analyze_garg_weat --config="$CONFIG"; then
        echo "  ERROR: analyze_garg_weat failed"
        STATUSES+=("analyze_failed")
        RESULT_PATHS+=("?")
        FIG_PATHS+=("?")
        continue
    fi

    SUMMARY_PATH="$RESULTS_DIR/garg_weat_summary_by_category.parquet"
    if [ ! -f "$SUMMARY_PATH" ]; then
        echo "  ERROR: expected summary file not written: $SUMMARY_PATH"
        STATUSES+=("no_results_written")
        RESULT_PATHS+=("$SUMMARY_PATH")
        FIG_PATHS+=("?")
        continue
    fi

    if ! python -m scripts.visualize main --config="$CONFIG"; then
        echo "  ERROR: visualize failed"
        STATUSES+=("visualize_failed")
        RESULT_PATHS+=("$SUMMARY_PATH")
        FIG_PATHS+=("?")
        continue
    fi

    FIG_PATH="$FIG_DIR/fig2_garg_weat_categories__${SRC}.pdf"
    if [ ! -f "$FIG_PATH" ]; then
        echo "  ERROR: expected figure not written: $FIG_PATH"
        echo "         (visualize exited 0 — check the analyze_garg_weat log"
        echo "          above for consistent-set / OOV diagnostics)"
        STATUSES+=("no_figure_written")
        RESULT_PATHS+=("$SUMMARY_PATH")
        FIG_PATHS+=("$FIG_PATH")
        continue
    fi
    STATUSES+=("ok")
    RESULT_PATHS+=("$SUMMARY_PATH")
    FIG_PATHS+=("$FIG_PATH")
    echo "  ok: $SUMMARY_PATH"
    echo "  fig: $FIG_PATH"
done

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
for j in "${!CONFIGS[@]}"; do
    printf "  %-60s  %-22s\n" "${CONFIGS[$j]}" "${STATUSES[$j]}"
    printf "      summary: %s\n" "${RESULT_PATHS[$j]}"
    printf "      fig:     %s\n" "${FIG_PATHS[$j]}"
done
echo "=========================================="
