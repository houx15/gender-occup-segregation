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

# ENGLISH garg_weat pipeline: analyze_garg_weat + visualize for every English
# embedding source on disk, one per-category-trend PDF per source. Mirrors
# fig2_all_sources.slurm. Runs the analyze_category_bias orchestrator, which
# reads analysis.metrics from each profile ([rnd, cohens_d]) and computes BOTH
# the per-category RND and the single-list Cohen's d in one model-load pass.
# The Chinese arms
# live in a separate script — slurm/garg_weat_zh.slurm — to keep the two
# pipelines (and their servers) apart.

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

    if ! python -m scripts.analyze_category_bias --config="$CONFIG"; then
        echo "  ERROR: analyze_category_bias failed"
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

    FIG_BOOT="$FIG_DIR/fig2_garg_weat_categories__${SRC}__bootstrap.pdf"
    FIG_SUB="$FIG_DIR/fig2_garg_weat_categories__${SRC}__subsample.pdf"
    MISSING=""
    [ ! -f "$FIG_BOOT" ] && MISSING="$FIG_BOOT"
    [ ! -f "$FIG_SUB" ] && MISSING="$MISSING $FIG_SUB"
    if [ -n "$MISSING" ]; then
        echo "  ERROR: expected figure(s) not written:$MISSING"
        echo "         (visualize exited 0 — check the analyze_garg_weat log"
        echo "          above for consistent-set / OOV diagnostics)"
        STATUSES+=("no_figure_written")
        RESULT_PATHS+=("$SUMMARY_PATH")
        FIG_PATHS+=("$MISSING")
        continue
    fi
    STATUSES+=("ok")
    RESULT_PATHS+=("$SUMMARY_PATH")
    FIG_PATHS+=("$FIG_BOOT ; $FIG_SUB")
    echo "  ok: $SUMMARY_PATH"
    echo "  fig (bootstrap): $FIG_BOOT"
    echo "  fig (subsample): $FIG_SUB"

    # Word-level ideation drivers (secondary, NON-FATAL). Runs only here — after
    # the primary figures are validated and status recorded — so a driver bug
    # never regresses the main deliverable. Do not touch STATUSES here.
    if python -m scripts.analyze_word_drivers --config="$CONFIG" \
        && python -m scripts.visualize_word_drivers --config="$CONFIG"; then
        echo "  word_drivers: ok ($RESULTS_DIR/word_drivers_*.{parquet,csv})"
    else
        echo "  WARN: word_drivers step failed (primary figures unaffected)"
    fi
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
