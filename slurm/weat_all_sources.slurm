#!/bin/bash
#SBATCH --job-name=weat_all_sources
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu
#SBATCH --output=logs/weat_all_sources_%j.out
#SBATCH --error=logs/weat_all_sources_%j.err

# Run analyze_weat for every embedding source we have on disk, producing one
# WEAT results CSV per source. Mirrors slurm/fig2_all_sources.slurm but for
# the WEAT (Cohen's d on stem / leadership / work-family) metric instead of
# Garg's RND.
#
# Usage:
#   sbatch slurm/weat_all_sources.slurm
#       Runs the default config list (trained COHA + HistWords SGNS + SVD).
#
#   sbatch slurm/weat_all_sources.slurm \
#       config/profiles/weat_coha_trained.yml \
#       config/profiles/weat_coha_histwords_sgns.yml
#       Custom list — exactly the configs you pass.
#
# Each config must already have its models_dir populated:
#   - weat_coha_trained.yml         → trained models from full_pipeline_en.slurm
#   - weat_coha_histwords_sgns.yml  → unzipped histwords_coha_sgns/sgns/<YYYY>-w.npy
#   - weat_coha_histwords_svd.yml   → unzipped histwords_coha_all/coha-word/svd/<YYYY>-w.npy

# NOTE: conda's activate script references $PS1, which is unset in non-
# interactive Slurm shells. Load + activate FIRST, then turn on `set -u`.
# We deliberately do NOT `set -e` — we want to continue past per-source failures.
module load anaconda3/2023.3
conda activate llm

set -u

DEFAULT_CONFIGS=(
    "config/profiles/weat_coha_trained.yml"
    "config/profiles/weat_coha_histwords_sgns.yml"
    "config/profiles/weat_coha_histwords_svd.yml"
)

if [ "$#" -gt 0 ]; then
    CONFIGS=("$@")
else
    CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

read_config_value() {
    # Args: <config_path> <python_expr_using_c_dict>
    python3 -c "import yaml; c = yaml.safe_load(open('$1')); print($2)"
}

declare -a STATUSES
declare -a SOURCES
declare -a RESULT_PATHS
declare -a FIG_PATHS

echo "=========================================="
echo "WEAT multi-source run"
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

    if ! python -m scripts.analyze_weat --config="$CONFIG"; then
        echo "  ERROR: analyze_weat failed"
        STATUSES+=("analyze_failed")
        RESULT_PATHS+=("?")
        FIG_PATHS+=("?")
        continue
    fi

    # Trust the file system: did analyze_weat actually write its primary
    # output? (weat_results.csv is the canonical Cohen's d table.)
    RESULT_PATH="$RESULTS_DIR/weat_results.csv"
    if [ ! -f "$RESULT_PATH" ]; then
        echo "  ERROR: expected results file not written: $RESULT_PATH"
        echo "         (analyze_weat exited 0 — check the log above for"
        echo "          OOV / gender-axis diagnostics)"
        STATUSES+=("no_results_written")
        RESULT_PATHS+=("$RESULT_PATH")
        FIG_PATHS+=("?")
        continue
    fi

    # Run visualize: produces heatmap, rankings, longitudinal trend PDFs
    # (filenames are date-prefixed, e.g. 20260518_weat_longitudinal_trend.pdf).
    if ! python -m scripts.visualize main --config="$CONFIG"; then
        echo "  ERROR: visualize failed (CSVs written but no figures)"
        STATUSES+=("visualize_failed")
        RESULT_PATHS+=("$RESULT_PATH")
        FIG_PATHS+=("?")
        continue
    fi

    # Confirm at least the longitudinal trend PDF landed. Date-prefix makes
    # an exact path impossible; glob for the most recent one.
    LATEST_FIG=$(ls -1t "$FIG_DIR"/*weat_longitudinal_trend.pdf 2>/dev/null | head -1)
    if [ -z "$LATEST_FIG" ]; then
        echo "  ERROR: no weat_longitudinal_trend.pdf under $FIG_DIR"
        STATUSES+=("no_figure_written")
        RESULT_PATHS+=("$RESULT_PATH")
        FIG_PATHS+=("?")
        continue
    fi
    STATUSES+=("ok")
    RESULT_PATHS+=("$RESULT_PATH")
    FIG_PATHS+=("$LATEST_FIG")
    echo "  ok: $RESULT_PATH"
    echo "  fig: $LATEST_FIG"
done

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
for j in "${!CONFIGS[@]}"; do
    printf "  %-55s  %-22s\n" "${CONFIGS[$j]}" "${STATUSES[$j]}"
    printf "      csv: %s\n" "${RESULT_PATHS[$j]}"
    printf "      fig: %s\n" "${FIG_PATHS[$j]}"
done
echo "=========================================="
