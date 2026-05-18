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
    SOURCES+=("$SRC")

    if [ ! -d "$MODELS_DIR" ]; then
        echo "  SKIP: models_dir does not exist: $MODELS_DIR"
        STATUSES+=("no_models_dir")
        RESULT_PATHS+=("?")
        continue
    fi
    NHIT=$(find "$MODELS_DIR" -maxdepth 2 \( -name '*.model' -o -name '*.kv' -o -name '*-w.npy' \) 2>/dev/null | head -1 | wc -l | tr -d ' ')
    if [ "$NHIT" -eq 0 ]; then
        echo "  SKIP: no .model / .kv / *-w.npy files under $MODELS_DIR"
        STATUSES+=("empty_models_dir")
        RESULT_PATHS+=("?")
        continue
    fi

    echo "  source=$SRC  models_dir=$MODELS_DIR"

    if ! python -m scripts.analyze_weat --config="$CONFIG"; then
        echo "  ERROR: analyze_weat failed"
        STATUSES+=("analyze_failed")
        RESULT_PATHS+=("?")
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
        continue
    fi
    STATUSES+=("ok")
    RESULT_PATHS+=("$RESULT_PATH")
    echo "  ok: $RESULT_PATH"
done

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
for j in "${!CONFIGS[@]}"; do
    printf "  %-55s  %-22s  %s\n" "${CONFIGS[$j]}" "${STATUSES[$j]}" "${RESULT_PATHS[$j]}"
done
echo "=========================================="
