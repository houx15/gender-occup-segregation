#!/bin/bash
#SBATCH --job-name=fig2_all_sources
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu
#SBATCH --output=logs/fig2_all_sources_%j.out
#SBATCH --error=logs/fig2_all_sources_%j.err

# Run analyze_garg + visualize for every embedding source we have on disk,
# producing one Fig 2 PDF per source. Each profile sets embedding_source so
# the figures land at fig2_garg_replication__<source>.pdf and don't
# overwrite each other.
#
# Usage:
#   sbatch slurm/fig2_all_sources.slurm
#       Runs the default config list (trained COHA + HistWords SGNS + SVD).
#
#   sbatch slurm/fig2_all_sources.slurm \
#       config/profiles/coha_garg.yml \
#       config/profiles/coha_histwords_sgns.yml
#       Custom list — exactly the configs you pass.
#
# Each config must already have its models_dir populated:
#   - coha_garg.yml          → trained models from full_pipeline_en.slurm
#   - coha_histwords_sgns.yml→ unzipped histwords_coha_sgns/<...>/<YYYY>-w.npy
#   - coha_histwords_svd.yml → unzipped histwords_coha_all/<...>/<YYYY>-w.npy
#
# After download_pretrained_embeddings.py finishes, look at MANIFEST.json
# and edit the matching profile's paths.models_dir to the resolved path.

# NOTE: conda's activate script references $PS1, which is unset in
# non-interactive Slurm shells. Load + activate FIRST, then turn on `set -u`.
# We deliberately do NOT `set -e` — we want to continue past per-source failures.
module load anaconda3/2023.3
conda activate llm

set -u

# Default config list — override by passing configs as positional args.
DEFAULT_CONFIGS=(
    "config/profiles/coha_garg.yml"
    "config/profiles/coha_histwords_sgns.yml"
    "config/profiles/coha_histwords_svd.yml"
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
declare -a FIGURES

echo "=========================================="
echo "Fig 2 multi-source run"
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
        FIGURES+=("?")
        continue
    fi

    SRC=$(read_config_value "$CONFIG" "c.get('embedding_source', '?')")
    MODELS_DIR=$(read_config_value "$CONFIG" "c['paths']['models_dir']")
    FIG_DIR=$(read_config_value "$CONFIG" "c['paths'].get('figures_dir', c['paths']['results_dir'] + '/figures')")
    SOURCES+=("$SRC")

    # Models dir must contain at least one usable model file.
    if [ ! -d "$MODELS_DIR" ]; then
        echo "  SKIP: models_dir does not exist: $MODELS_DIR"
        STATUSES+=("no_models_dir")
        FIGURES+=("?")
        continue
    fi
    NHIT=$(find "$MODELS_DIR" -maxdepth 2 \( -name '*.model' -o -name '*.kv' -o -name '*-w.npy' \) 2>/dev/null | head -1 | wc -l | tr -d ' ')
    if [ "$NHIT" -eq 0 ]; then
        echo "  SKIP: no .model / .kv / *-w.npy files under $MODELS_DIR"
        STATUSES+=("empty_models_dir")
        FIGURES+=("?")
        continue
    fi

    echo "  source=$SRC  models_dir=$MODELS_DIR"

    if ! python -m scripts.analyze_garg --config="$CONFIG"; then
        echo "  ERROR: analyze_garg failed"
        STATUSES+=("analyze_failed")
        FIGURES+=("?")
        continue
    fi

    if ! python -m scripts.visualize main --config="$CONFIG"; then
        echo "  ERROR: visualize failed"
        STATUSES+=("visualize_failed")
        FIGURES+=("?")
        continue
    fi

    FIG_PATH="$FIG_DIR/fig2_garg_replication__${SRC}.pdf"
    STATUSES+=("ok")
    FIGURES+=("$FIG_PATH")
    echo "  ok: $FIG_PATH"
done

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
for j in "${!CONFIGS[@]}"; do
    printf "  %-50s  %-20s  %s\n" "${CONFIGS[$j]}" "${STATUSES[$j]}" "${FIGURES[$j]}"
done
echo "=========================================="
