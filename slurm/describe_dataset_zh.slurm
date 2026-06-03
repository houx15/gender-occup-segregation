#!/bin/bash
#SBATCH --job-name=describe_dataset_zh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu
#SBATCH --output=logs/describe_dataset_zh_%j.out
#SBATCH --error=logs/describe_dataset_zh_%j.err

# CHINESE describe_dataset: produces a methods-section Markdown summary
# (<results_dir>/dataset_summary.md) per source, with corpus totals
# (documents, tokens, raw vocab, model vocab), raw-data footprint
# (files + bytes), training hyperparameters, and a per-unit breakdown.
#
# Defaults to the Princeton-hosted longitudinal arms (Renminribao + China
# Ngram). The provincial arms (Weibo + provincial newspaper, both on the PKU
# /lustre server) use a different Slurm style — run those with
# slurm/describe_dataset_pku.slurm instead.
#
# Re-runs are near-instant once the per-unit .dataset_stats.json sidecars
# are warm (cold first runs scan corpora; expect minutes for RMRB, longer
# for China Ngram). Pass --force in a one-off invocation to recompute.
#
# Pass explicit configs to scope the run, e.g.:
#   sbatch slurm/describe_dataset_zh.slurm config/profiles/garg_weat_renminribao.yml

module load anaconda3/2023.3
conda activate llm

set -u

DEFAULT_CONFIGS=(
    "config/profiles/garg_weat_renminribao.yml"
    "config/profiles/garg_weat_china_ngram.yml"
    "config/profiles/garg_weat_china_ngram_subsampled.yml"
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

echo "=========================================="
echo "describe_dataset (Chinese) — ${#CONFIGS[@]} config(s)"
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
        STATUSES+=("missing_config"); continue
    fi

    CORPORA_DIR=$(read_config_value "$CONFIG" "c['paths']['corpora_dir']")
    MODELS_DIR=$(read_config_value "$CONFIG" "c['paths'].get('models_dir', '')")
    RESULTS_DIR=$(read_config_value "$CONFIG" "c['paths']['results_dir']")

    if [ ! -d "$CORPORA_DIR" ] && [ ! -d "$MODELS_DIR" ]; then
        echo "  SKIP: neither corpora_dir nor models_dir exists"
        echo "        corpora_dir=$CORPORA_DIR"
        echo "        models_dir=$MODELS_DIR"
        STATUSES+=("no_corpora_or_models_dir"); continue
    fi

    if [ -d "$CORPORA_DIR" ]; then
        echo "  corpora_dir=$CORPORA_DIR"
    else
        echo "  corpora_dir missing — falling back to models_dir=$MODELS_DIR"
    fi

    if ! python -m scripts.describe_dataset --config="$CONFIG"; then
        echo "  ERROR: describe_dataset failed"
        STATUSES+=("describe_failed"); continue
    fi

    SUMMARY_PATH="$RESULTS_DIR/dataset_summary.md"
    if [ ! -f "$SUMMARY_PATH" ]; then
        echo "  ERROR: expected summary not written: $SUMMARY_PATH"
        STATUSES+=("no_summary_written"); continue
    fi

    SUMMARY_BYTES=$(wc -c < "$SUMMARY_PATH" | tr -d ' ')
    echo "  ok: $SUMMARY_PATH (${SUMMARY_BYTES} bytes)"
    STATUSES+=("ok")
done

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
for j in "${!CONFIGS[@]}"; do
    printf "  %-58s  %s\n" "${CONFIGS[$j]}" "${STATUSES[$j]}"
done
echo "=========================================="
