#!/bin/bash
#SBATCH --job-name=describe_dataset_pku
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# PKU-server CHINESE PROVINCIAL describe_dataset: produces a methods-section
# Markdown summary (<results_dir>/dataset_summary.md) per source, with corpus
# totals (documents, tokens, raw vocab, model vocab), raw-data footprint
# (files + bytes, plus parquet row counts for Weibo), training hyperparameters,
# and a per-unit breakdown (rows = province-year).
#   - Weibo     (informal lists, /lustre)
#   - Newspaper (formal lists, /lustre)
#
# Re-runs are near-instant once .dataset_stats.json sidecars are warm. The
# first cold run scans all corpora — for Weibo this also opens each parquet
# (cheap metadata read) to bucket files by province; expect that to dominate.
# Pass --force in a one-off invocation to recompute.
#
# Requires the 'opinion' env to have pyarrow (Weibo walker uses parquet
# metadata) — already used by analyze_category_bias.
#
# Pass explicit configs to scope the run, e.g.:
#   sbatch slurm/describe_dataset_pku.slurm config/profiles/garg_weat_weibo.yml

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

# Repo checkout (edit if yours is elsewhere).
REPO_DIR=/lustre/home/2401111059/gender-occup-segregation
cd "$REPO_DIR"

set -u

DEFAULT_CONFIGS=(
    "config/profiles/garg_weat_weibo.yml"
    "config/profiles/garg_weat_provincial_newspaper.yml"
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
echo "describe_dataset (PKU provincial) — ${#CONFIGS[@]} config(s)"
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
    RESULTS_DIR=$(read_config_value "$CONFIG" "c['paths']['results_dir']")

    if [ ! -d "$CORPORA_DIR" ]; then
        echo "  SKIP: corpora_dir does not exist: $CORPORA_DIR"
        STATUSES+=("no_corpora_dir"); continue
    fi

    echo "  corpora_dir=$CORPORA_DIR"

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
