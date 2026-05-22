#!/bin/bash
#SBATCH --job-name=garg_weat_zh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu
#SBATCH --output=logs/garg_weat_zh_%j.out
#SBATCH --error=logs/garg_weat_zh_%j.err

# CHINESE garg_weat pipeline: analyze_category_bias + visualize for the Chinese
# embedding sources. The orchestrator reads analysis.metrics ([rnd, cohens_d])
# from each profile, so this computes BOTH per-category RND and single-list
# Cohen's d. Kept separate from the English pipeline
# (slurm/garg_weat_all_sources.slurm).
#
# Defaults to the Princeton-hosted longitudinal arms (Renminribao + China
# Ngram). The provincial arms (Weibo + provincial newspaper, both on the PKU
# /lustre server) use a different Slurm style — run those with
# slurm/garg_weat_pku.slurm instead.
#
# Prereq: cleaned_*.txt must exist — run slurm/prepare_wordlists_zh.slurm first.

module load anaconda3/2023.3
conda activate llm

set -u

DEFAULT_CONFIGS=(
    "config/profiles/garg_weat_renminribao.yml"
    "config/profiles/garg_weat_china_ngram.yml"
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
echo "Garg-WEAT (Chinese) multi-source run"
echo "  Configs: ${#CONFIGS[@]}"
for cfg in "${CONFIGS[@]}"; do echo "    - $cfg"; done
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
        STATUSES+=("missing_config"); continue
    fi

    MODELS_DIR=$(read_config_value "$CONFIG" "c['paths']['models_dir']")
    RESULTS_DIR=$(read_config_value "$CONFIG" "c['paths']['results_dir']")
    FIG_DIR=$(read_config_value "$CONFIG" "c['paths'].get('figures_dir', c['paths']['results_dir'] + '/figures')")

    if [ ! -d "$MODELS_DIR" ]; then
        echo "  SKIP: models_dir does not exist: $MODELS_DIR"
        STATUSES+=("no_models_dir"); continue
    fi
    NHIT=$(find "$MODELS_DIR" -maxdepth 2 \( -name '*.model' -o -name '*.kv' \) 2>/dev/null | head -1 | wc -l | tr -d ' ')
    if [ "$NHIT" -eq 0 ]; then
        echo "  SKIP: no .model / .kv files under $MODELS_DIR"
        STATUSES+=("empty_models_dir"); continue
    fi

    echo "  models_dir=$MODELS_DIR"

    if ! python -m scripts.analyze_category_bias --config="$CONFIG"; then
        echo "  ERROR: analyze_category_bias failed"
        STATUSES+=("analyze_failed"); continue
    fi

    SUMMARY_PATH="$RESULTS_DIR/garg_weat_summary_by_category.parquet"
    if [ ! -f "$SUMMARY_PATH" ]; then
        echo "  ERROR: expected summary not written: $SUMMARY_PATH"
        STATUSES+=("no_results_written"); continue
    fi

    if ! python -m scripts.visualize main --config="$CONFIG"; then
        echo "  ERROR: visualize failed"
        STATUSES+=("visualize_failed"); continue
    fi

    # Longitudinal arms emit fig2_garg_weat_categories__*__{bootstrap,subsample};
    # provincial arms emit garg_weat_provincial_* / garg_weat_choropleth_*.
    # Don't assert specific names — just report what landed.
    NPDF=$(find "$FIG_DIR" -maxdepth 1 -name '*garg_weat*.pdf' 2>/dev/null | wc -l | tr -d ' ')
    if [ "$NPDF" -eq 0 ]; then
        echo "  WARN: no garg_weat PDFs under $FIG_DIR"
        echo "        (visualize exited 0 — check the analyze_garg_weat log above"
        echo "         for consistent-set / OOV diagnostics)"
        STATUSES+=("no_figure_written"); continue
    fi
    echo "  ok: $SUMMARY_PATH  ($NPDF figure(s) under $FIG_DIR)"
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
