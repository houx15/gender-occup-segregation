#!/bin/bash
#SBATCH --job-name=garg_weat_pku
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# PKU-server CHINESE PROVINCIAL garg_weat: analyze_garg_weat + visualize for the
# provincial arms, producing the cross-province RND views (rankings / heatmap /
# choropleth) that replace the Cohen's d provincial plots.
#   - Weibo     (informal lists, /lustre)
#   - Newspaper (formal lists, /lustre)
# Prereq: cleaned_*.txt exist — run slurm/prepare_wordlists_pku.slurm first.
#
# Requires the 'opinion' env to have pyarrow (analyze_garg_weat writes parquet)
# and geopandas (provincial choropleths) — both already used by the WEAT
# provincial pipeline. The choropleth uses the China shapefile at
# /lustre/home/2401111059/youth-analysis/configs/china_shp/...
#
# Pass explicit configs to scope the run, e.g.:
#   sbatch slurm/garg_weat_pku.slurm config/profiles/garg_weat_weibo.yml

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
echo "Garg-WEAT (PKU provincial) run — ${#CONFIGS[@]} config(s)"
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

    if ! python -m scripts.analyze_garg_weat --config="$CONFIG"; then
        echo "  ERROR: analyze_garg_weat failed"
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

    # Provincial arms emit garg_weat_provincial_* / garg_weat_choropleth_*.
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
