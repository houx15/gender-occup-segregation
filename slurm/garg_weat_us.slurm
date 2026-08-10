#!/bin/bash
#SBATCH --job-name=garg_weat_us
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/garg_weat_us_%j.out
#SBATCH --error=logs/garg_weat_us_%j.err

# US Garg-WEAT: analyze_category_bias (RND + Cohen's d) + visualize (trend +
# US choropleth) for the two US arms. Mirrors slurm/garg_weat_zh.slurm.
module load anaconda3/2023.3
conda activate llm
set -u

DEFAULT_CONFIGS=(
    "config/profiles/garg_weat_american_stories.yml"
    "config/profiles/garg_weat_dlnews.yml"
)
if [ "$#" -gt 0 ]; then CONFIGS=("$@"); else CONFIGS=("${DEFAULT_CONFIGS[@]}"); fi

read_config_value() { python3 -c "import yaml; c=yaml.safe_load(open('$1')); print($2)"; }
declare -a STATUSES

for CONFIG in "${CONFIGS[@]}"; do
    echo "----- $CONFIG -----"
    if [ ! -f "$CONFIG" ]; then STATUSES+=("missing_config"); continue; fi
    MODELS_DIR=$(read_config_value "$CONFIG" "c['paths']['models_dir']")
    RESULTS_DIR=$(read_config_value "$CONFIG" "c['paths']['results_dir']")
    if [ ! -d "$MODELS_DIR" ]; then echo "SKIP: no models_dir"; STATUSES+=("no_models_dir"); continue; fi
    NHIT=$(find "$MODELS_DIR" -maxdepth 2 -name '*.kv' | head -1 | wc -l | tr -d ' ')
    if [ "$NHIT" -eq 0 ]; then echo "SKIP: no .kv"; STATUSES+=("empty_models_dir"); continue; fi

    if ! python -m scripts.analyze_category_bias --config="$CONFIG"; then
        STATUSES+=("analyze_failed"); continue; fi
    SUMMARY="$RESULTS_DIR/garg_weat_summary_by_category.parquet"
    if [ ! -f "$SUMMARY" ]; then STATUSES+=("no_results_written"); continue; fi
    if ! python -m scripts.visualize main --config="$CONFIG"; then
        STATUSES+=("visualize_failed"); continue; fi
    STATUSES+=("ok")
done

echo "===== SUMMARY ====="
for j in "${!CONFIGS[@]}"; do printf "  %-56s %s\n" "${CONFIGS[$j]}" "${STATUSES[$j]}"; done
