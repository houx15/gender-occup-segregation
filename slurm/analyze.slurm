#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu

module purge
module load anaconda3/2023.3
conda activate llm

# Usage:
#   sbatch slurm/analyze.slurm                     # uses config defaults
#   sbatch slurm/analyze.slurm config/config.yml   # explicit config

CONFIG="${1:-config/config.yml}"

ANALYSIS_MODE=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('analysis_mode', 'prestige'))")
DATA_SOURCE=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data_source'])")

echo "Analysis: mode=$ANALYSIS_MODE, data_source=$DATA_SOURCE, config=$CONFIG"

# Run analysis
case $ANALYSIS_MODE in
    prestige) python -m scripts.analyze_prestige --config="$CONFIG" ;;
    weat)     python -m scripts.analyze_weat --config="$CONFIG" ;;
    *)        echo "Unknown analysis_mode: $ANALYSIS_MODE"; exit 1 ;;
esac

# Visualization
python -m scripts.visualize --config="$CONFIG"

# Correlation (provincial + weat only)
if [ "$ANALYSIS_MODE" = "weat" ]; then
    if [ "$DATA_SOURCE" = "weibo" ] || [ "$DATA_SOURCE" = "newspaper" ]; then
        python -m scripts.analyze_correlation --config="$CONFIG"
    fi
fi
