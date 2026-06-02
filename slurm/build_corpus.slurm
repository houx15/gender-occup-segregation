#!/bin/bash
#SBATCH --job-name=build_corpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

# Usage:
#   sbatch slurm/build_corpus.slurm                          # uses config/config.yml
#   sbatch slurm/build_corpus.slurm config/config.yml        # explicit config
#   sbatch slurm/build_corpus.slurm config/config.yml 1940_1949  # specific slice

CONFIG="${1:-config/config.yml}"
SLICE="${2:-}"

ARGS="--config=$CONFIG"
if [ -n "$SLICE" ]; then
    ARGS="$ARGS --slice=$SLICE"
fi

DATA_SOURCE=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data_source'])")

echo "Building corpus: data_source=$DATA_SOURCE, config=$CONFIG"

case $DATA_SOURCE in
    ngram)                python -m scripts.data_prep.build_corpora_ngram $ARGS ;;
    renminribao)          python -m scripts.data_prep.build_corpora_rmrb $ARGS ;;
    weibo)                python -m scripts.data_prep.build_corpora_weibo $ARGS ;;
    newspaper)            python -m scripts.data_prep.build_corpora_newspaper $ARGS ;;
    provincial_newspaper) python -m scripts.data_prep.build_corpora_provincial_newspaper $ARGS ;;
    *)                    echo "Unknown data_source: $DATA_SOURCE"; exit 1 ;;
esac
