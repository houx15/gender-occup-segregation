#!/bin/bash
#SBATCH --job-name=build_corpus_zh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu
#SBATCH --output=logs/build_corpus_zh_%j.out
#SBATCH --error=logs/build_corpus_zh_%j.err

# Princeton-llm-env build_corpus: dispatches per data_source. Same dispatch
# logic as the PKU-flavored slurm/build_corpus.slurm, just with the
# Princeton conda setup (anaconda3 module + llm env) and Princeton logging.
#
# Usage:
#   sbatch slurm/build_corpus_zh.slurm                                  # uses config/config.yml
#   sbatch slurm/build_corpus_zh.slurm config/profiles/<file>.yml       # explicit config
#   sbatch slurm/build_corpus_zh.slurm config/profiles/<file>.yml 1940_1949  # specific slice
#
# Typical Chinese-arm invocations (Princeton llm env):
#   sbatch slurm/build_corpus_zh.slurm config/profiles/garg_weat_china_ngram_weighted.yml
#   sbatch slurm/build_corpus_zh.slurm config/profiles/garg_weat_china_ngram.yml
#   sbatch slurm/build_corpus_zh.slurm config/profiles/garg_weat_renminribao.yml

module load anaconda3/2023.3
conda activate llm

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
