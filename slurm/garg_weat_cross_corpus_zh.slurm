#!/bin/bash
#SBATCH --job-name=garg_weat_cross_corpus_zh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/garg_weat_cross_corpus_zh_%j.out
#SBATCH --error=logs/garg_weat_cross_corpus_zh_%j.err

module load anaconda3/2023.3
conda activate llm

set -u

INST_CONFIG="config/profiles/garg_weat_renminribao.yml"
PUB_CONFIG="config/profiles/garg_weat_china_ngram.yml"

read_config_value() {
    python3 -c "import yaml; c = yaml.safe_load(open('$1')); print($2)"
}

# Output dir: sibling of the institutional results_dir (where cross_corpus
# defaults), so the validation below looks in the right place.
INST_RESULTS=$(read_config_value "$INST_CONFIG" "c['paths']['results_dir']")
FIG_DIR="$(dirname "$INST_RESULTS")/figures_garg_weat_cross_corpus_zh"

mkdir -p logs

echo "Rendering cross-corpus discourse figures into: $FIG_DIR"

if ! python -m scripts.visualize cross_corpus \
    --institutional_config="$INST_CONFIG" \
    --public_config="$PUB_CONFIG"; then
    echo "ERROR: cross_corpus rendering failed (see above)" >&2
    exit 1
fi

# Validation: exactly 8 cross-corpus PDFs.
NPDF=$(find "$FIG_DIR" -maxdepth 1 -name 'fig_crosscorpus_*.pdf' | wc -l | tr -d ' ')
echo "Found $NPDF cross-corpus PDF(s) in $FIG_DIR"
if [ "$NPDF" -ne 8 ]; then
    echo "ERROR: expected 8 cross-corpus PDFs, found $NPDF" >&2
    exit 1
fi
echo "OK: 8 cross-corpus figures rendered."
