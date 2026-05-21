#!/bin/bash
#SBATCH --job-name=oov_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu
#SBATCH --output=logs/oov_probe_%j.out
#SBATCH --error=logs/oov_probe_%j.err

# Dry-run OOV diagnostic for the analyze_garg_weat experiment.
# Runs scripts.oov_probe across all three embedding sources using the
# Garg configs (which already point at the right models_dir) and the
# candidate wordlists under wordlists/en/garg_weat/.
#
# Outputs one CSV per source under
#   /scratch/network/yh6580/gender-occup/data/oov_probe/oov_<source>.csv
# plus a combined per-word coverage table (printed at the end of the job
# log, sorted by worst coverage first).

module load anaconda3/2023.3
conda activate llm

set -u

OUT_DIR="/scratch/network/yh6580/gender-occup/data/oov_probe"
mkdir -p "$OUT_DIR"

CONFIGS=(
    "config/profiles/coha_garg.yml"
    "config/profiles/coha_histwords_sgns.yml"
    "config/profiles/coha_histwords_svd.yml"
)

CANDIDATES="candidates_leadership.txt,candidates_family.txt,candidates_science.txt"

for CFG in "${CONFIGS[@]}"; do
    SRC=$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG')).get('embedding_source','unknown'))")
    OUT="$OUT_DIR/oov_${SRC}.csv"
    echo ""
    echo "=========================================="
    echo "$CFG  ->  $OUT"
    echo "=========================================="
    python -m scripts.oov_probe \
        --config="$CFG" \
        --wordlist_dir="wordlists/en/garg_weat" \
        --candidates="$CANDIDATES" \
        --out="$OUT"
done

echo ""
echo "=========================================="
echo "COMBINED per-word coverage (ascending)"
echo "  rows near the top are the highest-OOV candidates."
echo "=========================================="
python3 - <<PY
import pandas as pd
from pathlib import Path
out_dir = Path("$OUT_DIR")
dfs = [pd.read_csv(p) for p in sorted(out_dir.glob("oov_*.csv"))]
if not dfs:
    print("(no oov_*.csv files found)")
    raise SystemExit(0)
df = pd.concat(dfs, ignore_index=True)
per_word = (
    df.groupby(["category", "word"])["in_vocab"]
      .agg(["sum", "count"])
      .rename(columns={"sum": "n_in_vocab", "count": "n_source_decade_pairs"})
)
per_word["coverage"] = per_word["n_in_vocab"] / per_word["n_source_decade_pairs"]
per_word = per_word.sort_values(["category", "coverage"])
print(per_word.to_string())
PY
