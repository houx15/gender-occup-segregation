#!/bin/bash
#SBATCH --job-name=diag_sizes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yh6580@princeton.edu
#SBATCH --output=logs/diag_sizes_%j.out
#SBATCH --error=logs/diag_sizes_%j.err

# Read-only diagnostic: explains the subsampled slice-size gap (e.g. 1975_1984
# at 85G vs 1970_1979 at 21G). Prints the per-year cap-vs-pass-through table and,
# for every slice dir still on disk, the emitted line count against the
# theoretical cap bound  sum_y min(year_total[y], cap).
#
#   lines/bound ~<= 1  -> real data under the cap (expected; cap may be a no-op)
#   lines/bound  >> 1  -> cap silently failed or data duplicated (a bug)
#
# Counting lines streams each corpus file (incl. the 85G slice) in 16 MB chunks,
# so memory stays tiny; wall-clock is dominated by reading bytes off disk.
#
# Usage:
#   sbatch slurm/diagnose_subsampled_sizes.slurm

module load anaconda3/2023.3
conda activate llm

set -u

CONFIG="config/profiles/garg_weat_china_ngram_subsampled.yml"

python -m scripts.diagnose_subsampled_sizes --config="$CONFIG"
