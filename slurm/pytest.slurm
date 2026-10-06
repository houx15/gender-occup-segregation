#!/bin/bash
#SBATCH --job-name=pytest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pytest_%j.out
#SBATCH --error=logs/pytest_%j.err

# Run tests in the llm env. pytest is not installed there, so its pure-Python
# packages are borrowed from the anaconda base install via a symlink shim.
#   sbatch slurm/pytest.slurm                          # all tests
#   sbatch slurm/pytest.slurm tests/test_us_analysis.py
module load anaconda3/2023.3
conda activate llm
set -eu
SHIM=/scratch/network/yh6580/gender-occup/pytest_shim
SITE=/usr/licensed/anaconda3/2023.3/lib/python3.10/site-packages
mkdir -p "$SHIM"
for m in pytest _pytest pluggy iniconfig exceptiongroup tomli py.py; do
    if [ -e "$SITE/$m" ]; then ln -sfn "$SITE/$m" "$SHIM/$m"; fi
done
PYTHONPATH="$SHIM" python -m pytest -q -p no:cacheprovider "${@:-tests/}"
