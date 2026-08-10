#!/bin/bash
#SBATCH --job-name=prepare_us_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/prepare_us_data_%j.out
#SBATCH --error=logs/prepare_us_data_%j.err

# Acquire + build US corpora. The download steps need internet — run on an
# internet-enabled node/partition. 3DLNews2 needs a one-time `globus login`
# on the login node BEFORE submitting. Pass the arm's config as $1.
#   sbatch slurm/prepare_us_data.slurm config/profiles/garg_weat_american_stories.yml
#   sbatch slurm/prepare_us_data.slurm config/profiles/garg_weat_dlnews.yml

module load anaconda3/2023.3
conda activate llm
set -u

CONFIG="${1:?pass a config profile}"
ARM=$(python3 -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['_arm'])")

# Shapefile (once; harmless if it already exists)
python -m scripts.data_prep.fetch_us_shapefile || echo "WARN: shapefile fetch failed"

if [ "$ARM" = "american_stories" ]; then
    python -m scripts.data_prep.us_state_mapper build --config="$CONFIG"
    python -m scripts.data_prep.download_american_stories --config="$CONFIG"
else
    python -m scripts.data_prep.download_dlnews --config="$CONFIG"
fi

python -m scripts.data_prep.build_corpora_us --config="$CONFIG" --arm="$ARM"
echo "DONE prepare_us_data: $ARM"
