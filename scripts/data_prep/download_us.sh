#!/bin/bash
# One-shot US download phase — run on a login node WITH internet.
#
# For each arm config:
#   american_stories -> build LCCN->state table, then download the configured years (HF)
#   dlnews           -> Globus transfer (needs a prior `globus login` and dest_endpoint
#                       filled in the config)
# Also fetches the US states shapefile once (used later by the map step).
#
# Prereqs (once): conda env active; `pip install datasets globus-cli`;
#   for the dlnews arm: `globus login --no-local-server` and dest_endpoint set
#   in config/profiles/garg_weat_dlnews.yml.
#
# Usage:
#   bash scripts/data_prep/download_us.sh                 # both arms
#   bash scripts/data_prep/download_us.sh config/profiles/garg_weat_american_stories.yml
#   bash scripts/data_prep/download_us.sh config/profiles/garg_weat_dlnews.yml

set -u

CONFIGS=("$@")
if [ "${#CONFIGS[@]}" -eq 0 ]; then
    CONFIGS=(
        "config/profiles/garg_weat_american_stories.yml"
        "config/profiles/garg_weat_dlnews.yml"
    )
fi

# Shapefile (once; harmless if it already exists). Non-fatal.
echo "==== fetching US states shapefile ===="
python -m scripts.data_prep.fetch_us_shapefile || echo "WARN: shapefile fetch failed (retry later)"

for CONFIG in "${CONFIGS[@]}"; do
    if [ ! -f "$CONFIG" ]; then
        echo "ERROR: config not found: $CONFIG" >&2
        exit 1
    fi
    ARM=$(python3 -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['_arm'])")
    echo ""
    echo "==== download arm=$ARM  ($CONFIG) ===="

    if [ "$ARM" = "american_stories" ]; then
        python -m scripts.data_prep.us_state_mapper build --config="$CONFIG" || exit 1
        # Download only the raw tarballs (light, network). The heavy article
        # extraction + state matching happens OFFLINE in build_corpora_us on a
        # compute node, so it never trips the login node's time/memory limits.
        python -m scripts.data_prep.prefetch_american_stories --config="$CONFIG" || exit 1
    elif [ "$ARM" = "dlnews" ]; then
        python -m scripts.data_prep.download_dlnews --config="$CONFIG" || exit 1
    else
        echo "ERROR: unknown _arm=$ARM in $CONFIG" >&2
        exit 1
    fi
done

echo ""
echo "==== ALL DOWNLOADS DONE ===="
echo "Next: build corpora, then train/analyze via slurm. E.g."
echo "  python -m scripts.data_prep.build_corpora_us --config=<config>"
