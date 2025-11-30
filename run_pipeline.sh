#!/bin/bash
# Pipeline runner for Chinese Google 5-gram Occupation-Gender-Prestige Project
# This script runs all four stages of the analysis pipeline

set -e  # Exit on error

echo "=========================================="
echo "Chinese Google 5-gram Analysis Pipeline"
echo "=========================================="
echo ""

CONFIG="config/config.yml"

# Parse command line arguments
SKIP_DOWNLOAD=false
SKIP_CORPUS=false
SKIP_TRAIN=false
EXPORT_PLOTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-corpus)
            SKIP_CORPUS=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --export-plots)
            EXPORT_PLOTS=true
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./run_pipeline.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-download   Skip the download step"
            echo "  --skip-corpus     Skip the corpus building step"
            echo "  --skip-train      Skip the embedding training step"
            echo "  --export-plots    Export diagnostic plots during analysis"
            echo "  --config PATH     Use custom config file (default: config/config.yml)"
            echo "  --help            Show this help message"
            echo ""
            echo "Example:"
            echo "  ./run_pipeline.sh --skip-download --export-plots"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Step 1: Download data
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "Step 1/4: Downloading Google 5-gram data..."
    echo "=========================================="
    python scripts/download_ngrams.py --config="$CONFIG"
    echo ""
else
    echo "Step 1/4: Skipping download (--skip-download flag set)"
    echo ""
fi

# Step 2: Build corpora
if [ "$SKIP_CORPUS" = false ]; then
    echo "Step 2/4: Building time-sliced corpora..."
    echo "=========================================="
    python scripts/build_corpora.py --config="$CONFIG"
    echo ""
else
    echo "Step 2/4: Skipping corpus building (--skip-corpus flag set)"
    echo ""
fi

# Step 3: Train embeddings
if [ "$SKIP_TRAIN" = false ]; then
    echo "Step 3/4: Training word embeddings..."
    echo "=========================================="
    python scripts/train_embeddings.py --config="$CONFIG"
    echo ""
else
    echo "Step 3/4: Skipping embedding training (--skip-train flag set)"
    echo ""
fi

# Step 4: Analyze embeddings
echo "Step 4/4: Analyzing embeddings..."
echo "=========================================="
if [ "$EXPORT_PLOTS" = true ]; then
    python scripts/analyze_embeddings.py --config="$CONFIG" --export_plots=True
else
    python scripts/analyze_embeddings.py --config="$CONFIG"
fi
echo ""

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo ""
echo "Results are available in the data/results/ directory:"
echo "  - occupation_gender_typing_by_decade.csv"
echo "  - occupation_prestige_by_decade.csv"
echo "  - occupation_gender_prestige_joint.csv"
echo "  - summary_statistics.csv"

if [ "$EXPORT_PLOTS" = true ]; then
    echo "  - plots/ (diagnostic visualizations)"
fi

echo ""
echo "Check logs/ directory for detailed execution logs."
