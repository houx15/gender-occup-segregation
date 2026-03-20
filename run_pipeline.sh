#!/bin/bash
# Unified pipeline runner for Gender-Occupation Segregation Analysis
#
# Auto-detects existing data at each stage and skips if found.
# Use --force-* flags to re-run a stage even if output exists.
#
# Usage:
#   ./run_pipeline.sh [OPTIONS]
#   ./run_pipeline.sh --config config/config.yml
#   ./run_pipeline.sh --force-train    # re-train even if models exist

set -e

CONFIG="config/config.yml"
FORCE_DOWNLOAD=false
FORCE_CORPUS=false
FORCE_TRAIN=false
FORCE_ANALYZE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force-download) FORCE_DOWNLOAD=true; shift ;;
        --force-corpus)   FORCE_CORPUS=true; shift ;;
        --force-train)    FORCE_TRAIN=true; shift ;;
        --force-analyze)  FORCE_ANALYZE=true; shift ;;
        --force-all)      FORCE_DOWNLOAD=true; FORCE_CORPUS=true; FORCE_TRAIN=true; FORCE_ANALYZE=true; shift ;;
        --config) CONFIG="$2"; shift 2 ;;
        --help)
            echo "Usage: ./run_pipeline.sh [OPTIONS]"
            echo ""
            echo "Each stage auto-skips if its output already exists."
            echo ""
            echo "Options:"
            echo "  --force-download  Force re-download even if raw data exists"
            echo "  --force-corpus    Force rebuild corpora even if they exist"
            echo "  --force-train     Force retrain models even if they exist"
            echo "  --force-analyze   Force re-analyze even if results exist"
            echo "  --force-all       Force re-run all stages"
            echo "  --config PATH     Config file (default: config/config.yml)"
            echo "  --help            Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Read config values
read_config() {
    python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print($1)"
}

DATA_SOURCE=$(read_config "c['data_source']")
ANALYSIS_MODE=$(read_config "c.get('analysis_mode', 'prestige')")
RAW_DIR=$(read_config "c['paths'].get('raw_ngram_dir', '')")
CORPORA_DIR=$(read_config "c['paths']['corpora_dir']")
MODELS_DIR=$(read_config "c['paths']['models_dir']")
RESULTS_DIR=$(read_config "c['paths']['results_dir']")
FIGURES_DIR=$(read_config "c['paths'].get('figures_dir', c['paths']['results_dir'] + '/figures')")

# Helper: count files matching a pattern in a directory
count_files() {
    local dir="$1"
    local pattern="$2"
    if [ -d "$dir" ]; then
        find "$dir" -name "$pattern" 2>/dev/null | head -100 | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

# Helper: check if directory has subdirectories with content
has_subdirs_with_content() {
    local dir="$1"
    if [ -d "$dir" ]; then
        local count=$(find "$dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1 | wc -l | tr -d ' ')
        [ "$count" -gt 0 ]
    else
        return 1
    fi
}

echo "=========================================="
echo "Gender-Occupation Segregation Pipeline"
echo "  Data source:    $DATA_SOURCE"
echo "  Analysis mode:  $ANALYSIS_MODE"
echo "  Auto-skip:      stages with existing output"
echo "=========================================="
echo ""

# ── Step 1: Download (ngram only) ──────────────────────────────────
if [ "$DATA_SOURCE" = "ngram" ]; then
    N_RAW=$(count_files "$RAW_DIR" "*.gz")
    if [ "$N_RAW" -gt 0 ] && [ "$FORCE_DOWNLOAD" = false ]; then
        echo "Step 1: SKIP download ($N_RAW .gz files found in raw_ngram_dir)"
    else
        echo "Step 1: Downloading Google 5-gram data..."
        python -m scripts.data_prep.download_ngrams --config="$CONFIG"
    fi
else
    echo "Step 1: SKIP download (not applicable for $DATA_SOURCE)"
fi
echo ""

# ── Step 2: Build corpora ──────────────────────────────────────────
N_CORPORA=0
if has_subdirs_with_content "$CORPORA_DIR"; then
    N_CORPORA=$(count_files "$CORPORA_DIR" "corpus_*")
fi

if [ "$N_CORPORA" -gt 0 ] && [ "$FORCE_CORPUS" = false ]; then
    echo "Step 2: SKIP corpus building ($N_CORPORA corpus files found)"
else
    echo "Step 2: Building corpora ($DATA_SOURCE)..."
    case $DATA_SOURCE in
        ngram)      python -m scripts.data_prep.build_corpora_ngram --config="$CONFIG" ;;
        renminribao) python -m scripts.data_prep.build_corpora_rmrb --config="$CONFIG" ;;
        weibo)      python -m scripts.data_prep.build_corpora_weibo --config="$CONFIG" ;;
        newspaper)  python -m scripts.data_prep.build_corpora_newspaper --config="$CONFIG" ;;
        *)          echo "Unknown data_source: $DATA_SOURCE"; exit 1 ;;
    esac
fi
echo ""

# ── Step 3: Train embeddings ──────────────────────────────────────
N_MODELS=$(count_files "$MODELS_DIR" "*.model")

if [ "$N_MODELS" -gt 0 ] && [ "$FORCE_TRAIN" = false ]; then
    echo "Step 3: SKIP training ($N_MODELS .model files found)"
else
    echo "Step 3: Training embeddings..."
    RETRAIN_FLAG=""
    if [ "$FORCE_TRAIN" = true ]; then RETRAIN_FLAG="--retrain"; fi
    python -m scripts.train_embeddings --config="$CONFIG" $RETRAIN_FLAG
fi
echo ""

# ── Step 4: Analysis ──────────────────────────────────────────────
# Check for existing result files
HAS_RESULTS=false
if [ "$ANALYSIS_MODE" = "prestige" ]; then
    if [ -f "$RESULTS_DIR/occupation_scores_by_slice.parquet" ] || [ -f "$RESULTS_DIR/occupation_scores_by_province.parquet" ]; then
        HAS_RESULTS=true
    fi
elif [ "$ANALYSIS_MODE" = "weat" ]; then
    if [ -f "$RESULTS_DIR/weat_results.csv" ]; then
        HAS_RESULTS=true
    fi
fi

if [ "$HAS_RESULTS" = true ] && [ "$FORCE_ANALYZE" = false ]; then
    echo "Step 4: SKIP analysis (result files found in results_dir)"
else
    echo "Step 4: Running analysis ($ANALYSIS_MODE)..."
    case $ANALYSIS_MODE in
        prestige) python -m scripts.analyze_prestige --config="$CONFIG" ;;
        weat)     python -m scripts.analyze_weat --config="$CONFIG" ;;
        *)        echo "Unknown analysis_mode: $ANALYSIS_MODE"; exit 1 ;;
    esac
fi
echo ""

# ── Step 5: Visualization ─────────────────────────────────────────
echo "Step 5: Creating visualizations..."
python -m scripts.visualize --config="$CONFIG"
echo ""

# ── Step 6: Correlation (provincial + weat only) ──────────────────
if [ "$ANALYSIS_MODE" = "weat" ]; then
    if [ "$DATA_SOURCE" = "weibo" ] || [ "$DATA_SOURCE" = "newspaper" ]; then
        echo "Step 6: Running correlation analysis..."
        python -m scripts.analyze_correlation --config="$CONFIG"
        echo ""
    fi
fi

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo ""
echo "Results: $RESULTS_DIR"
