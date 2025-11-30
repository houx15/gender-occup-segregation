#!/bin/bash
# Server setup script for Chinese Google 5-gram Occupation-Gender-Prestige Project
# Run this script after cloning the repository on your server

set -e  # Exit on error

echo "=========================================="
echo "Server Setup for Gender-Prestige Analysis"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on server (not macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${YELLOW}Warning: This appears to be macOS. This script is intended for Linux servers.${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo -e "${GREEN}Project root: $PROJECT_ROOT${NC}"
echo ""

# Step 1: Check Python version
echo "Step 1: Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    echo -e "${GREEN}✓ Python 3 found: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Step 2: Create virtual environment
echo ""
echo "Step 2: Setting up virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists${NC}"
    read -p "Recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo -e "${GREEN}✓ Virtual environment recreated${NC}"
    fi
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Step 3: Install dependencies
echo ""
echo "Step 3: Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 4: Create config file
echo ""
echo "Step 4: Creating configuration file..."
if [ -f "config/config.yml" ]; then
    echo -e "${YELLOW}config/config.yml already exists${NC}"
    read -p "Overwrite with template? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing config/config.yml"
    else
        cp config/config.example.yml config/config.yml
        echo -e "${GREEN}✓ Created config/config.yml from template${NC}"
    fi
else
    cp config/config.example.yml config/config.yml
    echo -e "${GREEN}✓ Created config/config.yml from template${NC}"
fi

# Step 5: Update paths in config
echo ""
echo "Step 5: Updating paths in config file..."
echo -e "${YELLOW}Please provide the following paths for your server:${NC}"
echo ""

# Get base directory
DEFAULT_BASE_DIR="$PROJECT_ROOT"
read -p "Base directory [$DEFAULT_BASE_DIR]: " BASE_DIR
BASE_DIR=${BASE_DIR:-$DEFAULT_BASE_DIR}

# Get data directory (may be on different partition)
read -p "Data directory (for large files) [$BASE_DIR/data]: " DATA_DIR
DATA_DIR=${DATA_DIR:-$BASE_DIR/data}

# Update config.yml
sed -i.bak "s|base_dir: \"/path/to/your/project\"|base_dir: \"$BASE_DIR\"|g" config/config.yml
sed -i.bak "s|/path/to/your/project/data/raw_ngrams|$DATA_DIR/raw_ngrams|g" config/config.yml
sed -i.bak "s|/path/to/your/project/data/raw_ngrams_decompressed|$DATA_DIR/raw_ngrams_decompressed|g" config/config.yml
sed -i.bak "s|/path/to/your/project/data/corpora|$DATA_DIR/corpora|g" config/config.yml
sed -i.bak "s|/path/to/your/project/data/models|$DATA_DIR/models|g" config/config.yml
sed -i.bak "s|/path/to/your/project/data/results|$DATA_DIR/results|g" config/config.yml
sed -i.bak "s|/path/to/your/project/logs|$BASE_DIR/logs|g" config/config.yml

rm -f config/config.yml.bak

echo -e "${GREEN}✓ Configuration updated${NC}"

# Step 6: Adjust worker count
echo ""
echo "Step 6: Setting CPU worker count..."
NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "16")
echo "Detected CPU cores: $NUM_CORES"
read -p "Workers to use for training [$NUM_CORES]: " WORKERS
WORKERS=${WORKERS:-$NUM_CORES}

sed -i.bak "s|workers: 16|workers: $WORKERS|g" config/config.yml
rm -f config/config.yml.bak

echo -e "${GREEN}✓ Worker count set to $WORKERS${NC}"

# Step 7: Create directory structure
echo ""
echo "Step 7: Creating directory structure..."
mkdir -p "$DATA_DIR"/{raw_ngrams,raw_ngrams_decompressed,corpora,models,results/plots}
mkdir -p "$BASE_DIR"/logs
mkdir -p notebooks

echo -e "${GREEN}✓ Directories created${NC}"

# Step 8: Set permissions
echo ""
echo "Step 8: Setting file permissions..."
chmod +x scripts/*.py
chmod +x run_pipeline.sh
chmod 600 config/config.yml  # Protect config file

echo -e "${GREEN}✓ Permissions set${NC}"

# Step 9: Run tests
echo ""
echo "Step 9: Running tests..."
if python -m unittest discover tests/ -v; then
    echo -e "${GREEN}✓ All tests passed${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo "You may still proceed, but check the errors above"
fi

# Step 10: Summary
echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Base directory: $BASE_DIR"
echo "  Data directory: $DATA_DIR"
echo "  CPU workers: $WORKERS"
echo ""
echo "Next steps:"
echo "  1. Review config/config.yml and adjust if needed"
echo "  2. Activate the environment: source venv/bin/activate"
echo "  3. Start the pipeline:"
echo "     - Full pipeline: ./run_pipeline.sh"
echo "     - Step by step: python scripts/download_ngrams.py --config config/config.yml"
echo ""
echo "For server deployment, consider using screen or tmux:"
echo "  screen -S gender-analysis"
echo "  source venv/bin/activate"
echo "  ./run_pipeline.sh 2>&1 | tee pipeline.log"
echo ""
echo "See DEPLOY.md for detailed server deployment instructions."
echo ""

# Check disk space
echo "Disk space check:"
df -h "$DATA_DIR" | tail -1 | awk '{print "  Available: " $4 " (" $5 " used)"}'
echo ""
AVAILABLE_GB=$(df -BG "$DATA_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -lt 100 ]; then
    echo -e "${RED}⚠ Warning: Less than 100GB available. You may need more space.${NC}"
else
    echo -e "${GREEN}✓ Sufficient disk space available${NC}"
fi
echo ""
