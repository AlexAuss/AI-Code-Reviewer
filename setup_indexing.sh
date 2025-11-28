#!/bin/bash

# Setup script for Phase 2 Indexing module
# Run this from the repository root

echo "=========================================="
echo "Phase 2 Indexing - Setup Script"
echo "=========================================="

if python3.10 -c "import venv" 2>/dev/null; then
    echo "python3.10-venv is already installed."
else
    echo "Installing python3.10-venv..."
    apt install -y python3.10-venv
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install Streamlit UI deps used by the code reviewer
# These installs are safe even if already present
echo ""
echo "=========================================="
echo "Installing Streamlit UI dependencies"
echo "=========================================="
pip install --upgrade \
    streamlit \
    streamlit-code-editor

# Optional quality of life packages for Streamlit dev
pip install --upgrade watchdog

# Run Data Merger Module to create Unified Dataset
echo ""
echo "=========================================="
echo "Creating Unified Dataset..."
echo "=========================================="
if [[ ! -f "Datasets/Unified_Dataset/train.jsonl" ]]; then
    python3 src/utils/Data_Merger_Module.py
else
    echo "Unified Dataset already exists at Datasets/Unified_Dataset/train.jsonl"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future:"
echo "  source .venv/bin/activate"
echo ""
echo "To test the indexing module:"
echo "  python3 src/indexing/demo_indexing.py"
echo ""
echo "To build indexes from your unified dataset:"
echo "  python3 src/indexing/build_indexes.py \\"
echo "    --dataset Datasets/Unified_Dataset/train.jsonl \\"
echo "    --output data/indexes"
echo ""
