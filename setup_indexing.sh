#!/bin/bash

# Setup script for Phase 2 Indexing module
# Run this from the repository root

echo "=========================================="
echo "Phase 2 Indexing - Setup Script"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run Data Merger Module to create Unified Dataset
echo ""
echo "=========================================="
echo "Creating Unified Dataset..."
echo "=========================================="
python3 Data_Merger_Module.py

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To test the indexing module:"
echo "  python3 src/indexing/demo_indexing.py"
echo ""
echo "To build indexes from your unified dataset:"
echo "  python3 src/indexing/build_indexes.py \\"
echo "    --dataset Datasets/Unified_Dataset/train.jsonl \\"
echo "    --output data/indexes"
echo ""
