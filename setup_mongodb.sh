#!/bin/bash

################################################################################
# MongoDB Setup Script for AI Code Reviewer
# 
# This script automates the complete MongoDB setup process:
# 1. Installs MongoDB Community Edition (if not installed)
# 2. Starts MongoDB service
# 3. Installs Python dependencies
# 4. Creates database, collection, and indexes
#
# Usage:
#   ./setup_mongodb.sh
#
# Requirements:
#   - macOS with Homebrew installed
#   - Python 3.10+ with virtual environment activated
################################################################################

set -e  # Exit on error

echo "🚀 AI Code Reviewer - MongoDB Setup"
echo "===================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running in virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${RED}❌ Error: Not in a virtual environment${NC}"
    echo "Please activate your virtual environment first:"
    echo "  source venv/bin/activate"
    exit 1
fi

echo -e "${GREEN}✅ Virtual environment detected: $VIRTUAL_ENV${NC}"
echo ""

# Step 1: Check if Homebrew is installed
echo "📦 Step 1: Checking Homebrew..."
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}⚠️  Homebrew not found. Installing Homebrew...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo -e "${GREEN}✅ Homebrew is installed${NC}"
fi
echo ""

# Step 2: Install MongoDB
echo "🗄️  Step 2: Installing MongoDB Community Edition..."
if brew list mongodb-community@7.0 &> /dev/null; then
    echo -e "${GREEN}✅ MongoDB already installed${NC}"
else
    echo "Installing MongoDB..."
    brew tap mongodb/brew
    brew install mongodb-community@7.0
    echo -e "${GREEN}✅ MongoDB installed successfully${NC}"
fi
echo ""

# Step 3: Start MongoDB service
echo "▶️  Step 3: Starting MongoDB service..."
brew services start mongodb-community@7.0

# Wait for MongoDB to start
echo "Waiting for MongoDB to start..."
sleep 3

# Check if MongoDB is running
if brew services list | grep mongodb-community | grep started &> /dev/null; then
    echo -e "${GREEN}✅ MongoDB service is running${NC}"
else
    echo -e "${RED}❌ MongoDB service failed to start${NC}"
    echo "Try manually: brew services start mongodb-community@7.0"
    exit 1
fi
echo ""

# Step 4: Install Python dependencies
echo "🐍 Step 4: Installing Python dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install pymongo python-dotenv > /dev/null 2>&1
echo -e "${GREEN}✅ Python dependencies installed${NC}"
echo ""

# Step 5: Check .env file
echo "📝 Step 5: Checking .env file..."
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating default .env file...${NC}"
    cat > .env << 'EOF'
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=ai_code_reviewer
MONGODB_COLLECTION=code_metadata

# MongoDB Connection Pool Settings
MONGODB_MAX_POOL_SIZE=50
MONGODB_MIN_POOL_SIZE=10

# Application Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
EOF
    echo -e "${GREEN}✅ .env file created${NC}"
else
    echo -e "${GREEN}✅ .env file exists${NC}"
fi
echo ""

# Step 6: Update .gitignore
echo "🔒 Step 6: Updating .gitignore..."
if ! grep -q "^\.env$" .gitignore 2>/dev/null; then
    echo ".env" >> .gitignore
    echo -e "${GREEN}✅ Added .env to .gitignore${NC}"
else
    echo -e "${GREEN}✅ .env already in .gitignore${NC}"
fi
echo ""

# Step 7: Create MongoDB database and collection
echo "🗃️  Step 7: Creating database and collection..."
python3 src/indexing/setup_mongodb.py

if [ $? -eq 0 ]; then
    echo ""
    echo "===================================="
    echo -e "${GREEN}✨ MongoDB Setup Complete!${NC}"
    echo "===================================="
    echo ""
    echo "Next steps:"
    echo "  1. Build indexes and metadata:"
    echo "     python src/indexing/build_indexes.py --dataset Datasets/Unified_Dataset/train_100.jsonl"
    echo ""
    echo "  2. Test retrieval:"
    echo "     python src/indexing/demo_retriever.py --patch 'def foo(): return 1'"
    echo ""
    echo "MongoDB Info:"
    echo "  • Database: ai_code_reviewer"
    echo "  • Collection: code_metadata"
    echo "  • Connection: mongodb://localhost:27017/"
    echo ""
    echo "View your data:"
    echo "  • Terminal: mongosh"
    echo "  • GUI: brew install --cask mongodb-compass"
    echo "  • VS Code: Install 'MongoDB for VS Code' extension"
    echo ""
else
    echo ""
    echo -e "${RED}❌ MongoDB setup failed${NC}"
    echo "Check the error messages above for details."
    exit 1
fi
