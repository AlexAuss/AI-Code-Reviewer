"""
Automated MongoDB setup - creates database, collection, and indexes.

Run this ONCE to initialize MongoDB for the project.

Your teammates just need to:
1. Install MongoDB (brew install mongodb-community@7.0)
2. Start service (brew services start mongodb-community@7.0)
3. Run this script (python src/indexing/setup_mongodb.py)
"""

import os
import sys
import logging
from pathlib import Path
from pymongo import MongoClient, ASCENDING
from pymongo.errors import CollectionInvalid, ConnectionFailure
from dotenv import load_dotenv
import subprocess

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load .env file
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)


def check_mongodb_status():
    """Check if MongoDB service is running."""
    logger.info("Checking MongoDB service status...")
    
    try:
        result = subprocess.run(
            ['brew', 'services', 'list'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if 'mongodb-community' in result.stdout and 'started' in result.stdout:
            logger.info("✅ MongoDB service is running")
            return True
        else:
            logger.error("❌ MongoDB service is NOT running")
            logger.error("Start it with: brew services start mongodb-community@7.0")
            return False
    except FileNotFoundError:
        logger.error("❌ Homebrew not found. Install MongoDB manually.")
        return False
    except Exception as e:
        logger.warning(f"⚠️  Could not check MongoDB status: {e}")
        logger.info("Attempting to connect anyway...")
        return True


def setup_mongodb():
    """
    Complete MongoDB setup - creates everything automatically.
    
    Steps:
    1. Connect to MongoDB (creates database if not exists)
    2. Create collection with schema validation
    3. Create indexes for fast queries
    4. Verify setup
    """
    
    # Read configuration from .env
    uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    db_name = os.getenv('MONGODB_DATABASE', 'ai_code_reviewer')
    collection_name = os.getenv('MONGODB_COLLECTION', 'code_metadata')
    
    logger.info("=" * 60)
    logger.info("Starting MongoDB Setup")
    logger.info("=" * 60)
    logger.info(f"MongoDB URI: {uri}")
    logger.info(f"Database: {db_name}")
    logger.info(f"Collection: {collection_name}")
    
    try:
        # Step 1: Connect to MongoDB
        logger.info("\n[1/4] Connecting to MongoDB...")
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        
        # Test connection
        client.admin.command('ping')
        logger.info("✅ MongoDB connection successful!")
        
        # Step 2: Create/get database (auto-created if not exists)
        logger.info(f"\n[2/4] Creating database '{db_name}'...")
        db = client[db_name]
        logger.info(f"✅ Database '{db_name}' ready")
        
        # Step 3: Create collection with schema validation
        logger.info(f"\n[3/4] Creating collection '{collection_name}'...")
        
        # Check if collection already exists
        if collection_name in db.list_collection_names():
            logger.info(f"⚠️  Collection '{collection_name}' already exists. Skipping creation.")
            collection = db[collection_name]
        else:
            # Create collection with schema validation
            db.create_collection(
                collection_name,
                validator={
                    '$jsonSchema': {
                        'bsonType': 'object',
                        'required': ['_id', 'original_patch', 'language', 'source_dataset'],
                        'properties': {
                            '_id': {'bsonType': 'int'},
                            'original_file': {'bsonType': ['string', 'null']},
                            'original_patch': {'bsonType': 'string'},
                            'refined_patch': {'bsonType': ['string', 'null']},
                            'review_comment': {'bsonType': 'string'},
                            'language': {'bsonType': 'string'},
                            'quality_label': {'bsonType': ['int', 'null']},
                            'source_dataset': {'bsonType': 'string'}
                        }
                    }
                }
            )
            collection = db[collection_name]
            logger.info(f"✅ Collection '{collection_name}' created with schema validation")
        
        # Step 4: Create indexes for fast queries
        logger.info("\n[4/4] Creating indexes...")
        
        # Note: _id index is automatically created and unique by MongoDB
        # No need to create it manually
        logger.info("✅ Index on '_id' exists (auto-created by MongoDB)")
        
        # Index on language for filtered retrieval
        collection.create_index([('language', ASCENDING)])
        logger.info("✅ Index on 'language' created")
        
        # Index on source_dataset for filtered retrieval
        collection.create_index([('source_dataset', ASCENDING)])
        logger.info("✅ Index on 'source_dataset' created")
        
        # Compound index for complex queries
        collection.create_index([('language', ASCENDING), ('source_dataset', ASCENDING)])
        logger.info("✅ Compound index on 'language + source_dataset' created")
        
        # Step 5: Verify setup
        logger.info("\n" + "=" * 60)
        logger.info("MongoDB Setup Complete!")
        logger.info("=" * 60)
        logger.info(f"Database: {db_name}")
        logger.info(f"Collection: {collection_name}")
        logger.info(f"Indexes: {len(list(collection.list_indexes()))}")
        logger.info(f"Document count: {collection.count_documents({})}")
        logger.info("=" * 60)
        
        # Test insert and delete
        logger.info("\n🧪 Testing insert/delete...")
        test_doc = {
            '_id': -1,
            'original_file': None,
            'original_patch': 'def test(): pass',
            'refined_patch': None,
            'review_comment': 'Test comment',
            'language': 'python',
            'quality_label': None,
            'source_dataset': 'test'
        }
        collection.insert_one(test_doc)
        collection.delete_one({'_id': -1})
        logger.info("✅ Insert/delete test passed")
        
        client.close()
        logger.info("\n✨ Setup successful! You can now run build_indexes.py")
        
        return True
        
    except ConnectionFailure as e:
        logger.error("\n❌ MongoDB connection failed!")
        logger.error("Make sure MongoDB is running:")
        logger.error("  brew services start mongodb-community@7.0")
        logger.error(f"Error: {e}")
        return False
    
    except Exception as e:
        logger.error(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # Check if .env file exists
    env_path = Path(__file__).parent.parent.parent / '.env'
    if not env_path.exists():
        logger.error("\n❌ .env file not found!")
        logger.error(f"Expected location: {env_path}")
        logger.error("\nCreate .env file with:")
        logger.error("  MONGODB_URI=mongodb://localhost:27017/")
        logger.error("  MONGODB_DATABASE=ai_code_reviewer")
        logger.error("  MONGODB_COLLECTION=code_metadata")
        sys.exit(1)
    
    # Check if MongoDB is running first
    if not check_mongodb_status():
        logger.error("\nPlease start MongoDB first:")
        logger.error("  brew services start mongodb-community@7.0")
        sys.exit(1)
    
    # Run setup
    success = setup_mongodb()
    sys.exit(0 if success else 1)
