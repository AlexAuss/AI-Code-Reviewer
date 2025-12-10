"""
Test script to verify MongoDB setup and metadata operations.

This script tests:
1. MongoDB connection
2. Metadata insertion
3. Metadata retrieval by ID
4. Batch retrieval
5. Document counting

Run this after setup_mongodb.sh to verify everything works.
"""

import sys
import logging
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.indexing.db_config import MongoDBManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_mongodb_operations():
    """Test MongoDB CRUD operations."""
    
    logger.info("=" * 60)
    logger.info("Testing MongoDB Operations")
    logger.info("=" * 60)
    
    # Test 1: Connection
    logger.info("\n[Test 1] Connecting to MongoDB...")
    try:
        db_manager = MongoDBManager()
        db_manager.connect()
        logger.info("✅ Connection successful")
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return False
    
    # Test 2: Clear collection
    logger.info("\n[Test 2] Clearing collection...")
    try:
        deleted = db_manager.clear_collection()
        logger.info(f"✅ Cleared {deleted} documents")
    except Exception as e:
        logger.error(f"❌ Clear failed: {e}")
        return False
    
    # Test 3: Insert single document
    logger.info("\n[Test 3] Inserting single document...")
    try:
        test_docs = [{
            '_id': 0,
            'original_file': None,
            'original_patch': 'def test(): pass',
            'refined_patch': 'def test():\n    return True',
            'review_comment': 'Added return value',
            'language': 'python',
            'quality_label': 1,
            'source_dataset': 'test'
        }]
        inserted = db_manager.insert_batch(test_docs)
        logger.info(f"✅ Inserted {inserted} document")
    except Exception as e:
        logger.error(f"❌ Insert failed: {e}")
        return False
    
    # Test 4: Insert batch
    logger.info("\n[Test 4] Inserting batch of documents...")
    try:
        batch_docs = [
            {
                '_id': i,
                'original_file': None,
                'original_patch': f'def test_{i}(): pass',
                'refined_patch': None,
                'review_comment': f'Test comment {i}',
                'language': 'python',
                'quality_label': i % 2,
                'source_dataset': 'test'
            }
            for i in range(1, 101)  # Insert 100 test documents
        ]
        inserted = db_manager.insert_batch(batch_docs)
        logger.info(f"✅ Inserted {inserted} documents")
    except Exception as e:
        logger.error(f"❌ Batch insert failed: {e}")
        return False
    
    # Test 5: Count documents
    logger.info("\n[Test 5] Counting documents...")
    try:
        count = db_manager.count()
        logger.info(f"✅ Total documents: {count}")
        if count != 101:
            logger.warning(f"⚠️  Expected 101 documents, got {count}")
    except Exception as e:
        logger.error(f"❌ Count failed: {e}")
        return False
    
    # Test 6: Get single document by ID
    logger.info("\n[Test 6] Retrieving single document by ID...")
    try:
        doc = db_manager.get_by_id(5)
        if doc:
            logger.info(f"✅ Retrieved document: _id={doc['_id']}, language={doc['language']}")
            logger.info(f"   Patch: {doc['original_patch'][:50]}...")
        else:
            logger.error("❌ Document not found")
            return False
    except Exception as e:
        logger.error(f"❌ Retrieval failed: {e}")
        return False
    
    # Test 7: Batch retrieval by IDs
    logger.info("\n[Test 7] Batch retrieval by multiple IDs...")
    try:
        doc_ids = [1, 5, 10, 20, 50]
        docs = db_manager.get_by_ids(doc_ids)
        logger.info(f"✅ Retrieved {len(docs)} documents")
        for doc in docs:
            logger.info(f"   - Document _id={doc['_id']}, language={doc['language']}")
        if len(docs) != len(doc_ids):
            logger.warning(f"⚠️  Expected {len(doc_ids)} documents, got {len(docs)}")
    except Exception as e:
        logger.error(f"❌ Batch retrieval failed: {e}")
        return False
    
    # Test 8: Query performance test
    logger.info("\n[Test 8] Performance test (1000 random retrievals)...")
    try:
        import time
        import random
        
        start_time = time.time()
        test_ids = [random.randint(0, 100) for _ in range(1000)]
        
        # Batch retrieval (efficient)
        docs = db_manager.get_by_ids(test_ids)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Retrieved {len(docs)} documents in {elapsed:.3f}s")
        logger.info(f"   Average: {elapsed/len(docs)*1000:.2f}ms per document")
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False
    
    # Test 9: Clean up
    logger.info("\n[Test 9] Cleaning up test data...")
    try:
        deleted = db_manager.clear_collection()
        logger.info(f"✅ Deleted {deleted} test documents")
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return False
    
    # Close connection
    logger.info("\n[Cleanup] Closing MongoDB connection...")
    db_manager.close()
    logger.info("✅ Connection closed")
    
    logger.info("\n" + "=" * 60)
    logger.info("✨ All tests passed!")
    logger.info("=" * 60)
    logger.info("\nMongoDB is ready for use!")
    logger.info("Next step: Run build_indexes.py to populate metadata")
    
    return True


if __name__ == '__main__':
    import sys
    
    success = test_mongodb_operations()
    sys.exit(0 if success else 1)
