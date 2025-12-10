"""MongoDB configuration and connection manager."""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, BulkWriteError
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)


class MongoDBConfig:
    """MongoDB connection configuration."""
    
    def __init__(self):
        # Read from environment variables
        self.uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        self.database = os.getenv('MONGODB_DATABASE', 'ai_code_reviewer')
        self.collection = os.getenv('MONGODB_COLLECTION', 'code_metadata')
        self.max_pool_size = int(os.getenv('MONGODB_MAX_POOL_SIZE', '50'))
        self.min_pool_size = int(os.getenv('MONGODB_MIN_POOL_SIZE', '10'))
        
        logger.info(f"MongoDB Config: {self.uri} -> {self.database}.{self.collection}")


class MongoDBManager:
    """Manage MongoDB connections and operations."""
    
    def __init__(self, config: Optional[MongoDBConfig] = None):
        self.config = config or MongoDBConfig()
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        self._collection: Optional[Collection] = None
    
    def connect(self) -> Collection:
        """Establish MongoDB connection and return collection."""
        if self._collection is not None:
            return self._collection
        
        try:
            logger.info("Connecting to MongoDB...")
            self._client = MongoClient(
                self.config.uri,
                maxPoolSize=self.config.max_pool_size,
                minPoolSize=self.config.min_pool_size,
                serverSelectionTimeoutMS=5000
            )
            
            # Test connection
            self._client.admin.command('ping')
            logger.info("MongoDB connection successful!")
            
            self._db = self._client[self.config.database]
            self._collection = self._db[self.config.collection]
            
            return self._collection
            
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            logger.error("Make sure MongoDB is running: brew services start mongodb-community@7.0")
            raise
        except Exception as e:
            logger.error(f"MongoDB connection error: {e}")
            raise
    
    def close(self):
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")
            self._client = None
            self._db = None
            self._collection = None
    
    def get_collection(self) -> Collection:
        """Get collection (connect if needed)."""
        if self._collection is None:
            return self.connect()
        return self._collection
    
    def insert_batch(self, documents: List[Dict[str, Any]], ordered: bool = False, upsert: bool = False) -> int:
        """
        Insert batch of documents into MongoDB.
        
        Args:
            documents: List of documents to insert
            ordered: If False, continues on error (faster for bulk inserts)
            upsert: If True, use upsert mode (insert or update) to handle duplicates
            
        Returns:
            Number of documents inserted/updated
        """
        if not documents:
            return 0
        
        collection = self.get_collection()
        
        if upsert:
            # Use bulk upsert to handle duplicates - ensures all _ids exist
            from pymongo import UpdateOne
            operations = [
                UpdateOne(
                    {'_id': doc['_id']},
                    {'$set': doc},
                    upsert=True
                )
                for doc in documents
            ]
            try:
                result = collection.bulk_write(operations, ordered=ordered)
                return result.upserted_count + result.modified_count
            except BulkWriteError as e:
                write_errors = e.details.get('writeErrors', [])
                logger.warning(f"Bulk upsert errors: {len(write_errors)} documents failed")
                if write_errors:
                    logger.warning(f"First error detail: {write_errors[0]}")
                return e.details.get('nUpserted', 0) + e.details.get('nModified', 0)
        else:
            # Original insert_many behavior
            try:
                result = collection.insert_many(documents, ordered=ordered)
                return len(result.inserted_ids)
            except BulkWriteError as e:
                # Log errors but continue
                logger.warning(f"Bulk write errors: {len(e.details.get('writeErrors', []))} documents failed")
                # Return number of successful insertions
                return e.details.get('nInserted', 0)
            except Exception as e:
                logger.error(f"Error inserting batch: {e}")
                raise
    
    def get_by_id(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Get single document by _id."""
        collection = self.get_collection()
        return collection.find_one({'_id': doc_id})
    
    def get_by_ids(self, doc_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Get multiple documents by _ids (batch query).
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            List of documents in same order as doc_ids
        """
        collection = self.get_collection()
        
        # Query all documents at once
        cursor = collection.find({'_id': {'$in': doc_ids}})
        
        # Create mapping from _id to document
        doc_map = {doc['_id']: doc for doc in cursor}
        
        # Return documents in same order as doc_ids
        results = []
        for doc_id in doc_ids:
            if doc_id in doc_map:
                results.append(doc_map[doc_id])
            else:
                logger.warning(f"Document with _id={doc_id} not found")
        
        return results
    
    def count(self) -> int:
        """Count total documents in collection."""
        collection = self.get_collection()
        return collection.count_documents({})
    
    def clear_collection(self):
        """Delete all documents from collection (use with caution!)."""
        collection = self.get_collection()
        result = collection.delete_many({})
        logger.info(f"Deleted {result.deleted_count} documents from collection")
        return result.deleted_count
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
