"""
Phase 2: Build dense vector index (FAISS) and sparse keyword index (BM25) 
from unified dataset for hybrid retrieval.

The unified dataset schema:
- original_file: full file content (oldf)
- language: programming language
- original_patch: the diff/patch (old_hunk or patch field)
- refined_patch: the refined hunk (hunk field from Code_Refinement, None for Comment_Generation)
- review_comment: human review/comment (comment or msg)
- quality_label: quality label (y from Comment_Generation, None for Code_Refinement)
- source_dataset: "code_refinement" or "comment_generation"
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Dense embeddings
from sentence_transformers import SentenceTransformer
from sentence_transformers import models as st_models
import faiss

# Sparse index (BM25)
from rank_bm25 import BM25Okapi
import pickle

# MongoDB for metadata storage
from src.indexing.db_config import MongoDBManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IndexConfig:
    """Configuration for index building."""
    dataset_path: str = "Datasets/Unified_Dataset/train.jsonl"
    index_output_dir: str = "data/indexes"
    
    # Dense index settings
    embedding_model: str = "microsoft/codebert-base"
    dense_index_name: str = "dense_faiss.index"
    
    # Sparse index settings
    sparse_index_name: str = "sparse_bm25.pkl"
    
    # Metadata storage
    metadata_file: str = "metadata.jsonl"
    
    # Processing batch size
    batch_size: int = 32
    
    # Field to embed for dense index
    dense_embedding_fields: List[str] = None
    
    def __post_init__(self):
        if self.dense_embedding_fields is None:
            # Default to code-centric fields since user queries are code-only
            # Include language token to aid cross-language discrimination
            self.dense_embedding_fields = ["language", "original_patch", "review_comment"]


class DenseIndexBuilder:
    """Build FAISS dense vector index from embeddings."""
    
    def __init__(self, config: IndexConfig):
        self.config = config
        self.model = None
        self.dimension = None
        
    def load_model(self):
        """Load embedding model. If CodeBERT/GraphCodeBERT is requested, wrap with pooling."""
        model_name = self.config.embedding_model
        logger.info(f"Loading embedding model: {model_name}")

        # Heuristic: for raw transformer checkpoints (e.g., microsoft/codebert-base),
        # build a SentenceTransformer with mean pooling.
        if any(tag in model_name.lower() for tag in ["codebert", "graphcodebert"]):
            transformer = st_models.Transformer(model_name, max_seq_length=512)
            pooling = st_models.Pooling(
                transformer.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,
                pooling_mode_cls_token=False,
                pooling_mode_max_tokens=False,
            )
            self.model = SentenceTransformer(modules=[transformer, pooling])
        else:
            # Standard Sentence-Transformers models
            self.model = SentenceTransformer(model_name)

        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")
        
    def create_embedding_text(self, record: Dict[str, Any]) -> str:
        """
        Create text to embed by combining fields specified in config.dense_embedding_fields.
        Default is code-centric: language + original_patch [+ refined_patch if present].
        """
        parts = []
        for field in self.config.dense_embedding_fields:
            if not record.get(field):
                continue
            if field == "language":
                parts.append(f"Language: {record['language']}")
            elif field == "original_patch":
                parts.append(f"Patch:\n{record['original_patch']}")
            elif field == "refined_patch":
                parts.append(f"Refined:\n{record['refined_patch']}")
            elif field == "review_comment":
                parts.append(f"Review:\n{record['review_comment']}")
            else:
                parts.append(str(record[field]))
        return "\n\n".join(parts)
    
    def build_index_from_stream(self, dataset_path: Path, output_path: Path) -> faiss.Index:
        """
        Build FAISS index by streaming dataset from disk.
        Memory-efficient for large datasets (300K+ records).
        
        Args:
            dataset_path: Path to unified dataset JSONL file
            output_path: Path to save FAISS index
            
        Returns:
            FAISS index object
        """
        if self.model is None:
            self.load_model()
        
        logger.info(f"Building dense index from {dataset_path} (streaming mode)...")
        
        # Create FAISS index first
        index = faiss.IndexFlatIP(self.dimension)
        
        # Process in chunks to avoid memory issues
        chunk_size = 1000
        total_processed = 0
        
        # Stream records in chunks
        chunk_records = []
        for record in stream_unified_dataset(dataset_path):
            chunk_records.append(record)
            
            # Process when chunk is full
            if len(chunk_records) >= chunk_size:
                self._process_chunk(chunk_records, index)
                total_processed += len(chunk_records)
                logger.info(f"Processed {total_processed} records")
                chunk_records = []  # Clear chunk
        
        # Process remaining records
        if chunk_records:
            self._process_chunk(chunk_records, index)
            total_processed += len(chunk_records)
            logger.info(f"Processed {total_processed} records (final)")
        
        logger.info(f"FAISS index created with {index.ntotal} vectors")
        
        # Save index
        output_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(output_path))
        logger.info(f"Dense index saved to {output_path}")
        
        return index
    
    def _process_chunk(self, chunk_records: List[Dict[str, Any]], index: faiss.Index):
        """Process a chunk of records and add to FAISS index."""
        # Create embedding texts for this chunk
        texts = [self.create_embedding_text(rec) for rec in chunk_records]
        
        # Generate embeddings in batches
        chunk_embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,  # Disable for cleaner logs
                batch_size=self.config.batch_size,
                convert_to_numpy=True
            )
            chunk_embeddings.append(batch_embeddings)
        
        # Concatenate chunk embeddings
        embeddings = np.vstack(chunk_embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index (incremental addition)
        index.add(embeddings)
        
        # Free memory
        del embeddings, chunk_embeddings, texts


class SparseIndexBuilder:
    """Build BM25 sparse keyword index."""
    
    def __init__(self, config: IndexConfig, use_codebert_tokenizer: bool = False):
        self.config = config
        self.use_codebert_tokenizer = use_codebert_tokenizer
        self.codebert_tokenizer = None
        
        if use_codebert_tokenizer:
            try:
                from transformers import AutoTokenizer
                logger.info("Loading CodeBERT tokenizer for BM25...")
                self.codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
                logger.info("CodeBERT tokenizer loaded successfully")
            except ImportError:
                logger.warning("transformers not installed. Falling back to basic tokenizer. Install with: pip install transformers")
                self.use_codebert_tokenizer = False
        
    def tokenize(self, text: str) -> List[str]:
        """
        Code-aware tokenization for better code matching.
        
        Supports two modes:
        1. CodeBERT tokenizer (better for code, subword tokenization)
        2. Enhanced regex tokenizer (fast, good enough)
        """
        if self.use_codebert_tokenizer and self.codebert_tokenizer:
            # Use CodeBERT's subword tokenization
            # This handles code much better: "calculateSum" → ["calculate", "Sum"]
            tokens = self.codebert_tokenizer.tokenize(text)
            # Convert subword tokens to strings (remove special tokens)
            tokens = [t for t in tokens if not t.startswith('Ġ') and t not in ['<s>', '</s>', '<pad>']]
            return tokens
        else:
            # Fall back to enhanced regex tokenizer
            return self._code_aware_tokenize(text)
    
    def _code_aware_tokenize(self, text: str) -> List[str]:
        """
        Enhanced regex-based tokenization for code.
        
        Improvements over simple tokenization:
        1. Split CamelCase: "CamelCase" → ["camel", "case"]
        2. Split snake_case: "snake_case" → ["snake", "case"]
        3. Keep operators as tokens
        4. Generate bigrams (phrase-level tokens)
        """
        import re
        
        # Preserve original for special code patterns
        original_text = text
        
        # Split CamelCase: "CamelCase" → "Camel Case"
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Split snake_case and kebab-case
        text = text.replace('_', ' ').replace('-', ' ')
        
        # Preserve operators as separate tokens
        # Keep common code operators: + - * / = < > ! & | etc.
        text = re.sub(r'([+\-*/=<>!&|{}()\[\]])', r' \1 ', text)
        
        # Basic tokenization (lowercase, word boundaries)
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Add bigrams for phrase-level matching
        # "null check" becomes both ["null", "check", "null_check"]
        bigrams = []
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i+1]}"
            bigrams.append(bigram)
        
        # Combine unigrams + bigrams
        all_tokens = tokens + bigrams
        
        # Add special code patterns from original text
        # Match common code patterns: function calls, null checks, etc.
        code_patterns = re.findall(r'\b(?:if|else|for|while|return|null|nullptr|None|undefined)\b', original_text.lower())
        all_tokens.extend(code_patterns)
        
        return all_tokens
    
    def create_sparse_text(self, record: Dict[str, Any]) -> str:
        """
        Create text for sparse indexing (keywords matter).
        Focus on code tokens and review terms.
        """
        parts = []
        
        if record.get("language"):
            parts.append(record["language"])
        
        if record.get("original_patch"):
            parts.append(record["original_patch"])
        
        if record.get("review_comment"):
            parts.append(record["review_comment"])
        
        if record.get("refined_patch"):
            parts.append(record["refined_patch"])
        
        return " ".join(parts)
    
    def build_index_from_stream(self, dataset_path: Path, output_path: Path) -> BM25Okapi:
        """
        Build BM25 index by streaming dataset from disk.
        Memory-efficient for large datasets (300K+ records).
        
        Args:
            dataset_path: Path to unified dataset JSONL file
            output_path: Path to save BM25 index
            
        Returns:
            BM25Okapi index object
        """
        logger.info(f"Building sparse BM25 index from {dataset_path} (streaming mode)...")
        
        # Process in chunks to avoid memory issues
        chunk_size = 5000  # BM25 tokenization is faster, use larger chunks
        tokenized_corpus = []
        total_processed = 0
        
        # Stream records in chunks
        chunk_records = []
        for record in stream_unified_dataset(dataset_path):
            chunk_records.append(record)
            
            # Process when chunk is full
            if len(chunk_records) >= chunk_size:
                chunk_tokenized = self._tokenize_chunk(chunk_records)
                tokenized_corpus.extend(chunk_tokenized)
                total_processed += len(chunk_records)
                logger.info(f"Tokenized {total_processed} records")
                chunk_records = []  # Clear chunk
        
        # Process remaining records
        if chunk_records:
            chunk_tokenized = self._tokenize_chunk(chunk_records)
            tokenized_corpus.extend(chunk_tokenized)
            total_processed += len(chunk_records)
            logger.info(f"Tokenized {total_processed} records (final)")
        
        # Build BM25 index
        logger.info(f"Building BM25 index from {len(tokenized_corpus)} tokenized documents...")
        bm25_index = BM25Okapi(tokenized_corpus)
        
        logger.info(f"BM25 index created with {len(tokenized_corpus)} documents")
        
        # Save index
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump({
                'bm25': bm25_index,
                'corpus': tokenized_corpus
            }, f)
        
        logger.info(f"Sparse index saved to {output_path}")
        
        return bm25_index
    
    def _tokenize_chunk(self, chunk_records: List[Dict[str, Any]]) -> List[List[str]]:
        """Tokenize a chunk of records."""
        # Create corpus texts for this chunk
        corpus_texts = [self.create_sparse_text(rec) for rec in chunk_records]
        
        # Tokenize corpus
        chunk_tokenized = [self.tokenize(text) for text in corpus_texts]
        
        # Free memory
        del corpus_texts
        
        return chunk_tokenized


class MetadataStore:
    """Store and retrieve record metadata in MongoDB."""
    
    def __init__(self, config: IndexConfig):
        self.config = config
        self.db_manager = MongoDBManager()
        
    def save_metadata_batch(self, metadata_batch: List[Dict[str, Any]], upsert: bool = True) -> int:
        """
        Save batch of metadata to MongoDB.
        
        Args:
            metadata_batch: List of metadata documents to insert
            upsert: If True, use upsert mode to handle duplicates (ensures all _ids exist)
            
        Returns:
            Number of documents inserted/updated
        """
        if not metadata_batch:
            return 0
        
        return self.db_manager.insert_batch(metadata_batch, ordered=False, upsert=upsert)
    
    def clear_metadata(self):
        """Clear all metadata from MongoDB collection."""
        logger.info("Clearing existing metadata from MongoDB...")
        self.db_manager.clear_collection()
    
    def count_metadata(self) -> int:
        """Count total metadata records in MongoDB."""
        return self.db_manager.count()
    
    def close(self):
        """Close MongoDB connection."""
        self.db_manager.close()


def stream_unified_dataset(dataset_path: Path):
    """
    Stream unified dataset from JSONL file without loading all into memory.
    Yields one record at a time.
    
    This is memory-efficient for large datasets (300K+ records).
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Please create the unified dataset first."
        )
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                yield record
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                continue


def build_all_indexes(config: IndexConfig, use_codebert_tokenizer: bool = False, skip_indexes: bool = False):
    """
    Main function to build all indexes (dense + sparse + metadata).
    OPTIMIZED: Streams dataset ONCE and builds all three simultaneously.
    
    Args:
        config: Index configuration
        use_codebert_tokenizer: If True, use CodeBERT tokenizer for BM25 (better for code)
        skip_indexes: If True, only build metadata (useful when indexes already exist)
    """
    dataset_path = Path(config.dataset_path)
    index_dir = Path(config.index_output_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify dataset exists
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}. Exiting.")
        return
    
    logger.info("=" * 60)
    logger.info("Starting index building (single-pass streaming mode)")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Output: {index_dir}")
    logger.info(f"Skip indexes: {skip_indexes}")
    logger.info("=" * 60)
    
    # Initialize builders
    dense_builder = None
    sparse_builder = None
    metadata_store = MetadataStore(config)
    
    if not skip_indexes:
        logger.info("\\n[Initializing] Dense and Sparse index builders...")
        dense_builder = DenseIndexBuilder(config)
        dense_builder.load_model()
        sparse_builder = SparseIndexBuilder(config, use_codebert_tokenizer=use_codebert_tokenizer)
    
    # Clear existing metadata
    logger.info("\\n[Preparing] MongoDB metadata storage...")
    metadata_store.clear_metadata()
    
    # Single-pass processing
    logger.info("\\n[Processing] Streaming dataset (single pass for all indexes + metadata)...")
    
    # Accumulators for dense and sparse
    dense_embeddings_list = [] if not skip_indexes else None
    tokenized_corpus = [] if not skip_indexes else None
    
    # Metadata batch
    metadata_batch = []
    metadata_batch_size = 1000
    
    # Counters
    total_processed = 0
    chunk_size = 1000
    chunk_records = []
    
    # Stream and process
    for record in stream_unified_dataset(dataset_path):
        chunk_records.append(record)
        
        # Process when chunk is full
        if len(chunk_records) >= chunk_size:
            # Process chunk for indexes and metadata
            if not skip_indexes:
                # Dense: create embeddings
                texts = [dense_builder.create_embedding_text(rec) for rec in chunk_records]
                chunk_embeddings = dense_builder.model.encode(
                    texts,
                    show_progress_bar=False,
                    batch_size=config.batch_size,
                    convert_to_numpy=True
                ).astype('float32')
                faiss.normalize_L2(chunk_embeddings)
                dense_embeddings_list.append(chunk_embeddings)
                
                # Sparse: tokenize
                for rec in chunk_records:
                    text = sparse_builder.create_sparse_text(rec)
                    tokens = sparse_builder.tokenize(text)
                    tokenized_corpus.append(tokens)
            
            # Metadata: prepare for MongoDB
            for idx, rec in enumerate(chunk_records):
                doc_id = total_processed + idx
                metadata = {
                    '_id': doc_id,  # Match FAISS index position
                    'original_file': rec.get('original_file'),
                    'original_patch': rec.get('original_patch') or '',
                    'refined_patch': rec.get('refined_patch'),
                    'review_comment': rec.get('review_comment') or '',
                    'language': rec.get('language') or '',
                    'quality_label': rec.get('quality_label'),
                    'source_dataset': rec.get('source_dataset') or '',
                }
                metadata_batch.append(metadata)
            
            # Insert metadata batch to MongoDB
            if len(metadata_batch) >= metadata_batch_size:
                inserted = metadata_store.save_metadata_batch(metadata_batch)
                logger.info(f"Inserted {inserted} metadata records to MongoDB")
                metadata_batch = []
            
            total_processed += len(chunk_records)
            logger.info(f"Processed {total_processed} records")
            chunk_records = []
    
    # Process remaining records
    if chunk_records:
        if not skip_indexes:
            texts = [dense_builder.create_embedding_text(rec) for rec in chunk_records]
            chunk_embeddings = dense_builder.model.encode(
                texts,
                show_progress_bar=False,
                batch_size=config.batch_size,
                convert_to_numpy=True
            ).astype('float32')
            faiss.normalize_L2(chunk_embeddings)
            dense_embeddings_list.append(chunk_embeddings)
            
            for rec in chunk_records:
                text = sparse_builder.create_sparse_text(rec)
                tokens = sparse_builder.tokenize(text)
                tokenized_corpus.append(tokens)
        
        for idx, rec in enumerate(chunk_records):
            doc_id = total_processed + idx
            metadata = {
                '_id': doc_id,
                'original_file': rec.get('original_file'),
                'original_patch': rec.get('original_patch') or '',
                'refined_patch': rec.get('refined_patch'),
                'review_comment': rec.get('review_comment') or '',
                'language': rec.get('language') or '',
                'quality_label': rec.get('quality_label'),
                'source_dataset': rec.get('source_dataset') or '',
            }
            metadata_batch.append(metadata)
        
        total_processed += len(chunk_records)
        logger.info(f"Processed {total_processed} records (final)")
    
    # Insert remaining metadata
    if metadata_batch:
        inserted = metadata_store.save_metadata_batch(metadata_batch)
        logger.info(f"Inserted {inserted} metadata records to MongoDB (final)")
    
    # Build indexes from accumulated data
    if not skip_indexes:
        # Dense index
        logger.info("\\n[Building] Dense FAISS index from embeddings...")
        all_embeddings = np.vstack(dense_embeddings_list)
        dense_index = faiss.IndexFlatIP(dense_builder.dimension)
        dense_index.add(all_embeddings)
        dense_index_path = index_dir / config.dense_index_name
        faiss.write_index(dense_index, str(dense_index_path))
        logger.info(f"Dense index saved: {dense_index_path} ({dense_index.ntotal} vectors)")
        
        # Sparse index
        logger.info("\\n[Building] Sparse BM25 index from tokenized corpus...")
        sparse_index = BM25Okapi(tokenized_corpus)
        sparse_index_path = index_dir / config.sparse_index_name
        with open(sparse_index_path, 'wb') as f:
            pickle.dump({'bm25': sparse_index, 'corpus': tokenized_corpus}, f)
        logger.info(f"Sparse index saved: {sparse_index_path} ({len(tokenized_corpus)} documents)")
    
    # Verify metadata
    total_metadata = metadata_store.count_metadata()
    logger.info(f"\\n[Verified] MongoDB metadata count: {total_metadata}")
    
    # Close MongoDB connection
    metadata_store.close()
    
    logger.info("\\n" + "=" * 60)
    logger.info("Index building complete!")
    logger.info(f"Total records processed: {total_processed}")
    if not skip_indexes:
        logger.info(f"Dense index: {dense_index_path} ({dense_index.ntotal} vectors)")
        logger.info(f"Sparse index: {sparse_index_path}")
    logger.info(f"Metadata (MongoDB): {total_metadata} records")
    logger.info("=" * 60)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build dense and sparse indexes from unified dataset"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='Datasets/Unified_Dataset/train.jsonl',
        help='Path to unified dataset JSONL file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/indexes',
        help='Output directory for indexes'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='microsoft/codebert-base',
        help='Embedding model for dense vectors (e.g., microsoft/codebert-base or sentence-transformers/all-mpnet-base-v2)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding generation'
    )
    parser.add_argument(
        '--use-codebert-tokenizer',
        action='store_true',
        help='Use CodeBERT tokenizer for BM25 (better for code, requires transformers)'
    )
    parser.add_argument(
        '--dense-code-only',
        action='store_true',
        help='Embed only code-centric fields (language, original_patch, refined_patch) and exclude review_comment'
    )
    parser.add_argument(
        '--dense-fields',
        type=str,
        default=None,
        help='Comma-separated list of fields to embed (overrides --dense-code-only). Possible fields: language,original_patch,refined_patch,review_comment'
    )
    parser.add_argument(
        '--skip-indexes',
        action='store_true',
        help='Skip building FAISS and BM25 indexes, only build metadata in MongoDB (useful when indexes already exist)'
    )
    
    args = parser.parse_args()
    
    # Resolve dense fields preference
    dense_fields = None
    if args.dense_fields:
        dense_fields = [f.strip() for f in args.dense_fields.split(',') if f.strip()]
    elif args.dense_code_only:
        dense_fields = ["language", "original_patch", "refined_patch"]

    config = IndexConfig(
        dataset_path=args.dataset,
        index_output_dir=args.output,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        dense_embedding_fields=dense_fields
    )
    
    build_all_indexes(config, use_codebert_tokenizer=args.use_codebert_tokenizer, skip_indexes=args.skip_indexes)


if __name__ == '__main__':
    main()
