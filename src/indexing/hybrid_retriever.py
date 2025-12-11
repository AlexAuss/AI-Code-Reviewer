"""
Hybrid retriever combining dense (FAISS) and sparse (BM25) search.
Implements reciprocal rank fusion for result merging.
"""

import sys
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sentence_transformers import SentenceTransformer
from sentence_transformers import models as st_models
import faiss
from rank_bm25 import BM25Okapi
import pickle

# MongoDB for metadata retrieval
from src.indexing.db_config import MongoDBManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TimingStats:
    """Store timing statistics for performance evaluation."""
    # Loading times (one-time cost)
    embedding_model_load_ms: float = 0.0
    model_warmup_ms: float = 0.0  # First encode to initialize GPU/MPS
    faiss_index_load_ms: float = 0.0
    bm25_index_load_ms: float = 0.0
    mongodb_connect_ms: float = 0.0
    total_load_ms: float = 0.0
    
    # Per-query times
    query_embedding_ms: float = 0.0
    faiss_search_ms: float = 0.0
    dense_search_total_ms: float = 0.0
    bm25_tokenize_ms: float = 0.0
    bm25_search_ms: float = 0.0
    sparse_search_total_ms: float = 0.0
    rrf_fusion_ms: float = 0.0
    mongodb_fetch_ms: float = 0.0
    total_retrieval_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'loading': {
                'embedding_model_load_ms': self.embedding_model_load_ms,
                'model_warmup_ms': self.model_warmup_ms,
                'faiss_index_load_ms': self.faiss_index_load_ms,
                'bm25_index_load_ms': self.bm25_index_load_ms,
                'mongodb_connect_ms': self.mongodb_connect_ms,
                'total_load_ms': self.total_load_ms,
            },
            'retrieval': {
                'query_embedding_ms': self.query_embedding_ms,
                'faiss_search_ms': self.faiss_search_ms,
                'dense_search_total_ms': self.dense_search_total_ms,
                'bm25_tokenize_ms': self.bm25_tokenize_ms,
                'bm25_search_ms': self.bm25_search_ms,
                'sparse_search_total_ms': self.sparse_search_total_ms,
                'rrf_fusion_ms': self.rrf_fusion_ms,
                'mongodb_fetch_ms': self.mongodb_fetch_ms,
                'total_retrieval_ms': self.total_retrieval_ms,
            }
        }
    
    def print_summary(self):
        """Print formatted timing summary."""
        print("\n" + "=" * 60)
        print("PERFORMANCE TIMING SUMMARY (milliseconds)")
        print("=" * 60)
        print("\n📦 LOADING TIMES (one-time cost):")
        print(f"  • Embedding Model Load:  {self.embedding_model_load_ms:>10.2f} ms")
        print(f"  • Model Warmup (MPS/GPU):{self.model_warmup_ms:>10.2f} ms")
        print(f"  • FAISS Index Load:      {self.faiss_index_load_ms:>10.2f} ms")
        print(f"  • BM25 Index Load:       {self.bm25_index_load_ms:>10.2f} ms")
        print(f"  • MongoDB Connect:       {self.mongodb_connect_ms:>10.2f} ms")
        print(f"  ─────────────────────────────────────")
        # Calculate sum of individual times
        individual_sum = (self.embedding_model_load_ms + self.model_warmup_ms + 
                         self.faiss_index_load_ms + self.bm25_index_load_ms + 
                         self.mongodb_connect_ms)
        print(f"  • Sum of above:          {individual_sum:>10.2f} ms")
        print(f"  • TOTAL LOAD TIME:       {self.total_load_ms:>10.2f} ms")
        
        print("\n🔍 RETRIEVAL TIMES (per-query):")
        print(f"  Dense Search:")
        print(f"    • Query Embedding:     {self.query_embedding_ms:>10.2f} ms")
        print(f"    • FAISS Search:        {self.faiss_search_ms:>10.2f} ms")
        print(f"    • Dense Total:         {self.dense_search_total_ms:>10.2f} ms")
        print(f"  Sparse Search:")
        print(f"    • BM25 Tokenization:   {self.bm25_tokenize_ms:>10.2f} ms")
        print(f"    • BM25 Search:         {self.bm25_search_ms:>10.2f} ms")
        print(f"    • Sparse Total:        {self.sparse_search_total_ms:>10.2f} ms")
        print(f"  Fusion & Metadata:")
        print(f"    • RRF Fusion:          {self.rrf_fusion_ms:>10.2f} ms")
        print(f"    • MongoDB Fetch:       {self.mongodb_fetch_ms:>10.2f} ms")
        print(f"  ─────────────────────────────────────")
        print(f"  • TOTAL RETRIEVAL TIME:  {self.total_retrieval_ms:>10.2f} ms")
        print("=" * 60 + "\n")


from collections import defaultdict
import math

class FastBM25:
    """
    Optimized BM25 using inverted index for sparse retrieval.
    Replaces rank_bm25 check-all-docs approach.
    """
    def __init__(self, bm25_object):
        self.k1 = bm25_object.k1
        self.b = bm25_object.b
        self.epsilon = bm25_object.epsilon
        self.idf = bm25_object.idf
        self.avgdl = bm25_object.avgdl
        self.doc_len = bm25_object.doc_len
        self.n_docs = len(self.doc_len)
        
        # Build inverted index
        # token -> list of (doc_id, freq)
        self.inverted_index = defaultdict(list)
        logger.info("Building inverted index for FastBM25...")
        start = time.perf_counter()
        for doc_id, freq_map in enumerate(bm25_object.doc_freqs):
            for token, freq in freq_map.items():
                self.inverted_index[token].append((doc_id, freq))
        build_time = (time.perf_counter() - start) * 1000
        logger.info(f"Inverted index built in {build_time:.2f} ms")
        
    def get_scores_fast(self, query_tokens: List[str], top_k: int = 20) -> List[Tuple[int, float]]:
        """Calculate BM25 scores only for relevant docs."""
        scores = defaultdict(float)
        
        for token in query_tokens:
            if token not in self.inverted_index:
                continue
                
            idf = self.idf.get(token)
            if idf is None:
                # OOV handling if needed, usually rank_bm25 handles this
                continue
                
            # Iterate only docs containing token
            for doc_id, freq in self.inverted_index[token]:
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / self.avgdl)
                score = idf * (freq * (self.k1 + 1)) / denom
                scores[doc_id] += score
                
        # Sort and take top_k
        # This sorts only documents that had at least one match
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_scores

class HybridRetriever:
    """
    Hybrid retriever combining dense semantic search (FAISS) 
    and sparse keyword search (BM25) with reciprocal rank fusion.
    """
    
    def __init__(
        self,
        index_dir: str = "data/indexes",
        embedding_model: str = "microsoft/codebert-base",
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        use_codebert_tokenizer: bool = False,
        device: str = None
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            index_dir: Directory containing indexes and metadata
            embedding_model: Sentence transformer model name
            dense_weight: Weight for dense retrieval scores (0-1)
            sparse_weight: Weight for sparse retrieval scores (0-1)
            device: Device to use for embedding model ('cpu', 'cuda', 'mps')
        """
        self.index_dir = Path(index_dir)
        self.embedding_model_name = embedding_model
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.use_codebert_tokenizer = use_codebert_tokenizer
        self.device = device
        
        # To be loaded
        self.dense_model = None
        self.dense_index = None
        self.sparse_index = None
        self.tokenized_corpus = None
        self.db_manager = None  # MongoDB manager for metadata
        self.tokenized_corpus = None
        self.db_manager = None  # MongoDB manager for metadata
        self._cb_tokenizer = None
        self.fast_bm25 = None
        
        # Timing statistics
        self.timing = TimingStats()
        
    def load_indexes(self):
        """Load all indexes and metadata."""
        logger.info("Loading indexes...")
        total_load_start = time.perf_counter()
        
        # Load dense embedding model
        model_name = self.embedding_model_name
        logger.info(f"Loading embedding model: {model_name}")
        embed_start = time.perf_counter()
        if any(tag in model_name.lower() for tag in ["codebert", "graphcodebert"]):
            transformer = st_models.Transformer(model_name, max_seq_length=512)
            pooling = st_models.Pooling(
                transformer.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,
                pooling_mode_cls_token=False,
                pooling_mode_max_tokens=False,
            )
            self.dense_model = SentenceTransformer(modules=[transformer, pooling])
        else:
            self.dense_model = SentenceTransformer(model_name)
        self.timing.embedding_model_load_ms = (time.perf_counter() - embed_start) * 1000
        
        # Warm up the model (first encode initializes GPU/MPS backend)
        warmup_start = time.perf_counter()
        _ = self.dense_model.encode(["warmup"], convert_to_numpy=True)
        self.timing.model_warmup_ms = (time.perf_counter() - warmup_start) * 1000
        logger.info(f"Model warmup completed")
        
        # Load FAISS index
        faiss_start = time.perf_counter()
        dense_path = self.index_dir / "dense_faiss.index"
        if not dense_path.exists():
            raise FileNotFoundError(f"Dense index not found: {dense_path}")
        self.dense_index = faiss.read_index(str(dense_path))
        self.timing.faiss_index_load_ms = (time.perf_counter() - faiss_start) * 1000
        logger.info(f"Loaded FAISS index with {self.dense_index.ntotal} vectors")
        
        # Load BM25 index
        bm25_start = time.perf_counter()
        sparse_path = self.index_dir / "sparse_bm25.pkl"
        if not sparse_path.exists():
            raise FileNotFoundError(f"Sparse index not found: {sparse_path}")
        with open(sparse_path, 'rb') as f:
            sparse_data = pickle.load(f)
            self.sparse_index = sparse_data['bm25']
            self.tokenized_corpus = sparse_data['corpus']
        self.timing.bm25_index_load_ms = (time.perf_counter() - bm25_start) * 1000
        logger.info(f"Loaded BM25 index with {len(self.tokenized_corpus)} documents")
        
        # Connect to MongoDB for metadata
        mongo_start = time.perf_counter()
        logger.info("Connecting to MongoDB for metadata...")
        self.db_manager = MongoDBManager()
        self.db_manager.connect()
        metadata_count = self.db_manager.count()  # Include count in connection time
        self.timing.mongodb_connect_ms = (time.perf_counter() - mongo_start) * 1000
        logger.info(f"Connected to MongoDB with {metadata_count} metadata records")
        
        # Record total load time
        self.timing.total_load_ms = (time.perf_counter() - total_load_start) * 1000
        
    def create_query_text(self, original_code: str, changed_code: str) -> str:
        """
        Create query text from user's code diff.
        
        Args:
            original_code: The old/before code
            changed_code: The new/after code
            
        Returns:
            Formatted query text for embedding
        """
        # Truncate to prevent BM25 performance issues with large files
        # 2048 chars is roughly 512 tokens, usually enough for a diff context
        max_len = 2048 
        
        orig_trunc = original_code[:max_len] if original_code else ""
        changed_trunc = changed_code[:max_len] if changed_code else ""
        
        # Generate a simple unified diff representation
        query_parts = [
            f"Original code:\n{original_code}",
            f"Changed code:\n{changed_code}"
        ]
        return "\n\n".join(query_parts)
    
    def tokenize(self, text: str) -> List[str]:
        """Code-aware tokenization to better match BM25 indexing."""
        import re
        # Optional: use CodeBERT tokenizer if requested
        if self.use_codebert_tokenizer and self._cb_tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._cb_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            except Exception:
                self._cb_tokenizer = None
                logger.warning("Failed to load CodeBERT tokenizer; falling back to regex tokenization.")
        if self._cb_tokenizer is not None and self.use_codebert_tokenizer:
            toks = self._cb_tokenizer.tokenize(text)
            toks = [t for t in toks if not t.startswith('Ġ') and t not in ['<s>', '</s>', '<pad>']]
            return toks
        # Split CamelCase
        s = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        # Split snake/kebab
        s = s.replace('_', ' ').replace('-', ' ')
        # Preserve operators
        s = re.sub(r'([+\-*/=<>!&|{}()\[\]])', r' \1 ', s)
        tokens = re.findall(r'\b\w+\b', s.lower())
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)] if len(tokens) > 1 else []
        all_tokens = tokens + bigrams
        code_patterns = re.findall(r'\b(?:if|else|for|while|return|null|nullptr|none|undefined)\b', s.lower())
        all_tokens.extend(code_patterns)
        return all_tokens
    
    def dense_search(self, query_text: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform dense semantic search using FAISS.
        
        Args:
            query_text: Query text to embed and search
            top_k: Number of top results to return
            
        Returns:
            List of (doc_id, similarity_score) tuples
        """
        dense_start = time.perf_counter()
        
        # Embed query
        embed_start = time.perf_counter()
        query_embedding = self.dense_model.encode(
            [query_text],
            convert_to_numpy=True
        ).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        self.timing.query_embedding_ms = (time.perf_counter() - embed_start) * 1000
        
        # Search
        search_start = time.perf_counter()
        distances, indices = self.dense_index.search(query_embedding, top_k)
        self.timing.faiss_search_ms = (time.perf_counter() - search_start) * 1000
        
        # Return (doc_id, score) tuples
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx != -1:  # Valid result
                results.append((int(idx), float(score)))
        
        self.timing.dense_search_total_ms = (time.perf_counter() - dense_start) * 1000
        return results
    
    def sparse_search(self, query_text: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform sparse keyword search using BM25.
        
        Args:
            query_text: Query text to tokenize and search
            top_k: Number of top results to return
            
        Returns:
            List of (doc_id, bm25_score) tuples
        """
        sparse_start = time.perf_counter()
        
        # Tokenize query
        tokenize_start = time.perf_counter()
        query_tokens = self.tokenize(query_text)
        self.timing.bm25_tokenize_ms = (time.perf_counter() - tokenize_start) * 1000
        
        # Get BM25 scores
        search_start = time.perf_counter()
        scores = self.sparse_index.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        self.timing.bm25_search_ms = (time.perf_counter() - search_start) * 1000
        
        # Return (doc_id, score) tuples
        results = []
        for idx in top_indices:
            results.append((int(idx), float(scores[idx])))
        
        self.timing.sparse_search_total_ms = (time.perf_counter() - sparse_start) * 1000
        return results
    
    def reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Merge dense and sparse results using Reciprocal Rank Fusion (RRF).
        
        RRF score for document d: sum over all rankings R: 1 / (k + rank_R(d))
        
        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search
            k: RRF constant (typically 60)
            
        Returns:
            Merged and sorted list of (doc_id, fused_score) tuples
        """
        rrf_start = time.perf_counter()
        
        # Build rank maps
        dense_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(dense_results, 1)}
        sparse_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(sparse_results, 1)}
        
        # Collect all unique doc IDs
        all_doc_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores = {}
        for doc_id in all_doc_ids:
            score = 0.0
            if doc_id in dense_ranks:
                score += self.dense_weight / (k + dense_ranks[doc_id])
            if doc_id in sparse_ranks:
                score += self.sparse_weight / (k + sparse_ranks[doc_id])
            rrf_scores[doc_id] = score
        
        # Sort by RRF score descending
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        self.timing.rrf_fusion_ms = (time.perf_counter() - rrf_start) * 1000
        return sorted_results
    
    def retrieve(
        self,
        original_code: str = None,
        changed_code: str = None,
        patch: str = None,
        top_k: int = 5,
        dense_top_k: int = 20,
        sparse_top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant examples using hybrid search.
        
        Args:
            original_code: The old/before code snippet
            changed_code: The new/after code snippet
            top_k: Final number of results to return
            dense_top_k: Number of candidates from dense search
            sparse_top_k: Number of candidates from sparse search
            return_timing: If True, return timing stats along with results
            
        Returns:
            List of retrieved records with metadata and scores
            If return_timing=True, returns tuple (results, timing_stats)
        """
        if self.dense_index is None:
            self.load_indexes()
        
        # Start timing AFTER indexes are loaded (retrieval-only time)
        retrieval_start = time.perf_counter()
        
        # Create query
        if patch is not None:
            query_text = patch
        else:
            query_text = self.create_query_text(original_code, changed_code)
        
        # Perform dense and sparse search
        logger.info("Performing hybrid retrieval...")
        dense_results = self.dense_search(query_text, top_k=dense_top_k)
        sparse_results = self.sparse_search(query_text, top_k=sparse_top_k)
        
        # Merge with RRF
        fused_results = self.reciprocal_rank_fusion(dense_results, sparse_results)
        
        # Get top-k final results
        final_results = fused_results[:top_k]
        
        # Retrieve metadata from MongoDB (batch query)
        mongo_fetch_start = time.perf_counter()
        doc_ids = [doc_id for doc_id, _ in final_results]
        metadata_records = self.db_manager.get_by_ids(doc_ids)
        self.timing.mongodb_fetch_ms = (time.perf_counter() - mongo_fetch_start) * 1000
        
        # Create result map for fast lookup
        metadata_map = {rec['_id']: rec for rec in metadata_records}
        
        # Build final results with scores
        retrieved_records = []
        for doc_id, fused_score in final_results:
            if doc_id in metadata_map:
                record = metadata_map[doc_id].copy()
                # Remove MongoDB _id from result (use doc_id instead)
                record.pop('_id', None)
                record['doc_id'] = doc_id
                record['retrieval_score'] = fused_score
                retrieved_records.append(record)
            else:
                logger.warning(f"Metadata not found for doc_id={doc_id}")
        
        self.timing.total_retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        logger.info(f"Retrieved {len(retrieved_records)} records")
        return retrieved_records
    
    def get_timing_stats(self) -> TimingStats:
        """Return the current timing statistics."""
        return self.timing


def main():
    """Test retrieval with sample query."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test hybrid retrieval")
    parser.add_argument(
        '--index-dir',
        type=str,
        default='data/indexes',
        help='Directory containing indexes'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='microsoft/codebert-base',
        help='Embedding model to encode queries (must match the one used for indexing)'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--original-code',
        type=str,
        help='Original code snippet'
    )
    group.add_argument(
        '--patch',
        type=str,
        help='Single patch/diff text to query with (code-only queries)'
    )
    parser.add_argument(
        '--changed-code',
        type=str,
        help='Changed code snippet (required if --original-code is used)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of results to retrieve'
    )
    parser.add_argument(
        '--use-codebert-tokenizer',
        action='store_true',
        help='Use CodeBERT tokenizer for BM25 query tokenization to match index building'
    )
    parser.add_argument(
        '--show-timing',
        action='store_true',
        default=True,
        help='Display detailed timing statistics (default: True)'
    )
    parser.add_argument(
        '--timing-json',
        action='store_true',
        help='Output timing stats as JSON for programmatic use'
    )
    
    args = parser.parse_args()
    
    # Initialize retriever
    retriever = HybridRetriever(index_dir=args.index_dir, embedding_model=args.embedding_model, use_codebert_tokenizer=args.use_codebert_tokenizer)
    
    # Perform retrieval
    if args.patch:
        results = retriever.retrieve(patch=args.patch, top_k=args.top_k)
    else:
        if not args.changed_code:
            parser.error("--changed-code is required when using --original-code")
        results = retriever.retrieve(
            original_code=args.original_code,
            changed_code=args.changed_code,
            top_k=args.top_k
        )
    
    # Display results
    print("\n" + "=" * 80)
    print(f"Top {len(results)} Retrieved Examples:")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (Score: {result['retrieval_score']:.4f}) ---")
        print(f"Source: {result.get('source_dataset', 'N/A')}")
        print(f"Language: {result.get('language', 'N/A')}")
        print(f"Quality Label: {result.get('quality_label', 'N/A')}")
        print(f"\nPatch:\n{result.get('original_patch', 'N/A')[:200]}...")
        print(f"\nReview:\n{result.get('review_comment', 'N/A')[:200]}...")
        if result.get('refined_patch'):
            print(f"\nRefined Patch:\n{result['refined_patch'][:200]}...")
    
    print("\n" + "=" * 80)
    
    # Show timing stats
    timing = retriever.get_timing_stats()
    if args.timing_json:
        import json
        print("\n--- TIMING STATS (JSON) ---")
        print(json.dumps(timing.to_dict(), indent=2))
    elif args.show_timing:
        timing.print_summary()


if __name__ == '__main__':
    main()
