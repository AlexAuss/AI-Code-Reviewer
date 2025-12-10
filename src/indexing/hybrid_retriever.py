"""
Hybrid retriever combining dense (FAISS) and sparse (BM25) search.
Implements reciprocal rank fusion for result merging.
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
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
        use_codebert_tokenizer: bool = False
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            index_dir: Directory containing indexes and metadata
            embedding_model: Sentence transformer model name
            dense_weight: Weight for dense retrieval scores (0-1)
            sparse_weight: Weight for sparse retrieval scores (0-1)
        """
        self.index_dir = Path(index_dir)
        self.embedding_model_name = embedding_model
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.use_codebert_tokenizer = use_codebert_tokenizer
        
        # To be loaded
        self.dense_model = None
        self.dense_index = None
        self.sparse_index = None
        self.tokenized_corpus = None
        self.db_manager = None  # MongoDB manager for metadata
        self._cb_tokenizer = None
        
    def load_indexes(self):
        """Load all indexes and metadata."""
        logger.info("Loading indexes...")
        
        # Load dense embedding model
        model_name = self.embedding_model_name
        logger.info(f"Loading embedding model: {model_name}")
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
        
        # Load FAISS index
        dense_path = self.index_dir / "dense_faiss.index"
        if not dense_path.exists():
            raise FileNotFoundError(f"Dense index not found: {dense_path}")
        self.dense_index = faiss.read_index(str(dense_path))
        logger.info(f"Loaded FAISS index with {self.dense_index.ntotal} vectors")
        
        # Load BM25 index
        sparse_path = self.index_dir / "sparse_bm25.pkl"
        if not sparse_path.exists():
            raise FileNotFoundError(f"Sparse index not found: {sparse_path}")
        with open(sparse_path, 'rb') as f:
            sparse_data = pickle.load(f)
            self.sparse_index = sparse_data['bm25']
            self.tokenized_corpus = sparse_data['corpus']
        logger.info(f"Loaded BM25 index with {len(self.tokenized_corpus)} documents")
        
        # Connect to MongoDB for metadata
        logger.info("Connecting to MongoDB for metadata...")
        self.db_manager = MongoDBManager()
        self.db_manager.connect()
        metadata_count = self.db_manager.count()
        logger.info(f"Connected to MongoDB with {metadata_count} metadata records")
        
    def create_query_text(self, original_code: str, changed_code: str) -> str:
        """
        Create query text from user's code diff.
        
        Args:
            original_code: The old/before code
            changed_code: The new/after code
            
        Returns:
            Formatted query text for embedding
        """
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
        # Embed query
        query_embedding = self.dense_model.encode(
            [query_text],
            convert_to_numpy=True
        ).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.dense_index.search(query_embedding, top_k)
        
        # Return (doc_id, score) tuples
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx != -1:  # Valid result
                results.append((int(idx), float(score)))
        
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
        # Tokenize query
        query_tokens = self.tokenize(query_text)
        
        # Get BM25 scores
        scores = self.sparse_index.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return (doc_id, score) tuples
        results = []
        for idx in top_indices:
            results.append((int(idx), float(scores[idx])))
        
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
            
        Returns:
            List of retrieved records with metadata and scores
        """
        if self.dense_index is None:
            self.load_indexes()
        
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
        doc_ids = [doc_id for doc_id, _ in final_results]
        metadata_records = self.db_manager.get_by_ids(doc_ids)
        
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
        
        logger.info(f"Retrieved {len(retrieved_records)} records")
        return retrieved_records


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


if __name__ == '__main__':
    main()
