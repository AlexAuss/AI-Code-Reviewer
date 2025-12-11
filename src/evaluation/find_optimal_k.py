"""
Find optimal K for hybrid retrieval using validation dataset.

OPTIMIZED VERSION:
- Batch embedding computation for similarity scoring (GPU accelerated)
- Uses the same device (MPS/CUDA) as hybrid retriever
- Efficient progress tracking
- Checkpoint/resume support

Computes Recall@K, Precision@K, and MAP@K metrics for different K values
to determine the best number of retrieved examples.

Usage:
    # Quick test with 100 samples
    python find_optimal_k.py --sample 100
    
    # Full evaluation
    python find_optimal_k.py --full
    
    # Resume from checkpoint
    python find_optimal_k.py --resume
"""

import sys
import json
import time
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Store retrieval metrics for different K values."""
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 10, 12])
    
    # Per-K metrics (aggregated across all queries)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    map_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    
    # For computing averages
    num_queries: int = 0
    total_relevant_found: Dict[int, int] = field(default_factory=dict)
    total_retrieved: Dict[int, int] = field(default_factory=dict)
    sum_ap: Dict[int, float] = field(default_factory=dict)
    sum_reciprocal_rank: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'k_values': self.k_values,
            'recall_at_k': self.recall_at_k,
            'precision_at_k': self.precision_at_k,
            'map_at_k': self.map_at_k,
            'mrr': self.mrr,
            'num_queries': self.num_queries
        }


def compute_cosine_similarity_batch(query_embeddings: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query embeddings and document embeddings.
    Vectorized implementation for efficiency.
    
    Args:
        query_embeddings: (N, D) array of N query embeddings
        doc_embeddings: (M, D) array of M document embeddings
        
    Returns:
        (N, M) similarity matrix
    """
    # Normalize
    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    
    query_normalized = query_embeddings / (query_norms + 1e-8)
    doc_normalized = doc_embeddings / (doc_norms + 1e-8)
    
    # Batch cosine similarity via matrix multiplication
    similarities = np.dot(query_normalized, doc_normalized.T)
    
    return similarities


def compute_average_precision(relevance: List[int], k: int) -> float:
    """
    Compute Average Precision at K.
    
    AP@K = (1/min(R,K)) * sum_{i=1}^{K} (Precision@i * rel_i)
    where R is total relevant docs
    """
    relevance_at_k = relevance[:k]
    
    if sum(relevance_at_k) == 0:
        return 0.0
    
    precision_sum = 0.0
    relevant_count = 0
    
    for i, rel in enumerate(relevance_at_k):
        if rel == 1:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i
    
    if relevant_count == 0:
        return 0.0
    
    return precision_sum / relevant_count


def compute_reciprocal_rank(relevance: List[int]) -> float:
    """Compute reciprocal rank (1/rank of first relevant doc)."""
    for i, rel in enumerate(relevance):
        if rel == 1:
            return 1.0 / (i + 1)
    return 0.0


class OptimalKFinder:
    """Find optimal K using validation dataset with batch optimization."""
    
    def __init__(
        self,
        validation_path: str = "Datasets/Unified_Dataset/valid.jsonl",
        index_dir: str = "data/indexes",
        k_values: List[int] = None,
        similarity_thresholds: List[float] = None,
        checkpoint_path: str = "data/evaluation/k_optimization_checkpoint.pkl",
        embedding_batch_size: int = 64  # Batch size for embedding computation
    ):
        self.validation_path = Path(validation_path)
        self.index_dir = index_dir
        self.k_values = k_values or [1, 3, 5, 7, 10, 12]
        self.similarity_thresholds = similarity_thresholds or [0.5, 0.6, 0.7, 0.8]
        self.checkpoint_path = Path(checkpoint_path)
        self.max_k = max(self.k_values)
        self.embedding_batch_size = embedding_batch_size
        
        # Will be initialized
        self.retriever = None
        self.embedding_model = None
        self.device = None
        
        # Results storage - metrics per threshold
        self.results_by_threshold: Dict[float, RetrievalMetrics] = {}
        
        # For checkpoint/resume
        self.processed_indices: set = set()
        self.all_relevance_data: List[Dict] = []
        
    def load_validation_data(self, sample_size: int = None) -> List[Dict]:
        """Load validation dataset."""
        logger.info(f"Loading validation data from {self.validation_path}")
        
        data = []
        with open(self.validation_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        logger.info(f"Loaded {len(data):,} samples")
        
        if sample_size and sample_size < len(data):
            np.random.seed(42)  # Reproducible
            indices = np.random.choice(len(data), size=sample_size, replace=False)
            data = [data[i] for i in indices]
            logger.info(f"Sampled {len(data):,} samples for evaluation")
        
        return data
    
    def initialize_retriever(self):
        """Initialize hybrid retriever with GPU/MPS acceleration."""
        from src.indexing.hybrid_retriever import HybridRetriever
        
        logger.info("Initializing hybrid retriever...")
        self.retriever = HybridRetriever(
            index_dir=self.index_dir,
            use_ivf_index=True,
            parallel_search=True
        )
        self.retriever.load_indexes()
        
        # Use the same embedding model and device for similarity computation
        self.embedding_model = self.retriever.dense_model
        self.device = self.retriever.device
        
        logger.info(f"Retriever initialized (device: {self.device})")
        logger.info(f"Embedding batch size: {self.embedding_batch_size}")
    
    def save_checkpoint(self):
        """Save progress for resume."""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'processed_indices': self.processed_indices,
            'all_relevance_data': self.all_relevance_data,
            'k_values': self.k_values,
            'similarity_thresholds': self.similarity_thresholds
        }
        
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.debug(f"Checkpoint saved: {len(self.processed_indices)} samples processed")
    
    def load_checkpoint(self) -> bool:
        """Load checkpoint if exists."""
        if not self.checkpoint_path.exists():
            return False
        
        with open(self.checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.processed_indices = checkpoint['processed_indices']
        self.all_relevance_data = checkpoint['all_relevance_data']
        
        logger.info(f"Checkpoint loaded: {len(self.processed_indices)} samples already processed")
        return True
    
    def evaluate_single_query(self, idx: int, sample: Dict) -> Dict:
        """
        Evaluate a single query and return relevance data.
        
        Retrieval is sequential (FAISS/BM25 limitation), but similarity
        computation uses batch embedding with GPU.
        """
        query_patch = sample.get('original_patch', '') or sample.get('patch', '') or ''
        ground_truth_review = sample.get('review_comment', '') or ''
        
        if not query_patch.strip():
            return {
                'idx': idx,
                'num_retrieved': 0,
                'similarities': [],
            }
        
        # Retrieve using hybrid search
        results = self.retriever.retrieve(
            patch=query_patch,
            top_k=self.max_k,
            dense_top_k=self.max_k * 2,
            sparse_top_k=self.max_k * 2
        )
        
        if not results:
            return {
                'idx': idx,
                'num_retrieved': 0,
                'similarities': [],
            }
        
        retrieved_reviews = [r.get('review_comment', '') or '' for r in results]
        
        # Batch encode ground truth + all retrieved reviews together for efficiency
        all_texts = [ground_truth_review] + retrieved_reviews
        
        # Batch embedding with GPU acceleration
        all_embeddings = self.embedding_model.encode(
            all_texts,
            convert_to_numpy=True,
            batch_size=self.embedding_batch_size,
            show_progress_bar=False,
            device=str(self.device) if self.device else None
        )
        
        gt_embedding = all_embeddings[0:1]  # (1, D)
        retrieved_embeddings = all_embeddings[1:]  # (K, D)
        
        # Compute cosine similarities (vectorized)
        similarities = compute_cosine_similarity_batch(gt_embedding, retrieved_embeddings)[0]
        
        return {
            'idx': idx,
            'num_retrieved': len(retrieved_reviews),
            'similarities': similarities.tolist(),
        }
    
    def compute_metrics_for_threshold(self, threshold: float) -> RetrievalMetrics:
        """Compute all metrics for a given similarity threshold."""
        metrics = RetrievalMetrics(k_values=self.k_values)
        
        # Initialize accumulators
        for k in self.k_values:
            metrics.total_relevant_found[k] = 0
            metrics.total_retrieved[k] = 0
            metrics.sum_ap[k] = 0.0
        
        for data in self.all_relevance_data:
            similarities = data['similarities']
            
            if not similarities:
                metrics.num_queries += 1
                continue
            
            # Convert similarities to binary relevance using threshold
            relevance = [1 if s >= threshold else 0 for s in similarities]
            
            # Compute metrics for each K
            for k in self.k_values:
                relevance_at_k = relevance[:k]
                
                # Count relevant in top-K
                relevant_found = sum(relevance_at_k)
                metrics.total_relevant_found[k] += relevant_found
                metrics.total_retrieved[k] += min(k, len(relevance_at_k))
                
                # Average Precision
                ap = compute_average_precision(relevance, k)
                metrics.sum_ap[k] += ap
            
            # Reciprocal Rank (for MRR)
            rr = compute_reciprocal_rank(relevance)
            metrics.sum_reciprocal_rank += rr
            
            metrics.num_queries += 1
        
        # Compute final averages
        if metrics.num_queries > 0:
            for k in self.k_values:
                metrics.precision_at_k[k] = (
                    metrics.total_relevant_found[k] / metrics.total_retrieved[k] 
                    if metrics.total_retrieved[k] > 0 else 0
                )
                metrics.recall_at_k[k] = metrics.total_relevant_found[k] / (metrics.num_queries * k)
                metrics.map_at_k[k] = metrics.sum_ap[k] / metrics.num_queries
            
            metrics.mrr = metrics.sum_reciprocal_rank / metrics.num_queries
        
        return metrics
    
    def run_evaluation(
        self,
        sample_size: int = None,
        resume: bool = False,
        checkpoint_interval: int = 100
    ):
        """
        Run full evaluation pipeline with GPU optimization.
        
        Args:
            sample_size: Number of samples to evaluate (None = all)
            resume: Whether to resume from checkpoint
            checkpoint_interval: Save checkpoint every N samples
        """
        # Load data
        validation_data = self.load_validation_data(sample_size)
        
        # Initialize retriever with GPU
        self.initialize_retriever()
        
        # Load checkpoint if resuming
        if resume:
            self.load_checkpoint()
        
        # Filter out already processed
        remaining_samples = [
            (i, sample) for i, sample in enumerate(validation_data)
            if i not in self.processed_indices
        ]
        
        total = len(validation_data)
        already_done = len(self.processed_indices)
        remaining = len(remaining_samples)
        
        logger.info(f"\nStarting evaluation: {remaining:,} remaining of {total:,} total")
        logger.info(f"Device: {self.device}")
        logger.info(f"K values: {self.k_values}")
        logger.info(f"Similarity thresholds: {self.similarity_thresholds}")
        
        start_time = time.time()
        processed_this_run = 0
        
        # Process queries
        for idx, sample in remaining_samples:
            # Evaluate single query
            result = self.evaluate_single_query(idx, sample)
            
            # Store result
            self.all_relevance_data.append(result)
            self.processed_indices.add(idx)
            processed_this_run += 1
            
            # Progress update
            total_processed = already_done + processed_this_run
            if total_processed % 50 == 0 or total_processed == total:
                elapsed = time.time() - start_time
                rate = processed_this_run / elapsed if elapsed > 0 else 0
                remaining_time = (remaining - processed_this_run) / rate if rate > 0 else 0
                
                logger.info(
                    f"Progress: {total_processed:,}/{total:,} ({total_processed/total*100:.1f}%) | "
                    f"Rate: {rate:.2f}/s | ETA: {remaining_time/60:.1f} min"
                )
            
            # Checkpoint
            if processed_this_run % checkpoint_interval == 0:
                self.save_checkpoint()
        
        # Final checkpoint
        self.save_checkpoint()
        
        # Compute metrics for each threshold
        logger.info("\nComputing metrics for different thresholds...")
        for threshold in self.similarity_thresholds:
            logger.info(f"  Computing metrics for threshold={threshold}")
            self.results_by_threshold[threshold] = self.compute_metrics_for_threshold(threshold)
        
        # Print and save results
        self.print_results()
        self.save_results()
        
        total_time = time.time() - start_time
        logger.info(f"\nTotal evaluation time: {total_time/60:.1f} minutes")
        
        return self.results_by_threshold
    
    def print_results(self):
        """Print formatted results table."""
        print("\n" + "=" * 80)
        print("RETRIEVAL METRICS EVALUATION RESULTS")
        print("=" * 80)
        print(f"Total queries evaluated: {len(self.all_relevance_data):,}")
        
        for threshold in self.similarity_thresholds:
            metrics = self.results_by_threshold[threshold]
            
            print(f"\n{'─' * 80}")
            print(f"SIMILARITY THRESHOLD: {threshold}")
            print(f"{'─' * 80}")
            
            # Header
            print(f"\n{'K':>5} │ {'Precision@K':>12} │ {'Recall@K':>10} │ {'MAP@K':>10} │ {'MRR':>8}")
            print(f"{'─' * 5}─┼─{'─' * 12}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 8}")
            
            for k in self.k_values:
                prec = metrics.precision_at_k.get(k, 0)
                recall = metrics.recall_at_k.get(k, 0)
                map_k = metrics.map_at_k.get(k, 0)
                mrr = metrics.mrr
                
                print(f"{k:>5} │ {prec:>12.4f} │ {recall:>10.4f} │ {map_k:>10.4f} │ {mrr:>8.4f}")
            
            # Find optimal K (best MAP@K)
            best_k = max(self.k_values, key=lambda k: metrics.map_at_k.get(k, 0))
            best_map = metrics.map_at_k.get(best_k, 0)
            print(f"\n✓ Optimal K for threshold {threshold}: K={best_k} (MAP@K={best_map:.4f})")
        
        # Overall recommendation
        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)
        
        # Find best threshold and K combination
        best_threshold = None
        best_k = None
        best_map = 0
        
        for threshold in self.similarity_thresholds:
            metrics = self.results_by_threshold[threshold]
            for k in self.k_values:
                map_k = metrics.map_at_k.get(k, 0)
                if map_k > best_map:
                    best_map = map_k
                    best_threshold = threshold
                    best_k = k
        
        print(f"\nBest configuration:")
        print(f"  • Similarity Threshold: {best_threshold}")
        print(f"  • K value: {best_k}")
        print(f"  • MAP@K: {best_map:.4f}")
        print("=" * 80)
    
    def save_results(self):
        """Save results to JSON file."""
        output_path = Path("data/evaluation/k_optimization_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'k_values': self.k_values,
            'similarity_thresholds': self.similarity_thresholds,
            'num_queries': len(self.all_relevance_data),
            'results_by_threshold': {
                str(t): m.to_dict() for t, m in self.results_by_threshold.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Find optimal K for retrieval")
    
    parser.add_argument('--validation-path', type=str,
                        default='Datasets/Unified_Dataset/valid.jsonl',
                        help='Path to validation dataset')
    parser.add_argument('--index-dir', type=str, default='data/indexes',
                        help='Directory containing indexes')
    parser.add_argument('--sample', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')
    parser.add_argument('--full', action='store_true',
                        help='Run full evaluation (all samples)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--k-values', type=str, default='1,3,5,7,10,12',
                        help='Comma-separated K values to test')
    parser.add_argument('--thresholds', type=str, default='0.5,0.6,0.7,0.8',
                        help='Comma-separated similarity thresholds')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                        help='Save checkpoint every N samples')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for embedding computation')
    
    args = parser.parse_args()
    
    # Parse K values and thresholds
    k_values = [int(k) for k in args.k_values.split(',')]
    thresholds = [float(t) for t in args.thresholds.split(',')]
    
    # Determine sample size
    if args.full:
        sample_size = None
    elif args.sample:
        sample_size = args.sample
    else:
        # Default: 100 samples for quick test
        sample_size = 100
        logger.info("Using default sample size of 100. Use --full for complete evaluation.")
    
    # Run evaluation
    finder = OptimalKFinder(
        validation_path=args.validation_path,
        index_dir=args.index_dir,
        k_values=k_values,
        similarity_thresholds=thresholds,
        embedding_batch_size=args.batch_size
    )
    
    finder.run_evaluation(
        sample_size=sample_size,
        resume=args.resume,
        checkpoint_interval=args.checkpoint_interval
    )


if __name__ == '__main__':
    main()
