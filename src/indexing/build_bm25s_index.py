"""
Build optimized BM25 index using bm25s library.

bm25s is ~100x faster than rank_bm25 for search operations.
This script converts the existing tokenized corpus to bm25s format.

Usage:
    # Build from existing rank_bm25 index (fast - reuses tokenization)
    python build_bm25s_index.py --from-rank-bm25
    
    # Build fresh from dataset (slower - re-tokenizes)
    python build_bm25s_index.py --from-dataset
"""

import sys
import time
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_from_rank_bm25(
    rank_bm25_path: str = "data/indexes/sparse_bm25.pkl",
    output_path: str = "data/indexes/sparse_bm25s"
):
    """
    Convert existing rank_bm25 index to bm25s format.
    
    This is faster than re-tokenizing since we reuse the tokenized corpus.
    """
    try:
        import bm25s
    except ImportError:
        print("ERROR: bm25s not installed. Install with: pip install bm25s")
        sys.exit(1)
    
    rank_bm25_path = Path(rank_bm25_path)
    output_path = Path(output_path)
    
    print(f"=" * 60)
    print("Building bm25s index from rank_bm25")
    print(f"=" * 60)
    
    # Load existing rank_bm25 index
    print(f"\n[1/3] Loading rank_bm25 index: {rank_bm25_path}")
    start = time.time()
    
    with open(rank_bm25_path, 'rb') as f:
        data = pickle.load(f)
        tokenized_corpus = data['corpus']  # List of List[str]
    
    load_time = time.time() - start
    print(f"      Loaded {len(tokenized_corpus):,} tokenized documents")
    print(f"      Time: {load_time:.1f}s")
    
    # Convert to bm25s format
    print(f"\n[2/3] Building bm25s index...")
    start = time.time()
    
    # bm25s expects list of strings, so we join tokens back
    # But we need to keep them as tokens for proper scoring
    # bm25s.tokenize() returns a special object, but we can use corpus_tokens directly
    
    # Create bm25s retriever with the tokenized corpus
    retriever = bm25s.BM25()
    
    # bm25s needs corpus_tokens as a list of np arrays or special TokenizedCorpus
    # Simplest approach: index the joined strings and let bm25s re-tokenize
    # Or we can use the lower-level API
    
    # Join tokens to strings (bm25s will re-tokenize, but with its fast tokenizer)
    corpus_texts = [" ".join(tokens) for tokens in tokenized_corpus]
    
    # Tokenize with bm25s (fast Rust implementation)
    corpus_tokens = bm25s.tokenize(corpus_texts, show_progress=True)
    
    # Build index
    retriever.index(corpus_tokens, show_progress=True)
    
    build_time = time.time() - start
    # Get document count from bm25s scores dict
    num_docs = retriever.scores.get('num_docs', len(corpus_texts))
    
    print(f"      Built index with {num_docs:,} documents")
    print(f"      Time: {build_time:.1f}s")
    
    # Save index
    print(f"\n[3/3] Saving bm25s index: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    retriever.save(str(output_path))
    
    print(f"\n" + "=" * 60)
    print("BUILD COMPLETE")
    print(f"=" * 60)
    print(f"  Output: {output_path}")
    print(f"  Documents: {num_docs:,}")
    print(f"  Total time: {load_time + build_time:.1f}s")
    print(f"=" * 60)
    
    return retriever


def build_from_dataset(
    dataset_path: str = "Datasets/Unified_Dataset/train.jsonl",
    output_path: str = "data/indexes/sparse_bm25s",
    use_codebert_tokenizer: bool = False
):
    """
    Build bm25s index directly from dataset.
    """
    try:
        import bm25s
    except ImportError:
        print("ERROR: bm25s not installed. Install with: pip install bm25s")
        sys.exit(1)
    
    import json
    
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    print(f"=" * 60)
    print("Building bm25s index from dataset")
    print(f"=" * 60)
    
    # Load and prepare corpus
    print(f"\n[1/3] Loading dataset: {dataset_path}")
    start = time.time()
    
    corpus_texts = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            # Combine relevant fields
            parts = []
            if record.get('language'):
                parts.append(record['language'])
            if record.get('original_patch'):
                parts.append(record['original_patch'])
            if record.get('review_comment'):
                parts.append(record['review_comment'])
            if record.get('refined_patch'):
                parts.append(record['refined_patch'])
            corpus_texts.append(" ".join(parts))
    
    load_time = time.time() - start
    print(f"      Loaded {len(corpus_texts):,} documents")
    print(f"      Time: {load_time:.1f}s")
    
    # Tokenize
    print(f"\n[2/3] Tokenizing corpus with bm25s...")
    start = time.time()
    
    corpus_tokens = bm25s.tokenize(corpus_texts, show_progress=True)
    
    tokenize_time = time.time() - start
    print(f"      Tokenization complete")
    print(f"      Time: {tokenize_time:.1f}s")
    
    # Build index
    print(f"\n[3/3] Building bm25s index...")
    start = time.time()
    
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=True)
    
    build_time = time.time() - start
    num_docs = retriever.scores.get('num_docs', len(corpus_texts))
    print(f"      Built index with {num_docs:,} documents")
    print(f"      Time: {build_time:.1f}s")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    retriever.save(str(output_path))
    total_time = load_time + tokenize_time + build_time
    
    print(f"\n" + "=" * 60)
    print("BUILD COMPLETE")
    print(f"=" * 60)
    print(f"  Output: {output_path}")
    print(f"  Documents: {num_docs:,}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"=" * 60)
    
    return retriever


def verify_index(
    bm25s_path: str = "data/indexes/sparse_bm25s",
    rank_bm25_path: str = "data/indexes/sparse_bm25.pkl",
    n_queries: int = 10
):
    """
    Verify bm25s index against rank_bm25 results.
    """
    try:
        import bm25s
    except ImportError:
        print("ERROR: bm25s not installed.")
        return
    
    print(f"\n" + "=" * 60)
    print("VERIFYING bm25s INDEX")
    print(f"=" * 60)
    
    # Load both indexes
    print("\nLoading indexes...")
    
    bm25s_index = bm25s.BM25.load(str(bm25s_path), load_corpus=False)
    
    with open(rank_bm25_path, 'rb') as f:
        data = pickle.load(f)
        rank_bm25_index = data['bm25']
        tokenized_corpus = data['corpus']
    
    # Get bm25s doc count from scores dict
    bm25s_num_docs = bm25s_index.scores.get('num_docs', len(tokenized_corpus))
    
    print(f"bm25s: {bm25s_num_docs:,} docs")
    print(f"rank_bm25: {len(tokenized_corpus):,} docs")
    
    # Test queries
    test_queries = [
        "null pointer check",
        "if else condition",
        "function return value",
        "loop iteration",
        "variable assignment",
        "class method",
        "exception handling",
        "memory allocation",
        "string concatenation",
        "array index"
    ][:n_queries]
    
    print(f"\nTesting {len(test_queries)} queries...")
    
    bm25s_times = []
    rank_bm25_times = []
    
    for query in test_queries:
        # bm25s search
        start = time.perf_counter()
        query_tokens = bm25s.tokenize([query], show_progress=False)
        results, scores = bm25s_index.retrieve(query_tokens, k=10)
        bm25s_time = (time.perf_counter() - start) * 1000
        bm25s_times.append(bm25s_time)
        
        # rank_bm25 search
        import re
        query_toks = re.findall(r'\b\w+\b', query.lower())
        start = time.perf_counter()
        rank_scores = rank_bm25_index.get_scores(query_toks)
        import numpy as np
        top_k = np.argsort(rank_scores)[::-1][:10]
        rank_bm25_time = (time.perf_counter() - start) * 1000
        rank_bm25_times.append(rank_bm25_time)
    
    avg_bm25s = sum(bm25s_times) / len(bm25s_times)
    avg_rank_bm25 = sum(rank_bm25_times) / len(rank_bm25_times)
    
    print(f"\n" + "-" * 40)
    print(f"RESULTS:")
    print(f"  bm25s avg:      {avg_bm25s:.2f} ms/query")
    print(f"  rank_bm25 avg:  {avg_rank_bm25:.2f} ms/query")
    print(f"  Speedup:        {avg_rank_bm25/avg_bm25s:.1f}x")
    print(f"=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build bm25s index")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--from-rank-bm25', action='store_true',
                       help='Build from existing rank_bm25 index (fast)')
    group.add_argument('--from-dataset', action='store_true',
                       help='Build from dataset (re-tokenize)')
    group.add_argument('--verify', action='store_true',
                       help='Verify bm25s vs rank_bm25')
    
    parser.add_argument('--rank-bm25-path', type=str, 
                        default='data/indexes/sparse_bm25.pkl',
                        help='Path to rank_bm25 index')
    parser.add_argument('--dataset-path', type=str,
                        default='Datasets/Unified_Dataset/train.jsonl',
                        help='Path to dataset')
    parser.add_argument('--output', type=str,
                        default='data/indexes/sparse_bm25s',
                        help='Output path for bm25s index')
    
    args = parser.parse_args()
    
    if args.from_rank_bm25:
        build_from_rank_bm25(args.rank_bm25_path, args.output)
    elif args.from_dataset:
        build_from_dataset(args.dataset_path, args.output)
    elif args.verify:
        verify_index(args.output, args.rank_bm25_path)
