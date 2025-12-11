#!/usr/bin/env python3
"""
Demo/Test script for HybridRetriever.

This file contains the CLI interface for testing retrieval.
The actual retriever logic is in hybrid_retriever.py.

Usage:
    python src/indexing/demo_retriever.py --patch 'def foo(): return 1'
    python src/indexing/demo_retriever.py --original-code 'def foo(): pass' --changed-code 'def foo(): return 1'
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.indexing.hybrid_retriever import HybridRetriever


def main():
    """Test retrieval with sample query."""
    parser = argparse.ArgumentParser(description="Demo/test hybrid retrieval")
    parser.add_argument('--index-dir', type=str, default='data/indexes',
                        help='Directory containing indexes')
    parser.add_argument('--embedding-model', type=str, default='microsoft/codebert-base',
                        help='Embedding model for queries')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--original-code', type=str, help='Original code snippet')
    group.add_argument('--patch', type=str, help='Patch/diff text to query')
    
    parser.add_argument('--changed-code', type=str, help='Changed code (with --original-code)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    parser.add_argument('--use-codebert-tokenizer', action='store_true',
                        help='Use CodeBERT tokenizer for BM25')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device for embeddings (auto-detect if not set)')
    parser.add_argument('--no-ivf', action='store_true',
                        help='Disable IVF index (use brute-force)')
    parser.add_argument('--nprobe', type=int, default=32,
                        help='FAISS nprobe for IVF search')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel search')
    parser.add_argument('--no-threshold', action='store_true',
                        help='Disable similarity threshold filtering')
    parser.add_argument('--timing-json', action='store_true',
                        help='Output timing as JSON')
    parser.add_argument('--no-timing', action='store_true',
                        help='Suppress timing output')
    parser.add_argument('--show-formatted', action='store_true',
                        help='Show LLM-formatted prompt')
    
    args = parser.parse_args()
    
    # Initialize retriever
    print("Initializing HybridRetriever...")
    retriever = HybridRetriever(
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        use_codebert_tokenizer=args.use_codebert_tokenizer,
        device=args.device,
        use_ivf_index=not args.no_ivf,
        faiss_nprobe=args.nprobe,
        parallel_search=not args.no_parallel
    )
    
    # Perform retrieval
    print("Retrieving examples...")
    if args.patch:
        results = retriever.retrieve(
            patch=args.patch, 
            top_k=args.top_k,
            apply_similarity_threshold=not args.no_threshold
        )
    else:
        if not args.changed_code:
            parser.error("--changed-code is required when using --original-code")
        results = retriever.retrieve(
            original_code=args.original_code,
            changed_code=args.changed_code,
            top_k=args.top_k,
            apply_similarity_threshold=not args.no_threshold
        )
    
    # Display results
    print("\n" + "=" * 80)
    print(f"Retrieved {len(results)} Examples:")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (Score: {result['retrieval_score']:.4f}) ---")
        if 'semantic_similarity' in result:
            print(f"Semantic Similarity: {result['semantic_similarity']:.4f}")
        print(f"Source: {result.get('source_dataset', 'N/A')}")
        print(f"Language: {result.get('language', 'N/A')}")
        print(f"Quality Label: {result.get('quality_label', 'N/A')}")
        
        patch_preview = (result.get('original_patch', 'N/A') or 'N/A')[:200]
        review_preview = (result.get('review_comment', 'N/A') or 'N/A')[:200]
        print(f"\nPatch:\n{patch_preview}...")
        print(f"\nReview:\n{review_preview}...")
        
        if result.get('refined_patch'):
            print(f"\nRefined Patch:\n{result['refined_patch'][:200]}...")
    
    print("\n" + "=" * 80)
    
    # Show LLM-formatted prompt if requested
    if args.show_formatted:
        print("\n" + "=" * 80)
        print("LLM-FORMATTED PROMPT:")
        print("=" * 80)
        formatted = retriever.format_for_llm_prompt(results)
        print(formatted)
        print("=" * 80)
    
    # Show timing stats
    if not args.no_timing:
        timing = retriever.get_timing_stats()
        if args.timing_json:
            print("\n--- TIMING STATS (JSON) ---")
            print(json.dumps(timing.to_dict(), indent=2))
        else:
            timing.print_summary()


if __name__ == '__main__':
    main()
