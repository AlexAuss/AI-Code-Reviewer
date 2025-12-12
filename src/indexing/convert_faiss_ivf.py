"""
Convert existing FAISS IndexFlatIP (brute-force) to IndexIVFFlat (fast approximate search).

This conversion process:
1. Loads existing dense_faiss.index (IndexFlatIP)
2. Extracts all vectors
3. Trains an IVF index on a sample
4. Adds all vectors to the new index
5. Saves as dense_faiss_ivf.index

Expected time: ~30 minutes for 268K vectors
Expected speedup: ~1000x search speed (14s → 5ms per query)
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import faiss


def get_optimal_ivf_params(n_vectors: int) -> dict:
    """
    Calculate optimal IVF parameters based on dataset size.
    
    Rules of thumb:
    - nlist: sqrt(n) to 4*sqrt(n) for good balance
    - nprobe: 1-10% of nlist for 95-99% accuracy
    
    For 268K vectors:
    - nlist = 1024 (sqrt(268K) ≈ 518, so 1024 is good)
    - nprobe = 32 (3% of nlist, gives ~99% recall)
    """
    sqrt_n = int(np.sqrt(n_vectors))
    
    # Round up to nearest power of 2 for efficiency
    nlist = 1 << (sqrt_n - 1).bit_length()  # Next power of 2
    nlist = max(256, min(nlist, 4096))  # Clamp between 256 and 4096
    
    # nprobe = ~3% of nlist for high accuracy
    nprobe = max(8, nlist // 32)
    
    return {
        'nlist': nlist,
        'nprobe': nprobe,
        'expected_recall': '~99%'
    }


def convert_flat_to_ivf(
    input_path: str = "data/indexes/dense_faiss.index",
    output_path: str = "data/indexes/dense_faiss_ivf.index",
    nlist: int = None,
    nprobe: int = None,
    training_sample_ratio: float = 0.1
):
    """
    Convert IndexFlatIP to IndexIVFFlat.
    
    Args:
        input_path: Path to existing IndexFlatIP
        output_path: Path to save IndexIVFFlat
        nlist: Number of clusters (auto-calculated if None)
        nprobe: Number of clusters to search (auto-calculated if None)
        training_sample_ratio: Fraction of vectors to use for training (0.1 = 10%)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input index not found: {input_path}")
    
    print(f"=" * 60)
    print("FAISS Index Conversion: IndexFlatIP → IndexIVFFlat")
    print(f"=" * 60)
    
    # Step 1: Load existing index
    print(f"\n[1/5] Loading existing index from {input_path}...")
    start_time = time.time()
    
    flat_index = faiss.read_index(str(input_path))
    n_vectors = flat_index.ntotal
    dimension = flat_index.d
    
    load_time = time.time() - start_time
    print(f"      Loaded {n_vectors:,} vectors of dimension {dimension}")
    print(f"      Time: {load_time:.1f}s")
    
    # Step 2: Calculate optimal parameters
    if nlist is None or nprobe is None:
        params = get_optimal_ivf_params(n_vectors)
        nlist = nlist or params['nlist']
        nprobe = nprobe or params['nprobe']
    
    print(f"\n[2/5] IVF Parameters:")
    print(f"      nlist (clusters): {nlist}")
    print(f"      nprobe (search): {nprobe}")
    print(f"      Expected recall: ~99%")
    
    # Step 3: Extract vectors from flat index
    print(f"\n[3/5] Extracting vectors from flat index...")
    start_time = time.time()
    
    # Reconstruct all vectors (works for IndexFlatIP)
    all_vectors = flat_index.reconstruct_n(0, n_vectors)
    
    extract_time = time.time() - start_time
    print(f"      Extracted {all_vectors.shape[0]:,} vectors")
    print(f"      Memory: {all_vectors.nbytes / (1024**3):.2f} GB")
    print(f"      Time: {extract_time:.1f}s")
    
    # Step 4: Create and train IVF index
    print(f"\n[4/5] Training IVF index (this takes a few minutes)...")
    start_time = time.time()
    
    # Create quantizer (inner product for cosine similarity on normalized vectors)
    quantizer = faiss.IndexFlatIP(dimension)
    
    # Create IVF index
    ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # Train on a sample (faster than full dataset, similar quality)
    n_train = min(int(n_vectors * training_sample_ratio), n_vectors)
    n_train = max(n_train, nlist * 40)  # Need at least 40 vectors per cluster
    
    # Random sample for training
    train_indices = np.random.choice(n_vectors, size=min(n_train, n_vectors), replace=False)
    train_vectors = all_vectors[train_indices]
    
    print(f"      Training on {len(train_vectors):,} vectors ({len(train_vectors)/n_vectors*100:.1f}% of data)...")
    ivf_index.train(train_vectors)
    
    train_time = time.time() - start_time
    print(f"      Training complete in {train_time:.1f}s")
    
    # Step 5: Add all vectors to trained index
    print(f"\n[5/5] Adding all vectors to IVF index...")
    start_time = time.time()
    
    # Add in batches to show progress
    batch_size = 50000
    for i in range(0, n_vectors, batch_size):
        end_idx = min(i + batch_size, n_vectors)
        ivf_index.add(all_vectors[i:end_idx])
    
    add_time = time.time() - start_time
    print(f"      Added {ivf_index.ntotal:,} vectors")
    print(f"      Time: {add_time:.1f}s")
    
    # Set default nprobe for searches
    ivf_index.nprobe = nprobe
    
    # Save the index
    print(f"\n[✓] Saving IVF index to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(ivf_index, str(output_path))
    
    # Get file sizes
    input_size = input_path.stat().st_size / (1024**2)
    output_size = output_path.stat().st_size / (1024**2)
    
    total_time = load_time + extract_time + train_time + add_time
    
    print(f"\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print(f"=" * 60)
    print(f"  Input:  {input_path} ({input_size:.1f} MB)")
    print(f"  Output: {output_path} ({output_size:.1f} MB)")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"\n  Index parameters:")
    print(f"    • Vectors: {ivf_index.ntotal:,}")
    print(f"    • Dimension: {dimension}")
    print(f"    • nlist: {nlist}")
    print(f"    • nprobe: {nprobe} (configurable at search time)")
    print(f"\n  Expected performance:")
    print(f"    • Search speedup: ~1000x (14s → 5-10ms)")
    print(f"    • Recall: ~99%")
    print(f"=" * 60)
    
    return ivf_index


def verify_index(
    original_path: str = "data/indexes/dense_faiss.index",
    ivf_path: str = "data/indexes/dense_faiss_ivf.index",
    n_test_queries: int = 100,
    top_k: int = 10
):
    """
    Verify IVF index quality by comparing results with brute-force.
    
    Args:
        original_path: Path to original IndexFlatIP
        ivf_path: Path to converted IndexIVFFlat
        n_test_queries: Number of random queries to test
        top_k: Number of results to compare
    """
    print(f"\n" + "=" * 60)
    print("VERIFYING IVF INDEX QUALITY")
    print(f"=" * 60)
    
    # Load both indexes
    print(f"\nLoading indexes...")
    flat_index = faiss.read_index(str(original_path))
    ivf_index = faiss.read_index(str(ivf_path))
    
    n_vectors = flat_index.ntotal
    dimension = flat_index.d
    
    # Generate random query vectors (simulate real queries)
    print(f"Testing with {n_test_queries} random queries...")
    
    # Use random vectors from the index as queries
    query_indices = np.random.choice(n_vectors, size=n_test_queries, replace=False)
    query_vectors = flat_index.reconstruct_n(0, n_vectors)[query_indices]
    
    # Search with both indexes
    print(f"Searching with brute-force (IndexFlatIP)...")
    start = time.time()
    flat_distances, flat_indices = flat_index.search(query_vectors, top_k)
    flat_time = time.time() - start
    
    print(f"Searching with IVF (IndexIVFFlat, nprobe={ivf_index.nprobe})...")
    start = time.time()
    ivf_distances, ivf_indices = ivf_index.search(query_vectors, top_k)
    ivf_time = time.time() - start
    
    # Calculate recall (how many of the true top-k are found)
    recalls = []
    for i in range(n_test_queries):
        flat_set = set(flat_indices[i])
        ivf_set = set(ivf_indices[i])
        recall = len(flat_set & ivf_set) / len(flat_set)
        recalls.append(recall)
    
    avg_recall = np.mean(recalls)
    min_recall = np.min(recalls)
    
    print(f"\n" + "-" * 40)
    print(f"RESULTS:")
    print(f"  Brute-force time: {flat_time*1000:.1f}ms ({flat_time*1000/n_test_queries:.2f}ms/query)")
    print(f"  IVF time: {ivf_time*1000:.1f}ms ({ivf_time*1000/n_test_queries:.2f}ms/query)")
    print(f"  Speedup: {flat_time/ivf_time:.1f}x")
    print(f"\n  Recall@{top_k}:")
    print(f"    • Average: {avg_recall*100:.1f}%")
    print(f"    • Minimum: {min_recall*100:.1f}%")
    print(f"=" * 60)
    
    return avg_recall, ivf_time / n_test_queries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FAISS IndexFlatIP to IndexIVFFlat")
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/indexes/dense_faiss.index',
        help='Path to input IndexFlatIP'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/indexes/dense_faiss_ivf.index',
        help='Path to output IndexIVFFlat'
    )
    parser.add_argument(
        '--nlist',
        type=int,
        default=None,
        help='Number of clusters (auto-calculated if not specified)'
    )
    parser.add_argument(
        '--nprobe',
        type=int,
        default=None,
        help='Number of clusters to search (auto-calculated if not specified)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify index quality after conversion'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only run verification (skip conversion)'
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_index(args.input, args.output)
    else:
        convert_flat_to_ivf(
            input_path=args.input,
            output_path=args.output,
            nlist=args.nlist,
            nprobe=args.nprobe
        )
        
        if args.verify:
            verify_index(args.input, args.output)
