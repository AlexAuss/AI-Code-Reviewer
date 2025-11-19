# Phase 2: Indexing Module

This module builds dense and sparse indexes from the unified dataset for hybrid retrieval in the AI Code Reviewer system.

## Architecture

```
Unified Dataset (train.jsonl)
        ↓
┌───────────────────────────┐
│  Phase 2: Index Building  │
├───────────────────────────┤
│ • Dense (FAISS) Index     │
│ • Sparse (BM25) Index     │
│ • Metadata Storage        │
└───────────────────────────┘
        ↓
    Indexes saved in data/indexes/
        ↓
┌───────────────────────────┐
│   Hybrid Retriever        │
├───────────────────────────┤
│ • Dense search (semantic) │
│ • Sparse search (keyword) │
│ • Reciprocal Rank Fusion  │
└───────────────────────────┘
```

## Dataset Schema

The unified dataset (`Datasets/Unified_Dataset/train.jsonl`) should contain:

```json
{
  "original_file": "full file content (oldf from both datasets)",
  "language": "python",
  "original_patch": "unified diff (old_hunk from Code_Refinement, patch from Comment_Generation)",
  "refined_patch": "refined hunk (from Code_Refinement) or null",
  "review_comment": "human review (comment from Code_Refinement, msg from Comment_Generation)",
  "quality_label": "quality label (y from Comment_Generation) or null",
  "source_dataset": "code_refinement or comment_generation"
}
```

## Files

- `build_indexes.py` - Main script to build FAISS + BM25 indexes
- `hybrid_retriever.py` - Hybrid retrieval combining dense + sparse search
- `README.md` - This file

## Usage

### 1. Build Indexes

```bash
# Ensure unified dataset exists
# Expected path: Datasets/Unified_Dataset/train.jsonl

# Build indexes (will create data/indexes/ directory)
python3 src/indexing/build_indexes.py \
  --dataset Datasets/Unified_Dataset/train.jsonl \
  --output data/indexes \
  --batch-size 32
```

Options:
- `--dataset`: Path to unified dataset JSONL file
- `--output`: Output directory for indexes
- `--embedding-model`: Sentence transformer model (default: `sentence-transformers/all-mpnet-base-v2`)
- `--batch-size`: Batch size for embedding generation

Output files in `data/indexes/`:
- `dense_faiss.index` - FAISS index (cosine similarity, normalized vectors)
- `sparse_bm25.pkl` - BM25 index with tokenized corpus
- `metadata.jsonl` - Record metadata (one line per indexed record)

### 2. Test Retrieval

```bash
# Test hybrid retrieval with sample code diff
python3 src/indexing/hybrid_retriever.py \
  --index-dir data/indexes \
  --original-code "def foo(): return 1" \
  --changed-code "def foo(): return 2" \
  --top-k 5
```

### 3. Use in Python

```python
from src.indexing.hybrid_retriever import HybridRetriever

# Initialize retriever
retriever = HybridRetriever(
    index_dir="data/indexes",
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    dense_weight=0.5,
    sparse_weight=0.5
)

# Retrieve similar examples
results = retriever.retrieve(
    original_code="def foo():\n    return 1",
    changed_code="def foo():\n    return 2",
    top_k=5
)

# Each result contains:
# - doc_id: Index position
# - retrieval_score: RRF fusion score
# - original_patch: The diff/patch
# - refined_patch: Refined patch (if available)
# - review_comment: Human review/comment
# - language: Programming language
# - quality_label: Quality label (if available)
# - source_dataset: Dataset source
```

## Index Building Details

### Dense Index (FAISS)
- **Embedding model**: `sentence-transformers/all-mpnet-base-v2` (768-dim)
- **Index type**: `IndexFlatIP` (inner product, cosine similarity with normalized vectors)
- **Text format**: Combines language, patch, review, and refined patch fields
- **Normalization**: L2 normalization for cosine similarity

### Sparse Index (BM25)
- **Algorithm**: BM25Okapi
- **Tokenization**: Simple word tokenization (lowercase, regex-based)
- **Corpus**: Combines patch, review, and refined patch text
- **Storage**: Pickled with tokenized corpus for query processing

### Hybrid Search
- **Method**: Reciprocal Rank Fusion (RRF)
- **Formula**: `score(d) = Σ_R [weight / (k + rank_R(d))]` where R ∈ {dense, sparse}
- **Default weights**: 0.5 dense, 0.5 sparse
- **RRF constant k**: 60 (standard)

## Performance Notes

- **Index building**: ~1-2 minutes per 10K records (GPU recommended for embeddings)
- **Query time**: <100ms for hybrid search with top-k=5
- **Memory**: ~4KB per record for dense vectors (768-dim float32)
- **Disk**: ~1MB per 1K records (FAISS index + metadata)

## Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `sentence-transformers>=2.2.2` - Dense embeddings
- `faiss-cpu>=1.7.4` - Vector similarity search
- `rank-bm25>=0.2.2` - Sparse keyword search
- `torch>=1.9.0` - PyTorch backend

## Next Steps

After building indexes:
1. Integrate with LLM review generation (Phase 3)
2. Use retrieved examples as context for GPT/CodeLLaMA prompts
3. Implement quality assessment using retrieved similar changes
4. Fine-tune retrieval weights based on downstream task performance
