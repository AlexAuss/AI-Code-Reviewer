"""
Demo script showing end-to-end index building and retrieval.
Creates a tiny sample dataset for testing if unified dataset doesn't exist.
"""

import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_dataset(output_path: Path, num_samples: int = 100):
    """
    Create a small sample unified dataset for testing.
    """
    logger.info(f"Creating sample dataset with {num_samples} records at {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sample templates
    samples = [
        {
            "original_file": "def calculate_sum(a, b):\n    return a + b\n",
            "language": "python",
            "original_patch": "@@ -1,2 +1,3 @@\n def calculate_sum(a, b):\n+    # Add two numbers\n     return a + b\n",
            "refined_patch": None,
            "review_comment": "Added documentation comment",
            "quality_label": 1,
            "source_dataset": "comment_generation"
        },
        {
            "original_file": "function validateEmail(email) {\n  return /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/.test(email);\n}\n",
            "language": "javascript",
            "original_patch": "@@ -1,3 +1,4 @@\n function validateEmail(email) {\n+  if (!email) return false;\n   return /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/.test(email);\n }\n",
            "refined_patch": "@@ -1,3 +1,4 @@\n function validateEmail(email) {\n+  if (!email || typeof email !== 'string') return false;\n   return /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/.test(email);\n }\n",
            "review_comment": "Should also check if email is a string",
            "quality_label": 0,
            "source_dataset": "code_refinement"
        },
        {
            "original_file": "public class User {\n  private String name;\n  public void setName(String name) { this.name = name; }\n}\n",
            "language": "java",
            "original_patch": "@@ -2,3 +2,4 @@\n   private String name;\n   public void setName(String name) {\n+    if (name == null) throw new IllegalArgumentException();\n     this.name = name;\n",
            "refined_patch": None,
            "review_comment": "Good: added null check",
            "quality_label": 1,
            "source_dataset": "comment_generation"
        },
        {
            "original_file": "#include <iostream>\nint main() {\n  int* ptr = new int(42);\n  delete ptr;\n  return 0;\n}\n",
            "language": "cpp",
            "original_patch": "@@ -3,4 +3,5 @@\n   int* ptr = new int(42);\n   delete ptr;\n+  ptr = nullptr;\n   return 0;\n }\n",
            "refined_patch": None,
            "review_comment": "Set pointer to nullptr after delete to prevent dangling pointer",
            "quality_label": 1,
            "source_dataset": "comment_generation"
        },
        {
            "original_file": "def divide(a, b):\n    return a / b\n",
            "language": "python",
            "original_patch": "@@ -1,2 +1,4 @@\n def divide(a, b):\n+    if b == 0:\n+        raise ValueError('Cannot divide by zero')\n     return a / b\n",
            "refined_patch": None,
            "review_comment": "Added zero division check",
            "quality_label": 1,
            "source_dataset": "comment_generation"
        }
    ]
    
    # Replicate samples to reach num_samples
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            sample = samples[i % len(samples)].copy()
            # Add some variation
            sample['id'] = i
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"Sample dataset created: {output_path}")


def main():
    """Run demo: create sample data, build indexes, test retrieval."""
    from src.indexing.build_indexes import build_all_indexes, IndexConfig
    from src.indexing.hybrid_retriever import HybridRetriever
    
    # Paths
    dataset_path = Path("Datasets/Unified_Dataset/train.jsonl")
    index_dir = Path("data/indexes")
    
    # Step 1: Check if dataset exists, if not create sample
    if not dataset_path.exists():
        logger.warning(f"Dataset not found at {dataset_path}")
        logger.info("Creating sample dataset for demo...")
        create_sample_dataset(dataset_path, num_samples=100)
    
    # Step 2: Build indexes
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Building Indexes")
    logger.info("=" * 60)
    
    config = IndexConfig(
        dataset_path=str(dataset_path),
        index_output_dir=str(index_dir),
        batch_size=16  # Small batch for demo
    )
    
    try:
        build_all_indexes(config)
    except Exception as e:
        logger.error(f"Error building indexes: {e}")
        return
    
    # Step 3: Test retrieval
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Testing Hybrid Retrieval")
    logger.info("=" * 60)
    
    retriever = HybridRetriever(index_dir=str(index_dir))
    
    # Test query 1: Division function with error handling
    original_code = """def divide(a, b):
    return a / b"""
    
    changed_code = """def divide(a, b):
    if b == 0:
        return None
    return a / b"""
    
    logger.info("\nQuery: Division function with zero check")
    logger.info(f"Original:\n{original_code}")
    logger.info(f"\nChanged:\n{changed_code}")
    
    results = retriever.retrieve(
        original_code=original_code,
        changed_code=changed_code,
        top_k=3
    )
    
    logger.info(f"\nTop {len(results)} Retrieved Examples:")
    for i, result in enumerate(results, 1):
        logger.info(f"\n--- Result {i} (Score: {result['retrieval_score']:.4f}) ---")
        logger.info(f"Language: {result.get('language')}")
        logger.info(f"Source: {result.get('source_dataset')}")
        logger.info(f"Quality: {result.get('quality_label')}")
        logger.info(f"Review: {result.get('review_comment', 'N/A')[:100]}...")
    
    # Test query 2: Null check in Java
    original_code_2 = """public void setName(String name) {
    this.name = name;
}"""
    
    changed_code_2 = """public void setName(String name) {
    if (name == null) throw new IllegalArgumentException();
    this.name = name;
}"""
    
    logger.info("\n" + "=" * 60)
    logger.info("\nQuery 2: Java null check")
    logger.info(f"Original:\n{original_code_2}")
    logger.info(f"\nChanged:\n{changed_code_2}")
    
    results_2 = retriever.retrieve(
        original_code=original_code_2,
        changed_code=changed_code_2,
        top_k=3
    )
    
    logger.info(f"\nTop {len(results_2)} Retrieved Examples:")
    for i, result in enumerate(results_2, 1):
        logger.info(f"\n--- Result {i} (Score: {result['retrieval_score']:.4f}) ---")
        logger.info(f"Language: {result.get('language')}")
        logger.info(f"Source: {result.get('source_dataset')}")
        logger.info(f"Review: {result.get('review_comment', 'N/A')[:100]}...")
    
    logger.info("\n" + "=" * 60)
    logger.info("Demo Complete!")
    logger.info("=" * 60)
    logger.info(f"\nIndexes saved to: {index_dir}")
    logger.info(f"- Dense FAISS index: {index_dir / 'dense_faiss.index'}")
    logger.info(f"- Sparse BM25 index: {index_dir / 'sparse_bm25.pkl'}")
    logger.info(f"- Metadata: {index_dir / 'metadata.jsonl'}")
    logger.info("\nYou can now use these indexes for RAG-based code review!")


if __name__ == '__main__':
    main()
