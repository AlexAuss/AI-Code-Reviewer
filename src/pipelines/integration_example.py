#!/usr/bin/env python3
"""
Complete Integration Example

This example shows exactly how your teammate will integrate the retriever
with their LLM and evaluation code. This is a working end-to-end example.

Run: python src/pipelines/integration_example.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.indexing.hybrid_retriever import HybridRetriever


# =============================================================================
# PART 1: YOUR PART (✅ DONE)
# =============================================================================

def initialize_retriever():
    """
    Initialize the retriever.
    This is all you need to provide - it's already done!
    """
    retriever = HybridRetriever(
        similarity_threshold=0.6,  # Optimal from evaluation
        use_ivf_index=True,
        parallel_search=True
    )
    return retriever


# =============================================================================
# PART 2: YOUR TEAMMATE'S PART (❌ TODO)
# =============================================================================

def generate_review_with_llm(code_patch: str, formatted_examples: str) -> str:
    """
    Generate review using LLM with few-shot examples.
    
    YOUR TEAMMATE IMPLEMENTS THIS.
    
    Args:
        code_patch: The code to review
        formatted_examples: Pre-formatted examples from retriever
        
    Returns:
        Generated review comment
    """
    # Build prompt
    prompt = f"""You are an expert code reviewer. Based on these examples of code patches and their reviews:

{formatted_examples}

Now review this code patch:
```
{code_patch}
```

Provide a constructive review comment:"""
    
    # TODO: Your teammate calls their LLM here
    # Option 1: OpenAI
    # import openai
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0.7,
    #     max_tokens=512
    # )
    # return response.choices[0].message.content
    
    # Option 2: Anthropic
    # import anthropic
    # client = anthropic.Anthropic()
    # response = client.messages.create(
    #     model="claude-3-opus-20240229",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return response.content[0].text
    
    # Option 3: Local model (e.g., via Ollama)
    # import requests
    # response = requests.post("http://localhost:11434/api/generate", 
    #     json={"model": "codellama", "prompt": prompt})
    # return response.json()["response"]
    
    # PLACEHOLDER for this example
    return "[LLM would generate review here based on examples]"


def compute_evaluation_metrics(generated_review: str, ground_truth_review: str) -> dict:
    """
    Compute metrics comparing generated vs ground truth.
    
    YOUR TEAMMATE IMPLEMENTS THIS (for evaluation pipeline only).
    
    Args:
        generated_review: LLM-generated review
        ground_truth_review: Ground truth from dataset
        
    Returns:
        Dictionary of metrics
    """
    # TODO: Your teammate implements metrics
    # 
    # from nltk.translate.bleu_score import sentence_bleu
    # from rouge_score import rouge_scorer
    # 
    # bleu = sentence_bleu([ground_truth_review.split()], 
    #                      generated_review.split())
    # 
    # scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    # rouge_scores = scorer.score(ground_truth_review, generated_review)
    
    # PLACEHOLDER for this example
    return {
        'bleu': 0.75,
        'rouge_1': 0.82,
        'rouge_2': 0.68,
        'rouge_l': 0.79
    }


# =============================================================================
# PART 3: INTEGRATION (How it all works together)
# =============================================================================

def complete_pipeline_example():
    """
    Shows the complete integration flow.
    """
    print("\n" + "=" * 80)
    print("COMPLETE INTEGRATION EXAMPLE")
    print("=" * 80)
    
    # Example code patch
    code_patch = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total / len(numbers)
    """
    
    # Ground truth (only for evaluation, not for UI)
    ground_truth_review = "Consider using sum() built-in function for cleaner code."
    
    print("\n1️⃣  Initializing retriever...")
    retriever = initialize_retriever()
    print(f"   ✅ Retriever ready (device: {retriever.device})")
    
    print("\n2️⃣  Retrieving similar examples...")
    retrieved_examples = retriever.retrieve(patch=code_patch, top_k=5)
    print(f"   ✅ Retrieved {len(retrieved_examples)} examples")
    
    print("\n3️⃣  Formatting examples for LLM prompt...")
    formatted_examples = retriever.format_for_llm_prompt(retrieved_examples)
    print(f"   ✅ Formatted {len(formatted_examples)} characters for prompt")
    
    print("\n4️⃣  Generating review with LLM...")
    generated_review = generate_review_with_llm(code_patch, formatted_examples)
    print(f"   ✅ Generated review")
    
    print("\n5️⃣  Computing evaluation metrics...")
    metrics = compute_evaluation_metrics(generated_review, ground_truth_review)
    print(f"   ✅ Computed metrics")
    
    # Show results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print("\n📥 INPUT (Code Patch):")
    print(code_patch)
    
    print("\n📤 OUTPUT (Generated Review):")
    print(generated_review)
    
    print("\n📊 EVALUATION METRICS:")
    for metric, value in metrics.items():
        print(f"   • {metric}: {value:.4f}")
    
    print("\n" + "=" * 80)
    print("\n✨ INTEGRATION COMPLETE!")
    print("\nWhat was used from YOUR part:")
    print("  ✅ HybridRetriever class")
    print("  ✅ retrieve() method")
    print("  ✅ format_for_llm_prompt() method")
    print("\nWhat YOUR TEAMMATE needs to implement:")
    print("  ❌ generate_review_with_llm() - Call their LLM")
    print("  ❌ compute_evaluation_metrics() - Implement BLEU/ROUGE")
    print("\nThat's it! Just 2 functions to implement. 🚀")
    print("=" * 80 + "\n")


def show_retriever_api():
    """Show all available retriever methods."""
    print("\n" + "=" * 80)
    print("RETRIEVER API REFERENCE")
    print("=" * 80)
    
    print("""
1. Initialize:
   retriever = HybridRetriever()

2. Retrieve examples:
   results = retriever.retrieve(
       patch=code_patch,
       top_k=5,
       apply_similarity_threshold=True
   )
   
   Returns: List[Dict] with keys:
   - 'original_patch': Code patch from dataset
   - 'review_comment': Review comment
   - 'retrieval_score': Fusion score
   - 'semantic_similarity': Cosine similarity
   - 'source_dataset': Dataset source
   - 'language': Programming language
   - Other metadata...

3. Format for LLM:
   formatted = retriever.format_for_llm_prompt(results)
   
   Returns: String formatted as:
   
   Example 1:
   Code Patch:
   <patch>
   
   Review Comment:
   <review>
   
   Example 2:
   ...

4. Convenience method (retrieve + format):
   results, formatted = retriever.retrieve_and_format(
       patch=code_patch,
       top_k=5
   )

5. Get timing stats:
   timing = retriever.get_timing_stats()
   timing.print_summary()
""")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Integration example")
    parser.add_argument('--api', action='store_true', help='Show API reference')
    args = parser.parse_args()
    
    if args.api:
        show_retriever_api()
    else:
        complete_pipeline_example()
