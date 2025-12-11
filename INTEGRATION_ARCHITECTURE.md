# 🔄 Integration Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     AI Code Reviewer System                          │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────┐     ┌──────────────────────────────┐
│   PIPELINE 1: EVALUATION     │     │   PIPELINE 2: STREAMLIT UI   │
│   (Test Dataset)             │     │   (Local Demo)               │
└──────────────────────────────┘     └──────────────────────────────┘
         │                                      │
         │                                      │
         ▼                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     YOUR PART (✅ IMPLEMENTED)                        │
│                                                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │               HybridRetriever                                │   │
│   │  • FAISS Dense Search (IVF, MPS/GPU)                        │   │
│   │  • BM25 Sparse Search (bm25s)                               │   │
│   │  • Reciprocal Rank Fusion                                   │   │
│   │  • Similarity Threshold Filtering (0.6)                     │   │
│   │  • K=5 (Optimal)                                            │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│   Methods:                                                            │
│   • retrieve(patch, top_k=5) → List[Dict]                           │
│   • format_for_llm_prompt(results) → str                            │
│   • retrieve_and_format(patch) → (results, formatted)               │
└──────────────────────────────────────────────────────────────────────┘
         │                                      │
         │ results, formatted_prompt            │ results, formatted_prompt
         ▼                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│              YOUR TEAMMATE'S PART (❌ TODO)                           │
│                                                                       │
│   ┌──────────────────────────┐     ┌──────────────────────────────┐ │
│   │  LLM Generation          │     │  LLM Generation              │ │
│   │  • Build prompt          │     │  • Build prompt              │ │
│   │  • Call LLM API          │     │  • Call LLM API              │ │
│   │  • Return review         │     │  • Return review             │ │
│   └──────────────────────────┘     └──────────────────────────────┘ │
│              │                                     │                  │
│              ▼                                     ▼                  │
│   ┌──────────────────────────┐     ┌──────────────────────────────┐ │
│   │  Evaluation Metrics      │     │  Streamlit Display           │ │
│   │  • BLEU                  │     │  • st.success(review)        │ │
│   │  • ROUGE                 │     │  • st.expander(examples)     │ │
│   │  • Semantic similarity   │     │  • Local server: port 8501   │ │
│   └──────────────────────────┘     └──────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Pipeline 1: Evaluation Flow

```
Test Dataset (23,273 samples)
    │
    │ for each sample:
    ├─► Sample: {patch, ground_truth_review}
    │
    ▼
┌─────────────────────────────────────┐
│  YOUR PART: Retrieval               │  ⏱️ ~2 seconds
│  retriever.retrieve(patch)          │
└─────────────────────────────────────┘
    │
    │ Retrieved examples (K=5)
    ▼
┌─────────────────────────────────────┐
│  YOUR TEAMMATE: LLM Generation      │  ⏱️ ~3-5 seconds
│  generate_review_with_llm()         │  (depends on LLM)
└─────────────────────────────────────┘
    │
    │ Generated review
    ▼
┌─────────────────────────────────────┐
│  YOUR TEAMMATE: Metrics             │  ⏱️ ~0.1 seconds
│  compute_evaluation_metrics()       │
└─────────────────────────────────────┘
    │
    │ Metrics (BLEU, ROUGE, etc.)
    ▼
Save results + aggregate metrics
```

**Total Time**: ~5-7 seconds per sample
**Full Evaluation**: ~32-45 hours (23,273 samples)

---

## 🖥️ Pipeline 2: Streamlit UI Flow

```
User opens browser → http://localhost:8501
    │
    │ Streamlit serves UI
    ▼
┌─────────────────────────────────────┐
│  Streamlit UI                       │
│  (UI/codeReviewerGUI.py)            │
└─────────────────────────────────────┘
    │
    │ User enters code & clicks button
    ▼
┌─────────────────────────────────────┐
│  YOUR PART: Retrieval               │  ⏱️ ~2 seconds
│  retriever.retrieve(patch)          │
└─────────────────────────────────────┘
    │
    │ Retrieved examples (K=5)
    ▼
┌─────────────────────────────────────┐
│  YOUR TEAMMATE: LLM Generation      │  ⏱️ ~3-5 seconds
│  generate_review_with_llm()         │
└─────────────────────────────────────┘
    │
    │ Generated review
    ▼
┌─────────────────────────────────────┐
│  Streamlit Display Results          │
│  st.success(review)                 │
│  st.expander(examples)              │
└─────────────────────────────────────┘
    │
    │ Rendered HTML
    ▼
User sees results in browser
```

**Total Response Time**: ~5-7 seconds
**User Experience**: Fast enough for local demo
**Server**: Built into Streamlit (port 8501)

---

## 📦 Module Dependencies

```
┌────────────────────────────────────────────────────────────┐
│  src/indexing/hybrid_retriever.py                          │
│  • HybridRetriever class                                   │
│  • No external pipeline dependencies                       │
│  • Can be imported anywhere                                │
└────────────────────────────────────────────────────────────┘
         │
         │ imported by
         ▼
┌────────────────────────────────────────────────────────────┐
│  src/pipelines/evaluation_pipeline_template.py             │
│  • Uses: HybridRetriever                                   │
│  • TODO: LLM call, metrics                                 │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  src/pipelines/ui_pipeline_template.py                     │
│  • Uses: HybridRetriever                                   │
│  • TODO: LLM call, Streamlit UI integration                │
│  • Template only - integrate into UI/codeReviewerGUI.py   │
└────────────────────────────────────────────────────────────┘
```

---

## 🔌 Integration Points (Detailed)

### Point 1: Initialize Retriever (One-time)

```python
from src.indexing.hybrid_retriever import HybridRetriever

# At startup (takes ~10 seconds)
retriever = HybridRetriever(
    index_dir="data/indexes",
    similarity_threshold=0.6,
    use_ivf_index=True,
    parallel_search=True
)

# Device auto-detected: MPS/CUDA/CPU
print(f"Ready on {retriever.device}")
```

### Point 2: Retrieve Examples (Per Request)

```python
# User's code patch
code_patch = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""

# Retrieve similar examples
results = retriever.retrieve(
    patch=code_patch,
    top_k=5,  # Get 5 examples
    apply_similarity_threshold=True  # Filter by 0.6 threshold
)

# Results structure:
# [
#   {
#     'original_patch': '...',
#     'review_comment': '...',
#     'retrieval_score': 0.95,
#     'semantic_similarity': 0.87,
#     'source_dataset': 'msg',
#     'language': 'python',
#     ...
#   },
#   ...
# ]
```

### Point 3: Format for LLM (Per Request)

```python
# Format examples for prompt
formatted = retriever.format_for_llm_prompt(results)

# Output format:
"""
Example 1:
Code Patch:
def foo():
    return bar()

Review Comment:
Consider error handling for bar() call.

Example 2:
...
"""
```

### Point 4: Generate Review (YOUR TEAMMATE)

```python
def generate_review_with_llm(code_patch, formatted_examples):
    prompt = f"""Based on these examples:
{formatted_examples}

Review this code:
{code_patch}

Review:"""
    
    # Call LLM
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
```

### Point 5: Streamlit Display (YOUR TEAMMATE)

```python
import streamlit as st

st.title("🤖 AI Code Reviewer")

code_input = st.text_area("Enter code:", height=200)

if st.button("Review"):
    with st.spinner("Retrieving examples..."):
        examples = retriever.retrieve(patch=code_input, top_k=5)
    
    with st.spinner("Generating review..."):
        review = generate_review_with_llm(code_input, examples)
    
    st.success(review)
```

---

## 📈 Performance Budget

| Component | Time | Notes |
|-----------|------|-------|
| **Retriever Init** | ~10s | One-time at startup |
| **Per Request** |  |  |
| - Retrieval | ~2s | Dense + Sparse + Fusion |
| - LLM Call | ~3-5s | Depends on model/API |
| - Metrics | ~0.1s | Evaluation only |
| **Total/Request** | ~5-7s | Acceptable for real-time |

---

## ✅ Implementation Status

### YOUR PART (✅ DONE)

- ✅ HybridRetriever class
- ✅ Optimal configuration (K=5, threshold=0.6)
- ✅ Helper methods (format_for_llm_prompt, retrieve_and_format)
- ✅ Demo script (demo_retriever.py)
- ✅ Pipeline templates
- ✅ Integration documentation
- ✅ Example code

### YOUR TEAMMATE'S PART (❌ TODO)

**For Both Pipelines:**
- ❌ Implement `generate_review_with_llm()`
  - Build prompt with examples
  - Call LLM API (OpenAI, Anthropic, local, etc.)
  - Return generated review

**For Evaluation Pipeline Only:**
- ❌ Implement `compute_evaluation_metrics()`
  - BLEU score
  - ROUGE scores (1, 2, L)
  - Semantic similarity (optional)

**For UI Pipeline Only:**
- ❌ Implement `generate_review_with_llm()`
  - Build prompt with examples
  - Call LLM API (OpenAI, Anthropic, local, etc.)
  - Return generated review
- ❌ Integrate with Streamlit UI
  - Import retriever in `UI/codeReviewerGUI.py`
  - Add button handler to call retriever
  - Display results with `st.success()`, `st.expander()`, etc.
  - No server code needed - Streamlit handles it!

---

## 🚦 Testing Checklist

### Before Integration

- ✅ Test retriever import: `python -c "from src.indexing.hybrid_retriever import HybridRetriever"`
- ✅ Test demo script: `python src/indexing/demo_retriever.py --patch 'test'`
- ✅ Test example: `python src/pipelines/integration_example.py`

### During Integration

- ⬜ Test LLM call with formatted examples
- ⬜ Test metrics computation
- ⬜ Test API endpoint
- ⬜ Test UI form submission

### After Integration

- ⬜ End-to-end evaluation pipeline test (10 samples)
- ⬜ End-to-end UI pipeline test
- ⬜ Performance test (measure latency)
- ⬜ Error handling test

---

## 📞 Support

**Questions about retriever?**
- See: `src/pipelines/README.md`
- Run: `python src/pipelines/integration_example.py --api`
- Test: `python src/indexing/demo_retriever.py --help`

**Ready to integrate?**
- Start with: `src/pipelines/evaluation_pipeline_template.py`
- Or: `src/pipelines/ui_pipeline_template.py`
- Just implement the TODO methods!

---

**🎉 Everything is modular and ready for integration!**
