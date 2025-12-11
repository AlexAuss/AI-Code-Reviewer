# Quick Reference: HybridRetriever Integration

## For Your Teammate 👨‍💻

### 1️⃣ Import (1 line)
```python
from src.indexing.hybrid_retriever import HybridRetriever
```

### 2️⃣ Initialize (1 line, at startup)
```python
retriever = HybridRetriever()  # Auto-configured with optimal settings
```

### 3️⃣ Retrieve (1 line, per request)
```python
results = retriever.retrieve(patch=user_code, top_k=5)
```

### 4️⃣ Format for LLM (1 line)
```python
formatted = retriever.format_for_llm_prompt(results)
```

### 5️⃣ Call YOUR LLM (implement this)
```python
review = your_llm_function(formatted)  # ← You implement this
```

---

## Complete Example (Copy-Paste Ready)

```python
from src.indexing.hybrid_retriever import HybridRetriever

# Initialize once
retriever = HybridRetriever()

# Process a request
code_patch = "def foo(): return 1"

# Get examples
results = retriever.retrieve(patch=code_patch, top_k=5)
formatted = retriever.format_for_llm_prompt(results)

# Build prompt
prompt = f"""Examples of code reviews:
{formatted}

Now review this code:
{code_patch}

Review:"""

# Call your LLM (OpenAI example)
import openai
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

review = response.choices[0].message.content
print(review)
```

---

## Files You Need

1. **Main Documentation**: `src/pipelines/README.md`
2. **Templates**:
   - Evaluation: `src/pipelines/evaluation_pipeline_template.py`
   - UI: `src/pipelines/ui_pipeline_template.py`
3. **Example**: `src/pipelines/integration_example.py`

---

## What You Implement (Just 2-3 Functions)

### For Evaluation Pipeline
```python
def generate_review_with_llm(code, examples):
    # Build prompt with examples
    # Call your LLM
    # Return generated review
    pass

def compute_evaluation_metrics(generated, ground_truth):
    # Compute BLEU, ROUGE, etc.
    # Return dict of metrics
    pass
```

### For Streamlit UI (Local Server)
```python
def generate_review_with_llm(code, examples):
    # Same as above
    pass

# No server code needed - Streamlit runs its own local server!
# Just integrate retriever directly in your Streamlit code:
import streamlit as st
retriever = HybridRetriever()
results = retriever.retrieve(patch=user_input)
st.write(results)  # Display in UI
```

---

## Test It First

```bash
# Test retriever
python src/indexing/demo_retriever.py --patch 'def test(): pass'

# Test integration example
python src/pipelines/integration_example.py

# See API reference
python src/pipelines/integration_example.py --api
```

---

## Performance

- **Initialization**: ~10 seconds (one-time)
- **Per retrieval**: ~2 seconds
- **Total per request**: ~5-7 seconds (including LLM)
- **Optimal K**: 5 examples
- **Optimal threshold**: 0.6
- **Quality**: MAP@K=1.0000 (validated on 23,422 samples)

---

## Quick Checklist

- [ ] Read `src/pipelines/README.md`
- [ ] Run demo: `python src/indexing/demo_retriever.py --patch 'test'`
- [ ] Copy template (evaluation OR streamlit UI)
- [ ] Implement `generate_review_with_llm()`
- [ ] Implement metrics (evaluation) OR integrate with Streamlit UI
- [ ] Test end-to-end
- [ ] Run locally: `streamlit run UI/codeReviewerGUI.py` 🚀

---

## Help?

- **Docs**: See `src/pipelines/README.md`
- **Examples**: See `src/pipelines/integration_example.py`
- **Architecture**: See `INTEGRATION_ARCHITECTURE.md`
- **Summary**: See `INTEGRATION_SUMMARY.md`

---

That's it! Just 5 lines of code to integrate retrieval! 🎉
