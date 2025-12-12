import streamlit as st
import sys
import time
import difflib
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.indexing.hybrid_retriever import HybridRetriever

# Page config
st.set_page_config(page_title="Generative AI Code Reviewer", layout="wide")

# Styles
st.markdown(
    """
<style>
.viewer {
    font-family: Consolas, monospace;
    font-size: 14px;
    line-height: 1.4;
    background: #0d1117;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 14px;
    max-height: 300px;
    overflow-y: auto;
    white-space: pre-wrap;
}
.diff_add { color: #22863a; }
.diff_del { color: #cb2431; }
.feedback { color: #6a737d; font-style: italic; }
.context { color: #9fb1c0; font-family: Consolas, monospace; }
textarea, .stTextArea textarea {
    font-family: Consolas, monospace !important;
    font-size: 14px !important;
    line-height: 1.4 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# Default example
default_before = """bool TransformationAddGlobalVariable::IsApplicable(
   if (!pointer_type) {
     return false;
   }
  // ... with Private storage class.
  if (pointer_type->storage_class() != SpvStorageClassPrivate) {
     return false;
   }
  // The initializer id must be the id of a constant. Check this with the
  // constant manager.
  auto constant_id = ir_context->get_constant_mgr()->GetConstantsFromIds(
      {message_.initializer_id()});
  if (constant_id.empty()) {
    return false;
  }
  assert(constant_id.size() == 1 &&
         "We asked for the constant associated with a single id; we should "
         "get a single constant.");
  // The type of the constant must match the pointee type of the pointer.
  if (pointer_type->pointee_type() != constant_id[0]->type()) {
    return false;
  }
"""

default_after = """bool TransformationAddGlobalVariable::IsApplicable(
   if (!pointer_type) {
     return false;
   }
  // ... with the right storage class.
  if (pointer_type->storage_class() != storage_class) {
     return false;
   }
  if (message_.initializer_id()) {
    // An initializer is not allowed if the storage class is Workgroup.
    if (storage_class == SpvStorageClassWorkgroup) {
      return false;
    }
"""


# -------------------------------------------------------------
# HELPER FUNCTIONS (Diffing & HTML Generation)
# -------------------------------------------------------------
def get_diff_operations(before_src: str, after_src: str):
    before_lines = before_src.splitlines()
    after_lines = after_src.splitlines()
    matcher = difflib.SequenceMatcher(None, before_lines, after_lines)
    return matcher.get_opcodes()

def get_comparison_results(before_src: str, after_src: str):
    before_lines = before_src.splitlines()
    after_lines = after_src.splitlines()
    opcodes = get_diff_operations(before_src, after_src)
    results = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'delete':
            for line in before_lines[i1:i2]:
                results.append({"old": line, "new": None, "type": "javascript", "feedback": "Removed line."})
        elif tag == 'insert':
            for line in after_lines[j1:j2]:
                results.append({"old": None, "new": line, "type": "javascript", "feedback": "Added line."})
        elif tag == 'replace':
            for line in before_lines[i1:i2]:
                results.append({"old": line, "new": None, "type": "javascript", "feedback": "Removed line."})
            for line in after_lines[j1:j2]:
                results.append({"old": None, "new": line, "type": "javascript", "feedback": "Added line."})
    return results

def generate_main_feedback(before_text: str, after_text: str, opcodes) -> str:
    total_deleted = 0
    total_added = 0
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'delete':
            total_deleted += i2 - i1
        elif tag == 'insert':
            total_added += j2 - j1
        elif tag == 'replace':
            total_deleted += i2 - i1
            total_added += j2 - j1
            
    if total_deleted == 0 and total_added == 0:
        return "No changes detected."
    elif total_deleted == 0:
        return f"Added {total_added} line(s)."
    elif total_added == 0:
        return f"Removed {total_deleted} line(s)."
    elif total_deleted == total_added:
        return f"Modified {total_deleted} line(s)."
    else:
        return f"Removed {total_deleted} line(s), added {total_added} line(s)."

def render_comparison_html(diff_data, before_text: str = None, after_text: str = None):
    if not diff_data or not before_text or not after_text:
        return "<div class='viewer'><span class='feedback'>No changes detected.</span></div>"

    before_lines = before_text.splitlines()
    after_lines = after_text.splitlines()
    opcodes = get_diff_operations(before_text, after_text)
    
    html = ["<div class='viewer'>"]
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            for line in before_lines[i1:i2]:
                html.append(f" {line}\n")
        elif tag == 'delete':
            for line in before_lines[i1:i2]:
                html.append(f"<span class='diff_del'>-{line}</span>\n")
        elif tag == 'insert':
            for line in after_lines[j1:j2]:
                html.append(f"<span class='diff_add'>+{line}</span>\n")
        elif tag == 'replace':
            for line in before_lines[i1:i2]:
                html.append(f"<span class='diff_del'>-{line}</span>\n")
            for line in after_lines[j1:j2]:
                html.append(f"<span class='diff_add'>+{line}</span>\n")
    
    main_feedback = generate_main_feedback(before_text, after_text, opcodes)
    
    if main_feedback:
        html.append("\n<div style='margin-top: 12px; padding-top: 12px; border-top: 1px solid #30363d;'>\n")
        html.append(f"<span class='feedback'><strong>Feedback/Review:</strong> {main_feedback}</span><br>\n")
        html.append("</div>\n")

    html.append("</div>")
    return ''.join(html)

def format_patch_as_display(diff_data, before_text: str = None, after_text: str = None) -> str:
    if not diff_data or not before_text or not after_text:
        return ""
    before_lines = before_text.splitlines()
    after_lines = after_text.splitlines()
    opcodes = get_diff_operations(before_text, after_text)
    out = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            for line in before_lines[i1:i2]:
                out.append(f" {line}")
        elif tag == 'delete':
            for line in before_lines[i1:i2]:
                out.append(f"-{line}")
        elif tag == 'insert':
            for line in after_lines[j1:j2]:
                out.append(f"+{line}")
        elif tag == 'replace':
            for line in before_lines[i1:i2]:
                out.append(f"-{line}")
            for line in after_lines[j1:j2]:
                out.append(f"+{line}")
    return "\n".join(out)

# -------------------------------------------------------------
# RETRIEVER INITIALIZATION (CACHED)
# -------------------------------------------------------------
@st.cache_resource
def load_retriever():
    """
    Load HybridRetriever with indexes. Runs once at startup and cached.
    This loads FAISS indexes, BM25 indexes, and connects to MongoDB.
    """
    with st.spinner("🔄 Loading retrieval system (first time only, ~30 seconds)..."):
        retriever = HybridRetriever(
            index_dir="data/indexes",
            dense_weight=0.5,
            sparse_weight=0.5,
            similarity_threshold=0.6,  # Optimal from evaluation
            use_ivf_index=True,
            parallel_search=True
        )
        # Load all indexes and connect to MongoDB
        retriever.load_indexes()
        return retriever

# Load retriever at startup (before UI)
try:
    retriever = load_retriever()
    st.success(f"✅ Retriever ready with {retriever.dense_index.ntotal} vectors")
except Exception as e:
    st.error(f"❌ Failed to load retriever: {e}")
    st.info("Make sure:\n1. Indexes exist in data/indexes/\n2. MongoDB is running: `brew services start mongodb-community`")
    retriever = None

# -------------------------------------------------------------
# RETRIEVAL FUNCTION (REPLACES MOCK API)
# -------------------------------------------------------------
def perform_retrieval(patch_text: str):
    """
    Perform hybrid retrieval using the loaded retriever.
    Returns retrieved examples and timing stats.
    """
    if retriever is None:
        raise Exception("Retriever not loaded")
    
    # Retrieve similar examples (K=5, threshold=0.6)
    retrieved_examples = retriever.retrieve(
        patch=patch_text,
        top_k=5,
        apply_similarity_threshold=True
    )
    
    # Get timing stats
    timing = retriever.get_timing_stats()
    
    return retrieved_examples, timing

# -------------------------------------------------------------
# UI LAYOUT
# -------------------------------------------------------------

st.title("Generative AI Code Reviewer")

left, right = st.columns(2)
with left:
    st.subheader("Before")
    before_text = st.text_area("Before", value=default_before, height=200, label_visibility="collapsed")
with right:
    st.subheader("After")
    after_text = st.text_area("After", value=default_after, height=200, label_visibility="collapsed")

btn_col, _ = st.columns([1, 6])
with btn_col:
    clicked = st.button("Review my Code", type="primary")

if "diff_html" not in st.session_state:
    st.session_state.diff_html = None

# -------------------------------------------------------------
# MAIN LOGIC - RETRIEVAL PIPELINE
# -------------------------------------------------------------
if clicked:
    if retriever is None:
        st.error("❌ Retriever not loaded. Cannot perform review.")
    else:
        # 1. Clear previous results
        st.session_state.diff_html = None
        st.session_state.retrieved_examples = None
        st.session_state.retrieval_time = None
        
        # 2. Analyze code changes
        with st.spinner("📝 Analyzing code changes..."):
            diff_data = get_comparison_results(before_text, after_text)
            patch_text = format_patch_as_display(
                diff_data, 
                before_text=before_text, 
                after_text=after_text
            )
        
        if not patch_text.strip():
            st.warning("⚠️ No changes detected between Before and After code.")
        else:
            # 3. Retrieve similar examples
            with st.spinner("🔍 Retrieving similar code reviews (~2 seconds)..."):
                try:
                    retrieved_examples, timing = perform_retrieval(patch_text)
                    
                    # Save results to session state
                    st.session_state['retrieved_examples'] = retrieved_examples
                    st.session_state['retrieval_time'] = timing.total_retrieval_ms / 1000
                    st.session_state['original_patch'] = patch_text
                    st.session_state['last_diff_data'] = diff_data
                    
                    # Generate comparison HTML
                    st.session_state.diff_html = render_comparison_html(
                        diff_data, 
                        before_text=before_text, 
                        after_text=after_text
                    )
                    
                    st.success(f"✅ Retrieved {len(retrieved_examples)} relevant examples in {timing.total_retrieval_ms/1000:.2f}s")
                    
                except Exception as e:
                    st.error(f"❌ Retrieval error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # TODO: Step 4 - LLM Generation (Your teammate will implement)
            # Once LLM is ready, add this:
            # 
            # with st.spinner("🤖 Generating AI review (~3-5 seconds)..."):
            #     formatted_examples = retriever.format_for_llm_prompt(retrieved_examples)
            #     llm_review = generate_review_with_llm(patch_text, formatted_examples)
            #     st.session_state.llm_review = llm_review

st.subheader("Comparison & Feedback")

if st.session_state.diff_html is None:
    st.info("👆 Press the 'Review my Code' button above to start the review process.")
else:
    # Show code comparison
    st.markdown("### 📋 Code Changes")
    st.markdown(st.session_state.diff_html, unsafe_allow_html=True)
    
    # Show retrieved examples
    if st.session_state.get('retrieved_examples'):
        st.markdown("---")
        st.markdown("### 📚 Similar Code Reviews from Training Data")
        
        examples = st.session_state['retrieved_examples']
        timing = st.session_state.get('retrieval_time', 0)
        st.caption(f"Found {len(examples)} relevant examples in {timing:.2f}s (K=5, threshold=0.6)")
        
        for i, example in enumerate(examples, 1):
            # Prepare display info
            lang = example.get('language', 'unknown')
            score = example.get('retrieval_score', 0)
            semantic_sim = example.get('semantic_similarity', 0)
            
            with st.expander(f"📝 Example {i} - {lang} (Retrieval Score: {score:.3f}, Similarity: {semantic_sim:.3f})", expanded=(i == 1)):
                # Similarity score
                if semantic_sim > 0:
                    st.caption(f"🎯 Semantic Similarity: {semantic_sim:.3f}")
                
                # Original patch
                st.markdown("**Original Patch:**")
                patch_code = example.get('original_patch') or example.get('patch', 'N/A')
                display_patch = patch_code[:800] + ('...' if len(patch_code) > 800 else '')
                st.code(display_patch, language='diff')
                
                # Review comment
                st.markdown("**Review Comment:**")
                review = example.get('review_comment', 'N/A')
                display_review = review[:800] + ('...' if len(review) > 800 else '')
                st.info(display_review)
                
                # Refined patch if available
                if example.get('refined_patch'):
                    st.markdown("**Refined/Fixed Code:**")
                    refined = example['refined_patch']
                    display_refined = refined[:800] + ('...' if len(refined) > 800 else '')
                    st.code(display_refined, language='diff')
                
                # Metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"📦 Source: {example.get('source_dataset', 'N/A')}")
                with col2:
                    st.caption(f"💻 Language: {lang}")
                with col3:
                    quality = example.get('quality_label', 'N/A')
                    st.caption(f"⭐ Quality: {quality}")
        
        # TODO: Show LLM-generated review (Your teammate will implement)
        st.markdown("---")
        st.markdown("### 🤖 AI-Generated Review")
        st.info("""
        **TODO: LLM Integration Pending**
        
        Once your teammate implements LLM integration, the AI-generated review will appear here.
        It will synthesize insights from the retrieved examples above.
        
        Implementation needed:
        1. Format examples using `retriever.format_for_llm_prompt(retrieved_examples)`
        2. Call LLM with formatted prompt + user's patch
        3. Display generated review here
        """)
        
        # Optional: Show performance stats
        with st.expander("🔧 Show Retrieval Performance Stats"):
            timing_obj = retriever.get_timing_stats()
            st.json(timing_obj.to_dict())