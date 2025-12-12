import streamlit as st
import time
import difflib

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
# MOCK API FUNCTION (UPDATED TO SIMULATE DELAY)
# -------------------------------------------------------------
def mock_api_call(before, after):
    """
    Simulates checking an API. 
    We sleep for 3 seconds so you can see the spinner in the UI.
    """
    time.sleep(3) # <--- Delay added here to simulate "working"
    return True

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
# MAIN LOGIC - TIMER/TIMEOUT LOOP
# -------------------------------------------------------------
if clicked:
    # 1. Clear previous result
    st.session_state.diff_html = None
    
    # 2. Setup Timeout variables
    start_time = time.time()
    max_duration_seconds = 120  # 2 Minutes
    api_success = False
    
    # 3. Processing Loop
    with st.spinner("Processing Code Review..."):
        while (time.time() - start_time) < max_duration_seconds:
            try:
                # Call the API (now takes 3 seconds)
                if mock_api_call(before_text, after_text):
                    api_success = True
                    break # Exit loop immediately on success
                else:
                    time.sleep(1) # Polling delay if API returns False (in-progress)
            except Exception as e:
                st.error(f"API Error: {e}")
                break
        
        # 4. Handle Result
        if not api_success:
            st.error(f"Operation timed out after {max_duration_seconds} seconds.")
        else:
            diff_data = get_comparison_results(before_text, after_text)
            
            # Save HTML
            st.session_state.diff_html = render_comparison_html(
                diff_data, 
                before_text=before_text, 
                after_text=after_text
            )
            
            # Save Debug/Patch data
            st.session_state['last_diff_data'] = diff_data
            st.session_state['original_patch'] = format_patch_as_display(
                diff_data, 
                before_text=before_text, 
                after_text=after_text
            )

st.subheader("Comparison & Feedback")

if st.session_state.diff_html is None:
    if not clicked:
        st.info("Press the button to compare the marked lines.")
else:
    st.markdown(st.session_state.diff_html, unsafe_allow_html=True)