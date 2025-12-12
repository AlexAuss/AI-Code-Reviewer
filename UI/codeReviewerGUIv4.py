import streamlit as st

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
.fix_semicolon { background-color: #e6ffed; font-weight: bold; padding: 0 2px; }
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

# Default example: users can paste their own code or edit this example
default_before = """bool TransformationAddGlobalVariable::IsApplicable(
   if (!pointer_type) {
     return false;
   }
  // ... with Private storage class.
  if (pointer_type->storage_class() != SpvStorageClassPrivate) {
     return false;
   }
  // The initializer id must be the id of a constant.  Check this with the
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


# Diff-based line comparison
def get_diff_operations(before_src: str, after_src: str):
    """Get diff operations (insert, delete, replace, equal) with proper handling.
    
    Returns list of tuples: (tag, i1, i2, j1, j2)
    - tag: 'replace', 'delete', 'insert', or 'equal'
    - i1, i2: range in before text
    - j1, j2: range in after text
    """
    import difflib
    
    before_lines = before_src.splitlines()
    after_lines = after_src.splitlines()
    
    matcher = difflib.SequenceMatcher(None, before_lines, after_lines)
    return matcher.get_opcodes()

def get_comparison_results(before_src: str, after_src: str):
    """Compare before and after code, automatically detecting changed lines using opcodes."""
    before_lines = before_src.splitlines()
    after_lines = after_src.splitlines()
    opcodes = get_diff_operations(before_src, after_src)
    
    results = []
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'delete':
            # Deleted lines
            for line in before_lines[i1:i2]:
                entry = {"old": line, "new": None, "type": "javascript", "feedback": "Removed line."}
                results.append(entry)
        elif tag == 'insert':
            # Inserted lines
            for line in after_lines[j1:j2]:
                entry = {"old": None, "new": line, "type": "javascript", "feedback": "Added line."}
                results.append(entry)
        elif tag == 'replace':
            # Replaced lines - show deletions first, then additions
            for line in before_lines[i1:i2]:
                entry = {"old": line, "new": None, "type": "javascript", "feedback": "Removed line."}
                results.append(entry)
            for line in after_lines[j1:j2]:
                entry = {"old": None, "new": line, "type": "javascript", "feedback": "Added line."}
                results.append(entry)
    
    return results

def render_comparison_html(diff_data, before_text: str = None, after_text: str = None):
    """Render comparison HTML with full code diff and feedback at the end.
    
    Uses difflib opcodes to properly handle variable-length change blocks.
    Includes main feedback based on the overall change.
    """
    if not diff_data or not before_text or not after_text:
        return "<div class='viewer'><span class='feedback'>No changes detected.</span></div>"

    before_lines = before_text.splitlines()
    after_lines = after_text.splitlines()
    opcodes = get_diff_operations(before_text, after_text)
    
    html = ["<div class='viewer'>"]
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            # Context lines - show all unchanged lines
            for line in before_lines[i1:i2]:
                html.append(f" {line}\n")
        elif tag == 'delete':
            # Deletions - show all deleted lines
            for line in before_lines[i1:i2]:
                html.append(f"<span class='diff_del'>-{line}</span>\n")
        elif tag == 'insert':
            # Insertions - show all added lines
            for line in after_lines[j1:j2]:
                html.append(f"<span class='diff_add'>+{line}</span>\n")
        elif tag == 'replace':
            # Replacements - show all deletions first, then all additions
            for line in before_lines[i1:i2]:
                html.append(f"<span class='diff_del'>-{line}</span>\n")
            for line in after_lines[j1:j2]:
                html.append(f"<span class='diff_add'>+{line}</span>\n")
    
    # Generate main feedback based on the overall change
    main_feedback = generate_main_feedback(before_text, after_text, opcodes)
    
    # Add feedback section at the end
    if main_feedback:
        html.append("\n<div style='margin-top: 12px; padding-top: 12px; border-top: 1px solid #30363d;'>\n")
        html.append(f"<span class='feedback'><strong>Feedback/Review:</strong> {main_feedback}</span><br>\n")
        html.append("</div>\n")

    
    html.append("</div>")
    return ''.join(html)

def generate_main_feedback(before_text: str, after_text: str, opcodes) -> str:
    """Generate main feedback based on the entire change.
    
    Analyzes the type and scope of changes to provide a summary.
    """
    total_deleted = 0
    total_added = 0
    replace_count = 0
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'delete':
            total_deleted += i2 - i1
        elif tag == 'insert':
            total_added += j2 - j1
        elif tag == 'replace':
            total_deleted += i2 - i1
            total_added += j2 - j1
            replace_count += 1
    
    # Build summary based on change patterns
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







def serialize_diff_to_patch(diff_data) -> str:
    """Serialize the marked comparison results into a patch-like string.

    Produces lines starting with '-' for removed/old lines and '+' for
    added/new lines. This format is suitable to store as the
    `original_patch` field in the dataset JSONL records.
    """
    if not diff_data:
        return ""
    lines = []
    for item in diff_data:
        old = item.get("old")
        new = item.get("new")
        if old is not None:
            lines.append(f"-{old}")
        if new is not None:
            lines.append(f"+{new}")
    return "\n".join(lines)


def build_patch_from_texts(before_text: str, after_text: str, context_lines: int = 3) -> str:
    """Build a patch from before/after textareas with surrounding context.

    Strategy:
    1. Extract marked lines (- and +) from before/after.
    2. For each marked line, include K surrounding context lines.
    3. Output the hunks in order with minimal duplication.
    """
    b_lines = before_text.splitlines()
    a_lines = after_text.splitlines()

    # Find all marked line indices
    b_marked_indices = set()
    a_marked_indices = set()
    
    for idx, line in enumerate(b_lines):
        if line.lstrip().startswith('-'):
            b_marked_indices.add(idx)
    
    for idx, line in enumerate(a_lines):
        if line.lstrip().startswith('+'):
            a_marked_indices.add(idx)

    if not b_marked_indices and not a_marked_indices:
        # No marked lines; return empty or full text (choose based on preference)
        return ""

    # Collect all line indices to include (marked + context)
    included_b = set()
    included_a = set()

    # For each marked line in before, include it plus context
    for idx in b_marked_indices:
        for j in range(max(0, idx - context_lines), min(len(b_lines), idx + context_lines + 1)):
            included_b.add(j)

    # For each marked line in after, include it plus context
    for idx in a_marked_indices:
        for j in range(max(0, idx - context_lines), min(len(a_lines), idx + context_lines + 1)):
            included_a.add(j)

    out = []
    last_output_idx = -10  # Track last output index to avoid duplication

    # Iterate through before text and output included lines
    for idx in sorted(included_b):
        line = b_lines[idx]
        s = line.lstrip()
        
        if s.startswith('-'):
            # Marked removal line
            content = s[1:].lstrip() if s[1:].startswith(' ') else s[1:]
            out.append('-' + content)
        else:
            # Context line from before
            out.append(line)
        
        last_output_idx = idx

    # Iterate through after text and output marked added lines (with minimal context duplication)
    for idx in sorted(included_a):
        line = a_lines[idx]
        s = line.lstrip()
        
        if s.startswith('+'):
            # Marked addition line
            content = s[1:].lstrip() if s[1:].startswith(' ') else s[1:]
            out.append('+' + content)

    return "\n".join(out)


def format_patch_as_display(diff_data, before_text: str = None, after_text: str = None) -> str:
    """Format the patch as a unified diff showing all lines with proper alignment.
    
    Uses difflib opcodes to properly handle variable-length change blocks.
    Groups consecutive deletions before additions in replace blocks.
    """
    if not diff_data or not before_text or not after_text:
        return ""

    before_lines = before_text.splitlines()
    after_lines = after_text.splitlines()
    opcodes = get_diff_operations(before_text, after_text)
    
    out = []
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            # Context lines - show all unchanged lines
            for line in before_lines[i1:i2]:
                out.append(f" {line}")
        elif tag == 'delete':
            # Deletions - show all deleted lines
            for line in before_lines[i1:i2]:
                out.append(f"-{line}")
        elif tag == 'insert':
            # Insertions - show all added lines
            for line in after_lines[j1:j2]:
                out.append(f"+{line}")
        elif tag == 'replace':
            # Replacements - show all deletions first, then all additions
            for line in before_lines[i1:i2]:
                out.append(f"-{line}")
            for line in after_lines[j1:j2]:
                out.append(f"+{line}")
    
    return "\n".join(out)

# No dropdown: users can paste multiple marked blocks directly into the textareas.

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
if "review_in_progress" not in st.session_state:
    st.session_state.review_in_progress = False
if "review_time" not in st.session_state:
    st.session_state.review_time = None

if clicked:
    # Mark review as in progress and record the time
    st.session_state.review_in_progress = True
    st.session_state.review_time = __import__('time').time()
    
    # Build comparison and feedback from current text areas
    diff_data = get_comparison_results(before_text, after_text)
    st.session_state.diff_html = render_comparison_html(diff_data, before_text=before_text, after_text=after_text)

    # Store raw diff_data for debugging / inspection
    st.session_state['last_diff_data'] = diff_data

    # Serialize the marked before/after into a patch string formatted as it appears
    # in the Comparison & Feedback display (context lines + diff markers + feedback).
    # This is stored as `original_patch` in the dataset.
    display_patch = format_patch_as_display(diff_data, before_text=before_text, after_text=after_text)
    st.session_state['original_patch'] = display_patch

st.subheader("Comparison & Feedback")
if st.session_state.diff_html is None:
    st.info("Press the button to compare the marked lines.")
else:
    # Check if we need to show the loading timer
    if st.session_state.review_in_progress and st.session_state.review_time is not None:
        import time
        elapsed = time.time() - st.session_state.review_time
        wait_duration = 3
        
        if elapsed < wait_duration:
            # Show progress bar with timer
            remaining = wait_duration - elapsed
            progress_placeholder = st.empty()
            progress_placeholder.progress(elapsed / wait_duration)
            progress_placeholder.text(f"Processing... {remaining:.1f}s remaining")
            
            # Trigger rerun after a short delay
            time.sleep(0.1)
            st.rerun()
        else:
            # Timer finished, show results
            st.session_state.review_in_progress = False
            st.markdown(st.session_state.diff_html, unsafe_allow_html=True)
    else:
        st.markdown(st.session_state.diff_html, unsafe_allow_html=True)