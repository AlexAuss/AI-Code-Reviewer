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

# Default multi-block content: users can paste multiple marked code blocks here.
default_before = """
# === Example 1: JavaScript (missing semicolon & formatting) ===
function incrementCounter() {
    // initialization
-    let count = 1
    for (let i = 0; i < 10; i++) {
        count += i
    }
    console.log(count)
}

# === Example 2: Python (None handling missing) ===
def process(item):
    # previously assumed item always present
-    return item.value

# === Example 3: Rename variable inside computation ===
def compute_all(values):
    # accumulate and report
-    total = compute_sum(values)
    logger.info("complete")

# === Example 4: Add type annotations (before) ===
def add(a, b):
-    return a + b

# === Example 5: Boolean correctness (before) ===
def is_valid(x):
-    return x

"""

default_after = """
# === Example 1: JavaScript (fixed semicolon & clean spacing) ===
function incrementCounter() {
    // initialization
+    let count = 1;
    for (let i = 0; i < 10; i++) {
        count += i;
    }
    console.log(count)
}

# === Example 2: Python (guard against None) ===
def process(item):
    # safely handle missing item
+    if item is None:
+        return None
+    return item.value

# === Example 3: Rename variable inside computation ===
def compute_all(values):
    # accumulate and report
+    sum_total = compute_sum(values)
    logger.info("complete")

# === Example 4: Add type annotations (after) ===
def add(a: int, b: int) -> int:
+    return a + b

# === Example 5: Boolean correctness (after) ===
def is_valid(x):
+    return bool(x)

"""


# Strict parsing of marked lines
def _parse_marked(text: str, marker: str):
    out = []
    for raw in text.splitlines():
        s = raw.lstrip()
        if s.startswith(marker):
            content = s[1:]
            if content.startswith(" "):
                content = content[1:]
            out.append(content)
    return out

def get_comparison_results(before_src: str, after_src: str):
    befores = _parse_marked(before_src, "-")
    afters  = _parse_marked(after_src, "+")
    results = []
    max_len = max(len(befores), len(afters))
    for i in range(max_len):
        old_line = befores[i] if i < len(befores) else None
        new_line = afters[i]  if i < len(afters)  else None
        entry = {"old": old_line, "new": new_line, "type": "none", "feedback": ""}
        if old_line is not None and new_line is not None:
            if old_line.rstrip() + ";" == new_line.rstrip():
                entry["type"] = "semicolon"
                entry["feedback"] = "Added missing semicolon."
            elif old_line.rstrip() == new_line.rstrip():
                entry["type"] = "no_change"
                entry["feedback"] = "No effective change in this pair."
            else:
                entry["type"] = "edit"
                entry["feedback"] = "Edited line."
        elif old_line is not None:
            entry["type"] = "remove"
            entry["feedback"] = "Removed line."
        elif new_line is not None:
            entry["type"] = "add"
            entry["feedback"] = "Added line."
        results.append(entry)
    return results

def render_comparison_html(diff_data, before_text: str = None, after_text: str = None):
    """Render comparison HTML including a nearby context line from the
    original Before/After text blocks.

    For each marked pair we attempt to show the previous non-marked line
    from the corresponding Before/After block as context.
    """
    if not diff_data:
        return "<div class='viewer'><span class='feedback'>No marked lines to review.</span></div>"

    def _collect_context_blocks(text: str, marker: str, k: int = 2):
        """Collect up to `k` non-marked context lines before and after each marked
        line in `text`. Returns a list aligned with the marked lines order where
        each element is a dict: {'pre': [...], 'post': [...]}.
        """
        if not text:
            return []
        lines = text.splitlines()
        blocks = []
        for idx, line in enumerate(lines):
            s = line.lstrip()
            if s.startswith(marker):
                # collect previous k non-marked lines
                pre = []
                for j in range(idx - 1, -1, -1):
                    prev = lines[j]
                    ps = prev.lstrip()
                    if ps.startswith('-') or ps.startswith('+'):
                        continue
                    if ps.strip() == '':
                        continue
                    pre.append(ps)
                    if len(pre) >= k:
                        break
                pre.reverse()

                # collect next k non-marked lines
                post = []
                for j in range(idx + 1, len(lines)):
                    nxt = lines[j]
                    ns = nxt.lstrip()
                    if ns.startswith('-') or ns.startswith('+'):
                        continue
                    if ns.strip() == '':
                        continue
                    post.append(ns)
                    if len(post) >= k:
                        break

                blocks.append({'pre': pre, 'post': post})
        return blocks

    # number of context lines to show before/after
    K = 2
    before_blocks = _collect_context_blocks(before_text or '', '-', K)
    after_blocks = _collect_context_blocks(after_text or '', '+', K)

    html = ["<div class='viewer'>Reviewing marked lines with nearby context:\n\n"]
    for i, item in enumerate(diff_data):
        old_line = item.get('old')
        new_line = item.get('new')
        kind = item.get('type')
        fb = item.get('feedback')

        # show pre-context from before_blocks if present
        if i < len(before_blocks):
            for ctx_line in before_blocks[i]['pre']:
                html.append(f"<div class='context'> {ctx_line}</div>\n")

        if old_line is not None:
            html.append(f"<span class='diff_del'>- {old_line}</span>\n")

        if new_line is not None:
            if kind == 'semicolon':
                prefix = new_line.rstrip(';')
                html.append(f"<span class='diff_add'>+ {prefix}<span class='fix_semicolon'>;</span></span>\n")
            else:
                html.append(f"<span class='diff_add'>+ {new_line}</span>\n")

        # show post-context from after_blocks if present
        if i < len(after_blocks):
            for ctx_line in after_blocks[i]['post']:
                html.append(f"<div class='context'> {ctx_line}</div>\n")

        if fb:
            html.append(f"<span class='feedback'>{fb}</span>\n")

        html.append('\n')

    html.append('</div>')
    return ''.join(html)


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

# No dropdown: users can paste multiple marked blocks directly into the textareas.

st.title("Generative AI Code Reviewer")

left, right = st.columns(2)
with left:
    st.subheader("Before (mark with '-')")
    before_text = st.text_area("Before", value=default_before, height=200, label_visibility="collapsed")
with right:
    st.subheader("After (mark with '+')")
    after_text = st.text_area("After", value=default_after, height=200, label_visibility="collapsed")

btn_col, _ = st.columns([1, 6])
with btn_col:
    clicked = st.button("Review my Code", type="primary")

if "diff_html" not in st.session_state:
    st.session_state.diff_html = None

if clicked:
    # Save the serialized patch into session state (no semicolon_pair tracking)

    # Build comparison and feedback from current text areas
    # You can also switch to the stored pair at any time:
    #   before_from_pair = st.session_state["semicolon_pair"].splitlines()[0]
    #   after_from_pair  = st.session_state["semicolon_pair"].splitlines()[1]
    diff_data = get_comparison_results(before_text, after_text)
    st.session_state.diff_html = render_comparison_html(diff_data, before_text=before_text, after_text=after_text)

    # Serialize the marked before/after into a patch string suitable for
    # storing as `original_patch` in the dataset. Keep it in session state
    # so other parts of the app or export scripts can pick it up.
    patch_text = serialize_diff_to_patch(diff_data)
    st.session_state['original_patch'] = patch_text

st.subheader("Comparison & Feedback")
if st.session_state.diff_html is None:
    st.info("Press the button to compare the marked lines.")
else:
    st.markdown(st.session_state.diff_html, unsafe_allow_html=True)
    