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
textarea, .stTextArea textarea {
  font-family: Consolas, monospace !important;
  font-size: 14px !important;
  line-height: 1.4 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Reusable pair variable ----------
SEMICOLON_PAIR = "- let count = 1\n+ let count = 1;"

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

def render_comparison_html(diff_data):
    if not diff_data:
        return "<div class='viewer'><span class='feedback'>No marked lines to review.</span></div>"
    html = ["<div class='viewer'>Reviewing marked lines only (- in Before, + in After):\n\n"]
    for item in diff_data:
        old_line = item["old"]
        new_line = item["new"]
        kind     = item["type"]
        fb       = item["feedback"]

        if old_line is not None:
            html.append(f"<span class='diff_del'>- {old_line}</span>\n")

        if new_line is not None:
            if kind == "semicolon":
                prefix = new_line.rstrip(";")
                html.append(f"<span class='diff_add'>+ {prefix}<span class='fix_semicolon'>;</span></span>\n")
            else:
                html.append(f"<span class='diff_add'>+ {new_line}</span>\n")

        if fb:
            html.append(f"<span class='feedback'>{fb}</span>\n")

        html.append("\n")
    html.append("</div>")
    return "".join(html)

# Defaults derived from the reusable pair
default_before = SEMICOLON_PAIR.splitlines()[0]
default_after  = SEMICOLON_PAIR.splitlines()[1]

st.title("Generative AI Code Reviewer")

left, right = st.columns(2)
with left:
    st.subheader("Before (mark with '-')")
    before_text = st.text_area("Before", value=default_before, height=120, label_visibility="collapsed")
with right:
    st.subheader("After (mark with '+')")
    after_text = st.text_area("After", value=default_after, height=120, label_visibility="collapsed")

btn_col, _ = st.columns([1, 6])
with btn_col:
    clicked = st.button("Review my Code", type="primary")

if "diff_html" not in st.session_state:
    st.session_state.diff_html = None

if clicked:
    # Save the reusable pair for later use elsewhere in the app if needed
    st.session_state["semicolon_pair"] = SEMICOLON_PAIR

    # Build comparison and feedback from current text areas
    # You can also switch to the stored pair at any time:
    #   before_from_pair = st.session_state["semicolon_pair"].splitlines()[0]
    #   after_from_pair  = st.session_state["semicolon_pair"].splitlines()[1]
    diff = get_comparison_results(before_text, after_text)
    st.session_state.diff_html = render_comparison_html(diff)

st.subheader("Comparison & Feedback")
if st.session_state.diff_html is None:
    st.info("Press the button to compare the marked lines.")
else:
    st.markdown(st.session_state.diff_html, unsafe_allow_html=True)
