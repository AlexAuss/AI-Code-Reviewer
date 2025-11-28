import tkinter as tk
import tkinter.font as tkfont

HEADER_FONT = ("Segoe UI", 20, "bold")
LABEL_FONT  = ("Segoe UI", 16, "bold")
TEXT_FONT   = ("Consolas", 13)
NOTES_FONT  = ("Segoe UI", 13)
TAB_SIZE_SPACES = 4

class LineNumbers(tk.Canvas):
    def __init__(self, master, text_widget, **kwargs):
        super().__init__(master, width=48, highlightthickness=0, **kwargs)
        self.text_widget = text_widget
        self.text_widget.bind("<<Change>>", self._on_change, add=True)
        self.text_widget.bind("<Configure>", self._on_change, add=True)
        self.text_widget.bind("<KeyRelease>", self._on_change, add=True)
        self.text_widget.bind("<MouseWheel>", self._on_change, add=True)
        self.text_widget.bind("<Button-4>", self._on_change, add=True)
        self.text_widget.bind("<Button-5>", self._on_change, add=True)

    def _on_change(self, event=None):
        self.redraw()

    def redraw(self):
        self.delete("all")
        i = self.text_widget.index("@0,0")
        while True:
            dline = self.text_widget.dlineinfo(i)
            if dline is None:
                break
            y = dline[1]
            line_no = str(i).split(".")[0]
            self.create_text(44, y, anchor="ne", text=line_no)
            i = self.text_widget.index(f"{i}+1line")

class CodeText(tk.Text):
    def __init__(self, master, **kwargs):
        super().__init__(master, wrap="none", undo=True, font=TEXT_FONT, **kwargs)
        font = tkfont.Font(font=self.cget("font"))
        tab_pixels = font.measure(" " * TAB_SIZE_SPACES)
        self.configure(tabs=(tab_pixels,))
        self.bind("<Return>", self._auto_indent)
        self.bind("<Tab>", self._soft_tab)
        self.bind("<BackSpace>", self._backspace_indent)
        for ch in ("(", "[", "{", "'", '"'):
            self.bind(ch, self._auto_pair, add=True)
        self.bind("<<Modified>>", self._flag_change)
        self.bind("<KeyRelease>", self._highlight_current_line, add=True)
        self.bind("<ButtonRelease-1>", self._highlight_current_line, add=True)
        self.tag_configure("current_line", background="#f0f6ff")
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="Cut", command=lambda: self.event_generate("<<Cut>>"))
        self.menu.add_command(label="Copy", command=lambda: self.event_generate("<<Copy>>"))
        self.menu.add_command(label="Paste", command=lambda: self.event_generate("<<Paste>>"))
        self.menu.add_separator()
        self.menu.add_command(label="Select All", command=lambda: self.event_generate("<<SelectAll>>"))
        self.bind("<Button-3>", self._show_menu)
        self._highlight_current_line()

    def _get_indent_of_line(self, index):
        line_start = self.index(f"{index} linestart")
        line_text = self.get(line_start, f"{line_start} lineend")
        return len(line_text) - len(line_text.lstrip(" "))

    def _auto_indent(self, event):
        cur = self.index("insert")
        indent_len = self._get_indent_of_line(cur)
        prev_char = self.get(f"{cur} -1c")
        extra = TAB_SIZE_SPACES if prev_char in (":", "{", "(", "[") else 0
        self.insert(cur, "\n" + " " * (indent_len + extra))
        return "break"

    def _soft_tab(self, event):
        self.insert("insert", " " * TAB_SIZE_SPACES)
        return "break"

    def _backspace_indent(self, event):
        line_start = self.index("insert linestart")
        cur = self.index("insert")
        if self.compare(cur, ">", line_start):
            text = self.get(line_start, cur)
            if text.isspace():
                to_delete = len(text) % TAB_SIZE_SPACES or TAB_SIZE_SPACES
                self.delete(f"insert -{to_delete}c", "insert")
                return "break"
        return None

    def _auto_pair(self, event):
        pairs = {"(": ")", "[": "]", "{": "}", "'": "'", '"': '"'}
        ch = event.char
        close = pairs.get(ch)
        if not close:
            return
        self.insert("insert", ch + close)
        self.mark_set("insert", "insert -1c")
        return "break"

    def _flag_change(self, event=None):
        self.event_generate("<<Change>>", when="tail")
        self.edit_modified(0)

    def _highlight_current_line(self, event=None):
        self.tag_remove("current_line", "1.0", "end")
        self.tag_add("current_line", "insert linestart", "insert lineend+1c")

    def _show_menu(self, event):
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

def on_button_click():
    before = text_box.get("1.0", "end-1c")
    after  = mid_text.get("1.0", "end-1c")
    ensure_viewer()
    viewer_text.configure(state="normal")
    viewer_text.delete("1.0", "end")
    populate_marked_comparison(before, after)
    viewer_text.configure(state="disabled")
    root.after(0, place_sashes)

def toggle_fullscreen(event=None):
    current = bool(root.attributes("-fullscreen"))
    root.attributes("-fullscreen", not current)

def quit_fullscreen(event=None):
    root.attributes("-fullscreen", False)

def ensure_viewer():
    global viewer_frame, viewer_text, viewer_gutter, viewer_ys, viewer_xs
    if viewer_frame is not None:
        return
    viewer_frame = tk.Frame(paned, padx=24, pady=16)
    viewer_frame.grid_columnconfigure(2, weight=1)
    viewer_frame.grid_rowconfigure(1, weight=1)

    vlabel = tk.Label(viewer_frame, text="Comparison & Feedback (marked lines only)", font=LABEL_FONT)
    vlabel.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))

    viewer_ys = tk.Scrollbar(viewer_frame, orient="vertical")
    viewer_xs = tk.Scrollbar(viewer_frame, orient="horizontal")

    # Third text field (viewer)
    global viewer_text
    viewer_text = CodeText(viewer_frame)
    viewer_text.grid(row=1, column=2, sticky="nsew")
    viewer_text.configure(xscrollcommand=viewer_xs.set, yscrollcommand=viewer_ys.set)

    # Diff/feedback tags
    viewer_text.tag_configure("fix_semicolon", background="#e6ffed")
    viewer_text.tag_configure("diff_add", foreground="#22863a")
    viewer_text.tag_configure("diff_del", foreground="#cb2431")
    viewer_text.tag_configure("feedback", foreground="#6a737d")

    def _viewer_wheel(e):
        viewer_text.yview_scroll(int(-e.delta / 120), "units")
        return "break"
    viewer_text.bind("<MouseWheel>", _viewer_wheel)

    # Line numbers for viewer
    global viewer_gutter
    viewer_gutter = LineNumbers(viewer_frame, viewer_text)
    viewer_gutter.grid(row=1, column=1, sticky="ns", padx=(0, 6))

    viewer_ys.configure(command=viewer_text.yview)
    viewer_ys.grid(row=1, column=3, sticky="ns")
    viewer_xs.configure(command=viewer_text.xview)
    viewer_xs.grid(row=2, column=2, sticky="ew", pady=(8, 0))

    # Add viewer AFTER the button pane so the button stays between editor 2 and viewer
    paned.add(viewer_frame)
    paned.paneconfigure(viewer_frame, stretch="always", minsize=220)

def _insert_feedback_line(msg: str):
    fb_start = viewer_text.index("end")
    viewer_text.insert("end", msg + "\n")
    viewer_text.tag_add("feedback", fb_start, f"{fb_start} lineend+1c")

def _parse_marked(text: str, marker: str):
    """
    Return a list of code lines marked with the given marker.
    Accepts lines starting with '+' or '-' with or without a space after.
    """
    out = []
    for raw in text.splitlines():
        s = raw.lstrip()
        if s.startswith(marker):
            content = s[1:]
            if content.startswith(" "):
                content = content[1:]
            out.append(content)
    return out

def populate_marked_comparison(before_src: str, after_src: str):
    """
    Compare ONLY user-marked lines:
      - Before editor: lines starting with '-'
      - After editor:  lines starting with '+'
    Pair them by order and show a focused diff with feedback.
    """
    if viewer_text is None:
        return

    viewer_text.tag_remove("fix_semicolon", "1.0", "end")
    viewer_text.tag_remove("diff_add", "1.0", "end")
    viewer_text.tag_remove("diff_del", "1.0", "end")
    viewer_text.tag_remove("feedback", "1.0", "end")

    befores = _parse_marked(before_src, "-")
    afters  = _parse_marked(after_src, "+")

    viewer_text.insert("end", "Reviewing marked lines only (- in Before, + in After):\n\n")

    max_len = max(len(befores), len(afters))
    if max_len == 0:
        viewer_text.insert("end", "No marked lines to review.\n")
        return

    for i in range(max_len):
        old = befores[i] if i < len(befores) else None
        new = afters[i]  if i < len(afters)  else None

        if old is not None and new is not None:
            del_start = viewer_text.index("end")
            viewer_text.insert("end", f"- {old}\n")
            viewer_text.tag_add("diff_del", del_start, f"{del_start} lineend+1c")

            add_start = viewer_text.index("end")
            viewer_text.insert("end", f"+ {new}\n")
            viewer_text.tag_add("diff_add", add_start, f"{add_start} lineend+1c")

            if old.rstrip() + ";" == new.rstrip():
                prefix_len = len(old.rstrip())
                semi_start = f"{add_start}+{2 + prefix_len}c"
                semi_end   = f"{add_start}+{2 + prefix_len + 1}c"
                viewer_text.tag_add("fix_semicolon", semi_start, semi_end)
                _insert_feedback_line("Added missing semicolon.")
            elif old.rstrip() == new.rstrip():
                _insert_feedback_line("No effective change in this pair.")
            else:
                _insert_feedback_line("Edited line.")
        elif old is not None:
            del_start = viewer_text.index("end")
            viewer_text.insert("end", f"- {old}\n")
            viewer_text.tag_add("diff_del", del_start, f"{del_start} lineend+1c")
            _insert_feedback_line("Removed line.")
        elif new is not None:
            add_start = viewer_text.index("end")
            viewer_text.insert("end", f"+ {new}\n")
            viewer_text.tag_add("diff_add", add_start, f"{add_start} lineend+1c")
            _insert_feedback_line("Added line.")

        viewer_text.insert("end", "\n")

# ---------- App setup ----------
root = tk.Tk()
root.title("Simple UI")

viewer_frame = None
viewer_text = None
viewer_gutter = None
viewer_ys = None
viewer_xs = None

# Middle editor globals
mid_frame = None
mid_text = None
mid_gutter = None
mid_ys = None
mid_xs = None

try:
    root.state("zoomed")
except tk.TclError:
    root.attributes("-fullscreen", True)

# Root grid
root.columnconfigure(0, weight=1)
root.rowconfigure(1, weight=1)

# Header (fixed)
header = tk.Frame(root, padx=24, pady=12)
header.grid(row=0, column=0, sticky="ew")
header.columnconfigure(0, weight=1)
header_label = tk.Label(header, text="Generative AI Code Reviewer", font=HEADER_FONT)
header_label.grid(row=0, column=0, sticky="w")

# Scrollable wrapper for everything under the header
scroll_wrap = tk.Frame(root)
scroll_wrap.grid(row=1, column=0, sticky="nsew")
scroll_wrap.columnconfigure(0, weight=1)
scroll_wrap.rowconfigure(0, weight=1)

page_canvas = tk.Canvas(scroll_wrap, highlightthickness=0)
page_vbar = tk.Scrollbar(scroll_wrap, orient="vertical", command=page_canvas.yview)
page_canvas.configure(yscrollcommand=page_vbar.set)

page_canvas.grid(row=0, column=0, sticky="nsew")
page_vbar.grid(row=0, column=1, sticky="ns")

# Inner frame placed inside the canvas
page = tk.Frame(page_canvas)
page.columnconfigure(0, weight=1)
page_win = page_canvas.create_window((0, 0), window=page, anchor="nw")

def _update_scrollregion(_=None):
    page_canvas.configure(scrollregion=page_canvas.bbox("all"))

def _sync_inner_width(event):
    page_canvas.itemconfigure(page_win, width=event.width)

page.bind("<Configure>", _update_scrollregion)
page_canvas.bind("<Configure>", _sync_inner_width)

def _canvas_mousewheel(event):
    page_canvas.yview_scroll(int(-event.delta / 120), "units")

page_canvas.bind_all("<MouseWheel>", _canvas_mousewheel)

# Paned layout inside the scrollable page
paned = tk.PanedWindow(page, orient="vertical")
paned.grid(row=0, column=0, sticky="nsew")
page.rowconfigure(0, weight=1)

# Top pane: BEFORE editor
top = tk.Frame(paned, padx=24, pady=16)
top.grid_columnconfigure(2, weight=1)
top.grid_rowconfigure(2, weight=1)

label = tk.Label(top, text="Before (mark review lines with '-' at line start)", font=LABEL_FONT)
label.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))

ys = tk.Scrollbar(top, orient="vertical")
xs = tk.Scrollbar(top, orient="horizontal")

text_box = CodeText(top)
text_box.grid(row=2, column=2, sticky="nsew")
text_box.configure(xscrollcommand=xs.set, yscrollcommand=ys.set)

gutter = LineNumbers(top, text_box)
gutter.grid(row=2, column=1, sticky="ns", padx=(0, 6))

ys.configure(command=text_box.yview)
ys.grid(row=2, column=3, sticky="ns")
xs.configure(command=text_box.xview)
xs.grid(row=3, column=2, sticky="ew", pady=(8, 0))

paned.add(top)

# Middle pane: AFTER editor
mid_frame = tk.Frame(paned, padx=24, pady=16)
mid_frame.grid_columnconfigure(2, weight=1)
mid_frame.grid_rowconfigure(2, weight=1)

mid_label = tk.Label(mid_frame, text="After (mark review lines with '+' at line start)", font=LABEL_FONT)
mid_label.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))

mid_ys = tk.Scrollbar(mid_frame, orient="vertical")
mid_xs = tk.Scrollbar(mid_frame, orient="horizontal")

mid_text = CodeText(mid_frame)
mid_text.grid(row=2, column=2, sticky="nsew")
mid_text.configure(xscrollcommand=mid_xs.set, yscrollcommand=mid_ys.set)

mid_gutter = LineNumbers(mid_frame, mid_text)
mid_gutter.grid(row=2, column=1, sticky="ns", padx=(0, 6))

mid_ys.configure(command=mid_text.yview)
mid_ys.grid(row=2, column=3, sticky="ns")
mid_xs.configure(command=mid_text.xview)
mid_xs.grid(row=3, column=2, sticky="ew", pady=(8, 0))

paned.add(mid_frame)

# Button pane (must be BEFORE viewer so it stays between editors and viewer)
bottom = tk.Frame(paned, padx=24, pady=16)
bottom.columnconfigure(0, weight=1)
button = tk.Button(bottom, text="Review my Code", command=on_button_click)
button.grid(row=0, column=0, sticky="e")
paned.add(bottom)
paned.paneconfigure(bottom, minsize=60)

def place_sashes():
    h = paned.winfo_height()
    if h <= 0:
        root.after(10, place_sashes)
        return
    count = len(paned.panes())
    # With 3 panes (before, after, button)
    if count >= 1:
        paned.sash_place(0, 0, int(h * 0.30))   # between before and after
    if count >= 2:
        paned.sash_place(1, 0, int(h * 0.60))   # between after and button
    if count >= 3:
        # If viewer exists (4 panes), this is temporary until viewer added.
        # When viewer is added, we’ll also set sash 2 below.
        pass
    if count >= 4:
        paned.sash_place(2, 0, int(h * 0.80))   # between button and viewer

# Initial sash placement and key bindings
root.after(0, place_sashes)
text_box.bind("<Control-Return>", lambda _e: on_button_click())
text_box.focus()
root.bind("<F11>", toggle_fullscreen)
root.bind("<Escape>", quit_fullscreen)

root.mainloop()
