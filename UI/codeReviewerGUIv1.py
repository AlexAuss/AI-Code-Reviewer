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
    src = text_box.get("1.0", "end-1c")
    ensure_viewer()
    viewer_text.configure(state="normal")
    viewer_text.delete("1.0", "end")
    viewer_text.insert("1.0", src)
    highlight_semicolon_errors()
    viewer_text.configure(state="disabled")
    ensure_notes()
    populate_notes_with_fixes(src)
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

    vlabel = tk.Label(viewer_frame, text="Code snapshot", font=LABEL_FONT)
    vlabel.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))

    # Scrollbars for the second text field
    viewer_ys = tk.Scrollbar(viewer_frame, orient="vertical")
    viewer_xs = tk.Scrollbar(viewer_frame, orient="horizontal")

    # Second text field
    viewer_text = CodeText(viewer_frame)
    viewer_text.grid(row=1, column=2, sticky="nsew")
    viewer_text.configure(xscrollcommand=viewer_xs.set, yscrollcommand=viewer_ys.set)
    # highlight tag uses background only (no text color change)
    viewer_text.tag_configure("error_semicolon", background="#ffe5e5")
    # Tag used to mark semicolon-at-EOL errors
    viewer_text.tag_configure("error_missing_semicolon", background="#ffe5e5")

    # Make mouse wheel scroll even when disabled
    def _viewer_wheel(e):
        viewer_text.yview_scroll(int(-e.delta / 120), "units")
        return "break"
    viewer_text.bind("<MouseWheel>", _viewer_wheel)

    # Line numbers for second field
    viewer_gutter = LineNumbers(viewer_frame, viewer_text)
    viewer_gutter.grid(row=1, column=1, sticky="ns", padx=(0, 6))

    viewer_ys.configure(command=viewer_text.yview)
    viewer_ys.grid(row=1, column=3, sticky="ns")
    viewer_xs.configure(command=viewer_text.xview)
    viewer_xs.grid(row=2, column=2, sticky="ew", pady=(8, 0))

    if notes_frame is not None:
        paned.add(viewer_frame, before=notes_frame)
    else:
        paned.add(viewer_frame)
    # Make the viewer pane grow and have a larger minimum height
    paned.paneconfigure(viewer_frame, stretch="always", minsize=220)


def highlight_semicolon_errors():
    """Highlight entire lines that do NOT end with a semicolon in the viewer_text."""
    if viewer_text is None:
        return
    # Clear any previous highlights
    viewer_text.tag_remove("error_semicolon", "1.0", "end")

    try:
        last_index = viewer_text.index("end-1c")
    except tk.TclError:
        return
    last_line = int(str(last_index).split(".")[0])

    for ln in range(1, last_line + 1):
        start = f"{ln}.0"
        end   = f"{ln}.end"
        line_text = viewer_text.get(start, end)
        rstrip = line_text.rstrip()
        # ignore empty/whitespace-only lines
        if not rstrip:
            continue
        # highlight lines whose final non-space char is NOT a semicolon
        if not rstrip.endswith(";"):
            viewer_text.tag_add("error_semicolon", start, end)


def populate_notes_with_fixes(src: str):
    """Fill the notes box with ONLY the corrected lines, followed by an explanation line.
    For each non-empty line in src that does not end with ';', append ';' and insert it,
    then insert the explanation: "You are missing a silicone".
    Newly added semicolons are highlighted in green.
    """
    if notes_text is None:
        return
    notes_text.configure(state="normal")
    notes_text.delete("1.0", "end")

    # Ensure the highlight tag exists
    try:
        notes_text.tag_configure("fix_semicolon")
    except tk.TclError:
        pass

    current_line = 1
    any_fixes = False
    for raw in src.splitlines():
        line = raw
        rstrip = line.rstrip()
        # Only include lines that are non-empty and missing a trailing semicolon
        if rstrip and not rstrip.endswith(";"):
            any_fixes = True
            prefix_len = len(rstrip)
            fixed = line[:prefix_len] + ";" + line[prefix_len:]

            # Insert corrected code line
            notes_text.insert("end", fixed + "\n")
            semi_start = f"{current_line}.{prefix_len}"
            semi_end   = f"{current_line}.{prefix_len + 1}"
            notes_text.tag_add("fix_semicolon", semi_start, semi_end)

            # Insert explanation line directly below
            notes_text.insert("end", "You are missing a silicone\n")

            # Move down two lines in the notes box
            current_line += 2

    if not any_fixes:
        notes_text.insert("end", "No lines required a trailing semicolon.\n")

    notes_text.configure(state="disabled")


def ensure_notes():
    global notes_frame, notes_text, notes_ys
    if notes_frame is not None:
        return
    notes_frame = tk.Frame(paned, padx=24, pady=16)
    notes_frame.grid_columnconfigure(0, weight=1)
    notes_frame.grid_rowconfigure(1, weight=1)

    nlabel = tk.Label(notes_frame, text="Describe the changes to fix the code", font=LABEL_FONT)
    nlabel.grid(row=0, column=0, sticky="w", pady=(0, 10))

    # Third text field, read only
    notes_text = tk.Text(notes_frame, wrap="word", font=NOTES_FONT, undo=False)
    notes_text.grid(row=1, column=0, sticky="nsew")
    # Tag for highlighting inserted semicolons in green
    notes_text.tag_configure("fix_semicolon", background="#e6ffed")

    # Read only state
    notes_text.configure(state="disabled")

    # Right click copy menu for read only field
    menu = tk.Menu(notes_text, tearoff=0)
    menu.add_command(label="Copy", command=lambda: notes_text.event_generate("<<Copy>>"))
    notes_text.bind("<Button-3>", lambda e: menu.tk_popup(e.x_root, e.y_root))

    # Scrollbar for the third field
    notes_ys = tk.Scrollbar(notes_frame, orient="vertical", command=notes_text.yview)
    notes_ys.grid(row=1, column=1, sticky="ns")
    notes_text.configure(yscrollcommand=notes_ys.set)

    paned.add(notes_frame)

def set_notes(text):
    """Utility to populate the read only notes box later."""
    if notes_text is None:
        return
    notes_text.configure(state="normal")
    notes_text.delete("1.0", "end")
    notes_text.insert("1.0", text)
    notes_text.configure(state="disabled")

def place_sashes():
    h = paned.winfo_height()
    if h <= 0:
        root.after(10, place_sashes)
        return
    # Give more space to the second (viewer) pane by default
    # Editor (top) gets ~35%, button a thin strip (~10%),
    # viewer gets from 45% down to either bottom or notes start.
    paned.sash_place(0, 0, int(h * 0.35))
    if len(paned.panes()) >= 3:
        paned.sash_place(1, 0, int(h * 0.45))
    if len(paned.panes()) >= 4:
        # With notes visible, leave viewer ~45% tall and notes ~10%
        paned.sash_place(2, 0, int(h * 0.90))

# ---------- App setup ----------
root = tk.Tk()
root.title("Simple UI")

viewer_frame = None
viewer_text = None
viewer_gutter = None
viewer_ys = None
viewer_xs = None
notes_frame = None
notes_text = None
notes_ys = None

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

# Top pane: entry label and code editor
top = tk.Frame(paned, padx=24, pady=16)
top.grid_columnconfigure(2, weight=1)
top.grid_rowconfigure(2, weight=1)

label = tk.Label(top, text="Enter the code you want reviewed", font=LABEL_FONT)
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

# Button pane
bottom = tk.Frame(paned, padx=24, pady=16)
bottom.columnconfigure(0, weight=1)
button = tk.Button(bottom, text="Review my Code", command=on_button_click)
button.grid(row=0, column=0, sticky="e")
paned.add(bottom)
paned.paneconfigure(bottom, minsize=60)

# Initial sash placement and key bindings
root.after(0, place_sashes)
text_box.bind("<Control-Return>", lambda _e: on_button_click())
text_box.focus()
root.bind("<F11>", toggle_fullscreen)
root.bind("<Escape>", quit_fullscreen)

root.mainloop()