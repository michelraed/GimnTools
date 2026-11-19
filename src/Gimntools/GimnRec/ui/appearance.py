"""Desktop typography and light/dark presentation for the VTK launcher."""

import sqlite3
import tkinter as tk
from tkinter import font, ttk
from weakref import WeakSet

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


_PALETTES = {
    "light": {
        "window": "#eef3f9", "surface": "#ffffff", "raised": "#e6edf6",
        "hover": "#d7e4ef", "text": "#17283e", "muted": "#50647e",
        "border": "#b8c8da", "accent": "#096b60", "accent_hover": "#07564d",
        "accent_text": "#ffffff", "selection": "#cdeae6",
    },
    "dark": {
        "window": "#101826", "surface": "#172334", "raised": "#223247",
        "hover": "#30465e", "text": "#edf4fe", "muted": "#a6b9d1",
        "border": "#3d526d", "accent": "#65e1c8", "accent_hover": "#8fead7",
        "accent_text": "#11253a", "selection": "#284a52",
    },
}


class _ThemedCanvas(FigureCanvasTkAgg):
    def __init__(self, figure, master, appearance):
        self.appearance = appearance
        super().__init__(figure, master=master)
        appearance.canvases.add(self)

    def draw(self):
        self.appearance.style_figure(self.figure)
        super().draw()


class DesktopTypography:
    """Select an installed family and update the named fonts in place."""

    AUTO = "Automatic"

    def __init__(self, root, session):
        self.root = root
        self.session = session
        self.installed = {name.casefold(): name for name in font.families(root=root)
                          if not name.startswith("@")}
        self.default_family = self._default_family()
        self.heading = font.Font(root=root, size=18, weight="bold")
        saved = session.get("ui.font")
        family = self.installed.get(saved.casefold()) if isinstance(saved, str) else None
        self.selection = tk.StringVar(root, value=family or self.AUTO)
        self.apply()

    def _default_family(self):
        candidates = ("Inter", "Segoe UI", "Helvetica Neue", "Noto Sans", "Ubuntu",
                      "DejaVu Sans", "Liberation Sans", "URW Gothic L", "Helvetica", "Arial")
        for candidate in candidates:
            family = self.installed.get(candidate.casefold())
            if family:
                trial = font.Font(root=self.root, family=family, size=11)
                if trial.measure("iii") < trial.measure("mmm"):
                    return trial.actual("family")
        return font.nametofont("TkDefaultFont", root=self.root).actual("family")

    def picker(self, parent, *, label_style="TLabel"):
        row = ttk.Frame(parent, style=parent.cget("style") or "TFrame")
        ttk.Label(row, text="UI font", style=label_style).pack(side=tk.LEFT, padx=(0, 6))
        combo = ttk.Combobox(row, textvariable=self.selection, state="readonly", width=22,
                             values=(self.AUTO, *sorted(self.installed.values(), key=str.casefold)))
        combo.pack(side=tk.LEFT)
        combo.bind("<<ComboboxSelected>>", self._choose)
        return row

    def _choose(self, _event):
        self.apply()
        self.session.remember("ui.font", self.selection.get())

    def apply(self):
        family = self.installed.get(self.selection.get().casefold(), self.default_family)
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkCaptionFont"):
            font.nametofont(name, root=self.root).configure(family=family, size=11)
        font.nametofont("TkHeadingFont", root=self.root).configure(family=family, size=11, weight="bold")
        self.heading.configure(family=family)
        style = ttk.Style(self.root)
        style.configure(".", font="TkDefaultFont")
        style.configure("TCombobox", font="TkDefaultFont")
        linespace = font.nametofont("TkDefaultFont", root=self.root).metrics("linespace")
        style.configure("Treeview", rowheight=linespace + 9)


class DesktopAppearance:
    def __init__(self, root, store, typography):
        self.root = root
        self.store = store
        self.canvases = WeakSet()
        self.style = ttk.Style(root)
        self.style.theme_create("gimntools", parent="clam")
        self.style.theme_use("gimntools")
        self.typography = typography
        self.typography.apply()
        self.strong = font.nametofont("TkHeadingFont", root=root)
        saved = store.get_setting("vtk_desktop_theme", "dark")
        self.mode = tk.StringVar(root, value=saved if saved in ("light", "dark") else "dark")
        self.apply()
        self._header()

    def _header(self):
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(18, 9))
        header.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(header, text="GimnTools", style="Brand.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text="PET Reconstruction", style="Header.TLabel").pack(
            side=tk.LEFT, padx=(14, 0))
        controls = ttk.Frame(header, style="Header.TFrame")
        controls.pack(side=tk.RIGHT)
        self.typography.picker(controls, label_style="Header.TLabel").pack(side=tk.LEFT, padx=(0, 14))
        ttk.Label(controls, text="Theme", style="Header.TLabel").pack(side=tk.LEFT, padx=(0, 8))
        for mode, label in (("light", "Light"), ("dark", "Dark")):
            ttk.Radiobutton(controls, text=label, value=mode, variable=self.mode,
                            style="Theme.TRadiobutton", command=self._switch).pack(
                side=tk.LEFT, padx=(0, 2))

    def _switch(self):
        self.apply()
        try:
            self.store.set_setting("vtk_desktop_theme", self.mode.get())
        except sqlite3.Error as exc:
            self.root.log(f"Could not save theme preference: {exc}")

    def apply(self):
        p = self.palette = _PALETTES[self.mode.get()]
        s = self.style
        s.configure(".", background=p["window"], foreground=p["text"],
                    bordercolor=p["border"], lightcolor=p["raised"], darkcolor=p["raised"],
                    troughcolor=p["surface"], selectbackground=p["selection"],
                    selectforeground=p["text"], focuscolor=p["accent"], font="TkDefaultFont")
        s.configure("Header.TFrame", background=p["surface"])
        s.configure("Header.TLabel", background=p["surface"], foreground=p["muted"])
        s.configure("Brand.TLabel", background=p["surface"], foreground=p["text"], font=self.typography.heading)
        s.configure("Hint.TLabel", foreground=p["muted"])
        s.configure("Emphasis.TLabel", foreground=p["accent"], font=self.strong)

        s.configure("TNotebook", borderwidth=0, tabmargins=(0, 0, 0, 2))
        s.configure("TNotebook.Tab", background=p["raised"], padding=(14, 7), borderwidth=0)
        s.map("TNotebook.Tab", background=[("selected", p["surface"]), ("active", p["hover"])],
              foreground=[("selected", p["accent"]), ("!selected", p["muted"])])
        s.configure("TLabelframe", borderwidth=1, relief="solid")
        s.configure("TLabelframe.Label", foreground=p["text"], font=self.strong)

        s.configure("TButton", background=p["raised"], padding=(8, 4), borderwidth=1)
        s.map("TButton", background=[("pressed", p["selection"]), ("active", p["hover"])])
        s.configure("Primary.TButton", background=p["accent"], foreground=p["accent_text"],
                    bordercolor=p["accent"], lightcolor=p["accent"], darkcolor=p["accent"],
                    padding=(10, 6), font=self.strong)
        s.map("Primary.TButton", background=[("active", p["accent_hover"])],
              foreground=[("disabled", p["muted"])])
        s.layout("Theme.TRadiobutton", s.layout("TButton"))
        s.configure("Theme.TRadiobutton", background=p["raised"], padding=(10, 4), font=self.strong)
        s.map("Theme.TRadiobutton",
              background=[("selected", p["selection"]), ("active", p["hover"])],
              foreground=[("selected", p["accent"])])

        for name in ("TEntry", "TSpinbox", "TCombobox"):
            s.configure(name, fieldbackground=p["surface"], background=p["raised"],
                        insertcolor=p["text"], arrowcolor=p["text"], padding=3)
            s.map(name, fieldbackground=[("readonly", p["surface"]), ("disabled", p["raised"])],
                  foreground=[("disabled", p["muted"])], bordercolor=[("focus", p["accent"])])
        for name in ("TCheckbutton", "TRadiobutton"):
            s.map(name, indicatorbackground=[("selected", p["accent"]), ("!selected", p["surface"])])
        s.configure("Treeview", background=p["surface"], fieldbackground=p["surface"])
        s.configure("Treeview.Heading", background=p["raised"], font=self.strong)
        s.map("Treeview", background=[("selected", p["selection"])],
              foreground=[("selected", p["text"])])
        for name in ("Vertical.TScrollbar", "Horizontal.TScrollbar", "Horizontal.TScale"):
            s.configure(name, background=p["hover"], arrowcolor=p["text"])
            s.map(name, background=[("active", p["border"])])

        for option, key in (("Background", "window"), ("Foreground", "text"),
                            ("selectBackground", "selection"), ("selectForeground", "text"),
                            ("insertBackground", "text"), ("troughColor", "raised")):
            self.root.option_add(f"*{option}", p[key], "widgetDefault")
        self.root.option_add("*TCombobox*Listbox.background", p["surface"], "widgetDefault")
        self._style_widgets(self.root)
        for canvas in self.canvases:
            if canvas.get_tk_widget().winfo_exists():
                canvas.draw_idle()

    def _style_widgets(self, widget):
        p = self.palette
        if not isinstance(widget, ttk.Widget):
            if isinstance(widget, (tk.Tk, tk.Toplevel, tk.Frame, tk.Canvas, tk.Label)):
                widget.configure(background=p["window"])
            if isinstance(widget, (tk.Label, tk.Scale, tk.Button, tk.Checkbutton, tk.Text, tk.Listbox)):
                widget.configure(foreground=p["text"])
            if isinstance(widget, (tk.Text, tk.Listbox)):
                widget.configure(background=p["surface"], selectbackground=p["selection"],
                                 selectforeground=p["text"])
            if isinstance(widget, tk.Text):
                widget.configure(insertbackground=p["text"], relief="flat", padx=10, pady=5)
            if isinstance(widget, (tk.Button, tk.Checkbutton, tk.Scale)):
                widget.configure(background=p["raised"], activebackground=p["hover"])
            if isinstance(widget, tk.Scale):
                widget.configure(troughcolor=p["surface"])
            if isinstance(widget, tk.Checkbutton):
                widget.configure(selectcolor=p["selection"])
        for child in widget.winfo_children():
            self._style_widgets(child)
        if isinstance(widget, NavigationToolbar2Tk) and hasattr(widget, "_set_image_for_button"):
            for button in widget._buttons.values():
                widget._set_image_for_button(button)

    def canvas(self, figure, master):
        return _ThemedCanvas(figure, master, self)

    def style_figure(self, figure):
        p = self.palette
        figure.set_facecolor(p["window"])
        for text in figure.texts:
            text.set_color(p["text"])
        for ax in figure.axes:
            ax.set_facecolor(p["surface"])
            ax.tick_params(colors=p["muted"])
            ax.title.set_color(p["text"])
            ax.xaxis.label.set_color(p["muted"])
            ax.yaxis.label.set_color(p["muted"])
            for spine in ax.spines.values():
                spine.set_color(p["border"])
            legend = ax.get_legend()
            if legend is not None:
                legend.get_frame().set_facecolor(p["surface"])
                legend.get_frame().set_edgecolor(p["border"])
                for text in legend.get_texts():
                    text.set_color(p["text"])
