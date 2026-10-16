"""
Gimntools.GimnRec.ui.app — graphical interface (Tkinter) for PET reconstruction.

Tab layout (a single tab bar):
  - "GATE"           : everything for GATE — create/load the detector_config IN
                       THE TAB ITSELF, angles/rsectors, read the ROOT file and
                       BUILD the sinogram.
  - "PETSYS"         : everything for PETSYS — create/load the detector_config
                       in the tab itself, angles/rsectors, and two sub-tabs:
                       "Data / Sinogram" (measurement + background + corrections)
                       and "Create calibration" (builds the calibration JSON).
  - "Sinogram"       : view of the built sinogram (SHARED between GATE and
                       PETSYS).
  - "Reconstruction" : reconstructs the shared sinogram (OSEM/MLEM/FBP), shows
                       the 3 axes and allows SAVING (MHD/NRRD).

Each tab builds its own configuration — there is no shared config tab.
The sinogram is the shared point: whether GATE or PETSYS builds it, the
Reconstruction tab operates on it.

Portable: only Tkinter (stdlib) + matplotlib. On Linux the system package
`python3-tk` may be required. The first reconstruction is slower (Numba
compiles).
"""

import os
import sys
import queue
import threading
import traceback

# Bootstrap: allows `python app.py` without installing the package.
try:
    import Gimntools  # noqa: F401
except ModuleNotFoundError:
    _src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if _src not in sys.path:
        sys.path.insert(0, _src)

import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)

try:
    _ALL_CMAPS = sorted(plt.colormaps())
except Exception:
    _ALL_CMAPS = ["gray", "hot", "viridis", "jet", "magma", "inferno", "plasma", "bone"]

from Gimntools.GimnRec.data_loaders import generate_rsector_angle
from Gimntools.GimnRec.storage import Store
from Gimntools.GimnRec.filters import FBP_WINDOWS, PAD_MODES, resolve_pad_length


PROCESS_AS_OPTIONS = ["GeneralGlobalPosition", "GateGlobalPositions", "CrystalID"]
METHOD_OPTIONS = ["osem", "mlem", "fbp"]


# ===========================================================================
#  Utility widgets
# ===========================================================================

def labeled_combo(parent, label, var, values, width=20):
    row = ttk.Frame(parent); row.pack(fill=tk.X, pady=1)
    ttk.Label(row, text=label, width=16).pack(side=tk.LEFT)
    ttk.Combobox(row, textvariable=var, values=values, state="readonly",
                 width=width).pack(side=tk.LEFT, fill=tk.X, expand=True)


def labeled_entry(parent, label, var, width=10):
    row = ttk.Frame(parent); row.pack(fill=tk.X, pady=1)
    ttk.Label(row, text=label, width=16).pack(side=tk.LEFT)
    ttk.Entry(row, textvariable=var, width=width).pack(side=tk.LEFT, fill=tk.X, expand=True)


def labeled_spin(parent, label, var, lo, hi, is_double=False, width=10):
    row = ttk.Frame(parent); row.pack(fill=tk.X, pady=1)
    ttk.Label(row, text=label, width=16).pack(side=tk.LEFT)
    ttk.Spinbox(row, from_=lo, to=hi, textvariable=var, width=width,
                increment=(0.1 if is_double else 1)).pack(side=tk.LEFT)


def file_picker(parent, label, var, kind="open", patterns=(("ROOT", "*.root"),)):
    row = ttk.Frame(parent); row.pack(fill=tk.X, pady=1)

    def pick():
        if kind == "open":
            p = filedialog.askopenfilename(title=label, filetypes=list(patterns))
        else:
            p = filedialog.asksaveasfilename(title=label, filetypes=list(patterns))
        if p:
            var.set(p)
    ttk.Button(row, text=label, width=18, command=pick).pack(side=tk.LEFT)
    ttk.Label(row, textvariable=var, anchor="w", relief="sunken",
              width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)


# ===========================================================================
#  Scrollable side column (canvas + vertical scrollbar)
# ===========================================================================

class ScrollableColumn(ttk.Frame):
    """A vertical column whose content can be scrolled.

    Widgets must be packed into ``.body`` (never into the ScrollableColumn
    itself).  The canvas width follows the requested width of ``.body``
    (clamped to ``max_width``), so the column keeps the same apparent width
    as a plain ttk.Frame while the vertical scrollbar gives access to the
    widgets that fall below the bottom of the window.
    """

    #: widgets that consume the wheel themselves — do not scroll the column
    _WHEEL_CONSUMERS = (tk.Spinbox, ttk.Spinbox, ttk.Combobox, ttk.Treeview,
                        tk.Listbox, tk.Text)

    def __init__(self, master, max_width=520, **kw):
        super().__init__(master, **kw)
        self.max_width = max_width
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical",
                                 command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.body = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.body,
                                              anchor="nw")
        self.body.bind("<Configure>", self._on_body_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # wheel only while the pointer is over this column
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)

    # ---- geometry ----
    def _on_body_configure(self, _=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        w = min(self.body.winfo_reqwidth(), self.max_width)
        if w > 1 and self.canvas.winfo_width() != w:
            self.canvas.configure(width=w)

    def _on_canvas_configure(self, event):
        # let the content use the full canvas width (no horizontal gap)
        if event.width > 1:
            self.canvas.itemconfigure(self._win, width=event.width)

    # ---- mouse wheel ----
    def _bind_wheel(self, _=None):
        self.canvas.bind_all("<MouseWheel>", self._on_wheel, add="+")
        self.canvas.bind_all("<Button-4>", self._on_wheel, add="+")
        self.canvas.bind_all("<Button-5>", self._on_wheel, add="+")

    def _unbind_wheel(self, _=None):
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.unbind_all(seq)

    def _on_wheel(self, event):
        try:
            w = self.winfo_containing(event.x_root, event.y_root)
        except Exception:
            w = None
        if isinstance(w, self._WHEEL_CONSUMERS):
            return                      # spinbox/combobox/tree handle it
        if event.num == 4:              # X11
            delta = -1
        elif event.num == 5:
            delta = 1
        else:                           # Windows / macOS
            delta = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(delta, "units")


# ===========================================================================
#  Reusable image control: colormap + vmin/vmax (slice/volume/manual)
# ===========================================================================

class ImageControls(ttk.LabelFrame):
    """Controls shared by all images: colormap (all of the
    matplotlib), and vmin/vmax scaling — from the current slice, the whole
    volume, or manual (editable). on_change is called whenever something
    changes (to trigger a redraw).
    """

    def __init__(self, parent, on_change, default_cmap="gray", default_mode="volume"):
        super().__init__(parent, text="Image: colour / scale", padding=4)
        self.on_change = on_change
        self.cmap = tk.StringVar(value=default_cmap if default_cmap in _ALL_CMAPS else "gray")
        self.mode = tk.StringVar(value=default_mode)   # slice | volume | manual
        self.vmin = tk.StringVar(value="")
        self.vmax = tk.StringVar(value="")

        r1 = ttk.Frame(self); r1.pack(fill=tk.X)
        ttk.Label(r1, text="colormap").pack(side=tk.LEFT)
        cb = ttk.Combobox(r1, textvariable=self.cmap, values=_ALL_CMAPS, width=16,
                          state="readonly")
        cb.pack(side=tk.LEFT, padx=3)
        cb.bind("<<ComboboxSelected>>", lambda e: self.on_change())

        r2 = ttk.Frame(self); r2.pack(fill=tk.X, pady=1)
        ttk.Label(r2, text="escala").pack(side=tk.LEFT)
        m = ttk.Combobox(r2, textvariable=self.mode, values=["slice", "volume", "manual"],
                         width=8, state="readonly")
        m.pack(side=tk.LEFT, padx=3)
        m.bind("<<ComboboxSelected>>", lambda e: self.on_change())
        ttk.Label(r2, text="vmin").pack(side=tk.LEFT)
        e1 = ttk.Entry(r2, textvariable=self.vmin, width=8); e1.pack(side=tk.LEFT)
        ttk.Label(r2, text="vmax").pack(side=tk.LEFT)
        e2 = ttk.Entry(r2, textvariable=self.vmax, width=8); e2.pack(side=tk.LEFT)
        ttk.Button(r2, text="aplicar", command=self._manual).pack(side=tk.LEFT, padx=3)
        e1.bind("<Return>", lambda e: self._manual())
        e2.bind("<Return>", lambda e: self._manual())

    def _manual(self):
        # editing/applying values implies manual mode
        self.mode.set("manual")
        self.on_change()

    def get_cmap(self):
        return self.cmap.get()

    def clim(self, slice_arr, volume_arr, update_display=True):
        """Return (vmin, vmax) according to the mode. In slice/volume it updates
        the boxes to reflect the values used; in manual mode it reads them."""
        mode = self.mode.get()
        if mode == "manual":
            try:
                return float(self.vmin.get()), float(self.vmax.get())
            except ValueError:
                src = slice_arr  # invalid values -> fall back to the slice
        else:
            src = volume_arr if mode == "volume" else slice_arr
        arr = np.asarray(src, dtype=float)
        if arr.size == 0 or not np.isfinite(arr).any():
            lo, hi = 0.0, 1.0
        else:
            lo = float(np.nanmin(arr)); hi = float(np.nanmax(arr))
            if hi <= lo:
                hi = lo + 1e-9
        if update_display and mode != "manual":
            self.vmin.set(f"{lo:.4g}"); self.vmax.set(f"{hi:.4g}")
        return lo, hi


def bind_wheel_slices(canvas, handler):
    """Connect the mouse scroll (over the image) to the handler. Uses the
    matplotlib toolbar, which is cross-platform and reports the axis under
    the mouse."""
    canvas.mpl_connect("scroll_event", handler)


# ===========================================================================
#  ROIs: masks and NEMA metrics (square / circle / free contour)
# ===========================================================================

from math import floor, ceil
from matplotlib.path import Path as _MplPath


def calc_nema_fwhm(profile, pixel_size):
    """FWHM per NEMA NU 4-2008: peak by parabolic fit + linear interpolation
    of the half-maximum edge crossings. Returns (px, mm)."""
    profile = np.asarray(profile, dtype=float)
    if len(profile) < 3:
        return 0.0, 0.0
    max_idx = int(np.argmax(profile))
    if max_idx == 0 or max_idx == len(profile) - 1:
        true_max = profile[max_idx]
    else:
        y1, y2, y3 = profile[max_idx - 1], profile[max_idx], profile[max_idx + 1]
        c = y2; a = (y1 + y3 - 2 * y2) / 2.0; b = (y3 - y1) / 2.0
        if a < 0:
            xp = -b / (2.0 * a); true_max = a * xp * xp + b * xp + c
        else:
            true_max = y2
    hm = true_max / 2.0
    li = max_idx
    while li > 0 and profile[li] > hm:
        li -= 1
    if profile[li] <= hm < profile[li + 1]:
        y0, y1v = profile[li], profile[li + 1]
        x_left = li + (hm - y0) / (y1v - y0) if y1v != y0 else li
    else:
        x_left = li
    ri = max_idx
    while ri < len(profile) - 1 and profile[ri] > hm:
        ri += 1
    if profile[ri] <= hm < profile[ri - 1]:
        y0, y1v = profile[ri - 1], profile[ri]
        x_right = (ri - 1) + (hm - y0) / (y1v - y0) if y1v != y0 else ri
    else:
        x_right = ri
    fwhm_px = x_right - x_left
    return round(fwhm_px, 2), round(fwhm_px * pixel_size, 2)


def roi_box_mask(shape, geom, H, W):
    """Bounding box (clipped to the image) and boolean mask of the ROI.
    shape: 'square' (x0,y0,size) | 'circle' (cx,cy,r) | 'free' (points)."""
    if shape == "square":
        ix0 = int(round(geom["x0"])); iy0 = int(round(geom["y0"]))
        iw = ih = max(1, int(round(geom["size"])))
    elif shape == "circle":
        ix0 = int(round(geom["cx"] - geom["r"])); iy0 = int(round(geom["cy"] - geom["r"]))
        iw = ih = max(1, int(round(2 * geom["r"])))
    elif shape == "free":
        pts = geom["points"]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ix0 = int(floor(min(xs))); iy0 = int(floor(min(ys)))
        iw = max(1, int(ceil(max(xs))) - ix0); ih = max(1, int(ceil(max(ys))) - iy0)
    else:
        raise ValueError(f"unknown ROI shape: {shape}")
    xs0 = max(0, ix0); ys0 = max(0, iy0)
    xe = min(W, ix0 + iw); ye = min(H, iy0 + ih)
    w = xe - xs0; h = ye - ys0
    if w <= 0 or h <= 0:
        return xs0, ys0, 0, 0, None
    yy, xx = np.mgrid[ys0:ye, xs0:xe]
    if shape == "square":
        mask = np.ones((h, w), bool)
    elif shape == "circle":
        mask = ((xx - geom["cx"]) ** 2 + (yy - geom["cy"]) ** 2) <= geom["r"] ** 2
    else:
        path = _MplPath([(p[0], p[1]) for p in geom["points"]])
        pts = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5])
        mask = path.contains_points(pts).reshape(h, w)
    return xs0, ys0, w, h, mask


def roi_metrics(data, shape, geom, sp_v, sp_h):
    """Statistics + FWHM (X and Y) over the masked ROI region.
    Generalizes the rectangular case: the 1-D profile sums use only the pixels
    inside the mask (outside = 0); the statistics use only the inside pixels."""
    H, W = data.shape
    x0, y0, w, h, mask = roi_box_mask(shape, geom, H, W)
    if w <= 0 or h <= 0 or mask is None or not mask.any():
        return None
    crop = data[y0:y0 + h, x0:x0 + w].astype(float)
    vals = crop[mask]
    masked = np.where(mask, crop, 0.0)
    fx_px, fx_mm = calc_nema_fwhm(masked.sum(axis=0), sp_h)
    fy_px, fy_mm = calc_nema_fwhm(masked.sum(axis=1), sp_v)
    return {
        "FWHM_X_px": fx_px, "FWHM_X_mm": fx_mm,
        "FWHM_Y_px": fy_px, "FWHM_Y_mm": fy_mm,
        "Max": round(float(vals.max()), 2), "Min": round(float(vals.min()), 2),
        "Mean": round(float(vals.mean()), 2), "Std": round(float(vals.std()), 2),
    }


# ===========================================================================
#  detector_config form (embedded in each tab)
# ===========================================================================

_COMMON_COMBOS = [("crystal_type", ["segmented", "monolithic"]),
                  ("rotate_along", ["x", "y", "z"])]
_GATE_COMBOS = [("process_as", PROCESS_AS_OPTIONS)]
_COMMON_SCALARS = [  # (key, label, kind)
    ("pla_thickness",  "pla_thickness (mm)", "float"),
    ("radius",         "radius (mm)",        "float"),
    ("rotation_angles", "rotation_angles",   "int"),
    ("arround",        "arround (180/360)",  "float"),
    ("detNum",         "detNum",             "int"),
    ("Half_FOV",       "Half_FOV (mm)",      "float"),
    ("sino_range_mm",  "sino_range_mm (mm)", "float"),
]
_PETSYS_SCALARS = [
    ("e_min",          "e_min (keV)",        "float"),
    ("e_max",          "e_max (keV)",        "float"),
    ("det_z_rotation", "det_z_rotation (°)", "float"),
]
_CFG_ARRAYS = [
    ("crystal_size", "crystal_size (3, mm)", "float"),
    ("sipm_pitch",   "sipm_pitch (2, mm)",   "float"),
    ("sipm_pixel_n", "sipm_pixel_n (2)",     "int"),
]
_CFG_DEFAULTS = {
    "process_as": "GeneralGlobalPosition", "crystal_type": "segmented",
    "rotate_along": "z", "pla_thickness": "0.0", "radius": "53",
    "rotation_angles": "10", "arround": "180", "detNum": "12",
    "Half_FOV": "53", "sino_range_mm": "53", "e_min": "450", "e_max": "650",
    "det_z_rotation": "90", "crystal_size": "20, 26.88, 26.88",
    "sipm_pitch": "3.36, 3.36", "sipm_pixel_n": "8, 8",
}


class DetectorConfigForm(ttk.LabelFrame):
    """Embedded detector_config form. `fmt` in {'gate','petsys'} fixes the
    data_from value and which format-specific fields are shown."""

    def __init__(self, master, app, fmt):
        super().__init__(master, text=f"detector_config ({fmt.upper()})", padding=6)
        self.app = app
        self.fmt = fmt.lower()
        self.vars = {}

        combos = _COMMON_COMBOS + (_GATE_COMBOS if self.fmt == "gate" else [])
        scalars = _COMMON_SCALARS + ([] if self.fmt == "gate" else _PETSYS_SCALARS)

        form = ttk.Frame(self); form.pack(fill=tk.X)
        r = 0
        for key, values in combos:
            self.vars[key] = tk.StringVar(value=_CFG_DEFAULTS.get(key, ""))
            ttk.Label(form, text=key, width=16).grid(row=r, column=0, sticky="w", pady=1)
            ttk.Combobox(form, textvariable=self.vars[key], values=values,
                         state="readonly", width=22).grid(row=r, column=1, sticky="w")
            r += 1
        for key, label, _k in scalars:
            self.vars[key] = tk.StringVar(value=_CFG_DEFAULTS.get(key, ""))
            ttk.Label(form, text=label, width=16).grid(row=r, column=0, sticky="w", pady=1)
            ttk.Entry(form, textvariable=self.vars[key], width=24).grid(
                row=r, column=1, sticky="w")
            r += 1
        for key, label, _k in _CFG_ARRAYS:
            self.vars[key] = tk.StringVar(value=_CFG_DEFAULTS.get(key, ""))
            ttk.Label(form, text=label, width=16).grid(row=r, column=0, sticky="w", pady=1)
            ttk.Entry(form, textvariable=self.vars[key], width=24).grid(
                row=r, column=1, sticky="w")
            r += 1

        btns = ttk.Frame(self); btns.pack(fill=tk.X, pady=4)
        ttk.Button(btns, text="Load JSON…", command=self._load).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Export JSON…", command=self._save).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Validar", command=self._validate).pack(side=tk.LEFT, padx=2)

        self._scalar_kinds = {k: kind for k, _l, kind in (_COMMON_SCALARS + _PETSYS_SCALARS)}
        self._array_kinds = {k: kind for k, _l, kind in _CFG_ARRAYS}

    def get_config(self):
        cfg = {"data_from": "GATE" if self.fmt == "gate" else "PETSYS"}
        for key in self.vars:
            txt = self.vars[key].get().strip()
            if txt == "":
                continue
            if key in self._array_kinds:
                kind = self._array_kinds[key]
                parts = [p for p in txt.replace(";", ",").split(",") if p.strip() != ""]
                cfg[key] = np.asarray(
                    [int(float(p)) if kind == "int" else float(p) for p in parts])
            elif key in self._scalar_kinds:
                kind = self._scalar_kinds[key]
                cfg[key] = int(float(txt)) if kind == "int" else float(txt)
            else:  # combos (strings)
                cfg[key] = txt
        return cfg

    def _populate(self, cfg):
        for key in self.vars:
            if key not in cfg:
                continue
            val = cfg[key]
            if isinstance(val, (list, tuple, np.ndarray)):
                self.vars[key].set(", ".join(str(x) for x in np.asarray(val).tolist()))
            else:
                self.vars[key].set(str(val))

    def _validate(self):
        from Gimntools.GimnRec.data_loaders import validate_detector_config
        ws = validate_detector_config(dict(self.get_config()), self.fmt)
        if ws:
            for w in ws:
                self.app.log(f"[AVISO] {w}")
        else:
            self.app.log(f"[{self.fmt.upper()}] config valid.")

    def _load(self):
        from Gimntools.GimnRec.data_loaders import load_detector_config
        p = filedialog.askopenfilename(title="detector_config.json",
                                       filetypes=[("JSON", "*.json")])
        if not p:
            return
        try:
            cfg = load_detector_config(p)
            self._populate(cfg)
            self.app.log(f"[{self.fmt.upper()}] config loaded from {os.path.basename(p)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _save(self):
        from Gimntools.GimnRec.data_loaders import save_detector_config
        p = filedialog.asksaveasfilename(title="Exportar detector_config.json",
                                         defaultextension=".json",
                                         filetypes=[("JSON", "*.json")],
                                         initialfile="detector_config.json")
        if not p:
            return
        try:
            save_detector_config(self.get_config(), p)
            self.app.log(f"[{self.fmt.upper()}] config exportado para {os.path.basename(p)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


# ===========================================================================
#  Reusable panel: angles + rsectors (generate / load / export / view)
# ===========================================================================

class AngleRsectorPanel(ttk.LabelFrame):
    def __init__(self, master, log_fn=None, default_n=12, **kw):
        super().__init__(master, text="Angles / rsectors", padding=6, **kw)
        self._log = log_fn or (lambda m: None)

        top = ttk.Frame(self); top.pack(fill=tk.X)
        ttk.Label(top, text="N detetores").pack(side=tk.LEFT)
        self.var_n = tk.IntVar(value=default_n)
        ttk.Spinbox(top, from_=1, to=512, width=6, textvariable=self.var_n).pack(
            side=tk.LEFT, padx=4)
        ttk.Button(top, text="Generate (0-360)", command=self.generate).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Load angles…", command=self._load_angles).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Load rsectors…", command=self._load_rsectors).pack(side=tk.LEFT, padx=2)

        row2 = ttk.Frame(self); row2.pack(fill=tk.X)
        ttk.Button(row2, text="Export angles JSON…", command=self._save_angles).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Export rsectors JSON…", command=self._save_rsectors).pack(side=tk.LEFT, padx=2)

        cols = ("rsectorID", "detector", "angle_deg")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=5)
        for c, w in zip(cols, (70, 110, 90)):
            self.tree.heading(c, text=c); self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True, pady=4)
        self.tree.bind("<Double-1>", self._edit_angle)
        self.generate()

    def generate(self):
        n = int(self.var_n.get())
        angles, rsectors = generate_rsector_angle(n)
        self._fill(angles, rsectors)
        self._log(f"[angles] generated {n} detectors (0-360).")

    def _load_angles(self):
        from Gimntools.GimnRec.data_loaders import load_detectors_angles
        p = filedialog.askopenfilename(title="detectors_angles.json",
                                       filetypes=[("JSON", "*.json")])
        if not p:
            return
        try:
            angles = load_detectors_angles(p)
            rsec = self.get_rsectors()
            if set(rsec.values()) != set(angles.keys()):
                rsec = {i: name for i, name in enumerate(angles.keys())}
            self._fill(angles, rsec)
            self._log(f"[angles] loaded from {os.path.basename(p)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _load_rsectors(self):
        from Gimntools.GimnRec.data_loaders import load_rsectors
        p = filedialog.askopenfilename(title="rsectors.json",
                                       filetypes=[("JSON", "*.json")])
        if not p:
            return
        try:
            rsec = load_rsectors(p)
            angles = self.get_angles()
            for name in rsec.values():
                angles.setdefault(name, 0.0)
            self._fill(angles, rsec)
            self._log(f"[rsectors] carregados de {os.path.basename(p)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _save_angles(self):
        import json
        p = filedialog.asksaveasfilename(title="Exportar detectors_angles.json",
                                         defaultextension=".json",
                                         filetypes=[("JSON", "*.json")],
                                         initialfile="detectors_angles.json")
        if not p:
            return
        try:
            with open(p, "w") as f:
                json.dump(self.get_angles(), f, indent=2)
            self._log(f"[angles] exported to {os.path.basename(p)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _save_rsectors(self):
        import json
        p = filedialog.asksaveasfilename(title="Exportar rsectors.json",
                                         defaultextension=".json",
                                         filetypes=[("JSON", "*.json")],
                                         initialfile="rsectors.json")
        if not p:
            return
        try:
            rsec = {str(k): v for k, v in self.get_rsectors().items()}
            with open(p, "w") as f:
                json.dump(rsec, f, indent=2)
            self._log(f"[rsectors] exportados para {os.path.basename(p)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _edit_angle(self, event):
        item = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        if not item or col != "#3":
            return
        x, y, w, h = self.tree.bbox(item, col)
        ent = ttk.Entry(self.tree)
        ent.place(x=x, y=y, width=w, height=h)
        ent.insert(0, self.tree.set(item, "angle_deg")); ent.focus()

        def commit(_=None):
            try:
                float(ent.get()); self.tree.set(item, "angle_deg", ent.get())
            except ValueError:
                pass
            ent.destroy()
        ent.bind("<Return>", commit); ent.bind("<FocusOut>", commit)

    def _fill(self, angles, rsectors):
        self.tree.delete(*self.tree.get_children())
        for rid in sorted(rsectors.keys(), key=lambda k: int(k)):
            name = rsectors[rid]; ang = angles.get(name, 0.0)
            self.tree.insert("", tk.END, values=(int(rid), name, f"{float(ang):.4g}"))

    def set_data(self, angles, rsectors):
        """Set the table from dicts (used when loading a profile).
        Accepts rsectors keys as int or str (converted to int)."""
        rsec = {int(k): v for k, v in (rsectors or {}).items()}
        ang = {k: float(v) for k, v in (angles or {}).items()}
        if not rsec and ang:
            rsec = {i: name for i, name in enumerate(ang.keys())}
        self._fill(ang, rsec)

    def get_angles(self):
        out = {}
        for iid in self.tree.get_children():
            _, name, ang = self.tree.item(iid, "values")
            out[name] = float(ang)
        return out

    def get_rsectors(self):
        out = {}
        for iid in self.tree.get_children():
            rid, name, _ = self.tree.item(iid, "values")
            out[int(rid)] = name
        return out


# ===========================================================================
#  GATE tab
# ===========================================================================

class GatePanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=6)
        self.app = app

        self.cfg_form = DetectorConfigForm(self, app, "gate")
        self.cfg_form.pack(fill=tk.X, pady=3)

        box_files = ttk.LabelFrame(self, text="GATE data", padding=6)
        box_files.pack(fill=tk.X, pady=3)
        self.var_root = tk.StringVar()
        file_picker(box_files, "ROOT GATE", self.var_root)
        self.var_interp = tk.BooleanVar(value=False)
        self.var_gnorm = tk.BooleanVar(value=False)
        self.var_ndist = tk.IntVar(value=128)
        self.var_asigma = tk.DoubleVar(value=0.5)
        ttk.Checkbutton(box_files, text="Interpolar radialmente (crystalID)",
                        variable=self.var_interp).pack(anchor="w", pady=2)
        ttk.Checkbutton(box_files, text="Geometric norm (dual-head sensitivity)",
                        variable=self.var_gnorm).pack(anchor="w", pady=2)
        labeled_spin(box_files, "n_dist_out", self.var_ndist, 16, 1024)
        labeled_spin(box_files, "angular_sigma", self.var_asigma, 0.0, 10.0, is_double=True)

        self.angles = AngleRsectorPanel(self, log_fn=app.log, default_n=12)
        self.angles.pack(fill=tk.BOTH, expand=True, pady=3)

        rown = ttk.Frame(self); rown.pack(fill=tk.X)
        ttk.Label(rown, text="sinogram name", width=16).pack(side=tk.LEFT)
        self.var_name = tk.StringVar(value="")
        ttk.Entry(rown, textvariable=self.var_name, width=24).pack(side=tk.LEFT)
        ttk.Button(self, text="Build sinogram",
                   command=self._build_async).pack(fill=tk.X, pady=4)

    def _build_async(self):
        if not self.var_root.get():
            messagebox.showwarning("No ROOT", "Select a GATE ROOT file.")
            return
        try:
            spec = dict(path=self.var_root.get(), cfg=self.cfg_form.get_config(),
                        angles=self.angles.get_angles(),
                        rsec=self.angles.get_rsectors(),
                        interpolate=self.var_interp.get(),
                        n_dist=self.var_ndist.get(),
                        angular_sigma=self.var_asigma.get(),
                        use_norm=self.var_gnorm.get(),
                        name=self.var_name.get())
            self.app.start_worker("build sinogram", self._build_worker, spec)
        except Exception:
            self.app.error(traceback.format_exc())

    def _build_worker(self, spec):
        try:
            from Gimntools.GimnRec.utils import coincidence_to_lor
            from Gimntools.GimnRec.filters import interpolate_distance
            from Gimntools.GimnRec.data_loaders import (validate_detector_config,
                                                        get_default_parameters)
            cfg = spec['cfg']; angles = spec['angles']; rsec = spec['rsec']
            for w in validate_detector_config(dict(cfg), "gate"):
                self.app.log(f"[WARNING] {w}")
            sino, _ = coincidence_to_lor(spec['path'], cfg, angles,
                                         get_default_parameters("gate"),
                                         rsectors=rsec, verbose=True)
            if spec['interpolate']:
                sino = interpolate_distance(sino, n_out=spec['n_dist'],
                                            kind="conservative")
                self.app.log("[GATE] count-conserving rebin (correlated bins)")
            if spec['angular_sigma'] > 0:
                self.app.log("[GATE] angular smoothing disabled for EM count histograms; "
                             "use FBP preprocessing if required")
            norm = None
            if spec['use_norm']:
                from Gimntools.GimnRec.normalization import geometric_norm_sinogram
                norm = geometric_norm_sinogram(cfg, angles,
                                               sino.shape[1], sino.shape[2])
            fov = 2.0 * cfg["Half_FOV"]
            srange = cfg.get("sino_range_mm", float(cfg["crystal_size"][1]))
            nm = spec['name'].strip() or os.path.basename(spec['path']).replace('.root','')
            self.app.set_sinogram(sino, fov, srange, source="GATE",
                                  detector_config=cfg, name=nm, norm=norm)
        except Exception:
            self.app.error(traceback.format_exc())


# ===========================================================================
#  PETSYS tab (own config + data/sinogram + create calibration)
# ===========================================================================

class PetsysPanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=6)
        self.app = app
        self.calibration = None

        self.cfg_form = DetectorConfigForm(self, app, "petsys")
        self.cfg_form.pack(fill=tk.X, pady=3)

        self.angles = AngleRsectorPanel(self, log_fn=app.log, default_n=12)
        self.angles.pack(fill=tk.X, pady=3)

        sub = ttk.Notebook(self); sub.pack(fill=tk.BOTH, expand=True, pady=3)
        self.tab_data = ttk.Frame(sub, padding=4)
        self.tab_cal = ttk.Frame(sub, padding=4)
        sub.add(self.tab_data, text="Data / Sinogram")
        sub.add(self.tab_cal, text="Create calibration")
        self._build_data_tab()
        self._build_cal_tab()

    # ---- sub-tab: data / sinogram ----
    def _build_data_tab(self):
        t = self.tab_data
        self.var_meas = tk.StringVar(); self.var_bg = tk.StringVar(); self.var_cal = tk.StringVar()
        file_picker(t, "measurement ROOT", self.var_meas)
        file_picker(t, "ROOT background", self.var_bg)
        file_picker(t, "calibration (opt.)", self.var_cal, patterns=(("JSON", "*.json"),))
        self.var_apply_corr = tk.BooleanVar(value=False)
        ttk.Checkbutton(t, text="Apply corrections (COR + encoder offset)",
                        variable=self.var_apply_corr).pack(anchor="w", pady=2)
        self.var_gnorm = tk.BooleanVar(value=False)
        ttk.Checkbutton(t, text="Geometric norm (dual-head sensitivity; prefer a "
                                "measured flood when available)",
                        variable=self.var_gnorm).pack(anchor="w", pady=2)
        self.var_bg_scale = tk.DoubleVar(value=1.0)
        labeled_entry(t, "background scale (Tprompt/Tbg)", self.var_bg_scale, width=12)
        self.var_ndist = tk.IntVar(value=128)
        self.var_asigma = tk.DoubleVar(value=0.5)
        labeled_spin(t, "n_dist_out", self.var_ndist, 16, 1024)
        labeled_spin(t, "angular_sigma", self.var_asigma, 0.0, 10.0, is_double=True)
        rown = ttk.Frame(t); rown.pack(fill=tk.X)
        ttk.Label(rown, text="sinogram name", width=16).pack(side=tk.LEFT)
        self.var_name = tk.StringVar(value="")
        ttk.Entry(rown, textvariable=self.var_name, width=24).pack(side=tk.LEFT)
        ttk.Button(t, text="Build sinogram",
                   command=self._build_async).pack(fill=tk.X, pady=4)

    def _build_async(self):
        if not self.var_meas.get():
            messagebox.showwarning("No ROOT", "Select the measurement ROOT.")
            return
        try:
            spec = dict(path=self.var_meas.get(), bg=self.var_bg.get(),
                        bg_scale=float(self.var_bg_scale.get()),
                        cfg=self.cfg_form.get_config(),
                        angles_cfg=self.angles.get_angles(),
                        rsec=self.angles.get_rsectors(),
                        cal_file=self.var_cal.get(), calibration=self.calibration,
                        apply_corr=self.var_apply_corr.get(),
                        use_norm=self.var_gnorm.get(),
                        n_dist=self.var_ndist.get(),
                        angular_sigma=self.var_asigma.get(),
                        name=self.var_name.get())
            if not np.isfinite(spec['bg_scale']) or spec['bg_scale'] < 0:
                raise ValueError("background scale (Tprompt/Tbg) must be finite and nonnegative")
            if spec['bg_scale'] != 1:
                self.app.log(f"[PETSYS] scaling background by {spec['bg_scale']:.6g}")
            self.app.start_worker("build sinogram", self._build_worker, spec)
        except Exception:
            self.app.error(traceback.format_exc())

    def _build_worker(self, spec):
        try:
            from Gimntools.GimnRec.utils import coincidence_to_lor
            from Gimntools.GimnRec.data_loaders import (validate_detector_config,
                                                        get_default_parameters)
            cfg = spec['cfg']; angles_cfg = spec['angles_cfg']; rsec = spec['rsec']
            for w in validate_detector_config(dict(cfg), "petsys"):
                self.app.log(f"[WARNING] {w}")
            params = get_default_parameters("petsys")
            sino, _ = coincidence_to_lor(spec['path'], cfg, angles_cfg, params,
                                         rsectors=rsec, verbose=True)
            background = None
            if spec['bg']:
                if not os.path.isfile(spec['bg']):
                    raise FileNotFoundError(f"Background ROOT not found: {spec['bg']}")
                bg, _ = coincidence_to_lor(spec['bg'], cfg, angles_cfg, params,
                                            rsectors=rsec)
                if bg.shape != sino.shape:
                    raise ValueError("background and prompt sinograms have different dimensions")
                background = np.asarray(bg, float) * spec['bg_scale']
            sino = np.asarray(sino, float)
            if not np.isfinite(sino).all() or (sino < 0).any() or (background is not None and
                    (not np.isfinite(background).all() or (background < 0).any())):
                raise ValueError("invalid prompt or background counts")
            self.app.log(f"[PETSYS] prompts {sino.sum():.0f}; "
                         f"scaled background {0 if background is None else background.sum():.1f}")
            if spec['angular_sigma'] > 0:
                self.app.log("[PETSYS] angular smoothing disabled for Poisson EM")
            angles = (np.arange(sino.shape[2]) + 0.5) * (180.0 / sino.shape[2])
            dx = dy = 0.0
            if spec['apply_corr']:
                cal = spec['calibration']
                if cal is None and spec['cal_file']:
                    from Gimntools.GimnRec.calibration import load_calibration
                    cal = load_calibration(spec['cal_file'])
                if cal is None:
                    raise ValueError("Corrections enabled without a valid calibration")
                dx = float(cal.get('dx_cor_mm', 0.0))
                dy = float(cal.get('dy_cor_mm', 0.0))
                phi = float(cal.get('phi_offset_deg', 0.0))
                if not np.isfinite([dx, dy, phi]).all():
                    raise ValueError("invalid calibration offsets")
                angles = angles + phi
                self.app.log(f"[PETSYS] geometry calibration dx={dx:.3f} "
                             f"dy={dy:.3f} phi={phi:.3f}° (raw bins unchanged)")
            norm = None
            if spec['use_norm']:
                from Gimntools.GimnRec.normalization import geometric_norm_sinogram
                norm = geometric_norm_sinogram(cfg, angles_cfg,
                                               sino.shape[1], sino.shape[2])
            fov = 2.0 * cfg["Half_FOV"]
            srange = cfg.get("sino_range_mm", float(cfg["crystal_size"][1]))
            nm = spec['name'].strip() or os.path.basename(spec['path']).replace('.root','')
            self.app.set_sinogram(sino, fov, srange, source="PETSYS",
                                  detector_config=cfg, angles_override=angles, name=nm,
                                  norm=norm, background=background,
                                  dx_cor_mm=dx, dy_cor_mm=dy)
        except Exception:
            self.app.error(traceback.format_exc())

    def _build_cal_tab(self):
        t = self.tab_cal
        ttk.Label(t, text="Each row = acquisition with source(s) of KNOWN position "
                  "(mm). 'sources': Sx,Sy pairs separated by ';' (e.g. '0,0' centred | "
                  "'7.5,0' off-centre | '7.5,0;-7.5,0' two sources).",
                  wraplength=440, foreground="#444").pack(anchor="w", pady=2)
        self.cal_bg = tk.StringVar()
        file_picker(t, "ROOT background", self.cal_bg)

        cols = ("label", "root", "sources", "I")
        self.cal_tree = ttk.Treeview(t, columns=cols, show="headings", height=5)
        for c, w in zip(cols, (100, 200, 140, 50)):
            self.cal_tree.heading(c, text=c); self.cal_tree.column(c, width=w, anchor="w")
        self.cal_tree.pack(fill=tk.BOTH, expand=True, pady=4)

        b = ttk.Frame(t); b.pack(fill=tk.X)
        ttk.Button(b, text="+ Centred source",
                   command=lambda: self._add_row("centre", "0,0", "1")).pack(side=tk.LEFT)
        ttk.Button(b, text="+ Off-centre source",
                   command=lambda: self._add_row("off_centre", "7.5,0", "1")).pack(side=tk.LEFT)
        ttk.Button(b, text="+ 2 sources",
                   command=lambda: self._add_row("2sources", "7.5,0;-7.5,0", "1;1")).pack(side=tk.LEFT)
        ttk.Button(b, text="Set row ROOT", command=self._set_row_root).pack(side=tk.LEFT, padx=4)
        ttk.Button(b, text="Remove row", command=self._remove_row).pack(side=tk.LEFT)

        bo = ttk.Frame(t); bo.pack(fill=tk.X, pady=4)
        self.cal_out = tk.StringVar(value="calibration.json")
        file_picker(bo, "Save to", self.cal_out, kind="save", patterns=(("JSON", "*.json"),))
        ttk.Button(t, text="Generate calibration (joint fit)",
                   command=self._run_cal_async).pack(fill=tk.X, pady=3)

    def _add_row(self, label, sources, I):
        self.cal_tree.insert("", tk.END, values=(label, "(choose ROOT)", sources, I))

    def _set_row_root(self):
        sel = self.cal_tree.selection()
        if not sel:
            messagebox.showinfo("Select", "Select a row first.")
            return
        p = filedialog.askopenfilename(title="Acquisition ROOT", filetypes=[("ROOT", "*.root")])
        if p:
            self.cal_tree.set(sel[0], "root", p)

    def _remove_row(self):
        for s in self.cal_tree.selection():
            self.cal_tree.delete(s)

    def _parse_sources(self, text):
        out = []
        for pair in text.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            sx, sy = pair.split(","); out.append((float(sx), float(sy)))
        return out

    def _run_cal_async(self):
        if not self.cal_tree.get_children():
            messagebox.showwarning("No acquisitions", "Add at least one row.")
            return
        try:
            cfg = self.cfg_form.get_config()
            specs = []
            for iid in self.cal_tree.get_children():
                label, root, sources, I = self.cal_tree.item(iid, "values")
                if not os.path.exists(root):
                    self.app.log(f"[CAL] skipping missing ROOT: {root}")
                    continue
                srcs = self._parse_sources(sources)
                Is = [float(x) for x in str(I).split(';') if x.strip()]
                if len(Is) != len(srcs):
                    Is = [1.0] * len(srcs)
                specs.append({'path':root,'sources':srcs,'I':Is,'label':label,
                              'n_gantry_angles':int(cfg.get('rotation_angles',12))})
            if not specs:
                raise ValueError("No valid calibration acquisition")
            options=dict(specs=specs, cfg=cfg,
                         angles=self.angles.get_angles(),
                         rsec=self.angles.get_rsectors(),
                         bg=self.cal_bg.get() or None,
                         n_dist=self.var_ndist.get(),
                         sigma=self.var_asigma.get(),
                         output=self.cal_out.get() or 'calibration.json')
            self.app.start_worker('calibration', self._cal_worker, options)
        except Exception:
            self.app.error(traceback.format_exc())

    def _cal_worker(self, options):
        try:
            from Gimntools.GimnRec.calibration import joint_calibration, save_calibration
            from Gimntools.GimnRec.utils import coincidence_to_lor
            from Gimntools.GimnRec.filters import interpolate_distance, smooth_angular
            from Gimntools.GimnRec.data_loaders import get_default_parameters
            par, _report = joint_calibration(
                options['specs'], options['bg'], options['cfg'],
                options['angles'], get_default_parameters('petsys'), options['rsec'],
                coincidence_to_lor_fn=coincidence_to_lor,
                interpolate_distance_fn=interpolate_distance,
                smooth_angular_fn=smooth_angular,
                n_dist_out=options['n_dist'],
                angular_sigma=options['sigma'], verbose=True)
            save_calibration(par, options['output'])
            self.app._q.put(('calibration_ready', (par, options['output'])))
        except Exception:
            self.app.error(traceback.format_exc())


# ===========================================================================
#  Sinogram tab (SHARED)
# ===========================================================================

class SinogramPanel(ttk.Frame):
    """Sinogram viewer with an interactive centre-of-rotation (COR) line.

    The COR is a CONSTANT radial offset of every angular column, so in this
    view it is a horizontal line. It is stored per sinogram and consumed by
    the reconstructor as ``cor_mm`` — applied to the projector geometry, so
    the sinogram itself is never shifted or resampled.
    """

    def __init__(self, master, app):
        super().__init__(master, padding=6)
        self.app = app
        self._cor_line = None
        self._axis_line = None
        self._dragging = False
        self.fig = Figure(figsize=(7, 5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, self)
        bind_wheel_slices(self.canvas, self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        c = ttk.Frame(self); c.pack(fill=tk.X)
        ttk.Label(c, text="slice (or scroll over the image)").pack(side=tk.LEFT)
        self.scale = ttk.Scale(c, from_=0, to=0, orient=tk.HORIZONTAL,
                               command=lambda e: self.draw())
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.ctrl = ImageControls(self, self.draw, default_cmap="hot", default_mode="volume")
        self.ctrl.pack(fill=tk.X, pady=2)

        # ---- centre of rotation --------------------------------------
        cor = ttk.LabelFrame(self, text="Centre of rotation (constant offset)",
                             padding=6)
        cor.pack(fill=tk.X, pady=2)
        row = ttk.Frame(cor); row.pack(fill=tk.X)
        ttk.Label(row, text="COR (mm)", width=10).pack(side=tk.LEFT)
        self.var_cor = tk.DoubleVar(value=0.0)
        self.var_step = tk.DoubleVar(value=0.10)
        self.spin_cor = ttk.Spinbox(row, from_=-1e4, to=1e4, width=10,
                                    textvariable=self.var_cor,
                                    increment=0.1, command=self._on_cor_changed)
        self.spin_cor.pack(side=tk.LEFT)
        ttk.Button(row, text="−", width=3,
                   command=lambda: self._nudge(-1)).pack(side=tk.LEFT, padx=1)
        ttk.Button(row, text="+", width=3,
                   command=lambda: self._nudge(+1)).pack(side=tk.LEFT, padx=1)
        ttk.Label(row, text="step").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Spinbox(row, from_=0.001, to=10.0, width=6, increment=0.05,
                    textvariable=self.var_step).pack(side=tk.LEFT)
        self.var_cor.trace_add("write", lambda *a: self._on_cor_changed())

        row2 = ttk.Frame(cor); row2.pack(fill=tk.X, pady=2)
        self.var_click = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, text="click/drag on the image to set",
                        variable=self.var_click).pack(side=tk.LEFT)
        ttk.Button(row2, text="Estimate from data",
                   command=self._estimate).pack(side=tk.LEFT, padx=4)
        ttk.Button(row2, text="Reset to 0",
                   command=lambda: self.var_cor.set(0.0)).pack(side=tk.LEFT)
        self.cor_info = ttk.Label(cor, foreground="#666", wraplength=620,
                                  text="0.000 mm — the line marks where the "
                                       "rotation axis sits on the radial axis. "
                                       "It is applied to the projector "
                                       "geometry, not by shifting the "
                                       "sinogram.")
        self.cor_info.pack(anchor="w")

        self.info = ttk.Label(self, text="(no sinogram yet)")
        self.info.pack(anchor="w")

    # ---- COR helpers ---------------------------------------------------
    def _bin_mm(self):
        if self.app.sino is None or not self.app.sino_range_mm:
            return None
        return float(self.app.sino_range_mm) / self.app.sino.shape[1]

    def _nudge(self, sign):
        try:
            step = float(self.var_step.get())
        except (tk.TclError, ValueError):
            step = 0.1
        self.var_cor.set(round(float(self.var_cor.get()) + sign * step, 6))

    def _on_cor_changed(self):
        """Store the COR on the current sinogram entry and move the line."""
        try:
            cor = float(self.var_cor.get())
        except (tk.TclError, ValueError):
            return
        entry = self.app.get_sinogram(self.app.current_sino_name)
        if entry is not None:
            entry["cor_mm"] = cor
        self.app.cor_mm = cor
        bw = self._bin_mm()
        txt = f"{cor:+.4f} mm"
        if bw:
            txt += f"  ({cor/bw:+.3f} radial bins of {bw:.4f} mm)"
        txt += ("  — applied to the projector geometry (no resampling of "
                "the counts).")
        self.cor_info.configure(text=txt)
        # keep the Reconstruction tab in sync
        try:
            self.app.recon_panel.v["cor_mm"].set(cor)
        except Exception:
            pass
        self._update_cor_line()

    def _update_cor_line(self):
        if self.app.sino is None:
            return
        try:
            cor = float(self.var_cor.get())
        except (tk.TclError, ValueError):
            return
        if self._axis_line is None or self._axis_line.axes is not self.ax:
            self._axis_line = self.ax.axhline(0.0, color="#7fd1ff", lw=0.8,
                                              ls=":", alpha=0.8)
        if self._cor_line is None or self._cor_line.axes is not self.ax:
            self._cor_line = self.ax.axhline(cor, color="#00ff66", lw=1.4,
                                             ls="--")
        else:
            self._cor_line.set_ydata([cor, cor])
        self.canvas.draw_idle()

    def _on_click(self, event):
        if (not self.var_click.get() or event.inaxes is not self.ax
                or event.ydata is None or event.button != 1):
            return
        # do not fight the matplotlib pan/zoom tools
        if getattr(self.canvas, "toolbar", None) is not None \
                and getattr(self.canvas.toolbar, "mode", ""):
            return
        self._dragging = True
        self.var_cor.set(round(float(event.ydata), 6))

    def _on_motion(self, event):
        if (self._dragging and event.inaxes is self.ax
                and event.ydata is not None):
            self.var_cor.set(round(float(event.ydata), 6))

    def _on_release(self, event):
        self._dragging = False

    def _estimate(self):
        if self.app.sino is None:
            messagebox.showwarning("No sinogram", "Build a sinogram first.")
            return
        try:
            from Gimntools.GimnRec.corrections import estimate_cor_offset
            n_dist = self.app.sino.shape[1]
            srange = float(self.app.sino_range_mm or n_dist)
            bw = srange / n_dist
            bins = np.linspace(-srange/2 + bw/2, srange/2 - bw/2, n_dist)
            z = int(float(self.scale.get()))
            d, info = estimate_cor_offset(self.app.sino, bins,
                                          self.app.angles, slice_idx=z)
            self.var_cor.set(round(float(d), 6))
            amp = info.get("sinusoid_amplitude_mm", float("nan"))
            self.app.log(f"[COR] estimate {d:+.4f} mm from slice {z} "
                         f"({info['n_used']} columns, object sinusoid "
                         f"amplitude {amp:.4f} mm)")
            if np.isfinite(amp) and amp > 2.0 * abs(d) and amp > 1e-3:
                messagebox.showwarning(
                    "COR estimate unreliable",
                    "The angle-dependent (object off-centre) term dominates "
                    "the constant term.\n\nA constant COR cannot correct "
                    "that: either the object is not centred on the axis, or "
                    "the misalignment is a translation rather than an axis "
                    "offset. Treat this number as unvalidated.")
        except Exception:
            self.app.error(traceback.format_exc())

    def _on_scroll(self, event):
        if self.app.sino is None:
            return
        nz = self.app.sino.shape[0]
        cur = int(float(self.scale.get()))
        new = min(max(cur + (1 if event.button == "up" else -1), 0), nz - 1)
        self.scale.set(new); self.draw()

    def on_new(self):
        nz = self.app.sino.shape[0]
        self.scale.configure(to=max(0, nz - 1)); self.scale.set(nz // 2)
        entry = self.app.get_sinogram(self.app.current_sino_name)
        self.var_cor.set(float((entry or {}).get("cor_mm", 0.0)))
        bw = self._bin_mm()
        if bw:
            self.spin_cor.configure(increment=round(bw, 6))
            self.var_step.set(round(bw, 6))
        self.info.configure(text=f"Source: {self.app.last_source} · "
                            f"shape {self.app.sino.shape} · "
                            f"counts {self.app.sino.sum():.0f}")
        self.draw()

    def draw(self):
        if self.app.sino is None:
            return
        z = min(max(int(float(self.scale.get())), 0), self.app.sino.shape[0] - 1)
        img = self.app.sino[z]
        vmin, vmax = self.ctrl.clim(img, self.app.sino)
        self.ax.clear()
        self._cor_line = None
        self._axis_line = None
        r = self.app.sino_range_mm or 1.0
        self.ax.imshow(img, aspect="auto", origin="lower", cmap=self.ctrl.get_cmap(),
                       extent=[0, 180, -r/2, r/2], vmin=vmin, vmax=vmax)
        self.ax.set_xlabel("Angle (deg)"); self.ax.set_ylabel("Distance (mm)")
        self.ax.set_title(f"Sinogram ({self.app.last_source}) — slice {z}")
        self._update_cor_line()
        self.canvas.draw_idle()


# ===========================================================================
#  Reconstruction tab (+ save) — operates on the shared sinogram
# ===========================================================================

class ReconstructionPanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=6)
        self.app = app
        self.current_recon = None   # entry atualmente mostrado

        # Left column inside a scrollable canvas: the control stack (parameters,
        # image grid, COR, reconstruction list, "Save volume") is taller than
        # the window on small screens, which used to hide the Save button.
        self.ctrl_scroll = ScrollableColumn(self)
        self.ctrl_scroll.pack(side=tk.LEFT, fill=tk.Y, padx=4)
        ctrl = self.ctrl_scroll.body

        selbox = ttk.LabelFrame(ctrl, text="Sinogram to reconstruct", padding=6)
        selbox.pack(fill=tk.X)
        self.var_sino = tk.StringVar()
        self.sino_combo = ttk.Combobox(selbox, textvariable=self.var_sino,
                                       state="readonly", width=24)
        self.sino_combo.pack(fill=tk.X)
        self.sino_combo.bind("<<ComboboxSelected>>",
                             lambda e: self._sync_cor_from_sino())

        box = ttk.LabelFrame(ctrl, text="Parameters", padding=6); box.pack(fill=tk.X)
        self.v = {
            "method": tk.StringVar(value="osem"), "iters": tk.IntVar(value=10),
            "subsets": tk.IntVar(value=4), "gevery": tk.IntVar(value=5),
            "gsigma": tk.DoubleVar(value=0.4), "use_psf": tk.BooleanVar(value=False),
            "fwhm": tk.DoubleVar(value=3.0), "legacy": tk.BooleanVar(value=False),
            # --- image grid (volume-driven reconstruction) ---
            "grid_mode": tk.StringVar(value="sinogram"),
            "n_img": tk.IntVar(value=128),
            "voxel_mm": tk.DoubleVar(value=0.0),
            "rays_auto": tk.BooleanVar(value=True),
            "rays_per_bin": tk.IntVar(value=1),
            "bp_chunks": tk.IntVar(value=1),
            # --- centre of rotation (constant radial offset) ---
            "cor_mm": tk.DoubleVar(value=0.0),
            # --- FBP filter / padding / radial interpolation ---
            "fbp_window": tk.StringVar(value="ramp"),
            "fbp_cutoff": tk.DoubleVar(value=1.0),
            "fbp_order": tk.IntVar(value=5),
            "fbp_hard_cut": tk.BooleanVar(value=True),
            "fbp_pad_mode": tk.StringVar(value="pow2"),
            "fbp_pad_factor": tk.IntVar(value=2),
            "fbp_interp": tk.StringVar(value="none"),
            "fbp_interp_factor": tk.IntVar(value=2),
            "fbp_legacy_scale": tk.BooleanVar(value=False),
        }
        labeled_combo(box, "method", self.v["method"], METHOD_OPTIONS)
        labeled_spin(box, "iterations", self.v["iters"], 1, 500)
        labeled_spin(box, "subsets (OSEM)", self.v["subsets"], 1, 64)
        labeled_spin(box, "gauss_every", self.v["gevery"], 0, 100)
        labeled_spin(box, "gauss_sigma", self.v["gsigma"], 0.0, 10.0, is_double=True)
        labeled_spin(box, "PSF FWHM (mm)", self.v["fwhm"], 0.0, 30.0, is_double=True)
        ttk.Checkbutton(box, text="Use PSF (radial, in the model)",
                        variable=self.v["use_psf"]).pack(anchor="w")
        ttk.Checkbutton(box, text="Legacy EM (old behaviour)",
                        variable=self.v["legacy"]).pack(anchor="w")
        self.v["zoom"] = tk.IntVar(value=1)
        labeled_spin(box, "Z upsample (visual)", self.v["zoom"], 1, 12)

        # ---- image grid: sinogram-driven (legacy) vs volume-driven --------
        gbox = ttk.LabelFrame(ctrl, text="Image grid", padding=6)
        gbox.pack(fill=tk.X, pady=2)
        labeled_combo(gbox, "mode", self.v["grid_mode"],
                      ["sinogram", "matrix", "voxel", "matrix+voxel"])
        labeled_spin(gbox, "n_img (px)", self.v["n_img"], 8, 2048)
        labeled_spin(gbox, "voxel (mm)", self.v["voxel_mm"], 0.0, 50.0,
                     is_double=True)
        ttk.Checkbutton(gbox, text="rays/bin automatic",
                        variable=self.v["rays_auto"]).pack(anchor="w")
        labeled_spin(gbox, "rays/bin (manual)", self.v["rays_per_bin"], 1, 32)
        labeled_spin(gbox, "backproj. chunks", self.v["bp_chunks"], 1, 32)

        # ---- FBP-only options: filter, cutoff, padding, interpolation -----
        fbox = ttk.LabelFrame(ctrl, text="FBP filter (method = fbp only)",
                              padding=6)
        fbox.pack(fill=tk.X, pady=2)
        labeled_combo(fbox, "window", self.v["fbp_window"], list(FBP_WINDOWS),
                      width=14)
        labeled_spin(fbox, "cutoff (×Nyquist)", self.v["fbp_cutoff"],
                     0.05, 1.0, is_double=True, width=8)
        self.lbl_cutoff = ttk.Label(fbox, foreground="#666", text="")
        self.lbl_cutoff.pack(anchor="w")
        labeled_spin(fbox, "Butterworth order", self.v["fbp_order"], 1, 20,
                     width=8)
        ttk.Checkbutton(fbox, text="rescale window onto the cutoff",
                        variable=self.v["fbp_hard_cut"]).pack(anchor="w")
        labeled_combo(fbox, "zero-padding", self.v["fbp_pad_mode"],
                      list(PAD_MODES), width=14)
        labeled_spin(fbox, "padding factor", self.v["fbp_pad_factor"], 1, 8,
                     width=8)
        labeled_combo(fbox, "radial interp.", self.v["fbp_interp"],
                      ["none", "linear", "cubic", "conservative"], width=14)
        labeled_spin(fbox, "interp. factor", self.v["fbp_interp_factor"],
                     1, 8, width=8)
        ttk.Checkbutton(fbox, text="legacy FBP scale (pre-0.4.1)",
                        variable=self.v["fbp_legacy_scale"]).pack(anchor="w")
        ttk.Label(fbox, wraplength=210, foreground="#666",
                  text="cutoff is a fraction of the RADIAL Nyquist "
                       "(0.5 cycles/bin); the mm⁻¹ value depends on the bin "
                       "size and is printed in the log. Lowering it trades "
                       "spatial resolution for noise — it is not a "
                       "correction. Butterworth ignores the rescale box: its "
                       "cutoff is the −3 dB point, not a hard limit. "
                       "'radial interp.' resamples the PROJECTIONS before "
                       "back-projection; this is not the same as an "
                       "interpolating back-projector — the projector here is "
                       "a Siddon ray tracer whose sub-sampling is set by "
                       "'rays/bin'. In grid mode 'sinogram' it also changes "
                       "the output matrix size.").pack(anchor="w", pady=2)
        for _k in ("fbp_window", "fbp_cutoff", "fbp_interp",
                   "fbp_interp_factor"):
            self.v[_k].trace_add("write", lambda *_a: self._update_fbp_hint())

        cbox = ttk.LabelFrame(ctrl, text="Centre of rotation", padding=6)
        cbox.pack(fill=tk.X, pady=2)
        crow = ttk.Frame(cbox); crow.pack(fill=tk.X)
        ttk.Label(crow, text="COR (mm)", width=10).pack(side=tk.LEFT)
        ttk.Spinbox(crow, from_=-1e4, to=1e4, width=9, increment=0.05,
                    textvariable=self.v["cor_mm"]).pack(side=tk.LEFT)
        ttk.Button(crow, text="edit on sinogram",
                   command=self._goto_sinogram).pack(side=tk.LEFT, padx=3)
        ttk.Label(cbox, wraplength=210, foreground="#666",
                  text="Constant radial offset of the rotation axis. Set it "
                       "here or by clicking/dragging the green line in the "
                       "Sinogram tab; the two stay in sync. Applied to the "
                       "projector geometry — the sinogram is never "
                       "resampled.").pack(anchor="w", pady=2)
        ttk.Label(gbox, wraplength=210, foreground="#666",
                  text="sinogram = legacy (matrix follows the radial bins). "
                       "matrix/voxel = reconstruct on the grid you ask for, "
                       "from the measured bins. matrix+voxel also redefines "
                       "the FOV as n_img*voxel.").pack(anchor="w", pady=2)

        rn = ttk.Frame(ctrl); rn.pack(fill=tk.X, pady=2)
        ttk.Label(rn, text="recon name", width=10).pack(side=tk.LEFT)
        self.var_recon_name = tk.StringVar(value="")
        ttk.Entry(rn, textvariable=self.var_recon_name, width=18).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Reconstruct",
                   command=self._recon_async).pack(fill=tk.X, pady=4)

        regbox = ttk.LabelFrame(ctrl, text="Reconstructions", padding=4)
        regbox.pack(fill=tk.X, pady=2)
        cols = ("name", "shape", "spacing(mm)", "source")
        self.recon_tree = ttk.Treeview(regbox, columns=cols, show="headings", height=8)
        for c, w in zip(cols, (110, 78, 92, 64)):
            self.recon_tree.heading(c, text=c); self.recon_tree.column(c, width=w, anchor="w")
        self.recon_tree.pack(fill=tk.BOTH, expand=True)
        self.recon_tree.bind("<<TreeviewSelect>>", lambda e: self._show_selected())
        rb = ttk.Frame(regbox); rb.pack(fill=tk.X, pady=2)
        ttk.Button(rb, text="+ Add MHD", command=self._add_mhd).pack(side=tk.LEFT, padx=2)
        ttk.Button(rb, text="− Remove", command=self._remove).pack(side=tk.LEFT, padx=2)

        box_exp = ttk.LabelFrame(ctrl, text="Save volume", padding=6)
        box_exp.pack(fill=tk.X, pady=(2, 8))
        self.var_fmt = tk.StringVar(value="mhd")
        labeled_combo(box_exp, "formato", self.var_fmt, ["mhd", "nrrd"])
        ttk.Button(box_exp, text="Save…", command=self._save).pack(fill=tk.X, pady=2)

        view = ttk.Frame(self); view.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(9, 5))
        self.ax_ax = self.fig.add_subplot(131)
        self.ax_co = self.fig.add_subplot(132)
        self.ax_sa = self.fig.add_subplot(133)
        self.canvas = FigureCanvasTkAgg(self.fig, master=view)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, view)
        bind_wheel_slices(self.canvas, self._on_scroll)
        self.ctrl = ImageControls(view, self.draw, default_cmap="gray", default_mode="volume")
        self.ctrl.pack(fill=tk.X, pady=2)
        cc = ttk.Frame(view); cc.pack(fill=tk.X)
        ttk.Label(view, text="(mouse scroll over each view changes the slice)",
                  foreground="#666").pack(anchor="w")
        self.scales = {}
        for key, lbl in [("z", "Z axial"), ("y", "Y coronal"), ("x", "X sagital")]:
            f = ttk.Frame(cc); f.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
            ttk.Label(f, text=lbl).pack()
            s = ttk.Scale(f, from_=0, to=0, orient=tk.HORIZONTAL,
                          command=lambda e: self.draw())
            s.pack(fill=tk.X)
            self.scales[key] = s

    # ---- listas (chamadas a partir do _drain) ----
    def refresh_sino_list(self, select=None):
        names = [s["name"] for s in self.app.sinograms]
        self.sino_combo.configure(values=names)
        if select and select in names:
            self.var_sino.set(select)
        elif names and not self.var_sino.get():
            self.var_sino.set(names[-1])
        self._sync_cor_from_sino()

    def refresh_recon_list(self):
        self.recon_tree.delete(*self.recon_tree.get_children())
        for r in self.app.reconstructions:
            sx, sy, sz = r.get("spacing", (1, 1, 1))
            shape = "×".join(str(d) for d in r["volume"].shape)
            self.recon_tree.insert("", tk.END, iid=r["name"],
                                   values=(r["name"], shape,
                                           f"{sx:.2f},{sy:.2f},{sz:.2f}",
                                           r.get("source", "")))

    def _goto_sinogram(self):
        """Jump to the Sinogram tab to place the COR line visually."""
        entry = self.app.get_sinogram(self.var_sino.get())
        if entry is not None:
            self.app.sino = entry["sino"]
            self.app.sino_range_mm = entry["sino_range_mm"]
            self.app.fov_mm = entry["fov_mm"]
            self.app.angles = entry["angles"]
            self.app.last_source = entry["source"]
            self.app.current_sino_name = entry["name"]
            self.app.sino_panel.on_new()
        self.app.notebook.select(self.app.sino_panel)

    def _sync_cor_from_sino(self):
        """Load the COR stored on the selected sinogram into the spinbox."""
        entry = self.app.get_sinogram(self.var_sino.get())
        if entry is not None:
            self.v["cor_mm"].set(float(entry.get("cor_mm", 0.0)))
        self._update_fbp_hint()

    def _update_fbp_hint(self):
        """Show the cutoff in physical units for the SELECTED sinogram.

        'Fraction of Nyquist' is only meaningful once the radial bin width is
        known, and the bin width changes with the sinogram and with the
        radial interpolation factor. Showing mm^-1 here is what makes the
        number reportable in a methods section.
        """
        try:
            entry = self.app.get_sinogram(self.var_sino.get())
            if entry is None:
                self.lbl_cutoff.configure(text="(select a sinogram)")
                return
            srange = float(entry.get("sino_range_mm") or 0.0)
            n_dist = int(entry["sino"].shape[1])
            if self.v["fbp_interp"].get() != "none":
                n_dist *= max(1, int(self.v["fbp_interp_factor"].get()))
            if srange <= 0 or n_dist <= 0:
                self.lbl_cutoff.configure(text="(unknown radial range)")
                return
            bin_mm = srange / n_dist
            nyq = 1.0 / (2.0 * bin_mm)              # mm^-1
            cut = float(self.v["fbp_cutoff"].get())
            txt = (f"bin {bin_mm:.4f} mm → Nyquist {nyq:.3f} mm⁻¹; "
                   f"cutoff = {cut * nyq:.3f} mm⁻¹ "
                   f"({cut * nyq * 10:.2f} cycles/cm)")
            if self.v["fbp_window"].get() == "butterworth":
                txt += "  [−3 dB point]"
            self.lbl_cutoff.configure(text=txt)
        except Exception:
            self.lbl_cutoff.configure(text="")

    def _recon_async(self):
        name = self.var_sino.get()
        if not name and self.app.sinograms:
            name = self.app.sinograms[-1]["name"]
        if not name:
            messagebox.showwarning(
                "No sinogram", "Build a sinogram (GATE/PETSYS tab) first.")
            return
        rname = self.var_recon_name.get()
        from types import SimpleNamespace
        snapshot = {key: SimpleNamespace(get=lambda value=var.get(): value)
                    for key, var in self.v.items()}
        self.app.start_worker("reconstruction", self.app.run_recon, snapshot, name, rname)

    def _add_mhd(self):
        path = filedialog.askopenfilename(
            filetypes=[("MHD/NRRD", "*.mhd *.nrrd"), ("All files", "*.*")])
        if not path:
            return
        try:
            import SimpleITK as sitk
            img = sitk.ReadImage(path); arr = np.asarray(sitk.GetArrayFromImage(img), float)
            sp = img.GetSpacing()
            nm = os.path.basename(path).rsplit(".", 1)[0]
            self.app.add_reconstruction({
                "name": nm, "volume": arr,
                "spacing": (float(sp[0]), float(sp[1]), float(sp[2])),
                "source": "MHD", "method": "external", "info": f"{nm} {arr.shape}"})
            self.app.log(f"[recon+] MHD: {nm} {arr.shape} spacing {tuple(round(s,3) for s in sp)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _remove(self):
        sel = self.recon_tree.selection()
        if not sel:
            return
        name = sel[0]
        self.app.reconstructions[:] = [r for r in self.app.reconstructions
                                       if r["name"] != name]
        if self.current_recon and self.current_recon.get("name") == name:
            self.current_recon = None
        self.refresh_recon_list()
        self.app.log(f"[recon−] removido: {name}")

    def _show_selected(self):
        sel = self.recon_tree.selection()
        if not sel:
            return
        name = sel[0]
        entry = next((r for r in self.app.reconstructions if r["name"] == name), None)
        if entry is None:
            return
        self.current_recon = entry
        self.app.volume = entry["volume"]
        self.app.last_source = entry.get("source", "-")
        self.app.last_method = entry.get("method", "-")
        self.on_new_volume()

    def _voxel_sizes(self):
        """(dxy, dz) in mm. Uses the spacing of the selected reconstruction (respects MHD
        loaded); it only falls back to detector_config when there is no
        current reconstruction."""
        if self.current_recon and self.current_recon.get("spacing"):
            sx, sy, sz = self.current_recon["spacing"]
            return float(sx), float(sz)
        nx = self.app.volume.shape[2] if self.app.volume is not None else 128
        dxy = (self.app.fov_mm / nx) if self.app.fov_mm else 1.0
        dz = dxy
        try:
            dz = float(np.asarray(self.app.detector_config["sipm_pitch"])[1])
        except Exception:
            dz = dxy
        return dxy, dz / (getattr(self.app, "vol_zoom", 1) or 1)

    def on_new_volume(self):
        nz, ny, nx = self.app.volume.shape
        self.scales["z"].configure(to=max(0, nz - 1)); self.scales["z"].set(nz // 2)
        self.scales["y"].configure(to=max(0, ny - 1)); self.scales["y"].set(ny // 2)
        self.scales["x"].configure(to=max(0, nx - 1)); self.scales["x"].set(nx // 2)
        self.draw()

    def _on_scroll(self, event):
        if self.app.volume is None or event.inaxes is None:
            return
        key = {id(self.ax_ax): "z", id(self.ax_co): "y",
               id(self.ax_sa): "x"}.get(id(event.inaxes))
        if key is None:
            return
        maxv = {"z": self.app.volume.shape[0] - 1,
                "y": self.app.volume.shape[1] - 1,
                "x": self.app.volume.shape[2] - 1}[key]
        s = self.scales[key]
        cur = int(float(s.get()))
        s.set(min(max(cur + (1 if event.button == "up" else -1), 0), maxv))
        self.draw()

    def draw(self):
        if self.app.volume is None:
            return
        v = self.app.volume; nz, ny, nx = v.shape
        z = min(max(int(float(self.scales["z"].get())), 0), nz - 1)
        y = min(max(int(float(self.scales["y"].get())), 0), ny - 1)
        x = min(max(int(float(self.scales["x"].get())), 0), nx - 1)
        dxy, dz = self._voxel_sizes()
        ext_ax = [0, nx * dxy, 0, ny * dxy]
        ext_co = [0, nx * dxy, 0, nz * dz]
        ext_sa = [0, ny * dxy, 0, nz * dz]
        cmap = self.ctrl.get_cmap()
        for i, (ax, img, title, ext) in enumerate([
            (self.ax_ax, v[z], f"Axial Z={z}", ext_ax),
            (self.ax_co, v[:, y, :], f"Coronal Y={y}", ext_co),
            (self.ax_sa, v[:, :, x], f"Sagital X={x}", ext_sa),
        ]):
            vmin, vmax = self.ctrl.clim(img, v, update_display=(i == 0))
            ax.clear()
            ax.imshow(img, cmap=cmap, origin="lower", extent=ext, aspect="equal",
                      vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("mm", fontsize=7); ax.set_ylabel("mm", fontsize=7)
        nm = self.current_recon["name"] if self.current_recon else self.app.last_source
        self.fig.suptitle(f"{nm} · {self.app.last_method}", fontsize=11)
        self.canvas.draw_idle()

    def _save(self):
        if self.app.volume is None:
            messagebox.showwarning("No volume", "Reconstruct or select a volume.")
            return
        fmt = self.var_fmt.get()
        base = self.current_recon["name"] if self.current_recon else self.app.last_source
        path = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}", filetypes=[(fmt.upper(), f"*.{fmt}")],
            initialfile=f"{base}.{fmt}")
        if not path:
            return
        try:
            import SimpleITK as sitk
            dxy, dz = self._voxel_sizes()
            arr = np.asarray(self.app.volume).astype(np.float32)
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing([dxy, dxy, dz])
            img.SetOrigin([-(arr.shape[2]*dxy)/2 + dxy/2,
                           -(arr.shape[1]*dxy)/2 + dxy/2,
                           -(arr.shape[0]*dz)/2 + dz/2])
            sitk.WriteImage(img, path)
            self.app.log(f"[{fmt.upper()} saved] {path} "
                         f"(voxel {dxy:.3f}×{dxy:.3f}×{dz:.3f} mm)")
            messagebox.showinfo("Saved", f"Volume saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))


# ===========================================================================
#  Profiles tab (researcher / config / rsectors / calibration / experiment)
# ===========================================================================

class ProfilesPanel(ttk.Frame):
    """CRUD for profiles stored in the local SQLite database.

    A profile groups: name, researcher, experiment, format, detector_config,
    rsectors, angles and calibration. On save, it captures the CURRENT state
    of the chosen format's tab; on load, it restores that state into the tab.
    """

    def __init__(self, master, app):
        super().__init__(master, padding=8)
        self.app = app

        form = ttk.LabelFrame(self, text="Profile", padding=6); form.pack(fill=tk.X)
        self.var_name = tk.StringVar()
        self.var_pesq = tk.StringVar()
        self.var_exp = tk.StringVar()
        self.var_fmt = tk.StringVar(value="petsys")
        for lbl, var in [("profile name", self.var_name),
                         ("researcher", self.var_pesq),
                         ("experiment", self.var_exp)]:
            row = ttk.Frame(form); row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=lbl, width=16).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=36).pack(side=tk.LEFT, fill=tk.X, expand=True)
        labeled_combo(form, "formato", self.var_fmt, ["gate", "petsys"])
        ttk.Label(form, text="notas").pack(anchor="w")
        self.txt_notes = tk.Text(form, height=3, width=44)
        self.txt_notes.pack(fill=tk.X)

        b = ttk.Frame(self); b.pack(fill=tk.X, pady=4)
        ttk.Button(b, text="Save profile (snapshot of the format tab)",
                   command=self._save).pack(side=tk.LEFT, padx=2)
        ttk.Button(b, text="Refresh list", command=self.refresh).pack(side=tk.LEFT, padx=2)

        cols = ("name", "researcher", "experiment", "fmt", "updated")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=10)
        for c, w in zip(cols, (140, 130, 160, 60, 150)):
            self.tree.heading(c, text=c); self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill=tk.BOTH, expand=True, pady=4)

        b2 = ttk.Frame(self); b2.pack(fill=tk.X)
        ttk.Button(b2, text="Load profile", command=self._load).pack(side=tk.LEFT, padx=2)
        ttk.Button(b2, text="Remove profile", command=self._delete).pack(side=tk.LEFT, padx=2)

        self.refresh()

    def refresh(self):
        self.tree.delete(*self.tree.get_children())
        import datetime
        for p in self.app.store.list_profiles():
            ts = p.get("updated_at") or 0
            when = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else ""
            self.tree.insert("", tk.END, values=(p["name"], p.get("pesquisador", ""),
                             p.get("experimento", ""), p.get("fmt", ""), when))

    def _tab_for(self, fmt):
        return self.app.gate if fmt == "gate" else self.app.petsys

    def _save(self):
        name = self.var_name.get().strip()
        if not name:
            messagebox.showwarning("No name", "Give the profile a name.")
            return
        fmt = self.var_fmt.get()
        tab = self._tab_for(fmt)
        try:
            cfg = tab.cfg_form.get_config()
            angles = tab.angles.get_angles()
            rsec = tab.angles.get_rsectors()
            calib = getattr(self.app.petsys, "calibration", None) if fmt == "petsys" else None
            self.app.store.save_profile(
                name, pesquisador=self.var_pesq.get(), experimento=self.var_exp.get(),
                fmt=fmt, detector_config=cfg, rsectors=rsec, angles=angles,
                calibration=calib, notes=self.txt_notes.get("1.0", tk.END).strip())
            self.app.log(f"[profile] '{name}' saved (snapshot of the {fmt.upper()} tab).")
            self.refresh()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Select", "Select a profile in the list.")
            return None
        return self.tree.item(sel[0], "values")[0]

    def _load(self):
        name = self._selected()
        if not name:
            return
        p = self.app.store.get_profile(name)
        if not p:
            return
        fmt = p.get("fmt", "petsys")
        tab = self._tab_for(fmt)
        tab.cfg_form._populate(p.get("detector_config") or {})
        tab.angles.set_data(p.get("angles") or {}, p.get("rsectors") or {})
        if fmt == "petsys" and p.get("calibration"):
            self.app.petsys.calibration = p["calibration"]
        # restore the profile form fields
        self.var_name.set(p["name"]); self.var_pesq.set(p.get("pesquisador", ""))
        self.var_exp.set(p.get("experimento", "")); self.var_fmt.set(fmt)
        self.txt_notes.delete("1.0", tk.END); self.txt_notes.insert("1.0", p.get("notes", "") or "")
        self.app.notebook.select(tab)
        self.app.log(f"[profile] '{name}' loaded into the {fmt.upper()} tab.")

    def _delete(self):
        name = self._selected()
        if not name:
            return
        if messagebox.askyesno("Remove", f"Remove profile '{name}'?"):
            self.app.store.delete_profile(name)
            self.refresh()
            self.app.log(f"[profile] '{name}' removed.")


# ===========================================================================
#  NEMA tab — FWHM analysis (NEMA NU 4-2008) with ROIs
# ===========================================================================

class NemaPanel(ttk.Frame):
    """Multi-image FWHM analyzer (NEMA NU 4-2008).

    - Image list (loaded MHD and/or imported reconstructions); each image
      keeps ITS OWN ROIs/measurements and is individually selectable.
    - ROIs: square/circle of user-defined size, free contour, and
      drag-resizable; copy/replicate and reposition.
    - Exports an Excel file with ONE SHEET per image (its ROI metrics).
    - Shared image controls (colormap, vmin/vmax) + slice scrolling.
    """

    def __init__(self, master, app):
        super().__init__(master, padding=4)
        self.app = app
        self.images = []        # [{name, array, spacing(dx,dy,dz), rois:[...]}]
        self.cur = -1
        self.plane = tk.StringVar(value="Axial")
        self.current_slice = 0
        # ferramentas de ROI
        self.roi_shape = tk.StringVar(value="square")   # square|circle|free|move
        self.side = tk.DoubleVar(value=21)
        self.radius = tk.DoubleVar(value=10)
        self.selected_roi = None
        self.pending = None
        self._drawing = False
        self._moving = False
        self._press = None
        self._free_pts = []
        self.current_patch = None
        self._build_ui()

    def _build_ui(self):
        left = ttk.Frame(self, width=250, padding=6); left.pack(side=tk.LEFT, fill=tk.Y)

        gi = ttk.LabelFrame(left, text="Images", padding=4); gi.pack(fill=tk.X)
        self.img_tree = ttk.Treeview(gi, columns=("name", "shape"), show="headings", height=6)
        self.img_tree.heading("name", text="name"); self.img_tree.column("name", width=120)
        self.img_tree.heading("shape", text="shape"); self.img_tree.column("shape", width=90)
        self.img_tree.pack(fill=tk.X)
        self.img_tree.bind("<<TreeviewSelect>>", lambda e: self._on_image_select())
        ib = ttk.Frame(gi); ib.pack(fill=tk.X, pady=2)
        ttk.Button(ib, text="+ MHD", command=self._add_mhd).pack(side=tk.LEFT, padx=1)
        ttk.Button(ib, text="↧ reconstructions", command=self._import_recons).pack(side=tk.LEFT, padx=1)
        ttk.Button(ib, text="−", width=3, command=self._remove_image).pack(side=tk.LEFT, padx=1)

        gp = ttk.LabelFrame(left, text="Plano", padding=4); gp.pack(fill=tk.X, pady=4)
        for p in ["Axial", "Coronal", "Sagittal"]:
            ttk.Radiobutton(gp, text=p, variable=self.plane, value=p,
                            command=self.update_image).pack(anchor=tk.W)
        ttk.Label(left, text="Slice (or scroll on the image):").pack()
        self.slice_slider = ttk.Scale(left, from_=0, to=0, orient=tk.HORIZONTAL,
                                      command=self.on_slice_change)
        self.slice_slider.pack(fill=tk.X)

        gr = ttk.LabelFrame(left, text="ROI", padding=4); gr.pack(fill=tk.X, pady=4)
        for txt, val in [("Square (side)", "square"), ("Circle (radius)", "circle"),
                         ("Free contour", "free"), ("Move/select", "move")]:
            ttk.Radiobutton(gr, text=txt, variable=self.roi_shape, value=val).pack(anchor=tk.W)
        rs = ttk.Frame(gr); rs.pack(fill=tk.X)
        ttk.Label(rs, text="lado(px)").pack(side=tk.LEFT)
        ttk.Entry(rs, textvariable=self.side, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(rs, text="raio(px)").pack(side=tk.LEFT)
        ttk.Entry(rs, textvariable=self.radius, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(gr, text="(click = fixed size; drag = variable)",
                  foreground="#666", font=("", 7)).pack(anchor="w")

        self.ctrl = ImageControls(left, self.update_image, default_cmap="gray",
                                  default_mode="volume")
        self.ctrl.pack(fill=tk.X, pady=4)

        center = ttk.Frame(self); center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=center)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        bind_wheel_slices(self.canvas, self._on_scroll)

        right = ttk.Frame(self, width=430, padding=6); right.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(right, text="ROI pendente (NEMA)", font=("Arial", 10, "bold")).pack()
        self.stats_text = tk.Text(right, height=8, width=48, font=("Courier", 9))
        self.stats_text.pack(pady=4)
        rb = ttk.Frame(right); rb.pack(fill=tk.X)
        ttk.Button(rb, text="Add to table", command=self._commit_pending).pack(side=tk.LEFT, padx=2)
        ttk.Button(rb, text="Copy sel. ROI", command=self.copy_roi).pack(side=tk.LEFT, padx=2)
        ttk.Label(right, text="ROIs of the current image", font=("Arial", 10, "bold")).pack(pady=(8, 0))
        cols = ("ID", "shape", "FW_X(mm)", "FW_Y(mm)", "Max", "Mean", "Std")
        self.tree = ttk.Treeview(right, columns=cols, show='headings', height=11)
        for c, w in zip(cols, (28, 56, 64, 64, 50, 50, 50)):
            self.tree.heading(c, text=c); self.tree.column(c, width=w, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True, pady=4)
        ttk.Button(right, text="Remove selected ROI", command=self.remove_roi).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Export Excel (1 sheet per image)",
                   command=self.save_excel).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Export CSV (current image)", command=self.save_csv).pack(fill=tk.X)

    # ---------- imagens ----------
    def cur_image(self):
        return self.images[self.cur] if 0 <= self.cur < len(self.images) else None

    def _cur_rois(self):
        im = self.cur_image()
        return im["rois"] if im else []

    def refresh_images(self):
        self.img_tree.delete(*self.img_tree.get_children())
        for i, im in enumerate(self.images):
            self.img_tree.insert("", tk.END, iid=str(i),
                                 values=(im["name"], "×".join(str(d) for d in im["array"].shape)))

    def _add_mhd(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("MHD/NRRD", "*.mhd *.nrrd"), ("All files", "*.*")])
        if not paths:
            return
        import SimpleITK as sitk
        for path in paths:
            try:
                img = sitk.ReadImage(path)
                arr = np.asarray(sitk.GetArrayFromImage(img), float)
                sp = img.GetSpacing()
                nm = os.path.basename(path).rsplit(".", 1)[0]
                self.images.append({"name": nm, "array": arr,
                                    "spacing": (float(sp[0]), float(sp[1]), float(sp[2])),
                                    "rois": []})
                self.app.log(f"[NEMA+] {nm} {arr.shape}")
            except Exception as e:
                messagebox.showerror("Error", f"{path}: {e}")
        self.refresh_images()

    def _import_recons(self):
        existing = {im["name"] for im in self.images}
        n = 0
        for r in self.app.reconstructions:
            if r["name"] in existing:
                continue
            self.images.append({"name": r["name"],
                                "array": np.asarray(r["volume"], float),
                                "spacing": tuple(r.get("spacing", (1, 1, 1))),
                                "rois": []})
            n += 1
        self.refresh_images()
        self.app.log(f"[NEMA] imported {n} reconstruction(s) from the list.")

    def _remove_image(self):
        sel = self.img_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        self.images.pop(idx)
        self.cur = -1
        self.refresh_images()
        self.ax.clear(); self.canvas.draw()
        self.refresh_table()

    def _on_image_select(self):
        sel = self.img_tree.selection()
        if not sel:
            return
        self.cur = int(sel[0])
        self.current_slice = 0
        self.selected_roi = None; self.pending = None
        self.update_image(); self.refresh_table()

    # ---------- vistas ----------
    def get_current_view(self):
        im = self.cur_image()
        if im is None:
            return None, (1, 1)
        dx, dy, dz = im["spacing"]
        idx = self.current_slice
        arr = im["array"]
        if self.plane.get() == "Axial":
            return arr[idx, :, :], (dy, dx)
        elif self.plane.get() == "Coronal":
            return arr[:, idx, :], (dz, dx)
        else:
            return arr[:, :, idx], (dz, dy)

    def max_slice(self):
        im = self.cur_image()
        if im is None:
            return 0
        z, y, x = im["array"].shape
        return {"Axial": z - 1, "Coronal": y - 1, "Sagittal": x - 1}[self.plane.get()]

    def update_image(self, *args):
        im = self.cur_image()
        if im is None:
            return
        ms = self.max_slice()
        self.slice_slider.configure(to=ms)
        if self.current_slice > ms:
            self.current_slice = ms
        data, _ = self.get_current_view()
        vmin, vmax = self.ctrl.clim(data, im["array"])
        self.ax.clear()
        self.ax.imshow(data, cmap=self.ctrl.get_cmap(), vmin=vmin, vmax=vmax)
        self.ax.set_title(f"{im['name']} · {self.plane.get()} · slice {self.current_slice}",
                          fontsize=9)
        self.ax.axis('off')
        for i, roi in enumerate(im["rois"]):
            if roi["plane"] == self.plane.get() and roi["slice"] == self.current_slice:
                color = 'yellow' if i == self.selected_roi else 'lime'
                self._add_patch(self.ax, roi["shape"], roi["geom"], color)
                cx, cy = self._center(roi["shape"], roi["geom"])
                self.ax.text(cx, cy, str(i + 1), color=color, fontsize=10, weight='bold',
                             ha='center', va='center')
        self.current_patch = None
        self.canvas.draw()

    def on_slice_change(self, val):
        self.current_slice = int(float(val))
        self.update_image()

    def _on_scroll(self, event):
        if self.cur_image() is None:
            return
        self.current_slice = min(max(self.current_slice + (1 if event.button == "up" else -1),
                                     0), self.max_slice())
        self.slice_slider.set(self.current_slice)
        self.update_image()

    # ---------- geometria de ROI ----------
    @staticmethod
    def _center(shape, geom):
        if shape == "square":
            return geom["x0"] + geom["size"] / 2, geom["y0"] + geom["size"] / 2
        if shape == "circle":
            return geom["cx"], geom["cy"]
        pts = geom["points"]
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    @staticmethod
    def _translate(shape, geom, dx, dy):
        if shape == "square":
            geom["x0"] += dx; geom["y0"] += dy
        elif shape == "circle":
            geom["cx"] += dx; geom["cy"] += dy
        else:
            geom["points"] = [(p[0] + dx, p[1] + dy) for p in geom["points"]]

    @staticmethod
    def _add_patch(ax, shape, geom, color, ls='-'):
        if shape == "square":
            ax.add_patch(patches.Rectangle((geom["x0"], geom["y0"]), geom["size"],
                         geom["size"], linewidth=1.5, edgecolor=color, facecolor='none',
                         linestyle=ls))
        elif shape == "circle":
            ax.add_patch(patches.Circle((geom["cx"], geom["cy"]), geom["r"], linewidth=1.5,
                         edgecolor=color, facecolor='none', linestyle=ls))
        else:
            ax.add_patch(patches.Polygon(geom["points"], closed=True, linewidth=1.5,
                         edgecolor=color, facecolor='none', linestyle=ls))

    def create_temp_patch(self, shape, geom):
        # redraw the base image and overlay the temporary patch (dashed cyan)
        self.update_image()
        self._add_patch(self.ax, shape, geom, 'cyan', ls='--')
        self.canvas.draw()

    def _pick_roi(self, x, y):
        best, bestd = None, 1e9
        for i, roi in enumerate(self._cur_rois()):
            if roi["plane"] != self.plane.get() or roi["slice"] != self.current_slice:
                continue
            cx, cy = self._center(roi["shape"], roi["geom"])
            d = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
            if d < bestd:
                best, bestd = i, d
        thr = max(15.0, float(self.side.get()))
        return best if bestd <= thr else None

    # ---------- interaction ----------
    def on_click(self, event):
        if event.inaxes != self.ax or self.cur_image() is None:
            return
        self._press = (event.xdata, event.ydata)
        mode = self.roi_shape.get()
        if mode == "move":
            self.selected_roi = self._pick_roi(event.xdata, event.ydata)
            self._moving = self.selected_roi is not None
            self.update_image()
            return
        if mode == "free":
            self._free_pts = [(event.xdata, event.ydata)]
        self._drawing = True

    def on_drag(self, event):
        if event.inaxes != self.ax or self.cur_image() is None or event.xdata is None:
            return
        mode = self.roi_shape.get()
        if mode == "move" and self._moving and self.selected_roi is not None:
            sx, sy = self._press
            roi = self._cur_rois()[self.selected_roi]
            self._translate(roi["shape"], roi["geom"], event.xdata - sx, event.ydata - sy)
            self._press = (event.xdata, event.ydata)
            self.update_image()
            return
        if not self._drawing:
            return
        sx, sy = self._press
        if mode == "square":
            side = max(abs(event.xdata - sx), abs(event.ydata - sy))
            x0 = sx if event.xdata > sx else sx - side
            y0 = sy if event.ydata > sy else sy - side
            self.create_temp_patch("square", {"x0": x0, "y0": y0, "size": side})
        elif mode == "circle":
            r = max(1.0, ((event.xdata - sx) ** 2 + (event.ydata - sy) ** 2) ** 0.5)
            self.create_temp_patch("circle", {"cx": sx, "cy": sy, "r": r})
        elif mode == "free":
            self._free_pts.append((event.xdata, event.ydata))
            self.create_temp_patch("free", {"points": self._free_pts})

    def on_release(self, event):
        mode = self.roi_shape.get()
        if mode == "move":
            if self._moving and self.selected_roi is not None:
                self._recompute(self.selected_roi); self.refresh_table()
            self._moving = False
            return
        if not self._drawing:
            return
        self._drawing = False
        sx, sy = self._press
        ex = event.xdata if event.xdata is not None else sx
        ey = event.ydata if event.ydata is not None else sy
        moved = (abs(ex - sx) + abs(ey - sy)) > 2
        if mode == "square":
            if moved:
                side = max(abs(ex - sx), abs(ey - sy))
                x0 = sx if ex > sx else sx - side; y0 = sy if ey > sy else sy - side
            else:
                side = float(self.side.get()); x0 = sx - side / 2; y0 = sy - side / 2
            self._make_pending("square", {"x0": x0, "y0": y0, "size": max(1.0, side)})
        elif mode == "circle":
            r = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5 if moved else float(self.radius.get())
            self._make_pending("circle", {"cx": sx, "cy": sy, "r": max(1.0, r)})
        elif mode == "free" and len(self._free_pts) >= 3:
            self._make_pending("free", {"points": list(self._free_pts)})

    # ---------- metrics / table ----------
    def _make_pending(self, shape, geom):
        data, (sp_v, sp_h) = self.get_current_view()
        m = roi_metrics(data, shape, geom, sp_v, sp_h)
        if m is None:
            return
        self.pending = {"shape": shape, "geom": geom, "plane": self.plane.get(),
                        "slice": self.current_slice, "metrics": m}
        self.update_image()
        self._add_patch(self.ax, shape, geom, 'cyan', ls='--')
        self.canvas.draw()
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END,
            f"[NEMA NU 4-2008] forma={shape}\n"
            f"FWHM X: {m['FWHM_X_px']:.2f} px | {m['FWHM_X_mm']:.2f} mm\n"
            f"FWHM Y: {m['FWHM_Y_px']:.2f} px | {m['FWHM_Y_mm']:.2f} mm\n"
            "----------------------------------------\n"
            f"Max: {m['Max']} | Min: {m['Min']}\n"
            f"Mean: {m['Mean']} | Std: {m['Std']}")

    def _recompute(self, idx):
        roi = self._cur_rois()[idx]
        data, (sp_v, sp_h) = self.get_current_view()
        m = roi_metrics(data, roi["shape"], roi["geom"], sp_v, sp_h)
        if m:
            roi["metrics"] = m

    def _commit_pending(self):
        im = self.cur_image()
        if im is None or self.pending is None:
            return
        im["rois"].append(self.pending)
        self.pending = None
        self.refresh_table(); self.update_image()

    def copy_roi(self):
        im = self.cur_image()
        if im is None or self.selected_roi is None:
            messagebox.showinfo("Copy", "Select a ROI (Move mode) to copy.")
            return
        import copy as _copy
        roi = _copy.deepcopy(im["rois"][self.selected_roi])
        self._translate(roi["shape"], roi["geom"], 6, 6)   # offset the copy
        self._recompute_into(roi)
        im["rois"].append(roi)
        self.selected_roi = len(im["rois"]) - 1
        self.refresh_table(); self.update_image()

    def _recompute_into(self, roi):
        data, (sp_v, sp_h) = self.get_current_view()
        m = roi_metrics(data, roi["shape"], roi["geom"], sp_v, sp_h)
        if m:
            roi["metrics"] = m

    def remove_roi(self):
        im = self.cur_image()
        sel = self.tree.selection()
        if im is None or not sel:
            return
        idx = self.tree.index(sel[0])
        im["rois"].pop(idx)
        self.selected_roi = None
        self.refresh_table(); self.update_image()

    def refresh_table(self):
        self.tree.delete(*self.tree.get_children())
        for i, roi in enumerate(self._cur_rois()):
            m = roi["metrics"]
            self.tree.insert("", tk.END, values=(i + 1, roi["shape"], m["FWHM_X_mm"],
                             m["FWHM_Y_mm"], m["Max"], m["Mean"], m["Std"]))

    # ---------- export ----------
    def _rois_dataframe(self, im):
        import pandas as pd
        rows = []
        for i, roi in enumerate(im["rois"]):
            cx, cy = self._center(roi["shape"], roi["geom"])
            m = roi["metrics"]
            rows.append({"ID": i + 1, "shape": roi["shape"], "plane": roi["plane"],
                         "slice": roi["slice"], "center_x_px": round(cx, 1),
                         "center_y_px": round(cy, 1), **m})
        cols = ["ID", "shape", "plane", "slice", "center_x_px", "center_y_px",
                "FWHM_X_px", "FWHM_X_mm", "FWHM_Y_px", "FWHM_Y_mm",
                "Max", "Min", "Mean", "Std"]
        return pd.DataFrame(rows, columns=cols)

    @staticmethod
    def _sheet_name(name, used):
        bad = '[]:*?/\\'
        s = "".join(c for c in str(name) if c not in bad)[:31] or "image"
        base = s; i = 2
        while s in used:
            suf = f"_{i}"; s = base[:31 - len(suf)] + suf; i += 1
        used.add(s)
        return s

    def save_excel(self):
        if not self.images:
            messagebox.showinfo("No images", "Load images first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                            filetypes=[("Excel", "*.xlsx")])
        if not path:
            return
        try:
            import pandas as pd
            used = set()
            with pd.ExcelWriter(path, engine="openpyxl") as xw:
                for im in self.images:
                    df = self._rois_dataframe(im)
                    df.to_excel(xw, sheet_name=self._sheet_name(im["name"], used), index=False)
            self.app.log(f"[NEMA] Excel saved: {path} ({len(self.images)} sheet(s))")
            messagebox.showinfo("Saved", f"Excel saved:\n{path}\n(1 sheet per image)")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_csv(self):
        im = self.cur_image()
        if im is None or not im["rois"]:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
        if path:
            self._rois_dataframe(im).to_csv(path, index=False)
            messagebox.showinfo("Saved", "CSV exported.")



# ===========================================================================
#  Plot Recon tab — publication-ready figure composer
# ===========================================================================

PLANES = ("Axial", "Coronal", "Sagittal")
CMAPS = ["gray", "gray_r", "viridis", "inferno", "magma", "plasma", "hot",
         "cividis", "bone", "jet", "turbo"]


class PlotReconPanel(ttk.Frame):
    """Compose a publication figure as a ROW x COLUMN grid of panels.

    Model
    -----
    * A **column** is one *case* being compared. It fixes the anatomical
      location once — plane, slice index and column heading — so that every
      row of that column shows the SAME slice of the SAME object. That is
      the only way an FBP-vs-MLEM comparison down a column is meaningful.
    * A **row** is one representation of that case, and is one of two kinds:

      ``sinogram``  the measured data. The panel does not ask the user which
                    sinogram to draw: it is looked up from the reconstruction
                    placed in a chosen *reference row* of the same column,
                    via that reconstruction's ``source`` field, which
                    ``run_recon`` sets to the name of the sinogram it
                    actually reconstructed. The figure therefore cannot show
                    a sinogram next to a reconstruction that did not come
                    from it.
      ``image``     a reconstruction. Each cell of the row picks one
                    reconstruction from the registry, so a row can be "FBP"
                    and the next row "MLEM".

    Every row carries its own supertitle, colormap, scaling mode and default
    vmin/vmax; every cell may override vmin/vmax. Sinogram rows are scaled
    independently of image rows, which is required because counts and
    reconstructed concentration are different quantities.

    Honest note on slice-matched sinograms: a sinogram is indexed by the
    AXIAL index of the volume, so a slice-matched sinogram only exists when
    the column's plane is Axial. For Coronal/Sagittal the panel shows the
    sinogram summed over slices and says so in the panel title, rather than
    pretending a per-slice sinogram exists.
    """

    ROW_KINDS = ("image", "sinogram")

    def __init__(self, master, app):
        super().__init__(master, padding=6)
        self.app = app
        self.rows = []             # ordered list of row dicts
        self.columns = []          # ordered list of column dicts
        self._preview_job = None
        self._uid = 0
        self._build_ui()

    # ------------------------------------------------------------------
    #  UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        left = ScrollableColumn(self)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        ctrl = left.body
        right = ttk.Frame(self)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- available reconstructions -------------------------------
        box = ttk.LabelFrame(ctrl, text="Available reconstructions", padding=4)
        box.pack(fill=tk.X)
        self.avail = tk.Listbox(box, selectmode=tk.EXTENDED, height=6,
                                exportselection=False, width=38)
        self.avail.pack(fill=tk.X)
        b = ttk.Frame(box); b.pack(fill=tk.X, pady=2)
        ttk.Button(b, text="Refresh",
                   command=self.refresh_available).pack(side=tk.LEFT, padx=1)
        ttk.Button(b, text="Add as columns →",
                   command=self._add_selected_as_columns).pack(side=tk.LEFT, padx=1)
        ttk.Button(b, text="Add as a row ↓",
                   command=self._add_selected_as_row).pack(side=tk.LEFT, padx=1)
        ttk.Label(box, wraplength=300, foreground="#666",
                  text="'Add as columns' makes one column per selected "
                       "reconstruction (the classic side-by-side). "
                       "'Add as a row' makes ONE new image row and fills the "
                       "existing columns with the selected reconstructions, "
                       "in order — that is how you stack FBP over MLEM."
                  ).pack(anchor="w")

        # --- rows -----------------------------------------------------
        rbox = ttk.LabelFrame(ctrl, text="Rows (top to bottom)", padding=3)
        rbox.pack(fill=tk.X, pady=4)
        rb = ttk.Frame(rbox); rb.pack(fill=tk.X)
        ttk.Button(rb, text="+ image row",
                   command=lambda: self._add_row("image")).pack(side=tk.LEFT, padx=1)
        ttk.Button(rb, text="+ sinogram row",
                   command=lambda: self._add_row("sinogram")).pack(side=tk.LEFT, padx=1)
        ttk.Button(rb, text="Clear rows",
                   command=self._clear_rows).pack(side=tk.LEFT, padx=1)
        self.row_host = ttk.Frame(rbox)
        self.row_host.pack(fill=tk.X, pady=2)

        # --- columns --------------------------------------------------
        cbox = ttk.LabelFrame(ctrl, text="Columns (left to right)", padding=3)
        cbox.pack(fill=tk.X, pady=4)
        cb = ttk.Frame(cbox); cb.pack(fill=tk.X)
        ttk.Button(cb, text="+ empty column",
                   command=lambda: self._add_column()).pack(side=tk.LEFT, padx=1)
        ttk.Button(cb, text="Clear columns",
                   command=self._clear_columns).pack(side=tk.LEFT, padx=1)
        self.col_host = ttk.Frame(cbox)
        self.col_host.pack(fill=tk.X, pady=2)

        # --- style ----------------------------------------------------
        st = ttk.LabelFrame(ctrl, text="Style", padding=4)
        st.pack(fill=tk.X)
        self.v_aspect = tk.StringVar(value="equal")
        self.v_row_title_pos = tk.StringVar(value="left")
        labeled_combo(st, "aspect (images)", self.v_aspect,
                      ["equal", "auto"], width=12)
        labeled_combo(st, "row supertitle at", self.v_row_title_pos,
                      ["left", "above", "off"], width=12)

        self.v_fs_title = tk.DoubleVar(value=11)
        self.v_fs_row = tk.DoubleVar(value=12)
        self.v_fs_label = tk.DoubleVar(value=10)
        self.v_fs_tick = tk.DoubleVar(value=9)
        self.v_fs_cbar = tk.DoubleVar(value=8)
        for lbl, var in [("font: col titles", self.v_fs_title),
                         ("font: row titles", self.v_fs_row),
                         ("font: axis labels", self.v_fs_label),
                         ("font: ticks", self.v_fs_tick),
                         ("font: colorbar", self.v_fs_cbar)]:
            labeled_entry(st, lbl, var, width=6)

        self.v_colw = tk.DoubleVar(value=3.2)
        self.v_dpi = tk.IntVar(value=300)
        self.v_suptitle = tk.StringVar(value="")
        self.v_letters = tk.BooleanVar(value=False)
        self.v_letter_color = tk.StringVar(value="white")
        self.v_cbars = tk.BooleanVar(value=True)
        self.v_family = tk.StringVar(value="sans-serif")
        self.v_axes_off = tk.BooleanVar(value=False)
        self.v_col_titles_top = tk.BooleanVar(value=True)
        labeled_entry(st, "column width (in)", self.v_colw, width=6)
        labeled_entry(st, "DPI (export)", self.v_dpi, width=6)
        labeled_entry(st, "figure suptitle", self.v_suptitle, width=12)
        labeled_combo(st, "font family", self.v_family,
                      ["sans-serif", "serif", "monospace"], width=12)
        ttk.Checkbutton(st, text="show colorbars", variable=self.v_cbars,
                        command=self.update_preview).pack(anchor="w")
        ttk.Checkbutton(st, text="column titles only on the top row",
                        variable=self.v_col_titles_top,
                        command=self.update_preview).pack(anchor="w")
        ttk.Checkbutton(st, text="hide axes (ticks and mm labels)",
                        variable=self.v_axes_off,
                        command=self.update_preview).pack(anchor="w")
        ttk.Checkbutton(st, text="panel letters (a), (b), …",
                        variable=self.v_letters,
                        command=self.update_preview).pack(anchor="w")
        labeled_combo(st, "letter colour", self.v_letter_color,
                      ["white", "black", "yellow", "red"], width=12)

        ex = ttk.Frame(ctrl); ex.pack(fill=tk.X, pady=4)
        ttk.Button(ex, text="Update preview",
                   command=self.update_preview).pack(fill=tk.X, pady=1)
        self.v_fmt = tk.StringVar(value="png")
        labeled_combo(ex, "format", self.v_fmt,
                      ["png", "pdf", "svg", "tiff"], width=8)
        ttk.Button(ex, text="Save figure…",
                   command=self.save_figure).pack(fill=tk.X, pady=1)

        # --- preview --------------------------------------------------
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, right).update()
        ttk.Label(right, foreground="#666",
                  text="(preview is rendered at screen DPI; the exported file "
                       "uses the DPI above)").pack(anchor="w")
        self.refresh_available()

    # ------------------------------------------------------------------
    #  registry helpers
    # ------------------------------------------------------------------
    def refresh_available(self):
        self.avail.delete(0, tk.END)
        for r in self.app.reconstructions:
            self.avail.insert(tk.END, f"{r['name']}  {r['volume'].shape}")
        for row in self.rows:
            if row["kind"].get() == "image":
                self._refresh_row_cell_choices(row)
        self._refresh_ref_row_choices()

    def _rec_names(self):
        return [r["name"] for r in self.app.reconstructions]

    def _rec_by_name(self, name):
        return next((r for r in self.app.reconstructions
                     if r["name"] == name), None)

    def _selected_recs(self):
        return [self.app.reconstructions[i] for i in self.avail.curselection()]

    def _next_uid(self):
        self._uid += 1
        return self._uid

    # ------------------------------------------------------------------
    #  rows
    # ------------------------------------------------------------------
    def _add_row(self, kind, title=None, recs=None):
        uid = self._next_uid()
        f = ttk.Frame(self.row_host, relief="groove", borderwidth=1, padding=3)
        f.pack(fill=tk.X, pady=2)
        row = {
            "uid": uid, "frame": f,
            "kind": tk.StringVar(value=kind),
            "title": tk.StringVar(value=title if title is not None
                                  else ("Sinogram" if kind == "sinogram"
                                        else "Reconstruction")),
            "cmap": tk.StringVar(value="viridis" if kind == "sinogram"
                                 else "gray"),
            "scale": tk.StringVar(value="per-panel"),
            "vmin": tk.StringVar(value=""), "vmax": tk.StringVar(value=""),
            "cbar_label": tk.StringVar(value="counts" if kind == "sinogram"
                                       else ""),
            "height": tk.DoubleVar(value=2.2 if kind == "sinogram" else 3.0),
            "ref_row": tk.StringVar(value=""),   # sinogram rows only
            "cells": [],                         # image rows: one var / column
        }

        hdr = ttk.Frame(f); hdr.pack(fill=tk.X)
        ttk.Label(hdr, text=f"{kind}", width=9,
                  foreground="#036").pack(side=tk.LEFT)
        ttk.Label(hdr, text="supertitle").pack(side=tk.LEFT)
        ttk.Entry(hdr, textvariable=row["title"], width=16).pack(side=tk.LEFT, padx=2)
        ttk.Button(hdr, text="▲", width=2,
                   command=lambda r=row: self._move_row(r, -1)).pack(side=tk.RIGHT)
        ttk.Button(hdr, text="▼", width=2,
                   command=lambda r=row: self._move_row(r, +1)).pack(side=tk.RIGHT)
        ttk.Button(hdr, text="✕", width=2,
                   command=lambda r=row: self._remove_row(r)).pack(side=tk.RIGHT)

        s1 = ttk.Frame(f); s1.pack(fill=tk.X)
        ttk.Label(s1, text="cmap", width=6).pack(side=tk.LEFT)
        ttk.Combobox(s1, textvariable=row["cmap"], values=CMAPS,
                     state="readonly", width=9).pack(side=tk.LEFT, padx=1)
        ttk.Label(s1, text="scale").pack(side=tk.LEFT, padx=(6, 1))
        ttk.Combobox(s1, textvariable=row["scale"],
                     values=["per-panel", "shared in row"],
                     state="readonly", width=13).pack(side=tk.LEFT)

        s2 = ttk.Frame(f); s2.pack(fill=tk.X)
        ttk.Label(s2, text="vmin", width=6).pack(side=tk.LEFT)
        ttk.Entry(s2, textvariable=row["vmin"], width=7).pack(side=tk.LEFT)
        ttk.Label(s2, text="vmax").pack(side=tk.LEFT, padx=(4, 1))
        ttk.Entry(s2, textvariable=row["vmax"], width=7).pack(side=tk.LEFT)
        ttk.Label(s2, text="h(in)").pack(side=tk.LEFT, padx=(4, 1))
        ttk.Entry(s2, textvariable=row["height"], width=5).pack(side=tk.LEFT)

        s3 = ttk.Frame(f); s3.pack(fill=tk.X)
        ttk.Label(s3, text="cbar", width=6).pack(side=tk.LEFT)
        ttk.Entry(s3, textvariable=row["cbar_label"], width=12).pack(side=tk.LEFT)

        if kind == "sinogram":
            s4 = ttk.Frame(f); s4.pack(fill=tk.X)
            ttk.Label(s4, text="from", width=6).pack(side=tk.LEFT)
            row["ref_combo"] = ttk.Combobox(s4, textvariable=row["ref_row"],
                                            state="readonly", width=22)
            row["ref_combo"].pack(side=tk.LEFT, padx=1)
            ttk.Label(f, wraplength=300, foreground="#666",
                      text="The sinogram of each column is looked up from the "
                           "reconstruction in the chosen row, through the "
                           "'source' recorded when it was reconstructed. "
                           "Per-column vmin/vmax are set in the column boxes "
                           "below.").pack(anchor="w")
        else:
            row["cell_host"] = ttk.Frame(f)
            row["cell_host"].pack(fill=tk.X)

        self.rows.append(row)
        if kind == "image":
            self._sync_row_cells(row)
            if recs:
                for i, rec in enumerate(recs):
                    if i < len(row["cells"]):
                        row["cells"][i]["rec"].set(rec["name"])
        self._refresh_ref_row_choices()
        if kind == "image" and recs:
            self._on_cell_changed()      # cells now define the slice range
        else:
            self.update_preview()
        return row

    def _remove_row(self, row):
        row["frame"].destroy()
        self.rows.remove(row)
        self._refresh_ref_row_choices()
        self.update_preview()

    def _clear_rows(self):
        for r in list(self.rows):
            r["frame"].destroy()
        self.rows = []
        self._refresh_ref_row_choices()
        self.update_preview()

    def _move_row(self, row, delta):
        i = self.rows.index(row)
        j = i + delta
        if not (0 <= j < len(self.rows)):
            return
        self.rows[i], self.rows[j] = self.rows[j], self.rows[i]
        for r in self.rows:
            r["frame"].pack_forget()
        for r in self.rows:
            r["frame"].pack(fill=tk.X, pady=2)
        self.update_preview()

    def _image_rows(self):
        return [r for r in self.rows if r["kind"].get() == "image"]

    def _row_label(self, row):
        return f"row {self.rows.index(row) + 1}: {row['title'].get()}"

    def _refresh_ref_row_choices(self):
        labels = [self._row_label(r) for r in self._image_rows()]
        for row in self.rows:
            if row["kind"].get() != "sinogram" or "ref_combo" not in row:
                continue
            row["ref_combo"].configure(values=labels)
            if row["ref_row"].get() not in labels:
                row["ref_row"].set(labels[0] if labels else "")

    def _ref_image_row(self, sino_row):
        """Resolve the reference image row of a sinogram row."""
        img_rows = self._image_rows()
        if not img_rows:
            return None
        want = sino_row["ref_row"].get()
        for r in img_rows:
            if self._row_label(r) == want:
                return r
        return img_rows[0]

    # ------------------------------------------------------------------
    #  columns
    # ------------------------------------------------------------------
    def _add_column(self, title=None):
        uid = self._next_uid()
        f = ttk.Frame(self.col_host, relief="groove", borderwidth=1, padding=3)
        f.pack(fill=tk.X, pady=2)
        col = {
            "uid": uid, "frame": f,
            "title": tk.StringVar(value=title or f"case {len(self.columns)+1}"),
            "plane": tk.StringVar(value="Axial"),
            "slice": tk.IntVar(value=0),
            "sino_vmin": tk.StringVar(value=""),
            "sino_vmax": tk.StringVar(value=""),
        }

        r1 = ttk.Frame(f); r1.pack(fill=tk.X)
        ttk.Label(r1, text="title", width=6).pack(side=tk.LEFT)
        ttk.Entry(r1, textvariable=col["title"], width=16).pack(side=tk.LEFT, padx=2)
        ttk.Button(r1, text="◀", width=2,
                   command=lambda c=col: self._move_column(c, -1)).pack(side=tk.RIGHT)
        ttk.Button(r1, text="▶", width=2,
                   command=lambda c=col: self._move_column(c, +1)).pack(side=tk.RIGHT)
        ttk.Button(r1, text="✕", width=2,
                   command=lambda c=col: self._remove_column(c)).pack(side=tk.RIGHT)

        r2 = ttk.Frame(f); r2.pack(fill=tk.X)
        ttk.Label(r2, text="plane", width=6).pack(side=tk.LEFT)
        cb = ttk.Combobox(r2, textvariable=col["plane"], values=list(PLANES),
                          state="readonly", width=9)
        cb.pack(side=tk.LEFT, padx=2)
        cb.bind("<<ComboboxSelected>>", lambda e, c=col: self._on_plane(c))
        col["lbl_mm"] = ttk.Label(r2, text="", width=11)
        col["lbl_mm"].pack(side=tk.RIGHT)

        r3 = ttk.Frame(f); r3.pack(fill=tk.X)
        ttk.Label(r3, text="slice", width=6).pack(side=tk.LEFT)
        col["scale_w"] = tk.Scale(r3, from_=0, to=0, orient=tk.HORIZONTAL,
                                  variable=col["slice"],
                                  command=lambda v: self._debounced_preview(),
                                  length=210)
        col["scale_w"].pack(side=tk.LEFT, fill=tk.X, expand=True)

        r4 = ttk.Frame(f); r4.pack(fill=tk.X)
        ttk.Label(r4, text="sino", width=6).pack(side=tk.LEFT)
        ttk.Label(r4, text="vmin").pack(side=tk.LEFT, padx=(0, 1))
        ttk.Entry(r4, textvariable=col["sino_vmin"], width=7).pack(side=tk.LEFT)
        ttk.Label(r4, text="vmax").pack(side=tk.LEFT, padx=(4, 1))
        ttk.Entry(r4, textvariable=col["sino_vmax"], width=7).pack(side=tk.LEFT)

        col["cell_frames"] = ttk.Frame(f)
        col["cell_frames"].pack(fill=tk.X)

        self.columns.append(col)
        for row in self._image_rows():
            self._sync_row_cells(row)
        self._on_plane(col)
        return col

    def _remove_column(self, col):
        i = self.columns.index(col)
        col["frame"].destroy()
        self.columns.remove(col)
        for row in self._image_rows():
            if i < len(row["cells"]):
                row["cells"][i]["frame"].destroy()
                del row["cells"][i]
        self.update_preview()

    def _clear_columns(self):
        for c in list(self.columns):
            c["frame"].destroy()
        self.columns = []
        for row in self._image_rows():
            for cell in row["cells"]:
                cell["frame"].destroy()
            row["cells"] = []
        self.update_preview()

    def _move_column(self, col, delta):
        i = self.columns.index(col)
        j = i + delta
        if not (0 <= j < len(self.columns)):
            return
        self.columns[i], self.columns[j] = self.columns[j], self.columns[i]
        for row in self._image_rows():
            if max(i, j) < len(row["cells"]):
                row["cells"][i], row["cells"][j] = row["cells"][j], row["cells"][i]
                for cell in row["cells"]:
                    cell["frame"].pack_forget()
                for cell in row["cells"]:
                    cell["frame"].pack(fill=tk.X, pady=1)
        for c in self.columns:
            c["frame"].pack_forget()
        for c in self.columns:
            c["frame"].pack(fill=tk.X, pady=2)
        self.update_preview()

    def _add_selected_as_columns(self):
        recs = self._selected_recs()
        if not recs:
            messagebox.showinfo("Select", "Select one or more reconstructions.")
            return
        rows = self._image_rows()
        if not rows:
            rows = [self._add_row("image", title="Reconstruction")]
        for rec in recs:
            col = self._add_column(title=rec["name"])
            i = self.columns.index(col)
            if i < len(rows[0]["cells"]):
                rows[0]["cells"][i]["rec"].set(rec["name"])
        # The slice slider is bounded by the column's reference volume, which
        # only exists once the cells above have been filled in.
        self._on_cell_changed()

    def _add_selected_as_row(self):
        recs = self._selected_recs()
        if not recs:
            messagebox.showinfo("Select", "Select one or more reconstructions.")
            return
        while len(self.columns) < len(recs):
            self._add_column()
        title = recs[0].get("method", "Reconstruction")
        if all(r.get("method") == title for r in recs):
            title = str(title).upper()
        self._add_row("image", title=title, recs=recs)

    def _on_plane(self, col):
        n = self._n_slices(col)
        col["scale_w"].configure(to=max(0, n - 1))
        if col["slice"].get() >= n:
            col["slice"].set(max(0, n // 2))
        self._debounced_preview()

    def _column_ref_rec(self, col):
        """First reconstruction present in this column, top row first.

        Used to define the column's slice range and its physical geometry.
        """
        j = self.columns.index(col)
        for row in self.rows:
            if row["kind"].get() != "image":
                continue
            if j < len(row["cells"]):
                rec = self._rec_by_name(row["cells"][j]["rec"].get())
                if rec is not None:
                    return rec
        return None

    def _n_slices(self, col):
        rec = self._column_ref_rec(col)
        if rec is None:
            return 1
        vol = rec["volume"]
        p = col["plane"].get()
        return (vol.shape[0] if p == "Axial"
                else vol.shape[1] if p == "Coronal" else vol.shape[2])

    # ------------------------------------------------------------------
    #  cells (one per image row per column)
    # ------------------------------------------------------------------
    def _sync_row_cells(self, row):
        """Make the row have exactly one cell per existing column."""
        host = row.get("cell_host")
        if host is None:
            return
        while len(row["cells"]) > len(self.columns):
            row["cells"].pop()["frame"].destroy()
        while len(row["cells"]) < len(self.columns):
            j = len(row["cells"])
            f = ttk.Frame(host); f.pack(fill=tk.X, pady=1)
            cell = {"frame": f,
                    "rec": tk.StringVar(value=""),
                    "vmin": tk.StringVar(value=""),
                    "vmax": tk.StringVar(value="")}
            ttk.Label(f, text=f"c{j+1}", width=3).pack(side=tk.LEFT)
            cell["combo"] = ttk.Combobox(f, textvariable=cell["rec"],
                                         state="readonly", width=16)
            cell["combo"].pack(side=tk.LEFT, padx=1)
            cell["combo"].bind("<<ComboboxSelected>>",
                               lambda e: self._on_cell_changed())
            ttk.Entry(f, textvariable=cell["vmin"], width=6).pack(side=tk.LEFT, padx=1)
            ttk.Entry(f, textvariable=cell["vmax"], width=6).pack(side=tk.LEFT)
            row["cells"].append(cell)
        self._refresh_row_cell_choices(row)

    def _refresh_row_cell_choices(self, row):
        names = [""] + self._rec_names()
        for cell in row["cells"]:
            cell["combo"].configure(values=names)

    def _on_cell_changed(self):
        for col in self.columns:
            self._on_plane(col)
        self.update_preview()

    # ------------------------------------------------------------------
    #  data extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _view(rec, plane, idx):
        """(image, extent_mm, xlabel, ylabel, position_mm) for one slice.

        Volume is (Z, Y, X); spacing is (dx, dy, dz). Extents are centred on
        the origin, consistent with the exported MHD/NRRD geometry.
        """
        vol = rec["volume"]
        dx, dy, dz = rec["spacing"]
        nz, ny, nx = vol.shape
        idx = int(np.clip(idx, 0,
                          {"Axial": nz, "Coronal": ny, "Sagittal": nx}[plane] - 1))
        if plane == "Axial":
            return (vol[idx, :, :],
                    [-nx*dx/2, nx*dx/2, -ny*dy/2, ny*dy/2],
                    "x (mm)", "y (mm)", idx*dz - nz*dz/2 + dz/2)
        if plane == "Coronal":
            return (vol[:, idx, :],
                    [-nx*dx/2, nx*dx/2, -nz*dz/2, nz*dz/2],
                    "x (mm)", "z (mm)", idx*dy - ny*dy/2 + dy/2)
        return (vol[:, :, idx],
                [-ny*dy/2, ny*dy/2, -nz*dz/2, nz*dz/2],
                "y (mm)", "z (mm)", idx*dx - nx*dx/2 + dx/2)

    def _sino_for(self, sino_row, col):
        """Sinogram of a column: (array, extent, note, source_name) or None.

        The sinogram is NOT chosen by hand: it is the one recorded as the
        ``source`` of the reconstruction sitting in the reference row of the
        same column. If that reconstruction was imported from disk rather
        than reconstructed here, no source exists and the panel says so.
        """
        ref_row = self._ref_image_row(sino_row)
        if ref_row is None:
            return None
        j = self.columns.index(col)
        if j >= len(ref_row["cells"]):
            return None
        rec = self._rec_by_name(ref_row["cells"][j]["rec"].get())
        if rec is None:
            return None
        entry = self.app.get_sinogram(rec.get("source", ""))
        if entry is None:
            return None
        sino = entry["sino"]                      # (n_slices, n_dist, n_ang)
        srange = float(entry.get("sino_range_mm") or 0.0)
        ext = ([0.0, 180.0, -srange/2, srange/2] if srange > 0
               else [0.0, 180.0, 0, sino.shape[1]])
        if col["plane"].get() == "Axial":
            z = int(np.clip(col["slice"].get(), 0, sino.shape[0] - 1))
            return sino[z], ext, f"slice {z}", entry["name"]
        return (sino.sum(axis=0), ext, "summed over slices", entry["name"])

    # ------------------------------------------------------------------
    #  figure
    # ------------------------------------------------------------------
    def _build_figure(self, fig):
        with matplotlib.rc_context({"font.family": self.v_family.get()}):
            self._build_figure_inner(fig)

    def _msg(self, fig, text):
        ax = fig.add_subplot(111); ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", color="#888")

    def _build_figure_inner(self, fig):
        fig.clear()
        if not self.rows or not self.columns:
            self._msg(fig, "Add at least one row and one column.\n"
                           "'Add as columns' then '+ image row' / "
                           "'+ sinogram row'.")
            return

        rows, cols = self.rows, self.columns
        nrows, ncols = len(rows), len(cols)
        fs_t = float(self.v_fs_title.get()); fs_r = float(self.v_fs_row.get())
        fs_l = float(self.v_fs_label.get()); fs_k = float(self.v_fs_tick.get())
        fs_c = float(self.v_fs_cbar.get())
        show_cbar = bool(self.v_cbars.get())
        letters = bool(self.v_letters.get())
        axes_off = bool(self.v_axes_off.get())
        titles_top = bool(self.v_col_titles_top.get())
        row_pos = self.v_row_title_pos.get()
        aspect = self.v_aspect.get()

        w = max(0.5, float(self.v_colw.get()))
        heights = [max(0.3, float(r["height"].get())) for r in rows]
        fig.set_size_inches(ncols * w, sum(heights))
        gs = fig.add_gridspec(nrows, ncols, height_ratios=heights)

        # Per-column mm position of the displayed slice, from the column's
        # reference reconstruction; also refreshes the sidebar label.
        pos_mm = {}
        for col in cols:
            rec = self._column_ref_rec(col)
            if rec is None:
                pos_mm[col["uid"]] = None
                col["lbl_mm"].configure(text="—")
            else:
                _, _, _, _, p = self._view(rec, col["plane"].get(),
                                           col["slice"].get())
                pos_mm[col["uid"]] = p
                col["lbl_mm"].configure(text=f"{p:+.2f} mm")

        # --- gather every panel's data first, so 'shared in row' can be
        #     computed from exactly the slices that will be displayed ---
        panels = {}
        for i, row in enumerate(rows):
            kind = row["kind"].get()
            for j, col in enumerate(cols):
                if kind == "sinogram":
                    s = self._sino_for(row, col)
                    panels[(i, j)] = (
                        None if s is None
                        else {"data": s[0], "ext": s[1], "note": s[2],
                              "src": s[3], "xl": "projection angle (deg)",
                              "yl": "r (mm)"})
                else:
                    rec = (self._rec_by_name(row["cells"][j]["rec"].get())
                           if j < len(row["cells"]) else None)
                    if rec is None:
                        panels[(i, j)] = None
                    else:
                        img, ext, xl, yl, _ = self._view(
                            rec, col["plane"].get(), col["slice"].get())
                        panels[(i, j)] = {"data": img, "ext": ext, "xl": xl,
                                          "yl": yl, "rec": rec,
                                          "note": rec["name"]}

        shared = {}
        for i, row in enumerate(rows):
            if row["scale"].get() != "shared in row":
                continue
            vals = [panels[(i, j)]["data"] for j in range(ncols)
                    if panels[(i, j)] is not None]
            if vals:
                shared[i] = (min(float(v.min()) for v in vals),
                             max(float(v.max()) for v in vals))

        k = 0
        for i, row in enumerate(rows):
            kind = row["kind"].get()
            cmap = row["cmap"].get()
            row_vmin, row_vmax = _f(row["vmin"].get()), _f(row["vmax"].get())
            for j, col in enumerate(cols):
                ax = fig.add_subplot(gs[i, j])
                p = panels[(i, j)]
                if p is None:
                    ax.axis("off")
                    ax.text(0.5, 0.5,
                            "no sinogram\n(source not recorded)"
                            if kind == "sinogram" else "no reconstruction",
                            ha="center", va="center", fontsize=fs_l,
                            color="#888")
                    k += 1
                    continue

                # vmin/vmax precedence: cell override > row default >
                # shared-in-row > matplotlib autoscale.
                if kind == "sinogram":
                    vmin = _f(col["sino_vmin"].get())
                    vmax = _f(col["sino_vmax"].get())
                else:
                    cell = row["cells"][j]
                    vmin, vmax = _f(cell["vmin"].get()), _f(cell["vmax"].get())
                if vmin is None:
                    vmin = row_vmin
                if vmax is None:
                    vmax = row_vmax
                if i in shared:
                    vmin = shared[i][0] if vmin is None else vmin
                    vmax = shared[i][1] if vmax is None else vmax

                im = ax.imshow(p["data"], cmap=cmap, origin="lower",
                               extent=p["ext"], vmin=vmin, vmax=vmax,
                               aspect="auto" if kind == "sinogram" else aspect)

                if axes_off:
                    ax.set_xticks([]); ax.set_yticks([])
                else:
                    ax.set_xlabel(p["xl"], fontsize=fs_l)
                    ax.set_ylabel(p["yl"], fontsize=fs_l)
                    ax.tick_params(labelsize=fs_k)

                # Column heading: on the top row only (default) or on each.
                if j >= 0 and (i == 0 or not titles_top):
                    head = col["title"].get()
                    if kind != "sinogram" and pos_mm[col["uid"]] is not None:
                        head += (f"\n{col['plane'].get()} · "
                                 f"{pos_mm[col['uid']]:+.2f} mm")
                    elif kind == "sinogram":
                        head += f"\n{p['src']} · {p['note']}"
                    ax.set_title(head, fontsize=fs_t)
                elif kind == "sinogram":
                    ax.set_title(f"{p['src']} · {p['note']}", fontsize=fs_t)

                # Row supertitle, on the leftmost panel.
                if j == 0 and row_pos != "off" and row["title"].get():
                    if row_pos == "left":
                        ax.annotate(row["title"].get(),
                                    xy=(0, 0.5), xycoords="axes fraction",
                                    xytext=(-fs_r * 4.2, 0),
                                    textcoords="offset points",
                                    rotation=90, ha="center", va="center",
                                    fontsize=fs_r, fontweight="bold")
                    else:  # above — must clear the column title, which may
                           # itself be two lines tall on the top row
                        title_lines = len(ax.get_title().split("\n")) \
                            if ax.get_title() else 0
                        dy = fs_r * 0.8 + title_lines * fs_t * 1.5
                        ax.annotate(row["title"].get(),
                                    xy=(0, 1), xycoords="axes fraction",
                                    xytext=(0, dy),
                                    textcoords="offset points",
                                    ha="left", va="bottom",
                                    fontsize=fs_r, fontweight="bold")

                if show_cbar:
                    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                    cb.ax.tick_params(labelsize=fs_c)
                    if row["cbar_label"].get():
                        cb.set_label(row["cbar_label"].get(), fontsize=fs_c)

                if letters:
                    ax.text(0.02, 0.97, f"({chr(ord('a') + k)})",
                            transform=ax.transAxes, ha="left", va="top",
                            fontsize=fs_t, color=self.v_letter_color.get(),
                            fontweight="bold")
                k += 1

        if self.v_suptitle.get():
            fig.suptitle(self.v_suptitle.get(), fontsize=fs_r + 2)
        try:
            fig.tight_layout()
        except Exception:
            pass

    # ------------------------------------------------------------------
    #  preview / export
    # ------------------------------------------------------------------
    def _debounced_preview(self):
        if self._preview_job is not None:
            try:
                self.after_cancel(self._preview_job)
            except Exception:
                pass
        self._preview_job = self.after(180, self.update_preview)

    def update_preview(self):
        self._preview_job = None
        try:
            self._build_figure(self.fig)
            self.canvas.draw_idle()
        except Exception as e:
            self.app.error(f"[plot] {e}")

    def save_figure(self):
        if not self.rows or not self.columns:
            messagebox.showinfo("Nothing to save",
                                "Add at least one row and one column.")
            return
        fmt = self.v_fmt.get()
        path = filedialog.asksaveasfilename(defaultextension=f".{fmt}",
                                            filetypes=[(fmt.upper(), f"*.{fmt}")])
        if not path:
            return
        try:
            dpi = int(self.v_dpi.get())
            out = Figure(figsize=(4, 3), dpi=dpi)
            self._build_figure(out)
            out.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            self.app.log(f"[plot] figure saved: {path} ({dpi} dpi, "
                         f"{out.get_size_inches()[0]:.1f}×"
                         f"{out.get_size_inches()[1]:.1f} in, "
                         f"{len(self.rows)}×{len(self.columns)} panels)")
            messagebox.showinfo("Saved", f"Figure saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))


def _f(txt):
    """Parse an optional float entry ('' -> None)."""
    txt = (txt or "").strip()
    if txt == "":
        return None
    try:
        return float(txt)
    except ValueError:
        return None


# ===========================================================================
#  Main application
# ===========================================================================

class ReconUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GimnToolsV2 — PET Reconstruction (GATE / PETSYS)")
        self.geometry("1300x850")

        # shared state
        self.sino = None
        self.angles = None
        self.volume = None
        self.detector_config = None
        self.fov_mm = None
        self.sino_range_mm = None
        self.cor_mm = 0.0          # constant radial offset of the rotation axis
        self.last_source = "-"
        self.last_method = "-"
        self.vol_zoom = 1
        # shared registries
        self.sinograms = []        # [{name, sino, fov_mm, sino_range_mm, source, detector_config, angles}]
        self.reconstructions = []  # [{name, volume, spacing(dx,dy,dz), source, method, info}]
        self.current_sino_name = None
        self._q = queue.Queue()
        self._active_task = None
        self._study_generation = 0

        # local database (SQLite, cross-platform)
        self.store = Store()

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.gate = GatePanel(self.notebook, self)
        self.petsys = PetsysPanel(self.notebook, self)
        self.sino_panel = SinogramPanel(self.notebook, self)
        self.recon_panel = ReconstructionPanel(self.notebook, self)
        self.profiles_panel = ProfilesPanel(self.notebook, self)
        self.plot_panel = PlotReconPanel(self.notebook, self)
        self.nema_panel = NemaPanel(self.notebook, self)
        self.notebook.add(self.gate, text="GATE")
        self.notebook.add(self.petsys, text="PETSYS")
        self.notebook.add(self.sino_panel, text="Sinogram")
        self.notebook.add(self.recon_panel, text="Reconstruction")
        self.notebook.add(self.plot_panel, text="Plot Recon")
        self.notebook.add(self.nema_panel, text="NEMA FWHM")
        self.notebook.add(self.profiles_panel, text="Profiles")

        self.log_text = tk.Text(self, height=7, state="disabled", font=("monospace", 8))
        self.log_text.pack(fill=tk.X, side=tk.BOTTOM)

        # Files remembered between runs (data pickers)
        self._file_vars = {
            "gate_root": self.gate.var_root,
            "petsys_meas": self.petsys.var_meas,
            "petsys_bg": self.petsys.var_bg,
            "petsys_cal": self.petsys.var_cal,
            "cal_bg": self.petsys.cal_bg,
            "cal_out": self.petsys.cal_out,
        }
        self._restore_last_files()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after(120, self._drain)
        self.after(400, self._notify_last_session)

    # ---- file memory across sessions ----
    def _restore_last_files(self):
        data = self.store.get_setting("last_files", {}) or {}
        self._had_last_files = False
        for k, var in self._file_vars.items():
            val = data.get(k, "")
            if val:
                var.set(val); self._had_last_files = True

    def _save_files(self):
        try:
            self.store.set_setting(
                "last_files", {k: v.get() for k, v in self._file_vars.items()})
        except Exception:
            pass

    def _notify_last_session(self):
        if getattr(self, "_had_last_files", False):
            messagebox.showinfo(
                "Previous session",
                "The pre-filled files are those from the LAST RUN "
                "(defaults from the previous session). Confirm or change them before "
                "building the sinogram.")

    def _on_close(self):
        self._save_files()
        self.destroy()

    # ---- thread-safe communication ----
    def log(self, msg):
        self._q.put(("log", msg))

    def error(self, msg):
        self._q.put(("error", msg))

    def start_worker(self, operation, target, *args):
        """Run one job at a time; Tk reads belong to caller, not worker."""
        if self._active_task is not None:
            self.log(f"[BUSY] Cannot start {operation}: {self._active_task} still running")
            return False
        self._active_task = operation
        def run():
            try:
                target(*args)
            finally:
                self._q.put(('task_done', operation))
        threading.Thread(target=run, daemon=True).start()
        return True

    def set_sinogram(self, sino, fov_mm, sino_range_mm, source,
                     detector_config=None, angles_override=None, name=None,
                     norm=None, background=None, dx_cor_mm=0.0, dy_cor_mm=0.0):
        """Queue a complete acquisition; never update Tk state from a worker."""
        data = np.asarray(sino, dtype=np.float64).copy()
        if data.ndim != 3 or not np.isfinite(data).all() or (data < 0).any():
            raise ValueError('measured sinogram must be finite and nonnegative')
        if background is not None:
            background = np.asarray(background, dtype=float).copy()
            if (background.shape != data.shape or not np.isfinite(background).all()
                    or (background < 0).any()):
                raise ValueError('background has invalid shape or values')
        angles = (np.asarray(angles_override, float).copy() if angles_override is not None
                  else (np.arange(data.shape[2]) + .5) * (180.0 / data.shape[2]))
        if angles.shape != (data.shape[2],) or not np.isfinite(angles).all():
            raise ValueError('invalid acquisition angles')
        entry = {
            'name': name or f'{source}_{len(self.sinograms)+1}',
            'sino': data, 'fov_mm': float(fov_mm),
            'sino_range_mm': float(sino_range_mm), 'source': source,
            'detector_config': detector_config, 'angles': angles,
            'norm': norm, 'background': background, 'cor_mm': 0.0,
            'dx_cor_mm': float(dx_cor_mm), 'dy_cor_mm': float(dy_cor_mm),
        }
        if threading.current_thread() is threading.main_thread():
            self._register_sinogram(entry)
        else:
            self._q.put(('new_sinogram', entry))

    def _register_sinogram(self, entry):
        entry['name'] = self._unique_name(entry['name'],
                    [s['name'] for s in self.sinograms])
        self._study_generation += 1
        self.sino = entry['sino']
        self.angles = entry['angles']
        self.volume = None            # do not display a previous acquisition
        self.recon_panel.current_recon = None
        self.fov_mm = entry['fov_mm']
        self.sino_range_mm = entry['sino_range_mm']
        self.last_source = entry['source']
        self.detector_config = entry['detector_config']
        self.sinograms.append(entry)
        self.current_sino_name = entry['name']
        self._q.put(('sino_refresh', entry['name']))
        self._q.put(('persist', None))

    @staticmethod
    def _unique_name(name, existing):
        if name not in existing:
            return name
        i = 2
        while f"{name}_{i}" in existing:
            i += 1
        return f"{name}_{i}"

    def get_sinogram(self, name):
        for s in self.sinograms:
            if s["name"] == name:
                return s
        return None

    def add_reconstruction(self, entry):
        entry["name"] = self._unique_name(
            entry.get("name", "recon"), [r["name"] for r in self.reconstructions])
        self.reconstructions.append(entry)
        self._q.put(("recon_list_refresh", None))
        return entry["name"]

    def set_volume(self, vol, method):
        self.volume = vol
        self.last_method = method
        self._q.put(("vol_refresh", None))

    def _publish_reconstruction(self, generation, rec, source, method):
        if generation != self._study_generation:
            self.log('[STALE] Discarded reconstruction from previous acquisition')
            return
        self.volume = rec['volume']
        self.last_source = source
        self.last_method = method
        self.vol_zoom = 1
        self.recon_panel.current_recon = rec
        self.add_reconstruction(rec)
        self._q.put(('vol_refresh', None))

    def _drain(self):
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "log":
                    self.log_text.configure(state="normal")
                    self.log_text.insert(tk.END, payload + "\n")
                    self.log_text.see(tk.END)
                    self.log_text.configure(state="disabled")
                elif kind == "error":
                    messagebox.showerror("Error", payload)
                elif kind == "task_done":
                    if self._active_task == payload:
                        self._active_task = None
                elif kind == "new_sinogram":
                    self._register_sinogram(payload)
                elif kind == 'calibration_ready':
                    par, output = payload
                    self.petsys.calibration = par
                    self.petsys.var_cal.set(output)
                    self.log(f'[CAL] calibration saved: {output}')
                    messagebox.showinfo('Calibration', f'Calibration saved to:\n{output}')
                elif kind == 'reconstruction_ready':
                    self._publish_reconstruction(*payload)
                elif kind == "sino_refresh":
                    self.sino_panel.on_new()
                    self.recon_panel.refresh_sino_list(select=payload)
                    self.notebook.select(self.sino_panel)
                elif kind == "vol_refresh":
                    self.recon_panel.on_new_volume()
                    self.notebook.select(self.recon_panel)
                elif kind == "recon_list_refresh":
                    self.recon_panel.refresh_recon_list()
                    self.plot_panel.refresh_available()
                elif kind == "persist":
                    self._save_files()
        except queue.Empty:
            pass
        self.after(120, self._drain)

    # ---- image-grid resolution for the Reconstruction tab ----
    @staticmethod
    def _grid_kwargs(params, fov_mm, srange):
        """Translate the GUI 'Image grid' controls into constructor kwargs.

        'sinogram' reproduces the legacy behaviour exactly (image matrix =
        number of radial bins). The other modes define the grid explicitly.
        """
        mode = params["grid_mode"].get() if "grid_mode" in params else "sinogram"
        rays = ("auto" if (("rays_auto" not in params)
                           or params["rays_auto"].get())
                else int(params["rays_per_bin"].get()))
        chunks = int(params["bp_chunks"].get()) if "bp_chunks" in params else 1
        cor = float(params["cor_mm"].get()) if "cor_mm" in params else 0.0
        kw = dict(fov_mm=fov_mm, sino_range_mm=srange,
                  rays_per_bin=rays, backproject_chunks=chunks,
                  cor_mm=cor)
        if mode == "sinogram":
            return kw
        n_img = int(params["n_img"].get())
        voxel = float(params["voxel_mm"].get())
        if mode == "matrix":
            kw["n_img"] = n_img
        elif mode == "voxel":
            if voxel <= 0:
                raise ValueError("voxel (mm) must be > 0 in 'voxel' grid mode")
            kw["voxel_mm"] = voxel
        elif mode == "matrix+voxel":
            if voxel <= 0:
                raise ValueError("voxel (mm) must be > 0 in 'matrix+voxel' mode")
            kw["n_img"] = n_img
            kw["voxel_mm"] = voxel
            kw["fov_mm"] = None       # FOV becomes n_img * voxel
        else:
            raise ValueError(f"unknown grid mode: {mode!r}")
        return kw

    # ---- reconstruction (from the Reconstruction tab) ----
    def run_recon(self, params, sino_name=None, recon_name=None):
        try:
            from Gimntools.GimnRec.reconstructor import siddon_reconstructor
            from Gimntools.GimnRec.filters import ramLak, make_psf_kernel
            entry = self.get_sinogram(sino_name) if sino_name else None
            if entry is None and self.sinograms:
                entry = self.sinograms[-1]
            if entry is None:
                self.error("No sinogram to reconstruct.")
                return
            generation = self._study_generation
            sino = entry["sino"]; angles = entry["angles"]
            fov_mm = entry["fov_mm"]; srange = entry["sino_range_mm"]
            cfg = entry.get("detector_config"); source = entry["source"]
            norm = entry.get("norm")
            background = entry.get('background')
            method = params["method"].get()
            self.log(f"[{entry['name']}] reconstructing ({method}"
                     f"{', geom-norm' if norm is not None else ''})… "
                     "(first time compiles Numba)")
            sino_for_r = sino
            if method == 'fbp' and background is not None:
                sino_for_r = sino - background
            if method == "fbp" and norm is not None:
                from Gimntools.GimnRec.normalization import apply_norm_to_sinogram
                sino_for_r = apply_norm_to_sinogram(sino_for_r, norm)
            # Radial resampling of the PROJECTIONS (FBP only). Applied after
            # the flood normalization, which is defined per measured
            # detector bin and would be meaningless on a resampled axis.
            interp_kind, interp_k = "none", 1
            if method == "fbp":
                interp_kind = params["fbp_interp"].get()
                interp_k = max(1, int(params["fbp_interp_factor"].get()))
                if interp_kind != "none" and interp_k > 1:
                    from Gimntools.GimnRec.filters import interpolate_distance
                    n_out = interp_k * sino_for_r.shape[1]
                    sino_for_r = interpolate_distance(
                        sino_for_r, n_out=n_out, kind=interp_kind,
                        align="centers")
                    self.log(f"[{entry['name']}] radial interpolation "
                             f"'{interp_kind}' x{interp_k}: "
                             f"{sino.shape[1]} -> {n_out} bins over the same "
                             f"{srange:.3f} mm (bin "
                             f"{srange/n_out:.4f} mm). This resamples the "
                             f"projections; it is NOT an interpolating "
                             f"back-projector.")
                    if params["grid_mode"].get() == "sinogram":
                        self.log(f"[{entry['name']}] NOTE: grid mode is "
                                 f"'sinogram', so the image matrix follows "
                                 f"the radial bins and also becomes "
                                 f"{n_out}x{n_out}. Use grid mode "
                                 f"'matrix'/'voxel' to interpolate the "
                                 f"projections without changing the volume.")
            grid_kw = self._grid_kwargs(params, fov_mm, srange)
            grid_kw['dx_cor_mm'] = entry.get('dx_cor_mm', 0.0)
            grid_kw['dy_cor_mm'] = entry.get('dy_cor_mm', 0.0)
            r = siddon_reconstructor(sino_for_r, **grid_kw)
            self.log(f"[{entry['name']}] grid: {r.n_img}x{r.n_img} px @ "
                     f"{r.voxel_mm:.4f} mm | radial bin {r.sino_pixel:.4f} mm "
                     f"| rays/bin {r.rays_per_bin} | COR {r.cor_mm:+.4f} mm "
                     f"({r.cor_mm/r.sino_pixel:+.3f} bins)")
            psf = None
            if params["use_psf"].get():
                psf = make_psf_kernel(float(params["fwhm"].get()),
                                      srange / sino.shape[1])
            if method == "osem":
                vol = r.osem(angles, iterations=int(params["iters"].get()),
                             subsets_n=int(params["subsets"].get()), psf_kernel=psf,
                             norm_sino=norm, background_sino=background,
                             regularization_every=int(params["gevery"].get()),
                             regularization_sigma=float(params["gsigma"].get()),
                             legacy=params["legacy"].get(), verbose=False)
            elif method == "mlem":
                vol = r.mlem(angles, iterations=int(params["iters"].get()),
                             psf_kernel=psf, norm_sino=norm,
                             background_sino=background,
                             regularization_every=int(params["gevery"].get()),
                             regularization_sigma=float(params["gsigma"].get()),
                             legacy=params["legacy"].get(), verbose=False)
            else:
                from Gimntools.GimnRec.filters import make_fbp_filter
                window = params["fbp_window"].get()
                cutoff = float(params["fbp_cutoff"].get())
                order = int(params["fbp_order"].get())
                hard = bool(params["fbp_hard_cut"].get())
                pad_mode = params["fbp_pad_mode"].get()
                pad_factor = int(params["fbp_pad_factor"].get())
                legacy_scale = bool(params["fbp_legacy_scale"].get())
                flt = make_fbp_filter(window, cutoff=cutoff, order=order,
                                      hard_cutoff=hard)
                n_pad = resolve_pad_length(r.n_dist, pad_factor, pad_mode)
                nyq_mm = 1.0 / (2.0 * r.sino_pixel)
                self.log(
                    f"[{entry['name']}] FBP filter: {window}"
                    + (f" (order {order})" if window == "butterworth" else "")
                    + f", cutoff {cutoff:.3g}x Nyquist = "
                      f"{cutoff * nyq_mm:.3f} mm^-1 "
                      f"({cutoff * nyq_mm * 10:.2f} cycles/cm)"
                    + ("" if window in ("ramp", "butterworth")
                       else f", window {'rescaled onto' if hard else 'shaped to Nyquist, truncated at'} the cutoff")
                    + f" | zero-padding '{pad_mode}' x{pad_factor} -> "
                      f"{n_pad} bins (from {r.n_dist})"
                    + f" | radial interp '{interp_kind}' x{interp_k}"
                    + (" | LEGACY scale" if legacy_scale else ""))
                vol = r.fbp(angles, filter_type=flt, pad_factor=pad_factor,
                            pad_mode=pad_mode, legacy_scale=legacy_scale)
            zoom = int(params["zoom"].get()) if "zoom" in params else 1
            if zoom > 1:
                from scipy.ndimage import zoom as ndzoom
                vol = ndzoom(vol, (zoom, 1, 1), order=1)
                self.log(f"[{entry['name']}] Z upsample x{zoom} (interpolation) -> {vol.shape}")
            # physical spacing (upsample already folded into dz).
            # dxy comes from the RESOLVED grid, not from fov/n_dist: the two
            # are independent once a volume-driven grid is selected.
            dxy = r.voxel_mm; dz = dxy
            try:
                dz = float(np.asarray(cfg["sipm_pitch"])[1])
            except Exception:
                dz = dxy
            dz = dz / zoom
            nm = (recon_name or "").strip() or f"{entry['name']}_{method}"
            rec = {"name": nm, "volume": vol, "spacing": (dxy, dxy, dz),
                   "source": entry["name"], "method": method,
                   "geometry": r.geometry(),
                   "info": f"{entry['name']} · {method} · {vol.shape} · "
                           f"COR {r.cor_mm:+.3f} mm"}
            if method == "fbp":
                rec["fbp"] = {
                    "window": window, "cutoff_nyquist": cutoff,
                    "cutoff_mm_inv": cutoff / (2.0 * r.sino_pixel),
                    "order": order, "hard_cutoff": hard,
                    "pad_mode": pad_mode, "pad_factor": pad_factor,
                    "pad_length": n_pad,
                    "interp": interp_kind, "interp_factor": interp_k,
                    "legacy_scale": legacy_scale,
                }
                rec["info"] += (f" · {window} @ {cutoff:.3g}xNyq"
                                + (f" · interp {interp_kind} x{interp_k}"
                                   if interp_kind != "none" else ""))
            if threading.current_thread() is threading.main_thread():
                self._publish_reconstruction(generation, rec, source, method)
            else:
                self._q.put(('reconstruction_ready', (generation, rec, source, method)))
            self.log(f"[{nm}] reconstruction ready: {vol.shape}, "
                     f"voxel {dxy:.3f}×{dxy:.3f}×{dz:.3f} mm")
        except Exception:
            self.error(traceback.format_exc())


def main():
    ReconUI().mainloop()


if __name__ == "__main__":
    main()