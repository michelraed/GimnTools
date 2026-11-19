"""Browse EM subset images and export the masks used in convergence metrics."""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.pyplot import get_cmap
import numpy as np

from Gimntools.GimnRec.iteration_history import grid_labels
from Gimntools.GimnRec.ui.rendering import plot_canvas, show_image
from Gimntools.GimnRec.ui.session import SessionToolbar


_PLANES = {"Axial": (0, 0, 1), "Coronal": (1, 0, 2), "Sagittal": (2, 1, 2)}


def _plane_mask(region, shape, axis, index, field):
    nz, ny, nx = shape
    flat = getattr(region, field)
    if axis == 0:
        mask = np.zeros((ny, nx), dtype=bool)
        if index in region.axial:
            mask.flat[flat[region.axial.index(index)]] = True
        return mask
    mask = np.zeros((nz, nx if axis == 1 else ny), dtype=bool)
    for z, pixels in zip(region.axial, flat):
        y, x = np.divmod(pixels, nx)
        if axis == 1:
            mask[z, x[y == index]] = True
        else:
            mask[z, y[x == index]] = True
    return mask


def _mask_edges(mask, x0, y0, width, height):
    """Return physical edges where adjacent pixels differ in ROI membership."""
    boundaries = []
    for y, x in np.argwhere(np.diff(np.pad(mask, ((0, 0), (1, 1))), axis=1)):
        px, py = x0 + x * width, y0 + y * height
        boundaries.append(((px, py), (px, py + height)))
    for y, x in np.argwhere(np.diff(np.pad(mask, ((1, 1), (0, 0))), axis=0)):
        px, py = x0 + x * width, y0 + y * height
        boundaries.append(((px, py), (px + width, py)))
    return boundaries


class IterationPanel(ttk.Frame):
    MODES = ("Iterations", "Source ROIs")

    def __init__(self, parent, app):
        super().__init__(parent, padding=8)
        self.app = app
        self.histories = {}
        self.result = tk.StringVar(self)
        self.mode = tk.StringVar(self, self.MODES[0])
        self.case = tk.StringVar(self)
        self.subsets = tk.StringVar(self)
        self.iteration = tk.StringVar(self)
        self.subset = tk.StringVar(self)
        self.source = tk.StringVar(self, "All")
        self.plane = tk.StringVar(self, "Axial")
        self.slice_index = tk.IntVar(self, 0)
        self.fixed_scale = tk.BooleanVar(self, True)
        self.info = tk.StringVar(self)
        self._frame_key = self._limit_key = None
        self._frame = None
        self._limit = None

        row = ttk.Frame(self)
        row.pack(fill=tk.X)
        self.result_combo = self._combo(row, "Result", self.result, (), self._result_changed, width=29)
        self._combo(row, "Show", self.mode, self.MODES, self._result_changed, width=12)
        ttk.Button(row, text="Save figure...", command=self._save).pack(side=tk.RIGHT)

        self.run_row = ttk.Frame(self)
        self.case_combo = self._combo(self.run_row, "COR case", self.case, (), self._case_changed, width=15)
        self.subsets_combo = self._combo(self.run_row, "Subsets per iteration", self.subsets, (),
                                        self._run_changed, width=4)
        self.iteration_combo = self._combo(self.run_row, "Iteration", self.iteration, (), self.redraw, width=6)
        self.subset_combo = self._combo(self.run_row, "Subset", self.subset, (), self.redraw, width=5)
        self.roi_row = ttk.Frame(self)
        self.roi_combo = self._combo(self.roi_row, "Source", self.source, ("All",), self._plane_changed, width=8)
        self.view_row = ttk.Frame(self)
        self.view_row.pack(fill=tk.X, pady=4)
        self._combo(self.view_row, "Plane", self.plane, tuple(_PLANES), self._plane_changed, width=10)
        ttk.Label(self.view_row, text="Slice").pack(side=tk.LEFT)
        self.slider = ttk.Scale(self.view_row, from_=0, to=1, command=self._slice_changed)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 8))
        ttk.Checkbutton(self.view_row, text="Fixed intensity scale", variable=self.fixed_scale,
                        command=self.redraw).pack(side=tk.LEFT)
        ttk.Label(self, textvariable=self.info, style="Hint.TLabel").pack(anchor="w")

        self.figure = Figure(figsize=(6, 5), dpi=100, constrained_layout=True)
        self.canvas = plot_canvas(app, self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        SessionToolbar(self.canvas, self).update()
        self._result_changed()

    @staticmethod
    def _combo(parent, label, var, values, command, *, width):
        ttk.Label(parent, text=label).pack(side=tk.LEFT, padx=(0, 4))
        combo = ttk.Combobox(parent, textvariable=var, values=values, width=width, state="readonly")
        combo.pack(side=tk.LEFT, padx=(0, 9))
        combo.bind("<<ComboboxSelected>>", lambda _event: command())
        return combo

    @staticmethod
    def _choices(combo, values):
        values = tuple(map(str, values))
        combo.configure(values=values)
        if combo.get() not in values:
            combo.set(values[0] if values else "")

    def add_history(self, name, history):
        self.histories[name] = history
        self._choices(self.result_combo, self.histories)
        self.select_result(name)

    def select_result(self, name):
        if name in self.histories:
            self.result.set(name)
            self._result_changed()

    def close_histories(self):
        for history in self.histories.values():
            history.close()
        self.histories.clear()

    def _result_changed(self):
        history = self.histories.get(self.result.get())
        self.run_row.pack_forget()
        self.roi_row.pack_forget()
        if history is None:
            self.redraw()
            return
        self._choices(self.roi_combo, ["All", *range(1, len(history.regions) + 1)])
        if not history.runs:
            self.mode.set(self.MODES[1])
        if self.mode.get() == self.MODES[1]:
            self.roi_row.pack(fill=tk.X, pady=(4, 0), before=self.view_row)
            self._plane_changed()
        else:
            self.run_row.pack(fill=tk.X, pady=(4, 0), before=self.view_row)
            self._choices(self.case_combo, dict.fromkeys(case for case, _ in history.runs))
            self._case_changed()

    def _case_changed(self):
        history = self.histories[self.result.get()]
        self._choices(self.subsets_combo, [count for case, count in history.runs if case == self.case.get()])
        self._run_changed()

    def _run_changed(self):
        series = self._series()
        self._choices(self.iteration_combo, range(1, series.iterations + 1))
        self._choices(self.subset_combo, range(1, series.subsets + 1))
        self._plane_changed()

    def _series(self):
        history = self.histories.get(self.result.get())
        if history is None:
            return None
        if self.mode.get() == self.MODES[1]:
            return history.reference
        return history.runs.get((self.case.get(), int(self.subsets.get())))

    def _plane_changed(self):
        series = self._series()
        if series is None:
            return
        axis = _PLANES[self.plane.get()][0]
        history = self.histories[self.result.get()]
        if self.source.get() == "All":
            source = history.regions[0]
        else:
            source = history.regions[int(self.source.get()) - 1]
        best = source.center_zyx[axis]
        self.slider.configure(to=max(series.shape[axis] - 1, 0))
        self.slice_index.set(best)
        self.slider.set(best)
        self.redraw()

    def _slice_changed(self, value):
        self.slice_index.set(round(float(value)))
        self.redraw()

    def _draw(self, figure, *, desktop):
        figure.clear()
        series = self._series()
        if series is None:
            figure.suptitle("Run a convergence sweep to view iterations and source ROIs.")
            return
        axis, h, v = _PLANES[self.plane.get()]
        index = min(series.shape[axis] - 1, max(0, self.slice_index.get()))
        is_roi = self.mode.get() == self.MODES[1]
        iteration, subset = (1, 1) if is_roi else (int(self.iteration.get()), int(self.subset.get()))
        frame_key = (series.path, iteration, subset)
        if frame_key != self._frame_key:
            self._frame = series.read(iteration, subset)
            self._frame_key = frame_key
        volume = self._frame
        image = np.take(volume, index, axis=axis)
        width, height = series.spacing[h], series.spacing[v]
        span_x, span_y = image.shape[1] * width, image.shape[0] * height
        extent = (-span_x / 2, span_x / 2, -span_y / 2, span_y / 2)
        if self.fixed_scale.get() and not is_roi:
            if self._limit_key != series.path:
                final = series.read(series.iterations, series.subsets)
                self._limit = max(float(np.max(final)), float(np.max(volume)), np.finfo(np.float32).tiny)
                self._limit_key = series.path
            vmax = self._limit
        else:
            vmax = max(float(image.max()), np.finfo(np.float32).tiny)
        axes = figure.subplots()
        renderer = show_image if desktop else lambda _app, ax, data, **kw: ax.imshow(data, **kw)
        renderer(self.app, axes, image, extent=extent, cmap="gray", origin="lower",
                 interpolation="nearest", vmin=0, vmax=vmax)
        axes.set(xlabel=f"{'XYZ'[h]} (mm)", ylabel=f"{'XYZ'[v]} (mm)")
        label = f"{self.plane.get()} · slice {index + 1}/{series.shape[axis]}"
        grid = grid_labels(series.shape, series.spacing)
        if is_roi:
            self._draw_regions(axes, series, axis, index, extent, width, height)
            axes.set_title(f"{self.result.get()} · MLEM reference\n{label}")
        else:
            axes.set_title(f"{self.result.get()} · {self.case.get()}\n"
                           f"Iteration {iteration}/{series.iterations}, subset {subset}/{series.subsets} · {label}")
        if desktop:
            self.info.set(f"{grid['matrix_xyz']} voxels · {grid['voxel_xyz_mm']} mm · {label}")

    def _draw_regions(self, axes, series, axis, index, extent, width, height):
        history = self.histories[self.result.get()]
        cmap = get_cmap("tab10")
        handles = []
        for number, region in enumerate(history.regions, 1):
            if self.source.get() not in ("All", str(number)):
                continue
            color = cmap((number - 1) % 10)
            for field, style, label in (("foreground", "-", "source"), ("background", "--", "background")):
                mask = _plane_mask(region, series.shape, axis, index, field)
                edges = _mask_edges(mask, extent[0], extent[2], width, height)
                if edges:
                    axes.add_collection(LineCollection(edges, colors=[color], linestyles=style, linewidths=1.35))
                    handles.append(Line2D([], [], color=color, linestyle=style, label=f"{number} · {label}"))
        if handles:
            axes.legend(handles=handles, loc="upper right", fontsize=8)

    def redraw(self):
        try:
            self._draw(self.figure, desktop=True)
            self.canvas.draw_idle()
        except (OSError, ValueError, tk.TclError) as exc:
            self.app.error(f"Iteration viewer: {exc}")

    def _save(self):
        if self._series() is None:
            return
        kind = "rois" if self.mode.get() == self.MODES[1] else "iteration"
        default = f"{Path(self.result.get()).name}-{kind}.png"
        path = self.app.session.choose(
            "figure.save", filedialog.asksaveasfilename, parent=self,
            defaultextension=".png", initialfile=default,
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
        )
        if path:
            try:
                figure = Figure(figsize=(7, 6), constrained_layout=True, facecolor="white")
                self._draw(figure, desktop=False)
                if self.figure.axes:
                    figure.axes[0].set(xlim=self.figure.axes[0].get_xlim(), ylim=self.figure.axes[0].get_ylim())
                figure.savefig(path, dpi=300)
            except (OSError, ValueError, tk.TclError) as exc:
                self.app.error(f"Iteration viewer: {exc}")
