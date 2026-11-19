"""Shared list-mode reconstruction and EM convergence controls."""

from pathlib import Path
from types import SimpleNamespace
import re
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure

from Gimntools.GimnRec.convergence import (
    convergence_operating_point, convergence_series, detect_source_regions, save_convergence_csv,
)
from Gimntools.GimnRec.corrections import estimate_sinogram_cor
from Gimntools.GimnRec.listmode import load_listmode, reconstruct_listmode
from Gimntools.GimnRec.ui.rendering import plot_canvas


def _field(parent, label, var, values=None):
    row = ttk.Frame(parent)
    row.pack(fill=tk.X, pady=2)
    ttk.Label(row, text=label).pack(side=tk.LEFT, padx=(0, 8))
    if values is None:
        widget = ttk.Entry(row, textvariable=var, width=15)
    else:
        widget = ttk.Combobox(row, textvariable=var, values=values, width=15, state="readonly")
    widget.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    return widget


def _numbers(value, length, convert, label):
    try:
        result = tuple(convert(part) for part in re.split(r"[\s,;]+", value.strip()))
        if length is not None and len(result) != length:
            raise ValueError
    except ValueError as exc:
        raise ValueError(f"Enter {label} as numbers separated by spaces or commas") from exc
    return result


class ListmodePanel(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=12)
        self.app = app
        self.path = tk.StringVar(self)
        self.matrix = tk.StringVar(self, "75 75 31")
        self.voxel = tk.StringVar(self, "0.84 0.84 0.84")
        self.center = tk.StringVar(self, "0 0 0")
        self.energy_enabled = tk.BooleanVar(self, False)
        self.energy = tk.StringVar(self, "450 650")
        self.iterations = tk.IntVar(self, 5)
        self.subsets = tk.IntVar(self, 1)
        self.relaxation = tk.DoubleVar(self, 0.8)
        self.ratio_cap = tk.DoubleVar(self, 1e6)
        self.result = tk.StringVar(self)

        ttk.Label(self, text="3D list-mode reconstruction", style="Emphasis.TLabel").pack(anchor="w", pady=(0, 8))
        path_row = ttk.Frame(self)
        path_row.pack(fill=tk.X)
        ttk.Label(path_row, text="Coincidences NPZ").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Entry(path_row, textvariable=self.path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_row, text="Browse...", command=self._browse).pack(side=tk.LEFT, padx=4)
        ttk.Label(self, text="x1, y1, z1, E1, x2, y2, z2, E2 · millimetres and keV",
                  style="Hint.TLabel").pack(anchor="w", pady=(2, 8))
        controls = ttk.LabelFrame(self, text="Volume and EM settings", padding=8)
        controls.pack(anchor="nw", ipadx=5)
        _field(controls, "Matrix X Y Z", self.matrix)
        _field(controls, "Voxel X Y Z (mm)", self.voxel)
        _field(controls, "FOV centre X Y Z (mm)", self.center)
        ttk.Checkbutton(controls, text="Filter both energies", variable=self.energy_enabled).pack(anchor="w")
        _field(controls, "Energy low high (keV)", self.energy)
        _field(controls, "Iterations", self.iterations)
        _field(controls, "Subsets (1 = MLEM)", self.subsets)
        _field(controls, "EM relaxation", self.relaxation)
        _field(controls, "EM ratio cap", self.ratio_cap)
        ttk.Label(self, text="A calibrated sensitivity map or normalization LORs can be included in the NPZ. "
                  "Otherwise the model uses uniform sensitivity.",
                  wraplength=600, style="Hint.TLabel").pack(anchor="w", pady=9)
        ttk.Button(self, text="Reconstruct list mode", style="Primary.TButton",
                   command=self._start).pack(anchor="w")
        ttk.Label(self, textvariable=self.result, wraplength=670).pack(anchor="w", pady=10)

    def _browse(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("List-mode NPZ", "*.npz")])
        if path:
            self.path.set(path)
            self.app.convergence_panel.refresh()

    def settings(self):
        path = self.path.get().strip()
        if not path:
            raise ValueError("Select a list-mode NPZ file")
        shape = _numbers(self.matrix.get(), 3, int, "matrix X Y Z")[::-1]
        spacing = _numbers(self.voxel.get(), 3, float, "voxel X Y Z")
        center = _numbers(self.center.get(), 3, float, "FOV centre X Y Z")
        energy = _numbers(self.energy.get(), 2, float, "energy low high") if self.energy_enabled.get() else None
        return Path(path), shape, spacing, energy, dict(center=center, relaxation=self.relaxation.get(),
                                                          ratio_cap=self.ratio_cap.get())

    def _start(self):
        try:
            path, shape, spacing, energy, options = self.settings()
            iterations, subsets = self.iterations.get(), self.subsets.get()
        except (ValueError, tk.TclError) as exc:
            self.app.error(str(exc))
            return

        def run():
            try:
                rays, metadata = load_listmode(path, energy)
                calibration = {k: metadata.pop(k) for k in
                               ("sensitivity", "normalization_lors", "normalization_weights")}
                volume, geometry = reconstruct_listmode(
                    rays, shape, spacing, iterations=iterations, subsets=subsets,
                    progress=lambda it: self.app.log(f"[List mode] {path.stem} · iteration {it}/{iterations}"),
                    **options, **calibration)
                geometry.update(metadata)
                method = "LM-MLEM" if subsets == 1 else "LM-OSEM"
                info = (f"{path.stem} · {method} · {geometry['traced_events']:,} of "
                        f"{geometry['total_events']:,} coincidences traced · {geometry['sensitivity']}")
                entry = {"name": f"{path.stem}_{method.lower()}", "source": path.stem, "method": method,
                         "volume": volume, "spacing": spacing, "origin": geometry["origin"],
                         "geometry": geometry, "info": info}
                self.app._q.put(("listmode_ready", entry))
                self.app.log(info)
            except Exception as exc:
                self.app.error(f"{path.name}: {exc}")

        if self.app.start_worker("list-mode reconstruction", run):
            self.result.set(f"Reconstructing {path.name}...")


class ConvergencePanel(ttk.Frame):
    LISTMODE = "List mode (selected NPZ)"

    def __init__(self, parent, app):
        super().__init__(parent, padding=8)
        self.app = app
        self.results = {}
        self.source = tk.StringVar(self)
        self.history = tk.StringVar(self)
        self.iterations = tk.StringVar(self, "1, 2, 4, 6, 8, 12")
        self.subsets = tk.StringVar(self, "1, 2, 4")
        self.radius = tk.DoubleVar(self, 2.0)
        self.source_count = tk.IntVar(self, 0)
        self.threshold = tk.DoubleVar(self, 25.)
        self.search_radius = tk.DoubleVar(self, 0.)
        self.compare_cor = tk.BooleanVar(self, False)
        self.status = tk.StringVar(self)
        self.x_axis = tk.StringVar(self, "Iterations")
        self.metric = tk.StringVar(self, "Contrast (%)")
        self.title = tk.StringVar(self)
        self.font = tk.StringVar(self, "sans-serif")
        self.legend = tk.StringVar(self, "best")
        self.style = tk.StringVar(self, "Lines + points")
        self.palette = tk.StringVar(self, "tab10")
        self.dpi = tk.IntVar(self, 200)
        self.selected_source = tk.StringVar(self, "All")

        side = ttk.Notebook(self)
        side.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        study = ttk.Frame(side, padding=8)
        style = ttk.Frame(side, padding=8)
        side.add(study, text="Sweep")
        side.add(style, text="Chart")
        self.source_selector = _field(study, "Acquisition", self.source, ())
        _field(study, "Iterations", self.iterations)
        _field(study, "Subsets", self.subsets)
        _field(study, "Source ROI (mm)", self.radius)
        _field(study, "Source count (0 = auto)", self.source_count)
        _field(study, "Peak threshold (%)", self.threshold)
        _field(study, "Search radius (mm, 0 = auto)", self.search_radius)
        ttk.Checkbutton(study, text="Compare estimated COR", variable=self.compare_cor).pack(anchor="w", pady=3)
        ttk.Label(study, text="Uses the current reconstruction grid and detector response. "
                  "A short MLEM run locates fixed source and background regions.",
                  style="Hint.TLabel", wraplength=250).pack(anchor="w", pady=7)
        ttk.Button(study, text="Run convergence sweep", style="Primary.TButton",
                   command=self._start).pack(fill=tk.X)
        ttk.Label(study, textvariable=self.status, wraplength=250).pack(fill=tk.X, pady=8)
        _field(style, "X axis", self.x_axis, ("Iterations", "Subset updates"))
        _field(style, "Signal metric", self.metric, ("Contrast (%)", "CNR"))
        self.source_view = _field(style, "Source", self.selected_source, ("All",))
        _field(style, "Title", self.title)
        _field(style, "Font", self.font, ("sans-serif", "serif", "monospace"))
        _field(style, "Legend", self.legend, ("best", "upper right", "upper left", "lower right", "off"))
        _field(style, "Line style", self.style, ("Lines + points", "Lines", "Points"))
        _field(style, "Colours", self.palette, ("tab10", "Dark2", "Set2", "viridis"))
        _field(style, "Export DPI", self.dpi)
        ttk.Button(style, text="Apply chart settings", command=self.redraw).pack(fill=tk.X, pady=4)

        right = ttk.Frame(self)
        right.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        results = ttk.Frame(right)
        results.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(results, text="Result").pack(side=tk.LEFT)
        self.history_combo = ttk.Combobox(results, textvariable=self.history, width=30, state="readonly")
        self.history_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.history_combo.bind("<<ComboboxSelected>>", lambda _event: self._select_result())
        ttk.Button(results, text="Save CSV...", command=self._save_csv).pack(side=tk.RIGHT)
        ttk.Button(results, text="Save figure...", command=self._save_figure).pack(side=tk.RIGHT, padx=5)
        self.figure = Figure(figsize=(7, 5), dpi=100, constrained_layout=True)
        self.canvas = plot_canvas(app, self.figure, right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, right).update()
        self.bind("<Visibility>", lambda _event: self.refresh())
        self.redraw()

    def refresh(self):
        names = [entry["name"] for entry in self.app.sinograms]
        names.append(self.LISTMODE)
        self.source_selector.configure(values=names)
        if self.source.get() not in names:
            self.source.set(self.app.current_sino_name or names[0])
        elif self.source.get() == self.LISTMODE and len(names) > 1 and not self.app.listmode_panel.path.get():
            self.source.set(self.app.current_sino_name or names[0])

    def _start(self):
        try:
            name = self.source.get()
            iterations = _numbers(self.iterations.get(), None, int, "iterations")
            subsets = _numbers(self.subsets.get(), None, int, "subsets")
            if min(iterations + subsets) < 1:
                raise ValueError("Iterations and subsets must be positive")
            options = dict(iterations=iterations, subsets=subsets, radius_mm=self.radius.get(),
                           source_count=self.source_count.get(), threshold=self.threshold.get() / 100,
                           search_radius_mm=self.search_radius.get())
            compare = self.compare_cor.get()
            if name == self.LISTMODE:
                if compare:
                    raise ValueError("Estimated COR comparison requires a sinogram")
                spec = self.app.listmode_panel.settings()
            else:
                entry = self.app.get_sinogram(name)
                if entry is None:
                    raise ValueError("Load a sinogram or select a list-mode NPZ")
                snapshot = self.app.recon_panel._snapshot()
                if name != self.app.recon_panel.var_sino.get():
                    self._use_acquisition_response(snapshot, entry)
        except (ValueError, tk.TclError) as exc:
            self.app.error(str(exc))
            return

        def run():
            try:
                if name == self.LISTMODE:
                    output_name, rows, regions, summary = self._run_listmode(spec, options)
                else:
                    output_name = name
                    rows, regions, summary = self._run_sinogram(entry, snapshot, options, compare)
                self.app._q.put(("convergence_ready", (output_name, rows, regions, summary)))
            except Exception as exc:
                self.app.error(f"Convergence: {exc}")
                self.app._q.put(("convergence_failed", None))

        if self.app.start_worker("EM convergence", run):
            self.status.set("Computing the reference and iteration checkpoints...")

    @staticmethod
    def _use_acquisition_response(snapshot, entry):
        from Gimntools.GimnRec.detector_response import CRYSTAL_MU_511, DEFAULT_FACE_WIDTH_MM, crystal_material

        config = entry["detector_config"] or {}
        if snapshot["response_mode"].get() in ("DOI-aware", "DOI + residual SV response",
                                               "Analytical detector response"):
            material = crystal_material(config.get("crystal_material"))
            snapshot.update({
                "doi_material": SimpleNamespace(get=lambda: material),
                "doi_mu": SimpleNamespace(get=lambda: config.get("mu_511_mm_inv", CRYSTAL_MU_511.get(material, ""))),
                "doi_face": SimpleNamespace(get=lambda: config.get("crystal_face_width_mm", DEFAULT_FACE_WIDTH_MM)),
            })

    def _run_sinogram(self, entry, snapshot, options, compare):
        snapshot["method"] = SimpleNamespace(get=lambda: "mlem")
        current_cor = entry["cor_mm"]
        snapshot["cor_mm"] = SimpleNamespace(get=lambda: current_cor)
        rows, regions = self.app.run_recon(snapshot, entry["name"],
                                          sweep=dict(options, case="Saved COR"))
        summary = f"Saved COR {current_cor:+.3f} mm."
        if compare:
            candidate, info = estimate_sinogram_cor(entry["sino"], entry["sino_range_mm"],
                                                    entry["angles"], background=entry["background"])
            snapshot["cor_mm"] = SimpleNamespace(get=lambda: candidate)
            extra, _ = self.app.run_recon(snapshot, entry["name"],
                                          sweep=dict(options, case="Estimated COR", regions=regions))
            rows.extend(extra)
            summary += f" Estimate {candidate:+.3f} mm ± {info['stability_mm']:.3f} mm (angular variation)."
        return rows, regions, summary

    @staticmethod
    def _run_listmode(spec, options):
        path, shape, spacing, energy, settings = spec
        rays, meta = load_listmode(path, energy)
        calibration = {k: meta[k] for k in ("sensitivity", "normalization_lors", "normalization_weights")}

        def run(iterations, subsets, callback=None):
            def visit(iteration, volume):
                for z, image in enumerate(volume):
                    callback(z, iteration, image)

            return reconstruct_listmode(rays, shape, spacing, iterations=iterations, subsets=subsets,
                                        checkpoint=visit if callback else None, **settings, **calibration)

        reference, geometry = run(min(4, max(options["iterations"])), 1)
        regions = detect_source_regions(reference, spacing, radius_mm=options["radius_mm"],
                                        source_count=options["source_count"],
                                        relative_threshold=options["threshold"], support=reference > 0,
                                        search_radius_mm=options["search_radius_mm"] or None)
        rows = convergence_series(run, regions, options["iterations"], options["subsets"],
                                  cor_mm=None, label="List mode")
        return path.stem, rows, regions, str(geometry["sensitivity"])

    def show_result(self, name, rows, regions, summary):
        result_name = self.app._unique_name(name, self.results)
        self.results[result_name] = rows, len(regions), summary
        self.history_combo.configure(values=list(self.results))
        self.history.set(result_name)
        self._select_result()

    def _select_result(self):
        rows, count, summary = self.results[self.history.get()]
        point = convergence_operating_point(rows)
        suggestion = f" Suggested checkpoint: {point[0]} iteration(s), {point[1]} subset(s)." if point else ""
        self.status.set(f"Detected sources: {count}. {summary}{suggestion}")
        self.source_view.configure(values=["All", *map(str, range(1, count + 1))])
        self.selected_source.set("All")
        self.redraw()

    def redraw(self):
        from matplotlib.pyplot import get_cmap

        self.figure.clear()
        signal, noise = self.figure.subplots(2, 1, sharex=True)
        result = self.results.get(self.history.get())
        if result is None:
            self.figure.suptitle("Run a convergence sweep to compare EM settings.")
        else:
            rows, _, _ = result
            group = {}
            for row in rows:
                if self.selected_source.get() in ("All", str(row["source"])):
                    group.setdefault((row["case"], row["source"], row["subsets"]), []).append(row)
            contrast = "cnr" if self.metric.get() == "CNR" else "contrast_pct"
            key = "updates" if self.x_axis.get() == "Subset updates" else "iterations"
            cmap = get_cmap(self.palette.get())
            marker = None if self.style.get() == "Lines" else "o"
            line = "None" if self.style.get() == "Points" else "-"
            for i, ((case, source, subsets), points) in enumerate(group.items()):
                x = [row[key] for row in points]
                color = cmap((i % 10) / 9)
                label = f"Source {source} · {subsets} subset(s) · {case}"
                signal.plot(x, [row[contrast] for row in points], linestyle=line,
                            marker=marker, markersize=3.5, color=color, label=label)
                noise.plot(x, [row["noise_pct"] for row in points], linestyle=line,
                           marker=marker, markersize=3.5, color=color)
            if group and self.legend.get() != "off":
                signal.legend(loc=self.legend.get(), prop={"family": self.font.get(), "size": 8})
        signal.set_title(self.title.get() or self.history.get(), fontfamily=self.font.get())
        signal.set_ylabel(self.metric.get(), fontfamily=self.font.get())
        noise.set_ylabel("Background noise (%)", fontfamily=self.font.get())
        noise.set_xlabel(self.x_axis.get(), fontfamily=self.font.get())
        for axes in (signal, noise):
            axes.grid(True, alpha=0.22)
            for label in axes.get_xticklabels() + axes.get_yticklabels():
                label.set_fontfamily(self.font.get())
        self.canvas.draw_idle()

    def _save_csv(self):
        result = self.results.get(self.history.get())
        if result is None:
            return
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".csv",
                                           initialfile=f"{Path(self.history.get()).name}-convergence.csv", filetypes=[("CSV", "*.csv")])
        if path:
            try:
                save_convergence_csv(path, result[0])
            except OSError as exc:
                self.app.error(str(exc))

    def _save_figure(self):
        if self.history.get() not in self.results:
            return
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".png",
                                           initialfile=f"{Path(self.history.get()).name}-convergence.png",
                                           filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if path:
            try:
                self.redraw()
                if self.app.appearance:
                    self.app.appearance.style_figure(self.figure)
                self.figure.savefig(path, dpi=self.dpi.get())
            except (OSError, ValueError, tk.TclError) as exc:
                self.app.error(str(exc))
