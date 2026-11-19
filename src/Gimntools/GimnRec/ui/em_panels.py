"""Shared list-mode reconstruction and EM convergence controls."""

from functools import partial
from pathlib import Path
from types import SimpleNamespace
import re
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from matplotlib.figure import Figure

from Gimntools.GimnRec.convergence import (
    DEFAULT_BACKGROUND_GAP_MM, convergence_operating_point, convergence_series, detect_source_regions,
    save_convergence_csv,
)
from Gimntools.GimnRec.convergence_plot import (
    NOISE_METRICS, ROBUST_CV, convergence_metrics, draw_convergence_chart,
)
from Gimntools.GimnRec.corrections import estimate_sinogram_cor
from Gimntools.GimnRec.iteration_history import IterationHistory
from Gimntools.GimnRec.listmode import CALIBRATION_FIELDS, load_listmode, reconstruct_listmode
from Gimntools.GimnRec.listmode_psf import prepare_listmode_psf
from Gimntools.GimnRec.listmode_response import prepare_listmode_response
from Gimntools.GimnRec.ui.rendering import plot_canvas
from Gimntools.GimnRec.ui.selection import BatchSelection
from Gimntools.GimnRec.ui.session import SessionToolbar


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
        self.sensitivity_floor = tk.StringVar(self)
        self.phantom_support = tk.BooleanVar(self, True)
        self.psf_mode = tk.StringVar(self, "Off")
        self.psf_fwhm = tk.StringVar(self, "1.0 1.2")
        self.psf_growth = tk.DoubleVar(self, 50.)
        self.psf_lut = tk.StringVar(self)
        self.result = tk.StringVar(self)

        ttk.Label(self, text="3D list-mode reconstruction", style="Emphasis.TLabel").pack(anchor="w", pady=(0, 8))
        path_row = ttk.Frame(self)
        path_row.pack(fill=tk.X)
        ttk.Label(path_row, text="Coincidences NPZ").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Entry(path_row, textvariable=self.path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_row, text="Browse...", command=self._browse).pack(side=tk.LEFT, padx=4)
        ttk.Label(self, text="x1, y1, z1, E1, x2, y2, z2, E2 · millimetres and keV",
                  style="Hint.TLabel").pack(anchor="w", pady=(2, 8))
        settings = ttk.Frame(self)
        settings.pack(anchor="nw", fill=tk.X)
        controls = ttk.LabelFrame(settings, text="Volume and EM settings", padding=8)
        controls.pack(side=tk.LEFT, anchor="n", ipadx=5)
        _field(controls, "Matrix X Y Z", self.matrix)
        _field(controls, "Voxel X Y Z (mm)", self.voxel)
        _field(controls, "FOV centre X Y Z (mm)", self.center)
        ttk.Checkbutton(controls, text="Filter both energies", variable=self.energy_enabled).pack(anchor="w")
        _field(controls, "Energy low high (keV)", self.energy)
        _field(controls, "Iterations", self.iterations)
        _field(controls, "Subsets (1 = MLEM)", self.subsets)
        _field(controls, "EM relaxation", self.relaxation)
        _field(controls, "EM ratio cap", self.ratio_cap)
        response = ttk.LabelFrame(settings, text="Detector response and support", padding=8)
        response.pack(side=tk.LEFT, anchor="n", padx=(12, 0), fill=tk.X, expand=True)
        _field(response, "Min. sensitivity (% of peak, blank = auto)", self.sensitivity_floor)
        ttk.Checkbutton(response, text="Use simulated phantom cylinder when available",
                        variable=self.phantom_support).pack(anchor="w")
        model = _field(response, "Residual spatially variant PSF", self.psf_mode,
                       ("Off", "3D Gaussian field", "Calibrated LUT"))
        self.fwhm_field = _field(response, "Central FWHM XY Z (mm)", self.psf_fwhm)
        self.growth_field = _field(response, "Peripheral FWHM increase (%)", self.psf_growth)
        row = ttk.Frame(response)
        row.pack(fill=tk.X, pady=2)
        self.lut_button = ttk.Button(row, text="Browse LUT…", command=self._browse_lut)
        self.lut_button.pack(side=tk.LEFT, padx=(0, 4))
        self.lut_field = ttk.Entry(row, textvariable=self.psf_lut)
        self.lut_field.pack(side=tk.LEFT, fill=tk.X, expand=True)
        model.bind("<<ComboboxSelected>>", lambda _event: self.sync_response_fields())
        self.sync_response_fields()
        ttk.Label(self, text="Pseudo-simulator files use their saved detector response and sensitivity. "
                  "Other files use NPZ calibration when supplied, or uniform sensitivity.",
                  wraplength=600, style="Hint.TLabel").pack(anchor="w", pady=9)
        ttk.Button(self, text="Reconstruct list mode", style="Primary.TButton",
                   command=self._start).pack(anchor="w")
        ttk.Label(self, textvariable=self.result, wraplength=670).pack(anchor="w", pady=10)

    def sync_response_fields(self):
        mode = self.psf_mode.get()
        for field in (self.fwhm_field, self.growth_field):
            field.configure(state="normal" if mode == "3D Gaussian field" else "disabled")
        for field in (self.lut_button, self.lut_field):
            field.configure(state="normal" if mode == "Calibrated LUT" else "disabled")

    def _browse_lut(self):
        path = self.app.session.choose("listmode.psf.open", filedialog.askopenfilename, parent=self,
                                       filetypes=[("Residual SV-PSF LUT", "*.npz")])
        if path:
            self.psf_lut.set(path)

    def _browse(self):
        path = self.app.session.choose(
            "listmode.open", filedialog.askopenfilename, parent=self, filetypes=[("List-mode NPZ", "*.npz")],
        )
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
        psf = None
        if self.psf_mode.get() == "3D Gaussian field":
            psf = dict(fwhm_mm=_numbers(self.psf_fwhm.get(), 2, float, "XY and Z FWHM"),
                       growth=self.psf_growth.get() / 100.)
        elif self.psf_mode.get() == "Calibrated LUT":
            lut = self.psf_lut.get().strip()
            if not lut:
                raise ValueError("Select a residual SV-PSF LUT for list mode")
            psf = dict(lut=lut)
        floor = self.sensitivity_floor.get().strip()
        return Path(path), shape, spacing, energy, dict(center=center, relaxation=self.relaxation.get(),
            ratio_cap=self.ratio_cap.get(), sensitivity_floor=float(floor) / 100. if floor else None,
            restrict_to_phantom="auto" if self.phantom_support.get() else False, psf=psf)

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
                calibration = {k: metadata.pop(k) for k in CALIBRATION_FIELDS}
                reported = 0

                def model_progress(done, total):
                    nonlocal reported
                    milestone = done * 4 // total
                    if milestone > reported:
                        reported = milestone
                        self.app.log(f"[List mode] Scanner response · {done:,}/{total:,} detector pairs")

                volume, geometry = reconstruct_listmode(
                    rays, shape, spacing, iterations=iterations, subsets=subsets,
                    model_progress=model_progress,
                    progress=lambda it: self.app.log(f"[List mode] {path.stem} · iteration {it}/{iterations}"),
                    **options, **calibration)
                geometry.update(metadata)
                method = "LM-MLEM" if subsets == 1 else "LM-OSEM"
                info = (f"{path.stem} · {method} · {geometry['traced_events']:,} of "
                        f"{geometry['total_events']:,} coincidences traced · {geometry['sensitivity']}")
                if geometry["sv_psf"]:
                    info += f" · {geometry['sv_psf']}"
                if geometry["sensitivity_floor"]:
                    info += f" · sensitivity ≥ {100 * geometry['sensitivity_floor']:g}%"
                if geometry["phantom_support"]:
                    info += " · phantom cylinder support"
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

    def __init__(self, parent, app, scroll_column):
        super().__init__(parent, padding=8)
        self.app = app
        self.results = {}
        self.comparison = ()
        self.source = tk.StringVar(self)
        self.history = tk.StringVar(self)
        self.matrix = tk.StringVar(self)
        self.voxel = tk.StringVar(self)
        self.record_steps = tk.BooleanVar(self, True)
        self.iterations = tk.StringVar(self, "1, 2, 4, 6, 8, 12")
        self.subsets = tk.StringVar(self, "1, 2, 4")
        self.radius = tk.DoubleVar(self, 2.0)
        self.background_gap = tk.DoubleVar(self, DEFAULT_BACKGROUND_GAP_MM)
        self.source_count = tk.IntVar(self, 0)
        self.threshold = tk.DoubleVar(self, 25.)
        self.search_radius = tk.DoubleVar(self, 0.)
        self.compare_cor = tk.BooleanVar(self, False)
        self.status = tk.StringVar(self)
        self.x_axis = tk.StringVar(self, "Iterations")
        self.metric = tk.StringVar(self, "Contrast (%)")
        self.noise_metric = tk.StringVar(self, ROBUST_CV)
        self.title = tk.StringVar(self)
        self.font = tk.StringVar(self, "sans-serif")
        self.legend = tk.StringVar(self, "best")
        self.style = tk.StringVar(self, "Lines + points")
        self.palette = tk.StringVar(self, "tab10")
        self.dpi = tk.IntVar(self, 200)
        self.selected_source = tk.StringVar(self, "All")

        side = ttk.Notebook(self)
        side.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sweep_tab = ttk.Frame(side)
        footer = ttk.Frame(sweep_tab, padding=(8, 5))
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        sweep_column = scroll_column(sweep_tab, max_width=370)
        sweep_column.pack(fill=tk.BOTH, expand=True)
        chart_column = scroll_column(side, max_width=370)
        side.add(sweep_tab, text="Sweep")
        side.add(chart_column, text="Chart")
        study = ttk.Frame(sweep_column.body, padding=8)
        chart = ttk.Frame(chart_column.body, padding=8)
        study.pack(fill=tk.X)
        chart.pack(fill=tk.X)
        self.source_selector = _field(study, "Acquisition", self.source, ())
        _field(study, "Matrix", self.matrix)
        _field(study, "Voxel (mm)", self.voxel)
        ttk.Label(study, text="Blank uses the current grid. Use one XY value or X Y Z for list mode.",
                  style="Hint.TLabel", wraplength=245).pack(anchor="w", pady=(1, 5))
        _field(study, "Iterations", self.iterations)
        _field(study, "Subsets", self.subsets)
        _field(study, "Source ROI (mm)", self.radius)
        _field(study, "Background gap (mm)", self.background_gap)
        ttk.Label(study, text="Transverse clearance from every source ROI, across all slices.",
                  style="Hint.TLabel", wraplength=245).pack(anchor="w")
        _field(study, "Source count (0 = auto)", self.source_count)
        _field(study, "Peak threshold (%)", self.threshold)
        _field(study, "Search radius (mm, 0 = auto)", self.search_radius)
        ttk.Checkbutton(study, text="Compare estimated COR", variable=self.compare_cor).pack(anchor="w", pady=3)
        ttk.Checkbutton(study, text="Record iteration images", variable=self.record_steps).pack(anchor="w", pady=(0, 5))
        ttk.Button(footer, text="Run convergence sweep", style="Primary.TButton",
                   command=self._start).pack(fill=tk.X)
        ttk.Button(footer, text="Run batch...", command=self._start_batch).pack(fill=tk.X, pady=(4, 0))
        ttk.Label(footer, textvariable=self.status, wraplength=250).pack(fill=tk.X, pady=(4, 0))
        _field(chart, "X axis", self.x_axis, ("Iterations", "Subset updates"))
        _field(chart, "Single-result signal", self.metric, ("Contrast (%)", "CNR"))
        noise_choice = _field(chart, "Background noise", self.noise_metric, NOISE_METRICS)
        noise_choice.bind("<<ComboboxSelected>>", lambda _event: self.redraw())
        self.source_view = _field(chart, "Source", self.selected_source, ("All",))
        _field(chart, "Title", self.title)
        _field(chart, "Font", self.font, ("sans-serif", "serif", "monospace"))
        _field(chart, "Legend", self.legend, ("best", "upper right", "upper left", "lower right", "off"))
        _field(chart, "Line style", self.style, ("Lines + points", "Lines", "Points"))
        _field(chart, "Colours", self.palette, ("tab10", "Dark2", "Set2", "viridis"))
        _field(chart, "Export DPI", self.dpi)
        ttk.Button(chart, text="Apply chart settings", command=self.redraw).pack(fill=tk.X, pady=4)

        right = ttk.Frame(self)
        right.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        results = ttk.Frame(right)
        results.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(results, text="Result").pack(side=tk.LEFT)
        self.history_combo = ttk.Combobox(results, textvariable=self.history, width=30, state="readonly")
        self.history_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.history_combo.bind("<<ComboboxSelected>>", lambda _event: self._select_single_result())
        actions = ttk.Frame(right)
        actions.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(actions, text="Iterations & ROIs", command=self._open_history).pack(side=tk.LEFT)
        ttk.Button(actions, text="Compare results...", command=self._compare_results).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Save figure...", command=self._save_figure).pack(side=tk.RIGHT)
        ttk.Button(actions, text="Save CSV...", command=self._save_csv).pack(side=tk.RIGHT, padx=5)
        viewport = ttk.Frame(right)
        viewport.pack(fill=tk.BOTH, expand=True)
        viewport.rowconfigure(0, weight=1)
        viewport.columnconfigure(0, weight=1)
        self.plot_view = tk.Canvas(viewport, highlightthickness=0, width=1, height=1)
        vertical = ttk.Scrollbar(viewport, orient=tk.VERTICAL, command=self.plot_view.yview)
        horizontal = ttk.Scrollbar(viewport, orient=tk.HORIZONTAL, command=self.plot_view.xview)
        self.plot_view.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.plot_view.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        plot = ttk.Frame(self.plot_view)
        self._plot_window = self.plot_view.create_window(0, 0, window=plot, anchor="nw")
        self._plot_minimum = (360, 300)
        self.plot_view.bind("<Configure>", self._resize_plot)
        self.figure = Figure(figsize=(7, 5), dpi=100, constrained_layout=True)
        self.canvas = plot_canvas(app, self.figure, plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        SessionToolbar(self.canvas, right).update()
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

    def _open_history(self):
        name = self.history.get()
        if name:
            self.app.iteration_panel.select_result(name)
            self.app.recon_workspace.select(self.app.iteration_panel)

    @staticmethod
    def _grid_value(raw, convert, label, *, listmode):
        if not raw.strip():
            return ()
        values = _numbers(raw, None, convert, label)
        if len(values) not in ((1, 3) if listmode else (1,)) or not np.isfinite(values).all() or min(values) <= 0:
            description = "one XY value or three X Y Z values" if listmode else "one positive number"
            raise ValueError(f"Enter {label} as {description}")
        return values

    def _start_batch(self):
        entries = list(self.app.sinograms)
        if self.app.listmode_panel.path.get().strip():
            entries.append({"name": self.LISTMODE})
        if not entries:
            self.app.error("Load a sinogram or select a list-mode NPZ")
            return
        selected = BatchSelection(
            self, entries, title="Batch convergence",
            prompt="Select acquisitions to sweep and plot side by side.",
            hint="The same iteration and subset settings apply to each acquisition.",
        ).result
        if selected:
            self._start(tuple(entry["name"] for entry in selected))

    def _sweep_spec(self, name, matrix_text, voxel_text, snapshot):
        is_listmode = name == self.LISTMODE
        matrix = self._grid_value(matrix_text, int, "matrix", listmode=is_listmode)
        voxel = self._grid_value(voxel_text, float, "voxel size", listmode=is_listmode)
        if is_listmode:
            path, shape, spacing, energy, settings = self.app.listmode_panel.settings()
            if matrix:
                shape = matrix[::-1] if len(matrix) == 3 else (shape[0], matrix[0], matrix[0])
            if voxel:
                spacing = voxel if len(voxel) == 3 else (voxel[0], voxel[0], spacing[2])
            return path, shape, spacing, energy, settings

        entry = self.app.get_sinogram(name)
        if entry is None:
            raise ValueError("Load a sinogram or select a list-mode NPZ")
        snapshot = snapshot.copy()
        if matrix or voxel:
            mode = "matrix+voxel" if matrix and voxel else "matrix" if matrix else "voxel"
            snapshot["grid_mode"] = SimpleNamespace(get=lambda: mode)
            if matrix:
                snapshot["n_img"] = SimpleNamespace(get=lambda: matrix[0])
            if voxel:
                snapshot["voxel_mm"] = SimpleNamespace(get=lambda: voxel[0])
        if name != self.app.recon_panel.var_sino.get():
            self._use_acquisition_response(snapshot, entry)
        return entry, snapshot

    def _start(self, names=None):
        names = names if names is not None else (self.source.get(),)
        try:
            iterations = _numbers(self.iterations.get(), None, int, "iterations")
            subsets = _numbers(self.subsets.get(), None, int, "subsets")
            if min(iterations + subsets) < 1:
                raise ValueError("Iterations and subsets must be positive")
            options = dict(iterations=iterations, subsets=subsets, radius_mm=self.radius.get(),
                           background_gap_mm=self.background_gap.get(),
                           source_count=self.source_count.get(), threshold=self.threshold.get() / 100,
                           search_radius_mm=self.search_radius.get(), record_steps=self.record_steps.get())
            compare = self.compare_cor.get()
            has_sinogram = any(name != self.LISTMODE for name in names)
            if compare and not has_sinogram:
                raise ValueError("Estimated COR comparison requires a sinogram")
            snapshot = self.app.recon_panel._snapshot() if has_sinogram else None
            matrix, voxel = self.matrix.get(), self.voxel.get()
            jobs = [(name, self._sweep_spec(name, matrix, voxel, snapshot)) for name in names]
        except (ValueError, tk.TclError) as exc:
            self.app.error(str(exc))
            return

        def run():
            outputs, failures = [], []
            for name, spec in jobs:
                history = None
                try:
                    history = IterationHistory()
                    spec_options = dict(options, history=history)
                    self.app.log(f"[convergence] Sweeping {name}")
                    if name == self.LISTMODE:
                        output_name, rows, regions, summary = self._run_listmode(spec, spec_options)
                    else:
                        output_name = name
                        rows, regions, summary = self._run_sinogram(*spec, spec_options, compare)
                    outputs.append((output_name, rows, regions, summary, history))
                except Exception as exc:
                    if history is not None:
                        history.close()
                    failures.append(f"{name}: {exc}")
            if outputs:
                self.app._q.put(("convergence_ready", outputs))
            else:
                self.app._q.put(("convergence_failed", None))
            if failures:
                self.app.error("Convergence:\n" + "\n".join(failures))

        if self.app.start_worker("EM convergence", run):
            self.status.set(f"Computing iteration checkpoints for {len(jobs)} acquisition(s)...")

    @staticmethod
    def _use_acquisition_response(snapshot, entry):
        from Gimntools.GimnRec.detector_response import CRYSTAL_MU_511, DEFAULT_FACE_WIDTH_MM, crystal_material

        config = entry["detector_config"] or {}
        if snapshot["response_mode"].get() in ("DOI-aware", "DOI + residual SV response",
                                               "Analytical detector response", "Automatic SV-PSF"):
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
        calibration = {k: meta[k] for k in CALIBRATION_FIELDS}

        def run(iterations, subsets, callback=None, *, update_callback=None):
            def visit(iteration, volume):
                for z, image in enumerate(volume):
                    callback(z, iteration, image)

            def record(iteration, subset, volume):
                update_callback(None, iteration, subset, volume)

            return reconstruct_listmode(
                rays, shape, spacing, iterations=iterations, subsets=subsets,
                checkpoint=visit if callback else None, update_callback=record if update_callback else None,
                **settings, **calibration,
            )

        if calibration["detector_model"] is not None:
            calibration["detector_model"] = prepare_listmode_response(
                rays, shape, spacing, settings["center"], calibration["detector_model"],
            )
        if settings.get("psf"):
            response = calibration["detector_model"]
            psf = prepare_listmode_psf(shape, spacing, settings["center"], settings["psf"],
                                       response.detector_config if response else None)
            settings = dict(settings, psf=psf)
        reference, geometry = run(min(4, max(options["iterations"])), 1)
        regions = detect_source_regions(reference, spacing, radius_mm=options["radius_mm"],
                                        background_gap_mm=options["background_gap_mm"],
                                        source_count=options["source_count"],
                                        relative_threshold=options["threshold"], support=reference > 0,
                                        search_radius_mm=options["search_radius_mm"] or None)
        history = options["history"]
        history.set_reference(reference, spacing, regions)
        recording = partial(history.record, "List mode", shape=shape, spacing=spacing) if options["record_steps"] else None
        rows = convergence_series(run, regions, options["iterations"], options["subsets"],
                                  cor_mm=None, label="List mode", recording=recording)
        model = " · ".join(name for name in (geometry["sensitivity"], geometry["sv_psf"]) if name)
        return path.stem, rows, regions, model

    def _register_result(self, name, rows, regions, summary, history):
        result_name = self.app._unique_name(name, self.results)
        if history is not None:
            from Gimntools.GimnRec.iteration_history import grid_labels

            grid = grid_labels(history.reference.shape, history.reference.spacing)
            summary = f"{summary} {grid['matrix_xyz']} voxels at {grid['voxel_xyz_mm']} mm."
            for row in rows:
                row.update(grid)
            self.app.iteration_panel.add_history(result_name, history)
        self.results[result_name] = rows, len(regions), summary
        return result_name

    def show_results(self, outputs):
        names = tuple(self._register_result(*output) for output in outputs)
        self.history_combo.configure(values=list(self.results))
        self.history.set(names[-1])
        self.comparison = names if len(names) > 1 else ()
        self._select_result()

    def _compare_results(self):
        if len(self.results) < 2:
            self.app.error("Run convergence sweeps for at least two results first")
            return
        selected = BatchSelection(
            self, [{"name": name} for name in self.results], title="Compare convergence",
            prompt="Each selected result will occupy one column of the comparison.",
        ).result
        if selected:
            names = tuple(item["name"] for item in selected)
            self.history.set(names[0])
            self.comparison = names if len(names) > 1 else ()
            self._select_result()

    def _select_single_result(self):
        self.comparison = ()
        self._select_result()

    def _selected_results(self):
        names = self.comparison or (self.history.get(),)
        return {name: self.results[name][0] for name in names if name in self.results}

    def _select_result(self):
        if self.comparison:
            count = max(self.results[name][1] for name in self.comparison)
            self.status.set(f"Comparing {len(self.comparison)} results.")
        else:
            rows, count, summary = self.results[self.history.get()]
            point = convergence_operating_point(rows)
            suggestion = f" Suggested checkpoint: {point[0]} iteration(s), {point[1]} subset(s)." if point else ""
            self.status.set(f"Detected sources: {count}. {summary}{suggestion}")
        self.source_view.configure(values=["All", *map(str, range(1, count + 1))])
        self.selected_source.set("All")
        self.redraw()

    def _resize_plot(self, _event=None):
        width = max(self.plot_view.winfo_width(), self._plot_minimum[0])
        height = max(self.plot_view.winfo_height(), self._plot_minimum[1])
        self.plot_view.itemconfigure(self._plot_window, width=width, height=height)
        self.plot_view.configure(scrollregion=(0, 0, width, height))

    def _metrics(self, selected):
        return convergence_metrics(self.metric.get(), self.noise_metric.get(), comparison=len(selected) > 1)

    def _draw_chart(self, figure, selected, metrics):
        draw_convergence_chart(
            figure, selected, metrics, x_axis=self.x_axis.get(), source=self.selected_source.get(),
            title=self.title.get(), family=self.font.get(), legend=self.legend.get(),
            style=self.style.get(), palette=self.palette.get(),
        )

    def redraw(self):
        selected = self._selected_results()
        metrics = self._metrics(selected)
        self._plot_minimum = (max(360, 245 * len(selected)), 145 * len(metrics))
        self._resize_plot()
        self._draw_chart(self.figure, selected, metrics)
        self.plot_view.xview_moveto(0)
        self.plot_view.yview_moveto(0)
        self.canvas.draw_idle()

    def _export_stem(self):
        return "convergence-comparison" if self.comparison else f"{Path(self.history.get()).name}-convergence"

    def _save_csv(self):
        selected = self._selected_results()
        if not selected:
            return
        path = self.app.session.choose(
            "convergence.table.save", filedialog.asksaveasfilename, parent=self, defaultextension=".csv",
            initialfile=f"{self._export_stem()}.csv", filetypes=[("CSV", "*.csv")],
        )
        if path:
            try:
                rows = ([dict(acquisition=name, **row) for name, series in selected.items() for row in series]
                        if self.comparison else next(iter(selected.values())))
                save_convergence_csv(path, rows)
            except OSError as exc:
                self.app.error(str(exc))

    def _save_figure(self):
        selected = self._selected_results()
        if not selected:
            return
        path = self.app.session.choose(
            "figure.save", filedialog.asksaveasfilename, parent=self, defaultextension=".png",
            initialfile=f"{self._export_stem()}.png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
        )
        if path:
            try:
                metrics = self._metrics(selected)
                size = (max(7, 3.4 * len(selected)), max(5, 2.25 * len(metrics)))
                figure = Figure(figsize=size, constrained_layout=True)
                self._draw_chart(figure, selected, metrics)
                if self.app.appearance:
                    self.app.appearance.style_figure(figure)
                figure.savefig(path, dpi=self.dpi.get())
            except (OSError, ValueError, tk.TclError) as exc:
                self.app.error(str(exc))
