"""Shared phantom design, synthetic acquisition and ROOT export workspace."""

from functools import partial
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

from matplotlib.figure import Figure
from matplotlib.patches import Circle

from Gimntools.GimnRec.phantoms import make_phantom
from Gimntools.GimnRec.pseudo_simulator import acquisition_plan, simulate_listmode
from Gimntools.GimnRec.root_listmode import export_root_listmode
from Gimntools.GimnRec.ui.em_panels import _field, _numbers
from Gimntools.GimnRec.ui.rendering import plot_canvas


class PseudoSimulatorPanel(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self.last_output = None
        self.phantom_type = tk.StringVar(self, "Derenzo")
        self.diameter = tk.DoubleVar(self, 24.)
        self.height = tk.DoubleVar(self, 20.)
        self.sectors = tk.IntVar(self, 6)
        self.sector_angles = tk.StringVar(self)
        self.rods = tk.StringVar(self, "0.9, 1.2, 1.5, 1.8, 2.1, 2.4")
        self.spheres = tk.StringVar(self, "2, 2.5, 3, 3.5, 4, 5")
        self.contrast = tk.DoubleVar(self, 4.)
        self.background = tk.DoubleVar(self, 1.)
        self.center_clearance = tk.DoubleVar(self, 1.5)
        self.sector_gap = tk.DoubleVar(self, 1.)
        self.geometry = tk.StringVar(self, "PETSYS")
        self.primaries = tk.IntVar(self, 100000)
        self.steps = tk.StringVar(self)
        self.arc = tk.StringVar(self)
        self.start_angle = tk.DoubleVar(self, 0.)
        self.seed = tk.IntVar(self, 0)
        self.hit_mode = tk.StringVar(self, "Match acquisition")
        self.root_format = tk.StringVar(self, "PETSYS")
        self.root_path = tk.StringVar(self)
        self.gate_unit = tk.StringVar(self, "MeV")
        self.gate_global = tk.BooleanVar(self, False)
        self.description = tk.StringVar(self)
        self.status = tk.StringVar(self)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        phantom = ttk.Frame(self.notebook, padding=12)
        acquisition = ttk.Frame(self.notebook, padding=12)
        root = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(phantom, text="Phantom")
        self.notebook.add(acquisition, text="Acquisition")
        self.notebook.add(root, text="ROOT to list mode")
        self._build_phantom(phantom)
        self._build_acquisition(acquisition)
        self._build_root(root)

        footer = ttk.Frame(self)
        footer.pack(fill=tk.X, pady=(7, 0))
        self.use_button = ttk.Button(footer, text="Open in List mode", command=self._use, state="disabled")
        self.use_button.pack(side=tk.RIGHT)
        ttk.Label(footer, textvariable=self.status).pack(side=tk.LEFT, fill=tk.X)
        self.notebook.bind("<<NotebookTabChanged>>", lambda _event: self._update_geometry())

    def _build_phantom(self, parent):
        controls = ttk.LabelFrame(parent, text="Activity distribution", padding=9)
        controls.pack(side=tk.LEFT, anchor=tk.N, padx=(0, 10))
        field = _field(controls, "Phantom", self.phantom_type, ("Derenzo", "IEC-inspired", "Homogeneous"))
        _field(controls, "Body diameter (mm)", self.diameter)
        _field(controls, "Axial length (mm)", self.height)
        self.contrast_field = _field(controls, "Insert activity (relative)", self.contrast)
        self.background_field = _field(controls, "Background activity (relative)", self.background)
        self.sector_field = _field(controls, "Derenzo sectors", self.sectors)
        self.sector_angles_field = _field(controls, "Sector angles ° (blank = equal)", self.sector_angles)
        self.rods_field = _field(controls, "Rod diameters (mm)", self.rods)
        self.clearance_field = _field(controls, "Rod-free centre radius (mm)", self.center_clearance)
        self.gap_field = _field(controls, "Gap between sectors (mm)", self.sector_gap)
        self.spheres_field = _field(controls, "Sphere diameters (mm)", self.spheres)
        ttk.Label(controls, text="Enter one rod diameter per sector, or a list of sphere diameters. "
                  "Use zero activity for cold inserts or a cold background.", style="Hint.TLabel",
                  wraplength=330).pack(anchor=tk.W, pady=5)
        ttk.Button(controls, text="Preview phantom", command=self.preview).pack(fill=tk.X)
        self.figure = Figure(figsize=(5.5, 4.2), layout="constrained")
        self.canvas = plot_canvas(self.app, self.figure, parent)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        field.bind("<<ComboboxSelected>>", lambda _event: self.preview())
        self.preview()

    def _build_acquisition(self, parent):
        controls = ttk.LabelFrame(parent, text="Acquisition sequence", padding=9)
        controls.pack(anchor=tk.NW)
        source = _field(controls, "Detector setup", self.geometry, ("PETSYS", "GATE"))
        source.bind("<<ComboboxSelected>>", lambda _event: self._update_geometry())
        for label, var in (("Primaries per step", self.primaries), ("Steps (blank = detector setup)", self.steps),
                           ("Rotation arc ° (blank = setup)", self.arc), ("Start angle °", self.start_angle),
                           ("Random seed", self.seed)):
            field = _field(controls, label, var)
            field.bind("<FocusOut>", lambda _event: self._update_geometry())
        _field(controls, "Saved hit positions", self.hit_mode,
               ("Match acquisition", "Crystal centres", "Interaction positions"))
        ttk.Label(parent, textvariable=self.description, style="Hint.TLabel",
                  wraplength=570).pack(anchor=tk.W, pady=(8, 4))
        ttk.Label(parent, text="Each step emits a new batch of back-to-back 511 keV pairs, then rotates "
                  "the source. The model includes crystal gaps and first interactions.",
                  style="Hint.TLabel", wraplength=570).pack(anchor=tk.W, pady=(0, 8))
        ttk.Button(parent, text="Simulate and save NPZ…", style="Primary.TButton",
                   command=self._simulate).pack(anchor=tk.W)

    def _build_root(self, parent):
        controls = ttk.LabelFrame(parent, text="Original coincidence ROOT", padding=9)
        controls.pack(anchor=tk.NW, fill=tk.X)
        _field(controls, "Acquisition format", self.root_format, ("PETSYS", "GATE"))
        row = ttk.Frame(controls)
        row.pack(fill=tk.X, pady=3)
        ttk.Entry(row, textvariable=self.root_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse…", command=self._browse_root).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Use ROOT from acquisition tab", command=self._use_current_root).pack(side=tk.LEFT)
        _field(controls, "GATE input energy unit", self.gate_unit, ("MeV", "keV"))
        ttk.Checkbutton(controls, text="Export GATE global XYZ when using CrystalID",
                        variable=self.gate_global).pack(anchor=tk.W)
        ttk.Label(parent, text="Export uses the geometry from the matching acquisition tab and the original "
                  "coincidence tree. Binned sinograms have no individual hit coordinates or energies.",
                  style="Hint.TLabel", wraplength=590).pack(anchor=tk.W, pady=8)
        ttk.Button(parent, text="Export list-mode NPZ…", style="Primary.TButton",
                   command=self._export).pack(anchor=tk.W)

    def phantom(self):
        kind = self.phantom_type.get()
        values = ()
        if kind == "Derenzo":
            values = _numbers(self.rods.get(), None, float, "rod diameters")
        elif kind == "IEC-inspired":
            values = _numbers(self.spheres.get(), None, float, "sphere diameters")
        angles = self.sector_angles.get().strip() if kind == "Derenzo" else ""
        return make_phantom(kind, self.diameter.get(), self.height.get(),
                            contrast=1. if kind == "Homogeneous" else self.contrast.get(),
                            background_activity=1. if kind == "Homogeneous" else self.background.get(),
                            center_clearance_mm=self.center_clearance.get() if kind == "Derenzo" else 0.,
                            sector_gap_mm=self.sector_gap.get() if kind == "Derenzo" else 0.,
                            sectors=self.sectors.get() if kind == "Derenzo" else 0, diameters=values,
                            sector_angles_deg=_numbers(angles, None, float, "sector angles") if angles else ())

    def preview(self):
        kind = self.phantom_type.get()
        for field, enabled in ((self.contrast_field, kind != "Homogeneous"),
                               (self.background_field, kind != "Homogeneous"),
                               (self.clearance_field, kind == "Derenzo"), (self.gap_field, kind == "Derenzo"),
                               (self.sector_field, kind == "Derenzo"), (self.rods_field, kind == "Derenzo"),
                               (self.sector_angles_field, kind == "Derenzo"),
                               (self.spheres_field, kind == "IEC-inspired")):
            field.state(["!disabled" if enabled else "disabled"])
        try:
            phantom = self.phantom()
        except (tk.TclError, ValueError) as exc:
            self.status.set(str(exc))
            return
        self.figure.clear()
        ax = self.figure.subplots()
        radius = phantom.diameter_mm / 2
        bg_color = "#dbe7ea" if phantom.background_activity > 0 else "none"
        ax.add_patch(Circle((0, 0), radius, edgecolor="#728898", facecolor=bg_color, linewidth=1.5))
        if phantom.center_clearance_mm:
            ax.add_patch(Circle((0, 0), phantom.center_clearance_mm, fill=False,
                                edgecolor="#728898", linestyle="--", linewidth=1))
        for insert in phantom.inserts:
            shade = "#147c83" if phantom.insert_activity >= phantom.background_activity else "#52687a"
            ax.add_patch(Circle(insert.center[:2], insert.radius, color=shade))
        ax.set(xlabel="X (mm)", ylabel="Y (mm)", title=f"{phantom.kind} · Central plane",
               xlim=(-1.12 * radius, 1.12 * radius), ylim=(-1.12 * radius, 1.12 * radius), aspect="equal")
        self.canvas.draw_idle()
        self.status.set(f"{phantom.kind} · Ø {phantom.diameter_mm:g} × {phantom.height_mm:g} mm · "
                        f"{len(phantom.inserts)} inserts · background activity {phantom.background_activity:g}")

    def _detector_setup(self, fmt):
        panel = getattr(self.app, fmt.lower())
        return panel.cfg_form.get_config(), panel.angles.get_angles(), panel.angles.get_rsectors()

    def _simulation_options(self):
        return dict(primaries=self.primaries.get(), steps=int(self.steps.get()) if self.steps.get().strip() else None,
                    arc_deg=float(self.arc.get()) if self.arc.get().strip() else None, start_deg=self.start_angle.get(),
                    seed=self.seed.get(), hit_mode=self.hit_mode.get())

    def _update_geometry(self):
        try:
            config, angles, mapping = self._detector_setup(self.geometry.get())
            options = self._simulation_options()
            plan = acquisition_plan(config, options["steps"], options["arc_deg"], options["start_deg"])
            count = len({name for name in mapping.values() if name in angles})
            self.description.set(f"{count} mapped heads · {len(plan)} steps · "
                                 f"{plan[0]:g}° to {plan[-1]:g}° · "
                                 f"{options['primaries'] * len(plan):,} total primaries")
        except (tk.TclError, ValueError) as exc:
            self.description.set(str(exc))

    def _browse_root(self):
        path = self.app.session.choose("pseudo.root.open", filedialog.askopenfilename, parent=self,
                                       filetypes=[("Coincidence ROOT", "*.root")])
        if path:
            self.root_path.set(path)

    def _use_current_root(self):
        value = self.app.gate.var_root if self.root_format.get() == "GATE" else self.app.petsys.var_meas
        self.root_path.set(value.get())

    def _output(self, name):
        return self.app.session.choose("pseudo.listmode.save", filedialog.asksaveasfilename, parent=self,
                                       initialfile=name + ".npz", defaultextension=".npz",
                                       filetypes=[("List-mode NPZ", "*.npz")])

    def _simulate(self):
        try:
            config, angles, mapping = self._detector_setup(self.geometry.get())
            phantom, options = self.phantom(), self._simulation_options()
            action = partial(simulate_listmode, phantom=phantom, config=config, detector_angles=angles,
                             mapping=mapping, **options,
                             progress=lambda i, n, count: self.app.log(
                                 f"[Pseudo-simulator] Step {i}/{n} · {count:,} coincidences"))
        except (ValueError, tk.TclError) as exc:
            self.app.error(str(exc))
            return
        name = f"{phantom.kind.lower()}_{config['data_from'].lower()}_listmode"
        if output := self._output(name):
            self._start(output, action)

    def _export(self):
        try:
            source = Path(self.root_path.get().strip())
            if not source.is_file():
                raise ValueError("Select the original coincidence ROOT file")
            config, angles, mapping = self._detector_setup(self.root_format.get())
            unit, global_hits = self.gate_unit.get(), self.gate_global.get()
        except (ValueError, tk.TclError) as exc:
            self.app.error(str(exc))
            return
        if output := self._output(f"{source.stem}_listmode"):
            def export(path):
                return export_root_listmode(source, path, config, angles, mapping, gate_global_hits=global_hits,
                                            gate_energy_unit=unit)
            self._start(output, export)

    def _start(self, output, action):
        def run():
            try:
                metadata = action(output)
                self.app._q.put(("pseudo_complete", (output, metadata)))
            except Exception as exc:
                self.app._q.put(("pseudo_complete", None))
                self.app.error(str(exc))

        if self.app.start_worker("list-mode generation", run):
            self.status.set("Generating list-mode NPZ…")

    def complete(self, result):
        if result is None:
            self.status.set("List-mode export could not be completed.")
            return
        self.last_output, metadata = result
        info = f"{Path(self.last_output).name} · {metadata['coincidences']:,} coincidences"
        self.status.set(info)
        self.app.log(f"[List-mode export] {info}")
        self.use_button.configure(state="normal")

    def _use(self):
        self.app.listmode_panel.path.set(self.last_output)
        self.app.convergence_panel.refresh()
        self.app.notebook.select(self.app.recon_workspace)
        self.app.recon_workspace.select(self.app.listmode_panel)
