"""Analytical LUT controls for the Tk interface."""

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from Gimntools.GimnRec.analytical_response import generate_analytical_lut, kernel_summary
from Gimntools.GimnRec.detector_response import (
    CRYSTAL_MATERIALS, CRYSTAL_MU_511, crystal_response_settings,
)
from Gimntools.GimnRec.reconstructor import siddon_reconstructor
from Gimntools.GimnRec.ui.rendering import plot_canvas, show_image


class _Settings(simpledialog.Dialog):
    fields = (
        ("511 keV μ (mm⁻¹)", "doi_mu"),
        ("Active segment width (mm)", "doi_face"),
        ("Face quadrature", "doi_face_order"),
        ("Depth quadrature", "doi_depth_order"),
        ("Source samples per axis", "5"),
        ("Kernel half width (samples)", "2"),
        ("Kernel step (voxels)", "1"),
        ("Fit regularization", "0.0001"),
    )

    def __init__(self, panel, config):
        self.panel = panel
        self.config = config
        super().__init__(panel, title="Analytical Detector Response / LUT Generator")

    def body(self, parent):
        self.material = tk.StringVar(parent, value=self.panel.v["doi_material"].get())
        ttk.Label(parent, text="Crystal material").grid(row=0, column=0, sticky="w")
        select = ttk.Combobox(parent, textvariable=self.material, values=CRYSTAL_MATERIALS,
                              state="readonly")
        select.grid(row=0, column=1, sticky="ew")
        self.variables = []
        for row, (label, default) in enumerate(self.fields, 1):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
            value = self.panel.v[default].get() if default in self.panel.v else default
            variable = tk.StringVar(parent, value=value)
            ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew")
            self.variables.append(variable)
        self.material.trace_add("write", lambda *_: self.variables[0].set(
            str(CRYSTAL_MU_511.get(self.material.get(), ""))))
        ttk.Label(parent, text="Uses the selected image grid. The full response includes DOI.",
                  wraplength=320).grid(row=len(self.fields) + 1, columnspan=2, sticky="w")
        return select

    def validate(self):
        mu, face, face_order, depth_order, samples, support, stride, reg = (
            variable.get().strip() for variable in self.variables
        )
        try:
            settings = crystal_response_settings(
                self.config, material=self.material.get(), mu=mu, face=face,
                face_order=face_order, depth_order=depth_order, analytical=True,
            )
        except ValueError as exc:
            messagebox.showerror("Analytical LUT", str(exc), parent=self)
            return False
        try:
            options = dict(sample_count=int(samples), half_support=int(support),
                           kernel_stride=int(stride), regularization=float(reg.replace(",", ".")))
        except ValueError:
            messagebox.showerror("Analytical LUT",
                                 "Enter valid source samples, kernel support, step and regularization.",
                                 parent=self)
            return False
        self.result = settings, options
        return True


def open_generator(panel):
    app = panel.app
    entry = app.get_sinogram(panel.var_sino.get())
    if entry is None or not entry.get("acquisition") or entry.get("rebinned"):
        messagebox.showerror("Analytical LUT", "Build and select an unrebinned sinogram from ROOT.",
                             parent=panel)
        return
    panel._sync_cor_from_sino()
    selection = _Settings(panel, entry["detector_config"]).result
    if selection is None:
        return
    settings, options = selection
    path = filedialog.asksaveasfilename(parent=panel, title="Save analytical LUT",
                                       defaultextension=".npz", filetypes=[("NPZ LUT", "*.npz")])
    if not path:
        return
    try:
        grid = app._grid_kwargs(panel.v, entry["fov_mm"], entry["sino_range_mm"])
    except (ValueError, tk.TclError) as exc:
        messagebox.showerror("Analytical LUT", str(exc), parent=panel)
        return
    grid.update(dx_cor_mm=entry["dx_cor_mm"], dy_cor_mm=entry["dy_cor_mm"])

    def run():
        try:
            recon = siddon_reconstructor(entry["sino"], **grid)
            result = generate_analytical_lut(
                recon, entry["angles"], settings, entry["acquisition"], path,
                **options,
            )
            app._q.put(("analytical_lut_ready", (path, result)))
        except Exception as exc:
            app.error(str(exc))

    app.start_worker("analytical LUT", run)


def show_lut(panel, path, result):
    from matplotlib.figure import Figure

    panel.v["sv_lut"].set(path)
    panel.v["response_mode"].set("Spatially variant PSF")
    window = tk.Toplevel(panel)
    window.title("Analytical LUT preview")
    selection = ttk.Combobox(window, state="readonly", values=[
        f"x {x:.2f} mm, y {y:.2f} mm" for y in result["y_mm"] for x in result["x_mm"]
    ])
    selection.pack(fill=tk.X)
    figure = Figure(figsize=(7.4, 3.1), layout="constrained")
    image, profiles = figure.subplots(1, 2)
    canvas = plot_canvas(panel.app, figure, window)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    stats = ttk.Label(window, wraplength=700)
    stats.pack(fill=tk.X)

    def update(_event=None):
        iy, ix = divmod(selection.current(), len(result["x_mm"]))
        dx, dy = result["dx_mm"], result["dy_mm"]
        summary = kernel_summary(result, iy, ix)
        image.clear()
        show_image(panel.app, image, result["kernels"][iy, ix], origin="lower", cmap="viridis")
        image.set_title("Kernel weights")
        profiles.clear()
        profiles.plot(dx, summary["x_profile"], label="X")
        profiles.plot(dy, summary["y_profile"], label="Y")
        profiles.set_xlabel("Offset (mm)")
        profiles.legend()
        cx, cy = summary["centroid"]
        wx, wy = summary["fwhm"]
        error = result["relative_error"][iy, ix]
        extra = "\nHigh residual. Prefer the analytical operator directly." if error > 0.30 else ""
        stats.configure(text=f"Centroid ({cx:.2f}, {cy:.2f}) mm · FWHM ({wx:.2f}, {wy:.2f}) mm\n"
                             f"Asymmetry {summary['asymmetry']:.3f} · Fit error {error:.1%} · "
                             f"{result['elapsed_seconds']:.1f} s{extra}")
        canvas.draw_idle()

    selection.current(0)
    selection.bind("<<ComboboxSelected>>", update)
    update()
