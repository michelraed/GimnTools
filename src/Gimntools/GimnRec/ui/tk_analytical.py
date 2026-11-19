"""Analytical LUT controls for the Tk interface."""

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from Gimntools.GimnRec.analytical_response import generate_analytical_lut, kernel_summary
from Gimntools.GimnRec.detector_response import ResponseSettings
from Gimntools.GimnRec.reconstructor import siddon_reconstructor
from Gimntools.GimnRec.ui.rendering import show_image


class _Settings(simpledialog.Dialog):
    fields = (
        ("511 keV μ (mm⁻¹)", "", float),
        ("Active segment width (mm)", "", float),
        ("Face quadrature", "3", int),
        ("Depth quadrature", "3", int),
        ("Source samples per axis", "5", int),
        ("Kernel half width (samples)", "2", int),
        ("Kernel step (voxels)", "1", int),
        ("Fit regularization", "0.0001", float),
    )

    def body(self, parent):
        self.variables = []
        for row, (label, default, _) in enumerate(self.fields):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
            value = tk.StringVar(parent, value=default)
            widget = ttk.Entry(parent, textvariable=value)
            widget.grid(row=row, column=1, sticky="ew")
            self.variables.append(value)
            if row == 0:
                first = widget
        ttk.Label(parent, text="Uses the selected image grid. The full response includes DOI.",
                  wraplength=320).grid(row=len(self.fields), columnspan=2, sticky="w")
        return first

    def validate(self):
        try:
            self.result = [
                converter(value.get())
                for value, (_, _, converter) in zip(self.variables, self.fields)
            ]
        except ValueError:
            messagebox.showerror("Analytical LUT", "Supply valid numerical model settings.", parent=self)
            return False
        return True


def open_generator(panel):
    app = panel.app
    entry = app.get_sinogram(panel.var_sino.get())
    if entry is None or not entry.get("acquisition") or entry.get("rebinned"):
        messagebox.showerror("Analytical LUT", "Build and select an unrebinned sinogram from ROOT.",
                             parent=panel)
        return
    fields = _Settings(panel, title="Analytical Detector Response / LUT Generator").result
    if fields is None:
        return
    mu, face, face_order, depth_order, samples, support, stride, regularization = fields
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
    settings = ResponseSettings(mu, face, face_order, depth_order)

    def run():
        try:
            recon = siddon_reconstructor(entry["sino"], **grid)
            result = generate_analytical_lut(
                recon, entry["angles"], settings, entry["acquisition"], path,
                sample_count=samples, half_support=support,
                kernel_stride=stride, regularization=regularization,
            )
            app._q.put(("analytical_lut_ready", (path, result)))
        except Exception as exc:
            app.error(str(exc))

    app.start_worker("analytical LUT", run)


def show_lut(panel, path, result):
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
    canvas = FigureCanvasTkAgg(figure, master=window)
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
