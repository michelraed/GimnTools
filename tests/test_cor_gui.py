"""Headless GUI test for the interactive centre-of-rotation control.

Runs the real Tk widgets under a virtual display (xvfb-run) and exercises:
  * the COR line following the spinbox value,
  * a simulated click/drag on the sinogram setting the value,
  * the value being stored on the sinogram entry and reaching the
    reconstructor as `cor_mm`.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# tkinter is stdlib but is packaged separately on many Linux distributions
# (python3-tk), and a Tk root also needs a display. Skip cleanly instead of
# breaking collection of the rest of the suite.
try:
    import tkinter as _tk  # noqa: F401
    _tk.Tk().destroy()
    _TK_ERROR = None
except Exception as exc:                      # pragma: no cover
    _TK_ERROR = exc

if _TK_ERROR is not None:
    if __name__ == "__main__":
        print(f"[SKIP] Tk unavailable ({_TK_ERROR}); GUI test not run.")
        sys.exit(0)
    import pytest
    pytest.skip(f"Tk unavailable: {_TK_ERROR}", allow_module_level=True)

ok = []
labelled = []


def check(label, cond):
    ok.append(bool(cond))
    labelled.append((label, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {label}")


from Gimntools.GimnRec.ui.app import ReconUI  # noqa: E402

app = ReconUI()
app.withdraw()
# The app pops a modal "previous session" dialog when the local DB remembers
# file paths; that would block a headless run.
app._had_last_files = False

# --- a synthetic sinogram with a deliberate axis offset -------------------
SR, ND, NA = 32.0, 64, 36
bw = SR / ND
z = np.zeros((3, ND, NA))
shift_bins = int(round(4.0 / bw))          # +4 mm offset
z[:, ND // 2 + shift_bins, :] = 100.0
angles = (np.arange(NA) + 0.5) * (180.0 / NA)
app.set_sinogram(z, fov_mm=SR, sino_range_mm=SR, source="TEST",
                 detector_config={"Half_FOV": SR / 2,
                                  "sipm_pitch": [1.0, 1.0]},
                 angles_override=angles, name="synthetic")
app.sino_panel.on_new()
app.recon_panel.refresh_sino_list(select="synthetic")
app.update()

sp = app.sino_panel
check("COR starts at zero", abs(float(sp.var_cor.get())) < 1e-12)
check("COR line exists after drawing", sp._cor_line is not None)

# --- 1. spinbox value moves the line -------------------------------------
sp.var_cor.set(2.5)
app.update()
check("line follows the spinbox",
      abs(float(np.asarray(sp._cor_line.get_ydata())[0]) - 2.5) < 1e-9)

# --- 2. +/- nudge uses the step, which defaults to one radial bin ---------
check("step defaults to one radial bin",
      abs(float(sp.var_step.get()) - bw) < 1e-6)
sp._nudge(+1)
app.update()
check("+ nudges by exactly one step",
      abs(float(sp.var_cor.get()) - (2.5 + bw)) < 1e-6)
sp._nudge(-1)
app.update()

# --- 3. simulated click on the image sets the COR -------------------------
class _Ev:
    def __init__(self, ax, y):
        self.inaxes = ax
        self.ydata = y
        self.xdata = 90.0
        self.button = 1


sp._on_click(_Ev(sp.ax, -3.25))
app.update()
check("click sets the COR", abs(float(sp.var_cor.get()) + 3.25) < 1e-9)
check("line moved to the clicked position",
      abs(float(np.asarray(sp._cor_line.get_ydata())[0]) + 3.25) < 1e-9)

# drag: motion while the button is held keeps updating
sp._on_motion(_Ev(sp.ax, 1.75))
app.update()
check("drag updates the COR", abs(float(sp.var_cor.get()) - 1.75) < 1e-9)
sp._on_release(_Ev(sp.ax, 1.75))
sp._on_motion(_Ev(sp.ax, 9.0))
check("motion after release is ignored",
      abs(float(sp.var_cor.get()) - 1.75) < 1e-9)

# clicking with the interaction disabled must do nothing
sp.var_click.set(False)
sp._on_click(_Ev(sp.ax, -8.0))
check("click ignored when the checkbox is off",
      abs(float(sp.var_cor.get()) - 1.75) < 1e-9)
sp.var_click.set(True)

# --- 4. the value is stored and synced with the Reconstruction tab --------
entry = app.get_sinogram("synthetic")
check("COR stored on the sinogram entry",
      abs(float(entry["cor_mm"]) - 1.75) < 1e-9)
check("Reconstruction tab spinbox in sync",
      abs(float(app.recon_panel.v["cor_mm"].get()) - 1.75) < 1e-9)

# --- 5. estimation from the data -----------------------------------------
sp._estimate()
app.update()
est = float(sp.var_cor.get())
check(f"estimate recovers the injected offset (got {est:+.3f} mm, "
      f"expected {shift_bins*bw:+.3f})",
      abs(est - shift_bins * bw) < bw)

# --- 6. the COR actually reaches the reconstructor ------------------------
sp.var_cor.set(1.25)
app.update()
kw = app._grid_kwargs(app.recon_panel.v, SR, SR)
check("run_recon kwargs carry cor_mm", abs(kw.get("cor_mm", 0.0) - 1.25) < 1e-9)

app.recon_panel.v["method"].set("fbp")
app.run_recon(app.recon_panel.v, "synthetic", "test_recon")
app.update()
rec = app.reconstructions[-1]
check("reconstruction records the COR in its geometry",
      abs(rec["geometry"]["cor_mm"] - 1.25) < 1e-9)
check("reconstruction produced a finite volume",
      np.isfinite(rec["volume"]).all())

app.destroy()
print(f"\n{sum(ok)}/{len(ok)} checks passed")
def test_all_checks_pass():
    """pytest entry point: every check above must have passed."""
    failed = [lbl for lbl, good in labelled if not good]
    assert not failed, "failed checks: " + ", ".join(failed)


if __name__ == "__main__":
    sys.exit(0 if all(ok) else 1)
