"""DICOM series export from the shared reconstruction list."""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from Gimntools.GimnRec.dicom_export import DEFAULT_EQUIPMENT, export_dicom_series


class DicomPanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=14)
        self.app = app
        box = ttk.LabelFrame(self, text="DICOM export", padding=12)
        box.pack(anchor="nw", fill=tk.X)
        box.columnconfigure(1, weight=1)
        self.reconstruction = tk.StringVar(self)
        self.study = tk.StringVar(self)
        self.equipment = tk.StringVar(self, value=DEFAULT_EQUIPMENT)
        self.selection = ttk.Combobox(box, textvariable=self.reconstruction, state="readonly")
        for row, (label, widget) in enumerate((
            ("Reconstruction", self.selection),
            ("Study name", ttk.Entry(box, textvariable=self.study)),
            ("Equipment name", ttk.Entry(box, textvariable=self.equipment)),
        )):
            ttk.Label(box, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=3)
            widget.grid(row=row, column=1, sticky="ew", pady=3)
        self.detail = ttk.Label(box, style="Hint.TLabel")
        self.detail.grid(row=3, column=1, sticky="w", pady=4)
        self.result = ttk.Label(self, wraplength=800)
        self.result.pack(fill=tk.X, pady=(10, 0))
        self.save = ttk.Button(box, text="Export DICOM series...", style="Primary.TButton",
                               command=self._export)
        self.save.grid(row=4, column=1, sticky="w", pady=(6, 2))
        self.reconstruction.trace_add("write", self._update_selection)
        self.refresh()

    def _selected(self):
        name = self.reconstruction.get()
        return next((entry for entry in self.app.reconstructions if entry["name"] == name), None)

    def refresh(self):
        names = [entry["name"] for entry in self.app.reconstructions]
        self.selection.configure(values=names)
        if self.reconstruction.get() not in names:
            self.reconstruction.set(names[-1] if names else "")
        else:
            self._update_selection()

    def _update_selection(self, *_):
        entry = self._selected()
        self.save.configure(state="normal" if entry else "disabled")
        if entry:
            dimensions = " × ".join(map(str, entry["volume"].shape[::-1]))
            spacing = " × ".join(f"{value:.4g}" for value in entry["spacing"])
            self.detail.configure(text=f"{dimensions} voxels · {spacing} mm")
        else:
            self.detail.configure(text="No reconstruction available")

    def _export(self):
        entry = self._selected()
        if entry is None:
            return
        study, equipment = self.study.get().strip(), self.equipment.get().strip()
        if not study or not equipment:
            messagebox.showwarning("DICOM export", "Enter the study and equipment names.", parent=self)
            return
        directory = filedialog.askdirectory(parent=self, title="Choose the DICOM destination")
        if not directory:
            return

        def run():
            try:
                output = export_dicom_series(
                    entry["volume"], entry["spacing"], directory,
                    study_name=study, series_name=entry["name"], equipment_name=equipment,
                    origin=entry.get("origin"), direction=entry.get("direction"),
                )
            except (OSError, ValueError) as exc:
                self.app.error(f"DICOM export: {exc}")
            else:
                self.app._q.put(("dicom_ready", str(output)))

        if self.app.start_worker("DICOM export", run):
            self.result.configure(text="Exporting DICOM series...")

    def export_ready(self, directory):
        self.result.configure(text=f"Saved series in {directory}")
