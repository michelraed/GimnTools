"""Multi-selection dialog for loaded acquisitions and comparison results."""

import tkinter as tk
from tkinter import simpledialog, ttk


class BatchSelection(simpledialog.Dialog):
    def __init__(self, parent, entries, *, title, prompt, hint=None):
        self.entries = entries
        self.prompt = prompt
        self.hint = hint
        super().__init__(parent, title=title)

    def body(self, frame):
        ttk.Label(frame, text=self.prompt, wraplength=450).pack(anchor="w")
        holder = ttk.Frame(frame)
        holder.pack(fill=tk.BOTH, expand=True, pady=6)
        self.list = tk.Listbox(holder, selectmode=tk.EXTENDED, width=58,
                               height=min(12, len(self.entries)), exportselection=False)
        scrollbar = ttk.Scrollbar(holder, orient=tk.VERTICAL, command=self.list.yview)
        self.list.configure(yscrollcommand=scrollbar.set)
        self.list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        for entry in self.entries:
            self.list.insert(tk.END, entry["name"])
        self.list.selection_set(0, tk.END)
        if self.hint:
            ttk.Label(frame, text=self.hint, style="Hint.TLabel").pack(anchor="w")
        return self.list

    def apply(self):
        self.result = tuple(self.entries[i] for i in self.list.curselection())
