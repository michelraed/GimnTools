"""Desktop controls and file-dialog locations shared by the Tk launchers."""

import math
import os
from pathlib import Path
import sqlite3
import tkinter as tk

from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk


_PREFIX = "desktop."


class DesktopSession:
    def __init__(self, root, store):
        self.root = root
        self.store = store
        self.saved = store.get_settings(_PREFIX)
        self.pending = {}
        self._job = None

    def get(self, key, default=None):
        key = _PREFIX + key
        return self.pending.get(key, self.saved.get(key, default))

    def remember(self, key, value):
        key = _PREFIX + key
        if value == self.saved.get(key):
            self.pending.pop(key, None)
        else:
            self.pending[key] = value
            if self._job is None:
                self._job = self.root.after(300, self.flush)

    def flush(self):
        if self._job is not None:
            self.root.after_cancel(self._job)
            self._job = None
        if not self.pending:
            return
        try:
            self.store.set_settings(self.pending)
        except sqlite3.Error as exc:
            self.root.log(f"Could not save desktop preferences: {exc}")
        else:
            self.saved.update(self.pending)
            self.pending.clear()

    def watch(self, group, fields, *, defaults=None):
        for name, var in fields.items():
            key = f"{group}.{name}"
            default = var.get()
            value = self.get(key, (defaults or {}).get(name, default))
            if type(value) is type(default) and (not isinstance(value, float) or math.isfinite(value)):
                var.set(value)
            var.trace_add("write", self._capture(key, var))

    def _capture(self, key, var):
        def capture(*_):
            try:
                value = var.get()
            except (tk.TclError, ValueError):
                return  # An entry can be temporarily empty while it is being edited.
            if not isinstance(value, float) or math.isfinite(value):
                self.remember(key, value)

        return capture

    def watch_notebook(self, key, notebook):
        tabs = {notebook.tab(tab, "text"): tab for tab in notebook.tabs()}
        wanted = self.get(key)
        if isinstance(wanted, str) and wanted in tabs:
            notebook.select(tabs[wanted])
        notebook.bind("<<NotebookTabChanged>>", lambda _event: self.remember(
            key, notebook.tab(notebook.select(), "text")), add="+")

    def directory(self, purpose, fallback=None):
        for value in (self.get(f"folder.{purpose}"), fallback, self.get("folder.last")):
            if not isinstance(value, (str, os.PathLike)) or not value:
                continue
            path = Path(value).expanduser()
            try:
                if path.is_dir():
                    return str(path)
            except (OSError, ValueError):
                continue
        return None

    def remember_directory(self, purpose, path):
        absolute = os.path.abspath(os.path.expanduser(path))
        self.remember(f"folder.{purpose}", absolute)
        self.remember("folder.last", absolute)

    def choose(self, purpose, picker, *, directory=False, **options):
        initial = self.directory(purpose, options.get("initialdir"))
        if initial:
            options["initialdir"] = initial
        selected = picker(**options)
        if selected:
            path = selected[0] if isinstance(selected, (list, tuple)) else selected
            self.remember_directory(purpose, path if directory else Path(path).parent)
        return selected


class SessionToolbar(NavigationToolbar2Tk):
    """Keep Matplotlib's save dialog in the same folder as other plot exports."""

    def save_figure(self, *args):
        import matplotlib

        session = self.winfo_toplevel().session
        default = matplotlib.rcParams["savefig.directory"]
        initial = session.directory("figure.save") or default
        with matplotlib.rc_context({"savefig.directory": initial}):
            filename = super().save_figure(*args)
            last = matplotlib.rcParams["savefig.directory"]
        if filename:
            session.remember_directory("figure.save", Path(filename).parent)
        elif last != initial:
            session.remember_directory("figure.save", last)
        return filename
