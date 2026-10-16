"""
Gimntools.GimnRec.storage — armazenamento local multiplataforma (SQLite).

Stores:
  - "settings": key/value pairs (e.g. last files used).
  - "profiles": profiles with researcher, experiment, detector_config,
    rsectors, angles and calibration.

Why SQLite: it is part of the Python standard library (`sqlite3`, no
external dependencies), it is a single file, it is transactional, and it
works the same on Windows, macOS and Linux. The file is placed in the
dados do utilizador apropriada a cada SO.

Everything structured (config, rsectors, etc.) is serialized as JSON in
text columns — simple, portable and human-readable.

Note on column names: ``pesquisador`` (researcher) and ``experimento``
(experiment) are kept in Portuguese for backward compatibility with
existing databases; renaming them would break saved profiles.
"""

import os
import sys
import json
import time
import sqlite3

import numpy as np

APP_NAME = "GimnTools"


def user_data_dir(app=APP_NAME):
    """Per-OS user data directory (fallback when no project root is found)."""
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:  # Linux / outros POSIX
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    d = os.path.join(base, app)
    os.makedirs(d, exist_ok=True)
    return d


def _find_project_root(start=None):
    """Walk up from this file until the project root (the directory that
    contains pyproject.toml). Works when the package is installed in
    editable mode (`pip install -e .`), i.e. from a project folder."""
    d = os.path.dirname(os.path.abspath(start or __file__))
    for _ in range(8):
        if os.path.exists(os.path.join(d, "pyproject.toml")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def db_path():
    """Location of gimntools.db, in order of priority:

      1) the GIMNTOOLS_DB environment variable (explicit path), if set;
      2) INSIDE THE PROJECT: <project_root>/data/gimntools.db
         (root = folder with pyproject.toml; typical with `pip install -e .`);
      3) fallback: the OS user data directory (wheel installs,
         when there is no local project).
    """
    env = os.environ.get("GIMNTOOLS_DB")
    if env:
        parent = os.path.dirname(os.path.abspath(env))
        if parent:
            os.makedirs(parent, exist_ok=True)
        return env

    root = _find_project_root()
    if root:
        d = os.path.join(root, "data")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "gimntools.db")

    return os.path.join(user_data_dir(), "gimntools.db")


def _json_safe(obj):
    """Convert NumPy arrays (and scalars) into JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def dumps(obj):
    return json.dumps(_json_safe(obj))


class Store:
    """Access to the local SQLite database. Every operation opens/closes
    its own connection (simple and safe for a single-user desktop app)."""

    def __init__(self, path=None):
        self.path = path or db_path()
        self._init()

    def _conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def _init(self):
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS settings("
                      "key TEXT PRIMARY KEY, value TEXT)")
            c.execute("""CREATE TABLE IF NOT EXISTS profiles(
                name TEXT PRIMARY KEY,
                pesquisador TEXT,
                experimento TEXT,
                fmt TEXT,
                detector_config TEXT,
                rsectors TEXT,
                angles TEXT,
                calibration TEXT,
                notes TEXT,
                updated_at REAL)""")

    # ---------------- settings (key/value) ----------------
    def get_setting(self, key, default=None):
        with self._conn() as c:
            r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        if r is None:
            return default
        try:
            return json.loads(r["value"])
        except (json.JSONDecodeError, TypeError):
            return default

    def set_setting(self, key, value):
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO settings(key, value) VALUES(?, ?)",
                      (key, dumps(value)))

    # ---------------- profiles ----------------
    def save_profile(self, name, *, pesquisador="", experimento="", fmt="petsys",
                     detector_config=None, rsectors=None, angles=None,
                     calibration=None, notes=""):
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO profiles("
                "name, pesquisador, experimento, fmt, detector_config, "
                "rsectors, angles, calibration, notes, updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (name, pesquisador, experimento, fmt,
                 dumps(detector_config or {}), dumps(rsectors or {}),
                 dumps(angles or {}), dumps(calibration or {}),
                 notes, time.time()))

    def list_profiles(self):
        with self._conn() as c:
            rows = c.execute(
                "SELECT name, pesquisador, experimento, fmt, updated_at "
                "FROM profiles ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_profile(self, name):
        with self._conn() as c:
            r = c.execute("SELECT * FROM profiles WHERE name=?", (name,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        for k in ("detector_config", "rsectors", "angles", "calibration"):
            try:
                d[k] = json.loads(d[k]) if d[k] else {}
            except (json.JSONDecodeError, TypeError):
                d[k] = {}
        return d

    def delete_profile(self, name):
        with self._conn() as c:
            c.execute("DELETE FROM profiles WHERE name=?", (name,))