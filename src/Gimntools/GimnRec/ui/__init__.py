"""Tkinter and optional PySide6/VTK graphical entry points."""


def qt_main():
    """Launch the optional Qt/VTK interface."""
    try:
        from Gimntools.GimnRec.ui.qt_app import main
    except ModuleNotFoundError as exc:
        if exc.name.split(".")[0] in {"PySide6", "vtkmodules"}:
            raise SystemExit("Install the Qt interface with pip install 'GimnTools[qt]'") from exc
        raise
    return main()
