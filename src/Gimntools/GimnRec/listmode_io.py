"""Bounded-memory NPZ export for coincidence batches."""

from contextlib import ExitStack
import json
import mmap
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from Gimntools.GimnRec.listmode import EVENT_COLUMNS
from Gimntools.GimnRec.sinogramer import Sinogramer
from Gimntools.GimnRec.utils import _resolve_axes


def object_axes(config):
    """List-mode XYZ uses the in-plane sinogram axes followed by the slice axis."""
    axis, axial = _resolve_axes(config)
    return axis, (*Sinogramer._PLANE_AXES[axis], axial)


def event_rows(first, second, energies):
    rows = np.empty((len(first), len(EVENT_COLUMNS)), dtype=np.float64)
    rows[:, :3], rows[:, 4:7] = first, second
    rows[:, 3], rows[:, 7] = energies
    return rows


def write_listmode(path, batches, metadata):
    """Stream (events, pose angles, step indices) into an atomic portable archive.

    Metadata can be filled by the batch iterator. All event positions must
    already be in the common object frame; pose_deg is descriptive only.
    """
    path = Path(path)
    with TemporaryDirectory(prefix=".listmode-", dir=path.parent) as directory, ExitStack() as stack:
        staging = Path(directory)
        events = stack.enter_context((staging / "events").open("w+b"))
        poses = stack.enter_context((staging / "poses").open("w+b"))
        steps = stack.enter_context((staging / "steps").open("w+b"))
        count = 0
        for rows, angles, indices in batches:
            np.asarray(rows, dtype=np.float64).tofile(events)
            np.broadcast_to(angles, (len(rows),)).astype(np.float64).tofile(poses)
            np.broadcast_to(indices, (len(rows),)).astype(np.int64).tofile(steps)
            count += len(rows)
        if not count:
            raise ValueError("No coincidences were recorded. Check the geometry, energy window or primaries per step")
        metadata.update(coincidences=count, columns=EVENT_COLUMNS, coordinate_units="mm", energy_units="keV")
        for stream in (events, poses, steps):
            stream.flush()
        mappings = [stack.enter_context(mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ))
                    for stream in (events, poses, steps)]
        archive = staging / "archive.npz"
        np.savez_compressed(archive,
                            events=np.ndarray((count, 8), dtype=np.float64, buffer=mappings[0]),
                            pose_deg=np.ndarray(count, dtype=np.float64, buffer=mappings[1]),
                            step_index=np.ndarray(count, dtype=np.int64, buffer=mappings[2]),
                            metadata_json=np.array(json.dumps(metadata, allow_nan=False)))
        os.replace(archive, path)
    return metadata
