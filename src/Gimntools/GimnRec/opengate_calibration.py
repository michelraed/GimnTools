"""Optional OpenGATE point of entry for independent detector calibration."""

from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.transform import Rotation

from Gimntools.GimnRec.physical_response import signature


PHYSICS_LIST = "G4EmStandardPhysics_option3"
READOUT = "Energy summed per head, crystal assigned to largest deposited energy, window on both heads"


def _open_gate():
    try:
        import opengate
    except ImportError as exc:
        raise RuntimeError("Install GimnTools[opengate] to generate independent calibration tallies") from exc
    return opengate


def _simulation(gate, geometry, pose, voxel, primaries, seed, directory):
    detector = geometry["detector"]
    if len(detector["heads"]) != 2:
        raise ValueError("The OpenGATE calibration workflow currently requires two heads")
    shape = tuple(geometry["image_grid"]["shape_zyx"])
    if not 0 <= pose < len(geometry["pose_degrees"]) or not 0 <= voxel < np.prod(shape) or primaries <= 0 or seed <= 0:
        raise ValueError("Check the pose index, voxel index, positive primary count and positive seed")
    millimetre = gate.g4_units.mm
    density = gate.g4_units.g_cm3
    sim = gate.Simulation()
    sim.output_dir = directory
    sim.number_of_threads = 1
    sim.random_seed = seed
    sim.physics_manager.physics_list_name = PHYSICS_LIST
    sim.world.material = "G4_AIR"
    lower, upper = np.array(detector["lower_mm"]), np.array(detector["upper_mm"])
    translations = np.array(detector["translations_xyz_mm"])
    spacing = np.array(geometry["image_grid"]["spacing_xyz_mm"])
    centre = np.array(geometry["image_grid"]["center_xyz_mm"])
    clearance = 2 * (np.linalg.norm(centre) + np.linalg.norm(spacing * shape[::-1])
                     + np.linalg.norm(upper) + np.max(np.abs(translations)) + 1.)
    sim.world.size = [clearance * millimetre] * 3
    materials = {
        "LYSO": (["Lu", "Y", "Si", "O"], [18, 2, 10, 50], 7.1),
        "BGO": (["Bi", "Ge", "O"], [4, 3, 12], 7.13),
    }
    material = detector["material"].upper()
    if material not in materials:
        raise ValueError("OpenGATE calibration supports LYSO and BGO composition presets")
    elements, atoms, rho = materials[material]
    sim.volume_manager.material_database.add_material_nb_atoms(material, elements, atoms, rho * density)

    pixels = np.array(detector["pixels"])
    pitch = np.array(detector["pitch_mm"])
    grid = np.array(detector["grid_to_head"])
    face = detector["face_mm"]
    local_rotation = np.eye(3)
    local_rotation[1:, 1:] = grid.T @ np.diag([1., np.linalg.det(grid)])
    channels = []
    for head, orient in enumerate(np.array(detector["head_rotations"])):
        for i, pixel in enumerate(np.ndindex(*pixels)):
            centre_uv = (pixel - (pixels - 1) / 2) * pitch
            local = np.r_[(lower[0] + upper[0]) / 2, centre_uv @ grid]
            size = np.array([upper[0] - lower[0], face, face])
            transverse = np.abs(local_rotation[1:, 1:]) @ (size[1:] / 2)
            if np.any(local[1:] - transverse < lower[1:] - 1e-7) or np.any(
                    local[1:] + transverse > upper[1:] + 1e-7):
                raise ValueError("The OpenGATE crystal geometry would extend beyond the configured module")
            channel = head * int(np.prod(pixels)) + i
            crystal = sim.add_volume("Box", f"crystal_{channel}")
            crystal.material = material
            crystal.size = size * millimetre
            crystal.rotation = orient @ local_rotation
            crystal.translation = (orient @ local + translations[head]) * millimetre
            name = f"hits_{channel}"
            detector_hits = sim.add_actor("DigitizerHitsCollectionActor", name)
            detector_hits.attached_to = crystal.name
            detector_hits.output_filename = f"{name}.root"
            detector_hits.attributes = ["EventID", "TotalEnergyDeposit"]
            channels.append(name)

    axes = np.argsort(["xyz".index(a) for a in geometry["object_axes"]])
    rotation = Rotation.from_euler(geometry["rotation_axis"], geometry["pose_degrees"][pose], degrees=True)
    xyz = np.asarray(np.unravel_index(voxel, shape)[::-1])
    position = centre + (xyz + .5 - np.asarray(shape[::-1]) / 2) * spacing
    source = sim.add_source("GenericSource", "coincidences")
    source.particle = "back_to_back"
    source.number_of_primaries = primaries
    source.position.type = "box"
    source.position.size = spacing[axes] * millimetre
    source.position.rotation = rotation.as_matrix()
    source.position.translation = rotation.apply(position[axes]) * millimetre
    source.direction.type = "iso"
    source.energy.mono = 511 * gate.g4_units.keV
    stats = sim.add_actor("SimulationStatisticsActor", "counts")
    return sim, stats, channels


def _readout(event, channel, energy, primaries, crystals, window):
    """Select a deterministic crystal and apply the energy window to each simulated head."""
    from scipy.sparse import coo_matrix

    hits = coo_matrix((energy, (event, channel)), shape=(primaries, 2*crystals)).tocsr()
    winners, accepted = [], np.ones(primaries, dtype=bool)
    for head in range(2):
        deposited = hits[:, head*crystals:(head+1)*crystals]
        total = np.asarray(deposited.sum(axis=1)).ravel()
        accepted &= (total > 0) & (total >= window[0]) & (total <= window[1])
        winners.append(np.asarray(deposited.argmax(axis=1)).ravel() + head*crystals)
    return winners[0][accepted], winners[1][accepted]


def simulate_voxel(geometry, output, *, pose, voxel, primaries, seed, energy_window=None):
    """Simulate one independent pose and source voxel, then emit registered coincidence tallies."""
    import json
    import uproot

    from Gimntools.GimnRec.physical_response import _dimensions

    gate = _open_gate()
    window = energy_window if energy_window is not None else geometry["detector"]["energy_window_kev"]
    if any(v is None for v in window) or not np.isfinite(window).all() or not 0 <= window[0] < window[1]:
        raise ValueError("Provide the two acquisition energy bounds in keV")
    expected = geometry["detector"]["energy_window_kev"]
    if all(v is not None for v in expected) and list(window) != expected:
        raise ValueError("Calibration energy window differs from the acquisition plan")
    poses, voxels, _, crystals = _dimensions(geometry)
    with TemporaryDirectory(prefix="gimntools-gate-") as directory:
        sim, stats, channels = _simulation(gate, geometry, pose, voxel, primaries, seed, directory)
        sim.run()
        emitted = int(stats.counts.events)
        if emitted != primaries:
            raise RuntimeError(f"Geant4 simulated {emitted} of {primaries} requested primaries")
        events, energies, identifiers = [], [], []
        for index, name in enumerate(channels):
            path = Path(directory) / f"{name}.root"
            if not path.exists():
                continue
            with uproot.open(path) as root:
                data = root[name].arrays(["EventID", "TotalEnergyDeposit"], library="np")
            events.extend(data["EventID"])
            energies.extend(data["TotalEnergyDeposit"] / gate.g4_units.keV)
            identifiers.extend(np.full(len(data["EventID"]), index, dtype=np.int64))
        first, second = _readout(np.asarray(events, dtype=np.int64), np.asarray(identifiers, dtype=np.int64),
                                 energies, emitted, crystals, window)
    input_counts = np.zeros((poses, voxels), dtype=np.int64)
    input_counts[pose, voxel] = emitted
    np.savez_compressed(output, emitted=input_counts, pose=np.full(len(first), pose, dtype=np.int64),
                       voxel=np.full(len(first), voxel, dtype=np.int64), first=first, second=second,
                       geometry_sha256=signature(geometry), seed=seed,
                       provenance_json=json.dumps({"engine": "OpenGATE", "version": version("opengate"),
                           "physics_list": PHYSICS_LIST, "energy_window_kev": list(window),
                           "readout": READOUT}))
