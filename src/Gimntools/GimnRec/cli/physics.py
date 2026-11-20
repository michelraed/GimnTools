"""Prepare, inspect and use calibrated crystal-pair response archives."""

import argparse
import json
from pathlib import Path

import numpy as np

from Gimntools.GimnRec.listmode import CALIBRATION_FIELDS, load_listmode, reconstruct_listmode
from Gimntools.GimnRec.listmode_response import prepare_listmode_response
from Gimntools.GimnRec.physical_response import read_physical_pairs, signature, write_physical_pairs


def _grid(parser):
    parser.add_argument("events", type=Path, help="List-mode acquisition NPZ")
    parser.add_argument("--matrix", nargs=3, type=int, metavar=("X", "Y", "Z"), required=True)
    parser.add_argument("--voxel", nargs=3, type=float, metavar=("X", "Y", "Z"), required=True)
    parser.add_argument("--center", nargs=3, type=float, default=(0.,)*3)
    parser.add_argument("--petsys-radius", type=float)
    parser.add_argument("--energy-window", nargs=2, type=float, metavar=("MIN_KEV", "MAX_KEV"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Save the grid and readout geometry for offline calibration")
    _grid(plan)
    plan.add_argument("output", type=Path)
    build = commands.add_parser("build", help="Create a physical operator from an independent coincidence tally")
    build.add_argument("plan", type=Path)
    build.add_argument("output", type=Path)
    build.add_argument("provenance", type=Path)
    build.add_argument("tally", type=Path, nargs="+")
    inspect = commands.add_parser("inspect", help="Validate a physical response and print its metadata")
    inspect.add_argument("lut", type=Path)
    reconstruct = commands.add_parser("reconstruct", help="Reconstruct a list-mode archive to MetaImage")
    _grid(reconstruct)
    reconstruct.add_argument("output", type=Path)
    reconstruct.add_argument("--physical-lut", type=Path)
    reconstruct.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    reconstruct.add_argument("--iterations", type=int, default=5)
    reconstruct.add_argument("--subsets", type=int, default=1)
    gate = commands.add_parser("opengate-voxel", help="Simulate one independent calibration voxel and pose")
    gate.add_argument("plan", type=Path)
    gate.add_argument("output", type=Path)
    gate.add_argument("--pose", type=int, required=True)
    gate.add_argument("--voxel-index", type=int, required=True)
    gate.add_argument("--primaries", type=int, required=True)
    gate.add_argument("--seed", type=int, required=True)
    gate.add_argument("--energy-window", nargs=2, type=float, metavar=("MIN_KEV", "MAX_KEV"))
    args = parser.parse_args(argv)
    if args.command == "inspect":
        model = read_physical_pairs(args.lut)
        probability, _, error = model.efficiency()
        print(json.dumps({"geometry": model.geometry, "provenance": model.provenance,
                          "accepted": int(model.counts.sum()), "primaries": int(model.emitted.sum()),
                          "maximum_efficiency": float(probability.max()), "maximum_standard_error": float(error.max())},
                         indent=2))
        return
    if args.command in ("build", "opengate-voxel"):
        spec = json.loads(args.plan.read_text(encoding="utf-8"))
        if args.command == "opengate-voxel":
            from Gimntools.GimnRec.opengate_calibration import simulate_voxel

            simulate_voxel(spec, args.output, pose=args.pose, voxel=args.voxel_index,
                           primaries=args.primaries, seed=args.seed, energy_window=args.energy_window)
            return
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
        emitted = None
        columns = ("pose", "voxel", "first", "second")
        chunks = {name: [] for name in columns}
        seeds = []
        for path in args.tally:
            with np.load(path, allow_pickle=False) as data:
                if "geometry_sha256" in data and data["geometry_sha256"].item() != signature(spec):
                    raise ValueError(f"{path.name} was simulated with a different geometry")
                if "seed" in data:
                    seeds.append(int(data["seed"].item()))
                if "provenance_json" in data:
                    for key, value in json.loads(data["provenance_json"].item()).items():
                        if key in provenance and provenance[key] != value:
                            raise ValueError(f"{path.name} has inconsistent calibration provenance for {key}")
                        provenance[key] = value
                primaries = data["emitted"]
                if primaries.dtype.kind not in "iu" or np.any(primaries < 0):
                    raise ValueError(f"{path.name} has invalid emitted primary counts")
                if emitted is None:
                    emitted = primaries.astype(np.int64)
                else:
                    if primaries.shape != emitted.shape:
                        raise ValueError(f"{path.name} has a different tally grid")
                    emitted += primaries.astype(np.int64)
                for name in columns:
                    chunks[name].append(data[name])
        if len(seeds) != len(set(seeds)):
            raise ValueError("OpenGATE tallies in a calibration must have independent seeds")
        if seeds:
            provenance["seeds"] = seeds
        write_physical_pairs(args.output, spec, provenance, emitted=emitted,
                             **{key: np.concatenate(parts) for key, parts in chunks.items()})
        return

    rays, metadata = load_listmode(args.events, args.energy_window)
    shape = tuple(args.matrix[::-1])
    if args.command == "plan":
        if metadata["detector_model"] is None:
            raise ValueError("An acquisition NPZ with saved discrete crystal geometry is required")
        spec = prepare_listmode_response(rays, shape, args.voxel, args.center, metadata["detector_model"],
                                         detector_radius_mm=args.petsys_radius, describe=True,
                                         energy_window=args.energy_window)
        args.output.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    else:
        import SimpleITK as sitk

        volume, geometry = reconstruct_listmode(rays, shape, args.voxel, center=args.center,
            detector_radius_mm=args.petsys_radius, physical_lut=args.physical_lut, backend=args.backend,
            iterations=args.iterations, subsets=args.subsets, energy_window=args.energy_window,
            **{name: metadata[name] for name in CALIBRATION_FIELDS})
        image = sitk.GetImageFromArray(volume.astype(np.float32))
        image.SetSpacing(tuple(args.voxel))
        image.SetOrigin(geometry["origin"])
        sitk.WriteImage(image, str(args.output), True)
        print(json.dumps(geometry, indent=2))


if __name__ == "__main__":
    main()
