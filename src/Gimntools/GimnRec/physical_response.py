"""Independent Monte Carlo detector tallies as a reusable sparse PET operator."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


FORMAT = "gimntools-physical-pairs-v1"


def signature(geometry):
    return hashlib.sha256(json.dumps(geometry, allow_nan=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _integer_vector(values, size=None):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu" or (size is not None and values.size != size):
        raise ValueError("Expected a one-dimensional integer tally column")
    return values.astype(np.int64, copy=False)


def pair_codes(heads, pixels):
    """Return the sorted IDs for every cross-head detector pair."""
    left, right = np.triu_indices(heads, k=1)
    per_head = np.arange(pixels)
    a = left[:, None]*pixels + per_head
    b = right[:, None]*pixels + per_head
    return np.sort((a[:, :, None]*(heads*pixels) + b[:, None, :]).ravel())


def _dimensions(geometry):
    grid = geometry["image_grid"]["shape_zyx"]
    pixels = geometry["detector"]["pixels"]
    heads = len(geometry["detector"]["heads"])
    angles = geometry["pose_degrees"]
    if (len(grid) != 3 or len(pixels) != 2 or heads < 2 or not angles
            or any(type(n) is not int or n <= 0 for n in [*grid, *pixels])):
        raise ValueError("Physical response geometry needs a positive image grid, crystal grid and discrete poses")
    return len(angles), int(np.prod(grid)), heads, int(np.prod(pixels))


def _probabilities(counts, emitted, pairs):
    observed = np.zeros_like(emitted)
    for i, total in enumerate(emitted):
        chunk = counts[i*pairs:(i+1)*pairs]
        observed[i] = np.asarray(chunk.sum(axis=0)).ravel()
    if np.any(observed > emitted):
        raise ValueError("Accepted coincidences exceed simulated primaries in a pose and voxel")
    return observed


@dataclass(frozen=True)
class PhysicalPairs:
    geometry: dict
    provenance: dict
    counts: csr_matrix
    emitted: np.ndarray
    sensitivity_by_pose: np.ndarray

    def pose_matrix(self, index):
        """Expected registered pairs per emitted annihilation at this pose."""
        pairs = self.counts.shape[0] // len(self.emitted)
        matrix = self.counts[index*pairs:(index+1)*pairs].astype(np.float64)
        matrix.data /= self.emitted[index, matrix.indices]
        return matrix

    def efficiency(self):
        """Detection and miss probabilities, and binomial error on detection."""
        p = self.sensitivity_by_pose
        error = np.sqrt(np.divide(p*(1-p), self.emitted, out=np.zeros_like(p), where=self.emitted > 0))
        return p, 1-p, error


def write_physical_pairs(path, geometry, provenance, *, emitted, pose, voxel, first, second, count=None):
    """Write registered pair tallies from an independent coincidence/digitizer simulation."""
    phases, voxels, heads, pixels = _dimensions(geometry)
    if not isinstance(provenance, dict) or any(not provenance.get(k) for k in ("engine", "version", "readout")):
        raise ValueError("Physical response provenance requires engine, version and readout description")
    emitted = np.asarray(emitted)
    if emitted.shape != (phases, voxels) or emitted.dtype.kind not in "iu" or np.any(emitted < 0):
        raise ValueError("emitted must count primaries as a nonnegative integer array (poses, voxels)")
    emitted = emitted.astype(np.int64)
    pose = _integer_vector(pose)
    voxel, first, second = (_integer_vector(v, len(pose)) for v in (voxel, first, second))
    weights = np.ones(len(pose), dtype=np.int64) if count is None else _integer_vector(count, len(pose))
    if (np.any((pose < 0) | (pose >= phases)) or np.any((voxel < 0) | (voxel >= voxels))
            or np.any((first < 0) | (first >= heads*pixels)) or np.any((second < 0) | (second >= heads*pixels))
            or np.any(weights <= 0) or np.any(first//pixels == second//pixels)):
        raise ValueError("Invalid pose, voxel, cross-head channel or multiplicity in the coincidence tallies")
    codes = pair_codes(heads, pixels)
    pair_indices = np.searchsorted(codes, np.minimum(first, second)*(heads*pixels) + np.maximum(first, second))
    counts = coo_matrix((weights, (pose*len(codes) + pair_indices, voxel)),
                        shape=(phases*len(codes), voxels), dtype=np.int64).tocsr()
    if not counts.nnz:
        raise ValueError("The simulation tally has no accepted coincidences")
    accepted = _probabilities(counts, emitted, len(codes))
    sensitivity = np.divide(accepted, emitted, out=np.zeros(accepted.shape), where=emitted > 0)
    path = Path(path)
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        target = Path(tmp.name)
        try:
            np.savez_compressed(tmp, format=FORMAT, geometry_json=json.dumps(geometry, allow_nan=False),
                                geometry_sha256=signature(geometry), provenance_json=json.dumps(provenance, allow_nan=False),
                                data=counts.data, indices=counts.indices, indptr=counts.indptr, emitted=emitted,
                                sensitivity_by_pose=sensitivity)
            tmp.close()
            os.replace(target, path)
        finally:
            target.unlink(missing_ok=True)


def _differences(reference, received):
    return ", ".join(section for section in sorted(reference.keys() | received.keys())
                     if reference.get(section) != received.get(section))


def read_physical_pairs(path, geometry=None):
    """Load and verify a physical operator without a simulator dependency."""
    with np.load(path, allow_pickle=False) as archive:
        try:
            if archive["format"].item() != FORMAT:
                raise ValueError("Choose a GimnTools physical crystal-pair response NPZ")
            saved = json.loads(archive["geometry_json"].item())
            provenance = json.loads(archive["provenance_json"].item())
            if signature(saved) != archive["geometry_sha256"].item():
                raise ValueError("Physical response geometry signature is invalid")
            if geometry is not None and signature(saved) != signature(geometry):
                raise ValueError("Physical response differs in " + _differences(geometry, saved))
            phases, voxels, heads, pixels = _dimensions(saved)
            emitted = archive["emitted"]
            sensitivity = archive["sensitivity_by_pose"]
            sizes = (phases*len(pair_codes(heads, pixels)), voxels)
            values = _integer_vector(archive["data"])
            indices = _integer_vector(archive["indices"], len(values))
            ptr = _integer_vector(archive["indptr"], sizes[0]+1)
            if (ptr[0] != 0 or ptr[-1] != len(values) or np.any(np.diff(ptr) < 0)
                    or np.any((indices < 0) | (indices >= voxels)) or np.any(values <= 0)):
                raise ValueError("Invalid sparse physical response")
            counts = csr_matrix((values, indices, ptr), shape=sizes)
        except KeyError as exc:
            raise ValueError(f"Physical response NPZ lacks {exc.args[0]}") from exc
    if emitted.shape != (phases, voxels) or emitted.dtype.kind not in "iu" or np.any(emitted < 0):
        raise ValueError("Invalid physical response primaries")
    accepted = _probabilities(counts, emitted, sizes[0]//phases)
    expected = np.divide(accepted, emitted, out=np.zeros_like(accepted, dtype=float), where=emitted > 0)
    if not np.array_equal(sensitivity, expected):
        raise ValueError("Stored physical sensitivity disagrees with the validated coincidence tallies")
    return PhysicalPairs(saved, provenance, counts, emitted, sensitivity)
