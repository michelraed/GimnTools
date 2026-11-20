"""Benchmark the physical CSR format and CPU geometric baseline on synthetic data."""

import argparse
import csv
import json
import platform
import resource
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from Gimntools.GimnRec.physical_response import read_physical_pairs, write_physical_pairs
from Gimntools.GimnRec.reconstructor import siddon_reconstructor


def timed(operation, repeats=1):
    started = perf_counter()
    for _ in range(repeats):
        result = operation()
    return result, (perf_counter() - started) / repeats


def benchmark(n, count, updates, directory):
    angles = np.arange(count) * 180. / count
    voxels = np.arange(n*n)
    yy, xx = np.divmod(voxels, n)
    active = ((xx - n/2)**2 + (yy - n/2)**2) < (.43*n)**2
    xx, yy = xx[active] - n/2, yy[active] - n/2
    phase = np.repeat(np.arange(count), np.count_nonzero(active))
    radians = np.deg2rad(angles)[:, None]
    channel = np.rint(np.cos(radians)*xx + np.sin(radians)*yy + n/2).astype(int).ravel()
    voxel = np.tile(voxels[active], count)
    geometry = {"image_grid": {"shape_zyx": [1, n, n]}, "pose_degrees": angles.tolist(),
                "detector": {"heads": ["left", "right"], "pixels": [n, 1]}}
    emitted = np.full((count, n*n), 100, dtype=np.int64)
    archive = directory / "response.npz"
    _, write_s = timed(lambda: write_physical_pairs(archive, geometry,
        {"engine": "synthetic", "version": "1", "readout": "one rotated nearest channel"},
        emitted=emitted, pose=phase, voxel=voxel, first=channel, second=channel+n, count=np.full(len(phase), 10)))
    loaded, load_s = timed(lambda: read_physical_pairs(archive, geometry))
    matrices, prepare_s = timed(lambda: [loaded.pose_matrix(i)[np.unique(channel[phase == i])*n +
                              np.unique(channel[phase == i])] for i in range(count)])
    sensitivity, sensitivity_s = timed(lambda: loaded.sensitivity_by_pose.mean(axis=0))
    _, sensitivity_recompute_s = timed(lambda: np.mean([np.asarray(a.sum(axis=0)).ravel() for a in matrices], axis=0))
    x = np.ones(n*n, dtype=float)
    data = [np.ones(a.shape[0]) for a in matrices]
    _, forward_s = timed(lambda: [a @ x for a in matrices], repeats=10)
    _, backward_s = timed(lambda: sum(a.T @ b for a, b in zip(matrices, data)), repeats=10)
    left = sum((a @ x) @ b for a, b in zip(matrices, data))
    right = x @ sum(a.T @ b for a, b in zip(matrices, data))
    started = perf_counter()
    for _ in range(updates):
        ratio = [np.divide(b, pred, out=np.zeros_like(b), where=pred > 0)
                 for a, b in zip(matrices, data) for pred in (a @ x,)]
        back = sum(a.T @ b for a, b in zip(matrices, ratio))
        x[active] *= back[active] / (sensitivity[active] * count)
    recon_s = perf_counter() - started

    baseline = siddon_reconstructor(np.ones((1, n, count)), fov_mm=60., sino_range_mm=60.,
                                     n_img=n, rays_per_bin=1, verbose=False)
    _, baseline_first_s = timed(lambda: baseline.mlem(angles, iterations=1, verbose=False))
    _, baseline_s = timed(lambda: baseline.mlem(angles, iterations=updates, verbose=False))
    return dict(matrix=n, poses=count, updates=updates, physical_write_s=write_s,
                physical_load_s=load_s, operator_prepare_s=prepare_s, sensitivity_load_s=sensitivity_s,
                sensitivity_recompute_s=sensitivity_recompute_s, forward_s=forward_s, backward_s=backward_s,
                update_s=recon_s/updates, reconstruction_s=recon_s, baseline_first_update_s=baseline_first_s,
                baseline_reconstruction_s=baseline_s, relative_adjoint_error=abs(left-right)/max(1., abs(left)),
                archive_bytes=archive.stat().st_size, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--updates", type=int, default=40)
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    cold = siddon_reconstructor(np.ones((1, 8, 4)), fov_mm=8., n_img=8, verbose=False)
    _, warmup_s = timed(lambda: cold.mlem(np.arange(4)*45., iterations=1, verbose=False))
    cpu = platform.processor() or platform.machine()
    if Path("/proc/cpuinfo").is_file():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    with TemporaryDirectory() as directory:
        rows = [benchmark(n, poses, args.updates, Path(directory))
                for n in (64, 128, 256) for poses in (12, 24)]
    with (args.output / "benchmark.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "benchmark.json").write_text(json.dumps(
        {"cpu": cpu, "platform": platform.platform(), "process_warmup_s": warmup_s, "scenarios": rows},
        indent=2), encoding="utf-8")
    print(f"{len(rows)} scenarios · {cpu} · initial call {warmup_s:.3f} s")


if __name__ == "__main__":
    main()
