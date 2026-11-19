"""Fixed physical regions for comparing contrast and noise across EM sweeps."""

from contextlib import nullcontext
import csv
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

DEFAULT_BACKGROUND_GAP_MM = 3.


@dataclass(frozen=True)
class SourceRegion:
    center_zyx: tuple
    axial: tuple
    foreground: tuple
    background: tuple
    background_gap_mm: float | None = None


def detect_source_regions(volume, spacing, *, radius_mm=2., source_count=0,
                          relative_threshold=0.25, search_radius_mm=None, support=None,
                          background_gap_mm=DEFAULT_BACKGROUND_GAP_MM):
    """Locate smoothed local peaks in 3D, separated by two ROI radii.

    Regions are fixed once on a low-iteration reference image. Background
    is a local shell with a transverse exclusion zone around every source.
    """
    image = np.asarray(volume, dtype=float)
    step = np.asarray(spacing, dtype=float)[::-1]
    if (image.ndim != 3 or not image.size or not np.isfinite(image).all()
            or step.shape != (3,) or not np.isfinite(step).all() or np.any(step <= 0)
            or not np.isfinite(radius_mm) or radius_mm <= 0
            or not np.isfinite(background_gap_mm) or background_gap_mm <= 0
            or not 0 < relative_threshold <= 1 or source_count < 0):
        raise ValueError("Check the source ROI radius, voxel spacing and detection settings")
    allowed = np.ones(image.shape, dtype=bool) if support is None else np.broadcast_to(support, image.shape).copy()
    z, y, x = np.ogrid[:image.shape[0], :image.shape[1], :image.shape[2]]
    if search_radius_mm is not None:
        allowed &= (((y - (image.shape[1] - 1) / 2) * step[1])**2
                    + ((x - (image.shape[2] - 1) / 2) * step[2])**2 <= search_radius_mm**2)
    if search_radius_mm is not None and (not np.isfinite(search_radius_mm) or search_radius_mm <= 0):
        raise ValueError("The source search radius must be positive")
    if not allowed.any():
        raise ValueError("The ROI search field has no model support")
    sigma = np.maximum(0.4, 0.6 * radius_mm / step)
    clean = np.maximum(image, 0)
    neighbors = np.ones((3, 3, 3), dtype=bool)
    neighbors[1, 1, 1] = False
    local_maximum = maximum_filter(clean, footprint=neighbors, mode="constant")
    upper = np.maximum(8 * local_maximum, np.quantile(clean[allowed], 0.995))
    clipped = np.minimum(clean, upper)
    if clipped[allowed].max() == 0:
        clipped = clean
    score = gaussian_filter(clipped, sigma=sigma, mode="nearest")
    score = np.sqrt(score * gaussian_filter(clipped, sigma=1.7 * sigma, mode="nearest"))
    noise = score[allowed]
    typical = np.median(noise)
    spread = 1.4826 * np.median(np.abs(noise - typical))
    peak = noise.max()
    if peak <= typical or peak <= 0:
        raise ValueError("No sources were found in the reference image")
    threshold = typical + max(4 * spread, (peak - typical) * relative_threshold)
    if source_count:
        threshold = typical + min(threshold - typical, 0.05 * (peak - typical))
    maxima = np.argwhere(allowed & (score >= threshold) & (score == maximum_filter(score, size=3)))
    centres = []
    ranked = sorted(maxima, key=lambda index: score[tuple(index)], reverse=True)
    for index in ranked:
        if all(np.linalg.norm((index - chosen) * step) >= 2 * radius_mm for chosen in centres):
            centres.append(index)
        if source_count and len(centres) == source_count:
            break
    if not centres:
        raise ValueError("No source met the detection threshold. Set an expected source count or lower the threshold")

    transverse = [((y - p[1]) * step[1])**2 + ((x - p[2]) * step[2])**2 for p in centres]
    inner = radius_mm + background_gap_mm
    outer = inner + 1.95 * radius_mm
    clear = np.minimum.reduce(transverse) >= inner**2
    regions = []
    for index, in_plane in zip(centres, transverse):
        distance = ((z - index[0]) * step[0])**2 + in_plane
        foreground = (distance <= radius_mm**2) & allowed
        background = (distance <= outer**2) & clear & allowed
        if np.count_nonzero(background) < 8:
            p = tuple(int(i) for i in index)
            raise ValueError(f"Fewer than 8 background voxels near source {p}. "
                             "Check the source ROI, background gap and search radius.")
        axial = tuple(np.flatnonzero(np.any(foreground | background, axis=(1, 2))))
        regions.append(SourceRegion(tuple(int(i) for i in index), axial,
                                    tuple(np.flatnonzero(foreground[k]) for k in axial),
                                    tuple(np.flatnonzero(background[k]) for k in axial),
                                    background_gap_mm))
    return regions


def _sample_stats(samples):
    if samples.size == 0:
        return np.nan, np.nan
    return float(np.mean(samples)), float(np.std(samples, ddof=1)) if samples.size > 1 else np.nan


def _winsor_stats(samples):
    if samples.size >= 12:
        limits = np.quantile(samples, [0.05, 0.95])
        samples = np.clip(samples, *limits)
    return _sample_stats(samples)


def convergence_series(run, regions, iterations, subsets, *, cor_mm, label, recording=None):
    """Measure selected checkpoints using one EM run per subset setting.

    run(iterations, subsets, checkpoint) calls checkpoint(z, iteration, XY)
    after each complete pass on that slice. Only samples in the ROIs are kept.
    """
    rows = []
    iterations = sorted(set(iterations))
    for subset in sorted(set(subsets)):
        samples = {it: [[[], []] for _ in regions] for it in iterations}
        masks = {}
        for index, region in enumerate(regions):
            for z, signal, background in zip(region.axial, region.foreground, region.background):
                masks.setdefault(z, []).append((index, signal, background))

        def collect(z, iteration, image):
            if iteration in samples:
                values = image.ravel()
                for index, signal, background in masks.get(z, ()):
                    samples[iteration][index][0].append(values[signal])
                    samples[iteration][index][1].append(values[background])

        context = recording(subset, iterations[-1]) if recording else nullcontext()
        with context as write:
            if write is None:
                run(iterations[-1], subset, collect)
            else:
                run(iterations[-1], subset, collect, update_callback=write)
        for iteration in iterations:
            for source, (signal, background) in enumerate(samples[iteration], 1):
                mean, _ = _winsor_stats(np.concatenate(signal) if signal else np.empty(0))
                background = np.concatenate(background) if background else np.empty(0)
                bg, noise = _winsor_stats(background)
                raw_bg, raw_noise = _sample_stats(background)
                rows.append({
                    "case": label, "cor_mm": cor_mm, "source": source, "iterations": iteration,
                    **dict(zip(("z_index", "y_index", "x_index"), regions[source - 1].center_zyx)),
                    "subsets": subset, "updates": iteration * subset,
                    "signal": mean, "background": bg, "background_sd": noise,
                    "background_voxels": background.size,
                    "background_gap_mm": regions[source - 1].background_gap_mm,
                    "background_raw_mean": raw_bg, "background_raw_sd": raw_noise,
                    "contrast_pct": 100 * (mean - bg) / bg if bg > 0 else np.nan,
                    "noise_pct": 100 * noise / bg if bg > 0 else np.nan,
                    "background_cv_pct": 100 * raw_noise / raw_bg if raw_bg > 0 else np.nan,
                    "cnr": (mean - bg) / noise if noise > 0 else np.nan,
                })
    return rows


def save_convergence_csv(path, rows):
    """Write one row per source, checkpoint and reconstruction setting."""
    with open(path, "w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def convergence_operating_point(rows):
    """Return the cheapest setting within 95% of peak normalized median CNR."""
    if not rows:
        return None
    per_setting, peaks = {}, {}
    for row in rows:
        if row["case"] != rows[0]["case"] or not np.isfinite(row["cnr"]) or row["cnr"] <= 0:
            continue
        source = row["source"]
        peaks[source] = max(peaks.get(source, 0.), row["cnr"])
        key = (row["updates"], row["iterations"], row["subsets"])
        per_setting.setdefault(key, {})[source] = row["cnr"]
    scores = {key: np.median([cnr / peaks[src] for src, cnr in sample.items()])
              for key, sample in per_setting.items() if sample.keys() == peaks.keys()}
    if not scores:
        return None
    viable = [key for key, score in scores.items() if score >= .95 * max(scores.values())]
    _, iterations, subsets = min(viable)
    return iterations, subsets
