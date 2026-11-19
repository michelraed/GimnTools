"""Analytical PET activity phantoms in object XYZ coordinates (mm)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Insert:
    center: tuple
    radius: float
    kind: str


@dataclass(frozen=True)
class Phantom:
    kind: str
    diameter_mm: float
    height_mm: float
    insert_to_background: float
    inserts: tuple
    sector_angles_deg: tuple = ()

    def sample(self, rng, count):
        """Sample activity by concentration times volume in disjoint regions."""
        radius = self.diameter_mm / 2
        volumes = np.array([
            np.pi * i.radius**2 * self.height_mm if i.kind == "rod"
            else 4 * np.pi / 3 * i.radius**3 for i in self.inserts
        ])
        background_volume = np.pi * radius**2 * self.height_mm - volumes.sum()
        # Rescale before multiplication so large finite activity ratios still work.
        scale = max(1., self.insert_to_background)
        weights = np.r_[background_volume / scale,
                        volumes * (self.insert_to_background / scale)]
        counts = rng.multinomial(count, weights / weights.sum())
        points = np.empty((count, 3))
        remaining = np.arange(counts[0])
        while remaining.size:
            candidates = _cylinder(rng, remaining.size, radius, self.height_mm)
            allowed = np.ones(remaining.size, dtype=bool)
            for insert in self.inserts:
                dimensions = 2 if insert.kind == "rod" else 3
                delta = candidates[:, :dimensions] - insert.center[:dimensions]
                allowed &= np.einsum("ij,ij->i", delta, delta) >= insert.radius**2
            points[remaining[allowed]] = candidates[allowed]
            remaining = remaining[~allowed]
        first = counts[0]
        for i, size in zip(self.inserts, counts[1:]):
            if i.kind == "rod":
                other = _cylinder(rng, size, i.radius, self.height_mm)
            else:
                other = unit_directions(rng, size) * (i.radius * rng.random(size)**(1 / 3))[:, None]
            points[first:first + size] = other + i.center
            first += size
        rng.shuffle(points)
        return points


def unit_directions(rng, count):
    azimuth = rng.uniform(0, 2 * np.pi, count)
    z = rng.uniform(-1, 1, count)
    radial = np.sqrt(1 - z*z)
    return np.column_stack((radial * np.cos(azimuth), radial * np.sin(azimuth), z))


def _cylinder(rng, count, radius, height):
    azimuth = rng.uniform(0, 2 * np.pi, count)
    radial = radius * np.sqrt(rng.random(count))
    return np.column_stack((radial * np.cos(azimuth), radial * np.sin(azimuth),
                            rng.uniform(-height / 2, height / 2, count)))


def _derenzo_inserts(body_radius, diameters, widths_deg):
    sectors = len(diameters)
    edge = -np.deg2rad(widths_deg[0]) / 2
    for sector, (diameter, width_deg) in enumerate(zip(diameters, widths_deg)):
        radius = diameter / 2
        pitch = 2 * diameter
        half_angle = np.deg2rad(width_deg) / 2
        angle = edge + half_angle
        edge += 2 * half_angle
        cosine, sine = np.cos(angle), np.sin(angle)
        narrow = sectors > 1 and half_angle <= np.pi / 2
        base_x = radius / np.sin(half_angle) if narrow else 0.
        limit = body_radius - radius
        rods = []
        if limit >= 0:
            row_pitch = np.sqrt(3) / 2 * pitch
            rows = int(limit / row_pitch)
            for row in range(-rows, rows + 1):
                y = row * row_pitch
                offset = (row % 2) * pitch / 2
                xmax = np.sqrt(max(0., limit**2 - y**2))
                xmin = base_x + abs(y) / np.tan(half_angle) if narrow else -xmax
                # Snap to a triangular lattice, offset to fit a rod near the sector tip.
                base = offset + base_x
                for col in range(int(np.ceil((xmin - base) / pitch)), int(np.floor((xmax - base) / pitch)) + 1):
                    x = base + col * pitch
                    if sectors > 1 and not narrow:
                        separation = half_angle - abs(np.arctan2(y, x))
                        clearance = np.hypot(x, y) * np.sin(min(separation, np.pi / 2))
                        if clearance < radius:
                            continue
                    rods.append(Insert((cosine*x - sine*y, sine*x + cosine*y, 0.), radius, "rod"))
        if not rods:
            raise ValueError(f"Derenzo sector {sector + 1} cannot fit a {diameter:g} mm rod")
        yield from rods


def _iec_inserts(radius, height, diameters):
    radii = diameters / 2
    if np.max(radii) > min(radius, height / 2):
        raise ValueError("An IEC sphere extends outside the phantom")
    if len(radii) == 1:
        return (Insert((0., 0., 0.), radii[0], "sphere"),)
    angles = np.linspace(0., 2 * np.pi, len(radii), endpoint=False)
    xy = np.column_stack((np.cos(angles), np.sin(angles)))
    pairs = np.triu_indices(len(radii), k=1)
    min_ring = np.max((radii[pairs[0]] + radii[pairs[1]]) /
                      np.linalg.norm(xy[pairs[0]] - xy[pairs[1]], axis=1))
    max_ring = radius - np.max(radii)
    if min_ring > max_ring:
        raise ValueError("The IEC spheres overlap at this phantom diameter. Decrease sphere diameters or enlarge the body")
    return tuple(Insert(tuple(center) + (0.,), r, "sphere")
                 for center, r in zip(xy * ((min_ring + max_ring) / 2), radii))


def make_phantom(kind, diameter_mm, height_mm, *, contrast=4., sectors=6, diameters=(),
                 sector_angles_deg=()):
    """Build a homogeneous cylinder, a Derenzo rod phantom or IEC-inspired spheres."""
    body = np.array([diameter_mm, height_mm, contrast], dtype=float)
    if not np.isfinite(body).all() or np.any(body[:2] <= 0) or contrast < 0:
        raise ValueError("Enter positive phantom dimensions and a nonnegative insert/background ratio")
    angles = ()
    if kind == "Homogeneous":
        inserts = ()
    else:
        sizes = np.asarray(diameters, dtype=float)
        if sizes.ndim != 1 or not sizes.size or not np.isfinite(sizes).all() or np.any(sizes <= 0):
            raise ValueError("Enter positive insert diameters")
        if kind == "Derenzo":
            if sectors != len(sizes):
                raise ValueError("Enter one rod diameter per Derenzo sector")
            widths = np.asarray(sector_angles_deg, dtype=float)
            if widths.size == 0:
                widths = np.full(sectors, 360. / sectors)
            if (widths.shape != (sectors,) or not np.isfinite(widths).all() or np.any(widths <= 0)
                    or not np.isclose(widths.sum(), 360., rtol=0., atol=1e-6)):
                raise ValueError("Enter one positive angle per Derenzo sector. The angles must total 360°")
            angles = tuple(widths.tolist())
            inserts = tuple(_derenzo_inserts(diameter_mm / 2, sizes, angles))
        elif kind == "IEC-inspired":
            inserts = _iec_inserts(diameter_mm / 2, height_mm, sizes)
        else:
            raise ValueError(f"Unknown phantom type {kind!r}")
    return Phantom(kind, float(diameter_mm), float(height_mm), float(contrast), inserts, angles)
