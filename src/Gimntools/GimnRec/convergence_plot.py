"""Convergence curves on comparable axes, one acquisition per column."""

from textwrap import fill

from matplotlib.pyplot import get_cmap
import numpy as np


ROBUST_CV = "Robust CV (%)"
CLASSICAL_CV = "Classical CV (%)"
NOISE_METRICS = (ROBUST_CV, CLASSICAL_CV, "Both")


def convergence_metrics(signal, noise, *, comparison=False):
    values = [("contrast_pct", "Contrast (%)"), ("cnr", "CNR")]
    if not comparison:
        values = [values[1] if signal == "CNR" else values[0]]
    if noise != CLASSICAL_CV:
        values.append(("noise_pct", "Robust background CV (%)"))
    if noise != ROBUST_CV:
        values.append(("background_cv_pct", "Classical background CV (%)"))
    return values


def draw_convergence_chart(figure, acquisitions, metrics, *, x_axis="Iterations", source="All",
                           title="", family="sans-serif", legend="best", style="Lines + points",
                           palette="tab10"):
    """Fill a figure with identical metric rows and shared scales across acquisitions."""
    figure.clear()
    axes = figure.subplots(len(metrics), max(1, len(acquisitions)),
                           squeeze=False, sharex=True, sharey="row")
    x_field = "updates" if x_axis == "Subset updates" else "iterations"
    if (title and len(acquisitions) > 1) or not acquisitions:
        figure.suptitle(title or "Run a convergence sweep to compare EM settings.", fontfamily=family)
    groups = {}
    for name, rows in acquisitions.items():
        series = groups[name] = {}
        for row in rows:
            if source in ("All", str(row["source"])):
                series.setdefault((row["case"], row["source"], row["subsets"]), []).append(row)
    subsets = sorted({key[2] for series in groups.values() for key in series})
    cmap = get_cmap(palette)
    colors = dict(zip(subsets, cmap(np.linspace(0, 0.9, len(subsets)))))
    markers = ("o", "s", "D", "P", "v", "X", "*")
    solid_case = ("-", ":", "-.")
    estimated_case = ("--", (0, (5, 2, 1, 2)), (0, (4, 2, 2, 2)))

    for col, (name, series) in enumerate(groups.items()):
        axes[0, col].set_title(fill(name, width=32) if len(groups) > 1 else title or name,
                               fontfamily=family)
        show_case = len({key[0] for key in series}) > 1
        for (case, source_id, subset), points in series.items():
            points = sorted(points, key=lambda point: point[x_field])
            x = [point[x_field] for point in points]
            mark = markers[(source_id - 1) % len(markers)] if style != "Lines" else None
            estimated = case == "Estimated COR"
            patterns = estimated_case if estimated else solid_case
            line = patterns[(source_id - 1) % len(patterns)]
            label = f"Source {source_id} · {subset} subset(s)"
            if show_case:
                label += f" · {case}"
            for row, (key, _label) in enumerate(metrics):
                axes[row, col].plot(
                    x, [point.get(key, np.nan) for point in points],
                    color=colors[subset], linestyle="None" if style == "Points" else line,
                    marker=mark, markersize=3.5, markerfacecolor="none" if estimated else colors[subset],
                    label=label,
                )
        if series and legend != "off":
            axes[0, col].legend(loc=legend, prop={"family": family, "size": 8})
        elif not series and source != "All":
            axes[0, col].annotate(f"Source {source} unavailable", (.5, .5), xycoords="axes fraction",
                                  ha="center", fontfamily=family)

    for row, (_key, label) in enumerate(metrics):
        axes[row, 0].set_ylabel(label, fontfamily=family)
        for axis in axes[row, :]:
            axis.grid(True, alpha=0.22)
            for tick in axis.get_xticklabels() + axis.get_yticklabels():
                tick.set_fontfamily(family)
    for axis in axes[-1, :]:
        axis.set_xlabel(x_axis, fontfamily=family)
    return axes
