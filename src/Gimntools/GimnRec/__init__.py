"""GimnRec — sinogram building and tomographic reconstruction (PET).

Public API (re-exported here for convenience)
---------------------------------------------
Sinogram building
    :func:`~Gimntools.GimnRec.utils.coincidence_to_lor`
Reconstruction
    :class:`~Gimntools.GimnRec.reconstructor.siddon_reconstructor`
    (FBP / MLEM / OSEM), :func:`~Gimntools.GimnRec.reconstructor.build_subsets`,
    :func:`~Gimntools.GimnRec.reconstructor.check_subset_count`
Pre-processing
    :func:`~Gimntools.GimnRec.filters.interpolate_distance`,
    :func:`~Gimntools.GimnRec.filters.smooth_angular`,
    :func:`~Gimntools.GimnRec.filters.make_psf_kernel`,
    :func:`~Gimntools.GimnRec.filters.ramLak`,
    :func:`~Gimntools.GimnRec.filters.cosine_filter`
Saving volumes
    :func:`~Gimntools.GimnRec.utils.save_reconstructions`
    (Portuguese alias kept: ``salvar_reconstrucoes``)
High-level pipeline
    :func:`~Gimntools.GimnRec.reconstruction_v2.process_file_v2`

Import example
--------------
>>> from Gimntools.GimnRec import siddon_reconstructor, coincidence_to_lor

Heavy optional dependencies (uproot, SimpleITK, matplotlib) are only
imported when the corresponding submodule is first used; this ``__init__``
therefore imports lazily and never fails at package-import time just
because an optional dependency is missing.
"""

__version__ = "2.0.0"

__all__ = [
    "siddon_reconstructor",
    "build_subsets",
    "check_subset_count",
    "coincidence_to_lor",
    "save_reconstructions",
    "salvar_reconstrucoes",
    "interpolate_distance",
    "smooth_angular",
    "make_psf_kernel",
    "ramLak",
    "cosine_filter",
    "hann_filter",
    "hamming_filter",
    "shepp_logan_filter",
    "process_file_v2",
    "geometric_norm_sinogram",
    "apply_norm_to_sinogram",
]

_LAZY = {
    "siddon_reconstructor": ("Gimntools.GimnRec.reconstructor", "siddon_reconstructor"),
    "build_subsets": ("Gimntools.GimnRec.reconstructor", "build_subsets"),
    "check_subset_count": ("Gimntools.GimnRec.reconstructor", "check_subset_count"),
    "coincidence_to_lor": ("Gimntools.GimnRec.utils", "coincidence_to_lor"),
    "save_reconstructions": ("Gimntools.GimnRec.utils", "save_reconstructions"),
    "salvar_reconstrucoes": ("Gimntools.GimnRec.utils", "save_reconstructions"),
    "interpolate_distance": ("Gimntools.GimnRec.filters", "interpolate_distance"),
    "smooth_angular": ("Gimntools.GimnRec.filters", "smooth_angular"),
    "make_psf_kernel": ("Gimntools.GimnRec.filters", "make_psf_kernel"),
    "ramLak": ("Gimntools.GimnRec.filters", "ramLak"),
    "cosine_filter": ("Gimntools.GimnRec.filters", "cosine_filter"),
    "hann_filter": ("Gimntools.GimnRec.filters", "hann_filter"),
    "hamming_filter": ("Gimntools.GimnRec.filters", "hamming_filter"),
    "shepp_logan_filter": ("Gimntools.GimnRec.filters", "shepp_logan_filter"),
    "process_file_v2": ("Gimntools.GimnRec.reconstruction_v2", "process_file_v2"),
    "geometric_norm_sinogram": ("Gimntools.GimnRec.normalization", "geometric_norm_sinogram"),
    "apply_norm_to_sinogram": ("Gimntools.GimnRec.normalization", "apply_norm_to_sinogram"),
}


def __getattr__(name):
    """PEP 562 lazy attribute access for the public API."""
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
