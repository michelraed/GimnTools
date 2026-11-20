"""Backend selection shared by the reconstruction solvers."""


def reconstruction_accelerator(backend):
    name = str(backend).strip().lower()
    if name == "cpu":
        return None
    if name != "cuda":
        raise ValueError("Select the CPU or CUDA reconstruction backend")
    try:
        from Gimntools.GimnRec.cuda_reconstruction import CudaReconstruction
    except ImportError as exc:
        raise RuntimeError("CUDA support is unavailable. Install GimnTools[cuda] or select CPU") from exc
    return CudaReconstruction()


def sparse_product(matrix, values, accelerator=None, *, transpose=False):
    if accelerator is not None:
        return accelerator.sparse_product(matrix, values, transpose=transpose)
    return (matrix.T if transpose else matrix) @ values
