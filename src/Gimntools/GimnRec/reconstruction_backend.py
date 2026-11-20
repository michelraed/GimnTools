"""Backend selection shared by the reconstruction solvers."""


def reconstruction_accelerator(backend):
    name = str(backend).strip().lower()
    if name == "cpu":
        return None
    if name != "cuda":
        raise ValueError("Select the CPU or CUDA reconstruction backend")
    from Gimntools.GimnRec.cuda_worker import CudaWorker

    return CudaWorker()


def sparse_product(matrix, values, accelerator=None, *, transpose=False):
    if accelerator is not None:
        return accelerator.sparse_product(matrix, values, transpose=transpose)
    return (matrix.T if transpose else matrix) @ values
