"""Run the CUDA runtime in a subprocess so a native failure cannot close the desktop."""

from collections import OrderedDict
from itertools import count
import os
from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import TemporaryFile
from threading import Lock
from traceback import format_exc
from weakref import finalize, ref


_WORKER_COMMAND = (sys.executable, "-u", "-X", "faulthandler", "-m", __spec__.name)
_HOST_CACHE_BYTES = 128 * 1024**2


class CudaWorkerError(RuntimeError):
    def __init__(self, message, details=""):
        super().__init__(message)
        self.details = details

    def __str__(self):
        if self.details:
            return f"{super().__str__()}\n\n{self.details}"
        return super().__str__()


def _send(stream, value):
    pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    stream.flush()


def _stop_worker(process, errors):
    try:
        process.stdin.close()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    finally:
        process.stdout.close()
        errors.close()


class CudaWorker:
    """Expose the accelerator operations while the worker owns the GPU context."""

    def __init__(self):
        self._errors = TemporaryFile()
        # Also support a source checkout opened directly through ui/app.py.
        env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, env.get("PYTHONPATH"))))
        try:
            self._process = subprocess.Popen(_WORKER_COMMAND, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=self._errors, env=env)
        except OSError as exc:
            self._errors.close()
            raise CudaWorkerError(f"Could not start the CUDA worker: {exc}") from exc
        self.close = finalize(self, _stop_worker, self._process, self._errors)
        self._lock = Lock()
        self._owners = OrderedDict()
        self._keys = count()
        try:
            self._receive("initialization")
        except BaseException:
            self.close()
            raise

    def _stopped(self, stage):
        try:
            status = self._process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self._process.kill()
            status = self._process.wait()
        self._errors.seek(0, os.SEEK_END)
        self._errors.seek(max(0, self._errors.tell() - 8192))
        details = self._errors.read().decode("utf-8", errors="replace").strip()
        self.close()
        message = f"CUDA worker stopped during {stage} (process status {status}). Select CPU to continue."
        return CudaWorkerError(message, details)

    def _receive(self, stage):
        try:
            state, result = pickle.load(self._process.stdout)
        except (EOFError, OSError, ValueError, pickle.UnpicklingError):
            raise self._stopped(stage) from None
        if state == "error":
            raise CudaWorkerError(f"CUDA failed during {stage}. Select CPU to continue.", result)
        return state, result

    def _call(self, stage, operation, *args):
        if not self.close.alive:
            raise CudaWorkerError("The CUDA worker is closed. Start a new reconstruction or select CPU.")
        try:
            _send(self._process.stdin, (operation, args))
        except OSError:
            raise self._stopped(stage) from None
        return self._receive(stage)

    def trace_parallel(self, angles, distances, n, pixel, corner, bin_width, samples):
        with self._lock:
            return self._call("sinogram ray tracing", "trace_parallel", angles, distances, n,
                              pixel, corner, bin_width, samples)[1]

    def trace_lors(self, rays, shape, spacing, corner):
        with self._lock:
            return self._call("list-mode ray tracing", "trace_lors", rays, shape, spacing, corner)[1]

    def sparse_product(self, matrix, vector, *, transpose=False):
        with self._lock:
            identity = id(matrix)
            owner, key = self._owners.pop(identity, (None, None))
            present = owner is not None and owner() is matrix
            if not present:
                owner, key = ref(matrix), next(self._keys)
            self._owners[identity] = owner, key
            if len(self._owners) > 256:
                self._owners.popitem(last=False)
            stage = "backprojection" if transpose else "forward projection"
            state, result = self._call(stage, "sparse_product", key, None if present else matrix, vector, transpose)
            if state == "miss":
                _, result = self._call(stage, "sparse_product", key, matrix, vector, transpose)
            return result


def _serve(incoming, outgoing):
    try:
        from Gimntools.GimnRec.cuda_reconstruction import CudaReconstruction

        accelerator = CudaReconstruction()
    except Exception:
        _send(outgoing, ("error", format_exc()))
        return
    _send(outgoing, ("ok", None))

    trace = {"trace_parallel": accelerator.trace_parallel, "trace_lors": accelerator.trace_lors}
    matrices = OrderedDict()
    cached_bytes = 0
    while True:
        try:
            operation, args = pickle.load(incoming)
        except EOFError:
            return
        try:
            if operation == "sparse_product":
                key, matrix, vector, transpose = args
                if matrix is None:
                    if key not in matrices:
                        _send(outgoing, ("miss", None))
                        continue
                    matrix, _ = matrices[key]
                    matrices.move_to_end(key)
                else:
                    matrix = matrix.tocsr()
                    size = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
                    if size <= _HOST_CACHE_BYTES:
                        while cached_bytes + size > _HOST_CACHE_BYTES:
                            _, (_, previous_size) = matrices.popitem(last=False)
                            cached_bytes -= previous_size
                        matrices[key] = matrix, size
                        cached_bytes += size
                result = accelerator.sparse_product(matrix, vector, transpose=transpose)
            else:
                result = trace[operation](*args)
        except Exception:
            _send(outgoing, ("error", format_exc()))
        else:
            _send(outgoing, ("ok", result))


if __name__ == "__main__":
    # Reserve the original stdout pipe for replies, including when a native
    # library prints directly to stdout before Python imports it.
    with os.fdopen(os.dup(sys.stdout.fileno()), "wb") as replies:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        try:
            _serve(sys.stdin.buffer, replies)
        except BrokenPipeError:
            pass
