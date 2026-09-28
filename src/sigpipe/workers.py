"""The threads of the processes jobs run their work in: one each, so that a job's workers are
the cores it takes. Importable before numpy loads (nothing of sigpipe's is imported here)."""

import os

# What starts threads of its own in a process: numpy's BLAS (OpenBLAS, MKL, Accelerate),
# OpenMP (numba, oneDNN), numexpr, TensorFlow's pools (the Silex models). Each reads its
# variable once, when it loads.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
)


def one_thread_each() -> None:
    """The processes started from now on run one thread each, unless the environment says
    otherwise. Measured on 2026-09-29 (12 cores): 2 inversion workers took 8 to 12 of them (the
    chains' processes, then numpy's BLAS threads), 2 petrophysical ones 7 (TensorFlow's pools);
    2 each, with one thread each. A process reads it when it starts, and a library when it
    loads: call it before the first process pool (the processes of a forkserver started before
    keep its environment), and before numpy loads for this process to keep to one as well."""
    for name in THREAD_VARIABLES:
        os.environ.setdefault(name, "1")
