"""Displacement-field generator: Slicer TPS chain -> displacement field, in pure numpy.

Public API:
    load_tps, SlicerTPS            (tps)
    ReferenceGrid, CCF_10UM        (grids)
    TransformChain                 (compose)
    sample_reference, calibrate    (validate)
"""
# Cap BLAS threads to 1 BEFORE numpy is imported. We parallelize with processes (--jobs); if each
# worker also multi-threads BLAS you get N x N thread oversubscription and the CPU thrashes
# (this turned a ~2h job into 60h). Must run before the numpy import below.
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

# Code Ocean can launch the reproducible run PINNED to a single CPU on a multi-core instance
# (nproc then reports 1, and the worker pool can't spread out — the whole job runs serially).
# Re-expand this process's CPU affinity to every installed core so children inherit all of them.
# Best-effort: no-op on macOS (no sched_setaffinity) or if a cpuset genuinely restricts us.
try:
    if hasattr(_os, "sched_setaffinity"):
        _os.sched_setaffinity(0, set(range(_os.cpu_count() or 1)))
except Exception:
    pass

from .tps import SlicerTPS, load_tps, peek_type
from .grids import ReferenceGrid, CCF_10UM
from .compose import TransformChain
from .validate import sample_reference, calibrate

__all__ = [
    "SlicerTPS", "load_tps", "peek_type",
    "ReferenceGrid", "CCF_10UM",
    "TransformChain",
    "sample_reference", "calibrate",
]
