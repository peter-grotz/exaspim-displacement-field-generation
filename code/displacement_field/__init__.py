"""Displacement-field generator: Slicer TPS chain -> displacement field, in pure numpy.

Public API:
    load_tps, SlicerTPS            (tps)
    ReferenceGrid, CCF_10UM        (grids)
    TransformChain                 (compose)
    sample_reference, calibrate    (validate)
"""
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
