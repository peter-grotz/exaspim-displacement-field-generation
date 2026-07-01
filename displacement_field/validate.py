"""Memory-safe validation + auto-calibration against a true vector displacement-field .nrrd.

The reference field can be enormous (the CCF 10um vector field is ~14 GB uncompressed), so we
NEVER load it whole. We sample K voxels:
  * raw encoding  -> np.memmap, random access, instant.
  * gzip encoding -> single streaming pass with bounded memory, strided sampling.

``calibrate`` sweeps the open convention flags (chain order x forward/inverse) and reports
which reproduces the reference, so the manual Slicer step is pinned to numeric parity.
"""
from __future__ import annotations

import zlib
from typing import Dict, List

import numpy as np
import nrrd

from .compose import TransformChain
from .grids import ReferenceGrid


def _meta(path: str):
    h = nrrd.read_header(path)
    sizes = list(map(int, h["sizes"]))
    if h.get("dimension") != 4 or 3 not in sizes:
        raise ValueError(f"{path}: not a 3-component vector field (dimension={h.get('dimension')}, "
                         f"sizes={sizes}). This is the scalar-vs-vector check.")
    comp_axis = sizes.index(3)
    if comp_axis != 0:
        raise ValueError(f"{path}: component axis expected first (fastest); got axis {comp_axis}.")
    grid = ReferenceGrid.from_nrrd_header(path)
    return h, grid


def _linear_to_ijk(v: np.ndarray, size):
    ni, nj, nk = size
    i = v % ni
    j = (v // ni) % nj
    k = v // (ni * nj)
    return np.stack([i, j, k], axis=1).astype(np.float64)


def _data_offset(path: str) -> int:
    """Byte offset of the data blob (just past the header's blank line). Reads only a few KB."""
    with open(path, "rb") as fh:
        head = fh.read(65536)
    i = head.find(b"\n\n")
    if i < 0:
        raise ValueError("could not find NRRD header terminator in first 64KB")
    return i + 2


def sample_reference(path: str, k: int = 3000, seed: int = 0):
    """Return (ijk (k,3), world (k,3), ref_disp (k,3), grid) sampled from the reference field.

    MEMORY-SAFE: never loads the whole field. raw -> memmap; gzip -> bounded streaming pass.
    """
    h, grid = _meta(path)
    total = grid.n_voxels
    rng = np.random.default_rng(seed)
    targets = np.sort(rng.choice(total, size=min(k, total), replace=False))
    off = _data_offset(path)

    encoding = h.get("encoding", "raw")
    if encoding in ("raw",):
        arr = np.memmap(path, dtype="<f4", mode="r", offset=off, shape=(total, 3))
        samp = np.array(arr[targets])          # copies only k rows
        del arr
    elif encoding in ("gzip", "gz"):
        samp = _stream_sample_gzip(path, off, targets)
    else:
        raise ValueError(f"unsupported encoding {encoding!r}")

    ijk = _linear_to_ijk(targets, grid.size)
    world = grid.world_from_indices(ijk)
    return ijk, world, samp, grid


def _stream_sample_gzip(path: str, off: int, targets: np.ndarray) -> np.ndarray:
    """One bounded-memory streaming pass over the gzip data; grab 3 floats per target voxel.

    Reads the compressed file from disk in fixed chunks (never the whole file in RAM) and
    feeds a zlib decompressor incrementally. Stops as soon as the last target is captured.
    """
    dobj = zlib.decompressobj(31)              # 31 => gzip container
    out = np.empty((len(targets), 3), dtype=np.float32)
    tgt_ptr = 0
    voxel = 0
    leftover = b""
    CH = 8 << 20
    with open(path, "rb") as fh:
        fh.seek(off)
        while tgt_ptr < len(targets):
            comp = fh.read(CH)
            dec = dobj.decompress(comp) if comp else dobj.flush()
            buf = leftover + dec
            nvox = len(buf) // 12
            if nvox:
                vals = np.frombuffer(buf[: nvox * 12], dtype="<f4").reshape(nvox, 3)
                leftover = buf[nvox * 12:]
                while tgt_ptr < len(targets) and targets[tgt_ptr] < voxel + nvox:
                    out[tgt_ptr] = vals[targets[tgt_ptr] - voxel]
                    tgt_ptr += 1
                voxel += nvox
            else:
                leftover = buf
            if not comp:
                break
    if tgt_ptr < len(targets):
        raise RuntimeError("stream ended before all targets were read")
    return out


def calibrate(ref_path: str, transform_files: List[str], k: int = 2000) -> Dict:
    """Sweep the open convention flags against the reference; report best match (mm error)."""
    _, world, ref_disp, grid = sample_reference(ref_path, k=k)
    results = []
    for order in ("ascending", "descending"):
        for use_inverse in (True, False):
            chain = TransformChain.from_files(transform_files, order=order, use_inverse=use_inverse)
            pred = chain.displacement_at(world)
            err = np.linalg.norm(pred - ref_disp, axis=1)
            results.append({"order": order, "use_inverse": use_inverse,
                            "max_err_mm": float(err.max()), "mean_err_mm": float(err.mean())})
    results.sort(key=lambda r: r["mean_err_mm"])
    return {"best": results[0], "all": results,
            "ref_disp_mag_mean_mm": float(np.linalg.norm(ref_disp, axis=1).mean())}
