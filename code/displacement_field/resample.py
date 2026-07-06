"""Warp a scalar volume by a displacement field (.nrrd), memory-safely.

warp(y) = V(y + D(y)): for each output voxel y on the field's grid, look up the displacement
D(y) (mm, LPS), convert it to a voxel offset via the grid directions, and trilinear-sample the
input volume there. The displacement field is streamed (never fully in RAM); the volume must be
an array on the same grid (held in RAM — ~5 GB for the CCF 10 um grid, so run on the capsule).

Used by the round-trip validation: forward-warp a volume, then inverse-warp the result, and the
recovery should match the original (up to interpolation blur) — a ground-truth test of the
inverse field.
"""
from __future__ import annotations

import zlib
import numpy as np
from scipy.ndimage import map_coordinates

from .grids import ReferenceGrid


def _data_offset(path: str) -> int:
    with open(path, "rb") as f:
        head = f.read(65536)
    return head.index(b"\n\n") + 2


def _stream_field(field_path: str, voxels_per_chunk: int):
    """Yield (start_voxel, D) where D is (m,3) float64 displacement (mm), streaming the gzip field."""
    off = _data_offset(field_path)
    dobj = zlib.decompressobj(31)
    buf = b""
    start = 0
    step = voxels_per_chunk * 12  # 3 float32 per voxel
    with open(field_path, "rb") as f:
        f.seek(off)
        while True:
            c = f.read(16 << 20)
            buf += dobj.decompress(c) if c else dobj.flush()
            while len(buf) >= step:
                D = np.frombuffer(buf[:step], dtype="<f4").reshape(-1, 3).astype(np.float64)
                yield start, D
                start += len(D)
                buf = buf[step:]
            if not c:
                n = (len(buf) // 12) * 12
                if n:
                    D = np.frombuffer(buf[:n], dtype="<f4").reshape(-1, 3).astype(np.float64)
                    yield start, D
                break


def _scalar_nrrd_header(grid: ReferenceGrid) -> str:
    ni, nj, nk = grid.size
    d = grid.directions
    rows = " ".join(f"({d[0,a]},{d[1,a]},{d[2,a]})" for a in range(3))
    return (
        "NRRD0004\n" "type: float\n" "dimension: 3\n" f"space: {grid.space}\n"
        f"sizes: {ni} {nj} {nk}\n" f"space directions: {rows}\n"
        "kinds: domain domain domain\n" "endian: little\n" "encoding: gzip\n"
        f"space origin: ({grid.origin[0]},{grid.origin[1]},{grid.origin[2]})\n\n"
    )


def warp_volume(volume: np.ndarray, grid: ReferenceGrid, field_path: str, out_path: str,
                chunk: int = 2_000_000, order: int = 1, cval: float = 0.0,
                gzip_level: int = 1, verbose: bool = True) -> None:
    """Resample ``volume`` (shape == grid.size) through the displacement field at ``field_path``,
    writing a scalar .nrrd on the same grid. Streams the field; holds only ``volume`` + a chunk."""
    if tuple(volume.shape) != tuple(grid.size):
        raise ValueError(f"volume shape {volume.shape} != grid {grid.size}")
    volume = np.ascontiguousarray(volume, dtype=np.float32)
    ni, nj, nk = grid.size
    inv_dirs_T = np.linalg.inv(grid.directions).T   # world-displacement -> index-offset (row-vec)
    total = grid.n_voxels
    comp = zlib.compressobj(gzip_level, zlib.DEFLATED, 31)
    done = 0
    with open(out_path, "wb") as fh:
        fh.write(_scalar_nrrd_header(grid).encode("ascii"))
        for start, D in _stream_field(field_path, chunk):
            m = len(D)
            lin = np.arange(start, start + m)
            i = lin % ni
            j = (lin // ni) % nj
            k = lin // (ni * nj)
            samp = np.stack([i, j, k], axis=1).astype(np.float64) + D @ inv_dirs_T  # (m,3) idx
            vals = map_coordinates(volume, samp.T, order=order, mode="constant",
                                   cval=cval).astype("<f4")
            fh.write(comp.compress(np.ascontiguousarray(vals).tobytes()))
            done += m
            if verbose:
                print(f"\r  warped {done:,}/{total:,} voxels ({100*done/total:.1f}%)", end="")
        fh.write(comp.flush())
    if verbose:
        print()


def load_volume(path: str, grid: ReferenceGrid) -> np.ndarray:
    """Load a .nrrd or .nii(.gz) volume as an array on ``grid`` (shape must match grid.size)."""
    if path.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        arr = np.asarray(nib.load(path).dataobj)
    else:
        import nrrd
        arr, _ = nrrd.read(path)
    if tuple(arr.shape) != tuple(grid.size):
        raise ValueError(f"volume {path} shape {arr.shape} != field grid {grid.size}; "
                         "the volume must be on the same grid as the displacement field")
    return np.ascontiguousarray(arr, dtype=np.float32)


# ---- NIfTI output path: warp into an in-memory array (optionally multi-core), write .nii.gz ----
#
# The dense warps are embarrassingly parallel over output voxels: each voxel independently reads
# its displacement D(y), converts it to an index offset, and trilinear-samples the (shared,
# read-only) input volume. The field is gzip and not seekable, so the MAIN process decompresses
# it serially (cheap, ~tens of seconds for the whole field) and hands each chunk to a fork-pool
# worker for the map_coordinates work. Workers inherit the volume via copy-on-write fork — no
# per-worker copy of the ~5 GB volume.

def _warp_chunk(volume, size, inv_dirs_T, start, D, order, cval):
    ni, nj, _ = size
    m = len(D)
    lin = np.arange(start, start + m)
    i = lin % ni
    j = (lin // ni) % nj
    k = lin // (ni * nj)
    samp = np.stack([i, j, k], axis=1).astype(np.float64) + D @ inv_dirs_T  # (m,3) index coords
    return map_coordinates(volume, samp.T, order=order, mode="constant", cval=cval).astype(np.float32)


_WARP: dict = {}


def _init_warp(volume, size, inv_dirs_T, order, cval):
    _WARP.update(volume=volume, size=size, inv=inv_dirs_T, order=order, cval=cval)


def _warp_worker(item):
    start, D = item
    vals = _warp_chunk(_WARP["volume"], _WARP["size"], _WARP["inv"], start, D,
                       _WARP["order"], _WARP["cval"])
    return start, vals


def warp_volume_array(volume: np.ndarray, grid: ReferenceGrid, field_path: str,
                      jobs: int = 1, order: int = 1, cval: float = 0.0,
                      chunk: int = 2_000_000, verbose: bool = True) -> np.ndarray:
    """Resample ``volume`` through the displacement field, returning the (ni,nj,nk) array.

    Same math as ``warp_volume`` but returns an array (for NIfTI writing) instead of streaming to
    .nrrd, and parallelizes the per-chunk resampling across ``jobs`` processes.
    """
    if tuple(volume.shape) != tuple(grid.size):
        raise ValueError(f"volume shape {volume.shape} != grid {grid.size}")
    volume = np.ascontiguousarray(volume, dtype=np.float32)
    inv_dirs_T = np.linalg.inv(grid.directions).T
    total = grid.n_voxels
    out = np.empty(total, dtype=np.float32)   # linear (i-fastest) buffer
    done = 0

    def _place(start, vals):
        nonlocal done
        out[start:start + len(vals)] = vals
        done += len(vals)
        if verbose:
            print(f"\r  warped {done:,}/{total:,} voxels ({100*done/total:.1f}%)", end="")

    if jobs <= 1:
        for start, D in _stream_field(field_path, chunk):
            _place(start, _warp_chunk(volume, grid.size, inv_dirs_T, start, D, order, cval))
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork")  # workers inherit `volume` copy-on-write (Linux/CO)
        with ctx.Pool(jobs, initializer=_init_warp,
                      initargs=(volume, grid.size, inv_dirs_T, order, cval)) as pool:
            for start, vals in pool.imap_unordered(_warp_worker,
                                                   _stream_field(field_path, chunk), chunksize=1):
                _place(start, vals)
    if verbose:
        print()
    return out.reshape(grid.size, order="F")   # i-fastest linear -> (ni,nj,nk)


def load_volume_affine(path: str, grid: ReferenceGrid):
    """Load a volume as (array, affine, header). NIfTI keeps its own affine; .nrrd derives one
    from the grid (LPS->RAS). Shape must match ``grid.size``."""
    if path.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        img = nib.load(path)
        arr = np.asarray(img.dataobj, dtype=np.float32)
        affine, header = img.affine, img.header
    else:
        arr = load_volume(path, grid)
        affine, header = grid_to_nifti_affine(grid), None
    if tuple(arr.shape) != tuple(grid.size):
        raise ValueError(f"volume {path} shape {arr.shape} != grid {grid.size}")
    return np.ascontiguousarray(arr, dtype=np.float32), affine, header


def grid_to_nifti_affine(grid: ReferenceGrid) -> np.ndarray:
    """Build a NIfTI (RAS) affine from an LPS reference grid: flip x,y signs of the LPS affine."""
    a = np.eye(4)
    a[:3, :3] = grid.directions
    a[:3, 3] = grid.origin
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    return lps_to_ras @ a


def write_nifti(arr: np.ndarray, affine, out_path: str, header=None) -> None:
    """Write a float32 volume as a gzip-compressed .nii.gz (reusing the input header if given)."""
    import nibabel as nib
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    img = nib.Nifti1Image(arr, affine, header)
    img.set_data_dtype(np.float32)
    img.header.set_slope_inter(1.0, 0.0)   # no intensity scaling
    nib.save(img, out_path)
