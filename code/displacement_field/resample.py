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
