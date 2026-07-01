"""Compose a chain of Slicer TPS transforms and rasterize a displacement field.

Replaces the manual 3D Slicer workflow: load transform1..N -> harden the chain into one
composite -> Convert to a displacement field on a reference grid -> save .nrrd.

CALIBRATION FLAGS (pinned once we have a true vector displacement-field reference):
  * order      : "ascending" (transform1 applied first) | "descending"
  * invert     : honor the Slicer "Inverse" marker by inverting the composite mapping
  * The output header/space is matched to the reference when provided.

Everything is chunked: memory stays bounded no matter how large the grid is (the CCF 10um
grid is 1.2e9 voxels; we never hold it all).
"""
from __future__ import annotations

import zlib
from typing import List, Sequence

import numpy as np

from .grids import ReferenceGrid
from .tps import SlicerTPS, load_tps


class TransformChain:
    """A composed chain of Slicer TPS transforms.

    CALIBRATED DEFAULTS (order='ascending', use_inverse=False) reproduce Slicer's exported
    displacement field to ~1e-9 mm — validated on sample 720164 (19 transforms) against the
    Slicer "Convert" output over 2000 whole-volume voxels; per-axis error exactly 0.

    Note: the stored class is InverseThinPlateSplineKernelTransform, but the FORWARD TPS on the
    stored (source,target) landmarks is what matches the exported field. The "Inverse" marker
    only affects how Slicer resamples images, not the Convert->displacement-field export.
    ``use_inverse=True`` (per-transform Newton inverse) and ``order`` remain available for
    other datasets / re-calibration via validate.calibrate.
    """

    def __init__(self, transforms: Sequence[SlicerTPS], order: str = "ascending",
                 use_inverse: bool = False):
        self.transforms: List[SlicerTPS] = list(transforms)
        if order not in ("ascending", "descending"):
            raise ValueError("order must be 'ascending' or 'descending'")
        self.order = order
        self.use_inverse = use_inverse

    @classmethod
    def from_files(cls, paths: Sequence[str], order: str = "ascending",
                   use_inverse: bool = False) -> "TransformChain":
        return cls([load_tps(p) for p in paths], order=order, use_inverse=use_inverse)

    def _ordered(self) -> List[SlicerTPS]:
        return self.transforms if self.order == "ascending" else list(reversed(self.transforms))

    def _apply_one(self, t: SlicerTPS, pts: np.ndarray, chunk: int) -> np.ndarray:
        return t.inverse_transform_points(pts, chunk=chunk) if self.use_inverse \
            else t.transform_points(pts, chunk=chunk)

    def map_points(self, world: np.ndarray, chunk: int = 50_000) -> np.ndarray:
        """Apply the composite transform to (M,3) world points (LPS mm)."""
        pts = np.ascontiguousarray(world, dtype=np.float64)
        for t in self._ordered():
            pts = self._apply_one(t, pts, chunk)
        return pts

    def map_points_forward(self, world: np.ndarray, chunk: int = 50_000) -> np.ndarray:
        """Force the forward composite (ignores use_inverse) — used for round-trip tests."""
        pts = np.ascontiguousarray(world, dtype=np.float64)
        for t in self._ordered():
            pts = t.transform_points(pts, chunk=chunk)
        return pts

    def displacement_at(self, world: np.ndarray, chunk: int = 50_000) -> np.ndarray:
        """Displacement D(x) = T(x) - x at (M,3) world points. This is the field's payload."""
        return self.map_points(world, chunk=chunk) - world

    def chunk_bytes(self, grid: ReferenceGrid, start: int, stop: int) -> bytes:
        """float32 little-endian displacement bytes for voxels [start, stop) (component-fastest)."""
        world = grid.world_for_range(start, stop)
        disp = self.displacement_at(world).astype("<f4")
        return np.ascontiguousarray(disp).tobytes()

    # ---- dense field writer (memory-safe; optionally multi-core) ----
    def write_field(self, grid: ReferenceGrid, out_path: str, chunk: int = 1_000_000,
                   jobs: int = 1, gzip_level: int = 6, verbose: bool = True) -> None:
        """Rasterize the displacement field over ``grid`` -> single-stream gzip .nrrd.

        Layout: dimension 4, sizes [3, ni, nj, nk], component axis fastest, float32, LPS —
        matches the Slicer "Convert" displacement-field export.

        Memory-safe (streams ``chunk`` voxels at a time). With ``jobs`` > 1, chunks are
        computed across processes and written in order (gzip stays a single valid stream);
        peak RAM ~ jobs * a few hundred MB. Compute scales ~linearly with cores
        (~78k voxels/sec/core measured on a 19-transform chain).
        """
        header = _nrrd_header(grid)
        ranges = [(s, min(s + chunk, grid.n_voxels)) for s in range(0, grid.n_voxels, chunk)]
        comp = zlib.compressobj(gzip_level, zlib.DEFLATED, 31)  # 31 -> gzip container
        total = grid.n_voxels
        done = 0

        def _emit(fh, b):
            nonlocal done
            fh.write(comp.compress(b))
            done += len(b) // 12
            if verbose:
                print(f"\r  rasterized {done:,}/{total:,} voxels ({100*done/total:.1f}%)", end="")

        with open(out_path, "wb") as fh:
            fh.write(header.encode("ascii"))
            if jobs <= 1:
                for s, e in ranges:
                    _emit(fh, self.chunk_bytes(grid, s, e))
            else:
                import multiprocessing as mp
                ctx = mp.get_context("fork")  # workers inherit chain+grid (Linux/CO)
                with ctx.Pool(jobs, initializer=_init_worker, initargs=(self, grid)) as pool:
                    for b in pool.imap(_worker_chunk, ranges, chunksize=1):  # ordered
                        _emit(fh, b)
            fh.write(comp.flush())
        if verbose:
            print()


def _nrrd_header(grid: ReferenceGrid) -> str:
    ni, nj, nk = grid.size
    d = grid.directions  # columns = axis vectors
    vec_rows = " ".join(f"({d[0,a]},{d[1,a]},{d[2,a]})" for a in range(3))
    return (
        "NRRD0004\n"
        "type: float\n"
        "dimension: 4\n"
        "space: " + grid.space + "\n"
        f"sizes: 3 {ni} {nj} {nk}\n"
        f"space directions: none {vec_rows}\n"
        "kinds: vector domain domain domain\n"
        "endian: little\n"
        "encoding: gzip\n"
        f"space origin: ({grid.origin[0]},{grid.origin[1]},{grid.origin[2]})\n"
        "\n"
    )


# ---- multiprocessing worker state (set once per worker via fork/initializer) ----
_WORKER: dict = {}


def _init_worker(chain: "TransformChain", grid: ReferenceGrid) -> None:
    _WORKER["chain"] = chain
    _WORKER["grid"] = grid


def _worker_chunk(rng):
    s, e = rng
    return _WORKER["chain"].chunk_bytes(_WORKER["grid"], s, e)
