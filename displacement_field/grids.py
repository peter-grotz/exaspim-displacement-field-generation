"""Reference-grid geometry for rasterizing a displacement field.

A grid is defined the NRRD/ITK way: an index (i, j, k) maps to a world point via
    world = origin + directions @ index
where ``directions`` is a 3x3 whose COLUMNS are the per-axis vectors (spacing already
folded in). We stay in LPS (Slicer writes displacement-field .nrrd in LPS).

Everything here is header-only / lazy: we never materialize the full voxel array.
The sample 703070 reference established the target grid = Allen CCF 10 um template.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
import nrrd


@dataclass
class ReferenceGrid:
    size: Tuple[int, int, int]          # (ni, nj, nk) voxel counts
    directions: np.ndarray              # 3x3, columns = axis direction*spacing (LPS mm)
    origin: np.ndarray                  # (3,) world origin (LPS mm)
    space: str = "left-posterior-superior"

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.size))

    @classmethod
    def from_nrrd_header(cls, path: str) -> "ReferenceGrid":
        """Build a grid from a .nrrd by reading ONLY its header (no data load).

        Handles vector fields whose component axis has a non-spatial 'space directions' row
        (encoded by pynrrd as None, the string 'none', or a row of NaN).
        """
        h = nrrd.read_header(path)
        sizes = list(map(int, h["sizes"]))
        raw = list(h["space directions"])
        dirs, keep = [], []
        for r in raw:
            if r is None or (isinstance(r, str) and r == "none"):
                keep.append(False)
                continue
            arr = np.asarray(r, dtype=float)
            if arr.ndim == 0 or np.isnan(arr).any():
                keep.append(False)
                continue
            dirs.append(arr)
            keep.append(True)
        dirs = np.array(dirs)                                   # (3,3) spatial rows
        spatial_sizes = [s for s, k in zip(sizes, keep) if k]
        origin = np.asarray(h["space origin"], dtype=float)
        # rows are per-axis vectors; store as columns so `origin + directions @ index` works
        return cls(size=tuple(spatial_sizes), directions=dirs.T.copy(),
                   origin=origin, space=str(h.get("space", "left-posterior-superior")))

    def world_from_indices(self, idx: np.ndarray) -> np.ndarray:
        """Map (M, 3) integer/float indices to (M, 3) world points (LPS mm)."""
        return self.origin[None, :] + idx @ self.directions.T

    def iter_chunks(self, chunk: int = 200_000) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (indices (m,3), world (m,3)) covering the whole grid, in NRRD memory order.

        NRRD order: first spatial axis fastest, then second, then third. Memory-safe:
        holds at most ``chunk`` points at a time regardless of total grid size.
        """
        ni, nj, nk = self.size
        # linear index l -> (i, j, k) with i fastest
        total = ni * nj * nk
        for start in range(0, total, chunk):
            stop = min(start + chunk, total)
            lin = np.arange(start, stop)
            i = lin % ni
            j = (lin // ni) % nj
            k = lin // (ni * nj)
            idx = np.stack([i, j, k], axis=1).astype(np.float64)
            yield idx, self.world_from_indices(idx)


# Known preset (confirmed from the 703070 reference header).
CCF_10UM = ReferenceGrid(
    size=(1320, 800, 1140),
    directions=np.array([[0.0, 0.0, -0.01],
                         [0.01, 0.0, 0.0],
                         [0.0, -0.01, 0.0]]),  # columns = axis vectors (LPS mm)
    origin=np.zeros(3),
    space="left-posterior-superior",
)
