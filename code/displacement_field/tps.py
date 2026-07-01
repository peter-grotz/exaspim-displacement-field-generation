"""Thin-plate-spline reconstruction from Slicer InverseThinPlateSplineKernelTransform .h5 files.

Why this exists: Slicer saves warping (landmark) transforms as the class
`InverseThinPlateSplineKernelTransform_double_3_3`, which stock ITK/SimpleITK/antspyx
CANNOT read (the class is a Slicer-specific serialization marker whose TransformPoint()
only throws). But the payload is a plain thin-plate spline: two landmark sets (source,
target) with kernel U(r)=|r| in 3D. We read the landmarks with h5py and reconstruct the
TPS in pure numpy. Validated against ITK's own evaluator to ~1e-13 mm (see
code/prototypes/tps_numpy.py).

Coordinate frame: landmarks are stored in LPS (ITK convention). We stay LPS-native.
"""
from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
from scipy.spatial.distance import cdist

DIM = 3


@dataclass
class SlicerTPS:
    """A single reconstructed thin-plate spline.

    source/target are (N, 3) landmark arrays in LPS mm. ``is_inverse`` records whether the
    stored Slicer class was the "Inverse" variant (Slicer applies the inverse mapping on
    load); how we honor that is decided at compose time (see compose.py) once calibrated.
    ``W`` is the solved coefficient matrix ((N+4, 3)): first N rows are the non-affine
    weights, last 4 are the affine part [a0; a_x; a_y; a_z].
    """

    source: np.ndarray
    target: np.ndarray
    is_inverse: bool
    W: np.ndarray

    @property
    def n_landmarks(self) -> int:
        return self.source.shape[0]

    def transform_points(self, pts: np.ndarray, chunk: int = 50_000) -> np.ndarray:
        """Apply the forward TPS to (M, 3) points. Chunked to bound memory."""
        pts = np.ascontiguousarray(pts, dtype=np.float64)
        out = np.empty_like(pts)
        wl, wa = self.W[: self.n_landmarks], self.W[self.n_landmarks :]
        for i in range(0, len(pts), chunk):
            xi = pts[i : i + chunk]
            u = cdist(xi, self.source)  # (m, N), kernel U(r)=r
            aff = np.hstack([np.ones((len(xi), 1)), xi])
            out[i : i + chunk] = u @ wl + aff @ wa
        return out

    def jacobian(self, pts: np.ndarray, chunk: int = 20_000) -> np.ndarray:
        """Analytic Jacobian dT/dx of the forward TPS at (M,3) points -> (M,3,3).

        T_d(x) = sum_i wl[i,d]*|x-c_i| + wa[0,d] + sum_e wa[1+e,d]*x_e
        dT_d/dx_e = sum_i wl[i,d]*(x_e-c_i,e)/|x-c_i| + wa[1+e,d]
        """
        pts = np.ascontiguousarray(pts, dtype=np.float64)
        wl, wa = self.W[: self.n_landmarks], self.W[self.n_landmarks :]
        aff = wa[1:].T  # (3,3): aff[d,e] = wa[1+e, d]
        out = np.empty((len(pts), DIM, DIM))
        for i in range(0, len(pts), chunk):
            xi = pts[i : i + chunk]                             # (m,3)
            diff = xi[:, None, :] - self.source[None, :, :]     # (m,N,3)
            r = np.linalg.norm(diff, axis=2)                    # (m,N)
            r = np.where(r < 1e-12, 1e-12, r)
            grad = np.einsum("id,mie->mde", wl, diff / r[..., None])  # (m,3,3)
            out[i : i + chunk] = grad + aff[None, :, :]
        return out

    def inverse_transform_points(self, y: np.ndarray, iters: int = 12, tol: float = 1e-8,
                                 chunk: int = 20_000) -> np.ndarray:
        """Invert the forward TPS: solve T(x)=y via vectorized Newton (robust at boundaries)."""
        y = np.ascontiguousarray(y, dtype=np.float64)
        x = y.copy()
        for _ in range(iters):
            resid = self.transform_points(x, chunk=chunk) - y
            if np.max(np.abs(resid)) < tol:
                break
            J = self.jacobian(x, chunk=chunk)
            x = x - np.linalg.solve(J, resid)
        return x


def _read_landmarks(path: str, group: str = "TransformGroup/0"):
    """Extract (is_inverse, source, target) in LPS from a Slicer TPS .h5.

    ITK KernelTransform layout: TransformParameters = source landmarks,
    TransformFixedParameters = target landmarks. Both flat (N*3,), point-major.
    """
    with h5py.File(path, "r") as f:
        g = f[group]
        ttype = g["TransformType"][()][0].decode()
        source = np.asarray(g["TransformParameters"], dtype=np.float64).reshape(-1, DIM)
        target = np.asarray(g["TransformFixedParameters"], dtype=np.float64).reshape(-1, DIM)
    if "ThinPlateSpline" not in ttype:
        raise ValueError(f"{path}: unexpected transform type {ttype!r} (expected a ThinPlateSpline)")
    return ("Inverse" in ttype), source, target


def _solve_tps(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve the Bookstein TPS system (kernel U(r)=r) mapping source -> target."""
    n = len(source)
    K = cdist(source, source)  # U(r)=r on the diagonal-block kernel
    P = np.hstack([np.ones((n, 1)), source])  # (n, 4)
    L = np.zeros((n + 4, n + 4))
    L[:n, :n] = K
    L[:n, n:] = P
    L[n:, :n] = P.T
    Y = np.vstack([target, np.zeros((4, DIM))])
    return np.linalg.solve(L, Y)


def load_tps(path: str, group: str = "TransformGroup/0") -> SlicerTPS:
    """Read a Slicer TPS .h5 and reconstruct a ready-to-evaluate SlicerTPS."""
    is_inverse, source, target = _read_landmarks(path, group)
    W = _solve_tps(source, target)
    return SlicerTPS(source=source, target=target, is_inverse=is_inverse, W=W)


def peek_type(path: str, group: str = "TransformGroup/0") -> dict:
    """Header-only peek: transform class + landmark count. No solve, negligible memory."""
    with h5py.File(path, "r") as f:
        g = f[group]
        ttype = g["TransformType"][()][0].decode()
        n = g["TransformParameters"].shape[0] // DIM
    return {"type": ttype, "is_inverse": "Inverse" in ttype, "n_landmarks": int(n)}
