"""Memory-safe tests for the displacement_field package.

Run: python3 code/tests/test_displacement_field.py
Never rasterizes a full grid; only evaluates a handful of points.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from displacement_field import load_tps, TransformChain, CCF_10UM  # noqa: E402
from displacement_field.tps import _solve_tps, SlicerTPS  # noqa: E402

REAL = "/Users/peter.grotz/Downloads/capsule-5590475-703070+transforms"
REAL_FILES = sorted(glob.glob(os.path.join(REAL, "Transform.*h5.h5")),
                    key=lambda p: int(os.path.basename(p).split(".")[1].replace("h5", "")))


def test_identity_landmarks_zero_displacement():
    rng = np.random.default_rng(1)
    lm = rng.random((30, 3)) * 10
    tps = SlicerTPS(lm, lm.copy(), False, _solve_tps(lm, lm))
    pts = rng.random((50, 3)) * 10
    out = tps.transform_points(pts)
    assert np.allclose(out, pts, atol=1e-8), np.abs(out - pts).max()
    print("  [ok] identity landmarks -> zero displacement")


def test_tps_reproduces_affine():
    rng = np.random.default_rng(2)
    src = rng.random((40, 3)) * 10
    A = np.array([[1.05, 0.02, 0.0], [-0.01, 0.98, 0.03], [0.0, 0.0, 1.02]])
    b = np.array([1.5, -2.0, 0.5])
    tgt = src @ A.T + b
    tps = SlicerTPS(src, tgt, False, _solve_tps(src, tgt))
    pts = rng.random((25, 3)) * 10
    assert np.allclose(tps.transform_points(pts), pts @ A.T + b, atol=1e-6)
    print("  [ok] TPS reproduces a global affine")


def test_real_interpolation_and_chain():
    if not REAL_FILES:
        print("  [skip] real transforms not present")
        return
    tps = load_tps(REAL_FILES[1])
    # a correct TPS maps each source landmark onto its target exactly
    out = tps.transform_points(tps.source[:20])
    err = np.linalg.norm(out - tps.target[:20], axis=1).max()
    assert err < 1e-8, err
    print(f"  [ok] real TPS interpolates landmarks (max err {err:.1e} mm)")

    chain = TransformChain.from_files(REAL_FILES, order="ascending")
    world = tps.source[:200]
    disp = chain.displacement_at(world)
    mag = np.linalg.norm(disp, axis=1)
    assert np.isfinite(mag).all() and mag.max() < 5.0, mag.max()
    print(f"  [ok] 10-transform chain displacement finite, max {mag.max():.3f} mm")


def test_newton_inverse_residual():
    if not REAL_FILES:
        print("  [skip] real transforms not present")
        return
    # Correct metric for a (possibly non-injective) TPS inverse: residual |T(inv(y)) - y| ~ 0.
    # (Round-trip inv(T(x))=x can differ where the TPS has multiple valid preimages — same as VTK.)
    worst = 0.0
    for f in REAL_FILES:
        tps = load_tps(f)
        y = tps.transform_points(tps.source[:120])
        inv = tps.inverse_transform_points(y, iters=15)
        worst = max(worst, np.linalg.norm(tps.transform_points(inv) - y, axis=1).max())
    assert worst < 1e-6, worst
    print(f"  [ok] per-transform Newton inverse residual (max {worst:.1e} mm)")


def test_chain_modes_finite():
    if not REAL_FILES:
        print("  [skip] real transforms not present")
        return
    world = load_tps(REAL_FILES[0]).source[:150]
    for order in ("ascending", "descending"):
        for use_inverse in (True, False):
            chain = TransformChain.from_files(REAL_FILES, order=order, use_inverse=use_inverse)
            mag = np.linalg.norm(chain.displacement_at(world), axis=1)
            assert np.isfinite(mag).all() and mag.max() < 10.0, (order, use_inverse, mag.max())
    print("  [ok] all 4 calibration modes produce finite displacements")


def test_itk_parity_optional():
    try:
        import itk
    except Exception:
        print("  [skip] itk not installed")
        return
    if not REAL_FILES:
        print("  [skip] real transforms not present")
        return
    tps = load_tps(REAL_FILES[1])
    itps = itk.ThinPlateSplineKernelTransform[itk.D, 3].New()
    PT = itk.Point[itk.D, 3]
    for lm, pts in ((itps.GetSourceLandmarks(), tps.source), (itps.GetTargetLandmarks(), tps.target)):
        c = lm.GetPoints(); c.Initialize()
        for i, xyz in enumerate(pts):
            p = PT(); p[0], p[1], p[2] = map(float, xyz); c.InsertElement(i, p)
    itps.ComputeWMatrix()
    rng = np.random.default_rng(3)
    lo, hi = tps.source.min(0) - 2, tps.source.max(0) + 2
    X = lo + (hi - lo) * rng.random((200, 3))
    ours = tps.transform_points(X)
    theirs = np.array([itps.TransformPoint([float(x) for x in p]) for p in X])
    err = np.linalg.norm(ours - theirs, axis=1).max()
    assert err < 1e-6, err
    print(f"  [ok] numpy TPS == ITK to {err:.1e} mm")


def test_ccf_grid_geometry():
    assert CCF_10UM.size == (1320, 800, 1140)
    assert CCF_10UM.n_voxels == 1320 * 800 * 1140
    # world extent ~ 13.2 x 8.0 x 11.4 mm (order per direction columns)
    corner = CCF_10UM.world_from_indices(np.array([[1319, 799, 1139]], float))[0]
    assert np.allclose(np.abs(corner), [11.39, 13.19, 7.99], atol=0.01), corner
    print("  [ok] CCF 10um grid geometry")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        print(fn.__name__)
        fn()
    print("\nALL TESTS PASSED")
