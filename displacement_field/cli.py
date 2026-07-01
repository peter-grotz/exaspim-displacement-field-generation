"""CLI for the displacement-field generator.

Examples:
  python -m displacement_field.cli inspect  /path/Transform.*h5.h5
  python -m displacement_field.cli calibrate --ref ref_field.nrrd --transforms t1.h5 t2.h5 ...
  python -m displacement_field.cli compose   --transforms t1.h5 ... --ref-grid ccf10 --out field.nrrd
  python -m displacement_field.cli points    --transforms t1.h5 ... --in pts.csv --out warped.csv
"""
from __future__ import annotations

import argparse
import glob
import re
import sys

import numpy as np

from .compose import TransformChain
from .grids import CCF_10UM, ReferenceGrid
from .tps import peek_type
from .validate import calibrate


def _natkey(s):
    """Natural sort key: order by the first integer in the basename (Transform2h < Transform10h)."""
    m = re.search(r"(\d+)", s.rsplit("/", 1)[-1])
    return (int(m.group(1)) if m else 0, s)


def _expand(paths):
    out = []
    for p in paths:
        out.extend(glob.glob(p) or [p])
    return sorted(out, key=_natkey)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="displacement_field")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ins = sub.add_parser("inspect", help="report transform type + landmark count (header only)")
    p_ins.add_argument("files", nargs="+")

    p_cal = sub.add_parser("calibrate", help="find the convention matching a reference vector field")
    p_cal.add_argument("--ref", required=True)
    p_cal.add_argument("--transforms", nargs="+", required=True)
    p_cal.add_argument("-k", type=int, default=2000)

    p_cmp = sub.add_parser("compose", help="rasterize a dense displacement field to .nrrd")
    p_cmp.add_argument("--transforms", nargs="+", required=True)
    p_cmp.add_argument("--ref-grid", default="ccf10", help="'ccf10' or a .nrrd to clone geometry from")
    p_cmp.add_argument("--out", required=True)
    p_cmp.add_argument("--order", default="ascending", choices=["ascending", "descending"])
    p_cmp.add_argument("--inverse", action="store_true",
                       help="apply per-transform inverse (default: forward — the calibrated mode)")

    p_pts = sub.add_parser("points", help="warp (x,y,z) points through the chain (CSV in/out)")
    p_pts.add_argument("--transforms", nargs="+", required=True)
    p_pts.add_argument("--in", dest="inp", required=True)
    p_pts.add_argument("--out", required=True)
    p_pts.add_argument("--order", default="ascending", choices=["ascending", "descending"])
    p_pts.add_argument("--inverse", action="store_true")

    args = ap.parse_args(argv)

    if args.cmd == "inspect":
        for f in _expand(args.files):
            info = peek_type(f)
            print(f"{f}: {info['type']}  N={info['n_landmarks']}  inverse={info['is_inverse']}")

    elif args.cmd == "calibrate":
        res = calibrate(args.ref, _expand(args.transforms), k=args.k)
        print(f"reference mean |disp| = {res['ref_disp_mag_mean_mm']:.4f} mm")
        for r in res["all"]:
            print(f"  order={r['order']:<10} use_inverse={r['use_inverse']!s:<5} "
                  f"mean_err={r['mean_err_mm']:.4e} mm  max_err={r['max_err_mm']:.4e} mm")
        b = res["best"]
        print(f"BEST: order={b['order']} use_inverse={b['use_inverse']} mean_err={b['mean_err_mm']:.4e} mm")

    elif args.cmd == "compose":
        grid = CCF_10UM if args.ref_grid == "ccf10" else ReferenceGrid.from_nrrd_header(args.ref_grid)
        chain = TransformChain.from_files(_expand(args.transforms), order=args.order,
                                          use_inverse=args.inverse)
        print(f"grid {grid.size} ({grid.n_voxels:,} voxels) -> {args.out}")
        chain.write_field(grid, args.out)

    elif args.cmd == "points":
        pts = np.loadtxt(args.inp, delimiter=",", ndmin=2)[:, :3]
        chain = TransformChain.from_files(_expand(args.transforms), order=args.order,
                                          use_inverse=args.inverse)
        mapped = chain.map_points(pts)
        np.savetxt(args.out, mapped, delimiter=",")
        print(f"wrote {len(mapped)} warped points -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
