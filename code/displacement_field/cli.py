"""CLI for the displacement-field generator.

Examples:
  python -m displacement_field.cli inspect  /path/Transform.*h5.h5
  python -m displacement_field.cli calibrate --ref ref_field.nrrd --transforms t1.h5 t2.h5 ...
  python -m displacement_field.cli compose   --transforms t1.h5 ... --ref-grid ccf10 --out field.nrrd
  python -m displacement_field.cli points    --transforms t1.h5 ... --in pts.csv --out warped.csv
  python -m displacement_field.cli roundtrip --transforms t1.h5 ... --field fwd.nrrd --volume V.nrrd --out-dir DIR
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys

import numpy as np

from .compose import TransformChain
from .grids import CCF_10UM, ReferenceGrid
from .resample import warp_volume, load_volume
from .tps import peek_type
from .validate import calibrate


def _natkey(s):
    """Natural sort key: order by the first integer in the basename (Transform2h < Transform10h)."""
    m = re.search(r"(\d+)", s.rsplit("/", 1)[-1])
    return (int(m.group(1)) if m else 0, s)


def _emit_transforms(tfiles, out_dir, sid):
    """Copy the transforms actually used into <out_dir>/<sid>_transforms/transform_1..N.h5.

    ``tfiles`` is already in APPLIED order (natural sort by filename integer), so transform_1.h5
    is the first-applied transform. Provides self-contained provenance alongside the fields.
    """
    tdir = os.path.join(out_dir, f"{sid}_transforms")
    os.makedirs(tdir, exist_ok=True)
    for i, src in enumerate(tfiles, start=1):
        shutil.copy2(src, os.path.join(tdir, f"transform_{i}.h5"))
    return tdir


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
    p_cmp.add_argument("--direction", default="forward", choices=["forward", "reverse", "both"],
                       help="forward field (default), reverse/inverse field, or both")
    p_cmp.add_argument("--out-reverse", help="reverse-field output path (for --direction both)")
    p_cmp.add_argument("--inv-iters", type=int, default=25, help="reverse: fixed-point iterations")
    p_cmp.add_argument("--inv-relax", type=float, default=0.7, help="reverse: relaxation factor")
    p_cmp.add_argument("--order", default="ascending", choices=["ascending", "descending"],
                       help="advanced: chain order (calibrated = ascending)")
    p_cmp.add_argument("--jobs", type=int, default=1,
                       help="parallel worker processes (use ~#cores for the full CCF grid)")
    p_cmp.add_argument("--chunk", type=int, default=1_000_000, help="voxels per work chunk")
    p_cmp.add_argument("--gzip-level", type=int, default=6, help="1=fast/larger .. 9=slow/smaller")

    p_pts = sub.add_parser("points", help="warp (x,y,z) points through the chain (CSV in/out)")
    p_pts.add_argument("--transforms", nargs="+", required=True)
    p_pts.add_argument("--in", dest="inp", required=True)
    p_pts.add_argument("--out", required=True)
    p_pts.add_argument("--order", default="ascending", choices=["ascending", "descending"])
    p_pts.add_argument("--inverse", action="store_true", help="warp via the reverse (inverse) map")

    p_rt = sub.add_parser("roundtrip",
                          help="save inverse field + forward-warped volume + reverse-warped volume")
    p_rt.add_argument("--transforms", nargs="+", required=True, help="Transform*h.h5 (for inverse field)")
    p_rt.add_argument("--field", required=True, help="the (forward) displacement field .nrrd already generated")
    p_rt.add_argument("--volume", required=True, help="sample volume (.nrrd/.nii) on the field's grid")
    p_rt.add_argument("--out-dir", required=True)
    p_rt.add_argument("--order", default="ascending", choices=["ascending", "descending"])
    p_rt.add_argument("--jobs", type=int, default=1, help="cores for inverse-field generation")
    p_rt.add_argument("--inv-iters", type=int, default=25)
    p_rt.add_argument("--inv-relax", type=float, default=0.7)
    p_rt.add_argument("--chunk", type=int, default=1_000_000)
    p_rt.add_argument("--gzip-level", type=int, default=1)

    p_fin = sub.add_parser("finalize",
                           help="full pipeline -> 4 named artifacts from a transform folder + input volume "
                                "(both may be local paths or s3:// URIs)")
    p_fin.add_argument("--transforms-dir", required=True, help="folder (or s3:// prefix) of Transform*.h5")
    p_fin.add_argument("--volume", required=True, help="input moved volume (.nii.gz, local or s3://)")
    p_fin.add_argument("--out-dir", required=True)
    p_fin.add_argument("--sample-id", default=None, help="override; else inferred from paths")
    p_fin.add_argument("--jobs", type=int, default=1, help="worker processes (use ~#cores)")
    p_fin.add_argument("--order", default="ascending", choices=["ascending", "descending"])
    p_fin.add_argument("--inv-iters", type=int, default=25)
    p_fin.add_argument("--inv-relax", type=float, default=0.7)
    p_fin.add_argument("--chunk", type=int, default=1_000_000)
    p_fin.add_argument("--gzip-level", type=int, default=1)
    p_fin.add_argument("--stage-dir", default=None, help="where to download s3 inputs (default: <out-dir>/_staging)")

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
        chain = TransformChain.from_files(_expand(args.transforms), order=args.order)  # forward
        jobs = [("forward", args.out)] if args.direction in ("forward", "both") else []
        if args.direction in ("reverse", "both"):
            root, ext = os.path.splitext(args.out)
            rev_out = args.out_reverse or (f"{root}_reverse{ext}" if args.direction == "both" else args.out)
            jobs.append(("reverse", rev_out))
        for direction, outp in jobs:
            print(f"[{direction}] grid {grid.size} ({grid.n_voxels:,} voxels), jobs={args.jobs} -> {outp}")
            chain.write_field(grid, outp, direction=direction, iters=args.inv_iters, relax=args.inv_relax,
                              jobs=args.jobs, chunk=args.chunk, gzip_level=args.gzip_level)

    elif args.cmd == "points":
        pts = np.loadtxt(args.inp, delimiter=",", ndmin=2)[:, :3]
        chain = TransformChain.from_files(_expand(args.transforms), order=args.order)
        mapped = pts + chain.reverse_displacement_at(pts) if args.inverse else chain.map_points(pts)
        np.savetxt(args.out, mapped, delimiter=",")
        print(f"wrote {len(mapped)} {'reverse-' if args.inverse else ''}warped points -> {args.out}")

    elif args.cmd == "roundtrip":
        grid = ReferenceGrid.from_nrrd_header(args.field)
        os.makedirs(args.out_dir, exist_ok=True)
        inv_field = os.path.join(args.out_dir, "inverse_displacement_field.nrrd")
        fwd_vol = os.path.join(args.out_dir, "transformed_volume.nrrd")
        rev_vol = os.path.join(args.out_dir, "reverse_transformed_volume.nrrd")
        chain = TransformChain.from_files(_expand(args.transforms), order=args.order)
        print(f"grid {grid.size} ({grid.n_voxels:,} voxels)")

        print(f"[1/3] forward-warp volume by {args.field} -> {fwd_vol}")
        warp_volume(load_volume(args.volume, grid), grid, args.field, fwd_vol,
                    chunk=args.chunk, gzip_level=args.gzip_level)

        print(f"[2/3] generate inverse field (jobs={args.jobs}) -> {inv_field}")
        chain.write_field(grid, inv_field, direction="reverse", iters=args.inv_iters,
                          relax=args.inv_relax, jobs=args.jobs, chunk=args.chunk,
                          gzip_level=args.gzip_level)

        print(f"[3/3] inverse-warp transformed volume -> {rev_vol}")
        warp_volume(load_volume(fwd_vol, grid), grid, inv_field, rev_vol,
                    chunk=args.chunk, gzip_level=args.gzip_level)

        print(f"done. saved:\n  {inv_field}\n  {fwd_vol}\n  {rev_vol}")

    elif args.cmd == "finalize":
        _finalize(args)


def _finalize(args):
    """Generate the four required artifacts for one sample.

    [1/4] forward displacement field   (T(x)-x on the CCF grid)          -> _manual_displacement_field.nrrd
    [2/4] inverse displacement field   (relaxed fixed-point inversion)   -> _manual_inverse_displacement_field.nrrd
    [3/4] transformed volume           (forward-warp the input)          -> _transformed_final.nii.gz
    [4/4] inverted volume              (inverse-warp the transformed)    -> _inverted_transform.nii.gz

    Fields (steps 1-2) run first, while no 5 GB volume array is held, so peak RAM stays low during
    the parallel field passes. The expensive step is [2/4]; both fields and both warps parallelize
    across --jobs.
    """
    import time
    from .naming import infer_sample_id, output_names
    from .resample import warp_volume_array, load_volume_affine, write_nifti
    from . import s3_stage

    stage_dir = args.stage_dir or os.path.join(args.out_dir, "_staging")

    transforms_dir = args.transforms_dir
    volume_path = args.volume
    if s3_stage.is_s3(transforms_dir):
        print(f"[stage] transforms <- {transforms_dir}")
        transforms_dir = s3_stage.stage_transforms(transforms_dir, os.path.join(stage_dir, "transforms"))
    if s3_stage.is_s3(volume_path):
        print(f"[stage] volume     <- {volume_path}")
        volume_path = s3_stage.stage_volume(volume_path, stage_dir)

    tfiles = [f for f in _expand([os.path.join(transforms_dir, "*")]) if f.lower().endswith(".h5")]
    if not tfiles:
        sys.exit(f"ERROR: no *.h5 transforms found in {transforms_dir}")

    sid = args.sample_id or infer_sample_id(volume_path, transforms_dir, tfiles[0])
    if not sid:
        sys.exit("ERROR: could not infer sample id from paths; pass --sample-id")
    names = output_names(sid)

    os.makedirs(args.out_dir, exist_ok=True)
    grid = CCF_10UM
    fwd_field = os.path.join(args.out_dir, names["forward_field"])
    inv_field = os.path.join(args.out_dir, names["inverse_field"])
    tr_vol = os.path.join(args.out_dir, names["transformed"])
    iv_vol = os.path.join(args.out_dir, names["inverted"])

    # Provenance: copy the exact transforms used (applied order) -> <sid>_transforms/transform_N.h5
    tdir = _emit_transforms(tfiles, args.out_dir, sid)
    print(f"[transforms] copied {len(tfiles)} -> {tdir}/transform_1..{len(tfiles)}.h5")

    chain = TransformChain.from_files(tfiles, order=args.order)
    print(f"sample {sid}: {len(tfiles)} transform(s), grid {grid.size} "
          f"({grid.n_voxels:,} voxels), jobs={args.jobs}")
    t0 = time.time()

    print(f"[1/4] forward field  -> {fwd_field}")
    chain.write_field(grid, fwd_field, direction="forward", jobs=args.jobs,
                      chunk=args.chunk, gzip_level=args.gzip_level)
    t1 = time.time(); print(f"      ({t1 - t0:.0f}s)")

    print(f"[2/4] inverse field  -> {inv_field}   (expensive step)")
    chain.write_field(grid, inv_field, direction="reverse", iters=args.inv_iters,
                      relax=args.inv_relax, jobs=args.jobs, chunk=args.chunk, gzip_level=args.gzip_level)
    t2 = time.time(); print(f"      ({t2 - t1:.0f}s)")

    print(f"[3/4] transformed    -> {tr_vol}")
    vol, affine, header = load_volume_affine(volume_path, grid)
    transformed = warp_volume_array(vol, grid, fwd_field, jobs=args.jobs)
    write_nifti(transformed, affine, tr_vol, header)
    del vol
    t3 = time.time(); print(f"      ({t3 - t2:.0f}s)")

    print(f"[4/4] inverted       -> {iv_vol}")
    inverted = warp_volume_array(transformed, grid, inv_field, jobs=args.jobs)
    write_nifti(inverted, affine, iv_vol, header)
    t4 = time.time(); print(f"      ({t4 - t3:.0f}s)")

    print(f"\ndone in {(t4 - t0)/60:.1f} min. saved to {args.out_dir}:")
    for p in (fwd_field, inv_field, tr_vol, iv_vol):
        sz = os.path.getsize(p) if os.path.exists(p) else 0
        print(f"  {os.path.basename(p)}  ({sz/1e9:.2f} GB)")
    print(f"  {os.path.basename(tdir)}/  ({len(tfiles)} transforms: transform_1..{len(tfiles)}.h5)")


if __name__ == "__main__":
    sys.exit(main())
