# displacement_field — Slicer TPS chain → displacement field, in pure Python

Replaces the manual 3D Slicer workflow (load `transform1.h5 … transformN.h5` → harden the
chain into one composite → *Convert* to a displacement field on a reference volume → save
`.nrrd`) with a headless, reproducible pipeline.

## Why this is non-trivial

Slicer saves warping transforms as `InverseThinPlateSplineKernelTransform_double_3_3` — a
**Slicer-specific ITK subclass that stock ITK / SimpleITK / antspyx cannot read** (its
`TransformPoint()` is a storage-only stub; the class isn't in ITK's transform factory). That
is the "incompatible files" wall. But the payload is a plain **thin-plate spline** (source +
target landmarks, kernel `U(r)=|r|`). We read the landmarks with `h5py` and reconstruct the
TPS in numpy — validated against ITK's own evaluator to **7×10⁻¹⁴ mm**. No Slicer or ITK is
needed at runtime; the Code Ocean capsule needs only `h5py`, `numpy`, `scipy`, `pynrrd`.

## Memory safety

The reference field is on the **Allen CCF 10 µm grid = 1320×800×1140 = 1.2 billion voxels**
(the dense vector field is ~14 GB). **Nothing here ever loads a full field.** Rasterization
streams to a gzip `.nrrd` in bounded-memory chunks; validation samples the reference via
memmap (raw) or a single bounded streaming pass (gzip).

## Usage

```bash
pip install -r ../../requirements.txt      # h5py, numpy, scipy, pynrrd (itk optional, for parity tests)

# what are these files?
python -m displacement_field.cli inspect "/path/Transform.*h5.h5"

# once a TRUE vector displacement-field .nrrd reference exists — find the exact convention:
python -m displacement_field.cli calibrate --ref ref_vector_field.nrrd --transforms "/path/Transform.*h5.h5"

# rasterize a dense field onto the CCF 10um grid (or clone another .nrrd's geometry):
python -m displacement_field.cli compose --transforms "/path/Transform.*h5.h5" --ref-grid ccf10 --out field.nrrd

# warp points directly through the chain (cheaper than a dense field; basis for SWC warping):
python -m displacement_field.cli points --transforms "/path/Transform.*h5.h5" --in pts.csv --out warped.csv
```

Python API:
```python
from displacement_field import TransformChain, CCF_10UM, calibrate
chain = TransformChain.from_files(sorted_h5_paths, order="ascending", use_inverse=True)
disp  = chain.displacement_at(world_points_lps)     # (M,3) displacement D(x)=T(x)-x
chain.write_field(CCF_10UM, "field.nrrd")           # streaming, memory-safe
```

## Status — CALIBRATED ✅

- ✅ Read Slicer InverseTPS `.h5`, reconstruct TPS, forward eval — **ITK-parity 7e-14 mm**.
- ✅ Chain composition, analytic-Jacobian Newton inverse, streaming vector-`.nrrd` writer,
  memory-safe reference sampling + auto-calibration. Unit-tested (`../tests/test_displacement_field.py`).
- ✅ **Calibrated to Slicer parity on sample 720164** (19 transforms) against the real
  13 GB Slicer displacement field: **mean 1.3e-9 mm / max 9e-9 mm** over 2000 whole-volume
  voxels, per-axis error 0. Winning convention = **order='ascending', forward TPS** (LPS-native,
  no coordinate flips). These are the locked defaults in `TransformChain`.
  - Subtlety: the class is `Inverse`TPS, but the FORWARD TPS on the stored landmarks matches
    the exported field (the marker only governs Slicer's image resampling). `use_inverse=True`
    and `order` stay available + swept by `calibrate` for re-validation on other datasets.

## Layout
- `tps.py` — read landmarks (`h5py`), reconstruct TPS, forward eval + analytic Jacobian + Newton inverse.
- `grids.py` — `ReferenceGrid` (from `.nrrd` header, memory-safe), `CCF_10UM` preset.
- `compose.py` — `TransformChain` (compose, map/inverse, `displacement_at`, streaming `write_field`).
- `validate.py` — memory-safe reference sampling + `calibrate` convention sweep.
- `cli.py` — `inspect | calibrate | compose | points`.

The `../prototypes/` scripts document the investigation (feasibility, `.h5` dissection, ITK
parity) and are not part of the package.
