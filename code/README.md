# exaspim-displacement-field-generation

Headless, dependency-light replacement for the manual **3D Slicer** step in the exaSPIM / CCF
neuron-registration workflow: take a chain of Slicer landmark transforms
(`Transform1h.h5 … TransformNh.h5`), compose them, and produce the combined **displacement
field** (`.nrrd`) — or warp point coordinates (e.g. `.swc` nodes) straight through the chain.

Reproduces Slicer's output to **~1×10⁻⁹ mm** (validated below). Pure Python — **no 3D Slicer
and no ITK required at runtime** (`numpy`, `scipy`, `h5py`, `pynrrd`).

## Why this exists

Slicer saves warping (thin-plate-spline) transforms as the class
`InverseThinPlateSplineKernelTransform_double_3_3`, a **Slicer-specific ITK subclass that
stock ITK / SimpleITK / antspyx cannot read** (its `TransformPoint()` is a storage-only stub;
the class isn't registered in ITK's transform factory). That is the long-standing "incompatible
files" wall. But the payload is a plain thin-plate spline — source + target landmarks with
kernel `U(r)=|r|`. We read the landmarks with `h5py` and reconstruct the TPS in numpy, matching
ITK's own evaluator to ~1e-13 mm and Slicer's exported field to ~1e-9 mm.

## Memory safety

The reference field is on the **Allen CCF 10 µm grid = 1320×800×1140 ≈ 1.2 billion voxels**
(the dense vector field is ~14 GB). **Nothing here ever loads a full field.** Rasterization
streams to a gzip `.nrrd` in bounded chunks; validation samples the reference via memmap (raw)
or a single bounded streaming pass (gzip).

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# what are these transforms? (header-only; instant)
./run inspect "/data/transforms/Transform*h.h5"

# rasterize the dense displacement field onto the CCF 10um grid (or clone another .nrrd's grid)
./run compose --transforms "/data/transforms/Transform*h.h5" --ref-grid ccf10 --out /results/field.nrrd

# warp points (x,y,z CSV) directly through the chain — the basis for .swc transformation
./run points --transforms "/data/transforms/Transform*h.h5" --in /data/pts.csv --out /results/warped.csv

# re-validate the convention against a known Slicer displacement field for a new dataset
./run calibrate --ref /data/ref_field.nrrd --transforms "/data/transforms/Transform*h.h5"
```

Python API:
```python
from displacement_field import TransformChain, CCF_10UM
chain = TransformChain.from_files(sorted_h5_paths)     # calibrated defaults
disp  = chain.displacement_at(world_points_lps)        # (M,3) displacement, D(x)=T(x)-x
chain.write_field(CCF_10UM, "field.nrrd")              # streaming, memory-safe
warped = chain.map_points(swc_xyz_lps)                 # warp coordinates directly
```

Transform files are sorted **numerically** by the integer in the filename
(`Transform2h.h5` before `Transform10h.h5`). Coordinates are **LPS mm** throughout (ITK/Slicer
file convention); no RAS flip is applied.

## Calibration / validation

Validated against a real Slicer displacement field for sample **720164** (19 TPS transforms,
13 GB reference field), sampling 2000 voxels across the whole volume:

| convention | mean err | max err |
|---|---|---|
| **order=ascending, forward** (default) | **1.3e-9 mm** | **9.1e-9 mm** |
| order=descending, forward | 3.7e-3 mm | 1.4e-1 mm |
| order=ascending/descending, inverse | 1.2e-1 mm | 7.7e-1 mm |

The winning convention (transforms applied in ascending order, **forward** TPS on the stored
landmarks, LPS-native) is the locked default. Note the class is labelled `Inverse…`, but the
forward TPS is what matches the *exported field* — the "Inverse" marker only governs how Slicer
resamples images. `order` and `use_inverse` remain configurable and are swept by `calibrate`
if a future dataset needs re-checking.

## Layout

- `displacement_field/tps.py` — read landmarks (h5py), reconstruct TPS, forward eval + analytic Jacobian + Newton inverse.
- `displacement_field/grids.py` — `ReferenceGrid` (from `.nrrd` header, memory-safe) + `CCF_10UM` preset.
- `displacement_field/compose.py` — `TransformChain` (compose, `displacement_at`, streaming `write_field`).
- `displacement_field/validate.py` — memory-safe reference sampling + `calibrate` convention sweep.
- `displacement_field/cli.py` — `inspect | compose | points | calibrate`.
- `tests/test_displacement_field.py` — synthetic + real-data + optional ITK-parity tests.
- `run` — Code Ocean entrypoint (forwards to the CLI).

## Tests

```bash
python tests/test_displacement_field.py
```
Tests that need the sample-720164 files skip cleanly when those files aren't present.
