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
# add --jobs <cores> for the full grid (see "Generating the full dense field" below)
./run compose --transforms "/data/transforms/Transform*h.h5" --ref-grid ccf10 \
              --jobs 16 --out /results/field.nrrd

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

## App Panel — Reproducible Run (finalize pipeline)

The Code Ocean **`run`** entrypoint drives the one-shot `finalize` pipeline: from a transform
folder + input volume it produces the four sample artifacts to `/results`:

| output | what |
|---|---|
| `<sampleID>_manual_displacement_field.nrrd` | forward field `T(x)−x` on the CCF grid |
| `<sampleID>_manual_inverse_displacement_field.nrrd` | inverse field (relaxed fixed-point) — the expensive step |
| `<sampleID>_transformed_final.nii.gz` | input volume forward-warped |
| `<sampleID>_inverted_transform.nii.gz` | transformed volume inverse-warped (round-trip) |
| `<sampleID>_transforms/transform_1..N.h5` | the exact transforms used, renamed in applied order (provenance) |

**App Panel parameters** are UI-defined (App Builder) and map to `run`'s positional args in
top-to-bottom order. Create three **Text** parameters, in this exact order:

| # | Display name | maps to | Default | Notes |
|---|---|---|---|---|
| 1 | Transforms folder (S3 URI or blank) | `$1` | *(blank)* | `s3://…/ccf_alignment/transforms`; blank → attached `*_transform_files` asset |
| 2 | Input volume (S3 URI or blank) | `$2` | *(blank)* | `s3://…/registration_metadata/<sid>_10um_to_ccf_moved.nii.gz`; blank → attached `*_10um_to_ccf_moved` asset |
| 3 | Sample ID (optional) | `$3` | *(blank)* | override; else inferred from the paths |

Each parameter accepts an `s3://` URI (staged anonymously from `aind-open-data`) **or** may be
left blank to use the data asset mounted under `/data`. Then hit **Reproducible Run**.

Performance: the inverse field is ~95% of the cost and runs across all cores with BLAS threads
capped to one per worker (avoids the oversubscription that once turned a ~2 h job into 60 h).
Both fields and both warps parallelize across `nproc`. On the pinned `c6a.16xlarge` (64 vCPU),
expect **~2–2.5 h / ~$5** for the full CCF 10 µm grid.

## Generating the full dense field (CCF 10 µm = 1.2 B voxels, ~13 GB output)

Throughput is **~78,000 voxels/sec/core** (measured, 19-transform chain), so the full grid is
~4.3 core-hours. It parallelizes near-linearly; `--jobs N` computes chunks across N processes
and writes them in order (the gzip output stays a single valid stream — verified byte-identical
to the serial writer).

| `--jobs` | wall-clock (approx) | RAM needed |
|---|---|---|
| 8  | ~35 min | ~16 GB |
| 16 | ~18 min | ~32 GB |
| 32 | ~10 min | ~64 GB |

**Recommended Code Ocean machine: 16–32 vCPU, 32–64 GB RAM**, CPU-only. RAM is not the
constraint (~0.3–0.5 GB/core; the writer streams — it never holds the field), so ~1–2 GB/core
plus headroom is ample; pick cores for speed. Beyond ~32 cores the single-stream gzip step
starts to dominate — drop `--gzip-level` (e.g. `--gzip-level 1`) to keep it fed. Output is
~13 GB; make sure `/results` (or scratch) has room.

```bash
./run compose --transforms "/data/720164_transform_files/Transform*h.h5" \
              --ref-grid ccf10 --jobs 16 --out /results/720164_displacement_field.nrrd
```

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
