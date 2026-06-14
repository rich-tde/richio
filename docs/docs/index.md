# RICHIO

**RICHIO** (RICH I/O) is a lightweight Python library for reading and
post-processing the output of
[RICH](https://gitlab.com/eladtan/RICH/-/tree/master?ref_type=heads), an
open-source moving-mesh radiative-hydrodynamic simulation code for astrophysics.
It was built for the study of tidal disruption events (TDEs), but the I/O and
analysis tools are general.

A plain `import richio` stays a compact reading/analysis helper — the heavy
optional capabilities (3-D volume rendering, shock finding) are kept behind
extra installs so the core library is fast to import and easy to deploy.

```python
import richio as rio

snap = rio.load("snap_0042.h5")   # load an HDF5 snapshot
snap.info()                        # print metadata + field table
rho = snap["Density"].in_cgs()     # field access, unit-aware (unyt)
snap.plots.peek("density")         # quick mid-plane density slice
```

## What you can do with it

| Capability | Entry point | Install |
|------------|-------------|---------|
| Load snapshots (HDF5 / NPY, multi-rank) | [`rio.load`](api/data.md) | core |
| Unit-aware field access with aliases | `snap["rho"]`, `snap.density` | core |
| Convert between code / cgs / mks units | [`unyt`](https://unyt.readthedocs.io) integration | core |
| Clip to a sub-region (lazy, no copy) | [`snap.clip`](guide/selecting-regions.md) | core |
| Slices & column projections | [`snap.slice` / `snap.project`](guide/slices-and-projections.md) | core |
| Publication-style plots | [`snap.plots`](guide/plotting.md) | core |
| Depth-cued 3-D volume renders & movies | [`richio.render`](guide/volume-rendering.md) | `pip install "richio[render]"` |
| Shock-zone & shock-surface detection | [`richio.shockfinder`](guide/shock-finding.md) | needs `scipy` + `numba` |

## Where to go next

- **New here?** Start with [Installation](getting-started/installation.md) then the
  [Quickstart](getting-started/quickstart.md).
- **Working with data?** The User guide covers
  [loading snapshots](guide/loading-snapshots.md),
  [fields & aliases](guide/fields-and-aliases.md),
  [units](guide/units.md),
  [selecting regions](guide/selecting-regions.md),
  [slices & projections](guide/slices-and-projections.md), and
  [plotting](guide/plotting.md).
- **Optional extras:** [volume rendering](guide/volume-rendering.md) and
  [shock finding](guide/shock-finding.md).
- **Looking up a function?** See the [API reference](api/data.md).

!!! note "Units everywhere"
    Every field RICHIO returns is a [`unyt`](https://unyt.readthedocs.io) array
    carrying physical units. Code values live in RICH's solar unit system
    (mass in M☉, length in R☉, time set by G = 1); convert at any time with
    `.in_cgs()`, `.in_mks()`, or `.in_base("rich")`. See [Units](guide/units.md).
