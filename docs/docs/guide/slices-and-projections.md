# Slices and projections

RICH stores an unstructured set of Voronoi cells. To analyse or visualise a field
you usually want it on a **regular grid** — either a 2-D slice through the volume
or a column-integrated projection. RICHIO resamples with a k-d tree
(nearest-neighbour) under the hood.

This page covers the data-producing methods on `Snapshot`. To get a *plotted*
figure in one call, see [Plotting](plotting.md) — those methods wrap these.

## Slice: a plane through the volume

[`snap.slice`](../api/data.md) returns the field sampled on a 2-D plane:

```python
sliced, xspace, yspace = snap.slice("density", res=512)
```

- `data` — field name (or a raw array of shape `(N,)`).
- `res` — grid resolution: an `int` for a square grid, or `(nx, ny)`.
- `plane` — `"xy"` (default), `"yz"`, `"zx"`.
- `slice_coord` — coordinate along the normal axis where the cut is taken (default `0`).
- `unit_system` — output unit system (default `"cgs"`).

The return is `(sliced_data, xspace, yspace)`: a 2-D `unyt` array plus the two 1-D
coordinate axes, ready to hand to `pcolormesh` (or to
[`scalar_map`](plotting.md)).

## Project: integrate along a line of sight

[`snap.project`](../api/data.md) integrates a field along the axis normal to a
plane, producing a column map (e.g. column density):

```python
column, xspace, yspace = snap.project("density", res=256, plane="xy")
```

It samples onto a 3-D grid and sums `field * dz` along the integration axis.
`plane=None` (the default) integrates along Z.

## Index maps: resample once, look up many fields

Both `slice` and `project` build their grid by first computing a
**nearest-neighbour index map** — an array of absolute indices into the original
cell array. Building the k-d tree is the expensive part, so if you need several
fields on the *same* geometry, build the map once and index each field yourself:

```python
# 3-D index map (nx, ny, nz) of absolute particle indices
i, x, y, z = snap.to_3dgrid(res=128)

rho = snap.density[i]        # all on the identical grid — no extra tree builds
T   = snap.temperature[i]
P   = snap.pressure[i]
```

The same idea applies in 2-D with [`to_2dgrid`](../api/data.md):

```python
i, x, y = snap.to_2dgrid(res=512, plane="xy", slice_coord=0)
rho = snap.density[i]
T   = snap.temperature[i]
```

Because `i` holds absolute indices into the unfiltered array, `snap.<field>[i]`
gives that field on the grid directly.

## Common options

| Option | Meaning |
|--------|---------|
| `box_size` | Domain bounds `[x0,y0,z0,x1,y1,z1]`; auto-read from `snap.box` when `None`. |
| `selection` | Boolean mask `(N,)` restricting which cells are used. |
| `plane` | Which plane / integration axis to use. |
| `volume_selection` (2-D) | Pre-filter cells near the slice plane to speed up the k-d tree (default `True`). |
| `workers` (`to_3dgrid`) | Threads for the k-d tree query (`-1` = all cores). |

!!! tip "Custom coordinates"
    `X`, `Y`, `Z` default to the `"X"/"Y"/"Z"` fields but accept any field name or
    array — e.g. pass `X="CMx", Y="CMy", Z="CMz"` to grid on centre-of-mass
    coordinates.

## Next

- [Plotting](plotting.md) — the one-call plotting wrappers around these methods
- [Volume rendering](volume-rendering.md) — full 3-D resampling and rendering
