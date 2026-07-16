# Slices and projections

RICH does not lay the gas out on a neat grid. Instead it tracks a cloud of
irregular cells (a Voronoi mesh) that move with the flow, packed closely where
the gas is interesting and sparse where it is not. That is good for the
simulation, but most analysis and almost every plot wants the data on a plain
rectangular grid. This page is about getting there: taking a flat 2-D **slice**
through the volume, or a **projection** that adds the gas up along a line of
sight.

To turn an irregular cloud into a grid, RICHIO lays down the grid points and, for
each one, finds the nearest cell and copies its value. The "find the nearest
cell" step uses a k-d tree, a structure built for that kind of nearest-neighbour
search.

The methods here return the data. If you would rather get a finished figure in
one call, see [Plotting](plotting.md); those methods wrap these and add the
drawing.

## Slice: a plane through the volume

[`snap.slice`](../api/data.md) samples a field on a flat plane:

```python
sliced, xspace, yspace = snap.slice("density", res=512)
```

You get back three things: the 2-D array of values, and the x and y coordinates
of its columns and rows. That is the shape `pcolormesh` (or
[`scalar_map`](plotting.md)) wants.

The arguments you will reach for most:

- `data`: the field name (or a raw array, if you have computed something
  yourself).
- `res`: the grid resolution. A single number gives a square grid; a pair
  `(nx, ny)` gives a rectangle.
- `plane`: which plane to cut, `"xy"` (the default), `"yz"`, or `"zx"`.
- `slice_coord`: where along the perpendicular axis to take the cut (default `0`,
  the middle).
- `box_size`: the region to cover (see below); taken from the snapshot if you
  leave it out.
- `unit_system`: the units of the returned values (default `"cgs"`).

## Project: add up along a line of sight

[`snap.project`](../api/data.md) integrates a field through the volume,
collapsing one axis to produce a column map, column density for instance:

```python
column, xspace, yspace = snap.project("density", res=256, plane="xy")
```

It samples onto a 3-D grid and sums the field times the cell depth along the
viewing axis. With `plane=None` (the default) it adds up along z.

## Choosing the region with `box_size`

Both methods accept a `box_size` that picks out the rectangle (or box) to sample,
in RICH length units unless you pass `unyt` quantities. For a slice, you can give
it two ways:

```python
snap.slice("density", res=512, box_size=[-20, -10, 20, 10])           # 4 numbers: x0, y0, x1, y1
snap.slice("density", res=512, box_size=[-20, -10, -5, 20, 10, 5])    # 6 numbers: full box
```

The four-number form is the convenient one for a slice: it is the in-plane
rectangle. The six-number form gives the full box bounds, the same layout as
`snap.box`. Projections, which work in 3-D, use the six-number form.

## Sampling several fields on the same grid

Building the k-d tree is the slow part. If you want several fields on the *same*
grid, say density, temperature, and pressure for one slice, rebuilding that grid
three times is wasteful. So RICHIO lets you build it once.

`to_3dgrid` (and its 2-D companion `to_2dgrid`) do the geometry and the
nearest-cell search, then hand back an array of indices: for each grid point, the
position of the cell that landed there in the original data. Index any field with
that array and you get the field on the grid, with no extra work:

```python
# 3-D grid: i has shape (nx, ny, nz)
i, x, y, z = snap.to_3dgrid(res=128)

rho = snap.density[i]        # all three are on the identical grid,
T   = snap.temperature[i]    # and the tree was only built once
P   = snap.pressure[i]
```

The same idea in 2-D:

```python
i, x, y = snap.to_2dgrid(res=512, plane="xy", slice_coord=0)
rho = snap.density[i]
T   = snap.temperature[i]
```

`slice` and `project` are thin wrappers over these: they call them to get `i`,
then look up the field for you.

## A few more options

Beyond `res`, `plane`, `slice_coord`, and `box_size`, these methods take:

- `selection`: a boolean mask that restricts which cells are considered before
  sampling. See the warning below before using it for a picture.
- `X`, `Y`, `Z`: which fields to use as coordinates. They default to the
  `X`/`Y`/`Z` position fields, but you can point them at others, for example
  `X="CMx", Y="CMy", Z="CMz"` to grid on centre-of-mass positions, or pass your
  own arrays.
- `volume_selection` (slices only, on by default): first drop cells far from the
  slice plane, so the k-d tree has fewer points to search. It speeds the slice up
  without changing the result.
- `workers` (`to_3dgrid`): how many threads to use for the search; `-1` uses
  every core.

!!! warning "Don't restrict a slice or projection with a value-based mask"
    The nearest-cell fill assumes the cells cover the volume. If you pass a
    `selection` (or slice a mask-based clip) that removes cells from the interior
    of the gas, the grid points over those holes get filled by the nearest
    surviving cell, which can be far away, and the picture stops meaning what it
    looks like. To restrict a picture to a region, clip by `box` instead, which
    keeps a connected chunk. Value-based masks (density, star, and the like)
    belong in analysis and statistics rather than in slices and projections. See
    [Selecting regions](selecting-regions.md).

## Next

- [Plotting](plotting.md): the one-call versions that draw the figure for you
- [Volume rendering](volume-rendering.md): full 3-D rendering, not just a grid
