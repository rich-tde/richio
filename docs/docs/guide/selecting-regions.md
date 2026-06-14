# Selecting regions

Often you only care about part of a snapshot — the disrupted star, the inner
region, the dense gas. [`snap.clip`](../api/data.md) gives you a sub-snapshot
restricted to a region of cells.

## A clip is a lazy view

`clip` does **not** copy any field data. It stores only the parent snapshot, a
boolean selection mask, and the clip box. Each field access on the clip re-reads
the parent field and applies the mask on the fly. The result is a drop-in
`Snapshot`: field access, [slices and projections](slices-and-projections.md),
plotting, and the [shock finder](shock-finding.md) all work on it unchanged.

```python
clip = snap.clip(box=[-1, -1, -1, 1, 1, 1])
clip["density"]            # same API as a full snapshot, restricted to the box
clip.plots.peek("density")
```

## Three ways to specify the region

```python
# 1. Explicit bounds [x0, y0, z0, x1, y1, z1]
clip = snap.clip(box=[-1, -1, -1, 1, 1, 1])

# 2. A box centred on a point, given a side length (scalar or per-axis)
clip = snap.clip(center=[0, 0, 0], width=snap.box[3] / 4)

# 3. An arbitrary boolean mask (or integer index array) over the cells
clip = snap.clip(mask=snap.density.v > 1e-15)
```

When you pass more than one of `box`, `center`+`width`, and `mask`, they are
combined with logical **AND** — handy for "dense cells within this box":

```python
clip = snap.clip(box=[-2, -2, -2, 2, 2, 2], mask=snap.mask_star_ratio())
```

!!! note "Bare numbers are code length units"
    Unitless values for `box`, `center`, or `width` are interpreted in RICH code
    length units (R☉). Pass a `unyt` quantity to use any other unit, e.g.
    `width=1e11 * u.cm`.

Passing none of the three (or `center` without `width`, or vice versa) raises
`ValueError`.

## Inspecting a clip

`ClippedSnapshot` exposes the same metadata as its parent, plus the selection it
applied:

```python
clip.mask      # boolean mask over the parent's cells
clip.box       # the clip bounds
len(clip)      # number of cells kept
clip.parent    # the original snapshot
```

## Next

- [Slices and projections](slices-and-projections.md) — resample a clip (or full snapshot) to a grid
- [Plotting](plotting.md) — visualise it
