# Selecting regions

Often you only care about part of a snapshot: the disrupted star, the inner few
solar radii, the dense gas rather than the empty surroundings.
[`snap.clip`](../api/data.md) gives you a smaller snapshot restricted to a region
of cells.

```python
clip = snap.clip(box=[-1, -1, -1, 1, 1, 1])
clip["density"]            # same as a full snapshot, but only inside the box
clip.plots.peek("density")
```

The result acts like a full snapshot. You can read fields from it, slice and
project it, plot it, and run the shock finder on it, all with the same methods.
They act on the cells you kept.

A clip does not copy any data. It remembers which cells you selected, and each
time you read a field it reads from the original snapshot and keeps your cells. So
clipping is cheap, and clipping a clip is fine.

## Three ways to say where

You can describe the region you want in three ways:

```python
# 1. Explicit bounds: [x0, y0, z0, x1, y1, z1]
clip = snap.clip(box=[-1, -1, -1, 1, 1, 1])

# 2. A box of a given side length, centred on a point
clip = snap.clip(center=[0, 0, 0], width=snap.box[3] / 4)

# 3. Any boolean mask (or list of cell indices) you have worked out yourself
clip = snap.clip(mask=snap.density.v > 1e-15)
```

For the second form, `width` can be a single number (a cube) or three numbers
(one side length per axis). For the third, the mask is the same kind of
True/False array you would get from `snap.mask_star_ratio()` or any comparison.

If you pass more than one of these at once, you get the cells that satisfy all of
them. That is useful for "the dense cells inside this box":

```python
clip = snap.clip(box=[-2, -2, -2, 2, 2, 2], mask=snap.mask_star_ratio())
```

!!! note "Plain numbers mean RICH length units"
    When you give `box`, `center`, or `width` as bare numbers, they are read in
    RICH's length unit (solar radii). To use any other unit, pass a `unyt`
    quantity instead, for example `width=1e11 * u.cm`.

Passing none of the three raises `ValueError`, and so does giving `center`
without `width` (or the reverse), since a centre with no size is not a region.

## Box clips for pictures, masks for statistics

A box clip keeps a solid, connected chunk of the simulation, so a slice or
projection of it looks the way you expect. This is the right tool when you want to
*see* a region.

A mask (whether you pass it to `clip` or use `snap.mask_density()` and friends
directly) keeps a scattered set of cells and removes the ones in between. That is
fine, and often exactly right, for analysis and statistics: summing the mass of
stellar material, building a histogram of shocked densities, and so on. It is the
wrong tool for a slice or projection. Those plots fill the grid by copying the
nearest cell to each grid point, and once cells are missing from the interior,
the nearest surviving cell can be far away, so the holes fill in with whatever is
closest and the picture no longer means what it appears to.

In short: clip by `box` when you are making an image, and reach for value-based
masks when you are computing numbers.

## What a clip remembers

Alongside the usual snapshot metadata, a clip knows what it selected:

```python
clip.mask      # the True/False array over the original cells
clip.box       # the bounds of the clipped region
len(clip)      # how many cells were kept
clip.parent    # the snapshot it came from
```

## Next

- [Slices and projections](slices-and-projections.md): turn a box clip (or a
  full snapshot) into a grid
- [Plotting](plotting.md): draw it
