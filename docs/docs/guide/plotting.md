# Plotting

Every snapshot comes with a plotter, reached as `snap.plots`. Its methods take
the [slice and projection](slices-and-projections.md) machinery from the last
page and hand you back a finished matplotlib figure: log-scaled, with a colorbar
already labelled with the right units.

Each plotting method returns three things:

```python
ax, im, data = snap.plots.peek("density")
```

`ax` is the matplotlib axes (use it to add a title, save the figure, and so on),
`im` is the image object (use it to adjust the colorbar without redrawing), and
`data` is the 2-D array that was plotted. For a quick look you can ignore all
three.

## A quick look

[`peek`](../api/plots.md) is the fastest way to an image: a 512×512 density slice
through the middle of the box.

```python
ax, im, data = snap.plots.peek()              # density by default
ax, im, data = snap.plots.peek("temperature")
```

If the snapshot does not have ordinary position fields, `peek` falls back to the
centre-of-mass positions, so it still produces an image.

## Slices

[`slice`](../api/plots.md) is the version for when you want control:

```python
ax, im, data = snap.plots.slice(
    "density",
    res=512,
    plane="xy",
    slice_coord=0,
    box_size=[-20, -10, 20, 10],
    cmap="magma",
)
```

The geometry arguments (`res`, `plane`, `slice_coord`, `box_size`, `selection`,
`unit_system`, `workers`) are the ones from
[Slices and projections](slices-and-projections.md), and mean the same here. The
rest control how the figure looks:

- `cmap`: the colormap (default `"twilight"`).
- `log_scale`: plot the base-10 logarithm of the data, on by default, which is
  usually what you want for densities and pressures.
- `vmin`, `vmax`: the colour limits. Because the plot is logarithmic, these are
  log values too, so `vmin=37` means 10³⁷ in the data's units.
- `label_latex`, `unit_latex`: LaTeX for the colorbar's symbol and unit. The unit
  is read off the data automatically; set `label_latex` to give the quantity a
  symbol, for example `label_latex=r"\rho"`.
- `ax`: an existing axes to draw onto, instead of making a new figure.
- `aspect_equal`: keep the axes square (on by default).

Any other keyword is passed straight through to matplotlib's `pcolormesh`.

## Projections

[`projection`](../api/plots.md) draws a column-integrated map, with the same
styling options as `slice`:

```python
ax, im, data = snap.plots.projection("density", res=256, plane="xy")
```

As with `snap.project`, `plane=None` integrates along z. The nearest-neighbour
query uses eight threads by default; pass `workers=1` for serial execution or a
different count appropriate for your job allocation.

## Plotting something you computed

You don't have to plot a stored field. Anywhere a method takes `data`, you can
pass an array you have worked out yourself, which is useful for derived
quantities. For example, the energy dissipated per cell is the dissipation rate
times the cell volume:

```python
ax, im, _ = snap.plots.slice(
    snap.dissipation * snap.volume,
    res=256,
    cmap="viridis",
    label_latex=r"\dot E_{\rm diss}",
    vmin=37,
)
```

If you have already resampled with [`slice` or `project`](slices-and-projections.md)
(the data-only versions) and only want to draw the result, use the standalone
[`scalar_map`](../api/plots.md):

```python
from richio.plots import scalar_map

sliced, x, y = snap.slice("temperature", res=512)
ax, im = scalar_map(sliced, x, y, cmap="inferno", label_latex=r"T")
```

`scalar_map` takes a 2-D field and its two coordinate axes, picks sensible
log-scale colour limits, and reads the unit straight off the array for the label.

## Saving and combining figures

Since you get the axes back, everything you already know about matplotlib
applies:

```python
import matplotlib.pyplot as plt

ax, im, data = snap.plots.slice("density", res=512)
ax.set_title("Density slice")
ax.figure.savefig("density.png", dpi=200, bbox_inches="tight")
```

Pass `ax=` to draw into your own layout, for example density and temperature side
by side:

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
snap.plots.slice("density", res=512, ax=axes[0])
snap.plots.slice("temperature", res=512, ax=axes[1], cmap="inferno")
```

## Going 3-D

The plotter also has [`volume`](../api/plots.md) and
[`volume_movie`](../api/plots.md) for 3-D renders and movies:

```python
snap.plots.volume("density", res=256, filename="vr.png")
snap.plots.volume_movie("density", res=256, n_frames=180, filename="spin.mp4")
```

These need the rendering extra (`pip install "richio[render]"`) and are covered in
[Volume rendering](volume-rendering.md).
