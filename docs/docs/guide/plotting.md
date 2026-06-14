# Plotting

Every snapshot carries a plotter at `snap.plots` (a `SnapshotPlotter`). Its
methods wrap the [slice and projection](slices-and-projections.md) machinery and
hand you back a rendered matplotlib figure — log-scaled, with an automatic,
unit-aware colorbar.

All plotting methods return a `(ax, im, data)` tuple: the matplotlib `Axes`, the
`pcolormesh` artist, and the underlying 2-D `unyt` array.

## A quick look

[`peek`](../api/plots.md) is the fastest path to a picture — a 512² mid-plane
`xy` density slice:

```python
ax, im, data = snap.plots.peek()            # density by default
ax, im, data = snap.plots.peek("temperature")
```

It falls back to centre-of-mass coordinates (`CMx/CMy/CMz`) automatically if the
`X/Y/Z` fields are absent.

## Slices

[`slice`](../api/plots.md) is the configurable version:

```python
ax, im, data = snap.plots.slice(
    "density",
    res=512,
    plane="xy",
    slice_coord=0,
    cmap="magma",
    unit_system="cgs",
)
```

Useful keywords:

| Keyword | Purpose |
|---------|---------|
| `res` | grid resolution — `int` (square) or `(nx, ny)` |
| `plane`, `slice_coord` | which plane, and where along its normal |
| `cmap` | colormap (default `"twilight"`) |
| `log_scale` | plot log₁₀ of the data (default `True`) |
| `vmin`, `vmax` | colour limits (forwarded to `pcolormesh`) |
| `ax` | draw onto an existing `Axes` instead of a new figure |
| `label_latex`, `unit_latex` | LaTeX symbol / unit for the colorbar label |
| `aspect_equal` | equal aspect ratio (default `True`) |

Any extra keyword is forwarded to `matplotlib.pyplot.pcolormesh`.

## Projections

[`projection`](../api/plots.md) renders a column-integrated map:

```python
ax, im, data = snap.plots.projection("density", res=256, plane="xy")
```

Same styling options as `slice`; `plane=None` integrates along Z.

## Plotting from arrays you already have

If you've already resampled with [`to_2dgrid` / `slice` / `project`](slices-and-projections.md),
render the result directly with the module-level
[`scalar_map`](../api/plots.md):

```python
from richio.plots import scalar_map

sliced, x, y = snap.slice("temperature", res=512)
ax, im = scalar_map(sliced, x, y, cmap="inferno", label_latex=r"T")
```

`scalar_map` takes a 2-D field plus its two coordinate axes, picks sensible
log-scale colour limits, and reads the unit straight off the `unyt` array for the
colorbar.

## Saving and composing

Because you get the `Axes` back, normal matplotlib applies:

```python
import matplotlib.pyplot as plt

ax, im, data = snap.plots.slice("density", res=512)
ax.set_title("Density slice")
ax.figure.savefig("density.png", dpi=200, bbox_inches="tight")

# draw onto your own subplot grid
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
snap.plots.slice("density", res=512, ax=axes[0])
snap.plots.slice("temperature", res=512, ax=axes[1], cmap="inferno")
```

## 3-D from the plotter

The plotter also exposes [`volume`](../api/plots.md) and
[`volume_movie`](../api/plots.md) as thin wrappers over the optional render
backend (they lazily import `yt`):

```python
snap.plots.volume("density", res=256, filename="vr.png")
snap.plots.volume_movie("density", res=256, n_frames=180, filename="spin.mp4")
```

These need `pip install "richio[render]"` — see [Volume rendering](volume-rendering.md).
