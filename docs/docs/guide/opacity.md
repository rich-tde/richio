# Opacity

RICHIO can look up Rosseland, Planck and scattering opacities from the same STA
tables RICH itself uses during a run. These are plain functions of temperature
and density rather than stored fields, so you call them with whatever arrays you
have:

```python
import richio

snap = richio.load("snap_0042.h5")
alpha = richio.opacity.rosseland_alpha(snap.T, snap.rho)
```

The tables ship with RICHIO, so nothing needs to be downloaded or configured.

## Which quantity you get

The tables store an **extinction coefficient**, usually written α, with units of
one over length. This is already the product of the specific opacity and the
density, κρ. It is the quantity you integrate along a ray to get an optical
depth:

$$\tau = \int \alpha \, \mathrm{d}r$$

Because both forms are useful, there are two families of functions:

```python
richio.opacity.rosseland_alpha(snap.T, snap.rho)     # α, in 1/cm
richio.opacity.rosseland_opacity(snap.T, snap.rho)   # κ = α/ρ, in cm**2/g
```

and likewise `planck_alpha` / `planck_opacity` and `scattering_alpha` /
`scattering_opacity`. If you want an optical depth, reach for the `_alpha` one.
If you want the number people quote in papers as "the opacity", in cm² per gram,
reach for `_opacity`.

## Units are converted for you

The tables are tabulated in Kelvin and g/cm³, but you do not have to convert
anything first. Pass the fields straight off the snapshot, still in RICH's own
units, and the conversion happens inside:

```python
snap.rho.units          # code density, not g/cm**3
alpha = richio.opacity.rosseland_alpha(snap.T, snap.rho)   # converted internally
alpha.units             # 1/cm
```

This is worth trusting rather than working around. If you hand in something with
the wrong dimensions, you get an error naming the problem instead of a plausible
looking but wrong answer. Arrays with no units attached are assumed to be in cgs
already, and warn you so, since there is nothing there to check.

## Slices, projections and renders

The functions return an ordinary per-cell array, and every RICHIO entry point
that takes a `data` argument accepts an array as readily as a field name. So
there is nothing extra to learn:

```python
alpha = richio.opacity.rosseland_alpha(snap.T, snap.rho)

snap.plots.slice(data=alpha, res=512)
snap.plots.projection(data=alpha, res=512)
```

A projection of α is an optical depth map, since projecting integrates along the
line of sight. For volume rendering, pass the array through the grid builder as a
named field:

```python
from richio.render import to_uniform_grid, volume_image

grid = to_uniform_grid(snap, {"rosseland_alpha": alpha}, res=512)
volume_image(snap, "rosseland_alpha", grid=grid, mode="projection", weight=None)
```

`weight=None` keeps the projection a plain line integral, which is what makes the
result a true optical depth rather than a weighted average.

## Working on part of a snapshot

Because these are functions and not fields, they work on anything of the right
shape, including a clipped region or your own test values:

```python
inner = snap.clip(center=(0, 0, 0), width=200)
alpha = richio.opacity.rosseland_alpha(inner.T, inner.rho)
```

```python
import unyt as u
import numpy as np

T = np.array([1e4, 1e5]) * u.K
rho = np.array([1e-8, 1e-6]) * u.g / u.cm**3
richio.opacity.rosseland_alpha(T, rho)
```

## Using a different table

The bundled tables are used by default. To point at another copy, set the
`RICHIO_OPACITY_DIR` environment variable, or load one explicitly and pass it in:

```python
table = richio.opacity.load_opacity_table("/path/to/STA")
richio.opacity.rosseland_alpha(snap.T, snap.rho, table=table)
```

The directory needs to hold `T.txt`, `rho.txt`, `ross.txt`, `planck.txt` and
`scatter.txt`. Loaded tables are cached, so repeated calls do not re-read them.

## A note on cost

Nothing is cached between calls: each call reads the temperature and density and
runs the interpolation again. If you need the result more than once, keep it in a
variable rather than calling the function repeatedly.
