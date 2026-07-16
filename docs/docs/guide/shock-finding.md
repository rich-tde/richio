# Shock finding (optional)

!!! info "This is an optional extra"
    The shock finder lives in `richio.shockfinder` and needs `scipy` and
    `numba`, which aren't installed automatically:

    ```bash
    pip install scipy numba
    ```

A shock is a thin front where gas slams into slower gas ahead of it and is
abruptly compressed and heated. The shock finder locates those fronts in a
snapshot and measures how strong each one is: its Mach number, the factor by
which the gas crosses the sound speed. It follows the method of
[Schaal et al. (2015)](https://arxiv.org/abs/1407.4117).

## The three steps

Finding shocks happens in three stages: work out which cells neighbour which,
flag the cells that lie in a shock, then pin down the front itself and its Mach
number.

```python
import richio as rio
from richio.shockfinder import build_voronoi, find_shock_zone, find_shock_surface

snap = rio.load("snap_0100.h5")

vor  = build_voronoi(snap)                   # 1. who neighbours whom
zone = find_shock_zone(snap, vor)            # 2. cells that lie in a shock
surf = find_shock_surface(snap, vor, zone)   # 3. the front, plus Mach numbers
```

## 1. Find the neighbours

To tell a shock from smooth flow, the algorithm compares each cell with the ones
next to it, so it first needs to know which cells are neighbours. There are two
ways to build that, and they return the same kind of result, so the later steps
do not care which you used:

- `build_voronoi(snap)` uses the exact Voronoi neighbours (the cells sharing a
  face). It is the most faithful, but only practical up to roughly a million
  cells.
- `build_knn(snap, k=48)` instead links each cell to its 48 nearest cells. It
  copes with the clumpy cell distributions in a real run, scales to tens of
  millions of cells, and at `k=48` agrees with the exact Voronoi neighbours more
  than 99.7% of the time. This is the one for large snapshots.

```python
from richio.shockfinder import build_knn

vor = build_knn(snap)                        # for large snapshots
```

### Only build neighbours where it matters

Most cells obviously aren't in a shock, since the gas there is not being squeezed,
and two cheap local checks (the first two Schaal conditions, below) can rule them
out before doing any neighbour work. `shock_candidates` runs those checks and
returns the cells that survive, typically about a quarter of them. Hand that
subset to `build_knn` and it only does the expensive work where it could matter:
the same answer, roughly four times faster and lighter.

```python
from richio.shockfinder import shock_candidates

cand = shock_candidates(snap)
vor  = build_knn(snap, cells=cand.candidates)
zone = find_shock_zone(snap, vor)
```

`shock_candidates(snap)` gives you back an object with two attributes:
`candidates`, the indices of the surviving cells, and `ds`, the direction the
shock is travelling at each cell (down the temperature gradient).

## 2. Flag the shock zone

`find_shock_zone` applies all three of Schaal's tests and returns a True/False
array, one entry per cell, marking the ones that sit inside a shock:

1. The flow is converging, meaning gas is being compressed (`∇·v < 0`).
2. Temperature and pressure rise together, their gradients pointing the same way,
   as they must across a shock.
3. There is a real jump in temperature and pressure between a cell and its
   neighbour, not just a gentle gradient.

```python
zone = find_shock_zone(snap, vor, mach_min=1.3)
n_shock = zone.sum()
```

`mach_min` sets how strong a jump has to be to count, in Mach number. On a
many-core machine, `find_shock_zone_threaded` does the same work split across
threads.

## 3. Pin down the front and its strength

`find_shock_surface` takes the flagged zone (which is a few cells thick) and finds
the actual front within it. For each surface cell it follows the shock direction
out to the gas just ahead (pre-shock) and just behind (post-shock), then uses the
Rankine-Hugoniot jump conditions, the standard relations linking the two sides of
a shock, to estimate the Mach number three independent ways: from the temperature
jump, the pressure jump, and the density jump. Three estimates that agree are a
good sign the front is real.

It returns an object whose attributes are:

- `surface_mask`, `pre_mask`, `post_mask`: True/False arrays marking the front
  cells and the gas just ahead of and behind them.
- `mach_T`, `mach_P`, `mach_rho`: the Mach number from the temperature, pressure,
  and density jumps.

```python
surf = find_shock_surface(snap, vor, zone)

import numpy as np
print("surface cells:", surf.surface_mask.sum())
print("median Mach (from T):", np.median(surf.mach_T))
```

As with the zone step, there is a `find_shock_surface_threaded` for large
snapshots.

## Looking at the result

The masks are ordinary True/False arrays over the snapshot's cells, so they fit
into the rest of RICHIO. You can [clip](selecting-regions.md) to the front and
plot it:

```python
shock = snap.clip(mask=surf.surface_mask)
shock.plots.peek("density")
```

or paint a field only where the shock is, by multiplying through a mask:

```python
ax, im, _ = snap.plots.slice(
    snap.density * surf.surface_mask,
    res=512,
    box_size=[0, -10, 20, 10],
    label_latex=r"\rho_{\rm surface}",
)
```

Multiplying the field by the mask (as in the second example) keeps every cell in
place and just zeroes the ones outside the shock, so the slice still samples the
full mesh. That reads more faithfully than clipping to the scattered shock cells
and slicing *that*; see the note in
[Selecting regions](selecting-regions.md#box-clips-for-pictures-masks-for-statistics).

## The Rankine-Hugoniot relations on their own

The closed-form jump relations are also available directly, in case you want to
go from a Mach number to a density ratio or back. They all take scalars or arrays
and assume an adiabatic index of `gamma=5/3` by default:

- `M2R` / `R2M`: Mach number to density ratio and back
- `MT2T1` / `T2T1M`: Mach number to temperature ratio and back
- `MP2P1` / `P2P1M`: Mach number to pressure ratio and back
- `delta`: the entropy jump

See the [`richio.shockfinder` API reference](../api/shockfinder.md) for the full
signatures.
