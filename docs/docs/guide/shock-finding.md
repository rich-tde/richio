# Shock finding (optional)

!!! info "Optional capability"
    The shock finder lives in `richio.shockfinder` and depends on `scipy` and
    `numba`, which are **not** core dependencies. Install them first:

    ```bash
    pip install scipy numba
    ```

`richio.shockfinder` detects shock zones and shock surfaces in a snapshot and
estimates Mach numbers, following the algorithm of
[Schaal et al. (2015), MNRAS 446, 3992](https://arxiv.org/abs/1407.4117).

## The workflow

There are three stages: build a neighbour graph, flag the shock **zone**, then
extract the shock **surface** with Mach numbers.

```python
import richio as rio
from richio.shockfinder import build_voronoi, find_shock_zone, find_shock_surface

snap = rio.load("snap_0100.h5")

vor  = build_voronoi(snap)              # 1. neighbour graph
zone = find_shock_zone(snap, vor)       # 2. boolean shock-zone mask (N,)
surf = find_shock_surface(snap, vor, zone)   # 3. surface + Mach numbers
```

## 1. Build the neighbour graph

The kernels traverse a cell-adjacency graph in CSR form
(`SimpleNamespace(positions, neighbor_data, neighbor_ptr)`). Two builders return
the same structure:

- [`build_voronoi(snap)`](../api/shockfinder.md) — exact Voronoi face neighbours
  via `scipy.spatial.Voronoi`. Most accurate, but practical only below ~1M cells.
- [`build_knn(snap, k=48, ...)`](../api/shockfinder.md) — a *k*-nearest-neighbour
  graph that is a drop-in replacement, robust to clustering, and scales to
  ~80M cells. With `k=48` it matches exact Voronoi on >99.7% of queries.

```python
from richio.shockfinder import build_knn, shock_candidates

vor = build_knn(snap)                   # for large snapshots
```

### Restrict the k-NN query to candidates

Only cells passing the cheap local cuts (Schaal conditions 1 & 2) ever have their
neighbours used. [`shock_candidates`](../api/shockfinder.md) computes them; feed
them to `build_knn(cells=...)` to query only that subset (~a quarter of cells),
roughly a 4× speed/memory win with an identical result:

```python
cand = shock_candidates(snap)
vor  = build_knn(snap, cells=cand.candidates)
zone = find_shock_zone(snap, vor)
```

`shock_candidates(snap)` returns `SimpleNamespace(candidates, ds)` — the
candidate indices and the per-cell shock direction `ds = −∇T/|∇T|`.

## 2. Find the shock zone

[`find_shock_zone(snap, vor)`](../api/shockfinder.md) applies all three Schaal+14
criteria and returns a boolean mask over all cells:

- **Condition 1** — converging flow: ∇·v < 0.
- **Condition 2** — temperature gradient aligned with pressure: ∇T · ∇P > 0.
- **Condition 3** — a measurable T and P jump across a Voronoi face.

```python
zone = find_shock_zone(snap, vor, mach_min=1.3)
n_shock = zone.sum()
```

For multi-core machines, [`find_shock_zone_threaded`](../api/shockfinder.md) is a
drop-in that splits the work across a thread pool (numba releases the GIL).

## 3. Extract the shock surface and Mach numbers

[`find_shock_surface(snap, vor, zone)`](../api/shockfinder.md) traces rays along
±`ds` out of the zone and computes three independent Rankine-Hugoniot Mach
estimates per surface cell. It returns a `SimpleNamespace`:

| Field | Meaning |
|-------|---------|
| `surface_mask` | bool `(N,)` — shock-surface cells |
| `pre_mask` | bool `(N,)` — pre-shock cells |
| `post_mask` | bool `(N,)` — post-shock cells |
| `mach_T` | Mach from the temperature jump T₂/T₁ |
| `mach_P` | Mach from the pressure jump P₂/P₁ |
| `mach_rho` | Mach from the density jump ρ₂/ρ₁ |

```python
surf = find_shock_surface(snap, vor, zone)

import numpy as np
print("surface cells:", surf.surface_mask.sum())
print("median Mach (from T):", np.median(surf.mach_T))
```

A threaded variant, [`find_shock_surface_threaded`](../api/shockfinder.md), is
available for large snapshots.

## Visualising the result

The masks are ordinary boolean arrays over the snapshot's cells, so they plug
straight into the rest of RICHIO — for example, [clip](selecting-regions.md) to
the shock surface and plot it:

```python
shock = snap.clip(mask=surf.surface_mask)
shock.plots.peek("density")
```

## Rankine-Hugoniot helpers

The module also exposes closed-form Rankine-Hugoniot relations (all accept
scalars or arrays, `gamma=5/3` by default): `M2R`/`R2M` (Mach ↔ density ratio),
`MT2T1`/`T2T1M` (Mach ↔ temperature ratio), `MP2P1`/`P2P1M` (Mach ↔ pressure
ratio), and `delta` (entropy jump). See the
[`richio.shockfinder` API reference](../api/shockfinder.md).
