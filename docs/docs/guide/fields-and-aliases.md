# Fields and aliases

A snapshot behaves like a read-only, unit-aware dictionary of simulation fields.

## Two ways to access a field

```python
density = snap["Density"]   # item access
density = snap.density      # attribute access (delegates to item access)
```

Both return the same [`unyt`](https://unyt.readthedocs.io) array. Use whichever
reads better; attribute access is convenient in interactive sessions, item
access is required for names that aren't valid Python identifiers (e.g.
`snap["tracers/Star"]`).

List what's available with `keys()`:

```python
>>> snap.keys()
['CMx', 'CMy', 'CMz', 'Density', ..., 'X', 'Y', 'Z', 'divV', 'stickers', 'tracers']
```

## Aliases

You rarely need to remember the exact on-disk field name. RICHIO resolves a
generous set of **aliases** to each canonical field, so the following are all
equivalent:

```python
snap["Density"], snap["density"], snap["rho"], snap["Den"], snap.density
```

Aliases are defined once in
[`richio.config.FIELD_REGISTRY`](../api/config.md) — the single source of truth
for field names, aliases, and units. The common ones:

| Canonical field | Aliases (selection) | Quantity |
|-----------------|---------------------|----------|
| `Density` | `density`, `rho`, `Den` | mass density |
| `Pressure` | `pressure`, `P` | pressure |
| `Temperature` | `temperature`, `T`, `temp` | temperature |
| `InternalEnergy` | `internal_energy`, `sie`, `IE` | specific internal energy |
| `Erad` | `radiation_energy`, `Rad` | radiation energy |
| `Vx`, `Vy`, `Vz` | `velocity_x`, `vx`, … | velocity components |
| `divV` | `velocity_divergence`, `div_v` | velocity divergence |
| `X`, `Y`, `Z` | `x`, `pos_x`, `position_x`, … | cell-centre positions |
| `CMx`, `CMy`, `CMz` | `cm_x`, `center_of_mass_x`, … | centre-of-mass positions |
| `Volume` | `volume`, `Vol` | cell volume |
| `Dissipation` | `dissipation`, `Diss` | dissipation rate |
| `DpDx…`, `DrhoDx…`, `DsieDx…` | `grad_p_x`, `grad_rho_x`, … | spatial gradients |
| `tracers/Star` | `star`, `star_ratio`, `Star` | stellar-material fraction |
| `tracers/Entropy` | `entropy`, `S` | specific entropy |
| `Time`, `Box`, `Cycle`, `ID` | `time`, `box`, `step`, `id` | metadata |

The full, authoritative list lives in the
[`richio.config` API reference](../api/config.md).

!!! tip "NPY-only fields"
    `Mass` (`mass`, `m`) and `FallbackTime` (`tfb`, `fallback_time`) exist only in
    NPY-format snapshots. Several fields also carry a different canonical name in
    NPY files (e.g. `Density` → `Den`, `Pressure` → `P`); the registry's
    `npy_name` handles that mapping for you so the same alias works either way.

## Slicing without loading everything

Item access accepts an optional index/slice as the second element of a tuple, so
you can pull a subset directly:

```python
snap["density", 1:10]     # first nine density values
snap["density", ::-1]     # reversed
snap["density", mask]     # boolean-mask selection
```

## Convenience masks

Two boolean-mask helpers cover common selections:

```python
star = snap.mask_star_ratio()   # cells that are stellar material (star tracer ≈ 1)
gas  = snap.mask_density()      # cells above the density floor (excludes background)
```

Combine them with field access or feed them to
[`clip`](selecting-regions.md):

```python
rho_star = snap.density[snap.mask_star_ratio()]
```

## Next

- [Units](units.md) — interpreting and converting the values
- [Selecting regions](selecting-regions.md) — restrict to a sub-volume
