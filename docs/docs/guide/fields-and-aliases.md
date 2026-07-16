# Fields and aliases

A snapshot behaves like a read-only dictionary of simulation fields: you ask for
a field by name and get back its values as a unit-carrying array. You can't write
to it, since a snapshot reflects what is on disk.

## Reading a field

There are two ways to read a field, and they do exactly the same thing:

```python
density = snap["Density"]   # square brackets, like a dictionary
density = snap.density      # attribute shortcut
```

Use whichever reads better. The attribute form is convenient when you are typing
interactively; the bracket form is the one for names that aren't valid Python
identifiers, such as `snap["tracers/Star"]`.

To see what is available, ask for the keys:

```python
>>> snap.keys()
['CMx', 'CMy', 'CMz', 'Density', ..., 'X', 'Y', 'Z', 'divV', 'stickers', 'tracers']
```

## You don't need the exact name

The on-disk field names aren't always the ones you would reach for, so each field
also answers to a set of friendlier aliases. All of these return the same density
array:

```python
snap["Density"], snap["density"], snap["rho"], snap["Den"], snap.density
```

So you can write `snap["rho"]` or `snap["T"]` and not worry about capitalisation
or the precise spelling RICH happens to use.

Here are the common fields and a few of the names each one accepts. The full,
authoritative list lives in
[`richio.config.FIELD_REGISTRY`](../api/config.md), the single file where names,
aliases, and units are defined.

| Field | Some names it answers to | What it is |
|-------|--------------------------|------------|
| `Density` | `density`, `rho`, `Den` | mass density |
| `Pressure` | `pressure`, `P` | pressure |
| `Temperature` | `temperature`, `T`, `temp` | temperature |
| `InternalEnergy` | `internal_energy`, `sie`, `IE` | internal energy per unit mass |
| `Erad` | `radiation_energy`, `Rad` | radiation energy |
| `Vx`, `Vy`, `Vz` | `vx`, `velocity_x`, … | velocity components |
| `divV` | `velocity_divergence`, `div_v` | how fast the flow is converging or spreading |
| `X`, `Y`, `Z` | `x`, `pos_x`, `position_x`, … | cell positions |
| `CMx`, `CMy`, `CMz` | `cm_x`, `center_of_mass_x`, … | centre-of-mass positions |
| `Volume` | `volume`, `Vol` | cell volume |
| `Dissipation` | `dissipation`, `Diss` | dissipation rate |
| `tracers/Star` | `star`, `star_ratio`, `Star` | fraction of the cell that is stellar material |
| `tracers/Entropy` | `entropy`, `S` | entropy per unit mass |

A note on the two file formats: `Mass` (`mass`, `m`) and `FallbackTime` (`tfb`)
exist only in the NumPy-directory snapshots. A few fields are also spelled
differently on disk in that format (`Density` is stored as `Den`, `Pressure` as
`P`), but you don't have to think about that, since the same aliases work for
both.

## Reading only part of a field

To grab a subset without reading the whole array, put a slice or a mask after the
field name, separated by a comma:

```python
snap["density", 1:10]     # the first nine values
snap["density", ::-1]     # reversed
snap["density", mask]     # only the cells where `mask` is True
```

## Two ready-made masks

Two common selections come built in. One picks out the cells that are stellar
material, the other drops the near-empty background cells (those sitting at the
simulation's density floor):

```python
star = snap.mask_star_ratio()   # cells that are stellar material
gas  = snap.mask_density()      # cells above the background density floor
```

Each returns a boolean array, one entry per cell, that you can use directly to
pick out values or combine with `&`:

```python
rho_star = snap.density[snap.mask_star_ratio()]
dense_star = snap.mask_star_ratio() & snap.mask_density()
```

You can also hand a mask to [`clip`](selecting-regions.md) to get a smaller
snapshot back instead of a bare array.

!!! warning "Masks are for analysis, not for slices and projections"
    These value-based masks pull out a scattered set of cells. That is exactly
    what you want for statistics, such as the total mass of stellar material or a
    histogram of shocked densities. It is the wrong tool for making a slice or
    projection. Those plots fill the grid by copying the nearest cell to each
    grid point, and once you have removed cells from the middle of the volume,
    the nearest surviving cell can be far away, so the picture fills the holes
    with whatever happens to be closest. To restrict a *picture* to a region, use
    a box clip instead (see [Selecting regions](selecting-regions.md)).

## Next

- [Units](units.md): reading and converting the values
- [Selecting regions](selecting-regions.md): work with part of a snapshot
