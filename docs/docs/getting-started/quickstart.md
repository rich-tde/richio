# Quickstart

This page walks through loading a snapshot, inspecting its fields, reading a
field, and converting units — the everyday RICHIO loop. It assumes you've
already [installed](installation.md) the package.

## Load a snapshot

Suppose you have a RICH snapshot `./snap_32.h5`. Load it with
[`rio.load`](../api/data.md):

```python
>>> import richio as rio
>>> snap = rio.load("./snap_32.h5")
```

`load` dispatches on the path: an `.h5` / `.hdf5` file gives a `SnapshotH5`, while
a directory of per-field `.npy` / `.txt` files gives a `SnapshotNPY`. Both behave
the same from here on. See [Loading snapshots](../guide/loading-snapshots.md) for
details.

## See what's inside

`snap.info()` prints a metadata panel (path, snapshot number, time, box size,
cell count) and a table of every available field with its unit and aliases:

```python
>>> snap.info()
```

For just the field names, use `keys()`:

```python
>>> snap.keys()
['CMx', 'CMy', 'CMz', 'Density', 'Dissipation', 'DpDx', 'DpDy', 'DpDz',
 'DrhoDx', 'DrhoDy', 'DrhoDz', 'DsieDx', 'DsieDy', 'DsieDz', 'Eg_0', 'Erad',
 'ID', 'InternalEnergy', 'Pressure', 'Temperature', 'Volume', 'Vx', 'Vy', 'Vz',
 'X', 'Y', 'Z', 'divV', 'stickers', 'tracers']
```

## Read a field

Index a snapshot like a dictionary. Field names are case-sensitive but many
**aliases** are accepted, so `snap["Density"]`, `snap["density"]`, `snap["rho"]`,
and the attribute form `snap.density` all return the same array:

```python
>>> density = snap["Density"]
>>> density
unyt_array([4.85566085e-17, 4.94904353e-17, 4.84095341e-17, ...,
       2.43368753e-16, 1.48725291e-16, 1.37720716e-16], shape=(759004,), units='1988415860000000000000000000000*kg/Rsun**3')
```

Every field is a [`unyt`](https://unyt.readthedocs.io) array. The units above are
RICH's default code system, where the mass unit is the solar mass, the length
unit the solar radius, and the time unit is fixed by setting G = 1. See
[Fields and aliases](../guide/fields-and-aliases.md) for the full alias list.

## Convert units

Convert to cgs with `.in_cgs()`:

```python
>>> density.in_cgs()
unyt_array([2.86988266e-16, 2.92507542e-16, 2.86119000e-16, ...,
       1.43840311e-15, 8.79023777e-16, 8.13982497e-16], shape=(759004,), units='g/cm**3')
```

or to mks (m, kg, s) with `.in_mks()`:

```python
>>> density.in_mks()
unyt_array([2.86988266e-13, 2.92507542e-13, 2.86119000e-13, ...,
       1.43840311e-12, 8.79023777e-13, 8.13982497e-13], shape=(759004,), units='kg/m**3')
```

and back to RICH code units with `.in_base("rich")`:

```python
>>> density.in_base("rich")
unyt_array([4.85566085e-17, 4.94904353e-17, 4.84095341e-17, ...,
       2.43368753e-16, 1.48725291e-16, 1.37720716e-16], shape=(759004,), units='1988415860000000000000000000000*kg/Rsun**3')
```

`"rich"` works as a base because RICHIO registers a `'rich'` unit system with
unyt on import. See [Units](../guide/units.md) for the full story, and
[unyt's documentation](https://unyt.readthedocs.io/en/stable/) for everything
else you can do with a `unyt_array`.

## A first picture

A quick mid-plane density slice is one line:

```python
>>> ax, im, data = snap.plots.peek("density")
```

See [Plotting](../guide/plotting.md) for slices, projections, and styling.

## Where to go next

- [Loading snapshots](../guide/loading-snapshots.md) — HDF5 vs NPY, multi-rank files, metadata
- [Fields and aliases](../guide/fields-and-aliases.md) — the field registry and access patterns
- [Units](../guide/units.md) — the RICH code unit system in depth
- [Selecting regions](../guide/selecting-regions.md) — clip to a sub-volume
- [Slices and projections](../guide/slices-and-projections.md) — resample to grids
- [Plotting](../guide/plotting.md) — publication-style figures
