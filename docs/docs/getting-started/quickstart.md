# Quickstart

This page walks through the everyday loop: open a snapshot, see what's in it,
read a field, convert its units, and draw a first picture. It assumes you have
already [installed](installation.md) RICHIO. The examples use the `>>>` prompt to
show what you would type and what comes back; you can paste the code without the
prompt.

## Open a snapshot

Say you have a snapshot file `snap_32.h5`. Open it with
[`rio.load`](../api/data.md):

```python
>>> import richio as rio
>>> snap = rio.load("./snap_32.h5")
```

`snap` now stands in for the whole simulation state at that moment. RICH writes
snapshots in two different layouts on disk, and `load` looks at the path to work
out which one you have and read it accordingly. You get the same `snap` object
either way, so the rest of this page is identical no matter which format you
started from. The [loading guide](../guide/loading-snapshots.md) has the details.

## See what's inside

`snap.info()` prints a summary: where the file came from, its snapshot number,
the simulation time, the size of the box, the number of cells, and then a table
of every field it contains with each field's units and the other names you can
call it by:

```python
>>> snap.info()
```

If you only want the field names, ask for the keys:

```python
>>> snap.keys()
['CMx', 'CMy', 'CMz', 'Density', 'Dissipation', 'DpDx', 'DpDy', 'DpDz',
 'DrhoDx', 'DrhoDy', 'DrhoDz', 'DsieDx', 'DsieDy', 'DsieDz', 'Eg_0', 'Erad',
 'ID', 'InternalEnergy', 'Pressure', 'Temperature', 'Volume', 'Vx', 'Vy', 'Vz',
 'X', 'Y', 'Z', 'divV', 'stickers', 'tracers']
```

## Read a field

Read a field the way you read from a dictionary, with square brackets and the
field's name:

```python
>>> density = snap["Density"]
>>> density
unyt_array([4.85566085e-17, 4.94904353e-17, 4.84095341e-17, ...,
       2.43368753e-16, 1.48725291e-16, 1.37720716e-16], shape=(759004,), units='1988415860000000000000000000000*kg/Rsun**3')
```

You don't have to remember the exact spelling. Each field answers to a handful of
aliases, so `snap["Density"]`, `snap["density"]`, and `snap["rho"]` all give you
the same array. There is also an attribute shortcut, `snap.density`, for
interactive typing. The [fields guide](../guide/fields-and-aliases.md) lists the
names for every field.

What comes back is not a plain NumPy array. It is a `unyt_array`, a NumPy array
that remembers its units. The long units above (`...kg/Rsun**3`) are RICH's own
internal system, where masses are measured in solar masses and lengths in solar
radii. That leads to the next step.

## Convert units

Because the array knows its units, switching to a familiar system is one call.
Use `.in_cgs()` for centimetre-gram-second units:

```python
>>> density.in_cgs()
unyt_array([2.86988266e-16, 2.92507542e-16, 2.86119000e-16, ...,
       1.43840311e-15, 8.79023777e-16, 8.13982497e-16], shape=(759004,), units='g/cm**3')
```

`.in_mks()` for metre-kilogram-second:

```python
>>> density.in_mks()
unyt_array([2.86988266e-13, 2.92507542e-13, 2.86119000e-13, ...,
       1.43840311e-12, 8.79023777e-13, 8.13982497e-13], shape=(759004,), units='kg/m**3')
```

and `.in_base("rich")` to go back to RICH's units:

```python
>>> density.in_base("rich")
unyt_array([4.85566085e-17, 4.94904353e-17, 4.84095341e-17, ...,
       2.43368753e-16, 1.48725291e-16, 1.37720716e-16], shape=(759004,), units='1988415860000000000000000000000*kg/Rsun**3')
```

You can also ask for any specific unit by name, such as
`snap["Temperature"].to("K")` or `snap["Vx"].to("km/s")`. The
[Units](../guide/units.md) page covers the RICH unit system, and unyt's own
[documentation](https://unyt.readthedocs.io/en/stable/) covers everything else
you can do with these arrays.

## A first picture

A quick density slice through the middle of the box is one line:

```python
>>> ax, im, data = snap.plots.peek("density")
```

This returns three things: the matplotlib axes, the image object, and the 2-D
array that was drawn. You can ignore them for a quick look, or use them to adjust
the figure. The [Plotting](../guide/plotting.md) page builds on this.

## Where to go next

- [Loading snapshots](../guide/loading-snapshots.md): the two file formats,
  multi-rank files, and metadata
- [Fields and aliases](../guide/fields-and-aliases.md): all the ways to read a
  field
- [Units](../guide/units.md): the RICH unit system in depth
- [Selecting regions](../guide/selecting-regions.md): work with part of a
  snapshot
- [Slices and projections](../guide/slices-and-projections.md): turn the gas
  into a grid
- [Plotting](../guide/plotting.md): make and style figures
