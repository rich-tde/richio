# Loading snapshots

Everything in RICHIO starts the same way: you hand a path to
[`rio.load`](../api/data.md) and get back a snapshot, an object that represents
the whole simulation at one moment and lets you read its fields.

```python
import richio as rio
snap = rio.load("snap_0042.h5")
```

## Two file formats, one snapshot

RICH writes snapshots in two layouts, and you will come across both:

- A single HDF5 file (ending in `.h5` or `.hdf5`). This is RICH's native output.
- A directory of NumPy files, with one file per field, named like
  `Density_42.npy`. These come from a separate extraction step.

You don't have to tell `load` which kind you have. It looks at the path, a file
ending in `.h5` versus a directory, and reads it the right way:

```python
snap = rio.load("snap_0042.h5")     # an HDF5 file
snap = rio.load("/run/snap_0042/")  # a directory of .npy files
```

Either way you get a snapshot that behaves the same from here on, so the rest of
the documentation applies whichever format you started with. (If the path is
neither a readable file nor a directory, `load` raises `FileNotFoundError`.)

The two formats don't store quite the same things. A few fields exist only in the
NumPy format, such as `Mass` and `FallbackTime`, and some fields are spelled
differently on disk. RICHIO smooths the spelling differences over for you, so the
same name works either way. See [Fields and aliases](fields-and-aliases.md).

## Snapshots split across processes

When RICH runs on many processors at once, each one writes its share of the
cells, and a single HDF5 file can end up holding several such pieces (stored
internally as groups called `rank0`, `rank1`, and so on). This is the normal
shape of a large run.

You don't have to deal with the pieces yourself. When you read a field, RICHIO
stitches them back together and gives you one array covering the whole
simulation. If you ever need the detail, two attributes expose it:

```python
snap.rank   # how many pieces the file was written in (1 for a single-processor run)
snap.f      # the open h5py file handle, for direct access to the raw data
```

## Looking at the metadata

`snap.info()` prints the most useful summary: the path, the snapshot number, the
time, the box size, the number of cells (and the number of pieces, for a split
file), followed by a table of every field with its units and aliases. By default
it shows values in RICH units. You can ask for cgs instead, or drop the alias
column:

```python
snap.info()                      # RICH units (the default)
snap.info(unit_system="cgs")     # show units and values in cgs
snap.info(show_aliases=False)    # leave out the aliases column
```

The same metadata is also available one piece at a time:

```python
snap.snapnum      # the snapshot number read from the filename (-1 if there isn't one)
snap.keys()       # the list of available field names
len(snap)         # how many cells there are
snap.box          # the box bounds, [x0, y0, z0, x1, y1, z1]
snap.time         # the simulation time
```

Each of these except `snapnum` and `keys()` comes back with units attached, so
for example `snap.time.in_units("day")` gives the time in days.

## Next

- [Fields and aliases](fields-and-aliases.md): how to read the data
- [Units](units.md): what the values mean
