# Loading snapshots

Everything in RICHIO starts with a **snapshot** object returned by
[`rio.load`](../api/data.md).

```python
import richio as rio
snap = rio.load("snap_0042.h5")
```

## Two on-disk formats, one interface

`load` inspects the path and returns the right subclass; both share the same API,
so the rest of the docs apply regardless of which you have.

| Path | Returned object | Notes |
|------|-----------------|-------|
| an `.h5` / `.hdf5` file | `SnapshotH5` | native RICH output, single- or multi-rank |
| a directory of `.npy` / `.txt` files | `SnapshotNPY` | one file per field, `<Field>_<snapnum>.npy` |

```python
snap = rio.load("snap_0042.h5")     # -> SnapshotH5
snap = rio.load("/run/snap_0042/")  # -> SnapshotNPY
```

If the path is neither a readable file nor a directory, `load` raises
`FileNotFoundError`.

!!! note "NPY-only fields"
    A few fields exist only in the NPY format (e.g. `Mass`, `FallbackTime`).
    See [Fields and aliases](fields-and-aliases.md).

## Multi-rank HDF5 files

RICH can write a snapshot split across MPI ranks (`rank0/`, `rank1/`, … groups
inside one HDF5 file). `SnapshotH5` detects this transparently and concatenates
ranks on field access, so you never index a rank by hand. Two attributes expose
the detail when you need it:

- `snap.rank` — number of MPI ranks stored (≥ 1).
- `snap.f` — the open `h5py.File` handle, for direct low-level access.

## Inspecting a snapshot

`snap.info()` prints a formatted summary — path, snapshot number, time, box size,
cycle, cell count (and rank count for multi-rank files) — followed by a table of
every field with its unit and aliases:

```python
snap.info()                      # values in RICH code units (default)
snap.info(unit_system="cgs")     # show units/values in cgs instead
snap.info(show_aliases=False)    # drop the Aliases column
```

Other handy bits of metadata:

```python
snap.snapnum      # integer snapshot index parsed from the filename (-1 if none)
snap.keys()       # sorted list of available field names
len(snap)         # number of cells
snap.box          # domain bounds [x0, y0, z0, x1, y1, z1] (unyt)
snap.time         # simulation time (unyt)
```

## Next

- [Fields and aliases](fields-and-aliases.md) — how to actually read the data
- [Units](units.md) — what the returned values mean
