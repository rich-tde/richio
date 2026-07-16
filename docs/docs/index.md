# RICHIO

RICHIO (RICH I/O) is a small Python library for reading and analysing the output
of [RICH](https://gitlab.com/eladtan/RICH/-/tree/master?ref_type=heads), a
simulation code for astrophysical gas flows. It was written to study tidal
disruption events (stars torn apart by black holes), but the reading and
analysis tools work just as well for any RICH run.

A RICH simulation writes its state to disk as a *snapshot*: the gas density,
temperature, velocity, and so on, recorded at one moment in time. RICHIO opens a
snapshot, hands you those quantities as NumPy-style arrays, and gives you a few
ways to look at them.

A first example:

```python
import richio as rio

snap = rio.load("snap_0042.h5")    # open a snapshot
snap.info()                         # print what's inside
rho = snap["Density"].in_cgs()      # read a field, converted to g/cm^3
snap.plots.peek("density")          # draw a quick density slice
```

## What it can do

With a plain `pip install richio` you can:

- Open a snapshot in either of RICH's two on-disk formats with one call to
  [`rio.load`](getting-started/quickstart.md). You don't need to know which
  format you have.
- Read any field by name. Every field comes back carrying its physical units, so
  a density is a density and not a bare number whose scaling you have to
  remember.
- Convert units freely, to cgs, to SI, or back to RICH's own units, with one
  method call.
- Cut out a region of the simulation, such as the disrupted star or the dense
  centre, and work with just that.
- Make pictures: slices through the volume, column-integrated projections, and
  publication-ready figures.

Two heavier features are kept behind optional installs so the everyday library
stays small and quick to import:

- 3-D volume rendering: shaded images and rotating movies of the gas
  (`pip install "richio[render]"`).
- Shock finding: locating shock fronts and measuring their strength (needs
  `scipy` and `numba`).

## Where to go next

If you are just starting, read [Installation](getting-started/installation.md)
and then the [Quickstart](getting-started/quickstart.md). Together they take
about ten minutes and cover the everyday loop of loading, reading, and plotting.

After that, the user guide goes one topic at a time:

- [Loading snapshots](guide/loading-snapshots.md): the two file formats and the
  snapshot's metadata
- [Fields and aliases](guide/fields-and-aliases.md): reading data and the many
  names each field answers to
- [Units](guide/units.md): what the numbers mean and how to convert them
- [Selecting regions](guide/selecting-regions.md): working with part of a
  snapshot
- [Slices and projections](guide/slices-and-projections.md): turning the gas
  into a regular grid you can plot
- [Plotting](guide/plotting.md): one-line figures and how to style them
- [Volume rendering](guide/volume-rendering.md) and
  [Shock finding](guide/shock-finding.md): the optional extras

To look up a specific function, see the [API reference](api/data.md).

!!! note "Everything carries units"
    Every field RICHIO returns is a [`unyt`](https://unyt.readthedocs.io) array,
    a NumPy array that also knows its physical units. By default the numbers are
    in RICH's own unit system: masses in solar masses, lengths in solar radii,
    and a time unit fixed by setting the gravitational constant to 1. Convert
    whenever you like with `.in_cgs()`, `.in_mks()`, or `.in_base("rich")`. The
    [Units](guide/units.md) page explains this in full.
