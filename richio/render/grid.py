#  Copyright 2025 The RICHIO Contributors
#
#  This file is part of RICHIO.
#
#  RICHIO is free software: you can redistribute it and/or modify it under
#  the terms of the European Union Public License version 1.2 or later, as
#  published by the European Commission.
#
#  RICHIO is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
#  A PARTICULAR PURPOSE. See the European Union Public License for more details.
#
#  You should have received a copy of the EUPL in an/all official language(s) of
#  the European Union along with RICHIO.  If not, see <https://eupl.eu>.

"""Backend-neutral resampling of RICH snapshots onto a uniform Cartesian grid.

This module has **no** 3-D-rendering dependency (no ``yt`` / ``vtk``).  It only
needs ``numpy`` and an existing :class:`richio.data.Snapshot`, so it can be
imported anywhere and reused by any rendering backend.

The heavy lifting — resampling the unstructured Voronoi cells onto a regular
grid — is delegated to :meth:`richio.data.Snapshot.to_3dgrid`, which builds a
k-d tree once and returns nearest-neighbour indices.  A single call yields an
index map that is shared across all requested fields (one tree build, many
field look-ups).
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class UniformGrid:
    """A field cube sampled onto a regular Cartesian grid.

    :param fields: Mapping ``field name -> ndarray`` of shape ``(nx, ny, nz)``.
                   Values are plain (unit-stripped) floats expressed in the
                   units recorded in :attr:`units`.
    :param units: Mapping ``field name -> unit string`` (e.g. ``"g/cm**3"``).
    :param bbox: Domain bounds, shape ``(3, 2)`` =
                 ``[[x0, x1], [y0, y1], [z0, z1]]`` in :attr:`length_unit`.
    :param dims: Grid dimensions ``(nx, ny, nz)``.
    :param length_unit: Unit string for :attr:`bbox` (default ``"cm"``).
    :param time: Simulation time of the source snapshot (in :attr:`time_unit`),
                 or ``None``.
    :param time_unit: Unit string for :attr:`time`.
    """

    fields: dict[str, np.ndarray]
    units: dict[str, str]
    bbox: np.ndarray
    dims: tuple[int, int, int]
    length_unit: str = "cm"
    time: float | None = None
    time_unit: str | None = None
    #: Names of the coordinate fields used to build the grid.
    coords: tuple[str, str, str] = field(default=("X", "Y", "Z"))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        flds = ", ".join(self.fields)
        return (
            f"UniformGrid(dims={self.dims}, fields=[{flds}], "
            f"length_unit={self.length_unit!r})"
        )


def to_uniform_grid(
    snap,
    fields: str | Sequence[str],
    res: int | Sequence[int] = 256,
    *,
    X: str = "X",
    Y: str = "Y",
    Z: str = "Z",
    box_size=None,
    selection=None,
    unit_system: str = "cgs",
    length_unit: str = "cm",
    workers: int = 8,
) -> UniformGrid:
    """Resample one or more snapshot fields onto a uniform 3-D grid.

    Builds the nearest-neighbour index map **once** via
    :meth:`~richio.data.Snapshot.to_3dgrid` and applies it to every requested
    field, so the (expensive) k-d tree is constructed a single time regardless
    of how many fields are requested.

    :param snap: A loaded :class:`richio.data.Snapshot` (or ``ClippedSnapshot``).
    :param fields: A field name or sequence of field names to sample.
    :param res: Grid resolution — a single int for a cubic grid, or
                ``(nx, ny, nz)``.
    :param X: x-coordinate field name. Defaults to ``"X"``.
    :param Y: y-coordinate field name. Defaults to ``"Y"``.
    :param Z: z-coordinate field name. Defaults to ``"Z"``.
    :param box_size: Domain bounds ``[x0, y0, z0, x1, y1, z1]``; ``None`` uses
                     the snapshot's ``box``.
    :param selection: Boolean mask ``(N,)`` restricting which cells are used.
    :param unit_system: Base unit system the field values are expressed in
                        (``"cgs"`` by default, so e.g. density is g/cm**3).
    :param length_unit: Unit for the returned :attr:`UniformGrid.bbox`.
    :param workers: Threads for the k-d tree nearest-neighbour query passed to
                    ``scipy.spatial.KDTree.query``.  Default ``8``: benchmarked
                    on a 70 M-cell snapshot the query saturates by ~8 threads
                    (1.8 s) and *regresses* with more due to oversubscription;
                    ``-1`` uses all cores.  The single-threaded tree *build*
                    (~30 s at 70 M cells) and HDF5 I/O dominate regardless.
    :returns: A populated :class:`UniformGrid`.
    """
    # `fields` may be a name, a list of names, or a {name: (N,) array} mapping of
    # precomputed (e.g. derived) per-cell data.  Normalise to a name->source map
    # where the source is either ``None`` (read from the snapshot) or an array.
    if isinstance(fields, str):
        sources = {fields: None}
    elif isinstance(fields, dict):
        sources = dict(fields)
    else:
        sources = {f: None for f in fields}
    first_field = next(iter(sources))

    # Resolve an automatic tight bounding box around the dense region.
    if isinstance(box_size, str) and box_size == "auto":
        box_size = tight_box(snap, first_field, coords=(X, Y, Z))

    # One k-d tree build, shared by every field. `i` holds absolute indices
    # into the original (unmasked) particle array, shape (nx, ny, nz).
    i, xspace, yspace, zspace = snap.to_3dgrid(
        res, X=X, Y=Y, Z=Z, box_size=box_size, selection=selection, workers=workers
    )
    dims = tuple(int(s) for s in i.shape)

    field_arrays: dict[str, np.ndarray] = {}
    field_units: dict[str, str] = {}
    for f, src in sources.items():
        data = snap._get_data(f) if src is None else snap._get_data(src)
        cube = data[i].in_base(unit_system)
        field_arrays[f] = np.ascontiguousarray(np.asarray(cube), dtype="float64")
        field_units[f] = str(cube.units)

    # Domain bounds in `length_unit`. to_3dgrid uses endpoint=False, so the
    # space runs [lo, hi) with spacing d = (hi - lo) / n; the true upper edge
    # is xspace[-1] + d. Recovering it this way also respects a custom box_size.
    def _edges(space):
        s = space.to(length_unit)
        d = s[1] - s[0]
        return float(s[0].value), float((s[-1] + d).value)

    bbox = np.array([_edges(xspace), _edges(yspace), _edges(zspace)], dtype="float64")

    t = getattr(snap, "time", None)
    time_val = time_unit = None
    if t is not None:
        t = np.atleast_1d(t)
        time_val = float(np.asarray(t)[0])
        time_unit = str(t.units) if hasattr(t, "units") else None

    return UniformGrid(
        fields=field_arrays,
        units=field_units,
        bbox=bbox,
        dims=dims,
        length_unit=length_unit,
        time=time_val,
        time_unit=time_unit,
        coords=(X, Y, Z),
    )


def tight_box(snap, field="density", pct=98.0, pad=1.1, square=True, coords=("X", "Y", "Z")):
    """Bounding box around the cells in the top ``pct`` percentile of *field*.

    Useful as ``box_size`` so a small bright structure fills the frame instead of
    floating in a mostly-empty domain.  The box spans the **full extent** of the
    selected cells (nothing is cropped) and, by default, is made **cubic** so the
    rendered volume keeps its true proportions — a non-cube box squashes the data
    and makes an already-planar structure look artificially flat.

    :param snap: A loaded :class:`richio.data.Snapshot`.
    :param field: Field whose high-percentile cells define the region.
    :param pct: Percentile threshold (cells ``>=`` this percentile are kept).
                Default ``98`` keeps the densest few percent (disk + bright
                stream).
    :param pad: Multiplicative padding on the half-extent (``1.0`` = snug fit).
    :param square: If ``True`` (default) return a cube (half-extent = the largest
                   per-axis half-extent, applied to all axes).  ``False`` keeps
                   the true per-axis extent (anisotropic box).
    :param coords: Names of the coordinate fields.
    :returns: ``unyt`` array ``[x0, y0, z0, x1, y1, z1]`` in code length units,
              ready to pass as ``box_size``.
    """
    f = np.asarray(snap._get_data(field))
    finite = np.isfinite(f)
    thr = np.percentile(f[finite], pct)
    mask = finite & (f >= thr)

    cx = snap._get_data(coords[0])
    out_unit = cx.units
    pos = np.stack(
        [
            np.asarray(cx)[mask],
            np.asarray(snap._get_data(coords[1]))[mask],
            np.asarray(snap._get_data(coords[2]))[mask],
        ],
        axis=1,
    )

    lo = pos.min(axis=0)
    hi = pos.max(axis=0)
    centre = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    if square:
        half = np.full(3, half.max())
    half = half * pad
    box = np.concatenate([centre - half, centre + half])

    import unyt as u

    return u.unyt_array(box, out_unit)


def densest_box(snap, field="density", radius=250.0, coords=("X", "Y", "Z")):
    """Cubic box of half-width *radius* centred on the single densest cell.

    Handy for a disk/black-hole close-up: the densest cell sits in the forming
    disk, so this frames the central region regardless of where the debris is.

    :param snap: A loaded :class:`richio.data.Snapshot`.
    :param field: Field whose maximum defines the centre. Defaults ``"density"``.
    :param radius: Half-width of the cube in the coordinate units (code length).
    :param coords: Names of the coordinate fields.
    :returns: ``unyt`` array ``[x0, y0, z0, x1, y1, z1]`` in code length units.
    """
    import unyt as u

    f = np.asarray(snap._get_data(field))
    ic = int(np.nanargmax(f))
    cx = snap._get_data(coords[0])
    centre = np.array([
        float(np.asarray(cx)[ic]),
        float(np.asarray(snap._get_data(coords[1]))[ic]),
        float(np.asarray(snap._get_data(coords[2]))[ic]),
    ])
    return u.unyt_array(np.concatenate([centre - radius, centre + radius]), cx.units)
