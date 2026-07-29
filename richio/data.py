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

import os
import re
import warnings

import h5py
import numpy as np
from numpy.typing import ArrayLike
from rich.console import Console
from rich.table import Table
import unyt as u

from richio.config import FIELD_REGISTRY
from richio.plots import SnapshotPlotter
from richio.units import units


def _build_h5_aliases():
    """Build the ``_field_aliases`` mapping for :class:`SnapshotH5`.

    Iterates over :data:`~richio.config.FIELD_REGISTRY` and returns a
    dictionary ``{canonical_key: [alias, ...]}`` for every field that exists
    in HDF5 snapshots (i.e. entries without ``npy_only: True``).

    :returns: Mapping from canonical HDF5 key to its list of aliases.
    :rtype: dict[str, list[str]]
    """
    return {k: v["aliases"] for k, v in FIELD_REGISTRY.items() if not v.get("npy_only", False)}


def _build_npy_aliases():
    """Build the ``_field_aliases`` mapping for :class:`SnapshotNPY`.

    For each entry in :data:`~richio.config.FIELD_REGISTRY`, uses the
    ``npy_name`` value as the canonical key (falling back to the HDF5 key when
    ``npy_name`` is absent).  Returns ``{canonical_npy_key: [alias, ...]}``.

    :returns: Mapping from canonical NPY file-name stem to its list of aliases.
    :rtype: dict[str, list[str]]
    """
    result = {}
    for k, v in FIELD_REGISTRY.items():
        npy_key = v.get("npy_name", k)  # use npy_name if present, else h5 key
        result[npy_key] = v["aliases"]
    return result


def load(path):
    """Load a RICH snapshot from disk and return the appropriate Snapshot object.

    Dispatches to :class:`SnapshotH5` for HDF5 files (``.h5`` / ``.hdf5``) or
    to :class:`SnapshotNPY` for directories containing per-field ``.npy`` /
    ``.txt`` files.

    :param path: Path to an HDF5 snapshot file or to a directory of NPY files.
    :returns: The loaded snapshot object.
    :rtype: :class:`SnapshotH5` or :class:`SnapshotNPY`
    :raises FileNotFoundError: If *path* is neither a valid file nor a directory.

    Examples::

        snap = richio.load("snap_0042.h5")
        snap = richio.load("/run/snap_0042/")
    """
    if os.path.isfile(path) and (path.endswith("h5") or path.endswith("hdf5")):  # if hdf5 file
        with h5py.File(path) as f:
            pass
        return SnapshotH5(path)
    elif os.path.isdir(path):  # if a directory
        return SnapshotNPY(path)
    else:
        raise FileNotFoundError(f"{path} is neither a directory nor a file.")


class Snapshot:
    """Abstract base class for RICH simulation snapshots.

    Provides field access, unit handling, masking helpers, and grid
    interpolation methods shared by both :class:`SnapshotH5` and
    :class:`SnapshotNPY`.  Do not instantiate directly — use
    :func:`load` instead.

    :param path: Path to the snapshot file or directory.

    Attributes
    ----------
    path : str
        File or directory path as given to the constructor.
    plots : :class:`~richio.plots.SnapshotPlotter`
        Plotter bound to this snapshot.
    snapnum : int
        Snapshot index extracted from the path, or ``-1`` if not found.
    """

    def __init__(self, path: str):
        self.path = path
        self.plots = SnapshotPlotter(self)  # initialise a plotter object
        self.snapnum = self._get_snapnum()  # snapshot number

        # Build class mapping once (idempotent)
        if not self.__class__._alias_to_canonical:
            self._build_alias_mapping()

    def _get_snapnum(self):
        """Extract the snapshot index from :attr:`path`.

        Searches for the pattern ``snap_<digits>`` anywhere in the path string
        and returns the first match as an integer.

        :returns: Snapshot index, or ``-1`` if the pattern is not found.
        :rtype: int
        """

        # pattern to match 'snap_' followed by digits
        pattern = r"snap_(\d+)"  # (...) matches whatever regular expressions inside the parenthesis, \d matches 0-9, and + matches many digits
        match = re.search(pattern, self.path)  # search for patterns all inside the path

        if match:
            return int(match.group(1))
        else:
            # warnings.warn(f"No snapshot number found in path: {self.path}")
            return -1

    @classmethod
    def _build_alias_mapping(cls):
        """Populate the class-level ``_alias_to_canonical`` dict (called once).

        Iterates over ``cls._field_aliases`` and registers both the canonical
        name and every alias so that :meth:`_resolve_field_name` can perform
        O(1) lookups.  The method is idempotent; subsequent calls are no-ops
        because the dict is already populated.
        """
        for canonical, aliases in cls._field_aliases.items():
            cls._alias_to_canonical[canonical] = canonical
            for alias in aliases:
                cls._alias_to_canonical[alias] = canonical

    def __getattr__(self, name):
        """Enable attribute-style field access (e.g. ``snap.density``).

        Delegates to :meth:`__getitem__` so that any registered field name or
        alias can be used as an attribute.

        :param name: Field name or alias.
        :raises AttributeError: If *name* is not a known field or alias.
        """
        try:
            return self[name]
        except FileNotFoundError:
            raise AttributeError(f"Field '{name}' not found")

    def _resolve_field_name(self, key: str) -> str:
        """Return the canonical field name for *key*, resolving any alias.

        :param key: Field name or alias.
        :returns: Canonical key from :data:`~richio.config.FIELD_REGISTRY`,
                  or *key* unchanged if not found.
        :rtype: str
        """
        return self.__class__._alias_to_canonical.get(key, key)

    def mask_star_ratio(self) -> np.ndarray:
        """Boolean mask selecting stellar-material cells.

        A cell is considered stellar when its ``star`` tracer value is within
        ``1e-3`` of unity.

        :returns: Boolean array of shape ``(N,)``; ``True`` for star cells.
        :rtype: :class:`numpy.ndarray`
        """
        return np.abs(self.star - 1) < 1e-3

    def mask_density(self) -> np.ndarray:
        """Boolean mask excluding density-floor (background) cells.

        Selects cells whose density exceeds the floor threshold
        ``1e-19`` in code density units.

        :returns: Boolean array of shape ``(N,)``; ``True`` for non-floor cells.
        :rtype: :class:`numpy.ndarray`
        """
        return self.density > 1e-19 * units.get_unit("Density")

    @property
    def _field_info(self):
        """Per-field metadata for every field present in this snapshot.

        For fields registered in :data:`~richio.config.FIELD_REGISTRY` the
        ``unit`` and ``aliases`` keys are populated.  For unrecognised fields
        ``unit`` is ``None`` and ``aliases`` is an empty list — they are still
        included so users can see all available fields.

        :returns: Mapping ``{field: {"unit": <unit or None>, "aliases": [...]}}``
        :rtype: dict[str, dict]
        """
        info = {}
        for field in self.keys():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                unit = units.get_unit(field, default=None)
            info[field] = {
                "unit": unit,
                "aliases": self._field_aliases.get(field, []),
            }
        return info

    def info(self, unit_system="rich", show_aliases=True) -> None:
        """
        Display snapshot metadata and available fields in a rich-formatted table.

        Prints a metadata panel (path, snapshot number, time, box size, cycle,
        cell count) followed by a field table listing the name, unit, and
        aliases of every field present in the snapshot.

        :param unit_system: Unit system used for displaying values: ``'rich'``
                            (code solar units, default), ``'cgs'``, or
                            ``'mks'``.
        :param show_aliases: Whether to include an *Aliases* column in the
                             field table.  Defaults to ``True``.

        Examples::

            snap.info()
            snap.info(unit_system='cgs')
            snap.info(show_aliases=False)
        """
        console = Console()

        # ---- Metadata table ------------------------------------------------
        meta_table = Table(show_header=False, box=None, padding=(0, 1))
        meta_table.add_column(style="bold cyan", no_wrap=True)
        meta_table.add_column(style="white")

        meta_table.add_row("Path", str(self.path))
        meta_table.add_row("Snapshot number", str(self.snapnum))

        for attr, label in [("time", "Time"), ("box", "Box size"), ("cycle", "Cycle")]:
            try:
                val = getattr(self, attr).in_base(unit_system)
                # Squeeze single-element arrays to scalar unyt_quantity for cleaner display
                if hasattr(val, "ndim") and val.ndim <= 1 and val.size == 1:
                    val = val[0]
                meta_table.add_row(label, str(val))
            except Exception:
                pass

        try:
            meta_table.add_row("Number of cells", f"{len(self):,}")
        except Exception:
            pass

        if hasattr(self, "rank"):
            meta_table.add_row("Number of ranks", str(self.rank))

        console.rule("[bold]RICH Snapshot Information[/bold]")
        console.print(meta_table)

        # ---- Fields table --------------------------------------------------
        field_table = Table(
            title=f"Available Fields  (unit system: {unit_system.upper()})",
            show_lines=False,
            header_style="bold magenta",
        )
        field_table.add_column("Field", style="cyan", no_wrap=True)
        field_table.add_column("Unit", style="green")
        if show_aliases:
            field_table.add_column("Aliases", style="dim")

        for field, meta in self._field_info.items():
            unit = meta["unit"]

            if unit is None:
                unit_str = "[dim]?[/dim]"
            elif unit == 1:
                unit_str = "[dim](unused)[/dim]"
            elif unit_system == "rich":
                unit_str = str(unit)
            else:
                try:
                    unit_str = str((1.0 * unit).in_base(unit_system))
                except Exception:
                    unit_str = "[dim]?[/dim]"

            if show_aliases:
                aliases = meta.get("aliases", [])
                alias_str = ", ".join(aliases) if aliases else "[dim]-[/dim]"
                field_table.add_row(field, unit_str, alias_str)
            else:
                field_table.add_row(field, unit_str)

        console.print(field_table)
        console.print(f"Total: [bold]{len(self.keys())}[/bold] fields")
        console.rule()

    def _get_data(self, data: str | ArrayLike) -> u.unyt_array:
        """Resolve *data* to a :class:`unyt.unyt_array` with physical units.

        Accepts a field name string (looked up via :meth:`__getitem__`), a
        :class:`unyt.unyt_array` (passed through unchanged), or a bare
        :class:`numpy.ndarray` (treated as dimensionless with a warning).

        :param data: Field name, unit-bearing array, or dimensionless array.
        :returns: Data array with units attached.
        :rtype: :class:`unyt.unyt_array`
        :raises TypeError: For unsupported input types.
        """
        if isinstance(data, str):
            key = data
            data = self[key]
        elif isinstance(data, u.unyt_array):
            pass
        elif isinstance(data, np.ndarray):
            data = data * u.Dimensionless
            warnings.warn("No unit attached, assuming data is dimensionless.")
        else:
            raise TypeError(
                f"Data type {type(data)} unsupported."
                "Use either str, unyt.unyt_array, or numpy.ndarray."
            )

        return data

    def project(
        self,
        data: str | ArrayLike,
        res: int | ArrayLike,
        X: str | ArrayLike = "X",
        Y: str | ArrayLike = "Y",
        Z: str | ArrayLike = "Z",
        box_size: ArrayLike | None = None,
        unit_system: str = "cgs",
        selection: ArrayLike = None,
        plane: str | None = None,
    ):
        """Interpolate *data* onto a 3-D grid and integrate (project) along the
        normal axis of *plane*.

        Calls :meth:`to_3dgrid` to build the nearest-neighbour grid, then sums
        ``grid_data * dz`` along the integration axis to produce a
        column-integrated 2-D map.

        :param data: Field to project — field name string or array of shape
                     ``(N,)``.
        :param res: Grid resolution — single integer for a cubic grid, or a
                    three-element sequence ``(nx, ny, nz)``.
        :param X: x-coordinates of cell centres (field name or array).
                  Defaults to ``"X"``.
        :param Y: y-coordinates of cell centres. Defaults to ``"Y"``.
        :param Z: z-coordinates of cell centres. Defaults to ``"Z"``.
        :param box_size: Domain bounds ``[x0, y0, z0, x1, y1, z1]``.  Reads
                         from the snapshot's ``box`` field when ``None``.
        :param unit_system: Target unit system for the output (``'cgs'``,
                            ``'rich'``, etc.).  Defaults to ``'cgs'``.
        :param selection: Boolean mask of shape ``(N,)`` to restrict which
                          cells are used.  Defaults to ``None`` (all cells).
        :param plane: Projection plane (e.g. ``"xy"``, ``"xz"``, ``"yz"``).
                      Determines which axis is integrated over.  ``None``
                      (default) integrates along Z.
        :returns: Tuple ``(projected_data, xspace, yspace)`` where
                  *projected_data* has shape ``(nx-1, ny-1)`` and *xspace* /
                  *yspace* are 1-D coordinate arrays.
        :rtype: tuple[:class:`unyt.unyt_array`, :class:`unyt.unyt_array`,
                      :class:`unyt.unyt_array`]
        """
        i, xspace, yspace, zspace = self.to_3dgrid(res, X, Y, Z, box_size, selection, plane=plane)

        data = self._get_data(data)
        grid_data = data[i]

        dz = zspace[1:] - zspace[:-1]  # PM: dz = (z1 - z0) / (nz - 1)
        projected_data = np.sum(grid_data[:-1, :-1, :-1] * dz, axis=-1).in_base(
            unit_system
        )  # PM: grid_data[:, :, :-1]

        return projected_data, xspace, yspace

    def to_3dgrid(
        self,
        res: int | ArrayLike,
        X: str | ArrayLike = "X",
        Y: str | ArrayLike = "Y",
        Z: str | ArrayLike = "Z",
        box_size: ArrayLike | None = None,
        selection: ArrayLike = None,
        endpoint: bool = False,
        plane: str | None = None,
        workers: int = 1,
    ):
        """Interpolate cell centres onto a regular 3-D Cartesian grid.

        Builds a 3-D rectilinear grid spanning the domain bounds, then uses a
        k-d tree (nearest-neighbour) to find the closest cell for each grid point.
        Returns absolute indices into the original particle array; used by
        :meth:`project` to look up field values.

        :param res: Grid resolution — single integer for a cubic grid or a
                    three-element sequence ``(nx, ny, nz)``.
        :param X: x-coordinates of cell centres. Defaults to ``"X"``.
        :param Y: y-coordinates of cell centres. Defaults to ``"Y"``.
        :param Z: z-coordinates of cell centres. Defaults to ``"Z"``.
        :param box_size: Domain bounds ``[x0, y0, z0, x1, y1, z1]``.  Reads
                         from the snapshot's ``box`` field when ``None``.
        :param selection: Boolean mask ``(N,)`` to restrict which cells are
                          used.  Defaults to ``None``.
        :param endpoint: If ``True`` the grid spacing is
                         ``(hi-lo)/(n-1)``; if ``False`` (default) it is
                         ``(hi-lo)/n`` so the grid never reaches the upper
                         boundary.
        :param plane: Projection plane, e.g. ``"xy"``, ``"xz"``, ``"yz"``.
                      Permutes the coordinate axes so that the third axis (the
                      integration axis for :meth:`project`) matches the normal
                      of the requested plane.  ``None`` (default) leaves the
                      axis order unchanged (integrates along Z).
        :returns: Tuple ``(i, xspace, yspace, zspace)`` where *i* has shape
                  ``(nx, ny, nz)`` and contains **absolute** indices into the
                  original particle array, so ``snap.density[i]`` gives the
                  projected field directly.
        :rtype: tuple
        """
        # Fetch data
        X = self._get_data(X)
        Y = self._get_data(Y)
        Z = self._get_data(Z)

        # Select cells
        if selection is not None:
            X = X[selection]
            Y = Y[selection]
            Z = Z[selection]

        # Set boxsize
        if box_size is None:
            x0, y0, z0, x1, y1, z1 = self.box  # Load the box size
        else:
            x0, y0, z0, x1, y1, z1 = box_size

            # Assign the default (code length ~ R_sun) unit to any bare number,
            # so an explicit numeric box works like a unit-bearing one and the
            # returned coordinate spaces carry units.
            def _as_len(v):
                return v if isinstance(v, u.unyt_quantity) else v * units.lscale

            x0, y0, z0, x1, y1, z1 = (_as_len(v) for v in (x0, y0, z0, x1, y1, z1))

        # Permute axes so that Z is the integration axis for the requested plane
        if plane is not None:
            X, Y, Z = _parse_plane(plane, X, Y, Z)
            x0, y0, z0 = _parse_plane(plane, x0, y0, z0)
            x1, y1, z1 = _parse_plane(plane, x1, y1, z1)

        # Set resolution
        try:
            nx, ny, nz = res[0], res[1], res[2]
        except TypeError:
            nx = ny = nz = res

        # Make Euclidean grid
        # disable endpoints by default such that dz = (z1-z0)/res instead of (z1-z0)/(res-1)
        # PM: endpoint=True
        # TODO: add an option to use np.geomspace
        xspace = np.linspace(x0, x1, nx, endpoint=endpoint)
        yspace = np.linspace(y0, y1, ny, endpoint=endpoint)
        zspace = np.linspace(z0, z1, nz, endpoint=endpoint)

        # Nearest-neighbour resample.  Build the k-d tree once and query the grid
        # in x-slabs, filling the (nx, ny, nz) index cube, instead of
        # materialising the whole (nx, ny, nz, 3) query array — that array alone
        # is ~50 GB at res 1024 and would blow the node memory.  Peak transient is
        # one slab of query points; the result is identical to a single query.
        from scipy.spatial import KDTree

        coords = np.stack([X, Y, Z], axis=-1)  # coordinates of the particles
        tree = KDTree(np.asarray(coords))

        xs = np.asarray(xspace, dtype="float64")
        yy, zz = np.meshgrid(
            np.asarray(yspace, dtype="float64"), np.asarray(zspace, dtype="float64"), indexing="ij"
        )
        yz = np.column_stack([yy.ravel(), zz.ravel()])  # (ny*nz, 2)
        del yy, zz

        plane_pts = ny * nz
        slab = max(1, int(8_000_000 // max(plane_pts, 1)))  # ~8M query points/chunk
        block = np.empty((slab * plane_pts, 3), dtype="float64")
        i_local = np.empty((nx, ny, nz), dtype=np.intp)
        for a in range(0, nx, slab):
            m = min(slab, nx - a)
            b = block[: m * plane_pts]
            b[:, 0] = np.repeat(xs[a : a + m], plane_pts)
            b[:, 1] = np.tile(yz[:, 0], m)
            b[:, 2] = np.tile(yz[:, 1], m)
            _, idx = tree.query(b, k=1, eps=0, p=2, workers=workers)
            i_local[a : a + m] = idx.reshape(m, ny, nz)

        # Map local indices back to absolute indices in the original particle array
        if selection is not None:
            i = np.where(selection)[0][i_local]
        else:
            i = i_local

        return i, xspace, yspace, zspace

    def to_2dgrid(
        self,
        res: int | ArrayLike,
        X: str | ArrayLike = "X",
        Y: str | ArrayLike = "Y",
        Z: str | ArrayLike = "Z",
        plane: str = "xy",
        slice_coord: float | u.array.unyt_quantity = 0,
        box_size: ArrayLike | None = None,
        selection: ArrayLike | None = None,
        volume_selection: bool = True,
    ):
        """Compute the nearest-neighbour index map for a 2-D slice plane.

        Handles all geometry — plane permutation, volume-proximity filtering,
        and k-d tree interpolation — and returns **absolute** indices into the
        original (unfiltered) particle array so that multiple fields can be
        looked up with a single ``field[i]`` without repeating the interpolation.

        :param res: Grid resolution — single integer (square) or ``(nx, ny)``.
        :param X: x-coordinates of cell centres. Defaults to ``"X"``.
        :param Y: y-coordinates of cell centres. Defaults to ``"Y"``.
        :param Z: z-coordinates of cell centres. Defaults to ``"Z"``.
        :param plane: Slice plane, e.g. ``"xy"`` (default), ``"yz"``, ``"zx"``.
        :param slice_coord: Normal-axis coordinate at which to slice.
                            Defaults to ``0``.
        :param box_size: Domain bounds ``[x0, y0, z0, x1, y1, z1]`` (6-element)
                         or ``[x0, y0, x1, y1]`` (4-element, plane only).
                         Auto-detected when ``None``.
        :param selection: Boolean mask ``(N,)`` to restrict which cells are
                          used.  Defaults to ``None`` (all cells).
        :param volume_selection: Pre-filter to cells within one cell-size of
                                 the plane to speed up the k-d tree query.
                                 Defaults to ``True``.
        :returns: Tuple ``(i, xspace, yspace)`` where *i* has shape ``(nx, ny)``
                  and contains **absolute** indices into the original particle
                  array (before any masking), so ``snap.density[i]`` gives the
                  sliced field directly.
        :rtype: tuple[:class:`numpy.ndarray`, :class:`unyt.unyt_array`,
                      :class:`unyt.unyt_array`]
        """
        # TODO: implement star_mask : put data to zero instead of removing them
        # (which is bad if you use nn) and put them to the lowest color in
        # colormap when plotting (something like set_bad...)

        X = self._get_data(X)
        Y = self._get_data(Y)
        Z = self._get_data(Z)

        # Build combined boolean mask over the full particle set
        mask = np.ones(len(X), dtype=bool)
        if selection is not None:
            mask &= selection

        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]
        if volume_selection:
            volume = self._get_data("volume")[mask]

        # Set resolution
        try:
            nx, ny = res[0], res[1]
        except TypeError:
            nx = ny = res

        # Set boxsize
        if box_size is None:
            x0, y0, z0, x1, y1, z1 = self.box
            x0, y0, z0 = _parse_plane(plane, x0, y0, z0)
            x1, y1, z1 = _parse_plane(plane, x1, y1, z1)
        else:
            if not isinstance(box_size, u.unyt_array):
                box_size *= units.lscale
            if len(box_size) == 6:
                x0, y0, z0, x1, y1, z1 = box_size
                x0, y0, z0 = _parse_plane(plane, x0, y0, z0)
                x1, y1, z1 = _parse_plane(plane, x1, y1, z1)
            elif len(box_size) == 4:
                x0, y0, x1, y1 = box_size

        if not isinstance(slice_coord, u.unyt_quantity):
            slice_coord *= units.lscale

        # Redefine X, Y to be the in-plane axes and Z the normal axis
        X, Y, Z = _parse_plane(plane, X, Y, Z)

        if volume_selection:
            vol_mask = np.abs(Z - slice_coord) < volume ** (1 / 3)
            # assuming spherical cells, V^(1/3)=(4pi/3)^(1/3)R ~ 1.6R, we don't
            # include the factor such that if V is not round enough we won't
            # lose too much accuracy
            # Fold vol_mask back into the full-array mask so we can return
            # absolute indices
            mask[np.where(mask)[0][~vol_mask]] = False
            X = X[vol_mask]
            Y = Y[vol_mask]
            Z = Z[vol_mask]

        # Make Euclidean grid
        xspace = np.linspace(x0, x1, nx, endpoint=False)
        yspace = np.linspace(y0, y1, ny, endpoint=False)
        zspace = slice_coord

        grid_x, grid_y, grid_z = np.meshgrid(xspace, yspace, zspace, indexing="ij")

        coords = np.stack([X, Y, Z], axis=-1)
        grid_coords = np.squeeze(
            np.stack([grid_x, grid_y, grid_z], axis=-1)
        )  # (nx, ny, 1, 3) → (nx, ny, 3)

        i_local = _kdtree_interpolate(coords=coords, grid_coords=grid_coords)

        # Map local indices back to absolute indices in the original particle array
        i = np.where(mask)[0][i_local]

        return i, xspace, yspace

    def slice(
        self,
        data: str | ArrayLike,
        res: int | ArrayLike,
        X: str | ArrayLike = "X",
        Y: str | ArrayLike = "Y",
        Z: str | ArrayLike = "Z",
        plane: str = "xy",
        slice_coord: float | u.array.unyt_quantity = 0,
        box_size: ArrayLike | None = None,
        selection: ArrayLike | None = None,
        unit_system: str = "cgs",
        volume_selection: bool = True,
    ):
        """Make a slice of the simulation grid.

        Delegates geometry and interpolation to :meth:`to_2dgrid`, then looks
        up *data* using the returned absolute indices.  To slice multiple fields
        at the same slice plane without repeating the interpolation, call
        :meth:`to_2dgrid` directly and index each field with the returned *i*.

        :param data: Field to slice — name string or array of shape ``(N,)``.
        :param res: Grid resolution — integer (square) or ``(nx, ny)`` tuple.
        :param X: x-coordinates. Defaults to ``"X"``.
        :param Y: y-coordinates. Defaults to ``"Y"``.
        :param Z: z-coordinates. Defaults to ``"Z"``.
        :param plane: Slice plane. Defaults to ``"xy"``.
        :param slice_coord: Normal-axis coordinate at which to slice.
                            Defaults to ``0``.
        :param box_size: Domain bounds. Auto-detected when ``None``.
        :param selection: Boolean cell mask. Defaults to ``None`` (all cells).
        :param unit_system: Output unit system. Defaults to ``'cgs'``.
        :param volume_selection: Pre-filter cells near the slice plane.
                                 Defaults to ``True``.
        :returns: Tuple ``(sliced_data, xspace, yspace)``.
        :rtype: tuple
        """
        i, xspace, yspace = self.to_2dgrid(
            res=res,
            X=X,
            Y=Y,
            Z=Z,
            plane=plane,
            slice_coord=slice_coord,
            box_size=box_size,
            selection=selection,
            volume_selection=volume_selection,
        )
        sliced_data = self._get_data(data)[i].in_base(unit_system)
        return sliced_data, xspace, yspace

    def profile(
        self,
        data: str | ArrayLike,
        weights: str | ArrayLike = "none",
        res: int = 100,
        X: str = "X",
        Y: str = "Y",
        Z: str = "Z",
    ):
        if isinstance(weights, str):
            weights_name = weights
        else:
            weights_name = "none"

        data = self._get_data(data)
        if weights_name != "none":
            weights = self._get_data(weights_name)
        X = self._get_data(X)
        Y = self._get_data(Y)
        Z = self._get_data(Z)

        r = (X**2 + Y**2 + Z**2) ** (1 / 2)

        rbins = np.logspace(np.log10(np.min(r)), np.log10(np.max(r)), res)
        rbins *= r.units
        if weights_name != "none":
            profile, bin_edges = np.histogram(r, rbins, weights=weights * data)
        else:
            profile, bin_edges = np.histogram(r, rbins, weights=data)

        if weights_name == "volume":
            profile /= 4 / 3 * np.pi * ((rbins[1:]) ** 3 - (rbins[:-1]) ** 3)
        elif weights_name == "none":
            weights_hist, bin_edges = np.histogram(r, rbins)
            profile /= weights_hist
        else:
            weights_hist, bin_edges = np.histogram(r, rbins, weights=weights)
            profile /= weights_hist

        return profile, rbins[1:]

    def line_profile(
        self,
        data: str | ArrayLike,
        p0: ArrayLike,
        p1: ArrayLike,
        res: int = 200,
        X: str | ArrayLike = "X",
        Y: str | ArrayLike = "Y",
        Z: str | ArrayLike = "Z",
        selection: ArrayLike = None,
        unit_system: str = "cgs",
        workers: int = 1,
    ):
        """Sample *data* along the straight line from *p0* to *p1*.

        Lays down ``res`` evenly spaced points between the two 3-D coordinates
        and uses a nearest-neighbour k-d tree to pick the value of the cell
        closest to each point.  Handy for a 1-D cut through a structure (e.g.
        across a shock front) when neither the radial :meth:`profile` nor the
        planar :meth:`slice` matches the geometry you care about.

        Endpoints are unit-aware: a bare number is treated as a code-length
        coordinate (``units.lscale`` ~ solar radius), exactly like
        :meth:`to_3dgrid`; a :class:`unyt.unyt_quantity` is converted to the
        coordinate units.

        :param data: Field to sample — name string or array of shape ``(N,)``.
        :param p0: Start point ``(x, y, z)`` of the line.
        :param p1: End point ``(x, y, z)`` of the line.
        :param res: Number of sample points along the line.  Defaults to ``200``.
        :param X: x-coordinates of cell centres. Defaults to ``"X"``.
        :param Y: y-coordinates of cell centres. Defaults to ``"Y"``.
        :param Z: z-coordinates of cell centres. Defaults to ``"Z"``.
        :param selection: Boolean mask ``(N,)`` restricting which cells are
                          used.  Defaults to ``None`` (all cells).
        :param unit_system: Unit system for the returned values (``'cgs'`` by
                            default).
        :param workers: Threads for the k-d tree query.  Defaults to ``1``.
        :returns: Tuple ``(coords, distance, values)`` where *coords* is a
                  ``(res, 3)`` :class:`unyt.unyt_array` of the sample positions,
                  *distance* is a ``(res,)`` arc length from *p0* along the
                  line, and *values* is a ``(res,)`` array in *unit_system*.
                  Plot a profile with ``plt.plot(distance, values)``.
        :rtype: tuple
        """
        # Coordinates of the cell centres (optionally restricted to a subset).
        Xc = self._get_data(X)
        Yc = self._get_data(Y)
        Zc = self._get_data(Z)
        if selection is not None:
            Xc, Yc, Zc = Xc[selection], Yc[selection], Zc[selection]
        length_unit = Xc.units

        # Attach code-length units to any bare-number endpoint, mirroring the
        # `_as_len` handling in `to_3dgrid`, then express both in the cell units.
        def _as_len(v):
            return v.to(length_unit) if isinstance(v, u.unyt_quantity) else v * units.lscale

        p0 = u.unyt_array([_as_len(c) for c in p0]).to(length_unit)
        p1 = u.unyt_array([_as_len(c) for c in p1]).to(length_unit)

        # res points p0 + t*(p1 - p0), t in [0, 1]; distance is the arc length.
        t = np.linspace(0.0, 1.0, res)
        coords = p0 + t[:, None] * (p1 - p0)  # (res, 3), unit-aware
        distance = t * float(np.linalg.norm((p1 - p0).value)) * length_unit  # (res,)

        src = np.stack([np.asarray(Xc), np.asarray(Yc), np.asarray(Zc)], axis=-1)
        i_local = _kdtree_interpolate(
            coords=src, grid_coords=np.asarray(coords), workers=workers
        )

        # Map local indices back to absolute indices in the original array.
        i = np.where(selection)[0][i_local] if selection is not None else i_local

        values = self._get_data(data)[i].in_base(unit_system)
        return coords, distance, values

    def clip(
        self,
        box: ArrayLike | None = None,
        center: ArrayLike | None = None,
        width: float | u.unyt_quantity | ArrayLike | None = None,
        mask: ArrayLike | None = None,
        X: str | ArrayLike = "X",
        Y: str | ArrayLike = "Y",
        Z: str | ArrayLike = "Z",
    ) -> "ClippedSnapshot":
        """Return a :class:`ClippedSnapshot` holding only a region of cells.

        The clip is a *lazy view*: it stores only this snapshot, a boolean
        selection mask, and the clip box.  No field data is copied — each field
        access on the clip re-reads the parent field and applies the mask, so
        the clip is a drop-in :class:`Snapshot` that geometry methods
        (:meth:`project`, :meth:`slice`) and the shock finder operate on
        unchanged.

        The region can be specified in any combination of:

        * ``box`` — explicit bounds ``[x0, y0, z0, x1, y1, z1]``;
        * ``center`` + ``width`` — an axis-aligned box of side ``width``
          (scalar or per-axis) centred on ``center``;
        * ``mask`` — an arbitrary boolean array of shape ``(N,)`` or an array of
          integer indices.

        When several are given they are combined with logical AND.  Bare
        (unitless) numbers for ``box`` / ``center`` / ``width`` are interpreted
        in code length units (``units.lscale``).

        :param box: Explicit bounds ``[x0, y0, z0, x1, y1, z1]``.
        :param center: Box centre ``[cx, cy, cz]`` (used with ``width``).
        :param width: Box side length — scalar or ``[wx, wy, wz]`` (used with
                      ``center``).
        :param mask: Boolean mask ``(N,)`` or integer index array.
        :param X: x-coordinate field name or array. Defaults to ``"X"``.
        :param Y: y-coordinate field name or array. Defaults to ``"Y"``.
        :param Z: z-coordinate field name or array. Defaults to ``"Z"``.
        :returns: A lazy regional view of this snapshot.
        :rtype: :class:`ClippedSnapshot`
        :raises ValueError: If no region is specified, or if ``center`` is given
                            without ``width`` (or vice versa).

        Examples::

            clip = snap.clip(box=[-1, -1, -1, 1, 1, 1])
            clip = snap.clip(center=[0, 0, 0], width=snap.box[3] / 4)
            clip = snap.clip(mask=snap.density.v > 1e-15)
        """
        x = self._get_data(X)
        y = self._get_data(Y)
        z = self._get_data(Z)

        sel = np.ones(len(x), dtype=bool)
        clip_box = None

        def _as_length(val):
            # Attach code length units to bare numbers, mirroring to_2dgrid.
            if isinstance(val, (u.unyt_array, u.unyt_quantity)):
                return val
            return np.asarray(val, dtype=float) * units.lscale

        if (center is None) != (width is None):
            raise ValueError("`center` and `width` must be given together.")

        if center is not None:
            center = _as_length(center)
            half = _as_length(width) / 2
            # broadcast a scalar half-width to all three axes
            half = half * np.ones(3) if np.ndim(half.value) == 0 else half
            lo = center - half
            hi = center + half
            box = [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]

        if box is not None:
            x0, y0, z0, x1, y1, z1 = (_as_length(b) for b in box)
            sel &= (x >= x0) & (x <= x1)
            sel &= (y >= y0) & (y <= y1)
            sel &= (z >= z0) & (z <= z1)
            clip_box = u.unyt_array([x0, y0, z0, x1, y1, z1])

        if mask is not None:
            mask = np.asarray(mask)
            if mask.dtype == bool:
                sel &= mask
            else:  # integer indices
                idx_mask = np.zeros(len(x), dtype=bool)
                idx_mask[mask] = True
                sel &= idx_mask

        if box is None and mask is None:
            raise ValueError("No region specified: pass `box`, `center`+`width`, or `mask`.")

        # When no explicit box was given, frame the clip on the bounding box of
        # the selected cells so projections/slices default to the region.
        if clip_box is None:
            if sel.any():
                xs, ys, zs = x[sel], y[sel], z[sel]
                clip_box = u.unyt_array(
                    [xs.min(), ys.min(), zs.min(), xs.max(), ys.max(), zs.max()]
                )
            else:
                clip_box = self.box

        return ClippedSnapshot(self, sel, clip_box)


class SnapshotH5(Snapshot):
    """RICH snapshot backed by a single HDF5 file.

    Supports multi-rank HDF5 files where particle data is stored under
    ``rank0/``, ``rank1/``, … groups.  Single-rank files (no rank groups)
    are also supported — fields are read from the root.

    :param path: Path to the ``.h5`` or ``.hdf5`` snapshot file.

    Attributes
    ----------
    rank : int
        Number of MPI ranks detected in the file (1 for single-core runs).
    f : :class:`h5py.File`
        Open HDF5 file handle for direct access to the raw data.
    """

    _field_aliases = _build_h5_aliases()

    # Reverse mapping for quick lookup, build only once
    _alias_to_canonical = {}

    def __init__(self, path):
        self.path = path
        self.rank = self._get_rank()  # after setting path
        self.f = h5py.File(self.path, "r")  # allow easy access to the h5py file

        super().__init__(path)  # inherit all methods from parent class

    def _get_rank(self) -> int:
        """Detect the number of MPI ranks stored in the HDF5 file.

        Counts ``rank<N>`` top-level groups and returns the total count.
        Returns ``1`` for single-core files that have no rank groups.

        :returns: Number of ranks (≥ 1).
        :rtype: int
        """
        with h5py.File(self.path, "r") as f:
            maxrank = 0
            for key in f.keys():
                if "rank" in key:
                    rank = int(key[4:])
                    if maxrank < rank:
                        maxrank = rank  # get the max rank

        maxrank += 1  # number of rank (starts from 1) is max rank (starts from 0) + 1
        return maxrank

    def __getitem__(self, key) -> u.unyt_array:
        """Return a field from the snapshot as a unit-bearing array.

        Concatenates data across all MPI ranks.  Aliases are resolved to their
        canonical names before the HDF5 lookup.  For fields stored at the root
        level (e.g. ``Box``, ``Time``), the rank-based path is skipped and the
        root dataset is read directly.

        Supports optional slicing::

            snap['density']          # full array
            snap['density', 1:10]    # rows 1–9
            snap['density', ::-1]    # reversed

        :param key: Field name (or alias), or a tuple ``(field, slice)``.
        :returns: Field data with physical units attached.
        :rtype: :class:`unyt.unyt_array`
        :raises KeyError: If the field is not found in the HDF5 file.
        """
        # Parse key and slice
        if isinstance(key, tuple):
            field, idx = key[0], key[1]
        else:
            field, idx = key, slice(None)

        # Resolve alias to canonical name (class method, no instance data)
        field = self._resolve_field_name(field)

        with h5py.File(self.path, "r") as f:
            try:
                arr = np.concatenate([f[f"rank{i}/{field}"] for i in range(self.rank)])
            except KeyError:  # If field is not under rank, try on the root order
                arr = f[field][()]
            try: 
                arr *= units.get_unit(field)
            except ValueError:
                warnings.warn(f"Field {field} not recognized. Passing without unit.")
                pass

        if np.ndim(arr) == 0:
            return arr
        return arr[idx]

    def __len__(self) -> int:
        """Return the total number of cells across all MPI ranks.

        Uses the length of the ``X`` coordinate dataset as a proxy for the
        cell count.

        :returns: Total cell count.
        :rtype: int
        :raises Exception: If neither ``X`` nor ``rank0/X`` is found in the
                           file.
        """
        with h5py.File(self.path, "r") as f:
            try:
                n = np.sum([len(f[f"rank{i}/X"]) for i in range(self.rank)])
            except KeyError:
                try:
                    n = len(f["X"])
                except KeyError:
                    raise Exception("Failed to get length. Neither field X nor rank0/X exists.")

        return n

    def keys(self) -> list:
        """Return a sorted list of all dataset names in the HDF5 file.

        Recurses through all groups, strips the ``rank<N>/`` prefix from paths,
        and de-duplicates so that each field appears once regardless of how many
        ranks are present.

        :returns: Sorted list of canonical field names.
        :rtype: list[str]
        """

        def _list_group(f, keys, prefix="") -> list:

            # recursively list all datasets
            for key in list(f.keys()):
                if prefix == "":
                    full_key = key
                else:
                    full_key = prefix + "/" + key

                if isinstance(f[key], h5py._hl.dataset.Dataset):
                    key_norank = re.sub(
                        r"rank\d+/", "", full_key
                    )  # match and remove the 'rank<number>/' prefix
                    if key_norank not in keys:
                        keys.append(key_norank)
                elif isinstance(f[key], h5py._hl.group.Group):
                    _list_group(f[key], keys, prefix=full_key)

            return 0

        keys = []
        with h5py.File(self.path, "r") as f:
            _list_group(f, keys=keys)
        keys.sort()

        return keys


class SnapshotNPY(Snapshot):
    """RICH snapshot backed by a directory of per-field NumPy files.

    Each field is stored as ``<FieldName>_<snapnum>.npy`` (or ``.txt``).
    The snapshot number is extracted from :attr:`path` via the ``snap_<N>``
    pattern and used to locate the correct files.

    :param path: Path to the directory containing ``.npy`` / ``.txt`` files.
    """

    _field_aliases = _build_npy_aliases()

    # Reverse mapping for quick lookup, build only once
    _alias_to_canonical = {}

    def __init__(self, path):
        self.path = path

        super().__init__(path)

    def keys(self) -> list:
        """Return a sorted list of field names available in the directory.

        Scans for files ending in ``.npy`` or ``.txt``, strips the
        ``_<snapnum>`` suffix to recover the field name stem, and
        de-duplicates.

        :returns: Sorted list of field name stems.
        :rtype: list[str]
        """
        keys = []
        files = os.listdir(
            self.path
        )  # files should look like Mass_30.npy, 30 being the snapshot number
        for i in range(len(files)):
            if files[i].endswith("txt") or files[i].endswith("npy"):
                _ = files[i].find("_")
                keys.append(files[i][:_])
            else:
                continue

        keys.sort()

        return keys

    def __getitem__(self, key) -> u.unyt_array:
        """Return a field from the snapshot directory as a unit-bearing array.

        Resolves aliases, then loads ``<FieldName>_<snapnum>.npy`` (or
        ``.txt`` as a fallback) using memory-mapping for efficiency.

        Supports optional slicing::

            snap['density']          # full array
            snap['density', 1:10]    # rows 1–9

        :param key: Field name (or alias), or a tuple ``(field, slice)``.
        :returns: Field data with physical units attached.
        :rtype: :class:`unyt.unyt_array`
        :raises FileNotFoundError: If neither ``.npy`` nor ``.txt`` file is
                                   found for the requested field.
        """
        # Parse key and slice
        if isinstance(key, tuple):
            field, idx = key[0], key[1]
        else:
            field, idx = key, slice(None)

        # Resolve alias to canonical name (class method, no instance data)
        field = self._resolve_field_name(field)

        filename = os.path.join(
            self.path, field + f"_{self.snapnum}.npy"
        )  # format of extractor.py
        if os.path.isfile(filename):
            arr = np.load(filename, mmap_mode="r")  # try .npy
        else:  # if not .npy try .txt
            filename = os.path.join(
                self.path, field + f"_{self.snapnum}.txt"
            )  # format of extractor.py
            if os.path.isfile(filename):
                arr = np.loadtxt(filename)
            else:
                raise FileNotFoundError(
                    f"File {filename} is not found. Key '{key}' does not exist."
                )

        if idx != slice(None):
            arr = arr[idx]

        try: 
            arr *= units.get_unit(field)
        except ValueError:
            warnings.warn(f"Field {field} not recognized. Passing without unit.")
            pass

        return arr

    def __len__(self) -> int:
        """Return the number of cells by reading the length of the first field.

        :returns: Cell count.
        :rtype: int
        """
        for key in self.keys():
            length = len(self[key])
            break
        return length


class ClippedSnapshot(Snapshot):
    """A lazy regional view of a :class:`Snapshot` (a "sub-snapshot").

    Created via :meth:`Snapshot.clip`.  Holds only a reference to the parent
    snapshot, a boolean selection mask over the parent's cells, and the clip
    box — **no field data is copied**.  Each field access re-reads the parent
    field from disk and applies the mask, so the clip behaves as a drop-in
    :class:`Snapshot`: :meth:`project`, :meth:`slice`, the plotter, and the
    shock finder all work on it unchanged, returning per-cell results of length
    ``len(clip)``.

    Because :class:`SnapshotH5` already concatenates data across MPI ranks
    before returning a field, the mask is a single flat array in the parent's
    absolute index space — ranks need no special handling here.

    :param parent: The snapshot being clipped.
    :param mask: Boolean array of shape ``(len(parent),)`` selecting cells.
    :param box: Six-element ``[x0, y0, z0, x1, y1, z1]`` clip bounds returned by
                this clip's ``box`` field.

    Attributes
    ----------
    parent : :class:`Snapshot`
        The snapshot this clip is a view of.
    mask : :class:`numpy.ndarray`
        Boolean selection mask over the parent's cells.
    """

    def __init__(self, parent: "Snapshot", mask: np.ndarray, box: u.unyt_array):
        self.parent = parent
        self.mask = np.asarray(mask, dtype=bool)
        self._box = box
        self.path = parent.path
        self.snapnum = parent.snapnum
        self.plots = SnapshotPlotter(self)  # plotter rebound to this clip
        self._field_aliases = parent._field_aliases  # for info()/_field_info

    def _resolve_field_name(self, key: str) -> str:
        """Resolve *key* to a canonical field name via the parent's alias table."""
        return self.parent._resolve_field_name(key)

    def __getitem__(self, key) -> u.unyt_array:
        """Return a field restricted to the clipped region.

        Per-cell fields (length equal to the parent's cell count) are indexed
        with the selection mask; the ``Box`` field returns the clip box; all
        other metadata (e.g. scalar ``Time``/``Cycle``) passes through
        unchanged.

        :param key: Field name (or alias), or a tuple ``(field, slice)``.
        :returns: Masked field data with physical units attached.
        :rtype: :class:`unyt.unyt_array`
        """
        if isinstance(key, tuple):
            field, idx = key[0], key[1]
        else:
            field, idx = key, slice(None)

        canon = self._resolve_field_name(field)

        if canon == "Box":
            return self._box[idx]

        arr = self.parent[field]
        if np.ndim(arr) > 0 and len(arr) == len(self.parent):
            return arr[self.mask][idx]
        return arr  # scalar / metadata field: pass through unmasked

    def __len__(self) -> int:
        """Return the number of cells selected by the clip mask."""
        return int(self.mask.sum())

    def keys(self) -> list:
        """Return the parent snapshot's field names (the clip exposes the same fields)."""
        return self.parent.keys()


def _parse_plane(plane, x, y, z):
    """Permute ``(x, y, z)`` so that the first two axes match *plane*.

    Given a two-character plane string such as ``"xy"``, ``"yz"``, or ``"zx"``
    (any ordering), returns ``(axis1, axis2, normal_axis)`` where *axis1* and
    *axis2* span the slicing plane and *normal_axis* is the orthogonal
    direction to be sliced through.

    :param plane: Two-character string specifying the slice plane, e.g.
                  ``"xy"``, ``"yz"``, ``"zx"``, ``"yx"``, etc.
    :param x: x-data (scalar, array, or unyt_array).
    :param y: y-data.
    :param z: z-data.
    :returns: Tuple ``(axis1, axis2, normal)`` corresponding to the requested
              plane and its normal.
    :rtype: tuple
    :raises Exception: If *plane* contains characters other than ``'x'``,
                       ``'y'``, ``'z'``.
    """

    def _parse_xyz(char, x, y, z):
        if char == "x":
            return x
        elif char == "y":
            return y
        elif char == "z":
            return z

    x1 = _parse_xyz(plane[0], x, y, z)
    x2 = _parse_xyz(plane[1], x, y, z)

    if (x1 is x and x2 is y) or (x1 is y and x2 is x):
        x3 = z
    elif (x1 is x and x2 is z) or (x1 is z and x2 is x):
        x3 = y
    elif (x1 is y and x2 is z) or (x1 is z and x2 is y):
        x3 = x
    else:
        raise Exception(f"Plane {plane} is unrecognizable.")

    return x1, x2, x3


def _kdtree_interpolate(coords, grid_coords, k=1, eps=0, workers=1):
    """Nearest-neighbour interpolation using a k-d tree.

    Builds a :class:`scipy.spatial.KDTree` from *coords* and queries it at
    every point in *grid_coords*, returning the index of the nearest source
    point for each query.

    :param coords: Source point coordinates, shape ``(N, 3)``.
    :param grid_coords: Query point coordinates, shape ``(nx, ny[, nz], 3)``
                        or ``(M, 3)``.
    :param k: Number of nearest neighbours to find.  Defaults to ``1``.
    :param eps: Approximate search tolerance passed to
                :meth:`scipy.spatial.KDTree.query`.  Defaults to ``0``
                (exact).
    :param workers: Number of parallel workers for the query.  Defaults to
                    ``1``.
    :returns: Index array of nearest-source indices, same leading shape as
              *grid_coords* (minus the last coordinate dimension).
    :rtype: :class:`numpy.ndarray`
    """
    from scipy.spatial import KDTree

    kdtree = KDTree(coords)  # build tree
    d, i = kdtree.query(
        grid_coords, k=k, eps=eps, p=2, workers=workers
    )  # the most time-consuming step

    return i
