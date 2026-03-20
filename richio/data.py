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
import unyt as u
from numpy.typing import ArrayLike
from rich.console import Console
from rich.table import Table

from richio.config import FIELD_REGISTRY
from richio.plots import SnapshotPlotter
from richio.units import units


def _build_h5_aliases():
    """Build _field_aliases for SnapshotH5 from the central registry."""
    return {
        k: v["aliases"]
        for k, v in FIELD_REGISTRY.items()
        if not v.get("npy_only", False)
    }


def _build_npy_aliases():
    """Build _field_aliases for SnapshotNPY from the central registry."""
    result = {}
    for k, v in FIELD_REGISTRY.items():
        npy_key = v.get("npy_name", k)  # use npy_name if present, else h5 key
        result[npy_key] = v["aliases"]
    return result


def load(path):
    if os.path.isfile(path) and (path.endswith("h5") or path.endswith("hdf5")):  # if hdf5 file
        with h5py.File(path) as f:
            pass
        return SnapshotH5(path)
    elif os.path.isdir(path):  # if a directory
        return SnapshotNPY(path)
    else:
        raise FileNotFoundError(f"{path} is neither a directory nor a file.")


class Snapshot:
    def __init__(self, path: str):
        self.path = path
        self.plots = SnapshotPlotter(self)  # initialise a plotter object
        self.snapnum = self._get_snapnum()  # snapshot number

        # Build class mapping once (idempotent)
        if not self.__class__._alias_to_canonical:
            self._build_alias_mapping()

    def _get_snapnum(self):
        """
        Try to read the snapshot number from the path. Matches `snap_<num>`
        pattern across the path and returns the first math. Set to -1 if no
        match is found.
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
        """Build once per class, not per instance."""
        for canonical, aliases in cls._field_aliases.items():
            cls._alias_to_canonical[canonical] = canonical
            for alias in aliases:
                cls._alias_to_canonical[alias] = canonical

    def __getattr__(self, name):
        """Allow attribute-style access: snap.density"""

        try:
            return self[name]
        except FileNotFoundError:
            raise AttributeError(f"Field '{name}' not found")

    def _resolve_field_name(self, key: str) -> str:
        """Read from shared class dict."""
        return self.__class__._alias_to_canonical.get(key, key)

    def mask_star_ratio(self) -> np.ndarray:
        """
        Get mask for star particles only (tracers/Star = 1).
        """
        return np.abs(self.star - 1) < 1e-3

    def mask_density(self) -> np.ndarray:
        """
        Get mask for floor density gas.
        """
        return self.density > 1e-19 * units.get_unit("Density")

    @property
    def _field_info(self):
        """
        Dictionary mapping field names to metadata.

        For recognised fields the ``unit`` and ``aliases`` keys are populated.
        For unrecognised fields (not in the central registry) both values are
        ``None`` / empty — they are still listed so users know the field exists.

        Returns
        -------
        dict
            ``{field: {"unit": <unit or None>, "aliases": [...]}}``
        """
        info = {}
        for field in self.keys():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                unit = units.get_unit(field, default=None)
            info[field] = {
                "unit":    unit,
                "aliases": self._field_aliases.get(field, []),
            }
        return info

    def info(self, unit_system="rich", show_aliases=True) -> None:
        """
        Display snapshot information and available fields.

        Parameters
        ----------
        unit_system : str, optional
            Unit system: 'rich' (default), 'cgs', or 'mks'
        show_aliases : bool, optional
            Show field aliases (default: True)

        Examples
        --------
        >>> snap.info()
        >>> snap.info(unit_system='cgs')
        >>> snap.info(show_aliases=False)
        """
        console = Console()

        # ---- Metadata table ------------------------------------------------
        meta_table = Table(show_header=False, box=None, padding=(0, 1))
        meta_table.add_column(style="bold cyan",  no_wrap=True)
        meta_table.add_column(style="white")

        meta_table.add_row("Path",            str(self.path))
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
        field_table.add_column("Field",   style="cyan",  no_wrap=True)
        field_table.add_column("Unit",    style="green")
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


    def _get_data(
        self, data: str | ArrayLike
        ) -> u.unyt_array:
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
    ):
        """
        Calculate a quantity, interpolate on grid, and integrate along z axis.
        To make use of the unit system, use either str keys or unyt_array data
        for `data`, `X`, `Y`, `Z`, `box_size`.
        """
        grid_data, i, xspace, yspace, zspace = self.to_grid(data, res, 
                X, Y, Z, box_size, selection)

        dz = zspace[1:] - zspace[:-1]                                                 #PM: dz = (z1 - z0) / (nz - 1)
        projected_data = np.sum(grid_data[:-1, :-1, :-1] * dz, axis=-1).in_base(unit_system)#PM: grid_data[:, :, :-1]

        return projected_data, xspace, yspace



    def to_grid(
        self,
        data: str | ArrayLike,
        res: int | ArrayLike,
        X: str | ArrayLike = "X",
        Y: str | ArrayLike = "Y",
        Z: str | ArrayLike = "Z",
        box_size: ArrayLike | None = None,
        selection: ArrayLike = None,
        endpoint: bool = False,
    ):
        """
        Interpolate to a fixed grid.
        """
        # Fetch data
        data = self._get_data(data)
        X = self._get_data(X)
        Y = self._get_data(Y)
        Z = self._get_data(Z)

        # Select cells
        if selection is not None:
            data = data[selection]
            X = X[selection]
            Y = Y[selection]
            Z = Z[selection]

        # Set boxsize
        if box_size is None:
            x0, y0, z0, x1, y1, z1 = self.box  # Load the box size
        else:
            x0, y0, z0, x1, y1, z1 = box_size

            # Assign default unit if not provided
            for l in [x0, y0, z0, x1, y1, z1]:
                if isinstance(l, u.unyt_quantity):
                    continue
                else:
                    l = l * units.lscale

        # Set resolution
        try:
            nx, ny, nz = res[0], res[1], res[2]
        except TypeError:
            nx = ny = nz = res

        # Make Euclidean grid
        xspace = np.linspace(x0, x1, nx, endpoint=endpoint)  # disable endpoints by default such that dz = (z1-z0)/res instead of (z1-z0)/(res-1)
        yspace = np.linspace(y0, y1, ny, endpoint=endpoint)  #PM: endpoint=True
        zspace = np.linspace(z0, z1, nz, endpoint=endpoint)  # TODO: add an option to use np.geomspace

        grid_x, grid_y, grid_z = np.meshgrid(xspace, yspace, zspace, indexing="ij")

        coords = np.stack([X, Y, Z], axis=-1)  # coordinates of the particles
        grid_coords = np.stack([grid_x, grid_y, grid_z], axis=-1)  # coordinates of the grid (query points)

        i = _kdtree_interpolate(coords=coords, grid_coords=grid_coords)

        grid_data = data[i]

        return grid_data, i, xspace, yspace, zspace



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
        volume_selection: bool = True, # select based on volume to speed up calculation
    ):
        """Make a slice of the simulation grid. Outputs the intepolated data and
        the indices.

        :param data: The data colume to interpolate, can be the name of the
            column ("density") or an array of size (N,) (snap.density)
        :type data: str | ArrayLike
        :param res: Resolution, can be one integer for a square grid (256 means a
            256x256 grid) or a tuple/array (X, Y) for an irregular grid
        :type res: int | ArrayLike
        :param X: The x coordinates of mesh generating points, can be name of
            the column or an array, defaults to "X"
        :type X: str | ArrayLike, optional
        :param Y: The y coordinates of mesh generating points, can be name of
            the column or an array, defaults to "Y"
        :type Y: str | ArrayLike, optional
        :param Z: The z coordinates of mesh generating points, can be name of
            the column or an array, defaults to "Z"
        :type Z: str | ArrayLike, optional
        :param plane: "xy", "yz", "xz", or in any other order, like "yx",
            defaults to "xy"
        :type plane: str, optional
        :param slice_coord: The coordinates of the orthogonal axis to the
            slicing plane, defaults to 0
        :type slice_coord: float | u.array.unyt_quantity | None, optional
        :param box_size: An array of (x_lower, y_lower, z_lower, x_upper,
            y_upper, z_upper), or (x1_lower, x2_lower, x1_upper, x2_upper)
            depending on the 'plane' parameter. Will read from the box_size of
            the snapshot if not given, defaults to None
        :type box_size: ArrayLike | None, optional
        :param selection: A bool array of size (N,) which filters what particles
            in the simulation are plotted, defaults to None
        :type selection: ArrayLike | None, optional
        :param unit_system: The data will be converted to given 'unyt' unit
            system, can be "cgs", "rich"(code solar units) or other unyt units,
            defaults to "cgs"
        :type unit_system: str, optional
        :param volume_selection: If turned on, only cells within a distance of
            volume**(1/3) to the slicing plane will be used, which speeds up the
            computation a lot (without losing much accuracy), defaults to True
        :type volume_selection: bool, optional
        :return: The interpolated data on a grid data, linear space of the x
            grid, linear space of the y grid
        :rtype: (unyt.unyt_array, unyt.unyt_array, unyt.unyt_array)
        """
        # TODO: implement star_mask : put data to zero instead of removing them
        # (which is bad if you use nn) and put them to the lowest color in
        # colormap when plotting (something like set_bad...)

        # Fetch data
        data = self._get_data(data)
        X = self._get_data(X)
        Y = self._get_data(Y)
        Z = self._get_data(Z)

        if volume_selection:
            volume = self._get_data('volume')

        if selection is not None:
            data = data[selection]
            X = X[selection]
            Y = Y[selection]
            Z = Z[selection]
            if volume_selection:
                volume = volume[selection]

        # Set resolution
        try:
            nx, ny = res[0], res[1]
        except TypeError:
            nx = ny = res

        # Set boxsize
        if box_size is None:
            x0, y0, z0, x1, y1, z1 = self.box  # Load the box size
            x0, y0, z0 = _parse_plane(plane, x0, y0, z0)
            x1, y1, z1 = _parse_plane(plane, x1, y1, z1)
        else:
            if isinstance(box_size, u.unyt_array):
                pass
            else:
                box_size *= units.lscale

            if len(box_size) == 6:
                x0, y0, z0, x1, y1, z1 = box_size       # A 3d box
                x0, y0, z0 = _parse_plane(plane, x0, y0, z0)
                x1, y1, z1 = _parse_plane(plane, x1, y1, z1)
            elif len(box_size) == 4:
                x0, y0, x1, y1 = box_size

        # Assign code unit if slice_coord doesn't have a unit
        if isinstance(slice_coord, u.unyt_quantity):
            pass
        else:
            slice_coord *= units.lscale

        # x_slice, y_slice, z_slice should only have one that is not None
        X, Y, Z = _parse_plane(plane, X, Y, Z)      # redefine x y to be the plane, z the sliced direction

        # Select only cells in proximity
        if volume_selection:
            mask = np.abs(Z - slice_coord) < volume**(1/3)
            # assuming spherical cells, V^(1/3)=(4pi/3)^(1/3)R ~ 1.6R, we don't
            # include the factor such that if V is not round enough we won't
            # lose too much accuracy
            data = data[mask]
            X = X[mask]
            Y = Y[mask]
            Z = Z[mask]


        # Make Euclidean grid
        xspace = np.linspace(x0, x1, nx, endpoint=False)
        yspace = np.linspace(y0, y1, ny, endpoint=False)
        zspace = slice_coord

        grid_x, grid_y, grid_z = np.meshgrid(xspace, yspace, zspace, indexing="ij")

        coords = np.stack([X, Y, Z], axis=-1)  # coordinates of the particles 
        grid_coords = np.stack([grid_x, grid_y, grid_z], axis=-1)  # coordinates of the grid (query points)
        grid_coords = np.squeeze(grid_coords)            # remove extra dimension (nx, ny, 1, 3) to (nx, ny, 3)

        i = _kdtree_interpolate(coords=coords, grid_coords=grid_coords)

        sliced_data = data[i]
        sliced_data = sliced_data.in_base(unit_system)

        return sliced_data, xspace, yspace


class SnapshotH5(Snapshot):
    _field_aliases = _build_h5_aliases()

    # Reverse mapping for quick lookup, build only once
    _alias_to_canonical = {}

    def __init__(self, path):
        self.path = path
        self.rank = self._get_rank()        # after setting path
        self.f = h5py.File(self.path, "r")  # allow easy access to the h5py file

        super().__init__(path)  # inherit all methods from parent class

    def _get_rank(self) -> int:
        """
        Get the number of ranks.
        
        If ran on single core this should return 1, as no 'rank' group is found
        """
        with h5py.File(self.path, "r") as f:
            maxrank = 0
            for key in f.keys():
                if "rank" in key:
                    rank = int(key[4:])
                    if maxrank < rank:
                        maxrank = rank # get the max rank

        maxrank += 1  # number of rank (starts from 1) is max rank (starts from 0) + 1
        return maxrank

    def __getitem__(self, key) -> u.unyt_array:
        """
        Get numpy array of the desired quantity, combining different ranks.
        Supports slicing: obj['field'] or obj['field', 1:10] or obj['field', ::-1]
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
                arr *= units.get_unit(field)
            except KeyError:  # If field is not under rank, try on the root order
                arr = f[field][:] * units.get_unit(field)

        return arr[idx]

    def __len__(self) -> int:
        """
        Number of particles of the snapshot, combining different ranks.
        Use the X coordinate to get the number.
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

    def keys(self) -> list:  # Update this to recursively list all datasets
        """
        List all keys that a dataset have, omit the "rank0" prefix.
        """
        def _list_group(f, keys, prefix="") -> list:

            # recursively list all datasets
            for key in list(f.keys()):

                if prefix == "":
                    full_key = key
                else:
                    full_key = prefix + "/" + key

                if isinstance(f[key], h5py._hl.dataset.Dataset):
                    key_norank = re.sub(r"rank\d+/", "", full_key) # match and remove the 'rank<number>/' prefix
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
    """
    Loading Paola's .npy directories.
    """

    _field_aliases = _build_npy_aliases()

    # Reverse mapping for quick lookup, build only once
    _alias_to_canonical = {}

    def __init__(self, path):
        self.path = path  # TODO: rewrite with Path?

        super().__init__(path)

    def keys(self) -> list:  # TODO: rewrite this with re?
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
        """
        Get numpy array of the desired quantity, combining different ranks.
        Supports slicing: obj['field'] or obj['field', 1:10] or obj['field', ::-1]
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

        if idx == slice(None):  # do not slice if no index indicated
            return arr * units.get_unit(
                field
            )  # slice[idx] works generally so long as arr is not 0-dimensional
        else:  # which is the case for snap.time
            return arr[idx] * units.get_unit(field)

    def __len__(self) -> int:
        for key in self.keys():
            length = len(self[key])
            break
        return length




def _parse_plane(plane, x, y, z):
    """
    Parse a string input "xy" to data x, y, z; "yz" to y, z, x; "zx" to z, x, y,
    etc, in order to specify the slicing plane.
    """

    def _parse_xyz(char, x, y, z):
        if char == 'x':
            return x
        elif char == 'y':
            return y
        elif char == 'z':
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

    from scipy.spatial import KDTree

    kdtree = KDTree(coords)  # build tree
    d, i = kdtree.query(
        grid_coords, k=k, eps=eps, p=2, workers=workers
    )  # the most time-consuming step

    return i