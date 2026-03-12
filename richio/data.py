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

from richio.plots import SnapshotPlotter
from richio.units import units


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

        Returns
        -------
        dict
            Dictionary with field names as keys and metadata as values

        Examples
        --------
        >>> info = snap._field_info
        >>> print(info['Density'])
        {'unit': g/cm³, 'aliases': ['Den', 'density', 'rho']}
        """
        info = {}

        for field in self.keys():
            info[field] = {
                "unit": units.get_unit(field),
                "aliases": self._field_aliases[field],
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
        # Header
        print("=" * 100)
        print(f"RICH SNAPSHOT INFORMATION".center(100))
        print("=" * 100)

        # Metadata
        print(f"\n{'Snapshot Details':<40}")
        print("-" * 100)

        meta_info = [
            ("Path", self.path),
            ("Snapshot Number", self.snapnum),
        ]

        # Add optional metadata
        for attr, label in [("time", "Time"), ("box", "Box size"), ("cycle", "Cycle")]:
            try:
                val = getattr(self, attr)
                meta_info.append((label, val.in_base(unit_system)))
            except:
                pass

        try:
            meta_info.append(("Number of Cells", f"{len(self):,}"))
        except:
            pass

        if hasattr(self, "rank"):
            meta_info.append(("Number of Ranks", self.rank))

        for label, value in meta_info:
            print(f"  {label:<25} : {value}")

        # Fields
        print(f"\n{'Available Fields':<40} [Unit System: {unit_system.upper()}]")
        print("-" * 100)

        # Header row
        if show_aliases:
            print(f"{'Field':<15} {'Unit':<40} {'Aliases'}")
        else:
            print(f"{'Field':<15} {'Unit'}")
        print("-" * 100)

        # Field information
        field_info_dict = self._field_info

        for field in self.keys():
            if field not in field_info_dict:
                continue

            info = field_info_dict[field]

            # Format unit
            if unit_system == "rich":
                unit_str = str(info["unit"])
            else:
                unit_str = str((1 * info["unit"]).in_base(unit_system))

            if show_aliases:
                aliases = info.get("aliases", [])
                alias_str = ", ".join(aliases) if aliases else "-"
                print(f"{field:<15} {unit_str:<40} {alias_str}")
            else:
                print(f"{field:<15} {unit_str}")

        # Footer
        print("-" * 100)
        print(f"Total: {len(self.keys())} fields")

        print("=" * 100)


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
        method: str = "nn",
    ):
        """
        Calculate a quantity, interpolate on grid, and integrate along z axis.
        To make use of the unit system, use either str keys or unyt_array data
        for `data`, `X`, `Y`, `Z`, `box_size`.

        Parameters
        ----------
        method : {'nn', 'voronoi'}
            'nn'      – nearest-neighbour interpolation on a 3-D grid then
                        explicit z-integration (default, existing behaviour).
            'voronoi' – analytical 2-D Voronoi projection: each cell's 3-D
                        volume is distributed over the pixels it covers,
                        weighted by the fraction of its 2-D Voronoi polygon
                        (projected onto the XY plane) that overlaps each
                        pixel.  Mass-conservative and free of the z-resolution
                        artefacts of the NN method.
        """
        if method == "nn":
            grid_data, i, xspace, yspace, zspace = self.to_grid(data, res,
                    X, Y, Z, box_size, selection)
            dz = zspace[1:] - zspace[:-1]
            projected_data = np.sum(grid_data[:-1, :-1, :-1] * dz, axis=-1).in_base(unit_system)
            return projected_data, xspace, yspace

        elif method == "voronoi":
            from richio.voronoi_viz import _voronoi_project

            # --- data fetching (mirrors to_grid but also needs volume) ---
            data_arr = self._get_data(data)
            Xarr     = self._get_data(X)
            Yarr     = self._get_data(Y)
            vol_arr  = self._get_data("volume")

            if selection is not None:
                data_arr = data_arr[selection]
                Xarr     = Xarr[selection]
                Yarr     = Yarr[selection]
                vol_arr  = vol_arr[selection]

            # Resolution
            try:
                nx, ny = int(res[0]), int(res[1])
            except TypeError:
                nx = ny = int(res)

            # Box size
            if box_size is None:
                x0, y0, z0, x1, y1, z1 = self.box
            else:
                if not isinstance(box_size, u.unyt_array):
                    box_size = box_size * units.lscale
                x0, y0, z0, x1, y1, z1 = box_size

            # Convert to CGS scalars for the numba backend
            x0f = float(x0.in_base("cgs").v);  x1f = float(x1.in_base("cgs").v)
            y0f = float(y0.in_base("cgs").v);  y1f = float(y1.in_base("cgs").v)
            Xf  = np.asarray(Xarr.in_base("cgs"),    dtype=np.float64)
            Yf  = np.asarray(Yarr.in_base("cgs"),    dtype=np.float64)
            df  = np.asarray(data_arr.in_base("cgs"), dtype=np.float64)
            vf  = np.asarray(vol_arr.in_base("cgs"),  dtype=np.float64)

            result_np = _voronoi_project(df, vf, Xf, Yf, x0f, x1f, y0f, y1f, nx, ny)

            # Units: (data CGS unit) * cm³ / cm² = (data CGS unit) * cm
            data_cgs_unit = data_arr.in_base("cgs").units
            projected_data = (result_np * data_cgs_unit * u.cm).in_base(unit_system)

            xspace = np.linspace(x0, x1, nx, endpoint=False)
            yspace = np.linspace(y0, y1, ny, endpoint=False)
            return projected_data, xspace, yspace

        else:
            raise ValueError(f"project: unknown method '{method}'. Use 'nn' or 'voronoi'.")



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
        method: str = "nn",
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
        :param method: Interpolation method.  ``'nn'`` (default) uses a
            KD-tree nearest-neighbour lookup.  ``'voronoi'`` builds the 2-D
            Voronoi tessellation of the cells near the slice plane and assigns
            each pixel the area-weighted mean of all Voronoi cells that
            overlap it – exact polygon-pixel intersection, no aliasing.
        :type method: str, optional
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
            volume = volume[mask]


        # Make Euclidean grid
        xspace = np.linspace(x0, x1, nx, endpoint=False)
        yspace = np.linspace(y0, y1, ny, endpoint=False)
        zspace = slice_coord

        if method == "nn":
            grid_x, grid_y, grid_z = np.meshgrid(xspace, yspace, zspace, indexing="ij")
            coords      = np.stack([X, Y, Z], axis=-1)
            grid_coords = np.squeeze(np.stack([grid_x, grid_y, grid_z], axis=-1))
            i           = _kdtree_interpolate(coords=coords, grid_coords=grid_coords)
            sliced_data = data[i].in_base(unit_system)
            return sliced_data, xspace, yspace

        elif method == "voronoi":
            from richio.voronoi_viz import _voronoi_slice

            # Convert to CGS scalars for the numba backend
            x0f = float(x0.in_base("cgs").v);  x1f = float(x1.in_base("cgs").v)
            y0f = float(y0.in_base("cgs").v);  y1f = float(y1.in_base("cgs").v)
            Xf  = np.asarray(X.in_base("cgs"),    dtype=np.float64)
            Yf  = np.asarray(Y.in_base("cgs"),    dtype=np.float64)
            df  = np.asarray(data.in_base("cgs"), dtype=np.float64)
            vf  = (np.asarray(volume.in_base("cgs"), dtype=np.float64)
                   if volume_selection else None)

            result_np   = _voronoi_slice(df, Xf, Yf, x0f, x1f, y0f, y1f, nx, ny,
                                         vol_f=vf)
            data_cgs_unit = data.in_base("cgs").units
            sliced_data = (result_np * data_cgs_unit).in_base(unit_system)
            return sliced_data, xspace, yspace

        else:
            raise ValueError(f"slice: unknown method '{method}'. Use 'nn' or 'voronoi'.")


class SnapshotH5(
    Snapshot
):
    _field_aliases = {
        # Positions
        "X": ["position_x", "pos_x", "x", "particle_position_x"],
        "Y": ["position_y", "pos_y", "y", "particle_position_y"],
        "Z": ["position_z", "pos_z", "z", "particle_position_z"],
        # Center of mass
        "CMx": ["cm_x", "center_of_mass_x"],
        "CMy": ["cm_y", "center_of_mass_y"],
        "CMz": ["cm_z", "center_of_mass_z"],
        # Velocities
        "Vx": ["velocity_x", "vx", "vel_x", "particle_velocity_x"],
        "Vy": ["velocity_y", "vy", "vel_y", "particle_velocity_y"],
        "Vz": ["velocity_z", "vz", "vel_z", "particle_velocity_z"],
        "divV": ["velocity_divergence", "DivV", "div_v", "divergence"],
        # Thermodynamic
        "Density": [
            "density",
            "densitites",
            "Den",
            "rho",
        ],  # swiftsimio: 'densities', yt: 'density'
        "Pressure": ["pressure", "P"],
        "Temperature": ["temperature", "T", "temp"],
        "InternalEnergy": ["internal_energy", "IE", "specific_internal_energy", "sie"],
        "tracers/Entropy": ["entropy", "Entropy", "S"],
        "Dissipation": ["dissipation", "Diss", "dissipation_rate"],
        # Volume
        "Volume": ["volume", "Vol", "volumes"],
        # Radiation
        "Erad": ["radiation_energy", "Rad", "E_rad", "Erad"],
        # Gradients - Pressure
        "DpDx": ["pressure_gradient_x", "grad_p_x", "dp_dx"],
        "DpDy": ["pressure_gradient_y", "grad_p_y", "dp_dy"],
        "DpDz": ["pressure_gradient_z", "grad_p_z", "dp_dz"],
        # Gradients - Density
        "DrhoDx": ["density_gradient_x", "grad_rho_x", "drho_dx"],
        "DrhoDy": ["density_gradient_y", "grad_rho_y", "drho_dy"],
        "DrhoDz": ["density_gradient_z", "grad_rho_z", "drho_dz"],
        # Gradients - Internal Energy
        "DsieDx": ["sie_gradient_x", "grad_sie_x", "dsie_dx"],
        "DsieDy": ["sie_gradient_y", "grad_sie_y", "dsie_dy"],
        "DsieDz": ["sie_gradient_z", "grad_sie_z", "dsie_dz"],
        # The percentage of material that comes from star
        "tracers/Star": ["star_fraction", "Star", "star", "star_ratio", "stellar_fraction"],
        # Metadata
        "Box": ["box_size", "box", "boxsize"],
        "Time": ["time", "t", "tfb", "simulation_time", "current_time"],
        "Cycle": ["cycle", "step", "iteration"],
        "ID": ["particle_id", "id", "ids", "particle_ids"],
        # Unused
        "Eg_0": ["eg_0"],
        "stickers": ["stickers"],
        "tracers/WasRemoved": ["was_removed", "WasRemoved", "removed"],
    }

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

    _field_aliases = {
        # Positions
        "CMx": SnapshotH5._field_aliases["CMx"],
        "CMy": SnapshotH5._field_aliases["CMy"],
        "CMz": SnapshotH5._field_aliases["CMz"],
        # Velocities
        "Vx": SnapshotH5._field_aliases["Vx"],
        "Vy": SnapshotH5._field_aliases["Vy"],
        "Vz": SnapshotH5._field_aliases["Vz"],
        "DivV": SnapshotH5._field_aliases["divV"],
        # Mass and volume
        "Mass": [
            "mass",
            "masses",
            "particle_mass",
            "m",
        ],  # swiftsimio uses 'masses' # calculated by den*vol, not directly from output .h5
        "Vol": SnapshotH5._field_aliases["Volume"],
        # Thermodynamic
        "Den": SnapshotH5._field_aliases["Density"],
        "P": SnapshotH5._field_aliases["Pressure"],
        "T": SnapshotH5._field_aliases["Temperature"],
        "IE": SnapshotH5._field_aliases["InternalEnergy"],
        "Diss": SnapshotH5._field_aliases["Dissipation"],
        "Entropy": SnapshotH5._field_aliases["tracers/Entropy"],
        # Radiation
        "Rad": SnapshotH5._field_aliases["Erad"],
        # Gradients - Pressure
        "DpDx": SnapshotH5._field_aliases["DpDx"],
        "DpDy": SnapshotH5._field_aliases["DpDy"],
        "DpDz": SnapshotH5._field_aliases["DpDz"],
        # The star fraction
        "Star": SnapshotH5._field_aliases["tracers/Star"],
        # Metadata
        "box": SnapshotH5._field_aliases["Box"],
        "tfb": SnapshotH5._field_aliases["Time"],
    }

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