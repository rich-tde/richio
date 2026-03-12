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

from importlib.resources import files
from typing import Any
import warnings

from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import unyt as u

from richio.units import units

# from richio.config import FIGURES_DIR, PROCESSED_DATA_DIR

# import typer
# app = typer.Typer()

# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#     output_path: Path = FIGURES_DIR / "plot.png",
#     # -----------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Generating plot from data...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Plot generation complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()


def use_nice_style():
    style_path = files("richio.styles").joinpath("nice.mplstyle")
    plt.style.use(style_path)


class SnapshotPlotter:
    def __init__(self, snap):
        self.snap = snap

    def peek(self, data='density', **kwargs):
        """
        A quick peek at the data.
        """
        try:
            return self.slice(data=data, res=512, X='X', Y='Y', Z='Z', plane='xy', slice_coord=0, **kwargs)
        except FileNotFoundError:       # X, Y, Z are not found, do cmx, cmy, cmz (center of mass)
            return self.slice(data=data, res=512, X='CMx', Y='CMy', Z='CMz', plane='xy', slice_coord=0, **kwargs)

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
        method: str = "nn",
        ax: Any | None = None,
        cmap: str | Colormap = "twilight",
        label_latex: str = "\\rho",
        unit_latex: str | None = None,
        **kwargs
    ):
        """
        Make a slice plot.  Pass ``method='voronoi'`` for the analytical
        Voronoi-based slice (exact polygon-pixel intersection).
        """
        sliced_data, xspace, yspace = self.snap.slice(
            data=data,
            res=res,
            X=X,
            Y=Y,
            Z=Z,
            plane=plane,
            slice_coord=slice_coord,
            box_size=box_size,
            selection=selection,
            unit_system=unit_system,
            volume_selection=volume_selection,
            method=method,
        )

        if ax is None:
            fig, ax = plt.subplots()

        # Plot
        xx, yy = np.meshgrid(xspace, yspace, indexing="ij")
        im = ax.pcolormesh(xx, yy, np.log10(sliced_data), cmap=cmap, **kwargs)
        
        if unit_latex is None:      # read the unit from data if not specified
            unit_latex = sliced_data.units.latex_repr
        
        plt.colorbar(im, ax=ax, label=f"$\\log[{label_latex}/{unit_latex}]$")

        return ax, im, sliced_data


        

    def projection(
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
        ax: Any | None = None,
        cmap: str | Colormap = "twilight",
        label_latex: str = "\\Sigma",
        unit_latex: str | None = None,
        **kwargs,
    ):
        """
        Make a projection plot.  Pass ``method='voronoi'`` for the analytical
        Voronoi-based projection (mass-conservative column integral).
        To make use of the unit system, use either str keys or unyt_array data
        for `data`, `X`, `Y`, `Z`, `box_size`.
        """
        projected_data, xspace, yspace = self.snap.project(
            data=data,
            res=res,
            X=X,
            Y=Y,
            Z=Z,
            box_size=box_size,
            unit_system=unit_system,
            selection=selection,
            method=method,
        )

        ax, im = scalar_map(
            f=projected_data,
            xspace=xspace,
            yspace=yspace,
            ax=ax,
            cmap=cmap,
            label_latex=label_latex,
            unit_latex=unit_latex,
            **kwargs
            )

        return ax, im, projected_data





def scalar_map(f : u.unyt_array | ArrayLike, 
                xspace : u.unyt_array | ArrayLike, 
                yspace : u.unyt_array | ArrayLike,
                ax: Any | None = None,
                cmap: str | Colormap = "twilight",
                label_latex: str = "\\Sigma",
                unit_latex: str | None = None,
                **kwargs):
    """
    A general visualisation for any scalar field data.
    """
    
    # ensure we have an Axes
    if ax is None:
        fig, ax = plt.subplots()

    # compute log-space data and choose sensible defaults for vmin/vmax
    data_log = np.log10(f)

    # copy kwargs so we can set defaults without mutating caller's dict
    kw = kwargs.copy()

    # convert to ndarray for robust min/max computations
    arr = np.asarray(data_log)
    finite_mask = np.isfinite(arr)
    if finite_mask.any():
        dmin = float(np.min(arr[finite_mask]))
        dmax = float(np.max(arr[finite_mask]))

        # round to nearest half-integers outward
        vmin_default = np.floor(dmin * 2.0) / 2.0
        vmax_default = np.ceil(dmax * 2.0) / 2.0

        if 'vmin' not in kw:
            kw['vmin'] = vmin_default
        if 'vmax' not in kw:
            kw['vmax'] = vmax_default
    else:
        warnings.warn("No finite values found in data; leaving vmin/vmax to matplotlib defaults.")

    xgrid, ygrid = np.meshgrid(xspace, yspace, indexing="ij")
    im = ax.pcolormesh(xgrid, ygrid, data_log, cmap=cmap, **kw)  # return im as well in case you want to customise colorbar

    if unit_latex is None:      # read the unit from data if not specified
        unit_latex = f.units.latex_repr     #TODO: check dimensionality and raise warning

    plt.colorbar(im, ax=ax, label=f"$\\log[{label_latex}/{unit_latex}]$")

    return ax, im