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


class SnapshotPlotter:
    """High-level plotting interface bound to a :class:`~richio.data.Snapshot`.

    An instance is automatically created and attached as ``snap.plots`` when a
    snapshot is loaded; you should not instantiate this class directly.

    :param snap: The parent snapshot object.
    :type snap: :class:`~richio.data.Snapshot`
    """

    def __init__(self, snap):
        self.snap = snap

    def peek(self, data="density", **kwargs):
        """Produce a quick mid-plane xy slice of *data* at 512² resolution.

        Tries ``X``, ``Y``, ``Z`` coordinates first; falls back to centre-of-mass
        coordinates ``CMx``, ``CMy``, ``CMz`` if the positional fields are absent.

        :param data: Field to visualise.  Defaults to ``'density'``.
        :type data: str
        :param kwargs: Extra keyword arguments forwarded to :meth:`slice`.
        :returns: Tuple ``(ax, im, sliced_data)`` — see :meth:`slice`.
        :rtype: tuple
        """
        try:
            return self.slice(
                data=data, res=512, X="X", Y="Y", Z="Z", plane="xy", slice_coord=0, **kwargs
            )
        except FileNotFoundError:  # X, Y, Z are not found, do cmx, cmy, cmz (center of mass)
            return self.slice(
                data=data, res=512, X="CMx", Y="CMy", Z="CMz", plane="xy", slice_coord=0, **kwargs
            )

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
        volume_selection: bool = True,  # select based on volume to speed up calculation
        ax: Any | None = None,
        cmap: str | Colormap = "twilight",
        label_latex: str = "\\rho",
        unit_latex: str | None = None,
        aspect_equal: bool = True,
        **kwargs,
    ):
        """Compute a mid-plane slice and render it as a ``pcolormesh`` plot.

        Delegates the data interpolation to
        :meth:`~richio.data.Snapshot.slice` and renders log₁₀ of the result
        using :func:`matplotlib.pyplot.pcolormesh`.  A colorbar labelled with
        the LaTeX field symbol and unit is added automatically.

        :param data: Field to plot — name string or array of shape ``(N,)``.
        :type data: str or ArrayLike
        :param res: Grid resolution — integer (square) or ``(nx, ny)`` tuple.
        :type res: int or ArrayLike
        :param X: x-coordinate field name or array. Defaults to ``"X"``.
        :type X: str or ArrayLike
        :param Y: y-coordinate field name or array. Defaults to ``"Y"``.
        :type Y: str or ArrayLike
        :param Z: z-coordinate field name or array. Defaults to ``"Z"``.
        :type Z: str or ArrayLike
        :param plane: Slice plane, e.g. ``"xy"`` (default), ``"yz"``, ``"zx"``.
        :type plane: str
        :param slice_coord: Coordinate along the normal axis at which to
                            slice.  Defaults to ``0``.
        :type slice_coord: float or :class:`unyt.unyt_quantity`
        :param box_size: Domain bounds for the plot area.  Auto-detected from
                         the snapshot when ``None``.
        :type box_size: ArrayLike or None
        :param selection: Boolean cell mask. Defaults to ``None`` (all cells).
        :type selection: ArrayLike or None
        :param unit_system: Output unit system (``'cgs'``, ``'rich'``, etc.).
                            Defaults to ``'cgs'``.
        :type unit_system: str
        :param volume_selection: Pre-filter cells to within one cell-size of
                                 the slice plane to speed up computation.
                                 Defaults to ``True``.
        :type volume_selection: bool
        :param ax: Existing :class:`matplotlib.axes.Axes` to draw on.  A new
                   figure is created when ``None``.
        :type ax: :class:`matplotlib.axes.Axes` or None
        :param cmap: Colormap name or object.  Defaults to ``'twilight'``.
        :type cmap: str or :class:`matplotlib.colors.Colormap`
        :param label_latex: LaTeX symbol for the colorbar label.
                            Defaults to ``r'\\rho'``.
        :type label_latex: str
        :param unit_latex: LaTeX unit string for the colorbar.  Auto-read from
                           the data when ``None``.
        :type unit_latex: str or None
        :param aspect_equal: Set equal aspect ratio on the axes.
                             Defaults to ``True``.
        :type aspect_equal: bool
        :param kwargs: Additional keyword arguments forwarded to
                       :func:`~matplotlib.pyplot.pcolormesh` (e.g. ``vmin``,
                       ``vmax``).
        :returns: Tuple ``(ax, im, sliced_data)`` — the axes, the
                  :class:`~matplotlib.collections.QuadMesh`, and the
                  interpolated data array.
        :rtype: tuple
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
        )

        if ax is None:
            fig, ax = plt.subplots()

        # Plot
        xx, yy = np.meshgrid(xspace, yspace, indexing="ij")
        im = ax.pcolormesh(xx, yy, np.log10(sliced_data), cmap=cmap, **kwargs)

        if unit_latex is None:  # read the unit from data if not specified
            unit_latex = sliced_data.units.latex_repr

        plt.colorbar(im, ax=ax, label=f"$\\log[{label_latex}/{unit_latex}]$")

        if aspect_equal:
            ax.set_aspect("equal", adjustable="box")

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
        ax: Any | None = None,
        cmap: str | Colormap = "twilight",
        label_latex: str = "\\Sigma",  # TODO: make them automatic from data name
        unit_latex: str | None = None,
        aspect_equal: bool = True,
        **kwargs,
    ):
        """Compute a column-integrated projection and render it as a plot.

        Delegates integration to :meth:`~richio.data.Snapshot.project` and
        rendering to :func:`scalar_map`.  Pass field name strings or
        :class:`unyt.unyt_array` objects for *data*, *X*, *Y*, *Z*, and
        *box_size* to benefit from automatic unit handling.

        :param data: Field to project — name string or array of shape ``(N,)``.
        :type data: str or ArrayLike
        :param res: Grid resolution — integer or ``(nx, ny, nz)`` tuple.
        :type res: int or ArrayLike
        :param X: x-coordinates. Defaults to ``"X"``.
        :type X: str or ArrayLike
        :param Y: y-coordinates. Defaults to ``"Y"``.
        :type Y: str or ArrayLike
        :param Z: z-coordinates (integration axis). Defaults to ``"Z"``.
        :type Z: str or ArrayLike
        :param box_size: Domain bounds. Auto-detected when ``None``.
        :type box_size: ArrayLike or None
        :param unit_system: Output unit system. Defaults to ``'cgs'``.
        :type unit_system: str
        :param selection: Boolean cell mask. Defaults to ``None``.
        :type selection: ArrayLike or None
        :param ax: Existing axes to draw on; new figure created when ``None``.
        :type ax: :class:`matplotlib.axes.Axes` or None
        :param cmap: Colormap. Defaults to ``'twilight'``.
        :type cmap: str or :class:`matplotlib.colors.Colormap`
        :param label_latex: LaTeX symbol for colorbar label.
                            Defaults to ``r'\\Sigma'``.
        :type label_latex: str
        :param unit_latex: LaTeX unit string for colorbar.  Auto-read when
                           ``None``.
        :type unit_latex: str or None
        :param aspect_equal: Set equal aspect ratio. Defaults to ``True``.
        :type aspect_equal: bool
        :param kwargs: Extra keyword arguments forwarded to
                       :func:`~matplotlib.pyplot.pcolormesh`.
        :returns: Tuple ``(ax, im, projected_data)``.
        :rtype: tuple
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
        )

        ax, im = scalar_map(
            f=projected_data,
            xspace=xspace,
            yspace=yspace,
            ax=ax,
            cmap=cmap,
            label_latex=label_latex,
            unit_latex=unit_latex,
            aspect_equal=aspect_equal,
            **kwargs,
        )

        return ax, im, projected_data


def scalar_map(
    f: u.unyt_array | ArrayLike,
    xspace: u.unyt_array | ArrayLike,
    yspace: u.unyt_array | ArrayLike,
    ax: Any | None = None,
    cmap: str | Colormap = "twilight",
    label_latex: str = "\\Sigma",
    unit_latex: str | None = None,
    aspect_equal: bool = True,
    **kwargs,
):
    """Render a 2-D scalar field on a regular grid as a log-scale colour map.

    Computes ``log₁₀(f)``, auto-selects ``vmin``/``vmax`` rounded to the
    nearest half-integer, and plots using :func:`~matplotlib.pyplot.pcolormesh`.
    A colorbar labelled ``$\\log[symbol/unit]$`` is added automatically.

    :param f: 2-D scalar data of shape ``(nx, ny)``.  A :class:`unyt.unyt_array`
              is recommended for automatic unit labelling.
    :type f: :class:`unyt.unyt_array` or ArrayLike
    :param xspace: 1-D array of x-coordinates (length ``nx``).
    :type xspace: :class:`unyt.unyt_array` or ArrayLike
    :param yspace: 1-D array of y-coordinates (length ``ny``).
    :type yspace: :class:`unyt.unyt_array` or ArrayLike
    :param ax: Existing axes to draw on; a new figure is created when ``None``.
    :type ax: :class:`matplotlib.axes.Axes` or None
    :param cmap: Colormap.  Defaults to ``'twilight'``.
    :type cmap: str or :class:`matplotlib.colors.Colormap`
    :param label_latex: LaTeX symbol for the colorbar (e.g. ``r'\\Sigma'``).
    :type label_latex: str
    :param unit_latex: LaTeX unit string.  Auto-read from ``f.units`` when
                       ``None``.
    :type unit_latex: str or None
    :param aspect_equal: Set equal aspect ratio on the axes.
                         Defaults to ``True``.
    :type aspect_equal: bool
    :param kwargs: Additional keyword arguments forwarded to
                   :func:`~matplotlib.pyplot.pcolormesh` (e.g. ``vmin``,
                   ``vmax`` to override auto-scaling).
    :returns: Tuple ``(ax, im)`` — the axes and the
              :class:`~matplotlib.collections.QuadMesh`.
    :rtype: tuple[:class:`matplotlib.axes.Axes`,
                  :class:`matplotlib.collections.QuadMesh`]
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

        if "vmin" not in kw:
            kw["vmin"] = vmin_default
        if "vmax" not in kw:
            kw["vmax"] = vmax_default
    else:
        warnings.warn("No finite values found in data; leaving vmin/vmax to matplotlib defaults.")

    xgrid, ygrid = np.meshgrid(xspace, yspace, indexing="ij")
    im = ax.pcolormesh(
        xgrid, ygrid, data_log, cmap=cmap, **kw
    )  # return im as well in case you want to customise colorbar

    if unit_latex is None:  # read the unit from data if not specified
        unit_latex = f.units.latex_repr  # TODO: check dimensionality and raise warning

    plt.colorbar(im, ax=ax, label=f"$\\log[{label_latex}/{unit_latex}]$")

    if aspect_equal:
        plt.gca().set_aspect("equal", adjustable="box")

    return ax, im
