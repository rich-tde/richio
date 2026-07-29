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

"""Rosseland, Planck and scattering opacities from the STA tables.

These are plain functions of temperature and density rather than snapshot
fields, so the same call works on a snapshot, on a clipped subset, or on arrays
from anywhere else::

    import richio
    alpha = richio.opacity.rosseland_alpha(snap.T, snap.rho)

The result is a per-cell :class:`unyt.unyt_array`, which every richio entry
point that takes ``data`` already accepts — so slices, projections and volume
renders need no extra plumbing::

    snap.plots.slice(data=alpha, res=512)
    snap.project(alpha, res=512)

**alpha vs opacity.** The tables store ``ln(sigma)`` where ``sigma`` is used
directly as a **1/length extinction coefficient** in RICH's radiative transfer
(see ``source/Radiation/STAgreyOpacity.cpp``) — it is *already* the product
``kappa * rho``, not a per-gram opacity.  Both forms are therefore provided:

* ``*_alpha``   -> extinction coefficient alpha [1/cm], the thing you integrate
  along a ray for an optical depth, ``tau = int alpha dr``.
* ``*_opacity`` -> specific opacity ``kappa = alpha / rho`` [cm**2/g], the
  textbook quantity.

The interpolation kernels are a vectorised port of the same table lookup RICH
performs internally (bilinear in ``ln T`` / ``ln rho``, with linear
extrapolation off the edges of the grid).
"""

from dataclasses import dataclass
import functools
from importlib.resources import files
import os
import warnings

import numpy as np
import unyt as u

#: Environment variable overriding which table directory is used.
TABLE_DIR_ENV = "RICHIO_OPACITY_DIR"

#: Table files expected in a table directory, in load order.
_TABLE_FILES = ("T.txt", "rho.txt", "ross.txt", "planck.txt", "scatter.txt")


@dataclass(frozen=True)
class OpacityTable:
    """The STA grey opacity table, as natural logs on a ``ln T`` x ``ln rho`` grid.

    :param ln_T: ``ln(T[K])`` grid axis, shape ``(nT,)``.
    :param ln_rho: ``ln(rho[g/cm**3])`` grid axis, shape ``(nrho,)``.
    :param ln_ross: ``ln(sigma_Rosseland[1/cm])``, shape ``(nT, nrho)``.
    :param ln_planck: ``ln(sigma_Planck[1/cm])``, shape ``(nT, nrho)``.
    :param ln_scatter: ``ln(sigma_scattering[1/cm])``, shape ``(nT, nrho)``.
    :param path: Directory the table was read from.
    """

    ln_T: np.ndarray
    ln_rho: np.ndarray
    ln_ross: np.ndarray
    ln_planck: np.ndarray
    ln_scatter: np.ndarray
    path: str = ""


def default_table_dir() -> str:
    """Return the table directory used when none is given.

    Resolution order: the :data:`TABLE_DIR_ENV` environment variable, else the
    copy of the STA tables bundled with richio.

    :returns: Path to a directory containing the STA ``.txt`` tables.
    :rtype: str
    """
    env = os.environ.get(TABLE_DIR_ENV)
    if env:
        return env
    # NB: "tables", not "data" -- a richio/data/ directory would collide with
    # the richio/data.py module and could shadow it.
    return str(files("richio") / "tables" / "opacity" / "sta")


@functools.lru_cache(maxsize=4)
def load_opacity_table(table_dir: str | None = None) -> OpacityTable:
    """Load (and cache) the STA opacity table.

    :param table_dir: Directory holding ``T.txt``, ``rho.txt``, ``ross.txt``,
                      ``planck.txt`` and ``scatter.txt``.  Defaults to
                      :func:`default_table_dir`.
    :returns: The loaded table.
    :rtype: :class:`OpacityTable`
    :raises FileNotFoundError: If a table file is missing.
    """
    path = table_dir or default_table_dir()
    arrays = []
    for name in _TABLE_FILES:
        fname = os.path.join(path, name)
        if not os.path.exists(fname):
            raise FileNotFoundError(
                f"Opacity table file {fname!r} not found. Point {TABLE_DIR_ENV} at a "
                f"directory containing {', '.join(_TABLE_FILES)}."
            )
        arrays.append(np.loadtxt(fname))
    ln_T, ln_rho, ln_ross, ln_planck, ln_scatter = arrays
    return OpacityTable(ln_T, ln_rho, ln_ross, ln_planck, ln_scatter, path=str(path))


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def _to_cgs(value, unit: str, name: str) -> np.ndarray:
    """Return *value* as a plain float64 array in cgs *unit*.

    The tables are tabulated in cgs (K and g/cm**3), so the conversion is
    required, not cosmetic: richio's ``snap.density`` is in **code units**, and
    feeding those in raw would silently look up the wrong part of the table.
    Letting ``unyt`` convert also validates the dimensions for free.

    :param value: A :class:`unyt.unyt_array` (converted) or bare array
                  (assumed to be cgs already, with a warning).
    :param unit: Target cgs unit string, e.g. ``"K"``.
    :param name: Argument name, used in error messages.
    :returns: Values in *unit*, as ``float64``.
    :rtype: :class:`numpy.ndarray`
    :raises ValueError: If *value* carries units of the wrong dimension.
    """
    if isinstance(value, u.unyt_array):
        try:
            return np.asarray(value.to(unit), dtype=np.float64)
        except Exception as exc:  # unyt raises UnitConversionError / etc.
            raise ValueError(
                f"{name} has units {value.units!s}, which cannot be converted to "
                f"{unit!r}. Check the argument order — the signature is "
                f"(temperature, density)."
            ) from exc

    warnings.warn(
        f"{name} has no units attached; assuming it is already in {unit!r}. "
        f"Pass a unyt array (e.g. snap.T, snap.rho) to have this checked.",
        stacklevel=3,
    )
    return np.asarray(value, dtype=np.float64)


def _prepare(temperature, density, table):
    """Validate/convert inputs and return ``(table, ln_T, ln_rho, rho_cgs)``."""
    if table is None:
        table = load_opacity_table()
    temperature_cgs = _to_cgs(temperature, "K", "temperature")
    density_cgs = _to_cgs(density, "g/cm**3", "density")
    if temperature_cgs.shape != density_cgs.shape:
        raise ValueError(
            f"temperature and density must have the same shape, got "
            f"{temperature_cgs.shape} and {density_cgs.shape}."
        )
    return table, np.log(temperature_cgs), np.log(density_cgs), density_cgs


# ---------------------------------------------------------------------------
# Interpolation kernels (vectorised port of RICH's table lookup)
# ---------------------------------------------------------------------------

def bilinear_interpolation_vectorized(x_vec, y_vec, data, x_arr, y_arr):
    """Bilinear interpolation on a rectilinear grid, clamped to its edges.

    :param x_vec: Monotonic x grid, shape ``(nx,)``.
    :param y_vec: Monotonic y grid, shape ``(ny,)``.
    :param data: Values with ``data[i, j]`` at ``(x_vec[i], y_vec[j])``.
    :param x_arr: Query x values, shape ``(N,)``.
    :param y_arr: Query y values, shape ``(N,)``.
    :returns: Interpolated values, shape ``(N,)``.
    :rtype: :class:`numpy.ndarray`
    """
    x_vec = np.asarray(x_vec)
    y_vec = np.asarray(y_vec)
    data = np.asarray(data)

    x_arr = np.asarray(x_arr, dtype=np.float64)
    y_arr = np.asarray(y_arr, dtype=np.float64)

    x_arr = np.clip(x_arr, x_vec[0], x_vec[-1])
    y_arr = np.clip(y_arr, y_vec[0], y_vec[-1])

    i = np.searchsorted(x_vec, x_arr, side="right") - 1
    j = np.searchsorted(y_vec, y_arr, side="right") - 1

    i = np.clip(i, 0, len(x_vec) - 2).astype(np.intp)
    j = np.clip(j, 0, len(y_vec) - 2).astype(np.intp)

    x0, x1 = x_vec[i], x_vec[i + 1]
    y0, y1 = y_vec[j], y_vec[j + 1]

    tx = (x_arr - x0) / (x1 - x0)
    ty = (y_arr - y0) / (y1 - y0)

    d00 = data[i, j]
    d10 = data[i + 1, j]
    d01 = data[i, j + 1]
    d11 = data[i + 1, j + 1]

    return (
        d00 * (1 - tx) * (1 - ty)
        + d10 * tx * (1 - ty)
        + d01 * (1 - tx) * ty
        + d11 * tx * ty
    )


def interpolate_2d_table_vectorized(
    x_vec, y_vec, data, x_arr, y_arr, x_vec_high_slope=0.0, slope_length=7
):
    """Interpolate a log table, extrapolating linearly beyond its edges.

    *data* holds natural logs, so the return value is exponentiated back to
    linear space.  Points outside the grid are extrapolated with a slope
    estimated over the first *slope_length* nodes.

    :param x_vec: ``ln`` x grid (temperature axis).
    :param y_vec: ``ln`` y grid (density axis).
    :param data: ``ln`` values, ``data[i, j]`` at ``(x_vec[i], y_vec[j])``.
    :param x_arr: Query ``ln`` x values, shape ``(N,)``.
    :param y_arr: Query ``ln`` y values, shape ``(N,)``.
    :param x_vec_high_slope: Slope used above the top of the x grid.
    :param slope_length: Number of nodes used to estimate edge slopes.
    :returns: ``(values, x_slope, y_slope)``, each shape ``(N,)``.
    :rtype: tuple
    """
    x_vec = np.asarray(x_vec, dtype=np.float64)
    y_vec = np.asarray(y_vec, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)

    x_arr = np.asarray(x_arr, dtype=np.float64)
    y_arr = np.asarray(y_arr, dtype=np.float64)

    n = len(x_arr)
    interp_val = np.empty(n, dtype=np.float64)
    x_slope = np.empty(n, dtype=np.float64)
    y_slope = np.empty(n, dtype=np.float64)

    mask_x_low = x_arr < x_vec[0]
    mask_x_high = x_arr > x_vec[-1]
    mask_x_mid = ~mask_x_low & ~mask_x_high
    mask_y_low = y_arr < y_vec[0]

    # x below grid, y below grid
    mask = mask_x_low & mask_y_low
    if np.any(mask):
        x_slope[mask] = (data[slope_length - 1, 0] - data[0, 0]) / (
            x_vec[slope_length - 1] - x_vec[0]
        )
        y_slope[mask] = (data[0, slope_length - 1] - data[0, 0]) / (
            y_vec[slope_length - 1] - y_vec[0]
        )
        base = (
            data[0, 0]
            + y_slope[mask] * (y_arr[mask] - y_vec[0])
            + x_slope[mask] * (x_arr[mask] - x_vec[0])
        )
        interp_val[mask] = np.exp(base)

    # x below grid, y inside
    mask = mask_x_low & ~mask_y_low
    if np.any(mask):
        x0 = x_vec[0] * 1.00001
        data_x0 = bilinear_interpolation_vectorized(
            x_vec, y_vec, data, np.full_like(y_arr[mask], x0), y_arr[mask]
        )
        x_high = x_vec[slope_length - 1]
        data_xhigh = bilinear_interpolation_vectorized(
            x_vec, y_vec, data, np.full_like(y_arr[mask], x_high), y_arr[mask]
        )
        x_slope[mask] = (data_xhigh - data_x0) / (x_vec[slope_length - 1] - x_vec[0])
        interp_val[mask] = np.exp(data_x0 + x_slope[mask] * (x_arr[mask] - x_vec[0]))
        y_slope[mask] = 0.0

    # x above grid, y below grid
    mask = mask_x_high & mask_y_low
    if np.any(mask):
        y_slope[mask] = (data[-1, slope_length - 1] - data[-1, 0]) / (
            y_vec[slope_length - 1] - y_vec[0]
        )
        base = (
            data[-1, 0]
            + y_slope[mask] * (y_arr[mask] - y_vec[0])
            + x_vec_high_slope * (x_arr[mask] - x_vec[-1])
        )
        interp_val[mask] = np.exp(base)
        x_slope[mask] = x_vec_high_slope

    # x above grid, y inside
    mask = mask_x_high & ~mask_y_low
    if np.any(mask):
        x_near = x_vec[-1] * 0.99999
        base = bilinear_interpolation_vectorized(
            x_vec, y_vec, data, np.full_like(y_arr[mask], x_near), y_arr[mask]
        )
        interp_val[mask] = np.exp(base + x_vec_high_slope * (x_arr[mask] - x_vec[-1]))
        x_slope[mask] = x_vec_high_slope
        y_slope[mask] = 0.0

    # x inside, y below grid
    mask = mask_x_mid & mask_y_low
    if np.any(mask):
        y0 = y_vec[0] * 0.9999
        data_y0 = bilinear_interpolation_vectorized(
            x_vec, y_vec, data, x_arr[mask], np.full_like(x_arr[mask], y0)
        )
        y_high = y_vec[slope_length - 1]
        data_yhigh = bilinear_interpolation_vectorized(
            x_vec, y_vec, data, x_arr[mask], np.full_like(x_arr[mask], y_high)
        )
        y_slope[mask] = (data_yhigh - data_y0) / (y_vec[slope_length - 1] - y_vec[0])
        interp_val[mask] = np.exp(data_y0 + y_slope[mask] * (y_arr[mask] - y_vec[0]))
        x_slope[mask] = 0.0

    # fully inside the grid
    mask = mask_x_mid & ~mask_y_low
    if np.any(mask):
        interp_val[mask] = np.exp(
            bilinear_interpolation_vectorized(x_vec, y_vec, data, x_arr[mask], y_arr[mask])
        )
        x_slope[mask] = 0.0
        y_slope[mask] = 0.0

    return interp_val, x_slope, y_slope


def calc_scattering_opacity_vectorized(
    T_, rho_, scatter_, Tcell_arr, rhocell_arr, return_coeff=False
):
    """Scattering extinction coefficient from the table, in ``ln`` inputs.

    Outside the tabulated density range the coefficient is rescaled linearly
    with density (scattering is ``propto rho`` at fixed composition).

    :param T_: ``ln T`` grid axis.
    :param rho_: ``ln rho`` grid axis.
    :param scatter_: ``ln sigma_scattering`` table.
    :param Tcell_arr: Query ``ln T`` values.
    :param rhocell_arr: Query ``ln rho`` values.
    :param return_coeff: Also return the ``(T, rho)`` extrapolation slopes.
    :returns: ``sigma`` in 1/cm, or ``(sigma, T_slope, d_slope)``.
    """
    T_ = np.asarray(T_, dtype=np.float64)
    rho_ = np.asarray(rho_, dtype=np.float64)
    scatter_ = np.asarray(scatter_, dtype=np.float64)

    Tcell_arr = np.asarray(Tcell_arr, dtype=np.float64)
    rhocell_arr = np.asarray(rhocell_arr, dtype=np.float64)

    d_log = rhocell_arr.copy()
    d_ratio = np.ones_like(rhocell_arr)

    mask_low = rhocell_arr < rho_[0]
    mask_high = rhocell_arr > rho_[-1]

    rho_min, rho_max = rho_[0], rho_[-1]
    d_ratio[mask_low] = np.exp(rhocell_arr[mask_low]) / np.exp(rho_min)
    d_log[mask_low] = rho_min
    d_ratio[mask_high] = np.exp(rhocell_arr[mask_high]) / np.exp(rho_max)
    d_log[mask_high] = rho_max

    interp_val, T_slope, d_slope = interpolate_2d_table_vectorized(
        T_, rho_, scatter_, Tcell_arr, d_log
    )
    scatter = interp_val * d_ratio

    if return_coeff:
        # The density slope from the interpolator does not apply: rho was
        # shifted onto the table edge and rescaled by hand above.
        d_slope[mask_low] = 1
        d_slope[mask_high] = 1
        return scatter, T_slope, d_slope
    return scatter


def calc_ross_opacity_vectorized(
    T_, rho_, rossland_, scatter_, Tcell_arr, rhocell_arr, return_coeff=False
):
    """Rosseland extinction coefficient from the table, in ``ln`` inputs.

    Below the tabulated density range the Rosseland table is unreliable, so the
    smaller of it and the scattering coefficient is used.

    :param T_: ``ln T`` grid axis.
    :param rho_: ``ln rho`` grid axis.
    :param rossland_: ``ln sigma_Rosseland`` table.
    :param scatter_: ``ln sigma_scattering`` table (low-density fallback).
    :param Tcell_arr: Query ``ln T`` values.
    :param rhocell_arr: Query ``ln rho`` values.
    :param return_coeff: Also return the ``(T, rho)`` extrapolation slopes.
    :returns: ``sigma`` in 1/cm, or ``(sigma, T_slope, d_slope)``.
    """
    T_ = np.asarray(T_, dtype=np.float64)
    rho_ = np.asarray(rho_, dtype=np.float64)

    Tcell_arr = np.asarray(Tcell_arr, dtype=np.float64)
    rhocell_arr = np.asarray(rhocell_arr, dtype=np.float64)

    d_log = rhocell_arr.copy()
    d_ratio = np.ones_like(rhocell_arr)

    mask_low = rhocell_arr < rho_[0]
    mask_high = rhocell_arr > rho_[-1]

    rho_max = rho_[-1]
    d_log[mask_high] = rho_max
    d_ratio[mask_high] = np.exp(rhocell_arr[mask_high]) / np.exp(rho_max)

    interp_val, T_slope, d_slope = interpolate_2d_table_vectorized(
        T_, rho_, rossland_, Tcell_arr, d_log
    )
    rossland = interp_val * d_ratio

    if np.any(mask_low):
        scattering, Tscatt_slope, dscatt_slope = calc_scattering_opacity_vectorized(
            T_, rho_, scatter_, Tcell_arr[mask_low], rhocell_arr[mask_low], return_coeff=True
        )
        use_scatt = ~(rossland[mask_low] > scattering)

        rossland[mask_low] = np.where(use_scatt, scattering, rossland[mask_low])
        T_slope[mask_low] = np.where(use_scatt, Tscatt_slope, T_slope[mask_low])
        d_slope[mask_low] = np.where(use_scatt, dscatt_slope, d_slope[mask_low])

        if return_coeff:
            return rossland, T_slope, d_slope

    if return_coeff:
        return rossland, T_slope, d_slope
    return rossland


def calc_planck_opacity_vectorized(
    T_, rho_, planck_, Tcell_arr, rhocell_arr, return_coeff=False
):
    """Planck extinction coefficient from the table, in ``ln`` inputs.

    Off the low-density edge the coefficient is rescaled by ``(rho/rho_min)``
    raised to a slope read from the table, which captures the transition from
    scattering-like to absorption-like (``propto rho**2``) behaviour.

    :param T_: ``ln T`` grid axis.
    :param rho_: ``ln rho`` grid axis.
    :param planck_: ``ln sigma_Planck`` table.
    :param Tcell_arr: Query ``ln T`` values.
    :param rhocell_arr: Query ``ln rho`` values.
    :param return_coeff: Also return the ``(T, rho)`` extrapolation slopes.
    :returns: ``sigma`` in 1/cm, or ``(sigma, T_slope, d_slope)``.
    """
    T_ = np.asarray(T_, dtype=np.float64)
    rho_ = np.asarray(rho_, dtype=np.float64)
    planck_ = np.asarray(planck_, dtype=np.float64)

    Tcell_arr = np.asarray(Tcell_arr, dtype=np.float64)
    rhocell_arr = np.asarray(rhocell_arr, dtype=np.float64)

    d_ratio = np.ones_like(rhocell_arr)
    d_slope = np.full_like(rhocell_arr, 2.0)
    d_log = rhocell_arr.copy()

    mask_low = rhocell_arr < rho_[0]
    mask_high = rhocell_arr > rho_[-1]
    mask_mid = ~mask_low & ~mask_high

    d_slope[mask_mid] = 0.0

    if np.any(mask_low):
        mask_in_T = mask_low & (T_[0] < Tcell_arr) & (Tcell_arr < T_[-1])
        if np.any(mask_in_T):
            idx = np.searchsorted(T_, Tcell_arr[mask_in_T])
            d_slope[mask_in_T] = (planck_[idx, 10] - planck_[idx, 0]) / (rho_[10] - rho_[0])
        d_ratio[mask_low] = np.exp(rhocell_arr[mask_low]) / np.exp(rho_[0])
        d_log[mask_low] = rho_[0]

    if np.any(mask_high):
        d_ratio[mask_high] = np.exp(rhocell_arr[mask_high]) / np.exp(rho_[-1])
        d_log[mask_high] = rho_[-1]

    interp_val, T_slope, d_slope_out = interpolate_2d_table_vectorized(
        T_, rho_, planck_, Tcell_arr, d_log, x_vec_high_slope=-3.5
    )
    planck = interp_val * (d_ratio**d_slope)

    if return_coeff:
        return planck, T_slope, d_slope
    return planck


# ---------------------------------------------------------------------------
# Public API — extinction coefficients [1/cm]
# ---------------------------------------------------------------------------

def rosseland_alpha(temperature, density, *, table=None) -> u.unyt_array:
    """Rosseland extinction coefficient ``alpha`` [1/cm].

    This is the quantity to integrate along a ray for a Rosseland optical
    depth, ``tau = int alpha dr``.  Divide by density (or use
    :func:`rosseland_opacity`) for the specific opacity in cm**2/g.

    :param temperature: Per-cell temperature (any unit of temperature; a bare
                        array is assumed to be Kelvin).
    :param density: Per-cell mass density (any density unit, e.g. richio's
                    code units; a bare array is assumed to be g/cm**3).
    :param table: An :class:`OpacityTable`; defaults to the bundled table.
    :returns: ``alpha`` with units of 1/cm.
    :rtype: :class:`unyt.unyt_array`
    """
    table, ln_T, ln_rho, _ = _prepare(temperature, density, table)
    sigma = calc_ross_opacity_vectorized(
        table.ln_T, table.ln_rho, table.ln_ross, table.ln_scatter, ln_T, ln_rho
    )
    return u.unyt_array(sigma, "cm**-1")


def planck_alpha(temperature, density, *, table=None) -> u.unyt_array:
    """Planck extinction coefficient ``alpha`` [1/cm].

    :param temperature: Per-cell temperature (bare array assumed Kelvin).
    :param density: Per-cell mass density (bare array assumed g/cm**3).
    :param table: An :class:`OpacityTable`; defaults to the bundled table.
    :returns: ``alpha`` with units of 1/cm.
    :rtype: :class:`unyt.unyt_array`
    """
    table, ln_T, ln_rho, _ = _prepare(temperature, density, table)
    sigma = calc_planck_opacity_vectorized(
        table.ln_T, table.ln_rho, table.ln_planck, ln_T, ln_rho
    )
    return u.unyt_array(sigma, "cm**-1")


def scattering_alpha(temperature, density, *, table=None) -> u.unyt_array:
    """Scattering extinction coefficient ``alpha`` [1/cm].

    :param temperature: Per-cell temperature (bare array assumed Kelvin).
    :param density: Per-cell mass density (bare array assumed g/cm**3).
    :param table: An :class:`OpacityTable`; defaults to the bundled table.
    :returns: ``alpha`` with units of 1/cm.
    :rtype: :class:`unyt.unyt_array`
    """
    table, ln_T, ln_rho, _ = _prepare(temperature, density, table)
    sigma = calc_scattering_opacity_vectorized(
        table.ln_T, table.ln_rho, table.ln_scatter, ln_T, ln_rho
    )
    return u.unyt_array(sigma, "cm**-1")


# ---------------------------------------------------------------------------
# Public API — specific opacities [cm**2/g]
# ---------------------------------------------------------------------------

def rosseland_opacity(temperature, density, *, table=None) -> u.unyt_array:
    """Rosseland mean opacity ``kappa = alpha / rho`` [cm**2/g].

    :param temperature: Per-cell temperature (bare array assumed Kelvin).
    :param density: Per-cell mass density (bare array assumed g/cm**3).
    :param table: An :class:`OpacityTable`; defaults to the bundled table.
    :returns: ``kappa`` with units of cm**2/g.
    :rtype: :class:`unyt.unyt_array`
    """
    table, ln_T, ln_rho, rho_cgs = _prepare(temperature, density, table)
    sigma = calc_ross_opacity_vectorized(
        table.ln_T, table.ln_rho, table.ln_ross, table.ln_scatter, ln_T, ln_rho
    )
    return u.unyt_array(sigma / rho_cgs, "cm**2/g")


def planck_opacity(temperature, density, *, table=None) -> u.unyt_array:
    """Planck mean opacity ``kappa = alpha / rho`` [cm**2/g].

    :param temperature: Per-cell temperature (bare array assumed Kelvin).
    :param density: Per-cell mass density (bare array assumed g/cm**3).
    :param table: An :class:`OpacityTable`; defaults to the bundled table.
    :returns: ``kappa`` with units of cm**2/g.
    :rtype: :class:`unyt.unyt_array`
    """
    table, ln_T, ln_rho, rho_cgs = _prepare(temperature, density, table)
    sigma = calc_planck_opacity_vectorized(
        table.ln_T, table.ln_rho, table.ln_planck, ln_T, ln_rho
    )
    return u.unyt_array(sigma / rho_cgs, "cm**2/g")


def scattering_opacity(temperature, density, *, table=None) -> u.unyt_array:
    """Scattering opacity ``kappa = alpha / rho`` [cm**2/g].

    :param temperature: Per-cell temperature (bare array assumed Kelvin).
    :param density: Per-cell mass density (bare array assumed g/cm**3).
    :param table: An :class:`OpacityTable`; defaults to the bundled table.
    :returns: ``kappa`` with units of cm**2/g.
    :rtype: :class:`unyt.unyt_array`
    """
    table, ln_T, ln_rho, rho_cgs = _prepare(temperature, density, table)
    sigma = calc_scattering_opacity_vectorized(
        table.ln_T, table.ln_rho, table.ln_scatter, ln_T, ln_rho
    )
    return u.unyt_array(sigma / rho_cgs, "cm**2/g")
