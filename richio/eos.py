"""RICH's equation of state, with tables based on Tomida et al. (2013).

The interpolation, pressure floors, and analytic fallbacks reproduce RICH's
``source/newtonian/three_dimensional/OndrejEOS.cpp``. They are RICH-specific,
not a general implementation of the EOS in that paper.

Use unitful snapshot fields or independently constructed quantities::

    import richio
    pressure = richio.eos.dT2p(snap.rho, snap.T)
    snap.plots.slice(data=pressure, res=512)

All conversions accept broadcastable scalars/arrays and a keyword-only
``table`` argument from :func:`load_eos_table`. Bare inputs warn and assume
CGS. In the original conversion names, ``d`` means density [g/cm**3], ``p``
pressure [erg/cm**3], ``e`` specific internal energy [erg/g], ``T`` temperature
[K], ``s`` specific entropy [erg/(g*K)], ``c`` sound speed [cm/s], and ``cv``
**volumetric** heat capacity [erg/(cm**3*K)]. Outputs carry these units and the
RICH unit registry, so ``result.in_base('rich')`` is supported.

Tables are loaded lazily. Physical tables and fallback calculations retain
units; logarithms use dimensionless ratios to explicit CGS references. States
without a valid four-point interpolation stencil raise :class:`ValueError`
unless the requested conversion supplies an analytic fallback.
"""

from dataclasses import dataclass, field
import functools
from importlib.resources import files
import os
from pathlib import Path
import warnings

import numpy as np
import unyt as u
from unyt.exceptions import UnitConversionError

from richio.units import units

TABLE_DIR_ENV = "RICHIO_EOS_DIR"
"""Environment variable overriding the bundled EOS table directory."""


def _quantity(value, unit):
    return u.unyt_array(value, unit, registry=units.registry)


# Explicit references for dimensionless logarithms and dimensional constants.
_RHO = _quantity(1.0, "g/cm**3")
_P = _quantity(1.0, "erg/cm**3")
_E = _quantity(1.0, "erg/g")
_T = _quantity(1.0, "K")
_S = _quantity(1.0, "erg/(g*K)")
_CV = _quantity(1.0, "erg/(cm**3*K)")
_C2 = _quantity(1.0, "cm**2/s**2")
_REFS = {
    "density": _RHO,
    "pressure": _P,
    "energy": _E,
    "temperature": _T,
    "entropy": _S,
    "heat_capacity": _CV,
    "sound_speed_squared": _C2,
}
_GAS = 1.3419e8 * _S
_GAS_CV = 2.0128e8 * _S
_ENTROPY_SCALE = 10.0**8.128 * _S
_ENTROPY_SWITCH = (5.2 * units.lscale**2 / units.tscale**2).to("erg/g")
_FILES = {
    "pressure": "Pfile.txt",
    "sound_speed_squared": "csfile.txt",
    "entropy": "Sfile.txt",
    "energy": "Ufile.txt",
    "temperature": "Tfile.txt",
    "heat_capacity": "CVfile.txt",
}


def _log_ratio(value, reference):
    """Strip only a dimensionless interpolation/logarithm coordinate."""
    return np.log((value / reference).to_value("dimensionless"))


@dataclass(frozen=True)
class RichEOS:
    """Physical EOS tables based on Tomida et al. (2013), used by RICH.

    Obtain an instance with :func:`load_eos_table`. Density has shape ``(Nd,)``;
    all other quantities have shape ``(Nd, Nt)``, ordered density then
    temperature. Energy and entropy are specific quantities, heat capacity is
    volumetric, and sound speed is stored squared. Arrays are read-only so the
    cached logarithmic coordinates remain consistent with the physical tables.

    :param density: Density [g/cm**3], regularly spaced in natural logarithm.
    :param pressure: Pressure [erg/cm**3].
    :param sound_speed_squared: Squared sound speed [cm**2/s**2].
    :param entropy: Specific entropy [erg/(g*K)].
    :param energy: Specific internal energy [erg/g].
    :param temperature: Temperature [K].
    :param heat_capacity: Volumetric heat capacity [erg/(cm**3*K)].
    :param path: Source table directory, or an empty string for in-memory tables.
    """

    density: u.unyt_array
    pressure: u.unyt_array
    sound_speed_squared: u.unyt_array
    entropy: u.unyt_array
    energy: u.unyt_array
    temperature: u.unyt_array
    heat_capacity: u.unyt_array
    path: str = ""
    _logs: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        logs = {}
        for name, reference in _REFS.items():
            value = getattr(self, name)
            if not isinstance(value, u.unyt_array):
                raise ValueError(f"RichEOS.{name} must carry units")
            value = value.to(reference.units).copy()
            if not np.all(np.isfinite(value) & (value > 0 * reference)):
                raise ValueError(f"RichEOS.{name} must be finite and positive")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
            if name != "sound_speed_squared":
                logs[name] = _log_ratio(value, reference)
                logs[name].flags.writeable = False
        density = logs["density"]
        shape = self.temperature.shape
        if density.ndim != 1 or density.size < 4:
            raise ValueError("EOS density must be a one-dimensional axis with at least four rows")
        if len(shape) != 2 or shape[0] != density.size or shape[1] < 4:
            raise ValueError("EOS tables must have shape (Nd, Nt), with Nd and Nt at least four")
        if any(getattr(self, name).shape != shape for name in _FILES):
            raise ValueError("All EOS tables must have the same (Nd, Nt) shape")
        spacing = np.diff(density)
        if spacing[0] <= 0 or not np.allclose(spacing, spacing[0], rtol=1e-10, atol=1e-13):
            raise ValueError("EOS density must be uniformly increasing in natural logarithm")
        for name in ("temperature", "pressure", "energy", "entropy"):
            if not np.all(np.diff(logs[name], axis=1) > 0):
                raise ValueError(f"EOS {name} must increase strictly along each temperature row")
        object.__setattr__(self, "_logs", logs)


def default_table_dir() -> str:
    """Return ``RICHIO_EOS_DIR`` or the bundled Tomida-based RICH table directory."""
    return os.environ.get(TABLE_DIR_ENV) or str(files("richio") / "tables" / "eos")


def load_eos_table(table_dir: str | os.PathLike | None = None) -> RichEOS:
    """Load and cache the Tomida-based tables in RICH's natural-log file format.

    :param table_dir: Directory containing ``density.txt``, ``Pfile.txt``,
        ``csfile.txt``, ``Sfile.txt``, ``Ufile.txt``, ``Tfile.txt``, and
        ``CVfile.txt``. Defaults to :func:`default_table_dir`.
    :returns: A :class:`RichEOS` with physical, unitful arrays.
    :raises FileNotFoundError: If a required file is absent.
    :raises ValueError: If the table layout or coordinates are invalid.
    """
    path = Path(table_dir if table_dir is not None else default_table_dir())
    return _load_eos_table(str(path.expanduser().resolve()))


@functools.lru_cache(maxsize=4)
def _load_eos_table(path):
    arrays = {}
    for name, filename in {"density": "density.txt", **_FILES}.items():
        file = Path(path) / filename
        if not file.is_file():
            raise FileNotFoundError(
                f"EOS table file {file} not found; set {TABLE_DIR_ENV} "
                "or pass table_dir to load_eos_table()"
            )
        arrays[name] = np.loadtxt(file).ravel()
    density = arrays.pop("density")
    if density.size < 4 or not np.all(np.isfinite(density)):
        raise ValueError("EOS density.txt needs at least four finite log-density values")
    spacing = density[1] - density[0]
    if spacing <= 0 or not np.allclose(np.diff(density), spacing, rtol=1e-10, atol=1e-13):
        raise ValueError("EOS density.txt must be uniformly increasing in natural logarithm")
    # RICH reconstructs density from the first two entries, including when
    # converting energy/entropy densities to specific quantities.
    rho = np.exp(density[0] + spacing * np.arange(density.size)) * _RHO
    for name, values in arrays.items():
        if values.size % density.size or values.size // density.size < 4:
            raise ValueError(f"EOS {_FILES[name]} does not have an (Nd, Nt) layout")
        values = values.reshape(density.size, -1)
        if name == "sound_speed_squared":
            arrays[name] = values * _C2
        elif name in ("energy", "entropy"):
            arrays[name] = np.exp(values) * (_REFS[name] * _RHO) / rho[:, None]
        else:
            arrays[name] = np.exp(values) * _REFS[name]
    return RichEOS(density=rho, **arrays, path=path)


def _input(value, name):
    reference = _REFS[name]
    if not isinstance(value, u.unyt_array):
        warnings.warn(f"{name} has no units; assuming {reference.units}", stacklevel=4)
        value = _quantity(value, reference.units)
    try:
        value = value.to(reference.units)
    except UnitConversionError as exc:
        raise ValueError(f"{name} must have units convertible to {reference.units}") from exc
    if not np.all(np.isfinite(value) & (value > 0 * reference)):
        raise ValueError(f"{name} must be finite and positive")
    return value


def _prepare(density, other, name):
    density, other = _input(density, "density"), _input(other, name)
    shape = np.broadcast_shapes(density.shape, other.shape)
    # unyt's ndarray broadcasting helpers discard units; expand through unitful
    # arithmetic instead. All subsequent physical calculations retain units.
    ones = np.ones(shape)
    return (density * ones).reshape(-1), (other * ones).reshape(-1), shape


def _hermite(x, y, xi):
    """RICH's four-point Hermite polynomial, with its original arithmetic."""
    x0, x1, x2, x3 = x.T
    y0, y1, y2, y3 = y.T
    dx, dxL, dxR = x2 - x1, x1 - x0, x3 - x2
    left, right = dxL + dx, dxR + dx
    m0 = (
        -dx * y0 * (1 / (dxL * left))
        + (dx - dxL) * y1 * (1 / (dx * dxL))
        + dxL * y2 * (1 / (dx * left))
    )
    m00 = 2 * (y0 * dx - left * y1 + dxL * y2) * (1 / (dx * dxL * left))
    m1 = (
        -dxR * y1 * (1 / (dx * right))
        - (dx - dxR) * y2 * (1 / (dx * dxR))
        + dx * y3 * (1 / (dxR * right))
    )
    m11 = 2 * (y1 * dxR - right * y2 + dx * y3) * (1 / (dx * dxR * right))
    dx0, dx1 = xi - x1, xi - x2
    ddx = dx0 - dx1
    ddx2 = ddx * ddx
    ddx3 = ddx2 * ddx
    ddx4 = ddx3 * ddx
    ddx5 = ddx4 * ddx
    dx02 = dx0 * dx0
    dx03 = dx02 * dx0
    dx12 = dx1 * dx1
    temp0 = y1 + m0 * dx0 + 0.5 * m00 * dx02
    temp1 = (y2 - y1 - m0 * ddx - 0.5 * m00 * ddx2) * dx03 * (1 / ddx3)
    temp2 = (3 * y1 - 3 * y2 + (2 * m0 + m1) * ddx + 0.5 * m00 * ddx2) * dx03 * dx1 * (1 / ddx4)
    temp3 = (
        (6 * y2 - 6 * y1 - 3 * (m0 + m1) * ddx - 0.5 * (m11 - m00) * ddx2)
        * dx03
        * dx12
        * (1 / ddx5)
    )
    return temp0 + temp1 + temp2 + temp3


def _domain_error(conversion, coordinate, bad, indices):
    affected = indices[np.flatnonzero(bad)]
    raise ValueError(
        f"{conversion}: {coordinate} outside the usable four-point EOS stencil "
        f"at flattened broadcast indices {affected[:8].tolist()} "
        f"({affected.size} affected states)"
    )


def _interpolate(table, density, other, axis, output, conversion, indices):
    """Interpolate along four state rows, then along reconstructed log density."""
    ld = table._logs["density"]
    d = _log_ratio(density, _RHO)
    q = _log_ratio(other, _REFS[axis])
    step = ld[1] - ld[0]
    row = np.floor((d - ld[0]) / step).astype(np.intp)
    bad = (row < 1) | (row > ld.size - 3)
    if np.any(bad):
        _domain_error(conversion, "density", bad, indices)
    coordinates = table._logs[axis]
    data = table.sound_speed_squared if output == "sound_speed_squared" else table._logs[output]
    nt = coordinates.shape[1]
    interpolated = []
    for offset in (-1, 0, 1, 2):
        rows = row + offset
        # Row-wise upper_bound without allocating a query-by-table-sized array.
        low = np.zeros(row.size, dtype=np.intp)
        high = np.full(row.size, nt, dtype=np.intp)
        while np.any(low < high):
            mid = (low + high) // 2
            advance = (low < high) & (coordinates[rows, np.minimum(mid, nt - 1)] <= q)
            high = np.where((low < high) & ~advance, mid, high)
            low = np.where(advance, mid + 1, low)
        index = low
        # Match the source's tolerance at the lower valid endpoint. The other
        # special case (index == 3) applies only to four-entry row tables.
        index = np.where((index == 1) & (np.abs(coordinates[rows, 1] - q) < 1e-13), 2, index)
        if nt == 4:
            index = np.where((index == 3) & (np.abs(coordinates[rows, 2] - q) < 1e-13), 2, index)
        bad = (index < 2) | (index >= nt - 1)
        if np.any(bad):
            _domain_error(conversion, axis, bad, indices)
        cols = index[:, None] - 2 + np.arange(4)
        interpolated.append(
            _hermite(coordinates[rows[:, None], cols], data[rows[:, None], cols], q)
        )
    x = ld[0] + step * (row[:, None] - 1 + np.arange(4))
    # stack preserves unyt units, unlike np.asarray(list_of_unitful_arrays).
    value = _hermite(x, np.stack(interpolated, axis=1), d)
    return value if output == "sound_speed_squared" else np.exp(value) * _REFS[output]


def _evaluate(d, other, axis, output, mask, fallback, table, conversion, shape):
    result = _quantity(np.empty(d.size), _REFS[output].units)
    if np.any(mask):
        result[mask] = fallback(d[mask], other[mask])
    if np.any(~mask):
        table = load_eos_table() if table is None else table
        result[~mask] = _interpolate(
            table, d[~mask], other[~mask], axis, output, conversion, np.flatnonzero(~mask)
        )
    return result.reshape(shape)


def dp2e(d, p, *, table=None):
    """Specific energy [erg/g] from density and pressure; floors p/d at 1e8 erg/g."""
    d, p, shape = _prepare(d, p, "pressure")
    p = np.maximum(p, d * (1e8 * _E))
    return _evaluate(
        d,
        p,
        "pressure",
        "energy",
        p > d * (1e16 * _E),
        lambda d, p: 1.5 * p / d,
        table,
        "dp2e",
        shape,
    )


def dp2T(d, p, *, table=None):
    """Temperature [K] from density and pressure; no low-pressure floor."""
    d, p, shape = _prepare(d, p, "pressure")
    return _evaluate(
        d,
        p,
        "pressure",
        "temperature",
        p > d * (1e16 * _E),
        lambda d, p: p * (7.452e-9 * _T / _E) / d,
        table,
        "dp2T",
        shape,
    )


def de2T(d, e, *, table=None):
    """Temperature [K] from density and specific internal energy [erg/g]."""
    d, e, shape = _prepare(d, e, "energy")
    return _evaluate(
        d,
        e,
        "energy",
        "temperature",
        e > 1e16 * _E,
        lambda d, e: e / (1.5 * _GAS),
        table,
        "de2T",
        shape,
    )


def dT2p(d, T, *, table=None):
    """Pressure [erg/cm**3] from density and temperature; ideal gas above 5e7 K."""
    d, T, shape = _prepare(d, T, "temperature")
    return _evaluate(
        d,
        T,
        "temperature",
        "pressure",
        T > 5e7 * _T,
        lambda d, T: T * d * _GAS,
        table,
        "dT2p",
        shape,
    )


def dT2e(d, T, *, table=None):
    """Specific energy [erg/g] from density and temperature; ideal gas above 8e7 K."""
    d, T, shape = _prepare(d, T, "temperature")
    return _evaluate(
        d,
        T,
        "temperature",
        "energy",
        T > 8e7 * _T,
        lambda d, T: T * 1.5 * _GAS,
        table,
        "dT2e",
        shape,
    )


def de2p(d, e, *, table=None):
    """Pressure [erg/cm**3] from density and specific internal energy [erg/g]."""
    d, e, shape = _prepare(d, e, "energy")
    return _evaluate(
        d,
        e,
        "energy",
        "pressure",
        e > 1e16 * _E,
        lambda d, e: e * d * 0.66666666666,
        table,
        "de2p",
        shape,
    )


def dp2c(d, p, *, table=None):
    """Sound speed [cm/s] from density and pressure; floors p/d at 1e8 erg/g."""
    d, p, shape = _prepare(d, p, "pressure")
    p = np.maximum(p, d * (1e8 * _E))
    squared = _evaluate(
        d,
        p,
        "pressure",
        "sound_speed_squared",
        p > d * (1e16 * _E),
        lambda d, p: 5 * p / (3 * d),
        table,
        "dp2c",
        shape,
    )
    return np.sqrt(squared)


def dp2cv(d, p, *, table=None):
    """Volumetric heat capacity [erg/(cm**3*K)] from density and pressure."""
    d, p, shape = _prepare(d, p, "pressure")
    return _evaluate(
        d,
        p,
        "pressure",
        "heat_capacity",
        p > d * (1e15 * _E),
        lambda d, p: _GAS_CV * d,
        table,
        "dp2cv",
        shape,
    )


def dT2cv(d, T, *, table=None):
    """Volumetric heat capacity [erg/(cm**3*K)] from density and temperature.

    RICH uses the ideal-gas fallback when T > 1e6 K or density > 10 g/cm**3.
    """
    d, T, shape = _prepare(d, T, "temperature")
    return _evaluate(
        d,
        T,
        "temperature",
        "heat_capacity",
        (T > 1e6 * _T) | (d > 10 * _RHO),
        lambda d, T: _GAS_CV * d,
        table,
        "dT2cv",
        shape,
    )


def de2c(d, e, *, table=None):
    """Sound speed [cm/s] from density and specific energy, via de2p then dp2c."""
    d, e, shape = _prepare(d, e, "energy")
    try:
        return dp2c(d, de2p(d, e, table=table), table=table).reshape(shape)
    except ValueError as exc:
        raise ValueError(f"de2c: {exc}") from exc


def _entropy_fallback(d, p):
    # The fitted Sackur–Tetrode expression expects numerical CGS ratios.
    argument = -38.43 + np.log((p / _P) ** 1.5 * (d / _RHO) ** -2.5)
    return 10.0 ** (8.128 + np.log10(argument)) * _S


def dp2s(d, p, *, table=None):
    """Specific entropy [erg/(g*K)] from density and pressure.

    Above p/d = 1e16 erg/g, use RICH's fitted Sackur–Tetrode expression.
    """
    d, p, shape = _prepare(d, p, "pressure")
    return _evaluate(
        d, p, "pressure", "entropy", p > d * (1e16 * _E), _entropy_fallback, table, "dp2s", shape
    )


def dT2s(d, T, *, table=None):
    """Specific entropy [erg/(g*K)] from density and temperature, via dT2p then dp2s."""
    d, T, shape = _prepare(d, T, "temperature")
    try:
        return dp2s(d, dT2p(d, T, table=table), table=table).reshape(shape)
    except ValueError as exc:
        raise ValueError(f"dT2s: {exc}") from exc


def sd2p(s, d, *, table=None):
    """Pressure [erg/cm**3] from specific entropy and density (entropy first).

    RICH switches to its analytic inverse above ``dp2s(d, 5.2*d)`` in code
    units. Here 5.2 carries richio's code specific-energy unit; this threshold
    is independent of the units chosen to express the input arrays.
    """
    d, s, shape = _prepare(d, s, "entropy")
    try:
        smax = dp2s(d, d * _ENTROPY_SWITCH, table=table)
    except ValueError as exc:
        raise ValueError(f"sd2p threshold: {exc}") from exc
    return _evaluate(
        d,
        s,
        "entropy",
        "pressure",
        s > smax,
        lambda d, s: ((d / _RHO) ** 2.5 * np.exp(s / _ENTROPY_SCALE + 38.43)) ** 0.666666666 * _P,
        table,
        "sd2p",
        shape,
    )
