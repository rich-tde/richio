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

import warnings

import unyt as u
from unyt import Unit, UnitRegistry
from unyt.dimensions import length, mass, time
from unyt.unit_systems import UnitSystem

from richio.config import FIELD_REGISTRY

_MISSING = object()  # sentinel for optional default in get_unit()


class Units:
    """Container for RICH simulation code units.

    The code unit system is defined by:

    * **Mass** — 1 solar mass (M☉ ≈ 2 × 10³⁰ kg)
    * **Length** — 1 solar radius (R☉ ≈ 7 × 10⁸ m)
    * **G = 1** (fixes the time unit, t_code ≈ 1603 s)

    All named unit expressions are accessible as attributes (e.g.
    ``units.lscale``, ``units.tscale``) and are also registered in a custom
    ``'rich'`` :class:`unyt.unit_systems.UnitSystem` so that
    ``qty.in_base('rich')`` works on any :class:`unyt.unyt_array`.

    Attributes
    ----------
    mscale : :class:`unyt.Unit`
        Code mass unit (~ 1 M☉).
    lscale : :class:`unyt.Unit`
        Code length unit (~ 1 R☉).
    tscale : :class:`unyt.Unit`
        Code time unit (~ 1603 s, derived from G = 1).
    system : :class:`unyt.unit_systems.UnitSystem`
        The ``'rich'`` unit system registered with unyt.
    registry : :class:`unyt.UnitRegistry`
        Custom registry containing the three code-unit base definitions.
    """

    def __init__(self):
        reg = UnitRegistry(unit_system="cgs")

        # base_value default in mks
        reg.add("code_mass",   base_value=2e30,    dimensions=mass,   tex_repr=r"M_\odot")
        reg.add("code_length", base_value=7e8,     dimensions=length, tex_repr=r"R_\odot")
        reg.add("code_time",   base_value=1603.0,  dimensions=time,   tex_repr=r"t_\text{code}")

        # Base units
        self.mscale = Unit("code_mass",   registry=reg)  # ~ solar mass 1.988e33 g
        self.lscale = Unit("code_length", registry=reg)  # ~ solar radius 6.955e10 cm
        self.tscale = Unit("code_time",   registry=reg)  # G ≈ 1; ≈ 1592 s

        # The rich unit system
        rus = UnitSystem(
            "rich",
            mass_unit=self.mscale,
            length_unit=self.lscale,
            time_unit=self.tscale,
            registry=reg,
        )
        self.system = rus
        self.registry = reg

        # ------------------------------------------------------------------
        # Named unit expressions — the string keys used in FIELD_REGISTRY.
        # Add a new key here if you need a new unit expression.
        # ------------------------------------------------------------------
        _unit_keys = {
            "lscale":              self.lscale,
            "tscale":              self.tscale,
            "mscale":              self.mscale,
            "dimensionless":       u.Dimensionless,
            "density":             rus["density"],
            "pressure":            rus["pressure"],
            "temperature":         rus["temperature"],
            "volume":              rus["volume"],
            "velocity":            rus["velocity"],
            "specific_energy":     rus["energy"] / self.mscale,
            "dissipation":         rus["energy"] / self.lscale**3 / self.tscale,
            "pressure_gradient":   rus["pressure"] / self.lscale,
            "density_gradient":    rus["density"]  / self.lscale,
            "sie_gradient":        rus["energy"]   / self.mscale / self.tscale,
            "velocity_divergence": rus["velocity"] / self.lscale,
            "specific_entropy":    rus["energy"]   / rus["temperature"] / self.mscale,
            "tfb_unit":            2.577726 * u.day,  # NPY fallback time unit
            "unknown":             1,
        }

        # Build field→unit mapping from the central registry (one place to maintain)
        self._unit_per_field = {}
        for h5_key, info in FIELD_REGISTRY.items():
            self._unit_per_field[h5_key] = _unit_keys[info["unit"]]

            npy_name = info.get("npy_name")
            if npy_name and npy_name != h5_key:
                npy_unit_key = info.get("npy_unit", info["unit"])
                self._unit_per_field[npy_name] = _unit_keys[npy_unit_key]

    def get_unit(self, key: str, default=_MISSING):
        """Return the unit associated with a RICH output field.

        :param key: Canonical field name (e.g. ``"Density"``) or alias
                    (e.g. ``"density"``).
        :type key: str
        :param default: Value to return when *key* is not found in the
                        registry.  If omitted, a :exc:`ValueError` is raised
                        for unknown keys.
        :returns: The :class:`unyt.Unit` (or unit expression) for the field,
                  or *default* if provided and the key is unknown.
        :raises ValueError: If *key* is unknown and no *default* was supplied.
        """
        if key in self._unit_per_field:
            unit = self._unit_per_field[key]
            # if unit == 1:
                # warnings.warn(f"'{key}' is in the data output but not used in the simulation.")
            return unit

        if default is not _MISSING:
            return default

        raise ValueError(f"Unknown key '{key}'. Supported keys: {list(self._unit_per_field)}")


# Singleton instance for convenience
units = Units()


def to_rich_units(qty):
    """
    Convert a unyt_array or unyt_quantity object to using the custom RICH unit
    registry. This does NOT change the unit, only the registry, such that
    ``qty.in_base('rich')`` is allowed afterwards.

    :param qty: unyt_array or unyt_quantity object.
    :returns: a unyt_array or unyt_quantity object but in RICH unit registry.
    """
    if isinstance(qty, u.unyt_quantity):
        qty = u.unyt_quantity(qty.value, qty.units, registry=units.registry)
    elif isinstance(qty, u.unyt_array):
        qty = u.unyt_array(qty.value, qty.units, registry=units.registry)
    else:
        raise Exception("Quantity is neither an unyt_quantity nor an unyt_array.")

    return qty.in_base("rich")
