"""Detached particle records with collection and field-column access."""

import numpy as np
import unyt as u

from richio.config import FIELD_REGISTRY
from richio.units import units

_FIELD_NAMES = {}
for _canonical, _info in FIELD_REGISTRY.items():
    for _name in (_canonical, _info.get("npy_name", _canonical), *_info["aliases"]):
        _FIELD_NAMES[_name] = _canonical


def _canonical_field(name):
    """Resolve registered storage names and aliases, preserving unknown names."""
    return _FIELD_NAMES.get(name, name)


def _particle_index(index, size):
    """Validate an integer particle index and resolve negative positions."""
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
        raise TypeError("Particle indices must be integers (not booleans).")
    index = int(index)
    if index < 0:
        index += size
    if not 0 <= index < size:
        raise IndexError("Particle index out of range.")
    return index


class Particles:
    """An ordered collection of detached particle records.

    Construct from an iterable of single-particle :class:`Particles` objects,
    or start empty and :meth:`append` particles returned by ``snap[i]``.
    Integer indexing, slices, and iteration all return ``Particles`` objects.

    String indexing returns a field column, resolving snapshot field aliases.
    The particle axis is always retained: scalar fields have shape ``(N,)``
    and vector fields have shape ``(N, ...)``, including when ``N == 1``.
    Compatible units are converted to those of the first record. Columns are
    independent copies; changing them does not change the stored particles.

    ``dict(particles)`` returns canonical field names mapped to these columns.
    A field missing from any record raises ``KeyError`` when accessed, including
    during dictionary conversion. An empty collection converts to ``{}``.
    """

    def __init__(self, particles=()):
        self._records = []
        self.extend(particles)

    @classmethod
    def _from_records(cls, records):
        """Wrap prepared, private records without copying their data again."""
        result = cls()
        result._records = records
        return result

    @classmethod
    def _from_record(cls, raw_record):
        """Detach one stored row, attaching units before normalizing its names."""
        record = {}
        for name, value in raw_record.items():
            if isinstance(value, u.unyt_array):
                value = value.copy()
            else:
                value = np.array(value, copy=True)
                unit = units.get_unit(name, default=1)
                if isinstance(unit, u.Unit):
                    # Construct directly: multiplying by a Unit can cast integer IDs.
                    value = u.unyt_array(value, unit)
                elif isinstance(unit, u.unyt_array):
                    # Some storage units include a multiplier (NPY ``tfb``).
                    value = value * unit
            record[_canonical_field(name)] = value
        return cls._from_records([record])

    def __len__(self):
        return len(self._records)

    def __repr__(self):
        return f"{type(self).__name__}(n={len(self)})"

    def __iter__(self):
        for record in self._records:
            yield type(self)._from_records([record])

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._column(key)
        if isinstance(key, slice):
            records = self._records[key]
        else:
            records = [self._records[_particle_index(key, len(self))]]
        return type(self)._from_records(records)

    def keys(self):
        """Return canonical field names in order of their first occurrence."""
        return list(dict.fromkeys(key for record in self._records for key in record))

    def append(self, particle):
        """Append a ``Particles`` object containing exactly one particle."""
        if not isinstance(particle, Particles):
            raise TypeError("append() requires a single-particle Particles object.")
        if len(particle) != 1:
            raise ValueError("append() requires exactly one particle; use extend() for many.")
        self._records.append(particle._records[0])

    def extend(self, particles):
        """Append a collection or an iterable of single-particle objects."""
        if isinstance(particles, Particles):
            self._records.extend(particles._records)
        else:
            for particle in particles:
                self.append(particle)

    def _column(self, key):
        if not self._records:
            raise ValueError("Cannot read a field from an empty Particles collection.")
        key = _canonical_field(key)
        values = []
        for index, record in enumerate(self._records):
            if key not in record:
                raise KeyError(f"Field {key!r} is missing from particle {index}.")
            values.append(record[key])

        unitful = [isinstance(value, u.unyt_array) for value in values]
        if any(unitful) and not all(unitful):
            raise TypeError(f"Field {key!r} mixes values with and without units.")
        if not any(unitful):
            return np.stack(values)

        unit = values[0].units
        # Avoid converting identical units, preserving integer fields exactly.
        column = np.stack(
            [
                np.asarray(value) if value.units == unit else value.to_value(unit)
                for value in values
            ]
        )
        return u.unyt_array(column, unit)
