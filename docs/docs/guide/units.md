# Units

Every field RICHIO returns is a [`unyt`](https://unyt.readthedocs.io) array that
carries its physical units. This means you never have to track conversion factors
by hand — arithmetic propagates units, and conversions are one method call.

## The RICH code unit system

RICH runs in a solar-scaled code unit system. RICHIO defines it as a `'rich'`
unit system and registers it with unyt on import:

| Quantity | Code unit | Approx. value |
|----------|-----------|---------------|
| mass | 1 solar mass (M☉) | ≈ 2 × 10³⁰ kg |
| length | 1 solar radius (R☉) | ≈ 7 × 10⁸ m |
| time | set by fixing **G = 1** | ≈ 1603 s |

All other units (density, pressure, velocity, …) are derived from these three.
A freshly read field is in code units:

```python
>>> rho = snap["Density"]
>>> rho.units
1988415860000000000000000000000*kg/Rsun**3
```

## Converting

Convert to a standard base system with the usual unyt methods:

```python
rho.in_cgs()            # g/cm**3
rho.in_mks()            # kg/m**3
rho.in_base("rich")     # back to RICH code units
```

`"rich"` is available as a base because RICHIO registered the system at import
time. You can also target any explicit unit:

```python
snap["Temperature"].to("K")
snap["Vx"].to("km/s")
```

## Looking up a field's unit

The `units` singleton maps field names (and aliases) to their unit via
[`get_unit`](../api/units.md):

```python
from richio.units import units

units.get_unit("Density")             # the density code unit
units.get_unit("rho")                 # aliases work too
units.get_unit("mystery", default=None)  # None instead of raising for unknown keys
```

Without a `default`, an unknown key raises `ValueError`.

## Bringing external arrays into the RICH registry

If you build a `unyt` array yourself (or load one from elsewhere), it won't know
about the `'rich'` system until you re-home it in RICHIO's registry.
[`to_rich_units`](../api/units.md) does exactly that — it keeps the value and
unit unchanged but swaps in the RICH registry so `.in_base("rich")` works
afterward:

```python
from richio.units import to_rich_units
import unyt as u

q = u.unyt_quantity(1.0, "g/cm**3")   # plain unyt, no 'rich' system
q = to_rich_units(q)                  # now in the RICH registry, expressed in code units
```

## Further reading

unyt does far more than convert — array math, equality across units, custom
units, and more. See the [unyt documentation](https://unyt.readthedocs.io/en/stable/).
