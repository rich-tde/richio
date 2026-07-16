# Units

Every field RICHIO returns carries its physical units. Under the hood these are
[`unyt`](https://unyt.readthedocs.io) arrays, NumPy arrays that also know whether
they hold a density, a velocity, a temperature, and so on. This means you never
have to track conversion factors by hand: arithmetic keeps the units straight for
you, and switching systems is a single method call.

## RICH's own units

RICH does its computing in a unit system scaled to the Sun. RICHIO knows this
system as `'rich'` and registers it the moment you import the package. Three base
units define it:

| Quantity | Unit | Roughly |
|----------|------|---------|
| mass | one solar mass | 2 × 10³⁰ kg |
| length | one solar radius | 7 × 10⁸ m |
| time | set by fixing G = 1 | 1603 s |

Everything else (density, pressure, velocity, and the rest) follows from those
three. A field you have just read is in these units:

```python
>>> rho = snap["Density"]
>>> rho.units
1988415860000000000000000000000*kg/Rsun**3
```

That awkward-looking unit is a solar mass per solar radius cubed, written out in
kilograms and solar radii. It is correct, just not how you would want to report a
density, so convert it.

## Converting

For the two standard systems, there is a method each:

```python
rho.in_cgs()            # g/cm**3
rho.in_mks()            # kg/m**3
rho.in_base("rich")     # back to RICH's units
```

`"rich"` works as a target here because RICHIO registered that system on import.
You can also ask for any specific unit by name:

```python
snap["Temperature"].to("K")
snap["Vx"].to("km/s")
snap["Time"].to("day")
```

## Looking up a field's unit without reading it

Sometimes you want to know what unit a field would come in, without loading the
data. The `units` object maps a field name (or any of its aliases) to its unit:

```python
from richio.units import units

units.get_unit("Density")                # the density unit
units.get_unit("rho")                     # aliases work here too
units.get_unit("mystery", default=None)   # returns None instead of raising
```

Without a `default`, asking for an unknown field raises `ValueError`.

## Bringing your own arrays into the RICH system

If you build a `unyt` array yourself, or load one from somewhere else, it won't
know about the `'rich'` system, so `.in_base("rich")` would fail on it. The
`to_rich_units` helper fixes that. It leaves the value and the unit untouched and
re-registers the array against RICHIO's unit system:

```python
from richio.units import to_rich_units
import unyt as u

q = u.unyt_quantity(1.0, "g/cm**3")   # a plain unyt quantity
q = to_rich_units(q)                  # now convertible with .in_base("rich")
```

## Further reading

unyt does far more than convert. It handles array maths that respects units,
comparisons across units, defining your own units, and more. See its
[documentation](https://unyt.readthedocs.io/en/stable/).
