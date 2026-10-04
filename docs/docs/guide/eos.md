# Equation of state

`richio.eos` implements RICH's equation of state using tables based on
**Tomida et al. (2013)**, *Radiation Magnetohydrodynamic Simulations of
Protostellar Collapse: Protostellar Core Formation*. The four-point Hermite
interpolation, pressure floors, and analytic fallbacks follow RICH's
`source/newtonian/three_dimensional/OndrejEOS.cpp`. These numerical choices are
specific to RICH; they are not presented as a general implementation of that paper.

The tables ship with richio and load on the first table lookup:

```python
import richio

snap = richio.load("snap_0042.h5")
pressure = richio.eos.dT2p(snap.rho, snap.T)
snap.plots.slice(data=pressure, res=512)
```

Like opacity, EOS conversions are standalone functions. They accept snapshot
fields, clipped subsets, or independently constructed arrays. They do not replace
stored snapshot fields.

## Conversions and units

The names and argument order match RICH. Density `d` is in g/cm³, pressure `p`
in erg/cm³, temperature `T` in K, specific internal energy `e` in erg/g, and
specific entropy `s` in erg/(g K). All arguments can carry any compatible units.

| Function | Inputs, in order | Output | CGS output unit |
| --- | --- | --- | --- |
| `dp2e(d, p)` | Density, pressure | Specific internal energy | erg/g |
| `dp2T(d, p)` | Density, pressure | Temperature | K |
| `de2T(d, e)` | Density, specific energy | Temperature | K |
| `dT2p(d, T)` | Density, temperature | Pressure | erg/cm³ |
| `dT2e(d, T)` | Density, temperature | Specific internal energy | erg/g |
| `de2p(d, e)` | Density, specific energy | Pressure | erg/cm³ |
| `dp2c(d, p)` | Density, pressure | Sound speed | cm/s |
| `dp2cv(d, p)` | Density, pressure | Volumetric heat capacity | erg/(cm³ K) |
| `dT2cv(d, T)` | Density, temperature | Volumetric heat capacity | erg/(cm³ K) |
| `de2c(d, e)` | Density, specific energy | Sound speed | cm/s |
| `dp2s(d, p)` | Density, pressure | Specific entropy | erg/(g K) |
| `dT2s(d, T)` | Density, temperature | Specific entropy | erg/(g K) |
| `sd2p(s, d)` | Specific entropy, density | Pressure | erg/cm³ |

`cv` is heat capacity **per volume**, not per mass. Divide by unitful density if
you need a mass-specific heat capacity. Energy `e` is already per mass.

Physical tables, inputs, and fallback constants carry `unyt` units. Only at the
logarithmic interpolation boundary are quantities divided by explicit CGS
reference quantities to form dimensionless logarithms. Bare inputs assume CGS
and issue a warning; incorrect dimensions raise an error.

```python
import numpy as np
import unyt as u
from richio import eos

rho = u.unyt_array([[1e-10], [1e-8]], "g/cm**3")
T = u.unyt_array([1e4, 1e5, 1e6], "K")
p = eos.dT2p(rho, T)                  # shape (2, 3), NumPy broadcasting
p_code = p.in_base("rich")            # registry is attached to the result
T_back = eos.dp2T(rho, p)
scalar = eos.dT2p(1e-8 * u.g/u.cm**3, 1e5 * u.K)
```

Scalar results have shape `()`. Inverse-table interpolation need not exactly
invert the forward interpolant, and different fallback thresholds can prevent
round trips from agreeing across regimes.

## Tables and overrides

```python
table = eos.load_eos_table()           # RichEOS; cached by resolved directory
table.energy.units                   # erg/g
table.heat_capacity.units            # erg/(cm**3*K)
table.pressure.shape                  # (604, 495) for the bundled tables

custom = eos.load_eos_table("/path/to/EOS")
p = eos.dT2p(rho, T, table=custom)
```

Without an explicit directory, `RICHIO_EOS_DIR` overrides the bundled tables.
`RichEOS` holds read-only physical arrays and their cached interpolation
coordinates. Each two-dimensional array is ordered density then temperature.

The loader accepts the current RICH file format: `density.txt` contains natural
log density; `Pfile.txt`, `Tfile.txt`, `Ufile.txt`, `Sfile.txt`, and `CVfile.txt`
contain natural logs of CGS pressure, temperature, energy density, entropy
density, and volumetric heat capacity. `csfile.txt` contains **linear sound-speed
squared**. Quantity tables are flattened in density-major order. Loading converts
energy and entropy densities to specific quantities. Base-10 legacy tables are
not this format.

## Domain and RICH fallbacks

The bundled grid spans approximately 1e-22–912 g/cm³ and 500–9.92e7 K, but its
four-point stencil needs neighboring nodes. The usable interpolation domain is
smaller than the full tabulated range. For pressure, energy, and entropy queries,
the query must have a valid stencil in all four neighboring density rows.

The conversion-specific branches are preserved, with strict `>` comparisons:

| Conversion | Analytic fallback condition |
| --- | --- |
| `dp2e`, `dp2T`, `dp2c`, `dp2s` | p/ρ > 1e16 erg/g |
| `dp2cv` | p/ρ > 1e15 erg/g |
| `de2p`, `de2T` | e > 1e16 erg/g |
| `dT2p` | T > 5e7 K |
| `dT2e` | T > 8e7 K |
| `dT2cv` | T > 1e6 K or ρ > 10 g/cm³ |
| `de2c` | Calls `de2p` then `dp2c`, including their branches |
| `dT2s` | Calls `dT2p` then `dp2s`, including their branches |
| `sd2p` | s > `dp2s(d, d * E_switch)` |

`dp2s` uses RICH's fitted Sackur–Tetrode entropy expression in its analytic
branch. `sd2p` uses the corresponding fitted inverse, preserving the source's
constants. Its switching energy is `E_switch = 5.2 * code_length**2/code_time**2`,
using richio's existing RICH scales (7e10 cm and 1603 s). Merely expressing inputs
in another unit system does not change this physical switching threshold.

Only `dp2e` and `dp2c` floor p/ρ at 1e8 erg/g. This floor does not guarantee a
valid table lookup: with the bundled tables it can still lie below the usable
pressure range.

Inputs must be finite and positive. Unsupported interpolation states raise
`ValueError` identifying the conversion and flattened broadcast indices;
there is no added clamping or extrapolation. An array containing an unsupported
state fails as a whole. Valid analytic branches bypass table interpolation.
