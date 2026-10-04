# RICH EOS tables

These tables are based on Tomida et al. (2013), *Radiation Magnetohydrodynamic
Simulations of Protostellar Collapse: Protostellar Core Formation*. They are
copied unchanged from RICH's `data/EOS` directory and consumed by the port of
`source/newtonian/three_dimensional/OndrejEOS.cpp` in `richio.eos`.

The grid contains 604 density rows and 495 temperature columns. `density.txt`
stores natural log density in g/cm³. All other files flatten the grid in
density-major order. `Pfile.txt`, `Tfile.txt`, `Ufile.txt`, `Sfile.txt`, and
`CVfile.txt` store natural logs of CGS pressure, temperature, energy density,
entropy density, and volumetric heat capacity. `csfile.txt` stores linear c_s²
in cm²/s². The loader converts energy and entropy densities to specific quantities.

The interpolation, pressure floors, and analytic fallbacks are specific to RICH.
