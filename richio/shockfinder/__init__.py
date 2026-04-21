"""
richio.shockfinder — shock-zone and shock-surface detection for RICH snapshots.

The algorithm is based on Schaal et al. (2015), MNRAS 446, 3992
(arXiv:1407.4117).  The typical workflow is::

    import richio
    from richio.shockfinder import build_voronoi, find_shock_zone, find_shock_surface

    snap = richio.load("snap_0100.h5")
    vor  = build_voronoi(snap)
    zone = find_shock_zone(snap, vor)
    surf = find_shock_surface(snap, vor, zone)

Public API
----------
:func:`build_voronoi`
    Build the Voronoi adjacency graph (CSR format) from cell centres.
:func:`find_shock_zone`
    Flag shock-zone cells via the three Schaal+14 criteria.
:func:`find_shock_surface`
    Identify shock-surface cells and compute Rankine-Hugoniot Mach numbers.
"""

from richio.shockfinder.sf import (
    build_voronoi,
    find_shock_surface,
    find_shock_zone,
)
