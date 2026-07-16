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
:func:`build_knn`
    Build a *k*-NN graph (same CSR format); a cheap, clustering-robust
    drop-in for :func:`build_voronoi` that scales to ~80M cells.  Pass
    ``cells=shock_candidates(snap).candidates`` to build only the rows the
    shockfinder needs.
:func:`shock_candidates`
    Conditions 1 & 2 candidate cells — the only cells whose neighbours are
    ever used; feed to :func:`build_knn` ``cells`` to restrict the query.
:func:`find_shock_zone`
    Flag shock-zone cells via the three Schaal+14 criteria.
:func:`find_shock_surface`
    Identify shock-surface cells and compute Rankine-Hugoniot Mach numbers.
"""

from richio.shockfinder.sf import *
