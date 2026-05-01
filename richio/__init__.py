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

"""
richio — I/O and analysis library for RICH TDE simulations.

The top-level namespace re-exports everything from :mod:`richio.data` (the
primary entry-point is :func:`richio.load`) and exposes the
:mod:`richio.plots` and :mod:`richio.units` sub-modules.

Typical usage::

    import richio
    snap = richio.load("snap_0042.h5")
    snap.density.to("g/cm**3")
    snap.plots.slice(data="density", res=512)
"""

from richio import (
    config,  # noqa: F401
    plots,
    units,
)
from richio.data import *
from richio.units import *
