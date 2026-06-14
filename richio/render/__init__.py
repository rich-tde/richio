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
richio.render — depth-cued volume rendering of RICH snapshots.

This is an **optional** subpackage. The heavy 3-D rendering stack (``yt``,
``imageio``) is not pulled in by a plain ``pip install richio``; install it with::

    pip install "richio[render]"

The core ``import richio`` never touches this module, so the base library stays a
compact reading/analysis helper.

Two layers:

* :mod:`richio.render.grid` — backend-neutral resampling of the unstructured
  Voronoi cells onto a uniform Cartesian grid (reuses
  :meth:`richio.data.Snapshot.to_3dgrid`).  Has **no** ``yt`` dependency, so a
  future PyVista/VTK backend can consume the same :class:`UniformGrid`.
* :mod:`richio.render.yt_backend` — turns a grid into a ``yt`` dataset and
  renders depth-cued **static images** (:func:`volume_image`) and
  **rotating-camera movies** (:func:`volume_movie`).

Typical usage::

    import richio
    import richio.render as rr

    snap = richio.load("snap_18.h5")
    rr.volume_image(snap, "density", res=256, filename="vr.png")
    rr.volume_movie(snap, "density", res=256, n_frames=180, filename="spin.mp4")
"""

from richio.render.grid import UniformGrid, tight_box, to_uniform_grid

# yt-backed functions are imported lazily so that `import richio.render` works
# (e.g. just to build a UniformGrid) even when yt is not installed.
__all__ = [
    "UniformGrid",
    "to_uniform_grid",
    "tight_box",
    "to_yt",
    "volume_image",
    "volume_movie",
]


def __getattr__(name):
    if name in ("to_yt", "volume_image", "volume_movie"):
        from richio.render import yt_backend

        return getattr(yt_backend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
