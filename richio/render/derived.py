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

"""Derived (computed-on-the-fly) fields for rendering.

These are quantities that are not stored in the snapshot but are combined from
several stored fields — e.g. the specific Bernoulli parameter.  Render entry
points (``volume_image``, ``evolution_movie``) recognise the names in
:data:`DERIVED_FIELDS` and compute the per-cell array before resampling.
"""

import numpy as np
import unyt as u


def bernoulli(snap, *, m_bh=1e4, m_star=0.5, r_star=0.47,
              coords=("CMx", "CMy", "CMz"), normalize=True, soft=1e-2):
    """Specific Bernoulli parameter (specific total energy) per cell.

    ``B = -G·M_BH/r + ½v² + e_int + (P + u_rad/3)/ρ``, in code units (G=1,
    M☉/R☉).  Here ``Erad`` is the *specific* radiation energy (per mass), so the
    radiation-pressure term ``(u_rad/3)/ρ`` is just ``Erad/3``.  ``B < 0`` is
    gravitationally bound, ``B > 0`` is unbound.

    :param snap: A loaded :class:`richio.data.Snapshot`.
    :param m_bh: Black-hole mass (code mass, i.e. M☉).
    :param m_star: Disrupted star mass (M☉) — only used for ``Δε``.
    :param r_star: Disrupted star radius (R☉) — only used for ``Δε``.
    :param coords: Position fields giving distance from the BH at the origin.
    :param normalize: If ``True`` (default) return the dimensionless ``B/Δε``
                      where ``Δε = G·M_BH·R_*/R_t² = M_BH^{1/3} M_*^{2/3}/R_*`` is
                      the canonical TDE spread in specific orbital energy.
    :param soft: Floor on ``r`` (R☉) to avoid the singularity at the BH.
    :returns: ``unyt`` array — dimensionless ``B/Δε`` (``normalize``) or ``B`` in
              code specific-energy units.
    """
    def cu(name):
        return np.asarray(snap._get_data(name).in_base("rich"), dtype="float64")

    x, y, z = cu(coords[0]), cu(coords[1]), cu(coords[2])
    r = np.maximum(np.sqrt(x * x + y * y + z * z), soft)
    v2 = cu("Vx") ** 2 + cu("Vy") ** 2 + cu("Vz") ** 2

    phi = -m_bh / r                                     # G = 1
    kinetic = 0.5 * v2
    thermal = cu("internal_energy") + cu("pressure") / cu("density") + cu("radiation_energy") / 3.0
    bern = phi + kinetic + thermal                      # code specific energy

    if normalize:
        r_t = r_star * (m_bh / m_star) ** (1.0 / 3.0)
        delta_eps = m_bh * r_star / r_t ** 2            # = M_BH^{1/3} M_*^{2/3} / R_*
        return u.unyt_array(bern / delta_eps, "dimensionless")
    return u.unyt_array(bern, "code_length**2/code_time**2")


#: Field names the render path computes via this module (name -> callable).
DERIVED_FIELDS = {"bernoulli": bernoulli}
