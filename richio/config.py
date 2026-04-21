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
Central configuration for RICHIO field names, aliases, and units.

The single exported object is :data:`FIELD_REGISTRY`, a dictionary that maps
every HDF5 / NPY field to its metadata.  It is the **only** place that needs
to be edited when adding or renaming a simulation output field.

:data:`FIELD_REGISTRY` structure
---------------------------------
Each entry has the form::

    "H5_key": {
        "aliases":  [list of alternative names accepted by __getitem__],
        "unit":     "<string key resolved by Units.get_unit>",
        "npy_name": "<canonical name in the NPY format>",  # if different
        "npy_unit": "<unit key override for the NPY name>",  # optional
        "npy_only": True,  # if the field does not exist in HDF5 snapshots
    }

``unit`` values must be valid keys in the ``_unit_keys`` dict inside
:class:`richio.units.Units`.
"""

# ---------------------------------------------------------------------------
# FIELD_REGISTRY — single source of truth for field aliases and units.
#
# To add a new field, add ONE entry here and nowhere else.
#
# Keys per entry:
#   aliases  : list of alternative names accepted by __getitem__
#   unit     : string key resolved by Units (see units.py for available keys)
#   npy_name : canonical name used in the NPY format (if different from H5 key)
#   npy_unit : unit key override for the NPY canonical name (optional)
#   npy_only : True if the field only exists in the NPY format
# ---------------------------------------------------------------------------

FIELD_REGISTRY = {
    # ---- Metadata -----------------------------------------------------------
    "Box": {
        "aliases": ["box_size", "box", "boxsize"],
        "unit": "lscale",
        "npy_name": "box",
    },
    "Time": {
        "aliases": ["time", "t", "tfb", "simulation_time", "current_time"],
        "unit": "tscale",
        "npy_name": "tfb",
        "npy_unit": "tfb_unit",  # fallback time for NPY (Mbh=1e4, Mstar=0.5, Rstar=0.47)
    },
    "Cycle": {
        "aliases": ["cycle", "step", "iteration"],
        "unit": "dimensionless",
    },
    "ID": {
        "aliases": ["particle_id", "id", "ids", "particle_ids"],
        "unit": "dimensionless",
    },
    # ---- Positions ----------------------------------------------------------
    "X": {
        "aliases": ["position_x", "pos_x", "x", "particle_position_x"],
        "unit": "lscale",
    },
    "Y": {
        "aliases": ["position_y", "pos_y", "y", "particle_position_y"],
        "unit": "lscale",
    },
    "Z": {
        "aliases": ["position_z", "pos_z", "z", "particle_position_z"],
        "unit": "lscale",
    },
    # ---- Centre of mass -----------------------------------------------------
    "CMx": {
        "aliases": ["cm_x", "center_of_mass_x"],
        "unit": "lscale",
    },
    "CMy": {
        "aliases": ["cm_y", "center_of_mass_y"],
        "unit": "lscale",
    },
    "CMz": {
        "aliases": ["cm_z", "center_of_mass_z"],
        "unit": "lscale",
    },
    # ---- Velocities ---------------------------------------------------------
    "Vx": {
        "aliases": ["velocity_x", "vx", "vel_x", "particle_velocity_x"],
        "unit": "velocity",
    },
    "Vy": {
        "aliases": ["velocity_y", "vy", "vel_y", "particle_velocity_y"],
        "unit": "velocity",
    },
    "Vz": {
        "aliases": ["velocity_z", "vz", "vel_z", "particle_velocity_z"],
        "unit": "velocity",
    },
    "divV": {
        "aliases": ["velocity_divergence", "DivV", "div_v", "divergence"],
        "unit": "velocity_divergence",
        "npy_name": "DivV",
    },
    # ---- Thermodynamic ------------------------------------------------------
    "Density": {
        "aliases": ["density", "densitites", "Den", "rho"],  # 'densitites' kept for compat
        "unit": "density",
        "npy_name": "Den",
    },
    "Pressure": {
        "aliases": ["pressure", "P"],
        "unit": "pressure",
        "npy_name": "P",
    },
    "Temperature": {
        "aliases": ["temperature", "T", "temp"],
        "unit": "temperature",
        "npy_name": "T",
    },
    "InternalEnergy": {
        "aliases": ["internal_energy", "IE", "specific_internal_energy", "sie"],
        "unit": "specific_energy",
        "npy_name": "IE",
    },
    "tracers/Entropy": {
        "aliases": ["entropy", "Entropy", "S"],
        "unit": "specific_entropy",
        "npy_name": "Entropy",
    },
    "Dissipation": {
        "aliases": ["dissipation", "Diss", "dissipation_rate"],
        "unit": "dissipation",
        "npy_name": "Diss",
    },
    # ---- Volume -------------------------------------------------------------
    "Volume": {
        "aliases": ["volume", "Vol", "volumes"],
        "unit": "volume",
        "npy_name": "Vol",
    },
    # ---- Radiation ----------------------------------------------------------
    "Erad": {
        "aliases": ["radiation_energy", "Rad", "E_rad", "Erad"],
        "unit": "specific_energy",
        "npy_name": "Rad",
    },
    # ---- Gradients: Pressure ------------------------------------------------
    "DpDx": {
        "aliases": ["pressure_gradient_x", "grad_p_x", "dp_dx"],
        "unit": "pressure_gradient",
    },
    "DpDy": {
        "aliases": ["pressure_gradient_y", "grad_p_y", "dp_dy"],
        "unit": "pressure_gradient",
    },
    "DpDz": {
        "aliases": ["pressure_gradient_z", "grad_p_z", "dp_dz"],
        "unit": "pressure_gradient",
    },
    # ---- Gradients: Density -------------------------------------------------
    "DrhoDx": {
        "aliases": ["density_gradient_x", "grad_rho_x", "drho_dx"],
        "unit": "density_gradient",
    },
    "DrhoDy": {
        "aliases": ["density_gradient_y", "grad_rho_y", "drho_dy"],
        "unit": "density_gradient",
    },
    "DrhoDz": {
        "aliases": ["density_gradient_z", "grad_rho_z", "drho_dz"],
        "unit": "density_gradient",
    },
    # ---- Gradients: Specific Internal Energy --------------------------------
    "DsieDx": {
        "aliases": ["sie_gradient_x", "grad_sie_x", "dsie_dx"],
        "unit": "sie_gradient",
    },
    "DsieDy": {
        "aliases": ["sie_gradient_y", "grad_sie_y", "dsie_dy"],
        "unit": "sie_gradient",
    },
    "DsieDz": {
        "aliases": ["sie_gradient_z", "grad_sie_z", "dsie_dz"],
        "unit": "sie_gradient",
    },
    # ---- Tracers / composition ----------------------------------------------
    "tracers/Star": {
        "aliases": ["star_fraction", "Star", "star", "star_ratio", "stellar_fraction"],
        "unit": "dimensionless",
        "npy_name": "Star",
    },
    "tracers/WasRemoved": {
        "aliases": ["was_removed", "WasRemoved", "removed"],
        "unit": "unknown",
    },
    # ---- Unused / legacy ----------------------------------------------------
    "Eg_0": {
        "aliases": [],
        "unit": "unknown",
    },
    "stickers": {
        "aliases": ["stickers"],
        "unit": "unknown",
    },
    # ---- NPY-only fields ----------------------------------------------------
    "Mass": {
        "npy_only": True,
        "aliases": ["mass", "masses", "particle_mass", "m"],
        "unit": "mscale",
    },
}
