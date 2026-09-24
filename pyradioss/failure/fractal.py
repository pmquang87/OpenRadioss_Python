"""
Fractal Damage Precursor Model (/FAIL/FRACTAL_DMG).

Fortran origin: ``starter/source/materials/fail/fractal/hm_read_fractal_dmg.F90``,
``starter/source/materials/fail/fractal/random_walk_dmg.F90``.
Failure model IRUPT = 12.

Theory:
-------
Generates stochastic initial defect/damage patterns on shell elements via
Diffusion-Limited Aggregation (Brownian random walks on the mesh dual graph).

In the engine, the pre-seeded damage acts as initial damage.
"""

from __future__ import annotations

import numpy as np


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance Fractal damage (pre-seeded initial damage); returns broken mask."""
    p = fail.params
    initial_dam = float(p.get("damage", p.get("Damage", 0.1)))
    # If not already initialized, seed initial damage
    if np.all(dama == 0.0):
        dama[:] = initial_dam
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Fractal damage for a shell layer; returns broken mask."""
    return solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
