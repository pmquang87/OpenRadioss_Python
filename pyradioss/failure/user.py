"""
User Failure Subroutine Interface (/FAIL/USER1, /FAIL/USER2, /FAIL/USER3).

Fortran origin: ``starter/source/materials/fail/failuser/hm_read_fail_user.F``,
``engine/source/materials/mat_share/mmain.F90``, ``mulawc.F90``, and ``dyn_userlib.c``.
Failure models IRUPT = 4, 5, 6.

Provides interface hooks for user-programmed failure logic.
"""

from __future__ import annotations

import numpy as np


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """User-defined solid failure step."""
    # Check if a custom python hook is provided in fail.params
    hook = fail.params.get("hook")
    if callable(hook):
        return hook(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    # Default: evaluate plastic strain against optional threshold
    eps_max = fail.params.get("eps_max", 1.0e30)
    dama[:] = np.minimum(1.0, dama + np.asarray(d_epsp, dtype=float) / max(eps_max, 1e-20))
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """User-defined shell failure step."""
    hook = fail.params.get("hook")
    if callable(hook):
        return hook(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    eps_max = fail.params.get("eps_max", 1.0e30)
    dama[:] = np.minimum(1.0, dama + np.asarray(d_epsp, dtype=float) / max(eps_max, 1e-20))
    return dama >= 1.0
