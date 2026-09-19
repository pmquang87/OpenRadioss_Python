"""
Mullins Effect Pseudo-Elastic Damage Model (/FAIL/MULLINS_OR).

Fortran origin: ``starter/source/materials/fail/mullins_or/hm_read_fail_mullins_or.F``,
``engine/source/materials/fail/mullins_or/fail_mullins_OR_s.F`` (solids).
Failure model IRUPT = 33.

Theory (Ogden & Roxburgh, Proc. R. Soc. Lond. A 1999):
------------------------------------------------------
Pseudo-elastic model of the Mullins effect in rubbers.
Historical maximum strain energy density W_max is tracked.

Damage softening factor eta:
  eta = 1.0 - (1.0/r) * erf((W_max - W) / (m + beta * W_max))
  D = 1.0 - eta

Note: The Mullins model softens deviatoric stresses without element deletion.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

_TINY = 1e-20


def _get_param(params: dict, keys: list[str], default: float = 0.0) -> float:
    for k in keys:
        if k in params:
            val = params[k]
            if val is not None:
                return float(val)
    return default


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance Mullins damage for a solid element slice; returns broken mask (always False)."""
    p = fail.params
    r = _get_param(p, ["r", "COEFR", "coefr"], 1.0)
    beta = _get_param(p, ["beta", "BETA", "betaf"], 0.0)
    m = _get_param(p, ["m", "COEFM", "coefm"], 0.0)

    sig_arr = np.asarray(sig, dtype=float)
    deps_arr = np.asarray(deps, dtype=float) if deps is not None else np.zeros((len(dama), 6))

    # Approximate incremental strain energy W ~ sum(sig * deps)
    dW = np.sum(sig_arr * deps_arr, axis=1)
    # Track historical energy in dama
    current_w = np.maximum(0.0, dama + np.maximum(0.0, dW))
    w_max = np.maximum(dama, current_w)

    denom = m + beta * w_max
    arg = np.where(denom > 0, (w_max - current_w) / np.maximum(denom, _TINY), 0.0)
    eta = 1.0 - (1.0 / max(r, _TINY)) * erf(np.clip(arg, 0.0, 5.0))
    d = np.clip(1.0 - eta, 0.0, 1.0)

    dama[:] = w_max
    # Mullins damage never deletes elements
    return np.zeros(len(dama), dtype=bool)


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Mullins shell step compatibility."""
    return np.zeros(len(dama), dtype=bool)
