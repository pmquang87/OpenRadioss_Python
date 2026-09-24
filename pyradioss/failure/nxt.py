"""
Forming Limit Stress Failure Model (/FAIL/NXT).

Fortran origin: ``starter/source/materials/fail/nxt/hm_read_fail_nxt.F``,
``engine/source/materials/fail/nxt/fail_nxt_c.F`` (shells only).
Failure model IRUPT = 25.

Theory:
-------
Sheet metal forming limit criterion comparing in-plane principal stresses
(sigma_1, sigma_2) normalized by current yield stress against forming limit curves:
  sigma_sr : Shear-tension forming limit curve
  sigma_3d : Biaxial forming limit curve

Instability parameter:
  lambda_nxt = 1.0 + (sigma_1_norm - sigma_sr) / (sigma_3d - sigma_sr)

Rupture occurs when lambda_nxt >= 2.0 (normalized damage D = lambda_nxt / 2.0 >= 1.0).
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-20


def _get_param(params: dict, keys: list[str], default: float = 0.0) -> float:
    for k in keys:
        if k in params:
            val = params[k]
            if val is not None:
                return float(val)
    return default


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance NXT failure for a shell layer; returns broken mask."""
    p = fail.params
    sigma_y0 = _get_param(p, ["sigma_y0", "sig_y0", "hardm"], 3.0e8)
    limit_sr = _get_param(p, ["limit_sr", "sigma_sr"], 1.1)
    limit_3d = _get_param(p, ["limit_3d", "sigma_3d"], 1.3)

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)

    center = (sxx + syy) / 2.0
    radius = np.sqrt(((sxx - syy) / 2.0) ** 2 + sxy ** 2)
    s1 = center + radius
    s2 = center - radius

    s1_norm = s1 / max(sigma_y0, _TINY)

    # In quadrant 1 or 2, calculate instability factor
    denom = max(limit_3d - limit_sr, _TINY)
    lam = np.where((s1 < 0) & (s2 < 0), 0.0, 1.0 + (s1_norm - limit_sr) / denom)
    lam = np.clip(lam, 0.0, 2.0)

    d = lam / 2.0
    dama[:] = np.maximum(dama, d)
    return dama >= 1.0


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """NXT solid step compatibility."""
    sig_arr = np.asarray(sig, dtype=float)
    return shell_step(fail, sig_arr[:, :3], d_epsp, deps, dt, dama, tstar=tstar)
