"""
Hoffman quadratic failure criterion (/FAIL/HOFFMAN).

Fortran origin: ``starter/source/materials/fail/hoffman/hm_read_fail_hoffman.F``,
``engine/source/materials/fail/hoffman/fail_hoffman_s.F`` (solids) and
``engine/source/materials/fail/hoffman/fail_hoffman_c.F`` (shells).
Failure model IRUPT = 46.

Theory:
-------
Quadratic failure criterion for orthotropic materials accounting for differences
between tensile and compressive strengths.

Parameters:
  SIGMA_1T : Tensile strength in direction 1 (X_t)
  SIGMA_2T : Tensile strength in direction 2 (Y_t)
  SIGMA_1C : Compressive strength in direction 1 (X_c)
  SIGMA_2C : Compressive strength in direction 2 (Y_c)
  SIGMA_12 : In-plane shear strength (S_12)
  TAU_MAX  : Relaxation time for softening (default inf)
  FCUT     : Cutoff frequency for stress filtering (optional)

Coefficients:
  F1  = 1/SIGMA_1T - 1/SIGMA_1C
  F2  = 1/SIGMA_2T - 1/SIGMA_2C
  F11 = 1/(SIGMA_1T * SIGMA_1C)
  F22 = 1/(SIGMA_2T * SIGMA_2C)
  F66 = 1/(SIGMA_12**2)
  F12 = -F11  (Hoffman specific relation)

Shell quadratic form:
  A = F11*sxx**2 + F22*syy**2 + F66*sxy**2 + F12*sxx*syy
  B = F1*sxx + F2*syy
  FINDEX = max(0, A + B)

Solid quadratic form (transversely isotropic in direction 1):
  A = F11*sxx**2 + F22*(syy**2 + szz**2) + F66*(sxy**2 + szx**2) + F12*(sxx*syy + sxx*szz)
  B = F1*sxx + F2*(syy + szz)
  FINDEX = max(0, A + B)

Reserve factor:
  R = (-B + sqrt(B**2 + 4*A)) / (2*A)

Failure condition:
  D = min(1.0, max(D, FINDEX))
  Point fails when D >= 1.0.
"""

from __future__ import annotations

import numpy as np

_INF = 1e30
_TINY = 1e-20


def _get_param(params: dict, keys: list[str], default: float = _INF) -> float:
    for k in keys:
        if k in params:
            val = params[k]
            if val is not None:
                return float(val)
    return default


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance Hoffman failure for a solid element slice; returns broken mask."""
    p = fail.params
    sigt1 = max(_get_param(p, ["sigma_1t", "Sigma_1t", "xt", "Xt", "s1t"], _INF), _TINY)
    sigt2 = max(_get_param(p, ["sigma_2t", "Sigma_2t", "yt", "Yt", "s2t"], _INF), _TINY)
    sigc1 = max(_get_param(p, ["sigma_1c", "Sigma_1c", "xc", "Xc", "s1c"], _INF), _TINY)
    sigc2 = max(_get_param(p, ["sigma_2c", "Sigma_2c", "yc", "Yc", "s2c"], _INF), _TINY)
    sigt12 = max(_get_param(p, ["sigma_12", "Sigma_12", "s", "S", "s12"], _INF), _TINY)

    f1 = 1.0 / sigt1 - 1.0 / sigc1
    f2 = 1.0 / sigt2 - 1.0 / sigc2
    f11 = 1.0 / (sigt1 * sigc1)
    f22 = 1.0 / (sigt2 * sigc2)
    f66 = 1.0 / (sigt12 ** 2)
    f12 = -f11

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    szz = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)
    sxy = sig_arr[:, 3] if sig_arr.shape[1] > 3 else np.zeros_like(sxx)
    szx = sig_arr[:, 5] if sig_arr.shape[1] > 5 else np.zeros_like(sxx)

    a = f11 * (sxx ** 2) + f22 * (syy ** 2 + szz ** 2) + f66 * (sxy ** 2 + szx ** 2) + f12 * (sxx * syy + sxx * szz)
    b = f1 * sxx + f2 * (syy + szz)
    findex = np.maximum(0.0, a + b)

    dama[:] = np.minimum(1.0, np.maximum(dama, findex))
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Hoffman failure for a shell layer; returns broken mask."""
    p = fail.params
    sigt1 = max(_get_param(p, ["sigma_1t", "Sigma_1t", "xt", "Xt", "s1t"], _INF), _TINY)
    sigt2 = max(_get_param(p, ["sigma_2t", "Sigma_2t", "yt", "Yt", "s2t"], _INF), _TINY)
    sigc1 = max(_get_param(p, ["sigma_1c", "Sigma_1c", "xc", "Xc", "s1c"], _INF), _TINY)
    sigc2 = max(_get_param(p, ["sigma_2c", "Sigma_2c", "yc", "Yc", "s2c"], _INF), _TINY)
    sigt12 = max(_get_param(p, ["sigma_12", "Sigma_12", "s", "S", "s12"], _INF), _TINY)

    f1 = 1.0 / sigt1 - 1.0 / sigc1
    f2 = 1.0 / sigt2 - 1.0 / sigc2
    f11 = 1.0 / (sigt1 * sigc1)
    f22 = 1.0 / (sigt2 * sigc2)
    f66 = 1.0 / (sigt12 ** 2)
    f12 = -f11

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)

    a = f11 * (sxx ** 2) + f22 * (syy ** 2) + f66 * (sxy ** 2) + f12 * (sxx * syy)
    b = f1 * sxx + f2 * syy
    findex = np.maximum(0.0, a + b)

    dama[:] = np.minimum(1.0, np.maximum(dama, findex))
    return dama >= 1.0
