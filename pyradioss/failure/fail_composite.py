"""
3D Anisotropic Composite Failure Criterion (/FAIL/COMPOSITE).

Fortran origin: ``starter/source/materials/fail/composite/hm_read_fail_composite.F90``,
``engine/source/materials/fail/composite/fail_composite_s.F90`` (solids) and
``engine/source/materials/fail/composite/fail_composite_c.F90`` (shells).
Failure model IRUPT = 51.

Theory:
-------
Multi-mode failure criterion for fiber-reinforced composites with up to 9 distinct
failure modes (tension/compression in directions 1, 2, 3 and shears 12, 23, 31).

Parameters:
  SIGMA_1T, SIGMA_1C : Longitudinal tensile/compressive strength
  SIGMA_2T, SIGMA_2C : Transversal tensile/compressive strength
  SIGMA_12           : In-plane shear strength
  SIGMA_3T, SIGMA_3C : Out-of-plane tensile/compressive strength (solids)
  SIGMA_23, SIGMA_31 : Out-of-plane shear strengths (solids)
  BETA               : Shear interaction parameter
  EXPN               : Exponent n (default 1.0)
  TAU_MAX            : Relaxation time

Modes:
  Mode 1: Tension 1   (|s11|/Xt)**n + beta*(|s12|/S12)**n
  Mode 2: Compress 1  (|s11|/Xc)**n + beta*(|s12|/S12)**n
  Mode 3: Tension 2   (|s22|/Yt)**n + beta*(|s12|/S12)**n
  Mode 4: Compress 2  (|s22|/Yc)**n + beta*(|s12|/S12)**n
  Mode 5: Shear 12    (|s12|/S12)**n
  Mode 6: Tension 3   (|s33|/Zt)**n + beta*(|s23|/S23)**n + beta*(|s31|/S31)**n
  Mode 7: Compress 3  (|s33|/Zc)**n + beta*(|s23|/S23)**n + beta*(|s31|/S31)**n
  Mode 8: Shear 23    (|s23|/S23)**n
  Mode 9: Shear 31    (|s31|/S31)**n

Global damage:
  D = min(1.0, max(Mode_k))
  Fails when D >= 1.0.
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
    """Advance Composite failure for a solid element slice; returns broken mask."""
    p = fail.params
    sigt1 = max(_get_param(p, ["sigma_1t", "Sigma_1t", "xt", "Xt", "s1t"], _INF), _TINY)
    sigc1 = max(_get_param(p, ["sigma_1c", "Sigma_1c", "xc", "Xc", "s1c"], _INF), _TINY)
    sigt2 = max(_get_param(p, ["sigma_2t", "Sigma_2t", "yt", "Yt", "s2t"], _INF), _TINY)
    sigc2 = max(_get_param(p, ["sigma_2c", "Sigma_2c", "yc", "Yc", "s2c"], _INF), _TINY)
    sigt12 = max(_get_param(p, ["sigma_12", "Sigma_12", "s12", "S12", "s", "S"], _INF), _TINY)

    sigt3 = max(_get_param(p, ["sigma_3t", "Sigma_3t", "zt", "Zt", "s3t"], _INF), _TINY)
    sigc3 = max(_get_param(p, ["sigma_3c", "Sigma_3c", "zc", "Zc", "s3c"], _INF), _TINY)
    sigt23 = max(_get_param(p, ["sigma_23", "Sigma_23", "s23", "S23"], _INF), _TINY)
    sigt31 = max(_get_param(p, ["sigma_31", "Sigma_31", "s31", "S31"], _INF), _TINY)

    beta = _get_param(p, ["beta", "Beta"], 0.0)
    expn = _get_param(p, ["expn", "EXPN", "n", "N"], 1.0)

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    szz = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)
    sxy = sig_arr[:, 3] if sig_arr.shape[1] > 3 else np.zeros_like(sxx)
    syz = sig_arr[:, 4] if sig_arr.shape[1] > 4 else np.zeros_like(sxx)
    szx = sig_arr[:, 5] if sig_arr.shape[1] > 5 else np.zeros_like(sxx)

    abs_sxx = np.abs(sxx)
    abs_syy = np.abs(syy)
    abs_szz = np.abs(szz)
    abs_sxy = np.abs(sxy)
    abs_syz = np.abs(syz)
    abs_szx = np.abs(szx)

    # In-plane terms
    sh12_term = beta * ((abs_sxy / sigt12) ** expn)
    sh23_term = beta * ((abs_syz / sigt23) ** expn)
    sh31_term = beta * ((abs_szx / sigt31) ** expn)

    mode1 = np.where(sxx >= 0.0, (abs_sxx / sigt1) ** expn + sh12_term, 0.0)
    mode2 = np.where(sxx < 0.0, (abs_sxx / sigc1) ** expn + sh12_term, 0.0)
    mode3 = np.where(syy >= 0.0, (abs_syy / sigt2) ** expn + sh12_term, 0.0)
    mode4 = np.where(syy < 0.0, (abs_syy / sigc2) ** expn + sh12_term, 0.0)
    mode5 = (abs_sxy / sigt12) ** expn

    mode6 = np.where(szz >= 0.0, (abs_szz / sigt3) ** expn + sh23_term + sh31_term, 0.0)
    mode7 = np.where(szz < 0.0, (abs_szz / sigc3) ** expn + sh23_term + sh31_term, 0.0)
    mode8 = (abs_syz / sigt23) ** expn
    mode9 = (abs_szx / sigt31) ** expn

    findex = np.maximum.reduce([mode1, mode2, mode3, mode4, mode5, mode6, mode7, mode8, mode9])
    dama[:] = np.minimum(1.0, np.maximum(dama, findex))
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Composite failure for a shell layer; returns broken mask."""
    p = fail.params
    sigt1 = max(_get_param(p, ["sigma_1t", "Sigma_1t", "xt", "Xt", "s1t"], _INF), _TINY)
    sigc1 = max(_get_param(p, ["sigma_1c", "Sigma_1c", "xc", "Xc", "s1c"], _INF), _TINY)
    sigt2 = max(_get_param(p, ["sigma_2t", "Sigma_2t", "yt", "Yt", "s2t"], _INF), _TINY)
    sigc2 = max(_get_param(p, ["sigma_2c", "Sigma_2c", "yc", "Yc", "s2c"], _INF), _TINY)
    sigt12 = max(_get_param(p, ["sigma_12", "Sigma_12", "s12", "S12", "s", "S"], _INF), _TINY)

    beta = _get_param(p, ["beta", "Beta"], 0.0)
    expn = _get_param(p, ["expn", "EXPN", "n", "N"], 1.0)

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)

    abs_sxx = np.abs(sxx)
    abs_syy = np.abs(syy)
    abs_sxy = np.abs(sxy)

    sh12_term = beta * ((abs_sxy / sigt12) ** expn)

    mode1 = np.where(sxx >= 0.0, (abs_sxx / sigt1) ** expn + sh12_term, 0.0)
    mode2 = np.where(sxx < 0.0, (abs_sxx / sigc1) ** expn + sh12_term, 0.0)
    mode3 = np.where(syy >= 0.0, (abs_syy / sigt2) ** expn + sh12_term, 0.0)
    mode4 = np.where(syy < 0.0, (abs_syy / sigc2) ** expn + sh12_term, 0.0)
    mode5 = (abs_sxy / sigt12) ** expn

    findex = np.maximum.reduce([mode1, mode2, mode3, mode4, mode5])
    dama[:] = np.minimum(1.0, np.maximum(dama, findex))
    return dama >= 1.0
