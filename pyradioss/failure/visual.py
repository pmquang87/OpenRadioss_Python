"""
Visual Failure Indicator Model (/FAIL/VISUAL).

Fortran origin: ``starter/source/materials/fail/visual/hm_read_fail_visual.F``,
``engine/source/materials/fail/visual/fail_visual_s.F`` (solids) and
``engine/source/materials/fail/visual/fail_visual_c.F`` (shells).
Failure model IRUPT = 36.

Theory:
-------
Diagnostic and visual failure indicator designed to track peak principal stress
or strain without eroding/deleting elements:
  D = (E_11 - C_min) / (C_max - C_min)

Note: Elements are never deleted by this model.
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


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance Visual failure indicator for a solid slice; returns all False (no deletion)."""
    p = fail.params
    c_min = _get_param(p, ["c_min", "C_min", "cmin"], 0.0)
    c_max = _get_param(p, ["c_max", "C_max", "cmax"], 1.0e8)

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    szz = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)

    sig_max = np.maximum.reduce([sxx, syy, szz])
    denom = max(c_max - c_min, _TINY)
    d = np.clip((sig_max - c_min) / denom, 0.0, 1.0)
    dama[:] = np.maximum(dama, d)
    return np.zeros(len(dama), dtype=bool)


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Visual failure indicator for a shell layer; returns all False (no deletion)."""
    p = fail.params
    c_min = _get_param(p, ["c_min", "C_min", "cmin"], 0.0)
    c_max = _get_param(p, ["c_max", "C_max", "cmax"], 1.0e8)

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)

    center = (sxx + syy) / 2.0
    radius = np.sqrt(((sxx - syy) / 2.0) ** 2 + sxy ** 2)
    sig_max = center + radius

    denom = max(c_max - c_min, _TINY)
    d = np.clip((sig_max - c_min) / denom, 0.0, 1.0)
    dama[:] = np.maximum(dama, d)
    return np.zeros(len(dama), dtype=bool)
