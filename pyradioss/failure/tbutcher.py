"""
Tuler-Butcher dynamic spall failure criterion (/FAIL/TBUTCHER).

Fortran origin: ``engine/source/materials/fail/tuler_butcher/fail_tbutcher_s.F`` (solids)
and ``fail_tbutcher_c.F`` (shells); reader
``starter/source/materials/fail/tuler_butcher/hm_read_fail_tbutcher.F``.

Theory (F.R. Tuler, B.M. Butcher, Int. J. Fract. Mech. 4 (1968) 431-437)
-------------------------------------------------------------------------
Cumulative stress-time damage criterion for dynamic spalling and tensile fracture:

    D = integral max(0, sigma_1 - sigma_0)^lambda dt

where:
- sigma_1 is the maximum principal tensile stress.
- sigma_0 is the threshold tensile stress (Sigma_r).
- lambda is the stress-exponent (Lambda / TBA).
- K is the critical damage threshold (K / TBK).

Damage accumulation:
    dD = dt * max(0, sigma_1 - sigma_0)^lambda
    D += dD

Failure condition:
    D >= K -> element/layer broken.
"""

from __future__ import annotations

import numpy as np


def _get_param(params: dict, keys: list[str], default: float = 0.0) -> float:
    for k in keys:
        if k in params:
            val = params[k]
            if val is not None:
                return float(val)
    return default


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance Tuler-Butcher damage for a solid element slice; returns broken mask."""
    p = fail.params
    tba = _get_param(p, ["lambda", "Lambda", "tba", "TBA"], 1.0)
    tbk = _get_param(p, ["k", "K", "tbk", "TBK"], 1e30)
    sigr = _get_param(p, ["sigma_r", "Sigma_r", "sigr", "SIGR", "s0"], 0.0)

    sig_arr = np.asarray(sig, dtype=float)
    n = len(sig_arr)

    # Construct symmetric 3x3 stress tensor for each element
    sigma_3x3 = np.zeros((n, 3, 3), dtype=float)
    sigma_3x3[:, 0, 0] = sig_arr[:, 0]  # sxx
    sigma_3x3[:, 1, 1] = sig_arr[:, 1]  # syy
    sigma_3x3[:, 2, 2] = sig_arr[:, 2]  # szz
    sigma_3x3[:, 0, 1] = sigma_3x3[:, 1, 0] = sig_arr[:, 3]  # sxy
    if sig_arr.shape[1] > 4:
        sigma_3x3[:, 1, 2] = sigma_3x3[:, 2, 1] = sig_arr[:, 4]  # syz
    if sig_arr.shape[1] > 5:
        sigma_3x3[:, 2, 0] = sigma_3x3[:, 0, 2] = sig_arr[:, 5]  # szx

    # Eigenvalues sorted ascending: [sigma_3, sigma_2, sigma_1]
    eigvals = np.linalg.eigvalsh(sigma_3x3)
    sig1 = eigvals[:, 2]

    # Damage accumulation
    over_stress = np.maximum(0.0, sig1 - sigr)
    d_dama = dt * (over_stress ** tba)
    dama[:] += d_dama

    return dama >= tbk


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Tuler-Butcher damage for a shell layer; returns broken mask."""
    sig_arr = np.asarray(sig, dtype=float)
    n = len(sig_arr)
    sig_3d = np.zeros((n, 6), dtype=float)
    sig_3d[:, 0] = sig_arr[:, 0]  # sxx
    sig_3d[:, 1] = sig_arr[:, 1]  # syy
    if sig_arr.shape[1] > 2:
        sig_3d[:, 3] = sig_arr[:, 2]  # sxy
    if sig_arr.shape[1] > 3:
        sig_3d[:, 4] = sig_arr[:, 3]  # syz
    if sig_arr.shape[1] > 4:
        sig_3d[:, 5] = sig_arr[:, 4]  # szx

    return solid_step(fail, sig_3d, d_epsp, deps, dt, dama, tstar=tstar)
