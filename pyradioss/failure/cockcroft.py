"""Cockcroft-Latham ductile fracture criterion (/FAIL/COCKCROFT).

Fortran origin:
- ``engine/source/materials/fail/cockroft_latham/fail_cockroft_s.F`` (solids)
- ``engine/source/materials/fail/cockroft_latham/fail_cockroft_c.F`` (shells)
- ``starter/source/materials/fail/cockroft_latham/hm_read_fail_cockcroft.F`` (reader)
- CFG: ``radioss2020/FAIL/fail_cockcroft.cfg``, ``radioss2025/FAIL/fail_cockcroft.cfg``

Theory (M.G. Cockcroft, D.J. Latham, J. Inst. Metals 96 (1968) 33-39)
---------------------------------------------------------------------
Fracture occurs when the accumulated work modified by the maximum principal
tensile stress reaches a critical threshold:

    D = integral <sigma_1> d(eps_p) / W_crit

or normalized form:

    D = integral (<sigma_1> / sigma_vm) d(eps_p) / W_crit

where:
- sigma_1 is the first (maximum) principal stress.
- <sigma_1> = max(sigma_1, 0) is the tensile cut-off (Macaulay bracket).
  Under hydrostatic compression (sigma_1 < 0), <sigma_1> = 0, giving zero damage.
- d(eps_p) is the equivalent plastic strain increment (for C0 > 0)
  or total equivalent strain increment (for C0 < 0).
- W_crit = |C0| is the critical fracture work threshold.
- sigma_1 may be smoothed via an exponential moving average (EMA) filter:
      sigma_1 = alpha * sigma_1 + (1 - alpha) * sigma_1_prev
  where alpha = 1.0 means no filtering.

Failure condition:
    Integration point breaks when D >= 1.0.
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-20
_INF = 1e20


def _max_principal_stress_3d(sxx: np.ndarray, syy: np.ndarray, szz: np.ndarray,
                             sxy: np.ndarray, syz: np.ndarray, szx: np.ndarray) -> np.ndarray:
    """Compute maximum principal stress sigma_1 for a 3D stress state using Cardano's formula.

    Matches OpenRadioss fail_cockroft_s.F lines 187-217.
    """
    n = len(sxx)
    s1 = np.zeros(n, dtype=float)

    for i in range(n):
        xx, yy, zz = sxx[i], syy[i], szz[i]
        xy, yz, zx = sxy[i], syz[i], szx[i]

        i1 = xx + yy + zz
        i2 = xx * yy + yy * zz + zz * xx - xy * xy - zx * zx - yz * yz
        i3 = (xx * yy * zz - xx * yz * yz - yy * zx * zx - zz * xy * xy
              + 2.0 * xy * zx * yz)

        q = (3.0 * i2 - i1 * i1) / 9.0
        r = (2.0 * i1 * i1 * i1 - 9.0 * i1 * i2 + 27.0 * i3) / 54.0

        q_cubed_neg = max(1e-20, -q ** 3)
        r_inter = min(r / np.sqrt(q_cubed_neg), 1.0)
        phi = np.arccos(max(r_inter, -1.0))

        sqrt_neg_q = np.sqrt(max(0.0, -q))
        root1 = 2.0 * sqrt_neg_q * np.cos(phi / 3.0) + i1 / 3.0
        root2 = 2.0 * sqrt_neg_q * np.cos((phi + 2.0 * np.pi) / 3.0) + i1 / 3.0
        root3 = 2.0 * sqrt_neg_q * np.cos((phi + 4.0 * np.pi) / 3.0) + i1 / 3.0

        s1[i] = max(root1, root2, root3)

    return s1


def _von_mises(sxx: np.ndarray, syy: np.ndarray, szz: np.ndarray,
               sxy: np.ndarray, syz: np.ndarray, szx: np.ndarray) -> np.ndarray:
    """Compute Von Mises equivalent stress."""
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - xx_sq if 'xx_sq' in locals() else (szz - sxx) ** 2)
                          + 6.0 * (sxy ** 2 + syz ** 2 + szx ** 2)))


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance Cockcroft-Latham damage for a 3D solid slice.

    Parameters
    ----------
    fail : Failure object
        Model parameters:
        - c0 / w_crit : critical work threshold
        - alpha / ema : exponential filter (default 1.0)
        - normalized / inorm : optional bool for normalized Cockcroft-Latham
    sig : ndarray of shape (m, 6)
        Stress components [s_xx, s_yy, s_zz, s_xy, s_yz, s_zx].
    d_epsp : ndarray of shape (m,)
        Plastic strain increment.
    deps : ndarray of shape (m, 6) or None
        Total strain increment.
    dt : float
        Time step.
    dama : ndarray of shape (m,)
        Persistent accumulated damage array.
    tstar : optional

    Returns
    -------
    ndarray of bool
        Mask of broken integration points.
    """
    p = fail.params
    c0 = float(p.get("c0", p.get("C0", p.get("w_crit", p.get("W_crit", _INF)))))
    w_crit = max(abs(c0), _TINY)
    alpha = float(p.get("alpha", p.get("Alpha", p.get("ema", p.get("EMA", 1.0)))))
    normalized = bool(p.get("normalized", p.get("inorm", p.get("Inorm", False))))

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    szz = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)
    sxy = sig_arr[:, 3] if sig_arr.shape[1] > 3 else np.zeros_like(sxx)
    syz = sig_arr[:, 4] if sig_arr.shape[1] > 4 else np.zeros_like(sxx)
    szx = sig_arr[:, 5] if sig_arr.shape[1] > 5 else np.zeros_like(sxx)

    # 1. Strain increment d_eeq
    if c0 < 0.0 and deps is not None:
        deps_arr = np.asarray(deps, dtype=float)
        exx = deps_arr[:, 0]
        eyy = deps_arr[:, 1]
        ezz = deps_arr[:, 2] if deps_arr.shape[1] > 2 else np.zeros_like(exx)
        exy = deps_arr[:, 3] if deps_arr.shape[1] > 3 else np.zeros_like(exx)
        eyz = deps_arr[:, 4] if deps_arr.shape[1] > 4 else np.zeros_like(exx)
        ezx = deps_arr[:, 5] if deps_arr.shape[1] > 5 else np.zeros_like(exx)
        e_hyd = (exx + eyy + ezz) / 3.0
        e11, e22, e33 = exx - e_hyd, eyy - e_hyd, ezz - e_hyd
        e12, e23, e13 = 0.5 * exy, 0.5 * eyz, 0.5 * ezx
        eeq = (2.0 / 3.0) * (e11 ** 2 + e22 ** 2 + e33 ** 2 + 2.0 * (e12 ** 2 + e23 ** 2 + e13 ** 2))
        d_eeq = np.sqrt(np.maximum(0.0, eeq))
    else:
        d_eeq = np.asarray(d_epsp, dtype=float) if d_epsp is not None else np.zeros_like(sxx)

    # 2. Maximum principal stress sigma_1
    s1 = _max_principal_stress_3d(sxx, syy, szz, sxy, syz, szx)

    # 3. Tensile work increment
    s1_pos = np.maximum(0.0, s1)

    if normalized:
        svm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2
                             + 6.0 * (sxy ** 2 + syz ** 2 + szx ** 2)))
        ratio = np.where(svm > _TINY, s1_pos / np.maximum(_TINY, svm), 0.0)
        d_damage = ratio * d_eeq / w_crit
    else:
        d_damage = s1_pos * d_eeq / w_crit

    dama[:] = np.minimum(1.0, dama + d_damage)
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Cockcroft-Latham damage for a plane-stress shell layer.

    Parameters
    ----------
    fail : Failure object
    sig : ndarray of shape (m, 3) or (m, 5)
        Plane stress tensor [s_xx, s_yy, s_xy, (s_yz, s_zx)].
    d_epsp : ndarray of shape (m,)
    deps : ndarray
    dt : float
    dama : ndarray of shape (m,)
    tstar : optional
    eps_tot : optional

    Returns
    -------
    ndarray of bool
        Broken mask.
    """
    p = fail.params
    c0 = float(p.get("c0", p.get("C0", p.get("w_crit", p.get("W_crit", _INF)))))
    w_crit = max(abs(c0), _TINY)
    normalized = bool(p.get("normalized", p.get("inorm", p.get("Inorm", False))))

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)

    # 1. Strain increment d_eeq
    if c0 < 0.0 and deps is not None:
        deps_arr = np.asarray(deps, dtype=float)
        exx = deps_arr[:, 0]
        eyy = deps_arr[:, 1]
        exy = deps_arr[:, 2] if deps_arr.shape[1] > 2 else np.zeros_like(exx)
        e_hyd = (exx + eyy) / 3.0
        e11, e22, e33 = exx - e_hyd, eyy - e_hyd, -e_hyd
        e12 = 0.5 * exy
        eeq = (2.0 / 3.0) * (e11 ** 2 + e22 ** 2 + e33 ** 2 + 2.0 * (e12 ** 2))
        d_eeq = np.sqrt(np.maximum(0.0, eeq))
    else:
        d_eeq = np.asarray(d_epsp, dtype=float) if d_epsp is not None else np.zeros_like(sxx)

    # 2. Maximum principal stress for plane stress (fail_cockroft_c.F lines 144-146)
    sig_a = 0.5 * (sxx + syy)
    sig_b = np.sqrt(0.25 * (sxx - syy) ** 2 + sxy ** 2)
    s1 = sig_a + sig_b

    # 3. Tensile work increment
    s1_pos = np.maximum(0.0, s1)

    if normalized:
        svm = np.sqrt(sxx ** 2 + syy ** 2 - sxx * syy + 3.0 * (sxy ** 2))
        ratio = np.where(svm > _TINY, s1_pos / np.maximum(_TINY, svm), 0.0)
        d_damage = ratio * d_eeq / w_crit
    else:
        d_damage = s1_pos * d_eeq / w_crit

    dama[:] = np.minimum(1.0, dama + d_damage)
    return dama >= 1.0
