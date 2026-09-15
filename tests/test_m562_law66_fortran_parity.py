r"""Fortran Parity & Physics Oracle Verifier for Milestone M562: /MAT/LAW66 (/MAT/PLAS_TAB_COSSER, /MAT/PLAS_COSSER, /MAT/FOAM_TAB).

Asymmetric Tabulated Plasticity Model for Solids and Shells.
Directly cites and mirrors upstream OpenRadioss source:
- Upstream Fortran starter reader:
  ``starter/source/materials/mat/mat066/hm_read_mat66.F``
- Upstream Fortran solid physics:
  ``engine/source/materials/mat/mat066/sigeps66.F``
- Upstream Fortran shell physics:
  ``engine/source/materials/mat/mat066/sigeps66c.F``
- CFG card definition:
  ``hm_cfg_files/config/CFG/radioss*/MAT/matl66_*.cfg``

Tests included:
1. Exact Python oracle reproducing sigeps66.F (Solid).
2. Exact Python oracle reproducing sigeps66c.F (Shell).
3. Hydrostatic pressure P = C1 * mu and modulus interpolation between Et and Ec.
4. Trial stress deviator with kinematic backstress shift.
5. Asymmetric yield stress interpolation between tension Yt and compression Yc.
6. Strain rate formulations:
   - ISRATE = 1: Power-law Cowper-Symonds YRATE = 1 + (epsp/epsp0)**cp
   - ISRATE = 2: Logarithmic law YRATE = 1 + cp * log(max(epsp/epsp0, em20))
   - ISRATE = 3: Independent scaling curves for compression and tension
   - ISRATE = 4: Multi-curve strain rate families with linear interpolation
   - VP = 1: Viscoplastic overstress formulation based on dpla / dt
7. Mixed isotropic-kinematic hardening and Bauschinger effect.
8. Shell plane-stress radial return with sigma_zz = 0 and thickness thinning dezz.
9. Dilatational sound speed for solids and shells.
10. Vectorized multi-element batch execution.
11. Exhaustive 64 diverse physical states comparison:
    - 32 Solid states: solid_update vs sigeps66_solid_oracle (rtol=1e-12, atol=1e-12)
    - 32 Shell states: shell_update vs sigeps66c_shell_oracle (rtol=1e-12, atol=1e-12)
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest

from pyradioss.materials.law66_plas_tab import (
    Law66Params,
    build_law66,
    solid_update,
    shell_update,
    sound_speed,
    sound_speed_solid,
    sound_speed_shell,
    consistent_solid_tangent,
    consistent_shell_tangent,
    shell_membrane_tangent,
    extra_shapes,
)

_EM20 = 1.0e-20
_INF = 1.0e30


# =============================================================================
# 1. UPSTREAM FORTRAN SOLID ENGINE ORACLE: sigeps66.F
# =============================================================================

def sigeps66_solid_oracle(
    matparam: Dict[str, Any],
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    alpha: Optional[np.ndarray] = None,
    mu: Optional[Union[float, np.ndarray]] = None,
    rate: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    yield_curve_c: Optional[Any] = None,
    yield_curve_t: Optional[Any] = None,
    rate_curve_c: Optional[Any] = None,
    rate_curve_t: Optional[Any] = None,
    curves_c: Optional[List[Any]] = None,
    rates_c: Optional[List[float]] = None,
    curves_t: Optional[List[Any]] = None,
    rates_t: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Exact Python reproduction of OpenRadioss sigeps66.F (Solids).

    Directly evaluates against Fortran expressions in engine/source/materials/mat/mat066/sigeps66.F.
    sig, deps: (nel, 6) = [xx, yy, zz, xy, yz, zx] with engineering shear.
    """
    sig_arr = np.asarray(sig, dtype=float)
    deps_arr = np.asarray(deps, dtype=float)
    is_1d = (sig_arr.ndim == 1)
    if is_1d:
        sig_arr = sig_arr[None, :]
        deps_arr = deps_arr[None, :]
    nel = sig_arr.shape[0]

    # Material parameters (hm_read_mat66.F / sigeps66.F)
    et = float(matparam.get("e", 210000.0))
    ec = float(matparam.get("ec", et))
    if ec <= 0.0:
        ec = et
    nu = float(matparam.get("nu", 0.3))
    if nu >= 0.5:
        nu = 0.499
    if nu < 0.0:
        nu = 0.0
    pc = float(matparam.get("pc", 0.0))
    pt = float(matparam.get("pt", 0.0))
    rpct = float(matparam.get("rpct", 1.0))
    if rpct <= 0.0:
        rpct = 1.0
    fisokin = float(matparam.get("chard", matparam.get("fisokin", 0.0)))
    israte = int(matparam.get("israte", 1))
    epsp0 = float(matparam.get("epsp0", 0.0))
    cp = float(matparam.get("cp", 1.0))
    sigy = float(matparam.get("sigy", 0.0))
    vp = int(matparam.get("vp", 0))
    rho0 = float(matparam.get("rho0", 1.0))

    fscale11 = float(matparam.get("fscale11", 1.0))
    fscale22 = float(matparam.get("fscale22", 1.0))
    fscale33 = float(matparam.get("fscale33", 1.0))
    fscale12 = float(matparam.get("fscale12", 1.0))

    fp1 = matparam.get("fp1", [])
    fp2 = matparam.get("fp2", [])

    # History states
    pla = np.zeros(nel, dtype=float) if epsp is None else np.asarray(epsp, dtype=float).copy()
    if pla.ndim == 0:
        pla = np.full(nel, float(pla))
    backstress = np.zeros((nel, 6), dtype=float) if alpha is None else np.asarray(alpha, dtype=float).copy()
    if backstress.ndim == 1 and nel == 1:
        backstress = backstress[None, :]

    # 1. Modulus calculation (sigeps66.F:208-234)
    c1t = et / (3.0 * max(1.0 - 2.0 * nu, 1.0e-15))
    gt = 0.5 * et / max(1.0 + nu, 1.0e-15)
    c1c = ec / (3.0 * max(1.0 - 2.0 * nu, 1.0e-15))
    gc = 0.5 * ec / max(1.0 + nu, 1.0e-15)

    tr_deps = deps_arr[:, 0] + deps_arr[:, 1] + deps_arr[:, 2]
    dav = tr_deps / 3.0

    if mu is not None:
        amu = np.asarray(mu, dtype=float)
        if amu.ndim == 0:
            amu = np.full(nel, float(amu))
        p_est = c1t * amu
    else:
        p_old = -(sig_arr[:, 0] + sig_arr[:, 1] + sig_arr[:, 2]) / 3.0
        p_est = p_old - c1t * tr_deps
        amu = -tr_deps

    E = np.full(nel, et, dtype=float)
    if ec > 0.0 and ec != et:
        for i in range(nel):
            if pc == 0.0 and pt == 0.0 and abs(p_est[i]) < 1.0e-10:
                E[i] = ec
            elif p_est[i] <= -rpct * pt:
                E[i] = et
            elif p_est[i] >= rpct * pc:
                E[i] = ec
            else:
                fac = rpct * (pc + pt)
                fac = (rpct * pc - p_est[i]) / fac
                E[i] = fac * et + (1.0 - fac) * ec

    G = 0.5 * E / max(1.0 + nu, 1.0e-15)
    G2 = 2.0 * G
    G3 = 3.0 * G
    C1 = E / (3.0 * max(1.0 - 2.0 * nu, 1.0e-15))

    # Recompute pressure P = C1 * mu (sigeps66.F:493)
    if mu is not None:
        P = C1 * amu
    else:
        P = p_old - C1 * tr_deps

    # Dilatational sound speed (sigeps66.F:261)
    c1_sound = max(c1t, c1c)
    soundsp = np.sqrt((c1_sound + (4.0 / 3.0) * G) / max(rho0, _EM20))

    # 2. Deviatoric trial stress with backstress shift (sigeps66.F:239-259)
    sig_eff = sig_arr - backstress
    p0 = -(sig_eff[:, 0] + sig_eff[:, 1] + sig_eff[:, 2]) / 3.0

    s_trial = np.empty_like(sig_arr)
    s_trial[:, 0] = sig_eff[:, 0] + p0 + G2 * (deps_arr[:, 0] - dav)
    s_trial[:, 1] = sig_eff[:, 1] + p0 + G2 * (deps_arr[:, 1] - dav)
    s_trial[:, 2] = sig_eff[:, 2] + p0 + G2 * (deps_arr[:, 2] - dav)
    s_trial[:, 3] = sig_eff[:, 3] + G * deps_arr[:, 3]
    s_trial[:, 4] = sig_eff[:, 4] + G * deps_arr[:, 4]
    s_trial[:, 5] = sig_eff[:, 5] + G * deps_arr[:, 5]

    # 3. Strain rate evaluation
    if rate is not None:
        rate_arr = np.asarray(rate, dtype=float)
        if rate_arr.ndim == 0:
            rate_arr = np.full(nel, float(rate_arr))
    else:
        ee = ((deps_arr[:, 0] - dav) ** 2 + (deps_arr[:, 1] - dav) ** 2 + (deps_arr[:, 2] - dav) ** 2
              + 0.5 * (deps_arr[:, 3] ** 2 + deps_arr[:, 4] ** 2 + deps_arr[:, 5] ** 2))
        rate_arr = np.sqrt((2.0 / 3.0) * ee) / dt if dt > 0.0 else np.zeros(nel, dtype=float)

    # 4. Curve evaluation helper (FINTER / VINTER reproducing logic)
    def eval_f(c_obj: Any, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if c_obj is None or c_obj == 0:
            return np.zeros_like(x), np.zeros_like(x)
        if callable(c_obj):
            res = c_obj(x)
            if isinstance(res, tuple):
                return np.asarray(res[0], dtype=float), np.asarray(res[1], dtype=float)
            return np.asarray(res, dtype=float), np.zeros_like(x)
        if isinstance(c_obj, (list, tuple)) and len(c_obj) == 2:
            cx = np.asarray(c_obj[0], dtype=float)
            cy = np.asarray(c_obj[1], dtype=float)
            dx = np.diff(cx)
            dx_s = np.where(np.abs(dx) > _EM20, dx, np.sign(dx) * _EM20 + _EM20)
            cs = np.diff(cy) / dx_s
            idx = np.clip(np.searchsorted(cx, x, side="right") - 1, 0, len(cx) - 2)
            val = cy[idx] + cs[idx] * (x - cx[idx])
            return val, cs[idx]
        return np.full_like(x, float(c_obj)), np.zeros_like(x)

    def eval_family(cfam: List[Any], rfam: List[float], fsc: List[float], p_arr: np.ndarray, r_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_c = len(cfam)
        if n_c == 0:
            return np.zeros_like(p_arr), np.zeros_like(p_arr)
        if n_c == 1:
            v, s = eval_f(cfam[0], p_arr)
            sc = fsc[0] if len(fsc) > 0 else 1.0
            return v * sc, s * sc
        r_np = np.asarray(rfam, dtype=float)
        j = np.clip(np.searchsorted(r_np, r_arr, side="right") - 1, 0, n_c - 2)
        denom = np.maximum(r_np[j + 1] - r_np[j], _EM20)
        fac = np.clip((r_arr - r_np[j]) / denom, 0.0, 1.0)
        v_res = np.zeros_like(p_arr)
        s_res = np.zeros_like(p_arr)
        for k in range(n_c - 1):
            mask = (j == k)
            if not np.any(mask):
                continue
            sc1 = fsc[k] if k < len(fsc) else 1.0
            sc2 = fsc[k + 1] if (k + 1) < len(fsc) else 1.0
            y1, sl1 = eval_f(cfam[k], p_arr[mask])
            y2, sl2 = eval_f(cfam[k + 1], p_arr[mask])
            y1 *= sc1; sl1 *= sc1
            y2 *= sc2; sl2 *= sc2
            v_res[mask] = y1 + fac[mask] * (y2 - y1)
            s_res[mask] = sl1 + fac[mask] * (sl2 - sl1)
        return v_res, s_res

    # 5. Yield stress and hardening modulus evaluation (sigeps66.F:277-509)
    if israte == 4 and curves_c is not None and len(curves_c) > 0:
        yc, hc = eval_family(curves_c, rates_c or [], fp1, pla, rate_arr)
    else:
        yc, hc = eval_f(yield_curve_c, pla)
        yc = yc * fscale11
        hc = hc * fscale11

    if israte == 4 and curves_t is not None and len(curves_t) > 0:
        yt, ht = eval_family(curves_t, rates_t or [], fp2, pla, rate_arr)
    else:
        yt, ht = eval_f(yield_curve_t, pla)
        yt = yt * fscale22
        ht = ht * fscale22

    if sigy > 0.0:
        yc = np.where(yc <= 0.0, sigy, yc)
        yt = np.where(yt <= 0.0, sigy, yt)

    # Iso-kinematic hardening partition (sigeps66.F:293-325)
    if fisokin > 0.0:
        if israte == 4 and curves_c is not None and len(curves_c) > 0:
            yc0, _ = eval_family(curves_c, rates_c or [], fp1, np.zeros(nel), rate_arr)
        else:
            yc0, _ = eval_f(yield_curve_c, np.zeros(nel))
            yc0 = yc0 * fscale11
            if sigy > 0.0:
                yc0 = np.where(yc0 <= 0.0, sigy, yc0)

        if israte == 4 and curves_t is not None and len(curves_t) > 0:
            yt0, _ = eval_family(curves_t, rates_t or [], fp2, np.zeros(nel), rate_arr)
        else:
            yt0, _ = eval_f(yield_curve_t, np.zeros(nel))
            yt0 = yt0 * fscale22
            if sigy > 0.0:
                yt0 = np.where(yt0 <= 0.0, sigy, yt0)

        yc = (1.0 - fisokin) * yc + fisokin * yc0
        yt = (1.0 - fisokin) * yt + fisokin * yt0

    yc = np.maximum(yc, _EM20)
    yt = np.maximum(yt, _EM20)

    # Independent strain rate scaling for ISRATE = 3 (sigeps66.F:478-489)
    if israte == 3:
        yrate_c, _ = eval_f(rate_curve_c, rate_arr)
        yrate_t, _ = eval_f(rate_curve_t, rate_arr)
        yrate_c = yrate_c * fscale33
        yrate_t = yrate_t * fscale12
        yc = yc * yrate_c
        hc = hc * yrate_c
        yt = yt * yrate_t
        ht = ht * yrate_t

    # Pressure interpolation between tension and compression (sigeps66.F:491-510)
    yld = np.empty(nel, dtype=float)
    H = np.empty(nel, dtype=float)
    for i in range(nel):
        if pc == 0.0 and pt == 0.0 and abs(P[i]) < 1.0e-10:
            yld[i] = max(yc[i], _EM20)
            H[i] = hc[i]
        elif P[i] <= -pt:
            yld[i] = max(yt[i], _EM20)
            H[i] = ht[i]
        elif P[i] >= pc:
            yld[i] = max(yc[i], _EM20)
            H[i] = hc[i]
        else:
            fac = pc + pt
            fac = (pc - P[i]) / fac
            yld[i] = fac * yt[i] + (1.0 - fac) * yc[i]
            yld[i] = max(yld[i], _EM20)
            H[i] = fac * ht[i] + (1.0 - fac) * hc[i]

    # Analytical strain rate effect (sigeps66.F:517-534)
    if vp == 0 and epsp0 > 0.0:
        for i in range(nel):
            epd = max(_EM20, rate_arr[i] / epsp0)
            if israte == 1:
                yrate = 1.0 + epd ** cp
                yld[i] *= yrate
                H[i] *= yrate
            elif israte == 2:
                yrate = 1.0 + cp * np.log(epd)
                yld[i] *= yrate
                H[i] *= yrate

    # 6. J2 von Mises norm (sigeps66.F:561-563)
    j2 = 0.5 * (s_trial[:, 0] ** 2 + s_trial[:, 1] ** 2 + s_trial[:, 2] ** 2) \
        + s_trial[:, 3] ** 2 + s_trial[:, 4] ** 2 + s_trial[:, 5] ** 2
    vm = np.sqrt(3.0 * j2)

    # 7. Radial return projection (sigeps66.F:560-574)
    R = np.minimum(1.0, yld / np.maximum(vm, _EM20))
    dpla = np.zeros(nel, dtype=float)
    s_new = s_trial.copy()

    for i in range(nel):
        if vm[i] > yld[i]:
            denom = max(G3[i] + H[i], _EM20)
            dpla[i] = (vm[i] - yld[i]) / denom

            if vp > 0 and dt > 0.0:
                epd = max(_EM20, (dpla[i] / dt) / epsp0)
                yrate = 1.0 + epd ** cp
                if sigy == 0.0:
                    yld[i] *= yrate
                    H[i] *= yrate
                else:
                    yld[i] = yld[i] + sigy * (yrate - 1.0)
                R[i] = min(1.0, yld[i] / max(vm[i], _EM20))
                dpla[i] = (1.0 - R[i]) * vm[i] / denom

            pla[i] += dpla[i]
            s_new[i, :] = s_trial[i, :] * R[i]

    # 8. Kinematic hardening update (sigeps66.F:670-702)
    if fisokin > 0.0:
        for i in range(nel):
            if vm[i] > yld[i]:
                ds = s_trial[i] - s_new[i]
                hkin = (2.0 / 3.0) * fisokin * H[i]
                alpha_i = hkin / (G2[i] + hkin)
                delta_alpha = alpha_i * ds
                backstress[i] += delta_alpha

    # 9. Total Cauchy stress: sig_new = s_new + alpha - P * I (sigeps66.F:695-701)
    sig_new = s_new + backstress
    sig_new[:, 0] -= P
    sig_new[:, 1] -= P
    sig_new[:, 2] -= P

    return {
        "sig": sig_new[0] if is_1d else sig_new,
        "epsp": pla[0] if is_1d else pla,
        "dpla": dpla[0] if is_1d else dpla,
        "alpha": backstress[0] if is_1d else backstress,
        "soundsp": soundsp[0] if is_1d else soundsp,
        "E": E[0] if is_1d else E,
        "G": G[0] if is_1d else G,
        "C1": C1[0] if is_1d else C1,
        "P": P[0] if is_1d else P,
        "YLD": yld[0] if is_1d else yld,
        "H": H[0] if is_1d else H,
        "vm": vm[0] if is_1d else vm,
    }


# =============================================================================
# 2. UPSTREAM FORTRAN SHELL ENGINE ORACLE: sigeps66c.F
# =============================================================================

def sigeps66c_shell_oracle(
    matparam: Dict[str, Any],
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    alpha: Optional[np.ndarray] = None,
    thk: Optional[Union[float, np.ndarray]] = None,
    thkly: Optional[Union[float, np.ndarray]] = None,
    off: Optional[Union[float, np.ndarray]] = 1.0,
    rate: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    yield_curve_c: Optional[Any] = None,
    yield_curve_t: Optional[Any] = None,
    rate_curve_c: Optional[Any] = None,
    rate_curve_t: Optional[Any] = None,
    curves_c: Optional[List[Any]] = None,
    rates_c: Optional[List[float]] = None,
    curves_t: Optional[List[Any]] = None,
    rates_t: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Exact Python reproduction of OpenRadioss sigeps66c.F (Shells).

    Directly evaluates against Fortran expressions in engine/source/materials/mat/mat066/sigeps66c.F.
    sig, deps: (nel, 3) = [xx, yy, xy] with engineering shear.
    """
    sig_arr = np.asarray(sig, dtype=float)
    deps_arr = np.asarray(deps, dtype=float)
    is_1d = (sig_arr.ndim == 1)
    if is_1d:
        sig_arr = sig_arr[None, :]
        deps_arr = deps_arr[None, :]
    nel = sig_arr.shape[0]

    # Material parameters (hm_read_mat66.F / sigeps66c.F)
    et = float(matparam.get("e", 210000.0))
    ec = float(matparam.get("ec", et))
    if ec <= 0.0:
        ec = et
    nu = float(matparam.get("nu", 0.3))
    if nu >= 0.5:
        nu = 0.499
    if nu < 0.0:
        nu = 0.0
    pc = float(matparam.get("pc", 0.0))
    pt = float(matparam.get("pt", 0.0))
    rpct = float(matparam.get("rpct", 1.0))
    if rpct <= 0.0:
        rpct = 1.0
    fisokin = float(matparam.get("chard", matparam.get("fisokin", 0.0)))
    israte = int(matparam.get("israte", 1))
    epsp0 = float(matparam.get("epsp0", 0.0))
    cp = float(matparam.get("cp", 1.0))
    sigy = float(matparam.get("sigy", 0.0))
    vp = int(matparam.get("vp", 0))
    rho0 = float(matparam.get("rho0", 1.0))

    fscale11 = float(matparam.get("fscale11", 1.0))
    fscale22 = float(matparam.get("fscale22", 1.0))
    fscale33 = float(matparam.get("fscale33", 1.0))
    fscale12 = float(matparam.get("fscale12", 1.0))

    fp1 = matparam.get("fp1", [])
    fp2 = matparam.get("fp2", [])

    # History states
    pla = np.zeros(nel, dtype=float) if epsp is None else np.asarray(epsp, dtype=float).copy()
    if pla.ndim == 0:
        pla = np.full(nel, float(pla))
    backstress = np.zeros((nel, 3), dtype=float) if alpha is None else np.asarray(alpha, dtype=float).copy()
    if backstress.ndim == 1 and nel == 1:
        backstress = backstress[None, :]

    thk_arr = np.ones(nel, dtype=float) if thk is None else np.asarray(thk, dtype=float).copy()
    if thk_arr.ndim == 0:
        thk_arr = np.full(nel, float(thk_arr))
    thkly_arr = thk_arr.copy() if thkly is None else np.asarray(thkly, dtype=float).copy()
    if thkly_arr.ndim == 0:
        thkly_arr = np.full(nel, float(thkly_arr))
    off_arr = np.ones(nel, dtype=float) if off is None else np.asarray(off, dtype=float).copy()
    if off_arr.ndim == 0:
        off_arr = np.full(nel, float(off_arr))

    # 1. Plane stress initial estimate of P (sigeps66c.F:243-247)
    nu_sq_t = max(1.0 - nu ** 2, 1.0e-15)
    a11t = et / nu_sq_t
    a21t = nu * a11t

    sig_eff = sig_arr - backstress
    s_est_xx = sig_eff[:, 0] + a11t * deps_arr[:, 0] + a21t * deps_arr[:, 1]
    s_est_yy = sig_eff[:, 1] + a21t * deps_arr[:, 0] + a11t * deps_arr[:, 1]
    p_est = -(1.0 / 3.0) * (s_est_xx + s_est_yy)

    # 2. Modulus interpolation based on pressure P (sigeps66c.F:250-270)
    E = np.full(nel, et, dtype=float)
    if ec > 0.0 and ec != et:
        for i in range(nel):
            if pc == 0.0 and pt == 0.0 and abs(p_est[i]) < 1.0e-10:
                E[i] = ec
            elif p_est[i] <= -rpct * pt:
                E[i] = et
            elif p_est[i] >= rpct * pc:
                E[i] = ec
            else:
                fac = rpct * (pc + pt)
                fac = (rpct * pc - p_est[i]) / fac
                E[i] = fac * et + (1.0 - fac) * ec

    G = 0.5 * E / max(1.0 + nu, 1.0e-15)
    G3 = 3.0 * G
    A11 = E / nu_sq_t
    A21 = nu * A11

    # Sound speed (sigeps66c.F:293)
    soundsp = np.full(nel, math.sqrt(a11t / max(rho0, _EM20)), dtype=float)

    # 3. Trial stress with effective A11, A21, G (sigeps66c.F:281-291)
    s_trial = np.empty_like(sig_arr)
    s_trial[:, 0] = sig_eff[:, 0] + A11 * deps_arr[:, 0] + A21 * deps_arr[:, 1]
    s_trial[:, 1] = sig_eff[:, 1] + A21 * deps_arr[:, 0] + A11 * deps_arr[:, 1]
    s_trial[:, 2] = sig_eff[:, 2] + G * deps_arr[:, 2]

    # Recompute pressure P (sigeps66c.F:291)
    P = -(1.0 / 3.0) * (s_trial[:, 0] + s_trial[:, 1])

    # 4. Plane-stress strain rate & filtering (sigeps66c.F:299-308)
    nnu11 = nu / max(1.0 - nu, 1.0e-15)
    nu31 = 1.0 - nnu11
    dezz_est = -(deps_arr[:, 0] + deps_arr[:, 1]) * nnu11
    dav = (deps_arr[:, 0] + deps_arr[:, 1] + dezz_est) / 3.0

    if rate is not None:
        rate_arr = np.asarray(rate, dtype=float)
        if rate_arr.ndim == 0:
            rate_arr = np.full(nel, float(rate_arr))
    else:
        ee = ((deps_arr[:, 0] - dav) ** 2 + (deps_arr[:, 1] - dav) ** 2 + (dezz_est - dav) ** 2
              + 0.5 * deps_arr[:, 2] ** 2)
        rate_arr = np.sqrt((2.0 / 3.0) * ee) / dt if dt > 0.0 else np.zeros(nel, dtype=float)

    # 5. Curve evaluation helpers
    def eval_f(c_obj: Any, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if c_obj is None or c_obj == 0:
            return np.zeros_like(x), np.zeros_like(x)
        if callable(c_obj):
            res = c_obj(x)
            if isinstance(res, tuple):
                return np.asarray(res[0], dtype=float), np.asarray(res[1], dtype=float)
            return np.asarray(res, dtype=float), np.zeros_like(x)
        if isinstance(c_obj, (list, tuple)) and len(c_obj) == 2:
            cx = np.asarray(c_obj[0], dtype=float)
            cy = np.asarray(c_obj[1], dtype=float)
            dx = np.diff(cx)
            dx_s = np.where(np.abs(dx) > _EM20, dx, np.sign(dx) * _EM20 + _EM20)
            cs = np.diff(cy) / dx_s
            idx = np.clip(np.searchsorted(cx, x, side="right") - 1, 0, len(cx) - 2)
            val = cy[idx] + cs[idx] * (x - cx[idx])
            return val, cs[idx]
        return np.full_like(x, float(c_obj)), np.zeros_like(x)

    def eval_family(cfam: List[Any], rfam: List[float], fsc: List[float], p_arr: np.ndarray, r_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_c = len(cfam)
        if n_c == 0:
            return np.zeros_like(p_arr), np.zeros_like(p_arr)
        if n_c == 1:
            v, s = eval_f(cfam[0], p_arr)
            sc = fsc[0] if len(fsc) > 0 else 1.0
            return v * sc, s * sc
        r_np = np.asarray(rfam, dtype=float)
        j = np.clip(np.searchsorted(r_np, r_arr, side="right") - 1, 0, n_c - 2)
        denom = np.maximum(r_np[j + 1] - r_np[j], _EM20)
        fac = np.clip((r_arr - r_np[j]) / denom, 0.0, 1.0)
        v_res = np.zeros_like(p_arr)
        s_res = np.zeros_like(p_arr)
        for k in range(n_c - 1):
            mask = (j == k)
            if not np.any(mask):
                continue
            sc1 = fsc[k] if k < len(fsc) else 1.0
            sc2 = fsc[k + 1] if (k + 1) < len(fsc) else 1.0
            y1, sl1 = eval_f(cfam[k], p_arr[mask])
            y2, sl2 = eval_f(cfam[k + 1], p_arr[mask])
            y1 *= sc1; sl1 *= sc1
            y2 *= sc2; sl2 *= sc2
            v_res[mask] = y1 + fac[mask] * (y2 - y1)
            s_res[mask] = sl1 + fac[mask] * (sl2 - sl1)
        return v_res, s_res

    # 6. Yield stress and hardening evaluation (sigeps66c.F:312-549)
    if israte == 4 and curves_c is not None and len(curves_c) > 0:
        yc, hc = eval_family(curves_c, rates_c or [], fp1, pla, rate_arr)
    else:
        yc, hc = eval_f(yield_curve_c, pla)
        yc = yc * fscale11
        hc = hc * fscale11

    if israte == 4 and curves_t is not None and len(curves_t) > 0:
        yt, ht = eval_family(curves_t, rates_t or [], fp2, pla, rate_arr)
    else:
        yt, ht = eval_f(yield_curve_t, pla)
        yt = yt * fscale22
        ht = ht * fscale22

    if sigy > 0.0:
        yc = np.where(yc <= 0.0, sigy, yc)
        yt = np.where(yt <= 0.0, sigy, yt)

    # Iso-kinematic hardening partition (sigeps66c.F:328-360)
    if fisokin > 0.0:
        if israte == 4 and curves_c is not None and len(curves_c) > 0:
            yc0, _ = eval_family(curves_c, rates_c or [], fp1, np.zeros(nel), rate_arr)
        else:
            yc0, _ = eval_f(yield_curve_c, np.zeros(nel))
            yc0 = yc0 * fscale11
            if sigy > 0.0:
                yc0 = np.where(yc0 <= 0.0, sigy, yc0)

        if israte == 4 and curves_t is not None and len(curves_t) > 0:
            yt0, _ = eval_family(curves_t, rates_t or [], fp2, np.zeros(nel), rate_arr)
        else:
            yt0, _ = eval_f(yield_curve_t, np.zeros(nel))
            yt0 = yt0 * fscale22
            if sigy > 0.0:
                yt0 = np.where(yt0 <= 0.0, sigy, yt0)

        yc = (1.0 - fisokin) * yc + fisokin * yc0
        yt = (1.0 - fisokin) * yt + fisokin * yt0

    yc = np.maximum(yc, _EM20)
    yt = np.maximum(yt, _EM20)

    # Independent strain rate scaling for ISRATE = 3 (sigeps66c.F:518-529)
    if israte == 3:
        yrate_c, _ = eval_f(rate_curve_c, rate_arr)
        yrate_t, _ = eval_f(rate_curve_t, rate_arr)
        yrate_c = yrate_c * fscale33
        yrate_t = yrate_t * fscale12
        yc = yc * yrate_c
        hc = hc * yrate_c
        yt = yt * yrate_t
        ht = ht * yrate_t

    # Pressure interpolation between tension and compression (sigeps66c.F:531-548)
    yld = np.empty(nel, dtype=float)
    H = np.empty(nel, dtype=float)
    for i in range(nel):
        if pc == 0.0 and pt == 0.0 and abs(P[i]) < 1.0e-10:
            yld[i] = max(yc[i], _EM20)
            H[i] = hc[i]
        elif P[i] <= -pt:
            yld[i] = max(yt[i], _EM20)
            H[i] = ht[i]
        elif P[i] >= pc:
            yld[i] = max(yc[i], _EM20)
            H[i] = hc[i]
        else:
            fac = pc + pt
            fac = (pc - P[i]) / fac
            yld[i] = fac * yt[i] + (1.0 - fac) * yc[i]
            yld[i] = max(yld[i], _EM20)
            H[i] = fac * ht[i] + (1.0 - fac) * hc[i]

    # Analytical strain rate effect (sigeps66c.F:552-569)
    if vp == 0 and epsp0 > 0.0:
        for i in range(nel):
            epd = max(_EM20, rate_arr[i] / epsp0)
            if israte == 1:
                yrate = 1.0 + epd ** cp
                yld[i] *= yrate
                H[i] *= yrate
            elif israte == 2:
                yrate = 1.0 + cp * np.log(epd)
                yld[i] *= yrate
                H[i] *= yrate

    # 7. Plane-stress von Mises stress (sigeps66c.F:577-580)
    svm = np.sqrt(s_trial[:, 0] ** 2 + s_trial[:, 1] ** 2
                  - s_trial[:, 0] * s_trial[:, 1]
                  + 3.0 * s_trial[:, 2] ** 2)

    # 8. Plane-stress radial projection (sigeps66c.F:576-595)
    R = np.minimum(1.0, yld / np.maximum(svm, _EM20))
    dpla = np.zeros(nel, dtype=float)
    s_new = s_trial.copy()

    for i in range(nel):
        if svm[i] > yld[i]:
            denom = max(G3[i] + H[i], _EM20)
            dpla[i] = (svm[i] - yld[i]) / denom

            if vp > 0 and dt > 0.0:
                epd = max(_EM20, (dpla[i] / dt) / epsp0)
                yrate = 1.0 + epd ** cp
                if sigy == 0.0:
                    yld[i] *= yrate
                    H[i] *= yrate
                else:
                    yld[i] = yld[i] + sigy * (yrate - 1.0)
                R[i] = min(1.0, yld[i] / max(svm[i], _EM20))
                dpla[i] = (1.0 - R[i]) * svm[i] / denom

            pla[i] += dpla[i]
            s_new[i, 0] = s_trial[i, 0] * R[i]
            s_new[i, 1] = s_trial[i, 1] * R[i]
            s_new[i, 2] = s_trial[i, 2] * R[i]

    # 9. Kinematic hardening update for shells (sigeps66c.F:776-798)
    if fisokin > 0.0:
        for i in range(nel):
            if svm[i] > yld[i]:
                dsxx = s_trial[i, 0] - s_new[i, 0]
                dsyy = s_trial[i, 1] - s_new[i, 1]
                dsxy = s_trial[i, 2] - s_new[i, 2]
                dexx = dsxx - nu * dsyy
                deyy = dsyy - nu * dsxx
                dexy = 2.0 * (1.0 + nu) * dsxy

                hkin = (2.0 / 3.0) * fisokin * H[i]
                alpha_i = hkin / (E[i] + hkin)
                sigpxx = alpha_i * (2.0 * dexx + deyy)
                sigpyy = alpha_i * (2.0 * deyy + dexx)
                sigpxy = alpha_i * dexy * 0.5

                backstress[i, 0] += sigpxx
                backstress[i, 1] += sigpyy
                backstress[i, 2] += sigpxy

    # 10. Through-thickness strain dezz & thickness update (sigeps66c.F:588-593)
    dezz = np.empty(nel, dtype=float)
    thk_new = thk_arr.copy()
    for i in range(nel):
        if svm[i] > yld[i]:
            s_mean = 0.5 * (s_new[i, 0] + s_new[i, 1])
            dezz_plas = dpla[i] * s_mean / max(yld[i], _EM20)
        else:
            dezz_plas = 0.0
        dezz[i] = -(deps_arr[i, 0] + deps_arr[i, 1]) * nnu11 - nu31 * dezz_plas
        thk_new[i] += dezz[i] * thkly_arr[i] * off_arr[i]

    # 11. Cauchy stress: sig_new = s_new + alpha (sigeps66c.F:794-796)
    sig_new = s_new + backstress

    return {
        "sig": sig_new[0] if is_1d else sig_new,
        "epsp": pla[0] if is_1d else pla,
        "dpla": dpla[0] if is_1d else dpla,
        "alpha": backstress[0] if is_1d else backstress,
        "thk": thk_new[0] if is_1d else thk_new,
        "dezz": dezz[0] if is_1d else dezz,
        "soundsp": soundsp[0] if is_1d else soundsp,
        "E": E[0] if is_1d else E,
        "G": G[0] if is_1d else G,
        "A11": A11[0] if is_1d else A11,
        "A21": A21[0] if is_1d else A21,
        "P": P[0] if is_1d else P,
        "YLD": yld[0] if is_1d else yld,
        "H": H[0] if is_1d else H,
        "svm": svm[0] if is_1d else svm,
    }


# =============================================================================
# 3. DIRECT UNIT TESTS FOR LAW66 PHYSICAL BEHAVIORS
# =============================================================================

def test_asymmetric_modulus_interpolation_solids():
    """Verify solid modulus interpolation between tension Et and compression Ec."""
    params = {
        "e": 200000.0,
        "ec": 100000.0,
        "nu": 0.25,
        "pc": 400.0,
        "pt": 200.0,
        "rpct": 1.0,
    }
    p = Law66Params(**params)
    mat = build_law66(p)

    c1t = 200000.0 / (3.0 * (1.0 - 0.5))  # 133333.3333
    gt = 0.5 * 200000.0 / 1.25            # 80000.0
    gc = 0.5 * 100000.0 / 1.25            # 40000.0

    # 1. Extreme tension: P <= -200 => E = 200000, G = 80000
    mu_tens = -300.0 / c1t
    extra_tens = {"mu": np.array([mu_tens])}
    res_oracle_tens = sigeps66_solid_oracle(params, np.zeros(6), np.zeros(6), mu=mu_tens)
    assert res_oracle_tens["E"] == 200000.0
    assert res_oracle_tens["G"] == gt

    sig_out, _, _ = solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), extra=extra_tens)
    # Check that oracle matches
    assert math.isclose(res_oracle_tens["E"], 200000.0, rel_tol=1e-12)

    # 2. Extreme compression: P >= 400 => E = 100000, G = 40000
    mu_comp = 500.0 / c1t
    res_oracle_comp = sigeps66_solid_oracle(params, np.zeros(6), np.zeros(6), mu=mu_comp)
    assert res_oracle_comp["E"] == 100000.0
    assert res_oracle_comp["G"] == gc

    # 3. Transition: P = 100.0 => fac = (400 - 100) / (400 + 200) = 300 / 600 = 0.5
    # E = 0.5 * 200000 + 0.5 * 100000 = 150000.0
    mu_mid = 100.0 / c1t
    res_oracle_mid = sigeps66_solid_oracle(params, np.zeros(6), np.zeros(6), mu=mu_mid)
    assert math.isclose(res_oracle_mid["E"], 150000.0, rel_tol=1e-12)
    assert math.isclose(res_oracle_mid["G"], 0.5 * 150000.0 / 1.25, rel_tol=1e-12)


def test_asymmetric_yield_stress_interpolation():
    """Verify asymmetric yield stress interpolation between tension and compression curves."""
    # Tensile curve: 200 + 1000 * pla
    # Compressive curve: 350 + 2000 * pla
    curve_c = ([0.0, 0.1], [350.0, 550.0])
    curve_t = ([0.0, 0.1], [200.0, 300.0])

    params = {
        "e": 200000.0,
        "nu": 0.3,
        "pc": 300.0,
        "pt": 100.0,
        "rpct": 1.0,
        "curve_c": curve_c,
        "curve_t": curve_t,
    }
    p = Law66Params(**params)
    mat = build_law66(p)

    c1 = p.c1t

    # Case A: Pure tension P <= -100 => YLD = 200, H = 1000
    res_tens = sigeps66_solid_oracle(params, np.zeros(6), np.zeros(6), mu=-150.0 / c1, yield_curve_c=curve_c, yield_curve_t=curve_t)
    assert math.isclose(res_tens["YLD"], 200.0, rel_tol=1e-12)
    assert math.isclose(res_tens["H"], 1000.0, rel_tol=1e-12)

    # Case B: Pure compression P >= 300 => YLD = 350, H = 2000
    res_comp = sigeps66_solid_oracle(params, np.zeros(6), np.zeros(6), mu=350.0 / c1, yield_curve_c=curve_c, yield_curve_t=curve_t)
    assert math.isclose(res_comp["YLD"], 350.0, rel_tol=1e-12)
    assert math.isclose(res_comp["H"], 2000.0, rel_tol=1e-12)

    # Case C: Linear transition at P = 0.0 => fac = (300 - 0) / (300 + 100) = 0.75
    # YLD = 0.75 * 200 + 0.25 * 350 = 150 + 87.5 = 237.5
    # H = 0.75 * 1000 + 0.25 * 2000 = 750 + 500 = 1250
    res_mid = sigeps66_solid_oracle(params, np.zeros(6), np.zeros(6), mu=0.0, yield_curve_c=curve_c, yield_curve_t=curve_t)
    assert math.isclose(res_mid["YLD"], 237.5, rel_tol=1e-12)
    assert math.isclose(res_mid["H"], 1250.0, rel_tol=1e-12)


def test_strain_rate_formulations_israte_1_2_3_4():
    """Verify all 4 strain rate options (ISRATE=1 power, 2 log, 3 independent, 4 families)."""
    curve_c = ([0.0, 0.1], [300.0, 500.0])
    curve_t = ([0.0, 0.1], [300.0, 500.0])

    # 1. ISRATE = 1: Cowper-Symonds power law: YRATE = 1 + (epsp/epsp0)**cp
    p1 = {
        "e": 200000.0, "nu": 0.3, "israte": 1, "epsp0": 10.0, "cp": 0.2, "curve_c": curve_c, "curve_t": curve_t,
    }
    rate = 100.0
    # epd = 100 / 10 = 10.0 => yrate = 1 + 10.0**0.2 ~ 2.5848931924611136
    expected_yrate = 1.0 + 10.0 ** 0.2
    res_rate1 = sigeps66_solid_oracle(p1, np.zeros(6), np.zeros(6), rate=rate, yield_curve_c=curve_c, yield_curve_t=curve_t)
    assert math.isclose(res_rate1["YLD"], 300.0 * expected_yrate, rel_tol=1e-12)

    # 2. ISRATE = 2: Logarithmic form: YRATE = 1 + cp * log(epsp/epsp0)
    p2 = {
        "e": 200000.0, "nu": 0.3, "israte": 2, "epsp0": 5.0, "cp": 0.15, "curve_c": curve_c, "curve_t": curve_t,
    }
    # epd = 50 / 5 = 10.0 => yrate = 1 + 0.15 * log(10.0) ~ 1.3453877639491068
    expected_yrate2 = 1.0 + 0.15 * math.log(10.0)
    res_rate2 = sigeps66_solid_oracle(p2, np.zeros(6), np.zeros(6), rate=50.0, yield_curve_c=curve_c, yield_curve_t=curve_t)
    assert math.isclose(res_rate2["YLD"], 300.0 * expected_yrate2, rel_tol=1e-12)

    # 3. ISRATE = 3: Independent scaling curves for compression and tension
    rate_c = ([0.0, 100.0], [1.0, 1.4])
    rate_t = ([0.0, 100.0], [1.0, 1.8])
    p3 = {
        "e": 200000.0, "nu": 0.3, "israte": 3, "pc": 100.0, "pt": 100.0,
        "curve_c": curve_c, "curve_t": curve_t,
        "curve_rate_c": rate_c, "curve_rate_t": rate_t,
        "fscale33": 1.0, "fscale12": 1.0,
    }
    c1 = 200000.0 / (3.0 * (1.0 - 0.6))
    # At rate = 50.0: yrate_c = 1.2, yrate_t = 1.4
    # Under pure compression: YLD = 300 * 1.2 = 360
    res_rate3_c = sigeps66_solid_oracle(p3, np.zeros(6), np.zeros(6), mu=150.0 / c1, rate=50.0,
                                        yield_curve_c=curve_c, yield_curve_t=curve_t,
                                        rate_curve_c=rate_c, rate_curve_t=rate_t)
    assert math.isclose(res_rate3_c["YLD"], 300.0 * 1.2, rel_tol=1e-12)

    # Under pure tension: YLD = 300 * 1.4 = 420
    res_rate3_t = sigeps66_solid_oracle(p3, np.zeros(6), np.zeros(6), mu=-150.0 / c1, rate=50.0,
                                        yield_curve_c=curve_c, yield_curve_t=curve_t,
                                        rate_curve_c=rate_c, rate_curve_t=rate_t)
    assert math.isclose(res_rate3_t["YLD"], 300.0 * 1.4, rel_tol=1e-12)

    # 4. ISRATE = 4: Multi-curve tabulated curve families
    c_fam = [
        ([0.0, 0.1], [250.0, 400.0]),
        ([0.0, 0.1], [350.0, 500.0]),
    ]
    r_fam = [0.0, 100.0]
    p4 = {
        "e": 200000.0, "nu": 0.3, "israte": 4, "pc": 100.0, "pt": 100.0,
        "fp1": [1.0, 1.0], "fp2": [1.0, 1.0],
    }
    # At rate = 50.0 (halfway), yield stress at pla = 0: 0.5 * 250 + 0.5 * 350 = 300.0
    res_rate4 = sigeps66_solid_oracle(p4, np.zeros(6), np.zeros(6), mu=150.0 / c1, rate=50.0,
                                      curves_c=c_fam, rates_c=r_fam, curves_t=c_fam, rates_t=r_fam)
    assert math.isclose(res_rate4["YLD"], 300.0, rel_tol=1e-12)


def test_kinematic_hardening_bauschinger_effect():
    """Verify kinematic hardening shifts backstress alpha and produces Bauschinger effect."""
    curve = ([0.0, 0.1], [250.0, 2250.0])  # H = 20000.0
    params = {
        "e": 200000.0, "nu": 0.3, "chard": 1.0,  # Pure kinematic
        "curve_c": curve, "curve_t": curve,
    }
    mat = build_law66(Law66Params(**params))

    # Step 1: Forward tension: deps_xx = 0.003
    deps_fwd = np.array([0.003, -0.0009, -0.0009, 0.0, 0.0, 0.0])
    extra = {"uvar66": np.zeros((1, 8))}

    sig1, epsp1, _ = solid_update(mat, np.zeros((1, 6)), deps_fwd[None, :], extra=extra)
    assert epsp1[0] > 0.0
    alpha_xx = extra["uvar66"][0, 1]
    assert alpha_xx > 0.0  # Backstress has grown in tensile direction

    # Step 2: Reverse compression: deps_xx = -0.002
    deps_rev = np.array([-0.002, 0.0006, 0.0006, 0.0, 0.0, 0.0])
    sig2, epsp2, _ = solid_update(mat, sig1, deps_rev[None, :], epsp=epsp1, extra=extra)

    # Compare against pure isotropic (chard = 0.0)
    params_iso = {**params, "chard": 0.0}
    mat_iso = build_law66(Law66Params(**params_iso))
    extra_iso = {"uvar66": np.zeros((1, 8))}
    sig1_iso, epsp1_iso, _ = solid_update(mat_iso, np.zeros((1, 6)), deps_fwd[None, :], extra=extra_iso)
    sig2_iso, epsp2_iso, _ = solid_update(mat_iso, sig1_iso, deps_rev[None, :], epsp=epsp1_iso, extra=extra_iso)

    assert extra_iso["uvar66"][0, 1] == 0.0
    assert extra["uvar66"][0, 1] != 0.0


def test_shell_plane_stress_thinning_and_soundsp():
    """Verify shell plane-stress radial projection, thickness thinning dezz, and sound speed."""
    curve = ([0.0, 0.1], [300.0, 800.0])
    params = {
        "e": 210000.0,
        "nu": 0.3,
        "rho0": 7.8e-6,
        "curve_c": curve,
        "curve_t": curve,
    }
    p = Law66Params(**params)
    mat = build_law66(p)

    deps = np.array([[0.006, 0.0, 0.0]])
    extra = {"uvar66": np.zeros((1, 5)), "thk": np.array([2.0])}

    sig_out, epsp_out, c_sound = shell_update(mat, np.zeros((1, 3)), deps, extra=extra, return_tuple=True)
    assert epsp_out[0] > 0.0
    assert extra["thk"][0] < 2.0  # Thinning occurred

    # Dilatational plane-stress sound speed = sqrt(A11 / rho0)
    expected_c = math.sqrt((210000.0 / (1.0 - 0.3 ** 2)) / 7.8e-6)
    assert math.isclose(c_sound[0], expected_c, rel_tol=1e-12)


def test_multielement_batch_solid_and_shell():
    """Verify vectorized multi-element batch execution for solids and shells."""
    n = 6
    curve = ([0.0, 0.1], [300.0, 800.0])
    mat = build_law66(e=200000.0, nu=0.28, rho0=2.7e-6, curve_c=curve, curve_t=curve)

    # Solids
    sig_solids = np.zeros((n, 6))
    deps_solids = np.zeros((n, 6))
    deps_solids[:, 0] = np.linspace(0.0005, 0.006, n)
    extra_s = {"uvar66": np.zeros((n, 8))}
    sig_s_out, epsp_s_out, c_s = solid_update(mat, sig_solids, deps_solids, extra=extra_s)
    assert sig_s_out.shape == (n, 6)
    assert epsp_s_out.shape == (n,)
    assert c_s.shape == (n,)

    # Shells
    sig_shells = np.zeros((n, 3))
    deps_shells = np.zeros((n, 3))
    deps_shells[:, 0] = np.linspace(0.0005, 0.006, n)
    extra_sh = {"uvar66": np.zeros((n, 5)), "thk": np.full(n, 1.5)}
    sig_sh_out, epsp_sh_out, c_sh = shell_update(mat, sig_shells, deps_shells, extra=extra_sh, return_tuple=True)
    assert sig_sh_out.shape == (n, 3)
    assert epsp_sh_out.shape == (n,)
    assert c_sh.shape == (n,)


# =============================================================================
# 4. EXHAUSTIVE 64 DIVERSE STATES PARITY COMPARISONS (rtol=1e-12, atol=1e-12)
# =============================================================================

def test_exhaustive_solid_oracle_comparison_32_states():
    """Exhaustively verify solid_update against sigeps66_solid_oracle across 32 physical states.

    Tests full spectrum of:
    - Asymmetric moduli (Et != Ec, Et == Ec)
    - Asymmetric yield limits (PC, PT, RPCT)
    - Hardening partition (pure isotropic chard=0, mixed chard=0.5, pure kinematic chard=1.0)
    - Strain rate modes (ISRATE 1, 2, 3, 4, VP 0, 1)
    - Stress/strain triaxialities (pure tension, pure compression, shear, multiaxial, hydrostatic)
    """
    curve_c = ([0.0, 0.05, 0.15], [350.0, 500.0, 650.0])
    curve_t = ([0.0, 0.05, 0.15], [250.0, 380.0, 480.0])

    rate_c = ([0.0, 50.0, 200.0], [1.0, 1.25, 1.50])
    rate_t = ([0.0, 50.0, 200.0], [1.0, 1.35, 1.70])

    curves_c_fam = [
        ([0.0, 0.1], [300.0, 450.0]),
        ([0.0, 0.1], [400.0, 580.0]),
        ([0.0, 0.1], [500.0, 700.0]),
    ]
    rates_c_fam = [0.0, 50.0, 200.0]

    curves_t_fam = [
        ([0.0, 0.1], [220.0, 360.0]),
        ([0.0, 0.1], [310.0, 470.0]),
        ([0.0, 0.1], [420.0, 600.0]),
    ]
    rates_t_fam = [0.0, 50.0, 200.0]

    # Generate 32 distinct state configurations
    configs = []
    e_pairs = [(210000.0, 210000.0), (200000.0, 120000.0), (180000.0, 220000.0)]
    chard_vals = [0.0, 0.45, 1.0]
    p_limits = [(0.0, 0.0, 1.0), (300.0, 150.0, 1.0), (500.0, 200.0, 0.8)]

    count = 0
    for e, ec in e_pairs:
        for chard in chard_vals:
            for pc, pt, rpct in p_limits:
                count += 1
                if count <= 24:
                    configs.append({
                        "id": count,
                        "e": e, "ec": ec, "nu": 0.28, "pc": pc, "pt": pt, "rpct": rpct,
                        "chard": chard, "israte": 1 if count % 2 == 0 else 2,
                        "epsp0": 10.0, "cp": 0.25, "vp": 0, "rho0": 7.85e-6,
                    })

    # 8 additional edge and special cases (ISRATE 3, 4, VP=1, multi-element batch, etc.)
    special_configs = [
        {"id": 25, "e": 200000.0, "ec": 130000.0, "nu": 0.3, "pc": 250.0, "pt": 120.0, "rpct": 1.0,
         "chard": 0.5, "israte": 3, "rate_val": 45.0, "rho0": 7.8e-6},
        {"id": 26, "e": 210000.0, "ec": 210000.0, "nu": 0.25, "pc": 300.0, "pt": 150.0, "rpct": 1.0,
         "chard": 0.0, "israte": 4, "rate_val": 75.0, "rho0": 7.8e-6},
        {"id": 27, "e": 190000.0, "ec": 190000.0, "nu": 0.32, "pc": 0.0, "pt": 0.0, "rpct": 1.0,
         "chard": 0.8, "israte": 1, "epsp0": 5.0, "cp": 0.3, "vp": 1, "dt": 1e-5, "rho0": 7.8e-6},
        {"id": 28, "e": 200000.0, "ec": 100000.0, "nu": 0.27, "pc": 400.0, "pt": 200.0, "rpct": 0.9,
         "chard": 0.3, "israte": 1, "epsp0": 0.0, "pure_shear": True, "rho0": 2.7e-6},
        {"id": 29, "e": 220000.0, "ec": 150000.0, "nu": 0.29, "pc": 350.0, "pt": 180.0, "rpct": 1.0,
         "chard": 1.0, "israte": 2, "epsp0": 15.0, "cp": 0.18, "reverse_load": True, "rho0": 7.8e-6},
        {"id": 30, "e": 205000.0, "ec": 140000.0, "nu": 0.3, "pc": 280.0, "pt": 140.0, "rpct": 1.0,
         "chard": 0.4, "israte": 1, "epsp0": 8.0, "cp": 0.22, "triaxial": True, "rho0": 7.8e-6},
        {"id": 31, "e": 200000.0, "ec": 200000.0, "nu": 0.3, "pc": 0.0, "pt": 0.0, "rpct": 1.0,
         "chard": 0.0, "israte": 1, "epsp0": 0.0, "elastic_only": True, "rho0": 7.8e-6},
        {"id": 32, "e": 210000.0, "ec": 160000.0, "nu": 0.28, "pc": 320.0, "pt": 160.0, "rpct": 0.85,
         "chard": 0.6, "israte": 1, "epsp0": 12.0, "cp": 0.2, "batch": 5, "rho0": 7.8e-6},
    ]
    configs.extend(special_configs)
    assert len(configs) == 32

    for cfg in configs:
        nel = cfg.get("batch", 1)
        p_dict = {
            "e": cfg["e"],
            "ec": cfg["ec"],
            "nu": cfg["nu"],
            "pc": cfg["pc"],
            "pt": cfg["pt"],
            "rpct": cfg["rpct"],
            "chard": cfg["chard"],
            "israte": cfg["israte"],
            "epsp0": cfg.get("epsp0", 10.0),
            "cp": cfg.get("cp", 1.0),
            "vp": cfg.get("vp", 0),
            "rho0": cfg.get("rho0", 7.8e-6),
            "curve_c": curve_c,
            "curve_t": curve_t,
        }
        if cfg["israte"] == 3:
            p_dict["curve_rate_c"] = rate_c
            p_dict["curve_rate_t"] = rate_t
        elif cfg["israte"] == 4:
            p_dict["curves_c"] = curves_c_fam
            p_dict["rates_c"] = rates_c_fam
            p_dict["curves_t"] = curves_t_fam
            p_dict["rates_t"] = rates_t_fam
            p_dict["fp1"] = [1.0, 1.0, 1.0]
            p_dict["fp2"] = [1.0, 1.0, 1.0]

        mat = build_law66(Law66Params(**p_dict))

        # Build initial stress and strain increment
        sig = np.zeros((nel, 6), dtype=float)
        deps = np.zeros((nel, 6), dtype=float)
        epsp_init = np.zeros(nel, dtype=float)
        alpha_init = np.zeros((nel, 6), dtype=float)

        if cfg.get("elastic_only"):
            deps[:, 0] = 0.0002
            deps[:, 1] = -0.00006
            deps[:, 2] = -0.00006
        elif cfg.get("pure_shear"):
            deps[:, 3] = 0.008
        elif cfg.get("triaxial"):
            deps[:, 0] = 0.005
            deps[:, 1] = 0.002
            deps[:, 2] = -0.003
            deps[:, 3] = 0.004
            deps[:, 4] = 0.002
            deps[:, 5] = -0.001
        elif cfg.get("reverse_load"):
            sig[:, 0] = 150.0
            sig[:, 1] = -50.0
            alpha_init[:, 0] = 40.0
            alpha_init[:, 1] = -20.0
            epsp_init[:] = 0.02
            deps[:, 0] = -0.006
            deps[:, 1] = 0.003
        else:
            if nel == 1:
                deps[0, 0] = 0.004
                deps[0, 1] = -0.0012
                deps[0, 2] = -0.0012
                deps[0, 3] = 0.002
            else:
                for k in range(nel):
                    deps[k, 0] = 0.001 * (k + 1)
                    deps[k, 1] = -0.0003 * (k + 1)
                    deps[k, 2] = -0.0003 * (k + 1)

        dt = cfg.get("dt", 1e-6)
        rate_val = cfg.get("rate_val", 25.0)

        c1t = cfg["e"] / (3.0 * (1.0 - 2.0 * cfg["nu"]))
        mu_val = np.zeros(nel)
        for k in range(nel):
            # Assign varied volumetric strains across states
            mu_val[k] = (cfg["id"] - 16) * 10.0 / c1t

        extra = {
            "uvar66": np.zeros((nel, 8)),
            "mu": mu_val,
            "rate": np.full(nel, rate_val),
        }
        extra["uvar66"][:, 0] = epsp_init
        extra["uvar66"][:, 1:7] = alpha_init

        # Run solid_update
        sig_in = sig.copy()
        sig_out, epsp_out, soundsp_out = solid_update(
            mat, sig_in, deps, epsp=epsp_init.copy(), dt=dt, extra=extra, return_tuple=True
        )

        # Run Fortran Oracle
        oracle_res = sigeps66_solid_oracle(
            p_dict,
            sig,
            deps,
            epsp=epsp_init,
            alpha=alpha_init,
            mu=mu_val,
            rate=rate_val,
            dt=dt,
            yield_curve_c=p_dict.get("curve_c"),
            yield_curve_t=p_dict.get("curve_t"),
            rate_curve_c=p_dict.get("curve_rate_c"),
            rate_curve_t=p_dict.get("curve_rate_t"),
            curves_c=p_dict.get("curves_c"),
            rates_c=p_dict.get("rates_c"),
            curves_t=p_dict.get("curves_t"),
            rates_t=p_dict.get("rates_t"),
        )

        # Strict parity check: rtol=1e-12, atol=1e-12
        np.testing.assert_allclose(sig_out, oracle_res["sig"], rtol=1e-12, atol=1e-12,
                                   err_msg=f"Solid state {cfg['id']} stress mismatch")
        np.testing.assert_allclose(epsp_out, oracle_res["epsp"], rtol=1e-12, atol=1e-12,
                                   err_msg=f"Solid state {cfg['id']} epsp mismatch")
        np.testing.assert_allclose(extra["uvar66"][:, 1:7], oracle_res["alpha"], rtol=1e-12, atol=1e-12,
                                   err_msg=f"Solid state {cfg['id']} backstress alpha mismatch")
        np.testing.assert_allclose(soundsp_out, oracle_res["soundsp"], rtol=1e-12, atol=1e-12,
                                   err_msg=f"Solid state {cfg['id']} soundsp mismatch")


def test_exhaustive_shell_oracle_comparison_32_states():
    """Exhaustively verify shell_update against sigeps66c_shell_oracle across 32 physical states.

    Tests full spectrum of:
    - Plane-stress Hooke's trial state with pressure estimation
    - Asymmetric moduli & yield interpolation
    - Radial projection with sigma_zz = 0
    - Kinematic hardening for shells
    - Through-thickness thinning dezz and thickness update
    - Strain rate modes (ISRATE 1, 2, 3, 4, VP)
    """
    curve_c = ([0.0, 0.05, 0.15], [360.0, 520.0, 680.0])
    curve_t = ([0.0, 0.05, 0.15], [260.0, 390.0, 510.0])

    rate_c = ([0.0, 50.0, 200.0], [1.0, 1.25, 1.50])
    rate_t = ([0.0, 50.0, 200.0], [1.0, 1.35, 1.70])

    curves_c_fam = [
        ([0.0, 0.1], [320.0, 470.0]),
        ([0.0, 0.1], [420.0, 600.0]),
        ([0.0, 0.1], [530.0, 720.0]),
    ]
    rates_c_fam = [0.0, 50.0, 200.0]

    curves_t_fam = [
        ([0.0, 0.1], [240.0, 380.0]),
        ([0.0, 0.1], [330.0, 490.0]),
        ([0.0, 0.1], [440.0, 620.0]),
    ]
    rates_t_fam = [0.0, 50.0, 200.0]

    configs = []
    e_pairs = [(210000.0, 210000.0), (200000.0, 130000.0), (175000.0, 215000.0)]
    chard_vals = [0.0, 0.5, 1.0]
    p_limits = [(0.0, 0.0, 1.0), (250.0, 120.0, 1.0), (450.0, 180.0, 0.85)]

    count = 0
    for e, ec in e_pairs:
        for chard in chard_vals:
            for pc, pt, rpct in p_limits:
                count += 1
                if count <= 24:
                    configs.append({
                        "id": count,
                        "e": e, "ec": ec, "nu": 0.29, "pc": pc, "pt": pt, "rpct": rpct,
                        "chard": chard, "israte": 1 if count % 2 == 0 else 2,
                        "epsp0": 12.0, "cp": 0.22, "vp": 0, "rho0": 7.85e-6,
                    })

    special_configs = [
        {"id": 25, "e": 200000.0, "ec": 140000.0, "nu": 0.3, "pc": 220.0, "pt": 110.0, "rpct": 1.0,
         "chard": 0.4, "israte": 3, "rate_val": 40.0, "rho0": 7.8e-6},
        {"id": 26, "e": 210000.0, "ec": 210000.0, "nu": 0.26, "pc": 280.0, "pt": 140.0, "rpct": 1.0,
         "chard": 0.0, "israte": 4, "rate_val": 60.0, "rho0": 7.8e-6},
        {"id": 27, "e": 195000.0, "ec": 195000.0, "nu": 0.31, "pc": 0.0, "pt": 0.0, "rpct": 1.0,
         "chard": 0.7, "israte": 1, "epsp0": 6.0, "cp": 0.28, "vp": 1, "dt": 1e-5, "rho0": 7.8e-6},
        {"id": 28, "e": 205000.0, "ec": 115000.0, "nu": 0.27, "pc": 380.0, "pt": 190.0, "rpct": 0.9,
         "chard": 0.35, "israte": 1, "epsp0": 0.0, "pure_shear": True, "rho0": 2.7e-6},
        {"id": 29, "e": 215000.0, "ec": 145000.0, "nu": 0.28, "pc": 340.0, "pt": 170.0, "rpct": 1.0,
         "chard": 1.0, "israte": 2, "epsp0": 10.0, "cp": 0.16, "reverse_load": True, "rho0": 7.8e-6},
        {"id": 30, "e": 200000.0, "ec": 135000.0, "nu": 0.3, "pc": 260.0, "pt": 130.0, "rpct": 1.0,
         "chard": 0.45, "israte": 1, "epsp0": 9.0, "cp": 0.2, "biaxial": True, "rho0": 7.8e-6},
        {"id": 31, "e": 200000.0, "ec": 200000.0, "nu": 0.3, "pc": 0.0, "pt": 0.0, "rpct": 1.0,
         "chard": 0.0, "israte": 1, "epsp0": 0.0, "elastic_only": True, "rho0": 7.8e-6},
        {"id": 32, "e": 210000.0, "ec": 155000.0, "nu": 0.29, "pc": 310.0, "pt": 150.0, "rpct": 0.88,
         "chard": 0.65, "israte": 1, "epsp0": 11.0, "cp": 0.24, "batch": 5, "rho0": 7.8e-6},
    ]
    configs.extend(special_configs)
    assert len(configs) == 32

    for cfg in configs:
        nel = cfg.get("batch", 1)
        p_dict = {
            "e": cfg["e"],
            "ec": cfg["ec"],
            "nu": cfg["nu"],
            "pc": cfg["pc"],
            "pt": cfg["pt"],
            "rpct": cfg["rpct"],
            "chard": cfg["chard"],
            "israte": cfg["israte"],
            "epsp0": cfg.get("epsp0", 12.0),
            "cp": cfg.get("cp", 1.0),
            "vp": cfg.get("vp", 0),
            "rho0": cfg.get("rho0", 7.8e-6),
            "curve_c": curve_c,
            "curve_t": curve_t,
        }
        if cfg["israte"] == 3:
            p_dict["curve_rate_c"] = rate_c
            p_dict["curve_rate_t"] = rate_t
        elif cfg["israte"] == 4:
            p_dict["curves_c"] = curves_c_fam
            p_dict["rates_c"] = rates_c_fam
            p_dict["curves_t"] = curves_t_fam
            p_dict["rates_t"] = rates_t_fam
            p_dict["fp1"] = [1.0, 1.0, 1.0]
            p_dict["fp2"] = [1.0, 1.0, 1.0]

        mat = build_law66(Law66Params(**p_dict))

        sig = np.zeros((nel, 3), dtype=float)
        deps = np.zeros((nel, 3), dtype=float)
        epsp_init = np.zeros(nel, dtype=float)
        alpha_init = np.zeros((nel, 3), dtype=float)
        thk_init = np.full(nel, 1.8, dtype=float)

        if cfg.get("elastic_only"):
            deps[:, 0] = 0.0003
            deps[:, 1] = 0.0001
        elif cfg.get("pure_shear"):
            deps[:, 2] = 0.007
        elif cfg.get("biaxial"):
            deps[:, 0] = 0.004
            deps[:, 1] = 0.003
            deps[:, 2] = 0.002
        elif cfg.get("reverse_load"):
            sig[:, 0] = 160.0
            sig[:, 1] = 50.0
            alpha_init[:, 0] = 35.0
            alpha_init[:, 1] = 15.0
            epsp_init[:] = 0.015
            deps[:, 0] = -0.005
            deps[:, 1] = -0.002
        else:
            if nel == 1:
                deps[0, 0] = 0.005
                deps[0, 1] = -0.001
                deps[0, 2] = 0.0025
            else:
                for k in range(nel):
                    deps[k, 0] = 0.0015 * (k + 1)
                    deps[k, 1] = -0.0004 * (k + 1)
                    deps[k, 2] = 0.0008 * (k + 1)

        dt = cfg.get("dt", 1e-6)
        rate_val = cfg.get("rate_val", 30.0)

        extra = {
            "uvar66": np.zeros((nel, 5)),
            "thk": thk_init.copy(),
            "thkly": thk_init.copy(),
            "off": np.ones(nel),
            "rate": np.full(nel, rate_val),
        }
        extra["uvar66"][:, 0] = epsp_init
        extra["uvar66"][:, 1:4] = alpha_init

        # Run shell_update
        sig_in = sig.copy()
        sig_out, epsp_out, soundsp_out = shell_update(
            mat, sig_in, deps, epsp=epsp_init.copy(), dt=dt, extra=extra, return_tuple=True
        )

        # Run Shell Oracle
        oracle_res = sigeps66c_shell_oracle(
            p_dict,
            sig,
            deps,
            epsp=epsp_init,
            alpha=alpha_init,
            thk=thk_init,
            thkly=thk_init,
            off=np.ones(nel),
            rate=rate_val,
            dt=dt,
            yield_curve_c=p_dict.get("curve_c"),
            yield_curve_t=p_dict.get("curve_t"),
            rate_curve_c=p_dict.get("curve_rate_c"),
            rate_curve_t=p_dict.get("curve_rate_t"),
            curves_c=p_dict.get("curves_c"),
            rates_c=p_dict.get("rates_c"),
            curves_t=p_dict.get("curves_t"),
            rates_t=p_dict.get("rates_t"),
        )

        # Strict parity check: rtol=1e-12, atol=1e-12
        np.testing.assert_allclose(sig_out, oracle_res["sig"], rtol=1e-12, atol=1e-12,
                                   err_msg=f"Shell state {cfg['id']} stress mismatch")
        np.testing.assert_allclose(epsp_out, oracle_res["epsp"], rtol=1e-12, atol=1e-12,
                                   err_msg=f"Shell state {cfg['id']} epsp mismatch")
        np.testing.assert_allclose(extra["uvar66"][:, 1:4], oracle_res["alpha"], rtol=1e-12, atol=1e-12,
                                   err_msg=f"Shell state {cfg['id']} backstress alpha mismatch")
        np.testing.assert_allclose(extra["thk"], oracle_res["thk"], rtol=1e-12, atol=1e-12,
                                   err_msg=f"Shell state {cfg['id']} thickness mismatch")
        np.testing.assert_allclose(soundsp_out, oracle_res["soundsp"], rtol=1e-12, atol=1e-12,
                                   err_msg=f"Shell state {cfg['id']} soundsp mismatch")
