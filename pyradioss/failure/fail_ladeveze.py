"""Ladevèze composite micro-damage failure model (/FAIL/LADEVEZE / /FAIL/LAD_DAMA).

Fortran origin:
- ``engine/source/materials/fail/ladeveze/fail_ladeveze.F`` (engine damage step)
- ``starter/source/materials/fail/ladeveze/hm_read_fail_ladeveze.F`` (starter reader)
- CFG: ``radioss2018/FAIL/fail_lad_dama.cfg``, ``radioss2022/FAIL/fail_lad_dama.cfg``

Theory (P. Ladevèze & E. Le Dantec 1992, O. Allix & P. Ladevèze 1992)
----------------------------------------------------------------------
Thermodynamic damage model for fiber-reinforced composite laminates:
1. Micro-damage variables:
   - d: matrix micro-cracking damage variable, affecting transverse stiffness
     E_22 (under tension only) and in-plane shear stiffness G_12.
   - d_1: longitudinal fiber damage variable affecting fiber modulus E_11.

2. Thermodynamic conjugate damage forces:
   For in-plane ply / plane stress:
       Y_11 = <sigma_11>_+^2 / [2 * K_1 * (1 - d_1)^2]    (fiber tension)
       Y_22 = <sigma_22>_+^2 / [2 * K_2 * (1 - d)^2]      (transverse tension; cracks close under compression)
       Y_12 = sigma_12^2 / [2 * K_3 * (1 - d)^2]          (in-plane shear)

   For 3D solid delamination (fail_ladeveze.F lines 165-189):
       Y_33 = <sigma_33>_+^2 / [2 * K_3 * (1 - d)^2]      (out-of-plane normal tension)
       Y_23 = sigma_23^2 / [2 * K_2 * (1 - d)^2]          (transverse shear)
       Y_13 = sigma_13^2 / [2 * K_1 * (1 - d)^2]          (longitudinal shear)

3. Equivalent thermodynamic damage force:
   - In-plane (shells): Y_D = Y_22 + gamma_1 * Y_12
   - Out-of-plane (solids): Y_D = Y_33 + gamma_1 * Y_13 + gamma_2 * Y_23
   - Irreversibility: Y_bar = sup_{tau <= t} Y_D(tau) (damage cannot heal)

4. Damage evolution:
   Delta = max(0, sqrt(Y_bar) - Y_0)
   W = Delta / (Y_c - Y_0)
   With Ladevèze delay/rate-dependent regularization:
       dot(d) = (k / a) * (1 - exp(-a * <W - d>_+))
       d_{n+1} = d_n + (k * dt / a) * (1 - exp(-a * max(0, W - d_n)))
   Quasi-static limit (or when rate effect is uncalibrated / a -> inf):
       d = min(1.0, max(d_n, W))

5. Fiber damage evolution (tension along fibers):
   If fiber thresholds Y_11_0 and Y_11_c are defined:
       d_1 = min(1.0, max(0.0, (sqrt(Y_11) - Y_11_0) / (Y_11_c - Y_11_0)))
   Or if tensile strength sigma_1t is defined:
       d_1 = 1.0 if sigma_11 >= sigma_1t else 0.0

6. Failure condition:
   Element / integration point broken when D = max(d, d_1) >= 1.0.
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-20
_INF = 1e30


def compute_damage_forces_shell(
    sxx: np.ndarray,
    syy: np.ndarray,
    sxy: np.ndarray,
    k1: float,
    k2: float,
    k3: float,
    d: np.ndarray,
    d1: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute thermodynamic damage forces Y_11, Y_22, Y_12 for plane-stress shell plies.

    Parameters
    ----------
    sxx : ndarray
        Longitudinal stress sigma_11 (fiber direction).
    syy : ndarray
        Transverse stress sigma_22 (matrix direction).
    sxy : ndarray
        In-plane shear stress sigma_12.
    k1, k2, k3 : float
        Stiffness parameters in directions 1 (fiber), 2 (matrix), 3 (shear).
    d : ndarray
        Current matrix damage variable in [0, 1].
    d1 : ndarray, optional
        Current fiber damage variable in [0, 1]. Defaults to zeros.

    Returns
    -------
    tuple of ndarrays (Y11, Y22, Y12)
    """
    if d1 is None:
        d1 = np.zeros_like(d)

    # Effective matrix damage factor (1 - d)^2
    om_d = np.maximum(1e-4, 1.0 - d)
    om_d_sq = om_d**2

    # Effective fiber damage factor (1 - d1)^2
    om_d1 = np.maximum(1e-4, 1.0 - d1)
    om_d1_sq = om_d1**2

    # Y_11: tensile stress along fibers <sigma_11>_+^2 / (2 * K_1 * (1 - d_1)^2)
    s11_pos = np.maximum(0.0, sxx)
    k1_eff = max(_TINY, k1) * om_d1_sq
    y11 = 0.5 * (s11_pos**2) / np.maximum(_TINY, k1_eff)

    # Y_22: transverse tension <sigma_22>_+^2 / (2 * K_2 * (1 - d)^2)
    # Under transverse compression (sigma_22 < 0), micro-cracks close -> Y_22 = 0
    s22_pos = np.maximum(0.0, syy)
    k2_eff = max(_TINY, k2) * om_d_sq
    y22 = 0.5 * (s22_pos**2) / np.maximum(_TINY, k2_eff)

    # Y_12: in-plane shear sigma_12^2 / (2 * K_3 * (1 - d)^2)
    k3_eff = max(_TINY, k3) * om_d_sq
    y12 = 0.5 * (sxy**2) / np.maximum(_TINY, k3_eff)

    return y11, y22, y12


def compute_damage_forces_solid(
    sig: np.ndarray,
    k1: float,
    k2: float,
    k3: float,
    d: np.ndarray,
    d1: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute thermodynamic damage forces for 3D solid elements.

    Matches OpenRadioss engine/source/materials/fail/ladeveze/fail_ladeveze.F.
    """
    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    szz = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)
    sxy = sig_arr[:, 3] if sig_arr.shape[1] > 3 else np.zeros_like(sxx)
    syz = sig_arr[:, 4] if sig_arr.shape[1] > 4 else np.zeros_like(sxx)
    szx = sig_arr[:, 5] if sig_arr.shape[1] > 5 else np.zeros_like(sxx)

    if d1 is None:
        d1 = np.zeros_like(d)

    om_d = np.maximum(1e-4, 1.0 - d)
    om_d_sq = om_d**2
    om_d1 = np.maximum(1e-4, 1.0 - d1)
    om_d1_sq = om_d1**2

    # Direction 33 (out-of-plane normal tension)
    s33_pos = np.maximum(0.0, szz)
    yd3 = 0.5 * (s33_pos**2) / np.maximum(_TINY, max(_TINY, k3) * om_d_sq)

    # Direction 23 (transverse shear)
    yd2 = 0.5 * (syz**2) / np.maximum(_TINY, max(_TINY, k2) * om_d_sq)

    # Direction 13 (longitudinal shear)
    yd1 = 0.5 * (szx**2) / np.maximum(_TINY, max(_TINY, k1) * om_d_sq)

    # In-plane forces
    s11_pos = np.maximum(0.0, sxx)
    y11 = 0.5 * (s11_pos**2) / np.maximum(_TINY, max(_TINY, k1) * om_d1_sq)
    s22_pos = np.maximum(0.0, syy)
    y22 = 0.5 * (s22_pos**2) / np.maximum(_TINY, max(_TINY, k2) * om_d_sq)
    y12 = 0.5 * (sxy**2) / np.maximum(_TINY, max(_TINY, k3) * om_d_sq)

    return y11, y22, yd3, y12, yd2, yd1


def _extract_params(fail) -> dict[str, float]:
    """Extract and sanitize Ladevèze model parameters from card."""
    p = fail.params if hasattr(fail, "params") else {}
    k1 = float(p.get("k1", p.get("K1", 1.0e30)))
    k2 = float(p.get("k2", p.get("K2", 1.0e30)))
    k3 = float(p.get("k3", p.get("K3", 1.0e30)))
    if k1 == 0.0:
        k1 = 1.0e30
    if k2 == 0.0:
        k2 = 1.0e30
    if k3 == 0.0:
        k3 = 1.0e30

    gamma1 = float(p.get("gamma1", p.get("Gamma_1", p.get("GAMMA_1", 0.0))))
    gamma2 = float(p.get("gamma2", p.get("Gamma_2", p.get("GAMMA_2", 0.0))))

    y0_raw = float(p.get("y0", p.get("Y0", 0.0)))
    yc_raw = float(p.get("yc", p.get("Yc", p.get("YC", 0.0))))

    # In OpenRadioss hm_read_fail_ladeveze.F:
    # Y0 = SQRT(Y0), YC = SQRT(YC)
    # If Y0 or Yc were already passed in root-energy form, we honor them
    y0 = np.sqrt(y0_raw) if y0_raw > 0.0 else float(p.get("sqrt_y0", 0.0))
    yc = np.sqrt(yc_raw) if yc_raw > 0.0 else float(p.get("sqrt_yc", 0.0))
    if yc == 0.0 or yc <= y0:
        yc = 2.0 * max(y0, 1.0e-3)

    k_lad = float(p.get("k", p.get("k_lad", p.get("k_LAD_DAMA", p.get("K", 0.0)))))
    a_dama = float(p.get("a", p.get("a_dama", p.get("a_DAMA", p.get("A", 1.0e30)))))
    if a_dama == 0.0:
        a_dama = 1.0e30

    tau_max = float(p.get("tau_max", p.get("Tau_max", p.get("TAU_MAX", 1.0e20))))
    ifail_sh = int(p.get("ifail_sh", p.get("Ifail_sh", 1)))
    ifail_so = int(p.get("ifail_so", p.get("Ifail_so", 1)))

    # Optional fiber thresholds
    y11_0 = float(p.get("y11_0", p.get("Y11_0", p.get("y11_start", 0.0))))
    y11_c = float(p.get("y11_c", p.get("Y11_c", p.get("y11_crit", 0.0))))
    sigma_1t = float(p.get("sigma_1t", p.get("SIGMA_1T", p.get("xt", 0.0))))

    return {
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "gamma1": gamma1,
        "gamma2": gamma2,
        "y0": y0,
        "yc": yc,
        "k_lad": k_lad,
        "a_dama": a_dama,
        "tau_max": tau_max,
        "ifail_sh": ifail_sh,
        "ifail_so": ifail_so,
        "y11_0": y11_0,
        "y11_c": y11_c,
        "sigma_1t": sigma_1t,
    }


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance Ladevèze damage for a 3D solid slice.

    Parameters
    ----------
    fail : Failure object or FailureModel
        Model parameters.
    sig : ndarray of shape (m, 6)
        Stress tensor [s_xx, s_yy, s_zz, s_xy, s_yz, s_zx].
    d_epsp : ndarray of shape (m,)
        Plastic strain increment (not directly used for elasticity-damage coupling).
    deps : ndarray of shape (m, 6) or None
        Total strain increment tensor.
    dt : float
        Current time step size.
    dama : ndarray of shape (m,)
        Persistent accumulated damage array (updated in-place).
    tstar : optional

    Returns
    -------
    ndarray of bool
        Mask of broken elements (dama >= 1.0).
    """
    p = _extract_params(fail)
    k1, k2, k3 = p["k1"], p["k2"], p["k3"]
    gama1, gama2 = p["gamma1"], p["gamma2"]
    y0, yc = p["y0"], p["yc"]
    k_lad, a_dama = p["k_lad"], p["a_dama"]

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    szz = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)
    sxy = sig_arr[:, 3] if sig_arr.shape[1] > 3 else np.zeros_like(sxx)
    syz = sig_arr[:, 4] if sig_arr.shape[1] > 4 else np.zeros_like(sxx)
    szx = sig_arr[:, 5] if sig_arr.shape[1] > 5 else np.zeros_like(sxx)

    m = len(sxx)
    d_curr = np.clip(np.asarray(dama, dtype=float), 0.0, 1.0)

    # Compute thermodynamic damage forces
    y11, y22, yd3, y12, yd2, yd1 = compute_damage_forces_solid(sig_arr, k1, k2, k3, d_curr)

    # Equivalent thermodynamic force
    # If out-of-plane delamination formulation (fail_ladeveze.F):
    # YD = YD3 + GAMA1 * YD1 + GAMA2 * YD2
    # If in-plane dominated (szx == 0, syz == 0, szz == 0), fallback to in-plane:
    has_oop = np.any(np.abs(szz) > 1e-6) or np.any(np.abs(syz) > 1e-6) or np.any(np.abs(szx) > 1e-6)
    if has_oop or (gama2 > 0.0 and gama1 > 0.0):
        yd = yd3 + gama1 * yd1 + gama2 * yd2
    else:
        yd = y22 + (gama1 if gama1 > 0.0 else 1.0) * y12

    # Ladevèze damage evolution
    # Delta = max(0, sqrt(YD) - Y0)
    # W = Delta / (YC - Y0)
    sqrt_yd = np.sqrt(np.maximum(0.0, yd))
    delta = np.maximum(0.0, sqrt_yd - y0)
    denom = max(_TINY, yc - y0)
    w_target = np.clip(delta / denom, 0.0, 1.0)

    # Delay / rate-dependent update
    if k_lad > 0.0 and a_dama < 1.0e15 and dt > 0.0:
        cc = np.maximum(0.0, w_target - d_curr)
        fac = k_lad * dt / a_dama
        d_new = d_curr + fac * (1.0 - np.exp(-a_dama * cc))
    else:
        d_new = np.maximum(d_curr, w_target)

    # Fiber damage d_1 check
    d1 = np.zeros(m, dtype=float)
    if p["sigma_1t"] > 0.0:
        d1 = np.where(sxx >= p["sigma_1t"], 1.0, 0.0)
    elif p["y11_c"] > p["y11_0"] and p["y11_c"] > 0.0:
        sqrt_y11 = np.sqrt(np.maximum(0.0, y11))
        d1 = np.clip((sqrt_y11 - p["y11_0"]) / max(_TINY, p["y11_c"] - p["y11_0"]), 0.0, 1.0)

    # Update damage variable
    d_total = np.maximum(d_new, d1)
    dama[:] = np.clip(np.maximum(dama, d_total), 0.0, 1.0)

    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Ladevèze damage for a plane-stress shell layer.

    Parameters
    ----------
    fail : Failure object or FailureModel
    sig : ndarray of shape (m, 3) or (m, 5)
        Plane stress tensor [s_xx, s_yy, s_xy, (s_yz, s_zx)].
    d_epsp : ndarray of shape (m,)
    deps : ndarray
    dt : float
    dama : ndarray of shape (m,)
        Persistent accumulated damage array (updated in-place).
    tstar : optional
    eps_tot : optional

    Returns
    -------
    ndarray of bool
        Mask of broken points in this layer.
    """
    p = _extract_params(fail)
    k1, k2, k3 = p["k1"], p["k2"], p["k3"]
    gama1, gama2 = p["gamma1"], p["gamma2"]
    y0, yc = p["y0"], p["yc"]
    k_lad, a_dama = p["k_lad"], p["a_dama"]

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)

    m = len(sxx)
    d_curr = np.clip(np.asarray(dama, dtype=float), 0.0, 1.0)

    # In-plane thermodynamic damage forces
    y11, y22, y12 = compute_damage_forces_shell(sxx, syy, sxy, k1, k2, k3, d_curr)

    # Equivalent damage force for in-plane ply:
    # Transverse micro-cracking coupled with in-plane shear:
    # Y_D = Y_22 + gamma_1 * Y_12
    coupling = gama1 if gama1 > 0.0 else 1.0
    yd = y22 + coupling * y12

    # Damage evolution
    sqrt_yd = np.sqrt(np.maximum(0.0, yd))
    delta = np.maximum(0.0, sqrt_yd - y0)
    denom = max(_TINY, yc - y0)
    w_target = np.clip(delta / denom, 0.0, 1.0)

    # Delay / rate-dependent update
    if k_lad > 0.0 and a_dama < 1.0e15 and dt > 0.0:
        cc = np.maximum(0.0, w_target - d_curr)
        fac = k_lad * dt / a_dama
        d_new = d_curr + fac * (1.0 - np.exp(-a_dama * cc))
    else:
        d_new = np.maximum(d_curr, w_target)

    # Fiber damage d_1
    d1 = np.zeros(m, dtype=float)
    if p["sigma_1t"] > 0.0:
        d1 = np.where(sxx >= p["sigma_1t"], 1.0, 0.0)
    elif p["y11_c"] > p["y11_0"] and p["y11_c"] > 0.0:
        sqrt_y11 = np.sqrt(np.maximum(0.0, y11))
        d1 = np.clip((sqrt_y11 - p["y11_0"]) / max(_TINY, p["y11_c"] - p["y11_0"]), 0.0, 1.0)

    d_total = np.maximum(d_new, d1)
    dama[:] = np.clip(np.maximum(dama, d_total), 0.0, 1.0)

    return dama >= 1.0
