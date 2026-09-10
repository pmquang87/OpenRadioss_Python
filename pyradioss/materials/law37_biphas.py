"""LAW37 — Biphasic fluid-gas material (/MAT/LAW37, /MAT/BIPHAS, /MAT/BIPHASIC).

Upstream Fortran origins:
- Constitutive Engine Kernel: ``engine/source/materials/mat/mat037/sigeps37.F``
- Starter Card Reader: ``starter/source/materials/mat/mat037/hm_read_mat37.F``
- HyperMesh CFG Schema: ``hm_cfg_files/config/CFG/radioss2018/MAT/matl37_biphas.cfg``

Physics
-------
Biphasic material law: liquid / gas mixture.
- Liquid is modeled with a linear EOS:
    P_1 = P_0 + C_1 * mu_1 = R_1 * rho_1 - C_1 + P_0
  where C_1 is bulk modulus, R_1 = C_1 / rho_l0, and P_0 is reference pressure.
- Gas is modeled with an ideal gas isentropic EOS:
    P_2 = P_0 * (rho_2 / rho_g0)^gamma
- Mixture equilibrium requires pressure equilibrium P_1 = P_2 and volume conservation:
    F_1 = M_1 / rho_1 + M_2 / rho_2 - V = 0
    F_2 = P_1 - P_2 = 0
- Solvers:
  - ISOLVER = 1 (default legacy): 2-step algebraic iteration with chord linearization.
    Sound speed: c = sqrt(C_1 / rho_1).
  - ISOLVER = 2 (Newton-Raphson): 2D Newton algorithm with convergence tolerance 1e-10 (max 20 iterations).
    Sound speed: Wood's mixture formula:
      SSP_1 = R_1 * rho_1
      SSP_2 = gamma * P_0 * (rho_2 / rho_g0)^gamma
      1 / (rho * c^2) = alpha_v1 / SSP_1 + alpha_v2 / SSP_2
- Viscous stresses:
    mu = (B_1 * rho_1 * nu_l + B_2 * rho_2 * nu_g) / rho
    mu_vol = (B_1 * rho_1 * nu_vol_l + B_2 * rho_2 * nu_vol_g) / rho
    sigma_v = 2 * mu * edot + mu_vol * tr(edot) * I
- Total stress:
    sigma = -P_total * I + sigma_v
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM30 = 1e-30
_EM20 = 1e-20
_EM10 = 1e-10


def _get_val(p: dict, rec: Any, keys: Union[str, list[str]], default: float = 0.0) -> float:
    """Helper to extract float parameter from parameter dict or record."""
    if isinstance(keys, str):
        keys = [keys]
    for k in keys:
        if isinstance(p, dict) and k in p and p[k] is not None:
            try:
                return float(p[k])
            except (ValueError, TypeError):
                pass
        if isinstance(rec, dict) and k in rec and rec[k] is not None:
            try:
                return float(rec[k])
            except (ValueError, TypeError):
                pass
        if hasattr(rec, k) and getattr(rec, k) is not None:
            try:
                return float(getattr(rec, k))
            except (ValueError, TypeError):
                pass
    return default


def build_law37(rec: Any) -> Material:
    """Construct a Material entity for /MAT/LAW37 (/MAT/BIPHAS, /MAT/BIPHASIC).

    Fortran origin: ``starter/source/materials/mat/mat037/hm_read_mat37.F``.

    Parameters
    ----------
    rec : GenericMaterialRecord, dict, or object
        Parsed card record from CFG or deck reader.

    Returns
    -------
    Material
        Material entity configured with law=37, law_name="LAW37", rho0, title, and params.
    """
    if hasattr(rec, "params") and getattr(rec, "params") is not None:
        p = getattr(rec, "params")
    elif isinstance(rec, dict) and "params" in rec and isinstance(rec["params"], dict):
        p = rec["params"]
    elif isinstance(rec, dict):
        p = rec
    else:
        p = {}

    if isinstance(rec, dict):
        rec_id = int(rec.get("id", rec.get("mat_id", rec.get("user_id", 1))))
        title = str(rec.get("title", f"LAW37_{rec_id}"))
    else:
        rec_id = int(getattr(rec, "id", getattr(rec, "mat_id", getattr(rec, "user_id", 1))))
        title = str(getattr(rec, "title", f"LAW37_{rec_id}"))

    # Liquid parameters (submat 1)
    rho_l0 = _get_val(p, rec, ["rho_l0", "Lqud_Rho_l", "RHO_L0", "RHO10", "rho10", "rho_l"], 1000.0)
    c_l = _get_val(p, rec, ["c_l", "C_l", "C1", "c1", "bulk_l", "K_l"], 2.2e9)
    alpha1 = _get_val(p, rec, ["alpha1", "ALPHA1", "ALPHA_l", "alpha_l", "a1", "A1"], 1.0)
    nu_l = _get_val(p, rec, ["nu_l", "Nu_l", "NU_l", "vis_l", "visa1", "VISA1"], 0.0)
    nu_vol_l = _get_val(p, rec, ["nu_vol_l", "Bulk_Ratio_l", "bulk_ratio_l", "NU_VOL_l", "visb1", "VISB1"], 0.0)

    # Gas parameters (submat 2)
    rho_g0 = _get_val(p, rec, ["rho_g0", "Lqud_Rho_g", "RHO_G0", "RHO20", "rho20", "rho_g"], 1.2)
    gamma = _get_val(p, rec, ["gamma", "Lqud_Gamma_bulk", "GAM", "gam", "gamma_bulk"], 1.4)
    p0 = _get_val(p, rec, ["p0", "Lqud_P0", "P0", "p_0"], 1.01325e5)
    nu_g = _get_val(p, rec, ["nu_g", "Nu_g", "NU_g", "vis_g", "visa2", "VISA2"], 0.0)
    nu_vol_g = _get_val(p, rec, ["nu_vol_g", "Bulk_Ratio_g", "bulk_ratio_g", "NU_VOL_g", "visb2", "VISB2"], 0.0)

    # Input validation / clamping (hm_read_mat37.F lines 142-167)
    alpha1 = max(0.0, min(1.0, alpha1))
    if rho_l0 < 0.0:
        rho_l0 = _EM20
    if rho_g0 < 0.0:
        rho_g0 = _EM20

    # Initial density rho0 (MAT_RHO, defaults to rho_l0 * alpha1 + (1 - alpha1) * rho_g0)
    rho0_input = _get_val(p, rec, ["rho0", "MAT_RHO", "density", "rho", "Refer_Rho", "refer_rho"], 0.0)
    if rho0_input <= 0.0:
        rho0 = rho_l0 * alpha1 + (1.0 - alpha1) * rho_g0
    else:
        rho0 = rho0_input

    # Pressure shift (MAT_PSH: if psh == 0 then -p0 else -psh, stored as pshift)
    # hm_read_mat37.F lines 180-184
    psh_raw = _get_val(p, rec, ["psh", "MAT_PSH", "pshift", "PSHIFT"], 0.0)
    if psh_raw == 0.0:
        pshift = -p0
    else:
        pshift = -psh_raw

    # Solver choice: 1 (legacy 2-iteration) or 2 (Newton-Raphson). Default 1, or 2 if INT22 > 0.
    int22_val = _get_val(p, rec, ["INT22", "int22"], 0.0)
    isolver_raw = None
    for k in ("isolver", "ISOLVER", "solver"):
        if isinstance(p, dict) and k in p and p[k] is not None:
            isolver_raw = int(p[k])
            break
        if isinstance(rec, dict) and k in rec and rec[k] is not None:
            isolver_raw = int(rec[k])
            break
        if hasattr(rec, k) and getattr(rec, k) is not None:
            isolver_raw = int(getattr(rec, k))
            break
    if isolver_raw is not None:
        isolver = isolver_raw
    elif int22_val > 0.0:
        isolver = 2
    else:
        isolver = 1

    pmin = _get_val(p, rec, ["pmin", "PMIN"], -p0)
    r1 = c_l / rho_l0 if rho_l0 > 0.0 else 0.0

    # Wood's mixture bulk modulus at reference state for elastic estimates
    alpha_v1_0 = (alpha1 * rho0 / rho_l0) if (rho_l0 > 0.0) else alpha1
    alpha_v1_0 = max(0.0, min(1.0, alpha_v1_0))
    alpha_v2_0 = 1.0 - alpha_v1_0
    k1 = c_l
    k2 = gamma * p0
    if k1 > 0.0 and k2 > 0.0:
        comp_mix = alpha_v1_0 / k1 + alpha_v2_0 / k2
        k_mix = 1.0 / comp_mix if comp_mix > 0.0 else c_l
    else:
        k_mix = c_l

    params: Dict[str, Any] = {
        "rho_l0": rho_l0,
        "c_l": c_l,
        "alpha1": alpha1,
        "nu_l": nu_l,
        "nu_vol_l": nu_vol_l,
        "rho_g0": rho_g0,
        "gamma": gamma,
        "p0": p0,
        "nu_g": nu_g,
        "nu_vol_g": nu_vol_g,
        "rho0": rho0,
        "psh": psh_raw,
        "pshift": pshift,
        "isolver": isolver,
        "pmin": pmin,
        "r1": r1,
        "R1": r1,
        "PMIN": pmin,
        "PSH": psh_raw,
        "PSHIFT": pshift,
        "ISOLVER": isolver,
        "visa1": nu_l,
        "visb1": nu_vol_l,
        "visa2": nu_g,
        "visb2": nu_vol_g,
        # CFG aliases
        "Lqud_Rho_l": rho_l0,
        "C_l": c_l,
        "ALPHA1": alpha1,
        "Nu_l": nu_l,
        "Bulk_Ratio_l": nu_vol_l,
        "Lqud_Rho_g": rho_g0,
        "Lqud_Gamma_bulk": gamma,
        "Lqud_P0": p0,
        "Nu_g": nu_g,
        "Bulk_Ratio_g": nu_vol_g,
        "MAT_RHO": rho0,
        "MAT_PSH": psh_raw,
        # Elastic parameters
        "K": k_mix,
        "E": 0.0,
        "nu": 0.5,
        "G": 0.0,
    }

    mat = Material(id=rec_id, law=37, rho0=rho0, title=title, params=params, law_name="LAW37")
    return mat


def init_uv37(
    mat: Material,
    nel: int = 1,
    rho: Optional[Union[float, np.ndarray]] = None,
    sig: Optional[np.ndarray] = None,
    pshift: Optional[float] = None,
) -> np.ndarray:
    """Initialize history array uv37 for LAW37 elements.

    Fortran origins:
    - Starter: ``starter/source/materials/mat/mat037/m37init.F`` lines 83-105
    - Engine cycle 0: ``engine/source/materials/mat/mat037/sigeps37.F`` lines 180-197

    Variables in uv37:
    - uv37[:, 0]: Liquid mass per unit volume B_1 = alpha_v * rho_1
    - uv37[:, 1]: Gas density rho_2
    - uv37[:, 2]: Liquid density rho_1
    - uv37[:, 3]: Liquid volume fraction alpha_{v, 1}
    - uv37[:, 4]: Gas volume fraction alpha_{v, 2} = 1 - alpha_{v, 1}

    Returns
    -------
    uv37 : np.ndarray
        Array of shape (nel, 5).
    """
    p = mat.params if mat.params is not None else {}
    rho_l0 = float(p.get("rho_l0", 1000.0))
    c_l = float(p.get("c_l", 2.2e9))
    alpha1 = float(p.get("alpha1", 1.0))
    rho_g0 = float(p.get("rho_g0", 1.2))
    gamma = float(p.get("gamma", 1.4))
    p0 = float(p.get("p0", 1.01325e5))

    if pshift is None:
        psh_raw = float(p.get("psh", 0.0))
        pshift = float(p.get("pshift", -p0 if psh_raw == 0.0 else -psh_raw))

    if rho is None:
        rho0_ref = float(getattr(mat, "rho0", None) or (rho_l0 * alpha1 + (1.0 - alpha1) * rho_g0))
        rho_arr = np.full(nel, rho0_ref, dtype=float)
    elif np.isscalar(rho):
        rho_arr = np.full(nel, float(rho), dtype=float)
    else:
        rho_arr = np.asarray(rho, dtype=float).copy()
        if rho_arr.shape[0] != nel:
            nel = rho_arr.shape[0]

    if sig is None:
        sig_arr = np.zeros((nel, 6), dtype=float)
    else:
        sig_arr = np.asarray(sig, dtype=float)

    uv37 = np.zeros((nel, 5), dtype=float)

    # sigeps37.F line 181: P = MAX(EM30, (-SIGOXX - SIGOYY - SIGOZZ)*THIRD) - PSH
    sig_hydro = -(sig_arr[:, 0] + sig_arr[:, 1] + sig_arr[:, 2]) / 3.0
    p_init = np.maximum(_EM30, sig_hydro) - pshift

    if gamma * c_l >= _EM30:
        # Liquid and gas correctly defined (sigeps37.F lines 183-193 / m37init.F lines 91-105)
        mu1p1 = (p_init - p0) / c_l + 1.0
        base_p = np.maximum(_EM30, p_init / p0)
        mu2p1 = base_p ** (1.0 / gamma)
        rho1_init = rho_l0 * mu1p1
        rho2_init = rho_g0 * mu2p1

        if alpha1 >= 1.0 - 1e-10:
            a_init = np.ones(nel, dtype=float)
            b1 = rho_arr.copy()
        elif alpha1 <= 1e-10:
            a_init = np.zeros(nel, dtype=float)
            b1 = np.zeros(nel, dtype=float)
        else:
            denom = rho1_init - rho2_init
            denom = np.where(np.abs(denom) < _EM30, _EM30, denom)
            a_init = (rho_arr - rho2_init) / denom
            a_init = np.clip(a_init, 0.0, 1.0)
            a_init = np.where(a_init < _EM20, 0.0, a_init)
            b1 = a_init * rho1_init

        uv37[:, 0] = b1
        uv37[:, 1] = rho2_init
        uv37[:, 2] = rho1_init
        uv37[:, 3] = a_init
        uv37[:, 4] = 1.0 - a_init
    else:
        # Boundary element (sigeps37.F line 195)
        uv37[:, 2] = rho_arr

    return uv37


def solid_update(
    mat: Material,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    *args: Any,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Constitutive stress update for LAW37 biphasic liquid-gas material.

    Fortran origin: ``engine/source/materials/mat/mat037/sigeps37.F``.

    Parameters
    ----------
    mat : Material
        LAW37 material instance.
    sig : np.ndarray
        Old stress tensor (n, 6) in Voigt order [xx, yy, zz, xy, yz, zx]. Modified in-place.
    deps : np.ndarray
        Strain increment tensor (n, 6) with engineering shear [xx, yy, zz, xy, yz, zx].
    epsp : np.ndarray, optional
        Effective plastic strain (not modified for fluid).
    dt : float
        Current cycle time step.
    extra : dict, optional
        Simulation context, containing "rho" and persistent "uv37" history array.

    Returns
    -------
    sig : np.ndarray
        New Cauchy stress tensor (n, 6).
    epsp : np.ndarray or None
        Plastic strain array (unchanged).
    soundsp : np.ndarray
        Speed of sound per element (n,).
    """
    nel = sig.shape[0]
    if nel == 0:
        return sig, epsp, np.zeros(0, dtype=float)

    p = mat.params if mat.params is not None else {}
    rho_l0 = float(p.get("rho_l0", 1000.0))
    c_l = float(p.get("c_l", 2.2e9))
    alpha1 = float(p.get("alpha1", 1.0))
    nu_l = float(p.get("nu_l", 0.0))
    nu_vol_l = float(p.get("nu_vol_l", 0.0))

    rho_g0 = float(p.get("rho_g0", 1.2))
    gamma = float(p.get("gamma", 1.4))
    p0 = float(p.get("p0", 1.01325e5))
    nu_g = float(p.get("nu_g", 0.0))
    nu_vol_g = float(p.get("nu_vol_g", 0.0))

    pshift = float(p.get("pshift", -p0 if p.get("psh", 0.0) == 0.0 else -float(p.get("psh", 0.0))))
    isolver = int(p.get("isolver", 1))
    pmin = float(p.get("pmin", -p0))
    r1 = c_l / rho_l0 if rho_l0 > 0.0 else 0.0
    rho0_ref = float(getattr(mat, "rho0", None) or (rho_l0 * alpha1 + (1.0 - alpha1) * rho_g0))
    if rho0_ref <= _EM30:
        rho0_ref = rho_l0 * alpha1 + (1.0 - alpha1) * rho_g0

    if extra is None:
        extra = {}

    current_rho = extra.get("rho")
    if current_rho is None:
        rho = np.full(nel, float(getattr(mat, "rho0", 1000.0) or 1000.0), dtype=float)
    elif np.isscalar(current_rho):
        rho = np.full(nel, float(current_rho), dtype=float)
    elif len(current_rho) != nel:
        rho = np.full(nel, float(getattr(mat, "rho0", 1000.0) or 1000.0), dtype=float)
    else:
        rho = np.asarray(current_rho, dtype=float).copy()

    # History array uv37: shape (nel, 5)
    # uv37[:, 0]: liquid mass per unit volume B_1 = alpha_{v, 1} * rho_1
    # uv37[:, 1]: gas density rho_2
    # uv37[:, 2]: liquid density rho_1
    # uv37[:, 3]: liquid volume fraction alpha_{v, 1}
    # uv37[:, 4]: gas volume fraction alpha_{v, 2} = 1 - alpha_{v, 1}
    uv37 = extra.get("uv37")
    if uv37 is None or uv37.shape != (nel, 5):
        uv37 = np.zeros((nel, 5), dtype=float)
        extra["uv37"] = uv37

    # Initialization at cycle 0 (or uninitialized)
    # Follow sigeps37.F lines 179-197 and m37init.F lines 83-105
    is_time_zero = (extra.get("time", None) == 0.0) or (kwargs.get("time", None) == 0.0)
    uninit = (uv37[:, 1] <= 0.0) & (uv37[:, 2] <= 0.0)
    to_init = np.ones(nel, dtype=bool) if is_time_zero else uninit
    if np.any(to_init):
        # If uninitialized and density differs from rho0_ref, scale B1 by rho/rho0_ref for mass conservation
        rho_init = rho[to_init]
        uv37_init = init_uv37(
            mat=mat,
            nel=int(np.count_nonzero(to_init)),
            rho=rho_init,
            sig=sig[to_init],
            pshift=pshift,
        )
        if not is_time_zero and np.any(np.abs(rho_init - rho0_ref) > 1e-6) and 1e-10 < alpha1 < 1.0 - 1e-10:
            uv37_ref = init_uv37(
                mat=mat,
                nel=int(np.count_nonzero(to_init)),
                rho=np.full(int(np.count_nonzero(to_init)), rho0_ref),
                sig=sig[to_init],
                pshift=pshift,
            )
            b1_0 = uv37_ref[:, 0]
            uv37_init[:, 0] = np.clip(b1_0 * (rho_init / rho0_ref), 0.0, rho_init)
            uv37_init[:, 3] = np.clip(uv37_init[:, 0] / np.maximum(_EM30, uv37_init[:, 2]), 0.0, 1.0)
            uv37_init[:, 4] = 1.0 - uv37_init[:, 3]
        uv37[to_init] = uv37_init

    # Boundary element input check (sigeps37.F lines 208-232)
    if gamma * c_l < _EM30:
        soundsp = np.full(nel, _EM30, dtype=float)
        for i in range(nel):
            if uv37[i, 2] / rho_l0 < 0.5:
                uv37[i, 0] = 0.0
                uv37[i, 1] = rho[i]
                uv37[i, 3] = 0.0
                uv37[i, 4] = 1.0
            else:
                uv37[i, 0] = uv37[i, 2]
                uv37[i, 1] = rho_g0
                uv37[i, 3] = 1.0
                uv37[i, 4] = 0.0
        return sig, epsp, soundsp

    # Strain rates
    if dt > _EM20:
        deps_rate = deps / dt
    else:
        deps_rate = np.zeros_like(deps)
    edot_vol = deps_rate[:, 0] + deps_rate[:, 1] + deps_rate[:, 2]

    soundsp = np.zeros(nel, dtype=float)
    pressure = np.zeros(nel, dtype=float)

    if isolver == 2:
        # =====================================================================
        # Newton-Raphson Solver (sigeps37.F lines 234-336)
        # =====================================================================
        tol = _EM10
        niter = 20

        vol = extra.get("volume", extra.get("vol", None))
        if vol is None:
            vol_arr = np.ones(nel, dtype=float)
        elif np.isscalar(vol):
            vol_arr = np.full(nel, float(vol), dtype=float)
        else:
            vol_arr = np.asarray(vol, dtype=float)

        for i in range(nel):
            rho_i = rho[i]
            vol_i = vol_arr[i]
            mas = rho_i * vol_i
            mas1 = uv37[i, 0] * vol_i
            mas2 = mas - mas1
            rho2 = uv37[i, 1]
            rho1 = uv37[i, 2]

            if mas1 / mas < _EM10:
                # Phase 2 (pure gas, sigeps37.F lines 250-257)
                uv37[i, 0] = 0.0
                uv37[i, 3] = 0.0
                uv37[i, 4] = 1.0
                rho2 = mas / vol_i
                uv37[i, 1] = rho2
                p_eq = p0 * (rho2 / rho_g0) ** gamma
            elif mas2 / mas < _EM10:
                # Phase 1 (pure liquid, sigeps37.F lines 258-265)
                rho1 = mas / vol_i
                uv37[i, 0] = rho1
                uv37[i, 2] = rho1
                uv37[i, 3] = 1.0
                uv37[i, 4] = 0.0
                p_eq = r1 * rho1 - c_l + p0
            else:
                # 2D Newton iteration for (rho1, rho2, sigeps37.F lines 267-291)
                if rho1 <= 0.0:
                    rho1 = rho_l0
                if rho2 <= 0.0:
                    rho2 = rho_g0

                err = 1e30
                it = 1
                while it < niter and err > tol:
                    p1 = r1 * rho1 - c_l + p0
                    p2 = p0 * (rho2 / rho_g0) ** gamma
                    f1 = mas1 / rho1 + mas2 / rho2 - vol_i
                    f2 = p1 - p2
                    df11 = -mas1 / (rho1 * rho1)
                    df12 = -mas2 / (rho2 * rho2)
                    df21 = r1
                    df22 = -gamma * p0 / (rho_g0 ** gamma) * (rho2 ** (gamma - 1.0))
                    det = df11 * df22 - df12 * df21
                    if abs(det) < _EM30:
                        break
                    drho1 = (-df22 * f1 + df12 * f2) / det
                    drho2 = (df21 * f1 - df11 * f2) / det
                    drho1 = min(3.0 * rho1, max(drho1, -0.5 * rho1))
                    drho2 = min(3.0 * rho2, max(drho2, -0.5 * rho2))
                    rho1 += drho1
                    rho2 += drho2
                    err = abs(drho1 / rho1) + abs(drho2 / rho2)
                    it += 1

                p_eq = r1 * rho1 - c_l + p0

            # Sound speed Wood's mixture formula (sigeps37.F lines 293-314)
            ssp1 = r1 * rho1
            ssp2 = gamma * p0 * (rho2 / rho_g0) ** gamma
            uv37[i, 1] = rho2
            uv37[i, 2] = rho1
            if rho1 > _EM30:
                uv37[i, 3] = uv37[i, 0] / rho1
            else:
                uv37[i, 3] = 0.0
            if uv37[i, 3] < _EM20:
                uv37[i, 3] = 0.0
            uv37[i, 4] = 1.0 - uv37[i, 3]

            term1 = (uv37[i, 3] / ssp1) if ssp1 > 0.0 else 0.0
            term2 = (uv37[i, 4] / ssp2) if ssp2 > 0.0 else 0.0
            ssp_tot = term1 + term2
            if ssp_tot > 0.0 and rho_i > 0.0:
                soundsp[i] = math.sqrt(1.0 / (ssp_tot * rho_i))
            else:
                soundsp[i] = _EM30

            pressure[i] = max(pmin, p_eq) + pshift

    else:
        # =====================================================================
        # Legacy Solver (ISOLVER = 1, sigeps37.F lines 340-392)
        # =====================================================================
        rho2_arr = uv37[:, 1].copy()
        rho2_arr = np.where(rho2_arr <= 0.0, rho_g0, rho2_arr)

        # Iteration 1 (sigeps37.F lines 345-356)
        pold = p0 * (rho2_arr / rho_g0) ** gamma
        r2 = gamma * pold / rho2_arr
        c2 = -(1.0 - gamma) * pold + p0
        c12 = c_l - c2
        b1 = uv37[:, 0].copy()
        b2 = rho - b1
        a = r1
        b = 0.5 * (b1 * r1 + b2 * r2 + c12)
        c_quad = b1 * c12
        disc1 = np.maximum(0.0, b * b - a * c_quad)
        rho1_arr = (b + np.sqrt(disc1)) / a
        p_iter1 = r1 * rho1_arr - c_l
        rhn2 = np.maximum(_EM30, (p_iter1 + c2) / r2)

        # Iteration 2 (sigeps37.F lines 358-363)
        pn2 = pold + p0 * (rhn2 / rho_g0) ** gamma
        r2 = gamma * pn2 / (rho2_arr + rhn2)
        b = 0.5 * (b1 * r1 + b2 * r2 + c12)
        disc2 = np.maximum(0.0, b * b - a * c_quad)
        rho1_arr = (b + np.sqrt(disc2)) / a
        p_iter2 = r1 * rho1_arr - c_l
        rho2_arr = np.maximum(_EM30, (p_iter2 + c2) / r2)

        uv37[:, 1] = rho2_arr
        uv37[:, 2] = rho1_arr
        uv37[:, 3] = np.where(rho1_arr > _EM30, uv37[:, 0] / rho1_arr, 0.0)
        uv37[:, 3] = np.where(uv37[:, 3] < _EM20, 0.0, uv37[:, 3])
        uv37[:, 4] = 1.0 - uv37[:, 3]

        pressure = np.maximum(pmin, p_iter2) + p0 + pshift
        soundsp = np.sqrt(np.maximum(0.0, c_l / np.maximum(_EM30, rho1_arr)))

    # Ensure non-negative state variables (sigeps37.F lines 331-335 / 386-390)
    np.maximum(0.0, uv37, out=uv37)

    # Viscous stresses (sigeps37.F lines 319-328 / 374-383)
    b1 = uv37[:, 0]
    b2 = rho - b1
    rho1 = uv37[:, 2]
    rho2 = uv37[:, 1]

    mu = (b1 * rho1 * nu_l + b2 * rho2 * nu_g) / rho
    mu_vol = (b1 * rho1 * nu_vol_l + b2 * rho2 * nu_vol_g) / rho

    sig_v_xx = 2.0 * mu * deps_rate[:, 0] + mu_vol * edot_vol
    sig_v_yy = 2.0 * mu * deps_rate[:, 1] + mu_vol * edot_vol
    sig_v_zz = 2.0 * mu * deps_rate[:, 2] + mu_vol * edot_vol
    sig_v_xy = mu * deps_rate[:, 3]
    sig_v_yz = mu * deps_rate[:, 4]
    sig_v_zx = mu * deps_rate[:, 5]

    # Total stress (sigeps37.F lines 316-318, 371-373 and mulaw.F90 SIGN + SIGV)
    sig[:, 0] = -pressure + sig_v_xx
    sig[:, 1] = -pressure + sig_v_yy
    sig[:, 2] = -pressure + sig_v_zz
    sig[:, 3] = sig_v_xy
    sig[:, 4] = sig_v_yz
    sig[:, 5] = sig_v_zx

    if extra is not None:
        extra["viscmax"] = 2.0 * mu + mu_vol

    return sig, epsp, soundsp


def shell_update(mat: Material, sig: np.ndarray, deps: np.ndarray, *args: Any, **kwargs: Any) -> Any:
    """LAW37 is not formulated for plane-stress shells."""
    raise NotImplementedError("LAW37 (biphasic fluid/gas) is implemented for 3D solid and SPH elements only.")


def sound_speed(
    mat: Material,
    rho: Optional[Union[float, np.ndarray]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Union[float, np.ndarray]:
    """Compute mixture speed of sound for LAW37.

    Parameters
    ----------
    mat : Material
        LAW37 material instance.
    rho : float or np.ndarray, optional
        Current density.
    extra : dict, optional
        Context containing "uv37" and "rho".

    Returns
    -------
    float or np.ndarray
        Speed of sound.
    """
    p = mat.params if mat.params is not None else {}
    c_l = float(p.get("c_l", 2.2e9))
    rho_l0 = float(p.get("rho_l0", 1000.0))
    p0 = float(p.get("p0", 1.01325e5))
    gamma = float(p.get("gamma", 1.4))
    rho_g0 = float(p.get("rho_g0", 1.2))
    isolver = int(p.get("isolver", 1))
    r1 = c_l / rho_l0 if rho_l0 > 0.0 else 0.0

    if rho is None and extra is not None:
        rho = extra.get("rho")
    if rho is None:
        rho = float(getattr(mat, "rho0", 1000.0) or 1000.0)

    is_scalar = np.isscalar(rho)
    rho_arr = np.atleast_1d(np.asarray(rho, dtype=float))

    if extra is not None and "uv37" in extra and extra["uv37"] is not None:
        uv37 = extra["uv37"]
        if uv37.shape[0] == rho_arr.shape[0]:
            rho1 = uv37[:, 2]
            rho2 = uv37[:, 1]
            alpha_v1 = uv37[:, 3]
            alpha_v2 = uv37[:, 4]
        else:
            rho1 = np.full_like(rho_arr, rho_l0)
            rho2 = np.full_like(rho_arr, rho_g0)
            alpha_v1 = np.full_like(rho_arr, float(p.get("alpha1", 1.0)))
            alpha_v2 = 1.0 - alpha_v1
    else:
        alpha1 = float(p.get("alpha1", 1.0))
        if alpha1 >= 1.0 - 1e-10:
            rho1 = rho_arr
            rho2 = np.full_like(rho_arr, rho_g0)
            alpha_v1 = np.ones_like(rho_arr)
            alpha_v2 = np.zeros_like(rho_arr)
        elif alpha1 <= 1e-10:
            rho1 = np.full_like(rho_arr, rho_l0)
            rho2 = rho_arr
            alpha_v1 = np.zeros_like(rho_arr)
            alpha_v2 = np.ones_like(rho_arr)
        else:
            rho1 = np.full_like(rho_arr, rho_l0)
            rho2 = np.full_like(rho_arr, rho_g0)
            alpha_v1 = np.clip(alpha1 * rho_arr / rho_l0, 0.0, 1.0)
            alpha_v2 = 1.0 - alpha_v1

    if isolver == 2:
        ssp1 = r1 * rho1
        ssp2 = gamma * p0 * (rho2 / rho_g0) ** gamma
        ssp1 = np.where(ssp1 > 0.0, ssp1, _EM30)
        ssp2 = np.where(ssp2 > 0.0, ssp2, _EM30)
        ssp_tot = alpha_v1 / ssp1 + alpha_v2 / ssp2
        ssp_tot = np.where(ssp_tot > 0.0, ssp_tot, _EM30)
        c = np.sqrt(1.0 / (ssp_tot * np.maximum(_EM30, rho_arr)))
    else:
        c = np.sqrt(np.maximum(0.0, c_l / np.maximum(_EM30, rho1)))

    if is_scalar:
        return float(c[0])
    return c


def consistent_solid_tangent(
    mat: Material,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    dt: Optional[float] = None,
) -> np.ndarray:
    """Consistent algorithmic 6x6 solid tangent for LAW37 compressible viscous fluid.

    Formulation:
        K_t = rho * c^2
        G_t = mu / dt
        K_vol = mu_vol / dt
        D_{ijkl} = (K_t + K_vol) * delta_{ij} * delta_{kl} + 2 * G_t * I_{ijkl}^{dev}

    Parameters
    ----------
    mat : Material
        LAW37 material instance.
    sig : np.ndarray, optional
        Stress tensor (n, 6).
    epsp : np.ndarray, optional
        Plastic strain.
    epsp_incr : np.ndarray, optional
        Plastic strain increment.
    extra : dict, optional
        Extra context with "rho", "uv37", "dt".
    dt : float, optional
        Time step.

    Returns
    -------
    D : np.ndarray
        Algorithmic tangent matrix of shape (n, 6, 6).
    """
    if sig is not None:
        sig = np.asarray(sig, dtype=float)
        if sig.ndim == 1:
            sig = sig.reshape(1, -1)
        nel = sig.shape[0]
    elif extra is not None and "rho" in extra:
        rho_val = np.asarray(extra["rho"])
        nel = rho_val.shape[0] if rho_val.ndim > 0 else 1
    else:
        nel = 1

    if nel == 0:
        return np.zeros((0, 6, 6), dtype=float)

    if dt is None or dt <= 0.0:
        if extra is not None and "dt" in extra:
            dt = float(extra["dt"])
        else:
            dt = 0.0

    p = mat.params if mat.params is not None else {}
    nu_l = float(p.get("nu_l", 0.0))
    nu_vol_l = float(p.get("nu_vol_l", 0.0))
    nu_g = float(p.get("nu_g", 0.0))
    nu_vol_g = float(p.get("nu_vol_g", 0.0))
    rho_l0 = float(p.get("rho_l0", 1000.0))
    rho_g0 = float(p.get("rho_g0", 1.2))

    if extra is not None and "rho" in extra:
        rho_val = extra["rho"]
        if np.isscalar(rho_val):
            rho = np.full(nel, float(rho_val), dtype=float)
        else:
            rho = np.asarray(rho_val, dtype=float)
    else:
        rho = np.full(nel, float(getattr(mat, "rho0", 1000.0) or 1000.0), dtype=float)

    c = sound_speed(mat, rho=rho, extra=extra)
    if np.isscalar(c):
        c = np.full(nel, float(c), dtype=float)
    else:
        c = np.asarray(c, dtype=float)

    if extra is not None and "uv37" in extra and extra["uv37"] is not None and extra["uv37"].shape == (nel, 5):
        uv37 = extra["uv37"]
        b1 = uv37[:, 0]
        b2 = rho - b1
        rho1 = uv37[:, 2]
        rho2 = uv37[:, 1]
    else:
        alpha1 = float(p.get("alpha1", 1.0))
        b1 = alpha1 * rho
        b2 = (1.0 - alpha1) * rho
        rho1 = np.full(nel, rho_l0, dtype=float)
        rho2 = np.full(nel, rho_g0, dtype=float)

    safe_rho = np.maximum(_EM30, rho)
    mu = (b1 * rho1 * nu_l + b2 * rho2 * nu_g) / safe_rho
    mu_vol = (b1 * rho1 * nu_vol_l + b2 * rho2 * nu_vol_g) / safe_rho

    if extra is not None and extra.get("mixture_sound_speed", False):
        r1 = float(p.get("r1", 0.0))
        if r1 <= 0.0 and rho_l0 > 0.0:
            r1 = float(p.get("c_l", 2.2e9)) / rho_l0
        gamma = float(p.get("gamma", 1.4))
        p0 = float(p.get("p0", 1.01325e5))
        ssp1 = np.where(r1 * rho1 > 0.0, r1 * rho1, _EM30)
        ssp2 = np.where(gamma * p0 * (rho2 / rho_g0) ** gamma > 0.0, gamma * p0 * (rho2 / rho_g0) ** gamma, _EM30)
        alpha_v1 = np.clip(b1 / np.maximum(_EM30, rho1), 0.0, 1.0)
        alpha_v2 = 1.0 - alpha_v1
        ssp_tot = np.where(alpha_v1 / ssp1 + alpha_v2 / ssp2 > 0.0, alpha_v1 / ssp1 + alpha_v2 / ssp2, _EM30)
        c = np.sqrt(1.0 / (ssp_tot * safe_rho))

    kt = rho * (c ** 2)
    if dt > _EM20:
        gt = mu / dt
        k_vol = mu_vol / dt
    else:
        gt = np.zeros(nel, dtype=float)
        k_vol = np.zeros(nel, dtype=float)

    d = np.zeros((nel, 6, 6), dtype=float)
    bulk = kt + k_vol

    for i in range(3):
        for j in range(3):
            d[:, i, j] += bulk
            if i == j:
                d[:, i, j] += 2.0 * gt

    for s in (3, 4, 5):
        d[:, s, s] += gt

    return d


def _register() -> None:
    """Register LAW37 / BIPHAS / BIPHASIC in the global material physics registry."""
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    except (ImportError, ValueError):
        try:
            from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
        except ImportError:
            return

    for key in (37, "37", "LAW37", "BIPHAS", "BIPHASIC"):
        MAT_PHYSICS_REGISTRY[key] = build_law37


_register()
