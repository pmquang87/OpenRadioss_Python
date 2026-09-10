"""
LAW22 — Damaged Elasto-Plastic Material (/MAT/LAW22, /MAT/DAMA, /MAT/PLAS_DAMA).

Fortran origin:
    starter/source/materials/mat/mat022/hm_read_mat22.F
    engine/source/materials/mat/mat022/sigeps22c.F
    engine/source/materials/mat/mat022/sigeps22g.F
    engine/source/materials/mat/mat022/m22cplr.F
    engine/source/materials/mat/mat022/m22law.F
    hm_cfg_files/config/CFG/radioss110/MAT/matl22_dama.cfg

Theory
------
Isotropic elasto-plasticity with power-law hardening, strain-rate sensitivity,
and bilinear softening damage degradation.

Yield function:
    sigma_y0 = min(a + b * eps_p^n, sig_max)
    depsl = max(0, eps_p - eps_dam)
    sigma_y = min(sigma_y0, YLDL + HL * depsl)

Damage degradation factor alpha (modulus degradation):
    alpha = min(1.0, sigma_y / (sigma_y + E * depsl))   (shells)
    alpha = min(1.0, sigma_y / (sigma_y + 3*G * depsl)) (solids)
    E_curr = alpha * E
    G_curr = alpha * G

Strain rate effect (Cowper-Symonds / Johnson-Cook logarithmic form):
    rate_fac = 1.0 + c * ln(max(eps_dot, eps_dot_0) / eps_dot_0)
    sigma_y = sigma_y * rate_fac
    (capped at sig_max if ICC == 2)

Failure:
    When eps_p >= eps_max, element/layer fails: off22 = 0.0, stresses zeroed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM15 = 1e-15
_EM20 = 1e-20
_EM30 = 1e-30
_INF = float("inf")

# Plane-stress von Mises metric P (Voigt [xx, yy, xy] with engineering shear):
# q^2 = sig^T P sig = sxx^2 - sxx*syy + syy^2 + 3*sxy^2.
_P_PLANE = np.array([
    [1.0, -0.5, 0.0],
    [-0.5, 1.0, 0.0],
    [0.0, 0.0, 3.0],
], dtype=float)


# ===================================================================
# Physics Constructor
# ===================================================================

def build_law22(rec=None, **kwargs) -> Material:
    """Physics constructor for /MAT/LAW22 (/MAT/DAMA, /MAT/PLAS_DAMA).

    Parses parameters and computes derived elastic, plastic, damage,
    and acoustic constants following hm_read_mat22.F and matl22_dama.cfg.

    Parameters
    ----------
    rec : dict, MatLaw22, GenericMaterialRecord, or Material (optional)
    **kwargs : parameter overrides or direct specifications

    Returns
    -------
    Material
        Instantiated Material object with law=22 and complete params dict.
    """
    params: Dict[str, Any] = {}
    _id = 0
    _title = "LAW22"
    _rho0 = 0.0

    if rec is not None:
        if isinstance(rec, dict):
            params.update(rec.get("params", {}))
            params.update(rec)
            _id = rec.get("id", 0)
            _title = rec.get("title", "LAW22")
            _rho0 = rec.get("rho0", rec.get("density", rec.get("MAT_RHO", 0.0)))
        elif hasattr(rec, "params") and isinstance(rec.params, dict):
            params.update(rec.params)
            _id = getattr(rec, "id", 0)
            _title = getattr(rec, "title", "LAW22")
            _rho0 = getattr(rec, "rho0", 0.0)
        else:
            _id = getattr(rec, "id", 0)
            _title = getattr(rec, "title", "LAW22")
            _rho0 = getattr(rec, "rho0", 0.0)
            for attr in dir(rec):
                if not attr.startswith("_"):
                    val = getattr(rec, attr)
                    if not callable(val):
                        params[attr] = val

    params.update(kwargs)

    def _get(keys: list[str], default: Any = 0.0) -> Any:
        for k in keys:
            if k in params and params[k] is not None:
                val = params[k]
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return val
        return default

    # Card 1: rho0, refer_rho
    rho0 = _get(["rho0", "MAT_RHO", "density", "rho", "Init_dens"], default=_rho0 or 1.0)
    refer_rho = _get(["refer_rho", "Refer_Rho", "rhor", "rho_ref", "RHO_REF", "Ref_dens"], default=rho0)
    if rho0 <= 0.0:
        rho0 = 1.0
    if refer_rho <= 0.0:
        refer_rho = rho0

    # Card 2: E (MAT_E), nu (MAT_NU)
    E = _get(["E", "MAT_E", "young", "e", "YOUNG", "Young"], default=0.0)
    nu = _get(["nu", "MAT_NU", "ANU", "pr", "Poisson"], default=0.0)
    if E <= 0.0:
        raise ValueError(f"/MAT/LAW22/{_id}: Young modulus E must be > 0 (got {E})")
    if nu == 0.5:
        nu = 0.499  # ZEP499 clamp in hm_read_mat22.F line 141
    if not (0.0 <= nu < 0.5):
        raise ValueError(f"/MAT/LAW22/{_id}: Poisson ratio nu={nu:g} outside [0, 0.5)")

    # Card 3: a (MAT_SIGY), b (MAT_BETA), n (MAT_HARD), eps_max (MAT_EPS), sig_max (MAT_SIG)
    a = _get(["a", "MAT_SIGY", "sigy", "A", "CA", "SIG_Y", "Yield_stress"], default=0.0)
    b = _get(["b", "MAT_BETA", "beta", "B", "CB", "Hardening_parameter"], default=0.0)
    n = _get(["n", "MAT_HARD", "hard", "N", "CN", "Hardening_exponent"], default=1.0)
    eps_max = _get(["eps_max", "MAT_EPS", "epsm", "EPMX", "EPS_MAX", "Failure_strain"], default=_INF)
    sig_max = _get(["sig_max", "MAT_SIG", "sigm", "YMAX", "SIGMX", "SIG_MAX", "Maximum_stress"], default=_INF)

    if n == 0.0:
        n = 1.0  # hm_read_mat22.F line 143: IF(CN == ZERO) CN = ONE
    if eps_max == 0.0 or eps_max is None:
        eps_max = _INF
    if sig_max == 0.0 or sig_max is None:
        sig_max = _INF

    # Card 4: c (MAT_SRC), eps_dot_0 (MAT_SRP), ICC (STRFLAG)
    c = _get(["c", "MAT_SRC", "src", "CC", "Strain_rate_coeff"], default=0.0)
    eps_dot_0 = _get(["eps_dot_0", "MAT_SRP", "srp", "eps0", "EPDR", "EPS_DOT_0", "Ref_strain_rate"], default=1.0)
    icc_val = _get(["ICC", "STRFLAG", "icc", "strflag"], default=1)
    try:
        ICC = int(icc_val)
    except (ValueError, TypeError):
        ICC = 1
    if ICC == 0:
        ICC = 1
    if c == 0.0 or eps_dot_0 == 0.0:
        eps_dot_0 = 1.0  # hm_read_mat22.F line 146

    # Card 5: eps_dam (MAT_DAMAGE), E_tan (MAT_ETAN)
    eps_dam = _get(["eps_dam", "MAT_DAMAGE", "damage", "epsl", "EPSL", "EPS_DAM", "Damage_strain"], default=_EM15)
    E_tan = _get(["E_tan", "MAT_ETAN", "etan", "e_t", "EL", "E_TAN", "Softening_slope"], default=0.0)
    if eps_dam <= 0.0:
        eps_dam = _EM15

    # Computed constants (hm_read_mat22.F lines 148-158)
    G = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    E1MN2 = E / (1.0 - nu * nu)
    EN1N2 = nu * E1MN2
    denom_hl = max(_EM20, E - E_tan)
    HL = E * E_tan / denom_hl
    denom_hl_solid = max(_EM20, 3.0 * G + E_tan)
    HL_solid = 3.0 * G * E_tan / denom_hl_solid
    YLDL = a + b * (eps_dam ** n)
    YLDL = min(YLDL, sig_max)

    # Sound speeds
    c_sound = math.sqrt(E / max(rho0, _EM20))
    c_shell = math.sqrt(max(E1MN2, G) / max(rho0, _EM20))
    c_solid = math.sqrt((K + 4.0 / 3.0 * G) / max(rho0, _EM20))

    mat_params = {
        "E": E,
        "nu": nu,
        "G": G,
        "K": K,
        "E1MN2": E1MN2,
        "EN1N2": EN1N2,
        "a": a,
        "b": b,
        "n": n,
        "eps_max": eps_max,
        "eps_p_max": eps_max,
        "sig_max": sig_max,
        "c": c,
        "eps_dot_0": eps_dot_0,
        "ICC": ICC,
        "eps_dam": eps_dam,
        "E_tan": E_tan,
        "HL": HL,
        "HL_shell": HL,
        "HL_solid": HL_solid,
        "YLDL": YLDL,
        "rho0": rho0,
        "refer_rho": refer_rho,
        "c_sound": c_sound,
        "SDSP": c_sound,
        "c_shell": c_shell,
        "c_solid": c_solid,
        # Standard CFG and legacy aliases
        "MAT_E": E,
        "MAT_NU": nu,
        "MAT_SIGY": a,
        "MAT_BETA": b,
        "MAT_HARD": n,
        "MAT_EPS": eps_max,
        "MAT_SIG": sig_max,
        "MAT_SRC": c,
        "MAT_SRP": eps_dot_0,
        "STRFLAG": ICC,
        "MAT_DAMAGE": eps_dam,
        "MAT_ETAN": E_tan,
        "A": a,
        "B": b,
        "N": n,
    }

    return Material(
        id=_id,
        law=22,
        rho0=rho0,
        title=_title,
        law_name="LAW22",
        params=mat_params,
    )


# ===================================================================
# State Management Shapes
# ===================================================================

def extra_shapes(mat: Optional[Material] = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Per-element persistent state shapes for LAW22.

    Parameters
    ----------
    mat : Material, optional
    nip : int, optional
        Number of integration points / layers for shells. If None, solid shapes.

    Returns
    -------
    dict
        'epsp22': accumulated equivalent plastic strain
        'alpe22': damage degradation factor alpha
        'off22': active element flag (1.0 active, 0.0 failed)
    """
    if nip is not None and nip > 0:
        return {
            "epsp22": (nip,),
            "alpe22": (nip,),
            "off22": (nip,),
        }
    return {
        "epsp22": (),
        "alpe22": (),
        "off22": (),
    }


# ===================================================================
# Sound Speed
# ===================================================================

def sound_speed(mat: Material, rho: Optional[Union[float, np.ndarray]] = None,
                extra: Any = None) -> Union[float, np.ndarray]:
    """Exact bulk/solid sound speed calculation: c_solid = sqrt((K + 4/3*G) / rho)."""
    rho0 = mat.rho0 if rho is None else rho
    K = mat.params.get("K", 0.0)
    G = mat.params.get("G", 0.0)
    return np.sqrt((K + 4.0 / 3.0 * G) / np.maximum(rho0, _EM20))


def sound_speed_solid(mat: Material, rho: Optional[Union[float, np.ndarray]] = None,
                      extra: Any = None) -> Union[float, np.ndarray]:
    """Exact 3D solid sound speed: c_solid = sqrt((K + 4/3*G) / rho)."""
    return sound_speed(mat, rho=rho, extra=extra)


def sound_speed_shell(mat: Material, rho: Optional[Union[float, np.ndarray]] = None,
                      extra: Any = None, mode: str = "plane_stress") -> Union[float, np.ndarray]:
    """Exact 2D plane-stress shell sound speed.

    Parameters
    ----------
    mat : Material
        LAW22 material definition.
    rho : float or ndarray, optional
        Density override.
    extra : dict, optional
    mode : {'plane_stress', 'young'}
        'plane_stress' (default): c = sqrt(max(E1MN2, G) / rho).
        'young': c = sqrt(E / rho) matching SDSP from hm_read_mat22.F line 154.
    """
    rho0 = mat.rho0 if rho is None else rho
    if mode == "young":
        E = mat.params.get("E", 0.0)
        return np.sqrt(E / np.maximum(rho0, _EM20))
    E1MN2 = mat.params.get("E1MN2", 0.0)
    G = mat.params.get("G", 0.0)
    return np.sqrt(np.maximum(E1MN2, G) / np.maximum(rho0, _EM20))


# ===================================================================
# Shell Stress Update (Plane Stress, m22cplr.F)
# ===================================================================

def shell_update(mat: Material, sig: np.ndarray, deps: np.ndarray,
                 epsp: Optional[np.ndarray] = None, dt: float = 0.0,
                 extra: Optional[dict] = None,
                 ipla: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Vectorized plane-stress shell update for LAW22.

    Ports ``engine/source/materials/mat/mat022/m22cplr.F``.

    Parameters
    ----------
    mat : Material
        LAW22 material definition.
    sig : np.ndarray
        In-plane stress array (n, 3) = [s11, s22, s12] (or (n, >=5) with transverse shear).
    deps : np.ndarray
        In-plane strain increment array (n, 3) = [de11, de22, dgamma12].
    epsp : np.ndarray, optional
        Accumulated equivalent plastic strain (n,). Updated in place if provided.
    dt : float
        Time step increment for strain rate calculation.
    extra : dict, optional
        Dictionary holding persistent state variables:
        - 'epsp22': (n,) plastic strain
        - 'alpe22': (n,) modulus degradation factor alpha
        - 'off22': (n,) active status flag (1.0 active, 0.0 failed)
        - 'ezz22': (n,) through-thickness plastic/elastic strain
    ipla : int, optional
        0: radial projection (m22cplr.F lines 129-142)
        1: plane-stress iterative return mapping (m22cplr.F lines 144-223)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        (sig, epsp, c_shell)
    """
    n = sig.shape[0]
    p = mat.params
    c_shell = float(p.get("c_shell", 0.0))

    if n == 0:
        ep_ret = np.empty(0, dtype=float) if epsp is None else epsp
        return sig, ep_ret, c_shell

    E = float(p["E"])
    nu = float(p["nu"])
    a = float(p["a"])
    b = float(p["b"])
    n_exp = float(p["n"])
    sig_max = float(p["sig_max"])
    eps_dam = float(p["eps_dam"])
    YLDL = float(p["YLDL"])
    HL = float(p.get("HL_shell", p.get("HL", 0.0)))
    c_rate = float(p["c"])
    eps_dot_0 = float(p["eps_dot_0"])
    ICC = int(p["ICC"])
    eps_max = float(p["eps_max"])

    if ipla is None:
        if extra is not None and "ipla" in extra:
            ipla = int(extra["ipla"])
        elif "IPLA" in p:
            ipla = int(p["IPLA"])
        else:
            ipla = 0

    # Retrieve or initialize state arrays
    if epsp is not None:
        epseq = np.array(epsp, dtype=float, copy=True)
    elif extra is not None and "epsp22" in extra:
        epseq = np.array(extra["epsp22"], dtype=float, copy=True)
    else:
        epseq = np.zeros(n, dtype=float)

    if extra is not None and "off22" in extra:
        off = np.array(extra["off22"], dtype=float, copy=True)
    elif extra is not None and "off" in extra:
        off = np.array(extra["off"], dtype=float, copy=True)
    else:
        off = np.ones(n, dtype=float)

    # Active mask: not failed and not previously broken
    active = (off > 0.0) & (epseq < eps_max)
    off[~active] = 0.0

    # 1. Yield calculation with power-law hardening and bilinear softening (m22cplr.F lines 87-95)
    epseq_pos = np.maximum(epseq, 0.0)
    yld = a + b * (epseq_pos ** n_exp)
    yld = np.minimum(yld, sig_max)
    depsl = np.maximum(0.0, epseq - eps_dam)
    yld = np.minimum(yld, YLDL + HL * depsl)
    yld = np.maximum(yld, _EM30)

    # Damage degradation factor alpha
    alpe = np.minimum(1.0, yld / (yld + E * depsl))
    alpe = np.maximum(_EM30, alpe)

    # Degraded elastic moduli
    E_curr = alpe * E
    G_curr = 0.5 * E_curr / (1.0 + nu)
    a1 = E_curr / (1.0 - nu * nu)
    a2 = nu * a1

    # 2. Elastic predictor (m22cplr.F lines 101-105)
    de11 = deps[:, 0]
    de22 = deps[:, 1]
    dgamma12 = deps[:, 2]

    s11_trial = sig[:, 0] + a1 * de11 + a2 * de22
    s22_trial = sig[:, 1] + a2 * de11 + a1 * de22
    s12_trial = sig[:, 2] + G_curr * dgamma12

    # 3. von Mises equivalent stress (m22cplr.F lines 111-116)
    svm = np.sqrt(s11_trial ** 2 + s22_trial ** 2 - s11_trial * s22_trial + 3.0 * s12_trial ** 2)

    # 4. Strain rate scaling (m22cplr.F lines 120-125)
    if dt > 0.0 and c_rate > 0.0:
        epsp_rate = np.maximum.reduce([
            np.abs(de11),
            np.abs(de22),
            0.5 * np.abs(dgamma12)
        ]) / max(dt, _EM20)
        epsp_rate = np.maximum(epsp_rate, eps_dot_0)
        yld = yld * (1.0 + c_rate * np.log(epsp_rate / eps_dot_0))
        if ICC == 2:
            yld = np.minimum(yld, sig_max)

    # 5. Plastically admissible stresses
    if ipla == 1:
        # Plane-stress iterative return mapping (m22cplr.F lines 144-223)
        s1 = s11_trial + s22_trial
        s2 = s11_trial - s22_trial
        s3 = s12_trial
        a_vm = 0.25 * (s1 ** 2)
        b_vm = 0.75 * (s2 ** 2) + 3.0 * (s3 ** 2)
        svm_plane = np.sqrt(a_vm + b_vm)

        s11_new = s11_trial.copy()
        s22_new = s22_trial.copy()
        s12_new = s12_trial.copy()
        dpla = np.zeros(n, dtype=float)

        plastic = (svm_plane > yld) & active
        if np.any(plastic):
            idx = np.where(plastic)[0]
            nu1 = 1.0 / (1.0 - nu)
            nu2 = 1.0 / (1.0 + nu)
            ep_idx = epseq[idx]
            small = 1e-7
            if n_exp == 1.0:
                h_term = np.full(len(idx), b, dtype=float)
            else:
                h_term = np.where(
                    ep_idx + small > 0.0,
                    b * n_exp * np.exp((n_exp - 1.0) * np.log(np.maximum(ep_idx + small, _EM20))),
                    0.0
                )
            h_term[yld[idx] >= sig_max] = 0.0
            dpla_j = (svm_plane[idx] - yld[idx]) / np.maximum(3.0 * G_curr[idx] + h_term, 1e-10)

            for _ in range(3):
                dpla_i = dpla_j.copy()
                pla_i = ep_idx + dpla_i
                yld_i = np.minimum(sig_max, a + b * (np.maximum(pla_i, 0.0) ** n_exp))
                yld_i = np.maximum(yld_i, _EM30)
                dr = 0.5 * E_curr[idx] * dpla_i / yld_i
                p_fac = 1.0 / (1.0 + dr * nu1)
                q_fac = 1.0 / (1.0 + 3.0 * dr * nu2)
                p2 = p_fac * p_fac
                q2 = q_fac * q_fac
                f = a_vm[idx] * p2 + b_vm[idx] * q2 - yld_i * yld_i
                df = -(a_vm[idx] * nu1 * p2 * p_fac + 3.0 * b_vm[idx] * nu2 * q2 * q_fac) * (E_curr[idx] - 2.0 * dr * h_term) / yld_i - 2.0 * h_term * yld_i
                corr = np.where(np.abs(df) > _EM30, f / df, 0.0)
                dpla_j = np.where(dpla_i > 0.0, np.maximum(0.0, dpla_i - corr), 0.0)

            dpla[idx] = dpla_i
            epseq[idx] += dpla_i

            s1_new = (s11_trial[idx] + s22_trial[idx]) * p_fac
            s2_new = (s11_trial[idx] - s22_trial[idx]) * q_fac
            s11_new[idx] = 0.5 * (s1_new + s2_new)
            s22_new[idx] = 0.5 * (s1_new - s2_new)
            s12_new[idx] = s12_trial[idx] * q_fac
    else:
        # IPLA=0 radial projection (m22cplr.F lines 129-142)
        plastic = (svm > yld) & active
        dk = np.ones(n, dtype=float)
        dk[plastic] = yld[plastic] / np.maximum(svm[plastic], _EM30)

        s11_new = s11_trial * dk
        s22_new = s22_trial * dk
        s12_new = s12_trial * dk

        dpla = np.zeros(n, dtype=float)
        dpla[plastic] = (svm[plastic] - yld[plastic]) / np.maximum(E_curr[plastic], 1e-10)
        epseq += dpla

    # 6. Element failure: epseq >= eps_max or already inactive (m22cplr.F / sigeps22c.F)
    failed = (epseq >= eps_max) | (~active)
    off[failed] = 0.0
    epseq[failed] = np.maximum(epseq[failed], eps_max)
    s11_new[failed] = 0.0
    s22_new[failed] = 0.0
    s12_new[failed] = 0.0

    # Update stress array
    sig[:, 0] = s11_new
    sig[:, 1] = s22_new
    sig[:, 2] = s12_new

    # Transverse shear stresses if present in (n, >=5)
    if sig.shape[1] >= 5 and deps.shape[1] >= 5:
        gs = G_curr
        sig[:, 3] = (sig[:, 3] + gs * deps[:, 3]) * off
        sig[:, 4] = (sig[:, 4] + gs * deps[:, 4]) * off
        r_trans = np.maximum(0.0, 1.0 - E_curr * dpla / np.maximum(yld, _EM30))
        sig[plastic, 3] *= r_trans[plastic]
        sig[plastic, 4] *= r_trans[plastic]
        sig[failed, 3:] = 0.0

    # Update state in extra dict and epsp argument
    if extra is not None:
        if "epsp22" in extra and isinstance(extra["epsp22"], np.ndarray):
            try:
                extra["epsp22"][:] = epseq
            except Exception:
                extra["epsp22"] = epseq
        else:
            extra["epsp22"] = epseq

        if "alpe22" in extra and isinstance(extra["alpe22"], np.ndarray):
            try:
                extra["alpe22"][:] = alpe
            except Exception:
                extra["alpe22"] = alpe
        else:
            extra["alpe22"] = alpe

        if "off22" in extra and isinstance(extra["off22"], np.ndarray):
            try:
                extra["off22"][:] = off
            except Exception:
                extra["off22"] = off
        else:
            extra["off22"] = off

        if "off" in extra and isinstance(extra["off"], np.ndarray):
            try:
                extra["off"][:] = off
            except Exception:
                extra["off"] = off

        if "layfail" in extra and isinstance(extra["layfail"], np.ndarray):
            extra["layfail"][failed] = 0.0

        if "dpla" in extra and isinstance(extra["dpla"], np.ndarray):
            try:
                extra["dpla"][:] = dpla
            except Exception:
                extra["dpla"] = dpla
        else:
            extra["dpla"] = dpla

        # Through-thickness plastic strain (m22cplr.F line 139 / line 222)
        s1_mean = 0.5 * (s11_new + s22_new)
        ezz = dpla * s1_mean / np.maximum(yld, _EM30)
        if "ezz22" in extra and isinstance(extra["ezz22"], np.ndarray):
            try:
                extra["ezz22"][:] = ezz
            except Exception:
                extra["ezz22"] = ezz
        else:
            extra["ezz22"] = ezz

    if epsp is not None:
        epsp[:] = epseq

    return sig, epseq, c_shell


# ===================================================================
# Solid Stress Update (3D Solid, m22law.F)
# ===================================================================

def solid_update(mat: Material, sig: np.ndarray, deps: np.ndarray,
                 epsp: Optional[np.ndarray] = None, dt: float = 0.0,
                 extra: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Vectorized 3D solid stress update for LAW22.

    Ports ``engine/source/materials/mat/mat022/m22law.F``.

    Parameters
    ----------
    mat : Material
        LAW22 material definition.
    sig : np.ndarray
        Jaumann-rotated 3D stress array (n, 6) = [xx, yy, zz, xy, yz, zx].
    deps : np.ndarray
        Strain increment array (n, 6) with engineering shear.
    epsp : np.ndarray, optional
        Accumulated equivalent plastic strain (n,).
    dt : float
        Time step increment.
    extra : dict, optional
        Dictionary holding persistent state variables:
        - 'epsp22': (n,)
        - 'alpe22': (n,)
        - 'off22': (n,)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        (sig, epsp, c_solid)
    """
    n = sig.shape[0]
    p = mat.params
    c_solid = float(p.get("c_solid", 0.0))

    if n == 0:
        ep_ret = np.empty(0, dtype=float) if epsp is None else epsp
        return sig, ep_ret, c_solid

    E = float(p["E"])
    nu = float(p["nu"])
    G = float(p["G"])
    K = float(p["K"])
    a = float(p["a"])
    b = float(p["b"])
    n_exp = float(p["n"])
    sig_max = float(p["sig_max"])
    eps_dam = float(p["eps_dam"])
    E_tan = float(p["E_tan"])
    YLDL = float(p["YLDL"])
    c_rate = float(p["c"])
    eps_dot_0 = float(p["eps_dot_0"])
    ICC = int(p["ICC"])
    eps_max = float(p["eps_max"])
    rho0 = float(p["rho0"])

    # Retrieve or initialize state arrays
    if epsp is not None:
        epseq = np.array(epsp, dtype=float, copy=True)
    elif extra is not None and "epsp22" in extra:
        epseq = np.array(extra["epsp22"], dtype=float, copy=True)
    else:
        epseq = np.zeros(n, dtype=float)

    if extra is not None and "off22" in extra:
        off = np.array(extra["off22"], dtype=float, copy=True)
    elif extra is not None and "off" in extra:
        off = np.array(extra["off"], dtype=float, copy=True)
    else:
        off = np.ones(n, dtype=float)

    # Active mask: not failed and not previously broken
    active = (off > 0.0) & (epseq < eps_max)
    off[~active] = 0.0

    # 1. Yield and hardening slope (m22law.F lines 200-221)
    epseq_pos = np.maximum(epseq, 0.0)
    if n_exp == 1.0:
        ak = a + b * epseq_pos
        qh = np.full(n, b, dtype=float)
    else:
        ak = a + b * (epseq_pos ** n_exp)
        qh = np.where(epseq_pos > 0.0, b * n_exp * (np.maximum(epseq_pos, _EM20) ** (n_exp - 1.0)), 0.0)

    ak = np.minimum(ak, sig_max)
    qh = np.where(ak >= sig_max, 0.0, qh)

    # 2. Softening after eps_dam (m22law.F lines 240-247)
    depsl = np.maximum(0.0, epseq - eps_dam)
    denom_hl = max(_EM20, 3.0 * G + E_tan)
    HL = float(p.get("HL_solid", 3.0 * G * E_tan / denom_hl))
    ak = np.minimum(ak, YLDL + HL * depsl)
    ak = np.maximum(ak, 0.0)
    qh = np.where(epseq > eps_dam, HL, qh)

    # 3. Damage degradation factor alpha (m22law.F line 245)
    alpe = np.minimum(1.0, ak / np.maximum(ak + 3.0 * G * depsl, _EM15))
    alpe = np.maximum(_EM30, alpe)

    # 4. Strain rate scaling (m22law.F lines 194-198)
    if dt > 0.0 and c_rate > 0.0:
        epd = off * np.maximum.reduce([
            np.abs(deps[:, 0]),
            np.abs(deps[:, 1]),
            np.abs(deps[:, 2]),
            0.5 * np.abs(deps[:, 3]),
            0.5 * np.abs(deps[:, 4]),
            0.5 * np.abs(deps[:, 5]),
        ]) / max(dt, _EM20)
        epsp_rate = np.maximum(epd, eps_dot_0)
        ce = 1.0 + c_rate * np.log(epsp_rate / eps_dot_0)
        ak = ak * ce
        # In Fortran m22law.F:220, IF(EPXE(I)>EPSL(I)) QH(I)=QL(I) occurs after CE scaling
        qh = np.where(epseq > eps_dam, HL, qh * ce)
        if ICC == 2:
            ak = np.minimum(ak, sig_max)

    G_curr = alpe * G

    # 5. Deviatoric predictor and pressure (m22law.F lines 250-267)
    p_old = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
    tr3 = (deps[:, 0] + deps[:, 1] + deps[:, 2]) / 3.0

    s = np.empty_like(sig)
    s[:, 0] = sig[:, 0] - p_old + 2.0 * G_curr * (deps[:, 0] - tr3)
    s[:, 1] = sig[:, 1] - p_old + 2.0 * G_curr * (deps[:, 1] - tr3)
    s[:, 2] = sig[:, 2] - p_old + 2.0 * G_curr * (deps[:, 2] - tr3)
    s[:, 3] = sig[:, 3] + G_curr * deps[:, 3]
    s[:, 4] = sig[:, 4] + G_curr * deps[:, 4]
    s[:, 5] = sig[:, 5] + G_curr * deps[:, 5]

    j2 = 0.5 * (s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2) + s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2
    aj2 = np.sqrt(3.0 * j2)

    # 6. Radial return to J2 yield surface (m22law.F lines 350-359)
    scale = np.minimum(1.0, ak / np.maximum(aj2, _EM15))
    plastic = (aj2 > ak) & active
    s *= scale[:, None]

    dpla = np.zeros(n, dtype=float)
    dpla[plastic] = ((1.0 - scale[plastic]) * aj2[plastic]
                     / np.maximum(3.0 * G_curr[plastic] + qh[plastic], 1e-10))
    epseq += dpla

    # 7. Pressure update (PDAM = 0.0 for LAW22 in OpenRadioss engine)
    if extra is not None and "rho" in extra:
        p_new = -K * (extra["rho"] / rho0 - 1.0)
    else:
        p_new = p_old + K * 3.0 * tr3

    sig[:, 0] = (s[:, 0] + p_new) * off
    sig[:, 1] = (s[:, 1] + p_new) * off
    sig[:, 2] = (s[:, 2] + p_new) * off
    sig[:, 3] = s[:, 3] * off
    sig[:, 4] = s[:, 4] * off
    sig[:, 5] = s[:, 5] * off

    # 8. Deletion / Failure: epseq >= eps_max
    failed = (epseq >= eps_max) | (~active)
    off[failed] = 0.0
    epseq[failed] = np.maximum(epseq[failed], eps_max)
    sig[failed, :] = 0.0

    # 9. Update state in extra and epsp
    if extra is not None:
        if "epsp22" in extra and isinstance(extra["epsp22"], np.ndarray):
            try:
                extra["epsp22"][:] = epseq
            except Exception:
                extra["epsp22"] = epseq
        else:
            extra["epsp22"] = epseq

        if "alpe22" in extra and isinstance(extra["alpe22"], np.ndarray):
            try:
                extra["alpe22"][:] = alpe
            except Exception:
                extra["alpe22"] = alpe
        else:
            extra["alpe22"] = alpe

        if "off22" in extra and isinstance(extra["off22"], np.ndarray):
            try:
                extra["off22"][:] = off
            except Exception:
                extra["off22"] = off
        else:
            extra["off22"] = off

        if "off" in extra and isinstance(extra["off"], np.ndarray):
            try:
                extra["off"][:] = off
            except Exception:
                extra["off"] = off

        if "dpla" in extra and isinstance(extra["dpla"], np.ndarray):
            try:
                extra["dpla"][:] = dpla
            except Exception:
                extra["dpla"] = dpla
        else:
            extra["dpla"] = dpla

    if epsp is not None:
        epsp[:] = epseq

    return sig, epseq, c_solid


# ===================================================================
# Consistent Algorithmic Tangents for Implicit Solvers
# ===================================================================

def shell_membrane_tangent(mat: Material) -> np.ndarray:
    """(3, 3) elastic plane-stress membrane tangent."""
    p = mat.params
    a1 = float(p.get("E1MN2", p["E"] / (1.0 - p["nu"] ** 2)))
    a2 = float(p.get("EN1N2", p["nu"] * a1))
    G = float(p["G"])
    return np.array([
        [a1, a2, 0.0],
        [a2, a1, 0.0],
        [0.0, 0.0, G],
    ], dtype=float)


def solid_tangent(mat: Material) -> np.ndarray:
    """(6, 6) elastic solid tangent."""
    G = float(mat.params["G"])
    K = float(mat.params["K"])
    ee = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
    I_dev = np.diag([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.5, 0.5, 0.5])
    I_dev[0, 1] = I_dev[0, 2] = I_dev[1, 0] = I_dev[1, 2] = I_dev[2, 0] = I_dev[2, 1] = -1.0 / 3.0
    return K * np.outer(ee, ee) + 2.0 * G * I_dev


def consistent_shell_tangent(mat: Material, sig: np.ndarray,
                             epsp: Optional[Union[float, np.ndarray]] = None,
                             dt: Any = 0.0,
                             extra: Optional[dict] = None,
                             epsp_incr: Optional[Union[float, np.ndarray]] = None,
                             symmetric: bool = False,
                             deps: Optional[np.ndarray] = None,
                             **kwargs) -> np.ndarray:
    """Consistent (n, 3, 3) algorithmic plane-stress tangent tensor.

    Governs Newton's quadratic convergence for the plane-stress radial projection
    with modulus degradation alpha.
    """
    if isinstance(dt, np.ndarray):
        epsp_incr = dt
        dt = 0.0

    if not isinstance(sig, np.ndarray):
        sig = np.asarray(sig, dtype=float)

    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig[None, :]

    n = sig.shape[0]
    if n == 0:
        return np.empty((0, 3, 3), dtype=float)

    p = mat.params
    E, nu = float(p["E"]), float(p["nu"])
    G = float(p["G"])
    a1_base = float(p.get("E1MN2", E / (1.0 - nu * nu)))
    a2_base = float(p.get("EN1N2", nu * a1_base))

    alpe = None
    off = None
    if extra is not None:
        alpe = extra.get("alpe22", extra.get("alpe"))
        off = extra.get("off22", extra.get("off"))
        if epsp is None:
            epsp = extra.get("epsp22")

    if epsp is None:
        epsp_arr = np.zeros(n, dtype=float)
    else:
        epsp_arr = np.asarray(epsp, dtype=float).flatten()
        if len(epsp_arr) == 1 and n > 1:
            epsp_arr = np.full(n, epsp_arr[0], dtype=float)

    if off is not None:
        off_arr = np.asarray(off, dtype=float).flatten()
        if len(off_arr) == 1 and n > 1:
            off_arr = np.full(n, off_arr[0], dtype=float)
    else:
        off_arr = np.ones(n, dtype=float)

    # Resolve alpe (modulus degradation)
    if alpe is not None:
        alpe_arr = np.asarray(alpe, dtype=float).flatten()
        if len(alpe_arr) == 1 and n > 1:
            alpe_arr = np.full(n, alpe_arr[0], dtype=float)
    else:
        # Fallback: compute alpe from epsp if past damage threshold
        eps_dam = float(p.get("eps_dam", _EM15))
        epseq_pos = np.maximum(epsp_arr, 0.0)
        depsl = np.maximum(0.0, epseq_pos - eps_dam)
        if np.any(depsl > 0.0):
            a = float(p["a"])
            b = float(p["b"])
            n_exp = float(p["n"])
            sig_max = float(p["sig_max"])
            YLDL = float(p["YLDL"])
            HL = float(p["HL"])
            yld = a + b * (epseq_pos ** n_exp)
            yld = np.minimum(yld, sig_max)
            yld = np.minimum(yld, YLDL + HL * depsl)
            yld = np.maximum(yld, _EM30)
            alpe_arr = np.minimum(1.0, yld / (yld + E * depsl))
            alpe_arr = np.maximum(_EM30, alpe_arr)
        else:
            alpe_arr = np.ones(n, dtype=float)

    D = np.zeros((n, 3, 3), dtype=float)
    for i in range(n):
        ai = alpe_arr[i]
        D[i, 0, 0] = ai * a1_base
        D[i, 1, 1] = ai * a1_base
        D[i, 0, 1] = ai * a2_base
        D[i, 1, 0] = ai * a2_base
        D[i, 2, 2] = ai * G

    eps_max = float(p.get("eps_max", _INF))
    dead = (off_arr <= 0.0) | (epsp_arr >= eps_max)
    D[dead] = 0.0

    if epsp_incr is None:
        if extra is not None and "dpla" in extra:
            epsp_incr = extra["dpla"]

    if epsp_incr is None and deps is not None:
        deps_arr = np.asarray(deps, dtype=float)
        if deps_arr.ndim == 1:
            deps_arr = deps_arr[None, :]
        epsp_incr = np.zeros(n, dtype=float)
        for i in range(n):
            if not dead[i]:
                Ce_i = D[i]
                s_tr = sig[i, :3] + Ce_i @ deps_arr[i, :3]
                svm_tr = math.sqrt(max(0.0, float(s_tr @ _P_PLANE @ s_tr)))
                a = float(p["a"])
                b = float(p["b"])
                n_exp = float(p["n"])
                sig_max = float(p["sig_max"])
                YLDL = float(p["YLDL"])
                HL = float(p["HL"])
                eps_dam = float(p.get("eps_dam", _EM15))
                ep0 = epsp_arr[i]
                yld = min(sig_max, a + b * (ep0 ** n_exp))
                depsl = max(0.0, ep0 - eps_dam)
                yld = max(_EM30, min(yld, YLDL + HL * depsl))
                if svm_tr > yld:
                    E_curr = alpe_arr[i] * E
                    epsp_incr[i] = (svm_tr - yld) / max(E_curr, 1e-10)

    if epsp_incr is None:
        if symmetric:
            D = 0.5 * (D + np.swapaxes(D, -1, -2))
        return D[0] if is_1d else D

    epsp_incr = np.asarray(epsp_incr, dtype=float).flatten()
    if len(epsp_incr) == 1 and n > 1:
        epsp_incr = np.full(n, epsp_incr[0], dtype=float)

    plastic = (epsp_incr > 0.0) & (~dead)
    if not np.any(plastic):
        if symmetric:
            D = 0.5 * (D + np.swapaxes(D, -1, -2))
        return D[0] if is_1d else D

    idx = np.where(plastic)[0]
    hardening = kwargs.get("hardening", extra.get("hardening", None) if extra is not None else None)
    b = float(p["b"])
    n_exp = float(p["n"])
    sig_max = float(p["sig_max"])
    eps_dam = float(p.get("eps_dam", _EM15))
    HL = float(p["HL"])

    for ii in idx:
        dl = float(epsp_incr[ii])
        s_c = sig[ii, :3]
        sy = math.sqrt(max(0.0, float(s_c @ _P_PLANE @ s_c)))
        sy = max(sy, _EM30)

        E_curr = alpe_arr[ii] * E
        q_tr = sy + E_curr * dl
        s_scale = sy / max(q_tr, _EM30)
        s_tr = s_c / max(s_scale, _EM30)

        Ce_i = D[ii].copy()
        v = Ce_i @ _P_PLANE @ s_tr
        rank1 = np.outer(s_tr, v)

        if hardening is True:
            ep_start = max(0.0, float(epsp_arr[ii]) - dl)
            if ep_start > eps_dam:
                h_slope = HL
            elif ep_start > 0.0:
                h_slope = b * n_exp * (max(ep_start, _EM20) ** (n_exp - 1.0))
            else:
                h_slope = b if n_exp == 1.0 else 0.0
            if sy >= sig_max:
                h_slope = 0.0
            h_eff = max(h_slope, 0.0)
            h_fac = h_eff / max(E_curr + h_eff, _EM30)
        else:
            h_fac = 0.0

        gamma = (h_fac - s_scale) / max(q_tr * q_tr, _EM30)
        D[ii] = s_scale * Ce_i + gamma * rank1

    if symmetric:
        D = 0.5 * (D + np.swapaxes(D, -1, -2))

    return D[0] if is_1d else D


def consistent_solid_tangent(mat: Material, sig: np.ndarray,
                             epsp: Optional[Union[float, np.ndarray]] = None,
                             dt: Any = 0.0,
                             extra: Optional[dict] = None,
                             epsp_incr: Optional[Union[float, np.ndarray]] = None,
                             symmetric: bool = False,
                             deps: Optional[np.ndarray] = None,
                             **kwargs) -> np.ndarray:
    """Consistent (n, 6, 6) algorithmic 3D solid tangent tensor."""
    if isinstance(dt, np.ndarray):
        epsp_incr = dt
        dt = 0.0

    if not isinstance(sig, np.ndarray):
        sig = np.asarray(sig, dtype=float)

    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig[None, :]

    n = sig.shape[0]
    if n == 0:
        return np.empty((0, 6, 6), dtype=float)

    p = mat.params
    G = float(p["G"])
    K = float(p["K"])

    alpe = None
    off = None
    if extra is not None:
        alpe = extra.get("alpe22", extra.get("alpe"))
        off = extra.get("off22", extra.get("off"))
        if epsp is None:
            epsp = extra.get("epsp22")

    if epsp is None:
        epsp_arr = np.zeros(n, dtype=float)
    else:
        epsp_arr = np.asarray(epsp, dtype=float).flatten()
        if len(epsp_arr) == 1 and n > 1:
            epsp_arr = np.full(n, epsp_arr[0], dtype=float)

    if off is not None:
        off_arr = np.asarray(off, dtype=float).flatten()
        if len(off_arr) == 1 and n > 1:
            off_arr = np.full(n, off_arr[0], dtype=float)
    else:
        off_arr = np.ones(n, dtype=float)

    # Resolve alpe (modulus degradation)
    if alpe is not None:
        alpe_arr = np.asarray(alpe, dtype=float).flatten()
        if len(alpe_arr) == 1 and n > 1:
            alpe_arr = np.full(n, alpe_arr[0], dtype=float)
    else:
        # Fallback: compute alpe from epsp if past damage threshold
        eps_dam = float(p.get("eps_dam", _EM15))
        epseq_pos = np.maximum(epsp_arr, 0.0)
        depsl = np.maximum(0.0, epseq_pos - eps_dam)
        if np.any(depsl > 0.0):
            a = float(p["a"])
            b = float(p["b"])
            n_exp = float(p["n"])
            sig_max = float(p["sig_max"])
            E_tan = float(p["E_tan"])
            YLDL = float(p["YLDL"])
            denom_hl = 3.0 * G + E_tan
            HL = 3.0 * G * E_tan / denom_hl if abs(denom_hl) > _EM20 else 0.0
            ak = a + b * (epseq_pos ** n_exp)
            ak = np.minimum(ak, sig_max)
            ak = np.minimum(ak, YLDL + HL * depsl)
            ak = np.maximum(ak, 0.0)
            alpe_arr = np.minimum(1.0, ak / np.maximum(ak + 3.0 * G * depsl, _EM15))
            alpe_arr = np.maximum(_EM30, alpe_arr)
        else:
            alpe_arr = np.ones(n, dtype=float)

    ee = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
    KeeT = K * np.outer(ee, ee)

    I_dev = np.diag([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.5, 0.5, 0.5])
    I_dev[0, 1] = I_dev[0, 2] = I_dev[1, 0] = I_dev[1, 2] = I_dev[2, 0] = I_dev[2, 1] = -1.0 / 3.0

    D = np.zeros((n, 6, 6), dtype=float)
    for i in range(n):
        Gi = G * alpe_arr[i]
        D[i] = KeeT + 2.0 * Gi * I_dev

    eps_max = float(p.get("eps_max", _INF))
    dead = (off_arr <= 0.0) | (epsp_arr >= eps_max)
    D[dead] = 0.0

    if epsp_incr is None:
        if extra is not None and "dpla" in extra:
            epsp_incr = extra["dpla"]

    if epsp_incr is None and deps is not None:
        deps_arr = np.asarray(deps, dtype=float)
        if deps_arr.ndim == 1:
            deps_arr = deps_arr[None, :]
        epsp_incr = np.zeros(n, dtype=float)
        for i in range(n):
            if not dead[i]:
                Gi = G * alpe_arr[i]
                p_old = (sig[i, 0] + sig[i, 1] + sig[i, 2]) / 3.0
                tr3 = (deps_arr[i, 0] + deps_arr[i, 1] + deps_arr[i, 2]) / 3.0
                s_tr = np.zeros(6)
                s_tr[0] = sig[i, 0] - p_old + 2.0 * Gi * (deps_arr[i, 0] - tr3)
                s_tr[1] = sig[i, 1] - p_old + 2.0 * Gi * (deps_arr[i, 1] - tr3)
                s_tr[2] = sig[i, 2] - p_old + 2.0 * Gi * (deps_arr[i, 2] - tr3)
                s_tr[3] = sig[i, 3] + Gi * deps_arr[i, 3]
                s_tr[4] = sig[i, 4] + Gi * deps_arr[i, 4]
                s_tr[5] = sig[i, 5] + Gi * deps_arr[i, 5]
                j2 = 0.5 * (s_tr[0]**2 + s_tr[1]**2 + s_tr[2]**2) + s_tr[3]**2 + s_tr[4]**2 + s_tr[5]**2
                aj2 = math.sqrt(3.0 * j2)

                a = float(p["a"])
                b = float(p["b"])
                n_exp = float(p["n"])
                sig_max = float(p["sig_max"])
                E_tan = float(p["E_tan"])
                YLDL = float(p["YLDL"])
                eps_dam = float(p.get("eps_dam", _EM15))
                ep0 = epsp_arr[i]
                if n_exp == 1.0:
                    ak = a + b * ep0
                    qh = b
                else:
                    ak = a + b * (ep0 ** n_exp)
                    qh = b * n_exp * (max(ep0, _EM20) ** (n_exp - 1.0)) if ep0 > 0.0 else 0.0
                ak = min(ak, sig_max)
                depsl = max(0.0, ep0 - eps_dam)
                denom_hl = 3.0 * G + E_tan
                HL = 3.0 * G * E_tan / denom_hl if abs(denom_hl) > _EM20 else 0.0
                ak = min(ak, YLDL + HL * depsl)
                ak = max(ak, 0.0)
                if ep0 > eps_dam:
                    qh = HL
                if aj2 > ak:
                    epsp_incr[i] = (aj2 - ak) / max(3.0 * Gi + qh, 1e-10)

    if epsp_incr is None:
        if symmetric:
            D = 0.5 * (D + np.swapaxes(D, -1, -2))
        return D[0] if is_1d else D

    epsp_incr = np.asarray(epsp_incr, dtype=float).flatten()
    if len(epsp_incr) == 1 and n > 1:
        epsp_incr = np.full(n, epsp_incr[0], dtype=float)

    plastic = (epsp_incr > 0.0) & (~dead)
    if not np.any(plastic):
        if symmetric:
            D = 0.5 * (D + np.swapaxes(D, -1, -2))
        return D[0] if is_1d else D

    idx = np.where(plastic)[0]
    hardening = kwargs.get("hardening", extra.get("hardening", None) if extra is not None else None)
    eps_dam = float(p.get("eps_dam", _EM15))
    b = float(p["b"])
    n_exp = float(p["n"])
    sig_max = float(p["sig_max"])
    E_tan = float(p["E_tan"])

    for ii in idx:
        dep = float(epsp_incr[ii])
        Gi = G * alpe_arr[ii]

        s_vec = sig[ii, :6].copy()
        pm = (s_vec[0] + s_vec[1] + s_vec[2]) / 3.0
        s_vec[0] -= pm
        s_vec[1] -= pm
        s_vec[2] -= pm

        snorm = math.sqrt(s_vec[0] ** 2 + s_vec[1] ** 2 + s_vec[2] ** 2
                          + 2.0 * (s_vec[3] ** 2 + s_vec[4] ** 2 + s_vec[5] ** 2))
        snorm = max(snorm, _EM30)
        Nv = s_vec / snorm
        q = math.sqrt(1.5) * snorm

        ep_start = max(0.0, float(epsp_arr[ii]) - dep)
        if ep_start > eps_dam:
            denom_hl = 3.0 * G + E_tan
            qh = 3.0 * G * E_tan / denom_hl if abs(denom_hl) > _EM20 else 0.0
        else:
            if n_exp == 1.0:
                qh = b
            elif ep_start > 0.0:
                qh = b * n_exp * (max(ep_start, _EM20) ** (n_exp - 1.0))
            else:
                qh = 0.0
        if q >= sig_max:
            qh = 0.0

        q_tr = q + (3.0 * Gi + qh) * dep
        scale = q / max(q_tr, _EM30)

        if hardening is True:
            h_eff = max(qh, 0.0)
            h_fac = h_eff / max(3.0 * Gi + h_eff, _EM30)
        else:
            h_fac = 0.0

        NN = np.outer(Nv, Nv)
        C_dev = scale * 2.0 * Gi * I_dev + 2.0 * Gi * (h_fac - scale) * NN
        D[ii] = KeeT + C_dev

    if symmetric:
        D = 0.5 * (D + np.swapaxes(D, -1, -2))

    return D[0] if is_1d else D


# ===================================================================
# Registration
# ===================================================================

def _register() -> None:
    """Register LAW22 physics constructor with MAT_PHYSICS_REGISTRY."""
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for key in ("22", "LAW22", "DAMA", "PLAS_DAMA", "TSAI_WU", "MAT_DAMA", "MAT_TSAIWU"):
            MAT_PHYSICS_REGISTRY[key] = build_law22
    except ImportError:
        pass


_register()
