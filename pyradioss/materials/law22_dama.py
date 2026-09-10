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
    YLDL = a + b * (eps_dam ** n)
    YLDL = min(YLDL, sig_max)

    # Sound speeds
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
        "sig_max": sig_max,
        "c": c,
        "eps_dot_0": eps_dot_0,
        "ICC": ICC,
        "eps_dam": eps_dam,
        "E_tan": E_tan,
        "HL": HL,
        "YLDL": YLDL,
        "rho0": rho0,
        "refer_rho": refer_rho,
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
                      extra: Any = None) -> Union[float, np.ndarray]:
    """Exact 2D plane-stress shell sound speed: c_shell = sqrt(max(E1MN2, G) / rho)."""
    rho0 = mat.rho0 if rho is None else rho
    E1MN2 = mat.params.get("E1MN2", 0.0)
    G = mat.params.get("G", 0.0)
    return np.sqrt(np.maximum(E1MN2, G) / np.maximum(rho0, _EM20))


# ===================================================================
# Shell Stress Update (Plane Stress, m22cplr.F)
# ===================================================================

def shell_update(mat: Material, sig: np.ndarray, deps: np.ndarray,
                 epsp: Optional[np.ndarray] = None, dt: float = 0.0,
                 extra: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray, float]:
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
    HL = float(p["HL"])
    c_rate = float(p["c"])
    eps_dot_0 = float(p["eps_dot_0"])
    ICC = int(p["ICC"])
    eps_max = float(p["eps_max"])

    # Retrieve or initialize state arrays
    if epsp is not None:
        epseq = np.array(epsp, dtype=float, copy=True)
    elif extra is not None and "epsp22" in extra:
        epseq = np.array(extra["epsp22"], dtype=float, copy=True)
    else:
        epseq = np.zeros(n, dtype=float)

    if extra is not None and "off22" in extra:
        off = np.array(extra["off22"], dtype=float, copy=True)
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

    # 5. Plastically admissible stresses (m22cplr.F lines 129-142, IPLA=0 radial projection)
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

        if "dpla" in extra and isinstance(extra["dpla"], np.ndarray):
            try:
                extra["dpla"][:] = dpla
            except Exception:
                extra["dpla"] = dpla
        else:
            extra["dpla"] = dpla

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
    denom_hl = 3.0 * G + E_tan
    HL = 3.0 * G * E_tan / denom_hl if abs(denom_hl) > _EM20 else 0.0
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
        qh = qh * ce
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
                             epsp: Optional[np.ndarray] = None,
                             dt: Any = 0.0,
                             extra: Optional[dict] = None,
                             epsp_incr: Optional[np.ndarray] = None,
                             **kwargs) -> np.ndarray:
    """Consistent (n, 3, 3) algorithmic plane-stress tangent tensor.

    Governs Newton's quadratic convergence for the plane-stress radial projection
    with modulus degradation alpha.
    """
    if isinstance(dt, np.ndarray):
        epsp_incr = dt
        dt = 0.0

    n = sig.shape[0]
    if n == 0:
        return np.empty((0, 3, 3), dtype=float)

    p = mat.params
    E, nu = float(p["E"]), float(p["nu"])
    G = float(p["G"])
    a1 = float(p.get("E1MN2", E / (1.0 - nu * nu)))
    a2 = float(p.get("EN1N2", nu * a1))
    Ce = np.array([
        [a1, a2, 0.0],
        [a2, a1, 0.0],
        [0.0, 0.0, G],
    ], dtype=float)

    D = np.broadcast_to(Ce, (n, 3, 3)).copy()

    alpe = None
    off = None
    if extra is not None:
        alpe = extra.get("alpe22")
        off = extra.get("off22")
        if epsp is None:
            epsp = extra.get("epsp22")

    if alpe is not None:
        alpe_arr = np.asarray(alpe, dtype=float)
        if alpe_arr.ndim > 0 and len(alpe_arr) == n:
            D *= alpe_arr[:, None, None]

    if epsp is None:
        epsp = np.zeros(n, dtype=float)
    else:
        epsp = np.asarray(epsp, dtype=float)

    if off is not None:
        off_arr = np.asarray(off, dtype=float)
        if off_arr.ndim > 0 and len(off_arr) == n:
            dead = off_arr <= 0.0
            D[dead] = 0.0

    if epsp_incr is None:
        if extra is not None and "dpla" in extra:
            epsp_incr = extra["dpla"]
        else:
            return D

    epsp_incr = np.asarray(epsp_incr, dtype=float)
    plastic = (epsp_incr > 0.0)
    if off is not None:
        plastic = plastic & (np.asarray(off) > 0.0)

    if not np.any(plastic):
        return D

    idx = np.where(plastic)[0]
    dl = epsp_incr[idx]
    s_c = sig[idx, :3]

    sy = np.sqrt(np.maximum(
        np.einsum("mi,ij,mj->m", s_c, _P_PLANE, s_c), 0.0))
    sy = np.maximum(sy, _EM30)

    alpe_idx = alpe[idx] if (alpe is not None and np.ndim(alpe) > 0) else 1.0
    G_idx = G * alpe_idx

    q_tr = sy + 3.0 * G_idx * dl
    s_scale = sy / q_tr
    sig_tr = s_c / s_scale[:, None]

    ep_idx = epsp[idx]
    eps_dam = float(p["eps_dam"])
    b = float(p["b"])
    n_exp = float(p["n"])
    sig_max = float(p["sig_max"])
    HL = float(p["HL"])

    H = np.where(
        ep_idx > eps_dam,
        HL,
        np.where(
            ep_idx > 0.0,
            b * n_exp * (np.maximum(ep_idx, _EM20) ** (n_exp - 1.0)),
            b if n_exp == 1.0 else 0.0
        )
    )
    H = np.where(sy >= sig_max, 0.0, H)
    Hbar = np.maximum(H, 0.0)

    gamma = (Hbar / (3.0 * G_idx + Hbar) - s_scale) / (q_tr ** 2)

    for k, ii in enumerate(idx):
        Ce_i = D[ii]
        CP = Ce_i @ _P_PLANE
        v = CP @ sig_tr[k]
        rank1 = np.outer(sig_tr[k], v)
        D[ii] = s_scale[k] * Ce_i + gamma[k] * rank1

    return D


def consistent_solid_tangent(mat: Material, sig: np.ndarray,
                             epsp: Optional[np.ndarray] = None,
                             dt: Any = 0.0,
                             extra: Optional[dict] = None,
                             epsp_incr: Optional[np.ndarray] = None,
                             **kwargs) -> np.ndarray:
    """Consistent (n, 6, 6) algorithmic 3D solid tangent tensor."""
    if isinstance(dt, np.ndarray):
        epsp_incr = dt
        dt = 0.0

    n = sig.shape[0]
    if n == 0:
        return np.empty((0, 6, 6), dtype=float)

    p = mat.params
    G = float(p["G"])
    K = float(p["K"])

    alpe = None
    off = None
    if extra is not None:
        alpe = extra.get("alpe22")
        off = extra.get("off22")
        if epsp is None:
            epsp = extra.get("epsp22")

    if epsp is None:
        epsp = np.zeros(n, dtype=float)
    else:
        epsp = np.asarray(epsp, dtype=float)

    ee = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
    KeeT = K * np.outer(ee, ee)

    I_dev = np.diag([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.5, 0.5, 0.5])
    I_dev[0, 1] = I_dev[0, 2] = I_dev[1, 0] = I_dev[1, 2] = I_dev[2, 0] = I_dev[2, 1] = -1.0 / 3.0

    D = np.zeros((n, 6, 6), dtype=float)
    alpe_arr = np.asarray(alpe, dtype=float) if alpe is not None else np.ones(n, dtype=float)
    if alpe_arr.ndim == 0:
        alpe_arr = np.full(n, float(alpe_arr), dtype=float)

    for i in range(n):
        Gi = G * alpe_arr[i]
        D[i] = KeeT + 2.0 * Gi * I_dev

    if off is not None:
        off_arr = np.asarray(off, dtype=float)
        if off_arr.ndim > 0 and len(off_arr) == n:
            dead = off_arr <= 0.0
            D[dead] = 0.0

    if epsp_incr is None:
        if extra is not None and "dpla" in extra:
            epsp_incr = extra["dpla"]
        else:
            return D

    epsp_incr = np.asarray(epsp_incr, dtype=float)
    plastic = (epsp_incr > 0.0)
    if off is not None:
        plastic = plastic & (np.asarray(off) > 0.0)

    if not np.any(plastic):
        return D

    idx = np.where(plastic)[0]
    s = sig[idx].copy()
    pm = (s[:, 0] + s[:, 1] + s[:, 2]) / 3.0
    s[:, 0] -= pm
    s[:, 1] -= pm
    s[:, 2] -= pm

    snorm = np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2
                    + 2.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
    snorm = np.maximum(snorm, _EM30)
    Nv = s / snorm[:, None]
    q = np.sqrt(1.5) * snorm
    dep = epsp_incr[idx]

    eps_dam = float(p["eps_dam"])
    b = float(p["b"])
    n_exp = float(p["n"])
    sig_max = float(p["sig_max"])
    HL = float(p["HL"])
    ep_idx = epsp[idx]

    H = np.where(
        ep_idx > eps_dam,
        HL,
        np.where(
            ep_idx > 0.0,
            b * n_exp * (np.maximum(ep_idx, _EM20) ** (n_exp - 1.0)),
            b if n_exp == 1.0 else 0.0
        )
    )
    H = np.where(q >= sig_max, 0.0, H)
    Hbar = np.maximum(H, 0.0)

    for k, ii in enumerate(idx):
        Gi = G * alpe_arr[ii]
        q_tr = q[k] + 3.0 * Gi * dep[k]
        a_coef = 3.0 * Gi * dep[k] / q_tr
        b_coef = 6.0 * Gi * Gi * (dep[k] / q_tr - 1.0 / (3.0 * Gi + Hbar[k]))
        C_minus_vol = 2.0 * Gi * I_dev
        NN = np.outer(Nv[k], Nv[k])
        D[ii] = D[ii] - a_coef * C_minus_vol + b_coef * NN

    return D


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
