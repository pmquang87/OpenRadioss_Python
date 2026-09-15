"""
LAW48 — Zhao strain-rate hardening plasticity (/MAT/LAW48, /MAT/ZHAO, /MAT/PLAS_ZHAO).

Fortran origin:
- ``engine/source/materials/mat/mat048/sigeps48.F`` (solids)
- ``engine/source/materials/mat/mat048/sigeps48c.F`` (shells)
- ``starter/source/materials/mat/mat048/hm_read_mat48.F`` (starter keyword reader)

Theory
------
The Zhao constitutive model accounts for coupled strain hardening and strain-rate
sensitivity, together with a high-rate power law term and tensile failure damage.

Yield stress formulation:
    sigma_y = P_A + P_B + P_C

1. Work hardening:
    P_A = c_a * sigma_y0 + c_b * eps_p^(c_n)
    where eps_p is equivalent plastic strain (capped at eps_p_max).

2. Coupled strain rate sensitivity:
    P_B = (c_c - c_d * eps_p^(c_m)) * ln(eps_dot / eps_dot_0)   for eps_dot > eps_dot_0 (else 0)

3. High-rate power law:
    P_C = c_e * eps_dot^(c_k)                                    for eps_dot > 0 (else 0)

4. Maximum yield stress saturation:
    YY = P_A + P_B + P_C
    YLD = min(S_max + P_C, YY)

5. Plastic hardening slope:
    H = P_DA + P_DB   (set to 0 if capped at S_max + P_C)
    P_DA = c_b * c_n * eps_p^(c_n - 1)  (or upstream c_b * c_n * eps_p^(1 - c_n) if c_n < 1; E below yield)
    P_DB = c_d * c_m * eps_p^(c_m - 1) * ln(eps_dot / eps_dot_0) (if eps_dot > eps_dot_0, else 0)

6. Maximum principal strain tensile damage:
    FAIL = max(0, min(1, (eps_r2 - eps_t) / (eps_r2 - eps_r1)))
    with eps_t the maximum principal total strain (solved via the 3D deviatoric cubic
    in solids or 2D closed-form in shells).
    Both yield stress and hardening are scaled by FAIL:
    YLD = FAIL * YLD,  H = FAIL * H.

7. Mixed isotropic / kinematic hardening:
    F_isokin in [0, 1] (0 = pure isotropic, 1 = pure kinematic).
    Y_LO = c_a * sigma_y0 + P_C
    YLD = (1 - F_isokin) * YLD + F_isokin * (FAIL * Y_LO)
    H_iso = (1 - F_isokin) * H
    H_kin = F_isokin * H

8. Element deletion:
    When eps_p > eps_p_max or eps_t >= eps_r2, sigma = 0, off = 0.0.

9. Sound speed:
    Solid: c = sqrt((K + 4G/3) / rho0) = sqrt(E*(1-nu) / ((1+nu)*(1-2nu)*rho0))
    Shell: c = sqrt(E / ((1 - nu^2) * rho0))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from ..model.entities import Material

_EM20 = 1e-20
_INF = 1e30


@dataclass
class Law48Params:
    """Strongly-typed parameters for /MAT/LAW48 (Zhao)."""

    rho0: float = 1.0
    rhor: float = 1.0
    E: float = 2.1e5
    nu: float = 0.3
    ca: float = 1.0
    sigy0: float = 200.0
    cb: float = 0.0
    cn: float = 1.0001
    fisokin: float = 0.0
    sig_max: float = _INF
    cc: float = 0.0
    cd: float = 0.0
    cm: float = 1.0001
    ce: float = 0.0
    ck: float = 1.0
    eps0: float = 1.0
    fcut: float = _INF
    eps_max: float = _INF
    eps_t1: float = _INF
    eps_t2: float = 2.0 * _INF

    @property
    def G(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def K(self) -> float:
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))

    @property
    def A11(self) -> float:
        return self.E / (1.0 - self.nu * self.nu)

    @property
    def A21(self) -> float:
        return self.nu * self.A11


def _get_params(mat: Any) -> Law48Params:
    """Extract Law48Params from a Material, Law48Params, dict, or object."""
    if isinstance(mat, Law48Params):
        return mat

    p: dict[str, Any] = {}
    rho0 = 1.0
    if isinstance(mat, Material):
        p = mat.params
        rho0 = float(getattr(mat, "rho0", 1.0) or 1.0)
    elif isinstance(mat, dict):
        p = mat
        rho0 = float(p.get("rho0", p.get("rho", 1.0)) or 1.0)
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        p = mat.params
        rho0 = float(getattr(mat, "rho0", p.get("rho0", p.get("rho", 1.0))) or 1.0)
    elif hasattr(mat, "__dict__"):
        p = mat.__dict__
        rho0 = float(getattr(mat, "rho0", p.get("rho0", p.get("rho", 1.0))) or 1.0)

    e = float(p.get("MAT_E", p.get("E", p.get("e", 2.1e5))) or 2.1e5)
    nu = float(p.get("MAT_NU", p.get("NU", p.get("nu", 0.3))) if p.get("MAT_NU") is not None or p.get("NU") is not None or p.get("nu") is not None else 0.3)
    rhor = float(p.get("Refer_Rho", p.get("refer_rho", p.get("rhor", rho0))) or rho0)

    # Yield stress & ca multiplier
    ca = float(p.get("ca", p.get("c_a", 1.0)) or 1.0)
    if "sigma_y0" in p and p["sigma_y0"] is not None:
        sigy0 = float(p["sigma_y0"])
    elif "sigy0" in p and p["sigy0"] is not None:
        sigy0 = float(p["sigy0"])
    elif "MAT_SIGY" in p and p["MAT_SIGY"] is not None:
        sigy0 = float(p["MAT_SIGY"])
    elif "sigy" in p and p["sigy"] is not None:
        sigy0 = float(p["sigy"])
    elif "a" in p and p["a"] is not None:
        sigy0 = float(p["a"])
    elif "sig_y" in p and p["sig_y"] is not None:
        sigy0 = float(p["sig_y"])
    else:
        sigy0 = 0.0

    cb = float(p.get("MAT_B", p.get("cb", p.get("c_b", p.get("b", p.get("B", 0.0))))) or 0.0)
    cn = float(p.get("MAT_N", p.get("cn", p.get("c_n", p.get("n", p.get("N", 1.0001))))) or 0.0)
    if cn == 0.0 or cn == 1.0:
        cn = 1.0001

    fisokin = float(p.get("MAT_HARD", p.get("fisokin", p.get("chard", p.get("hard", 0.0)))) or 0.0)
    fisokin = min(max(fisokin, 0.0), 1.0)

    sig_max = float(p.get("MAT_SIG", p.get("sig_max", p.get("sigm", p.get("smax", 0.0)))) or 0.0)
    if sig_max <= 0.0:
        sig_max = _INF

    cc = float(p.get("MAT_C", p.get("cc", p.get("c_c", p.get("c", p.get("C", 0.0))))) or 0.0)
    cd = float(p.get("MAT_D", p.get("cd", p.get("c_d", p.get("d", p.get("D", 0.0))))) or 0.0)
    cm = float(p.get("MAT_M", p.get("cm", p.get("c_m", p.get("m", p.get("M", 1.0001))))) or 0.0)
    if cm == 0.0 or cm == 1.0:
        cm = 1.0001

    ce = float(p.get("MAT_E1", p.get("ce", p.get("c_e", p.get("e1", p.get("E1", 0.0))))) or 0.0)
    ck = float(p.get("MAT_K", p.get("ck", p.get("c_k", p.get("k", p.get("K", 1.0))))) or 0.0)
    if ck == 0.0:
        ck = 1.0

    eps0 = float(p.get("MAT_E0", p.get("eps0", p.get("eps_rate_0", p.get("eps_dot_0", p.get("e0", 0.0))))) or 0.0)
    if eps0 <= 0.0:
        eps0 = 1.0

    fcut = float(p.get("SCALE", p.get("fcut", p.get("Fcut", 0.0))) or 0.0)
    if fcut <= 0.0:
        fcut = _INF

    eps_max = float(p.get("MAT_EPS", p.get("eps_max", p.get("epsm", p.get("eps_p_max", 0.0)))) or 0.0)
    if eps_max <= 0.0:
        eps_max = _INF

    eps_t1 = float(p.get("MAT_ETA1", p.get("eps_t1", p.get("epsr1", p.get("eta1", 0.0)))) or 0.0)
    if eps_t1 <= 0.0:
        eps_t1 = _INF

    eps_t2 = float(p.get("MAT_ETA2", p.get("eps_t2", p.get("epsr2", p.get("eta2", 0.0)))) or 0.0)
    if eps_t2 <= 0.0:
        eps_t2 = 2.0 * _INF

    return Law48Params(
        rho0=rho0,
        rhor=rhor,
        E=e,
        nu=nu,
        ca=ca,
        sigy0=sigy0,
        cb=cb,
        cn=cn,
        fisokin=fisokin,
        sig_max=sig_max,
        cc=cc,
        cd=cd,
        cm=cm,
        ce=ce,
        ck=ck,
        eps0=eps0,
        fcut=fcut,
        eps_max=eps_max,
        eps_t1=eps_t1,
        eps_t2=eps_t2,
    )


def build_law48(rec: Any) -> Material:
    """Physics constructor for the cfg-parsed /MAT/LAW48 record (hm_read_mat48.F)."""
    p = rec.params if hasattr(rec, "params") else rec
    params_obj = _get_params(p)
    rho = float(getattr(rec, "density", getattr(rec, "rho", params_obj.rho0)) or params_obj.rho0)
    title = str(getattr(rec, "title", "MAT_LAW48"))
    rec_id = int(getattr(rec, "id", 0) or 0)

    params_dict = {
        "E": params_obj.E,
        "nu": params_obj.nu,
        "ca": params_obj.ca,
        "sigy0": params_obj.sigy0,
        "cb": params_obj.cb,
        "cn": params_obj.cn,
        "fisokin": params_obj.fisokin,
        "sig_max": params_obj.sig_max,
        "cc": params_obj.cc,
        "cd": params_obj.cd,
        "cm": params_obj.cm,
        "ce": params_obj.ce,
        "ck": params_obj.ck,
        "eps0": params_obj.eps0,
        "fcut": params_obj.fcut,
        "eps_max": params_obj.eps_max,
        "eps_t1": params_obj.eps_t1,
        "eps_t2": params_obj.eps_t2,
        # Standard radioss aliases
        "a": params_obj.ca * params_obj.sigy0,
        "b": params_obj.cb,
        "n": params_obj.cn,
        "c": params_obj.cc,
        "d": params_obj.cd,
        "m": params_obj.cm,
        "e1": params_obj.ce,
        "k": params_obj.ck,
        "chard": params_obj.fisokin,
        "eps_rate_0": params_obj.eps0,
        "eps_p_max": params_obj.eps_max,
    }

    return Material(id=rec_id, law=48, rho0=rho, title=title, params=params_dict)


# ----------------------------------------------------------------------------
# Yield stress and hardening evaluation
# ----------------------------------------------------------------------------

def eval_yield_and_hardening(
    params: Law48Params,
    epsp: np.ndarray | float,
    eps_dot: np.ndarray | float,
    fail: np.ndarray | float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Zhao yield stress and hardening slopes.

    Returns
    -------
    yld : current yield stress with FAIL and kinematic weighting applied
    H : total hardening slope
    H_iso : isotropic hardening slope (1 - F_isokin) * H
    H_kin : kinematic hardening slope F_isokin * H
    """
    pla = np.asarray(epsp, dtype=float)
    edot = np.asarray(eps_dot, dtype=float)
    fl = np.asarray(fail, dtype=float)

    # Work hardening P_A
    eff_pla = np.clip(pla, 0.0, params.eps_max)
    safe_pla = np.maximum(eff_pla, _EM20)
    pa = params.ca * params.sigy0 + params.cb * (safe_pla ** params.cn)
    pa = np.where(pla <= 0.0, params.ca * params.sigy0, pa)

    # Coupled strain rate sensitivity P_B
    safe_edot = np.maximum(edot, _EM20)
    pb_active = edot > params.eps0
    log_ratio = np.log(np.maximum(safe_edot / params.eps0, 1.0))
    pb = (params.cc - params.cd * (safe_pla ** params.cm)) * log_ratio
    pb = np.where(pla <= 0.0, params.cc * log_ratio, pb)
    pb = np.where(pb_active, pb, 0.0)

    # High-rate power law P_C
    pc_active = edot > 0.0
    pc = np.where(pc_active, params.ce * (safe_edot ** params.ck), 0.0)

    # Total yield stress & cap
    yy = pa + pb + pc
    smax_pc = params.sig_max + pc
    yld_raw = np.minimum(smax_pc, yy)

    # Plastic hardening slope H = P_DA + P_DB
    # P_DA (derivative of work hardening curve; E when pla <= 0 per sigeps48.F:285 / sigeps48c.F:279)
    if params.cn >= 1.0:
        pda = params.cb * params.cn * (safe_pla ** (params.cn - 1.0))
    else:
        pda = params.cb * params.cn * (safe_pla ** (1.0 - params.cn))
    pda = np.where(pla <= 0.0, params.E, pda)

    # P_DB (upstream sigeps48.F:287-293)
    if params.cm >= 1.0:
        pdb = params.cd * params.cm * (safe_pla ** (params.cm - 1.0)) * log_ratio
    else:
        pdb = params.cd * params.cm * (safe_pla ** (1.0 - params.cm)) * log_ratio
    pdb = np.where((pla > 0.0) & pb_active, pdb, 0.0)

    h_raw = pda + pdb
    # Zero hardening when capped
    capped = yld_raw < yy
    h_raw = np.where(capped, 0.0, h_raw)

    # Tension failure damage scaling
    yld_fail = fl * yld_raw
    h_fail = fl * h_raw

    # Kinematic hardening yield scaling (sigeps48c.F:307-315)
    ylo = params.ca * params.sigy0 + pc
    if params.fisokin > 0.0:
        yld = (1.0 - params.fisokin) * yld_fail + params.fisokin * (fl * ylo)
    else:
        yld = yld_fail

    h_iso = (1.0 - params.fisokin) * h_fail
    h_kin = params.fisokin * h_fail

    # Zero yield if past eps_max
    broken = pla > params.eps_max
    yld = np.where(broken, 0.0, yld)
    h_fail = np.where(broken, 0.0, h_fail)
    h_iso = np.where(broken, 0.0, h_iso)
    h_kin = np.where(broken, 0.0, h_kin)

    return yld, h_fail, h_iso, h_kin


# ----------------------------------------------------------------------------
# Maximum principal strain solvers & Tensile failure factor
# ----------------------------------------------------------------------------

def _principal_strain_3d(eps: np.ndarray) -> np.ndarray | float:
    """Maximum principal total strain from 3D strain tensor (sigeps48.F:201-245).

    Uses the exact 4-iteration Newton solver on the deviatoric cubic equation,
    including the convergence check ABS(Y) > 1e-8.
    """
    eps_arr = np.asarray(eps, dtype=float)
    single = eps_arr.ndim == 1
    if single:
        eps_arr = eps_arr.reshape(1, 6)

    dav = (eps_arr[:, 0] + eps_arr[:, 1] + eps_arr[:, 2]) / 3.0
    e1 = eps_arr[:, 0] - dav
    e2 = eps_arr[:, 1] - dav
    e3 = eps_arr[:, 2] - dav
    e4 = 0.5 * eps_arr[:, 3]
    e5 = 0.5 * eps_arr[:, 4]
    e6 = 0.5 * eps_arr[:, 5]

    e42 = e4 * e4
    e52 = e5 * e5
    e62 = e6 * e6

    c = - e1 * e1 - e2 * e2 - e3 * e3 - e42 - e52 - e62
    d = - e1 * e2 * e3 + e1 * e52 + e2 * e62 + e3 * e42 - 2.0 * e4 * e5 * e6
    cd = c / 3.0
    epst = np.sqrt(np.maximum(-cd, 0.0))
    epst2 = epst * epst
    y = (epst2 + c) * epst + d

    active = np.abs(y) > 1e-8
    x = np.where(active, 1.75 * epst, epst)
    for _ in range(4):
        x2 = x * x
        y_val = (x2 + c) * x + d
        yp = 3.0 * x2 + c
        step = np.where(active & (yp != 0.0), y_val / np.where(yp == 0.0, 1.0, yp), 0.0)
        x = x - step

    res = np.where(active, x + dav, epst)
    return res[0] if single else res


def _principal_strain_2d(eps: np.ndarray) -> np.ndarray | float:
    """Maximum in-plane principal total strain (sigeps48c.F:245-247)."""
    eps_arr = np.asarray(eps, dtype=float)
    single = eps_arr.ndim == 1
    if single:
        eps_arr = eps_arr.reshape(1, -1)
    exx = eps_arr[:, 0]
    eyy = eps_arr[:, 1]
    exy = eps_arr[:, 2]
    diff = exx - eyy
    res = 0.5 * (exx + eyy + np.sqrt(diff * diff + exy * exy))
    return res[0] if single else res


def tensile_failure_factor(params: Law48Params, epst: np.ndarray | float) -> np.ndarray:
    """FAIL = max(0, min(1, (eps_r2 - eps_t) / (eps_r2 - eps_r1)))."""
    t = np.asarray(epst, dtype=float)
    if params.eps_t1 >= _INF or params.eps_t2 <= params.eps_t1:
        return np.ones_like(t)
    denom = max(params.eps_t2 - params.eps_t1, _EM20)
    return np.clip((params.eps_t2 - t) / denom, 0.0, 1.0)


# ----------------------------------------------------------------------------
# Sound speeds
# ----------------------------------------------------------------------------

def sound_speed_solid_law48(mat: Any, rho0: float | np.ndarray | None = None, **kwargs: Any) -> float | np.ndarray:
    """Longitudinal sound speed for solid elements:
    c = sqrt((K + 4G/3) / rho0) = sqrt(E*(1-nu) / ((1+nu)*(1-2nu)*rho0)).
    """
    p = _get_params(mat)
    rho = rho0 if rho0 is not None else p.rho0
    if isinstance(rho, np.ndarray):
        r = np.where(rho > 0.0, rho, 1.0)
    else:
        r = rho if (rho is not None and rho > 0.0) else 1.0

    c_sq = (p.K + 4.0 * p.G / 3.0) / r
    return np.sqrt(c_sq) if isinstance(c_sq, np.ndarray) else math.sqrt(c_sq)


def sound_speed_shell_law48(mat: Any, rho0: float | np.ndarray | None = None, **kwargs: Any) -> float | np.ndarray:
    """Sound speed for shell elements:
    c = sqrt(E / ((1 - nu^2) * rho0)).
    """
    p = _get_params(mat)
    rho = rho0 if rho0 is not None else p.rho0
    if isinstance(rho, np.ndarray):
        r = np.where(rho > 0.0, rho, 1.0)
    else:
        r = rho if (rho is not None and rho > 0.0) else 1.0

    c_sq = p.A11 / r
    return np.sqrt(c_sq) if isinstance(c_sq, np.ndarray) else math.sqrt(c_sq)


# ----------------------------------------------------------------------------
# Solid update (sigeps48.F)
# ----------------------------------------------------------------------------

def solid_update_law48(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constitutive cycle for solid elements (/MAT/LAW48).

    Parameters
    ----------
    mat : Material or Law48Params
    sig : (NEL, 6) or (6,) stress tensor [xx, yy, zz, xy, yz, zx]
    deps : (NEL, 6) or (6,) strain increment (engineering shear)
    epsp : (NEL,) or float plastic strain
    dt : time increment
    extra : dict of persistent state views (sigb48, eps48, epsd48, off, rho, etc.)

    Returns
    -------
    sig : updated stress array
    epsp : updated equivalent plastic strain
    c : sound speed array
    """
    # Disambiguate if called as solid_update_law48(sig, deps, epsp, mat, ...)
    if isinstance(mat, np.ndarray) and not isinstance(sig, np.ndarray):
        mat, sig = sig, mat
        if isinstance(deps, (int, float)) and isinstance(epsp, np.ndarray):
            deps, epsp = epsp, deps

    p = _get_params(mat)

    sig_arr = np.asarray(sig, dtype=float)
    deps_arr = np.asarray(deps, dtype=float)
    single = sig_arr.ndim == 1

    if single:
        sig_arr = sig_arr.reshape(1, 6)
        deps_arr = deps_arr.reshape(1, 6)

    nel = sig_arr.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.asarray(epsp, dtype=float).copy()
        if epsp_arr.ndim == 0:
            epsp_arr = np.full(nel, float(epsp_arr))

    if extra is None:
        extra = {}

    G = p.G
    G2 = 2.0 * G
    G3 = 3.0 * G
    bulk = p.K

    # 1. Deviatoric elastic trial predictor
    pm_old = (sig_arr[:, 0] + sig_arr[:, 1] + sig_arr[:, 2]) / 3.0
    dav = (deps_arr[:, 0] + deps_arr[:, 1] + deps_arr[:, 2]) / 3.0
    s_tr = np.empty_like(sig_arr)
    s_tr[:, 0] = sig_arr[:, 0] - pm_old + G2 * (deps_arr[:, 0] - dav)
    s_tr[:, 1] = sig_arr[:, 1] - pm_old + G2 * (deps_arr[:, 1] - dav)
    s_tr[:, 2] = sig_arr[:, 2] - pm_old + G2 * (deps_arr[:, 2] - dav)
    s_tr[:, 3] = sig_arr[:, 3] + G * deps_arr[:, 3]
    s_tr[:, 4] = sig_arr[:, 4] + G * deps_arr[:, 4]
    s_tr[:, 5] = sig_arr[:, 5] + G * deps_arr[:, 5]

    # 2. Kinematic hardening backstress
    sigb_key = "sigb48" if "sigb48" in extra else ("sigb" if "sigb" in extra else None)
    if sigb_key is not None:
        sigb = extra[sigb_key]
        if single and sigb.ndim == 1:
            sigb = sigb.reshape(1, 6)
    elif p.fisokin > 0.0:
        sigb = np.zeros((nel, 6), dtype=float)
        extra["sigb48"] = sigb
    else:
        sigb = None

    if sigb is not None:
        s_rel = s_tr - sigb
    else:
        s_rel = s_tr

    # 3. Equivalent strain rate & filtering (mstrain_rate.F / sigeps48.F)
    if dt > 0.0:
        exx_d = deps_arr[:, 0] - dav
        eyy_d = deps_arr[:, 1] - dav
        ezz_d = deps_arr[:, 2] - dav
        ee = (
            0.5 * (exx_d ** 2 + eyy_d ** 2 + ezz_d ** 2)
            + 0.25 * (deps_arr[:, 3] ** 2 + deps_arr[:, 4] ** 2 + deps_arr[:, 5] ** 2)
        )
        raw_rate = (np.sqrt(3.0 * ee) / 1.5) / dt
    else:
        raw_rate = np.zeros(nel, dtype=float)

    epsd_key = "epsd48" if "epsd48" in extra else ("epsd" if "epsd" in extra else None)
    if p.fcut < _INF and dt > 0.0:
        alpha = min(1.0, 2.0 * math.pi * p.fcut * dt)
        if epsd_key is not None:
            prev_rate = np.asarray(extra[epsd_key], dtype=float)
            filtered_rate = alpha * raw_rate + (1.0 - alpha) * prev_rate
            if isinstance(extra[epsd_key], np.ndarray):
                if extra[epsd_key].ndim > 0:
                    extra[epsd_key][:] = filtered_rate
                else:
                    extra[epsd_key].fill(float(filtered_rate.flat[0]))
            else:
                extra[epsd_key] = float(filtered_rate.flat[0]) if single else filtered_rate
            epsd = filtered_rate
        else:
            extra["epsd48"] = raw_rate.copy()
            epsd = raw_rate.copy()
    else:
        epsd = raw_rate

    # 4. Total strain & maximum principal strain cubic solver (sigeps48.F:201-245)
    eps_key = "eps48" if "eps48" in extra else ("eps" if "eps" in extra else None)
    if eps_key is not None:
        eps_tot = extra[eps_key]
        if eps_tot.ndim == 1:
            if nel == 1:
                eps_tot = eps_tot.reshape(1, 6)
                extra[eps_key] = eps_tot
            else:
                eps_tot = np.broadcast_to(eps_tot, (nel, 6)).copy()
                extra[eps_key] = eps_tot
        eps_tot += deps_arr
    elif p.eps_t1 < _INF:
        eps_tot = deps_arr.copy()
        extra["eps48"] = eps_tot
    else:
        eps_tot = deps_arr

    epst = _principal_strain_3d(eps_tot)
    fail = tensile_failure_factor(p, epst)

    # 5. Yield stress and hardening slope
    yld, _, h_iso, h_kin = eval_yield_and_hardening(p, epsp_arr, epsd, fail)

    # 6. Von Mises stress and radial return (sigeps48.F:308-331)
    vm2 = (
        0.5 * (s_rel[:, 0] ** 2 + s_rel[:, 1] ** 2 + s_rel[:, 2] ** 2)
        + s_rel[:, 3] ** 2
        + s_rel[:, 4] ** 2
        + s_rel[:, 5] ** 2
    )
    vm = np.sqrt(3.0 * vm2)

    r = np.minimum(1.0, yld / np.maximum(vm, _EM20))
    dpla = (1.0 - r) * vm / np.maximum(G3 + h_iso, _EM20)
    plastic = dpla > 0.0

    # Two-step return to match sigeps48.F lines 316-318
    yld_corr = yld + dpla * h_iso
    yld_corr = np.where(epsp_arr + dpla > p.eps_max, 0.0, yld_corr)
    r_corr = np.minimum(1.0, yld_corr / np.maximum(vm, _EM20))

    s_rel_new = s_rel * r_corr[:, None]
    epsp_arr += dpla

    # Update backstress for kinematic hardening
    if sigb is not None:
        diff = s_tr - (s_rel_new + sigb)
        alpha_kin = h_kin / np.maximum(2.0 * G + h_kin, _EM20)
        sigb += np.where(plastic[:, None], alpha_kin[:, None] * diff, 0.0)
        s_new = s_rel_new + sigb
    else:
        s_new = s_rel_new

    # 7. Pressure update: P = K * mu or hypoelastic trace
    if "rho" in extra:
        rho = extra["rho"]
        amu = (rho / p.rho0 - 1.0) if isinstance(rho, np.ndarray) else np.full(nel, rho / p.rho0 - 1.0)
        p_mean = -bulk * amu
    elif "amu" in extra:
        p_mean = -bulk * extra["amu"]
    else:
        p_mean = pm_old + bulk * dav * 3.0

    sig_out = s_new.copy()
    sig_out[:, 0] += p_mean
    sig_out[:, 1] += p_mean
    sig_out[:, 2] += p_mean

    # 8. Element deletion check (eps_p > eps_p_max or eps_t >= eps_r2)
    deleted = (epsp_arr > p.eps_max) | (epst >= p.eps_t2)
    if np.any(deleted):
        sig_out[deleted] = 0.0
        if "off" in extra:
            if isinstance(extra["off"], np.ndarray):
                if extra["off"].ndim > 0:
                    extra["off"][deleted] = 0.0
                else:
                    extra["off"].fill(0.0)
            else:
                extra["off"] = 0.0
        else:
            extra["off"] = np.where(deleted, 0.0, 1.0)
        if "off48" in extra:
            if isinstance(extra["off48"], np.ndarray):
                if extra["off48"].ndim > 0:
                    extra["off48"][deleted] = 0.0
                else:
                    extra["off48"].fill(0.0)
            else:
                extra["off48"] = 0.0

    c = sound_speed_solid_law48(p, rho0=extra.get("rho"))
    if not isinstance(c, np.ndarray):
        c = np.full(nel, c)

    if single:
        return sig_out[0], float(epsp_arr[0]), c[0]
    return sig_out, epsp_arr, c


# ----------------------------------------------------------------------------
# Shell update (sigeps48c.F)
# ----------------------------------------------------------------------------

def plane_stress_return_newton_law48(
    params: Law48Params,
    sig_rel: np.ndarray,
    yld: np.ndarray,
    h_iso: np.ndarray,
    epsp_arr: np.ndarray,
    off: np.ndarray | None = None,
    nmax: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterative plane-stress plasticity return via Newton-Raphson (sigeps48c.F:346-423).

    Solves the nonlinear scalar constraint F(dpla) = 0 for in-plane plane stress
    using up to nmax (default 3) Newton-Raphson iterations.

    Returns
    -------
    sig_rel_new : (NEL, 3) updated plane stress tensor [xx, yy, xy]
    dpla : (NEL,) plastic strain increment
    dezz_pl : (NEL,) plastic through-thickness thinning increment
    """
    s1 = sig_rel[:, 0] + sig_rel[:, 1]
    s2 = sig_rel[:, 0] - sig_rel[:, 1]
    s3 = sig_rel[:, 2]

    aa = 0.25 * s1 * s1
    bb = 0.75 * s2 * s2 + 3.0 * s3 * s3
    svm = np.sqrt(np.maximum(aa + bb, 0.0))

    if off is None:
        off_arr = np.ones_like(svm)
    else:
        off_arr = np.asarray(off, dtype=float)

    plastic = (svm > yld) & (off_arr == 1.0)
    nel = len(svm)

    dpla_j = np.where(plastic, (svm - yld) / np.maximum(3.0 * params.G + h_iso, _EM20), 0.0)
    dpla_i = dpla_j.copy()

    nu11 = 1.0 / (1.0 - params.nu)
    nu21 = 1.0 / (1.0 + params.nu)
    nu31 = (1.0 - 2.0 * params.nu) / (1.0 - params.nu)

    pp = np.ones(nel, dtype=float)
    qq = np.ones(nel, dtype=float)
    dr = np.zeros(nel, dtype=float)

    if np.any(plastic):
        for _ in range(nmax):
            dpla_i = dpla_j.copy()
            yld_i = yld + h_iso * dpla_i
            dr = 0.5 * params.E * dpla_i / np.maximum(yld_i, _EM20)
            pp = 1.0 / (1.0 + dr * nu11)
            qq = 1.0 / (1.0 + 3.0 * dr * nu21)
            p2 = pp * pp
            q2 = qq * qq
            f = aa * p2 + bb * q2 - yld_i * yld_i
            df = -(aa * nu11 * p2 * pp + 3.0 * bb * nu21 * q2 * qq) * (params.E - 2.0 * dr * h_iso) / np.maximum(yld_i, _EM20) - 2.0 * h_iso * yld_i
            safe_df = np.where(df == 0.0, -1.0, df)
            step = np.where(plastic & (dpla_i > 0.0), f / safe_df, 0.0)
            dpla_j = np.where(plastic, np.maximum(0.0, dpla_i - step), 0.0)

    # Plastically admissible stresses (sigeps48c.F:409-417)
    s1_corr = s1 * pp
    s2_corr = s2 * qq

    sig_rel_new = sig_rel.copy()
    sig_rel_new[:, 0] = np.where(plastic, 0.5 * (s1_corr + s2_corr), sig_rel[:, 0])
    sig_rel_new[:, 1] = np.where(plastic, 0.5 * (s1_corr - s2_corr), sig_rel[:, 1])
    sig_rel_new[:, 2] = np.where(plastic, s3 * qq, sig_rel[:, 2])

    dezz_pl = np.where(plastic, -nu31 * dr * s1_corr / params.E, 0.0)
    dpla = np.where(plastic, dpla_i, 0.0)

    return sig_rel_new, dpla, dezz_pl


def shell_update_law48(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict[str, Any] | None = None,
    iflag: int | None = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Constitutive plane-stress cycle for shell elements (/MAT/LAW48).

    Parameters
    ----------
    mat : Material or Law48Params
    sig : (NEL, 3) or (NEL, 5) or (3,) stress tensor [xx, yy, xy]
    deps : (NEL, 3) or (NEL, 5) or (3,) strain increment (engineering shear)
    epsp : (NEL,) or float plastic strain
    dt : time increment
    extra : dict of persistent state views (sigb48, eps48, epsd48, off, thk, etc.)
    iflag : 0 for radial projection (sigeps48c.F:321-340), 1 for Newton-Raphson return (sigeps48c.F:346-423)

    Returns
    -------
    sig : updated stress array
    epsp : updated equivalent plastic strain
    """
    if isinstance(mat, np.ndarray) and not isinstance(sig, np.ndarray):
        mat, sig = sig, mat
        if isinstance(deps, (int, float)) and isinstance(epsp, np.ndarray):
            deps, epsp = epsp, deps

    p = _get_params(mat)

    sig_arr = np.asarray(sig, dtype=float)
    deps_arr = np.asarray(deps, dtype=float)
    single = sig_arr.ndim == 1

    if single:
        sig_arr = sig_arr.reshape(1, -1)
        deps_arr = deps_arr.reshape(1, -1)

    nel = sig_arr.shape[0]
    has_transverse = sig_arr.shape[1] >= 5 and deps_arr.shape[1] >= 5

    if epsp is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.asarray(epsp, dtype=float).copy()
        if epsp_arr.ndim == 0:
            epsp_arr = np.full(nel, float(epsp_arr))

    if extra is None:
        extra = {}

    G = p.G
    G3 = 3.0 * G
    A11 = p.A11
    A21 = p.A21

    # 1. Elastic trial update for in-plane components (sigeps48c.F:221-228)
    sig_tr = sig_arr.copy()
    sig_tr[:, 0] += A11 * deps_arr[:, 0] + A21 * deps_arr[:, 1]
    sig_tr[:, 1] += A21 * deps_arr[:, 0] + A11 * deps_arr[:, 1]
    sig_tr[:, 2] += G * deps_arr[:, 2]

    if has_transverse:
        sig_tr[:, 3] += G * deps_arr[:, 3]
        sig_tr[:, 4] += G * deps_arr[:, 4]

    # 2. Kinematic backstress
    sigb_key = "sigb48" if "sigb48" in extra else ("sigb" if "sigb" in extra else None)
    if sigb_key is not None:
        sigb = extra[sigb_key]
        if single and sigb.ndim == 1:
            sigb = sigb.reshape(1, 3)
    elif p.fisokin > 0.0:
        sigb = np.zeros((nel, 3), dtype=float)
        extra["sigb48"] = sigb
    else:
        sigb = None

    if sigb is not None:
        sig_rel = sig_tr[:, :3] - sigb
    else:
        sig_rel = sig_tr[:, :3]

    # 3. Strain rate & filtering (sigeps48c.F:235-241)
    if dt > 0.0:
        dxx = deps_arr[:, 0] / dt
        dyy = deps_arr[:, 1] / dt
        dxy = deps_arr[:, 2] / dt
        diff = dxx - dyy
        raw_rate = 0.5 * (np.abs(dxx + dyy) + np.sqrt(diff * diff + dxy * dxy))
    else:
        raw_rate = np.zeros(nel, dtype=float)

    epsd_key = "epsd48" if "epsd48" in extra else ("epsd" if "epsd" in extra else None)
    if p.fcut < _INF and dt > 0.0:
        alpha = min(1.0, 2.0 * math.pi * p.fcut * dt)
        if epsd_key is not None:
            extra[epsd_key][:] = alpha * raw_rate + (1.0 - alpha) * extra[epsd_key]
            epsd = extra[epsd_key].copy()
        else:
            extra["epsd48"] = raw_rate.copy()
            epsd = raw_rate.copy()
    else:
        epsd = raw_rate

    # 4. Total strain & maximum in-plane principal strain (sigeps48c.F:245-249)
    eps_key = "eps48" if "eps48" in extra else ("eps" if "eps" in extra else None)
    if eps_key is not None:
        eps_tot = extra[eps_key]
        if eps_tot.ndim == 1:
            if nel == 1:
                eps_tot = eps_tot.reshape(1, -1)
                extra[eps_key] = eps_tot
            else:
                eps_tot = np.broadcast_to(eps_tot, (nel, eps_tot.shape[0])).copy()
                extra[eps_key] = eps_tot
        eps_tot[:, :3] += deps_arr[:, :3]
    elif p.eps_t1 < _INF:
        eps_tot = deps_arr[:, :3].copy()
        extra["eps48"] = eps_tot
    else:
        eps_tot = deps_arr[:, :3]

    epst = _principal_strain_2d(eps_tot)
    fail = tensile_failure_factor(p, epst)

    # 5. Yield stress and hardening slope
    yld, h_total, h_iso, h_kin = eval_yield_and_hardening(p, epsp_arr, epsd, fail)

    # 6. Plane stress projection (IFLAG=0: radial projection, IFLAG=1: Newton-Raphson return)
    iflag_val = iflag
    if iflag_val is None:
        if "iflag" in extra:
            iflag_val = int(extra["iflag"])
        else:
            iflag_val = int(kwargs.get("iflag", 0))

    nnu11 = p.nu / (1.0 - p.nu)
    nu31 = (1.0 - 2.0 * p.nu) / (1.0 - p.nu)

    if iflag_val == 1:
        off_arr = extra.get("off", np.ones(nel))
        sig_rel_new, dpla, dezz_pl = plane_stress_return_newton_law48(
            p, sig_rel, yld, h_iso, epsp_arr, off=off_arr, nmax=int(kwargs.get("nmax", 3))
        )
        plastic = dpla > 0.0
        epsp_arr += dpla
        dezz_el = -(deps_arr[:, 0] + deps_arr[:, 1]) * nnu11
        dezz = dezz_el + dezz_pl
    else:
        # Radial projection (sigeps48c.F:323-340)
        sxx = sig_rel[:, 0]
        syy = sig_rel[:, 1]
        sxy = sig_rel[:, 2]
        svm2 = sxx * sxx + syy * syy - sxx * syy + 3.0 * sxy * sxy
        svm = np.sqrt(np.maximum(svm2, 0.0))

        r = np.minimum(1.0, yld / np.maximum(svm, _EM20))
        dpla = (1.0 - r) * svm / np.maximum(G3 + h_iso, _EM20)
        plastic = dpla > 0.0

        two_step = kwargs.get("two_step", True)
        if two_step:
            yld_corr = yld + dpla * h_iso
            yld_corr = np.where(epsp_arr + dpla > p.eps_max, 0.0, yld_corr)
            r_corr = np.minimum(1.0, yld_corr / np.maximum(svm, _EM20))
            sig_rel_new = sig_rel * r_corr[:, None]
        else:
            sig_rel_new = sig_rel * r[:, None]

        epsp_arr += dpla
        dezz_el = -(deps_arr[:, 0] + deps_arr[:, 1]) * nnu11
        s_mean = 0.5 * (sig_rel_new[:, 0] + sig_rel_new[:, 1])
        dezz_pl = -dpla * s_mean / np.maximum(yld, _EM20)
        dezz = dezz_el + nu31 * dezz_pl

    # 7. Kinematic hardening update (sigeps48c.F:476-496)
    if sigb is not None:
        dsxx = sig_tr[:, 0] - (sig_rel_new[:, 0] + sigb[:, 0])
        dsyy = sig_tr[:, 1] - (sig_rel_new[:, 1] + sigb[:, 1])
        dsxy = sig_tr[:, 2] - (sig_rel_new[:, 2] + sigb[:, 2])

        dexx = dsxx - p.nu * dsyy
        deyy = dsyy - p.nu * dsxx
        dexy = 2.0 * (1.0 + p.nu) * dsxy

        beta = p.fisokin * h_total / np.maximum(p.E + h_total, _EM20) / 3.0
        sigpxx = beta * (4.0 * dexx + 2.0 * deyy)
        sigpyy = beta * (4.0 * deyy + 2.0 * dexx)
        sigpxy = beta * dexy

        sig_out = sig_tr.copy()
        sig_out[:, 0] = sig_rel_new[:, 0] + sigb[:, 0]
        sig_out[:, 1] = sig_rel_new[:, 1] + sigb[:, 1]
        sig_out[:, 2] = sig_rel_new[:, 2] + sigb[:, 2]

        sigb[:, 0] += np.where(plastic, sigpxx, 0.0)
        sigb[:, 1] += np.where(plastic, sigpyy, 0.0)
        sigb[:, 2] += np.where(plastic, sigpxy, 0.0)
    else:
        sig_out = sig_tr.copy()
        sig_out[:, :3] = sig_rel_new

    # 8. Layer thinning update dezz (sigeps48c.F:336-339 / 419-421)
    nnu11 = p.nu / (1.0 - p.nu)
    nu31 = (1.0 - 2.0 * p.nu) / (1.0 - p.nu)
    dezz_el = -(deps_arr[:, 0] + deps_arr[:, 1]) * nnu11
    s_mean = 0.5 * (sig_out[:, 0] + sig_out[:, 1])
    dezz_pl = -dpla * s_mean / np.maximum(yld, _EM20)
    dezz = dezz_el + nu31 * dezz_pl

    if "thk" in extra:
        extra["thk"] += dezz * extra["thk"]

    # 9. Element deletion check (eps_p > eps_p_max or eps_t >= eps_r2)
    deleted = (epsp_arr > p.eps_max) | (epst >= p.eps_t2)
    if np.any(deleted):
        sig_out[deleted] = 0.0
        if "off" in extra:
            if isinstance(extra["off"], np.ndarray):
                if extra["off"].ndim > 0:
                    extra["off"][deleted] = 0.0
                else:
                    extra["off"].fill(0.0)
            else:
                extra["off"] = 0.0
        else:
            extra["off"] = np.where(deleted, 0.0, 1.0)
        if "off48" in extra:
            if isinstance(extra["off48"], np.ndarray):
                if extra["off48"].ndim > 0:
                    extra["off48"][deleted] = 0.0
                else:
                    extra["off48"].fill(0.0)
            else:
                extra["off48"] = 0.0
        if "layfail" in extra:
            if isinstance(extra["layfail"], np.ndarray):
                if extra["layfail"].ndim > 0:
                    extra["layfail"][deleted] = 0.0
                else:
                    extra["layfail"].fill(0.0)
            else:
                extra["layfail"] = 0.0

    if single:
        return sig_out[0], float(epsp_arr[0])
    return sig_out, epsp_arr


# ----------------------------------------------------------------------------
# Consistent tangents for implicit analysis
# ----------------------------------------------------------------------------

def shell_membrane_tangent(mat: Any) -> np.ndarray:
    """(3, 3) plane-stress elastic membrane tangent for LAW48."""
    p = _get_params(mat)
    c = p.A11
    return np.array([
        [c, p.nu * c, 0.0],
        [p.nu * c, c, 0.0],
        [0.0, 0.0, p.G],
    ], dtype=float)


def _copy_extra(extra: dict[str, Any] | None) -> dict[str, Any] | None:
    """Deep-copy arrays and nested dicts in state extra dictionary."""
    if extra is None:
        return None
    res: dict[str, Any] = {}
    for k, v in extra.items():
        if isinstance(v, np.ndarray):
            res[k] = v.copy()
        elif isinstance(v, dict):
            res[k] = _copy_extra(v)
        elif hasattr(v, "copy"):
            try:
                res[k] = v.copy()
            except Exception:
                res[k] = v
        else:
            res[k] = v
    return res


def tangent_law48_solid(
    mat: Any,
    sig: np.ndarray | None = None,
    epsp: np.ndarray | float | None = None,
    epsp_incr: np.ndarray | float | None = None,
    dt: float = 0.0,
    extra: dict[str, Any] | None = None,
    *args: Any,
    deps: np.ndarray | None = None,
    eps: np.ndarray | None = None,
    symmetric: bool = False,
    h: float = 1e-7,
    **kwargs: Any,
) -> np.ndarray:
    """The consistent (algorithmic) elastoplastic tangent of the radial return
    for solid elements, (NEL, 6, 6) or (6, 6), Voigt / engineering shear.

    Supports both:
    1. Direct numerical perturbation (when ``deps`` is supplied or rate/damage is active):
       D_ij = (sigma(deps + h*e_j) - sigma(deps - h*e_j)) / (2*h)
    2. Analytical Simo & Hughes / de Souza Neto return-mapping operator:
       D = C - a (C - K 1(x)1) + b (N (x) N)
       a = 3G Δεp / q_tr,   b = 6G^2 (Δεp/q_tr - 1/(3G+H_iso))
    """
    p = _get_params(mat)
    G = p.G
    Kb = p.K

    # Disambiguate arguments
    if "deps" in kwargs and deps is None:
        deps = kwargs["deps"]
    if "eps" in kwargs and eps is None:
        eps = kwargs["eps"]
    if deps is None and eps is not None:
        deps = eps
    if "epsp_incr" in kwargs and epsp_incr is None:
        epsp_incr = kwargs["epsp_incr"]
    if "extra" in kwargs and extra is None:
        extra = kwargs["extra"]
    if "dt" in kwargs:
        dt = float(kwargs["dt"])
    if "symmetric" in kwargs:
        symmetric = bool(kwargs["symmetric"])
    if "h" in kwargs:
        h = float(kwargs["h"])

    # Disambiguate positional args
    if len(args) > 0:
        if epsp_incr is None and len(args) >= 1:
            epsp_incr = args[0]
        if extra is None and len(args) >= 2 and isinstance(args[1], dict):
            extra = args[1]

    # If sig was passed as deps or deps passed in epsp position
    if deps is None and epsp is not None and isinstance(epsp, np.ndarray):
        if (epsp.ndim == 2 and epsp.shape[1] == 6) or (epsp.ndim == 1 and epsp.shape[0] == 6):
            deps = epsp
            epsp = None
    if deps is None and epsp_incr is not None and isinstance(epsp_incr, np.ndarray):
        if (epsp_incr.ndim == 2 and epsp_incr.shape[1] == 6) or (epsp_incr.ndim == 1 and epsp_incr.shape[0] == 6):
            deps = epsp_incr
            epsp_incr = None

    # Build 6x6 elastic matrix C
    C = np.zeros((6, 6), dtype=float)
    c11 = Kb + 4.0 * G / 3.0
    c12 = Kb - 2.0 * G / 3.0
    C[0:3, 0:3] = c12
    np.fill_diagonal(C[0:3, 0:3], c11)
    C[3, 3] = G
    C[4, 4] = G
    C[5, 5] = G

    if sig is None and deps is None:
        return C

    # Determine element count and shapes
    if deps is not None:
        deps_arr = np.asarray(deps, dtype=float)
        single = (deps_arr.ndim == 1)
        if single:
            deps_arr = deps_arr.reshape(1, 6)
        nel = deps_arr.shape[0]
        sig_arr = np.asarray(sig, dtype=float) if sig is not None else np.zeros((nel, 6), dtype=float)
        if sig_arr.ndim == 1:
            sig_arr = sig_arr.reshape(1, 6)
    else:
        sig_arr = np.asarray(sig, dtype=float)
        single = (sig_arr.ndim == 1)
        if single:
            sig_arr = sig_arr.reshape(1, 6)
        nel = sig_arr.shape[0]
        deps_arr = None

    if nel == 0:
        return np.empty((0, 6, 6), dtype=float) if not single else np.empty((6, 6), dtype=float)

    # Extract plastic strain
    if epsp is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.asarray(epsp, dtype=float).flatten()
        if len(epsp_arr) == 1 and nel > 1:
            epsp_arr = np.full(nel, epsp_arr[0], dtype=float)

    # Extract element deletion mask
    off_raw = None
    if extra is not None:
        off_raw = extra.get("off", extra.get("off48", extra.get("layfail", None)))
    if "off" in kwargs and off_raw is None:
        off_raw = kwargs["off"]

    deleted_mask = (epsp_arr > p.eps_max)
    if off_raw is not None:
        off_arr = np.asarray(off_raw, dtype=float).flatten()
        if len(off_arr) == 1 and nel > 1:
            off_arr = np.full(nel, off_arr[0], dtype=float)
        deleted_mask = deleted_mask | (off_arr <= 0.0)

    # Check total strain damage deletion if total strain is present
    eps_key = "eps48" if (extra is not None and "eps48" in extra) else ("eps" if (extra is not None and "eps" in extra) else None)
    if eps_key is not None and p.eps_t1 < _INF:
        eps_tot = np.asarray(extra[eps_key], dtype=float)
        if eps_tot.ndim == 1:
            eps_tot = eps_tot.reshape(1, -1)
        if eps_tot.shape[0] == nel:
            epst_chk = _principal_strain_3d(eps_tot)
            deleted_mask = deleted_mask | (epst_chk >= p.eps_t2)

    # Numerical perturbation path when deps is provided
    if deps_arr is not None:
        D = np.zeros((nel, 6, 6), dtype=float)
        active = ~deleted_mask
        if np.any(active):
            for j in range(6):
                ej = np.zeros_like(deps_arr)
                ej[:, j] = h
                ex_p = _copy_extra(extra)
                ex_m = _copy_extra(extra)
                sp, _, _ = solid_update_law48(p, sig_arr, deps_arr + ej, epsp=epsp_arr, dt=dt, extra=ex_p)
                sm, _, _ = solid_update_law48(p, sig_arr, deps_arr - ej, epsp=epsp_arr, dt=dt, extra=ex_m)
                if sp.ndim == 1:
                    sp = sp.reshape(1, 6)
                    sm = sm.reshape(1, 6)
                D[:, :, j] = (sp - sm) / (2.0 * h)
        if np.any(deleted_mask):
            D[deleted_mask] = 0.0
        if symmetric:
            D = 0.5 * (D + np.swapaxes(D, -1, -2))
        return D[0] if single else D

    # Analytical Simo & Hughes path
    D = np.broadcast_to(C, (nel, 6, 6)).copy()

    if epsp_incr is None:
        if np.any(deleted_mask):
            D[deleted_mask] = 0.0
        if symmetric:
            D = 0.5 * (D + np.swapaxes(D, -1, -2))
        return D[0] if single else D

    dep_arr = np.asarray(epsp_incr, dtype=float).flatten()
    if len(dep_arr) == 1 and nel > 1:
        dep_arr = np.full(nel, dep_arr[0], dtype=float)

    plastic = (dep_arr > 0.0) & (~deleted_mask)
    if not np.any(plastic):
        if np.any(deleted_mask):
            D[deleted_mask] = 0.0
        if symmetric:
            D = 0.5 * (D + np.swapaxes(D, -1, -2))
        return D[0] if single else D

    idx = np.where(plastic)[0]
    s = sig_arr[idx].copy()
    pm = (s[:, 0] + s[:, 1] + s[:, 2]) / 3.0
    s[:, 0] -= pm
    s[:, 1] -= pm
    s[:, 2] -= pm

    snorm = np.sqrt(
        s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2
        + 2.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
    )
    snorm = np.maximum(snorm, _EM20)
    Nv = s / snorm[:, None]
    q = np.sqrt(1.5) * snorm
    dep = dep_arr[idx]
    q_tr = q + 3.0 * G * dep

    fail = np.ones(len(idx), dtype=float)
    if eps_key is not None and p.eps_t1 < _INF:
        eps_tot = np.asarray(extra[eps_key], dtype=float)
        if eps_tot.ndim == 1:
            eps_tot = eps_tot.reshape(1, -1)
        epst = _principal_strain_3d(eps_tot[idx])
        fail = tensile_failure_factor(p, epst)

    edot = np.zeros(len(idx), dtype=float)
    epsd_key = "epsd48" if (extra is not None and "epsd48" in extra) else ("epsd" if (extra is not None and "epsd" in extra) else None)
    if epsd_key is not None:
        epsd_tot = np.asarray(extra[epsd_key], dtype=float)
        if epsd_tot.ndim == 0:
            edot = np.full(len(idx), float(epsd_tot))
        elif len(epsd_tot) == 1 and nel > 1:
            edot = np.full(len(idx), float(epsd_tot[0]))
        else:
            edot = epsd_tot[idx]

    _, _, h_iso, _ = eval_yield_and_hardening(p, epsp_arr[idx], edot, fail)
    Hbar = np.maximum(h_iso, 0.0)

    a = 3.0 * G * dep / q_tr
    b = 6.0 * G * G * (dep / q_tr - 1.0 / (3.0 * G + Hbar))

    ee = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    KeeT = Kb * np.outer(ee, ee)
    C_minus_vol = C - KeeT
    NN = np.einsum("mi,mj->mij", Nv, Nv)

    D[idx] = (
        C[None, :, :]
        - a[:, None, None] * C_minus_vol[None, :, :]
        + b[:, None, None] * NN
    )

    if np.any(deleted_mask):
        D[deleted_mask] = 0.0

    if symmetric:
        D = 0.5 * (D + np.swapaxes(D, -1, -2))

    if single:
        return D[0]
    return D


#: Plane-stress von Mises metric P (Voigt [xx, yy, xy], engineering shear):
#: q^2 = sig^T P sig = sxx^2 - sxx*syy + syy^2 + 3*sxy^2.
_P_PLANE = np.array([
    [1.0, -0.5, 0.0],
    [-0.5, 1.0, 0.0],
    [0.0, 0.0, 3.0],
], dtype=float)


def tangent_law48_shell(
    mat: Any,
    sig: np.ndarray | None = None,
    epsp: np.ndarray | float | None = None,
    epsp_incr: np.ndarray | float | None = None,
    dt: float = 0.0,
    extra: dict[str, Any] | None = None,
    *args: Any,
    deps: np.ndarray | None = None,
    eps: np.ndarray | None = None,
    symmetric: bool = False,
    h: float = 1e-7,
    **kwargs: Any,
) -> np.ndarray:
    """The consistent algorithmic plane-stress tangent for shells,
    (NEL, 3, 3) or (3, 3), Voigt [xx, yy, xy] with engineering shear.

    Supports both:
    1. Direct numerical perturbation (when ``deps`` is supplied or rate/damage is active):
       D_ij = (sigma(deps + h*e_j) - sigma(deps - h*e_j)) / (2*h)
    2. Analytical plane-stress radial return tangent:
       D = s C + [H_iso/(3G+H_iso) - s] / q_tr^2 * sig_tr (x) (C P sig_tr)
    """
    p = _get_params(mat)
    C = shell_membrane_tangent(p)

    # Disambiguate arguments
    if "deps" in kwargs and deps is None:
        deps = kwargs["deps"]
    if "eps" in kwargs and eps is None:
        eps = kwargs["eps"]
    if deps is None and eps is not None:
        deps = eps
    if "epsp_incr" in kwargs and epsp_incr is None:
        epsp_incr = kwargs["epsp_incr"]
    if "extra" in kwargs and extra is None:
        extra = kwargs["extra"]
    if "dt" in kwargs:
        dt = float(kwargs["dt"])
    if "symmetric" in kwargs:
        symmetric = bool(kwargs["symmetric"])
    if "h" in kwargs:
        h = float(kwargs["h"])

    # Disambiguate positional args
    if len(args) > 0:
        if epsp_incr is None and len(args) >= 1:
            epsp_incr = args[0]
        if extra is None and len(args) >= 2 and isinstance(args[1], dict):
            extra = args[1]

    # If sig was passed as deps or deps passed in epsp position
    if deps is None and epsp is not None and isinstance(epsp, np.ndarray):
        if (epsp.ndim == 2 and epsp.shape[1] in (3, 5)) or (epsp.ndim == 1 and epsp.shape[0] in (3, 5)):
            deps = epsp
            epsp = None
    if deps is None and epsp_incr is not None and isinstance(epsp_incr, np.ndarray):
        if (epsp_incr.ndim == 2 and epsp_incr.shape[1] in (3, 5)) or (epsp_incr.ndim == 1 and epsp_incr.shape[0] in (3, 5)):
            deps = epsp_incr
            epsp_incr = None

    if sig is None and deps is None:
        return C

    # Determine element count and shapes
    if deps is not None:
        deps_arr = np.asarray(deps, dtype=float)
        single = (deps_arr.ndim == 1)
        if single:
            deps_arr = deps_arr.reshape(1, -1)
        nel = deps_arr.shape[0]
        sig_arr = np.asarray(sig, dtype=float) if sig is not None else np.zeros((nel, deps_arr.shape[1]), dtype=float)
        if sig_arr.ndim == 1:
            sig_arr = sig_arr.reshape(1, -1)
    else:
        sig_arr = np.asarray(sig, dtype=float)
        single = (sig_arr.ndim == 1)
        if single:
            sig_arr = sig_arr.reshape(1, -1)
        nel = sig_arr.shape[0]
        deps_arr = None

    if nel == 0:
        return np.empty((0, 3, 3), dtype=float) if not single else np.empty((3, 3), dtype=float)

    # Extract plastic strain
    if epsp is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.asarray(epsp, dtype=float).flatten()
        if len(epsp_arr) == 1 and nel > 1:
            epsp_arr = np.full(nel, epsp_arr[0], dtype=float)

    # Extract element deletion mask
    off_raw = None
    if extra is not None:
        off_raw = extra.get("off", extra.get("off48", extra.get("layfail", None)))
    if "off" in kwargs and off_raw is None:
        off_raw = kwargs["off"]

    deleted_mask = (epsp_arr > p.eps_max)
    if off_raw is not None:
        off_arr = np.asarray(off_raw, dtype=float).flatten()
        if len(off_arr) == 1 and nel > 1:
            off_arr = np.full(nel, off_arr[0], dtype=float)
        deleted_mask = deleted_mask | (off_arr <= 0.0)

    eps_key = "eps48" if (extra is not None and "eps48" in extra) else ("eps" if (extra is not None and "eps" in extra) else None)
    if eps_key is not None and p.eps_t1 < _INF:
        eps_tot = np.asarray(extra[eps_key], dtype=float)
        if eps_tot.ndim == 1:
            eps_tot = eps_tot.reshape(1, -1)
        if eps_tot.shape[0] == nel:
            epst_chk = _principal_strain_2d(eps_tot[:, :3])
            deleted_mask = deleted_mask | (epst_chk >= p.eps_t2)

    # Numerical perturbation path when deps is provided
    if deps_arr is not None:
        D = np.zeros((nel, 3, 3), dtype=float)
        active = ~deleted_mask
        if np.any(active):
            for j in range(3):
                ej = np.zeros_like(deps_arr)
                ej[:, j] = h
                ex_p = _copy_extra(extra)
                ex_m = _copy_extra(extra)
                sp, _ = shell_update_law48(p, sig_arr, deps_arr + ej, epsp=epsp_arr, dt=dt, extra=ex_p)
                sm, _ = shell_update_law48(p, sig_arr, deps_arr - ej, epsp=epsp_arr, dt=dt, extra=ex_m)
                if sp.ndim == 1:
                    sp = sp.reshape(1, -1)
                    sm = sm.reshape(1, -1)
                D[:, :, j] = (sp[:, :3] - sm[:, :3]) / (2.0 * h)
        if np.any(deleted_mask):
            D[deleted_mask] = 0.0
        if symmetric:
            D = 0.5 * (D + np.swapaxes(D, -1, -2))
        return D[0] if single else D

    # Analytical plane-stress radial projection path
    D = np.broadcast_to(C, (nel, 3, 3)).copy()

    if epsp_incr is None:
        if np.any(deleted_mask):
            D[deleted_mask] = 0.0
        if symmetric:
            D = 0.5 * (D + np.swapaxes(D, -1, -2))
        return D[0] if single else D

    dep_arr = np.asarray(epsp_incr, dtype=float).flatten()
    if len(dep_arr) == 1 and nel > 1:
        dep_arr = np.full(nel, dep_arr[0], dtype=float)

    plastic = (dep_arr > 0.0) & (~deleted_mask)
    if not np.any(plastic):
        if np.any(deleted_mask):
            D[deleted_mask] = 0.0
        if symmetric:
            D = 0.5 * (D + np.swapaxes(D, -1, -2))
        return D[0] if single else D

    idx = np.where(plastic)[0]
    dl = dep_arr[idx]
    s_c = sig_arr[idx, :3]

    sy = np.sqrt(np.maximum(np.einsum("mi,ij,mj->m", s_c, _P_PLANE, s_c), 0.0))
    sy = np.maximum(sy, _EM20)
    q_tr = sy + 3.0 * p.G * dl
    s_factor = sy / q_tr
    sig_tr = s_c / s_factor[:, None]

    fail = np.ones(len(idx), dtype=float)
    if eps_key is not None and p.eps_t1 < _INF:
        eps_tot = np.asarray(extra[eps_key], dtype=float)
        if eps_tot.ndim == 1:
            eps_tot = eps_tot.reshape(1, -1)
        epst = _principal_strain_2d(eps_tot[idx, :3])
        fail = tensile_failure_factor(p, epst)

    edot = np.zeros(len(idx), dtype=float)
    epsd_key = "epsd48" if (extra is not None and "epsd48" in extra) else ("epsd" if (extra is not None and "epsd" in extra) else None)
    if epsd_key is not None:
        epsd_tot = np.asarray(extra[epsd_key], dtype=float)
        if epsd_tot.ndim == 0:
            edot = np.full(len(idx), float(epsd_tot))
        elif len(epsd_tot) == 1 and nel > 1:
            edot = np.full(len(idx), float(epsd_tot[0]))
        else:
            edot = epsd_tot[idx]

    _, _, h_iso, _ = eval_yield_and_hardening(p, epsp_arr[idx], edot, fail)
    Hbar = np.maximum(h_iso, 0.0)

    gamma = (Hbar / (3.0 * p.G + Hbar) - s_factor) / (q_tr ** 2)
    CP = C @ _P_PLANE
    v = np.einsum("ij,mj->mi", CP, sig_tr)
    rank1 = np.einsum("mi,mj->mij", sig_tr, v)
    D[idx] = s_factor[:, None, None] * C[None, :, :] + gamma[:, None, None] * rank1

    if np.any(deleted_mask):
        D[deleted_mask] = 0.0

    if symmetric:
        D = 0.5 * (D + np.swapaxes(D, -1, -2))

    if single:
        return D[0]
    return D


# ----------------------------------------------------------------------------
# Aliases & Registrations
# ----------------------------------------------------------------------------

solid_update = solid_update_law48
shell_update = shell_update_law48
solid_sound_speed = sound_speed_solid_law48
shell_sound_speed = sound_speed_shell_law48
sound_speed = sound_speed_solid_law48
consistent_solid_tangent = tangent_law48_solid
consistent_shell_tangent = tangent_law48_shell
solid_tangent = tangent_law48_solid
shell_tangent = tangent_law48_shell


def extra_shapes(mat: Any = None, nip: int | None = None) -> dict[str, tuple[int, ...]]:
    """Per-element persistent state shapes for LAW48.

    Solids (nip=None):
      eps48: (6,) total strain
      sigb48: (6,) backstress
      epsd48: () filtered strain rate
      off48: () alive mask
    Shells (nip given):
      eps48: (nip, 3) total in-plane strain
      sigb48: (nip, 3) in-plane backstress
      epsd48: (nip,) layer filtered strain rate
      off48: (nip,) layer alive mask
    """
    return {
        "eps48": (nip, 3) if nip else (6,),
        "sigb48": (nip, 3) if nip else (6,),
        "epsd48": (nip,) if nip else (),
        "off48": (nip,) if nip else (),
    }



def _register():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for key in ("48", 48, "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "LAW48_ZHAO"):
            MAT_PHYSICS_REGISTRY[key] = build_law48
    except Exception:
        pass


_register()
