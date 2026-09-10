"""
LAW25 — Composite Anisotropic Plasticity Model (/MAT/LAW25, /MAT/COMP_PLAS, /MAT/COMPSH, /MAT/TSAI_WU, /MAT/CRASURV).

Fortran origin:
  - starter/source/materials/mat/mat025/hm_read_mat25.F
  - starter/source/materials/mat/mat025/read_mat25_tsaiwu.F90
  - starter/source/materials/mat/mat025/read_mat25_crasurv.F90
  - engine/source/materials/mat/mat025/sigeps25c.F
  - engine/source/materials/mat/mat025/mat25_tsaiwu_c.F90 (shells)
  - engine/source/materials/mat/mat025/mat25_tsaiwu_s.F90 (solids)
  - engine/source/materials/mat/mat025/mat25_crasurv_c.F90 (shells CRASURV)
  - engine/source/materials/mat/mat025/mat25_crasurv_s.F90 (solids CRASURV)
  - engine/source/materials/mat/mat025/m25law.F
  - engine/source/materials/mat/mat025/m25crak.F
  - engine/source/materials/mat/mat025/m25delam.F
  - hm_cfg_files/config/CFG/radioss110/MAT/matl25_compsh.cfg
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_EM15 = 1.0e-15
_INF = 1.0e30


# ============================================================================
# Core Mathematical Formulations (Fortran parity)
# ============================================================================

def tsai_wu_coefficients(
    sigyt1: float,
    sigyc1: float,
    sigyt2: float,
    sigyc2: float,
    sig12: float | tuple[float, float] | list[float],
    alpha: float = 1.0,
) -> Dict[str, float]:
    """Calculate Tsai-Wu failure/yield coefficients.

    Fortran source: starter/source/materials/mat/mat025/read_mat25_tsaiwu.F90:310-316

    Formulas:
      F1  = 1/sigyt1 - 1/sigyc1
      F2  = 1/sigyt2 - 1/sigyc2
      F11 = 1 / (sigyt1 * sigyc1)
      F22 = 1 / (sigyt2 * sigyc2)
      F33 = 1 / (sig12^2)  [or 1 / (sigyt12 * sigyc12)]
      F12 = -0.5 * alpha * sqrt(F11 * F22)
          = -alpha / (2 * sqrt(sigyt1 * sigyc1 * sigyt2 * sigyc2))
    """
    if isinstance(sig12, (tuple, list)):
        sigyt12, sigyc12 = float(sig12[0]), float(sig12[1])
    else:
        sigyt12 = float(sig12)
        sigyc12 = float(sig12)

    s_yt1 = max(float(sigyt1), _EM20)
    s_yc1 = max(float(sigyc1), _EM20)
    s_yt2 = max(float(sigyt2), _EM20)
    s_yc2 = max(float(sigyc2), _EM20)
    s_yt12 = max(float(sigyt12), _EM20)
    s_yc12 = max(float(sigyc12), _EM20)

    f1 = 1.0 / s_yt1 - 1.0 / s_yc1
    f2 = 1.0 / s_yt2 - 1.0 / s_yc2
    f11 = 1.0 / (s_yt1 * s_yc1)
    f22 = 1.0 / (s_yt2 * s_yc2)
    f33 = 1.0 / (s_yt12 * s_yc12)
    denom = 2.0 * math.sqrt(s_yt1 * s_yc1 * s_yt2 * s_yc2)
    f12 = -float(alpha) / denom if denom > 0.0 else 0.0

    return {
        "F1": f1,
        "F2": f2,
        "F11": f11,
        "F22": f22,
        "F33": f33,
        "F12": f12,
    }


def tsai_wu_yield_criterion(
    s1: float | np.ndarray,
    s2: float | np.ndarray,
    s12: float | np.ndarray,
    F1: float,
    F2: float,
    F11: float,
    F22: float,
    F33: float,
    F12: float,
) -> float | np.ndarray:
    """Evaluate Tsai-Wu yield/failure function W.

    Fortran source: engine/source/materials/mat/mat025/mat25_tsaiwu_c.F90:480-484

    W = F1*s1 + F2*s2 + F11*s1^2 + F22*s2^2 + 2*F12*s1*s2 + F33*s12^2
    """
    return (
        F1 * s1
        + F2 * s2
        + F11 * (s1 ** 2)
        + F22 * (s2 ** 2)
        + 2.0 * F12 * s1 * s2
        + F33 * (s12 ** 2)
    )


def tsai_wu_flow_normal(
    s1: float | np.ndarray,
    s2: float | np.ndarray,
    s12: float | np.ndarray,
    F1: float,
    F2: float,
    F11: float,
    F22: float,
    F33: float,
    F12: float,
) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Compute gradient normal of Tsai-Wu yield function (flow vector).

    Fortran source: engine/source/materials/mat/mat025/mat25_tsaiwu_c.F90:501-503

    dF/ds1  = F1 + 2*F11*s1 + 2*F12*s2
    dF/ds2  = F2 + 2*F22*s2 + 2*F12*s1
    dF/ds12 = 2*F33*s12
    """
    df_ds1 = F1 + 2.0 * F11 * s1 + 2.0 * F12 * s2
    df_ds2 = F2 + 2.0 * F22 * s2 + 2.0 * F12 * s1
    df_ds12 = 2.0 * F33 * s12
    return df_ds1, df_ds2, df_ds12


def hardening_yield(
    wpla: float,
    b: float,
    n: float,
    epspfac: float = 1.0,
    fmax: float = _INF,
) -> float:
    """Hardening yield limit f_yld.

    Fortran source: engine/source/materials/mat/mat025/mat25_tsaiwu_c.F90:463-474

    f_yld = min(fmax, (1 + b * wpla^n) * epspfac)
    """
    if wpla > 0.0 and b > 0.0:
        base = 1.0 + b * (wpla ** n)
    else:
        base = 1.0
    return min(float(fmax), base * float(epspfac))


def strain_rate_factor(
    eps_dot: float,
    c: float = 0.0,
    epdr: float = 1.0,
    formulation: str = "log",
) -> float:
    """Strain rate factor epspfac.

    Fortran logarithmic rate formulation:
      mat25_tsaiwu_c.F90:453-462:
      epspfac = 1 + c * log(eps_dot / epdr)  for eps_dot > epdr

    Power-law (Cowper-Symonds) rate formulation:
      epspfac = 1 + (eps_dot / c) ** (1 / epdr)
    """
    ed = float(eps_dot)
    c_val = float(c)
    epdr_val = float(epdr)

    if ed <= 0.0 or c_val <= 0.0:
        return 1.0

    if formulation.lower() in ("cowper_symonds", "power"):
        if epdr_val > 0.0:
            return 1.0 + (ed / c_val) ** (1.0 / epdr_val)
        return 1.0
    else:
        # Default: Fortran log law
        if epdr_val > 0.0 and ed > epdr_val:
            return 1.0 + c_val * math.log(ed / epdr_val)
        return 1.0


def tensile_damage(
    epst: float,
    epst_limit: float,
    epsm_limit: float,
    dmax: float = 0.999,
    dmg_old: float = 0.0,
) -> float:
    """Damage evolution in tension.

    Fortran source: engine/source/materials/mat/mat025/m25crak.F:74-76

    dam1 = (epst - epst1) / (epsm1 - epst1)
    dam2 = dam1 * epsm1 / epst
    dmg  = min(max(dam2, dmg_old), dmax)
    """
    if epst > epst_limit and epsm_limit > epst_limit:
        dam1 = (epst - epst_limit) / (epsm_limit - epst_limit)
        dam2 = dam1 * epsm_limit / max(epst, _EM20)
        return float(min(max(dam2, dmg_old), dmax))
    return float(dmg_old)


def sound_speed_law25(
    e11: float,
    e22: float,
    nu12: float,
    g12: float,
    g23: float,
    g31: float,
    rho0: float,
) -> float:
    """Sound speed for LAW25 matching Fortran read_mat25_tsaiwu.F90:288.

    c = sqrt(max(C1, G12, G23, G31) / rho0)
    where C1 = max(E1, E2) / (1 - nu12 * nu21)
          nu21 = nu12 * E2 / E1
    """
    e1 = float(e11)
    e2 = float(e22)
    nu = float(nu12)
    nu21 = nu * e2 / max(e1, _EM20) if e1 > 0.0 else 0.0
    detc = max(1.0e-15, 1.0 - nu * nu21)
    c1 = max(e1, e2) / detc
    gmax = max(float(g12), float(g23), float(g31))
    mod_max = max(c1, gmax)
    r = float(rho0)
    return math.sqrt(mod_max / max(r, _EM20)) if r > 0.0 else math.sqrt(mod_max)


# ============================================================================
# Parameter Extraction & Constructor
# ============================================================================

def build_law25(rec: Any = None, **kwargs: Any) -> Material:
    """Physics constructor for the cfg-parsed /MAT/LAW25 composite record.

    Follows starter/source/materials/mat/mat025/read_mat25_tsaiwu.F90 and
    read_mat25_crasurv.F90.
    """
    if rec is None:
        p: Dict[str, Any] = dict(kwargs)
        _id = int(kwargs.get("id", 1))
        rho0_in = kwargs.get("rho0", kwargs.get("density", kwargs.get("MAT_RHO", 0.0)))
        _title = str(kwargs.get("title", "LAW25_COMPOSITE"))
    elif isinstance(rec, dict):
        base_params = rec.get("params", rec)
        p = {**base_params, **kwargs}
        _id = int(rec.get("id", kwargs.get("id", 1)))
        rho0_in = rec.get("rho0", rec.get("density", kwargs.get("rho0", kwargs.get("density", 0.0))))
        _title = str(rec.get("title", kwargs.get("title", "LAW25_COMPOSITE")))
    elif hasattr(rec, "params"):
        p = {**rec.params, **kwargs}
        _id = int(getattr(rec, "id", kwargs.get("id", 1)))
        rho0_in = getattr(rec, "rho0", getattr(rec, "density", kwargs.get("rho0", 0.0)))
        _title = str(getattr(rec, "title", kwargs.get("title", "LAW25_COMPOSITE")))
    elif hasattr(rec, "__dataclass_fields__"):
        base_dict = {k: getattr(rec, k) for k in rec.__dataclass_fields__ if hasattr(rec, k)}
        p = {**base_dict, **kwargs}
        _id = int(getattr(rec, "id", kwargs.get("id", 1)))
        rho0_in = getattr(rec, "rho0", kwargs.get("rho0", 0.0))
        _title = str(getattr(rec, "title", kwargs.get("title", "LAW25_COMPOSITE")))
    else:
        p = dict(kwargs)
        _id = int(kwargs.get("id", 1))
        rho0_in = kwargs.get("rho0", 0.0)
        _title = str(kwargs.get("title", "LAW25_COMPOSITE"))

    def _get(keys: Sequence[str], default: float = 0.0) -> float:
        for k in keys:
            if k in p and p[k] is not None:
                try:
                    return float(p[k])
                except (ValueError, TypeError):
                    pass
        return float(default)

    def _geti(keys: Sequence[str], default: int = 0) -> int:
        for k in keys:
            if k in p and p[k] is not None:
                try:
                    return int(p[k])
                except (ValueError, TypeError):
                    pass
        return int(default)

    rho0 = _get(["MAT_RHO", "Refer_Rho", "rho0", "density", "rho"], float(rho0_in))
    rhor = _get(["Refer_Rho", "rhor", "rho_ref"], 0.0)

    # Elasticity
    e11 = _get(["MAT_EA", "e11", "E11", "E1", "e1"], 0.0)
    e22 = _get(["MAT_EB", "e22", "E22", "E2", "e2"], 0.0)
    e33 = _get(["MAT_EC", "e33", "E33", "E3", "e3"], 0.0)
    if e33 <= 0.0:
        e33 = max(e11, e22)

    nu12 = _get(["MAT_PRAB", "nu12", "NU12", "prab", "nu"], 0.0)
    iform = _geti(["MAT_Iflag", "iform", "IFORM", "iflag"], 0)

    g12 = _get(["MAT_GAB", "g12", "G12"], 0.0)
    g23 = _get(["MAT_GBC", "g23", "G23"], 0.0)
    g31 = _get(["MAT_GCA", "g31", "G31"], 0.0)

    # Failure limits
    eps_f1 = _get(["MAT_EPSF1", "eps_f1", "EPS_F1", "epsf1"], _INF)
    eps_f2 = _get(["MAT_EPSF2", "eps_f2", "EPS_F2", "epsf2"], _INF)
    eps_t1 = _get(["MAT_EPST1", "eps_t1", "EPS_T1", "epst1"], _INF)
    eps_m1 = _get(["MAT_EPSM1", "eps_m1", "EPS_M1", "epsm1"], 1.1 * _INF)
    eps_t2 = _get(["MAT_EPST2", "eps_t2", "EPS_T2", "epst2"], _INF)
    eps_m2 = _get(["MAT_EPSM2", "eps_m2", "EPS_M2", "epsm2"], 1.1 * _INF)
    dmax = _get(["MAT_DAMAGE", "dmax", "DMAX"], 0.999)

    # Hardening / plasticity
    wpmax = _get(["WPMAX", "wpmax", "wplamx"], _INF)
    wpref = _get(["WPREF", "wpref", "wplaref"], 1.0)
    ioff = _geti(["Itype", "ioff", "IOFF"], 0)
    ratio = _get(["MAT_R00", "ratio"], 1.0)

    # Tsai-Wu specific
    b = _get(["MAT_BETA", "b", "B", "beta"], 0.0)
    n = _get(["MAT_HARD", "n", "N", "hard"], 1.0)
    fmax = _get(["MAT_SIG", "fmax", "FMAX"], _INF)
    sig_1yt = _get(["MAT_SIGYT1", "sig_1yt", "sigyt1"], _INF)
    sig_2yt = _get(["MAT_SIGYT2", "sig_2yt", "sigyt2"], _INF)
    sig_1yc = _get(["MAT_SIGYC1", "MAT_SIG1_yc", "sig_1yc", "sigyc1"], _INF)
    sig_2yc = _get(["MAT_SIGYC2", "MAT_SIG2_yc", "sig_2yc", "sigyc2"], _INF)
    alpha = _get(["MAT_ALPHA", "alpha", "ALPHA"], 1.0)
    sig_12 = _get(["MAT_SIG12", "sig12", "SIG12", "sig_12"], _INF)
    sig_12yc = _get(["MAT_SIGC12", "sig_12yc", "sigc12", "sigyc12"], sig_12)
    sig_12yt = _get(["MAT_SIGT12", "MAT_SIG12_yt", "sig_12yt", "sigt12", "sigyt12"], sig_12)
    if sig_12yc >= _INF and sig_12yt < _INF:
        sig_12yc = sig_12yt
    elif sig_12yt >= _INF and sig_12yc < _INF:
        sig_12yt = sig_12yc

    c = _get(["MAT_SRC", "c", "C", "src"], 0.0)
    eps_rate_0 = _get(["MAT_SRP", "eps_rate_0", "srp", "epdr"], 1.0)
    icc = _geti(["STRFLAG", "icc", "ICC", "strflag"], 1)

    # CRASURV specific
    iflawp = _geti(["MAT_Fl", "iflawp", "IFLAWP"], 0)
    b_1t = _get(["MAT_b1_t", "b_1t", "b1_t"], 0.0)
    n_1t = _get(["MAT_n1_t", "n_1t", "n1_t"], 1.0)
    sig_1maxt = _get(["MAT_SIG1max_t", "sig_1maxt", "sig1max_t"], _INF)
    c_1t = _get(["MAT_c1_t", "c_1t", "c1_t"], 0.0)
    eps_1t1 = _get(["MAT_EPS1_t1", "eps_1t1", "eps1_t1"], _INF)
    eps_2t1 = _get(["MAT_EPS2_t1", "eps_2t1", "eps2_t1"], 1.2 * _INF)
    sig_rst1 = _get(["MAT_SIGres_t1", "sig_rst1", "sigres_t1"], 0.0)
    wpmax_t1 = _get(["MAT_Wmax_pt1", "wpmax_t1"], _INF)

    b_2t = _get(["MAT_b2_t", "b_2t", "b2_t"], 0.0)
    n_2t = _get(["MAT_n2_t", "n_2t", "n2_t"], 1.0)
    sig_2maxt = _get(["MAT_SIG2max_t", "sig_2maxt", "sig2max_t"], _INF)
    c_2t = _get(["MAT_c2_t", "c_2t", "c2_t"], 0.0)
    eps_1t2 = _get(["MAT_EPS1_t2", "eps_1t2", "eps1_t2"], _INF)
    eps_2t2 = _get(["MAT_EPS2_t2", "eps_2t2", "eps2_t2"], 1.2 * _INF)
    sig_rst2 = _get(["MAT_SIGres_t2", "sig_rst2", "sigres_t2"], 0.0)
    wpmax_t2 = _get(["MAT_Wmax_pt2", "wpmax_t2"], _INF)

    b_1c = _get(["MAT_b1_c", "b_1c", "b1_c"], 0.0)
    n_1c = _get(["MAT_n1_c", "n_1c", "n1_c"], 1.0)
    sig_1maxc = _get(["MAT_SIG1max_c", "sig_1maxc", "sig1max_c"], _INF)
    c_1c = _get(["MAT_c1_c", "c_1c", "c1_c"], 0.0)
    eps_1c1 = _get(["MAT_EPS1_c1", "eps_1c1", "eps1_c1"], _INF)
    eps_2c1 = _get(["MAT_EPS2_c1", "eps_2c1", "eps2_c1"], 1.2 * _INF)
    sig_rsc1 = _get(["MAT_SIGres_c1", "sig_rsc1", "sigres_c1"], 0.0)
    wpmax_c1 = _get(["MAT_Wmax_pc1", "wpmax_c1"], _INF)

    b_2c = _get(["MAT_b2_c", "b_2c", "b2_c"], 0.0)
    n_2c = _get(["MAT_n2_c", "n_2c", "n2_c"], 1.0)
    sig_2maxc = _get(["MAT_SIG2max_c", "sig_2maxc", "sig2max_c"], _INF)
    c_2c = _get(["MAT_c2_c", "c_2c", "c2_c"], 0.0)
    eps_1c2 = _get(["MAT_EPS1_c2", "eps_1c2", "eps1_c2"], _INF)
    eps_2c2 = _get(["MAT_EPS2_c2", "eps_2c2", "eps2_c2"], 1.2 * _INF)
    sig_rsc2 = _get(["MAT_SIGres_c2", "sig_rsc2", "sigres_c2"], 0.0)
    wpmax_c2 = _get(["MAT_Wmax_pc2", "wpmax_c2"], _INF)

    b_12t = _get(["MAT_b12_t", "b_12t", "b12_t"], 0.0)
    n_12t = _get(["MAT_n12_t", "n_12t", "n12_t"], 1.0)
    sig_12maxt = _get(["MAT_SIG12max_t", "sig_12maxt", "sig12max_t"], _INF)
    c_12t = _get(["MAT_c12_t", "c_12t", "c12_t"], 0.0)
    eps_1t12 = _get(["MAT_EPS1_t12", "eps_1t12", "eps1_t12"], _INF)
    eps_2t12 = _get(["MAT_EPS2_t12", "eps_2t12", "eps2_t12"], 1.2 * _INF)
    sig_rst12 = _get(["MAT_SIGres_t12", "sig_rst12", "sigres_t12"], 0.0)
    wpmax_t12 = _get(["MAT_Wmax_pt12", "wpmax_t12"], _INF)

    # Delamination & strain rate smoothing
    gamma_ini = _get(["MAT_GAMAi", "gamma_ini", "gamai"], _INF)
    gamma_max = _get(["MAT_GAMAm", "gamma_max", "gamam"], 1.1 * _INF)
    d3max = _get(["MAT_DAMm", "d3max", "damm"], 1.0)
    fsmooth = _geti(["Fsmooth", "fsmooth"], 0)
    fcut = _get(["Fcut", "fcut"], _INF)

    # Kinematics & sound speed (read_mat25_tsaiwu.F90:282-288)
    nu21 = nu12 * e22 / max(e11, _EM20) if e11 > 0.0 else 0.0
    detc = 1.0 - nu12 * nu21
    if detc <= 0.0:
        detc = 1e-15

    c1 = max(e11, e22) / detc
    gmax = max(g12, g23, g31)
    c_sound = sound_speed_law25(e11, e22, nu12, g12, g23, g31, rho0)

    # Tsai-Wu yield coefficients (read_mat25_tsaiwu.F90:310-316)
    tw_coeffs = tsai_wu_coefficients(sig_1yt, sig_1yc, sig_2yt, sig_2yc, (sig_12yt, sig_12yc), alpha)
    f1 = tw_coeffs["F1"]
    f2 = tw_coeffs["F2"]
    f11 = tw_coeffs["F11"]
    f22 = tw_coeffs["F22"]
    f33 = tw_coeffs["F33"]
    f12 = tw_coeffs["F12"]

    params: Dict[str, Any] = {
        "E": max(e11, e22),
        "MAT_E": max(e11, e22),
        "nu": nu12 if 0.0 <= nu12 < 0.5 else 0.3,
        "MAT_NU": nu12 if 0.0 <= nu12 < 0.5 else 0.3,
        "G": g12,
        "MAT_G": g12,
        "rho": rho0,
        "MAT_RHO": rho0,
        "rho0": rho0,
        "rhor": rhor,
        "e11": e11,
        "e22": e22,
        "e33": e33,
        "E1": e11,
        "E2": e22,
        "E3": e33,
        "nu12": nu12,
        "nu21": nu21,
        "detc": detc,
        "C1": c1,
        "iform": iform,
        "iflag": iform,
        "g12": g12,
        "g23": g23,
        "g31": g31,
        "G12": g12,
        "G23": g23,
        "G31": g31,
        "eps_f1": eps_f1,
        "eps_f2": eps_f2,
        "epsf1": eps_f1,
        "epsf2": eps_f2,
        "eps_t1": eps_t1,
        "eps_m1": eps_m1,
        "epst1": eps_t1,
        "epsm1": eps_m1,
        "eps_t2": eps_t2,
        "eps_m2": eps_m2,
        "epst2": eps_t2,
        "epsm2": eps_m2,
        "dmax": dmax,
        "wpmax": wpmax,
        "wpref": wpref,
        "wplaref": wpref,
        "wplamx": wpmax,
        "ioff": ioff,
        "ratio": ratio,
        "b": b,
        "n": n,
        "fmax": fmax,
        "sig_1yt": sig_1yt,
        "sig_2yt": sig_2yt,
        "sig_1yc": sig_1yc,
        "sig_2yc": sig_2yc,
        "sigyt1": sig_1yt,
        "sigyc1": sig_1yc,
        "sigyt2": sig_2yt,
        "sigyc2": sig_2yc,
        "alpha": alpha,
        "sig_12yc": sig_12yc,
        "sig_12yt": sig_12yt,
        "sigc12": sig_12yc,
        "sigt12": sig_12yt,
        "sigyc12": sig_12yc,
        "sigyt12": sig_12yt,
        "c": c,
        "eps_rate_0": eps_rate_0,
        "epdr": eps_rate_0,
        "icc": icc,
        "strflag": icc,
        "iflawp": iflawp,
        "b_1t": b_1t,
        "b1_t": b_1t,
        "n_1t": n_1t,
        "n1_t": n_1t,
        "sig_1maxt": sig_1maxt,
        "sig1max_t": sig_1maxt,
        "c_1t": c_1t,
        "c1_t": c_1t,
        "eps_1t1": eps_1t1,
        "eps1_t1": eps_1t1,
        "eps_2t1": eps_2t1,
        "eps2_t1": eps_2t1,
        "sig_rst1": sig_rst1,
        "sigres_t1": sig_rst1,
        "wpmax_t1": wpmax_t1,
        "b_2t": b_2t,
        "b2_t": b_2t,
        "n_2t": n_2t,
        "n2_t": n_2t,
        "sig_2maxt": sig_2maxt,
        "sig2max_t": sig_2maxt,
        "c_2t": c_2t,
        "c2_t": c_2t,
        "eps_1t2": eps_1t2,
        "eps1_t2": eps_1t2,
        "eps_2t2": eps_2t2,
        "eps2_t2": eps_2t2,
        "sig_rst2": sig_rst2,
        "sigres_t2": sig_rst2,
        "wpmax_t2": wpmax_t2,
        "b_1c": b_1c,
        "b1_c": b_1c,
        "n_1c": n_1c,
        "n1_c": n_1c,
        "sig_1maxc": sig_1maxc,
        "sig1max_c": sig_1maxc,
        "c_1c": c_1c,
        "c1_c": c_1c,
        "eps_1c1": eps_1c1,
        "eps1_c1": eps_1c1,
        "eps_2c1": eps_2c1,
        "eps2_c1": eps_2c1,
        "sig_rsc1": sig_rsc1,
        "sigres_c1": sig_rsc1,
        "wpmax_c1": wpmax_c1,
        "b_2c": b_2c,
        "b2_c": b_2c,
        "n_2c": n_2c,
        "n2_c": n_2c,
        "sig_2maxc": sig_2maxc,
        "sig2max_c": sig_2maxc,
        "c_2c": c_2c,
        "c2_c": c_2c,
        "eps_1c2": eps_1c2,
        "eps1_c2": eps_1c2,
        "eps_2c2": eps_2c2,
        "eps2_c2": eps_2c2,
        "sig_rsc2": sig_rsc2,
        "sigres_c2": sig_rsc2,
        "wpmax_c2": wpmax_c2,
        "b_12t": b_12t,
        "b12_t": b_12t,
        "n_12t": n_12t,
        "n12_t": n_12t,
        "sig_12maxt": sig_12maxt,
        "sig12max_t": sig_12maxt,
        "c_12t": c_12t,
        "c12_t": c_12t,
        "eps_1t12": eps_1t12,
        "eps1_t12": eps_1t12,
        "eps_2t12": eps_2t12,
        "eps2_t12": eps_2t12,
        "sig_rst12": sig_rst12,
        "sigres_t12": sig_rst12,
        "wpmax_t12": wpmax_t12,
        "gamma_ini": gamma_ini,
        "gamai": gamma_ini,
        "gamma_max": gamma_max,
        "gamam": gamma_max,
        "d3max": d3max,
        "damm": d3max,
        "fsmooth": fsmooth,
        "fcut": fcut,
        "F1": f1,
        "F2": f2,
        "F11": f11,
        "F22": f22,
        "F33": f33,
        "F12": f12,
        "c_sound": c_sound,
        "ssp": c_sound,
    }

    mat = Material(
        id=_id,
        law=25,
        rho0=rho0,
        title=_title,
        law_name="LAW25",
        params=params,
    )
    mat.sound_speed_solid = lambda: c_sound
    mat.sound_speed_shell = lambda: c_sound
    return mat


build_comp_plas = build_law25
build_compsh = build_law25
build_tsai_wu = build_law25
build_crasurv = build_law25
build_composite_plas = build_law25


# ============================================================================
# Sound Speed & Extra Allocations
# ============================================================================

def sound_speed(mat: Any, rho: float | np.ndarray | None = None, extra: Any = None) -> float | np.ndarray:
    """Sound speed for LAW25 composite orthotropic material matching read_mat25_tsaiwu.F90:288."""
    params = getattr(mat, "params", {}) or {}
    rho0 = float(getattr(mat, "rho0", 0.0) or params.get("rho0", 0.0) or params.get("MAT_RHO", 1.0))
    r = rho if rho is not None else rho0
    c_stored = params.get("c_sound", params.get("ssp", None))
    if c_stored is not None and rho is None:
        return float(c_stored)

    e11 = float(params.get("e11", params.get("MAT_EA", 1.0)))
    e22 = float(params.get("e22", params.get("MAT_EB", 1.0)))
    nu12 = float(params.get("nu12", params.get("MAT_PRAB", 0.3)))
    g12 = float(params.get("g12", params.get("MAT_GAB", 0.0)))
    g23 = float(params.get("g23", params.get("MAT_GBC", 0.0)))
    g31 = float(params.get("g31", params.get("MAT_GCA", 0.0)))

    nu21 = nu12 * e22 / max(e11, _EM20) if e11 > 0.0 else 0.0
    detc = max(1.0e-15, 1.0 - nu12 * nu21)
    c1 = max(e11, e22) / detc
    gmax = max(g12, g23, g31)
    mod_max = max(c1, gmax)
    return np.sqrt(mod_max / np.maximum(r, _EM20))



def extra_shapes(mat: Any, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Per-element persistent state arrays for LAW25 (hm_read_mat25.F lines 121-136)."""
    p = getattr(mat, "params", {}) or {}
    iform = int(p.get("iform", p.get("MAT_Iflag", p.get("iflag", 0))))
    nmod = 6 if iform != 0 else 3
    ndmg = 1 + nmod
    if nip:
        return {
            "dmg25": (nip, ndmg),
            "crak25": (nip, 2),
            "stra25": (nip, 6),
            "tsaiwu25": (nip,),
            "epsd25": (nip,),
            "wpla25": (nip,),
            "off25": (nip,),
        }
    return {
        "dmg25": (ndmg,),
        "crak25": (2,),
        "stra25": (6,),
        "tsaiwu25": (),
        "epsd25": (),
        "wpla25": (),
        "off25": (),
    }


# ============================================================================
# Tangent Stiffness Formulations
# ============================================================================

def _extract_extra_state(extra: Any, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract damage (d1, d2) and failure status (off) arrays of shape (n,) from extra dict."""
    d1 = np.zeros(n, dtype=np.float64)
    d2 = np.zeros(n, dtype=np.float64)
    off = np.ones(n, dtype=np.float64)

    if extra is None or not isinstance(extra, dict):
        return d1, d2, off

    # 1. Failure status flag
    for off_key in ("off25", "off", "layfail"):
        if off_key in extra and extra[off_key] is not None:
            val = np.asarray(extra[off_key], dtype=np.float64).flatten()
            if len(val) == 1 and n > 1:
                off[:] = val[0]
            elif len(val) >= n:
                off[:] = val[:n]
            elif len(val) > 0:
                off[:len(val)] = val
            break

    # 2. Damage arrays
    for dmg_key in ("dmg25", "dmg"):
        if dmg_key in extra and extra[dmg_key] is not None:
            val = np.asarray(extra[dmg_key], dtype=np.float64)
            if val.ndim == 1:
                if len(val) >= 3 and n == 1:
                    d1[0] = val[1]
                    d2[0] = val[2]
                elif len(val) == 2 and n == 1:
                    d1[0] = val[0]
                    d2[0] = val[1]
                elif len(val) >= n:
                    d1[:] = val[:n]
            elif val.ndim == 2:
                m = min(n, val.shape[0])
                if val.shape[1] >= 3:
                    d1[:m] = val[:m, 1]
                    d2[:m] = val[:m, 2]
                elif val.shape[1] >= 2:
                    d1[:m] = val[:m, 0]
                    d2[:m] = val[:m, 1]
            break

    if "d1" in extra and extra["d1"] is not None:
        v1 = np.asarray(extra["d1"], dtype=np.float64).flatten()
        if len(v1) == 1 and n > 1:
            d1[:] = v1[0]
        elif len(v1) >= n:
            d1[:] = v1[:n]
        elif len(v1) > 0:
            d1[:len(v1)] = v1

    if "d2" in extra and extra["d2"] is not None:
        v2 = np.asarray(extra["d2"], dtype=np.float64).flatten()
        if len(v2) == 1 and n > 1:
            d2[:] = v2[0]
        elif len(v2) >= n:
            d2[:] = v2[:n]
        elif len(v2) > 0:
            d2[:len(v2)] = v2

    return d1, d2, off


def shell_membrane_tangent(mat: Any, extra: Any = None) -> np.ndarray:
    """(3, 3) orthotropic plane-stress elastic matrix for shells.

    Accounts for damaged moduli (d1, d2) if provided in extra.
    """
    params = getattr(mat, "params", {}) or {}
    e11 = float(params.get("e11", params.get("MAT_EA", 1.0)))
    e22 = float(params.get("e22", params.get("MAT_EB", 1.0)))
    nu12 = float(params.get("nu12", params.get("MAT_PRAB", 0.3)))
    g12 = float(params.get("g12", params.get("MAT_GAB", 0.5 * e11 / max(1.0 + nu12, 1e-15))))

    nu21 = nu12 * e22 / max(e11, _EM20) if e11 > 0.0 else 0.0

    d1, d2, off = _extract_extra_state(extra, 1)
    if off[0] <= 0.0:
        return np.zeros((3, 3), dtype=np.float64)

    de1 = max(_EM20, min(1.0, 1.0 - d1[0]))
    de2 = max(_EM20, min(1.0, 1.0 - d2[0]))
    scale1 = 1.0 if (de1 >= 1.0 - 1e-12 and de2 >= 1.0 - 1e-12) else 0.0
    scale2 = max(_EM20, 1.0 - nu12 * nu21 * scale1)

    c11 = e11 * de1 / scale2
    c22 = e22 * de2 / scale2
    c12 = nu21 * c11 * scale1
    g12_eff = de1 * de2 * g12

    return np.array([
        [c11, c12, 0.0],
        [c12, c22, 0.0],
        [0.0, 0.0, g12_eff],
    ], dtype=np.float64)


def consistent_shell_tangent(
    mat: Any,
    sig: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Any = None,
    extra: Any = None,
    dt: Any = None,
    deps: Optional[np.ndarray] = None,
    symmetric: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    """(n, 3, 3) consistent plane-stress tangent for shells.

    Differentiates the constitutive stress update d(sig) / d(deps).
    Accounts for:
      - Orthotropic elastic plane-stress matrix
      - Degraded moduli under tensile damage (d1, d2) and unilateral recovery under compression
      - Consistent algorithmic elastoplastic tangent under Tsai-Wu / CRASURV yielding
      - Element deletion / failure off state (zero stiffness)
      - Major symmetry enforcement if symmetric=True
    """
    if deps is None and "deps" in kwargs:
        deps = kwargs["deps"]
    if dt is not None and epsp_incr is None and not isinstance(dt, (float, int)):
        epsp_incr = dt

    sig_arr = np.asarray(sig, dtype=np.float64)
    is_1d = (sig_arr.ndim == 1)
    sig_2d = sig_arr[None, :] if is_1d else sig_arr
    n = sig_2d.shape[0]
    if n == 0:
        return np.empty((0, 3, 3), dtype=np.float64)

    params = getattr(mat, "params", {}) or {}
    e11 = float(params.get("e11", params.get("MAT_EA", 1.0)))
    e22 = float(params.get("e22", params.get("MAT_EB", 1.0)))
    nu12 = float(params.get("nu12", params.get("MAT_PRAB", 0.3)))
    g12 = float(params.get("g12", params.get("MAT_GAB", 0.5 * e11 / max(1.0 + nu12, 1e-15))))
    nu21 = nu12 * e22 / max(e11, _EM20) if e11 > 0.0 else 0.0

    f1 = float(params.get("F1", 0.0))
    f2 = float(params.get("F2", 0.0))
    f11 = float(params.get("F11", 0.0))
    f22 = float(params.get("F22", 0.0))
    f33 = float(params.get("F33", 0.0))
    f12 = float(params.get("F12", 0.0))
    b_val = float(params.get("b", params.get("MAT_BETA", 0.0)))
    n_val = float(params.get("n", params.get("MAT_HARD", 1.0)))
    fmax_val = float(params.get("fmax", params.get("MAT_SIG", _INF)))

    d1, d2, off = _extract_extra_state(extra, n)

    epsp_arr = None
    if epsp is not None:
        epsp_arr = np.asarray(epsp, dtype=np.float64).flatten()
    epsp_incr_arr = None
    if epsp_incr is not None:
        epsp_incr_arr = np.asarray(epsp_incr, dtype=np.float64).flatten()

    deps_2d = None
    if deps is not None:
        deps_arr = np.asarray(deps, dtype=np.float64)
        deps_2d = deps_arr[None, :] if deps_arr.ndim == 1 else deps_arr

    tangents = np.zeros((n, 3, 3), dtype=np.float64)

    for i in range(n):
        if off[i] <= 0.0:
            continue

        s1 = sig_2d[i, 0]
        s2 = sig_2d[i, 1]
        s12 = sig_2d[i, 2]

        de1 = 1.0 - d1[i] if s1 >= 0.0 else 1.0
        de2 = 1.0 - d2[i] if s2 >= 0.0 else 1.0
        de1 = max(_EM20, min(1.0, de1))
        de2 = max(_EM20, min(1.0, de2))

        scale1 = 1.0 if (de1 >= 1.0 - 1e-12 and de2 >= 1.0 - 1e-12) else 0.0
        scale2 = max(_EM20, 1.0 - nu12 * nu21 * scale1)

        a11 = e11 * de1 / scale2
        a22 = e22 * de2 / scale2
        a12 = nu21 * a11 * scale1
        g12_eff = de1 * de2 * g12

        c_el = np.array([
            [a11, a12, 0.0],
            [a12, a22, 0.0],
            [0.0, 0.0, g12_eff],
        ], dtype=np.float64)

        wp = float(epsp_arr[i]) if (epsp_arr is not None and i < len(epsp_arr)) else 0.0
        fyld = (1.0 + b_val * (wp ** n_val)) if wp > 0.0 else 1.0
        fyld = min(fmax_val, fyld)

        is_yielding = False
        beta = 1.0
        t_tr = np.array([s1, s2, s12], dtype=np.float64)

        if deps_2d is not None and i < len(deps_2d):
            t_tr = np.array([s1, s2, s12], dtype=np.float64) + c_el @ deps_2d[i]
            wvec_tr = f1 * t_tr[0] + f2 * t_tr[1] + f11 * (t_tr[0]**2) + f22 * (t_tr[1]**2) + f33 * (t_tr[2]**2) + 2.0 * f12 * t_tr[0] * t_tr[1]
            if wvec_tr > fyld:
                is_yielding = True
                coefa = f11 * (t_tr[0]**2) + f22 * (t_tr[1]**2) + f33 * (t_tr[2]**2) + 2.0 * f12 * t_tr[0] * t_tr[1]
                coefb = f1 * t_tr[0] + f2 * t_tr[1]
                delta = coefb**2 + 4.0 * coefa * fyld
                if delta >= 0.0 and coefa > _EM20:
                    beta = (-coefb + math.sqrt(delta)) / (2.0 * coefa)
                    beta = max(0.0, min(1.0, beta))
                else:
                    beta = 1.0 / math.sqrt(max(_EM20, wvec_tr / fyld))
        else:
            wvec = f1 * s1 + f2 * s2 + f11 * (s1**2) + f22 * (s2**2) + f33 * (s12**2) + 2.0 * f12 * s1 * s2
            if epsp_incr_arr is not None and i < len(epsp_incr_arr) and epsp_incr_arr[i] > 0.0:
                is_yielding = True
            elif wvec >= fyld - 1e-5:
                is_yielding = True

        if is_yielding:
            s_ret = beta * t_tr
            m_grad = np.array([
                f1 + 2.0 * f11 * s_ret[0] + 2.0 * f12 * s_ret[1],
                f2 + 2.0 * f22 * s_ret[1] + 2.0 * f12 * s_ret[0],
                2.0 * f33 * s_ret[2],
            ], dtype=np.float64)

            m_cel = m_grad @ c_el
            m_dot_t = float(m_grad @ t_tr)

            h_hard = 0.0
            if wp > 0.0 and b_val > 0.0:
                h_hard = (s_ret[0] * m_grad[0] + s_ret[1] * m_grad[1] + 2.0 * s_ret[2] * m_grad[2]) * n_val * b_val * (wp ** (n_val - 1.0))

            denom = m_dot_t + h_hard
            if denom > _EM20:
                c_ep = beta * (c_el - np.outer(t_tr, m_cel) / denom)
                if symmetric:
                    c_ep = 0.5 * (c_ep + c_ep.T)
                tangents[i] = c_ep
            else:
                tangents[i] = c_el if not symmetric else 0.5 * (c_el + c_el.T)
        else:
            tangents[i] = c_el if not symmetric else 0.5 * (c_el + c_el.T)

    return tangents[0] if is_1d else tangents


def solid_stiffness_matrix(mat: Any, extra: Any = None) -> np.ndarray:
    """(6, 6) 3D orthotropic elastic stiffness matrix.

    Accounts for damaged moduli (d1, d2) if provided in extra.
    """
    params = getattr(mat, "params", {}) or {}
    e11 = float(params.get("e11", params.get("MAT_EA", 1.0)))
    e22 = float(params.get("e22", params.get("MAT_EB", 1.0)))
    e33 = float(params.get("e33", params.get("MAT_EC", max(e11, e22))))
    nu12 = float(params.get("nu12", params.get("MAT_PRAB", 0.3)))
    g12 = float(params.get("g12", params.get("MAT_GAB", 0.5 * e11 / max(1.0 + nu12, 1e-15))))
    g23 = float(params.get("g23", params.get("MAT_GBC", g12)))
    g31 = float(params.get("g31", params.get("MAT_GCA", g12)))

    nu21 = nu12 * e22 / max(e11, _EM20) if e11 > 0.0 else 0.0

    d1, d2, off = _extract_extra_state(extra, 1)
    if off[0] <= 0.0:
        return np.zeros((6, 6), dtype=np.float64)

    de1 = max(_EM20, min(1.0, 1.0 - d1[0]))
    de2 = max(_EM20, min(1.0, 1.0 - d2[0]))
    scale1 = 1.0 if (de1 >= 1.0 - 1e-12 and de2 >= 1.0 - 1e-12) else 0.0
    scale2 = max(_EM20, 1.0 - nu12 * nu21 * scale1)

    a11 = e11 * de1 / scale2
    a22 = e22 * de2 / scale2
    a12 = nu21 * a11 * scale1
    g12_eff = de1 * de2 * g12
    g23_eff = de2 * g23
    g31_eff = de1 * g31

    c = np.zeros((6, 6), dtype=np.float64)
    c[0, 0] = a11
    c[0, 1] = a12
    c[1, 0] = a12
    c[1, 1] = a22
    c[2, 2] = e33
    c[3, 3] = g12_eff
    c[4, 4] = g23_eff
    c[5, 5] = g31_eff
    return c


def consistent_solid_tangent(
    mat: Any,
    sig: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Any = None,
    extra: Any = None,
    dt: Any = None,
    deps: Optional[np.ndarray] = None,
    symmetric: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    """(n, 6, 6) 3D solid consistent tangent.

    Differentiates the 3D solid constitutive update d(sig) / d(deps).
    Components: [11, 22, 33, 12, 23, 31].
    In-plane components [11, 22, 12] undergo Tsai-Wu yielding and damage;
    out-of-plane components [33, 23, 31] remain elastic with degraded shear moduli.
    """
    if deps is None and "deps" in kwargs:
        deps = kwargs["deps"]
    if dt is not None and epsp_incr is None and not isinstance(dt, (float, int)):
        epsp_incr = dt

    sig_arr = np.asarray(sig, dtype=np.float64)
    is_1d = (sig_arr.ndim == 1)
    sig_2d = sig_arr[None, :] if is_1d else sig_arr
    n = sig_2d.shape[0]
    if n == 0:
        return np.empty((0, 6, 6), dtype=np.float64)

    params = getattr(mat, "params", {}) or {}
    e11 = float(params.get("e11", params.get("MAT_EA", 1.0)))
    e22 = float(params.get("e22", params.get("MAT_EB", 1.0)))
    e33 = float(params.get("e33", params.get("MAT_EC", max(e11, e22))))
    nu12 = float(params.get("nu12", params.get("MAT_PRAB", 0.3)))
    g12 = float(params.get("g12", params.get("MAT_GAB", 0.5 * e11 / max(1.0 + nu12, 1e-15))))
    g23 = float(params.get("g23", params.get("MAT_GBC", g12)))
    g31 = float(params.get("g31", params.get("MAT_GCA", g12)))
    nu21 = nu12 * e22 / max(e11, _EM20) if e11 > 0.0 else 0.0

    f1 = float(params.get("F1", 0.0))
    f2 = float(params.get("F2", 0.0))
    f11 = float(params.get("F11", 0.0))
    f22 = float(params.get("F22", 0.0))
    f33 = float(params.get("F33", 0.0))
    f12 = float(params.get("F12", 0.0))
    b_val = float(params.get("b", params.get("MAT_BETA", 0.0)))
    n_val = float(params.get("n", params.get("MAT_HARD", 1.0)))
    fmax_val = float(params.get("fmax", params.get("MAT_SIG", _INF)))

    d1, d2, off = _extract_extra_state(extra, n)

    epsp_arr = None
    if epsp is not None:
        epsp_arr = np.asarray(epsp, dtype=np.float64).flatten()
    epsp_incr_arr = None
    if epsp_incr is not None:
        epsp_incr_arr = np.asarray(epsp_incr, dtype=np.float64).flatten()

    deps_2d = None
    if deps is not None:
        deps_arr = np.asarray(deps, dtype=np.float64)
        deps_2d = deps_arr[None, :] if deps_arr.ndim == 1 else deps_arr

    tangents = np.zeros((n, 6, 6), dtype=np.float64)

    for i in range(n):
        if off[i] <= 0.0:
            continue

        s1 = sig_2d[i, 0]
        s2 = sig_2d[i, 1]
        s3 = sig_2d[i, 2]
        s12 = sig_2d[i, 3]
        s23 = sig_2d[i, 4]
        s31 = sig_2d[i, 5]

        de1 = 1.0 - d1[i] if s1 >= 0.0 else 1.0
        de2 = 1.0 - d2[i] if s2 >= 0.0 else 1.0
        de1 = max(_EM20, min(1.0, de1))
        de2 = max(_EM20, min(1.0, de2))

        scale1 = 1.0 if (de1 >= 1.0 - 1e-12 and de2 >= 1.0 - 1e-12) else 0.0
        scale2 = max(_EM20, 1.0 - nu12 * nu21 * scale1)

        a11 = e11 * de1 / scale2
        a22 = e22 * de2 / scale2
        a12 = nu21 * a11 * scale1
        g12_eff = de1 * de2 * g12
        g23_eff = de2 * g23
        g31_eff = de1 * g31

        c_el = np.zeros((6, 6), dtype=np.float64)
        c_el[0, 0] = a11
        c_el[0, 1] = a12
        c_el[1, 0] = a12
        c_el[1, 1] = a22
        c_el[2, 2] = e33
        c_el[3, 3] = g12_eff
        c_el[4, 4] = g23_eff
        c_el[5, 5] = g31_eff

        c_el_in = np.array([
            [a11, a12, 0.0],
            [a12, a22, 0.0],
            [0.0, 0.0, g12_eff],
        ], dtype=np.float64)

        wp = float(epsp_arr[i]) if (epsp_arr is not None and i < len(epsp_arr)) else 0.0
        fyld = (1.0 + b_val * (wp ** n_val)) if wp > 0.0 else 1.0
        fyld = min(fmax_val, fyld)

        is_yielding = False
        beta = 1.0
        t_in = np.array([s1, s2, s12], dtype=np.float64)

        if deps_2d is not None and i < len(deps_2d):
            deps_in = np.array([deps_2d[i, 0], deps_2d[i, 1], deps_2d[i, 3]], dtype=np.float64)
            t_in = np.array([s1, s2, s12], dtype=np.float64) + c_el_in @ deps_in
            wvec_tr = f1 * t_in[0] + f2 * t_in[1] + f11 * (t_in[0]**2) + f22 * (t_in[1]**2) + f33 * (t_in[2]**2) + 2.0 * f12 * t_in[0] * t_in[1]
            if wvec_tr > fyld:
                is_yielding = True
                coefa = f11 * (t_in[0]**2) + f22 * (t_in[1]**2) + f33 * (t_in[2]**2) + 2.0 * f12 * t_in[0] * t_in[1]
                coefb = f1 * t_in[0] + f2 * t_in[1]
                delta = coefb**2 + 4.0 * coefa * fyld
                if delta >= 0.0 and coefa > _EM20:
                    beta = (-coefb + math.sqrt(delta)) / (2.0 * coefa)
                    beta = max(0.0, min(1.0, beta))
                else:
                    beta = 1.0 / math.sqrt(max(_EM20, wvec_tr / fyld))
        else:
            wvec = f1 * s1 + f2 * s2 + f11 * (s1**2) + f22 * (s2**2) + f33 * (s12**2) + 2.0 * f12 * s1 * s2
            if epsp_incr_arr is not None and i < len(epsp_incr_arr) and epsp_incr_arr[i] > 0.0:
                is_yielding = True
            elif wvec >= fyld - 1e-5:
                is_yielding = True

        if is_yielding:
            s_ret_in = beta * t_in
            m_in = np.array([
                f1 + 2.0 * f11 * s_ret_in[0] + 2.0 * f12 * s_ret_in[1],
                f2 + 2.0 * f22 * s_ret_in[1] + 2.0 * f12 * s_ret_in[0],
                2.0 * f33 * s_ret_in[2],
            ], dtype=np.float64)

            m_cel_in = m_in @ c_el_in
            m_dot_t = float(m_in @ t_in)

            h_hard = 0.0
            if wp > 0.0 and b_val > 0.0:
                h_hard = (s_ret_in[0] * m_in[0] + s_ret_in[1] * m_in[1] + 2.0 * s_ret_in[2] * m_in[2]) * n_val * b_val * (wp ** (n_val - 1.0))

            denom = m_dot_t + h_hard
            if denom > _EM20:
                c_ep_in = beta * (c_el_in - np.outer(t_in, m_cel_in) / denom)
                c_algo = c_el.copy()
                c_algo[0, [0, 1, 3]] = c_ep_in[0]
                c_algo[1, [0, 1, 3]] = c_ep_in[1]
                c_algo[3, [0, 1, 3]] = c_ep_in[2]
                if symmetric:
                    c_algo = 0.5 * (c_algo + c_algo.T)
                tangents[i] = c_algo
            else:
                tangents[i] = c_el if not symmetric else 0.5 * (c_el + c_el.T)
        else:
            tangents[i] = c_el if not symmetric else 0.5 * (c_el + c_el.T)

    return tangents[0] if is_1d else tangents



# ============================================================================
# Core Integration Kernel (mat25_tsaiwu_c.F90 / mat25_crasurv_c.F90)
# ============================================================================

def _update_point_law25(
    p: Dict[str, Any],
    sig_old: np.ndarray,
    deps: np.ndarray,
    epsp_val: float,
    dt: float,
    is_solid: bool,
    wpla_val: float,
    epst_val: np.ndarray,
    dmg_val: np.ndarray,
    off_val: float,
) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray, float]:
    """Single Gauss point constitutive update for LAW25."""
    if off_val <= 0.0:
        return np.zeros_like(sig_old), epsp_val, wpla_val, epst_val, dmg_val, 0.0

    iform = int(p.get("iform", p.get("MAT_Iflag", p.get("iflag", 0))))
    ioff = int(p.get("ioff", p.get("Itype", 0)))
    e11 = float(p.get("e11", p.get("MAT_EA", 1.0)))
    e22 = float(p.get("e22", p.get("MAT_EB", 1.0)))
    e33 = float(p.get("e33", p.get("MAT_EC", max(e11, e22))))
    nu12 = float(p.get("nu12", p.get("MAT_PRAB", 0.0)))
    nu21 = float(p.get("nu21", nu12 * e22 / max(e11, _EM20) if e11 > 0 else 0.0))
    g12 = float(p.get("g12", p.get("MAT_GAB", 0.0)))
    g23 = float(p.get("g23", p.get("MAT_GBC", 0.0)))
    g31 = float(p.get("g31", p.get("MAT_GCA", 0.0)))

    epst1 = float(p.get("epst1", p.get("MAT_EPST1", _INF)))
    epsm1 = float(p.get("epsm1", p.get("MAT_EPSM1", 1.1 * _INF)))
    epst2 = float(p.get("epst2", p.get("MAT_EPST2", _INF)))
    epsm2 = float(p.get("epsm2", p.get("MAT_EPSM2", 1.1 * _INF)))
    epsf1 = float(p.get("epsf1", p.get("MAT_EPSF1", _INF)))
    epsf2 = float(p.get("epsf2", p.get("MAT_EPSF2", _INF)))
    dmax = float(p.get("dmax", p.get("MAT_DAMAGE", 0.999)))
    wpmax = float(p.get("wpmax", p.get("WPMAX", _INF)))
    wplaref = float(p.get("wplaref", p.get("WPREF", 1.0)))
    c_rate = float(p.get("c", p.get("MAT_SRC", 0.0)))
    epdr = float(p.get("epdr", p.get("MAT_SRP", 1.0)))
    alpha = float(p.get("alpha", p.get("MAT_ALPHA", 1.0)))

    # Total strain tracking (m25law.F line 243)
    epst = np.copy(epst_val)
    n_d = min(len(epst), len(deps))
    epst[:n_d] += deps[:n_d]

    # Tensile damage accumulation (mat25_tsaiwu_c.F90 lines 384-402, m25law.F lines 263-293)
    dmg = np.copy(dmg_val)
    if epst[0] > epst1 and epsm1 > epst1:
        dam1 = (epst[0] - epst1) / (epsm1 - epst1)
        dam2 = dam1 * epsm1 / max(epst[0], _EM20)
        dmg[1] = min(max(dam2, dmg[1]), dmax)
    if epst[1] > epst2 and epsm2 > epst2:
        dam1 = (epst[1] - epst2) / (epsm2 - epst2)
        dam2 = dam1 * epsm2 / max(epst[1], _EM20)
        dmg[2] = min(max(dam2, dmg[2]), dmax)

    # Unilateral damage: damaged under tension, intact under compression
    # mat25_tsaiwu_c.F90 lines 406-412: de1 = 1 - max(0, sign(dmg(2), sig(1)))
    de1 = 1.0 - dmg[1] if sig_old[0] >= 0.0 else 1.0
    de2 = 1.0 - dmg[2] if sig_old[1] >= 0.0 else 1.0
    de1 = max(_EM20, min(1.0, de1))
    de2 = max(_EM20, min(1.0, de2))

    scale1 = 1.0 if (de1 >= 1.0 - 1e-12 and de2 >= 1.0 - 1e-12) else 0.0
    scale2 = max(_EM20, 1.0 - nu12 * nu21 * scale1)
    a11 = e11 * de1 / scale2
    a22 = e22 * de2 / scale2
    a12 = nu21 * a11 * scale1
    g12_eff = de1 * de2 * g12
    g23_eff = de2 * g23
    g31_eff = de1 * g31

    # Elastic trial increment
    if is_solid:
        t1 = sig_old[0] + a11 * deps[0] + a12 * deps[1]
        t2 = sig_old[1] + a12 * deps[0] + a22 * deps[1]
        t3 = sig_old[3] + g12_eff * deps[3]
        s3 = sig_old[2] + e33 * deps[2]
        s5 = sig_old[4] + g23_eff * deps[4]
        s6 = sig_old[5] + g31_eff * deps[5]
    else:
        t1 = sig_old[0] + a11 * deps[0] + a12 * deps[1]
        t2 = sig_old[1] + a12 * deps[0] + a22 * deps[1]
        t3 = sig_old[2] + g12_eff * deps[2]

    # Strain rate factor (mat25_tsaiwu_c.F90 lines 453-462)
    eps_dot = np.linalg.norm(deps) / max(dt, _EM20) if dt > 0.0 else 0.0
    epspfac = 1.0
    if c_rate > 0.0 and epdr > 0.0 and eps_dot > epdr:
        epspfac = 1.0 + c_rate * math.log(eps_dot / epdr)

    wpla = wpla_val
    so1, so2, so3 = (sig_old[0], sig_old[1], sig_old[3]) if is_solid else (sig_old[0], sig_old[1], sig_old[2])

    if iform == 0:
        # Standard Tsai-Wu formulation (mat25_tsaiwu_c.F90:463-541)
        f1 = float(p.get("F1", 0.0))
        f2 = float(p.get("F2", 0.0))
        f11 = float(p.get("F11", 0.0))
        f22 = float(p.get("F22", 0.0))
        f33 = float(p.get("F33", 0.0))
        f12 = float(p.get("F12", 0.0))
        b_val = float(p.get("b", p.get("MAT_BETA", 0.0)))
        n_val = float(p.get("n", p.get("MAT_HARD", 1.0)))
        fmax_val = float(p.get("fmax", p.get("MAT_SIG", _INF)))

        fyld = hardening_yield(wpla, b_val, n_val, epspfac, fmax_val)
        wvec = tsai_wu_yield_criterion(t1, t2, t3, f1, f2, f11, f22, f33, f12)

        if wvec > fyld:
            coefa = f11 * (t1 ** 2) + f22 * (t2 ** 2) + f33 * (t3 ** 2) + 2.0 * f12 * t1 * t2
            coefb = f1 * t1 + f2 * t2
            delta = coefb ** 2 + 4.0 * coefa * fyld
            if delta >= 0.0 and coefa > _EM20:
                beta = (-coefb + math.sqrt(delta)) / (2.0 * coefa)
                beta = max(0.0, min(1.0, beta))
            else:
                beta = 1.0 / math.sqrt(max(_EM20, wvec / fyld))

            so1 = beta * t1
            so2 = beta * t2
            so3 = beta * t3

            dp1, dp2, dp3 = tsai_wu_flow_normal(so1, so2, so3, f1, f2, f11, f22, f33, f12)

            ds1 = t1 - so1
            ds2 = t2 - so2
            ds3 = t3 - so3

            wvec_hard = epspfac * (wpla ** (n_val - 1.0)) if (wpla > 0.0 and fyld < fmax_val) else 0.0
            h_term = (so1 * dp1 + so2 * dp2 + 2.0 * so3 * dp3) * n_val * b_val * wvec_hard

            denom = (
                dp1 * (a11 * dp1 + a12 * dp2)
                + dp2 * (a12 * dp1 + a22 * dp2)
                + 2.0 * dp3 * g12_eff * dp3
                + h_term
            )

            t1_ret = so1
            t2_ret = so2
            t3_ret = so3

            dwpla = (
                0.5 * (
                    (t1 - t1_ret) * (t1 + t1_ret) / max(a11, _EM20)
                    + (t2 - t2_ret) * (t2 + t2_ret) / max(a22, _EM20)
                    + (t3 - t3_ret) * (t3 + t3_ret) / max(g12_eff, _EM20)
                )
                / wplaref
            )
            if denom > _EM20:
                lamda = max(0.0, (dp1 * ds1 + dp2 * ds2 + dp3 * ds3) / denom)
                dwpla_flow = 0.5 * (
                    dp1 * lamda * (t1 + so1)
                    + dp2 * lamda * (t2 + so2)
                    + 2.0 * dp3 * lamda * (t3 + so3)
                ) / wplaref
                dwpla = max(dwpla, dwpla_flow)

            wpla += max(0.0, dwpla)
            t1 = t1_ret
            t2 = t2_ret
            t3 = t3_ret


    else:
        # CRASURV formulation (mat25_crasurv_c.F90:538-755)
        sigy0_t1 = float(p.get("sig_1yt", p.get("sigyt1", _INF)))
        sigy0_t2 = float(p.get("sig_2yt", p.get("sigyt2", _INF)))
        sigy0_c1 = float(p.get("sig_1yc", p.get("sigyc1", _INF)))
        sigy0_c2 = float(p.get("sig_2yc", p.get("sigyc2", _INF)))
        sigy0_t12 = float(p.get("sig_12yt", p.get("sigt12", _INF)))

        b_1t = float(p.get("b_1t", p.get("b1_t", 0.0)))
        n_1t = float(p.get("n_1t", p.get("n1_t", 1.0)))
        b_2t = float(p.get("b_2t", p.get("b2_t", 0.0)))
        n_2t = float(p.get("n_2t", p.get("n2_t", 1.0)))
        b_1c = float(p.get("b_1c", p.get("b1_c", 0.0)))
        n_1c = float(p.get("n_1c", p.get("n1_c", 1.0)))
        b_2c = float(p.get("b_2c", p.get("b2_c", 0.0)))
        n_2c = float(p.get("n_2c", p.get("n2_c", 1.0)))
        b_12t = float(p.get("b_12t", p.get("b12_t", 0.0)))
        n_12t = float(p.get("n_12t", p.get("n12_t", 1.0)))

        sig1maxt = float(p.get("sig_1maxt", p.get("sig1max_t", _INF)))
        sig2maxt = float(p.get("sig_2maxt", p.get("sig2max_t", _INF)))
        sig1maxc = float(p.get("sig_1maxc", p.get("sig1max_c", _INF)))
        sig2maxc = float(p.get("sig_2maxc", p.get("sig2max_c", _INF)))
        sig12maxt = float(p.get("sig_12maxt", p.get("sig12max_t", _INF)))

        c_1t = float(p.get("c_1t", p.get("c1_t", 0.0)))
        c_2t = float(p.get("c_2t", p.get("c2_t", 0.0)))
        c_1c = float(p.get("c_1c", p.get("c1_c", 0.0)))
        c_2c = float(p.get("c_2c", p.get("c2_c", 0.0)))
        c_12t = float(p.get("c_12t", p.get("c12_t", 0.0)))

        eps1_t1 = float(p.get("eps_1t1", p.get("eps1_t1", _INF)))
        eps2_t1 = float(p.get("eps_2t1", p.get("eps2_t1", 1.2 * _INF)))
        sig_rst1 = float(p.get("sig_rst1", p.get("sigres_t1", 0.0)))

        eps1_t2 = float(p.get("eps_1t2", p.get("eps1_t2", _INF)))
        eps2_t2 = float(p.get("eps_2t2", p.get("eps2_t2", 1.2 * _INF)))
        sig_rst2 = float(p.get("sig_rst2", p.get("sigres_t2", 0.0)))

        eps1_c1 = float(p.get("eps_1c1", p.get("eps1_c1", _INF)))
        eps2_c1 = float(p.get("eps_2c1", p.get("eps2_c1", 1.2 * _INF)))
        sig_rsc1 = float(p.get("sig_rsc1", p.get("sigres_c1", 0.0)))

        eps1_c2 = float(p.get("eps_1c2", p.get("eps1_c2", _INF)))
        eps2_c2 = float(p.get("eps_2c2", p.get("eps2_c2", 1.2 * _INF)))
        sig_rsc2 = float(p.get("sig_rsc2", p.get("sigres_c2", 0.0)))

        eps1_t12 = float(p.get("eps_1t12", p.get("eps1_t12", _INF)))
        eps2_t12 = float(p.get("eps_2t12", p.get("eps2_t12", 1.2 * _INF)))
        sig_rst12 = float(p.get("sig_rst12", p.get("sigres_t12", 0.0)))

        # 1. Directional hardening
        if wpla > 0.0:
            s_yt1 = sigy0_t1 * (1.0 + b_1t * (wpla ** n_1t))
            s_yt2 = sigy0_t2 * (1.0 + b_2t * (wpla ** n_2t))
            s_yc1 = sigy0_c1 * (1.0 + b_1c * (wpla ** n_1c))
            s_yc2 = sigy0_c2 * (1.0 + b_2c * (wpla ** n_2c))
            s_yt12 = sigy0_t12 * (1.0 + b_12t * (wpla ** n_12t))
        else:
            s_yt1, s_yt2, s_yc1, s_yc2, s_yt12 = sigy0_t1, sigy0_t2, sigy0_c1, sigy0_c2, sigy0_t12

        # 2. Rate sensitivity & bounds
        s_yt1 = min(s_yt1, sig1maxt) * (1.0 + c_1t * (epspfac - 1.0))
        s_yt2 = min(s_yt2, sig2maxt) * (1.0 + c_2t * (epspfac - 1.0))
        s_yc1 = min(s_yc1, sig1maxc) * (1.0 + c_1c * (epspfac - 1.0))
        s_yc2 = min(s_yc2, sig2maxc) * (1.0 + c_2c * (epspfac - 1.0))
        s_yt12 = min(s_yt12, sig12maxt) * (1.0 + c_12t * (epspfac - 1.0))

        # 3. Softening from total/crack strains
        soft1 = 0.0
        if epst[0] >= eps1_t1 and eps2_t1 > eps1_t1:
            soft1 = min(1.0, (epst[0] - eps1_t1) / (eps2_t1 - eps1_t1))
            s_yt1 = min(s_yt1, (1.0 - soft1) * s_yt1 + soft1 * sig_rst1)
        elif epst[0] <= -eps1_c1 and eps2_c1 > eps1_c1:
            soft1 = min(1.0, (-epst[0] - eps1_c1) / (eps2_c1 - eps1_c1))
            s_yc1 = min(s_yc1, (1.0 - soft1) * s_yc1 + soft1 * sig_rsc1)

        soft2 = 0.0
        if epst[1] >= eps1_t2 and eps2_t2 > eps1_t2:
            soft2 = min(1.0, (epst[1] - eps1_t2) / (eps2_t2 - eps1_t2))
            s_yt2 = min(s_yt2, (1.0 - soft2) * s_yt2 + soft2 * sig_rst2)
        elif epst[1] <= -eps1_c2 and eps2_c2 > eps1_c2:
            soft2 = min(1.0, (-epst[1] - eps1_c2) / (eps2_c2 - eps1_c2))
            s_yc2 = min(s_yc2, (1.0 - soft2) * s_yc2 + soft2 * sig_rsc2)

        soft12 = 0.0
        gamma12 = epst[3] if is_solid else epst[2]
        if 0.5 * abs(gamma12) >= eps1_t12 and eps2_t12 > eps1_t12:
            soft12 = min(1.0, (0.5 * abs(gamma12) - eps1_t12) / (eps2_t12 - eps1_t12))
            s_yt12 = min(s_yt12, (1.0 - soft12) * s_yt12 + soft12 * sig_rst12)

        # Dynamic Tsai-Wu coefficients
        tw_cras = tsai_wu_coefficients(s_yt1, s_yc1, s_yt2, s_yc2, s_yt12, alpha)
        f1 = tw_cras["F1"]
        f2 = tw_cras["F2"]
        f11 = tw_cras["F11"]
        f22 = tw_cras["F22"]
        f33 = tw_cras["F33"]
        f12 = tw_cras["F12"]

        wvec = tsai_wu_yield_criterion(t1, t2, t3, f1, f2, f11, f22, f33, f12)

        if wvec > 1.0:
            coefa = f11 * (t1 ** 2) + f22 * (t2 ** 2) + f33 * (t3 ** 2) + 2.0 * f12 * t1 * t2
            coefb = f1 * t1 + f2 * t2
            delta = coefb ** 2 + 4.0 * coefa
            if delta >= 0.0 and coefa > _EM20:
                beta = (-coefb + math.sqrt(delta)) / (2.0 * coefa)
                beta = max(0.0, min(1.0, beta))
            else:
                beta = 1.0 / math.sqrt(max(_EM20, wvec))

            so1 = beta * t1
            so2 = beta * t2
            so3 = beta * t3

            dp1, dp2, dp3 = tsai_wu_flow_normal(so1, so2, so3, f1, f2, f11, f22, f33, f12)

            ds1 = t1 - so1
            ds2 = t2 - so2
            ds3 = t3 - so3

            denom = (
                dp1 * (a11 * dp1 + a12 * dp2)
                + dp2 * (a12 * dp1 + a22 * dp2)
                + 2.0 * dp3 * g12_eff * dp3
            )

            t1_ret = so1
            t2_ret = so2
            t3_ret = so3

            dwpla = (
                0.5 * (
                    (t1 - t1_ret) * (t1 + t1_ret) / max(a11, _EM20)
                    + (t2 - t2_ret) * (t2 + t2_ret) / max(a22, _EM20)
                    + (t3 - t3_ret) * (t3 + t3_ret) / max(g12_eff, _EM20)
                )
                / wplaref
            )
            if denom > _EM20:
                lamda = max(0.0, (dp1 * ds1 + dp2 * ds2 + dp3 * ds3) / denom)
                dwpla_flow = 0.5 * (
                    dp1 * lamda * (t1 + so1)
                    + dp2 * lamda * (t2 + so2)
                    + 2.0 * dp3 * lamda * (t3 + so3)
                ) / wplaref
                dwpla = max(dwpla, dwpla_flow)

            wpla += max(0.0, dwpla)
            t1 = t1_ret
            t2 = t2_ret
            t3 = t3_ret


    # Plastic work damage and global failure index (sigeps25c.F:288-302, m25law.F:308-323)
    if wpmax < 1e20 and wpmax > 0.0 and len(dmg) > 3:
        dmg[3] = min(1.0, wpla / wpmax)
    if len(dmg) > 0:
        cands = [float(dmg[1]), float(dmg[2])]
        if len(dmg) > 3:
            cands.append(float(dmg[3]))
        if len(dmg) > 6 and iform != 0:
            cands.extend([max(0.0, abs(float(dmg[4])) - 1.0), max(0.0, abs(float(dmg[5])) - 1.0), max(0.0, float(dmg[6]) - 1.0)])
        dmg[0] = max(cands)

    if is_solid:
        sig_new = np.array([t1, t2, s3, t3, s5, s6], dtype=float)
    else:
        sig_new = np.array([t1, t2, t3], dtype=float)

    # Element failure deletion criteria (m25law.F lines 336-350)
    fail1 = (epst[0] >= epsf1) or (dmg[1] >= dmax)
    fail2 = (epst[1] >= epsf2) or (dmg[2] >= dmax)
    fail_p = (wpla >= wpmax)
    shear_strain = epst[3] if is_solid else epst[2]
    fail_shear = (abs(shear_strain) >= epsf1)

    should_delete = False
    if ioff in (0, 1):
        should_delete = fail_p
    elif ioff == 2:
        should_delete = (fail1 or fail_p)
    elif ioff == 3:
        should_delete = (fail2 or fail_p)
    elif ioff == 4:
        should_delete = ((fail1 and fail2) or fail_p)
    elif ioff == 5:
        should_delete = (fail1 or fail2 or fail_p)
    elif ioff == 6:
        should_delete = (fail1 or fail2 or fail_shear or fail_p)

    off_new = off_val
    if should_delete:
        off_new = 0.0
        sig_new[:] = 0.0

    epsp_new = epsp_val + (wpla - wpla_val)
    return sig_new, epsp_new, wpla, epst, dmg, off_new


# ============================================================================
# Stress Update Routines (Vectorized & 1D)
# ============================================================================

def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, float | np.ndarray]:
    """Plane-stress (shell) constitutive update for LAW25."""
    p = getattr(mat, "params", {}) or {}
    is_1d = (sig.ndim == 1)
    s = np.atleast_2d(sig).copy()
    d = np.atleast_2d(deps).copy()
    n = s.shape[0]

    ep = np.zeros(n) if epsp is None else (np.atleast_1d(epsp).copy() if not np.isscalar(epsp) else np.full(n, float(epsp)))
    if extra is None:
        extra = {}

    wpla = extra.get("wpla25")
    if wpla is None or len(wpla) != n:
        wpla = extra.get("wpla", np.zeros(n))
    epst = extra.get("stra25")
    if epst is None or epst.shape[0] != n:
        epst = extra.get("epst", np.zeros((n, 3)))
    dmg = extra.get("dmg25")
    if dmg is None or dmg.shape[0] != n:
        dmg = extra.get("dmg", np.zeros((n, 8)))
    off = extra.get("off25")
    if off is None or len(off) != n:
        off = extra.get("off", np.ones(n))

    s_out = np.zeros_like(s)
    ep_out = np.zeros_like(ep)

    for i in range(n):
        s_i, ep_i, wpla_i, epst_i, dmg_i, off_i = _update_point_law25(
            p, s[i], d[i], ep[i], dt, False, wpla[i], epst[i], dmg[i], off[i]
        )
        s_out[i] = s_i
        ep_out[i] = ep_i
        wpla[i] = wpla_i
        epst[i] = epst_i
        dmg[i] = dmg_i
        off[i] = off_i

    extra["wpla25"] = wpla
    extra["stra25"] = epst
    extra["dmg25"] = dmg
    extra["off25"] = off
    extra["off"] = off
    if "layfail" in extra and extra["layfail"] is not None:
        extra["layfail"][:] = off

    c_val = sound_speed(mat)
    c_arr = np.full(n, float(c_val), dtype=float)

    if epsp is not None and isinstance(epsp, np.ndarray):
        epsp[:] = ep_out.reshape(epsp.shape)

    if is_1d:
        return s_out[0], ep_out[0], float(c_arr[0])
    return s_out, ep_out, c_arr


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, float | np.ndarray]:
    """3D solid constitutive update for LAW25."""
    p = getattr(mat, "params", {}) or {}
    is_1d = (sig.ndim == 1)
    s = np.atleast_2d(sig).copy()
    d = np.atleast_2d(deps).copy()
    n = s.shape[0]

    ep = np.zeros(n) if epsp is None else (np.atleast_1d(epsp).copy() if not np.isscalar(epsp) else np.full(n, float(epsp)))
    if extra is None:
        extra = {}

    wpla = extra.get("wpla25")
    if wpla is None or len(wpla) != n:
        wpla = extra.get("wpla", np.zeros(n))
    epst = extra.get("stra25")
    if epst is None or epst.shape[0] != n:
        epst = extra.get("epst", np.zeros((n, 6)))
    dmg = extra.get("dmg25")
    if dmg is None or dmg.shape[0] != n:
        dmg = extra.get("dmg", np.zeros((n, 8)))
    off = extra.get("off25")
    if off is None or len(off) != n:
        off = extra.get("off", np.ones(n))

    s_out = np.zeros_like(s)
    ep_out = np.zeros_like(ep)

    for i in range(n):
        s_i, ep_i, wpla_i, epst_i, dmg_i, off_i = _update_point_law25(
            p, s[i], d[i], ep[i], dt, True, wpla[i], epst[i], dmg[i], off[i]
        )
        s_out[i] = s_i
        ep_out[i] = ep_i
        wpla[i] = wpla_i
        epst[i] = epst_i
        dmg[i] = dmg_i
        off[i] = off_i

    extra["wpla25"] = wpla
    extra["stra25"] = epst
    extra["dmg25"] = dmg
    extra["off25"] = off
    extra["off"] = off

    c_val = sound_speed(mat)
    c_arr = np.full(n, float(c_val), dtype=float)

    if epsp is not None and isinstance(epsp, np.ndarray):
        epsp[:] = ep_out.reshape(epsp.shape)

    if is_1d:
        return s_out[0], ep_out[0], float(c_arr[0])
    return s_out, ep_out, c_arr


# ============================================================================
# Registry Hook
# ============================================================================

def _register() -> None:
    """Register LAW25 in pyradioss MAT_PHYSICS_REGISTRY."""
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for key in (25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV",
                    "COMPOSITE_PLAS", "MAT_LAW25", "MAT_COMP_PLAS", "MAT_COMPSH",
                    "MAT_TSAI_WU", "MAT_CRASURV", "MAT_COMPOSITE_PLAS"):
            MAT_PHYSICS_REGISTRY[key] = build_law25
    except Exception:
        pass


_register()
