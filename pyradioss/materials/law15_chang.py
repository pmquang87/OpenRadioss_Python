"""
LAW15 — Chang-Chang Composite Model with Tsai-Wu Plasticity
(/MAT/LAW15, /MAT/CHANG, /MAT/PLAS_ANISO, /MAT/COMP_CHANG).

Fortran origin:
  - starter/source/materials/mat/mat015/hm_read_mat15.F
  - engine/source/materials/mat/mat015/sigeps15c.F
  - engine/source/materials/mat/mat015/m15cplrc.F
  - engine/source/materials/mat/mat015/m15crak.F
  - hm_cfg_files/config/CFG/radioss110/MAT/matl15_chang.cfg

Constitutive model features:
  1. Orthotropic linear elasticity in shell plane stress: E1, E2, nu12, G12, G23, G31
  2. Tsai-Wu anisotropic yield surface W(sig) with isotropic power hardening (b, n)
     and Cowper-Symonds / logarithmic strain rate enhancement (c, epdr)
  3. Chang-Chang failure criteria for fiber breakage and matrix cracking:
     - Fiber failure (tensile/compressive)
     - Matrix cracking (tensile/compressive)
     - Post-failure exponential stress relaxation with relaxation time Tmax
  4. Total element / layer deletion (itype / ioff flag)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_EM30 = 1.0e-30
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

    Fortran source: starter/source/materials/mat/mat015/hm_read_mat15.F:327-332

    Formulas:
      F1  = 1/sigyt1 - 1/sigyc1
      F2  = 1/sigyt2 - 1/sigyc2
      F11 = 1 / (sigyt1 * sigyc1)
      F22 = 1 / (sigyt2 * sigyc2)
      F33 = 1 / (sigyt12 * sigyc12)  [or 1 / (sig12^2)]
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
    """Evaluate Tsai-Wu yield function W_vec.

    Fortran source: engine/source/materials/mat/mat015/m15cplrc.F:200-202

    W_vec = F1*s1 + F2*s2 + F11*s1^2 + F22*s2^2 + F33*s12^2 + 2*F12*s1*s2
    """
    return (
        F1 * s1
        + F2 * s2
        + F11 * (s1 ** 2)
        + F22 * (s2 ** 2)
        + F33 * (s12 ** 2)
        + 2.0 * F12 * s1 * s2
    )


def hardening_yield(
    wpla: float,
    b: float,
    n: float,
    epspfac: float = 1.0,
    fmax: float = _INF,
) -> float:
    """Hardening yield limit f_yld.

    Fortran source: engine/source/materials/mat/mat015/m15cplrc.F:187, 194

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
) -> float:
    """Strain rate enhancement factor.

    Fortran source: engine/source/materials/mat/mat015/m15cplrc.F:177-186

    If eps_dot > epdr:
      epspfac = 1 + c * log(eps_dot / epdr)
    Else:
      epspfac = 1.0
    """
    ed = float(eps_dot)
    c_val = float(c)
    epdr_val = float(epdr)

    if ed <= 0.0 or c_val <= 0.0:
        return 1.0

    if epdr_val > 0.0 and ed > epdr_val:
        return 1.0 + c_val * math.log(ed / epdr_val)
    return 1.0


def sound_speed_law15(
    e11: float,
    e22: float,
    nu12: float,
    g12: float,
    g23: float,
    g31: float,
    rho0: float,
) -> float:
    """Sound speed for LAW15 matching Fortran hm_read_mat15.F:266-269.

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

def build_law15(rec: Any = None, **kwargs: Any) -> Material:
    """Physics constructor for the cfg-parsed /MAT/LAW15 (CHANG) composite record.

    Follows starter/source/materials/mat/mat015/hm_read_mat15.F and
    hm_cfg_files/config/CFG/radioss110/MAT/matl15_chang.cfg.
    """
    if rec is None:
        p: Dict[str, Any] = dict(kwargs)
        _id = int(kwargs.get("id", 1))
        rho0_in = kwargs.get("rho0", kwargs.get("density", kwargs.get("MAT_RHO", 0.0)))
        _title = str(kwargs.get("title", "LAW15_CHANG"))
    elif isinstance(rec, dict):
        base_params = rec.get("params", rec)
        p = {**base_params, **kwargs}
        _id = int(rec.get("id", kwargs.get("id", 1)))
        rho0_in = rec.get("rho0", rec.get("density", kwargs.get("rho0", kwargs.get("density", 0.0))))
        _title = str(rec.get("title", kwargs.get("title", "LAW15_CHANG")))
    elif hasattr(rec, "params"):
        p = {**rec.params, **kwargs}
        _id = int(getattr(rec, "id", kwargs.get("id", 1)))
        rho0_in = getattr(rec, "rho0", getattr(rec, "density", kwargs.get("rho0", 0.0)))
        _title = str(getattr(rec, "title", kwargs.get("title", "LAW15_CHANG")))
    elif hasattr(rec, "__dataclass_fields__"):
        base_dict = {k: getattr(rec, k) for k in rec.__dataclass_fields__ if hasattr(rec, k)}
        p = {**base_dict, **kwargs}
        _id = int(getattr(rec, "id", kwargs.get("id", 1)))
        rho0_in = getattr(rec, "rho0", kwargs.get("rho0", 0.0))
        _title = str(getattr(rec, "title", kwargs.get("title", "LAW15_CHANG")))
    else:
        p = dict(kwargs)
        _id = int(kwargs.get("id", 1))
        rho0_in = kwargs.get("rho0", 0.0)
        _title = str(kwargs.get("title", "LAW15_CHANG"))

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

    # Initial density
    rho0 = _get(["MAT_RHO", "Refer_Rho", "rho0", "density", "rho"], float(rho0_in))
    rhor = _get(["Refer_Rho", "rhor", "rho_ref"], 0.0)
    if rhor <= 0.0:
        rhor = rho0

    # Elasticity
    e11 = _get(["MAT_EA", "e11", "E11", "E1", "e1"], 0.0)
    e22 = _get(["MAT_EB", "e22", "E22", "E2", "e2"], 0.0)
    nu12 = _get(["MAT_PRAB", "nu12", "NU12", "prab", "nu"], 0.0)
    g12 = _get(["MAT_GAB", "g12", "G12"], 0.0)
    g23 = _get(["MAT_GBC", "g23", "G23"], 0.0)
    g31 = _get(["MAT_GCA", "g31", "G31"], 0.0)

    # Hardening / plasticity
    # MAT_BETA is hardening b; MAT_Beta is shear scaling factor beta_s in failure
    b = _get(["MAT_BETA", "b", "B", "hardening_b"], 0.0)
    n = _get(["MAT_HARD", "hard", "n", "N"], 1.0)
    if n <= 0.0:
        n = 1.0
    fmax = _get(["MAT_SIG", "sig", "fmax", "FMAX"], _INF)
    if fmax <= 0.0:
        fmax = _INF

    wpmax = _get(["WPMAX", "wpmax", "wplamx"], _INF)
    if wpmax <= 0.0:
        wpmax = _INF
    wpref = _get(["WPREF", "wpref", "wplaref"], 1.0)
    if wpref <= 0.0:
        wpref = 1.0
    # Fortran: WPLAMX = WPLAMX / WPLAREF
    wpmax_norm = wpmax / wpref

    itype = _geti(["Itype", "itype", "IOFF", "ioff"], 0)

    # Tsai-Wu yield strengths
    sigyt1 = _get(["MAT_SIGYT1", "sigyt1", "sig_1yt"], 0.0)
    sigyt2 = _get(["MAT_SIGYT2", "sigyt2", "sig_2yt"], 0.0)
    sigyc1 = _get(["MAT_SIGYC1", "sigyc1", "sig_1yc"], 0.0)
    sigyc2 = _get(["MAT_SIGYC2", "sigyc2", "sig_2yc"], 0.0)
    alpha = _get(["MAT_ALPHA", "alpha", "ALPHA"], 1.0)
    if alpha <= 0.0:
        alpha = 1.0

    sigt12 = _get(["MAT_SIGT12", "sigt12", "sigyt12", "sig_t12"], 0.0)
    sigc12 = _get(["MAT_SIGC12", "sigc12", "sigyc12", "sig_c12"], 0.0)

    # Defaults and fallbacks for shear yield:
    if sigc12 <= 0.0 and sigt12 > 0.0:
        sigc12 = sigt12
    elif sigt12 <= 0.0 and sigc12 > 0.0:
        sigt12 = sigc12

    # Defaults and fallbacks for compressive yield:
    if sigyc1 <= 0.0 and sigyt1 > 0.0:
        sigyc1 = sigyt1
    if sigyc2 <= 0.0 and sigyt2 > 0.0:
        sigyc2 = sigyt2

    # Fallback to _INF if still zero:
    if sigyt1 <= 0.0:
        sigyt1 = _INF
    if sigyc1 <= 0.0:
        sigyc1 = _INF
    if sigyt2 <= 0.0:
        sigyt2 = _INF
    if sigyc2 <= 0.0:
        sigyc2 = _INF
    if sigt12 <= 0.0:
        sigt12 = _INF
    if sigc12 <= 0.0:
        sigc12 = _INF

    # Strain rate parameters
    src = _get(["MAT_SRC", "src", "c", "C"], 0.0)
    srp = _get(["MAT_SRP", "srp", "epdr", "eps0", "eps_rate_0"], 1.0)
    if srp <= 0.0:
        srp = 1.0
    strflag = _geti(["STRFLAG", "strflag", "icc", "ICC"], 1)
    if strflag == 0:
        strflag = 1

    # Chang-Chang failure parameters
    # Note: MAT_Beta is shear scaling factor in failure
    beta_s = _get(["MAT_Beta", "beta_s", "beta_failure", "mchang_beta"], 1.0)
    tmax = _get(["MAT_TMAX", "tmax", "TMAX"], _INF)
    if tmax <= 0.0:
        tmax = _INF

    s1 = _get(["MCHANG_S1", "s1", "S1"], 0.0)
    s2 = _get(["MCHANG_S2", "s2", "S2"], 0.0)
    s12 = _get(["MCHANG_S12", "s12", "S12"], 0.0)
    c1 = _get(["MCHANG_C1", "c1", "C1", "c11"], 0.0)
    c2 = _get(["MCHANG_C2", "c2", "C2", "c22"], 0.0)

    # Defaults and fallbacks for Chang-Chang strengths:
    if s1 <= 0.0:
        s1 = sigyt1
    if s2 <= 0.0:
        s2 = sigyt2
    if c1 <= 0.0:
        c1 = sigyc1
    if c2 <= 0.0:
        c2 = sigyc2
    if s12 <= 0.0:
        s12 = sigt12

    # Smoothing & cutoff frequency
    fsmooth = _geti(["Fsmooth", "fsmooth", "israte"], 0)
    fcut = _get(["Fcut", "fcut"], _INF)

    # Kinematics & sound speed (hm_read_mat15.F:209-210, 266-269)
    nu21 = nu12 * e22 / max(e11, _EM20) if e11 > 0.0 else 0.0
    detc = 1.0 - nu12 * nu21
    if detc <= 0.0:
        detc = 1.0e-15

    c1_mod = max(e11, e22) / detc
    gmax = max(g12, g23, g31)
    c_sound = sound_speed_law15(e11, e22, nu12, g12, g23, g31, rho0)

    # Tsai-Wu yield coefficients (hm_read_mat15.F:327-332)
    tw_coeffs = tsai_wu_coefficients(sigyt1, sigyc1, sigyt2, sigyc2, (sigt12, sigc12), alpha)
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
        "rho0": rho0,
        "density": rho0,
        "MAT_RHO": rho0,
        "rhor": rhor,
        "E1": e11,
        "E2": e22,
        "e11": e11,
        "e22": e22,
        "MAT_EA": e11,
        "MAT_EB": e22,
        "nu12": nu12,
        "nu21": nu21,
        "MAT_PRAB": nu12,
        "G12": g12,
        "G23": g23,
        "G31": g31,
        "g12": g12,
        "g23": g23,
        "g31": g31,
        "MAT_GAB": g12,
        "MAT_GBC": g23,
        "MAT_GCA": g31,
        "detc": detc,
        "C1": c1_mod,
        "Gmax": gmax,
        "c_sound": c_sound,
        "ssp": c_sound,
        "b": b,
        "MAT_BETA": b,
        "hard": n,
        "n": n,
        "MAT_HARD": n,
        "sig": fmax,
        "fmax": fmax,
        "MAT_SIG": fmax,
        "wpmax": wpmax_norm,
        "WPMAX": wpmax_norm,
        "wpref": wpref,
        "WPREF": wpref,
        "itype": itype,
        "Itype": itype,
        "ioff": itype,
        "IOFF": itype,
        "sigyt1": sigyt1,
        "sigyt2": sigyt2,
        "sigyc1": sigyc1,
        "sigyc2": sigyc2,
        "MAT_SIGYT1": sigyt1,
        "MAT_SIGYT2": sigyt2,
        "MAT_SIGYC1": sigyc1,
        "MAT_SIGYC2": sigyc2,
        "alpha": alpha,
        "MAT_ALPHA": alpha,
        "sigt12": sigt12,
        "sigc12": sigc12,
        "MAT_SIGT12": sigt12,
        "MAT_SIGC12": sigc12,
        "src": src,
        "c": src,
        "MAT_SRC": src,
        "srp": srp,
        "epdr": srp,
        "MAT_SRP": srp,
        "strflag": strflag,
        "icc": strflag,
        "STRFLAG": strflag,
        "beta_s": beta_s,
        "MAT_Beta": beta_s,
        "tmax": tmax,
        "MAT_TMAX": tmax,
        "s1": s1,
        "s2": s2,
        "s12": s12,
        "c1": c1,
        "c2": c2,
        "MCHANG_S1": s1,
        "MCHANG_S2": s2,
        "MCHANG_S12": s12,
        "MCHANG_C1": c1,
        "MCHANG_C2": c2,
        "fsmooth": fsmooth,
        "Fsmooth": fsmooth,
        "fcut": fcut,
        "Fcut": fcut,
        "F1": f1,
        "F2": f2,
        "F11": f11,
        "F22": f22,
        "F33": f33,
        "F12": f12,
    }

    mat = Material(
        id=_id,
        law=15,
        rho0=rho0,
        title=_title,
        law_name="LAW15",
        params=params,
    )
    return mat


build_chang = build_law15
build_plas_aniso = build_law15
build_comp_chang = build_law15


# ============================================================================
# Sound Speed & Extra Allocations
# ============================================================================

def sound_speed(mat: Any, rho: float | np.ndarray | None = None, extra: Any = None) -> float | np.ndarray:
    """Sound speed for LAW15 composite orthotropic material matching hm_read_mat15.F:266-269.

    c = sqrt(max(C1, G12, G23, G31) / rho0)
    where C1 = max(E1, E2) / (1 - nu12 * nu21)
    """
    params = getattr(mat, "params", {}) or {}
    rho0 = float(getattr(mat, "rho0", 0.0) or params.get("rho0", 0.0) or params.get("MAT_RHO", 1.0))
    r = rho if rho is not None else rho0
    c_stored = params.get("c_sound", params.get("ssp", None))
    if c_stored is not None and rho is None:
        return float(c_stored)

    e11 = float(params.get("e11", params.get("E1", params.get("MAT_EA", 1.0))))
    e22 = float(params.get("e22", params.get("E2", params.get("MAT_EB", 1.0))))
    nu12 = float(params.get("nu12", params.get("MAT_PRAB", 0.3)))
    g12 = float(params.get("g12", params.get("G12", params.get("MAT_GAB", 0.0))))
    g23 = float(params.get("g23", params.get("G23", params.get("MAT_GBC", 0.0))))
    g31 = float(params.get("g31", params.get("G31", params.get("MAT_GCA", 0.0))))

    nu21 = nu12 * e22 / max(e11, _EM20) if e11 > 0.0 else 0.0
    detc = max(1.0e-15, 1.0 - nu12 * nu21)
    c1 = max(e11, e22) / detc
    gmax = max(g12, g23, g31)
    mod_max = max(c1, gmax)
    return np.sqrt(mod_max / np.maximum(r, _EM20))


def extra_shapes(mat: Any, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Per-element persistent state arrays for LAW15 (hm_read_mat15.F lines 95-97, 386-389).

    State variables:
      damt15: (nip, 2) or (2,)  damage factors [damt_fiber, damt_matrix]
      sigr15: (nip, 6) or (6,)  stored stresses & failure time [s1, s2, s12, s23, s31, tfail]
      wpla15: (nip,) or ()      accumulated plastic work
      off15:  (nip,) or ()      active status flag (1.0 active, 0.0 failed)
    """
    if nip:
        return {
            "damt15": (nip, 2),
            "sigr15": (nip, 6),
            "wpla15": (nip,),
            "off15": (nip,),
        }
    return {
        "damt15": (2,),
        "sigr15": (6,),
        "wpla15": (),
        "off15": (),
    }


# ============================================================================
# Tangent Stiffness Formulations
# ============================================================================

def shell_membrane_tangent(mat: Any, extra: Any = None) -> np.ndarray:
    """(3, 3) orthotropic plane-stress elastic matrix for shells.

    Accounts for damage state in extra if provided.
    """
    params = getattr(mat, "params", {}) or {}
    e11 = float(params.get("e11", params.get("E1", params.get("MAT_EA", 1.0))))
    e22 = float(params.get("e22", params.get("E2", params.get("MAT_EB", 1.0))))
    nu12 = float(params.get("nu12", params.get("MAT_PRAB", 0.3)))
    g12 = float(params.get("g12", params.get("G12", params.get("MAT_GAB", 0.5 * e11 / max(1.0 + nu12, 1e-15)))))

    nu21 = nu12 * e22 / max(e11, _EM20) if e11 > 0.0 else 0.0

    off_val = 1.0
    damt = np.array([1.0, 1.0], dtype=np.float64)

    if extra is not None and isinstance(extra, dict):
        for off_key in ("off15", "off", "layfail"):
            if off_key in extra and extra[off_key] is not None:
                val = np.asarray(extra[off_key], dtype=np.float64).flatten()
                if len(val) > 0:
                    off_val = float(val[0])
                break
        for dam_key in ("damt15", "damt"):
            if dam_key in extra and extra[dam_key] is not None:
                val = np.asarray(extra[dam_key], dtype=np.float64)
                if val.ndim == 1 and len(val) >= 2:
                    damt[:] = val[:2]
                elif val.ndim == 2 and val.shape[1] >= 2:
                    damt[:] = val[0, :2]
                break

    if off_val <= 0.0:
        return np.zeros((3, 3), dtype=np.float64)

    # Fiber failure degrades all stiffness
    if damt[0] < 1.0:
        scale_f = max(0.0, float(damt[0]))
        detc = max(1e-15, 1.0 - nu12 * nu21)
        c11 = (e11 / detc) * scale_f
        c22 = (e22 / detc) * scale_f
        c12 = (nu21 * e11 / detc) * scale_f
        g12_eff = g12 * scale_f
        return np.array([
            [c11, c12, 0.0],
            [c12, c22, 0.0],
            [0.0, 0.0, g12_eff],
        ], dtype=np.float64)

    # Matrix failure degrades transverse stiffness and shear
    if damt[1] < 1.0:
        scale_m = max(0.0, float(damt[1]))
        e22_eff = max(_EM20, e22 * scale_m)
        g12_eff = g12 * scale_m
        return np.array([
            [e11, 0.0, 0.0],
            [0.0, e22_eff, 0.0],
            [0.0, 0.0, g12_eff],
        ], dtype=np.float64)

    detc = max(1e-15, 1.0 - nu12 * nu21)
    c11 = e11 / detc
    c22 = e22 / detc
    c12 = nu21 * c11

    return np.array([
        [c11, c12, 0.0],
        [c12, c22, 0.0],
        [0.0, 0.0, g12],
    ], dtype=np.float64)


def consistent_shell_tangent(
    mat: Any,
    sig: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: Any = 0.0,
    extra: Any = None,
    deps: Optional[np.ndarray] = None,
    symmetric: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    """(n, 3, 3) consistent plane-stress algorithmic tangent tensor for shells.

    Differentiates the constitutive stress update d(sig) / d(deps).
    Accounts for:
      - Orthotropic elastic plane-stress matrix
      - Degraded moduli under matrix cracking (damt[1]) and fiber breakage (damt[0])
      - Consistent algorithmic elastoplastic tangent under Tsai-Wu yielding
      - Element deletion / failure off state (zero stiffness)
      - Major symmetry enforcement if symmetric=True
    """
    if deps is None and "deps" in kwargs:
        deps = kwargs["deps"]

    sig_arr = np.asarray(sig, dtype=np.float64)
    is_1d = (sig_arr.ndim == 1)
    sig_2d = sig_arr[None, :] if is_1d else sig_arr
    n = sig_2d.shape[0]
    if n == 0:
        return np.empty((0, 3, 3), dtype=np.float64)

    params = getattr(mat, "params", {}) or {}
    e11 = float(params.get("e11", params.get("E1", params.get("MAT_EA", 1.0))))
    e22 = float(params.get("e22", params.get("E2", params.get("MAT_EB", 1.0))))
    nu12 = float(params.get("nu12", params.get("MAT_PRAB", 0.3)))
    g12 = float(params.get("g12", params.get("G12", params.get("MAT_GAB", 0.5 * e11 / max(1.0 + nu12, 1e-15)))))
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

    # Extract state variables
    damt = np.ones((n, 2), dtype=np.float64)
    off = np.ones(n, dtype=np.float64)
    wpla = np.zeros(n, dtype=np.float64)

    if extra is not None and isinstance(extra, dict):
        for dam_key in ("damt15", "damt"):
            if dam_key in extra and extra[dam_key] is not None:
                val = np.asarray(extra[dam_key], dtype=np.float64)
                if val.ndim == 1 and len(val) >= 2:
                    damt[:, 0] = val[0]
                    damt[:, 1] = val[1]
                elif val.ndim == 2:
                    m = min(n, val.shape[0])
                    damt[:m, :2] = val[:m, :2]
                break

        for off_key in ("off15", "off", "layfail"):
            if off_key in extra and extra[off_key] is not None:
                val = np.asarray(extra[off_key], dtype=np.float64).flatten()
                if len(val) == 1 and n > 1:
                    off[:] = val[0]
                elif len(val) >= n:
                    off[:] = val[:n]
                elif len(val) > 0:
                    off[:len(val)] = val
                break

        for wp_key in ("wpla15", "wpla"):
            if wp_key in extra and extra[wp_key] is not None:
                val = np.asarray(extra[wp_key], dtype=np.float64).flatten()
                if len(val) == 1 and n > 1:
                    wpla[:] = val[0]
                elif len(val) >= n:
                    wpla[:] = val[:n]
                elif len(val) > 0:
                    wpla[:len(val)] = val
                break

    tangents = np.zeros((n, 3, 3), dtype=np.float64)

    for i in range(n):
        if off[i] <= 0.0:
            continue

        s1 = sig_2d[i, 0]
        s2 = sig_2d[i, 1]
        s12 = sig_2d[i, 2]

        dam_f = damt[i, 0]
        dam_m = damt[i, 1]

        # Fiber failure
        if dam_f < 1.0:
            scale_f = max(0.0, dam_f)
            detc = max(1e-15, 1.0 - nu12 * nu21)
            c11 = (e11 / detc) * scale_f
            c22 = (e22 / detc) * scale_f
            c12 = (nu21 * e11 / detc) * scale_f
            g12_eff = g12 * scale_f
            c_el = np.array([
                [c11, c12, 0.0],
                [c12, c22, 0.0],
                [0.0, 0.0, g12_eff],
            ], dtype=np.float64)
            tangents[i] = c_el
            continue

        # Matrix failure
        if dam_m < 1.0:
            scale_m = max(0.0, dam_m)
            e22_eff = max(_EM20, e22 * scale_m)
            g12_eff = g12 * scale_m
            c_el = np.array([
                [e11, 0.0, 0.0],
                [0.0, e22_eff, 0.0],
                [0.0, 0.0, g12_eff],
            ], dtype=np.float64)
        else:
            detc = max(1e-15, 1.0 - nu12 * nu21)
            c11 = e11 / detc
            c22 = e22 / detc
            c12 = nu21 * c11
            c_el = np.array([
                [c11, c12, 0.0],
                [c12, c22, 0.0],
                [0.0, 0.0, g12],
            ], dtype=np.float64)

        wp = float(wpla[i])
        fyld = (1.0 + b_val * (wp ** n_val)) if wp > 0.0 else 1.0
        fyld = min(fmax_val, fyld)

        wvec = tsai_wu_yield_criterion(s1, s2, s12, f1, f2, f11, f22, f33, f12)

        if wvec >= fyld - 1e-5:
            # Consistent elastoplastic tangent
            m_grad = np.array([
                f1 + 2.0 * f11 * s1 + 2.0 * f12 * s2,
                f2 + 2.0 * f22 * s2 + 2.0 * f12 * s1,
                2.0 * f33 * s12,
            ], dtype=np.float64)

            m_cel = m_grad @ c_el
            denom_el = float(m_grad @ m_cel)

            h_hard = 0.0
            if wp > 0.0 and b_val > 0.0:
                h_hard = (s1 * m_grad[0] + s2 * m_grad[1] + 2.0 * s12 * m_grad[2]) * n_val * b_val * (wp ** (n_val - 1.0))

            denom = denom_el + h_hard
            if denom > _EM20:
                c_ep = c_el - np.outer(m_cel, m_cel) / denom
                if symmetric:
                    c_ep = 0.5 * (c_ep + c_ep.T)
                tangents[i] = c_ep
            else:
                tangents[i] = c_el if not symmetric else 0.5 * (c_el + c_el.T)
        else:
            tangents[i] = c_el if not symmetric else 0.5 * (c_el + c_el.T)

    return tangents[0] if is_1d else tangents


# ============================================================================
# Update Kernels
# ============================================================================

def solid_update(*args: Any, **kwargs: Any) -> Any:
    """3D solid constitutive update.

    LAW15 is strictly for shell elements in OpenRadioss
    (starter/source/materials/mat/mat015/hm_read_mat15.F:395).
    """
    raise NotImplementedError("LAW15 is for shell elements only")


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, float | np.ndarray]:
    """Plane-stress (shell) constitutive update for LAW15 (CHANG).

    Fortran origin:
      - engine/source/materials/mat/mat015/sigeps15c.F
      - engine/source/materials/mat/mat015/m15cplrc.F
      - engine/source/materials/mat/mat015/m15crak.F

    Parameters:
      mat: Material instance with LAW15 parameters
      sig: In-plane stress array (n, 3) = [s11, s22, s12] or (3,)
      deps: Strain increment array (n, 3) = [de11, de22, dgamma12] or (3,)
      epsp: Optional plastic strain array (n,) or scalar
      dt: Simulation time increment
      extra: Optional dict containing persistent state arrays:
             - 'damt15': (n, 2) [damt_fiber, damt_matrix], init 1.0
             - 'sigr15': (n, 6) [s1, s2, s12, s23, s31, tfail], init 0.0
             - 'wpla15': (n,) accumulated plastic work, init 0.0
             - 'off15': (n,) active status flag (1.0 active, 0.0 failed)
             - 'time': current simulation time

    Returns:
      (sig_new, epsp_new, sound_speed)
    """
    p = getattr(mat, "params", {}) or {}
    is_1d = (sig.ndim == 1)
    s = np.atleast_2d(sig).copy()
    d = np.atleast_2d(deps).copy()
    n = s.shape[0]

    ep = np.zeros(n) if epsp is None else (np.atleast_1d(epsp).copy() if not np.isscalar(epsp) else np.full(n, float(epsp)))
    if extra is None:
        extra = {}

    # Extract or allocate state arrays
    damt = extra.get("damt15")
    if damt is None or damt.shape[0] != n:
        damt = extra.get("damt")
    if damt is None or damt.shape[0] != n:
        damt = np.ones((n, 2), dtype=np.float64)
    else:
        damt = np.copy(damt)

    sigr = extra.get("sigr15")
    if sigr is None or sigr.shape[0] != n:
        sigr = extra.get("sigr")
    if sigr is None or sigr.shape[0] != n:
        sigr = np.zeros((n, 6), dtype=np.float64)
    else:
        sigr = np.copy(sigr)

    wpla = extra.get("wpla15")
    if wpla is None or len(wpla) != n:
        wpla = extra.get("wpla")
    if wpla is None or len(wpla) != n:
        wpla = np.zeros(n, dtype=np.float64)
    else:
        wpla = np.copy(wpla)

    off = extra.get("off15")
    if off is None or len(off) != n:
        off = extra.get("off")
    if off is None or len(off) != n:
        off = np.ones(n, dtype=np.float64)
    else:
        off = np.copy(off)

    time = float(extra.get("time", extra.get("t", extra.get("TT", 0.0))))

    # Material properties
    e11 = float(p.get("e11", p.get("E1", p.get("MAT_EA", 1.0))))
    e22 = float(p.get("e22", p.get("E2", p.get("MAT_EB", 1.0))))
    nu12 = float(p.get("nu12", p.get("MAT_PRAB", 0.0)))
    g12 = float(p.get("g12", p.get("G12", p.get("MAT_GAB", 0.0))))

    b_val = float(p.get("b", p.get("MAT_BETA", 0.0)))
    n_val = float(p.get("hard", p.get("n", p.get("MAT_HARD", 1.0))))
    fmax_val = float(p.get("sig", p.get("fmax", p.get("MAT_SIG", _INF))))
    wpmax_val = float(p.get("wpmax", p.get("WPMAX", _INF)))
    wpref_val = float(p.get("wpref", p.get("WPREF", 1.0)))
    itype = int(p.get("itype", p.get("Itype", p.get("ioff", 0))))

    f1 = float(p.get("F1", 0.0))
    f2 = float(p.get("F2", 0.0))
    f11 = float(p.get("F11", 0.0))
    f22 = float(p.get("F22", 0.0))
    f33 = float(p.get("F33", 0.0))
    f12 = float(p.get("F12", 0.0))

    src = float(p.get("src", p.get("c", p.get("MAT_SRC", 0.0))))
    srp = float(p.get("srp", p.get("epdr", p.get("MAT_SRP", 1.0))))
    strflag = int(p.get("strflag", p.get("icc", p.get("STRFLAG", 1))))

    beta_s = float(p.get("beta_s", p.get("MAT_Beta", 1.0)))
    tmax = float(p.get("tmax", p.get("MAT_TMAX", _INF)))

    s1 = float(p.get("s1", p.get("MCHANG_S1", _INF)))
    s2 = float(p.get("s2", p.get("MCHANG_S2", _INF)))
    s12_str = float(p.get("s12", p.get("MCHANG_S12", _INF)))
    c1 = float(p.get("c1", p.get("MCHANG_C1", _INF)))
    c2 = float(p.get("c2", p.get("MCHANG_C2", _INF)))

    s_out = np.zeros_like(s)
    ep_out = np.zeros_like(ep)

    for i in range(n):
        if off[i] <= 0.0:
            continue

        sig_old_1 = s[i, 0]
        sig_old_2 = s[i, 1]
        sig_old_3 = s[i, 2]

        deps_1 = d[i, 0]
        deps_2 = d[i, 1]
        deps_3 = d[i, 2]

        dam_f = damt[i, 0]
        dam_m = damt[i, 1]

        # --------------------------------------------------------------------
        # 1. Elastic Predictor (m15cplrc.F:118-166)
        # --------------------------------------------------------------------
        if dam_m < 1.0:
            # Matrix has cracked: transverse stiffness and Poisson effects zeroed
            # Fortran lines 148-152: E22 = EM20, NU12 = EM20, NU21 = EM20
            a11 = e11
            a22 = _EM20
            a12 = 0.0
            t1 = sig_old_1 + a11 * deps_1
            t2 = 0.0
            t3 = sig_old_3 + g12 * deps_3
        else:
            nu21 = nu12 * e22 / max(e11, _EM20) if e11 > 0.0 else 0.0
            scale2 = max(1.0e-15, 1.0 - nu12 * nu21)
            a11 = e11 / scale2
            a22 = e22 / scale2
            a12 = nu21 * a11
            t1 = sig_old_1 + a11 * deps_1 + a12 * deps_2
            t2 = sig_old_2 + a12 * deps_1 + a22 * deps_2
            t3 = sig_old_3 + g12 * deps_3

        # --------------------------------------------------------------------
        # 2. Strain Rate Effect (m15cplrc.F:170-195)
        # --------------------------------------------------------------------
        dt_eff = max(dt, _EM20)
        eps_dot = max(abs(deps_1), abs(deps_2), abs(deps_3)) / dt_eff
        epspfac = strain_rate_factor(eps_dot, src, srp)

        wp = float(wpla[i])
        fyld = (1.0 + b_val * (wp ** n_val)) * epspfac
        fmax_eff = fmax_val * epspfac if strflag in (1, 3) else fmax_val
        wpmax_eff = wpmax_val * epspfac if strflag in (3, 4) else wpmax_val
        fyld = min(fmax_eff, fyld)

        # --------------------------------------------------------------------
        # 3. Tsai-Wu Plasticity Return (m15cplrc.F:199-266)
        # --------------------------------------------------------------------
        wvec = tsai_wu_yield_criterion(t1, t2, t3, f1, f2, f11, f22, f33, f12)

        if wvec > fyld and off[i] > 0.0:
            coefa = f11 * (t1 ** 2) + f22 * (t2 ** 2) + f33 * (t3 ** 2) + 2.0 * f12 * t1 * t2
            coefb = f1 * t1 + f2 * t2
            delta = coefb ** 2 + 4.0 * coefa * fyld
            if delta >= 0.0 and coefa > _EM20:
                beta_yld = (-coefb + math.sqrt(delta)) / (2.0 * coefa)
                beta_yld = max(0.0, min(1.0, beta_yld))
            else:
                beta_yld = 1.0 / math.sqrt(max(_EM20, wvec / fyld))

            # If old stress was already at/near yield, evaluate gradient at sig_old;
            # otherwise (entering yield from elastic state), evaluate at beta_yld * T
            w_old = tsai_wu_yield_criterion(sig_old_1, sig_old_2, sig_old_3, f1, f2, f11, f22, f33, f12)
            if w_old >= 0.9 * fyld:
                so1 = sig_old_1
                so2 = sig_old_2
                so3 = sig_old_3
            else:
                so1 = beta_yld * t1
                so2 = beta_yld * t2
                so3 = beta_yld * t3

            dp1 = f1 + 2.0 * f11 * so1 + 2.0 * f12 * so2
            dp2 = f2 + 2.0 * f22 * so2 + 2.0 * f12 * so1
            dp3 = 2.0 * f33 * so3

            ds1 = t1 - so1
            ds2 = t2 - so2
            ds3 = t3 - so3

            lamda = dp1 * ds1 + dp2 * ds2 + dp3 * ds3
            if lamda <= 0.0:
                so1 = beta_yld * t1
                so2 = beta_yld * t2
                so3 = beta_yld * t3
                dp1 = f1 + 2.0 * f11 * so1 + 2.0 * f12 * so2
                dp2 = f2 + 2.0 * f22 * so2 + 2.0 * f12 * so1
                dp3 = 2.0 * f33 * so3
                ds1 = t1 - so1
                ds2 = t2 - so2
                ds3 = t3 - so3
                lamda = dp1 * ds1 + dp2 * ds2 + dp3 * ds3

            if lamda > 0.0:
                cnn = n_val - 1.0
                wvec_hard = epspfac * (wp ** cnn) if (wp > 0.0 and fyld < fmax_eff) else 0.0
                h_term = (so1 * dp1 + so2 * dp2 + 2.0 * so3 * dp3) * n_val * b_val * wvec_hard
                denom = (
                    dp1 * (a11 * dp1 + a12 * dp2)
                    + dp2 * (a12 * dp1 + a22 * dp2)
                    + 2.0 * dp3 * g12 * dp3
                    + h_term
                )
                if denom > _EM20:
                    lamda = lamda / denom
                    dp1_pl = lamda * dp1
                    dp2_pl = lamda * dp2
                    dp3_pl = lamda * dp3

                    t1 = t1 - a11 * dp1_pl - a12 * dp2_pl
                    t2 = t2 - a12 * dp1_pl - a22 * dp2_pl
                    t3 = t3 - 2.0 * g12 * dp3_pl

                    dwpla = 0.5 * (
                        dp1_pl * (t1 + so1)
                        + dp2_pl * (t2 + so2)
                        + 2.0 * dp3_pl * (t3 + so3)
                    )
                    wp = max(0.0, wp + dwpla / max(wpref_val, _EM20))
            else:
                t1 = beta_yld * t1
                t2 = beta_yld * t2
                t3 = beta_yld * t3
                dwpla = 0.5 * ((t1 - sig_old_1)**2 / a11 + (t2 - sig_old_2)**2 / a22)
                wp = max(0.0, wp + dwpla / max(wpref_val, _EM20))

        s11 = t1
        s22 = t2
        s12 = t3
        tfail = sigr[i, 5]

        # --------------------------------------------------------------------
        # 4. Chang-Chang Failure Checking (m15crak.F:102-188)
        # --------------------------------------------------------------------
        if dam_f < 1.0:
            # Mode A: Fiber already failed -> exponential relaxation of all stresses
            # Fortran line 104-105
            dam_f = math.exp(-(time - tfail) / max(tmax, _EM20))
            if dam_f < 0.01:
                dam_f = 0.0
            s11 = sigr[i, 0] * dam_f
            s22 = sigr[i, 1] * dam_f
            s12 = sigr[i, 2] * dam_f
        elif dam_m < 1.0:
            # Mode B: Matrix already failed -> exponential relaxation of s22, s12
            # Fortran lines 109-130
            dam_m = math.exp(-(time - tfail) / max(tmax, _EM20))
            if dam_m < 0.01:
                dam_m = 0.0
            s22 = sigr[i, 1] * dam_m
            s12 = sigr[i, 2] * dam_m

            # Check fiber failure mode while matrix is cracked
            if s11 > 0.0:
                ef2 = (s11 / max(s1, _EM20)) ** 2 + beta_s * (s12 / max(s12_str, _EM20)) ** 2
                efc2 = 0.0
            else:
                ef2 = 0.0
                efc2 = (s11 / max(c1, _EM20)) ** 2

            if ef2 >= 1.0 or efc2 >= 1.0:
                # Fiber failure triggers!
                tfail = time
                sigr[i, 5] = tfail
                dam_f = 0.999
                sigr[i, 0] = s11
                sigr[i, 1] = s22
                sigr[i, 2] = s12
        else:
            # Mode C: Intact -> check fiber and matrix failure criteria
            # Fortran lines 137-187
            if s11 > 0.0:
                ef2 = (s11 / max(s1, _EM20)) ** 2 + beta_s * (s12 / max(s12_str, _EM20)) ** 2
                efc2 = 0.0
            else:
                ef2 = 0.0
                efc2 = (s11 / max(c1, _EM20)) ** 2

            if ef2 >= 1.0 or efc2 >= 1.0:
                # Fiber breakage failure
                dam_f = 0.999
                tfail = time
                sigr[i, 5] = tfail
                sigr[i, 0] = s11
                sigr[i, 1] = s22
                sigr[i, 2] = s12
            else:
                # Matrix cracking failure
                if s22 >= 0.0:
                    em2 = (s22 / max(c2, _EM20)) ** 2 + (s12 / max(s12_str, _EM20)) ** 2
                    emc2 = 0.0
                else:
                    em2 = 0.0
                    term1 = (s22 / (2.0 * max(s12_str, _EM20))) ** 2
                    term2 = (s12 / max(s12_str, _EM20)) ** 2
                    ratio = (c2 / (2.0 * max(s12_str, _EM20))) ** 2 - 1.0
                    term3 = (s22 / max(c2, _EM20)) * ratio
                    emc2 = term1 + term2 + term3

                if em2 >= 1.0 or emc2 >= 1.0:
                    # Matrix cracking failure
                    dam_m = 0.999
                    tfail = time
                    sigr[i, 5] = tfail
                    sigr[i, 0] = s11
                    sigr[i, 1] = s22
                    sigr[i, 2] = s12

        damt[i, 0] = dam_f
        damt[i, 1] = dam_m

        # --------------------------------------------------------------------
        # 5. Layer Failure & Element Deletion (IOFF / Itype)
        # --------------------------------------------------------------------
        if itype != 0:
            if dam_f < 1.0 or wp >= wpmax_eff:
                off[i] = 0.0
                s11 = 0.0
                s22 = 0.0
                s12 = 0.0
        else:
            if wp >= wpmax_eff:
                off[i] = 0.0
                s11 = 0.0
                s22 = 0.0
                s12 = 0.0

        s_out[i, 0] = s11
        s_out[i, 1] = s22
        s_out[i, 2] = s12
        wpla[i] = wp
        ep_out[i] = wp

    # Save state back into extra
    extra["damt15"] = damt
    extra["damt"] = damt
    extra["sigr15"] = sigr
    extra["sigr"] = sigr
    extra["wpla15"] = wpla
    extra["wpla"] = wpla
    extra["off15"] = off
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


# ============================================================================
# Material Registration
# ============================================================================

def _register() -> None:
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for k in (15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG"):
            MAT_PHYSICS_REGISTRY[k] = build_law15
    except Exception:
        pass


_register()
