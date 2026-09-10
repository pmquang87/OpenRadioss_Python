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
    sig_12yc = _get(["MAT_SIGC12", "sig_12yc", "sigc12", "sigyc12"], _INF)
    sig_12yt = _get(["MAT_SIGT12", "MAT_SIG12_yt", "sig_12yt", "sigt12", "sigyt12"], _INF)
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
    mod_max = max(c1, gmax, e33)
    c_sound = math.sqrt(max(c1, gmax, e33) / max(rho0, _EM20)) if rho0 > 0.0 else math.sqrt(mod_max)

    # Tsai-Wu yield coefficients (read_mat25_tsaiwu.F90:310-316)
    f1 = (1.0 / sig_1yt - 1.0 / sig_1yc) if sig_1yt > 0 and sig_1yc > 0 else 0.0
    f2 = (1.0 / sig_2yt - 1.0 / sig_2yc) if sig_2yt > 0 and sig_2yc > 0 else 0.0
    f11 = 1.0 / max(_EM20, min(1e20, sig_1yt * sig_1yc)) if sig_1yt > 0 and sig_1yc > 0 else 0.0
    f22 = 1.0 / max(_EM20, min(1e20, sig_2yt * sig_2yc)) if sig_2yt > 0 and sig_2yc > 0 else 0.0
    f33 = 1.0 / max(_EM20, min(1e20, sig_12yt * sig_12yc)) if sig_12yt > 0 and sig_12yc > 0 else 0.0
    denom = 2.0 * math.sqrt(max(_EM20, min(1e20, sig_1yt * sig_1yc * sig_2yt * sig_2yc)))
    f12 = -alpha / denom if denom > 0 else 0.0

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
    e33 = float(params.get("e33", params.get("MAT_EC", max(e11, e22))))
    nu12 = float(params.get("nu12", params.get("MAT_PRAB", 0.3)))
    g12 = float(params.get("g12", params.get("MAT_GAB", 0.0)))
    g23 = float(params.get("g23", params.get("MAT_GBC", 0.0)))
    g31 = float(params.get("g31", params.get("MAT_GCA", 0.0)))

    nu21 = nu12 * e22 / max(e11, _EM20) if e11 > 0.0 else 0.0
    detc = max(1.0e-15, 1.0 - nu12 * nu21)
    c1 = max(e11, e22) / detc
    young_mod = max(c1, g12, g23, g31, e33)
    return np.sqrt(young_mod / np.maximum(r, _EM20))


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

def shell_membrane_tangent(mat: Any) -> np.ndarray:
    """(3, 3) orthotropic plane-stress elastic matrix for shells."""
    params = getattr(mat, "params", {}) or {}
    e11 = float(params.get("e11", params.get("MAT_EA", 1.0)))
    e22 = float(params.get("e22", params.get("MAT_EB", 1.0)))
    nu12 = float(params.get("nu12", params.get("MAT_PRAB", 0.3)))
    g12 = float(params.get("g12", params.get("MAT_GAB", 0.5 * e11 / max(1.0 + nu12, 1e-15))))

    nu21 = nu12 * e22 / max(e11, _EM20) if e11 > 0.0 else 0.0
    detc = max(1.0e-15, 1.0 - nu12 * nu21)

    c11 = e11 / detc
    c22 = e22 / detc
    c12 = nu12 * e22 / detc

    return np.array([
        [c11, c12, 0.0],
        [c12, c22, 0.0],
        [0.0, 0.0, g12],
    ], dtype=np.float64)


def consistent_shell_tangent(
    mat: Any,
    sig: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: Any = None,
    extra: Any = None,
    **kwargs: Any,
) -> np.ndarray:
    """(n, 3, 3) consistent plane-stress tangent for shells."""
    c_shell = shell_membrane_tangent(mat)
    n = 1 if sig.ndim == 1 else sig.shape[0]
    tangents = np.broadcast_to(c_shell, (n, 3, 3)).copy()

    params = getattr(mat, "params", {}) or {}
    f1 = float(params.get("F1", 0.0))
    f2 = float(params.get("F2", 0.0))
    f11 = float(params.get("F11", 0.0))
    f22 = float(params.get("F22", 0.0))
    f33 = float(params.get("F33", 0.0))
    f12 = float(params.get("F12", 0.0))
    b_val = float(params.get("b", params.get("MAT_BETA", 0.0)))
    n_val = float(params.get("n", params.get("MAT_HARD", 1.0)))
    fmax_val = float(params.get("fmax", params.get("MAT_SIG", _INF)))

    sig_2d = np.atleast_2d(sig)
    for i in range(n):
        s1 = sig_2d[i, 0]
        s2 = sig_2d[i, 1]
        s12 = sig_2d[i, 2]

        # Check Tsai-Wu yielding
        wvec = f1 * s1 + f2 * s2 + f11 * (s1 ** 2) + f22 * (s2 ** 2) + f33 * (s12 ** 2) + 2.0 * f12 * s1 * s2
        if wvec >= 1.0 - 1e-6:
            # Flow gradient m = df/ds
            m_grad = np.array([
                f1 + 2.0 * f11 * s1 + 2.0 * f12 * s2,
                f2 + 2.0 * f22 * s2 + 2.0 * f12 * s1,
                2.0 * f33 * s12,
            ], dtype=float)

            cm = c_shell @ m_grad
            h_hard = 0.0
            wp = float(epsp[i]) if (epsp is not None and i < len(epsp)) else 0.0
            if wp > 0.0 and b_val > 0.0:
                h_hard = (s1 * m_grad[0] + s2 * m_grad[1] + s12 * m_grad[2]) * n_val * b_val * (wp ** (n_val - 1.0))

            denom = float(m_grad @ cm + h_hard)
            if denom > _EM20:
                c_ep = c_shell - np.outer(cm, cm) / denom
                tangents[i] = c_ep

    return tangents


def solid_stiffness_matrix(mat: Any) -> np.ndarray:
    """(6, 6) 3D orthotropic elastic stiffness matrix."""
    params = getattr(mat, "params", {}) or {}
    e11 = float(params.get("e11", params.get("MAT_EA", 1.0)))
    e22 = float(params.get("e22", params.get("MAT_EB", 1.0)))
    e33 = float(params.get("e33", params.get("MAT_EC", max(e11, e22))))
    nu12 = float(params.get("nu12", params.get("MAT_PRAB", 0.3)))
    g12 = float(params.get("g12", params.get("MAT_GAB", 0.5 * e11 / max(1.0 + nu12, 1e-15))))
    g23 = float(params.get("g23", params.get("MAT_GBC", g12)))
    g31 = float(params.get("g31", params.get("MAT_GCA", g12)))

    nu21 = nu12 * e22 / max(e11, _EM20)
    detc = max(1.0e-15, 1.0 - nu12 * nu21)
    a11 = e11 / detc
    a22 = e22 / detc
    a12 = nu21 * a11

    c = np.zeros((6, 6), dtype=np.float64)
    c[0, 0] = a11
    c[0, 1] = a12
    c[1, 0] = a12
    c[1, 1] = a22
    c[2, 2] = e33
    c[3, 3] = g12
    c[4, 4] = g23
    c[5, 5] = g31
    return c


def consistent_solid_tangent(
    mat: Any,
    sig: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: Any = None,
    extra: Any = None,
    **kwargs: Any,
) -> np.ndarray:
    """(n, 6, 6) 3D solid consistent tangent."""
    c_mat = solid_stiffness_matrix(mat)
    n = 1 if sig.ndim == 1 else sig.shape[0]
    tangents = np.broadcast_to(c_mat, (n, 6, 6)).copy()

    params = getattr(mat, "params", {}) or {}
    f1 = float(params.get("F1", 0.0))
    f2 = float(params.get("F2", 0.0))
    f11 = float(params.get("F11", 0.0))
    f22 = float(params.get("F22", 0.0))
    f33 = float(params.get("F33", 0.0))
    f12 = float(params.get("F12", 0.0))
    b_val = float(params.get("b", params.get("MAT_BETA", 0.0)))
    n_val = float(params.get("n", params.get("MAT_HARD", 1.0)))

    sig_2d = np.atleast_2d(sig)
    for i in range(n):
        s1 = sig_2d[i, 0]
        s2 = sig_2d[i, 1]
        s12 = sig_2d[i, 3]

        wvec = f1 * s1 + f2 * s2 + f11 * (s1 ** 2) + f22 * (s2 ** 2) + f33 * (s12 ** 2) + 2.0 * f12 * s1 * s2
        if wvec >= 1.0 - 1e-6:
            m_grad = np.array([
                f1 + 2.0 * f11 * s1 + 2.0 * f12 * s2,
                f2 + 2.0 * f22 * s2 + 2.0 * f12 * s1,
                0.0,
                2.0 * f33 * s12,
                0.0,
                0.0,
            ], dtype=float)

            cm = c_mat @ m_grad
            h_hard = 0.0
            wp = float(epsp[i]) if (epsp is not None and i < len(epsp)) else 0.0
            if wp > 0.0 and b_val > 0.0:
                h_hard = (s1 * m_grad[0] + s2 * m_grad[1] + s12 * m_grad[3]) * n_val * b_val * (wp ** (n_val - 1.0))

            denom = float(m_grad @ cm + h_hard)
            if denom > _EM20:
                c_ep = c_mat - np.outer(cm, cm) / denom
                tangents[i] = c_ep

    return tangents


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
    epst = epst_val + deps

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

        fyld = ((1.0 + b_val * (wpla ** n_val)) if wpla > 0.0 else 1.0) * epspfac
        fyld = min(fmax_val, fyld)

        wvec = f1 * t1 + f2 * t2 + f11 * (t1 ** 2) + f22 * (t2 ** 2) + f33 * (t3 ** 2) + 2.0 * f12 * t1 * t2

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

            dp1 = f1 + 2.0 * f11 * so1 + 2.0 * f12 * so2
            dp2 = f2 + 2.0 * f22 * so2 + 2.0 * f12 * so1
            dp3 = 2.0 * f33 * so3

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
        f1 = (1.0 / s_yt1 - 1.0 / s_yc1) if (s_yt1 > 0 and s_yc1 > 0) else 0.0
        f2 = (1.0 / s_yt2 - 1.0 / s_yc2) if (s_yt2 > 0 and s_yc2 > 0) else 0.0
        f11 = 1.0 / max(_EM20, s_yt1 * s_yc1)
        f22 = 1.0 / max(_EM20, s_yt2 * s_yc2)
        f33 = 1.0 / max(_EM20, s_yt12 * s_yt12)
        denom_12 = 2.0 * math.sqrt(max(_EM20, s_yt1 * s_yc1 * s_yt2 * s_yc2))
        f12 = -alpha / denom_12 if denom_12 > 0 else 0.0

        wvec = f1 * t1 + f2 * t2 + f11 * (t1 ** 2) + f22 * (t2 ** 2) + f33 * (t3 ** 2) + 2.0 * f12 * t1 * t2

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

            dp1 = f1 + 2.0 * f11 * so1 + 2.0 * f12 * so2
            dp2 = f2 + 2.0 * f22 * so2 + 2.0 * f12 * so1
            dp3 = 2.0 * f33 * so3

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
    if "layfail" in extra and extra["layfail"] is not None:
        extra["layfail"][:] = off

    c_val = sound_speed(mat)
    c_arr = np.full(n, float(c_val), dtype=float)

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

    c_val = sound_speed(mat)
    c_arr = np.full(n, float(c_val), dtype=float)

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
                    "COMPOSITE_PLAS", "MAT_LAW25", "MAT_COMP_PLAS", "MAT_COMPSH"):
            MAT_PHYSICS_REGISTRY[key] = build_law25
    except Exception:
        pass


_register()
