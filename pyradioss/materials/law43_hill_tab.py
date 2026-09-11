"""
LAW43 — Tabulated Hill Orthotropic Anisotropic Plasticity (/MAT/LAW43, /MAT/HILL_TAB).

Upstream Fortran reference:
  - Starter reader: starter/source/materials/mat/mat043/hm_read_mat43.F
  - Engine shells: engine/source/materials/mat/mat043/sigeps43c.F
  - Engine solids / global shells: engine/source/materials/mat/mat043/sigeps43g.F
  - CFG file: hm_cfg_files/config/CFG/radioss140/MAT/matl43_HILL_TAB.cfg

Theory
------
Hill 1948 quadratic orthotropic yield criterion in plane stress:
    SVM = sqrt(A01 * sig_xx^2 + A02 * sig_yy^2 - A03 * sig_xx * sig_yy + A12 * sig_xy^2)

Hill Anisotropy Constants computed from Lankford coefficients (R00, R45, R90):
    R = 0.25 * (R00 + 2.0 * R45 + R90)
    H = R / (1.0 + R)
    A01 = H * (1.0 + 1.0 / R00)
    A02 = H * (1.0 + 1.0 / R90)
    A03 = 2.0 * H
    A12 = (2.0 * R45 + 1.0) * (A01 + A02 - A03)
    if Iyield > 0:
        A02 /= A01
        A03 /= A01
        A12 /= A01
        A01 = 1.0

Hardening:
  Mixed isotropic-kinematic hardening:
    YLD = (1 - FISOKIN) * FAIL * Y(pla, edot) + FISOKIN * FAIL * Y0(edot)
    Back-stress update alpha += (2*P1 + P2)*dr0, etc.

Dynamic Young's modulus degradation:
    if OPTE == 1 and curve: E = E0 * f_E(pla)
    elif CE > 0: E = E0 - (E0 - Einf) * (1 - exp(-CE * pla))

Failure:
    Tensile strain failure: FAIL = clamp((EPSR2 - epst) / (EPSR2 - EPSR1), 0.0, 1.0)
    Element deletion: if pla > EPSMAX: off = 0.8 * off; if off < 0.1: off = 0.0
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_INF = 1.0e30


# ============================================================================
# Curve Evaluation Utilities
# ============================================================================

def _curve_eval(cx: np.ndarray, cy: np.ndarray, cs: np.ndarray,
                e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear evaluation with end-slope extrapolation (FINTER).

    Returns (value, slope).
    """
    e_arr = np.asarray(e, dtype=float)
    if len(cx) == 1:
        return np.full_like(e_arr, cy[0]), np.zeros_like(e_arr)
    idx = np.minimum(np.maximum(np.searchsorted(cx, e_arr, side="right") - 1, 0),
                     len(cx) - 2)
    val = cy[idx] + cs[idx] * (e_arr - cx[idx])
    return val, cs[idx]


def _eval_yield_stress(mat: Material, epsp: np.ndarray, rate: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate current yield stress, hardening slope, and initial yield stress.

    Returns:
      Y: current yield stress at epsp
      H: current hardening slope dY/depsp at epsp
      Y0: initial yield stress at epsp=0
    """
    p = mat.params
    cxs = p.get("curve_x", [])
    cys = p.get("curve_y", [])
    css = p.get("curve_s", [])
    nfun = len(cxs)

    if nfun == 0:
        # Fallback flat yield
        sigy0 = float(p.get("sigy0", p.get("MAT_SIGY", 1.0e30)))
        zeros = np.zeros_like(epsp)
        return np.full_like(epsp, sigy0), zeros, np.full_like(epsp, sigy0)

    epsp_arr = np.asarray(epsp, dtype=float)
    zeros_arr = np.zeros_like(epsp_arr)

    if nfun == 1:
        y, h = _curve_eval(cxs[0], cys[0], css[0], epsp_arr)
        y0, _ = _curve_eval(cxs[0], cys[0], css[0], zeros_arr)
        return y, h, y0

    rates = np.asarray(p["rates"], dtype=float)
    rate_arr = np.asarray(rate, dtype=float)

    # Evaluate all curves
    vals = np.empty((nfun, len(epsp_arr)), dtype=float)
    slps = np.empty((nfun, len(epsp_arr)), dtype=float)
    vals0 = np.empty((nfun, len(epsp_arr)), dtype=float)
    for i in range(nfun):
        vals[i], slps[i] = _curve_eval(cxs[i], cys[i], css[i], epsp_arr)
        vals0[i], _ = _curve_eval(cxs[i], cys[i], css[i], zeros_arr)

    # Linear bracket interpolation in strain rate
    j = np.clip(np.searchsorted(rates, rate_arr, side="right") - 1, 0, nfun - 2)
    denom = np.maximum(rates[j + 1] - rates[j], _EM20)
    w = np.clip((rate_arr - rates[j]) / denom, 0.0, 1.0)

    cols = np.arange(len(epsp_arr))
    sy = (1.0 - w) * vals[j, cols] + w * vals[j + 1, cols]
    h_slope = (1.0 - w) * slps[j, cols] + w * slps[j + 1, cols]
    sy0 = (1.0 - w) * vals0[j, cols] + w * vals0[j + 1, cols]

    return sy, h_slope, sy0


def _eval_young_modulus(mat: Material, pla: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute degraded Young's modulus E, A1, A2, G, G3 given plastic strain pla.

    Follows sigeps43c.F lines 144-168.
    """
    p = mat.params
    e0 = float(p.get("E0", p.get("E", 1.0)))
    nu = float(p.get("nu", 0.0))
    opte = int(p.get("OPTE", 0))
    ce = float(p.get("CE", 0.0))
    einf = float(p.get("Einf", e0))

    pla_arr = np.asarray(pla, dtype=float)
    e = np.full_like(pla_arr, e0, dtype=float)

    if opte == 1 and "E_curve_x" in p and len(p["E_curve_x"]) > 0:
        pos = pla_arr > 0.0
        if np.any(pos):
            scale, _ = _curve_eval(p["E_curve_x"], p["E_curve_y"], p["E_curve_s"], pla_arr[pos])
            e[pos] = scale * e0
    elif ce > 0.0:
        pos = pla_arr > 0.0
        if np.any(pos):
            e[pos] = e0 - (e0 - einf) * (1.0 - np.exp(-ce * pla_arr[pos]))

    a1 = e / (1.0 - nu ** 2)
    a2 = nu * a1
    g = 0.5 * e / (1.0 + nu)
    g3 = 3.0 * g
    return e, a1, a2, g, g3


# ============================================================================
# Constructor: build_law43
# ============================================================================

def build_law43(rec: Any = None, **kwargs: Any) -> Material:
    """Physics constructor for /MAT/LAW43 (/MAT/HILL_TAB).

    Extracts parameters per starter/source/materials/mat/mat043/hm_read_mat43.F:
      - rho0, rhor
      - E0, nu
      - IFUNCE, Einf, CE
      - R00, R45, R90
      - FISOKIN (iso-kinematic factor [0..1])
      - Iyield (IR0)
      - EPSMAX, EPSR1, EPSR2
      - ASRATE, ISRATE
      - curves
    """
    if rec is None:
        p: Dict[str, Any] = dict(kwargs)
        _id = int(kwargs.get("id", 1))
        rho0_in = kwargs.get("rho0", kwargs.get("density", kwargs.get("MAT_RHO", 0.0)))
        _title = str(kwargs.get("title", "LAW43_HILL_TAB"))
    elif isinstance(rec, dict):
        base_params = rec.get("params", rec)
        p = {**base_params, **kwargs}
        _id = int(rec.get("id", kwargs.get("id", 1)))
        rho0_in = rec.get("rho0", rec.get("density", kwargs.get("rho0", kwargs.get("density", 0.0))))
        _title = str(rec.get("title", kwargs.get("title", "LAW43_HILL_TAB")))
    elif hasattr(rec, "params"):
        p = {**(rec.params if isinstance(rec.params, dict) else {}), **kwargs}
        _id = int(getattr(rec, "id", kwargs.get("id", 1)))
        rho0_in = getattr(rec, "rho0", getattr(rec, "density", kwargs.get("rho0", 0.0)))
        _title = str(getattr(rec, "title", kwargs.get("title", "LAW43_HILL_TAB")))
    elif hasattr(rec, "__dict__"):
        p = {**rec.__dict__, **kwargs}
        _id = int(getattr(rec, "id", kwargs.get("id", 1)))
        rho0_in = getattr(rec, "rho0", getattr(rec, "density", kwargs.get("rho0", 0.0)))
        _title = str(getattr(rec, "title", kwargs.get("title", "LAW43_HILL_TAB")))
    else:
        p = dict(kwargs)
        _id = int(kwargs.get("id", 1))
        rho0_in = kwargs.get("rho0", 0.0)
        _title = str(kwargs.get("title", "LAW43_HILL_TAB"))

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
    rhor = _get(["Refer_Rho", "rhor", "MAT_REFRHO"], rho0)
    if rhor == 0.0:
        rhor = rho0

    e0 = _get(["MAT_E", "E", "e", "young", "Young", "E0"], 0.0)
    if e0 <= 0.0:
        raise ValueError(f"/MAT/LAW43/{_id}: Young's modulus E must be > 0.")

    nu = _get(["MAT_NU", "NU", "nu", "anu"], 0.0)
    if nu >= 0.5:
        nu = 0.499
    if nu < 0.0:
        raise ValueError(f"/MAT/LAW43/{_id}: Poisson's ratio nu must be >= 0.")

    ifunce = _geti(["Yr_fun", "IFUNCE", "ifunce", "fct_e", "funct_e"], 0)
    einf = _get(["MAT_EFIB", "EINF", "Einf", "einf"], e0)
    ce = _get(["MAT_C", "CE", "ce", "c_e"], 0.0)
    opte = 1 if ifunce > 0 else 0

    r00 = _get(["MAT_R00", "R00", "r00", "r_00"], 1.0)
    if r00 <= 0.0:
        r00 = 1.0
    r45 = _get(["MAT_R45", "R45", "r45", "r_45"], 1.0)
    if r45 <= 0.0:
        r45 = 1.0
    r90 = _get(["MAT_R90", "R90", "r90", "r_90"], 1.0)
    if r90 <= 0.0:
        r90 = 1.0

    fisokin = _get(["MAT_CHARD", "C_hard", "c_hard", "FISOKIN", "fisokin", "chard"], 0.0)
    if fisokin < 0.0 or fisokin > 1.0:
        raise ValueError(f"/MAT/LAW43/{_id}: FISOKIN ({fisokin}) must be in [0, 1] (upstream error 913).")

    iyield = _geti(["MAT_Iyield", "Iyield", "iyield", "ir0", "IR0"], 0)

    epsmax = _get(["MAT_EPS", "EPS", "eps", "epsmax", "EPSMAX", "eps_max"], _INF)
    if epsmax <= 0.0:
        epsmax = _INF

    epsr1 = _get(["MAT_EPST1", "EPST1", "epsr1", "EPSR1"], _INF)
    if epsr1 <= 0.0:
        epsr1 = _INF
    epsr2 = _get(["MAT_EPST2", "EPST2", "epsr2", "EPSR2"], 2.0 * _INF)
    if epsr2 <= 0.0:
        epsr2 = 2.0 * _INF

    asrate = _get(["Fcut", "fcut", "ASRATE", "asrate"], 0.0)
    israte = _geti(["Fsmooth", "fsmooth", "ISRATE", "israte"], 0)
    if asrate != 0.0:
        israte = 1
    elif israte != 0:
        asrate = 10000.0
    else:
        asrate = 0.0

    # Hill Anisotropy Constants (hm_read_mat43.F:216-228)
    r = 0.25 * (r00 + 2.0 * r45 + r90)
    h = r / (1.0 + r)
    a01 = h * (1.0 + 1.0 / r00)
    a02 = h * (1.0 + 1.0 / r90)
    a03 = 2.0 * h
    a12 = (2.0 * r45 + 1.0) * (a01 + a02 - a03)

    if iyield > 0:
        a02 = a02 / a01
        a03 = a03 / a01
        a12 = a12 / a01
        a01 = 1.0

    # Elastic constants
    g = 0.5 * e0 / (1.0 + nu)
    a1 = e0 / (1.0 - nu ** 2)
    a2 = nu * a1
    c1 = e0 / (3.0 * (1.0 - 2.0 * nu))
    c_shell = math.sqrt(a1 / max(rho0, _EM20)) if rho0 > 0.0 else math.sqrt(a1)
    c_solid = math.sqrt((c1 + 4.0 / 3.0 * g) / max(rho0, _EM20)) if rho0 > 0.0 else math.sqrt(c1 + 4.0 / 3.0 * g)

    # Curve processing
    cxs: List[np.ndarray] = []
    cys: List[np.ndarray] = []
    css: List[np.ndarray] = []
    rates: List[float] = []
    funct_ids: List[int] = []
    yfacs: List[float] = []

    raw_curves = p.get("curves", kwargs.get("curves", None))
    if raw_curves is not None:
        parsed_entries = []
        for item in raw_curves:
            if isinstance(item, dict):
                fid = int(item.get("fct_id", item.get("funct_id", item.get("fid", 0))))
                yf = float(item.get("fscale", item.get("yfac", item.get("scale", 1.0))))
                rt = float(item.get("eps_dot", item.get("rate", item.get("srate", 0.0))))
                pts = item.get("points", item.get("pts", None))
                parsed_entries.append((fid, yf, rt, pts))
            elif isinstance(item, (tuple, list)):
                if len(item) == 2 and isinstance(item[0], (list, np.ndarray)):
                    # (x, y) direct curve arrays
                    x_arr = np.asarray(item[0], dtype=float)
                    y_arr = np.asarray(item[1], dtype=float)
                    parsed_entries.append((0, 1.0, 0.0, (x_arr, y_arr)))
                elif len(item) == 3 and isinstance(item[0], (list, np.ndarray)):
                    # (x, y, rate)
                    x_arr = np.asarray(item[0], dtype=float)
                    y_arr = np.asarray(item[1], dtype=float)
                    rt = float(item[2])
                    parsed_entries.append((0, 1.0, rt, (x_arr, y_arr)))
                elif len(item) >= 3:
                    fid = int(item[0])
                    yf = float(item[1]) if float(item[1]) != 0.0 else 1.0
                    rt = float(item[2])
                    parsed_entries.append((fid, yf, rt, None))
                elif len(item) == 1:
                    parsed_entries.append((int(item[0]), 1.0, 0.0, None))

        # Sort entries by strain rate
        parsed_entries.sort(key=lambda x: x[2])

        if len(parsed_entries) == 1:
            e1 = parsed_entries[0]
            parsed_entries = [
                (e1[0], e1[1], 0.0, e1[3]),
                (e1[0], e1[1], 1.0, e1[3]),
            ]
        elif len(parsed_entries) > 1 and parsed_entries[0][2] != 0.0:
            # hm_read_mat43.F: line 206: if RATE(1) != 0, prepend curve at rate 0
            e0_entry = (parsed_entries[0][0], parsed_entries[0][1], 0.0, parsed_entries[0][3])
            parsed_entries.insert(0, e0_entry)

        for fid, yf, rt, pts in parsed_entries:
            funct_ids.append(fid)
            yfacs.append(yf)
            rates.append(rt)
            if pts is not None:
                x_pts = np.asarray(pts[0], dtype=float)
                y_pts = np.asarray(pts[1], dtype=float) * yf
                if len(x_pts) > 1:
                    s_pts = np.diff(y_pts) / np.maximum(np.diff(x_pts), _EM20)
                else:
                    s_pts = np.zeros(0, dtype=float)
                cxs.append(x_pts)
                cys.append(y_pts)
                css.append(s_pts)

    if len(cxs) == 0 and "curve_x" in p and "curve_y" in p:
        raw_cxs = p["curve_x"]
        raw_cys = p["curve_y"]
        raw_css = p.get("curve_s", None)
        raw_rates = list(p.get("rates", [0.0] * len(raw_cxs)))

        if len(raw_cxs) == 1:
            raw_cxs = [raw_cxs[0], raw_cxs[0]]
            raw_cys = [raw_cys[0], raw_cys[0]]
            raw_rates = [0.0, 1.0]
            if raw_css is not None and len(raw_css) > 0:
                raw_css = [raw_css[0], raw_css[0]]

        for i in range(len(raw_cxs)):
            xi = np.asarray(raw_cxs[i], dtype=float)
            yi = np.asarray(raw_cys[i], dtype=float)
            if raw_css is not None and i < len(raw_css):
                si = np.asarray(raw_css[i], dtype=float)
            else:
                si = np.diff(yi) / np.maximum(np.diff(xi), _EM20) if len(xi) > 1 else np.zeros(0, dtype=float)
            cxs.append(xi)
            cys.append(yi)
            css.append(si)
            rates.append(float(raw_rates[i]))
    elif "FunctionIds" in p:
        fids = list(p["FunctionIds"]) if isinstance(p["FunctionIds"], (list, tuple, np.ndarray)) else [p["FunctionIds"]]
        yfs = list(p.get("ABG_cpa", [1.0] * len(fids)))
        rts = list(p.get("ABG_cpb", [0.0] * len(fids)))
        valid = [(int(f), float(y) if float(y) != 0.0 else 1.0, float(r))
                 for f, y, r in zip(fids, yfs, rts) if int(f) != 0]
        valid.sort(key=lambda x: x[2])
        if len(valid) == 1:
            valid = [(valid[0][0], valid[0][1], 0.0), (valid[0][0], valid[0][1], 1.0)]
        elif len(valid) > 1 and valid[0][2] != 0.0:
            valid.insert(0, (valid[0][0], valid[0][1], 0.0))
        for f, y, r in valid:
            funct_ids.append(f)
            yfacs.append(y)
            rates.append(r)

    params: Dict[str, Any] = {
        "E": e0,
        "E0": e0,
        "nu": nu,
        "G": g,
        "K": c1,
        "C1": c1,
        "A1": a1,
        "A2": a2,
        "c_shell": c_shell,
        "c_solid": c_solid,
        "c": c_shell,
        "rho0": rho0,
        "rhor": rhor,
        "R00": r00,
        "R45": r45,
        "R90": r90,
        "R_bar": r,
        "H_hill": h,
        "A01": a01,
        "A02": a02,
        "A03": a03,
        "A12": a12,
        "FISOKIN": fisokin,
        "Iyield": iyield,
        "EPSMAX": epsmax,
        "EPSR1": epsr1,
        "EPSR2": epsr2,
        "ASRATE": asrate,
        "ISRATE": israte,
        "IFUNCE": ifunce,
        "OPTE": opte,
        "Einf": einf,
        "CE": ce,
        "funct_ids": funct_ids,
        "yfac": yfacs,
        "rates": rates,
    }

    if len(cxs) > 0:
        params["curve_x"] = cxs
        params["curve_y"] = cys
        params["curve_s"] = css

    # Pass through any resolved E curve
    if "E_curve_x" in p and "E_curve_y" in p:
        x_e = np.asarray(p["E_curve_x"], dtype=float)
        y_e = np.asarray(p["E_curve_y"], dtype=float)
        params["E_curve_x"] = x_e
        params["E_curve_y"] = y_e
        if "E_curve_s" in p and p["E_curve_s"] is not None:
            params["E_curve_s"] = np.asarray(p["E_curve_s"], dtype=float)
        else:
            params["E_curve_s"] = np.diff(y_e) / np.maximum(np.diff(x_e), _EM20) if len(x_e) > 1 else np.zeros(0, dtype=float)

    mat = Material(
        id=_id,
        law=43,
        rho0=rho0,
        title=_title,
        law_name="LAW43",
        params=params,
    )
    return mat


# ============================================================================
# Curve Resolution Hook: resolve
# ============================================================================

def resolve(mat: Material, model: Any, log: Any = None) -> None:
    """Resolve /FUNCT references into stored numpy arrays in mat.params."""
    p = mat.params
    funct_ids = p.get("funct_ids", [])
    yfacs = p.get("yfac", [1.0] * len(funct_ids))
    rates = p.get("rates", [0.0] * len(funct_ids))

    if hasattr(model, "functions"):
        functions = model.functions
    elif isinstance(model, dict):
        functions = model
    else:
        functions = {}

    cxs: List[np.ndarray] = []
    cys: List[np.ndarray] = []
    css: List[np.ndarray] = []

    for fid, yf in zip(funct_ids, yfacs):
        fct = functions.get(fid)
        if fct is not None:
            if hasattr(fct, "x"):
                x_arr = np.asarray(fct.x, dtype=float).copy()
                y_arr = np.asarray(fct.y, dtype=float).copy() * yf
            else:
                x_arr = np.asarray(fct[0], dtype=float).copy()
                y_arr = np.asarray(fct[1], dtype=float).copy() * yf
            if hasattr(fct, "slope"):
                s_arr = np.asarray(fct.slope, dtype=float).copy() * yf
            elif len(x_arr) > 1:
                s_arr = np.diff(y_arr) / np.maximum(np.diff(x_arr), _EM20)
            else:
                s_arr = np.zeros(0, dtype=float)
            cxs.append(x_arr)
            cys.append(y_arr)
            css.append(s_arr)

    if len(cxs) > 0:
        p["curve_x"] = cxs
        p["curve_y"] = cys
        p["curve_s"] = css
        p["rates"] = rates

    if p.get("IFUNCE", 0) > 0:
        fct_e = functions.get(p["IFUNCE"])
        if fct_e is not None:
            if hasattr(fct_e, "x"):
                x_e = np.asarray(fct_e.x, dtype=float).copy()
                y_e = np.asarray(fct_e.y, dtype=float).copy()
            else:
                x_e = np.asarray(fct_e[0], dtype=float).copy()
                y_e = np.asarray(fct_e[1], dtype=float).copy()
            s_e = np.diff(y_e) / np.maximum(np.diff(x_e), _EM20) if len(x_e) > 1 else np.zeros(0, dtype=float)
            p["E_curve_x"] = x_e
            p["E_curve_y"] = y_e
            p["E_curve_s"] = s_e


# ============================================================================
# Sound Speed & Allocations
# ============================================================================

def sound_speed(mat: Material, rho: Optional[float] = None, extra: Optional[Dict[str, Any]] = None) -> float:
    """Return sound speed: sqrt(A1 / rho) for shells, sqrt((C1 + 4G/3)/rho) for solids."""
    rho_val = float(rho) if rho is not None else float(getattr(mat, "rho0", 0.0) or mat.params.get("rho0", 0.0))
    if rho_val <= 0.0:
        rho_val = 1.0

    is_solid = False
    if extra is not None:
        if extra.get("is_solid", False) or extra.get("element_type") == "solid" or extra.get("solid", False):
            is_solid = True

    p = mat.params if hasattr(mat, "params") else {}
    if is_solid:
        c1 = float(p.get("C1", p.get("K", 0.0)))
        g = float(p.get("G", 0.0))
        if c1 == 0.0 and g == 0.0:
            e0 = float(p.get("E0", p.get("E", p.get("e", getattr(mat, "E", getattr(mat, "e", 1.0))))))
            nu = float(p.get("nu", getattr(mat, "nu", 0.0)))
            c1 = e0 / (3.0 * max(1.0 - 2.0 * nu, 1e-15))
            g = 0.5 * e0 / max(1.0 + nu, 1e-15)
        return float(math.sqrt(max(0.0, (c1 + 4.0 / 3.0 * g) / rho_val)))
    else:
        a1 = float(p.get("A1", 0.0))
        if a1 == 0.0:
            e0 = float(p.get("E0", p.get("E", p.get("e", getattr(mat, "E", getattr(mat, "e", 1.0))))))
            nu = float(p.get("nu", getattr(mat, "nu", 0.0)))
            a1 = e0 / max(1.0 - nu ** 2, 1e-15)
        return float(math.sqrt(max(0.0, a1 / rho_val)))


def extra_shapes(mat: Material, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Extra allocations required in element layer state."""
    if nip:
        return {
            "pla43": (nip,),
            "uvar43": (nip, 4),
            "off43": (nip,),
            "edot43": (nip,),
            "thk43": (nip,),
        }
    return {
        "pla43": (),
        "uvar43": (4,),
        "off43": (),
        "edot43": (),
        "thk43": (),
    }


# ============================================================================
# Shell Constitutive Update: shell_update (sigeps43c.F)
# ============================================================================

def shell_update(mat: Material, sig: np.ndarray, deps: np.ndarray,
                 *args: Any, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plane-stress constitutive update for shells under Hill plasticity with tables.

    Follows engine/source/materials/mat/mat043/sigeps43c.F.
    """
    if not isinstance(sig, np.ndarray):
        sig = np.array(sig, dtype=float)
    if not isinstance(deps, np.ndarray):
        deps = np.array(deps, dtype=float)

    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig[None, :]
        deps = deps[None, :]

    n = sig.shape[0]
    p = mat.params
    rho0 = float(p.get("rho0", getattr(mat, "rho0", 1.0)))

    # Argument disambiguation
    epsp_in: Optional[np.ndarray] = None
    dt: float = 0.0
    extra: Optional[Dict[str, Any]] = None

    if len(args) >= 3:
        epsp_in = args[0]
        dt = float(args[1])
        extra = args[2]
    elif len(args) == 2:
        if isinstance(args[0], (int, float)) and not isinstance(args[0], np.ndarray):
            dt = float(args[0])
            extra = args[1]
        elif isinstance(args[1], dict) or args[1] is None:
            dt = float(args[0]) if isinstance(args[0], (int, float)) else 0.0
            extra = args[1]
        else:
            epsp_in = args[0]
            dt = float(args[1])
    elif len(args) == 1:
        if isinstance(args[0], (int, float)) and not isinstance(args[0], np.ndarray):
            dt = float(args[0])
        else:
            epsp_in = args[0]

    if "dt" in kwargs:
        dt = float(kwargs["dt"])
    if "extra" in kwargs:
        extra = kwargs["extra"]
    if "epsp" in kwargs and kwargs["epsp"] is not None:
        epsp_in = kwargs["epsp"]

    # History variables from extra
    if extra is not None and "pla43" in extra and extra["pla43"] is not None:
        pla = np.asarray(extra["pla43"], dtype=float).copy()
        if pla.ndim == 0:
            pla = np.full(n, float(pla), dtype=float)
        elif len(pla) == 1 and n > 1:
            pla = np.full(n, float(pla[0]), dtype=float)
    elif epsp_in is not None:
        pla = np.asarray(epsp_in, dtype=float).copy()
        if pla.ndim == 0:
            pla = np.full(n, float(pla), dtype=float)
        elif len(pla) == 1 and n > 1:
            pla = np.full(n, float(pla[0]), dtype=float)
    else:
        pla = np.zeros(n, dtype=float)

    if extra is not None and "uvar43" in extra and extra["uvar43"] is not None:
        uvar = np.asarray(extra["uvar43"], dtype=float)
        if uvar.ndim == 1:
            if n == 1:
                uvar = uvar[None, :4]
            else:
                uvar = np.tile(uvar[:4], (n, 1))
        elif uvar.shape[0] != n:
            uvar = np.tile(uvar[0, :4], (n, 1))
    elif extra is not None and "uv43" in extra and extra["uv43"] is not None:
        uvar = np.asarray(extra["uv43"], dtype=float)
        if uvar.ndim == 1:
            if n == 1:
                uvar = uvar[None, :4]
            else:
                uvar = np.tile(uvar[:4], (n, 1))
        elif uvar.shape[0] != n:
            uvar = np.tile(uvar[0, :4], (n, 1))
    else:
        uvar = np.zeros((n, 4), dtype=float)

    if extra is not None and "off43" in extra and extra["off43"] is not None:
        off = np.asarray(extra["off43"], dtype=float).flatten()
    elif extra is not None and "off" in extra and extra["off"] is not None:
        off = np.asarray(extra["off"], dtype=float).flatten()
    elif extra is not None and "layfail" in extra and extra["layfail"] is not None:
        off = np.asarray(extra["layfail"], dtype=float).flatten()
    else:
        off = np.ones(n, dtype=float)

    if len(off) == 1 and n > 1:
        off = np.full(n, float(off[0]), dtype=float)

    # Dynamic E modulus degradation (sigeps43c.F lines 144-168)
    e_mod, a1_arr, a2_arr, g_arr, g3_arr = _eval_young_modulus(mat, pla)
    nu = float(p.get("nu", 0.0))
    shf = float(extra.get("shf", 5.0 / 6.0) if extra is not None else 5.0 / 6.0)
    gs_arr = g_arr * shf

    # Back-stresses
    alpha_xx = uvar[:, 0].copy()
    alpha_yy = uvar[:, 1].copy()
    alpha_xy = uvar[:, 2].copy()

    # Elastic trial stresses relative to back-stress
    sxx_tr = (sig[:, 0] - alpha_xx) + a1_arr * deps[:, 0] + a2_arr * deps[:, 1]
    syy_tr = (sig[:, 1] - alpha_yy) + a2_arr * deps[:, 0] + a1_arr * deps[:, 1]
    sxy_tr = (sig[:, 2] - alpha_xy) + g_arr * deps[:, 2]

    has_shear = (sig.shape[1] >= 5 and deps.shape[1] >= 5)
    if has_shear:
        syz_tr = sig[:, 3] + gs_arr * deps[:, 3]
        szx_tr = sig[:, 4] + gs_arr * deps[:, 4]
    else:
        syz_tr = None
        szx_tr = None

    # Strain rate measure (sigeps43c.F lines 183-189)
    if dt > 0.0:
        edxx = deps[:, 0] / dt
        edyy = deps[:, 1] / dt
        edxy = deps[:, 2] / dt
        edot_inst = 0.5 * (np.abs(edxx + edyy) + np.sqrt((edxx - edyy) ** 2 + edxy ** 2))
    else:
        edot_inst = np.zeros(n, dtype=float)

    israte = int(p.get("ISRATE", 0))
    asrate = float(p.get("ASRATE", 0.0))
    if israte == 0:
        edot = edot_inst
    else:
        edot_prev = uvar[:, 3]
        # Exponential or factor smoothing
        if 0.0 < asrate <= 1.0:
            alpha_f = asrate
        elif asrate > 1.0 and dt > 0.0:
            alpha_f = 1.0 - math.exp(-asrate * dt)
        else:
            alpha_f = 1.0
        edot = alpha_f * edot_inst + (1.0 - alpha_f) * edot_prev
        uvar[:, 3] = edot

    # Tensile failure strain (sigeps43c.F lines 193-196)
    epsr1 = float(p.get("EPSR1", _INF))
    epsr2 = float(p.get("EPSR2", 2.0 * _INF))
    if extra is not None and "eps" in extra and extra["eps"] is not None:
        eps_tot = np.asarray(extra["eps"], dtype=float)
        exx = eps_tot[:, 0]
        eyy = eps_tot[:, 1]
        exy = eps_tot[:, 2]
        epst = 0.5 * (exx + eyy + np.sqrt((exx - eyy) ** 2 + exy ** 2))
    else:
        epst = pla

    if epsr2 > epsr1:
        fail = np.clip((epsr2 - epst) / (epsr2 - epsr1), 0.0, 1.0)
    else:
        fail = np.ones(n, dtype=float)

    # Yield stress and hardening modulus from curves
    fisokin = float(p.get("FISOKIN", 0.0))
    yld_curve, h_curve, yld0_curve = _eval_yield_stress(mat, pla, edot)

    yld = (1.0 - fisokin) * (fail * yld_curve) + fisokin * (fail * yld0_curve)
    yld = np.maximum(yld, _EM20)
    h_slope = np.maximum(fail * h_curve, 0.0)
    hk = h_slope * fisokin

    # Hill criterion (sigeps43c.F lines 251-255)
    a01 = float(p.get("A01", 1.0))
    a02 = float(p.get("A02", 1.0))
    a03 = float(p.get("A03", 1.0))
    a12 = float(p.get("A12", 3.0))

    s1_tr = a01 * sxx_tr * sxx_tr
    s2_tr = a02 * syy_tr * syy_tr
    s3_tr = a03 * sxx_tr * syy_tr
    axy_tr = a12 * sxy_tr * sxy_tr
    svm = np.sqrt(np.maximum(0.0, s1_tr + s2_tr - s3_tr + axy_tr))

    # Thickness strain (elastic part)
    nnu1 = nu / max(1.0 - nu, _EM20)
    nu5 = 1.0 - nnu1
    dezz_el = -(deps[:, 0] + deps[:, 1]) * nnu1
    dezz_tot = dezz_el.copy()

    # Plastic return mapping
    plastic = (svm > yld) & (off > 0.0)
    sxx_ret = sxx_tr.copy()
    syy_ret = syy_tr.copy()
    sxy_ret = sxy_tr.copy()
    dpla = np.zeros(n, dtype=float)

    if np.any(plastic):
        idx = np.where(plastic)[0]

        for i in idx:
            dpla_j = (svm[i] - yld[i]) / (g3_arr[i] + h_slope[i])
            fhk = (4.0 / 3.0) * hk[i] / a1_arr[i]
            fa01 = a01 * fhk
            fa02 = a02 * fhk
            fa03 = a03 * fhk
            nu1 = nu + 0.5 * fhk
            nu2_i = 1.0 - nu1 * nu1 + fhk * fhk
            nu3_i = nu1 * 0.5
            nu4_i = 0.5 * (1.0 - nu)

            s1 = a01 * nu1 * 2.0 - a03 - fa03
            s2 = a02 * nu1 * 2.0 - a03 - fa03
            s12 = a03 - nu1 * (a01 + a02) + fa03
            s3 = math.sqrt(max(0.0, nu2_i * (a01 - a02) ** 2 + s12 * s12))

            q12 = 0.0 if abs(s1) < _EM20 else -(a01 - a02 + s3 + fa01 - fa02) / s1
            q21 = 0.0 if abs(s2) < _EM20 else (a01 - a02 + s3 + fa01 - fa02) / s2
            jq = 1.0 / (1.0 - q12 * q21)
            jq2 = jq * jq

            a_val = a01 * q12
            b_val = a02 * q21
            a_1 = (a01 + a03 * q21 + b_val * q21) * jq2
            a_2 = (a02 + a03 * q12 + a_val * q12) * jq2
            a_3 = (a_val + b_val) * jq2 * 2.0 + a03 * (jq2 * 2.0 - jq)

            s11 = sxx_tr[i] + syy_tr[i] * q12
            s22 = q21 * sxx_tr[i] + syy_tr[i]

            axx = a_1 * s11 * s11
            ayy = a_2 * s22 * s22
            a_xy = a_3 * s11 * s22
            axy = a12 * sxy_tr[i] * sxy_tr[i]

            a_b = a03 * nu3_i
            b_b = s3 * jq
            b_1 = a02 - a_b - b_b + fa02
            b_2 = a01 - a_b + b_b + fa01
            b_3 = a12 * (nu4_i + 0.5 * fhk)

            h_iso = max(0.0, h_slope[i] - hk[i])

            # Newton-Raphson iterations (sigeps43c.F lines 327-351)
            for _ in range(4):
                if dpla_j > 0.0:
                    yld_i = yld[i] + h_iso * dpla_j
                    dr = a1_arr[i] * dpla_j / yld_i
                    p_1 = 1.0 / (1.0 + b_1 * dr)
                    p_2 = 1.0 / (1.0 + b_2 * dr)
                    p_3 = 1.0 / (1.0 + b_3 * dr)
                    pp1 = p_1 * p_1
                    pp2 = p_2 * p_2
                    pp3 = p_3 * p_3

                    f_res = axx * pp1 + ayy * pp2 - a_xy * p_1 * p_2 + axy * pp3 - yld_i * yld_i
                    df_res = -(
                        (axx * p_1 - a_xy * p_2 * 0.5) * pp1 * b_1
                        + (ayy * p_2 - a_xy * p_1 * 0.5) * pp2 * b_2
                        + axy * pp3 * p_3 * b_3
                    ) * (a1_arr[i] - dr * h_iso) / yld_i - h_iso * yld_i

                    if abs(df_res) > _EM20:
                        dpla_j = max(0.0, dpla_j - f_res * 0.5 / df_res)
                else:
                    dpla_j = 0.0

            dpla[i] = dpla_j
            yld_final = yld[i] + h_iso * dpla_j
            dr0 = dpla_j / yld_final
            dr = a1_arr[i] * dr0

            p_1 = 1.0 / (1.0 + b_1 * dr)
            p_2 = 1.0 / (1.0 + b_2 * dr)
            p_3 = 1.0 / (1.0 + b_3 * dr)

            s1_p = s11 * p_1
            s2_p = s22 * p_2
            sxx_ret[i] = jq * (s1_p - s2_p * q12)
            syy_ret[i] = jq * (s2_p - s1_p * q21)
            sxy_ret[i] = sxy_tr[i] * p_3

            # Plastic thinning (sigeps43c.F lines 369-374)
            s1_thin = a01 * sxx_ret[i] + a02 * syy_ret[i] - a03 * (sxx_ret[i] + syy_ret[i]) * 0.5
            dezz_pl = -nu5 * dpla_j * s1_thin / yld_final
            dezz_tot[i] += dezz_pl

            # Back-stresses (sigeps43c.F lines 375-382)
            s1_bs = a03 * 0.5
            p1_bs = a01 * sxx_ret[i] - s1_bs * syy_ret[i]
            p2_bs = a02 * syy_ret[i] - s1_bs * sxx_ret[i]
            p3_bs = a12 * sxy_ret[i]
            dr0_bs = (2.0 / 3.0) * dr0 * hk[i]

            alpha_xx[i] += (2.0 * p1_bs + p2_bs) * dr0_bs
            alpha_yy[i] += (2.0 * p2_bs + p1_bs) * dr0_bs
            alpha_xy[i] += 0.5 * p3_bs * dr0_bs

    # Update accumulated plastic strain
    pla += dpla

    # Failure element deletion (sigeps43c.F line 387)
    epsmax = float(p.get("EPSMAX", _INF))
    exceeded = (pla > epsmax) & (off > 0.0)
    if np.any(exceeded):
        off[exceeded] = 0.8 * off[exceeded]
        fully_failed = off < 0.1
        off[fully_failed] = 0.0

    # Add back-stresses to returned stresses
    active = off > 0.0
    sig[:, 0] = np.where(active, sxx_ret + alpha_xx, 0.0)
    sig[:, 1] = np.where(active, syy_ret + alpha_yy, 0.0)
    sig[:, 2] = np.where(active, sxy_ret + alpha_xy, 0.0)
    if has_shear:
        sig[:, 3] = np.where(active, syz_tr, 0.0)
        sig[:, 4] = np.where(active, szx_tr, 0.0)

    # Write state back to extra dict
    if extra is not None:
        if "uvar43" in extra and extra["uvar43"] is not None:
            arr = extra["uvar43"]
            if arr.ndim == 1:
                arr[0] = alpha_xx[0]
                arr[1] = alpha_yy[0]
                arr[2] = alpha_xy[0]
                arr[3] = edot[0]
            else:
                arr[:, 0] = alpha_xx
                arr[:, 1] = alpha_yy
                arr[:, 2] = alpha_xy
                arr[:, 3] = edot
        if "uv43" in extra and extra["uv43"] is not None:
            arr = extra["uv43"]
            if arr.ndim == 1:
                arr[0] = alpha_xx[0]
                arr[1] = alpha_yy[0]
                arr[2] = alpha_xy[0]
                arr[3] = edot[0]
            else:
                arr[:, 0] = alpha_xx
                arr[:, 1] = alpha_yy
                arr[:, 2] = alpha_xy
                arr[:, 3] = edot
        if "off43" in extra and extra["off43"] is not None:
            arr = extra["off43"]
            if arr.ndim == 0:
                arr[...] = off[0]
            else:
                arr[:] = off
        if "off" in extra and extra["off"] is not None:
            arr = extra["off"]
            if arr.ndim == 0:
                arr[...] = off[0]
            else:
                arr[:] = off
        if "pla43" in extra and extra["pla43"] is not None:
            arr = extra["pla43"]
            if arr.ndim == 0:
                arr[...] = pla[0]
            else:
                arr[:] = pla
        if "edot43" in extra and extra["edot43"] is not None:
            arr = extra["edot43"]
            if arr.ndim == 0:
                arr[...] = edot[0]
            else:
                arr[:] = edot
        if "thk43" in extra and extra["thk43"] is not None:
            thk0 = extra.get("thk0", 1.0)
            extra["thk43"] = extra["thk43"] + dezz_tot * thk0 * off
        if "thk" in extra and extra["thk"] is not None:
            thk0 = extra.get("thk0", 1.0)
            extra["thk"] = extra["thk"] + dezz_tot * thk0 * off

    if epsp_in is not None and isinstance(epsp_in, np.ndarray):
        epsp_in[:] = pla

    # Sound speed
    rho_cur = np.full(n, rho0, dtype=float)
    if extra is not None and "rho" in extra and extra["rho"] is not None:
        rho_cur = np.asarray(extra["rho"], dtype=float).flatten()
        if len(rho_cur) == 1 and n > 1:
            rho_cur = np.full(n, float(rho_cur[0]), dtype=float)
    c_sound = np.sqrt(a1_arr / np.maximum(rho_cur, _EM20))

    if is_1d:
        return sig[0], pla[0], c_sound[0]
    return sig, pla, c_sound


# ============================================================================
# Solid Constitutive Update: solid_update (sigeps43g.F / 3D Hill)
# ============================================================================

def solid_update(mat: Material, sig: np.ndarray, deps: np.ndarray,
                 *args: Any, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constitutive update for 3D solids and global/thick shells (sigeps43g.F).

    Handles:
      - 3D continuum solids (n, 6): [xx, yy, zz, xy, yz, zx]
      - Generalized shells with moments (n, 8): [xx, yy, xy, yz, zx, mxx, myy, mxy]
    """
    if not isinstance(sig, np.ndarray):
        sig = np.array(sig, dtype=float)
    if not isinstance(deps, np.ndarray):
        deps = np.array(deps, dtype=float)

    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig[None, :]
        deps = deps[None, :]

    n = sig.shape[0]
    p = mat.params
    rho0 = float(p.get("rho0", getattr(mat, "rho0", 1.0)))

    epsp_in: Optional[np.ndarray] = None
    dt: float = 0.0
    extra: Optional[Dict[str, Any]] = None

    if len(args) >= 3:
        epsp_in = args[0]
        dt = float(args[1])
        extra = args[2]
    elif len(args) == 2:
        if isinstance(args[0], (int, float)) and not isinstance(args[0], np.ndarray):
            dt = float(args[0])
            extra = args[1]
        elif isinstance(args[1], dict) or args[1] is None:
            dt = float(args[0]) if isinstance(args[0], (int, float)) else 0.0
            extra = args[1]
        else:
            epsp_in = args[0]
            dt = float(args[1])
    elif len(args) == 1:
        if isinstance(args[0], (int, float)) and not isinstance(args[0], np.ndarray):
            dt = float(args[0])
        else:
            epsp_in = args[0]

    if "dt" in kwargs:
        dt = float(kwargs["dt"])
    if "extra" in kwargs:
        extra = kwargs["extra"]
    if "epsp" in kwargs and kwargs["epsp"] is not None:
        epsp_in = kwargs["epsp"]

    # History variables
    if extra is not None and "pla43" in extra and extra["pla43"] is not None:
        pla = np.asarray(extra["pla43"], dtype=float).copy()
        if pla.ndim == 0:
            pla = np.full(n, float(pla), dtype=float)
        elif len(pla) == 1 and n > 1:
            pla = np.full(n, float(pla[0]), dtype=float)
    elif epsp_in is not None:
        pla = np.asarray(epsp_in, dtype=float).copy()
        if pla.ndim == 0:
            pla = np.full(n, float(pla), dtype=float)
        elif len(pla) == 1 and n > 1:
            pla = np.full(n, float(pla[0]), dtype=float)
    else:
        pla = np.zeros(n, dtype=float)

    if extra is not None and "off43" in extra and extra["off43"] is not None:
        off = np.asarray(extra["off43"], dtype=float).flatten()
    elif extra is not None and "off" in extra and extra["off"] is not None:
        off = np.asarray(extra["off"], dtype=float).flatten()
    else:
        off = np.ones(n, dtype=float)
    if len(off) == 1 and n > 1:
        off = np.full(n, float(off[0]), dtype=float)

    # Check whether this is a generalized shell with bending moments (sigeps43g.F)
    is_generalized_shell = (sig.shape[1] == 8) or (extra is not None and "mom" in extra)

    if is_generalized_shell:
        # sigeps43g.F exact coupled membrane-bending integration
        thk0 = np.ones(n, dtype=float)
        if extra is not None and "thk0" in extra:
            thk0 = np.asarray(extra["thk0"], dtype=float).flatten()
            if len(thk0) == 1 and n > 1:
                thk0 = np.full(n, float(thk0[0]), dtype=float)

        e_mod, a1_arr, a2_arr, g_arr, g3_arr = _eval_young_modulus(mat, pla)
        nu = float(p.get("nu", 0.0))
        shf = float(extra.get("shf", 5.0 / 6.0) if extra is not None else 5.0 / 6.0)
        gs_arr = g_arr * shf

        c1_factor = thk0 * (1.0 / 12.0)
        am1 = a1_arr * c1_factor
        am2 = a2_arr * c1_factor
        gm = g_arr * c1_factor

        signxx = sig[:, 0] + a1_arr * deps[:, 0] + a2_arr * deps[:, 1]
        signyy = sig[:, 1] + a2_arr * deps[:, 0] + a1_arr * deps[:, 1]
        signxy = sig[:, 2] + g_arr * deps[:, 2]

        if sig.shape[1] == 8 and deps.shape[1] == 8:
            signyz = sig[:, 3] + gs_arr * deps[:, 3]
            signzx = sig[:, 4] + gs_arr * deps[:, 4]
            momnxx = sig[:, 5] + am1 * deps[:, 5] + am2 * deps[:, 6]
            momnyy = sig[:, 6] + am2 * deps[:, 5] + am1 * deps[:, 6]
            momnxy = sig[:, 7] + gm * deps[:, 7]
        else:
            signyz = sig[:, 3] + gs_arr * deps[:, 3] if sig.shape[1] >= 5 else np.zeros(n)
            signzx = sig[:, 4] + gs_arr * deps[:, 4] if sig.shape[1] >= 5 else np.zeros(n)
            momnxx = np.zeros(n, dtype=float)
            momnyy = np.zeros(n, dtype=float)
            momnxy = np.zeros(n, dtype=float)

        # Strain rate
        if dt > 0.0:
            edxx = deps[:, 0] / dt
            edyy = deps[:, 1] / dt
            edxy = deps[:, 2] / dt
            edot = 0.5 * (np.abs(edxx + edyy) + np.sqrt((edxx - edyy) ** 2 + edxy ** 2))
        else:
            edot = np.zeros(n, dtype=float)

        yld_c, h_c, _ = _eval_yield_stress(mat, pla, edot)
        yld = np.maximum(yld_c, _EM20)
        h_slope = np.maximum(h_c, 0.0)

        a01 = float(p.get("A01", 1.0))
        a02 = float(p.get("A02", 1.0))
        a03 = float(p.get("A03", 1.0))
        a12 = float(p.get("A12", 3.0))

        c1_pla = pla * e_mod
        gama = 1.5 * (c1_pla + yld) / (1.5 * c1_pla + yld)
        gama2 = gama * gama
        cm = 16.0 * gama2
        cnm = (4.0 / math.sqrt(3.0)) * gama
        qtier = (4.0 / 3.0) * gama2

        s1_m = a01 * (signxx ** 2 + cm * momnxx ** 2)
        s2_m = a02 * (signyy ** 2 + cm * momnyy ** 2)
        s3_m = a03 * (signyy * signxx + cm * momnxx * momnyy)
        anxy = a12 * signxy ** 2
        amxy = a12 * momnxy ** 2 * cm
        f1 = s1_m + s2_m - s3_m + anxy + amxy

        s1_c = a01 * (signxx * momnxx)
        s2_c = a02 * (signyy * momnyy)
        s3_c = a03 * (signxx * momnyy + signyy * momnxx) * 0.5
        anmxy = a12 * signxy * momnxy
        f2 = cnm * (s1_c + s2_c - s3_c + anmxy)
        svm = np.sqrt(np.maximum(0.0, f1 + np.abs(f2)))

        plastic = (svm > yld) & (off > 0.0)
        dpla = np.zeros(n, dtype=float)

        if np.any(plastic):
            idx = np.where(plastic)[0]
            for i in idx:
                nu2_i = 1.0 - nu * nu
                nu3_i = nu * 0.5
                nu4_i = 0.5 - nu3_i

                dpla_i = (svm[i] - yld[i]) / (g3_arr[i] * qtier[i] + h_slope[i])

                s1 = a01 * nu * 2.0 - a03
                s2 = a02 * nu * 2.0 - a03
                s12 = a03 - nu * (a01 + a02)
                s3 = math.sqrt(max(0.0, nu2_i * (a01 - a02) ** 2 + s12 * s12))

                q12 = 0.0 if abs(s1) < _EM20 else -(a01 - a02 + s3) / s1
                q21 = 0.0 if abs(s2) < _EM20 else (a01 - a02 + s3) / s2
                jq = 1.0 / (1.0 - q12 * q21)
                jq2 = jq * jq

                a_v = a01 * q12
                b_v = a02 * q21
                a_1 = (a01 + a03 * q21 + b_v * q21) * jq2
                a_2 = (a02 + a03 * q12 + a_v * q12) * jq2
                a_3 = (a_v + b_v) * jq2 * 2.0 + a03 * (jq2 * 2.0 - jq)

                sn11 = signxx[i] + signyy[i] * q12
                sn22 = q21 * signxx[i] + signyy[i]
                sm11 = momnxx[i] + momnyy[i] * q12
                sm22 = q21 * momnxx[i] + momnyy[i]

                anxx_i = a_1 * sn11 ** 2
                anyy_i = a_2 * sn22 ** 2
                an_xy_i = a_3 * sn11 * sn22

                amxx_i = a_1 * sm11 ** 2 * cm[i]
                amyy_i = a_2 * sm22 ** 2 * cm[i]
                am_xy_i = a_3 * sm11 * sm22 * cm[i]

                anmxx_i = a_1 * sn11 * sm11 * cnm[i]
                anmyy_i = a_2 * sn22 * sm22 * cnm[i]
                anm_xy_i = a_3 * sn11 * sm22 * cnm[i] * 0.5
                amn_xy_i = a_3 * sm11 * sn22 * cnm[i] * 0.5

                b_1 = a02 - a03 * nu3_i - s3 * jq
                b_2 = a01 - a03 * nu3_i + s3 * jq
                b_3 = a12 * nu4_i

                # Newton iterations per sigeps43g.F lines 388-462
                twop444 = 22.0 / 9.0
                zep444 = 4.0 / 9.0
                onep8333 = 11.0 / 6.0

                for _ in range(3):
                    yld_i = yld[i] + h_slope[i] * dpla_i
                    dr = a1_arr[i] * dpla_i / yld_i
                    xp = [b_1 * dr, b_2 * dr, b_3 * dr]
                    c1_jac = 1.0 + qtier[i]
                    b_jac = twop444 * gama2[i]

                    jac = [1.0] * 3
                    jac_i = [1.0] * 3
                    jac_2 = [1.0] * 3
                    fn = [1.0] * 3
                    fm = [1.0] * 3
                    fnm = [1.0] * 3
                    dfn = [0.0] * 3
                    dfm = [0.0] * 3
                    dfnm = [0.0] * 3
                    djac = [0.0] * 3

                    for k in range(3):
                        djac[k] = c1_jac + b_jac * xp[k]
                        jac[k] = 1.0 + (djac[k] + c1_jac) * xp[k] * 0.5
                        jac_i[k] = 1.0 / max(jac[k], _EM20)
                        jac_2[k] = jac_i[k] * jac_i[k]

                        a_fn = xp[k] * zep444 * gama2[i]
                        dfn[k] = 5.5 + 16.5 * a_fn
                        fn[k] = 1.0 + (dfn[k] + 5.5) * a_fn * 0.5

                        a_fm = onep8333 * xp[k]
                        dfm[k] = onep8333 + a_fm
                        fm[k] = 1.0 + (dfm[k] * xp[k] + a_fm) * 0.5

                        dfnm[k] = -b_jac * xp[k]
                        fnm[k] = 1.0 + dfnm[k] * xp[k] * 0.5

                    sfn = jac_2[0] * fn[0] * (anxx_i - an_xy_i) + jac_2[1] * fn[1] * (anyy_i - an_xy_i) + jac_2[2] * fn[2] * anxy[i]
                    sfm = jac_2[0] * fm[0] * (amxx_i - am_xy_i) + jac_2[1] * fm[1] * (amyy_i - am_xy_i) + jac_2[2] * fm[2] * amxy[i]
                    sfnm = jac_2[0] * fnm[0] * (anmxx_i - anm_xy_i) + jac_2[1] * fnm[1] * (anmyy_i - amn_xy_i) + jac_2[2] * fnm[2] * anmxy[i]

                    c_tol = abs(sfnm) / max(sfn, sfm, _EM20)
                    s_sign = 0.0 if c_tol < 1e-6 else (-1.0 if sfnm < 0.0 else 1.0)

                    f_val = sfn + sfm + s_sign * sfnm - yld_i * yld_i
                    c1_const = zep444 * gama2[i]

                    dsfn = 0.0
                    dsfm = 0.0
                    dsfnm = 0.0
                    for k, (b_k, comp_n, comp_m, comp_nm) in enumerate([
                        (b_1, anxx_i - an_xy_i, amxx_i - am_xy_i, anmxx_i - anm_xy_i),
                        (b_2, anyy_i - an_xy_i, amyy_i - am_xy_i, anmyy_i - amn_xy_i),
                        (b_3, anxy[i], amxy[i], anmxy[i])
                    ]):
                        s12_k = jac_2[k] * b_k
                        s_k = s12_k * c1_const
                        aa_k = 2.0 * jac_i[k] * djac[k] * s12_k
                        dsfn += (s_k * dfn[k] - aa_k * fn[k]) * comp_n
                        dsfm += (s_k * dfm[k] - aa_k * fm[k]) * comp_m
                        dsfnm += (s_k * dfnm[k] - aa_k * fnm[k]) * comp_nm

                    df_val = (dsfn + dsfm + s_sign * dsfnm) * (a1_arr[i] - dr * h_slope[i]) / yld_i - 2.0 * h_slope[i] * yld_i
                    if abs(df_val) > _EM20:
                        dpla_i = max(0.0, dpla_i - f_val / df_val)

                dpla[i] = dpla_i
                yld_f = yld[i] + h_slope[i] * dpla_i
                dr = a1_arr[i] * dpla_i / yld_f
                xp = [b_1 * dr, b_2 * dr, b_3 * dr]
                c1_jac = 1.0 + qtier[i]
                b_jac = twop444 * gama2[i]

                jac = [1.0] * 3
                jac_i = [1.0] * 3
                jac_2 = [1.0] * 3
                fnm = [1.0] * 3
                pn = [1.0] * 3
                p_m = [1.0] * 3
                pnm1 = [0.0] * 3
                pnm2 = [0.0] * 3

                for k in range(3):
                    a_term = b_jac * xp[k]
                    jac[k] = 1.0 + c1_jac * xp[k] + a_term * xp[k]
                    jac_i[k] = 1.0 / max(jac[k], _EM20)
                    jac_2[k] = jac_i[k] * jac_i[k]
                    fnm[k] = 1.0 - a_term * xp[k] * 0.5
                    a_p = xp[k] * jac_i[k]
                    pn[k] = jac_i[k] + qtier[i] * a_p
                    p_m[k] = jac_i[k] + a_p
                    pnm1[k] = -(2.0 / math.sqrt(3.0)) * gama[i] * a_p
                    pnm2[k] = pnm1[k] * (1.0 / 12.0)

                sn1 = sn11 * pn[0] + sm11 * pnm1[0] * s_sign
                sn2 = sn22 * pn[1] + sm22 * pnm1[1] * s_sign
                s3_xy = signxy[i] * pn[2] + momnxy[i] * pnm1[2] * s_sign

                sm1 = sm11 * p_m[0] + sn11 * pnm2[0] * s_sign
                sm2 = sm22 * p_m[1] + sn22 * pnm2[1] * s_sign
                m3_xy = momnxy[i] * p_m[2] + signxy[i] * pnm2[2] * s_sign

                signxx[i] = jq * (sn1 - sn2 * q12)
                signyy[i] = jq * (sn2 - sn1 * q21)
                signxy[i] = s3_xy
                momnxx[i] = jq * (sm1 - sm2 * q12)
                momnyy[i] = jq * (sm2 - sm1 * q21)
                momnxy[i] = m3_xy

        pla += dpla
        epsmax = float(p.get("EPSMAX", _INF))
        exceeded = (pla > epsmax) & (off > 0.0)
        if np.any(exceeded):
            off[exceeded] = 0.8 * off[exceeded]
            off[off < 0.1] = 0.0

        active = off > 0.0
        sig[:, 0] = np.where(active, signxx, 0.0)
        sig[:, 1] = np.where(active, signyy, 0.0)
        sig[:, 2] = np.where(active, signxy, 0.0)
        if sig.shape[1] >= 5:
            sig[:, 3] = np.where(active, signyz, 0.0)
            sig[:, 4] = np.where(active, signzx, 0.0)
        if sig.shape[1] == 8:
            sig[:, 5] = np.where(active, momnxx, 0.0)
            sig[:, 6] = np.where(active, momnyy, 0.0)
            sig[:, 7] = np.where(active, momnxy, 0.0)

        c_sound = np.sqrt(a1_arr / max(rho0, _EM20))

    else:
        # Standard 3D continuum solid (n, 6): [xx, yy, zz, xy, yz, zx]
        e0 = float(p.get("E0", p.get("E", 1.0)))
        nu = float(p.get("nu", 0.0))
        c1_bulk = e0 / (3.0 * (1.0 - 2.0 * nu))
        g_shear = 0.5 * e0 / (1.0 + nu)

        # Dynamic E degradation
        e_mod, _, _, g_arr, g3_arr = _eval_young_modulus(mat, pla)
        k_arr = e_mod / (3.0 * (1.0 - 2.0 * nu))

        # Elastic trial
        p_old = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
        deps_v = deps[:, 0] + deps[:, 1] + deps[:, 2]
        p_new = p_old + k_arr * deps_v

        # Deviatoric stresses
        sxx_old = sig[:, 0] - p_old
        syy_old = sig[:, 1] - p_old
        szz_old = sig[:, 2] - p_old

        dev_deps_xx = deps[:, 0] - deps_v / 3.0
        dev_deps_yy = deps[:, 1] - deps_v / 3.0
        dev_deps_zz = deps[:, 2] - deps_v / 3.0

        sxx_tr = sxx_old + 2.0 * g_arr * dev_deps_xx
        syy_tr = syy_old + 2.0 * g_arr * dev_deps_yy
        szz_tr = szz_old + 2.0 * g_arr * dev_deps_zz
        sxy_tr = sig[:, 3] + g_arr * deps[:, 3]
        syz_tr = sig[:, 4] + g_arr * deps[:, 4]
        szx_tr = sig[:, 5] + g_arr * deps[:, 5]

        # Strain rate
        if dt > 0.0:
            edot = np.sqrt((2.0 / 3.0) * (dev_deps_xx ** 2 + dev_deps_yy ** 2 + dev_deps_zz ** 2
                                          + 0.5 * (deps[:, 3] ** 2 + deps[:, 4] ** 2 + deps[:, 5] ** 2))) / dt
        else:
            edot = np.zeros(n, dtype=float)

        yld_c, h_c, _ = _eval_yield_stress(mat, pla, edot)
        yld = np.maximum(yld_c, _EM20)
        h_slope = np.maximum(h_c, 0.0)

        # 3D Hill 1948 criterion
        a01 = float(p.get("A01", 1.0))
        a02 = float(p.get("A02", 1.0))
        a03 = float(p.get("A03", 1.0))
        a12 = float(p.get("A12", 3.0))

        h_h = 0.5 * a03
        g_h = a01 - h_h
        f_h = a02 - h_h
        n_h = 0.5 * a12
        l_h = n_h
        m_h = n_h

        svm_sq = (f_h * (syy_tr - szz_tr) ** 2 + g_h * (szz_tr - sxx_tr) ** 2 + h_h * (sxx_tr - syy_tr) ** 2
                  + 2.0 * l_h * syz_tr ** 2 + 2.0 * m_h * szx_tr ** 2 + 2.0 * n_h * sxy_tr ** 2)
        svm = np.sqrt(np.maximum(0.0, svm_sq))

        plastic = (svm > yld) & (off > 0.0)
        dpla = np.zeros(n, dtype=float)

        if np.any(plastic):
            idx = np.where(plastic)[0]
            dl = (svm[idx] - yld[idx]) / (3.0 * g_arr[idx] + h_slope[idx])
            dl = np.maximum(0.0, dl)
            dpla[idx] = dl

            scale = (yld[idx] + h_slope[idx] * dl) / np.maximum(svm[idx], _EM20)
            sxx_tr[idx] *= scale
            syy_tr[idx] *= scale
            szz_tr[idx] *= scale
            sxy_tr[idx] *= scale
            syz_tr[idx] *= scale
            szx_tr[idx] *= scale

        pla += dpla
        epsmax = float(p.get("EPSMAX", _INF))
        exceeded = (pla > epsmax) & (off > 0.0)
        if np.any(exceeded):
            off[exceeded] = 0.8 * off[exceeded]
            off[off < 0.1] = 0.0

        active = off > 0.0
        sig[:, 0] = np.where(active, sxx_tr + p_new, 0.0)
        sig[:, 1] = np.where(active, syy_tr + p_new, 0.0)
        sig[:, 2] = np.where(active, szz_tr + p_new, 0.0)
        sig[:, 3] = np.where(active, sxy_tr, 0.0)
        sig[:, 4] = np.where(active, syz_tr, 0.0)
        sig[:, 5] = np.where(active, szx_tr, 0.0)

        c_sound = np.sqrt((k_arr + 4.0 / 3.0 * g_arr) / max(rho0, _EM20))

    # Write state back
    if extra is not None:
        if "pla43" in extra and extra["pla43"] is not None:
            arr = extra["pla43"]
            if arr.ndim == 0:
                arr[...] = pla[0]
            else:
                arr[:] = pla
        if "off43" in extra and extra["off43"] is not None:
            arr = extra["off43"]
            if arr.ndim == 0:
                arr[...] = off[0]
            else:
                arr[:] = off
        if "off" in extra and extra["off"] is not None:
            arr = extra["off"]
            if arr.ndim == 0:
                arr[...] = off[0]
            else:
                arr[:] = off

    if epsp_in is not None and isinstance(epsp_in, np.ndarray):
        epsp_in[:] = pla

    if is_1d:
        return sig[0], pla[0], c_sound[0]
    return sig, pla, c_sound


# ============================================================================
# Consistent Algorithmic Tangents
# ============================================================================

def shell_membrane_tangent(mat: Material) -> np.ndarray:
    """Constant (3, 3) elastic plane-stress membrane constitutive matrix."""
    p = mat.params if hasattr(mat, "params") else {}
    e0 = float(p.get("E0", p.get("E", p.get("e", getattr(mat, "E", getattr(mat, "e", 1.0))))))
    nu = float(p.get("nu", getattr(mat, "nu", 0.0)))
    a1 = e0 / (1.0 - nu ** 2)
    a2 = nu * a1
    g = 0.5 * e0 / (1.0 + nu)

    return np.array([
        [a1, a2, 0.0],
        [a2, a1, 0.0],
        [0.0, 0.0, g],
    ], dtype=float)


def consistent_shell_tangent(mat: Material, sig: np.ndarray,
                             epsp: Optional[np.ndarray] = None,
                             epsp_incr: Optional[np.ndarray] = None,
                             extra: Optional[Dict[str, Any]] = None,
                             deps: Optional[np.ndarray] = None,
                             dt: float = 0.0,
                             symmetric: bool = False) -> np.ndarray:
    """Consistent algorithmic plane-stress tangent operator (n, 3, 3)."""
    if not isinstance(sig, np.ndarray):
        sig = np.array(sig, dtype=float)

    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig[None, :]

    n = sig.shape[0]
    c_el = shell_membrane_tangent(mat)
    d_tangent = np.tile(c_el, (n, 1, 1))

    if deps is not None:
        # High-accuracy numerical central difference verified tangent
        deps_arr = np.asarray(deps, dtype=float)
        if deps_arr.ndim == 1:
            deps_arr = deps_arr[None, :]
        h = 1.0e-7

        for i in range(n):
            d_num = np.zeros((3, 3), dtype=float)
            sig_i = sig[i].copy()
            deps_i = deps_arr[i].copy()
            epsp_i = float(epsp[i]) if epsp is not None else 0.0

            for j in range(3):
                deps_p = deps_i.copy()
                deps_m = deps_i.copy()
                deps_p[j] += h
                deps_m[j] -= h

                ext_p = {**extra} if extra is not None else {}
                ext_m = {**extra} if extra is not None else {}
                if "uvar43" in ext_p:
                    ext_p["uvar43"] = ext_p["uvar43"].copy()
                if "uvar43" in ext_m:
                    ext_m["uvar43"] = ext_m["uvar43"].copy()

                sig_p, _, _ = shell_update(mat, sig_i.copy(), deps_p, epsp=epsp_i, dt=dt, extra=ext_p)
                sig_m, _, _ = shell_update(mat, sig_i.copy(), deps_m, epsp=epsp_i, dt=dt, extra=ext_m)
                d_num[:, j] = (sig_p[:3] - sig_m[:3]) / (2.0 * h)

            if symmetric:
                d_num = 0.5 * (d_num + d_num.T)
            d_tangent[i] = d_num

        return d_tangent[0] if is_1d else d_tangent

    if epsp_incr is None:
        return d_tangent[0] if is_1d else d_tangent

    epsp_incr_arr = np.asarray(epsp_incr, dtype=float).flatten()
    if epsp_incr_arr.ndim == 0 or len(epsp_incr_arr) == 1:
        epsp_incr_arr = np.full(n, float(epsp_incr_arr.item() if epsp_incr_arr.ndim == 0 else epsp_incr_arr[0]), dtype=float)

    plastic = epsp_incr_arr > 0.0
    if not np.any(plastic):
        return d_tangent[0] if is_1d else d_tangent

    p = mat.params
    e0 = float(p.get("E0", p.get("E", 1.0)))
    a01 = float(p.get("A01", 1.0))
    a02 = float(p.get("A02", 1.0))
    a03 = float(p.get("A03", 1.0))
    a12 = float(p.get("A12", 3.0))

    p_hill = np.array([
        [a01, -0.5 * a03, 0.0],
        [-0.5 * a03, a02, 0.0],
        [0.0, 0.0, a12],
    ], dtype=float)

    idx = np.where(plastic)[0]
    for ii in idx:
        dl = float(epsp_incr_arr[ii])
        s_c = sig[ii, :3]
        seq_c = math.sqrt(max(0.0, float(s_c @ p_hill @ s_c)))
        seq_c = max(seq_c, _EM20)

        q_tr = seq_c + e0 * dl
        scale = seq_c / max(q_tr, _EM20)
        s_tr = s_c / max(scale, _EM20)

        g_vec = c_el @ p_hill @ s_tr
        denom = q_tr * q_tr
        gamma = -scale / max(denom, _EM20)

        rank1 = np.outer(s_tr, g_vec)
        d_tangent[ii] = scale * c_el + gamma * rank1

        if symmetric:
            d_tangent[ii] = 0.5 * (d_tangent[ii] + d_tangent[ii].T)

    return d_tangent[0] if is_1d else d_tangent


def consistent_solid_tangent(mat: Material, sig: np.ndarray,
                             epsp: Optional[np.ndarray] = None,
                             epsp_incr: Optional[np.ndarray] = None,
                             extra: Optional[Dict[str, Any]] = None,
                             deps: Optional[np.ndarray] = None,
                             dt: float = 0.0,
                             symmetric: bool = False) -> np.ndarray:
    """Consistent algorithmic solid tangent operator (n, 6, 6)."""
    if not isinstance(sig, np.ndarray):
        sig = np.array(sig, dtype=float)

    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig[None, :]

    n = sig.shape[0]
    p = mat.params
    e0 = float(p.get("E0", p.get("E", 1.0)))
    nu = float(p.get("nu", 0.0))
    k_bulk = e0 / (3.0 * (1.0 - 2.0 * nu))
    g_shear = 0.5 * e0 / (1.0 + nu)

    # Elastic 6x6 tangent
    c_el = np.zeros((6, 6), dtype=float)
    c11 = k_bulk + 4.0 / 3.0 * g_shear
    c12 = k_bulk - 2.0 / 3.0 * g_shear
    c_el[0, 0] = c_el[1, 1] = c_el[2, 2] = c11
    c_el[0, 1] = c_el[0, 2] = c_el[1, 0] = c_el[1, 2] = c_el[2, 0] = c_el[2, 1] = c12
    c_el[3, 3] = c_el[4, 4] = c_el[5, 5] = g_shear

    d_tangent = np.tile(c_el, (n, 1, 1))

    if deps is not None:
        deps_arr = np.asarray(deps, dtype=float)
        if deps_arr.ndim == 1:
            deps_arr = deps_arr[None, :]
        h = 1.0e-7

        for i in range(n):
            d_num = np.zeros((6, 6), dtype=float)
            sig_i = sig[i].copy()
            deps_i = deps_arr[i].copy()
            epsp_i = float(epsp[i]) if epsp is not None else 0.0

            for j in range(6):
                deps_p = deps_i.copy()
                deps_m = deps_i.copy()
                deps_p[j] += h
                deps_m[j] -= h

                ext_p = {**extra} if extra is not None else {}
                ext_m = {**extra} if extra is not None else {}

                sig_p, _, _ = solid_update(mat, sig_i.copy(), deps_p, epsp=epsp_i, dt=dt, extra=ext_p)
                sig_m, _, _ = solid_update(mat, sig_i.copy(), deps_m, epsp=epsp_i, dt=dt, extra=ext_m)
                d_num[:, j] = (sig_p[:6] - sig_m[:6]) / (2.0 * h)

            if symmetric:
                d_num = 0.5 * (d_num + d_num.T)
            d_tangent[i] = d_num

        return d_tangent[0] if is_1d else d_tangent

    return d_tangent[0] if is_1d else d_tangent


# ============================================================================
# Registration
# ============================================================================

def _register() -> None:
    """Register LAW43 in MAT_PHYSICS_REGISTRY."""
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for key in (43, "43", "LAW43", "HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "LAW43_HILL_TAB"):
            MAT_PHYSICS_REGISTRY[key] = build_law43
    except Exception:
        pass


_register()
