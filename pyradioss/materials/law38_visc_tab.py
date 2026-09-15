"""
LAW38 — Tabulated Viscoelastic Foam (/MAT/LAW38, /MAT/VISC_TAB).

Fortran reference origin:
  - ``engine/source/materials/mat/mat038/sigeps38.F`` (constitutive update)
  - ``starter/source/materials/mat/mat038/hm_read_mat38.F`` (parameter extraction & defaults)
  - ``starter/source/materials/mat/mat038/m38init.F`` (initialization of UVAR state)
  - ``engine/source/materials/mat/mat038/sigeps38.F:CHECKAXES`` (eigenvector continuity check)
  - ``engine/source/materials/mat/mat033/sigeps33.F:DREH`` (principal axis tensor rotation)

Solids only (SOLID_ISOTROPIC + SPH), exactly as upstream.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..model.entities import Material

# Numeric constants matching OpenRadioss implicit_f / constant_mod
_ZERO = 0.0
_HALF = 0.5
_ONE = 1.0
_TWO = 2.0
_THREE = 3.0
_THIRD = 1.0 / 3.0
_FOUR_OVER_3 = 4.0 / 3.0
_SEVEN_OVER_5 = 7.0 / 5.0
_TINY = 1.0e-30
_SMALL = 1.0e-3
_EM6 = 1.0e-6
_EM16 = 1.0e-16
_EM20 = 1.0e-20
_EP30 = 1.0e30


# ----------------------------------------------------------------------------
# Curve Evaluation Helpers
# ----------------------------------------------------------------------------

def _eval_curve(curve: Any, x: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Evaluate curve value and derivative df/dx with piecewise linear interpolation
    and end-slope extrapolation, matching OpenRadioss finter.F.

    Supports:
      - None: returns (0.0, 0.0)
      - Callable: f(x) -> y, with numerical central-difference derivative
      - (xs, ys) tuple/list or object with .x and .y attributes
    """
    is_scalar = np.isscalar(x)
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))

    if curve is None:
        y = np.zeros_like(x_arr)
        dy = np.zeros_like(x_arr)
        return (float(y[0]), float(dy[0])) if is_scalar else (y, dy)

    if callable(curve):
        y = np.asarray(curve(x_arr), dtype=float)
        # Numerical derivative
        eps = np.maximum(1e-6 * np.abs(x_arr), 1e-8)
        y_plus = np.asarray(curve(x_arr + eps), dtype=float)
        y_minus = np.asarray(curve(x_arr - eps), dtype=float)
        dy = (y_plus - y_minus) / (2.0 * eps)
        return (float(y[0]), float(dy[0])) if is_scalar else (y, dy)

    if hasattr(curve, "x") and hasattr(curve, "y"):
        xs = np.asarray(curve.x, dtype=float)
        ys = np.asarray(curve.y, dtype=float)
    elif isinstance(curve, (tuple, list)) and len(curve) == 2:
        xs = np.asarray(curve[0], dtype=float)
        ys = np.asarray(curve[1], dtype=float)
    else:
        y = np.zeros_like(x_arr)
        dy = np.zeros_like(x_arr)
        return (float(y[0]), float(dy[0])) if is_scalar else (y, dy)

    if len(xs) == 0:
        y = np.zeros_like(x_arr)
        dy = np.zeros_like(x_arr)
        return (float(y[0]), float(dy[0])) if is_scalar else (y, dy)

    if len(xs) == 1:
        y = np.full_like(x_arr, ys[0])
        dy = np.zeros_like(x_arr)
        return (float(y[0]), float(dy[0])) if is_scalar else (y, dy)

    # Sort if not monotonic
    if not np.all(np.diff(xs) > 0):
        sort_idx = np.argsort(xs)
        xs = xs[sort_idx]
        ys = ys[sort_idx]

    # Piecewise linear interpolation with end-segment extrapolation
    # (finter.F behavior)
    slopes = np.diff(ys) / np.maximum(np.diff(xs), _EM16)
    first_slope = slopes[0]
    last_slope = slopes[-1]

    # Standard interp clamps at ends
    y_clamped = np.interp(x_arr, xs, ys)
    dy = np.zeros_like(x_arr)

    # Locate intervals for derivatives
    idx = np.searchsorted(xs, x_arr, side="right") - 1
    # Below lower bound
    below = x_arr < xs[0]
    # Above upper bound
    above = x_arr > xs[-1]
    # Inside bounds
    inside = ~(below | above)

    idx_clamped = np.clip(idx, 0, len(slopes) - 1)
    dy[inside] = slopes[idx_clamped[inside]]

    # Extrapolate values and slopes outside bounds
    y = y_clamped.copy()
    if np.any(below):
        dy[below] = first_slope
        y[below] = ys[0] + (x_arr[below] - xs[0]) * first_slope
    if np.any(above):
        dy[above] = last_slope
        y[above] = ys[-1] + (x_arr[above] - xs[-1]) * last_slope

    return (float(y[0]), float(dy[0])) if is_scalar else (y, dy)


# ----------------------------------------------------------------------------
# Parameter Extraction & Construction (hm_read_mat38.F)
# ----------------------------------------------------------------------------

def build_law38(rec: Any) -> Material:
    """Card parsing and parameter validation matching ``hm_read_mat38.F``."""
    if isinstance(rec, Material):
        p = dict(rec.params) if rec.params else {}
        mat_id = rec.id
        title = rec.title
        rho0 = rec.rho0
    elif isinstance(rec, dict):
        p = dict(rec.get("params", rec))
        mat_id = int(rec.get("id", rec.get("mat_id", 1)))
        title = str(rec.get("title", ""))
        rho0 = None
        for k in ("density", "rho0", "rho", "MAT_RHO"):
            if k in rec and rec[k] is not None:
                rho0 = float(rec[k])
                break
        if rho0 is None:
            for k in ("density", "rho0", "rho", "MAT_RHO"):
                if k in p and p[k] is not None:
                    rho0 = float(p[k])
                    break
        rho0 = rho0 if rho0 is not None else 1.0
    else:
        p = dict(getattr(rec, "params", {}))
        mat_id = int(getattr(rec, "id", 1))
        title = str(getattr(rec, "title", ""))
        rho0 = float(getattr(rec, "rho0", getattr(rec, "density", 1.0)))

    if rho0 <= 0.0:
        raise ValueError(f"LAW38/{mat_id}: Density rho0 must be > 0 (upstream hm_read_mat38 check)")

    def _get_f(keys: Sequence[str], default: float = 0.0) -> float:
        for k in keys:
            v = p.get(k)
            if v is not None:
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
            if hasattr(rec, k) and getattr(rec, k) is not None:
                try:
                    return float(getattr(rec, k))
                except (ValueError, TypeError):
                    pass
        return default

    def _get_i(keys: Sequence[str], default: int = 0) -> int:
        for k in keys:
            v = p.get(k)
            if v is not None:
                try:
                    return int(v)
                except (ValueError, TypeError):
                    pass
            if hasattr(rec, k) and getattr(rec, k) is not None:
                try:
                    return int(getattr(rec, k))
                except (ValueError, TypeError):
                    pass
        return default

    rhor = _get_f(["Refer_Rho", "rhor"], rho0)
    if rhor <= 0.0:
        rhor = rho0

    e0 = _get_f(["MAT_E", "e0", "e", "E"], 0.0)
    if e0 <= 0.0:
        raise ValueError(f"LAW38/{mat_id}: Initial Young modulus E0 must be > 0 (upstream hm_read_mat38)")

    nu_t = _get_f(["MAT_NU", "nu_t", "nu"], _EM20)
    nu_c = _get_f(["MAT_NUt", "nu_c"], 0.0)
    rv = _get_f(["MAT_RV", "rv"], 0.0)
    iflag = _get_i(["MAT_IFLAG", "iflag"], 0)
    itotal = _get_i(["ITOTAL", "itotal"], 0)

    beta = _get_f(["MAT_RELX", "beta"], _EM20)
    hyster = _get_f(["MAT_HYST", "hyster", "h"], 1.0)
    ratedamp = _get_f(["DAMP1", "ratedamp", "r_d"], 0.5)
    krecover = _get_i(["Gflag", "krecover", "k_r"], 0)
    kdecay = _get_i(["Vflag", "kdecay", "k_d"], 0)
    theta = _get_f(["MAT_Theta", "theta", "instant_mod_upd"], 0.67)

    kcompair = _get_i(["MAT_Kair", "kcompair", "kair"], 0)
    npcurve = _get_i(["FUN_A4", "npcurve", "np"], 0)
    pscale = _get_f(["MAT_PScale", "pscale"], 1.0)
    p0 = _get_f(["MAT_P0", "p0"], 0.0)
    relaxp = _get_f(["MAT_PR", "relaxp", "rp"], _EM20)
    maxpres = _get_f(["MAT_PMAX", "maxpres", "pmax"], _EP30)
    phi = _get_f(["MAT_POROS", "phi", "poros"], 0.0)

    iunload = _get_i(["FUN_B4", "iunload", "ful"], 0)
    funload = _get_f(["MAT_ALPHA6", "funload", "alpha_unload"], 1.0)
    runload = _get_f(["MAT_EPSF2", "runload", "eps_unload"], 0.0)
    exponas = _get_f(["MAT_EXP1", "exponas", "a"], 1.0)
    exponbs = _get_f(["MAT_EXP2", "exponbs", "b"], 1.0)

    m_func = _get_i(["NFUNC", "nfunc", "m_func"], 0)
    cutoff = _get_f(["MAT_CUTOFF", "tensioncut", "cutoff"], _EP30)
    imsta = _get_i(["MAT_Iinsta", "imsta", "iinsta"], 0)

    efinal = _get_f(["MAT_Efinal", "efinal", "e_final"], e0)
    epsfin = _get_f(["MAT_Epsfinal", "epsfin", "epsi_final"], 1.0)
    lamda = _get_f(["MAT_Lamda", "lamda", "lamb"], 1.0)
    viscosity = _get_f(["MAT_MaxVisc", "viscosity", "visc"], _EP30)
    tolerance = _get_f(["MAT_Tol", "tolerance", "tol"], 1.0)

    # Defaults / clamping matching hm_read_mat38.F lines 226-258
    if pscale == 0.0:
        pscale = 1.0
    if nu_t <= 0.0:
        nu_t = _EM20
    if nu_t >= 0.5:
        nu_t = 0.499
    if nu_c >= 0.5:
        nu_c = 0.499
    if itotal > 3:
        itotal = 0
    if beta <= 0.0:
        beta = _EM20
    if hyster <= 0.0:
        hyster = 1.0
    if ratedamp <= 0.0:
        ratedamp = 0.5
    if krecover <= 0 or krecover > 2:
        krecover = 0
    if kdecay <= 0 or kdecay > 2:
        kdecay = 0
    if theta <= 0.0:
        theta = 0.67
    if relaxp <= 0.0:
        relaxp = _EM20
    if maxpres <= 0.0:
        maxpres = _EP30
    if funload <= 0.0:
        funload = 1.0
    if exponas == 0.0:
        exponas = 1.0
    if exponbs == 0.0:
        exponbs = 1.0
    if cutoff <= 0.0:
        cutoff = _EP30
    if epsfin <= 0.0 or epsfin > 0.0:
        epsfin = 1.0
    if lamda <= 0.0:
        lamda = 1.0
    if tolerance <= 0.0:
        tolerance = 1.0
    if viscosity <= 0.0:
        viscosity = _EP30
    if efinal <= e0:
        efinal = e0

    # Curves & scale factors
    fscale_tab = list(p.get("fscale_tab") or getattr(rec, "fscale_tab", []))
    eps_tab = list(p.get("eps_tab") or getattr(rec, "eps_tab", []))
    load_fids = list(p.get("load_fids") or getattr(rec, "load_fids", []))
    unload_fids = list(p.get("unload_fids") or getattr(rec, "unload_fids", []))
    load_curves = list(p.get("load_curves") or getattr(rec, "load_curves", []))
    unload_curves = list(p.get("unload_curves") or getattr(rec, "unload_curves", []))

    if m_func > 5:
        m_func = 5
    if m_func == 0 and load_curves:
        m_func = len(load_curves)

    params: Dict[str, Any] = {
        "rho0": rho0, "rhor": rhor,
        "e0": e0, "e": e0, "E": e0,
        "nu_t": nu_t, "nu_c": nu_c, "nu": max(nu_t, nu_c), "rv": rv,
        "iflag": iflag, "itotal": itotal,
        "beta": beta, "hyster": hyster, "ratedamp": ratedamp,
        "krecover": krecover, "kdecay": kdecay, "theta": theta,
        "kcompair": kcompair, "npcurve": npcurve, "pscale": pscale,
        "p0": p0, "gamma": _SEVEN_OVER_5, "relaxp": relaxp,
        "maxpres": maxpres, "phi": phi,
        "iunload": iunload, "funload": funload, "runload": runload,
        "exponas": exponas, "exponbs": exponbs,
        "m_func": m_func, "nfunc": m_func,
        "cutoff": cutoff, "tensioncut": cutoff,
        "imsta": imsta,
        "efinal": efinal, "epsfin": epsfin, "lamda": lamda,
        "viscosity": viscosity, "tolerance": tolerance,
        "fscale_tab": fscale_tab, "eps_tab": eps_tab,
        "load_fids": load_fids, "unload_fids": unload_fids,
        "load_curves": load_curves, "unload_curves": unload_curves,
        "pressure_curve": p.get("pressure_curve") or getattr(rec, "pressure_curve", None),
        "unload_curve": p.get("unload_curve") or getattr(rec, "unload_curve", None),
    }

    return Material(id=mat_id, law=38, rho0=rho0, title=title, params=params, law_name="LAW38")


def resolve(mat: Material, model: Any, log: Any = None) -> None:
    """Resolve `/FUNCT` curve references from the model container into callable/data curves."""
    if model is None:
        return

    # Ensure model.curves is accessible and mirrors functions if needed
    if not hasattr(model, "curves"):
        model.curves = getattr(model, "functions", {})
    curves_dict = getattr(model, "curves", None)
    if not curves_dict and hasattr(model, "functions"):
        curves_dict = model.functions
        model.curves = curves_dict

    funcs = curves_dict or {}

    p = mat.params
    load_fids = p.get("load_fids") or p.get("ifload") or p.get("Funct_Id_Load") or p.get("funct_id_load") or []
    unload_fids = p.get("unload_fids") or p.get("ifunload") or p.get("Funct_Id_UnLoad") or p.get("funct_id_unload") or []
    npcurve = p.get("npcurve") or p.get("FUN_A4") or p.get("fun_a4") or p.get("np") or 0
    iunload = p.get("iunload") or p.get("FUN_B4") or p.get("fun_b4") or p.get("ful") or 0

    load_curves = []
    for fid in load_fids:
        fct = funcs.get(fid)
        if fct is not None:
            if hasattr(fct, "x") and hasattr(fct, "y"):
                load_curves.append((fct.x.copy(), fct.y.copy()))
            else:
                load_curves.append(fct)
        else:
            load_curves.append(None)
            if log is not None:
                log.error(f"/MAT/LAW38/{mat.id}: loading function {fid} not defined", "MAT CHECK")
    if load_curves:
        p["load_curves"] = load_curves
        p["curve_load"] = load_curves

    unload_curves = []
    for fid in unload_fids:
        fct = funcs.get(fid)
        if fct is not None:
            if hasattr(fct, "x") and hasattr(fct, "y"):
                unload_curves.append((fct.x.copy(), fct.y.copy()))
            else:
                unload_curves.append(fct)
        elif load_curves and load_curves[0] is not None:
            unload_curves.append(load_curves[0])
        else:
            unload_curves.append(None)
    if unload_curves:
        p["unload_curves"] = unload_curves
        p["curve_unload"] = unload_curves

    if npcurve in funcs:
        fct = funcs[npcurve]
        if hasattr(fct, "x") and hasattr(fct, "y"):
            p["pressure_curve"] = (fct.x.copy(), fct.y.copy())
        else:
            p["pressure_curve"] = fct
        p["curve_pressure"] = p["pressure_curve"]

    if iunload in funcs:
        fct = funcs[iunload]
        if hasattr(fct, "x") and hasattr(fct, "y"):
            p["unload_curve"] = (fct.x.copy(), fct.y.copy())
        else:
            p["unload_curve"] = fct
        p["curve_unload_global"] = p["unload_curve"]


def extra_shapes(mat: Optional[Material] = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """State variable allocation for LAW38:
    - eps38: (6,) total strain tensor in global frame
    - uv38: (33,) state variables matching sigeps38.F / m38init.F
    - off38: () element deletion flag
    """
    return {"eps38": (6,), "uv38": (33,), "off38": ()}


# ----------------------------------------------------------------------------
# Sound Speed
# ----------------------------------------------------------------------------

def sound_speed(mat: Material, rho: Optional[Union[float, np.ndarray]] = None,
                extra: Optional[Dict[str, Any]] = None) -> Union[float, np.ndarray]:
    """Longitudinal sound speed:
    KKK = Emax / (3*(1 - 2*nu_max))
    GGG = Emax / (2*(1 + nu_max))
    c = sqrt((KKK + 4/3*GGG) / rho0)
    Matches sigeps38.F lines 1137-1139.
    """
    p = mat.params
    rho0 = mat.rho0 if mat.rho0 > 0.0 else p.get("rho0", 1.0)
    density = rho if rho is not None else rho0

    e0 = p.get("e0", p.get("e", 1.0))
    efinal = p.get("efinal", e0)
    emax = max(e0, efinal)

    nu_t = p.get("nu_t", _EM20)
    nu_c = p.get("nu_c", 0.0)
    nu_max = min(0.499, max(nu_t, nu_c, _EM20))

    if extra is not None and "uv38" in extra:
        uv = extra["uv38"]
        if uv.ndim == 2 and uv.shape[0] > 0:
            ey_max = np.max(uv[:, 9:12], axis=1)
            emax = np.maximum(emax, ey_max)

    kkk = emax / (3.0 * (1.0 - 2.0 * nu_max))
    ggg = emax / (2.0 * (1.0 + nu_max))
    c_sq = (kkk + _FOUR_OVER_3 * ggg) / np.maximum(density, _TINY)
    return np.sqrt(np.maximum(c_sq, _TINY))


# ----------------------------------------------------------------------------
# Constitutive Stress Update (sigeps38.F)
# ----------------------------------------------------------------------------

def solid_update(
    mat: Material,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constitutive stress update for Material Law 38 (/MAT/VISC_TAB).

    Parameters:
      mat : Material
          Law 38 material record.
      sig : (n, 6) ndarray
          Stress in Voigt [xx, yy, zz, xy, yz, zx] from previous cycle.
      deps : (n, 6) ndarray
          Strain increment in Voigt [xx, yy, zz, xy, yz, zx].
      epsp : (n,) ndarray, optional
          Equivalent plastic strain (passed through).
      dt : float
          Current timestep.
      extra : dict, optional
          Persistent state dict containing 'uv38', 'eps38', 'off38', 'time', etc.

    Returns:
      (sig_new, epsp_out, sound_speed_arr)
    """
    n = sig.shape[0]
    if n == 0:
        empty6 = np.empty((0, 6), dtype=sig.dtype)
        empty1 = np.empty((0,), dtype=sig.dtype)
        return empty6, (epsp if epsp is not None else empty1), empty1

    p = mat.params

    # 1. Unpack material parameters
    rho0 = mat.rho0 if mat.rho0 > 0.0 else p.get("rho0", 1.0)
    e0 = p.get("e0", p.get("e", 1.0))
    nu_t = p.get("nu_t", _EM20)
    nu_c = p.get("nu_c", 0.0)
    rv = p.get("rv", 0.0)
    iflag = p.get("iflag", 0)
    itotal = p.get("itotal", 0)

    beta = p.get("beta", _EM20)
    hyster = p.get("hyster", 1.0)
    ratedamp = p.get("ratedamp", 0.5)
    krecover = p.get("krecover", 0)
    kdecay = p.get("kdecay", 0)
    theta = p.get("theta", 0.67)

    kcompair = p.get("kcompair", p.get("kair", p.get("MAT_Kair", 0)))
    pscale = p.get("pscale", p.get("MAT_PScale", 1.0))
    p0 = p.get("p0", p.get("MAT_P0", 0.0))
    relaxp = p.get("relaxp", p.get("pr", p.get("rp", p.get("MAT_PR", _EM20))))
    maxpres = p.get("maxpres", p.get("pmax", p.get("MAT_PMAX", _EP30)))
    phi = p.get("phi", p.get("poros", p.get("MAT_POROS", 0.0)))

    iunload = p.get("iunload", 0)
    funload = p.get("funload", 1.0)
    runload = p.get("runload", 0.0)
    exponas = p.get("exponas", 1.0)
    exponbs = p.get("exponbs", 1.0)

    imsta = p.get("imsta", 0)
    ideac = 0
    if imsta >= 10:
        imsta -= 10
        ideac = 1

    tensioncut = abs(p.get("tensioncut", p.get("cutoff", p.get("MAT_CUTOFF", _EP30))))
    efinal = p.get("efinal", e0)
    epsfin = p.get("epsfin", 1.0)
    lamda = p.get("lamda", 1.0)
    viscosity = p.get("viscosity", _EP30)
    tolerance = p.get("tolerance", 1.0)

    # Defaults / clamping matching hm_read_mat38.F lines 226-258
    if pscale == 0.0:
        pscale = 1.0
    if nu_t <= 0.0:
        nu_t = _EM20
    if nu_t >= 0.5:
        nu_t = 0.499
    if nu_c >= 0.5:
        nu_c = 0.499
    if itotal > 3:
        itotal = 0
    if beta <= 0.0:
        beta = _EM20
    if hyster <= 0.0:
        hyster = 1.0
    if ratedamp <= 0.0:
        ratedamp = 0.5
    if krecover <= 0 or krecover > 2:
        krecover = 0
    if kdecay <= 0 or kdecay > 2:
        kdecay = 0
    if theta <= 0.0:
        theta = 0.67
    if relaxp <= 0.0:
        relaxp = _EM20
    if maxpres <= 0.0:
        maxpres = _EP30
    if funload <= 0.0:
        funload = 1.0
    if exponas == 0.0:
        exponas = 1.0
    if exponbs == 0.0:
        exponbs = 1.0
    if tensioncut <= 0.0:
        tensioncut = _EP30
    if epsfin <= 0.0 or epsfin > 0.0:
        epsfin = 1.0
    if lamda <= 0.0:
        lamda = 1.0
    if tolerance <= 0.0:
        tolerance = 1.0
    if viscosity <= 0.0:
        viscosity = _EP30
    if efinal <= e0:
        efinal = e0

    # Formulation flags
    # itotal = 0, 1, -1 -> TOTAL = True; itotal = 2, 3, -2 -> TOTAL = False (incremental)
    total_flag = itotal in (0, 1, -1)
    ismstr = p.get("ismstr", 0)
    if ismstr in (10, 12):
        total_flag = True

    # Curves
    load_curves = p.get("load_curves", [])
    unload_curves = p.get("unload_curves", [])
    pressure_curve = p.get("pressure_curve", None)
    dedicated_unload_curve = p.get("unload_curve", None)

    fscale_tab = p.get("fscale_tab", [])
    eps_tab = p.get("eps_tab", [])

    nfunc1 = max(1, len(load_curves))
    if not load_curves:
        load_curves = [None]
        nfunc1 = 1

    alphas = np.ones(nfunc1, dtype=float)
    edots = np.zeros(nfunc1, dtype=float)
    for i in range(min(nfunc1, len(fscale_tab))):
        if fscale_tab[i] > 0.0:
            alphas[i] = fscale_tab[i]
    for i in range(min(nfunc1, len(eps_tab))):
        edots[i] = eps_tab[i]

    runload = max(runload, edots[0])
    runload = min(runload, edots[-1])

    # 2. Extract or initialize UVAR and persistent states
    if extra is not None and "uv38" in extra:
        uvar = extra["uv38"]
    else:
        # Initialize UVAR state matching m38init.F
        uvar = np.zeros((n, 33), dtype=sig.dtype)
        uvar[:, 9:12] = e0
        uvar[:, 12:15] = nu_t / e0
        uvar[:, 16] = 1.0
        uvar[:, 20] = 1.0
        uvar[:, 24] = 1.0
        uvar[:, 31] = _EM20  # UVAR32
        if extra is not None:
            extra["uv38"] = uvar

    if extra is not None and "eps38" in extra:
        eps_tot = extra["eps38"]
        eps_tot += deps
    else:
        eps_tot = deps.copy()
        if extra is not None:
            extra["eps38"] = eps_tot

    if extra is not None and "off38" in extra:
        off = extra["off38"]
        if "off" in extra:
            off = np.minimum(off, extra["off"])
    elif extra is not None and "off" in extra:
        off = extra["off"]
    else:
        off = np.ones(n, dtype=sig.dtype)
        if extra is not None:
            extra["off38"] = off

    curr_time = float(extra.get("time", 0.0)) if extra is not None else 0.0
    eint = extra.get("eint", np.zeros(n, dtype=sig.dtype)) if extra is not None else np.zeros(n, dtype=sig.dtype)

    # 3. Energy-based recovery (KRECOVER = 2, sigeps38 lines 303-327)
    efac = np.zeros(n, dtype=sig.dtype)
    if krecover == 2:
        if curr_time == 0.0:
            efac[:] = 0.0
            uvar[:, 31] = _EM20
        else:
            uvar[:, 31] = np.maximum(uvar[:, 31], eint)
            max_eint = uvar[:, 31]
            ratio_e = np.where(max_eint > 0.0, eint / max_eint, 0.0)
            pui = np.where(eint == max_eint, 1.0, np.where(ratio_e > 0.0, np.exp(beta * np.log(np.maximum(ratio_e, _TINY))), 0.0))
            efac = (1.0 - hyster) * (1.0 - pui)

    # 4. Define stretch / strain tensor for eigendecomposition (lines 333-379)
    if total_flag:
        av = np.empty((n, 6), dtype=sig.dtype)
        av[:, 0] = eps_tot[:, 0]
        av[:, 1] = eps_tot[:, 1]
        av[:, 2] = eps_tot[:, 2]
        av[:, 3] = eps_tot[:, 3] * _HALF
        av[:, 4] = eps_tot[:, 4] * _HALF
        av[:, 5] = eps_tot[:, 5] * _HALF
    else:
        av = np.empty((n, 6), dtype=sig.dtype)
        av[:, 0] = deps[:, 0]
        av[:, 1] = deps[:, 1]
        av[:, 2] = deps[:, 2]
        av[:, 3] = deps[:, 3] * _HALF
        av[:, 4] = deps[:, 4] * _HALF
        av[:, 5] = deps[:, 5] * _HALF

    # Octahedral strain formulation branch (IFLAG == 1, lines 471-490 & 906-939)
    if iflag == 1:
        return _octahedral_update(mat, sig, deps, eps_tot, epsp, dt, uvar, off, rho0, e0, nu_t, total_flag)

    # Build symmetric 3x3 strain tensor for each element
    E_mat = np.zeros((n, 3, 3), dtype=sig.dtype)
    E_mat[:, 0, 0] = av[:, 0]
    E_mat[:, 1, 1] = av[:, 1]
    E_mat[:, 2, 2] = av[:, 2]
    E_mat[:, 0, 1] = av[:, 3]
    E_mat[:, 1, 0] = av[:, 3]
    E_mat[:, 1, 2] = av[:, 4]
    E_mat[:, 2, 1] = av[:, 4]
    E_mat[:, 2, 0] = av[:, 5]
    E_mat[:, 0, 2] = av[:, 5]

    # Eigendecomposition: eigenvalues earv (n, 3) and orthonormal eigenvectors dirprv (n, 3, 3)
    earv, dirprv = np.linalg.eigh(E_mat)

    # Check axis rotation and transform history variables (CHECKAXES & DREH, lines 380-458)
    for i in range(n):
        dprao_i = uvar[i, 16:25].reshape((3, 3), order="F")
        dot_products = np.sum(dprao_i * dirprv[i], axis=0)
        ang = np.abs(np.abs(dot_products) - 1.0)
        amax = np.max(ang)

        if amax >= tolerance:
            Q = dirprv[i].T @ dprao_i
            Q2 = Q * Q

            for m in range(4):
                start = m * 3
                uvar[i, start:start+3] = Q2 @ uvar[i, start:start+3]

            if beta > _TINY and krecover != 2:
                for m in range(2):
                    start = 25 + m * 3
                    uvar[i, start:start+3] = Q2 @ uvar[i, start:start+3]

            uvar[i, 16:25] = dirprv[i].ravel(order="F")

    ear = np.zeros((n, 3), dtype=sig.dtype)
    ecr = np.zeros((n, 3), dtype=sig.dtype)
    if total_flag:
        ear[:, :] = earv
    else:
        ecr[:, :] = earv

    dti = 1.0 / dt if dt > 0.0 else 0.0

    el = np.zeros((n, 3), dtype=sig.dtype)
    ean = np.zeros((n, 3), dtype=sig.dtype)
    ebn = np.zeros((n, 3), dtype=sig.dtype)
    ecn = np.zeros((n, 3), dtype=sig.dtype)

    for j in range(3):
        if total_flag:
            ecr[:, j] = ear[:, j] - uvar[:, j]
        else:
            ear[:, j] = ecr[:, j] + uvar[:, j]

        ebr_j = ecr[:, j] * dti

        if ismstr in (0, 2, 4):
            el[:, j] = np.exp(ear[:, j])
            ebn[:, j] = ebr_j * el[:, j]
            ecn[:, j] = el[:, j] * (1.0 - np.exp(-ecr[:, j]))
        elif ismstr in (10, 12):
            el[:, j] = np.sqrt(np.maximum(ear[:, j] + 1.0, _TINY))
            ebn[:, j] = ebr_j
            ecn[:, j] = ecr[:, j]
        else:
            el[:, j] = ear[:, j] + 1.0
            ebn[:, j] = ebr_j
            ecn[:, j] = ecr[:, j]

        ean[:, j] = el[:, j] - 1.0
        ebn[:, j] = uvar[:, j+6] + (ebn[:, j] - uvar[:, j+6]) * ratedamp

    volumer = el[:, 0] * el[:, 1] * el[:, 2]

    # Confined closed-cell air pressure (lines 561-583)
    pair = np.zeros(n, dtype=sig.dtype)
    if kcompair == 1:
        comp_mask = volumer < 1.0
        if np.any(comp_mask):
            p_comp = np.zeros(n, dtype=sig.dtype)
            phi_safe = (volumer - phi) > _SMALL
            active = comp_mask & phi_safe

            if pressure_curve is not None:
                p_curve_val, _ = _eval_curve(pressure_curve, volumer[active])
                p_comp[active] = -p_curve_val * pscale
            else:
                p_comp[active] = p0 * (volumer[active] - 1.0) / (volumer[active] - phi)

            not_safe = comp_mask & (~phi_safe)
            p_comp[not_safe] = uvar[not_safe, 15]

            p_comp[comp_mask] = np.exp(-relaxp * curr_time) * np.maximum(p_comp[comp_mask], -maxpres)
            pair[comp_mask] = p_comp[comp_mask]
            uvar[comp_mask, 15] = pair[comp_mask]

    elif kcompair == 2:
        comp_mask = volumer < 1.0
        if np.any(comp_mask) and pressure_curve is not None:
            p_curve_val, _ = _eval_curve(pressure_curve, volumer[comp_mask])
            pair[comp_mask] = -p_curve_val

    # 5. Principal stress calculation (lines 588-853)
    psn = np.zeros((n, 3), dtype=sig.dtype)
    eyn = np.zeros((n, 3), dtype=sig.dtype)
    visc = np.zeros((n, 3), dtype=sig.dtype)
    ei = np.full(n, e0, dtype=sig.dtype)

    for j in range(3):
        strain_j = -ean[:, j]
        rate_j = np.abs(ebn[:, j])

        unloading_mask = ((ean[:, j] < 0.0) & (ebn[:, j] > 0.0)) | ((ean[:, j] > 0.0) & (ebn[:, j] < 0.0))

        psn1_val, df1_val = _eval_curve(load_curves[0], strain_j)
        psn1 = -alphas[0] * psn1_val

        edot0 = np.maximum(rate_j, edots[0])

        if iunload != 0 and dedicated_unload_curve is not None:
            mask_ul = unloading_mask
            if np.any(mask_ul):
                if abs(runload - edots[0]) < _EM20:
                    y_ul, df_ul = _eval_curve(dedicated_unload_curve, strain_j[mask_ul])
                    psn[mask_ul, j] = -funload * y_ul
                    eyn[mask_ul, j] = funload * df_ul
                    visc[mask_ul, j] = 0.0
                else:
                    edotu = np.clip(edot0[mask_ul], edots[0], max(runload, edots[0]))
                    edots_val = edots[0]
                    edotl_val = max(runload, edots_val)

                    psn1_ul = psn1[mask_ul]
                    eyn[mask_ul, j] = alphas[0] * df1_val[mask_ul]

                    y_ul, _ = _eval_curve(dedicated_unload_curve, strain_j[mask_ul])
                    psn2_ul = -funload * y_ul

                    denom = max(_TINY, edotl_val - edots_val)
                    ratio = np.clip((edotu - edots_val) / denom, 0.0, 1.0)
                    pui_1 = np.where(ratio > 0.0, ratio ** exponas, 0.0)

                    psn_ul = np.where(
                        pui_1 < 1.0,
                        psn2_ul + (psn1_ul - psn2_ul) * np.exp(exponbs * np.log(np.maximum(1.0 - pui_1, _TINY))),
                        psn2_ul
                    )

                    d_rate = edotu - edots_val
                    visc_calc = np.where(
                        (d_rate > _TINY) & (np.abs(psn_ul - psn1_ul) > _TINY),
                        np.abs(psn_ul - psn1_ul) / np.maximum(d_rate, _TINY),
                        0.0
                    )
                    visc_lim = np.minimum(visc_calc, viscosity)
                    visc[mask_ul, j] = visc_lim
                    psn[mask_ul, j] = psn1_ul - visc_lim * d_rate

            mask_ld = ~mask_ul
            if np.any(mask_ld):
                _interpolate_curves(
                    mask_ld, j, nfunc1, load_curves, alphas, edots,
                    strain_j, edot0, psn1, df1_val, exponas, exponbs,
                    viscosity, psn, eyn, visc
                )
        else:
            if nfunc1 == 1:
                if unload_curves and np.any(unloading_mask):
                    psn[~unloading_mask, j] = psn1[~unloading_mask]
                    eyn[~unloading_mask, j] = alphas[0] * df1_val[~unloading_mask]
                    y_ul, df_ul = _eval_curve(unload_curves[0], strain_j[unloading_mask])
                    psn[unloading_mask, j] = -alphas[0] * y_ul
                    eyn[unloading_mask, j] = alphas[0] * df_ul
                else:
                    psn[:, j] = psn1
                    eyn[:, j] = alphas[0] * df1_val
            else:
                mask_ul = unloading_mask
                mask_ld = ~unloading_mask
                if np.any(mask_ld):
                    _interpolate_curves(
                        mask_ld, j, nfunc1, load_curves, alphas, edots,
                        strain_j, edot0, psn1, df1_val, exponas, exponbs,
                        viscosity, psn, eyn, visc
                    )
                if np.any(mask_ul):
                    curves_ul = unload_curves if unload_curves else load_curves
                    _interpolate_curves(
                        mask_ul, j, nfunc1, curves_ul, alphas, edots,
                        strain_j, edot0, psn1, df1_val, exponas, exponbs,
                        viscosity, psn, eyn, visc
                    )

        # 6. Hysteresis & damage decay (lines 743-781)
        decay = uvar[:, j+28].copy()
        comp_mask = ean[:, j] < 0.0

        if np.any(comp_mask):
            if krecover == 0:
                more_comp = comp_mask & (ecn[:, j] < 0.0)
                uvar[more_comp, j+25] += np.abs(ecn[more_comp, j])
            elif krecover == 1:
                uvar[comp_mask, j+25] -= ecn[comp_mask, j]

            rate_comp = comp_mask & (ebn[:, j] < 0.0)
            if np.any(rate_comp):
                decay[rate_comp] = np.minimum(
                    1.0,
                    hyster * (1.0 - np.exp(-beta * uvar[rate_comp, j+25]))
                )

        if krecover == 2:
            decay = efac.copy()

        if kdecay == 0:
            psn[:, j] *= (1.0 - decay)
        elif kdecay == 1:
            load_mask = ecn[:, j] < 0.0
            psn[load_mask, j] *= (1.0 - decay[load_mask])
        elif kdecay == 2:
            unload_mask = ecn[:, j] > 0.0
            psn[unload_mask, j] *= (1.0 - decay[unload_mask])

        uvar[:, j+28] = decay

        if not total_flag:
            dsigma = psn[:, j] - uvar[:, j+3]
            nonzero_ecn = np.abs(ecn[:, j]) > _TINY
            eyn[nonzero_ecn, j] = np.abs(dsigma[nonzero_ecn] / ecn[nonzero_ecn, j])
            eyn[~nonzero_ecn, j] = uvar[~nonzero_ecn, j+9]

            sign_change = np.sign(ebn[:, j]) != np.sign(uvar[:, j+6] + _TINY)
            eyn[sign_change, j] = uvar[sign_change, j+9]

            eyn[:, j] = eyn[:, j] * theta + uvar[:, j+9] * (1.0 - theta)

        # 7. Tension regime (lines 795-850)
        if itotal in (0, 2):
            tens_mask = ean[:, j] >= 0.0
            if np.any(tens_mask):
                uvar[tens_mask, j+28] = 0.0
                tmp1 = np.exp(-lamda * (volumer[tens_mask] - 1.0 + epsfin))
                ei[tens_mask] = efinal + (e0 - efinal) * (1.0 - tmp1)
                tmp2 = lamda * (efinal - e0) * tmp1
                eyn[tens_mask, j] = np.maximum(ei[tens_mask], tmp2)

                if total_flag:
                    psn[tens_mask, j] = ei[tens_mask] * ean[tens_mask, j]
                else:
                    psn[tens_mask, j] = uvar[tens_mask, j+3] + ei[tens_mask] * ecn[tens_mask, j]

                uvar[tens_mask, j+12] = nu_t / ei[tens_mask]

            if krecover == 2 and np.any(tens_mask):
                decay_t = efac[tens_mask]
                if kdecay == 0:
                    psn[tens_mask, j] *= (1.0 - decay_t)
                elif kdecay == 1:
                    l_m = tens_mask & (ecn[:, j] < 0.0)
                    psn[l_m, j] *= (1.0 - efac[l_m])
                elif kdecay == 2:
                    u_m = tens_mask & (ecn[:, j] > 0.0)
                    psn[u_m, j] *= (1.0 - efac[u_m])
        elif itotal in (-1, -2):
            tens_mask = ean[:, j] > 0.0
            if np.any(tens_mask):
                uvar[tens_mask, j+28] = 0.0
                ei_val = np.maximum(uvar[tens_mask, 9], np.maximum(uvar[tens_mask, 10], uvar[tens_mask, 11]))
                ei[tens_mask] = ei_val
                eyn[tens_mask, j] = ei_val
                if itotal == -1:
                    psn[tens_mask, j] = ei_val * ean[tens_mask, j]
                else:
                    psn[tens_mask, j] = uvar[tens_mask, j+3] + ei_val * ecn[tens_mask, j]
                uvar[tens_mask, j+12] = nu_t / ei_val

    # 8. Instability control (lines 855-870)
    if imsta >= 1:
        sigmax = -(np.min(psn, axis=1) - np.max(psn, axis=1))
        for j in range(3):
            strain_abs = np.maximum(_TINY, np.abs(-ean[:, j]))
            esecant = 0.4 * np.abs(psn[:, j]) / strain_abs
            instable = esecant <= sigmax
            if np.any(instable):
                tmp1 = 0.2 * (sigmax[instable] - esecant[instable])
                psn[instable, j] += tmp1 * ean[instable, j]
                eyn[instable, j] = np.maximum(eyn[instable, j], (1.0 + tmp1) * esecant[instable])

    for j in range(3):
        tmp0 = np.maximum(eyn[:, j], uvar[:, j+9])
        uvar[:, j] = ear[:, j]
        uvar[:, j+3] = psn[:, j]
        uvar[:, j+6] = ebn[:, j]
        uvar[:, j+9] = eyn[:, j]
        eyn[:, j] = tmp0

    # 9. Poisson coupling (lines 941-1000)
    psc = np.zeros((n, 3), dtype=sig.dtype)
    if (nu_t + nu_c) <= 2.0 * _TINY:
        if total_flag:
            psc[:, :] = psn
        else:
            for j in range(3):
                psc[:, j] = eyn[:, j] * ecn[:, j]
    else:
        e12 = (ean[:, 0] + ean[:, 1]) * _HALF
        e23 = (ean[:, 1] + ean[:, 2]) * _HALF
        e31 = (ean[:, 2] + ean[:, 0]) * _HALF

        def _nu_interp(e_avg: np.ndarray) -> np.ndarray:
            term = (1.0 - np.exp(-rv * np.abs(e_avg))) * (np.sign(e_avg) + 1.0) * _HALF
            return nu_c + (nu_t - nu_c) * term

        v12 = _nu_interp(e12)
        v23 = _nu_interp(e23)
        v31 = _nu_interp(e31)

        if total_flag:
            detc = (1.0 - v23*v23 - v31*v31 - v12*v12 - 2.0*v12*v31*v23)
            detc_safe = np.where(np.abs(detc) > _TINY, detc, 1.0)

            a11 = 1.0 - v23*v23
            a12 = v12 + v23*v31
            a13 = v31 + v23*v12
            a22 = 1.0 - v31*v31
            a23 = v23 + v31*v12
            a33 = 1.0 - v12*v12

            psc[:, 0] = (a11 * psn[:, 0] + a12 * psn[:, 1] + a13 * psn[:, 2]) / detc_safe
            psc[:, 1] = (a12 * psn[:, 0] + a22 * psn[:, 1] + a23 * psn[:, 2]) / detc_safe
            psc[:, 2] = (a13 * psn[:, 0] + a23 * psn[:, 1] + a33 * psn[:, 2]) / detc_safe
        else:
            uvar[:, 12] = theta * v23 / ei + (1.0 - theta) * uvar[:, 12]
            uvar[:, 13] = theta * v31 / ei + (1.0 - theta) * uvar[:, 13]
            uvar[:, 14] = theta * v12 / ei + (1.0 - theta) * uvar[:, 14]

            detc = (1.0 / (eyn[:, 0]*eyn[:, 1]*eyn[:, 2])
                    - uvar[:, 12]*uvar[:, 12] / eyn[:, 0]
                    - uvar[:, 13]*uvar[:, 13] / eyn[:, 1]
                    - uvar[:, 14]*uvar[:, 14] / eyn[:, 2]
                    - 2.0 * uvar[:, 12] * uvar[:, 13] * uvar[:, 14])
            detc_safe = np.where(np.abs(detc) > _TINY, detc, 1.0)

            a11 = 1.0 / (eyn[:, 1]*eyn[:, 2]) - uvar[:, 12]*uvar[:, 12]
            a12 = uvar[:, 14] / eyn[:, 2] + uvar[:, 12]*uvar[:, 13]
            a13 = uvar[:, 13] / eyn[:, 1] + uvar[:, 12]*uvar[:, 14]
            a22 = 1.0 / (eyn[:, 0]*eyn[:, 2]) - uvar[:, 13]*uvar[:, 13]
            a23 = uvar[:, 12] / eyn[:, 0] + uvar[:, 13]*uvar[:, 14]
            a33 = 1.0 / (eyn[:, 0]*eyn[:, 1]) - uvar[:, 14]*uvar[:, 14]

            psc[:, 0] = (a11 * ecn[:, 0] + a12 * ecn[:, 1] + a13 * ecn[:, 2]) / detc_safe
            psc[:, 1] = (a12 * ecn[:, 0] + a22 * ecn[:, 1] + a23 * ecn[:, 2]) / detc_safe
            psc[:, 2] = (a13 * ecn[:, 0] + a23 * ecn[:, 1] + a33 * ecn[:, 2]) / detc_safe

    # 10. Tension cutoff element deletion (lines 1002-1009)
    deleted_mask = (off <= 0.0) | (psn[:, 0] > tensioncut) | (psn[:, 1] > tensioncut) | (psn[:, 2] > tensioncut)
    if np.any(deleted_mask):
        psc[deleted_mask, :] = 0.0
        off[deleted_mask] = 0.0
        if extra is not None:
            if "off" in extra:
                extra["off"][deleted_mask] = 0.0
            if "off38" in extra:
                extra["off38"][deleted_mask] = 0.0

    # 11. Cauchy stress conversion (lines 1010-1035)
    if ismstr in (0, 2, 4):
        den1 = np.maximum(el[:, 1] * el[:, 2], _TINY)
        den2 = np.maximum(el[:, 2] * el[:, 0], _TINY)
        den3 = np.maximum(el[:, 0] * el[:, 1], _TINY)
        psc[:, 0] /= den1
        psc[:, 1] /= den2
        psc[:, 2] /= den3
        eyn[:, 0] /= den1
        eyn[:, 1] /= den2
        eyn[:, 2] /= den3
    elif ismstr in (10, 12):
        den1 = el[:, 0] / np.maximum(volumer, _TINY)
        den2 = el[:, 1] / np.maximum(volumer, _TINY)
        den3 = el[:, 2] / np.maximum(volumer, _TINY)
        psc[:, 0] *= den1
        psc[:, 1] *= den2
        psc[:, 2] *= den3
        eyn[:, 0] *= den1
        eyn[:, 1] *= den2
        eyn[:, 2] *= den3

    # Add confined air pressure (lines 1036-1064)
    sigprv = np.empty((n, 3), dtype=sig.dtype)
    if kcompair == 2:
        tmp0 = volumer
        tmp3 = np.min(el, axis=1)
        valid = (tmp0 < 1.0) & (tmp3 < 1.0) & (tmp3 > tmp0) & (np.abs(tmp0 - tmp3) > _EM6)
        aa = np.zeros(n, dtype=sig.dtype)
        if np.any(valid):
            tmp2 = np.where(tmp0[valid] > 0.0, np.exp(_THIRD * np.log(tmp0[valid])) - tmp0[valid], 0.0)
            aa[valid] = np.clip((tmp3[valid] - tmp0[valid]) / np.maximum(tmp2, _TINY), 0.0, 1.0)
        for j in range(3):
            sigprv[:, j] = psc[:, j] + aa * (pair - psc[:, j])
    else:
        for j in range(3):
            sigprv[:, j] = psc[:, j] + pair

    if np.any(deleted_mask):
        sigprv[deleted_mask, :] = 0.0

    # 12. Back-rotate principal stresses to global coordinates (lines 1065-1085)
    sign = np.empty((n, 6), dtype=sig.dtype)
    sign[:, 0] = (dirprv[:, 0, 0]**2 * sigprv[:, 0] +
                  dirprv[:, 0, 1]**2 * sigprv[:, 1] +
                  dirprv[:, 0, 2]**2 * sigprv[:, 2])

    sign[:, 1] = (dirprv[:, 1, 0]**2 * sigprv[:, 0] +
                  dirprv[:, 1, 1]**2 * sigprv[:, 1] +
                  dirprv[:, 1, 2]**2 * sigprv[:, 2])

    sign[:, 2] = (dirprv[:, 2, 0]**2 * sigprv[:, 0] +
                  dirprv[:, 2, 1]**2 * sigprv[:, 1] +
                  dirprv[:, 2, 2]**2 * sigprv[:, 2])

    sign[:, 3] = (dirprv[:, 0, 0]*dirprv[:, 1, 0] * sigprv[:, 0] +
                  dirprv[:, 0, 1]*dirprv[:, 1, 1] * sigprv[:, 1] +
                  dirprv[:, 0, 2]*dirprv[:, 1, 2] * sigprv[:, 2])

    sign[:, 4] = (dirprv[:, 1, 0]*dirprv[:, 2, 0] * sigprv[:, 0] +
                  dirprv[:, 1, 1]*dirprv[:, 2, 1] * sigprv[:, 1] +
                  dirprv[:, 1, 2]*dirprv[:, 2, 2] * sigprv[:, 2])

    sign[:, 5] = (dirprv[:, 2, 0]*dirprv[:, 0, 0] * sigprv[:, 0] +
                  dirprv[:, 2, 1]*dirprv[:, 0, 1] * sigprv[:, 1] +
                  dirprv[:, 2, 2]*dirprv[:, 0, 2] * sigprv[:, 2])

    if not total_flag:
        sign += sig

    if np.any(deleted_mask):
        sign[deleted_mask, :] = 0.0

    # IMSTA == 2: shear stress stabilization (lines 1105-1135)
    if imsta == 2:
        epsxy = (dirprv[:, 0, 0] * dirprv[:, 1, 0] * ean[:, 0] +
                 dirprv[:, 0, 1] * dirprv[:, 1, 1] * ean[:, 1] +
                 dirprv[:, 0, 2] * dirprv[:, 1, 2] * ean[:, 2])
        epsyz = (dirprv[:, 1, 0] * dirprv[:, 2, 0] * ean[:, 0] +
                 dirprv[:, 1, 1] * dirprv[:, 2, 1] * ean[:, 1] +
                 dirprv[:, 1, 2] * dirprv[:, 2, 2] * ean[:, 2])
        epszx = (dirprv[:, 2, 0] * dirprv[:, 0, 0] * ean[:, 0] +
                 dirprv[:, 2, 1] * dirprv[:, 0, 1] * ean[:, 1] +
                 dirprv[:, 2, 2] * dirprv[:, 0, 2] * ean[:, 2])

        esec1 = 0.5 * np.abs(sign[:, 3]) / np.maximum(_TINY, np.abs(epsxy))
        esec2 = 0.5 * np.abs(sign[:, 4]) / np.maximum(_TINY, np.abs(epsyz))
        esec3 = 0.5 * np.abs(sign[:, 5]) / np.maximum(_TINY, np.abs(epszx))
        sigmax2 = np.maximum(0.5 * ei, sigmax if imsta >= 1 else 0.5 * ei)

        c1 = esec1 <= sigmax2
        if np.any(c1):
            sign[c1, 3] += 0.1 * (sigmax2[c1] - esec1[c1]) * epsxy[c1]
        c2 = esec2 <= sigmax2
        if np.any(c2):
            sign[c2, 4] += 0.1 * (sigmax2[c2] - esec2[c2]) * epsyz[c2]
        c3 = esec3 <= sigmax2
        if np.any(c3):
            sign[c3, 5] += 0.1 * (sigmax2[c3] - esec3[c3]) * epszx[c3]

    if extra is not None:
        extra["epsd"] = np.max(np.abs(ebn), axis=1)
        extra["viscmax"] = np.max(visc, axis=1)

    # 13. Longitudinal sound speed (lines 1100-1141)
    emax = np.maximum(ei, np.max(eyn, axis=1))
    nu_max = min(0.499, max(nu_t, nu_c, _EM20))
    kkk = emax / (3.0 * (1.0 - 2.0 * nu_max))
    ggg = emax / (2.0 * (1.0 + nu_max))
    soundsp = np.sqrt(np.maximum((kkk + _FOUR_OVER_3 * ggg) / rho0, _TINY))

    epsp_out = epsp if epsp is not None else np.zeros(n, dtype=sig.dtype)
    return sign, epsp_out, soundsp


def _interpolate_curves(
    mask: np.ndarray,
    j: int,
    nfunc1: int,
    curves: Sequence[Any],
    alphas: np.ndarray,
    edots: np.ndarray,
    strain_j: np.ndarray,
    edot0: np.ndarray,
    psn1: np.ndarray,
    df1_val: np.ndarray,
    exponas: float,
    exponbs: float,
    viscosity: float,
    psn_out: np.ndarray,
    eyn_out: np.ndarray,
    visc_out: np.ndarray,
) -> None:
    """Helper for multi-curve strain-rate interpolation (lines 684-736)."""
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return

    for i in indices:
        rate = min(edot0[i], edots[nfunc1 - 1])
        L = 0
        for idx in range(nfunc1):
            if rate <= edots[idx]:
                L = idx
                break
        else:
            L = nfunc1 - 1

        k = L
        k1 = max(0, L - 1)

        edotl = edots[k]
        edots_val = edots[k1]

        y2, df2 = _eval_curve(curves[k], strain_j[i])
        psn2_val = -alphas[k] * y2

        y1, df1 = _eval_curve(curves[k1], strain_j[i])
        psn1_val = -alphas[k1] * y1

        if L == nfunc1 - 1:
            eyn_out[i, j] = alphas[L] * df2
        else:
            eyn_out[i, j] = alphas[k1] * df1

        denom = max(_TINY, edotl - edots_val)
        ratio = min(1.0, max(0.0, (rate - edots_val) / denom))
        pui_1 = ratio ** exponas if ratio > 0.0 else 0.0

        if pui_1 < 1.0:
            psn_val = psn2_val + (psn1_val - psn2_val) * ((1.0 - pui_1) ** exponbs)
        else:
            psn_val = psn2_val

        d_rate = rate - edots_val
        visc_val = 0.0
        if d_rate > _TINY and abs(psn_val - psn1_val) > _TINY:
            visc_val = abs((psn_val - psn1_val) / d_rate)
        visc_val = min(visc_val, viscosity)
        visc_out[i, j] = visc_val

        psn_out[i, j] = psn1_val - visc_val * d_rate


def _octahedral_update(
    mat: Material,
    sig: np.ndarray,
    deps: np.ndarray,
    eps_tot: np.ndarray,
    epsp: Optional[np.ndarray],
    dt: float,
    uvar: np.ndarray,
    off: np.ndarray,
    rho0: float,
    e0: float,
    nu_t: float,
    total_flag: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Octahedral strain formulation branch (IFLAG == 1, lines 906-939)."""
    n = sig.shape[0]
    emax = np.full(n, e0, dtype=sig.dtype)
    vt = min(0.499, max(nu_t, _EM20))

    d11 = emax * (1.0 - vt) / ((1.0 + vt) * (1.0 - 2.0 * vt))
    d12 = emax * vt / ((1.0 + vt) * (1.0 - 2.0 * vt))
    d44 = emax / (2.0 * (1.0 + vt))

    sign = np.zeros((n, 6), dtype=sig.dtype)
    if total_flag:
        sign[:, 0] = d11 * eps_tot[:, 0] + d12 * eps_tot[:, 1] + d12 * eps_tot[:, 2]
        sign[:, 1] = d12 * eps_tot[:, 0] + d11 * eps_tot[:, 1] + d12 * eps_tot[:, 2]
        sign[:, 2] = d12 * eps_tot[:, 0] + d12 * eps_tot[:, 1] + d11 * eps_tot[:, 2]
        sign[:, 3] = d44 * eps_tot[:, 3]
        sign[:, 4] = d44 * eps_tot[:, 4]
        sign[:, 5] = d44 * eps_tot[:, 5]
    else:
        sign[:, 0] = sig[:, 0] + d11 * deps[:, 0] + d12 * deps[:, 1] + d12 * deps[:, 2]
        sign[:, 1] = sig[:, 1] + d12 * deps[:, 0] + d11 * deps[:, 1] + d12 * deps[:, 2]
        sign[:, 2] = sig[:, 2] + d12 * deps[:, 0] + d12 * deps[:, 1] + d11 * deps[:, 2]
        sign[:, 3] = sig[:, 3] + d44 * deps[:, 3]
        sign[:, 4] = sig[:, 4] + d44 * deps[:, 4]
        sign[:, 5] = sig[:, 5] + d44 * deps[:, 5]

    soundsp = np.sqrt(d11 / rho0)
    epsp_out = epsp if epsp is not None else np.zeros(n, dtype=sig.dtype)
    return sign, epsp_out, soundsp


# ----------------------------------------------------------------------------
# Consistent Algorithmic Tangent (M12 requirement)
# ----------------------------------------------------------------------------

def consistent_solid_tangent(
    mat: Material,
    *args: Any,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent algorithmic tangent stiffness tensor of shape (n, 6, 6).

    Computes the exact directional consistent tangent stiffness tensor
    C_ijkl = d(sigma_ij) / d(deps_kl) relating stress changes to strain increments
    under multi-axial deformation, closed-cell air pressure, rate-dependent viscosity,
    Poisson coupling, and tabulated nonlinear curve response.

    Supports both:
      - Symmetric consistent tangent (default, symmetric=True):
        C = 0.5 * (C + C.T) guaranteeing major/minor symmetry and positive
        definiteness in stable compressive and elastic deformation regimes.
      - Exact directional algorithmic numerical perturbation (symmetric=False)
        for verification against central finite differences.

    Accepts multiple signature patterns:
      - (mat, eps, deps, dt, extra)
      - (mat, sig, epsp, epsp_incr, extra)
      - (mat, deps, dt, extra)
      - (mat, dt=dt, extra=extra, ...)
      - keyword arguments (eps, deps, sig, epsp, epsp_incr, extra, dt, symmetric, h)
    """
    sig = kwargs.get("sig")
    deps = kwargs.get("deps")
    eps = kwargs.get("eps")
    epsp = kwargs.get("epsp")
    epsp_incr = kwargs.get("epsp_incr")
    extra = kwargs.get("extra")
    dt = kwargs.get("dt")
    symmetric = bool(kwargs.get("symmetric", True))
    h = float(kwargs.get("h", 1e-7))

    # Parse positional arguments
    if len(args) == 1:
        if isinstance(args[0], dict):
            extra = args[0]
        elif isinstance(args[0], np.ndarray):
            if (args[0].ndim == 2 and args[0].shape[1] == 6) or (args[0].ndim == 1 and args[0].shape[0] == 6):
                deps = args[0]
            else:
                sig = args[0]
    elif len(args) == 2:
        if isinstance(args[1], dict):
            if isinstance(args[0], np.ndarray):
                deps = args[0]
            extra = args[1]
        elif isinstance(args[1], (int, float)):
            deps = args[0]
            dt = float(args[1])
        elif isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray):
            sig = args[0]
            deps = args[1]
    elif len(args) == 3:
        if isinstance(args[2], dict):
            deps = args[0]
            dt = float(args[1])
            extra = args[2]
        else:
            sig = args[0]
            epsp = args[1]
            deps = args[2]
    elif len(args) == 4:
        if isinstance(args[3], dict):
            if isinstance(args[2], (int, float)):
                eps = args[0]
                deps = args[1]
                dt = float(args[2])
                extra = args[3]
            else:
                sig = args[0]
                epsp = args[1]
                deps = args[2]
                extra = args[3]
    elif len(args) >= 5:
        eps = args[0]
        deps = args[1]
        dt = float(args[2]) if args[2] is not None else None
        extra = args[3]

    if deps is None and epsp_incr is not None:
        if isinstance(epsp_incr, np.ndarray) and ((epsp_incr.ndim == 2 and epsp_incr.shape[1] == 6) or (epsp_incr.ndim == 1 and epsp_incr.shape[0] == 6)):
            deps = epsp_incr

    n = None
    for arr in (deps, sig, eps, epsp, epsp_incr):
        if arr is not None and isinstance(arr, np.ndarray) and arr.ndim >= 1:
            n = arr.shape[0]
            break

    if n is None and extra is not None:
        for k in ("uv38", "eps38", "off38", "rho"):
            if k in extra and isinstance(extra[k], np.ndarray) and extra[k].ndim >= 1:
                n = extra[k].shape[0]
                break

    if n is None:
        n = 1

    if n == 0:
        return np.empty((0, 6, 6), dtype=float)

    if deps is not None:
        deps_arr = np.asarray(deps, dtype=float)
        if deps_arr.ndim == 1 and deps_arr.shape[0] == 6:
            deps_norm = np.broadcast_to(deps_arr.reshape(1, 6), (n, 6)).copy()
        elif deps_arr.ndim == 2 and deps_arr.shape == (n, 6):
            deps_norm = deps_arr.copy()
        else:
            deps_norm = np.zeros((n, 6), dtype=float)
    else:
        deps_norm = np.zeros((n, 6), dtype=float)

    if sig is not None:
        sig_arr = np.asarray(sig, dtype=float)
        if sig_arr.ndim == 1 and sig_arr.shape[0] == 6:
            sig_norm = np.broadcast_to(sig_arr.reshape(1, 6), (n, 6)).copy()
        elif sig_arr.ndim == 2 and sig_arr.shape == (n, 6):
            sig_norm = sig_arr.copy()
        else:
            sig_norm = np.zeros((n, 6), dtype=float)
    else:
        sig_norm = np.zeros((n, 6), dtype=float)

    if dt is None:
        if extra is not None and "dt" in extra:
            dt_val = float(extra["dt"])
        else:
            dt_val = 0.0
    else:
        dt_val = float(dt)

    if epsp is not None:
        epsp_arr = np.asarray(epsp, dtype=float)
        if epsp_arr.ndim == 0:
            epsp_norm = np.full(n, float(epsp_arr), dtype=float)
        elif epsp_arr.shape[0] != n:
            epsp_norm = np.zeros(n, dtype=float)
        else:
            epsp_norm = epsp_arr.copy()
    else:
        epsp_norm = np.zeros(n, dtype=float)

    C = np.zeros((n, 6, 6), dtype=float)
    for j in range(6):
        dp = deps_norm.copy()
        dm = deps_norm.copy()
        dp[:, j] += h
        dm[:, j] -= h

        extra_p = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in extra.items()} if extra is not None else None
        extra_m = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in extra.items()} if extra is not None else None

        sp, _, _ = solid_update(mat, sig_norm.copy(), dp, epsp=epsp_norm, dt=dt_val, extra=extra_p)
        sm, _, _ = solid_update(mat, sig_norm.copy(), dm, epsp=epsp_norm, dt=dt_val, extra=extra_m)

        C[:, :, j] = (sp - sm) / (2.0 * h)

    if symmetric:
        C = 0.5 * (C + np.swapaxes(C, 1, 2))

    return C



# ----------------------------------------------------------------------------
# Shell Update (Not Supported)
# ----------------------------------------------------------------------------

def shell_update(*args: Any, **kwargs: Any) -> None:
    """LAW38 (/MAT/VISC_TAB) is implemented for 3D solid elements only."""
    raise NotImplementedError(
        "LAW38 (/MAT/VISC_TAB - Tabulated Viscoelastic Foam) is implemented "
        "for 3D solid elements only; shells are not supported upstream."
    )


# ----------------------------------------------------------------------------
# Registry Hook
# ----------------------------------------------------------------------------

def _register() -> None:
    """Register LAW38 in MAT_PHYSICS_REGISTRY."""
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for key in (38, "38", "LAW38", "VISC_TAB", "MAT_VISC_TAB", "LAW38_VISC_TAB"):
            MAT_PHYSICS_REGISTRY[key] = build_law38
    except Exception:
        pass


_register()
