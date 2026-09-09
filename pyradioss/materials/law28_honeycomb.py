"""LAW28 — Orthotropic honeycomb crushable material (/MAT/LAW28, /MAT/HONEYCOMB).

Fortran origin:
   constitutive update: ``engine/source/materials/mat/mat028/sigeps28.F``
   parameter extraction: ``starter/source/materials/mat/mat028/hm_read_mat28.F``
   CFG layout: ``C:\\OpenRadioss\\hm_cfg_files\\config\\CFG\\radioss110\\MAT\\matl28_honeycomb.cfg``

Theory and Fortran Semantics (sigeps28.F):
-----------------------------------------
An orthotropic crushable honeycomb material model without Poisson coupling
(nu = 0.0), characterized by:
  1. Orthotropic elastic stiffness moduli:
     E11, E22, E33 (normal) and G12, G23, G31 (shear).
  2. Uncoupled elastic trial stress:
     sigma_11_trial = sigma_11_old + E11 * deps_11
     sigma_22_trial = sigma_22_old + E22 * deps_22
     sigma_33_trial = sigma_33_old + E33 * deps_33
     sigma_12_trial = sigma_12_old + G12 * deps_12
     sigma_23_trial = sigma_23_old + G23 * deps_23
     sigma_31_trial = sigma_31_old + G31 * deps_31
  3. Total strain tracking (extra["eps28"] += deps):
     eps = extra["eps28"]
  4. Element deletion check:
     If extra has "off28" (or "off"):
       rupture = (eps_11 > eps_max11) | (eps_22 > eps_max22) | (eps_33 > eps_max33) |
                 (|eps_12/2| > eps_max12) | (|eps_23/2| > eps_max23) | (|eps_31/2| > eps_max31)
       active elements with rupture set off = 0.0.
       Deleted elements (off == 0) have stress zeroed: sigma = 0.
  5. Yield stress clamping for each component k in {1..6}:
     Abscissa x:
       For normal components k in {1, 2, 3}:
         if Gflag == 0: x = mu = rho/rho0 - 1 = AMU (or -tr(eps))
         elif Gflag == 1: x = eps_k
         elif Gflag == -1: x = -eps_k
       For shear components k in {4, 5, 6}:
         if Vflag == 0: x = mu
         elif Vflag == 1: x = eps_shear_k
         elif Vflag == -1: x = -eps_shear_k
     Yield stress:
       Y_k = FScale_k * Funct_k(x)
     Clamping (sigeps28.F lines 292, 299, 306, 313, 320, 327):
       sigma_k = sign(sigma_k_trial) * min(|sigma_k_trial|, Y_k)
  6. Sound speed:
     c = sqrt(max(E11, E22, E33, G12, G23, G31) / rho0)

Solids only (SOLID_ORTHOTROPIC and SPH).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pyradioss.model.entities import Material

_DEFAULT_EPS_MAX = 1.0e30


# ------------------------------------------------------------------ #
# Parameter extraction (hm_read_mat28.F)
# ------------------------------------------------------------------ #

def build_law28(rec: Any) -> Material:
    """Build a LAW28 Material from a GenericMaterialRecord, dict, Material, or entity.

    Fortran origin: ``starter/source/materials/mat/mat028/hm_read_mat28.F``.
    """
    if isinstance(rec, Material):
        p = dict(rec.params) if rec.params else {}
        mat_id = rec.id
        title = rec.title
        obj = rec
    elif isinstance(rec, dict):
        p = dict(rec.get("params", rec))
        mat_id = rec.get("id", 1)
        title = rec.get("title", "")
        obj = rec
    else:
        p = dict(getattr(rec, "params", {}))
        mat_id = getattr(rec, "id", 1)
        title = getattr(rec, "title", "")
        obj = rec

    def _get_val(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if isinstance(obj, dict) and k in obj and obj[k] is not None:
                try:
                    return float(obj[k])
                except (ValueError, TypeError):
                    pass
            if hasattr(obj, k) and getattr(obj, k) is not None:
                try:
                    return float(getattr(obj, k))
                except (ValueError, TypeError):
                    pass
            if k in p and p[k] is not None:
                try:
                    return float(p[k])
                except (ValueError, TypeError):
                    pass
        return default

    def _get_int(keys: list[str], default: int = 0) -> int:
        for k in keys:
            if isinstance(obj, dict) and k in obj and obj[k] is not None:
                try:
                    return int(obj[k])
                except (ValueError, TypeError):
                    pass
            if hasattr(obj, k) and getattr(obj, k) is not None:
                try:
                    return int(getattr(obj, k))
                except (ValueError, TypeError):
                    pass
            if k in p and p[k] is not None:
                try:
                    return int(p[k])
                except (ValueError, TypeError):
                    pass
        return default

    # Density
    rho0 = _get_val(["MAT_RHO", "mat_rho", "rho0", "rho", "density", "RHO0", "DENSITY"], default=1.0)
    if rho0 <= 0.0:
        raise ValueError("LAW28: Density rho0 must be > 0 (hm_read_mat28 check)")

    rho_ref = _get_val(["Refer_Rho", "refer_rho", "rhor", "ref_rho", "REF_RHO", "rho_ref", "RHOR", "MAT_REFRHO"], default=0.0)
    if rho_ref == 0.0:
        rho_ref = rho0  # hm_read_mat28.F: IF(RHOR==ZERO) RHOR=RHO0

    # Moduli
    e11 = _get_val(["MAT_EA", "mat_ea", "E11", "e11", "EA", "ea"])
    e22 = _get_val(["MAT_EB", "mat_eb", "E22", "e22", "EB", "eb"])
    e33 = _get_val(["MAT_EC", "mat_ec", "E33", "e33", "EC", "ec"])

    g12 = _get_val(["MAT_GAB", "mat_gab", "G12", "g12", "GAB", "gab"])
    g23 = _get_val(["MAT_GBC", "mat_gbc", "G23", "g23", "GBC", "gbc"])
    g31 = _get_val(["MAT_GCA", "mat_gca", "G31", "g31", "GCA", "gca"])

    # Normal yield functions and formulation
    fun_id11 = _get_int(["FUN_A1", "fun_a1", "fun_id11", "FUN_ID11", "funa1", "i11"])
    fun_id22 = _get_int(["FUN_B1", "fun_b1", "fun_id22", "FUN_ID22", "funb1", "i22"])
    fun_id33 = _get_int(["FUN_A2", "fun_a2", "fun_id33", "FUN_ID33", "funa2", "i33"])
    gflag = _get_int(["Gflag", "gflag", "GFLAG", "iflag1", "IFLAG1", "if1", "IF1"], default=0)

    fscale11 = _get_val(["FScale11", "fscale11", "FSCALE11", "fac1", "FAC1"], default=1.0)
    fscale22 = _get_val(["FScale22", "fscale22", "FSCALE22", "fac2", "FAC2"], default=1.0)
    fscale33 = _get_val(["FScale33", "fscale33", "FSCALE33", "fac3", "FAC3"], default=1.0)
    if fscale11 == 0.0:
        fscale11 = 1.0  # hm_read_mat28.F: IF (FAC1 == ZERO) FAC1 = ONE * FAC_UNIT
    if fscale22 == 0.0:
        fscale22 = 1.0
    if fscale33 == 0.0:
        fscale33 = 1.0

    eps_max11 = _get_val(["MAT_EPSR1", "mat_epsr1", "eps_max11", "epsr1", "EPS_MAX11", "EPSR1", "emx11"], default=_DEFAULT_EPS_MAX)
    eps_max22 = _get_val(["MAT_EPSR2", "mat_epsr2", "eps_max22", "epsr2", "EPS_MAX22", "EPSR2", "emx22"], default=_DEFAULT_EPS_MAX)
    eps_max33 = _get_val(["MAT_EPSR3", "mat_epsr3", "eps_max33", "epsr3", "EPS_MAX33", "EPSR3", "emx33"], default=_DEFAULT_EPS_MAX)
    if eps_max11 == 0.0:
        eps_max11 = _DEFAULT_EPS_MAX  # hm_read_mat28.F: IF(UPARAM(9)==ZERO) UPARAM(9)=INFINITY
    if eps_max22 == 0.0:
        eps_max22 = _DEFAULT_EPS_MAX
    if eps_max33 == 0.0:
        eps_max33 = _DEFAULT_EPS_MAX

    # Shear yield functions and formulation
    fun_id12 = _get_int(["FUN_A3", "fun_a3", "fun_id12", "FUN_ID12", "funa3", "i12"])
    fun_id23 = _get_int(["FUN_B3", "fun_b3", "fun_id23", "FUN_ID23", "funb3", "i23"])
    fun_id31 = _get_int(["FUN_A4", "fun_a4", "fun_id31", "FUN_ID31", "funa4", "i31"])
    vflag = _get_int(["Vflag", "vflag", "VFLAG", "iflag2", "IFLAG2", "if2", "IF2"], default=0)

    fscale12 = _get_val(["FScale12", "fscale12", "FSCALE12", "fac4", "FAC4"], default=1.0)
    fscale23 = _get_val(["FScale23", "fscale23", "FSCALE23", "fac5", "FAC5"], default=1.0)
    fscale31 = _get_val(["FScale13", "fscale13", "FSCALE13", "FScale31", "fscale31", "FSCALE31", "fac6", "FAC6"], default=1.0)
    if fscale12 == 0.0:
        fscale12 = 1.0
    if fscale23 == 0.0:
        fscale23 = 1.0
    if fscale31 == 0.0:
        fscale31 = 1.0

    eps_max12 = _get_val(["MAT_EPSR4", "mat_epsr4", "eps_max12", "epsr4", "EPS_MAX12", "EPSR4", "emx12"], default=_DEFAULT_EPS_MAX)
    eps_max23 = _get_val(["MAT_EPSR5", "mat_epsr5", "eps_max23", "epsr5", "EPS_MAX23", "EPSR5", "emx23"], default=_DEFAULT_EPS_MAX)
    eps_max31 = _get_val(["MAT_EPSR6", "mat_epsr6", "eps_max31", "epsr6", "EPS_MAX31", "EPSR6", "emx31"], default=_DEFAULT_EPS_MAX)
    if eps_max12 == 0.0:
        eps_max12 = _DEFAULT_EPS_MAX
    if eps_max23 == 0.0:
        eps_max23 = _DEFAULT_EPS_MAX
    if eps_max31 == 0.0:
        eps_max31 = _DEFAULT_EPS_MAX

    e_max = max(e11, e22, e33)
    params = {
        "E": e_max if e_max > 0.0 else 1.0,
        "nu": 0.0,
        "rho0": rho0,
        "rho_ref": rho_ref,
        "E11": e11,
        "E22": e22,
        "E33": e33,
        "G12": g12,
        "G23": g23,
        "G31": g31,
        "fun_id11": fun_id11,
        "fun_id22": fun_id22,
        "fun_id33": fun_id33,
        "gflag": gflag,
        "fscale11": fscale11,
        "fscale22": fscale22,
        "fscale33": fscale33,
        "eps_max11": eps_max11,
        "eps_max22": eps_max22,
        "eps_max33": eps_max33,
        "fun_id12": fun_id12,
        "fun_id23": fun_id23,
        "fun_id31": fun_id31,
        "vflag": vflag,
        "fscale12": fscale12,
        "fscale23": fscale23,
        "fscale31": fscale31,
        "eps_max12": eps_max12,
        "eps_max23": eps_max23,
        "eps_max31": eps_max31,
    }

    # Preserve any pre-resolved curves if present
    for k in ("curve28_x", "curve28_y", "curve28_s", "curve28_fct"):
        if k in p:
            params[k] = p[k]

    return Material(id=mat_id, law=28, rho0=rho0, title=title, params=params)


build_honeycomb = build_law28


# ------------------------------------------------------------------ #
# Curve resolution (starter initialization)
# ------------------------------------------------------------------ #

def resolve(mat: Material, model: Any, log: Any = None) -> None:
    """Resolve curves fun_id11..31 from model.functions (deck-order-free).

    Stores evaluated or sampled curve arrays in mat.params:
    curve28_x, curve28_y, curve28_s, and curve28_fct for fast interpolation.
    """
    p = mat.params
    fids = [
        p.get("fun_id11", 0),
        p.get("fun_id22", 0),
        p.get("fun_id33", 0),
        p.get("fun_id12", 0),
        p.get("fun_id23", 0),
        p.get("fun_id31", 0),
    ]

    curve_x: list[np.ndarray | None] = [None] * 6
    curve_y: list[np.ndarray | None] = [None] * 6
    curve_s: list[np.ndarray | None] = [None] * 6
    curve_fct: list[Any | None] = [None] * 6

    funcs = getattr(model, "functions", {}) if model is not None else {}

    for k, fid in enumerate(fids):
        if fid and fid != 0:
            fct = funcs.get(fid, None)
            if fct is None:
                if hasattr(log, "error"):
                    log.error(f"/MAT/LAW28/{mat.id}: function {fid} not defined", "MAT CHECK")
                continue
            curve_x[k] = np.asarray(fct.x, dtype=float).copy()
            curve_y[k] = np.asarray(fct.y, dtype=float).copy()
            slope = getattr(fct, "slope", None)
            if slope is not None:
                curve_s[k] = np.asarray(slope, dtype=float).copy()
            elif len(curve_x[k]) > 1:
                curve_s[k] = np.diff(curve_y[k]) / np.diff(curve_x[k])
            curve_fct[k] = fct

    p["curve28_x"] = curve_x
    p["curve28_y"] = curve_y
    p["curve28_s"] = curve_s
    p["curve28_fct"] = curve_fct


def _eval_curve_k(mat: Material, k: int, x: np.ndarray) -> np.ndarray | None:
    """Evaluate curve k (0..5) at abscissae x adhering to finter.F extrapolation."""
    p = mat.params

    # 1. Direct function object if available
    fcts = p.get("curve28_fct")
    if fcts is not None and k < len(fcts) and fcts[k] is not None:
        return np.asarray(fcts[k].eval(x), dtype=float)

    # 2. Plain arrays (curve28_x, curve28_y, curve28_s)
    cxs = p.get("curve28_x")
    cys = p.get("curve28_y")
    css = p.get("curve28_s")
    if cxs is not None and cys is not None and k < len(cxs) and cxs[k] is not None and cys[k] is not None:
        xs = cxs[k]
        ys = cys[k]
        ss = css[k] if css is not None and k < len(css) else None
        out = np.interp(x, xs, ys)
        if ss is not None and len(ss) > 0:
            below = x < xs[0]
            above = x > xs[-1]
            if np.any(below):
                out = np.where(below, ys[0] + ss[0] * (x - xs[0]), out)
            if np.any(above):
                out = np.where(above, ys[-1] + ss[-1] * (x - xs[-1]), out)
        return out

    # 3. Fallback: check dictionary / list of curves in params
    curves = p.get("curves")
    if isinstance(curves, (list, tuple)) and k < len(curves) and curves[k] is not None:
        fct = curves[k]
        if hasattr(fct, "eval"):
            return np.asarray(fct.eval(x), dtype=float)
        if isinstance(fct, tuple) and len(fct) >= 2:
            xs, ys = fct[0], fct[1]
            return np.interp(x, xs, ys)

    return None


# ------------------------------------------------------------------ #
# Constitutive kernel: solid_update (sigeps28.F)
# ------------------------------------------------------------------ #

def solid_update(
    mat: Material,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict | None = None,
    return_tuple: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray] | np.ndarray:
    """Vectorized stress update for LAW28 honeycomb solid elements.

    Fortran origin: ``engine/source/materials/mat/mat028/sigeps28.F``.

    Parameters
    ----------
    mat : Material
        Material object with law=28 parameters.
    sig : (n, 6) ndarray
        Old stress tensor (Voigt components: xx, yy, zz, xy, yz, zx).
    deps : (n, 6) ndarray
        Strain increment (engineering shear: gamma_xy, gamma_yz, gamma_zx).
    epsp : (n,) ndarray or None
        Plastic strain (unused by law 28, passed through).
    dt : float
        Time step (unused in rate-independent rate integration).
    extra : dict or None
        Per-element environment and state views:
        - "eps28" (or "eps"): total strain tensor (n, 6)
        - "off28" (or "off"): element active/deleted flag (n,) (1.0 on, 0.0 off)
        - "rho": current density (n,)
        - "amu" / "AMU": relative volume expansion (n,)
    return_tuple : bool, default True
        If True, returns (sign, epsp, c). If False, returns sign.

    Returns
    -------
    (sign, epsp, c) or sign
    """
    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig.reshape(1, -1)
        deps = deps.reshape(1, -1)

    n = sig.shape[0]
    p = mat.params
    rho0 = float(mat.rho0 if mat.rho0 > 0 else p.get("rho0", 1.0))

    # Sound speed calculation: c = sqrt(max(E11, E22, E33, G12, G23, G31) / rho0)
    e11 = float(p.get("E11", 0.0))
    e22 = float(p.get("E22", 0.0))
    e33 = float(p.get("E33", 0.0))
    g12 = float(p.get("G12", 0.0))
    g23 = float(p.get("G23", 0.0))
    g31 = float(p.get("G31", 0.0))
    c_scalar = math.sqrt(max(max(e11, e22, e33, g12, g23, g31), 0.0) / rho0)

    if n == 0:
        c_empty = np.empty(0, dtype=sig.dtype)
        if return_tuple:
            return sig.copy(), epsp, c_empty
        return sig.copy()

    c = np.full(n, c_scalar, dtype=sig.dtype)

    # 1. Total strain tracking (sigeps28.F lines 212-221, 244-262)
    if extra is not None and "eps28" in extra:
        extra["eps28"] += deps
        eps = extra["eps28"]
    elif extra is not None and "eps" in extra:
        extra["eps"] += deps
        eps = extra["eps"]
    else:
        eps = deps.copy()

    if is_1d and eps.ndim == 1:
        eps = eps.reshape(1, -1)

    # 2. Element deletion check (sigeps28.F lines 197-222)
    eps_max11 = float(p.get("eps_max11", _DEFAULT_EPS_MAX))
    eps_max22 = float(p.get("eps_max22", _DEFAULT_EPS_MAX))
    eps_max33 = float(p.get("eps_max33", _DEFAULT_EPS_MAX))
    eps_max12 = float(p.get("eps_max12", _DEFAULT_EPS_MAX))
    eps_max23 = float(p.get("eps_max23", _DEFAULT_EPS_MAX))
    eps_max31 = float(p.get("eps_max31", _DEFAULT_EPS_MAX))

    off = None
    if extra is not None:
        if "off28" in extra:
            off = extra["off28"]
        elif "off" in extra:
            off = extra["off"]

    if off is not None:
        active = (off != 0.0)
        rupture = (
            (eps[:, 0] > eps_max11)
            | (eps[:, 1] > eps_max22)
            | (eps[:, 2] > eps_max33)
            | (np.abs(eps[:, 3] / 2.0) > eps_max12)
            | (np.abs(eps[:, 4] / 2.0) > eps_max23)
            | (np.abs(eps[:, 5] / 2.0) > eps_max31)
        )
        off[active & rupture] = 0.0

    # 3. Uncoupled elastic trial stress (sigeps28.F lines 175-195)
    sign = np.empty_like(sig)
    sign[:, 0] = sig[:, 0] + e11 * deps[:, 0]
    sign[:, 1] = sig[:, 1] + e22 * deps[:, 1]
    sign[:, 2] = sig[:, 2] + e33 * deps[:, 2]
    sign[:, 3] = sig[:, 3] + g12 * deps[:, 3]
    sign[:, 4] = sig[:, 4] + g23 * deps[:, 4]
    sign[:, 5] = sig[:, 5] + g31 * deps[:, 5]

    # 4. Abscissa determination and yield clamping (sigeps28.F lines 232-331)
    gflag = int(p.get("gflag", 0))
    vflag = int(p.get("vflag", 0))

    if extra is not None and "amu" in extra:
        mu = np.atleast_1d(np.asarray(extra["amu"], dtype=sig.dtype))
    elif extra is not None and "AMU" in extra:
        mu = np.atleast_1d(np.asarray(extra["AMU"], dtype=sig.dtype))
    elif extra is not None and "rho" in extra:
        rho_curr = np.atleast_1d(np.asarray(extra["rho"], dtype=sig.dtype))
        mu = rho_curr / rho0 - 1.0
    else:
        # Small-strain volumetric compression mu = rho/rho0 - 1 ≈ -tr(eps)
        mu = -(eps[:, 0] + eps[:, 1] + eps[:, 2])

    if mu.ndim == 0:
        mu = np.full(n, float(mu), dtype=sig.dtype)
    elif mu.shape[0] != n:
        mu = np.broadcast_to(mu, (n,)).astype(sig.dtype)

    fscales = [
        float(p.get("fscale11", 1.0)),
        float(p.get("fscale22", 1.0)),
        float(p.get("fscale33", 1.0)),
        float(p.get("fscale12", 1.0)),
        float(p.get("fscale23", 1.0)),
        float(p.get("fscale31", 1.0)),
    ]

    fids = [
        p.get("fun_id11", 0),
        p.get("fun_id22", 0),
        p.get("fun_id33", 0),
        p.get("fun_id12", 0),
        p.get("fun_id23", 0),
        p.get("fun_id31", 0),
    ]

    for k in range(6):
        # In sigeps28.F: if AUX == 0 (no function defined for this component), skip clamping
        if fids[k] == 0 and p.get("curve28_x") is None and p.get("curve28_fct") is None and p.get("curves") is None:
            continue

        if k < 3:
            if gflag == 1:
                xk = eps[:, k]
            elif gflag == -1:
                xk = -eps[:, k]
            else:
                xk = mu
        else:
            if vflag == 1:
                xk = eps[:, k]
            elif vflag == -1:
                xk = -eps[:, k]
            else:
                xk = mu

        yk = _eval_curve_k(mat, k, xk)
        if yk is not None:
            yk_arr = np.atleast_1d(np.asarray(yk, dtype=sig.dtype))
            if yk_arr.ndim == 0:
                yk_arr = np.full(n, float(yk_arr), dtype=sig.dtype)
            elif yk_arr.shape[0] != n:
                yk_arr = np.broadcast_to(yk_arr, (n,)).astype(sig.dtype)

            y_limit = np.maximum(0.0, yk_arr * fscales[k])
            # Clamping: sign(sigma) * min(|sigma|, Y)
            sign[:, k] = np.sign(sign[:, k]) * np.minimum(np.abs(sign[:, k]), y_limit)

    # 5. Zero stresses for deleted elements (off == 0)
    if off is not None:
        sign[off == 0.0] = 0.0

    if is_1d:
        sign = sign[0]
        c = c[0]

    if return_tuple:
        return sign, epsp, c
    return sign


# ------------------------------------------------------------------ #
# Sound speed & implicit tangent
# ------------------------------------------------------------------ #

def sound_speed(mat: Material, rho: float | None = None, extra: dict | None = None) -> float:
    """Acoustic sound speed for LAW28 honeycomb.

    Fortran origin: ``engine/source/materials/mat/mat028/sigeps28.F``:
        SOUNDSP(I) = SQRT(MAX(E11,E22,E33,G12,G23,G31)/RHO0(I))
    """
    p = mat.params
    e11 = float(p.get("E11", 0.0))
    e22 = float(p.get("E22", 0.0))
    e33 = float(p.get("E33", 0.0))
    g12 = float(p.get("G12", 0.0))
    g23 = float(p.get("G23", 0.0))
    g31 = float(p.get("G31", 0.0))
    r = rho if rho is not None and rho > 0.0 else mat.rho0
    if r <= 0.0:
        r = float(p.get("rho0", 1.0))
    c_max = max(e11, e22, e33, g12, g23, g31)
    return math.sqrt(max(c_max, 0.0) / r)


def consistent_solid_tangent(
    mat: Material,
    sig: np.ndarray,
    epsp: np.ndarray | None = None,
    epsp_incr: np.ndarray | None = None,
    extra: dict | None = None,
) -> np.ndarray:
    """Return the (n, 6, 6) algorithmic tangent matrix for LAW28 solids.

    Uncoupled orthotropic elastic tangent with zeroed rows/cols for deleted elements.
    """
    n = sig.shape[0] if sig is not None and hasattr(sig, "shape") else 0
    if n == 0:
        return np.empty((0, 6, 6), dtype=float if sig is None else sig.dtype)

    p = mat.params
    e11 = float(p.get("E11", 0.0))
    e22 = float(p.get("E22", 0.0))
    e33 = float(p.get("E33", 0.0))
    g12 = float(p.get("G12", 0.0))
    g23 = float(p.get("G23", 0.0))
    g31 = float(p.get("G31", 0.0))

    D = np.zeros((n, 6, 6), dtype=sig.dtype)
    D[:, 0, 0] = e11
    D[:, 1, 1] = e22
    D[:, 2, 2] = e33
    D[:, 3, 3] = g12
    D[:, 4, 4] = g23
    D[:, 5, 5] = g31

    if extra is not None:
        off = extra.get("off28", extra.get("off", None))
        if off is not None:
            off_arr = np.asarray(off)
            D[off_arr == 0.0] = 0.0

    return D


def shell_update(mat: Any, sig: Any, *args: Any, **kwargs: Any) -> Any:
    """LAW28 is implemented for 3D solid elements only (SOLID_ORTHOTROPIC/SPH)."""
    raise NotImplementedError("LAW28 (honeycomb) is implemented for 3D solid elements only.")


# ------------------------------------------------------------------ #
# Registration
# ------------------------------------------------------------------ #

def _register():
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY["LAW28"] = build_law28
    MAT_PHYSICS_REGISTRY["HONEYCOMB"] = build_law28
    MAT_PHYSICS_REGISTRY["HONEYCOMB_SOL"] = build_law28
    MAT_PHYSICS_REGISTRY[28] = build_law28
    MAT_PHYSICS_REGISTRY["28"] = build_law28


_register()
