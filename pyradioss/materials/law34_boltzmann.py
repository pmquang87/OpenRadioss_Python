"""
LAW34 — Boltzmann linear viscoelastic relaxation model (/MAT/LAW34, /MAT/BOLTZMAN, /MAT/VISC_MAXW).

Upstream Fortran origins:
- 3D Solids: ``engine/source/materials/mat/mat034/sigeps34.F``
- 2D Shells (Plane Stress): ``engine/source/materials/mat/mat034/sigeps34c.F``
- Starter Card Reader: ``starter/source/materials/mat/mat034/hm_read_mat34.F``
- HyperMesh CFG Schema: ``hm_cfg_files/config/CFG/radioss110/MAT/matl34_boltzman.cfg``

Theory
------
A linear viscoelastic Maxwell/Kelvin-like relaxation model with air-pressure coupling:
- Instantaneous shear modulus: G0 (short time)
- Long-term shear modulus: GI (long time)
- Elastic shear modulus: GE = GI
- Viscous shear modulus: GV = G0 - GI
- Relaxation decay constant: BETA (tau = 1 / BETA)
- Bulk modulus: BULK
- Closed-cell air pressure: P0, PHI (foam vs polymer density ratio), GAMA0 (initial volumetric strain)

Time-integration of the viscous deviatoric strain increment:
    C1 = 1 - exp(-BETA * dt)
    C2 = -C1 / BETA   (or -dt when BETA*dt -> 0)
    deps_v = C1 * (e_{n+1} - q_n) + C2 * (de / dt)
    q_{n+1} = q_n + deps_v + de

Deviatoric stress update:
    s_{n+1} = s_n + 2*GE * de - 2*GV * deps_v

Mean pressure update:
    gamma = rho0 / rho - 1 + gama0
    dp_dgama = -P0 * (1 - phi) / (1 + gamma - phi)
    DP = (3 * BULK - dp_dgama) * deps_m

Plane-Stress Shell Formulation (sigeps34c.F):
    Enforces sigma_{zz} = 0 analytically, yielding a closed-form solution for
    the through-thickness strain increment deps_zz without iterative projection.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material


# =============================================================================
# Physics Constructor / Starter Reader
# =============================================================================

def build_law34(rec: Any) -> Material:
    """Construct a Material entity for /MAT/LAW34 (/MAT/BOLTZMAN, /MAT/VISC_MAXW).

    Parameters
    ----------
    rec : GenericMaterialRecord, dict, or object
        Parsed card record from CFG or deck reader.

    Returns
    -------
    Material
        Material entity configured with law=34, elastic constants, and params.
    """
    if hasattr(rec, "params") and getattr(rec, "params") is not None:
        p = getattr(rec, "params")
    elif isinstance(rec, dict) and "params" in rec and isinstance(rec["params"], dict):
        p = rec["params"]
    elif isinstance(rec, dict):
        p = rec
    else:
        p = {}

    def _get(keys: Union[str, list[str]], default: float = 0.0) -> float:
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
    if isinstance(rec, dict):
        rec_id = rec.get("id", rec.get("mat_id", rec.get("user_id", 1)))
        title = rec.get("title", f"LAW34_{rec_id}")
    else:
        rec_id = getattr(rec, "id", getattr(rec, "mat_id", getattr(rec, "user_id", 1)))
        title = getattr(rec, "title", f"LAW34_{rec_id}")

    rho0 = _get(["MAT_RHO", "rho", "rho0", "density", "RHO", "RHO0"], default=1.0)
    if rho0 <= 0.0:
        rho0 = 1.0

    rhor = _get(["Refer_Rho", "refer_rho", "rhor", "RHOR", "REF_RHO", "ref_rho"], default=0.0)
    if rhor <= 0.0:
        rhor = rho0

    bulk = _get(["MAT_BULK", "bulk", "K", "k", "BULK"], default=0.0)
    g0 = _get(["MAT_G0", "g0", "G0", "G_INS", "g_ins", "G_ins"], default=0.0)
    gi = _get(["MAT_GI", "gi", "GI", "G_INF", "g_inf", "G_inf", "Gl", "gl", "GL"], default=0.0)
    beta = _get(["MAT_DECAY", "beta", "BETA", "decay", "Beta", "DECAY"], default=0.0)
    p0 = _get(["MAT_P0", "p0", "P0"], default=0.0)
    phi = _get(["MAT_PHI", "phi", "PHI"], default=0.0)
    gama0 = _get(["MAT_GAMA0", "gama0", "GAMA0", "gamma0", "GAMMA0"], default=0.0)

    # Derived elastic properties (hm_read_mat34.F lines 139-152)
    denom = 3.0 * bulk + g0
    if denom > 0.0:
        young = (9.0 * bulk * g0) / denom
        nu = (3.0 * bulk - 2.0 * g0) / (2.0 * denom)
    else:
        young = 0.0
        nu = 0.3

    dp_drho0 = bulk + (4.0 / 3.0) * g0
    parmat17 = (2.0 * g0) / dp_drho0 if dp_drho0 > 0.0 else 0.0
    c_solid = math.sqrt(max(0.0, dp_drho0 / rho0))
    c_bar = math.sqrt(max(0.0, young / rho0))

    params: Dict[str, Any] = {
        "rho0": rho0,
        "rhor": rhor,
        "bulk": bulk,
        "K": bulk,
        "g0": g0,
        "G0": g0,
        "gi": gi,
        "GI": gi,
        "beta": beta,
        "BETA": beta,
        "p0": p0,
        "P0": p0,
        "phi": phi,
        "PHI": phi,
        "gama0": gama0,
        "GAMA0": gama0,
        "gamma0": gama0,
        "E": young,
        "nu": nu,
        "G": g0,
        "young": young,
        "nuparam": 7,
        "nuvar": 8,
        "parmat1": bulk,
        "parmat2": young,
        "parmat16": 2,
        "parmat17": parmat17,
        "pm1": rhor,
        "pm12": math.sqrt(max(0.0, g0)),
        "pm22": g0,
        "pm27": c_bar,
        "sound_speed": c_solid,
    }

    mat = Material(id=rec_id, law=34, rho0=rho0, title=title, params=params)
    mat.law_name = "LAW34"
    return mat


def _compute_relaxation_coeffs(beta: float, dt: float) -> Tuple[float, float, float]:
    """Compute Maxwell relaxation coefficients C1, C2, and C2/dt (sigeps34.F lines 100-101).

    C1 = 1 - exp(-beta * dt)
    C2 = -C1 / beta  (with analytical limit -dt as beta -> 0)
    """
    if dt > 1e-20:
        if beta > 0.0:
            b_dt = beta * dt
            c1 = -math.expm1(-b_dt)
            c2 = -c1 / beta
        else:
            c1 = 0.0
            c2 = -dt
        c2_over_dt = c2 / dt
    else:
        c1 = 0.0
        c2 = 0.0
        c2_over_dt = -1.0
    return c1, c2, c2_over_dt


# =============================================================================
# Sound Speed
# =============================================================================

def sound_speed(mat: Material, rho: Any = None, extra: Any = None) -> Union[float, np.ndarray]:
    """Dilatational sound speed for LAW34 elements (sigeps34.F line 157-158).

    c = sqrt((BULK + 4/3 * G0) / rho0)

    Parameters
    ----------
    mat : Material
        LAW34 material.
    rho : float, ndarray, or None
        Current density.
    extra : dict or None
        Extra state.

    Returns
    -------
    float or ndarray
        Sound speed.
    """
    p = mat.params
    bulk = p.get("bulk", p.get("K", mat.K))
    g0 = p.get("g0", p.get("G0", mat.G))
    rho0 = mat.rho0 if mat.rho0 > 0.0 else 1.0

    dp_drho = bulk + (4.0 / 3.0) * g0
    c_val = math.sqrt(max(0.0, dp_drho / rho0))

    if rho is not None and isinstance(rho, np.ndarray):
        return np.full(rho.shape, c_val, dtype=float)
    return c_val


# =============================================================================
# 3D Solids: solid_update (sigeps34.F)
# =============================================================================

def solid_update(
    mat: Material,
    sig: np.ndarray,
    deps: np.ndarray,
    *args: Any,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Vectorized stress update for LAW34 3D solid elements (sigeps34.F).

    Parameters
    ----------
    mat : Material
        LAW34 material instance.
    sig : ndarray, shape (n, 6) or (6,)
        Cauchy stress in Voigt convention [xx, yy, zz, xy, yz, zx].
    deps : ndarray, shape (n, 6) or (6,)
        Strain increment with engineering shears [xx, yy, zz, xy, yz, zx].
    *args, **kwargs
        Flexible argument unpacking: (dt, extra) or (epsp, dt, extra) or
        dt=..., extra=..., epsp=...

    Returns
    -------
    (sign, epsp, c) : tuple of ndarrays
        sign : updated stress (n, 6)
        epsp : plastic strain (passed through)
        c : current dilatational sound speed (n,)
    """
    # Flexible unpack
    dt = kwargs.get("dt", None)
    extra = kwargs.get("extra", None)
    epsp = kwargs.get("epsp", None)

    if len(args) == 1:
        if dt is None:
            dt = args[0]
    elif len(args) == 2:
        a0, a1 = args
        if isinstance(a0, dict) or (isinstance(a1, dict) or a1 is None):
            dt = a0
            extra = a1
        elif isinstance(a1, (int, float, np.floating, np.integer)):
            epsp = a0
            dt = a1
        else:
            dt = a0
            extra = a1
    elif len(args) >= 3:
        epsp = args[0]
        dt = args[1]
        extra = args[2]

    if dt is None:
        dt = 0.0
    dt = float(dt)

    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig.reshape(1, -1)
        deps = deps.reshape(1, -1)

    n = sig.shape[0]
    if n == 0:
        empty_c = np.empty(0, dtype=sig.dtype)
        return (sig[0] if is_1d else sig), epsp, (empty_c[0] if is_1d else empty_c)

    if extra is None:
        extra = {}

    # State arrays: history q (deviatoric strain history) and total strain eps34
    # Fortran NUVAR=8 (hm_read_mat34.F line 129; sigeps34.F uses UVAR 1..6)
    if "uv34" not in extra or extra["uv34"] is None:
        extra["uv34"] = np.zeros((n, 8), dtype=sig.dtype)
    else:
        u = np.asarray(extra["uv34"], dtype=sig.dtype)
        if u.ndim == 1:
            u = u.reshape(n, -1) if u.size >= n * 6 else np.broadcast_to(u, (n, u.size)).copy()
        if u.ndim != 2 or u.shape[0] != n or u.shape[1] < 6:
            cols = max(8, u.shape[1] if u.ndim == 2 else 8)
            padded = np.zeros((n, cols), dtype=sig.dtype)
            if u.ndim == 2:
                r = min(n, u.shape[0])
                c_idx = min(cols, u.shape[1])
                padded[:r, :c_idx] = u[:r, :c_idx]
            extra["uv34"] = padded
        else:
            extra["uv34"] = u

    if "eps34" not in extra or extra["eps34"] is None:
        extra["eps34"] = np.zeros((n, 6), dtype=sig.dtype)
    else:
        e_arr = np.asarray(extra["eps34"], dtype=sig.dtype)
        if e_arr.ndim == 1:
            e_arr = e_arr.reshape(n, -1) if e_arr.size >= n * 6 else np.broadcast_to(e_arr, (n, e_arr.size)).copy()
        if e_arr.ndim != 2 or e_arr.shape[0] != n or e_arr.shape[1] < 6:
            cols = max(6, e_arr.shape[1] if e_arr.ndim == 2 else 6)
            padded = np.zeros((n, cols), dtype=sig.dtype)
            if e_arr.ndim == 2:
                r = min(n, e_arr.shape[0])
                c_idx = min(cols, e_arr.shape[1])
                padded[:r, :c_idx] = e_arr[:r, :c_idx]
            extra["eps34"] = padded
        else:
            extra["eps34"] = e_arr

    if "rho" not in extra or extra["rho"] is None:
        rho = np.full(n, mat.rho0, dtype=sig.dtype)
    elif np.isscalar(extra["rho"]):
        rho = np.full(n, extra["rho"], dtype=sig.dtype)
    elif len(extra["rho"]) != n:
        rho = np.full(n, mat.rho0, dtype=sig.dtype)
    else:
        rho = np.asarray(extra["rho"], dtype=sig.dtype)

    p = mat.params
    bulk = p.get("bulk", p.get("K", mat.K))
    g0 = p.get("g0", p.get("G0", mat.G))
    gi = p.get("gi", p.get("GI", g0))
    beta = p.get("beta", p.get("BETA", 0.0))
    p0 = p.get("p0", p.get("P0", 0.0))
    phi = p.get("phi", p.get("PHI", 0.0))
    gama0 = p.get("gama0", p.get("GAMA0", 0.0))

    ge = gi
    gv = g0 - gi
    ge2 = 2.0 * ge
    gv2 = 2.0 * gv
    bulk3 = 3.0 * bulk

    c1, c2, c2_over_dt = _compute_relaxation_coeffs(beta, dt)

    # Volumetric strain & air pressure (sigeps34.F lines 105-106, 146)
    rho0 = mat.rho0 if mat.rho0 > 0.0 else 1.0
    gama = rho0 / rho - 1.0 + gama0
    denom = 1.0 + gama - phi
    dpdgama = np.where(np.abs(denom) > 1e-15, -p0 * (1.0 - phi) / denom, 0.0)

    deps_m = (deps[:, 0] + deps[:, 1] + deps[:, 2]) / 3.0
    dp = (bulk3 - dpdgama) * deps_m

    # Update accumulated total strain
    eps = extra["eps34"]
    eps[:, :6] += deps[:, :6]

    em = (eps[:, 0] + eps[:, 1] + eps[:, 2]) / 3.0
    dexx = eps[:, 0] - em
    deyy = eps[:, 1] - em
    dezz = eps[:, 2] - em
    dexy = eps[:, 3]
    deyz = eps[:, 4]
    dezx = eps[:, 5]

    ddexx = deps[:, 0] - deps_m
    ddeyy = deps[:, 1] - deps_m
    ddezz = deps[:, 2] - deps_m
    ddexy = deps[:, 3]
    ddeyz = deps[:, 4]
    ddezx = deps[:, 5]

    q = extra["uv34"]
    depsvxx = c1 * (dexx - q[:, 0]) + c2_over_dt * ddexx
    depsvyy = c1 * (deyy - q[:, 1]) + c2_over_dt * ddeyy
    depsvzz = c1 * (dezz - q[:, 2]) + c2_over_dt * ddezz
    depsvxy = c1 * (dexy - q[:, 3]) + c2_over_dt * ddexy
    depsvyz = c1 * (deyz - q[:, 4]) + c2_over_dt * ddeyz
    depsvzx = c1 * (dezx - q[:, 5]) + c2_over_dt * ddezx

    sign = np.empty_like(sig)
    sign[:, 0] = sig[:, 0] + ge2 * ddexx - gv2 * depsvxx + dp
    sign[:, 1] = sig[:, 1] + ge2 * ddeyy - gv2 * depsvyy + dp
    sign[:, 2] = sig[:, 2] + ge2 * ddezz - gv2 * depsvzz + dp
    sign[:, 3] = sig[:, 3] + ge * ddexy - gv * depsvxy
    sign[:, 4] = sig[:, 4] + ge * ddeyz - gv * depsvyz
    sign[:, 5] = sig[:, 5] + ge * ddezx - gv * depsvzx

    # Update history q (sigeps34.F lines 161-166)
    q[:, 0] += depsvxx + ddexx
    q[:, 1] += depsvyy + ddeyy
    q[:, 2] += depsvzz + ddezz
    q[:, 3] += depsvxy + ddexy
    q[:, 4] += depsvyz + ddeyz
    q[:, 5] += depsvzx + ddezx

    # In-place update of sig
    sig[:] = sign

    # Dilatational sound speed
    c = np.full(n, math.sqrt(max(0.0, (bulk + (4.0 / 3.0) * g0) / rho0)), dtype=sig.dtype)

    if is_1d:
        epsp_out = epsp[0] if (epsp is not None and hasattr(epsp, "__len__")) else epsp
        return sign[0], epsp_out, c[0]
    return sign, epsp, c


# =============================================================================
# 2D Shells: shell_update (Plane Stress, sigeps34c.F)
# =============================================================================

def shell_update(
    mat: Material,
    sig: np.ndarray,
    deps: np.ndarray,
    *args: Any,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Vectorized plane-stress update for LAW34 shell elements (sigeps34c.F).

    Enforces sigma_{zz} = 0 analytically to solve for deps_zz in closed form.

    Parameters
    ----------
    mat : Material
        LAW34 material instance.
    sig : ndarray, shape (n, 3) or (n, 5) or (3,)
        In-plane stress components [xx, yy, xy] (and optional transverse shears).
    deps : ndarray, shape (n, 3) or (n, 5) or (3,)
        In-plane strain increments with engineering shear [xx, yy, xy].
    *args, **kwargs
        Flexible argument unpacking: (dt, extra) or (epsp, dt, extra) or
        dt=..., extra=..., epsp=...

    Returns
    -------
    (sign[:, :3], epsp) : tuple of ndarrays
        sign : updated in-plane stress (n, 3)
        epsp : plastic strain (passed through)
    """
    dt = kwargs.get("dt", None)
    extra = kwargs.get("extra", None)
    epsp = kwargs.get("epsp", None)

    if len(args) == 1:
        if dt is None:
            dt = args[0]
    elif len(args) == 2:
        a0, a1 = args
        if isinstance(a0, dict) or (isinstance(a1, dict) or a1 is None):
            dt = a0
            extra = a1
        elif isinstance(a1, (int, float, np.floating, np.integer)):
            epsp = a0
            dt = a1
        else:
            dt = a0
            extra = a1
    elif len(args) >= 3:
        epsp = args[0]
        dt = args[1]
        extra = args[2]

    if dt is None:
        dt = 0.0
    dt = float(dt)

    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig.reshape(1, -1)
        deps = deps.reshape(1, -1)

    n = sig.shape[0]
    if n == 0:
        return (sig[0, :3] if is_1d else sig[:, :3]), epsp

    if extra is None:
        extra = {}

    # Shell history requires at least 7 variables:
    # 0..5 = deviatoric strain history q, 6 = dezz (thickness deviatoric strain)
    # Fortran NUVAR=8 (hm_read_mat34.F line 129; sigeps34c.F uses UVAR 1..7)
    if "uv34" not in extra or extra["uv34"] is None:
        extra["uv34"] = np.zeros((n, 8), dtype=sig.dtype)
    else:
        u = np.asarray(extra["uv34"], dtype=sig.dtype)
        if u.ndim == 1:
            u = u.reshape(n, -1) if u.size >= n * 7 else np.broadcast_to(u, (n, u.size)).copy()
        if u.ndim != 2 or u.shape[0] != n or u.shape[1] < 7:
            cols = max(8, u.shape[1] if u.ndim == 2 else 8)
            padded = np.zeros((n, cols), dtype=sig.dtype)
            if u.ndim == 2:
                r = min(n, u.shape[0])
                c_idx = min(cols, u.shape[1])
                padded[:r, :c_idx] = u[:r, :c_idx]
            extra["uv34"] = padded
        else:
            extra["uv34"] = u

    if "eps34" not in extra or extra["eps34"] is None:
        extra["eps34"] = np.zeros((n, max(6, deps.shape[1])), dtype=sig.dtype)
    else:
        e_arr = np.asarray(extra["eps34"], dtype=sig.dtype)
        if e_arr.ndim == 1:
            e_arr = e_arr.reshape(n, -1) if e_arr.size >= n * 3 else np.broadcast_to(e_arr, (n, e_arr.size)).copy()
        if e_arr.ndim != 2 or e_arr.shape[0] != n or e_arr.shape[1] < 3:
            cols = max(6, deps.shape[1], e_arr.shape[1] if e_arr.ndim == 2 else 6)
            padded = np.zeros((n, cols), dtype=sig.dtype)
            if e_arr.ndim == 2:
                r = min(n, e_arr.shape[0])
                c_idx = min(cols, e_arr.shape[1])
                padded[:r, :c_idx] = e_arr[:r, :c_idx]
            extra["eps34"] = padded
        else:
            extra["eps34"] = e_arr

    p = mat.params
    bulk = p.get("bulk", p.get("K", mat.K))
    g0 = p.get("g0", p.get("G0", mat.G))
    gi = p.get("gi", p.get("GI", g0))
    beta = p.get("beta", p.get("BETA", 0.0))

    ge = gi
    gv = g0 - gi
    ge2 = 2.0 * ge
    gv2 = 2.0 * gv
    bulk3 = 3.0 * bulk

    c1, c2, c2_over_dt = _compute_relaxation_coeffs(beta, dt)

    cc2 = gv2 * (c1 + c2_over_dt)

    q = extra["uv34"]
    if "ezz34" in extra and extra["ezz34"] is not None:
        dezz_old = np.asarray(extra["ezz34"]).reshape(-1).copy()
    else:
        dezz_old = q[:, 6].copy()
    h3 = q[:, 2].copy()  # z-component of deviatoric strain history

    # Analytical solution for deps_zz assuming sign_zz = 0 (sigeps34c.F lines 107-111)
    aa = gv2 * c1 * (dezz_old - h3) + (1.0 / 3.0) * (ge2 - cc2 - bulk3) * (deps[:, 0] + deps[:, 1])
    bb = (2.0 / 3.0) * ge2 + bulk - (2.0 / 3.0) * cc2
    if np.isscalar(bb):
        bb_safe = bb if abs(bb) > 1e-30 else 1e-30
    else:
        bb_safe = np.where(np.abs(bb) < 1e-30, 1e-30, bb)
    deps_zz = aa / bb_safe

    ddezz = (2.0 / 3.0) * deps_zz - (1.0 / 3.0) * (deps[:, 0] + deps[:, 1])
    dezz_new = dezz_old + ddezz
    q[:, 6] = dezz_new
    if "ezz34" in extra and extra["ezz34"] is not None:
        extra["ezz34"][:] = dezz_new.reshape(extra["ezz34"].shape)

    # Accumulate total strains
    eps = extra["eps34"]
    n_comp = min(deps.shape[1], eps.shape[1])
    eps[:, :n_comp] += deps[:, :n_comp]

    eps_xx = eps[:, 0]
    eps_yy = eps[:, 1]
    eps_zz = 1.5 * dezz_new + 0.5 * (eps_xx + eps_yy)
    em = (eps_xx + eps_yy + eps_zz) / 3.0

    dexx = eps_xx - em
    deyy = eps_yy - em
    dezz = dezz_new
    dexy = eps[:, 2] if eps.shape[1] > 2 else np.zeros(n, dtype=sig.dtype)

    deps_m = (deps[:, 0] + deps[:, 1] + deps_zz) / 3.0
    ddexx = deps[:, 0] - deps_m
    ddeyy = deps[:, 1] - deps_m
    ddexy = deps[:, 2] if deps.shape[1] > 2 else np.zeros(n, dtype=sig.dtype)

    depsvxx = c1 * (dexx - q[:, 0]) + c2_over_dt * ddexx
    depsvyy = c1 * (deyy - q[:, 1]) + c2_over_dt * ddeyy
    depsvzz = c1 * (dezz - q[:, 2]) + c2_over_dt * ddezz
    depsvxy = c1 * (dexy - q[:, 3]) + c2_over_dt * ddexy

    dp = bulk3 * deps_m

    sign = np.empty_like(sig)
    sign[:, 0] = sig[:, 0] + ge2 * ddexx - gv2 * depsvxx + dp
    sign[:, 1] = sig[:, 1] + ge2 * ddeyy - gv2 * depsvyy + dp
    if sig.shape[1] > 2:
        sign[:, 2] = sig[:, 2] + ge * ddexy - gv * depsvxy

    # Transverse shear components if 5-component stress passed
    if sig.shape[1] > 3:
        ddeyz = deps[:, 3] if deps.shape[1] > 3 else np.zeros(n, dtype=sig.dtype)
        ddezx = deps[:, 4] if deps.shape[1] > 4 else np.zeros(n, dtype=sig.dtype)
        deyz = eps[:, 3] if eps.shape[1] > 3 else np.zeros(n, dtype=sig.dtype)
        dezx = eps[:, 4] if eps.shape[1] > 4 else np.zeros(n, dtype=sig.dtype)
        depsvyz = c1 * (deyz - q[:, 4]) + c2_over_dt * ddeyz
        depsvzx = c1 * (dezx - q[:, 5]) + c2_over_dt * ddezx
        sign[:, 3] = sig[:, 3] + ge * ddeyz - gv * depsvyz
        if sig.shape[1] > 4:
            sign[:, 4] = sig[:, 4] + ge * ddezx - gv * depsvzx
        q[:, 4] += depsvyz + ddeyz
        q[:, 5] += depsvzx + ddezx

    # Update history q (sigeps34c.F lines 174-180)
    q[:, 0] += depsvxx + ddexx
    q[:, 1] += depsvyy + ddeyy
    q[:, 2] += depsvzz + ddezz
    if sig.shape[1] > 2:
        q[:, 3] += depsvxy + ddexy

    # Thickness update (sigeps34c.F line 122)
    if "thk" in extra and extra["thk"] is not None:
        thkly = extra.get("thkly", 1.0)
        off = extra.get("off", 1.0)
        extra["thk"] += deps_zz * thkly * off

    # In-place update of sig
    sig[:, :sign.shape[1]] = sign

    out_sign = sign[:, :3]
    if is_1d:
        epsp_out = epsp[0] if (epsp is not None and hasattr(epsp, "__len__")) else epsp
        return out_sign[0], epsp_out
    return out_sign, epsp


# =============================================================================
# Consistent Tangents (Implicit Solver)
# =============================================================================

def consistent_solid_tangent(
    mat: Material,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    dt: Optional[float] = None,
) -> np.ndarray:
    """Return the (n, 6, 6) consistent algorithmic tangent for LAW34 solids.

    Combines the Maxwell/viscoelastic relaxation algorithmic shear modulus:
        G_alg = GE - GV * (C1 + C2 / dt)
    with the volumetric stiffness (Bulk modulus and air pressure coupling):
        K_t = BULK - DPDGAMA / 3

    Parameters
    ----------
    mat : Material
        LAW34 material instance.
    sig : ndarray, shape (n, 6), (6,), or None
        Stress tensor.
    epsp, epsp_incr : optional
        Plastic strain views (unused).
    extra : dict or None
        Extra dictionary containing environment density 'rho' or 'dt'.
    dt : float or None
        Current time step.

    Returns
    -------
    ndarray
        (n, 6, 6) or (6, 6) algorithmic tangent tensor.
    """
    if isinstance(sig, (int, float, np.floating, np.integer)):
        if dt is None:
            dt = float(sig)
        sig = None
    if dt is None:
        if extra is not None and isinstance(extra, dict):
            dt = extra.get("dt", 0.0)
        elif extra is not None and isinstance(extra, (int, float, np.floating, np.integer)):
            dt = float(extra)
            extra = None
        else:
            dt = 0.0
    dt = float(dt) if dt is not None else 0.0

    p = mat.params
    bulk = p.get("bulk", p.get("K", mat.K))
    g0 = p.get("g0", p.get("G0", mat.G))
    gi = p.get("gi", p.get("GI", g0))
    beta = p.get("beta", p.get("BETA", 0.0))
    p0 = p.get("p0", p.get("P0", 0.0))
    phi = p.get("phi", p.get("PHI", 0.0))
    gama0 = p.get("gama0", p.get("GAMA0", 0.0))

    ge = gi
    gv = g0 - gi

    c1, c2, c2_over_dt = _compute_relaxation_coeffs(beta, dt)
    cc = c1 + c2_over_dt
    g_alg = ge - gv * cc

    is_1d = (sig is not None and isinstance(sig, np.ndarray) and sig.ndim == 1)
    if is_1d:
        sig = sig.reshape(1, -1)

    n = sig.shape[0] if sig is not None else 1
    if sig is not None and n == 0:
        return np.zeros((0, 6, 6), dtype=sig.dtype)

    if extra is not None and isinstance(extra, dict) and "rho" in extra and extra["rho"] is not None:
        rho = np.asarray(extra["rho"], dtype=float)
        if rho.ndim == 0 or len(rho) != n:
            rho = np.full(n, rho.item() if rho.ndim == 0 else mat.rho0)
    else:
        rho = np.full(n, mat.rho0)

    rho0 = mat.rho0 if mat.rho0 > 0.0 else 1.0
    gama = rho0 / rho - 1.0 + gama0
    denom = 1.0 + gama - phi
    dpdgama = np.where(np.abs(denom) > 1e-15, -p0 * (1.0 - phi) / denom, 0.0)

    k_t = bulk - dpdgama / 3.0

    # Lamé parameters
    c11 = k_t + (4.0 / 3.0) * g_alg
    c12 = k_t - (2.0 / 3.0) * g_alg
    c44 = g_alg

    C = np.zeros((n, 6, 6), dtype=float)
    for i in range(3):
        C[:, i, i] = c11
        for j in range(3):
            if i != j:
                C[:, i, j] = c12
    for i in range(3, 6):
        C[:, i, i] = c44

    if sig is None or is_1d:
        return C[0]
    return C


def shell_membrane_tangent(
    mat: Material,
    dt: Optional[Union[float, Dict[str, Any]]] = None,
) -> np.ndarray:
    """(3, 3) plane-stress consistent tangent matrix for LAW34 shells.

    Parameters
    ----------
    mat : Material
        LAW34 material instance.
    dt : float, dict, or None
        Time step or extra state dict.

    Returns
    -------
    ndarray
        (3, 3) plane-stress tangent matrix [xx, yy, xy].
    """
    if isinstance(dt, dict):
        dt = dt.get("dt", 0.0)
    dt = float(dt) if dt is not None else 0.0

    p = mat.params
    bulk = p.get("bulk", p.get("K", mat.K))
    g0 = p.get("g0", p.get("G0", mat.G))
    gi = p.get("gi", p.get("GI", g0))
    beta = p.get("beta", p.get("BETA", 0.0))

    ge = gi
    gv = g0 - gi

    c1, c2, c2_over_dt = _compute_relaxation_coeffs(beta, dt)
    cc = c1 + c2_over_dt
    g_alg = ge - gv * cc

    denom = bulk + (4.0 / 3.0) * g_alg
    if abs(denom) < 1e-30:
        denom = 1e-30
    c11 = 4.0 * g_alg * (bulk + g_alg / 3.0) / denom
    c12 = 2.0 * g_alg * (bulk - 2.0 * g_alg / 3.0) / denom
    c33 = g_alg

    return np.array([
        [c11, c12, 0.0],
        [c12, c11, 0.0],
        [0.0, 0.0, c33],
    ], dtype=float)


def consistent_shell_tangent(
    mat: Material,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    dt: Optional[float] = None,
) -> np.ndarray:
    """(n, 3, 3) plane-stress consistent tangent for LAW34 shell layers."""
    if isinstance(sig, (int, float, np.floating, np.integer)):
        if dt is None:
            dt = float(sig)
        sig = None
    if dt is None and extra is not None and isinstance(extra, dict):
        dt = extra.get("dt", 0.0)
    elif dt is None and isinstance(extra, (int, float, np.floating, np.integer)):
        dt = float(extra)
        extra = None
    C_mat = shell_membrane_tangent(mat, dt=dt)
    if sig is None or (isinstance(sig, np.ndarray) and sig.ndim == 1):
        return C_mat
    n = sig.shape[0]
    return np.broadcast_to(C_mat, (n, 3, 3)).copy()


# =============================================================================
# 1D Truss: truss_update (sigeps34t.F)
# =============================================================================

def truss_update(
    mat: Material,
    force: np.ndarray,
    deps: np.ndarray,
    area: np.ndarray,
    al0: np.ndarray,
    al: np.ndarray,
    dt: float,
    extra: Optional[Dict[str, Any]] = None,
    off: Optional[np.ndarray] = None,
    gap: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """1D truss element stress and force update for LAW34 (sigeps34t.F).

    Parameters
    ----------
    mat : Material
        LAW34 material instance.
    force : ndarray
        Current axial force (n,).
    deps : ndarray
        Axial strain increment (n,).
    area : ndarray
        Current cross-sectional area (n,).
    al0 : ndarray
        Initial element length (n,).
    al : ndarray
        Current element length (n,).
    dt : float
        Current time step.
    extra : dict or None
        Extra dictionary containing state 'uv34_t' (n, 2) where:
        uvar[:, 0] = deviatoric strain history q_1
        uvar[:, 1] = total axial strain eps
    off : ndarray or None
        Active element flag (1=active, 0=inactive).
    gap : ndarray or None
        Initial slack/gap distance.

    Returns
    -------
    (force_new, area_new, dsig, sti) : tuple of ndarrays
        force_new : updated axial force (n,)
        area_new : updated cross-sectional area (n,)
        dsig : axial stress increment (n,)
        sti : axial stiffness (n,)
    """
    dt = float(dt) if dt is not None else 0.0
    is_1d = (np.ndim(force) == 0)
    force = np.atleast_1d(np.asarray(force, dtype=float)).copy()
    deps = np.atleast_1d(np.asarray(deps, dtype=float))
    area = np.atleast_1d(np.asarray(area, dtype=float)).copy()
    al0 = np.atleast_1d(np.asarray(al0, dtype=float))
    al = np.atleast_1d(np.asarray(al, dtype=float))
    n = len(force)

    if off is None:
        off = np.ones(n, dtype=float)
    else:
        off = np.atleast_1d(np.asarray(off, dtype=float)).copy()

    if gap is None:
        gap = np.zeros(n, dtype=float)
    else:
        gap = np.atleast_1d(np.asarray(gap, dtype=float))

    if extra is None:
        extra = {}

    if "uv34_t" not in extra or extra["uv34_t"] is None:
        extra["uv34_t"] = np.zeros((n, 2), dtype=float)
    elif extra["uv34_t"].shape[0] != n or extra["uv34_t"].shape[1] < 2:
        padded = np.zeros((n, 2), dtype=float)
        if extra["uv34_t"].ndim == 2:
            r = min(n, extra["uv34_t"].shape[0])
            c_idx = min(2, extra["uv34_t"].shape[1])
            padded[:r, :c_idx] = extra["uv34_t"][:r, :c_idx]
        extra["uv34_t"] = padded

    p = mat.params
    bulk = p.get("bulk", p.get("K", mat.K))
    g0 = p.get("g0", p.get("G0", mat.G))
    gi = p.get("gi", p.get("GI", g0))
    beta = p.get("beta", p.get("BETA", 0.0))

    ge = gi
    gv = g0 - gi
    ge2 = 2.0 * ge
    gv2 = 2.0 * gv

    c1, c2, c2_over_dt = _compute_relaxation_coeffs(beta, dt)

    uvar = extra["uv34_t"]
    # Total normal strain (sigeps34t.F lines 80-81)
    eps = uvar[:, 1] + deps
    uvar[:, 1] = eps

    # Gap activation (sigeps34t.F lines 83-85)
    gap_mask = (gap > 0.0) & (al <= (al0 - gap))
    off = np.where(gap_mask, 1.0, off)

    # Poisson contraction and stiffness (sigeps34t.F lines 87-94)
    k3 = 3.0 * bulk
    nu2_denom = k3 + ge
    nu2 = (k3 - ge2) / nu2_denom if abs(nu2_denom) > 1e-30 else 1.0
    nu2 = max(float(nu2), 1.0)

    area_new = area * (1.0 - nu2 * deps * off)
    sti = np.full(n, k3, dtype=float)

    # Strain deviators (sigeps34t.F lines 97-99)
    ddexx = deps * (2.0 / 3.0)
    depsdxx = ddexx * (1.0 / dt) if dt > 1e-20 else np.zeros(n, dtype=float)
    dexx = eps * (2.0 / 3.0)

    # Viscous strain & mean pressure (sigeps34t.F lines 101-102)
    depsvxx = c1 * (dexx - uvar[:, 0]) + c2 * depsdxx
    dp = bulk * deps

    # Stress increment & force update (sigeps34t.F lines 104-109)
    dsig = ge2 * ddexx - gv2 * depsvxx + dp
    force_new = (force + dsig * area_new) * off

    ddexx_safe = np.where(np.abs(ddexx) < 1e-20, 1e-20, np.abs(ddexx))
    sti = np.maximum(sti, np.abs(dsig / ddexx_safe)) * off
    uvar[:, 0] += depsvxx + ddexx

    if is_1d:
        return force_new[0], area_new[0], dsig[0], sti[0]
    return force_new, area_new, dsig, sti


# =============================================================================
# Integrated Beam: beam_update (sigeps34pi.F)
# =============================================================================

def beam_update(
    mat: Material,
    sig: np.ndarray,
    deps: np.ndarray,
    dt: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrated beam element stress update for LAW34 (sigeps34pi.F).

    Components: [xx (axial), xy (shear y), xz (shear z)].

    Parameters
    ----------
    mat : Material
        LAW34 material instance.
    sig : ndarray, shape (n, 3) or (3,)
        Current beam stresses [xx, xy, xz].
    deps : ndarray, shape (n, 3) or (3,)
        Strain increments [xx, xy, xz].
    dt : float
        Time step.
    extra : dict or None
        Extra dictionary containing state 'uv34_b' (n, 3) for history q.

    Returns
    -------
    (sign, eps_tot) : tuple of ndarrays
        sign : updated stress [xx, xy, xz]
        eps_tot : accumulated total strain [xx, xy, xz]
    """
    dt = float(dt) if dt is not None else 0.0
    is_1d = (np.ndim(sig) == 1)
    sig = np.atleast_2d(np.asarray(sig, dtype=float)).copy()
    deps = np.atleast_2d(np.asarray(deps, dtype=float))
    n = sig.shape[0]

    if extra is None:
        extra = {}

    if "uv34_b" not in extra or extra["uv34_b"] is None:
        extra["uv34_b"] = np.zeros((n, 3), dtype=float)
    elif extra["uv34_b"].shape[0] != n or extra["uv34_b"].shape[1] < 3:
        padded = np.zeros((n, 3), dtype=float)
        if extra["uv34_b"].ndim == 2:
            r = min(n, extra["uv34_b"].shape[0])
            c_idx = min(3, extra["uv34_b"].shape[1])
            padded[:r, :c_idx] = extra["uv34_b"][:r, :c_idx]
        extra["uv34_b"] = padded

    if "eps34_b" not in extra or extra["eps34_b"] is None:
        extra["eps34_b"] = np.zeros((n, 3), dtype=float)
    elif extra["eps34_b"].shape[0] != n or extra["eps34_b"].shape[1] < 3:
        padded = np.zeros((n, 3), dtype=float)
        if extra["eps34_b"].ndim == 2:
            r = min(n, extra["eps34_b"].shape[0])
            c_idx = min(3, extra["eps34_b"].shape[1])
            padded[:r, :c_idx] = extra["eps34_b"][:r, :c_idx]
        extra["eps34_b"] = padded

    eps = extra["eps34_b"]
    eps += deps

    p = mat.params
    bulk = p.get("bulk", p.get("K", mat.K))
    g0 = p.get("g0", p.get("G0", mat.G))
    gi = p.get("gi", p.get("GI", g0))
    beta = p.get("beta", p.get("BETA", 0.0))

    ge = gi
    gv = g0 - gi
    ge2 = 2.0 * ge
    gv2 = 2.0 * gv

    c1, c2, c2_over_dt = _compute_relaxation_coeffs(beta, dt)

    uvar = extra["uv34_b"]
    ddexx = (2.0 / 3.0) * deps[:, 0]
    ddexy = deps[:, 1]
    ddexz = deps[:, 2]

    dexx = (2.0 / 3.0) * eps[:, 0]
    dexy = eps[:, 1]
    dexz = eps[:, 2]

    depsvxx = c1 * (dexx - uvar[:, 0]) + c2_over_dt * ddexx
    depsvxy = c1 * (dexy - uvar[:, 1]) + c2_over_dt * ddexy
    depsvxz = c1 * (dexz - uvar[:, 2]) + c2_over_dt * ddexz

    dp = bulk * deps[:, 0]

    sign = np.empty_like(sig)
    sign[:, 0] = sig[:, 0] + ge2 * ddexx - gv2 * depsvxx + dp
    sign[:, 1] = sig[:, 1] + ge * ddexy - gv * depsvxy
    sign[:, 2] = sig[:, 2] + ge * ddexz - gv * depsvxz

    uvar[:, 0] += depsvxx + ddexx
    uvar[:, 1] += depsvxy + ddexy
    uvar[:, 2] += depsvxz + ddexz

    if is_1d:
        return sign[0], eps[0]
    return sign, eps


# =============================================================================
# Registry Hook
# =============================================================================

def _register() -> None:
    """Register build_law34 into MAT_PHYSICS_REGISTRY."""
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for k in (34, "34", "LAW34", "BOLTZMAN", "VISC_MAXW", "BOLTZMANN"):
            MAT_PHYSICS_REGISTRY[k] = build_law34
    except Exception:
        pass


_register()
