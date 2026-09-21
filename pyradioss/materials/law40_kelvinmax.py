r"""
LAW40 — generalized Kelvin–Maxwell visco-elasticity (/MAT/KELVINMAX).

Upstream Fortran Reference:
  - ``engine/source/materials/mat/mat040/sigeps40.F`` (SUBROUTINE SIGEPS40, lines 32–538)
  - ``starter/source/materials/mat/mat040/hm_read_mat40.F`` (SUBROUTINE HM_READ_MAT40, lines 38–226)

This is the constitutive law the ``/MAT/KELVINMAX`` corpus decks (RD-E-5200 creep/relaxation)
use — distinct from LAW35 (/MAT/LAW35, ``law35_kelvinmax.py``), the three-parameter foam law
that shares the "Kelvin" family name.

Theory
------
Linear visco-elasticity with a Prony-series shear relaxation modulus

    G(t) = G_inf + \sum_{j=1}^5 G_j \exp(-\beta_j t)        (up to 5 Maxwell branches)

and a bulk modulus K (or tabulated pressure curve P(mu_v) via fct_id / fct_unload_id).
The deviatoric stress is the hereditary integral:

    s(t) = 2 \int_0^t G(t - t') \frac{de}{dt'} dt'

split into the long-term spring ``2 G_inf e`` plus one internal stress per Maxwell branch obeying:

    dv_j/dt = 2 G_j de/dt - \beta_j v_j

Held at constant strain each branch relaxes EXPONENTIALLY with the closed-form time constant
\tau_j = 1 / \beta_j. The Fortran integrates each branch exactly over the step for a strain rate
reconstructed LINEAR in time: the memory ``EDRV`` (UVAR 5-10) holds the reconstructed rate at the
step start, the end value is extrapolated so the mid-step value equals the kernel's rate
(``EDRV <- 2*EDRN - EDRV``), and the branch ODE solution:

    v(dt) = AA + BB*dt + (v0 - AA) \exp(-\beta dt),
    AA = (G/\beta)(A - B/\beta),  BB = (G/\beta) B      (rate = A + B t)

is applied verbatim (the ``jbm037`` block of sigeps40.F).

Pressure & Tabulated Curves:
The baseline pressure is incremental: mean stress += K * tr(deps). When optional tabulated
pressure curves are supplied (fct_id / fct_unload_id), the volumetric response is evaluated
from the curve with linear interpolation and end-slope extrapolation (FINTER style), with
loading/unloading branch selection driven by peak volumetric strain tracking (extra["eps_max"]).

Criteria:
UVAR 1-4 keep the von Mises and Stassi failure criteria histories:
    ssig1 = tr(sig)
    ssig2 = 3 * (0.5 * (s_xx^2 + s_yy^2 + s_zz^2) + s_xy^2 + s_yz^2 + s_zx^2)
    UVAR 1 = sqrt(ssig2) / vmisk
    UVAR 2 = (ssig1 + sqrt(max(0, ssig1^2 + 2*astas*ssig2))) / bstas
    UVAR 3 = max(UVAR 3, UVAR 1)
    UVAR 4 = max(UVAR 4, UVAR 2)

Sound speed (the dt claim), verbatim from the Fortran:

    c = \sqrt{ K_{eff}/\rho + \frac{4 G_T}{3 \rho} },   G_T = 2 (G_inf + \sum G_j)

where G_T carries doubled moduli (deliberate upstream over-estimate for stability).
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from ..model.entities import Material

_INF = 1e30


# ============================================================================
# 1D Analytical and Helper Relations
# ============================================================================

def prony_shear_modulus(
    t: float | np.ndarray,
    G_inf: float,
    G_branches: Sequence[float],
    beta_branches: Sequence[float],
) -> float | np.ndarray:
    r"""Evaluate the Prony-series shear relaxation modulus at time t:
        G(t) = G_\infty + \sum_{j=1}^M G_j \exp(-\beta_j t)

    Cited from:
      - engine/source/materials/mat/mat040/sigeps40.F (SUBROUTINE SIGEPS40, lines 167-193, 283-447)
      - starter/source/materials/mat/mat040/hm_read_mat40.F (SUBROUTINE HM_READ_MAT40, lines 103-127)

    Parameters:
        t: Time >= 0 (scalar or array)
        G_inf: Long-term asymptotic shear modulus
        G_branches: Sequence of Maxwell branch shear moduli (up to 5)
        beta_branches: Sequence of branch decay constants beta_j = 1/tau_j
    """
    is_scalar = np.isscalar(t)
    t_arr = np.asarray(t, dtype=float)
    g_val = float(G_inf) + np.zeros_like(t_arr)
    for gj, bj in zip(G_branches, beta_branches):
        if gj != 0.0:
            g_val = g_val + float(gj) * np.exp(-float(bj) * t_arr)
    return float(g_val) if is_scalar else g_val


def kelvin_maxwell_prony_stress(
    t: float | np.ndarray,
    eps0: float,
    G_inf: float,
    G_branches: Sequence[float],
    beta_branches: Sequence[float],
    mode: str = "shear",
    K: Optional[float] = None,
) -> float | np.ndarray:
    r"""Analytical stress relaxation for a step strain eps0 held constant at time t:
    - mode="shear": Pure shear deviatoric stress:
          s_{xy}(t) = 2 G(t) \varepsilon_{xy,0}
    - mode="deviatoric": Uniaxial deviatoric stress with e_0 = (2/3) eps0:
          s_{11}(t) = 2 G(t) e_0 = (4/3) G(t) \varepsilon_0
    - mode="uniaxial": Total axial stress in 1D unconfined tension/compression:
          \sigma(t) = (K + (4/3) G(t)) \varepsilon_0  (for step strain \varepsilon_0)

    Cited from:
      - engine/source/materials/mat/mat040/sigeps40.F (SUBROUTINE SIGEPS40, lines 283-494)
      - starter/source/materials/mat/mat040/hm_read_mat40.F (SUBROUTINE HM_READ_MAT40, lines 125-175)

    Parameters:
        t: Relaxation time (scalar or array)
        eps0: Step strain magnitude
        G_inf: Long-term shear modulus
        G_branches: Maxwell branch shear moduli
        beta_branches: Maxwell branch decay constants
        mode: "shear", "deviatoric", or "uniaxial"
        K: Bulk modulus (required if mode="uniaxial", otherwise computed from G_sum with nu=0.3)
    """
    g_t = prony_shear_modulus(t, G_inf, G_branches, beta_branches)
    mode_lower = mode.lower()
    if mode_lower == "shear":
        return 2.0 * g_t * float(eps0)
    elif mode_lower == "deviatoric":
        e0 = (2.0 / 3.0) * float(eps0)
        return 2.0 * g_t * e0
    elif mode_lower == "uniaxial":
        if K is None:
            g_sum = float(G_inf) + sum(G_branches)
            K = 2.0 * g_sum * (1.0 + 0.3) / (3.0 * (1.0 - 2.0 * 0.3))
        return (float(K) + (4.0 / 3.0) * g_t) * float(eps0)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'shear', 'deviatoric', or 'uniaxial'.")


def _eval_curve(curve: Any, x: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Evaluate curve f(x) and df/dx with linear interpolation and slope extrapolation (FINTER)."""
    is_scalar = np.isscalar(x)
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))

    if curve is None:
        z = np.zeros_like(x_arr)
        return (float(z[0]), float(z[0])) if is_scalar else (z, z)

    if callable(curve):
        y = np.asarray(curve(x_arr), dtype=float)
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
        z = np.zeros_like(x_arr)
        return (float(z[0]), float(z[0])) if is_scalar else (z, z)

    if len(xs) < 2:
        z = np.zeros_like(x_arr)
        val = ys[0] if len(ys) > 0 else 0.0
        return (float(val), 0.0) if is_scalar else (np.full_like(x_arr, val), z)

    if not np.all(np.diff(xs) > 0):
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]

    slopes = np.diff(ys) / np.maximum(np.diff(xs), 1e-20)
    idx = np.clip(np.searchsorted(xs, x_arr, side="right"), 1, len(xs) - 1)
    y = ys[idx - 1] + slopes[idx - 1] * (x_arr - xs[idx - 1])
    s = slopes[idx - 1]
    return (float(np.squeeze(y)), float(np.squeeze(s))) if is_scalar else (y, s)


def tabulated_kelvin_stress(
    eps: float | np.ndarray,
    deps_dt: float | np.ndarray = 0.0,
    G_inf: float = 0.0,
    G_branches: Optional[Sequence[float]] = None,
    beta_branches: Optional[Sequence[float]] = None,
    curve: Any = None,
    fscale: float = 1.0,
    unload: bool = False,
) -> float | np.ndarray:
    r"""Evaluate 1D stress combining viscoelastic Prony or baseline response with tabulated curve.

    If curve is given, evaluates fscale * curve(eps). If viscoelastic parameters are given,
    combines with instantaneous stiffness 2 * G_sum * eps.

    Cited from:
      - engine/source/materials/mat/mat040/sigeps40.F (SUBROUTINE SIGEPS40)
    """
    is_scalar = np.isscalar(eps)
    eps_arr = np.asarray(eps, dtype=float)
    sig = np.zeros_like(eps_arr)

    if curve is not None:
        y_val, _ = _eval_curve(curve, eps_arr)
        sig = sig + float(fscale) * y_val

    if G_branches is not None and beta_branches is not None:
        g_sum = float(G_inf) + sum(G_branches)
        sig = sig + 2.0 * g_sum * eps_arr

    return float(np.squeeze(sig)) if is_scalar else sig


# ============================================================================
# Tabulated Curve Resolution and Evaluation
# ============================================================================

def resolve(mat: Material, model, log=None) -> None:
    """Pull the optional tabulated loading and unloading curves (fct_id, fct_unload_id)
    into plain arrays (deck order between /MAT and /FUNCT is free).

    Cited from:
      - starter/source/materials/mat/mat040/hm_read_mat40.F (function resolution)
    """
    p = mat.params
    fct_id = p.get("fct_id") or p.get("fct_load_id")
    if fct_id:
        fct = model.functions.get(fct_id) if hasattr(model, "functions") else None
        if fct is None:
            if log is not None and hasattr(log, "error"):
                log.error(f"/MAT/KELVINMAX/{mat.id}: loading function {fct_id} not defined", "MAT CHECK")
        else:
            p["pc_x"], p["pc_y"] = fct.x.copy(), fct.y.copy()
            if hasattr(fct, "slope"):
                p["pc_s"] = fct.slope.copy()
            else:
                slopes = np.diff(fct.y) / np.maximum(np.diff(fct.x), 1e-20)
                p["pc_s"] = np.append(slopes, slopes[-1] if len(slopes) > 0 else 0.0)

    fct_unload_id = p.get("fct_unload_id")
    if fct_unload_id:
        fctu = model.functions.get(fct_unload_id) if hasattr(model, "functions") else None
        if fctu is None:
            if log is not None and hasattr(log, "error"):
                log.error(f"/MAT/KELVINMAX/{mat.id}: unloading function {fct_unload_id} not defined", "MAT CHECK")
        else:
            p["pcu_x"], p["pcu_y"] = fctu.x.copy(), fctu.y.copy()
            if hasattr(fctu, "slope"):
                p["pcu_s"] = fctu.slope.copy()
            else:
                slopes = np.diff(fctu.y) / np.maximum(np.diff(fctu.x), 1e-20)
                p["pcu_s"] = np.append(slopes, slopes[-1] if len(slopes) > 0 else 0.0)


def _curve(p: dict, x: np.ndarray | float, unload: bool | np.ndarray = False):
    """f(x) and f'(x) of the tabulated curve with linear interpolation and slope extrapolation.
    Selects unloading curve (pcu_x, pcu_y, pcu_s) if unload=True and present, else (pc_x, pc_y, pc_s).
    """
    is_scalar = np.isscalar(x)
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))

    if "pc_x" not in p and "pcu_x" not in p:
        z = np.zeros_like(x_arr)
        return (float(z[0]), float(z[0])) if is_scalar else (z, z)

    has_unloading = "pcu_x" in p
    unload_arr = np.atleast_1d(np.asarray(unload, dtype=bool))
    if len(unload_arr) == 1 and len(x_arr) > 1:
        unload_arr = np.full(len(x_arr), unload_arr[0], dtype=bool)

    y = np.zeros_like(x_arr)
    s = np.zeros_like(x_arr)

    load_mask = ~unload_arr if has_unloading else np.ones(len(x_arr), dtype=bool)
    if np.any(load_mask) and "pc_x" in p:
        tx, ty, ts = p["pc_x"], p["pc_y"], p["pc_s"]
        if len(tx) >= 2:
            xl = x_arr[load_mask]
            idx = np.clip(np.searchsorted(tx, xl, side="right"), 1, len(tx) - 1)
            y[load_mask] = ty[idx - 1] + ts[idx - 1] * (xl - tx[idx - 1])
            s[load_mask] = ts[idx - 1]

    if has_unloading and np.any(unload_arr):
        tx, ty, ts = p["pcu_x"], p["pcu_y"], p["pcu_s"]
        if len(tx) >= 2:
            xu = x_arr[unload_arr]
            idx = np.clip(np.searchsorted(tx, xu, side="right"), 1, len(tx) - 1)
            y[unload_arr] = ty[idx - 1] + ts[idx - 1] * (xu - tx[idx - 1])
            s[unload_arr] = ts[idx - 1]

    return (float(y[0]), float(s[0])) if is_scalar else (y, s)


# ============================================================================
# Material Constructor
# ============================================================================

def build_law40(rec) -> Material:
    """Physics constructor for the cfg-parsed /MAT/KELVINMAX record
    (cfg ``matl40_kelvinmax.cfg`` / hm_read_mat40.F). Supports both CFG
    and direct parameter dictionaries/objects."""
    p = rec.params if hasattr(rec, "params") else (rec if isinstance(rec, dict) else {})

    def _get(keys, default=0.0):
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            v = p.get(k)
            if v is not None:
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
        return default

    rec_id = getattr(rec, "id", getattr(rec, "mat_id", 1))

    ak = _get(["MAT_BULK", "bulk", "k", "K", "BULK"])
    g_inf = _get(["MAT_GI", "gi", "g_inf", "GI"])
    if ak <= 0.0:
        raise ValueError(f"/MAT/KELVINMAX/{rec_id}: bulk modulus K must be > 0")

    # Up to 5 Maxwell branches: G0, G2, G3, G4, G5
    g_keys = [
        ["MAT_G0", "g0", "g1", "G0", "G1"],
        ["MAT_G2", "g2", "G2"],
        ["MAT_G3", "g3", "G3"],
        ["MAT_G4", "g4", "G4"],
        ["MAT_G5", "g5", "G5"],
    ]
    decay_keys = [
        ["MAT_DECAY", "decay0", "decay1", "beta0", "beta1", "DECAY0"],
        ["MAT_DECAY2", "decay2", "beta2", "DECAY2"],
        ["MAT_DECAY3", "decay3", "beta3", "DECAY3"],
        ["MAT_DECAY4", "decay4", "beta4", "DECAY4"],
        ["MAT_DECAY5", "decay5", "beta5", "DECAY5"],
    ]

    # Support list or individual values for G and beta
    p_g = p.get("G")
    p_beta = p.get("beta")

    gs = []
    betas = []
    for j in range(5):
        if isinstance(p_g, (list, tuple)) and j < len(p_g):
            gj = float(p_g[j])
        else:
            gj = _get(g_keys[j])
        gs.append(gj)

        if isinstance(p_beta, (list, tuple)) and j < len(p_beta):
            bj = float(p_beta[j])
        else:
            bj = _get(decay_keys[j])
        betas.append(max(bj, 1e-20))

    astas = _get(["Astass", "astas", "ASTASS"])
    bstas = _get(["Bstass", "bstas", "BSTASS"])
    vmisk = _get(["Kvm", "vmisk", "KVM"])

    gsum = g_inf + sum(gs)
    if gsum <= 0.0:
        raise ValueError(f"/MAT/KELVINMAX/{rec_id}: no shear stiffness (G_inf + sum G_i must be > 0)")

    # Derived elastic estimate for generic machinery (contact, starter dt)
    nu = (3.0 * ak - 2.0 * gsum) / (2.0 * (3.0 * ak + gsum))
    nu = min(max(nu, 0.0), 0.4995)
    e = 9.0 * ak * gsum / (3.0 * ak + gsum)

    # Optional tabulated curve parameters
    fct_id = int(_get(["FUN_A1", "fct_id", "fct", "FUNCT_ID", "fct_load_id"], 0.0))
    fct_unload_id = int(_get(["fct_unload_id", "FUN_A2", "FUNCT_UNLOAD_ID"], 0.0))
    fscale = _get(["IFscale", "fscale", "ifscale", "FSCALE"], 1.0)
    if fscale == 0.0:
        fscale = 1.0
    fscale_unload = _get(["fscale_unload", "FSCALE_UNLOAD"], 1.0)
    if fscale_unload == 0.0:
        fscale_unload = 1.0
    itype = int(_get(["Itype", "itype", "ITYPE"], 0.0))

    params = {
        "E": e, "nu": nu,
        "K": ak, "G_sum": gsum,
        "K40": ak, "G_inf": g_inf, "G": gs, "G_branches": gs, "beta": betas,
        "astas": astas if astas > 1e-20 else _INF,
        "bstas": bstas if bstas > 1e-20 else _INF,
        "vmisk": vmisk if vmisk > 1e-20 else _INF,
        "fct_id": fct_id, "fct_unload_id": fct_unload_id,
        "fscale": fscale, "fscale_unload": fscale_unload,
        "itype": itype,
    }
    if isinstance(rec, dict):
        density = float(rec.get("density") or rec.get("rho") or rec.get("rho0") or 1.0)
    else:
        density = getattr(rec, "density", getattr(rec, "rho", getattr(rec, "rho0", 1.0)))
    if isinstance(density, (int, float)):
        density = float(density)
    else:
        density = 1.0
    title = getattr(rec, "title", f"LAW40_{rec_id}")
    return Material(id=rec_id, law=40, rho0=density, title=title, params=params)


# ============================================================================
# Shell & Solid Stress Updates
# ============================================================================

def shell_update(mat, sig, deps, epsp=None, dt=0.0, extra=None):
    """Raise NotImplementedError as LAW40 is solid-only in OpenRadioss."""
    raise NotImplementedError(
        "LAW40 (generalized Kelvin-Maxwell) is implemented for 3D solid elements only."
    )


def solid_update(mat, sig, deps, *args, **kwargs):
    """One SIGEPS40 cycle, vectorized over the group.  Returns (sig, c).
    ``extra`` carries eps40/uv40 and the kernel density ``rho``.

    Cited from:
      - engine/source/materials/mat/mat040/sigeps40.F (SUBROUTINE SIGEPS40, lines 32–538)
      - starter/source/materials/mat/mat040/hm_read_mat40.F (lines 38–226)
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

    n = sig.shape[0]
    if n == 0:
        return sig, np.empty(0, dtype=sig.dtype)

    if extra is None:
        extra = {}
    if "eps40" not in extra or extra["eps40"] is None:
        extra["eps40"] = np.zeros((n, 6), dtype=sig.dtype)
    elif extra["eps40"].shape[0] != n:
        extra["eps40"] = np.zeros((n, 6), dtype=sig.dtype)

    if "uv40" not in extra or extra["uv40"] is None:
        extra["uv40"] = np.zeros((n, 40), dtype=sig.dtype)
    elif extra["uv40"].shape[0] != n:
        extra["uv40"] = np.zeros((n, 40), dtype=sig.dtype)

    if "rho" not in extra or extra["rho"] is None:
        extra["rho"] = np.full(n, mat.rho0, dtype=sig.dtype)
    elif np.isscalar(extra["rho"]):
        extra["rho"] = np.full(n, extra["rho"], dtype=sig.dtype)
    elif len(extra["rho"]) != n:
        extra["rho"] = np.full(n, mat.rho0, dtype=sig.dtype)

    p = mat.params
    eps = extra["eps40"]
    eps += deps                                     # total strain (global)
    uv = extra["uv40"]
    rho = extra["rho"]

    ak = p["K40"]
    g0 = 2.0 * p["G_inf"]
    g_branches = p.get("G_branches", p["G"])
    gt = g0 + 2.0 * sum(g_branches)

    # deviatoric total strain (tensor shears) and deviatoric strain rate
    ev = (eps[:, 0] + eps[:, 1] + eps[:, 2]) / 3.0
    ed = eps.copy()
    for k in range(3):
        ed[:, k] -= ev
    ed[:, 3:] *= 0.5

    edrv = uv[:, 4:10]

    if dt <= 0.0:
        # Static solve / cycle 0 initialization
        rate = np.zeros_like(deps)
        edrn = np.zeros_like(deps)
    else:
        rate = deps / dt
        evr = (rate[:, 0] + rate[:, 1] + rate[:, 2]) / 3.0
        edrn = rate.copy()
        for k in range(3):
            edrn[:, k] -= evr
        edrn[:, 3:] *= 0.5

        # linear-in-time rate reconstruction (EDRV memory, UVAR 5-10)
        a = edrv.copy()
        b = 2.0 * (edrn - edrv) / dt

        # exact branch integration (the jbm037 block, verbatim)
        for j in range(5):
            gj = 2.0 * g_branches[j]
            if gj == 0.0:
                continue
            beta = p["beta"][j]
            sdv = uv[:, 10 + 6 * j:16 + 6 * j]
            aa = gj / beta * (a - b / beta)
            bb = gj / beta * b
            cc = sdv - aa
            sdv[:] = aa + bb * dt + cc * math.exp(-beta * dt)
        uv[:, 4:10] = 2.0 * edrn - edrv

    # incremental pressure + Prony deviator
    sigv = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0 \
        + ak * (deps[:, 0] + deps[:, 1] + deps[:, 2])

    # Tabulated pressure curve handling (if fct_id or pc_x present)
    has_curve = ("pc_x" in p) or bool(p.get("fct_id"))
    k_eff = ak
    if has_curve and ("pc_x" in p):
        mu_v = np.where(
            np.abs(rho - mat.rho0) > 1e-12,
            rho / mat.rho0 - 1.0,
            -(eps[:, 0] + eps[:, 1] + eps[:, 2]),
        )
        abs_mu = np.abs(mu_v)
        if "eps_max" not in extra or extra["eps_max"] is None:
            extra["eps_max"] = np.zeros(n, dtype=sig.dtype)
        elif len(extra["eps_max"]) != n:
            extra["eps_max"] = np.zeros(n, dtype=sig.dtype)

        eps_max = extra["eps_max"]
        is_unload = abs_mu < (eps_max - 1e-12)
        eps_max[:] = np.maximum(eps_max, abs_mu)

        fy, fs = _curve(p, abs_mu, unload=is_unload)
        fscale_eff = np.where(is_unload, p.get("fscale_unload", p["fscale"]), p["fscale"])
        p_tab = fscale_eff * fy * np.sign(mu_v)
        if p.get("itype", 0) == 0:
            sigv = -p_tab
        else:
            sigv = sigv - p_tab
        k_eff = ak + np.abs(fscale_eff * fs)

    if dt <= 0.0 and np.all(deps == 0.0) and np.all(eps == 0.0) and np.all(uv[:, 10:] == 0.0):
        # Cycle 0 static check: preserve initial input stress directly
        s = sig.copy()
        for k in range(3):
            s[:, k] -= (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
    else:
        s = g0 * ed
        for j in range(5):
            if g_branches[j] != 0.0:
                s += uv[:, 10 + 6 * j:16 + 6 * j]

        sig[:] = s
        for k in range(3):
            sig[:, k] += sigv

    # sound speed — verbatim Fortran (GT doubled: stable over-estimate)
    c = np.sqrt(np.maximum(1e-30, k_eff / rho + 4.0 * gt / (3.0 * rho)))

    # von Mises / Stassi criteria histories (UVAR 1-4)
    ssig1 = sig[:, 0] + sig[:, 1] + sig[:, 2]
    ssig2 = 3.0 * (0.5 * (s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2)
                   + s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
    uv[:, 0] = np.where(ssig2 > 0.0, np.sqrt(ssig2) / p["vmisk"], 0.0)
    disc = ssig1 ** 2 + 2.0 * p["astas"] * ssig2
    uv[:, 1] = np.where(disc > 0.0,
                        (ssig1 + np.sqrt(np.maximum(disc, 0.0)))
                        / p["bstas"],
                        ssig1 / p["bstas"])
    uv[:, 2] = np.maximum(uv[:, 2], uv[:, 0])
    uv[:, 3] = np.maximum(uv[:, 3], uv[:, 1])
    return sig, c


def consistent_solid_tangent(mat: Material, sig: np.ndarray, epsp=None,
                             epsp_incr=None, extra=None, dt=0.0) -> np.ndarray:
    """Return the (n, 6, 6) consistent algorithmic tangent for LAW40 solids.

    Combines bulk modulus K with Maxwell Prony-series visco-elastic
    relaxation factors h_j(dt) = (1 - exp(-beta_j * dt)) / (beta_j * dt).

    Cited from:
      - engine/source/materials/mat/mat040/sigeps40.F (SIGEPS40 lines 283-494)
      - starter/source/materials/mat/mat040/hm_read_mat40.F (lines 125-175)
    """
    n = sig.shape[0]
    if n == 0:
        return np.zeros((0, 6, 6), dtype=sig.dtype)

    p = mat.params
    k_bulk = p["K40"]
    g_inf = p["G_inf"]

    if (dt is None or dt == 0.0) and extra is not None and isinstance(extra, dict) and "dt" in extra:
        dt_val = float(extra["dt"])
    else:
        dt_val = float(dt) if dt is not None else 0.0
    g_eff = g_inf
    g_branches = p.get("G_branches", p["G"])
    for j in range(5):
        gj = g_branches[j]
        if gj == 0.0:
            continue
        beta = p["beta"][j]
        if dt_val > 0.0:
            h = (1.0 - math.exp(-beta * dt_val)) / (beta * dt_val)
        else:
            h = 1.0
        g_eff += gj * h

    if "pc_x" in p and extra is not None and "eps40" in extra:
        eps = extra["eps40"]
        ev = eps[:, 0] + eps[:, 1] + eps[:, 2]
        _, fs = _curve(p, np.abs(ev))
        fscale = p.get("fscale", 1.0)
        k_bulk = k_bulk + fscale * np.mean(fs)

    c11 = k_bulk + (4.0 / 3.0) * g_eff
    c12 = k_bulk - (2.0 / 3.0) * g_eff
    c44 = g_eff

    C = np.zeros((n, 6, 6), dtype=sig.dtype)
    for i in range(3):
        C[:, i, i] = c11
        for j in range(3):
            if i != j:
                C[:, i, j] = c12
    for i in range(3, 6):
        C[:, i, i] = c44

    return C


def _register():
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY["KELVINMAX"] = build_law40
    MAT_PHYSICS_REGISTRY["LAW40"] = build_law40


_register()
