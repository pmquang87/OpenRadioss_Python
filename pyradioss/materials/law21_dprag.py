r"""LAW21 — Drucker-Prager parabolic/linear yield surface material model with compaction EOS (/MAT/LAW21, /MAT/DPRAG).

Fortran origins:
- ``engine/source/materials/mat/mat021/m21law.F`` (solid constitutive update)
- ``starter/source/materials/mat/mat021/hm_read_mat21.F`` (starter card reader, defaults & parameter estimation)
- ``hm_cfg_files/config/CFG/radioss110/MAT/matl21_dprag.cfg`` (CFG attributes & card format)

Theory
------
LAW21 models geological media, concrete, rock, and soils using:
1. Pressure-dependent parabolic / linear Drucker-Prager yield surface:
   \(F = J_2 - G_0(P_{tot}) \le 0\)
   where
   \(G_0(P_{tot}) = A_0 + A_1 P_{tot} + A_2 P_{tot}^2\)
   with \(P_{tot} = P + P_{ext}\) (\(P_{ext}\) external pressure shift / PSH).
   Capped by a von Mises limit \(A_{max}\), tensile fracture pressure \(P_{min}\) (tensile cutoff),
   and pressure axis root \(P^*\) closure:
   - If \(A_2 = 0\) and \(A_1 \ne 0\): \(P^* = -A_0 / A_1\)
   - If \(A_2 \ne 0\): \(\Delta = A_1^2 - 4 A_0 A_2\); if \(\Delta \ge 0\), \(P^* = (-A_1 + \sqrt{\Delta}) / (2 A_2)\) else \(P^* = -\infty\)
   - If \(P < P_{min}\) or \(P_{tot} \le P^*\), the yield envelope collapses to zero (\(G_0 = 0\)).
2. Radial return projection of deviatoric trial stress onto the yield envelope:
   \(s_{ij}^{trial} = s_{ij}^{old} + 2 G (\Delta\varepsilon_{ij} - \frac{1}{3} tr(\Delta\boldsymbol{\varepsilon})\delta_{ij})\)
   \(J_2 = \frac{1}{2} \mathbf{s}^{trial} : \mathbf{s}^{trial}\)
   \(\text{ratio} = 1.0\) if \(J_2 \le G_0\) and \(G_0 > 0\) else \(\sqrt{\frac{G_0}{J_2 + 10^{-14}}}\) (0 if \(G_0 \le 0\))
   \(s_{ij} = \text{ratio} \cdot s_{ij}^{trial} \cdot \text{off}\)
   Cauchy stress: \(\sigma_{ij} = s_{ij} - P \delta_{ij}\) (compression positive pressure convention).
   Plastic strain increment: \(\Delta\varepsilon_p = (1 - \text{ratio}) \sqrt{J_2} / (3 G)\).
3. Compaction equation of state for volumetric behavior:
   Volumetric strain: \(\mu = \rho / \rho_0 - 1\)
   Loading pressure from function ``ifunc``:
   \(P_{load}(\mu) = F_{scale} \cdot finter(ifunc, \mu)\) with tangent bulk modulus \(K_t = \partial P / \partial \mu\).
   (If no curve is provided, linear bulk elasticity \(P_{load}(\mu) = C_1 \mu\)).
   Maximum historical compaction: \(\mu_{bak} \leftarrow \max(\mu_{bak}, \min(\mu_{max}, \mu))\).
   Evolving unloading bulk modulus:
   \(\alpha = \min(1, \max(0, \mu_{bak} / \mu_{max}))\) if \(\mu_{max} > 0\) else 1.0
   \(K_{unload} = \alpha B_{max} + (1 - \alpha) B_{min}\) with \(B_{max} = B_{unl}\), \(B_{min} = C_1\).
   Hysteretic unloading pressure:
   \(P_{unl} = P_{load}(\mu_{bak}) - (\mu_{bak} - \mu) K_{unload}\).
   Effective pressure:
   \(P = \min(P_{unl}, P_{load})\) if \(\mu_{bak} > \mu_{min}\) else \(P_{load}\).
   Tensile cutoff: \(P = \max(P, P_{min}) \cdot \text{off}\).
4. Acoustic sound speed:
   \(K_{eff} = \max(K_{unload}, \partial P / \partial \mu)\)
   \(c_{solid} = \sqrt{|\frac{4}{3} G + K_{eff}| / \rho_0}\).
5. Algorithmic consistent solid tangent tensor \(\mathbf{D} = \frac{\partial \boldsymbol{\sigma}}{\partial \boldsymbol{\varepsilon}}\) (Voigt 6x6):
   \(\mathbf{D} = K_t (\mathbf{m} \otimes \mathbf{m}) + r \mathbf{C}_{dev} - \frac{r G}{J_2} (\mathbf{s}^{trial} \otimes \mathbf{s}^{trial}) - \frac{r K_t}{2 G_0} \left(\frac{\partial G_0}{\partial P_{tot}}\right) (\mathbf{s}^{trial} \otimes \mathbf{m})\).

Solids only (SOLID_ISOTROPIC + SPH). Plane-stress shells raise NotImplementedError.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pyradioss.model.entities import Material

_EM14 = 1e-14
_EM20 = 1e-20
_EP20 = 1e20
_INF = 1e30
_MUMIN = -1e20


# -----------------------------------------------------------------------------
# Parameter extraction and validation (hm_read_mat21.F)
# -----------------------------------------------------------------------------

def _ensure_params(mat: Material | dict) -> dict[str, Any]:
    """Ensure material params contain both CFG and direct keys with robust defaults.

    Cites hm_read_mat21.F lines 90-220.
    """
    if hasattr(mat, "params") and mat.params is not None:
        p = mat.params
    elif isinstance(mat, dict):
        p = mat.get("params", mat)
    else:
        p = getattr(mat, "params", {})

    # Density rho0
    rho0_val = (
        getattr(mat, "rho0", None)
        if hasattr(mat, "rho0") and getattr(mat, "rho0") is not None
        else (
            p.get("rho0")
            if p.get("rho0") is not None
            else (
                p.get("density")
                if p.get("density") is not None
                else (p.get("rho") if p.get("rho") is not None else (p.get("MAT_RHO") if p.get("MAT_RHO") is not None else p.get("Refer_Rho")))
            )
        )
    )
    if rho0_val is None or float(rho0_val) <= 0.0:
        raise ValueError(f"LAW21: Density rho0 must be > 0 (got {rho0_val})")
    rho0 = float(rho0_val)

    # Young's modulus E
    e_val = p.get("E") if p.get("E") is not None else (p.get("MAT_E") if p.get("MAT_E") is not None else p.get("e"))
    if e_val is None or float(e_val) <= 0.0:
        raise ValueError(f"LAW21: Young's modulus E must be > 0 (got {e_val})")
    e = float(e_val)

    # Poisson's ratio nu in [0, 0.5)
    nu_val = p.get("nu") if p.get("nu") is not None else (p.get("MAT_NU") if p.get("MAT_NU") is not None else p.get("poisson"))
    if nu_val is None:
        raise ValueError("LAW21: Poisson's ratio nu must be defined")
    nu = float(nu_val)
    if not (0.0 <= nu < 0.5):
        raise ValueError(f"LAW21: Poisson's ratio nu must be in [0, 0.5) (got {nu})")

    # Elastic shear and bulk moduli
    g = e / (2.0 * (1.0 + nu))
    k = e / (3.0 * (1.0 - 2.0 * nu))

    # Drucker-Prager coefficients A0, A1, A2
    a0 = float(p.get("a0") if p.get("a0") is not None else (p.get("A0") if p.get("A0") is not None else (p.get("MAT_A0") or 0.0)))
    a1 = float(p.get("a1") if p.get("a1") is not None else (p.get("A1") if p.get("A1") is not None else (p.get("MAT_A1") or 0.0)))
    a2 = float(p.get("a2") if p.get("a2") is not None else (p.get("A2") if p.get("A2") is not None else (p.get("MAT_A2") or 0.0)))

    # von Mises cap Amax (defaults to 1e20 if 0 or None, hm_read_mat21.F line 122)
    amax_val = (
        p.get("amax")
        if p.get("amax") is not None
        else (
            p.get("Amax")
            if p.get("Amax") is not None
            else (p.get("AMAX") if p.get("AMAX") is not None else (p.get("AMX") if p.get("AMX") is not None else p.get("MAT_AMAX")))
        )
    )
    amax = float(amax_val) if amax_val is not None and float(amax_val) != 0.0 else _EP20

    # Function describing P vs Mu
    ifunc_val = p.get("ifunc") if p.get("ifunc") is not None else (p.get("IFUNC") if p.get("IFUNC") is not None else p.get("FUN_A1"))
    ifunc = int(ifunc_val) if ifunc_val is not None else 0

    # PFscale: Y-scale factor for pressure function (default 1.0)
    pfscale_val = (
        p.get("pfscale")
        if p.get("pfscale") is not None
        else (
            p.get("PFscale")
            if p.get("PFscale") is not None
            else (p.get("fscale") if p.get("fscale") is not None else (p.get("fac_y") if p.get("fac_y") is not None else p.get("FACY")))
        )
    )
    pfscale = float(pfscale_val) if pfscale_val is not None and float(pfscale_val) != 0.0 else 1.0

    # Tensile bulk modulus C1 (MAT_BULK / bmin > 0, default K)
    c1_val = (
        p.get("c1")
        if p.get("c1") is not None
        else (
            p.get("C1")
            if p.get("C1") is not None
            else (p.get("bmin") if p.get("bmin") is not None else (p.get("BMIN") if p.get("BMIN") is not None else p.get("MAT_BULK")))
        )
    )
    c1 = float(c1_val) if c1_val is not None and float(c1_val) != 0.0 else k
    if c1 <= 0.0:
        raise ValueError(f"LAW21: Tensile bulk modulus C1 must be > 0 (got {c1})")

    # Unloading bulk modulus bunl (MAT_K_UNLOAD / bmax, default C1)
    bunl_val = (
        p.get("bunl")
        if p.get("bunl") is not None
        else (
            p.get("BUNL")
            if p.get("BUNL") is not None
            else (p.get("bmax") if p.get("bmax") is not None else (p.get("BMAX") if p.get("BMAX") is not None else p.get("MAT_K_UNLOAD")))
        )
    )
    bunl = float(bunl_val) if bunl_val is not None and float(bunl_val) != 0.0 else c1

    # Tension fracture pressure pmin (MAT_PC, defaults to -1e30 if 0 or None, hm_read_mat21.F line 119)
    pmin_val = (
        p.get("pmin")
        if p.get("pmin") is not None
        else (
            p.get("PMIN")
            if p.get("PMIN") is not None
            else (p.get("MAT_PC") if p.get("MAT_PC") is not None else p.get("p_min"))
        )
    )
    pmin = float(pmin_val) if pmin_val is not None and float(pmin_val) != 0.0 else -_INF

    # External pressure shift pext / psh (PEXT, default 0.0)
    pext_val = (
        p.get("pext")
        if p.get("pext") is not None
        else (
            p.get("PEXT")
            if p.get("PEXT") is not None
            else (p.get("psh") if p.get("psh") is not None else (p.get("PSH") if p.get("PSH") is not None else p.get("MAT_PSH")))
        )
    )
    pext = float(pext_val) if pext_val is not None else 0.0

    # Maximum volumetric compression mumax (MAT_SIG / xmumx, default 1e20)
    mumax_val = (
        p.get("mumax")
        if p.get("mumax") is not None
        else (
            p.get("MUMAX")
            if p.get("MUMAX") is not None
            else (p.get("xmumx") if p.get("xmumx") is not None else (p.get("XMUMX") if p.get("XMUMX") is not None else p.get("MAT_SIG")))
        )
    )
    mumax = float(mumax_val) if mumax_val is not None and float(mumax_val) != 0.0 else _EP20

    # Pressure root pstar (hm_read_mat21.F lines 146-160)
    pstar_val = p.get("pstar") if p.get("pstar") is not None else p.get("PSTAR")
    if pstar_val is not None:
        pstar = float(pstar_val)
    elif a2 == 0.0 and a1 != 0.0:
        pstar = -a0 / a1
    elif a2 != 0.0:
        delta = a1**2 - 4.0 * a0 * a2
        if delta >= 0.0:
            pstar = (-a1 + math.sqrt(delta)) / (2.0 * a2)
        else:
            pstar = -_INF
    else:
        pstar = -_INF

    # Store normalized parameters back into dictionary
    p["rho0"] = rho0
    p["E"] = e
    p["nu"] = nu
    p["G"] = g
    p["K"] = k
    p["a0"] = a0
    p["a1"] = a1
    p["a2"] = a2
    p["A0"] = a0
    p["A1"] = a1
    p["A2"] = a2
    p["amax"] = amax
    p["Amax"] = amax
    p["ifunc"] = ifunc
    p["pfscale"] = pfscale
    p["c1"] = c1
    p["bunl"] = bunl
    p["bmin"] = c1
    p["bmax"] = bunl
    p["pmin"] = pmin
    p["pext"] = pext
    p["psh"] = pext
    p["mumax"] = mumax
    p["pstar"] = pstar

    # Populate CFG keys
    p["MAT_RHO"] = rho0
    p["MAT_E"] = e
    p["MAT_NU"] = nu
    p["MAT_A0"] = a0
    p["MAT_A1"] = a1
    p["MAT_A2"] = a2
    p["MAT_AMAX"] = amax
    p["FUN_A1"] = ifunc
    p["PFscale"] = pfscale
    p["MAT_BULK"] = c1
    p["MAT_K_UNLOAD"] = bunl
    p["MAT_PC"] = pmin
    p["PEXT"] = pext
    p["MAT_SIG"] = mumax

    return p


def build_law21(rec: Any) -> Material:
    """Card parsing and validation -> Material for LAW21 (/MAT/LAW21, /MAT/DPRAG).

    Cites starter/source/materials/mat/mat021/hm_read_mat21.F.
    """
    if isinstance(rec, Material):
        p = dict(rec.params) if rec.params is not None else {}
        mat_id = rec.id
        title = rec.title
        density = rec.rho0
    elif isinstance(rec, dict):
        p = dict(rec.get("params", rec))
        mat_id = rec.get("id", 1)
        title = rec.get("title", "LAW21")
        density = None
        for k in ("density", "rho0", "rho", "MAT_RHO", "Refer_Rho"):
            if k in rec and rec[k] is not None:
                density = float(rec[k])
                break
        if density is None:
            for k in ("density", "rho0", "rho", "MAT_RHO", "Refer_Rho"):
                if k in p and p[k] is not None:
                    density = float(p[k])
                    break
        if density is None:
            density = 0.0
    else:
        p = dict(getattr(rec, "params", {}))
        mat_id = getattr(rec, "id", 1)
        title = getattr(rec, "title", "LAW21")
        density = float(getattr(rec, "rho0", getattr(rec, "density", 0.0)))

    params = dict(p)
    if density > 0.0:
        params["rho0"] = density
        params["MAT_RHO"] = density

    # Populate and validate
    _ensure_params(params)
    density = params["rho0"]

    mat = Material(id=mat_id, law=21, rho0=density, title=title, params=params)
    return mat


# -----------------------------------------------------------------------------
# Compaction curve interpolation (FINTER)
# -----------------------------------------------------------------------------

def _finter(xy: tuple[np.ndarray, np.ndarray], x: np.ndarray | float) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Piecewise-linear value AND local slope with end-slope linear extrapolation (finter.F)."""
    xs = np.asarray(xy[0], dtype=float)
    ys = np.asarray(xy[1], dtype=float)
    if xs.size == 0:
        raise ValueError("Empty curve points array")
    if xs.size == 1:
        y = np.full_like(x, ys[0], dtype=float)
        der = np.zeros_like(x, dtype=float)
        return (float(y), float(der)) if np.ndim(x) == 0 else (y, der)

    slopes = np.diff(ys) / np.diff(xs)
    is_scalar = np.ndim(x) == 0
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))

    y = np.interp(x_arr, xs, ys)
    idx = np.clip(np.searchsorted(xs, x_arr, side="right") - 1, 0, len(slopes) - 1)
    der = slopes[idx].copy()

    below = x_arr < xs[0]
    above = x_arr > xs[-1]
    if np.any(below):
        y = np.where(below, ys[0] + slopes[0] * (x_arr - xs[0]), y)
        der = np.where(below, slopes[0], der)
    if np.any(above):
        y = np.where(above, ys[-1] + slopes[-1] * (x_arr - xs[-1]), y)
        der = np.where(above, slopes[-1], der)

    if is_scalar:
        return float(y[0]), float(der[0])
    return y, der


def _eval_pressure_curve(
    p: dict[str, Any],
    extra: dict | None,
    mu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate loading pressure P_load(mu) and derivative dP/dmu = K_t.

    Cites m21law.F line 161.
    """
    pfscale = float(p.get("pfscale", 1.0))
    c1 = float(p["c1"])
    ifunc = int(p.get("ifunc", 0))

    curve = None
    if extra is not None:
        curve = extra.get("curve") or extra.get("f_table")
        if curve is None and "functions" in extra and ifunc in extra["functions"]:
            curve = extra["functions"][ifunc]
    if curve is None:
        curve = p.get("curve") or p.get("func_curve") or p.get("f_table")
        if curve is None and "functions" in p and ifunc in p["functions"]:
            curve = p["functions"][ifunc]

    if curve is None:
        # Default linear elastic pressure response
        p_load = c1 * mu
        dpdm = np.full_like(mu, c1, dtype=float)
        return p_load, dpdm

    if hasattr(curve, "eval"):
        y = np.asarray(curve.eval(mu), dtype=float)
        if hasattr(curve, "slope") and hasattr(curve, "x"):
            xs = np.asarray(curve.x, dtype=float)
            slopes = np.asarray(curve.slope, dtype=float)
            idx = np.clip(np.searchsorted(xs, mu, side="right") - 1, 0, len(slopes) - 1)
            der = slopes[idx].copy()
            below = mu < xs[0]
            above = mu > xs[-1]
            if np.any(below):
                der = np.where(below, slopes[0], der)
            if np.any(above):
                der = np.where(above, slopes[-1], der)
        else:
            h = 1e-6
            der = (np.asarray(curve.eval(mu + h), dtype=float) - np.asarray(curve.eval(mu - h), dtype=float)) / (2.0 * h)
    elif callable(curve):
        y = np.asarray(curve(mu), dtype=float)
        h = 1e-6
        der = (np.asarray(curve(mu + h), dtype=float) - np.asarray(curve(mu - h), dtype=float)) / (2.0 * h)
    elif isinstance(curve, (tuple, list)) and len(curve) >= 2:
        y, der = _finter(curve, mu)
    else:
        p_load = c1 * mu
        dpdm = np.full_like(mu, c1, dtype=float)
        return p_load, dpdm

    p_load = pfscale * y
    dpdm = pfscale * der
    return p_load, dpdm


# -----------------------------------------------------------------------------
# Sound speed calculation
# -----------------------------------------------------------------------------

def sound_speed_solid_law21(
    mat: Material | dict,
    rho: float | np.ndarray | None = None,
    extra: dict | None = None,
    **kwargs,
) -> float | np.ndarray:
    """Longitudinal acoustic wave speed for LAW21 solids.

    Cites m21law.F lines 178-182:
    K_eff = max(bulk, dpdm)
    c = sqrt(|4/3 * G + K_eff| / rho0).
    """
    p = _ensure_params(mat)
    g = float(p["G"])
    c1 = float(p["c1"])
    bunl = float(p["bunl"])
    mumax = float(p["mumax"])
    rho0 = float(p["rho0"])
    g43 = (4.0 / 3.0) * g

    if extra is not None:
        mu = extra.get("mu")
        if mu is not None:
            mu_arr = np.atleast_1d(np.asarray(mu, dtype=float))
            mu_bak = np.asarray(extra.get("mu_bak", mu_arr), dtype=float)
            if mumax > 0.0:
                alpha = np.clip(mu_bak / mumax, 0.0, 1.0)
            else:
                alpha = np.ones_like(mu_bak)
            bulk = alpha * bunl + (1.0 - alpha) * c1
            _, dpdm = _eval_pressure_curve(p, extra, mu_arr)
            kt_eff = np.maximum(bulk, dpdm)
        else:
            kt_eff = max(c1, bunl)
    else:
        kt_eff = max(c1, bunl)

    current_rho = rho0 if rho is None else rho
    c_sq = np.abs(g43 + kt_eff) / current_rho
    c = np.sqrt(c_sq)
    if isinstance(c, np.ndarray):
        if c.ndim == 0 or (c.size == 1 and (rho is None or np.ndim(rho) == 0)):
            return float(c.flat[0])
    return c


sound_speed_solid = sound_speed_solid_law21
sound_speed = sound_speed_solid_law21


# -----------------------------------------------------------------------------
# Constitutive stress update (solid only)
# -----------------------------------------------------------------------------

def solid_update_law21(
    mat: Material | dict,
    sig: np.ndarray,
    eps_dot: np.ndarray | None = None,
    dt: float = 0.0,
    *args,
    d_eps: np.ndarray | None = None,
    deps: np.ndarray | None = None,
    epsp: np.ndarray | None = None,
    extra: dict | None = None,
    return_tuple: bool = False,
    **kwargs,
) -> np.ndarray | tuple[np.ndarray, np.ndarray | None, np.ndarray | float | None]:
    """Vectorized 3D solid stress update for LAW21 (Drucker-Prager yield surface + Compaction EOS).

    Ports:
    - ``engine/source/materials/mat/mat021/m21law.F``
    - ``starter/source/materials/mat/mat021/hm_read_mat21.F``

    Parameters
    ----------
    mat : Material or dict
        Material definition or parameters.
    sig : (6,) or (n, 6) ndarray
        Old Cauchy stress tensor [xx, yy, zz, xy, yz, zx] in Voigt notation.
    eps_dot : (n, 6) ndarray, optional
        Strain rate tensor or strain increment if called positionally.
    dt : float, optional
        Time step increment.
    *args :
        Positional fallbacks (deps, epsp, dt, extra).
    d_eps, deps : (n, 6) ndarray, optional
        Engineering strain increment tensor.
    epsp : (n,) ndarray, optional
        Equivalent plastic strain history.
    extra : dict, optional
        State variables (e.g. 'mu', 'mu_bak', 'epxe', 'off', 'rho', 'curve').
    return_tuple : bool, default False
        If True, returns (sig_new, epsp, sound_speed).

    Returns
    -------
    sig_new : (6,) or (n, 6) ndarray (or tuple if return_tuple=True)
        Updated Cauchy stress tensor.
    """
    # 0. Handle argument permutations
    if len(args) >= 1:
        if isinstance(dt, (int, float)):
            if extra is None and isinstance(args[0], dict):
                extra = args[0]
        else:
            actual_deps = eps_dot
            epsp = dt
            dt = float(args[0])
            if len(args) >= 2 and isinstance(args[1], dict):
                extra = args[1]
            if deps is None and d_eps is None:
                deps = actual_deps

    if extra is None:
        extra = {}

    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(np.asarray(sig, dtype=float))
    n = sig_arr.shape[0]

    if n == 0:
        sig_ret = sig.copy()
        if return_tuple:
            return sig_ret, epsp, None
        return sig_ret

    if dt <= 0.0:
        sig_ret = sig.copy()
        if return_tuple:
            c_val = sound_speed_solid_law21(mat, extra=extra)
            return sig_ret, epsp, c_val
        return sig_ret

    # Strain increment tensor
    if d_eps is not None:
        d_e = np.atleast_2d(np.asarray(d_eps, dtype=float))
    elif deps is not None:
        d_e = np.atleast_2d(np.asarray(deps, dtype=float))
    elif eps_dot is not None:
        d_e = np.atleast_2d(np.asarray(eps_dot, dtype=float)) * dt
    else:
        d_e = np.zeros_like(sig_arr, dtype=float)

    # Material parameters
    p = _ensure_params(mat)
    g = float(p["G"])
    a0 = float(p["A0"])
    a1 = float(p["A1"])
    a2 = float(p["A2"])
    amax = float(p["Amax"])
    c1 = float(p["c1"])
    bunl = float(p["bunl"])
    pmin = float(p["pmin"])
    pext = float(p["pext"])
    mumax = float(p["mumax"])
    pstar = float(p["pstar"])
    rho0 = float(p["rho0"])

    # State variables setup
    if "mu_bak" not in extra or extra["mu_bak"] is None:
        extra["mu_bak"] = np.zeros(n, dtype=float)
    if "epxe" not in extra or extra["epxe"] is None:
        extra["epxe"] = np.zeros(n, dtype=float)

    mu_bak = np.asarray(extra["mu_bak"], dtype=float).copy()
    if mu_bak.shape != (n,):
        mu_bak = np.full(n, float(mu_bak.flat[0]) if mu_bak.size > 0 else 0.0, dtype=float)

    off = np.asarray(extra.get("off", np.ones(n, dtype=float)), dtype=float)
    if off.shape != (n,):
        off = np.full(n, float(off.flat[0]) if off.size > 0 else 1.0, dtype=float)

    # 1. Deviatoric trial stress increment (m21law.F lines 141-155, 206-214)
    p_old = -(sig_arr[:, 0] + sig_arr[:, 1] + sig_arr[:, 2]) / 3.0
    tr_deps = d_e[:, 0] + d_e[:, 1] + d_e[:, 2]

    s_tr = np.empty_like(sig_arr, dtype=float)
    s_tr[:, 0] = sig_arr[:, 0] + p_old + 2.0 * g * (d_e[:, 0] - tr_deps / 3.0)
    s_tr[:, 1] = sig_arr[:, 1] + p_old + 2.0 * g * (d_e[:, 1] - tr_deps / 3.0)
    s_tr[:, 2] = sig_arr[:, 2] + p_old + 2.0 * g * (d_e[:, 2] - tr_deps / 3.0)
    s_tr[:, 3] = sig_arr[:, 3] + g * d_e[:, 3]
    s_tr[:, 4] = sig_arr[:, 4] + g * d_e[:, 4]
    s_tr[:, 5] = sig_arr[:, 5] + g * d_e[:, 5]

    # 2. Volumetric strain mu = rho / rho0 - 1
    if "rho" in extra and extra["rho"] is not None:
        rho_arr = np.asarray(extra["rho"], dtype=float)
        mu = rho_arr / rho0 - 1.0
    elif "mu_total" in extra and extra["mu_total"] is not None:
        mu = np.asarray(extra["mu_total"], dtype=float)
    else:
        mu_prev = np.asarray(extra.get("mu", 0.0), dtype=float)
        if mu_prev.shape != (n,):
            mu_prev = np.full(n, float(mu_prev.flat[0]) if mu_prev.size > 0 else 0.0, dtype=float)
        mu = mu_prev - tr_deps

    if mu.shape != (n,):
        mu = np.full(n, float(mu.flat[0]) if mu.size > 0 else 0.0, dtype=float)
    extra["mu"] = mu

    # 3. Compaction EOS & Unloading (m21law.F lines 160-173)
    p_load, dpdm = _eval_pressure_curve(p, extra, mu)
    p_bak, _ = _eval_pressure_curve(p, extra, mu_bak)

    if mumax > 0.0:
        alpha = np.clip(mu_bak / mumax, 0.0, 1.0)
    else:
        alpha = np.ones(n, dtype=float)
    bulk = alpha * bunl + (1.0 - alpha) * c1

    p_unl = p_bak - (mu_bak - mu) * bulk
    p_eff = np.where(mu_bak > _MUMIN, np.minimum(p_unl, p_load), p_load)
    p_eff = np.maximum(p_eff, pmin) * off
    p_new = p_eff
    p_tot = p_new + pext

    # Historical compaction update (m21law.F line 172)
    mu_bak_new = np.where(mu > mu_bak, np.minimum(mumax, mu), mu_bak)
    extra["mu_bak"] = mu_bak_new

    # 4. Sound speed (m21law.F lines 178-182)
    kt_eff = np.maximum(bulk, dpdm)
    g43 = (4.0 / 3.0) * g
    c_solid = np.sqrt(np.abs(g43 + kt_eff) / rho0)

    # 5. Second invariant J2 of trial deviatoric stress
    j2 = (
        0.5 * (s_tr[:, 0] ** 2 + s_tr[:, 1] ** 2 + s_tr[:, 2] ** 2)
        + s_tr[:, 3] ** 2
        + s_tr[:, 4] ** 2
        + s_tr[:, 5] ** 2
    )

    # 6. Drucker-Prager yield envelope G0 (m21law.F lines 195-201, 221-227)
    g0 = a0 + a1 * p_tot + a2 * (p_tot ** 2)
    g0 = np.clip(g0, 0.0, amax)
    g0 = np.where(p_new <= pmin, 0.0, g0)
    g0 = np.where(p_tot <= pstar, 0.0, g0)

    # 7. Projection factor ratio (m21law.F lines 232-239)
    yield2 = j2 - g0
    ratio = np.where((yield2 <= 0.0) & (g0 > 0.0), 1.0, np.sqrt(g0 / (j2 + _EM14)))
    ratio = np.where(g0 <= 0.0, 0.0, ratio)

    # 8. Deviatoric and total Cauchy stress (m21law.F lines 245-252)
    s_new = ratio[:, None] * s_tr * off[:, None]

    sig_new = np.empty_like(sig_arr, dtype=float)
    sig_new[:, 0] = s_new[:, 0] - p_new
    sig_new[:, 1] = s_new[:, 1] - p_new
    sig_new[:, 2] = s_new[:, 2] - p_new
    sig_new[:, 3] = s_new[:, 3]
    sig_new[:, 4] = s_new[:, 4]
    sig_new[:, 5] = s_new[:, 5]

    # 9. Plastic strain increment (m21law.F line 253)
    # dpla = (1 - ratio) * sqrt(j2) / (3 * g)
    denom = max(_EM20, 3.0 * g)
    dpla = (1.0 - ratio) * np.sqrt(np.maximum(0.0, j2)) / denom

    epxe_cur = np.asarray(extra.get("epxe", 0.0), dtype=float)
    if epxe_cur.shape != (n,):
        epxe_cur = np.full(n, float(epxe_cur.flat[0]) if epxe_cur.size > 0 else 0.0, dtype=float)
    epxe_new = epxe_cur + dpla

    # Store state variables in extra
    extra["epxe"] = epxe_new
    extra["epsq"] = mu_bak_new
    extra["sigy"] = g0
    extra["dpla"] = dpla
    extra["ratio"] = ratio
    extra["j2"] = j2
    extra["g0"] = g0
    extra["p_new"] = p_new
    extra["ptot"] = p_tot
    extra["s_new"] = s_new
    extra["p_old"] = p_new
    extra["bulk"] = bulk
    extra["dpdm"] = dpdm
    extra["kt"] = kt_eff
    extra["c_solid"] = c_solid
    extra["sound_speed"] = c_solid

    if epsp is not None and hasattr(epsp, "__setitem__"):
        try:
            epsp[:] = epxe_new
        except Exception:
            pass

    out_sig = sig_new[0] if is_1d else sig_new
    out_epsp = epxe_new[0] if is_1d else epxe_new
    out_c = float(c_solid[0]) if is_1d else c_solid

    if return_tuple:
        return out_sig, out_epsp, out_c

    return out_sig


solid_update = solid_update_law21


# -----------------------------------------------------------------------------
# Plane-stress Shell Update (Not Supported)
# -----------------------------------------------------------------------------

def shell_update_law21(*args: Any, **kwargs: Any) -> Any:
    """Plane-stress shell update is not supported for LAW21."""
    raise NotImplementedError("LAW21 (Drucker-Prager) is implemented for 3D solid elements only.")


shell_update = shell_update_law21


# -----------------------------------------------------------------------------
# Algorithmic Consistent Tangent Stiffness Tensor
# -----------------------------------------------------------------------------

def tangent_law21_solid(
    mat: Material | dict,
    sig: np.ndarray,
    eps_dot: np.ndarray | None = None,
    dt: float = 0.0,
    *args,
    epsp: np.ndarray | None = None,
    epsp_incr: np.ndarray | None = None,
    extra: dict | None = None,
    symmetric: bool = False,
    **kwargs,
) -> np.ndarray:
    """Algorithmic consistent elastoplastic tangent stiffness matrix for LAW21 in Voigt notation.

    Voigt convention: [xx, yy, zz, xy, yz, zx] with engineering shear.

    Parameters
    ----------
    mat : Material or dict
        Material definition.
    sig : (6,) or (n, 6) ndarray
        Stress state.
    eps_dot, dt :
        Optional rate and time step.
    epsp, epsp_incr :
        Optional plastic strain history.
    extra : dict, optional
        Extra state views (e.g. 'mu', 'mu_bak', 'off', 'curve').
    symmetric : bool, default False
        If True, returns symmetrized matrix 0.5 * (D + D^T).

    Returns
    -------
    D : (6, 6) or (n, 6, 6) ndarray
        Consistent tangent stiffness tensor.
    """
    if extra is None:
        for a in args:
            if isinstance(a, dict):
                extra = a
                break

    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(np.asarray(sig, dtype=float))
    n = sig_arr.shape[0]

    if n == 0:
        return np.empty((0, 6, 6), dtype=sig.dtype)

    p = _ensure_params(mat)
    g = float(p["G"])
    c1 = float(p["c1"])
    bunl = float(p["bunl"])
    a0 = float(p["A0"])
    a1 = float(p["A1"])
    a2 = float(p["A2"])
    amax = float(p["Amax"])
    pmin = float(p["pmin"])
    pext = float(p["pext"])
    mumax = float(p["mumax"])
    pstar = float(p["pstar"])

    ee = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
    c_dev = np.zeros((6, 6), dtype=float)
    c_dev[0, 0] = c_dev[1, 1] = c_dev[2, 2] = (4.0 / 3.0) * g
    c_dev[0, 1] = c_dev[0, 2] = c_dev[1, 0] = c_dev[1, 2] = c_dev[2, 0] = c_dev[2, 1] = -(2.0 / 3.0) * g
    c_dev[3, 3] = c_dev[4, 4] = c_dev[5, 5] = g

    d_tangent = np.zeros((n, 6, 6), dtype=float)

    off = np.asarray(extra.get("off", np.ones(n, dtype=float)), dtype=float) if extra else np.ones(n, dtype=float)
    if off.shape != (n,):
        off = np.full(n, float(off.flat[0]) if off.size > 0 else 1.0, dtype=float)

    p_cur = -(sig_arr[:, 0] + sig_arr[:, 1] + sig_arr[:, 2]) / 3.0
    p_tot = p_cur + pext
    s_cur = sig_arr.copy()
    s_cur[:, 0] += p_cur
    s_cur[:, 1] += p_cur
    s_cur[:, 2] += p_cur

    mu = extra.get("mu") if extra is not None else None
    mu_bak = extra.get("mu_bak") if extra is not None else None

    for i in range(n):
        if off[i] <= 0.0 or p_cur[i] <= pmin:
            # Inactive element or tensile fracture pressure cutoff
            d_tangent[i] = np.zeros((6, 6), dtype=float)
            continue

        # 1. Tangent bulk modulus K_t = dP/dmu
        if mu is not None:
            mu_arr = np.asarray(mu, dtype=float)
            mu_i = float(mu_arr.flat[i if i < mu_arr.size else 0])
            if mu_bak is not None:
                mu_bak_arr = np.asarray(mu_bak, dtype=float)
                mu_bak_i = float(mu_bak_arr.flat[i if i < mu_bak_arr.size else 0])
            else:
                mu_bak_i = 0.0

            alpha_i = min(1.0, max(0.0, mu_bak_i / mumax)) if mumax > 0.0 else 1.0
            bulk_i = alpha_i * bunl + (1.0 - alpha_i) * c1

            p_load_i, dpdm_i = _eval_pressure_curve(p, extra, np.array([mu_i]))
            p_bak_i, _ = _eval_pressure_curve(p, extra, np.array([mu_bak_i]))
            p_unl_i = p_bak_i[0] - (mu_bak_i - mu_i) * bulk_i

            if mu_bak_i > _MUMIN and p_unl_i < p_load_i[0]:
                k_t = bulk_i
            else:
                k_t = float(dpdm_i[0])
        else:
            k_t = max(c1, bunl)

        keet = k_t * np.outer(ee, ee)
        c_elastic = keet + c_dev

        # 2. Yield surface and ratio evaluation
        s_i = s_cur[i]
        j2_i = (
            0.5 * (s_i[0] ** 2 + s_i[1] ** 2 + s_i[2] ** 2)
            + s_i[3] ** 2
            + s_i[4] ** 2
            + s_i[5] ** 2
        )
        ptot_i = p_tot[i]

        g0_uncapped = a0 + a1 * ptot_i + a2 * (ptot_i ** 2)
        g0_val = min(max(0.0, g0_uncapped), amax)
        if ptot_i <= pstar:
            g0_val = 0.0

        if extra is not None and "ratio" in extra:
            ratio_arr = np.asarray(extra["ratio"], dtype=float)
            r_i = float(ratio_arr.flat[i if i < ratio_arr.size else 0])
        else:
            if j2_i <= g0_val and g0_val > 0.0:
                r_i = 1.0
            elif g0_val <= 0.0:
                r_i = 0.0
            else:
                r_i = math.sqrt(g0_val / (j2_i + _EM14))

        # 3. Tangent stiffness tensor assembly
        if r_i >= 1.0:
            # Purely elastic response
            d_tangent[i] = c_elastic
        elif r_i <= 0.0:
            # Yield surface collapsed to a point (deviatoric stiffness vanishes)
            d_tangent[i] = keet
        else:
            # Elastoplastic radial return tangent
            norm_s = math.sqrt(max(j2_i, _EM20))
            s_hat = s_i / norm_s
            nn = np.outer(s_hat, s_hat)

            # Pressure coupling derivative dG0/dP_tot
            if g0_uncapped >= amax or g0_uncapped <= 0.0:
                dg0_dptot = 0.0
            else:
                dg0_dptot = a1 + 2.0 * a2 * ptot_i

            d_coupling = - (k_t * dg0_dptot / (2.0 * max(g0_val, _EM14))) * np.outer(s_i, ee)
            d_tangent[i] = keet + r_i * c_dev - g * r_i * nn + d_coupling

        if symmetric or kwargs.get("symmetric", False):
            d_tangent[i] = 0.5 * (d_tangent[i] + d_tangent[i].T)

    if is_1d:
        return d_tangent[0]
    return d_tangent


consistent_solid_tangent = tangent_law21_solid
solid_tangent = tangent_law21_solid


# -----------------------------------------------------------------------------
# Physics Registry
# -----------------------------------------------------------------------------

def _register() -> None:
    """Register LAW21 constructors in pyradioss MAT_PHYSICS_REGISTRY."""
    try:
        from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY

        MAT_PHYSICS_REGISTRY["LAW21"] = build_law21
        MAT_PHYSICS_REGISTRY["DPRAG"] = build_law21
        MAT_PHYSICS_REGISTRY["21"] = build_law21
        MAT_PHYSICS_REGISTRY[21] = build_law21
    except Exception:
        pass


_register()
