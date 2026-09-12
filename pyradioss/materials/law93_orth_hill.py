"""
LAW93 — Orthotropic Hill 1948 Elasto-Plastic Material (/MAT/LAW93, /MAT/ORTH_HILL).

Implements the Hill 1948 anisotropic yield criterion with Voce exponential
continuous hardening and optional multi-rate tabulated plasticity curves
for 3D continuum solids and 2D shells/membranes.

Fortran reference sources:
- starter/source/materials/mat/mat093/hm_read_mat93.F   (starter card reader & stability)
- engine/source/materials/mat/mat093/sigeps93.F         (3D continuum solid kernel)
- engine/source/materials/mat/mat093/sigeps93c.F        (2D shell/membrane kernel)
- hm_cfg_files/config/CFG/radioss2021/MAT/matl93_ORTH_HILL.cfg (card layout & formats)

Constitutive Formulation
------------------------
1. Hill 1948 Quadratic Anisotropic Yield Surface (3D):
   sigma_HL = sqrt(F*(s_yy - s_zz)^2 + G*(s_zz - s_xx)^2 + H*(s_xx - s_yy)^2
                   + 2*L*s_yz^2 + 2*M*s_zx^2 + 2*N*s_xy^2)

   where the Hill constants are computed from directional normalized yield stress ratios:
   A_1 = 1 / R_11^2,  A_2 = 1 / R_22^2,  A_3 = 1 / R_33^2
   F = 0.5 * (A_2 + A_3 - A_1)
   G = 0.5 * (A_3 + A_1 - A_2)
   H = 0.5 * (A_1 + A_2 - A_3)
   L = 1.5 / R_23^2
   M = 1.5 / R_13^2
   N = 1.5 / R_12^2

   In the isotropic limit (R_11 = R_22 = R_33 = R_12 = R_23 = R_13 = 1):
   A_i = 1 -> F = G = H = 0.5, L = M = N = 1.5
   sigma_HL reduces identically to the standard von Mises equivalent stress.

2. Plane Stress Yield Surface (2D Shells):
   sigma_HL = sqrt((F + H)*s_yy^2 + (G + H)*s_xx^2 - 2*H*s_xx*s_yy + 2*N*s_xy^2)

3. Hardening Law:
   Continuous Voce combination:
   sigma_y(p) = sigma_y0 + Q_R1 * (1 - exp(-C_R1 * p)) + Q_R2 * (1 - exp(-C_R2 * p))
   H(p) = d(sigma_y)/dp = Q_R1 * C_R1 * exp(-C_R1 * p) + Q_R2 * C_R2 * exp(-C_R2 * p)

4. Return Mapping:
   Cutting-plane Newton-Raphson scheme with backward Euler consistency
   (matching sigeps93.F and sigeps93c.F).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_EM10 = 1.0e-10
_EM20 = 1.0e-20
_INF = 1.0e30


@dataclass
class OrthHillParams:
    """Parameters for /MAT/LAW93 (/MAT/ORTH_HILL).

    Attributes
    ----------
    id : int
        Material ID.
    title : str
        Material title.
    rho0 : float
        Initial density.
    rhor : float
        Reference density.
    e11, e22, e33 : float
        Directional Young's moduli.
    g12, g13, g23 : float
        Directional shear moduli.
    nu12, nu13, nu23 : float
        Directional Poisson's ratios.
    nu21, nu31, nu32 : float
        Symmetric directional Poisson's ratios.
    nl : int
        Number of strain-rate plasticity curves.
    fcut : float
        Strain-rate filtering cut frequency.
    vp : int
        Strain-rate computation flag (1: plastic, 2: total, 3: deviatoric).
    curves : list of dict
        Tabulated plasticity curves [{fct_id, fscale, eps_dot}, ...].
    sigma_y : float
        Initial yield stress.
    qr1, cr1, qr2, cr2 : float
        Voce 2-term exponential hardening parameters.
    r11, r22, r12, r33, r13, r23 : float
        Hill anisotropic yield stress ratios.
    a1, a2, a3 : float
        Hill squared-inverse ratios (1/R_ii^2).
    ff, gg, hh, ll, mm, nn : float
        Hill 1948 quadratic yield surface coefficients.
    a11, a22, a12 : float
        2D plane-stress orthotropic elasticity stiffness coefficients.
    d11, d12, d13, d22, d23, d33 : float
        3D orthotropic elasticity stiffness coefficients.
    c_solid : float
        Acoustic wave speed for continuum solids: sqrt(max(D11, D22, D33) / rho0).
    c_shell : float
        Acoustic wave speed for shells/membranes: sqrt(max(A11, A22) / rho0).
    e : float
        Effective Young's modulus: max(E11, E22, E33).
    nu : float
        Effective Poisson's ratio: max(Nu12, Nu13, Nu23).
    g : float
        Effective shear modulus: G12.
    """

    id: int = 1
    title: str = ""
    rho0: float = 1.0
    rhor: float = 0.0

    # Orthotropic elastic moduli
    e11: float = 210000.0
    e22: float = 210000.0
    e33: float = 210000.0
    g12: float = 80000.0
    g13: float = 80000.0
    g23: float = 80000.0
    nu12: float = 0.3
    nu13: float = 0.3
    nu23: float = 0.3
    nu21: float = 0.3
    nu31: float = 0.3
    nu32: float = 0.3

    # Hardening & rate options
    nl: int = 0
    fcut: float = 0.0
    vp: int = 0
    curves: List[Dict[str, Any]] = field(default_factory=list)

    # Voce continuous hardening
    sigma_y: float = _INF
    qr1: float = 0.0
    cr1: float = 0.0
    qr2: float = 0.0
    cr2: float = 0.0

    # Hill 1948 anisotropic yield stress ratios
    r11: float = 1.0
    r22: float = 1.0
    r33: float = 1.0
    r12: float = 1.0
    r13: float = 1.0
    r23: float = 1.0

    # Derived Hill coefficients
    a1: float = 1.0
    a2: float = 1.0
    a3: float = 1.0
    ff: float = 0.5
    gg: float = 0.5
    hh: float = 0.5
    ll: float = 1.5
    mm: float = 1.5
    nn: float = 1.5

    # 2D plane stress elasticity matrix
    a11: float = 0.0
    a22: float = 0.0
    a12: float = 0.0

    # 3D continuum elasticity matrix
    d11: float = 0.0
    d12: float = 0.0
    d13: float = 0.0
    d22: float = 0.0
    d23: float = 0.0
    d33: float = 0.0

    # Derived speeds & scalars
    c_solid: float = 0.0
    c_shell: float = 0.0
    e: float = 210000.0
    nu: float = 0.3
    g: float = 80000.0

    @property
    def young(self) -> float:
        return self.e

    @property
    def poisson(self) -> float:
        return self.nu

    @property
    def shear(self) -> float:
        return self.g

    @property
    def sound_speed(self) -> float:
        return self.c_solid

    def sound_speed_solid(self) -> float:
        return self.c_solid

    def sound_speed_shell(self) -> float:
        return self.c_shell


# ============================================================================
# Curve Evaluation Helpers
# ============================================================================

def _eval_plasticity_curve(curve: Any, eps: float) -> Tuple[float, float]:
    """Evaluate curve for plastic strain returning (yield_stress, slope)."""
    if curve is None:
        return 0.0, 0.0
    if hasattr(curve, "eval"):
        y = float(curve.eval(eps))
        slope = getattr(curve, "slope", 0.0)
        return y, float(slope)
    if hasattr(curve, "x") and hasattr(curve, "y"):
        xs = np.asarray(curve.x, dtype=float)
        ys = np.asarray(curve.y, dtype=float)
    elif hasattr(curve, "data"):
        data = np.asarray(curve.data, dtype=float)
        if data.ndim == 2 and data.shape[1] >= 2:
            xs, ys = data[:, 0], data[:, 1]
        else:
            return 0.0, 0.0
    elif isinstance(curve, (list, tuple)) and len(curve) >= 2:
        xs = np.asarray(curve[0], dtype=float)
        ys = np.asarray(curve[1], dtype=float)
    else:
        return 0.0, 0.0

    if len(xs) == 0:
        return 0.0, 0.0
    if len(xs) == 1:
        return float(ys[0]), 0.0

    if eps <= xs[0]:
        slope = (ys[1] - ys[0]) / max(xs[1] - xs[0], _EM10)
        return float(ys[0] + slope * (eps - xs[0])), float(slope)
    if eps >= xs[-1]:
        slope = (ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], _EM10)
        return float(ys[-1] + slope * (eps - xs[-1])), float(slope)

    idx = int(np.searchsorted(xs, eps))
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    slope = (y1 - y0) / max(x1 - x0, _EM10)
    return float(y0 + slope * (eps - x0)), float(slope)


def eval_yield_stress(
    p: OrthHillParams,
    pla: float | np.ndarray,
    rate: float | np.ndarray = 0.0,
) -> Tuple[float | np.ndarray, float | np.ndarray]:
    """Evaluate current yield stress and hardening slope H = d(sigma_y)/dp.

    Returns (sigma_y, H).
    """
    is_scalar = np.isscalar(pla)
    p_arr = np.asarray(pla, dtype=float)
    r_arr = np.asarray(rate, dtype=float) if not np.isscalar(rate) else np.full_like(p_arr, float(rate))

    # If no tabulated curves, continuous Voce hardening
    if p.nl == 0 or len(p.curves) == 0:
        # sigma_y(p) = sigy + QR1*(1 - exp(-CR1*p)) + QR2*(1 - exp(-CR2*p))
        # H(p) = QR1*CR1*exp(-CR1*p) + QR2*CR2*exp(-CR2*p)
        exp1 = np.exp(-p.cr1 * np.maximum(p_arr, 0.0))
        exp2 = np.exp(-p.cr2 * np.maximum(p_arr, 0.0))
        y_val = p.sigma_y + p.qr1 * (1.0 - exp1) + p.qr2 * (1.0 - exp2)
        h_val = p.qr1 * p.cr1 * exp1 + p.qr2 * p.cr2 * exp2
        if is_scalar:
            return float(y_val.item()), float(h_val.item())
        return y_val, h_val

    # Tabulated hardening
    ncurves = len(p.curves)
    if ncurves == 1:
        c_entry = p.curves[0]
        fscale = float(c_entry.get("fscale", 1.0))
        curve_obj = c_entry.get("curve", None)
        y_out = np.zeros_like(p_arr)
        h_out = np.zeros_like(p_arr)
        for i, ep in enumerate(p_arr.flat):
            val, slp = _eval_plasticity_curve(curve_obj, float(ep))
            y_out.flat[i] = fscale * val
            h_out.flat[i] = fscale * slp
        y_out = np.maximum(y_out, _EM20)
        if is_scalar:
            return float(y_out.item()), float(h_out.item())
        return y_out, h_out

    # Multi-rate interpolation matching sigeps93.F:206-238
    rates = np.array([float(c.get("eps_dot", 0.0)) for c in p.curves], dtype=float)
    fscales = np.array([float(c.get("fscale", 1.0)) for c in p.curves], dtype=float)
    curve_objs = [c.get("curve", None) for c in p.curves]

    y_out = np.zeros_like(p_arr)
    h_out = np.zeros_like(p_arr)

    for i, (ep, er) in enumerate(zip(p_arr.flat, r_arr.flat)):
        # Find bracketing rate curves
        j1 = 0
        for j in range(1, ncurves - 1):
            if er >= rates[j]:
                j1 = j
        j2 = j1 + 1

        val1, slp1 = _eval_plasticity_curve(curve_objs[j1], float(ep))
        val2, slp2 = _eval_plasticity_curve(curve_objs[j2], float(ep))
        y1 = fscales[j1] * val1
        y2 = fscales[j2] * val2
        s1 = fscales[j1] * slp1
        s2 = fscales[j2] * slp2

        dr = rates[j2] - rates[j1]
        fac = (er - rates[j1]) / dr if dr > _EM10 else 0.0
        fac = max(0.0, min(1.0, fac))

        y_out.flat[i] = max(y1 + fac * (y2 - y1), _EM20)
        h_out.flat[i] = s1 + fac * (s2 - s1)

    if is_scalar:
        return float(y_out.item()), float(h_out.item())
    return y_out, h_out


# ============================================================================
# Parameter Builder: build_law93
# ============================================================================

def build_law93(mat: Any = None, **kwargs: Any) -> OrthHillParams:
    """Build and validate an OrthHillParams instance from a Material object or dict.

    Follows starter/source/materials/mat/mat093/hm_read_mat93.F.
    """
    if isinstance(mat, OrthHillParams) and not kwargs:
        return mat

    if mat is None:
        params = dict(kwargs)
    else:
        params = getattr(mat, "params", {}) or {}
        if not isinstance(params, dict):
            params = {}
        else:
            params = dict(params)
        params.update(kwargs)

    def _get(key: str, default: Any = 0.0) -> Any:
        if key in params:
            return params[key]
        if key.lower() in params:
            return params[key.lower()]
        if key.upper() in params:
            return params[key.upper()]
        if mat is not None:
            for attr in (key.lower(), key.upper(), key):
                if hasattr(mat, attr):
                    val = getattr(mat, attr)
                    if val is not None:
                        return val
        return default

    def _f(key: str, default: float = 0.0) -> float:
        val = _get(key, default)
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _i(key: str, default: int = 0) -> int:
        val = _get(key, default)
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    mid = _i("id", getattr(mat, "id", 1))
    title = str(_get("title", getattr(mat, "title", "")))
    rho0 = _f("rho0", _f("rho", _f("MAT_RHO", 1.0)))
    rhor = _f("rhor", _f("ref_rho", 0.0))

    # Young moduli (defaults: E22 = E11, E33 = E22)
    e11 = _f("e11", _f("E11", 210000.0))
    e22 = _f("e22", _f("E22", 0.0))
    if e22 <= 0.0:
        e22 = e11
    e33 = _f("e33", _f("E33", 0.0))
    if e33 <= 0.0:
        e33 = e22

    # Shear moduli (defaults: G13 = G12, G23 = G12)
    g12 = _f("g12", _f("G12", 80000.0))
    g13 = _f("g13", _f("G13", 0.0))
    if g13 <= 0.0:
        g13 = g12
    g23 = _f("g23", _f("G23", 0.0))
    if g23 <= 0.0:
        g23 = g12

    # Poisson ratios
    nu12 = _f("nu12", _f("Nu12", 0.3))
    nu13 = _f("nu13", _f("Nu13", 0.3))
    nu23 = _f("nu23", _f("Nu23", 0.3))

    # Remaining Poisson ratios: hm_read_mat93.F:192-194
    nu21 = nu12 * e22 / e11
    nu31 = nu13 * e33 / e11
    nu32 = nu23 * e33 / e22

    # Rate parameters
    nl = _i("nl", _i("LAW93_NL", 0))
    fcut = _f("fcut", _f("FCUT", 0.0))
    vp = _i("vp", _i("VP", 0))
    curves = _get("curves", getattr(mat, "curves", [])) or []

    # Voce continuous hardening
    sigma_y = _f("sigma_y", _f("SIGY", _f("LAW93_Sigma_y", 0.0)))
    if sigma_y <= 0.0:
        sigma_y = _INF
    qr1 = _f("qr1", _f("QR1", _f("LAW93_QR1", 0.0)))
    cr1 = _f("cr1", _f("CR1", _f("LAW93_CR1", 0.0)))
    qr2 = _f("qr2", _f("QR2", _f("LAW93_QR2", 0.0)))
    cr2 = _f("cr2", _f("CR2", _f("LAW93_CR2", 0.0)))

    # Hill yield stress ratios (defaults: 1.0)
    r11 = _f("r11", _f("R11", _f("LAW93_R11", 1.0)))
    if r11 <= 0.0:
        r11 = 1.0
    r22 = _f("r22", _f("R22", _f("LAW93_R22", 1.0)))
    if r22 <= 0.0:
        r22 = 1.0
    r33 = _f("r33", _f("R33", _f("LAW93_R33", 1.0)))
    if r33 <= 0.0:
        r33 = 1.0
    r12 = _f("r12", _f("R12", _f("LAW93_R12", 1.0)))
    if r12 <= 0.0:
        r12 = 1.0
    r13 = _f("r13", _f("R13", _f("LAW93_R13", 1.0)))
    if r13 <= 0.0:
        r13 = 1.0
    r23 = _f("r23", _f("R23", _f("LAW93_R23", 1.0)))
    if r23 <= 0.0:
        r23 = 1.0

    # Hill coefficients: hm_read_mat93.F:218-226
    a1 = 1.0 / (r11 * r11)
    a2 = 1.0 / (r22 * r22)
    a3 = 1.0 / (r33 * r33)
    ff = 0.5 * (a2 + a3 - a1)
    gg = 0.5 * (a3 + a1 - a2)
    hh = 0.5 * (a1 + a2 - a3)
    ll = 1.5 / (r23 * r23)
    mm = 1.5 / (r13 * r13)
    nn = 1.5 / (r12 * r12)

    # 2D plane stress elasticity matrix: hm_read_mat93.F:229-232
    fac = 1.0 / max(1.0 - nu12 * nu21, 1.0e-12)
    a11 = e11 * fac
    a12 = nu21 * a11
    a22 = e22 * fac

    # 3D compliance matrix & inverse: hm_read_mat93.F:234-256
    c11 = 1.0 / e11
    c22 = 1.0 / e22
    c33 = 1.0 / e33
    c12 = -nu12 / e11
    c13 = -nu31 / e33
    c23 = -nu23 / e22

    detc = (
        c11 * c22 * c33
        - c11 * c23 * c23
        - c12 * c12 * c33
        + c12 * c13 * c23
        + c13 * c12 * c23
        - c13 * c22 * c13
    )
    if detc <= 0.0:
        detc = 1.0e-15

    d11 = (c22 * c33 - c23 * c23) / detc
    d12 = -(c12 * c33 - c13 * c23) / detc
    d13 = (c12 * c23 - c13 * c22) / detc
    d22 = (c11 * c33 - c13 * c13) / detc
    d23 = -(c11 * c23 - c13 * c12) / detc
    d33 = (c11 * c22 - c12 * c12) / detc

    # Sound speeds: hm_read_mat93.F:280, sigeps93.F:458, sigeps93c.F:449
    rho_eff = max(rho0, 1.0e-12)
    c_solid = math.sqrt(max(d11, d22, d33) / rho_eff)
    c_shell = math.sqrt(max(a11, a22) / rho_eff)

    e_eff = max(e11, e22, e33)
    nu_eff = max(nu12, nu13, nu23)
    g_eff = g12

    return OrthHillParams(
        id=mid,
        title=title,
        rho0=rho0,
        rhor=rhor,
        e11=e11,
        e22=e22,
        e33=e33,
        g12=g12,
        g13=g13,
        g23=g23,
        nu12=nu12,
        nu13=nu13,
        nu23=nu23,
        nu21=nu21,
        nu31=nu31,
        nu32=nu32,
        nl=nl,
        fcut=fcut,
        vp=vp,
        curves=curves,
        sigma_y=sigma_y,
        qr1=qr1,
        cr1=cr1,
        qr2=qr2,
        cr2=cr2,
        r11=r11,
        r22=r22,
        r33=r33,
        r12=r12,
        r13=r13,
        r23=r23,
        a1=a1,
        a2=a2,
        a3=a3,
        ff=ff,
        gg=gg,
        hh=hh,
        ll=ll,
        mm=mm,
        nn=nn,
        a11=a11,
        a22=a22,
        a12=a12,
        d11=d11,
        d12=d12,
        d13=d13,
        d22=d22,
        d23=d23,
        d33=d33,
        c_solid=c_solid,
        c_shell=c_shell,
        e=e_eff,
        nu=nu_eff,
        g=g_eff,
    )


def resolve(mat: Any, model: Any = None, log: Any = None) -> OrthHillParams:
    """Resolve and cache OrthHillParams from a Material or dict."""
    if isinstance(mat, OrthHillParams):
        return mat
    cached = getattr(mat, "_cached_law93", None)
    if cached is None:
        cached = build_law93(mat)
        if model is not None and getattr(model, "functions", None):
            fns = model.functions
            for c in cached.curves:
                fid = getattr(c, "fct_id", 0)
                if fid in fns:
                    c.func = fns[fid]
        try:
            setattr(mat, "_cached_law93", cached)
        except Exception:
            pass
    return cached


def sound_speed(mat: Any, rho: float | np.ndarray | None = None) -> float | np.ndarray:
    """Acoustic sound speed for 3D continuum solids: sqrt(max(D11, D22, D33) / rho)."""
    p = resolve(mat)
    if rho is None or np.all(rho == 0.0):
        return p.c_solid
    return np.sqrt(max(p.d11, p.d22, p.d33) / np.maximum(rho, 1.0e-12))


def sound_speed_shell(mat: Any, rho: float | np.ndarray | None = None) -> float | np.ndarray:
    """Acoustic sound speed for shells/membranes: sqrt(max(A11, A22) / rho)."""
    p = resolve(mat)
    if rho is None or np.all(rho == 0.0):
        return p.c_shell
    return np.sqrt(max(p.a11, p.a22) / np.maximum(rho, 1.0e-12))


def extra_shapes(mat: Any = None, nip: int | None = None) -> Dict[str, Tuple[int, ...]]:
    """Persistent state arrays required by LAW93 in element kernels.

    Stores:
      - uvar: (nip, 1) or (1,) for filtered plastic/deviatoric strain rate if VP active.
      - dpla: (nip,) or (1,) plastic strain increment per cycle.
    """
    if nip is not None and nip > 0:
        return {
            "uvar": (nip, 1),
            "dpla": (nip,),
        }
    return {
        "uvar": (1,),
        "dpla": (1,),
    }


# ============================================================================
# 3D Continuum Solid Kernel: solid_update
# ============================================================================

def solid_update(
    mat: Any = None,
    sig: np.ndarray | None = None,
    deps: np.ndarray | None = None,
    epsp: np.ndarray | None = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    *,
    pla: np.ndarray | None = None,
    sig_old: np.ndarray | None = None,
    rho: Optional[Union[float, np.ndarray]] = None,
    return_tuple: bool = True,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D continuum solid elasto-plastic stress update for LAW93 (sigeps93.F).

    Accepts both (mat, sig, deps, epsp, dt, extra) and (mat, deps, sig_old, pla, rho, extra).
    """
    p = resolve(mat)

    # Disambiguate argument order
    if sig_old is not None:
        sig_in = sig_old
        deps_in = deps if deps is not None else sig
    elif deps is not None and sig is not None:
        if isinstance(dt, dict):
            extra = dt
            rho = epsp
            epsp = None
            deps_in = sig
            sig_in = deps
        else:
            sig_in = sig
            deps_in = deps
    elif sig is not None:
        deps_in = sig
        sig_in = np.zeros_like(sig)
    else:
        raise ValueError("Neither sig nor deps was provided to solid_update.")

    deps_arr = np.asarray(deps_in, dtype=float)
    sig_arr = np.asarray(sig_in, dtype=float)

    is_1d = (deps_arr.ndim == 1)
    if is_1d:
        deps_arr = deps_arr.reshape(1, 6)
        sig_arr = sig_arr.reshape(1, 6)

    nel = deps_arr.shape[0]

    pla_in = pla if pla is not None else (epsp if epsp is not None else np.zeros(nel, dtype=float))
    pla_arr = np.asarray(pla_in, dtype=float)
    if pla_arr.ndim == 0 or (is_1d and pla_arr.ndim == 1 and pla_arr.shape[0] != nel):
        pla_arr = np.full(nel, float(pla_arr.item()))

    if extra is None:
        extra = kwargs.get("extra", None)
    if rho is None and extra is not None and "rho" in extra:
        rho = extra["rho"]

    dt_val = 1.0e-6
    if isinstance(dt, (int, float)) and dt > 0.0:
        dt_val = float(dt)
    elif extra is not None and "dt" in extra:
        dt_val = float(extra["dt"])

    # 1. Trial stresses using 3D orthotropic elasticity (sigeps93.F:248-253)
    # [xx, yy, zz, xy, yz, zx]
    sign = np.empty_like(sig_arr)
    sign[:, 0] = sig_arr[:, 0] + p.d11 * deps_arr[:, 0] + p.d12 * deps_arr[:, 1] + p.d13 * deps_arr[:, 2]
    sign[:, 1] = sig_arr[:, 1] + p.d12 * deps_arr[:, 0] + p.d22 * deps_arr[:, 1] + p.d23 * deps_arr[:, 2]
    sign[:, 2] = sig_arr[:, 2] + p.d13 * deps_arr[:, 0] + p.d23 * deps_arr[:, 1] + p.d33 * deps_arr[:, 2]
    sign[:, 3] = sig_arr[:, 3] + p.g12 * deps_arr[:, 3]
    sign[:, 4] = sig_arr[:, 4] + p.g23 * deps_arr[:, 4]
    sign[:, 5] = sig_arr[:, 5] + p.g13 * deps_arr[:, 5]

    # 2. Hill equivalent stress: sigeps93.F:256-259
    # sig_hl = sqrt(FF*(yy-zz)^2 + GG*(zz-xx)^2 + HH*(xx-yy)^2 + 2*LL*yz^2 + 2*MM*zx^2 + 2*NN*xy^2)
    diff_yz = sign[:, 1] - sign[:, 2]
    diff_zx = sign[:, 2] - sign[:, 0]
    diff_xy = sign[:, 0] - sign[:, 1]
    sig_hl2 = (
        p.ff * (diff_yz ** 2)
        + p.gg * (diff_zx ** 2)
        + p.hh * (diff_xy ** 2)
        + 2.0 * p.ll * (sign[:, 4] ** 2)
        + 2.0 * p.mm * (sign[:, 5] ** 2)
        + 2.0 * p.nn * (sign[:, 3] ** 2)
    )
    sig_hl = np.sqrt(np.maximum(sig_hl2, 0.0))

    # Evaluate yield stress & hardening modulus
    rate_arr = np.zeros(nel, dtype=float)
    if extra is not None and "uvar" in extra:
        uvar = extra["uvar"]
        if hasattr(uvar, "shape") and uvar.size >= nel:
            rate_arr = np.asarray(uvar, dtype=float).reshape(nel, -1)[:, 0]

    yld, h_slope = eval_yield_stress(p, pla_arr, rate_arr)

    # 3. Yield check: phi = sig_hl - yld (sigeps93.F:265)
    phi = sig_hl - yld
    yielding = (phi > 0.0)

    pla_new = pla_arr.copy()
    dpla = np.zeros(nel, dtype=float)

    # 4. Plastic correction with cutting plane algorithm (NITER = 3 iterations)
    # sigeps93.F:281-448
    if np.any(yielding):
        niter = 3
        for _ in range(niter):
            mask = (phi > 0.0)
            if not np.any(mask):
                break

            sxx = sign[mask, 0]
            syy = sign[mask, 1]
            szz = sign[mask, 2]
            sxy = sign[mask, 3]
            syz = sign[mask, 4]
            szx = sign[mask, 5]
            shl = np.maximum(sig_hl[mask], _EM20)
            y_cur = np.maximum(yld[mask], _EM20)
            h_cur = h_slope[mask]
            phi_cur = phi[mask]

            # Normal to yield surface DPHI/DSIG: sigeps93.F:302-307
            norm_xx = (p.gg * (sxx - szz) + p.hh * (sxx - syy)) / shl
            norm_yy = (p.ff * (syy - szz) + p.hh * (syy - sxx)) / shl
            norm_zz = (p.ff * (szz - syy) + p.gg * (szz - sxx)) / shl
            norm_xy = 2.0 * p.nn * sxy / shl
            norm_yz = 2.0 * p.ll * syz / shl
            norm_zx = 2.0 * p.mm * szx / shl

            # DPHI/DLAMBDA: sigeps93.F:314-343
            dfdsig2 = (
                norm_xx * (p.d11 * norm_xx + p.d12 * norm_yy + p.d13 * norm_zz)
                + norm_yy * (p.d12 * norm_xx + p.d22 * norm_yy + p.d23 * norm_zz)
                + norm_zz * (p.d13 * norm_xx + p.d23 * norm_yy + p.d33 * norm_zz)
                + (norm_xy ** 2) * p.g12
                + (norm_yz ** 2) * p.g23
                + (norm_zx ** 2) * p.g13
            )

            sig_dfdsig = (
                sxx * norm_xx
                + syy * norm_yy
                + szz * norm_zz
                + sxy * norm_xy
                + syz * norm_yz
                + szx * norm_zx
            )
            dpla_dlam = sig_dfdsig / y_cur

            dphi_dlam = -dfdsig2 - h_cur * dpla_dlam
            # Guard against zero division
            dphi_dlam_safe = np.where(
                np.abs(dphi_dlam) < _EM20,
                np.sign(dphi_dlam + _EM20) * _EM20,
                dphi_dlam,
            )

            dlam = -phi_cur / dphi_dlam_safe

            # Plastic strains tensor increment: sigeps93.F:349-354
            dpxx = dlam * norm_xx
            dpyy = dlam * norm_yy
            dpzz = dlam * norm_zz
            dpxy = dlam * norm_xy
            dpyz = dlam * norm_yz
            dpzx = dlam * norm_zx

            # Elasto-plastic stresses update: sigeps93.F:357-362
            sign[mask, 0] -= (p.d11 * dpxx + p.d12 * dpyy + p.d13 * dpzz)
            sign[mask, 1] -= (p.d12 * dpxx + p.d22 * dpyy + p.d23 * dpzz)
            sign[mask, 2] -= (p.d13 * dpxx + p.d23 * dpyy + p.d33 * dpzz)
            sign[mask, 3] -= dpxy * p.g12
            sign[mask, 4] -= dpyz * p.g23
            sign[mask, 5] -= dpzx * p.g13

            # Cumulated plastic strain update: sigeps93.F:365-367
            ddep = dlam * dpla_dlam
            dpla[mask] += np.maximum(0.0, ddep)
            pla_new[mask] += ddep

            # Recompute Hill equivalent stress: sigeps93.F:370-373
            diff_yz = sign[mask, 1] - sign[mask, 2]
            diff_zx = sign[mask, 2] - sign[mask, 0]
            diff_xy = sign[mask, 0] - sign[mask, 1]
            shl_new2 = (
                p.ff * (diff_yz ** 2)
                + p.gg * (diff_zx ** 2)
                + p.hh * (diff_xy ** 2)
                + 2.0 * p.ll * (sign[mask, 4] ** 2)
                + 2.0 * p.mm * (sign[mask, 5] ** 2)
                + 2.0 * p.nn * (sign[mask, 3] ** 2)
            )
            sig_hl[mask] = np.sqrt(np.maximum(shl_new2, 0.0))

            # Recompute yield stress
            y_sub, h_sub = eval_yield_stress(p, pla_new[mask], rate_arr[mask])
            yld[mask] = y_sub
            h_slope[mask] = h_sub
            phi[mask] = sig_hl[mask] - yld[mask]

    # Store internal variable updates
    if extra is not None:
        extra["dpla"] = dpla[0] if is_1d else dpla
        extra["pla"] = pla_new[0] if is_1d else pla_new
        if "uvar" in extra and p.nl > 1 and p.vp == 1:
            dpdt = dpla / max(_EM20, dt_val)
            asrate = p.fcut
            uvar_arr = np.asarray(extra["uvar"], dtype=float).reshape(nel, -1)
            uvar_arr[:, 0] = asrate * dpdt + (1.0 - asrate) * uvar_arr[:, 0]

    # Sound speed: sqrt(max(D11, D22, D33) / rho0)
    rho_eff = rho if rho is not None else p.rho0
    if np.isscalar(rho_eff):
        c_sound = np.full(nel, math.sqrt(max(p.d11, p.d22, p.d33) / max(float(rho_eff), 1.0e-12)))
    else:
        c_sound = np.sqrt(max(p.d11, p.d22, p.d33) / np.maximum(rho_eff, 1.0e-12))

    if is_1d:
        return sign[0], pla_new[0], c_sound[0]
    return sign, pla_new, c_sound


# ============================================================================
# 2D Shell Kernel: shell_update
# ============================================================================

def shell_update(
    mat: Any = None,
    sig: np.ndarray | None = None,
    deps: np.ndarray | None = None,
    epsp: np.ndarray | None = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    *,
    pla: np.ndarray | None = None,
    sig_old: np.ndarray | None = None,
    rho: Optional[Union[float, np.ndarray]] = None,
    return_tuple: bool = True,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D shell plane-stress elasto-plastic stress update for LAW93 (sigeps93c.F).

    Accepts both (mat, sig, deps, epsp, dt, extra) and (mat, deps, sig_old, pla, rho, extra).
    """
    p = resolve(mat)

    # Disambiguate argument order
    if sig_old is not None:
        sig_in = sig_old
        deps_in = deps if deps is not None else sig
    elif deps is not None and sig is not None:
        if isinstance(dt, dict):
            extra = dt
            rho = epsp
            epsp = None
            deps_in = sig
            sig_in = deps
        else:
            sig_in = sig
            deps_in = deps
    elif sig is not None:
        deps_in = sig
        sig_in = np.zeros_like(sig)
    else:
        raise ValueError("Neither sig nor deps was provided to shell_update.")

    deps_arr = np.asarray(deps_in, dtype=float)
    sig_arr = np.asarray(sig_in, dtype=float)

    is_1d = (deps_arr.ndim == 1)
    if is_1d:
        deps_arr = deps_arr.reshape(1, -1)
        sig_arr = sig_arr.reshape(1, -1)

    nel = deps_arr.shape[0]
    ncols = deps_arr.shape[1]
    has_transverse = (ncols >= 5)

    pla_in = pla if pla is not None else (epsp if epsp is not None else np.zeros(nel, dtype=float))
    pla_arr = np.asarray(pla_in, dtype=float)
    if pla_arr.ndim == 0 or (is_1d and pla_arr.ndim == 1 and pla_arr.shape[0] != nel):
        pla_arr = np.full(nel, float(pla_arr.item()))

    if extra is None:
        extra = kwargs.get("extra", None)
    if rho is None and extra is not None and "rho" in extra:
        rho = extra["rho"]

    dt_val = 1.0e-6
    if isinstance(dt, (int, float)) and dt > 0.0:
        dt_val = float(dt)
    elif extra is not None and "dt" in extra:
        dt_val = float(extra["dt"])

    # 1. Trial stresses using 2D plane-stress orthotropic elasticity (sigeps93c.F:253-257)
    # [xx, yy, xy]
    sign = np.empty_like(sig_arr)
    sign[:, 0] = sig_arr[:, 0] + p.a11 * deps_arr[:, 0] + p.a12 * deps_arr[:, 1]
    sign[:, 1] = sig_arr[:, 1] + p.a12 * deps_arr[:, 0] + p.a22 * deps_arr[:, 1]
    sign[:, 2] = sig_arr[:, 2] + p.g12 * deps_arr[:, 2]
    if has_transverse:
        # Transverse shear (elastically integrated with shear factor SHF = 5/6)
        shf = 5.0 / 6.0
        sign[:, 3] = sig_old[:, 3] + shf * p.g23 * deps[:, 3]
        sign[:, 4] = sig_old[:, 4] + shf * p.g13 * deps[:, 4]

    # 2. Hill plane-stress equivalent stress: sigeps93c.F:260-262
    # sig_hl = sqrt((FF+HH)*yy^2 + (GG+HH)*xx^2 - 2*HH*xx*yy + 2*NN*xy^2)
    sxx = sign[:, 0]
    syy = sign[:, 1]
    sxy = sign[:, 2]
    sig_hl2 = (
        (p.ff + p.hh) * (syy ** 2)
        + (p.gg + p.hh) * (sxx ** 2)
        - 2.0 * p.hh * sxx * syy
        + 2.0 * p.nn * (sxy ** 2)
    )
    sig_hl = np.sqrt(np.maximum(sig_hl2, 0.0))

    # Evaluate yield stress & hardening slope
    rate_arr = np.zeros(nel, dtype=float)
    if extra is not None and "uvar" in extra:
        uvar = extra["uvar"]
        if hasattr(uvar, "shape") and uvar.size >= nel:
            rate_arr = np.asarray(uvar, dtype=float).reshape(nel, -1)[:, 0]

    yld, h_slope = eval_yield_stress(p, pla_arr, rate_arr)

    # 3. Yield check: phi = sig_hl - yld (sigeps93c.F:269)
    phi = sig_hl - yld
    yielding = (phi > 0.0)

    pla_new = pla_arr.copy()
    dpla = np.zeros(nel, dtype=float)
    dpzz_accum = np.zeros(nel, dtype=float)

    # 4. Plastic correction with cutting plane algorithm (NITER = 3 iterations)
    # sigeps93c.F:284-438
    if np.any(yielding):
        niter = 3
        for _ in range(niter):
            mask = (phi > 0.0)
            if not np.any(mask):
                break

            sxx_m = sign[mask, 0]
            syy_m = sign[mask, 1]
            sxy_m = sign[mask, 2]
            shl_cur = sig_hl[mask]
            y_cur = yld[mask]
            h_cur = h_slope[mask]
            phi_cur = phi[mask]

            # Yield surface gradient: sigeps93c.F:312-320
            # df/dsxx = (GG*sxx + HH*(sxx - syy)) / shl
            # df/dsyy = (FF*syy + HH*(syy - sxx)) / shl
            # df/dsxy = 2 * NN * sxy / shl
            norm_xx = (p.gg * sxx_m + p.hh * (sxx_m - syy_m)) / shl_cur
            norm_yy = (p.ff * syy_m + p.hh * (syy_m - sxx_m)) / shl_cur
            norm_xy = (2.0 * p.nn * sxy_m) / shl_cur

            # Newton denominator dphi/dlam: sigeps93c.F:333-338
            # dfdsig2 = norm_xx*(A11*norm_xx + A12*norm_yy) + norm_yy*(A12*norm_xx + A22*norm_yy) + norm_xy^2 * G12
            dfdsig2 = (
                norm_xx * (p.a11 * norm_xx + p.a12 * norm_yy)
                + norm_yy * (p.a12 * norm_xx + p.a22 * norm_yy)
                + (norm_xy ** 2) * p.g12
            )

            sig_dfdsig = sxx_m * norm_xx + syy_m * norm_yy + sxy_m * norm_xy
            dpla_dlam = sig_dfdsig / y_cur

            dphi_dlam = -dfdsig2 - h_cur * dpla_dlam
            dphi_dlam_safe = np.where(
                np.abs(dphi_dlam) < _EM20,
                np.sign(dphi_dlam + _EM20) * _EM20,
                dphi_dlam,
            )

            dlam = -phi_cur / dphi_dlam_safe

            # Plastic strains increment: sigeps93c.F:344-346
            dpxx = dlam * norm_xx
            dpyy = dlam * norm_yy
            dpxy = dlam * norm_xy

            # Stress update: sigeps93c.F:349-351
            sign[mask, 0] -= (p.a11 * dpxx + p.a12 * dpyy)
            sign[mask, 1] -= (p.a22 * dpyy + p.a12 * dpxx)
            sign[mask, 2] -= dpxy * p.g12

            # Cumulated plastic strain: sigeps93c.F:354-356
            ddep = dlam * dpla_dlam
            dpla[mask] += np.maximum(0.0, ddep)
            pla_new[mask] += ddep

            # Transverse plastic strain: sigeps93c.F:364 (dpzz = dpzz - (dpxx + dpyy))
            dpzz_accum[mask] -= (dpxx + dpyy)

            # Recompute Hill equivalent stress: sigeps93c.F:359-361
            sxx_m = sign[mask, 0]
            syy_m = sign[mask, 1]
            sxy_m = sign[mask, 2]
            shl_new2 = (
                (p.ff + p.hh) * (syy_m ** 2)
                + (p.gg + p.hh) * (sxx_m ** 2)
                - 2.0 * p.hh * sxx_m * syy_m
                + 2.0 * p.nn * (sxy_m ** 2)
            )
            sig_hl[mask] = np.sqrt(np.maximum(shl_new2, 0.0))

            # Recompute yield stress
            y_sub, h_sub = eval_yield_stress(p, pla_new[mask], rate_arr[mask])
            yld[mask] = y_sub
            h_slope[mask] = h_sub
            phi[mask] = sig_hl[mask] - yld[mask]

    # 5. Shell thinning (transverse normal strain dezz & thickness update)
    # sigeps93c.F:460-468
    # deelzz = -(NU13 / E11) * (sign_xx - sigo_xx) - (NU23 / E22) * (sign_yy - sigo_yy)
    deelzz = -(p.nu13 / p.e11) * (sign[:, 0] - sig_arr[:, 0]) - (p.nu23 / p.e22) * (sign[:, 1] - sig_arr[:, 1])
    dezz = deelzz + dpzz_accum

    if extra is not None or "thick" in kwargs or "thk" in kwargs:
        if extra is None:
            extra = {}
        thk_val = extra.get("thick", extra.get("thk", kwargs.get("thick", kwargs.get("thk", 1.0))))
        thk_arr = np.asarray(thk_val, dtype=float)
        thk_new = thk_arr * (1.0 + dezz)
        extra["thick"] = thk_new[0] if is_1d else thk_new
        extra["thk"] = extra["thick"]
        extra["dpla"] = dpla[0] if is_1d else dpla
        extra["pla"] = pla_new[0] if is_1d else pla_new
        if "uvar" in extra and p.nl > 1 and p.vp == 1:
            dpdt = dpla / max(_EM20, dt_val)
            asrate = p.fcut
            uvar_arr = np.asarray(extra["uvar"], dtype=float).reshape(nel, -1)
            uvar_arr[:, 0] = asrate * dpdt + (1.0 - asrate) * uvar_arr[:, 0]

    # Sound speed: sqrt(max(A11, A22) / rho0)
    rho_eff = rho if rho is not None else p.rho0
    if np.isscalar(rho_eff):
        c_sound = np.full(nel, math.sqrt(max(p.a11, p.a22) / max(float(rho_eff), 1.0e-12)))
    else:
        c_sound = np.sqrt(max(p.a11, p.a22) / np.maximum(rho_eff, 1.0e-12))

    if is_1d:
        return sign[0], pla_new[0], c_sound[0]
    return sign, pla_new, c_sound


# ============================================================================
# Algorithmic Consistent Tangent Operators
# ============================================================================

def consistent_tangent(
    mat: Any,
    sig: np.ndarray,
    pla: float = 0.0,
    eps_dot: float = 0.0,
    plane_stress: bool = False,
    is_shell: bool | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Compute the algorithmic consistent tangent matrix for LAW93.

    Parameters
    ----------
    mat : Material or OrthHillParams
    sig : (6,) or (3,) current stress tensor
    pla : float, cumulative equivalent plastic strain
    eps_dot : float, strain rate
    plane_stress : bool, if True returns (3, 3) shell tangent, else (6, 6) solid
    is_shell : bool, optional alias for plane_stress
    **kwargs : dict, additional keyword arguments

    Returns
    -------
    C_ep : (6, 6) or (3, 3) elasto-plastic consistent tangent matrix
    """
    if is_shell is not None:
        plane_stress = is_shell
    p = resolve(mat)
    sig = np.asarray(sig, dtype=float).ravel()

    if plane_stress or sig.shape[0] <= 3:
        # 2D Plane-stress (3 x 3)
        c_el = np.array([
            [p.a11, p.a12, 0.0],
            [p.a12, p.a22, 0.0],
            [0.0, 0.0, p.g12],
        ], dtype=float)

        sxx = sig[0]
        syy = sig[1]
        sxy = sig[2] if len(sig) > 2 else 0.0
        shl2 = (
            (p.ff + p.hh) * (syy ** 2)
            + (p.gg + p.hh) * (sxx ** 2)
            - 2.0 * p.hh * sxx * syy
            + 2.0 * p.nn * (sxy ** 2)
        )
        shl = math.sqrt(max(shl2, 0.0))
        yld, h_slope = eval_yield_stress(p, pla, eps_dot)

        if shl < yld * (1.0 - 1.0e-5) or shl < _EM10:
            return c_el

        # Plastic normal:
        norm = np.array([
            (p.gg * sxx + p.hh * (sxx - syy)) / shl,
            (p.ff * syy + p.hh * (syy - sxx)) / shl,
            2.0 * p.nn * sxy / shl,
        ], dtype=float)

        cn = c_el @ norm
        denom = float(norm @ cn + h_slope)
        if abs(denom) < _EM20:
            return c_el
        return c_el - np.outer(cn, cn) / denom

    # 3D Solid (6 x 6)
    c_el = np.array([
        [p.d11, p.d12, p.d13, 0.0, 0.0, 0.0],
        [p.d12, p.d22, p.d23, 0.0, 0.0, 0.0],
        [p.d13, p.d23, p.d33, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, p.g12, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, p.g23, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, p.g13],
    ], dtype=float)

    sxx, syy, szz = sig[0], sig[1], sig[2]
    sxy = sig[3] if len(sig) > 3 else 0.0
    syz = sig[4] if len(sig) > 4 else 0.0
    szx = sig[5] if len(sig) > 5 else 0.0

    shl2 = (
        p.ff * ((syy - szz) ** 2)
        + p.gg * ((szz - sxx) ** 2)
        + p.hh * ((sxx - syy) ** 2)
        + 2.0 * p.ll * (syz ** 2)
        + 2.0 * p.mm * (szx ** 2)
        + 2.0 * p.nn * (sxy ** 2)
    )
    shl = math.sqrt(max(shl2, 0.0))
    yld, h_slope = eval_yield_stress(p, pla, eps_dot)

    if shl < yld * (1.0 - 1.0e-5) or shl < _EM10:
        return c_el

    norm = np.array([
        (p.gg * (sxx - szz) + p.hh * (sxx - syy)) / shl,
        (p.ff * (syy - szz) + p.hh * (syy - sxx)) / shl,
        (p.ff * (szz - syy) + p.gg * (szz - sxx)) / shl,
        2.0 * p.nn * sxy / shl,
        2.0 * p.ll * syz / shl,
        2.0 * p.mm * szx / shl,
    ], dtype=float)

    cn = c_el @ norm
    denom = float(norm @ cn + h_slope)
    if abs(denom) < _EM20:
        return c_el
    return c_el - np.outer(cn, cn) / denom
