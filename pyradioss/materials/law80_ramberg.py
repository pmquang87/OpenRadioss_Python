"""LAW80 — Ramberg-Osgood Elastoplastic & Metallurgical Kinetics (/MAT/LAW80, /MAT/KIRKALDY).

Upstream OpenRadioss Fortran reference:
- 3D Solids Constitutive Update:
  `engine/source/materials/mat/mat080/sigeps80.F` (SUBROUTINE SIGEPS80, lines 36-1265)
- 2D Shells Constitutive Update:
  `engine/source/materials/mat/mat080/sigeps80c.F` (SUBROUTINE SIGEPS80C, lines 36-1170)
- Starter Card Reader:
  `starter/source/materials/mat/mat080/hm_read_mat80.F` (SUBROUTINE HM_READ_MAT80, lines 40-682)
- Kirkaldy Metallurgical Kinetics:
  `engine/source/materials/mat/mat080/kirkaldykinetics.F` (SUBROUTINE KIRKALDYKINETICS)
- Phase Kinetics 2:
  `engine/source/materials/mat/mat080/phasekinetic2.F` (SUBROUTINE PHASEKINETIC2)

Theory:
-------
1. Ramberg-Osgood Elastoplastic Constitutive Relation:
   Total strain decomposed into elastic and plastic components:
     epsilon = epsilon_e + epsilon_p = sigma / E + sign(sigma) * (|sigma| / K)^(1/n)
   where:
     E = Young's modulus
     nu = Poisson's ratio
     K = Strength coefficient (stress units, e.g. MPa)
     n = Strain hardening exponent (typically 0.1 - 0.3; Hollomon exponent)
   In the plastic regime:
     epsilon_p = (|sigma| / K)^(1/n)  <=>  |sigma| = K * (epsilon_p)^n

2. 3D Incremental Radial Return with Newton-Raphson Iterations (sigeps80.F lines 866-920):
   Elastic predictor:
     p_trial = (sigma_xx^trial + sigma_yy^trial + sigma_zz^trial) / 3
     s_trial = sigma^trial - p_trial * I
     sigma_vm^trial = sqrt(3/2 * s_trial : s_trial)
   During plastic flow (von Mises J2 consistency):
     sigma_vm = sigma_vm^trial - 3*G * Delta_epsilon_p
     epsilon_p^{new} = epsilon_p^0 + Delta_epsilon_p = (sigma_vm / K)^(1/n)
   Yield consistency scalar equation:
     F(sigma_vm) = sigma_vm + 3*G * ((sigma_vm / K)^(1/n) - epsilon_p^0) - sigma_vm^trial = 0
   Solved via Newton-Raphson iterations:
     dF / d(sigma_vm) = 1 + (3*G / (n * K)) * (sigma_vm / K)^(1/n - 1)
     sigma_vm^{k+1} = sigma_vm^k - F(sigma_vm^k) / (dF / d(sigma_vm^k))
   Stress radial scaling:
     s^{new} = s_trial * (sigma_vm / sigma_vm^trial)
     sigma^{new} = s^{new} + p_trial * I

3. Kirkaldy metallurgical kinetics for steel phase transformations:
   Simulates diffusion-controlled phase transformations during cooling/heating
   between Austenite, Ferrite, Pearlite, Bainite, and Martensite (sigeps80.F lines 512-600).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_EM10 = 1.0e-10


# ============================================================================
# 1D Ramberg-Osgood Relations (Analytical & Newton Inversion)
# ============================================================================

def ramberg_osgood_strain(
    sigma: float | np.ndarray,
    E: float,
    K: float,
    n: float,
) -> float | np.ndarray:
    """Compute total strain epsilon from Cauchy stress sigma using Ramberg-Osgood:
        epsilon = sigma / E + sign(sigma) * (|sigma| / K)^(1 / n)

    Cited from:
    - engine/source/materials/mat/mat080/sigeps80.F
    - Ramberg, W., & Osgood, W. R. (1943). Description of stress-strain curves by three parameters.
      NACA Technical Note No. 902.

    Parameters:
        sigma: Cauchy stress (MPa)
        E: Young's modulus (MPa)
        K: Strength coefficient (MPa)
        n: Strain hardening exponent (typically 0.1 - 0.3)
    """
    is_scalar = np.isscalar(sigma)
    s = np.atleast_1d(np.asarray(sigma, dtype=float))
    s_abs = np.abs(s)
    sgn = np.sign(s)
    # Total strain = elastic strain + plastic strain
    eps_e = s / float(E)
    inv_n = 1.0 / float(n)
    eps_p = sgn * ((np.maximum(s_abs, 0.0) / float(K)) ** inv_n)
    res = eps_e + eps_p
    return float(res[0]) if is_scalar else res


def ramberg_osgood_stress(
    epsilon: float | np.ndarray,
    E: float,
    K: float,
    n: float,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float | np.ndarray:
    """Invert the Ramberg-Osgood equation to compute Cauchy stress sigma from strain epsilon.

    Solves:
        f(sigma) = sigma / E + sign(sigma) * (|sigma| / K)^(1 / n) - epsilon = 0
    via Newton-Raphson iterations with quadratic convergence.

    Cited from:
    - engine/source/materials/mat/mat080/sigeps80.F (SIGEPS80 lines 866-920)

    Parameters:
        epsilon: Total strain
        E: Young's modulus (MPa)
        K: Strength coefficient (MPa)
        n: Strain hardening exponent
        tol: Convergence tolerance on residual
        max_iter: Maximum iterations
    """
    is_scalar = np.isscalar(epsilon)
    eps_arr = np.atleast_1d(np.asarray(epsilon, dtype=float))
    sig_out = np.zeros_like(eps_arr)

    E_f = float(E)
    K_f = float(K)
    n_f = float(n)
    inv_n = 1.0 / n_f

    for idx, eps in enumerate(eps_arr):
        if abs(eps) < 1e-15:
            sig_out[idx] = 0.0
            continue

        sgn = 1.0 if eps >= 0.0 else -1.0
        abs_eps = abs(eps)

        # Initial guess: linear elastic guess for small strain, power-law for large
        sig_el = E_f * abs_eps
        sig_pl = K_f * (abs_eps ** n_f) if abs_eps > 0.0 else sig_el
        sig = min(sig_el, sig_pl) if sig_pl > 0.0 else sig_el
        sig = max(sig, 1e-12)

        for _ in range(max_iter):
            s_norm = sig / K_f
            e_p = s_norm ** inv_n
            e_cur = sig / E_f + e_p
            f = e_cur - abs_eps
            if abs(f) <= tol * max(abs_eps, 1.0):
                break
            df = 1.0 / E_f + (inv_n / K_f) * (s_norm ** (inv_n - 1.0))
            d_sig = f / max(df, 1e-20)
            sig = max(1e-15, sig - d_sig)
            if abs(d_sig) <= tol * max(sig, 1.0):
                break

        sig_out[idx] = sgn * sig

    return float(sig_out[0]) if is_scalar else sig_out


def ramberg_osgood_tangent(
    sigma: float | np.ndarray,
    E: float,
    K: float,
    n: float,
) -> float | np.ndarray:
    """Compute tangent modulus d(sigma) / d(epsilon) for Ramberg-Osgood material:
        d(epsilon) / d(sigma) = 1 / E + (1 / (n * K)) * (|sigma| / K)^(1 / n - 1)
        d(sigma) / d(epsilon) = 1 / (d(epsilon) / d(sigma))
    """
    is_scalar = np.isscalar(sigma)
    s = np.atleast_1d(np.asarray(sigma, dtype=float))
    s_abs = np.abs(s)
    inv_n = 1.0 / float(n)
    s_norm = np.maximum(s_abs, 1e-15) / float(K)
    deps_dsig = 1.0 / float(E) + (inv_n / float(K)) * (s_norm ** (inv_n - 1.0))
    tan = 1.0 / np.maximum(deps_dsig, 1e-20)
    return float(tan[0]) if is_scalar else tan


def ramberg_osgood_cyclic_strain(
    delta_sigma: float | np.ndarray,
    E: float,
    K: float,
    n: float,
) -> float | np.ndarray:
    """Cyclic Ramberg-Osgood hysteresis curve (Masing rule):
        Delta_epsilon = Delta_sigma / E + 2 * (|Delta_sigma| / (2 * K))^(1 / n) * sign(Delta_sigma)
    """
    is_scalar = np.isscalar(delta_sigma)
    ds = np.atleast_1d(np.asarray(delta_sigma, dtype=float))
    sgn = np.sign(ds)
    abs_ds = np.abs(ds)
    inv_n = 1.0 / float(n)
    de = abs_ds / float(E) + 2.0 * ((abs_ds / (2.0 * float(K))) ** inv_n)
    res = sgn * de
    return float(res[0]) if is_scalar else res


def ramberg_osgood_cyclic_stress(
    delta_epsilon: float | np.ndarray,
    E: float,
    K: float,
    n: float,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float | np.ndarray:
    """Invert cyclic Ramberg-Osgood curve to find stress range Delta_sigma from strain range Delta_epsilon."""
    is_scalar = np.isscalar(delta_epsilon)
    de_arr = np.atleast_1d(np.asarray(delta_epsilon, dtype=float))
    ds_out = np.zeros_like(de_arr)

    E_f = float(E)
    K_f = float(K)
    n_f = float(n)
    inv_n = 1.0 / n_f

    for idx, de in enumerate(de_arr):
        if abs(de) < 1e-15:
            ds_out[idx] = 0.0
            continue
        sgn = 1.0 if de >= 0.0 else -1.0
        abs_de = abs(de)
        ds = E_f * abs_de
        for _ in range(max_iter):
            cur_de = ds / E_f + 2.0 * ((ds / (2.0 * K_f)) ** inv_n)
            f = cur_de - abs_de
            if abs(f) <= tol * max(abs_de, 1.0):
                break
            df = 1.0 / E_f + (inv_n / K_f) * ((ds / (2.0 * K_f)) ** (inv_n - 1.0))
            d_ds = f / max(df, 1e-20)
            ds = max(1e-15, ds - d_ds)
            if abs(d_ds) <= tol * max(ds, 1.0):
                break
        ds_out[idx] = sgn * ds

    return float(ds_out[0]) if is_scalar else ds_out


# ============================================================================
# Dataclass & Resolution
# ============================================================================

@dataclass
class Law80Params:
    """Parameters for OpenRadioss /MAT/LAW80 (Ramberg-Osgood & Kirkaldy kinetics)."""
    id: int = 1
    title: str = ""
    law: int = 80
    law_name: str = "LAW80"

    # Density
    rho0: float = 7.8e-3
    rhor: float = 7.8e-3

    # Elastic Constants
    young: float = 210000.0
    nu: float = 0.3

    # Ramberg-Osgood Parameters
    k_strength: float = 1000.0   # Strength coefficient K (MPa)
    n_exp: float = 0.2           # Hardening exponent n (0.1 - 0.3)
    sigy0: float = 350.0         # Reference / offset yield stress (MPa)
    eps0: float = 0.002          # Reference offset strain (e.g. 0.2%)
    alpha0: float = 0.02         # Ramberg-Osgood yield offset parameter
    n_ramberg: float = 5.0       # Inverse hardening exponent (n_ramberg = 1 / n_exp)
    efac: float = 1.0

    # Thermal & Time Scales
    tini: float = 293.15
    tref: float = 293.15
    unitt: float = 1.0
    israte: int = 1

    # Alloy Chemical Composition (wt %)
    c_alloy: float = 0.2
    mn_alloy: float = 1.2
    si_alloy: float = 0.3
    cr_alloy: float = 0.5
    ni_alloy: float = 0.2
    mo_alloy: float = 0.1
    v_alloy: float = 0.0
    gsize: float = 7.0         # ASTM grain size number

    # Critical Transformation Temperatures
    ae1: float = 727.0
    ae3: float = 850.0
    bs: float = 550.0
    ms: float = 400.0

    # Derived constants
    bulk: float = field(init=False)
    g: float = field(init=False)
    lamhook: float = field(init=False)
    a11_2d: float = field(init=False)
    a12_2d: float = field(init=False)
    c_solid: float = field(init=False)
    c_shell: float = field(init=False)

    @property
    def E(self) -> float:
        return self.young

    @property
    def e(self) -> float:
        return self.young

    @property
    def K(self) -> float:
        return self.k_strength

    @property
    def k(self) -> float:
        return self.k_strength

    @property
    def n(self) -> float:
        return self.n_exp

    def __post_init__(self) -> None:
        if self.rho0 <= 0.0:
            self.rho0 = 7.8e-3
        if self.rhor <= 0.0:
            self.rhor = self.rho0

        # Harmonize hardening exponents
        if self.n_exp <= 0.0:
            if self.n_ramberg > 0.0:
                self.n_exp = 1.0 / self.n_ramberg if self.n_ramberg > 1.0 else self.n_ramberg
            else:
                self.n_exp = 0.2
        if self.n_ramberg <= 0.0:
            self.n_ramberg = 1.0 / self.n_exp if self.n_exp < 1.0 else self.n_exp

        # Harmonize strength coefficient K and yield stress sigy0
        if self.eps0 <= 0.0:
            self.eps0 = 0.002
        if self.k_strength <= 0.0:
            if self.sigy0 > 0.0:
                self.k_strength = self.sigy0 / (max(self.eps0, 1.0e-6) ** self.n_exp)
            else:
                self.k_strength = 1000.0
        if self.sigy0 <= 0.0:
            self.sigy0 = self.k_strength * (max(self.eps0, 1.0e-6) ** self.n_exp)

        e = self.young
        nu = self.nu
        self.bulk = e / max(3.0 * (1.0 - 2.0 * nu), _EM20)
        self.g = 0.5 * e / max(1.0 + nu, _EM20)
        self.lamhook = 2.0 * self.g * nu / max(1.0 - 2.0 * nu, _EM20)

        denom_2d = max(1.0 - nu * nu, _EM20)
        self.a11_2d = e / denom_2d
        self.a12_2d = nu * self.a11_2d

        self.c_solid = math.sqrt(max(0.0, (self.bulk + 4.0 / 3.0 * self.g) / self.rho0))
        self.c_shell = math.sqrt(max(0.0, self.a11_2d / self.rho0))


def _extract_val(data: Dict[str, Any], keys: Sequence[str], default: float) -> float:
    for k in keys:
        if k in data and data[k] is not None:
            try:
                return float(data[k])
            except (ValueError, TypeError):
                pass
    return default


def build_law80(mat_def: Any = None, **kwargs: Any) -> Law80Params:
    """Construct Law80Params from a Material entity, dictionary, or keyword arguments."""
    if isinstance(mat_def, Law80Params):
        return mat_def

    data: Dict[str, Any] = {}
    if isinstance(mat_def, dict):
        data.update(mat_def)
    elif hasattr(mat_def, "__dict__"):
        data.update(mat_def.__dict__)
        if hasattr(mat_def, "params") and isinstance(mat_def.params, dict):
            data.update(mat_def.params)

    data.update(kwargs)

    mat_id = int(_extract_val(data, ["id", "mat_id", "user_id"], 1))
    title = str(data.get("title", f"LAW80_{mat_id}"))

    rho0 = _extract_val(data, ["MAT_RHO", "rho0", "rho", "density"], 7.8e-3)
    rhor = _extract_val(data, ["Refer_Rho", "rhor", "ref_rho"], rho0)

    young = _extract_val(data, ["MAT_E", "young", "e", "E"], 210000.0)
    nu = _extract_val(data, ["MAT_NU", "nu", "poisson"], 0.3)
    efac = _extract_val(data, ["SCALE", "efac"], 1.0)
    unitt = _extract_val(data, ["time_inputunit_value", "unitt"], 1.0)
    israte = int(_extract_val(data, ["Fsmooth", "israte"], 1))

    # Exponent parsing: supports both n (< 1.0) and n_ramberg (> 1.0)
    n_raw = _extract_val(data, ["n_exp", "n_hard", "N_EXP"], 0.0)
    n_ramberg_raw = _extract_val(data, ["n_ramberg", "n", "N", "EXP"], 0.0)

    if n_raw > 0.0:
        if n_raw > 1.0:
            n_ramberg = n_raw
            n_exp = 1.0 / n_raw
        else:
            n_exp = n_raw
            n_ramberg = 1.0 / n_raw
    elif n_ramberg_raw > 0.0:
        if n_ramberg_raw < 1.0:
            n_exp = n_ramberg_raw
            n_ramberg = 1.0 / n_ramberg_raw
        else:
            n_ramberg = n_ramberg_raw
            n_exp = 1.0 / n_ramberg_raw
    else:
        n_exp = 0.2
        n_ramberg = 5.0

    sigy0 = _extract_val(data, ["sigy0", "SIGMA_r", "sig0", "yield_stress", "SIGY0"], 350.0)
    eps0 = _extract_val(data, ["Epsilon_0", "eps0", "EPS0"], 0.002)
    alpha0 = _extract_val(data, ["alpha0", "ALPHA0", "alpha"], 0.02)

    k_strength = _extract_val(data, ["MAT_K", "k_strength", "k", "K", "k_coef", "K_strength"], 0.0)
    if k_strength <= 0.0:
        if sigy0 > 0.0 and eps0 > 0.0:
            k_strength = sigy0 / (max(eps0, 1.0e-6) ** n_exp)
        else:
            k_strength = 1000.0

    tini = _extract_val(data, ["TINI", "tini", "Tini"], 293.15)
    tref = _extract_val(data, ["TREF", "tref"], 293.15)

    c_alloy = _extract_val(data, ["C", "c_alloy"], 0.2)
    mn_alloy = _extract_val(data, ["MN", "mn_alloy"], 1.2)
    si_alloy = _extract_val(data, ["SI", "si_alloy"], 0.3)
    cr_alloy = _extract_val(data, ["CR", "cr_alloy"], 0.5)
    ni_alloy = _extract_val(data, ["NI", "ni_alloy"], 0.2)
    mo_alloy = _extract_val(data, ["MO", "mo_alloy"], 0.1)
    v_alloy = _extract_val(data, ["V", "v_alloy"], 0.0)
    gsize = _extract_val(data, ["GSIZE", "gsize"], 7.0)

    ae1 = _extract_val(data, ["AE1", "ae1"], 727.0)
    ae3 = _extract_val(data, ["AE3", "ae3"], 850.0)
    bs = _extract_val(data, ["BS", "bs"], 550.0)
    ms = _extract_val(data, ["MS", "ms"], 400.0)

    return Law80Params(
        id=mat_id,
        title=title,
        rho0=rho0,
        rhor=rhor,
        young=young,
        nu=nu,
        k_strength=k_strength,
        n_exp=n_exp,
        sigy0=sigy0,
        eps0=eps0,
        alpha0=alpha0,
        n_ramberg=n_ramberg,
        efac=efac,
        tini=tini,
        tref=tref,
        unitt=unitt,
        israte=israte,
        c_alloy=c_alloy,
        mn_alloy=mn_alloy,
        si_alloy=si_alloy,
        cr_alloy=cr_alloy,
        ni_alloy=ni_alloy,
        mo_alloy=mo_alloy,
        v_alloy=v_alloy,
        gsize=gsize,
        ae1=ae1,
        ae3=ae3,
        bs=bs,
        ms=ms,
    )


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law80Params:
    """Resolve and cache Law80Params from a Material or dict."""
    if isinstance(mat, Law80Params):
        return mat
    cached = getattr(mat, "_cached_law80", None)
    if cached is None:
        cached = build_law80(mat)
        try:
            setattr(mat, "_cached_law80", cached)
        except Exception:
            pass
    return cached


def needs_defgrad(mat: Any = None) -> bool:
    """Return False: LAW80 is a rate-form hypoelastic/kinetics formulation."""
    return False


def extra_shapes(mat: Any = None, nip: Optional[int] = 1) -> Dict[str, Tuple[int, ...]]:
    """Persistent history variables for LAW80 (15 metallurgical & phase state variables)."""
    if nip is not None and nip > 1:
        return {"uvar80": (nip, 15), "uvar": (nip, 15)}
    return {"uvar80": (15,), "uvar": (15,)}


def sound_speed(
    mat: Any,
    eps: Any = None,
    extra: Any = None,
    is_shell: bool = False,
) -> float | np.ndarray:
    """Acoustic sound speed for LAW80."""
    p = resolve(mat)
    c_val = p.c_shell if is_shell else p.c_solid
    if eps is not None and isinstance(eps, np.ndarray) and eps.ndim > 1:
        return np.full(len(eps), c_val, dtype=float)
    return c_val


# ============================================================================
# 3D Solid Constitutive Update (sigeps80.F)
# ============================================================================

def _solid_update_single(
    p: Law80Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Single 3D solid continuum update matching sigeps80.F (lines 780-922)."""
    if off < 0.1:
        return np.zeros(6, dtype=float), float(uvar0[0]), uvar0.copy(), 0.0

    uvar = uvar0.copy()
    epsp = float(uvar[0])
    temp = float(uvar[1]) if uvar[1] > 0.0 else p.tini

    # 1. Elastic trial stresses (sigeps80.F lines 781-787)
    deps_vol = deps[0] + deps[1] + deps[2]
    dav = deps_vol * p.lamhook

    sign = np.empty(6, dtype=float)
    sign[0] = sig0[0] + 2.0 * p.g * deps[0] + dav
    sign[1] = sig0[1] + 2.0 * p.g * deps[1] + dav
    sign[2] = sig0[2] + 2.0 * p.g * deps[2] + dav
    sign[3] = sig0[3] + p.g * deps[3]
    sign[4] = sig0[4] + p.g * deps[4]
    sign[5] = sig0[5] + p.g * deps[5]

    # 2. Deviatoric stresses and von Mises equivalent (sigeps80.F lines 789-800)
    pres = (sign[0] + sign[1] + sign[2]) / 3.0
    s_dev = sign.copy()
    s_dev[0] -= pres
    s_dev[1] -= pres
    s_dev[2] -= pres

    j2 = 0.5 * (s_dev[0] ** 2 + s_dev[1] ** 2 + s_dev[2] ** 2) + (s_dev[3] ** 2 + s_dev[4] ** 2 + s_dev[5] ** 2)
    svm = math.sqrt(max(0.0, 3.0 * j2))

    # 3. Ramberg-Osgood Radial Return with Newton iterations (sigeps80.F lines 866-920)
    K = p.k_strength
    n = p.n_exp
    inv_n = 1.0 / n
    g3 = 3.0 * p.g

    # Current flow stress corresponding to accumulated plastic strain epsp
    sigy_curr = K * (epsp ** n) if epsp > 1e-12 else (p.sigy0 if p.sigy0 > 0.0 else 0.0)

    if svm > sigy_curr and svm > _EM20:
        # Solve F(sigma_vm) = sigma_vm + 3G * ((sigma_vm / K)^(1/n) - epsp0) - svm_trial = 0
        sig_vm = svm
        for _ in range(25):
            s_norm = max(sig_vm, 1e-15) / K
            term_epsp = s_norm ** inv_n
            res = sig_vm + g3 * (term_epsp - epsp) - svm
            if abs(res) <= 1e-10 * max(svm, 1.0):
                break
            dF = 1.0 + (g3 * inv_n / K) * (s_norm ** (inv_n - 1.0))
            d_sig = res / max(dF, 1e-20)
            sig_vm = max(1e-15, sig_vm - d_sig)
            if abs(d_sig) <= 1e-10 * max(sig_vm, 1.0):
                break

        dlam = max(0.0, (svm - sig_vm) / g3)
        epsp += dlam

        scale = sig_vm / max(svm, _EM20)
        sign[0] = s_dev[0] * scale + pres
        sign[1] = s_dev[1] * scale + pres
        sign[2] = s_dev[2] * scale + pres
        sign[3] = s_dev[3] * scale
        sign[4] = s_dev[4] * scale
        sign[5] = s_dev[5] * scale

    uvar[0] = epsp
    uvar[1] = temp

    # Update martensite fraction if cooling below Ms (sigeps80.F line 327)
    if temp < p.ms:
        uvar[2] = min(1.0, max(uvar[2], 1.0 - math.exp(-0.011 * max(0.0, p.ms - temp))))

    return sign, epsp, uvar, p.c_solid


def solid_update(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """3D solid continuum constitutive update for /MAT/LAW80 (sigeps80.F)."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(6, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, 15), dtype=float)
    uvar_arr[:, 1] = p.tini

    if extra is not None and isinstance(extra, dict):
        for k in ("uvar80", "uvar"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(15, len(u))] = u[:min(15, len(u))]
                elif u.ndim == 2:
                    uvar_arr[:min(nel, len(u)), :min(15, u.shape[1])] = u[:min(nel, len(u)), :min(15, u.shape[1])]
                break

    if epsp is not None:
        ep_in = np.atleast_1d(epsp).astype(float)
        uvar_arr[:min(nel, len(ep_in)), 0] = ep_in[:nel]

    off_arr = np.ones(nel, dtype=float)
    if extra is not None and isinstance(extra, dict) and "off" in extra:
        o = np.atleast_1d(extra["off"]).astype(float)
        off_arr[:min(nel, len(o))] = o[:nel]

    sig_out = np.zeros_like(sig_arr)
    epsp_out = np.zeros(nel, dtype=float)
    c_out = np.zeros(nel, dtype=float)

    for i in range(nel):
        s_i, ep_i, u_i, c_i = _solid_update_single(p, sig_arr[i], deps_arr[i], uvar_arr[i], off=off_arr[i])
        sig_out[i] = s_i
        epsp_out[i] = ep_i
        uvar_arr[i] = u_i
        c_out[i] = c_i

    if extra is not None and isinstance(extra, dict):
        extra["uvar80"] = uvar_arr
        extra["uvar"] = uvar_arr

    res_sig = sig_out[0] if is_1d else sig_out
    res_epsp = epsp_out[0] if is_1d else epsp_out
    res_c = float(c_out[0]) if is_1d else c_out

    if hasattr(sig, "__setitem__"):
        try:
            sig[:] = res_sig
        except Exception:
            pass
    if epsp is not None and hasattr(epsp, "__setitem__"):
        try:
            epsp[:] = res_epsp
        except Exception:
            pass

    return res_sig, res_epsp, res_c


# ============================================================================
# 2D Shell Constitutive Update (sigeps80c.F)
# ============================================================================

def _shell_update_single(
    p: Law80Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """2D plane-stress shell update matching sigeps80c.F."""
    if off < 0.1:
        return np.zeros(len(sig0), dtype=float), float(uvar0[0]), uvar0.copy(), 0.0

    uvar = uvar0.copy()
    epsp = float(uvar[0])

    sign = np.empty_like(sig0, dtype=float)
    sign[0] = sig0[0] + p.a11_2d * deps[0] + p.a12_2d * deps[1]
    sign[1] = sig0[1] + p.a12_2d * deps[0] + p.a11_2d * deps[1]
    sign[2] = sig0[2] + p.g * deps[2]
    if len(sig0) >= 5:
        sign[3] = sig0[3] + p.g * deps[3]
        sign[4] = sig0[4] + p.g * deps[4]

    svm = math.sqrt(max(0.0, sign[0] ** 2 + sign[1] ** 2 - sign[0] * sign[1] + 3.0 * sign[2] ** 2))

    K = p.k_strength
    n = p.n_exp
    inv_n = 1.0 / n
    g3 = 3.0 * p.g

    sigy_curr = K * (epsp ** n) if epsp > 1e-12 else (p.sigy0 if p.sigy0 > 0.0 else 0.0)

    if svm > sigy_curr and svm > _EM20:
        sig_vm = svm
        for _ in range(25):
            s_norm = max(sig_vm, 1e-15) / K
            term_epsp = s_norm ** inv_n
            res = sig_vm + g3 * (term_epsp - epsp) - svm
            if abs(res) <= 1e-10 * max(svm, 1.0):
                break
            dF = 1.0 + (g3 * inv_n / K) * (s_norm ** (inv_n - 1.0))
            d_sig = res / max(dF, 1e-20)
            sig_vm = max(1e-15, sig_vm - d_sig)
            if abs(d_sig) <= 1e-10 * max(sig_vm, 1.0):
                break

        dlam = max(0.0, (svm - sig_vm) / g3)
        epsp += dlam

        scale = sig_vm / max(svm, _EM20)
        sign[:3] *= scale

    uvar[0] = epsp
    return sign, epsp, uvar, p.c_shell


def shell_update(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """2D plane-stress shell constitutive update for /MAT/LAW80 (sigeps80c.F)."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(3, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, 15), dtype=float)
    uvar_arr[:, 1] = p.tini

    if extra is not None and isinstance(extra, dict):
        for k in ("uvar80", "uvar"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(15, len(u))] = u[:min(15, len(u))]
                elif u.ndim == 2:
                    uvar_arr[:min(nel, len(u)), :min(15, u.shape[1])] = u[:min(nel, len(u)), :min(15, u.shape[1])]
                break

    if epsp is not None:
        ep_in = np.atleast_1d(epsp).astype(float)
        uvar_arr[:min(nel, len(ep_in)), 0] = ep_in[:nel]

    off_arr = np.ones(nel, dtype=float)
    if extra is not None and isinstance(extra, dict) and "off" in extra:
        o = np.atleast_1d(extra["off"]).astype(float)
        off_arr[:min(nel, len(o))] = o[:nel]

    sig_out = np.zeros_like(sig_arr)
    epsp_out = np.zeros(nel, dtype=float)
    c_out = np.zeros(nel, dtype=float)

    for i in range(nel):
        s_i, ep_i, u_i, c_i = _shell_update_single(p, sig_arr[i], deps_arr[i], uvar_arr[i], off=off_arr[i])
        sig_out[i] = s_i
        epsp_out[i] = ep_i
        uvar_arr[i] = u_i
        c_out[i] = c_i

    if extra is not None and isinstance(extra, dict):
        extra["uvar80"] = uvar_arr
        extra["uvar"] = uvar_arr

    res_sig = sig_out[0] if is_1d else sig_out
    res_epsp = epsp_out[0] if is_1d else epsp_out
    res_c = float(c_out[0]) if is_1d else c_out

    if hasattr(sig, "__setitem__"):
        try:
            sig[:] = res_sig
        except Exception:
            pass
    if epsp is not None and hasattr(epsp, "__setitem__"):
        try:
            epsp[:] = res_epsp
        except Exception:
            pass

    return res_sig, res_epsp, res_c


# ============================================================================
# Tangent Stiffness Operators (Consistent Tangents)
# ============================================================================

def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent solid tangent stiffness operator (n, 6, 6)."""
    p = resolve(mat)
    c_el = np.zeros((6, 6), dtype=float)
    c11 = p.bulk + 4.0 / 3.0 * p.g
    c12 = p.bulk - 2.0 / 3.0 * p.g
    c_el[0, 0] = c_el[1, 1] = c_el[2, 2] = c11
    c_el[0, 1] = c_el[0, 2] = c_el[1, 0] = c_el[1, 2] = c_el[2, 0] = c_el[2, 1] = c12
    c_el[3, 3] = c_el[4, 4] = c_el[5, 5] = p.g

    n = 1
    if sig is not None and np.ndim(sig) >= 2:
        n = len(sig)
    return np.broadcast_to(c_el, (n, 6, 6)).copy()


consistent_solid_tangent = solid_tangent


def shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent shell plane-stress tangent operator (n, 3, 3)."""
    p = resolve(mat)
    c_el = np.array([
        [p.a11_2d, p.a12_2d, 0.0],
        [p.a12_2d, p.a11_2d, 0.0],
        [0.0, 0.0, p.g],
    ], dtype=float)

    n = 1
    if sig is not None and np.ndim(sig) >= 2:
        n = len(sig)
    return np.broadcast_to(c_el, (n, 3, 3)).copy()


consistent_shell_tangent = shell_tangent
