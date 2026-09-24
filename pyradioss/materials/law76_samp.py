"""
LAW76 — SAMP-1 Semi-Analytical Model for Polymers (/MAT/LAW76, /MAT/SAMP, /MAT/SAMP-1).

Upstream Fortran reference:
- Starter reader: ``starter/source/materials/mat/mat076/hm_read_mat76.F``
- Hardening synthesis: ``starter/source/materials/mat/mat076/law76_func_comp.F90``
- Engine 3D solids: ``engine/source/materials/mat/mat076/sigeps76.F``
- 3D non-associated flow: ``engine/source/materials/mat/mat076/no_asso_plas76.F``
- 3D associated flow: ``engine/source/materials/mat/mat076/asso_plas76.F``
- Engine 2D shells: ``engine/source/materials/mat/mat076/sigeps76c.F``
- 2D quadratic yield: ``engine/source/materials/mat/mat076/no_asso_qplas76c.F``
- 2D linear yield: ``engine/source/materials/mat/mat076/no_asso_lplas76c.F``
- 2D associated quadratic: ``engine/source/materials/mat/mat076/asso_qplas76c.F``
- HyperMesh CFG: ``hm_cfg_files/config/CFG/radioss2018/MAT/matl76_76.cfg``

Theory
------
SAMP-1 (Semi-Analytical Model for Polymers) represents thermoplastics and general
polymers undergoing large deformations, with:
1. Pressure-dependent yield surface:
   - Quadratic yield criterion (IQUAD = 1, default):
     f = sigma_vm^2 - A_0 - A_1*p - A_2*p^2 <= 0
     where:
       A_0 = 3 * sigma_s^2
       A_1 = 9 * sigma_s^2 * (sigma_c - sigma_t) / (sigma_c * sigma_t)
       A_2 = 9 * (sigma_c * sigma_t - 3 * sigma_s^2) / (sigma_c * sigma_t)
       p   = -1/3 * tr(sigma) (hydrostatic pressure, positive in compression)
   - Linear yield criterion (IQUAD = 0):
     f = sigma_vm - A_0 - A_1*p - A_2*p^2 <= 0
     where:
       A_0 = sqrt(3) * sigma_s
       A_1 = 3 * [ (sigma_t - sigma_c)/(sigma_t + sigma_c) - A_0 * (sigma_t - sigma_c)/(sigma_t * sigma_c) ]
       A_2 = 18 * [ 1/(sigma_t + sigma_c) - A_0 / (2 * sigma_t * sigma_c) ]
2. Non-associated plastic potential (IFORM = 0, default):
   - g(sigma, p) = sqrt(sigma_vm^2 + alpha * p^2)
     where alpha = 4.5 * (1 - 2*nu_p) / (1 + nu_p)
     nu_p is the plastic Poisson's ratio. If nu_p = 0.5, alpha = 0 (isochoric plasticity).
     If nu_p < 0.5, alpha > 0 (plastic dilatation in tension, compaction in compression).
3. Hardening curves:
   - fct_ID_T: Uniaxial tension yield stress vs plastic strain (and strain rate).
   - fct_ID_C: Uniaxial compression yield stress vs plastic strain.
   - fct_ID_S: Shear yield stress vs plastic strain.
   - fct_ID_B: Biaxial yield stress vs plastic strain.
   - ICAS flag:
     -1: All 3 curves (T, C, S) supplied.
      0: Tension only -> sigma_c = sigma_t, sigma_s = sigma_t / sqrt(3).
      1: Tension + Compression -> sigma_s = sqrt(sigma_c * sigma_t / 3) (IQUAD=1).
      2: Tension + Shear -> Compression curve synthesized via law76_func_comp.F90.
4. Convexity enforcement (ICONV = 1):
   Ensures sigma_s >= 1.05 * sqrt(sigma_c * sigma_t / 3) (or linear equivalent).
5. Damage & stiffness degradation:
   D = min(1.0, max(0.0, (eps_p - eps_f) / (eps_r - eps_f)))
   sigma = (1 - D) * sigma_undamaged
   E_eff = (1 - D) * E.
6. Shell plane stress (sigeps76c.F):
   Enforces sigma_zz = 0, computes through-thickness strain increment:
   deps_zz = -nu/(1-nu)*(deps_xx + deps_yy) + nu/(1-nu)*(deps^p_xx + deps^p_yy) + deps^p_zz
   and updates shell thickness h = h + deps_zz * h_0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_INF = 1.0e30
_SQR3 = math.sqrt(3.0)
_SFAC = 1.05  # Security factor for ICONV convexity check (sigeps76.F line 105)


# =============================================================================
# Parameter Dataclass: Law76Params
# =============================================================================

@dataclass
class Law76Params:
    """Strongly-typed parameters for /MAT/LAW76 (/MAT/SAMP, /MAT/SAMP-1).

    Follows starter/source/materials/mat/mat076/hm_read_mat76.F.
    """
    id: int = 1
    title: str = ""
    law: int = 76
    law_name: str = "LAW76"

    # Card 1: Density
    rho0: float = 1.0
    refer_rho: float = 1.0

    # Card 2: Linear Elastic Constants
    e: float = 2100.0          # Initial Young's modulus
    nu: float = 0.35           # Initial Poisson's ratio
    g: float = 0.0             # Shear modulus G = E / (2*(1+nu))
    bulk: float = 0.0          # Bulk modulus K = C1 = E / (3*(1-2*nu))
    a1: float = 0.0            # First component of 3D elasticity matrix
    a2: float = 0.0            # Second component of 3D elasticity matrix
    a11_2d: float = 0.0        # E / (1 - nu^2) for plane stress
    a12_2d: float = 0.0        # nu * E / (1 - nu^2) for plane stress

    # Card 3: Function / Table Identifiers
    fun_d1: Any = 0            # Tension yield stress table/curve (TAB_ID1 / fct_ID_T)
    fun_d2: Any = 0            # Compression yield stress table/curve (TAB_ID2 / fct_ID_C)
    fun_d3: Any = 0            # Shear yield stress table/curve (TAB_ID3 / fct_ID_S)
    fun_d4: Any = 0            # Biaxial yield stress table/curve (fct_ID_B)

    # Card 4: Scale Factors
    fscale_t: float = 1.0      # Stress scale factor for tension (FScale11)
    fscale_c: float = 1.0      # Stress scale factor for compression (FScale22)
    fscale_s: float = 1.0      # Stress scale factor for shear (FScale33)
    fscale_b: float = 1.0      # Stress scale factor for biaxial (FScale12)
    facx: float = 1.0          # Strain rate scale factor (FACX)

    # Card 5: Plastic Poisson's Ratio & Strain Rate Filtering
    mat_nut: float = 0.3       # Plastic Poisson's ratio (Nu_p)
    fun_b5: Any = 0            # Plastic Poisson's ratio function (fct_IDpr)
    mat_pscale: float = 1.0    # Scale factor for Nu_p curve (Fscale_pr)
    israte: int = 0            # Strain rate smoothing flag (0: none, 1: smoothed)
    asrate: float = 1.0e30     # Cutoff frequency for strain rate filtering (Fcut)

    # Card 6: Failure & Rupture
    eps_f: float = _INF        # Failure plastic strain (Epsilon_f_p)
    eps_r: float = 2.0 * _INF  # Rupture plastic strain (Epsilon_r_p)
    dc: float = 1.0            # Critical damage (D_c)

    # Card 7: Damage Function
    fun_a1: Any = 0            # Damage vs plastic strain function (fct_ID1)
    fun_a2: Any = 0            # Multiplier on Dc depending on triaxiality
    fun_a3: Any = 0            # Multiplier on Dc depending on element size
    scale_dmg: float = 1.0     # Scale factor for damage curve

    # Card 8: Formulation Flags
    iform: int = 0             # 0: non-associated flow, 1: associated flow
    iquad: int = 1             # 0: linear yield surface, 1: quadratic yield surface
    iconv: int = 0             # 0: unconstrained, 1: ensure convexity (material stability)
    icas: int = 0              # -1: T+C+S, 0: T only, 1: T+C, 2: T+S

    # Resolved Curve Objects or Tables
    tens_curve: Any = None
    comp_curve: Any = None
    shear_curve: Any = None
    biax_curve: Any = None
    nup_curve: Any = None
    dmg_curve: Any = None

    # Fallback initial yield values if curves not provided
    sig_t0: float = 0.0        # Initial tensile yield stress
    sig_c0: float = 0.0        # Initial compressive yield stress
    sig_s0: float = 0.0        # Initial shear yield stress
    h_t: float = 0.0           # Tensile linear hardening modulus
    h_c: float = 0.0           # Compressive linear hardening modulus
    h_s: float = 0.0           # Shear linear hardening modulus

    def __post_init__(self) -> None:
        """Derive and populate elastic constants if not explicitly provided."""
        if self.g == 0.0 and self.e > 0.0:
            self.g = self.e / (2.0 * max(1.0e-12, 1.0 + self.nu))
        if self.bulk == 0.0 and self.e > 0.0:
            self.bulk = self.e / (3.0 * max(1.0e-12, 1.0 - 2.0 * self.nu))
        if self.a1 == 0.0 and self.bulk > 0.0:
            self.a1 = self.bulk + (4.0 / 3.0) * self.g
        if self.a2 == 0.0 and self.bulk > 0.0:
            self.a2 = self.bulk - (2.0 / 3.0) * self.g
        if self.a11_2d == 0.0 and self.e > 0.0:
            denom = max(1.0e-12, 1.0 - self.nu**2)
            self.a11_2d = self.e / denom
            self.a12_2d = self.nu * self.a11_2d

    @property
    def E(self) -> float:
        return self.e

    @property
    def K(self) -> float:
        return self.bulk

    @property
    def G(self) -> float:
        return self.g

    @property
    def nu_p(self) -> float:
        return self.mat_nut


# =============================================================================
# Helper: Curve Evaluation & law76_func_comp
# =============================================================================

def _eval_curve_or_val(
    curve_obj: Any,
    x: float,
    rate: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    default_val: float = 0.0,
    default_slope: float = 0.0,
) -> Tuple[float, float]:
    """Evaluate curve or scalar with slope at (x, rate).

    Returns
    -------
    (val, slope) : Tuple[float, float]
        Evaluated value and tangent slope d(val)/dx.
    """
    if curve_obj is None:
        return default_val + default_slope * x, default_slope

    # Numeric scalar fallback
    if isinstance(curve_obj, (int, float, np.integer, np.floating)):
        if curve_obj == 0:
            return default_val + default_slope * x, default_slope
        val = float(curve_obj) * scale_y
        return val + default_slope * x, default_slope

    # Callable (e.g., lambda x, r=0: ...)
    if callable(curve_obj):
        try:
            val = float(curve_obj(x * scale_x, rate)) * scale_y
            h = max(1.0e-7, 1.0e-5 * abs(x))
            val_plus = float(curve_obj((x + h) * scale_x, rate)) * scale_y
            slope = (val_plus - val) / h
            return max(0.0, val), max(0.0, slope)
        except TypeError:
            val = float(curve_obj(x * scale_x)) * scale_y
            h = max(1.0e-7, 1.0e-5 * abs(x))
            val_plus = float(curve_obj((x + h) * scale_x)) * scale_y
            slope = (val_plus - val) / h
            return max(0.0, val), max(0.0, slope)

    # 2D numpy array of shape (N, 2)
    if isinstance(curve_obj, np.ndarray):
        if curve_obj.ndim == 2 and curve_obj.shape[1] >= 2:
            xs, ys = curve_obj[:, 0], curve_obj[:, 1]
            if len(xs) == 0:
                return default_val, default_slope
            if len(xs) == 1:
                return float(ys[0]) * scale_y, 0.0
            idx = np.searchsorted(xs, x * scale_x, side="right") - 1
            idx = max(0, min(idx, len(xs) - 2))
            dx = xs[idx + 1] - xs[idx]
            if abs(dx) < 1.0e-14:
                return float(ys[idx]) * scale_y, 0.0
            slp = (ys[idx + 1] - ys[idx]) / dx
            val = (ys[idx] + slp * (x * scale_x - xs[idx])) * scale_y
            return max(0.0, float(val)), max(0.0, float(slp * scale_y * scale_x))

    # 2D/1D table or list/tuple of (x_vals, y_vals)
    if isinstance(curve_obj, (list, tuple)) and len(curve_obj) == 2:
        xs, ys = np.asarray(curve_obj[0], dtype=float), np.asarray(curve_obj[1], dtype=float)
        if len(xs) == 0:
            return default_val, default_slope
        if len(xs) == 1:
            return float(ys[0]) * scale_y, 0.0
        idx = np.searchsorted(xs, x * scale_x, side="right") - 1
        idx = max(0, min(idx, len(xs) - 2))
        dx = xs[idx + 1] - xs[idx]
        if abs(dx) < 1.0e-14:
            return float(ys[idx]) * scale_y, 0.0
        slp = (ys[idx + 1] - ys[idx]) / dx
        val = (ys[idx] + slp * (x * scale_x - xs[idx])) * scale_y
        return max(0.0, float(val)), max(0.0, float(slp * scale_y * scale_x))

    # Object with x/y or values
    if hasattr(curve_obj, "x") and hasattr(curve_obj, "y"):
        xs, ys = np.asarray(curve_obj.x, dtype=float), np.asarray(curve_obj.y, dtype=float)
        if len(xs) == 0:
            return default_val, default_slope
        if len(xs) == 1:
            return float(ys[0]) * scale_y, 0.0
        idx = np.searchsorted(xs, x * scale_x, side="right") - 1
        idx = max(0, min(idx, len(xs) - 2))
        dx = xs[idx + 1] - xs[idx]
        if abs(dx) < 1.0e-14:
            return float(ys[idx]) * scale_y, 0.0
        slp = (ys[idx + 1] - ys[idx]) / dx
        val = (ys[idx] + slp * (x * scale_x - xs[idx])) * scale_y
        return max(0.0, float(val)), max(0.0, float(slp * scale_y * scale_x))

    return default_val, default_slope


def law76_func_comp(
    tens_curve: Any,
    shear_curve: Any,
    nup: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesize compression curve from tension and shear data.

    Fortran port: ``starter/source/materials/mat/mat076/law76_func_comp.F90``.
    When ICAS = 2 (tension and shear provided, compression missing),
    computes equivalent compression hardening points.
    """
    scale_x_s = _SQR3 / max(1.0e-6, (1.0 + nup))

    if isinstance(tens_curve, np.ndarray) and tens_curve.ndim == 2 and tens_curve.shape[1] >= 2:
        xt, yt = tens_curve[:, 0], tens_curve[:, 1]
    elif isinstance(tens_curve, (list, tuple)) and len(tens_curve) == 2:
        xt, yt = np.asarray(tens_curve[0], dtype=float), np.asarray(tens_curve[1], dtype=float)
    elif hasattr(tens_curve, "x") and hasattr(tens_curve, "y"):
        xt, yt = np.asarray(tens_curve.x, dtype=float), np.asarray(tens_curve.y, dtype=float)
    else:
        xt = np.array([0.0, 0.05, 0.2, 0.5, 1.0])
        yt = np.array([30.0, 40.0, 55.0, 70.0, 85.0])

    if isinstance(shear_curve, np.ndarray) and shear_curve.ndim == 2 and shear_curve.shape[1] >= 2:
        xs, ys = shear_curve[:, 0], shear_curve[:, 1]
    elif isinstance(shear_curve, (list, tuple)) and len(shear_curve) == 2:
        xs, ys = np.asarray(shear_curve[0], dtype=float), np.asarray(shear_curve[1], dtype=float)
    elif hasattr(shear_curve, "x") and hasattr(shear_curve, "y"):
        xs, ys = np.asarray(shear_curve.x, dtype=float), np.asarray(shear_curve.y, dtype=float)
    else:
        xs = xt / scale_x_s
        ys = yt / _SQR3

    xs_scaled = scale_x_s * xs
    # Union of strain points
    x_comp = np.unique(np.sort(np.concatenate([xt, xs_scaled])))
    y_t = np.interp(x_comp, xt, yt)
    y_s = np.interp(x_comp, xs_scaled, ys)

    # Compute slopes and alpha_max (law76_func_comp.F90 lines 162-170)
    alphamax = 1.0
    for k in range(1, len(x_comp)):
        dx = x_comp[k] - x_comp[k - 1]
        if dx > 1.0e-12:
            st = (y_t[k] - y_t[k - 1]) / dx
            ss = (y_s[k] - y_s[k - 1]) / dx
            if st > 0.0 and ss > 0.0 and y_t[k] > 0.0:
                alpha_k = _SQR3 * 0.5 * (st / ss) * (y_s[k] / y_t[k]) ** 2
                alphamax = max(alphamax, alpha_k)

    # Compute y_comp (law76_func_comp.F90 lines 171-175)
    y_comp = np.zeros_like(x_comp)
    for k in range(len(x_comp)):
        num = _SQR3 * alphamax * y_t[k] * y_s[k]
        den = 2.0 * alphamax * y_t[k] - _SQR3 * y_s[k]
        y_comp[k] = num / max(_EM20, den)

    return x_comp, y_comp


# =============================================================================
# Material Builder: build_law76
# =============================================================================

def build_law76(rec: Any) -> Material:
    """Construct a Material entity for /MAT/LAW76 (/MAT/SAMP, /MAT/SAMP-1).

    Parameters
    ----------
    rec : GenericMaterialRecord, dict, Material, or object
        Parsed card record from CFG or deck reader.

    Returns
    -------
    Material
        Material entity configured with law=76, parameters, and derived constants.
    """
    if isinstance(rec, Material) and hasattr(rec, "params") and "law76_params" in rec.params:
        return rec

    p = getattr(rec, "params", rec) if hasattr(rec, "params") else rec
    if not isinstance(p, dict):
        p = getattr(rec, "__dict__", {})

    mat_id = int(p.get("id", getattr(rec, "id", 1)) or 1)
    title = str(p.get("title", getattr(rec, "title", "LAW76_SAMP")) or "LAW76_SAMP")

    rho0 = float(p.get("rho0", p.get("rho", p.get("MAT_RHO", getattr(rec, "rho0", 1.0)))) or 1.0)
    refer_rho = float(p.get("refer_rho", p.get("Refer_Rho", rho0)) or rho0)

    e = float(p.get("E", p.get("e", p.get("young", p.get("MAT_E", 2100.0)))) or 2100.0)
    nu = float(p.get("nu", p.get("pr", p.get("poisson", p.get("MAT_NU", 0.35)))) or 0.35)
    # Ensure physical stability
    nu = min(0.49999, max(0.0, nu))

    # Derived linear elastic constants (hm_read_mat76.F lines 204-207)
    g = 0.5 * e / (1.0 + nu)
    bulk = e / (3.0 * (1.0 - 2.0 * nu))  # C1 in Fortran
    a1 = e * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    a2 = a1 * nu / (1.0 - nu)
    a11_2d = e / (1.0 - nu * nu)
    a12_2d = nu * a11_2d

    fun_d1 = p.get("fun_d1", p.get("fct_id_t", p.get("FUN_D1", p.get("tab_id1", 0)))) or 0
    fun_d2 = p.get("fun_d2", p.get("fct_id_c", p.get("FUN_D2", p.get("tab_id2", 0)))) or 0
    fun_d3 = p.get("fun_d3", p.get("fct_id_s", p.get("FUN_D3", p.get("tab_id3", 0)))) or 0
    fun_d4 = p.get("fun_d4", p.get("fct_id_b", p.get("FUN_D4", 0))) or 0

    fscale_t = float(p.get("fscale_t", p.get("fscale11", p.get("FScale11", 1.0))) or 1.0)
    fscale_c = float(p.get("fscale_c", p.get("fscale22", p.get("FScale22", 1.0))) or 1.0)
    fscale_s = float(p.get("fscale_s", p.get("fscale33", p.get("FScale33", 1.0))) or 1.0)
    fscale_b = float(p.get("fscale_b", p.get("fscale12", p.get("FScale12", 1.0))) or 1.0)
    facx = float(p.get("facx", p.get("FACX", 1.0)) or 1.0)

    mat_nut = float(p.get("mat_nut", p.get("nu_p", p.get("nup", p.get("MAT_NUt", 0.35)))) or 0.35)
    mat_nut = max(0.0, min(0.5, mat_nut))
    fun_b5 = p.get("fun_b5", p.get("fct_id_nu", p.get("fct_idpr", p.get("FUN_B5", 0)))) or 0
    mat_pscale = float(p.get("mat_pscale", p.get("fscale_pr", p.get("MAT_PScale", 1.0))) or 1.0)

    israte = int(p.get("israte", p.get("ISRATE", 0)) or 0)
    asrate = float(p.get("asrate", p.get("fcut", p.get("MAT_asrate", 1.0e30))) or 1.0e30)

    eps_f = float(p.get("eps_f", p.get("epsilon_f", p.get("MAT_Epsilon_F", _INF))) or _INF)
    eps_r = float(p.get("eps_r", p.get("epsilon_0", p.get("Epsilon_0", 2.0 * eps_f))) or 2.0 * eps_f)
    if eps_r <= eps_f:
        eps_r = max(eps_f + 1.0e-5, 2.0 * eps_f)
    dc = float(p.get("dc", p.get("MAT_Dc", 1.0)) or 1.0)

    fun_a1 = p.get("fun_a1", p.get("fct_id_dmg", p.get("FUN_A1", 0))) or 0
    fun_a2 = p.get("fun_a2", p.get("FUN_A2", 0)) or 0
    fun_a3 = p.get("fun_a3", p.get("FUN_A3", 0)) or 0
    scale_dmg = float(p.get("scale_dmg", p.get("scale", p.get("SCALE", 1.0))) or 1.0)

    iform = int(p.get("iform", p.get("IFORM", 0)) or 0)
    iquad = int(p.get("iquad", p.get("iflag", p.get("MAT_Iflag", 1))) or 1)
    iconv = int(p.get("iconv", p.get("gflag", p.get("Gflag", 0))) or 0)

    # Resolve ICAS case (hm_read_mat76.F lines 183-196)
    has_t = (fun_d1 != 0 or "sig_t0" in p or "tens_curve" in p)
    has_c = (fun_d2 != 0 or "sig_c0" in p or "comp_curve" in p)
    has_s = (fun_d3 != 0 or "sig_s0" in p or "shear_curve" in p)

    if has_c and has_s:
        icas = -1
    elif has_c and not has_s:
        icas = 1
    elif has_s and not has_c:
        icas = 2
    else:
        icas = 0

    # User provided curves in rec
    tens_curve = p.get("tens_curve", fun_d1)
    comp_curve = p.get("comp_curve", fun_d2)
    shear_curve = p.get("shear_curve", fun_d3)
    biax_curve = p.get("biax_curve", fun_d4)
    nup_curve = p.get("nup_curve", fun_b5)
    dmg_curve = p.get("dmg_curve", fun_a1)

    # Initial fallback values
    sig_t0 = float(p.get("sig_t0", p.get("sig0", p.get("yield_stress", 30.0))) or 30.0)
    sig_c0 = float(p.get("sig_c0", sig_t0 if icas == 0 else 1.2 * sig_t0) or sig_t0)
    sig_s0 = float(p.get("sig_s0", sig_t0 / _SQR3) or sig_t0 / _SQR3)
    h_t = float(p.get("h_t", p.get("hardening", 100.0)) or 0.0)
    h_c = float(p.get("h_c", h_t) or 0.0)
    h_s = float(p.get("h_s", h_t / _SQR3) or 0.0)

    # If ICAS == 2, synthesize compression curve
    if icas == 2:
        try:
            x_syn, y_syn = law76_func_comp(tens_curve, shear_curve, mat_nut)
            comp_curve = (x_syn, y_syn)
            icas = -1
            iconv = 1
        except Exception:
            icas = 0

    law76_p = Law76Params(
        id=mat_id,
        title=title,
        law=76,
        law_name="LAW76",
        rho0=rho0,
        refer_rho=refer_rho,
        e=e,
        nu=nu,
        g=g,
        bulk=bulk,
        a1=a1,
        a2=a2,
        a11_2d=a11_2d,
        a12_2d=a12_2d,
        fun_d1=fun_d1,
        fun_d2=fun_d2,
        fun_d3=fun_d3,
        fun_d4=fun_d4,
        fscale_t=fscale_t,
        fscale_c=fscale_c,
        fscale_s=fscale_s,
        fscale_b=fscale_b,
        facx=facx,
        mat_nut=mat_nut,
        fun_b5=fun_b5,
        mat_pscale=mat_pscale,
        israte=israte,
        asrate=asrate,
        eps_f=eps_f,
        eps_r=eps_r,
        dc=dc,
        fun_a1=fun_a1,
        fun_a2=fun_a2,
        fun_a3=fun_a3,
        scale_dmg=scale_dmg,
        iform=iform,
        iquad=iquad,
        iconv=iconv,
        icas=icas,
        tens_curve=tens_curve,
        comp_curve=comp_curve,
        shear_curve=shear_curve,
        biax_curve=biax_curve,
        nup_curve=nup_curve,
        dmg_curve=dmg_curve,
        sig_t0=sig_t0,
        sig_c0=sig_c0,
        sig_s0=sig_s0,
        h_t=h_t,
        h_c=h_c,
        h_s=h_s,
    )

    out_params = dict(p)
    out_params.update({
        "rho": rho0, "rho0": rho0, "refer_rho": refer_rho,
        "e": e, "E": e, "young": e, "nu": nu, "pr": nu,
        "g": g, "bulk": bulk, "K": bulk,
        "fun_d1": fun_d1, "fun_d2": fun_d2, "fun_d3": fun_d3, "fun_d4": fun_d4,
        "fscale_t": fscale_t, "fscale_c": fscale_c, "fscale_s": fscale_s, "fscale_b": fscale_b,
        "mat_nut": mat_nut, "nu_p": mat_nut, "nup": mat_nut,
        "eps_f": eps_f, "eps_r": eps_r, "dc": dc,
        "iform": iform, "iquad": iquad, "iconv": iconv, "icas": icas,
        "law76_params": law76_p,
        "law": 76, "law_name": "LAW76",
    })

    mat = Material(id=mat_id, law=76, law_name="LAW76", rho0=rho0, title=title, params=out_params)
    return mat


def _get_law76_params(mat: Any) -> Law76Params:
    """Retrieve or build Law76Params from Material or dict."""
    if isinstance(mat, Law76Params):
        return mat
    if hasattr(mat, "params") and isinstance(mat.params, dict) and "law76_params" in mat.params:
        return mat.params["law76_params"]
    built = build_law76(mat)
    return built.params["law76_params"]


# =============================================================================
# Helper: Yield Surface Parameter Computation
# =============================================================================

def _compute_yield_coeffs(
    p_lp: Law76Params,
    sig_t: float,
    sig_c: float,
    sig_s: float,
) -> Tuple[float, float, float, float]:
    """Compute yield surface parameters A0, A1, A2, and convexity-adjusted sig_s.

    Fortran reference:
      sigeps76.F lines 241-256 (no_asso_plas76.F lines 241-256)
    """
    s_t = max(1.0e-12, sig_t)
    s_c = max(1.0e-12, sig_c)
    s_s = max(1.0e-12, sig_s)

    if p_lp.icas == 0:
        s_c = s_t
        s_s = s_t / _SQR3
    elif p_lp.icas == 1:
        if p_lp.iquad == 1:
            s_s = math.sqrt(s_c * s_t / 3.0)
        else:
            s_s = 2.0 * s_t * s_c / (_SQR3 * (s_t + s_c))

    # Ensured convexity (ICONV == 1)
    if p_lp.iconv == 1:
        if p_lp.iquad == 1:
            s_s_min = _SFAC * math.sqrt(s_c * s_t / 3.0)
            if s_s < s_s_min:
                s_s = s_s_min
        else:
            aa = 1.0 / ((s_t + s_c) * _SQR3)
            s_s_min = _SFAC * 2.0 * s_t * s_c * aa
            if s_s < s_s_min:
                s_s = s_s_min

    if p_lp.iquad == 1:
        aa = 1.0 / (s_c * s_t)
        a0 = 3.0 * (s_s ** 2)
        a1 = 9.0 * (s_s ** 2) * (s_c - s_t) * aa
        a2 = 9.0 * (s_c * s_t - 3.0 * (s_s ** 2)) * aa
    else:
        a0 = s_s * _SQR3
        a1 = 3.0 * (((s_t - s_c) / (s_t + s_c)) - a0 * ((s_t - s_c) / (s_t * s_c)))
        a2 = 18.0 * ((1.0 / (s_t + s_c)) - a0 / (2.0 * s_t * s_c))

    return a0, a1, a2, s_s


# =============================================================================
# 3D Continuum Solid Formulation (sigeps76.F, no_asso_plas76.F, asso_plas76.F)
# =============================================================================

def solid_step(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Fortran-faithful stress update for LAW76 3D solid elements (sigeps76.F).

    Parameters
    ----------
    mat : Material or Law76Params
        Material instance.
    sig : ndarray, shape (6,) or (n, 6)
        Cauchy stress [xx, yy, zz, xy, yz, zx].
    deps : ndarray, shape (6,) or (n, 6)
        Strain increment [xx, yy, zz, xy, yz, zx].
    epsp : ndarray, optional
        Accumulated equivalent plastic strain.
    dt : float
        Current time step.
    extra : dict, optional
        Persistent element state dictionary containing 'uvar', 'dmg', 'off'.

    Returns
    -------
    (sign, epsp, soundsp) : Tuple[np.ndarray, np.ndarray, np.ndarray]
        sign : Updated Cauchy stress.
        epsp : Updated equivalent plastic strain.
        soundsp : Exact dilatational acoustic wave speed.
    """
    p_lp = _get_law76_params(mat)
    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig.reshape(1, -1)
        deps = deps.reshape(1, -1)

    n = sig.shape[0]
    if n == 0:
        c_speed = math.sqrt((p_lp.bulk + 4.0 / 3.0 * p_lp.g) / max(_EM20, p_lp.rho0))
        return (sig[0] if is_1d else sig), epsp, (np.array(c_speed) if is_1d else np.full(n, c_speed))

    if extra is None:
        extra = {}

    # State arrays: uvar (n, 7), dmg (n,), off (n,)
    if "uvar" not in extra or extra["uvar"] is None:
        uvar = np.zeros((n, 7), dtype=float)
        uvar[:, 6] = p_lp.mat_nut
        extra["uvar"] = uvar
    else:
        uvar = np.asarray(extra["uvar"], dtype=float)
        if uvar.ndim == 1:
            uvar = uvar.reshape(n, -1)
        if uvar.shape != (n, 7):
            padded = np.zeros((n, 7), dtype=float)
            padded[:, 6] = p_lp.mat_nut
            r = min(n, uvar.shape[0])
            c_idx = min(7, uvar.shape[1])
            padded[:r, :c_idx] = uvar[:r, :c_idx]
            uvar = padded
            extra["uvar"] = uvar

    if "dmg" not in extra or extra["dmg"] is None:
        dmg = np.zeros(n, dtype=float)
        extra["dmg"] = dmg
    else:
        dmg = np.asarray(extra["dmg"], dtype=float)
        if dmg.ndim == 0:
            dmg = np.full(n, float(dmg))
        elif dmg.shape != (n,):
            dmg = np.resize(dmg, n)
        extra["dmg"] = dmg

    if "off" not in extra or extra["off"] is None:
        off = np.ones(n, dtype=float)
        extra["off"] = off
    else:
        off = np.asarray(extra["off"], dtype=float)
        if off.ndim == 0:
            off = np.full(n, float(off))
        elif off.shape != (n,):
            off = np.resize(off, n)
        extra["off"] = off

    # Handle element deletion / damage status
    for i in range(n):
        if off[i] < 0.1:
            off[i] = 0.0
        elif off[i] < 1.0:
            off[i] *= 0.8

    # Recover initial undamaged stresses: sig0 = sig / max(1 - D, EM20) (sigeps76.F lines 108-115)
    sig0 = np.empty_like(sig)
    for i in range(n):
        factor = max(1.0 - dmg[i], _EM20)
        sig0[i] = sig[i] / factor

    sign = np.empty_like(sig)
    soundsp = np.empty(n, dtype=float)
    c_bulk_g = math.sqrt((p_lp.bulk + 4.0 / 3.0 * p_lp.g) / max(_EM20, p_lp.rho0))
    soundsp.fill(c_bulk_g)

    asrate = min(1.0, 2.0 * math.pi * p_lp.asrate * dt) if dt > 0.0 else 1.0

    if epsp is None:
        epsp = np.zeros(n, dtype=float)
    else:
        epsp = np.asarray(epsp, dtype=float).copy()
        if epsp.ndim == 0:
            epsp = np.full(n, float(epsp))

    # Loop over each Gauss / integration point
    for i in range(n):
        if off[i] == 0.0 or dmg[i] >= 1.0:
            sign[i] = 0.0
            continue

        # 1. Trial stress calculation (no_asso_plas76.F lines 198-207)
        dxx, dyy, dzz = deps[i, 0], deps[i, 1], deps[i, 2]
        dxy, dyz, dzx = deps[i, 3], deps[i, 4], deps[i, 5]

        sxx_tr = sig0[i, 0] + (p_lp.a1 * dxx + p_lp.a2 * (dyy + dzz))
        syy_tr = sig0[i, 1] + (p_lp.a1 * dyy + p_lp.a2 * (dxx + dzz))
        szz_tr = sig0[i, 2] + (p_lp.a1 * dzz + p_lp.a2 * (dxx + dyy))
        sxy_tr = sig0[i, 3] + p_lp.g * dxy
        syz_tr = sig0[i, 4] + p_lp.g * dyz
        szx_tr = sig0[i, 5] + p_lp.g * dzx

        # Hydrostatic pressure (p = -1/3 tr(sigma)) and deviatoric stress
        p = - (sxx_tr + syy_tr + szz_tr) / 3.0
        sdxx = sxx_tr + p
        sdyy = syy_tr + p
        sdzz = szz_tr + p
        sdxy = sxy_tr
        sdyz = syz_tr
        sdzx = szx_tr

        svm2 = 0.5 * (sdxx**2 + sdyy**2 + sdzz**2) + (sdxy**2 + sdyz**2 + sdzx**2)
        svm = math.sqrt(max(0.0, 3.0 * svm2))

        # Current internal state variables
        plat = uvar[i, 0] if uvar[i, 0] > 0.0 else epsp[i]
        plac = uvar[i, 1] if uvar[i, 1] > 0.0 else epsp[i]
        plas = uvar[i, 2] if uvar[i, 2] > 0.0 else (epsp[i] * (_SQR3 / 2.0))
        epspt = uvar[i, 3]
        epspc = uvar[i, 4]
        epsps = uvar[i, 5]
        nu_p_curr = uvar[i, 6] if uvar[i, 6] > 0.0 else p_lp.mat_nut

        # Plastic potential parameter alpha
        alpha = 4.5 * ((1.0 - 2.0 * nu_p_curr) / max(0.1, 1.0 + nu_p_curr))

        # Evaluate hardening curves
        sig_t, dsig_t = _eval_curve_or_val(
            p_lp.tens_curve, plat, epspt * p_lp.facx, 1.0, p_lp.fscale_t, p_lp.sig_t0, p_lp.h_t
        )
        sig_c, dsig_c = _eval_curve_or_val(
            p_lp.comp_curve, plac, epspc * p_lp.facx, 1.0, p_lp.fscale_c, p_lp.sig_c0, p_lp.h_c
        )
        sig_s, dsig_s = _eval_curve_or_val(
            p_lp.shear_curve, plas, epsps * p_lp.facx, 1.0, p_lp.fscale_s, p_lp.sig_s0, p_lp.h_s
        )

        a0, a1_y, a2_y, sig_s_adj = _compute_yield_coeffs(p_lp, sig_t, sig_c, sig_s)

        # Evaluate yield function
        if p_lp.iquad == 1:
            phi = (svm**2) - a0 - a1_y * p - a2_y * (p**2)
        else:
            phi = svm - a0 - a1_y * p - a2_y * (p**2)

        dpla = 0.0
        dplat = 0.0
        dplac = 0.0
        dplas = 0.0

        # Plastic return mapping (cutting-plane iterations)
        if phi > 0.0 and off[i] == 1.0 and dmg[i] < 1.0:
            for _ in range(8):
                if p_lp.iform == 1:
                    # Associated flow
                    cb = a1_y + 2.0 * a2_y * p
                    norm_denom = max(_EM20, math.sqrt(6.0 * (svm**2) + (cb**2) / 3.0))
                    if p_lp.iquad == 1:
                        normxx = 3.0 * sdxx + cb / 3.0
                        normyy = 3.0 * sdyy + cb / 3.0
                        normzz = 3.0 * sdzz + cb / 3.0
                        normxy = 3.0 * sdxy
                        normyz = 3.0 * sdyz
                        normzx = 3.0 * sdzx
                    else:
                        normxx = 1.5 * sdxx / max(_EM20, svm) + cb / 3.0
                        normyy = 1.5 * sdyy / max(_EM20, svm) + cb / 3.0
                        normzz = 1.5 * sdzz / max(_EM20, svm) + cb / 3.0
                        normxy = 1.5 * sdxy / max(_EM20, svm)
                        normyz = 1.5 * sdyz / max(_EM20, svm)
                        normzx = 1.5 * sdzx / max(_EM20, svm)

                    normxx_n = normxx / norm_denom
                    normyy_n = normyy / norm_denom
                    normzz_n = normzz / norm_denom
                    normxy_n = normxy / norm_denom
                    normyz_n = normyz / norm_denom
                    normzx_n = normzx / norm_denom
                    psi = norm_denom
                else:
                    # Non-associated flow (no_asso_plas76.F lines 297-323)
                    psi = math.sqrt(max(_EM20, (svm**2) + alpha * (p**2)))
                    normxx_n = 1.5 * sdxx / psi - (alpha * p / (3.0 * psi))
                    normyy_n = 1.5 * sdyy / psi - (alpha * p / (3.0 * psi))
                    normzz_n = 1.5 * sdzz / psi - (alpha * p / (3.0 * psi))
                    normxy_n = 1.5 * sdxy / psi
                    normyz_n = 1.5 * sdyz / psi
                    normzx_n = 1.5 * sdzx / psi

                    if p_lp.iquad == 1:
                        normxx = 3.0 * sdxx + (a1_y + 2.0 * a2_y * p) / 3.0
                        normyy = 3.0 * sdyy + (a1_y + 2.0 * a2_y * p) / 3.0
                        normzz = 3.0 * sdzz + (a1_y + 2.0 * a2_y * p) / 3.0
                        normxy = 3.0 * sdxy
                        normyz = 3.0 * sdyz
                        normzx = 3.0 * sdzx
                    else:
                        normxx = 1.5 * (sdxx / max(_EM20, svm)) + (a1_y + 2.0 * a2_y * p) / 3.0
                        normyy = 1.5 * (sdyy / max(_EM20, svm)) + (a1_y + 2.0 * a2_y * p) / 3.0
                        normzz = 1.5 * (sdzz / max(_EM20, svm)) + (a1_y + 2.0 * a2_y * p) / 3.0
                        normxy = 1.5 * (sdxy / max(_EM20, svm))
                        normyz = 1.5 * (sdyz / max(_EM20, svm))
                        normzx = 1.5 * (sdzx / max(_EM20, svm))

                # df/dsigma : C : dg/dsigma
                dfdsig2 = (
                    normxx * (p_lp.a1 * normxx_n + p_lp.a2 * (normyy_n + normzz_n))
                    + normyy * (p_lp.a1 * normyy_n + p_lp.a2 * (normxx_n + normzz_n))
                    + normzz * (p_lp.a1 * normzz_n + p_lp.a2 * (normxx_n + normyy_n))
                    + 4.0 * p_lp.g * (normxy * normxy_n + normyz * normyz_n + normzx * normzx_n)
                )

                # Plastic strain rates and hardening derivatives
                yld_ratio = (svm / psi) if psi > _EM20 else 1.0
                bb = 1.5 / (1.0 + nu_p_curr)
                dsigt_dlam = dsig_t * yld_ratio * bb
                dsigc_dlam = dsig_c * yld_ratio * bb
                dsigs_dlam = dsig_s * (_SQR3 / 2.0) * yld_ratio

                if p_lp.icas == 0:
                    dsigc_dlam = dsigt_dlam
                    dsigs_dlam = (1.0 / _SQR3) * dsigt_dlam
                elif p_lp.icas == 1:
                    if p_lp.iquad == 1:
                        dsigs_dlam = (1.0 / _SQR3) * (0.5 / math.sqrt(max(_EM20, sig_t * sig_c))) * (
                            dsigc_dlam * sig_t + sig_c * dsigt_dlam
                        )
                    else:
                        aa_sc = 1.0 / ((sig_t + sig_c) * _SQR3)
                        dsigs_dlam = 2.0 * (dsigt_dlam * sig_c + dsigc_dlam * sig_t) * aa_sc \
                                     - 2.0 * _SQR3 * sig_c * sig_t * (dsigt_dlam + dsigc_dlam) * (aa_sc ** 2)

                # Derivatives of A0, A1, A2
                if p_lp.iquad == 1:
                    da0_dsigs = 6.0 * sig_s_adj
                    cc = sig_s_adj / (sig_c * sig_t)
                    da1_dsigs = 18.0 * (sig_c - sig_t) * cc
                    da1_dsigc = 9.0 * (sig_s_adj / sig_c) ** 2
                    da1_dsigt = -9.0 * (sig_s_adj / sig_t) ** 2
                    da2_dsigs = -54.0 * cc
                    da2_dsigc = 27.0 * cc * (sig_s_adj / sig_c)
                    da2_dsigt = 27.0 * cc * (sig_s_adj / sig_t)
                else:
                    da0_dsigs = _SQR3
                    da1_dsigs = -3.0 * _SQR3 * (sig_t - sig_c) / (sig_t * sig_c)
                    da1_dsigc = 3.0 * ((sig_s_adj * _SQR3 / (sig_c ** 2)) - 2.0 * sig_t / ((sig_t + sig_c) ** 2))
                    da1_dsigt = 3.0 * (2.0 * sig_c / ((sig_t + sig_c) ** 2) - (sig_s_adj * _SQR3 / (sig_t ** 2)))
                    da2_dsigs = -9.0 * _SQR3 / (sig_t * sig_c)
                    da2_dsigc = 18.0 * ((sig_s_adj * _SQR3 / (2.0 * sig_t * (sig_c ** 2))) - 1.0 / ((sig_t + sig_c) ** 2))
                    da2_dsigt = 18.0 * ((sig_s_adj * _SQR3 / (2.0 * sig_c * (sig_t ** 2))) - 1.0 / ((sig_t + sig_c) ** 2))

                da0_dlam = da0_dsigs * dsigs_dlam
                da1_dlam = da1_dsigs * dsigs_dlam + da1_dsigt * dsigt_dlam + da1_dsigc * dsigc_dlam
                da2_dlam = da2_dsigs * dsigs_dlam + da2_dsigt * dsigt_dlam + da2_dsigc * dsigc_dlam

                dphi_dlam = - dfdsig2 - da0_dlam - p * da1_dlam - (p ** 2) * da2_dlam
                if abs(dphi_dlam) < _EM20:
                    dphi_dlam = -_EM20 if dphi_dlam <= 0.0 else _EM20

                dlam = - phi / dphi_dlam
                if dlam < 0.0:
                    dlam = 0.0

                # Plastic strain tensor update
                dpxx = dlam * normxx_n
                dpyy = dlam * normyy_n
                dpzz = dlam * normzz_n
                dpxy = dlam * normxy_n
                dpyz = dlam * normyz_n
                dpzx = dlam * normzx_n

                # Elasto-plastic stress update
                sxx_tr -= (p_lp.a1 * dpxx + p_lp.a2 * (dpyy + dpzz))
                syy_tr -= (p_lp.a1 * dpyy + p_lp.a2 * (dpxx + dpzz))
                szz_tr -= (p_lp.a1 * dpzz + p_lp.a2 * (dpxx + dpyy))
                sxy_tr -= 2.0 * p_lp.g * dpxy
                syz_tr -= 2.0 * p_lp.g * dpyz
                szx_tr -= 2.0 * p_lp.g * dpzx

                # Accumulate plastic strains
                inc_t = dlam * yld_ratio * bb
                inc_s = (_SQR3 / 2.0) * yld_ratio * dlam
                inc_eq = yld_ratio * dlam

                plat = max(0.0, plat + inc_t)
                plac = plat
                plas = max(0.0, plas + inc_s)
                dplat += inc_t
                dplac += inc_t
                dplas += inc_s
                dpla += inc_eq

                # Update pressure and deviator
                p = - (sxx_tr + syy_tr + szz_tr) / 3.0
                sdxx = sxx_tr + p
                sdyy = syy_tr + p
                sdzz = szz_tr + p
                sdxy = sxy_tr
                sdyz = syz_tr
                sdzx = szx_tr
                svm2 = 0.5 * (sdxx**2 + sdyy**2 + sdzz**2) + (sdxy**2 + sdyz**2 + sdzx**2)
                svm = math.sqrt(max(0.0, 3.0 * svm2))

                # Re-evaluate hardening curves
                sig_t, dsig_t = _eval_curve_or_val(
                    p_lp.tens_curve, plat, epspt * p_lp.facx, 1.0, p_lp.fscale_t, p_lp.sig_t0, p_lp.h_t
                )
                sig_c, dsig_c = _eval_curve_or_val(
                    p_lp.comp_curve, plac, epspc * p_lp.facx, 1.0, p_lp.fscale_c, p_lp.sig_c0, p_lp.h_c
                )
                sig_s, dsig_s = _eval_curve_or_val(
                    p_lp.shear_curve, plas, epsps * p_lp.facx, 1.0, p_lp.fscale_s, p_lp.sig_s0, p_lp.h_s
                )

                a0, a1_y, a2_y, sig_s_adj = _compute_yield_coeffs(p_lp, sig_t, sig_c, sig_s)

                if p_lp.iquad == 1:
                    phi = (svm**2) - a0 - a1_y * p - a2_y * (p**2)
                    tol = 1.0e-6 * max(1.0, a0)
                else:
                    phi = svm - a0 - a1_y * p - a2_y * (p**2)
                    tol = 1.0e-6 * max(1.0, a0)

                if abs(phi) <= tol or phi <= 0.0:
                    break

        # Update plastic Poisson ratio if function provided
        if p_lp.fun_b5 != 0 or p_lp.nup_curve is not None:
            nup_val, _ = _eval_curve_or_val(
                p_lp.nup_curve or p_lp.fun_b5, epsp[i] + dpla, scale_y=p_lp.mat_pscale, default_val=p_lp.mat_nut
            )
            nu_p_curr = max(0.0, min(0.5, nup_val))

        # Store internal state variables (sigeps76.F lines 561-574)
        uvar[i, 0] = plat
        uvar[i, 1] = plac
        uvar[i, 2] = plas
        if dt > 0.0:
            dt_inv = 1.0 / dt
            asrate = min(1.0, 2.0 * math.pi * p_lp.asrate * dt)
            uvar[i, 3] = asrate * (dplat * dt_inv) + (1.0 - asrate) * epspt
            uvar[i, 4] = asrate * (dplac * dt_inv) + (1.0 - asrate) * epspc
            uvar[i, 5] = asrate * (dplas * dt_inv) + (1.0 - asrate) * epsps
        else:
            uvar[i, 3] = 0.0
            uvar[i, 4] = 0.0
            uvar[i, 5] = 0.0
        uvar[i, 6] = nu_p_curr

        epsp[i] += dpla

        # 2. Damage accumulation (sigeps76.F lines 150-182)
        if p_lp.fun_a1 != 0 or p_lp.dmg_curve is not None:
            d_val, _ = _eval_curve_or_val(
                p_lp.dmg_curve or p_lp.fun_a1, epsp[i], scale_y=p_lp.scale_dmg, default_val=0.0
            )
            dmg[i] = max(dmg[i], min(1.0, d_val))
        elif epsp[i] >= p_lp.eps_f:
            d_val = (epsp[i] - p_lp.eps_f) / max(_EM20, p_lp.eps_r - p_lp.eps_f)
            dmg[i] = max(dmg[i], min(1.0, d_val))

        if dmg[i] >= 1.0:
            dmg[i] = 1.0
            if off[i] == 1.0:
                off[i] = 0.8

        # 3. Damaged stresses: sign = sig_undamaged * (1 - D)
        degrade = max(0.0, 1.0 - dmg[i]) * off[i]
        sign[i, 0] = sxx_tr * degrade
        sign[i, 1] = syy_tr * degrade
        sign[i, 2] = szz_tr * degrade
        sign[i, 3] = sxy_tr * degrade
        sign[i, 4] = syz_tr * degrade
        sign[i, 5] = szx_tr * degrade

    res_sig = sign[0] if is_1d else sign
    res_epsp = epsp[0] if is_1d else epsp
    res_soundsp = soundsp[0] if is_1d else soundsp
    return res_sig, res_epsp, res_soundsp


# =============================================================================
# 2D Plane-Stress Shell Formulation (sigeps76c.F, no_asso_qplas76c.F)
# =============================================================================

def shell_step(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Plane-stress shell update for LAW76 (sigeps76c.F).

    Enforces sigma_{zz} = 0 with through-thickness plastic strain increment
    and thickness updates.

    Parameters
    ----------
    mat : Material or Law76Params
        Material instance.
    sig : ndarray, shape (3,) or (5,) or (n, 3) or (n, 5)
        In-plane stress components [xx, yy, xy] (and optional [yz, zx]).
    deps : ndarray, shape (3,) or (5,) or (n, 3) or (n, 5)
        In-plane strain increments with engineering shear [xx, yy, xy] (and optional [yz, zx]).
    epsp : ndarray, optional
        Accumulated equivalent plastic strain.
    dt : float
        Time step.
    extra : dict, optional
        Persistent element extra state dictionary.

    Returns
    -------
    (sign, epsp) : Tuple[np.ndarray, np.ndarray]
        Updated stress tensor and plastic strain.
    """
    p_lp = _get_law76_params(mat)
    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig.reshape(1, -1)
        deps = deps.reshape(1, -1)

    n = sig.shape[0]
    n_comp = sig.shape[1]
    if n == 0:
        return (sig[0] if is_1d else sig), epsp

    if extra is None:
        extra = {}

    if "uvar" not in extra or extra["uvar"] is None:
        uvar = np.zeros((n, 7), dtype=float)
        uvar[:, 6] = p_lp.mat_nut
        extra["uvar"] = uvar
    else:
        uvar = np.asarray(extra["uvar"], dtype=float)
        if uvar.ndim == 1:
            uvar = uvar.reshape(n, -1)
        if uvar.shape != (n, 7):
            padded = np.zeros((n, 7), dtype=float)
            padded[:, 6] = p_lp.mat_nut
            r = min(n, uvar.shape[0])
            c_idx = min(7, uvar.shape[1])
            padded[:r, :c_idx] = uvar[:r, :c_idx]
            uvar = padded
            extra["uvar"] = uvar

    if "dmg" not in extra or extra["dmg"] is None:
        dmg = np.zeros(n, dtype=float)
        extra["dmg"] = dmg
    else:
        dmg = np.asarray(extra["dmg"], dtype=float)
        if dmg.ndim == 0:
            dmg = np.full(n, float(dmg))
        elif dmg.shape != (n,):
            dmg = np.resize(dmg, n)
        extra["dmg"] = dmg

    if "off" not in extra or extra["off"] is None:
        off = np.ones(n, dtype=float)
        extra["off"] = off
    else:
        off = np.asarray(extra["off"], dtype=float)
        if off.ndim == 0:
            off = np.full(n, float(off))
        elif off.shape != (n,):
            off = np.resize(off, n)
        extra["off"] = off

    # Thickness tracking in extra
    if "thk" not in extra or extra["thk"] is None:
        thk = np.ones(n, dtype=float)
        extra["thk"] = thk
    else:
        thk = np.asarray(extra["thk"], dtype=float)
        if thk.ndim == 0:
            thk = np.full(n, float(thk))
        extra["thk"] = thk

    if "thk0" not in extra or extra["thk0"] is None:
        extra["thk0"] = thk.copy()
    thk0 = extra["thk0"]

    if "dezz" not in extra or extra["dezz"] is None:
        dezz = np.zeros(n, dtype=float)
        extra["dezz"] = dezz
    else:
        dezz = np.asarray(extra["dezz"], dtype=float)
        if dezz.ndim == 0:
            dezz = np.zeros(n, dtype=float)
        extra["dezz"] = dezz

    if epsp is None:
        epsp = np.zeros(n, dtype=float)
    else:
        epsp = np.asarray(epsp, dtype=float).copy()
        if epsp.ndim == 0:
            epsp = np.full(n, float(epsp))

    for i in range(n):
        if off[i] < 0.1:
            off[i] = 0.0
        elif off[i] < 1.0:
            off[i] *= 0.8

    sig0 = np.empty_like(sig)
    for i in range(n):
        factor = max(1.0 - dmg[i], _EM20)
        sig0[i] = sig[i] / factor

    sign = np.empty_like(sig)
    nu1 = p_lp.nu / max(1.0e-6, 1.0 - p_lp.nu)  # aa2 / aa1 in plane stress
    asrate = min(1.0, 2.0 * math.pi * p_lp.asrate * dt) if dt > 0.0 else 1.0

    for i in range(n):
        if off[i] == 0.0 or dmg[i] >= 1.0:
            sign[i] = 0.0
            continue

        dxx, dyy = deps[i, 0], deps[i, 1]
        dxy = deps[i, 2]
        dyz = deps[i, 3] if n_comp >= 5 else 0.0
        dzx = deps[i, 4] if n_comp >= 5 else 0.0

        # In-plane elastic trial stress (no_asso_qplas76c.F lines 215-220)
        sxx_tr = sig0[i, 0] + p_lp.a11_2d * dxx + p_lp.a12_2d * dyy
        syy_tr = sig0[i, 1] + p_lp.a11_2d * dyy + p_lp.a12_2d * dxx
        sxy_tr = sig0[i, 2] + p_lp.g * dxy

        # Transverse shear (linear elastic, uncoupled from plane-stress yield)
        syz_tr = (sig0[i, 3] + p_lp.g * dyz) if n_comp >= 5 else 0.0
        szx_tr = (sig0[i, 4] + p_lp.g * dzx) if n_comp >= 5 else 0.0

        # Plane-stress pressure (sigma_zz = 0)
        p = - (sxx_tr + syy_tr) / 3.0
        sxx_dev = sxx_tr + p
        syy_dev = syy_tr + p
        szz_dev = p  # 0 + p = p

        # Elastic through-thickness strain increment (sigeps76c.F line 225)
        dezz_i = - nu1 * (dxx + dyy)

        svm2 = 1.5 * (sxx_dev**2 + syy_dev**2 + szz_dev**2) + 3.0 * (sxy_tr**2)
        svm = math.sqrt(max(0.0, svm2))

        plat = uvar[i, 0] if uvar[i, 0] > 0.0 else epsp[i]
        plac = uvar[i, 1] if uvar[i, 1] > 0.0 else epsp[i]
        plas = uvar[i, 2] if uvar[i, 2] > 0.0 else (epsp[i] * (_SQR3 / 2.0))
        epspt = uvar[i, 3]
        epspc = uvar[i, 4]
        epsps = uvar[i, 5]
        nu_p_curr = uvar[i, 6] if uvar[i, 6] > 0.0 else p_lp.mat_nut

        alpha = 4.5 * ((1.0 - 2.0 * nu_p_curr) / max(0.1, 1.0 + nu_p_curr))

        sig_t, dsig_t = _eval_curve_or_val(
            p_lp.tens_curve, plat, epspt * p_lp.facx, 1.0, p_lp.fscale_t, p_lp.sig_t0, p_lp.h_t
        )
        sig_c, dsig_c = _eval_curve_or_val(
            p_lp.comp_curve, plac, epspc * p_lp.facx, 1.0, p_lp.fscale_c, p_lp.sig_c0, p_lp.h_c
        )
        sig_s, dsig_s = _eval_curve_or_val(
            p_lp.shear_curve, plas, epsps * p_lp.facx, 1.0, p_lp.fscale_s, p_lp.sig_s0, p_lp.h_s
        )

        a0, a1_y, a2_y, sig_s_adj = _compute_yield_coeffs(p_lp, sig_t, sig_c, sig_s)

        if p_lp.iquad == 1:
            phi = (svm**2) - a0 - a1_y * p - a2_y * (p**2)
        else:
            phi = svm - a0 - a1_y * p - a2_y * (p**2)

        dpla = 0.0
        dplat = 0.0
        dplac = 0.0
        dplas = 0.0

        if phi > 0.0 and off[i] == 1.0 and dmg[i] < 1.0:
            for _ in range(8):
                if p_lp.iform == 1:
                    cb = a1_y + 2.0 * a2_y * p
                    norm_denom = max(_EM20, math.sqrt(6.0 * (svm**2) + (cb**2) / 3.0))
                    if p_lp.iquad == 1:
                        normxx = 3.0 * sxx_dev + cb / 3.0
                        normyy = 3.0 * syy_dev + cb / 3.0
                        normzz = 3.0 * szz_dev + cb / 3.0
                        normxy = 6.0 * sxy_tr
                    else:
                        normxx = 1.5 * sxx_dev / max(_EM20, svm) + cb / 3.0
                        normyy = 1.5 * syy_dev / max(_EM20, svm) + cb / 3.0
                        normzz = 1.5 * szz_dev / max(_EM20, svm) + cb / 3.0
                        normxy = 3.0 * sxy_tr / max(_EM20, svm)
                    normxx_n = normxx / norm_denom
                    normyy_n = normyy / norm_denom
                    normzz_n = normzz / norm_denom
                    normxy_n = normxy / norm_denom
                    gf = norm_denom
                else:
                    # Non-associated plastic potential (no_asso_qplas76c.F lines 270-288)
                    gf = math.sqrt(max(_EM20, (svm**2) + alpha * (p**2)))
                    normxx_n = (1.5 * sxx_dev - alpha * p / 3.0) / gf
                    normyy_n = (1.5 * syy_dev - alpha * p / 3.0) / gf
                    normzz_n = (1.5 * szz_dev - alpha * p / 3.0) / gf
                    normxy_n = 3.0 * sxy_tr / gf

                    cb = a1_y + 2.0 * a2_y * p
                    if p_lp.iquad == 1:
                        normxx = 3.0 * sxx_dev + cb / 3.0
                        normyy = 3.0 * syy_dev + cb / 3.0
                        normxy = 6.0 * sxy_tr
                    else:
                        normxx = 1.5 * sxx_dev / max(_EM20, svm) + cb / 3.0
                        normyy = 1.5 * syy_dev / max(_EM20, svm) + cb / 3.0
                        normxy = 3.0 * sxy_tr / max(_EM20, svm)

                dfdsigdlam = (
                    normxx * (p_lp.a11_2d * normxx_n + p_lp.a12_2d * normyy_n)
                    + normyy * (p_lp.a11_2d * normyy_n + p_lp.a12_2d * normxx_n)
                    + normxy * normxy_n * p_lp.g
                )

                yld_norm = (svm / gf) if gf > _EM20 else 1.0
                bb = 1.5 / (1.0 + nu_p_curr)
                dsigt_dlam = dsig_t * yld_norm * bb
                dsigc_dlam = dsig_c * yld_norm * bb
                dsigs_dlam = dsig_s * yld_norm * (_SQR3 / 2.0)

                if p_lp.icas == 0:
                    dsigc_dlam = dsigt_dlam
                    dsigs_dlam = (1.0 / _SQR3) * dsigt_dlam
                elif p_lp.icas == 1:
                    if p_lp.iquad == 1:
                        dsigs_dlam = (1.0 / _SQR3) * (0.5 / math.sqrt(max(_EM20, sig_t * sig_c))) * (
                            dsigc_dlam * sig_t + sig_c * dsigt_dlam
                        )
                    else:
                        aa_sc = 1.0 / ((sig_t + sig_c) * _SQR3)
                        dsigs_dlam = 2.0 * (dsigt_dlam * sig_c + dsigc_dlam * sig_t) * aa_sc \
                                     - 2.0 * _SQR3 * sig_c * sig_t * (dsigt_dlam + dsigc_dlam) * (aa_sc ** 2)

                if p_lp.iquad == 1:
                    da0_dsigs = 6.0 * sig_s_adj
                    cc = sig_s_adj / (sig_c * sig_t)
                    da1_dsigs = 18.0 * (sig_c - sig_t) * cc
                    da1_dsigc = 9.0 * (sig_s_adj / sig_c) ** 2
                    da1_dsigt = -9.0 * (sig_s_adj / sig_t) ** 2
                    da2_dsigs = -54.0 * cc
                    da2_dsigc = 27.0 * cc * (sig_s_adj / sig_c)
                    da2_dsigt = 27.0 * cc * (sig_s_adj / sig_t)
                else:
                    da0_dsigs = _SQR3
                    da1_dsigs = -3.0 * _SQR3 * (sig_t - sig_c) / (sig_t * sig_c)
                    da1_dsigc = 3.0 * ((sig_s_adj * _SQR3 / (sig_c ** 2)) - 2.0 * sig_t / ((sig_t + sig_c) ** 2))
                    da1_dsigt = 3.0 * (2.0 * sig_c / ((sig_t + sig_c) ** 2) - (sig_s_adj * _SQR3 / (sig_t ** 2)))
                    da2_dsigs = -9.0 * _SQR3 / (sig_t * sig_c)
                    da2_dsigc = 18.0 * ((sig_s_adj * _SQR3 / (2.0 * sig_t * (sig_c ** 2))) - 1.0 / ((sig_t + sig_c) ** 2))
                    da2_dsigt = 18.0 * ((sig_s_adj * _SQR3 / (2.0 * sig_c * (sig_t ** 2))) - 1.0 / ((sig_t + sig_c) ** 2))

                da0 = da0_dsigs * dsigs_dlam
                da1 = da1_dsigs * dsigs_dlam + da1_dsigt * dsigt_dlam + da1_dsigc * dsigc_dlam
                da2 = da2_dsigs * dsigs_dlam + da2_dsigt * dsigt_dlam + da2_dsigc * dsigc_dlam

                ff = dfdsigdlam + da0 + p * da1 + (p ** 2) * da2
                if abs(ff) < _EM20:
                    ff = _EM20 if ff >= 0.0 else -_EM20

                dlam = phi / ff
                if dlam < 0.0:
                    dlam = 0.0

                dpxx = dlam * normxx_n
                dpyy = dlam * normyy_n
                dpzz = dlam * normzz_n
                dpxy = dlam * normxy_n

                sxx_tr -= (p_lp.a11_2d * dpxx + p_lp.a12_2d * dpyy)
                syy_tr -= (p_lp.a11_2d * dpyy + p_lp.a12_2d * dpxx)
                sxy_tr -= p_lp.g * dpxy

                inc_t = dlam * yld_norm * bb
                inc_s = dlam * yld_norm * (_SQR3 / 2.0)
                inc_eq = dlam * yld_norm

                plat = max(0.0, plat + inc_t)
                plac = plat
                plas = max(0.0, plas + inc_s)
                dplat += inc_t
                dplac += inc_t
                dplas += inc_s
                dpla += inc_eq

                # Through-thickness plastic strain contribution (no_asso_qplas76c.F line 413)
                dezz_i += nu1 * (dpxx + dpyy) + dpzz

                p = - (sxx_tr + syy_tr) / 3.0
                sxx_dev = sxx_tr + p
                syy_dev = syy_tr + p
                szz_dev = p
                svm2 = 1.5 * (sxx_dev**2 + syy_dev**2 + szz_dev**2) + 3.0 * (sxy_tr**2)
                svm = math.sqrt(max(0.0, svm2))

                sig_t, dsig_t = _eval_curve_or_val(
                    p_lp.tens_curve, plat, epspt * p_lp.facx, 1.0, p_lp.fscale_t, p_lp.sig_t0, p_lp.h_t
                )
                sig_c, dsig_c = _eval_curve_or_val(
                    p_lp.comp_curve, plac, epspc * p_lp.facx, 1.0, p_lp.fscale_c, p_lp.sig_c0, p_lp.h_c
                )
                sig_s, dsig_s = _eval_curve_or_val(
                    p_lp.shear_curve, plas, epsps * p_lp.facx, 1.0, p_lp.fscale_s, p_lp.sig_s0, p_lp.h_s
                )

                a0, a1_y, a2_y, sig_s_adj = _compute_yield_coeffs(p_lp, sig_t, sig_c, sig_s)

                if p_lp.iquad == 1:
                    phi = (svm**2) - a0 - a1_y * p - a2_y * (p**2)
                    tol = 1.0e-6 * max(1.0, a0)
                else:
                    phi = svm - a0 - a1_y * p - a2_y * (p**2)
                    tol = 1.0e-6 * max(1.0, a0)

                if abs(phi) <= tol or phi <= 0.0:
                    break

        # Thickness evolution (sigeps76c.F line 223)
        dezz[i] = dezz_i
        thk[i] = thk[i] + dezz_i * thk0[i] * off[i]
        # Prevent degenerate thickness
        thk[i] = max(0.01 * thk0[i], thk[i])

        if p_lp.fun_b5 != 0 or p_lp.nup_curve is not None:
            nup_val, _ = _eval_curve_or_val(
                p_lp.nup_curve or p_lp.fun_b5, epsp[i] + dpla, scale_y=p_lp.mat_pscale, default_val=p_lp.mat_nut
            )
            nu_p_curr = max(0.0, min(0.5, nup_val))

        uvar[i, 0] = plat
        uvar[i, 1] = plac
        uvar[i, 2] = plas
        if dt > 0.0:
            dt_inv = 1.0 / dt
            asrate = min(1.0, 2.0 * math.pi * p_lp.asrate * dt)
            uvar[i, 3] = asrate * (dplat * dt_inv) + (1.0 - asrate) * epspt
            uvar[i, 4] = asrate * (dplac * dt_inv) + (1.0 - asrate) * epspc
            uvar[i, 5] = asrate * (dplas * dt_inv) + (1.0 - asrate) * epsps
        else:
            uvar[i, 3] = 0.0
            uvar[i, 4] = 0.0
            uvar[i, 5] = 0.0
        uvar[i, 6] = nu_p_curr

        epsp[i] += dpla

        if p_lp.fun_a1 != 0 or p_lp.dmg_curve is not None:
            d_val, _ = _eval_curve_or_val(
                p_lp.dmg_curve or p_lp.fun_a1, epsp[i], scale_y=p_lp.scale_dmg, default_val=0.0
            )
            dmg[i] = max(dmg[i], min(1.0, d_val))
        elif epsp[i] >= p_lp.eps_f:
            d_val = (epsp[i] - p_lp.eps_f) / max(_EM20, p_lp.eps_r - p_lp.eps_f)
            dmg[i] = max(dmg[i], min(1.0, d_val))

        if dmg[i] >= 1.0:
            dmg[i] = 1.0
            if off[i] == 1.0:
                off[i] = 0.8

        degrade = max(0.0, 1.0 - dmg[i]) * off[i]
        sign[i, 0] = sxx_tr * degrade
        sign[i, 1] = syy_tr * degrade
        sign[i, 2] = sxy_tr * degrade
        if n_comp >= 5:
            sign[i, 3] = syz_tr * degrade
            sign[i, 4] = szx_tr * degrade

    res_sig = sign[0] if is_1d else sign
    res_epsp = epsp[0] if is_1d else epsp
    return res_sig, res_epsp


# =============================================================================
# Updaters, Sound Speed, Tangents, and State Helpers
# =============================================================================

def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_tuple: bool = False,
    **kwargs: Any,
) -> Union[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray], Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Standard pyradioss wrapper for LAW76 solid stress update."""
    res_sig, res_epsp, res_c = solid_step(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, **kwargs)
    if return_tuple:
        return res_sig, res_epsp, res_c
    return res_sig, res_epsp


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Standard pyradioss wrapper for LAW76 shell stress update."""
    return shell_step(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, **kwargs)


def sound_speed(
    mat: Any,
    rho: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    is_shell: bool = False,
    **kwargs: Any,
) -> float:
    """Exact acoustic longitudinal sound speed for /MAT/LAW76.

    Solid: c = sqrt((K + 4/3 G) / rho)
    Shell: c = sqrt(E / ((1 - nu^2) * rho))
    """
    p_lp = _get_law76_params(mat)
    rho_val = float(rho if rho is not None else p_lp.rho0)
    rho_val = max(_EM20, rho_val)

    if is_shell or (extra and extra.get("is_shell", False)):
        return math.sqrt(p_lp.a11_2d / rho_val)
    return math.sqrt((p_lp.bulk + 4.0 / 3.0 * p_lp.g) / rho_val)


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent algorithmic tangent stiffness tensor C (6, 6) for LAW76 solids."""
    p_lp = _get_law76_params(mat)
    c_elastic = np.zeros((6, 6), dtype=float)
    c_elastic[0, 0] = c_elastic[1, 1] = c_elastic[2, 2] = p_lp.a1
    c_elastic[0, 1] = c_elastic[0, 2] = c_elastic[1, 0] = p_lp.a2
    c_elastic[1, 2] = c_elastic[2, 0] = c_elastic[2, 1] = p_lp.a2
    c_elastic[3, 3] = c_elastic[4, 4] = c_elastic[5, 5] = p_lp.g

    if sig is None or deps is None:
        return c_elastic

    sig_arr = np.asarray(sig, dtype=float)
    deps_arr = np.asarray(deps, dtype=float)

    if sig_arr.ndim == 2 and sig_arr.shape[0] > 1:
        n = sig_arr.shape[0]
        return np.broadcast_to(c_elastic, (n, 6, 6)).copy()

    # Numerical perturbation for algorithmic consistency
    sig0 = sig_arr.flatten()
    deps0 = deps_arr.flatten()
    if len(sig0) < 6:
        s_pad = np.zeros(6, dtype=float)
        s_pad[:len(sig0)] = sig0
        sig0 = s_pad
    if len(deps0) < 6:
        d_pad = np.zeros(6, dtype=float)
        d_pad[:len(deps0)] = deps0
        deps0 = d_pad

    ex0 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in (extra or {}).items()}
    sig_base, _, _ = solid_step(mat, sig0, deps0, dt=dt, extra=ex0)

    h = 1.0e-7
    c_algo = np.zeros((6, 6), dtype=float)
    for j in range(6):
        deps_p = deps0.copy()
        deps_p[j] += h
        ex_p = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in (extra or {}).items()}
        sig_p, _, _ = solid_step(mat, sig0, deps_p, dt=dt, extra=ex_p)
        c_algo[:, j] = (sig_p - sig_base) / h

    return c_algo


def consistent_solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Alias for solid_tangent."""
    return solid_tangent(mat, sig=sig, deps=deps, dt=dt, extra=extra, **kwargs)


def shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent plane-stress tangent stiffness matrix C (3, 3) for LAW76 shells."""
    p_lp = _get_law76_params(mat)
    c_elastic = np.array([
        [p_lp.a11_2d, p_lp.a12_2d, 0.0],
        [p_lp.a12_2d, p_lp.a11_2d, 0.0],
        [0.0, 0.0, p_lp.g],
    ], dtype=float)

    if sig is None or deps is None:
        return c_elastic

    sig_arr = np.asarray(sig, dtype=float)
    deps_arr = np.asarray(deps, dtype=float)
    if sig_arr.ndim == 2 and sig_arr.shape[0] > 1:
        n = sig_arr.shape[0]
        return np.broadcast_to(c_elastic, (n, 3, 3)).copy()

    sig0 = sig_arr.flatten()[:3]
    deps0 = deps_arr.flatten()[:3]
    ex0 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in (extra or {}).items()}
    sig_base, _ = shell_step(mat, sig0, deps0, dt=dt, extra=ex0)

    h = 1.0e-7
    c_algo = np.zeros((3, 3), dtype=float)
    for j in range(3):
        deps_p = deps0.copy()
        deps_p[j] += h
        ex_p = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in (extra or {}).items()}
        sig_p, _ = shell_step(mat, sig0, deps_p, dt=dt, extra=ex_p)
        c_algo[:, j] = (sig_p[:3] - sig_base[:3]) / h

    return c_algo


def consistent_shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Alias for shell_tangent."""
    return shell_tangent(mat, sig=sig, deps=deps, dt=dt, extra=extra, **kwargs)


def shell_membrane_tangent(mat: Any) -> np.ndarray:
    """Plane-stress 3x3 elastic membrane tangent for LAW76."""
    p_lp = _get_law76_params(mat)
    return np.array([
        [p_lp.a11_2d, p_lp.a12_2d, 0.0],
        [p_lp.a12_2d, p_lp.a11_2d, 0.0],
        [0.0, 0.0, p_lp.g],
    ], dtype=float)


def extra_shapes(mat: Any, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Define persistent internal state variables for LAW76 (NUVAR = 7)."""
    return {
        "uvar": (nip, 7) if nip > 1 else (7,),
        "dmg": (nip,) if nip > 1 else (1,),
        "off": (nip,) if nip > 1 else (1,),
        "thk": (nip,) if nip > 1 else (1,),
        "thk0": (nip,) if nip > 1 else (1,),
        "dezz": (nip,) if nip > 1 else (1,),
    }
