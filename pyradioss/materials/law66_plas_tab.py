"""
LAW66 — Asymmetric Tension/Compression Tabulated Plasticity Model for Solids and Shells
(/MAT/LAW66, /MAT/PLAS_TAB_COSSER, /MAT/PLAS_COSSER, /MAT/FOAM_TAB).

Upstream Fortran reference:
  - Starter reader: starter/source/materials/mat/mat066/hm_read_mat66.F
  - Solid engine physics: engine/source/materials/mat/mat066/sigeps66.F
  - Shell engine physics: engine/source/materials/mat/mat066/sigeps66c.F
  - Card layout / CFG: hm_cfg_files/config/CFG/radioss*/MAT/matl66_*.cfg

Theory & Algorithm
------------------
1. Asymmetric elastic response under hydrostatic pressure P:
   - Hydrostatic pressure P = C1T * mu (with mu = rho/rho0 - 1).
   - In tension (P <= -RPCT * PT): E = ET (tensile Young's modulus).
   - In compression (P >= RPCT * PC): E = EC (compressive Young's modulus).
   - In transition (-RPCT * PT < P < RPCT * PC):
       fac = (RPCT * PC - P) / (RPCT * (PC + PT))
       E = fac * ET + (1 - fac) * EC
   - Derived shear G = 0.5 * E / (1 + nu), Bulk modulus C1 = E / (3 * (1 - 2*nu)).

2. Trial stress shifted by kinematic backstress alpha:
   - Solids:
       P0 = -trace(sig - alpha) / 3
       dav = trace(deps) / 3
       s_trial = dev(sig - alpha) + 2*G*(dev(deps))
       s_trial_shear = (sig - alpha)_shear + G * deps_shear
   - Shells (plane stress):
       s_trial_xx = (sig - alpha)_xx + A11*deps_xx + A21*deps_yy
       s_trial_yy = (sig - alpha)_yy + A21*deps_xx + A11*deps_yy
       s_trial_xy = (sig - alpha)_xy + G*deps_xy

3. Asymmetric yield function and pressure interpolation:
   - Compression yield YC(pla) and tension yield YT(pla) evaluated from curves/tables.
   - Mixed isotropic-kinematic hardening:
       YC = (1 - fisokin) * YC(pla) + fisokin * YC(0)
       YT = (1 - fisokin) * YT(pla) + fisokin * YT(0)
   - Pressure interpolation between YT and YC:
       If P <= -PT: YLD = YT, H = HT
       Else if P >= PC: YLD = YC, H = HC
       Else:
         fac = (PC - P) / (PC + PT)
         YLD = fac * YT + (1 - fac) * YC
         H = fac * HT + (1 - fac) * HC

4. Strain rate formulations:
   - ISRATE = 1: Cowper-Symonds power law: YRATE = 1 + (rate / epsp0)**(1/cp)
   - ISRATE = 2: Logarithmic form: YRATE = 1 + cp * ln(rate / epsp0)
   - ISRATE = 3: Independent scaling curves for compression (fun_b1) and tension (fun_b2)
   - ISRATE = 4: Multi-curve strain rate families with linear rate interpolation
   - VP = 1: Viscoplastic overstress formulation based on dpla / dt.

5. Radial return / projection:
   - Solids: J2 von Mises norm AJ2 = sqrt(3 * J2).
     If AJ2 > YLD:
       dpla = (AJ2 - YLD) / (3*G + H)
       s_new = (YLD / AJ2) * s_trial
       delta_alpha = [hkin / (2*G + hkin)] * (s_trial - s_new)
       alpha += delta_alpha
       sig_new = s_new + alpha - P * I
   - Shells: Plane-stress von Mises norm SVM.
     If SVM > YLD:
       dpla = (SVM - YLD) / (3*G + H)
       s_new = (YLD / SVM) * s_trial
       delta_alpha = [hkin / (E + hkin)] * ...
       alpha += delta_alpha
       sig_new = s_new + alpha
       dezz through-thickness thinning updates shell thickness thk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_INF = 1.0e30
_P_PLANE = np.array([[1.0, -0.5, 0.0],
                     [-0.5, 1.0, 0.0],
                     [0.0, 0.0, 3.0]], dtype=float)


# ============================================================================
# Parameter Dataclass: Law66Params
# ============================================================================

@dataclass
class Law66Params:
    """Parameters for /MAT/LAW66 (/MAT/PLAS_TAB_COSSER, /MAT/PLAS_COSSER, /MAT/FOAM_TAB).

    Follows starter/source/materials/mat/mat066/hm_read_mat66.F.
    """
    rho0: float = 1.0
    refer_rho: float = 1.0
    e: float = 210000.0          # Et: tension Young's modulus
    nu: float = 0.3
    ec: float = 0.0              # Ec: compression Young's modulus (default = e)
    pc: float = 0.0              # Limit pressure in compression
    pt: float = 0.0              # Limit pressure in tension
    rpct: float = 1.0            # Fraction of pt and pc used for modulus interpolation
    chard: float = 0.0           # fisokin: iso-kinematic hardening factor [0..1]
    asrate: float = 0.0          # fcut: cutoff frequency for strain rate filtering
    fsmooth: int = 0             # Strain rate smoothing flag
    israte: int = 1              # 1: CS power, 2: CS log, 3: independent curves, 4: families
    epsp0: float = 0.0           # Reference strain rate (0.0: rate sensitivity disabled)
    cp: float = 1.0              # Strain rate parameter
    sigy: float = 0.0            # Initial yield stress
    vp: int = 0                  # Viscoplastic flag (0: static scaling, 1: rate from dpla/dt)

    # Function IDs or curve representations
    fun_a1: Any = 0              # Compression yield curve / ID
    fun_a2: Any = 0              # Tension yield curve / ID
    fscale11: float = 1.0        # Scale factor for compression yield curve
    fscale22: float = 1.0        # Scale factor for tension yield curve

    fun_b1: Any = 0              # Compression strain-rate scaling curve / ID (ISRATE=3)
    fun_b2: Any = 0              # Tension strain-rate scaling curve / ID (ISRATE=3)
    fscale33: float = 1.0        # Scale factor for fun_b1
    fscale12: float = 1.0        # Scale factor for fun_b2

    # ISRATE = 4 curve families
    abg_ipt: list = field(default_factory=list)    # Compression curve IDs
    k_a1: list = field(default_factory=list)       # Compression strain rates
    fp1: list = field(default_factory=list)        # Compression curve scale factors

    abg_ipdel: list = field(default_factory=list)  # Tension curve IDs
    k_b1: list = field(default_factory=list)       # Tension strain rates
    fp2: list = field(default_factory=list)        # Tension curve scale factors

    id: int = 1
    title: str = "LAW66"

    # Resolved curve objects / data
    curve_c: Any = None
    curve_t: Any = None
    curve_rate_c: Any = None
    curve_rate_t: Any = None
    curves_c: list = field(default_factory=list)
    rates_c: list = field(default_factory=list)
    curves_t: list = field(default_factory=list)
    rates_t: list = field(default_factory=list)

    # Derived constants computed in __post_init__
    gt: float = field(init=False, default=0.0)
    c1t: float = field(init=False, default=0.0)
    gc: float = field(init=False, default=0.0)
    c1c: float = field(init=False, default=0.0)
    a11t: float = field(init=False, default=0.0)
    a21t: float = field(init=False, default=0.0)
    a11c: float = field(init=False, default=0.0)
    a21c: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.refer_rho <= 0.0:
            self.refer_rho = self.rho0
        if self.rho0 <= 0.0:
            self.rho0 = self.refer_rho if self.refer_rho > 0.0 else 1.0

        if self.ec <= 0.0:
            self.ec = self.e

        if self.nu >= 0.5:
            self.nu = 0.499
        if self.nu < 0.0:
            self.nu = 0.0

        if self.rpct <= 0.0:
            self.rpct = 1.0

        if self.israte == 0:
            self.israte = 1

        if self.fscale11 <= 0.0:
            self.fscale11 = 1.0
        if self.fscale22 <= 0.0:
            self.fscale22 = 1.0
        if self.fscale33 <= 0.0:
            self.fscale33 = 1.0
        if self.fscale12 <= 0.0:
            self.fscale12 = 1.0

        # Derived elastic constants (Et, Ec)
        nu_denom = max(1.0 - 2.0 * self.nu, 1.0e-15)
        nu_sq = max(1.0 - self.nu ** 2, 1.0e-15)

        self.gt = 0.5 * self.e / max(1.0 + self.nu, 1.0e-15)
        self.c1t = self.e / (3.0 * nu_denom)
        self.a11t = self.e / nu_sq
        self.a21t = self.nu * self.a11t

        self.gc = 0.5 * self.ec / max(1.0 + self.nu, 1.0e-15)
        self.c1c = self.ec / (3.0 * nu_denom)
        self.a11c = self.ec / nu_sq
        self.a21c = self.nu * self.a11c

    @property
    def E(self) -> float:
        return self.e

    @property
    def nu0(self) -> float:
        return self.nu

    @property
    def G(self) -> float:
        return self.gt

    @property
    def K(self) -> float:
        return self.c1t

    @property
    def fisokin(self) -> float:
        return self.chard

    @property
    def soundsp(self) -> float:
        """Dilatational sound speed for solid elements."""
        c1_max = max(self.c1t, self.c1c)
        return math.sqrt((c1_max + (4.0 / 3.0) * self.gt) / max(self.rho0, _EM20))

    @property
    def soundsp_shell(self) -> float:
        """Dilatational sound speed for shell elements."""
        return math.sqrt(self.a11t / max(self.rho0, _EM20))

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)


# ============================================================================
# Curve Evaluation Helpers
# ============================================================================

def _eval_curve(curve: Any, x: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear evaluation of 1D curve returning (value, slope).

    Extrapolates using end-segment slope matching OpenRadioss FINTER.
    """
    x_arr = np.asarray(x, dtype=float)
    if curve is None or curve == 0:
        return np.zeros_like(x_arr), np.zeros_like(x_arr)

    if hasattr(curve, "eval"):
        try:
            val = np.asarray(curve.eval(x_arr), dtype=float)
            slope = getattr(curve, "slope", None)
            if slope is not None:
                slp = np.asarray(slope, dtype=float)
                if slp.size == 1:
                    slp_arr = np.full_like(val, float(slp))
                elif len(slp) > 0:
                    slp_arr = np.full_like(val, float(slp[-1]))
                else:
                    slp_arr = np.zeros_like(val)
            else:
                slp_arr = np.zeros_like(val)
            return val, slp_arr
        except Exception:
            pass

    cx, cy = None, None
    if hasattr(curve, "x") and hasattr(curve, "y"):
        cx = np.asarray(curve.x, dtype=float)
        cy = np.asarray(curve.y, dtype=float)
    elif isinstance(curve, tuple) and len(curve) == 2 and isinstance(curve[0], (list, tuple, np.ndarray)):
        cx = np.asarray(curve[0], dtype=float)
        cy = np.asarray(curve[1], dtype=float)
    elif isinstance(curve, (tuple, list)):
        try:
            pts = np.asarray(curve, dtype=float)
            if pts.ndim == 2 and pts.shape[0] == 2 and pts.shape[1] != 2:
                cx, cy = pts[0, :], pts[1, :]
            elif pts.ndim == 2 and pts.shape[1] >= 2:
                cx, cy = pts[:, 0], pts[:, 1]
        except Exception:
            pass
    elif isinstance(curve, dict) and "x" in curve and "y" in curve:
        cx = np.asarray(curve["x"], dtype=float)
        cy = np.asarray(curve["y"], dtype=float)
    elif callable(curve):
        res = curve(x_arr)
        if isinstance(res, tuple):
            return np.asarray(res[0], dtype=float), np.asarray(res[1], dtype=float)
        return np.asarray(res, dtype=float), np.zeros_like(x_arr)
    elif isinstance(curve, (int, float)):
        return np.full_like(x_arr, float(curve)), np.zeros_like(x_arr)

    if cx is None or len(cx) == 0:
        return np.zeros_like(x_arr), np.zeros_like(x_arr)
    if len(cx) == 1:
        return np.full_like(x_arr, cy[0]), np.zeros_like(x_arr)

    dx = np.diff(cx)
    dx_safe = np.where(np.abs(dx) > _EM20, dx, np.sign(dx) * _EM20 + _EM20)
    cs = np.diff(cy) / dx_safe
    idx = np.minimum(np.maximum(np.searchsorted(cx, x_arr, side="right") - 1, 0), len(cx) - 2)
    val = cy[idx] + cs[idx] * (x_arr - cx[idx])
    return val, cs[idx]


def _eval_rate_family(curves: list, rates: list, fscales: list,
                      epsp: np.ndarray, rate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate multi-curve strain rate family (ISRATE=4) at (epsp, rate)."""
    n_pts = len(epsp)
    n_curves = len(curves)
    if n_curves == 0:
        return np.zeros(n_pts, dtype=float), np.zeros(n_pts, dtype=float)
    if n_curves == 1:
        val, slp = _eval_curve(curves[0], epsp)
        sc = fscales[0] if len(fscales) > 0 else 1.0
        return val * sc, slp * sc

    rates_arr = np.asarray(rates, dtype=float)
    j = np.clip(np.searchsorted(rates_arr, rate, side="right") - 1, 0, n_curves - 2)
    denom = np.maximum(rates_arr[j + 1] - rates_arr[j], _EM20)
    fac = np.clip((rate - rates_arr[j]) / denom, 0.0, 1.0)

    val_res = np.zeros(n_pts, dtype=float)
    slp_res = np.zeros(n_pts, dtype=float)

    for k in range(n_curves - 1):
        mask = (j == k)
        if not np.any(mask):
            continue
        sc1 = fscales[k] if k < len(fscales) else 1.0
        sc2 = fscales[k + 1] if (k + 1) < len(fscales) else 1.0
        y1, s1 = _eval_curve(curves[k], epsp[mask])
        y2, s2 = _eval_curve(curves[k + 1], epsp[mask])
        y1 *= sc1
        s1 *= sc1
        y2 *= sc2
        s2 *= sc2
        f = fac[mask]
        val_res[mask] = y1 + f * (y2 - y1)
        slp_res[mask] = s1 + f * (s2 - s1)

    return val_res, slp_res


# ============================================================================
# Moduli and Yield Calculation
# ============================================================================

def _compute_pressure_moduli(p: Law66Params, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute effective E, G, C1 based on hydrostatic pressure P (sigeps66.F:211-234)."""
    n = len(P)
    E = np.full(n, p.e, dtype=float)
    if p.ec > 0.0 and p.ec != p.e:
        p_comp = p.rpct * p.pc
        p_tens = -p.rpct * p.pt
        if p.pc == 0.0 and p.pt == 0.0:
            E = np.where(P >= 0.0, p.ec, p.e)
        else:
            denom = p.rpct * (p.pc + p.pt)
            tens_mask = P <= p_tens
            comp_mask = P >= p_comp
            interp_mask = (~tens_mask) & (~comp_mask)
            E[tens_mask] = p.e
            E[comp_mask] = p.ec
            if denom > 0.0 and np.any(interp_mask):
                fac = (p_comp - P[interp_mask]) / denom
                E[interp_mask] = fac * p.e + (1.0 - fac) * p.ec

    G = 0.5 * E / max(1.0 + p.nu, 1.0e-15)
    C1 = E / (3.0 * max(1.0 - 2.0 * p.nu, 1.0e-15))
    return E, G, C1


_compute_elastic_moduli = _compute_pressure_moduli


def _compute_yield_and_hardening(p: Law66Params, P: np.ndarray, epsp: np.ndarray,
                                 rate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute asymmetric yield stress YLD and hardening slope H (sigeps66.F:277-534)."""
    n = len(epsp)

    # 1. Compression curve
    if p.israte == 4 and len(p.curves_c) > 0:
        yc, hc = _eval_rate_family(p.curves_c, p.rates_c, p.fp1, epsp, rate)
    else:
        c_curve = p.curve_c if p.curve_c is not None else p.fun_a1
        yc, hc = _eval_curve(c_curve, epsp)
        yc = yc * p.fscale11
        hc = hc * p.fscale11

    # 2. Tension curve
    if p.israte == 4 and len(p.curves_t) > 0:
        yt, ht = _eval_rate_family(p.curves_t, p.rates_t, p.fp2, epsp, rate)
    else:
        t_curve = p.curve_t if p.curve_t is not None else p.fun_a2
        yt, ht = _eval_curve(t_curve, epsp)
        yt = yt * p.fscale22
        ht = ht * p.fscale22

    # Fallback to sigy if curves evaluate to zero
    if p.sigy > 0.0:
        yc = np.where(yc <= 0.0, p.sigy, yc)
        yt = np.where(yt <= 0.0, p.sigy, yt)

    # 3. Iso-kinematic hardening partition (CHARD / fisokin in [0..1])
    fisokin = p.chard
    if fisokin > 0.0:
        if p.israte == 4 and len(p.curves_c) > 0:
            yc0, _ = _eval_rate_family(p.curves_c, p.rates_c, p.fp1, np.zeros(n), rate)
        else:
            c_curve = p.curve_c if p.curve_c is not None else p.fun_a1
            yc0, _ = _eval_curve(c_curve, np.zeros(n))
            yc0 = yc0 * p.fscale11
            if p.sigy > 0.0:
                yc0 = np.where(yc0 <= 0.0, p.sigy, yc0)

        if p.israte == 4 and len(p.curves_t) > 0:
            yt0, _ = _eval_rate_family(p.curves_t, p.rates_t, p.fp2, np.zeros(n), rate)
        else:
            t_curve = p.curve_t if p.curve_t is not None else p.fun_a2
            yt0, _ = _eval_curve(t_curve, np.zeros(n))
            yt0 = yt0 * p.fscale22
            if p.sigy > 0.0:
                yt0 = np.where(yt0 <= 0.0, p.sigy, yt0)

        yc = (1.0 - fisokin) * yc + fisokin * yc0
        yt = (1.0 - fisokin) * yt + fisokin * yt0

    yc = np.maximum(yc, _EM20)
    yt = np.maximum(yt, _EM20)

    # 4. Independent strain rate scaling for ISRATE = 3
    if p.israte == 3:
        rate_c_obj = p.curve_rate_c if p.curve_rate_c is not None else p.fun_b1
        rate_t_obj = p.curve_rate_t if p.curve_rate_t is not None else p.fun_b2
        yrate_c, _ = _eval_curve(rate_c_obj, rate)
        yrate_t, _ = _eval_curve(rate_t_obj, rate)
        yrate_c = yrate_c * p.fscale33
        yrate_t = yrate_t * p.fscale12
        yc = yc * yrate_c
        hc = hc * yrate_c
        yt = yt * yrate_t
        ht = ht * yrate_t

    # 5. Interpolate yield stress and hardening modulus based on pressure P
    if p.pc == 0.0 and p.pt == 0.0:
        comp_mask = P >= 0.0
        yld = np.where(comp_mask, yc, yt)
        h = np.where(comp_mask, hc, ht)
    else:
        tens_mask = P <= -p.pt
        comp_mask = P >= p.pc
        interp_mask = (~tens_mask) & (~comp_mask)
        yld = np.empty(n, dtype=float)
        h = np.empty(n, dtype=float)
        yld[tens_mask] = yt[tens_mask]
        h[tens_mask] = ht[tens_mask]
        yld[comp_mask] = yc[comp_mask]
        h[comp_mask] = hc[comp_mask]
        denom = p.pc + p.pt
        if denom > 0.0 and np.any(interp_mask):
            fac = (p.pc - P[interp_mask]) / denom
            yld[interp_mask] = fac * yt[interp_mask] + (1.0 - fac) * yc[interp_mask]
            h[interp_mask] = fac * ht[interp_mask] + (1.0 - fac) * hc[interp_mask]

    yld = np.maximum(yld, _EM20)

    # 6. Analytical strain rate scaling for ISRATE = 1 or 2 (when VP == 0)
    # Follows sigeps66.F:517-534. Active only when epsp0 > 0.
    if p.vp == 0 and p.epsp0 > 0.0 and p.epsp0 < 1.0e19:
        rate_val = np.maximum(rate, 0.0)
        has_rate = rate_val > 0.0
        if p.israte == 1 and np.any(has_rate):
            epd = np.maximum(rate_val / p.epsp0, _EM20)
            power = p.cp if p.cp != 0.0 else 1.0
            yrate = np.where(has_rate, 1.0 + epd ** power, 1.0)
            yld = yld * yrate
            h = h * yrate
        elif p.israte == 2 and np.any(has_rate):
            epd = np.maximum(rate_val / p.epsp0, _EM20)
            yrate = np.where(has_rate, 1.0 + p.cp * np.log(epd), 1.0)
            yld = yld * yrate
            h = h * yrate

    return yld, h


# ============================================================================
# Parameter Extraction Helper
# ============================================================================

def _extract_param(p: Dict[str, Any], keys: Sequence[str], default: Any) -> Any:
    for k in keys:
        if k in p and p[k] is not None:
            return p[k]
    if "params" in p and isinstance(p["params"], dict):
        for k in keys:
            if k in p["params"] and p["params"][k] is not None:
                return p["params"][k]
    return default


def _get_params(mat: Any) -> Law66Params:
    if isinstance(mat, Law66Params):
        return mat
    if hasattr(mat, "param_obj") and isinstance(mat.param_obj, Law66Params):
        return mat.param_obj
    if hasattr(mat, "law66_params") and isinstance(mat.law66_params, Law66Params):
        return mat.law66_params
    if hasattr(mat, "params") and isinstance(mat.params, dict):
        if "_obj" in mat.params and isinstance(mat.params["_obj"], Law66Params):
            return mat.params["_obj"]
    return build_law66(mat).params["_obj"]


# ============================================================================
# Factory: build_law66
# ============================================================================

def build_law66(rec: Any = None, **kwargs: Any) -> Material:
    """Physics constructor for /MAT/LAW66 (/MAT/PLAS_TAB_COSSER, /MAT/PLAS_COSSER).

    Accepts Law66Params, MaterialLaw66 dataclass, dict, or keyword arguments.
    """
    if isinstance(rec, Law66Params):
        p_obj = rec
        mid = rec.id
        rho0 = rec.rho0
        title = rec.title
    elif rec is None:
        p_dict = dict(kwargs)
        if "params" in p_dict and isinstance(p_dict["params"], dict):
            p_dict = {**p_dict["params"], **p_dict}
        mid = int(p_dict.get("id", 1))
        rho0 = float(p_dict.get("rho0", p_dict.get("MAT_RHO", p_dict.get("density", 1.0))))
        title = str(p_dict.get("title", "LAW66"))
        p_obj = None
    elif isinstance(rec, dict):
        base = rec.get("params", rec)
        p_dict = {**base, **kwargs}
        mid = int(rec.get("id", p_dict.get("id", 1)))
        rho0 = float(p_dict.get("rho0", p_dict.get("MAT_RHO", p_dict.get("density", 1.0))))
        title = str(rec.get("title", p_dict.get("title", "LAW66")))
        p_obj = None
    elif hasattr(rec, "params") and isinstance(rec.params, dict):
        base_dict = {}
        if hasattr(rec, "__dict__"):
            base_dict.update(rec.__dict__)
        base_dict.update(rec.params)
        p_dict = {**base_dict, **kwargs}
        mid = int(getattr(rec, "id", p_dict.get("id", 1)))
        rho0 = float(getattr(rec, "rho0", getattr(rec, "rho", p_dict.get("rho0", 1.0))))
        title = str(getattr(rec, "title", p_dict.get("title", "LAW66")))
        p_obj = None
    elif hasattr(rec, "__dict__"):
        p_dict = {**rec.__dict__, **kwargs}
        mid = int(getattr(rec, "id", p_dict.get("id", 1)))
        rho0 = float(getattr(rec, "rho0", getattr(rec, "rho", p_dict.get("rho0", 1.0))))
        title = str(getattr(rec, "title", p_dict.get("title", "LAW66")))
        p_obj = None
    else:
        p_dict = dict(kwargs)
        mid = int(p_dict.get("id", 1))
        rho0 = float(p_dict.get("rho0", 1.0))
        title = str(p_dict.get("title", "LAW66"))
        p_obj = None

    if p_obj is None:
        def _get_f(keys: Sequence[str], default: float) -> float:
            val = _extract_param(p_dict, keys, default)
            try:
                return float(val)
            except (ValueError, TypeError):
                return float(default)

        def _get_i(keys: Sequence[str], default: int) -> int:
            val = _extract_param(p_dict, keys, default)
            try:
                return int(val)
            except (ValueError, TypeError):
                return int(default)

        rho0_val = _get_f(["MAT_RHO", "Refer_Rho", "rho0", "density", "rho"], rho0)
        rhor_val = _get_f(["Refer_Rho", "ref_rho", "rhor", "MAT_REFRHO"], rho0_val)
        e_val = _get_f(["MAT_E", "E", "e", "young", "Young"], 210000.0)
        nu_val = _get_f(["MAT_NU", "NU", "nu"], 0.3)
        ec_val = _get_f(["MAT_EC", "EC", "ec"], 0.0)
        pc_val = _get_f(["MAT_PC", "P_c", "PC", "pc"], 0.0)
        pt_val = _get_f(["MAT_PT", "P_t", "PT", "pt"], 0.0)
        rpct_val = _get_f(["MAT_RPCT", "RPCT", "rpct"], 1.0)
        chard_val = _get_f(["MAT_HARD", "C_hard", "c_hard", "chard", "CHARD", "FISOKIN", "fisokin"], 0.0)
        asrate_val = _get_f(["MAT_asrate", "F_cut", "asrate", "fcut", "f_cut"], 0.0)
        fsmooth_val = _get_i(["Fsmooth", "fsmooth"], 0)
        israte_val = _get_i(["ISRATE", "israte", "irate"], 1)

        fun_a1_val = _extract_param(p_dict, ["FUN_A1", "funct_IDc", "fun_a1", "curve_c"], 0)
        fun_a2_val = _extract_param(p_dict, ["FUN_A2", "funct_IDt", "fun_a2", "curve_t"], 0)
        fscale11_val = _get_f(["FScale11", "Fscalec", "fscale11", "fscalec"], 1.0)
        fscale22_val = _get_f(["FScale22", "Fscalet", "fscale22", "fscalet"], 1.0)

        epsp0_val = _get_f(["Epsilon_0", "epsp0", "EPSP0", "epsilon_0"], 0.0)
        cp_val = _get_f(["MAT_C0", "c", "cp", "CP"], 1.0)
        sigy_val = _get_f(["SIGMA_r", "Sigma_Y0", "SIG_Y0", "sigy", "sigma_y0", "sig_y0"], 0.0)
        vp_val = _get_i(["VP", "vp"], 0)

        fun_b1_val = _extract_param(p_dict, ["FUN_B1", "fnYrt_IDc", "fun_b1", "curve_rate_c"], 0)
        fun_b2_val = _extract_param(p_dict, ["FUN_B2", "fnYrt_IDt", "fun_b2", "curve_rate_t"], 0)
        fscale33_val = _get_f(["FScale33", "Yrate_Fscalec", "fscale33"], 1.0)
        fscale12_val = _get_f(["FScale12", "Yrate_Fscalet", "fscale12"], 1.0)

        abg_ipt_val = list(_extract_param(p_dict, ["ABG_IPt", "func_c_list", "abg_ipt"], []))
        k_a1_val = list(_extract_param(p_dict, ["K_A1", "eps_c_list", "k_a1"], []))
        fp1_val = list(_extract_param(p_dict, ["Fp1", "fscale_c_list", "fp1"], []))

        abg_ipdel_val = list(_extract_param(p_dict, ["ABG_IPdel", "func_t_list", "abg_ipdel"], []))
        k_b1_val = list(_extract_param(p_dict, ["K_B1", "eps_t_list", "k_b1"], []))
        fp2_val = list(_extract_param(p_dict, ["Fp2", "fscale_t_list", "fp2"], []))

        curve_c_obj = _extract_param(p_dict, ["curve_c", "curve_comp"], None)
        curve_t_obj = _extract_param(p_dict, ["curve_t", "curve_tens"], None)
        curve_rate_c_obj = _extract_param(p_dict, ["curve_rate_c"], None)
        curve_rate_t_obj = _extract_param(p_dict, ["curve_rate_t"], None)

        p_obj = Law66Params(
            rho0=rho0_val,
            refer_rho=rhor_val,
            e=e_val,
            nu=nu_val,
            ec=ec_val,
            pc=pc_val,
            pt=pt_val,
            rpct=rpct_val,
            chard=chard_val,
            asrate=asrate_val,
            fsmooth=fsmooth_val,
            israte=israte_val,
            epsp0=epsp0_val,
            cp=cp_val,
            sigy=sigy_val,
            vp=vp_val,
            fun_a1=fun_a1_val,
            fun_a2=fun_a2_val,
            fscale11=fscale11_val,
            fscale22=fscale22_val,
            fun_b1=fun_b1_val,
            fun_b2=fun_b2_val,
            fscale33=fscale33_val,
            fscale12=fscale12_val,
            abg_ipt=abg_ipt_val,
            k_a1=k_a1_val,
            fp1=fp1_val,
            abg_ipdel=abg_ipdel_val,
            k_b1=k_b1_val,
            fp2=fp2_val,
            id=mid,
            title=title,
            curve_c=curve_c_obj,
            curve_t=curve_t_obj,
            curve_rate_c=curve_rate_c_obj,
            curve_rate_t=curve_rate_t_obj,
        )

    params_dict = {
        "rho0": p_obj.rho0,
        "refer_rho": p_obj.refer_rho,
        "e": p_obj.e,
        "E": p_obj.e,
        "nu": p_obj.nu,
        "ec": p_obj.ec,
        "EC": p_obj.ec,
        "pc": p_obj.pc,
        "P_c": p_obj.pc,
        "pt": p_obj.pt,
        "P_t": p_obj.pt,
        "rpct": p_obj.rpct,
        "RPCT": p_obj.rpct,
        "chard": p_obj.chard,
        "fisokin": p_obj.chard,
        "asrate": p_obj.asrate,
        "f_cut": p_obj.asrate,
        "fsmooth": p_obj.fsmooth,
        "israte": p_obj.israte,
        "ISRATE": p_obj.israte,
        "epsp0": p_obj.epsp0,
        "cp": p_obj.cp,
        "sigy": p_obj.sigy,
        "vp": p_obj.vp,
        "fun_a1": p_obj.fun_a1,
        "fun_a2": p_obj.fun_a2,
        "fscale11": p_obj.fscale11,
        "fscale22": p_obj.fscale22,
        "fun_b1": p_obj.fun_b1,
        "fun_b2": p_obj.fun_b2,
        "fscale33": p_obj.fscale33,
        "fscale12": p_obj.fscale12,
        "abg_ipt": p_obj.abg_ipt,
        "k_a1": p_obj.k_a1,
        "fp1": p_obj.fp1,
        "abg_ipdel": p_obj.abg_ipdel,
        "k_b1": p_obj.k_b1,
        "fp2": p_obj.fp2,
        "G": p_obj.gt,
        "K": p_obj.c1t,
        "A11": p_obj.a11t,
        "A21": p_obj.a21t,
        "C1": p_obj.c1t,
        "_obj": p_obj,
    }

    mat = Material(
        id=p_obj.id,
        law=66,
        rho0=p_obj.rho0,
        title=p_obj.title,
        params=params_dict,
        law_name="LAW66",
    )
    mat.param_obj = p_obj
    mat.law66_params = p_obj
    return mat


def resolve(mat: Any, model: Any, log: Any = None) -> None:
    """Resolve /FUNCT references into material curve arrays."""
    p = _get_params(mat)

    def _resolve_one(fid: Any) -> Any:
        if fid == 0 or fid is None:
            return None
        if hasattr(model, "functions") and fid in model.functions:
            return model.functions[fid]
        if hasattr(model, "curves") and fid in model.curves:
            return model.curves[fid]
        if isinstance(model, dict):
            if "functions" in model and fid in model["functions"]:
                return model["functions"][fid]
            if "curves" in model and fid in model["curves"]:
                return model["curves"][fid]
            if fid in model:
                return model[fid]
        return None

    if p.curve_c is None and p.fun_a1:
        p.curve_c = _resolve_one(p.fun_a1)
    if p.curve_t is None and p.fun_a2:
        p.curve_t = _resolve_one(p.fun_a2)
    if p.curve_rate_c is None and p.fun_b1:
        p.curve_rate_c = _resolve_one(p.fun_b1)
    if p.curve_rate_t is None and p.fun_b2:
        p.curve_rate_t = _resolve_one(p.fun_b2)

    if p.israte == 4:
        if len(p.abg_ipt) > 0 and len(p.curves_c) == 0:
            p.curves_c = [_resolve_one(fid) for fid in p.abg_ipt]
            p.rates_c = list(p.k_a1)
        if len(p.abg_ipdel) > 0 and len(p.curves_t) == 0:
            p.curves_t = [_resolve_one(fid) for fid in p.abg_ipdel]
            p.rates_t = list(p.k_b1)

    if hasattr(mat, "params") and isinstance(mat.params, dict):
        mat.params["curve_c"] = p.curve_c
        mat.params["curve_t"] = p.curve_t
        mat.params["curve_rate_c"] = p.curve_rate_c
        mat.params["curve_rate_t"] = p.curve_rate_t
        mat.params["curves_c"] = p.curves_c
        mat.params["curves_t"] = p.curves_t


# ============================================================================
# Engine Solid Kernel: solid_update
# ============================================================================

def solid_update(mat: Any, sig: np.ndarray, deps: np.ndarray,
                 epsp: Optional[np.ndarray] = None, dt: float = 0.0,
                 extra: Optional[dict] = None,
                 return_tuple: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Radial-return solid stress update for /MAT/LAW66.

    Follows engine/source/materials/mat/mat066/sigeps66.F.
    sig, deps: (n, 6) = [xx, yy, zz, xy, yz, zx] with engineering shear.
    """
    n = sig.shape[0]
    p = _get_params(mat)

    if n == 0:
        c_empty = np.empty(0, dtype=float)
        return (sig, epsp, c_empty) if return_tuple else (sig, epsp)

    if epsp is None:
        epsp = np.zeros(n, dtype=float)

    # 1. State array: uvar66 [0: pla, 1..6: alpha (6 components), 7: filtered strain rate]
    alpha = np.zeros((n, 6), dtype=float)
    rate_filtered = np.zeros(n, dtype=float)

    if extra is not None:
        if "uvar66" in extra:
            uvar = np.atleast_2d(extra["uvar66"])
            if uvar.shape[-1] >= 7:
                alpha[:, :] = uvar[:, 1:7]
            if uvar.shape[-1] >= 8:
                rate_filtered[:] = uvar[:, 7]
        elif "sigb66" in extra:
            sigb = np.atleast_2d(extra["sigb66"])
            alpha[:, :] = sigb[:, :6]

    # 2. Hydrostatic pressure P = C1T * mu (with mu = rho/rho0 - 1)
    tr_deps = deps[:, 0] + deps[:, 1] + deps[:, 2]
    dav = tr_deps / 3.0

    has_mu = False
    if extra is not None and "rho" in extra:
        mu = np.asarray(extra["rho"], dtype=float) / p.rho0 - 1.0
        P_est = p.c1t * mu
        has_mu = True
    elif extra is not None and "mu" in extra:
        mu = np.asarray(extra["mu"], dtype=float)
        P_est = p.c1t * mu
        has_mu = True
    else:
        # Fallback to incremental trace tracking: P_new = P_old - C1T * tr_deps
        p_old = -(sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
        P_est = p_old - p.c1t * tr_deps

    # Effective moduli based on pressure P_est (sigeps66.F:211-234)
    E, G, C1 = _compute_pressure_moduli(p, P_est)
    G2 = 2.0 * G
    G3 = 3.0 * G

    # Recompute hydrostatic pressure P = C1 * mu (sigeps66.F:493)
    if has_mu:
        P = C1 * mu
    else:
        P = p_old - C1 * tr_deps

    # Dilatational sound speed (sigeps66.F:261)
    c1_sound = max(p.c1t, p.c1c)
    soundsp = np.sqrt((c1_sound + (4.0 / 3.0) * G) / max(p.rho0, _EM20))

    # 3. Deviatoric trial stress shifted by backstress alpha (sigeps66.F:241-260)
    sig_eff = sig - alpha
    p0 = -(sig_eff[:, 0] + sig_eff[:, 1] + sig_eff[:, 2]) / 3.0

    s_trial = np.empty_like(sig)
    s_trial[:, 0] = sig_eff[:, 0] + p0 + G2 * (deps[:, 0] - dav)
    s_trial[:, 1] = sig_eff[:, 1] + p0 + G2 * (deps[:, 1] - dav)
    s_trial[:, 2] = sig_eff[:, 2] + p0 + G2 * (deps[:, 2] - dav)
    s_trial[:, 3] = sig_eff[:, 3] + G * deps[:, 3]
    s_trial[:, 4] = sig_eff[:, 4] + G * deps[:, 4]
    s_trial[:, 5] = sig_eff[:, 5] + G * deps[:, 5]

    # 4. Deviatoric strain rate & filtering
    if extra is not None and "rate" in extra:
        rate = np.asarray(extra["rate"], dtype=float)
        if rate.ndim == 0:
            rate = np.full(n, float(rate))
    elif extra is not None and "epsp_rate" in extra:
        rate = np.asarray(extra["epsp_rate"], dtype=float)
        if rate.ndim == 0:
            rate = np.full(n, float(rate))
    else:
        ee = ((deps[:, 0] - dav) ** 2 + (deps[:, 1] - dav) ** 2 + (deps[:, 2] - dav) ** 2
              + 0.5 * (deps[:, 3] ** 2 + deps[:, 4] ** 2 + deps[:, 5] ** 2))
        inst_rate = np.sqrt((2.0 / 3.0) * ee) / dt if dt > 0.0 else np.zeros(n, dtype=float)

        if p.asrate > 0.0:
            omega = 2.0 * math.pi * p.asrate
            alpha_filt = min(1.0, omega * dt) if dt > 0.0 else 0.0
            rate = alpha_filt * inst_rate + (1.0 - alpha_filt) * rate_filtered
            rate_filtered[:] = rate
        else:
            rate = inst_rate

    # 5. Yield stress and hardening modulus evaluation (sigeps66.F:491-534)
    yld, H = _compute_yield_and_hardening(p, P, epsp, rate)

    # 6. J2 von Mises norm
    j2 = 0.5 * (s_trial[:, 0] ** 2 + s_trial[:, 1] ** 2 + s_trial[:, 2] ** 2) \
        + s_trial[:, 3] ** 2 + s_trial[:, 4] ** 2 + s_trial[:, 5] ** 2
    vm = np.sqrt(3.0 * j2)

    # 7. Radial return (sigeps66.F:560-574)
    R = np.minimum(1.0, yld / np.maximum(vm, _EM20))
    plastic = vm > yld
    s_new = s_trial.copy()
    dpla = np.zeros(n, dtype=float)

    if np.any(plastic):
        idx = np.where(plastic)[0]
        denom = G3[idx] + H[idx]
        denom = np.maximum(denom, _EM20)
        dpla[idx] = (vm[idx] - yld[idx]) / denom

        # Viscoplastic overstress iteration (VP == 1) (sigeps66.F:540-558)
        if p.vp > 0 and dt > 0.0:
            epd = np.maximum(_EM20, (dpla[idx] / dt) / p.epsp0)
            power = p.cp if p.cp != 0.0 else 1.0
            yrate = 1.0 + epd ** power
            if p.sigy == 0.0:
                yld[idx] = yld[idx] * yrate
            else:
                yld[idx] = yld[idx] + p.sigy * (yrate - 1.0)
            R[idx] = np.minimum(1.0, yld[idx] / np.maximum(vm[idx], _EM20))
            dpla[idx] = (1.0 - R[idx]) * vm[idx] / denom

        epsp[idx] += dpla[idx]
        for k in range(6):
            s_new[idx, k] = s_trial[idx, k] * R[idx]

        # 8. Kinematic hardening update (sigeps66.F:670-702)
        if p.chard > 0.0:
            fisokin = p.chard
            hkin = (2.0 / 3.0) * fisokin * H[idx]
            alpha_fac = hkin / np.maximum(G2[idx] + hkin, _EM20)
            ds = s_trial[idx] - s_new[idx]
            delta_alpha = alpha_fac[:, None] * ds
            alpha[idx] += delta_alpha

    # 9. Total Cauchy stress: sig_new = s_new + alpha - P * I
    sig[:, :] = s_new + alpha
    sig[:, 0] -= P
    sig[:, 1] -= P
    sig[:, 2] -= P

    # Update extra persistent variables
    if extra is not None:
        if "uvar66" in extra:
            extra["uvar66"][:, 0] = epsp
            extra["uvar66"][:, 1:7] = alpha
            if extra["uvar66"].shape[-1] >= 8:
                extra["uvar66"][:, 7] = rate_filtered
        elif "sigb66" in extra:
            extra["sigb66"][:, :] = alpha

    return (sig, epsp, soundsp) if return_tuple else (sig, epsp)


# ============================================================================
# Engine Shell Kernel: shell_update
# ============================================================================

def shell_update(mat: Any, sig: np.ndarray, deps: np.ndarray,
                 epsp: Optional[np.ndarray] = None, dt: float = 0.0,
                 extra: Optional[dict] = None,
                 return_tuple: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Plane-stress radial projection shell update for /MAT/LAW66.

    Follows engine/source/materials/mat/mat066/sigeps66c.F.
    sig, deps: (n, 3) = [xx, yy, xy] with engineering shear.
    """
    n = sig.shape[0]
    p = _get_params(mat)

    if n == 0:
        c_empty = np.empty(0, dtype=float)
        return (sig, epsp, c_empty) if return_tuple else (sig, epsp)

    if epsp is None:
        epsp = np.zeros(n, dtype=float)

    alpha = np.zeros((n, 3), dtype=float)
    rate_filtered = np.zeros(n, dtype=float)

    if extra is not None:
        if "uvar66" in extra:
            uvar = np.atleast_2d(extra["uvar66"])
            if uvar.shape[-1] >= 4:
                alpha[:, :] = uvar[:, 1:4]
            if uvar.shape[-1] >= 5:
                rate_filtered[:] = uvar[:, 4]
        elif "sigb66" in extra:
            sigb = np.atleast_2d(extra["sigb66"])
            alpha[:, :] = sigb[:, :3]

    # 1. First estimate of plane-stress normal stresses to evaluate P (sigeps66c.F:243-247)
    sig_eff = sig - alpha
    s_est_xx = sig_eff[:, 0] + p.a11t * deps[:, 0] + p.a21t * deps[:, 1]
    s_est_yy = sig_eff[:, 1] + p.a21t * deps[:, 0] + p.a11t * deps[:, 1]
    P = -(1.0 / 3.0) * (s_est_xx + s_est_yy)

    # 2. Effective moduli based on pressure P (sigeps66c.F:250-270)
    E, G, _ = _compute_pressure_moduli(p, P)
    nu_sq = max(1.0 - p.nu ** 2, 1.0e-15)
    A11 = E / nu_sq
    A21 = p.nu * A11
    G3 = 3.0 * G
    soundsp = np.full(n, math.sqrt(p.a11t / max(p.rho0, _EM20)), dtype=float)

    # 3. Trial stress with effective A11, A21, G (sigeps66c.F:281-291)
    s_trial = np.empty_like(sig)
    s_trial[:, 0] = sig_eff[:, 0] + A11 * deps[:, 0] + A21 * deps[:, 1]
    s_trial[:, 1] = sig_eff[:, 1] + A21 * deps[:, 0] + A11 * deps[:, 1]
    s_trial[:, 2] = sig_eff[:, 2] + G * deps[:, 2]
    P = -(1.0 / 3.0) * (s_trial[:, 0] + s_trial[:, 1])

    # 4. Plane-stress strain rate & filtering
    nnu11 = p.nu / max(1.0 - p.nu, 1.0e-15)
    nu31 = 1.0 - nnu11
    dezz_est = -(deps[:, 0] + deps[:, 1]) * nnu11
    dav = (deps[:, 0] + deps[:, 1] + dezz_est) / 3.0

    if extra is not None and "rate" in extra:
        rate = np.asarray(extra["rate"], dtype=float)
        if rate.ndim == 0:
            rate = np.full(n, float(rate))
    elif extra is not None and "epsp_rate" in extra:
        rate = np.asarray(extra["epsp_rate"], dtype=float)
        if rate.ndim == 0:
            rate = np.full(n, float(rate))
    else:
        ee = ((deps[:, 0] - dav) ** 2 + (deps[:, 1] - dav) ** 2 + (dezz_est - dav) ** 2
              + 0.5 * deps[:, 2] ** 2)
        inst_rate = np.sqrt((2.0 / 3.0) * ee) / dt if dt > 0.0 else np.zeros(n, dtype=float)

        if p.asrate > 0.0:
            omega = 2.0 * math.pi * p.asrate
            alpha_filt = min(1.0, omega * dt) if dt > 0.0 else 0.0
            rate = alpha_filt * inst_rate + (1.0 - alpha_filt) * rate_filtered
            rate_filtered[:] = rate
        else:
            rate = inst_rate

    # 5. Yield stress and hardening modulus evaluation (sigeps66c.F:530-569)
    yld, H = _compute_yield_and_hardening(p, P, epsp, rate)

    # 6. Plane-stress von Mises stress
    svm = np.sqrt(s_trial[:, 0] ** 2 + s_trial[:, 1] ** 2
                  - s_trial[:, 0] * s_trial[:, 1]
                  + 3.0 * s_trial[:, 2] ** 2)

    # 7. Radial projection (sigeps66c.F:576-595)
    R = np.minimum(1.0, yld / np.maximum(svm, _EM20))
    plastic = svm > yld
    s_new = s_trial.copy()
    dpla = np.zeros(n, dtype=float)

    if np.any(plastic):
        idx = np.where(plastic)[0]
        denom = G3[idx] + H[idx]
        denom = np.maximum(denom, _EM20)
        dpla[idx] = (svm[idx] - yld[idx]) / denom

        # Viscoplastic overstress iteration (VP == 1)
        if p.vp > 0 and dt > 0.0:
            epd = np.maximum(_EM20, (dpla[idx] / dt) / p.epsp0)
            power = p.cp if p.cp != 0.0 else 1.0
            yrate = 1.0 + epd ** power
            if p.sigy == 0.0:
                yld[idx] = yld[idx] * yrate
            else:
                yld[idx] = yld[idx] + p.sigy * (yrate - 1.0)
            R[idx] = np.minimum(1.0, yld[idx] / np.maximum(svm[idx], _EM20))
            dpla[idx] = (1.0 - R[idx]) * svm[idx] / denom

        epsp[idx] += dpla[idx]
        s_new[idx, 0] = s_trial[idx, 0] * R[idx]
        s_new[idx, 1] = s_trial[idx, 1] * R[idx]
        s_new[idx, 2] = s_trial[idx, 2] * R[idx]

        # 8. Kinematic hardening update for shells (sigeps66c.F:776-798)
        if p.chard > 0.0:
            fisokin = p.chard
            hkin = (2.0 / 3.0) * fisokin * H[idx]
            alpha_fac = hkin / np.maximum(E[idx] + hkin, _EM20)
            dsxx = s_trial[idx, 0] - s_new[idx, 0]
            dsyy = s_trial[idx, 1] - s_new[idx, 1]
            dsxy = s_trial[idx, 2] - s_new[idx, 2]
            dexx = dsxx - p.nu * dsyy
            deyy = dsyy - p.nu * dsxx
            dexy = 2.0 * (1.0 + p.nu) * dsxy
            sigpxx = alpha_fac * (2.0 * dexx + deyy)
            sigpyy = alpha_fac * (2.0 * deyy + dexx)
            sigpxy = alpha_fac * dexy * 0.5
            alpha[idx, 0] += sigpxx
            alpha[idx, 1] += sigpyy
            alpha[idx, 2] += sigpxy

    # 9. Through-thickness strain dezz & thickness update (sigeps66c.F:588-593)
    s_mean_new = 0.5 * (s_new[:, 0] + s_new[:, 1])
    dezz_plas = np.where(plastic, dpla * s_mean_new / np.maximum(yld, _EM20), 0.0)
    dezz = -(deps[:, 0] + deps[:, 1]) * nnu11 - nu31 * dezz_plas
    if extra is not None:
        thkly = np.asarray(extra.get("thkly", extra.get("thklyl", extra.get("thk", 1.0))), dtype=float)
        off = np.asarray(extra.get("off", 1.0), dtype=float)
        if "thk" in extra:
            extra["thk"] += dezz * thkly * off
        if "thk66" in extra:
            extra["thk66"] += dezz * thkly * off

    # 10. Cauchy stress: sig_new = s_new + alpha (sigeps66c.F:794-796)
    sig[:, :] = s_new + alpha

    if extra is not None:
        if "uvar66" in extra:
            extra["uvar66"][:, 0] = epsp
            extra["uvar66"][:, 1:4] = alpha
            if extra["uvar66"].shape[-1] >= 5:
                extra["uvar66"][:, 4] = rate_filtered
        elif "sigb66" in extra:
            extra["sigb66"][:, :3] = alpha

    return (sig, epsp, soundsp) if return_tuple else (sig, epsp)


# Aliases for material dispatch
solid_update_law66 = solid_update
shell_update_law66 = shell_update


# ============================================================================
# Sound Speed Functions
# ============================================================================

def sound_speed_solid(mat: Any, rho: Optional[Any] = None, extra: Optional[dict] = None) -> Any:
    """Dilatational sound speed for solid elements."""
    p = _get_params(mat)
    r = rho if rho is not None else (extra.get("rho") if extra and "rho" in extra else p.rho0)
    c1 = max(p.c1t, p.c1c)
    num = c1 + (4.0 / 3.0) * p.gt
    if isinstance(r, np.ndarray):
        return np.sqrt(num / np.maximum(r, _EM20))
    return math.sqrt(num / max(float(r), _EM20))


def sound_speed_shell(mat: Any, rho: Optional[Any] = None, extra: Optional[dict] = None) -> Any:
    """Dilatational sound speed for shell elements."""
    p = _get_params(mat)
    r = rho if rho is not None else (extra.get("rho") if extra and "rho" in extra else p.rho0)
    num = p.a11t
    if isinstance(r, np.ndarray):
        return np.sqrt(num / np.maximum(r, _EM20))
    return math.sqrt(num / max(float(r), _EM20))


def sound_speed(mat: Any, rho: Optional[Any] = None, extra: Optional[dict] = None) -> Any:
    return sound_speed_solid(mat, rho=rho, extra=extra)


# ============================================================================
# Consistent Tangents for Implicit Solver
# ============================================================================

def solid_tangent(mat: Any) -> np.ndarray:
    """Constant elastic (6, 6) tangent matrix for solid elements."""
    p = _get_params(mat)
    G = p.gt
    Kb = p.c1t
    lam = Kb - (2.0 / 3.0) * G
    c11 = lam + 2.0 * G
    c12 = lam
    C = np.zeros((6, 6), dtype=float)
    C[0:3, 0:3] = c12
    np.fill_diagonal(C[0:3, 0:3], c11)
    C[3, 3] = G
    C[4, 4] = G
    C[5, 5] = G
    return C


def shell_membrane_tangent(mat: Any) -> np.ndarray:
    """Constant elastic (3, 3) plane-stress tangent matrix for shell elements."""
    p = _get_params(mat)
    c = p.a11t
    nu_c = p.a21t
    g = p.gt
    return np.array([
        [c, nu_c, 0.0],
        [nu_c, c, 0.0],
        [0.0, 0.0, g],
    ], dtype=float)


def consistent_solid_tangent(mat: Any, sig: np.ndarray, epsp: Optional[np.ndarray] = None,
                             epsp_incr: Optional[np.ndarray] = None,
                             extra: Optional[dict] = None) -> np.ndarray:
    """Algorithmic consistent elastoplastic solid tangent (Simo & Hughes Box 7.3).

    Returns (n, 6, 6) Voigt engineering shear tensor.
    """
    p = _get_params(mat)
    sig_arr = np.asarray(sig, dtype=float)
    is_1d = (sig_arr.ndim == 1)
    if is_1d:
        sig_arr = sig_arr[None, :]
    n = sig_arr.shape[0]

    C = solid_tangent(p)
    D = np.broadcast_to(C, (n, 6, 6)).copy()

    if epsp_incr is None or n == 0:
        return D[0] if is_1d else D

    epsp_incr_arr = np.asarray(epsp_incr, dtype=float)
    if epsp_incr_arr.ndim == 0:
        epsp_incr_arr = np.full(n, float(epsp_incr_arr))

    plastic = epsp_incr_arr > 0.0
    if not np.any(plastic):
        return D[0] if is_1d else D

    idx = np.where(plastic)[0]
    s = sig_arr[idx].copy()
    pm = (s[:, 0] + s[:, 1] + s[:, 2]) / 3.0
    s[:, 0] -= pm
    s[:, 1] -= pm
    s[:, 2] -= pm

    snorm = np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2
                    + 2.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
    snorm = np.maximum(snorm, 1.0e-30)
    Nv = s / snorm[:, None]
    q = np.sqrt(1.5) * snorm
    dep = epsp_incr_arr[idx]

    G = p.gt
    Kb = p.c1t
    q_tr = q + 3.0 * G * dep

    # Hardening slope H
    epsp_idx = epsp[idx] if epsp is not None else np.zeros(len(idx))
    P_est = -pm
    _, H = _compute_yield_and_hardening(p, P_est, epsp_idx, np.zeros(len(idx)))
    Hd = np.maximum(3.0 * G + H, 0.03 * G)

    a = 3.0 * G * dep / q_tr
    b = 6.0 * G * G * (dep / q_tr - 1.0 / Hd)

    ee = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    KeeT = Kb * np.outer(ee, ee)
    C_minus_vol = C - KeeT
    NN = np.einsum("mi,mj->mij", Nv, Nv)

    D[idx] = (C[None, :, :]
              - a[:, None, None] * C_minus_vol[None, :, :]
              + b[:, None, None] * NN)

    return D[0] if is_1d else D


def consistent_shell_tangent(mat: Any, sig: np.ndarray, epsp: Optional[np.ndarray] = None,
                             epsp_incr: Optional[np.ndarray] = None,
                             extra: Optional[dict] = None) -> np.ndarray:
    """Algorithmic consistent plane-stress shell tangent.

    Returns (n, 3, 3) Voigt membrane tensor [xx, yy, xy].
    """
    p = _get_params(mat)
    sig_arr = np.asarray(sig, dtype=float)
    is_1d = (sig_arr.ndim == 1)
    if is_1d:
        sig_arr = sig_arr[None, :]
    n = sig_arr.shape[0]

    C = shell_membrane_tangent(p)
    D = np.broadcast_to(C, (n, 3, 3)).copy()

    if epsp_incr is None or n == 0:
        return D[0] if is_1d else D

    epsp_incr_arr = np.asarray(epsp_incr, dtype=float)
    if epsp_incr_arr.ndim == 0:
        epsp_incr_arr = np.full(n, float(epsp_incr_arr))

    plastic = epsp_incr_arr > 0.0
    if not np.any(plastic):
        return D[0] if is_1d else D

    idx = np.where(plastic)[0]
    dl = epsp_incr_arr[idx]
    s_c = sig_arr[idx]

    G = p.gt
    sy = np.sqrt(np.maximum(np.einsum("mi,ij,mj->m", s_c, _P_PLANE, s_c), 0.0))
    sy = np.maximum(sy, 1.0e-30)
    q_tr = sy + 3.0 * G * dl
    sfac = sy / q_tr
    sig_tr = s_c / sfac[:, None]

    epsp_idx = epsp[idx] if epsp is not None else np.zeros(len(idx))
    P_est = -(1.0 / 3.0) * (s_c[:, 0] + s_c[:, 1])
    _, H = _compute_yield_and_hardening(p, P_est, epsp_idx, np.zeros(len(idx)))

    Hd = np.maximum(3.0 * G + H, 0.03 * G)
    Hfrac = (Hd - 3.0 * G) / Hd

    CP = C @ _P_PLANE
    gvec = np.einsum("ij,mj->mi", CP, sig_tr)
    coef = (Hfrac - sfac) / (q_tr * q_tr)

    D[idx] = (sfac[:, None, None] * C[None, :, :]
              + coef[:, None, None] * np.einsum("mi,mj->mij", sig_tr, gvec))

    return D[0] if is_1d else D


# ============================================================================
# History Variables Allocation
# ============================================================================

def extra_shapes(mat: Any, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Per-element persistent state arrays needed by LAW66.

    Solids (nip is None):
      - uvar66: (8,) -> [0: epsp, 1..6: alpha (6 components), 7: filtered_rate]
    Shells (nip is not None):
      - uvar66: (nip, 8) -> [0: epsp, 1..3: alpha (3 components), 4: filtered_rate]
      - thk: (nip,)
    """
    if nip is not None and nip > 0:
        return {
            "uvar66": (nip, 8),
            "thk66": (nip,),
            "thk": (nip,),
        }
    return {
        "uvar66": (8,),
    }
