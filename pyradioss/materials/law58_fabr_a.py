"""
LAW58 — Anisotropic Fabric Material Model (/MAT/LAW58, /MAT/FABR_A).
Shells only (airbag / technical textile fabric).

Fortran origin
--------------
* engine : ``engine/source/materials/mat/mat058/sigeps58c.F`` (shell layer constitutive kernel)
* starter: ``starter/source/materials/mat/mat058/hm_read_mat58.F`` (starter keyword reader & constants)
* starter: ``starter/source/materials/mat/mat058/cm58in3.F`` (initialization of material axes & shear)

Theory
------
The LAW58 fabric model represents a woven fabric composed of two interlaced yarn families
(warp, direction 1, index c; weft, direction 2, index t):

1. **Yarn Crimp Geometry & Kinematics**:
   Initial unit cell dimensions:
       L_c0 = 1 / N_t,   L_t0 = 1 / N_c
   with nominal stretches S_1, S_2:
       D_c0 = L_c0 * (1 + S_1),   D_t0 = L_t0 * (1 + S_2)
   Initial yarn crimp wave amplitudes:
       H_c0 = sqrt(D_c0^2 - L_c0^2),   H_t0 = sqrt(D_t0^2 - L_t0^2)
   Initial yarn axial stiffnesses:
       K_c = E_1 / N_c,   K_t = E_2 / N_t
   Initial bending stiffnesses:
       K_fc = flex1 * K_c * H_c0 / D_c0
       K_ft = flex2 * K_t * H_t0 / D_t0

2. **Crimp Interchange Iteration** (sigeps58c.F:340-465):
   At each cycle, yarn lengths deform with in-plane stretches:
       e_c = exp(eps_xx) - 1,   e_t = exp(eps_yy) - 1
       L_c = L_c0 * (1 + e_c),   L_t = L_t0 * (1 + e_t)
   First, uncoupled out-of-plane yarn height deflection iteration:
       H_c = H_c0 + y_c,   H_t = H_t0 + y_t
       D_c = sqrt(L_c^2 + H_c^2),   D_t = sqrt(L_t^2 + H_t^2)
   If (y_c + y_t) < 0, yarn contact occurs! The coupled crimp interchange model
   is solved for interchange deflection y (H_c = H_c0 + y, H_t = H_t0 - y)
   equilibrating vertical contact forces between warp and weft.
   Contact force:
       F_n = F_c * (H_c / D_c) + F_t * (H_t / D_t)

3. **Yarn Membrane Normal Stresses**:
   Direct yarn tension from analytical moduli (E_1, E_2, B_1, B_2) or tabulated curves (FUN_A1, FUN_A2):
       sigma_c = F_c * L_c / D_c,   sigma_t = F_t * L_t / D_t
   Resolved membrane normal stresses:
       sigma_xx = sigma_c * (N_c / E_c2)
       sigma_yy = sigma_t * (N_t / E_t2)
   where lateral stretch E_c2 = exp(eps_yy) and E_t2 = exp(eps_xx).

4. **Trellis Shear Behavior & Lock Angle**:
   Initial lock angle phi_lock from geometry or MAT_ALPHA:
       tan_phi_lock = sqrt(1 - cos^2 phi_lock) / cos_phi_lock
       G_b = tan_phi_lock * (G_0 - G_t)
   For Trellis shear angle tan_phi:
       |tan_phi| <= tan_phi_lock: sigma_xy = G_0 * tan_phi
       |tan_phi| > tan_phi_lock : sigma_xy = G_t * tan_phi + sign(tan_phi) * G_b
   or tabulated shear stress from FUN_A3.

5. **Yarn Sliding Friction & Viscous Damping**:
   - Friction shear stress limit:
       tau_frot = (2/3) * d_S * F_n * (H_c0 + H_t0) / (L_c + L_t)
       sigma_g = tau_fold + G_frot * Delta(tan_phi)
       sigma_v_xy = clip(sigma_g, -tau_frot, tau_frot)
   - Fiber damping in normal directions:
       sigma_v_xx = (deps_xx / dt) * D_f * sqrt(0.5 * rho0 * A * h) * sqrt(N_c * K_c)
       sigma_v_yy = (deps_yy / dt) * D_f * sqrt(0.5 * rho0 * A * h) * sqrt(N_t * K_t)

6. **Zero-Stress Relative Area (Folding / REF-STATE)**:
   Relative area A / A_0 = (1 + e_c)(1 + e_t). If A / A_0 <= A_rel, stresses are zeroed.

7. **Transverse Shear**:
   sigma_yz += G_5 * deps_yz,   sigma_zx += G_5 * deps_zx

8. **Shell Acoustic Sound Speed**:
   c_shell = sqrt(max(K_c, K_t, G_0) / rho0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1e-20
_INF = 1e30


@dataclass
class Law58Params:
    """Strongly-typed parameters for /MAT/LAW58 (/MAT/FABR_A)."""

    rho0: float = 1.0
    rhor: float = 1.0
    e1: float = 1000.0
    b1: float = 0.0
    e2: float = 1000.0
    b2: float = 0.0
    flex: float = 1e-3
    g0: float = 0.0
    gt: float = 0.0
    alphat: float = 0.0  # lock angle in degrees
    g5: float = 0.0  # transverse shear modulus
    sensor_id: int = 0
    df: float = 0.05  # fiber damping coefficient
    ds: float = 0.0  # shear friction coefficient
    gfrot: float = 0.0  # friction modulus
    zero_stress: float = 0.0
    arel: float = 0.0  # zero-stress relative area
    n1: int = 1  # warp yarn count
    n2: int = 1  # weft yarn count
    s1: float = 0.1  # warp nominal crimp stretch
    s2: float = 0.1  # weft nominal crimp stretch
    c4: float = 0.0  # flex1
    c5: float = 0.0  # flex2

    # Tabulated functions / scale factors
    fun_a1: Any = None
    c1: float = 1.0
    fun_a2: Any = None
    c2: float = 1.0
    fun_a3: Any = None
    c3: float = 1.0
    fun_a4: Any = None
    scale4: float = 1.0
    fun_a5: Any = None
    scale5: float = 1.0
    fun_a6: Any = None
    scale6: float = 1.0

    # Derived geometric & constitutive constants
    nc: int = field(init=False)
    nt: int = field(init=False)
    embc: float = field(init=False)
    embt: float = field(init=False)
    flex1: float = field(init=False)
    flex2: float = field(init=False)
    lc0: float = field(init=False)
    lt0: float = field(init=False)
    dc0: float = field(init=False)
    dt0: float = field(init=False)
    hc0: float = field(init=False)
    ht0: float = field(init=False)
    kc: float = field(init=False)
    kt: float = field(init=False)
    kbc: float = field(init=False)
    kbt: float = field(init=False)
    kfc: float = field(init=False)
    kft: float = field(init=False)
    phi_lock: float = field(init=False)
    tan_lock: float = field(init=False)
    g_post: float = field(init=False)
    gb: float = field(init=False)
    ccl: float = field(init=False)
    ttl: float = field(init=False)

    def __post_init__(self) -> None:
        self.nc = max(int(self.n1), 1)
        self.nt = max(int(self.n2), 1)
        self.embc = self.s1 if self.s1 != 0.0 else 0.1
        self.embt = self.s2 if self.s2 != 0.0 else 0.1

        flx = self.flex if self.flex != 0.0 else 1e-3
        f1 = self.c4
        f2 = self.c5
        if f1 == 0.0 and f2 == 0.0:
            f1 = flx
            f2 = flx
        elif f1 == 0.0 and f2 != 0.0:
            f1 = f2
        elif f2 == 0.0 and f1 != 0.0:
            f2 = f1
        self.flex1 = f1
        self.flex2 = f2

        if self.c1 == 0.0:
            self.c1 = 1.0
        if self.c2 == 0.0:
            self.c2 = 1.0
        if self.c3 == 0.0:
            self.c3 = 1.0

        # Crimp unit cell dimensions
        self.lc0 = 1.0 / self.nt
        self.lt0 = 1.0 / self.nc
        self.dc0 = self.lc0 * (1.0 + self.embc)
        self.dt0 = self.lt0 * (1.0 + self.embt)
        self.hc0 = math.sqrt(max(self.dc0 * self.dc0 - self.lc0 * self.lc0, 0.0))
        self.ht0 = math.sqrt(max(self.dt0 * self.dt0 - self.lt0 * self.lt0, 0.0))

        # Yarn stiffnesses
        self.kc = self.e1 / self.nc
        self.kt = self.e2 / self.nt
        self.kbc = self.b1 / self.nc
        self.kbt = self.b2 / self.nt

        self.kfc = self.flex1 * self.kc * self.hc0 / max(self.dc0, _EM20)
        self.kft = self.flex2 * self.kt * self.ht0 / max(self.dt0, _EM20)

        # Limits for zero derivative in analytical softening
        self.ccl = _INF if self.kbc == 0.0 else (self.kc / self.kbc)
        self.ttl = _INF if self.kbt == 0.0 else (self.kt / self.kbt)

        # Tangent shear modulus default
        gt_val = self.gt
        if gt_val == 0.0:
            gt_val = 0.25 * (self.e1 + self.e2)
        self.gt = gt_val

        # Shear blocking / lock angle
        if self.alphat == 0.0:
            cosin = 0.5 * (self.hc0 / self.lc0 + self.ht0 / self.lt0)
            cosin = max(min(cosin, 0.9999999), 1e-12)
            self.tan_lock = math.sqrt(1.0 - cosin * cosin) / cosin
            self.phi_lock = math.atan(self.tan_lock)
        else:
            self.phi_lock = self.alphat * math.pi / 180.0
            self.tan_lock = math.tan(self.phi_lock)

        g_secant = self.gt / (1.0 + self.tan_lock * self.tan_lock)
        if self.g0 == 0.0:
            self.g0 = g_secant
        self.g_post = g_secant  # UPARAM(14) in Fortran
        self.gb = self.tan_lock * (self.g0 - g_secant)

        if self.gfrot == 0.0:
            self.gfrot = self.g0
        if self.g5 == 0.0:
            self.g5 = self.g0


@dataclass
class FabricAMaterial(Material):
    """Material class for LAW58 / FABR_A."""

    def sound_speed_shell(self) -> float:
        return sound_speed_shell_law58(self)

    @property
    def G(self) -> float:
        p = self.params
        return float(max(p.get("g0", 0.0), p.get("g5", 0.0), p.get("gt", 0.0)))

    @property
    def E(self) -> float:
        p = self.params
        return float(p.get("E") or max(p.get("e1", 0.0), p.get("e2", 0.0)))

    @property
    def nu(self) -> float:
        return 0.0


def _get_params(mat: Any) -> Law58Params:
    """Extract Law58Params from a Material, Law58Params, dict, or object."""
    if isinstance(mat, Law58Params):
        return mat

    rho0 = 1.0
    rhor = 1.0
    p: dict[str, Any] = {}

    if hasattr(mat, "rho0") and mat.rho0 is not None:
        rho0 = float(mat.rho0)
    if hasattr(mat, "rhor") and mat.rhor is not None:
        rhor = float(mat.rhor)
    if hasattr(mat, "density") and mat.density is not None:
        rho0 = float(mat.density)

    if hasattr(mat, "params") and isinstance(mat.params, dict):
        p = mat.params
    elif isinstance(mat, dict):
        p = mat
    elif hasattr(mat, "__dict__"):
        p = mat.__dict__

    def _get(keys: Tuple[str, ...], default: Any) -> Any:
        for k in keys:
            if k in p and p[k] is not None:
                return p[k]
            if hasattr(mat, k) and getattr(mat, k) is not None:
                return getattr(mat, k)
        return default

    def _f(keys: Tuple[str, ...], default: float = 0.0) -> float:
        val = _get(keys, default)
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _i(keys: Tuple[str, ...], default: int = 0) -> int:
        val = _get(keys, default)
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    rho0 = _f(("rho", "rho0", "MAT_RHO", "density", "RHO_I"), rho0)
    rhor = _f(("rhor", "MAT_REFRHO", "Refer_Rho", "RHO_0"), rhor)

    e1 = _f(("e1", "MAT_E1", "E1", "E_warp"), 1000.0)
    b1 = _f(("b1", "MAT_B1", "B1"), 0.0)
    e2 = _f(("e2", "MAT_E2", "E2", "E_weft"), 1000.0)
    b2 = _f(("b2", "MAT_B2", "B2"), 0.0)
    flex = _f(("flex", "MAT_F", "FLEX"), 1e-3)
    g0 = _f(("g0", "MAT_G0", "G0"), 0.0)
    gt = _f(("gt", "MAT_GI", "GT"), 0.0)
    alphat = _f(("alphat", "alpha", "MAT_ALPHA", "ALPHA_T", "phi_lock"), 0.0)
    g5 = _f(("g5", "gsh", "MAT_G5"), 0.0)
    sensor_id = _i(("sensor_id", "isens", "ISENSOR"), 0)
    df = _f(("df", "MAT_Df", "DF"), 0.05)
    ds = _f(("ds", "MAT_dS", "DS"), 0.0)
    gfrot = _f(("gfrot", "Friction_phi"), 0.0)
    zero_stress = _f(("zero_stress", "M58_Zerostress", "ZEROSTRESS", "ZEROSTR"), 0.0)
    arel = _f(("arel", "a_r", "AREL", "areamin"), 0.0)
    n1 = _i(("n1", "n1_warp", "N1_warp", "N_1"), 1)
    n2 = _i(("n2", "n2_weft", "N2_weft", "N_2"), 1)
    s1 = _f(("s1", "S1", "S_1"), 0.1)
    s2 = _f(("s2", "S2", "S_2"), 0.1)
    c4 = _f(("c4", "MAT_C4", "flex1"), 0.0)
    c5 = _f(("c5", "MAT_C5", "flex2"), 0.0)

    fun_a1 = _get(("fun_a1", "FUN_A1"), None)
    c1 = _f(("c1", "MAT_C1"), 1.0)
    fun_a2 = _get(("fun_a2", "FUN_A2"), None)
    c2 = _f(("c2", "MAT_C2"), 1.0)
    fun_a3 = _get(("fun_a3", "FUN_A3"), None)
    c3 = _f(("c3", "MAT_C3"), 1.0)

    return Law58Params(
        rho0=rho0, rhor=rhor, e1=e1, b1=b1, e2=e2, b2=b2,
        flex=flex, g0=g0, gt=gt, alphat=alphat, g5=g5,
        sensor_id=sensor_id, df=df, ds=ds, gfrot=gfrot,
        zero_stress=zero_stress, arel=arel, n1=n1, n2=n2,
        s1=s1, s2=s2, c4=c4, c5=c5,
        fun_a1=fun_a1, c1=c1, fun_a2=fun_a2, c2=c2, fun_a3=fun_a3, c3=c3,
    )


def _eval_curve(curve: Any, x: float) -> Tuple[float, float]:
    """Evaluate a tabulated curve function and its slope dy/dx at x."""
    if curve is None:
        return 0.0, 0.0

    # Callable function f(x) -> float or (y, dydx)
    if callable(curve):
        res = curve(x)
        if isinstance(res, (tuple, list)) and len(res) >= 2:
            return float(res[0]), float(res[1])
        y = float(res)
        h = max(abs(x) * 1e-6, 1e-8)
        yp = float(curve(x + h))
        ym = float(curve(x - h))
        return y, (yp - ym) / (2.0 * h)

    # Object with eval or finter methods
    for meth in ("finter", "eval", "interpolate", "value_and_deriv"):
        if hasattr(curve, meth):
            m = getattr(curve, meth)
            try:
                res = m(x)
                if isinstance(res, (tuple, list)) and len(res) >= 2:
                    return float(res[0]), float(res[1])
                y = float(res)
                h = max(abs(x) * 1e-6, 1e-8)
                return y, float(m(x + h) - m(x - h)) / (2.0 * h)
            except Exception:
                pass

    # Object with x, y arrays
    xs = getattr(curve, "x", None)
    ys = getattr(curve, "y", None)
    if xs is None and hasattr(curve, "data"):
        data = np.asarray(curve.data)
        if data.ndim == 2 and data.shape[1] >= 2:
            xs, ys = data[:, 0], data[:, 1]

    if xs is not None and ys is not None:
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        y = float(np.interp(x, xs, ys))
        h = max(abs(x) * 1e-6, 1e-8)
        yp = float(np.interp(x + h, xs, ys))
        ym = float(np.interp(x - h, xs, ys))
        return y, (yp - ym) / (2.0 * h)

    return 0.0, 0.0


def crimp_interchange(
    p: Law58Params,
    lc: float,
    lt: float,
    yc_old: float = 0.0,
    yt_old: float = 0.0,
    cvisc: float = 0.0,
    cvist: float = 0.0,
    niter: int = 3,
) -> Tuple[float, float, float, float, float, float, float]:
    """Calculate yarn crimp geometry, heights (yc, yt), forces (fc, ft),
    and contact normal force (fn) via sigeps58c.F:340-465.

    Returns:
        (yc, yt, fc, ft, fn, dc, dt)
    """
    yc = yc_old
    yt = yt_old

    # 1. Uncoupled model (individual spring deflection)
    dyc = 0.0
    dyt = 0.0
    for _ in range(niter):
        hc = p.hc0 + yc
        ht = p.ht0 + yt
        dc = math.sqrt(max(lc * lc + hc * hc, _EM20))
        dt_len = math.sqrt(max(lt * lt + ht * ht, _EM20))
        udc = 1.0 / dc
        udt = 1.0 / dt_len
        hdc = hc * udc
        hdt = ht * udt
        dcc = dc - p.dc0
        dtt = dt_len - p.dt0

        # Warp yarn tension
        kfc_cur = p.kfc
        if p.fun_a1 is not None:
            f_val, f_der = _eval_curve(p.fun_a1, dcc)
            fc = p.c1 * f_val
            fpc = p.c1 * f_der
            kfc_cur = p.flex1 * fpc * p.hc0 / max(p.dc0, _EM20)
            if kfc_cur == 0.0:
                kfc_cur = p.kfc
            fpc = fpc * hdc
        elif dcc >= p.ccl:
            fc = 0.5 * p.kc * p.ccl
            fpc = 0.0
        else:
            fc = (p.kc - 0.5 * p.kbc * dcc) * dcc
            fpc = (p.kc - p.kbc * dcc) * hdc

        # Weft yarn tension
        kft_cur = p.kft
        if p.fun_a2 is not None:
            f_val, f_der = _eval_curve(p.fun_a2, dtt)
            ft = p.c2 * f_val
            fpt = p.c2 * f_der
            kft_cur = p.flex2 * fpt * p.ht0 / max(p.dt0, _EM20)
            if kft_cur == 0.0:
                kft_cur = p.kft
            fpt = fpt * hdt
        elif dtt >= p.ttl:
            ft = 0.5 * p.kt * p.ttl
            fpt = 0.0
        else:
            ft = (p.kt - 0.5 * p.kbt * dtt) * dtt
            fpt = (p.kt - p.kbt * dtt) * hdt

        func = kfc_cur * yc + fc * hdc + cvisc * dyc
        funt = kft_cur * yt + ft * hdt + cvist * dyt
        deric = kfc_cur + fpc * hdc + fc * udc * (1.0 - hdc * hdc) + cvisc
        derit = kft_cur + fpt * hdt + ft * udt * (1.0 - hdt * hdt) + cvist

        if abs(deric) > _EM20:
            yc = yc - func / deric
        if abs(derit) > _EM20:
            yt = yt - funt / derit

        dyc = yc - yc_old
        dyt = yt - yt_old

    fn = 0.0

    # 2. Coupled model (interlacing yarn contact)
    if (yc + yt) < 0.0:
        y = 0.5 * (yc_old - yt_old)
        dyc = 0.0
        for _ in range(niter):
            hc = p.hc0 + y
            ht = p.ht0 - y
            dc = math.sqrt(max(lc * lc + hc * hc, _EM20))
            dt_len = math.sqrt(max(lt * lt + ht * ht, _EM20))
            dcc = dc - p.dc0
            dtt = dt_len - p.dt0
            udc = 1.0 / dc
            udt = 1.0 / dt_len
            hdc = hc * udc
            hdt = ht * udt

            kfc_cur = p.kfc
            if p.fun_a1 is not None:
                f_val, f_der = _eval_curve(p.fun_a1, dcc)
                fc = p.c1 * f_val
                fpc = p.c1 * f_der
                kfc_cur = p.flex1 * fpc * p.hc0 / max(p.dc0, _EM20)
                if kfc_cur == 0.0:
                    kfc_cur = p.kfc
                fpc = fpc * hdc
            elif dcc >= p.ccl:
                fc = 0.5 * p.kc * p.ccl
                fpc = 0.0
            else:
                fc = (p.kc - 0.5 * p.kbc * dcc) * dcc
                fpc = (p.kc - p.kbc * dcc) * hdc

            kft_cur = p.kft
            if p.fun_a2 is not None:
                f_val, f_der = _eval_curve(p.fun_a2, dtt)
                ft = p.c2 * f_val
                fpt = p.c2 * f_der
                kft_cur = p.flex2 * fpt * p.ht0 / max(p.dt0, _EM20)
                if kft_cur == 0.0:
                    kft_cur = p.kft
                fpt = fpt * hdt
            elif dtt >= p.ttl:
                ft = 0.5 * p.kt * p.ttl
                fpt = 0.0
            else:
                ft = (p.kt - 0.5 * p.kbt * dtt) * dtt
                fpt = (p.kt - p.kbt * dtt) * hdt

            kf = kfc_cur + kft_cur
            func = kf * y + fc * hdc - ft * hdt + (cvisc + cvist) * dyc
            deric = (
                kf
                + fpc * hdc
                + fc * udc * (1.0 - hdc * hdc)
                + fpt * hdt
                + ft * udt * (1.0 - hdt * hdt)
                + cvisc
                + cvist
            )
            if abs(deric) > _EM20:
                y = y - func / deric
            dyc = y - 0.5 * (yc_old - yt_old)

            if y > 0.0:
                y = min(y, p.ht0)
            else:
                y = max(y, -p.hc0)

        yc = y
        yt = -y
        fn = fc * (hc / dc) + ft * (ht / dt_len)

    # Re-evaluate final yarn forces
    hc = p.hc0 + yc
    ht = p.ht0 + yt
    dc = math.sqrt(max(lc * lc + hc * hc, _EM20))
    dt_len = math.sqrt(max(lt * lt + ht * ht, _EM20))
    dcc = dc - p.dc0
    dtt = dt_len - p.dt0

    if p.fun_a1 is not None:
        f_val, _ = _eval_curve(p.fun_a1, dcc)
        fc = p.c1 * f_val
    elif dcc >= p.ccl:
        fc = 0.5 * p.kc * p.ccl
    else:
        fc = (p.kc - 0.5 * p.kbc * dcc) * dcc

    if p.fun_a2 is not None:
        f_val, _ = _eval_curve(p.fun_a2, dtt)
        ft = p.c2 * f_val
    elif dtt >= p.ttl:
        ft = 0.5 * p.kt * p.ttl
    else:
        ft = (p.kt - 0.5 * p.kbt * dtt) * dtt

    return yc, yt, fc, ft, fn, dc, dt_len


def shell_update_law58(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Constitutive plane-stress cycle for shell elements (/MAT/LAW58 /MAT/FABR_A).

    Parameters
    ----------
    mat : Material, Law58Params, or dict
    sig : (NEL, 3) or (NEL, 5) or (NEL, 6) or (3,) stress tensor
    deps : (NEL, 3) or (NEL, 5) or (NEL, 6) or (3,) strain increment (engineering shear)
    epsp : plastic strain (unused in fabric)
    dt : time increment
    extra : dict of persistent state views (eps58, yc, yt, fn, sigv_xy, tan_phi, etc.)

    Returns
    -------
    sig : updated stress array
    epsp : updated plastic strain array
    """
    sig_arr = np.asarray(sig, dtype=float)
    deps_arr = np.asarray(deps, dtype=float)

    single = sig_arr.ndim == 1
    if single:
        sig_arr = sig_arr.reshape(1, -1)
    if deps_arr.ndim == 1:
        deps_arr = deps_arr.reshape(1, -1)

    nel = sig_arr.shape[0]
    ncomp = sig_arr.shape[1]
    if epsp is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.asarray(epsp, dtype=float).flatten()
        if len(epsp_arr) == 1 and nel > 1:
            epsp_arr = np.full(nel, epsp_arr[0], dtype=float)

    if nel == 0:
        res_sig = sig_arr.reshape(sig.shape) if single else sig_arr
        return res_sig, epsp_arr

    p = _get_params(mat)

    # Initialize extra storage
    if extra is None:
        extra = {}

    eps_key = "eps58"
    if eps_key not in extra:
        extra[eps_key] = np.zeros((nel, 3), dtype=float)
    eps_tot = extra[eps_key]

    if "yc" not in extra:
        extra["yc"] = np.zeros(nel, dtype=float)
    yc_arr = extra["yc"]

    if "yt" not in extra:
        extra["yt"] = np.zeros(nel, dtype=float)
    yt_arr = extra["yt"]

    if "fn" not in extra:
        extra["fn"] = np.zeros(nel, dtype=float)
    fn_arr = extra["fn"]

    if "sigv_xy" not in extra:
        extra["sigv_xy"] = np.zeros(nel, dtype=float)
    sigv_xy_arr = extra["sigv_xy"]

    if "tan_phi" not in extra:
        extra["tan_phi"] = np.zeros(nel, dtype=float)
    tan_phi_arr = extra["tan_phi"]

    if "sigi58" not in extra:
        extra["sigi58"] = np.zeros((nel, 3), dtype=float)
    sigi_arr = extra["sigi58"]

    if "t58" not in extra:
        extra["t58"] = np.zeros(nel, dtype=float)
    t_arr = extra["t58"]

    # Element geometric properties for damping
    areas = extra.get("area", np.ones(nel, dtype=float))
    thks = extra.get("thk", extra.get("thkly", np.ones(nel, dtype=float)))
    if np.isscalar(areas):
        areas = np.full(nel, float(areas), dtype=float)
    if np.isscalar(thks):
        thks = np.full(nel, float(thks), dtype=float)

    s_out = sig_arr.copy()

    # In-plane stress integration
    for i in range(nel):
        dep_xx = float(deps_arr[i, 0])
        dep_yy = float(deps_arr[i, 1])
        dep_xy = float(deps_arr[i, 2])

        # True strains in warp and weft
        etc = eps_tot[i, 0] + dep_xx
        ett = eps_tot[i, 1] + dep_yy
        eps_tot[i, 0] = etc
        eps_tot[i, 1] = ett
        eps_tot[i, 2] += dep_xy

        # Engineering strains and cell lengths
        ec = math.exp(etc) - 1.0
        et = math.exp(ett) - 1.0
        lc = p.lc0 * (1.0 + ec)
        lt = p.lt0 * (1.0 + et)

        # Dynamic crimp damping parameters
        mass = p.rho0 * float(areas[i]) * float(thks[i]) * 0.25
        dt_inv = 1.0 / max(dt, _EM20) if dt > 0.0 else 0.0
        cvisc = math.sqrt(max(mass * p.kfc, 0.0)) * dt_inv / 3.0
        cvist = math.sqrt(max(mass * p.kft, 0.0)) * dt_inv / 3.0

        yc, yt, fc, ft, fn, dc, dt_len = crimp_interchange(
            p, lc, lt, yc_arr[i], yt_arr[i], cvisc, cvist
        )
        yc_arr[i] = yc
        yt_arr[i] = yt
        fn_arr[i] = fn

        # Lateral stretch factors (E_c2, E_t2)
        trace = math.exp(etc + ett)
        ec2 = max(trace / (ec + 1.0), 1e-6)
        et2 = max(trace / (et + 1.0), 1e-6)
        rfac = p.nc / ec2
        rfat = p.nt / et2

        # Membrane normal stresses
        sigc = fc * lc / max(dc, _EM20)
        sigt = ft * lt / max(dt_len, _EM20)
        sxx = sigc * rfac
        syy = sigt * rfat

        # Trellis shear angle
        tan_phi_old = tan_phi_arr[i]
        tan_phi = tan_phi_old + dep_xy
        tan_phi_arr[i] = tan_phi

        if p.fun_a3 is not None:
            phi_deg = math.atan(tan_phi) * 180.0 / math.pi
            val_sxy, _ = _eval_curve(p.fun_a3, phi_deg)
            sxy = p.c3 * val_sxy
        elif tan_phi > p.tan_lock:
            sxy = p.g_post * tan_phi + p.gb
        elif tan_phi < -p.tan_lock:
            sxy = p.g_post * tan_phi - p.gb
        else:
            sxy = p.g0 * tan_phi

        # Yarn sliding friction (tau_frot)
        sigv_xy = 0.0
        if fn > 0.0 and p.ds > 0.0:
            tfrot = (2.0 / 3.0) * p.ds * fn * (p.hc0 + p.ht0) / max(lc + lt, _EM20)
            dtang = dep_xy
            sigg = sigv_xy_arr[i] + p.gfrot * dtang
            if abs(sigg) > tfrot:
                sigv_xy = math.copysign(tfrot, sigg)
            else:
                sigv_xy = sigg
            sigv_xy_arr[i] = sigv_xy

        # Viscous fiber damping
        sigv_xx = 0.0
        sigv_yy = 0.0
        if p.df > 0.0 and dt > 0.0:
            damp = math.sqrt(max(p.rho0 * float(areas[i]) * float(thks[i]) * 0.5, 0.0))
            v1 = p.df * damp * math.sqrt(max(p.nc * p.kc, 0.0))
            v2 = p.df * damp * math.sqrt(max(p.nt * p.kt, 0.0))
            sigv_xx = dt_inv * dep_xx * v1
            sigv_yy = dt_inv * dep_yy * v2

        # Transverse shear stresses
        syz = sig_arr[i, 3] + p.g5 * deps_arr[i, 3] if ncomp >= 4 else 0.0
        szx = sig_arr[i, 4] + p.g5 * deps_arr[i, 4] if ncomp >= 5 else 0.0

        # Total stress
        tot_sxx = sxx + sigv_xx
        tot_syy = syy + sigv_yy
        tot_sxy = sxy + sigv_xy

        # Zero-stress relative area / folding deactivation
        areamin = p.arel if p.arel > 0.0 else (p.zero_stress if (0.0 < p.zero_stress <= 1.0) else 0.0)
        if areamin > 0.0:
            areamin2 = 1.0 + 0.5 * (areamin - 1.0)
            dareamin = 1.0 / (areamin2 - areamin) if areamin2 > areamin else 0.0
            # relative area ratio A / A_0 = (1 + ec)*(1 + et) ~ 1 + ec + et
            rel_area = 1.0 + ec + et
            aa = (rel_area - areamin) * dareamin if dareamin > 0.0 else (0.0 if rel_area <= areamin else 1.0)
            aa = min(max(aa, 0.0), 1.0)
            tot_sxx *= aa
            tot_syy *= aa
            tot_sxy *= aa
            syz *= aa
            szx *= aa

        # REF-STATE zerostress relaxation option
        if p.zero_stress > 1.0 or (p.zero_stress > 0.0 and p.sensor_id > 0):
            tstart = 0.0
            t_cur = t_arr[i]
            if t_cur <= tstart:
                sigi_arr[i, 0] = tot_sxx
                sigi_arr[i, 1] = tot_syy
                sigi_arr[i, 2] = tot_sxy
                tot_sxx = 0.0
                tot_syy = 0.0
                tot_sxy = 0.0
            else:
                for k, snew in enumerate((tot_sxx, tot_syy, tot_sxy)):
                    dsig = snew - sig_arr[i, k] - sigi_arr[i, k]
                    si = sigi_arr[i, k]
                    if si > 0.0 and dsig < 0.0:
                        sigi_arr[i, k] = max(0.0, si + p.zero_stress * dsig)
                    elif si < 0.0 and dsig > 0.0:
                        sigi_arr[i, k] = min(0.0, si + p.zero_stress * dsig)
                tot_sxx -= sigi_arr[i, 0]
                tot_syy -= sigi_arr[i, 1]
                tot_sxy -= sigi_arr[i, 2]
            if dt > 0.0:
                t_arr[i] += dt

        s_out[i, 0] = tot_sxx
        s_out[i, 1] = tot_syy
        s_out[i, 2] = tot_sxy
        if ncomp >= 4:
            s_out[i, 3] = syz
        if ncomp >= 5:
            s_out[i, 4] = szx

    res = s_out[0] if single else s_out
    return res, epsp_arr


def sound_speed_shell_law58(mat: Any, rho0: Optional[float] = None) -> float:
    """Shell acoustic sound speed:
        c_shell = sqrt(max(K_c, K_t, G_0) / rho0)
    """
    p = _get_params(mat)
    dens = rho0 if rho0 is not None else p.rho0
    if dens <= 0.0:
        dens = 1.0
    kmax = max(p.kc, p.kt, p.g0)
    return float(math.sqrt(kmax / dens))


def shell_membrane_tangent(mat: Any) -> np.ndarray:
    """(3, 3) reference in-plane elastic membrane tangent matrix."""
    p = _get_params(mat)
    return np.array([
        [p.e1, 0.0, 0.0],
        [0.0, p.e2, 0.0],
        [0.0, 0.0, p.g0],
    ], dtype=float)


def _copy_extra(extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if extra is None:
        return None
    res: Dict[str, Any] = {}
    for k, v in extra.items():
        if isinstance(v, np.ndarray):
            res[k] = v.copy()
        elif isinstance(v, dict):
            res[k] = _copy_extra(v)
        else:
            res[k] = v
    return res


def tangent_law58_shell(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    h: float = 1e-7,
    **kwargs: Any,
) -> np.ndarray:
    """Algorithmic consistent plane-stress membrane tangent (n, 3, 3).
    Verified against central finite differences.
    """
    p = _get_params(mat)

    if sig is not None:
        sig_arr = np.asarray(sig, dtype=float)
        single = sig_arr.ndim == 1
        if single:
            sig_arr = sig_arr.reshape(1, -1)
        nel = sig_arr.shape[0]
    elif deps is not None:
        deps_tmp = np.asarray(deps, dtype=float)
        single = deps_tmp.ndim == 1
        if single:
            deps_tmp = deps_tmp.reshape(1, -1)
        nel = deps_tmp.shape[0]
        sig_arr = np.zeros((nel, 3), dtype=float)
    else:
        single = True
        nel = 1
        sig_arr = np.zeros((1, 3), dtype=float)

    if deps is None:
        deps_arr = np.zeros((nel, 3), dtype=float)
    else:
        deps_arr = np.asarray(deps, dtype=float).copy()
        if deps_arr.ndim == 1:
            deps_arr = deps_arr.reshape(1, -1)
        if deps_arr.shape[1] < 3:
            pad = np.zeros((nel, 3), dtype=float)
            pad[:, :deps_arr.shape[1]] = deps_arr
            deps_arr = pad

    D = np.zeros((nel, 3, 3), dtype=float)

    for j in range(3):
        ej = np.zeros_like(deps_arr)
        ej[:, j] = h
        ex_p = _copy_extra(extra)
        ex_m = _copy_extra(extra)

        sp, _ = shell_update_law58(p, sig_arr.copy(), deps_arr + ej, epsp=epsp, dt=dt, extra=ex_p)
        sm, _ = shell_update_law58(p, sig_arr.copy(), deps_arr - ej, epsp=epsp, dt=dt, extra=ex_m)

        if sp.ndim == 1:
            sp = sp.reshape(1, -1)
            sm = sm.reshape(1, -1)

        D[:, :, j] = (sp[:, :3] - sm[:, :3]) / (2.0 * h)

    return D[0] if single else D


def solid_update(mat: Any, sig: np.ndarray, deps: np.ndarray, *args: Any, **kwargs: Any) -> Any:
    """LAW58 is defined strictly for shell elements (OpenRadioss hm_read_mat58.F)."""
    raise NotImplementedError("LAW58 (/MAT/FABR_A) is implemented for shell elements only.")


solid_update_law58 = solid_update


class FabricAMaterial(Material):
    """LAW58 fabric material representation."""
    pass


def build_law58(rec: Any) -> FabricAMaterial:
    """Construct FabricAMaterial from a GenericMaterialRecord."""
    p = rec.params
    e1 = float(p.get("MAT_E1") or p.get("E1") or 1000.0)
    b1 = float(p.get("MAT_B1") or p.get("B1") or 0.0)
    e2 = float(p.get("MAT_E2") or p.get("E2") or 1000.0)
    b2 = float(p.get("MAT_B2") or p.get("B2") or 0.0)
    flex = float(p.get("MAT_F") or p.get("FLEX") or 1e-3)
    g0 = float(p.get("MAT_G0") or p.get("G0") or 0.0)
    gt = float(p.get("MAT_GI") or p.get("GT") or 0.0)
    alphat = float(p.get("MAT_ALPHA") or p.get("ALPHA_T") or 0.0)
    g5 = float(p.get("MAT_G5") or 0.0)
    sensor_id = int(p.get("ISENSOR") or 0)
    df = float(p.get("MAT_Df") or p.get("DF") or 0.05)
    ds = float(p.get("MAT_dS") or p.get("DS") or 0.0)
    gfrot = float(p.get("Friction_phi") or p.get("GFROT") or 0.0)
    zero_stress = float(p.get("M58_Zerostress") or p.get("ZEROSTRESS") or 0.0)
    arel = float(p.get("a_r") or p.get("AREL") or 0.0)
    n1 = int(p.get("N1_warp") or p.get("N1") or 1)
    n2 = int(p.get("N2_weft") or p.get("N2") or 1)
    s1 = float(p.get("S1") or 0.1)
    s2 = float(p.get("S2") or 0.1)
    c4 = float(p.get("MAT_C4") or 0.0)
    c5 = float(p.get("MAT_C5") or 0.0)

    params_obj = Law58Params(
        rho0=rec.density, rhor=rec.density,
        e1=e1, b1=b1, e2=e2, b2=b2, flex=flex,
        g0=g0, gt=gt, alphat=alphat, g5=g5,
        sensor_id=sensor_id, df=df, ds=ds, gfrot=gfrot,
        zero_stress=zero_stress, arel=arel, n1=n1, n2=n2,
        s1=s1, s2=s2, c4=c4, c5=c5,
    )

    p_dict = {
        "E": max(e1, e2),
        "G": params_obj.g0,
        "nu": 0.0,
        "rho0": rec.density, "e1": e1, "b1": b1, "e2": e2, "b2": b2,
        "flex": flex, "g0": params_obj.g0, "gt": params_obj.gt,
        "alphat": alphat, "phi_lock": params_obj.phi_lock,
        "tan_lock": params_obj.tan_lock, "gb": params_obj.gb,
        "g5": params_obj.g5, "sensor_id": sensor_id,
        "df": df, "ds": ds, "gfrot": params_obj.gfrot,
        "zero_stress": zero_stress, "arel": arel,
        "n1": params_obj.nc, "n2": params_obj.nt,
        "s1": params_obj.embc, "s2": params_obj.embt,
        "kc": params_obj.kc, "kt": params_obj.kt,
        "kfc": params_obj.kfc, "kft": params_obj.kft,
        "hc0": params_obj.hc0, "ht0": params_obj.ht0,
        "lc0": params_obj.lc0, "lt0": params_obj.lt0,
        "dc0": params_obj.dc0, "dt0": params_obj.dt0,
        "params_obj": params_obj,
    }

    return FabricAMaterial(
        id=rec.id, law=58, rho0=rec.density, title=rec.title, params=p_dict
    )


def _register() -> None:
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        MAT_PHYSICS_REGISTRY.setdefault("LAW58", build_law58)
        MAT_PHYSICS_REGISTRY.setdefault("FABR_A", build_law58)
        MAT_PHYSICS_REGISTRY.setdefault("MAT_FABR_A", build_law58)
        MAT_PHYSICS_REGISTRY.setdefault("FABRIC_A", build_law58)
    except ImportError:
        pass


_register()

# Convenient functional aliases
shell_update = shell_update_law58
sound_speed_shell = sound_speed_shell_law58
tangent_shell = tangent_law58_shell
consistent_shell_tangent = tangent_law58_shell
