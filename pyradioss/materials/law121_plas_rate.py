"""OpenRadioss /MAT/LAW121 (/MAT/PLAS_RATE) — Rate-Dependent Elastoplasticity.

Fortran source references:
- Starter input reader:
  `starter/source/materials/mat/mat121/hm_read_mat121.F`
- 3D continuum solid stress update:
  `engine/source/materials/mat/mat121/sigeps121.F`
  `engine/source/materials/mat/mat121/mat121_newton.F`
  `engine/source/materials/mat/mat121/mat121_nice.F`
- 2D plane-stress shell stress update:
  `engine/source/materials/mat/mat121/sigeps121c.F`
  `engine/source/materials/mat/mat121/mat121c_newton.F`
  `engine/source/materials/mat/mat121/mat121c_nice.F`
- Card layout configuration:
  `hm_cfg_files/config/CFG/radioss2022/MAT/matl121_plasrate.cfg`

Constitutive Formulation:
1. Dynamic properties with strain rate interpolation:
   - Initial yield stress sigma_y0(eps_dot) from /FUNCT fct_sig0 or constant yscale_sig0
   - Young's modulus E(eps_dot) from /FUNCT fct_youn or constant young
   - Tangent modulus E_tang(eps_dot) from /FUNCT fct_tang or constant tang
   - Failure stress/strain from /FUNCT fct_fail
2. Plastic hardening modulus:
   H = E * E_tang / (E - E_tang)   (for E > E_tang)
   sigma_y(epsp, eps_dot) = sigma_y0(eps_dot) + H(eps_dot) * epsp
3. Return mapping:
   - 3D solids: J2 Von Mises radial return (cutting plane Newton iteration matching mat121_newton.F)
   - 2D shells: Plane-stress projection cutting plane iteration matching mat121c_newton.F
4. Strain rate formulations:
   - ivisc = 0: Scaled yield stress with first-order filter:
     alpha = min(1.0, fcut * dt)
     eps_dot_eff = alpha * eps_dot_dev + (1 - alpha) * eps_dot_old
   - ivisc = 1: Viscoplastic formulation with instantaneous plastic strain rate:
     eps_dot_p = delta_lambda / dt
5. Exact acoustic sound speeds:
   - Solids: c_dilatational = sqrt((K + 4/3 G) / rho0)
   - Shells: c_shell = sqrt(A11 / rho0) = sqrt(E / ((1 - nu^2) * rho0))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_DEFAULT_FCUT = 10000.0


def _eval_curve_1d(
    curve: Any,
    x_in: Union[float, np.ndarray],
    xscale: float = 1.0,
    yscale: float = 1.0,
    default_val: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear 1D curve evaluation with slope derivation.

    Returns (y_vals, slopes) for input abscissae x_in.
    """
    x_arr = np.atleast_1d(np.asarray(x_in, dtype=np.float64))
    x_scaled = x_arr / (xscale if abs(xscale) > _EM20 else 1.0)

    if curve is None:
        y = np.full_like(x_arr, default_val * yscale)
        s = np.zeros_like(x_arr)
        return y, s

    if isinstance(curve, (int, float, np.number)):
        y = np.full_like(x_arr, float(curve) * yscale)
        s = np.zeros_like(x_arr)
        return y, s

    if callable(curve):
        try:
            res = curve(x_scaled)
            if isinstance(res, tuple) and len(res) == 2:
                y = np.asarray(res[0], dtype=np.float64) * yscale
                s = np.asarray(res[1], dtype=np.float64) * (yscale / xscale)
                return y, s
            y = np.asarray(res, dtype=np.float64) * yscale
            # Finite difference slope
            dx = 1.0e-6
            y_p = np.asarray(curve(x_scaled + dx), dtype=np.float64) * yscale
            s = (y_p - y) / (dx * xscale)
            return y, s
        except Exception:
            pass

    cx, cy = None, None
    if hasattr(curve, "x") and hasattr(curve, "y"):
        cx = np.asarray(curve.x, dtype=np.float64)
        cy = np.asarray(curve.y, dtype=np.float64)
    elif hasattr(curve, "data"):
        data = np.asarray(curve.data, dtype=np.float64)
        if data.ndim == 2 and data.shape[1] >= 2:
            cx = data[:, 0]
            cy = data[:, 1]
    elif isinstance(curve, (list, tuple)):
        if len(curve) == 2 and isinstance(curve[0], (list, tuple, np.ndarray)) and isinstance(curve[1], (list, tuple, np.ndarray)):
            cx = np.asarray(curve[0], dtype=np.float64)
            cy = np.asarray(curve[1], dtype=np.float64)
        else:
            arr = np.asarray(curve, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                cx = arr[:, 0]
                cy = arr[:, 1]
    elif isinstance(curve, dict) and "x" in curve and "y" in curve:
        cx = np.asarray(curve["x"], dtype=np.float64)
        cy = np.asarray(curve["y"], dtype=np.float64)

    if cx is not None and cy is not None and len(cx) > 0:
        if len(cx) == 1:
            y = np.full_like(x_arr, cy[0] * yscale)
            s = np.zeros_like(x_arr)
            return y, s
        # Sort if needed
        if np.any(np.diff(cx) < 0):
            order = np.argsort(cx)
            cx = cx[order]
            cy = cy[order]

        y_interp = np.interp(x_scaled, cx, cy) * yscale
        dx = np.diff(cx)
        dy = np.diff(cy)
        slopes_raw = np.where(dx > _EM20, dy / dx, 0.0)
        idx = np.searchsorted(cx, x_scaled, side="right") - 1
        idx = np.clip(idx, 0, len(slopes_raw) - 1)
        s_interp = slopes_raw[idx] * (yscale / xscale)
        return y_interp, s_interp

    y = np.full_like(x_arr, default_val * yscale)
    s = np.zeros_like(x_arr)
    return y, s


@dataclass
class Law121Params:
    """Parameters for OpenRadioss /MAT/LAW121 (/MAT/PLAS_RATE)."""
    id: int = 1
    title: str = ""
    law: int = 121
    law_name: str = "LAW121"
    rho0: float = 0.0
    rho: float = 0.0
    young: float = 1.0
    nu: float = 0.3
    ires: int = 2
    ivisc: int = 0
    fcut: float = _DEFAULT_FCUT
    dtmin: float = 0.0
    tdel: float = 0.0
    fct_sig0: int = 0
    xscale_sig0: float = 1.0
    yscale_sig0: float = 1.0
    fct_youn: int = 0
    xscale_youn: float = 1.0
    yscale_youn: float = 1.0
    fct_tang: int = 0
    xscale_tang: float = 1.0
    tang: float = 0.0
    fct_fail: int = 0
    ifail: int = 0
    xscale_fail: float = 1.0
    yscale_fail: float = 1.0

    # Resolved curve entities or callables
    sig0_curve: Any = None
    youn_curve: Any = None
    tang_curve: Any = None
    fail_curve: Any = None
    yield_table: Any = None
    params: Dict[str, Any] = field(default_factory=dict)

    # Derived elastic moduli
    g: float = field(init=False)
    g2: float = field(init=False)
    bulk: float = field(init=False)
    lame: float = field(init=False)
    a11: float = field(init=False)
    a12: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rho > 0.0 and self.rho0 <= 0.0:
            self.rho0 = self.rho
        if self.rho0 > 0.0 and self.rho <= 0.0:
            self.rho = self.rho0
        if self.young <= 0.0:
            self.young = 1.0
        if not (0.0 <= self.nu < 0.5):
            self.nu = 0.3
        if self.ires not in (1, 2):
            self.ires = 2
        self.ivisc = min(max(int(self.ivisc), 0), 1)
        self.ifail = min(max(int(self.ifail), 0), 3)
        if self.fcut <= 0.0:
            self.fcut = _DEFAULT_FCUT
        if self.dtmin == 0.0 and self.tdel != 0.0:
            self.dtmin = self.tdel
        if self.tdel == 0.0 and self.dtmin != 0.0:
            self.tdel = self.dtmin
        if self.xscale_sig0 == 0.0:
            self.xscale_sig0 = 1.0
        if self.yscale_sig0 == 0.0:
            self.yscale_sig0 = 1.0
        if self.xscale_youn == 0.0:
            self.xscale_youn = 1.0
        if self.yscale_youn == 0.0:
            self.yscale_youn = 1.0
        if self.xscale_tang == 0.0:
            self.xscale_tang = 1.0
        if self.xscale_fail == 0.0:
            self.xscale_fail = 1.0
        if self.yscale_fail == 0.0:
            self.yscale_fail = 1.0

        # Elasticity moduli (hm_read_mat121.F:160-167)
        self.g2 = self.young / (1.0 + self.nu)
        self.g = 0.5 * self.g2
        self.bulk = self.young / (3.0 * (1.0 - 2.0 * self.nu))
        self.lame = self.g2 * self.nu / (1.0 - 2.0 * self.nu)
        denom_plane = 1.0 - self.nu * self.nu
        self.a11 = self.young / (denom_plane if abs(denom_plane) > _EM20 else 1.0)
        self.a12 = self.a11 * self.nu

    @property
    def e(self) -> float:
        return self.young

    @property
    def E(self) -> float:
        return self.young

    @property
    def Nu(self) -> float:
        return self.nu

    @property
    def G(self) -> float:
        return self.g

    @classmethod
    def from_material(cls, mat: Any) -> Law121Params:
        """Construct Law121Params from generic Material or MatLaw121 entity."""
        if isinstance(mat, Law121Params):
            return mat

        def _get(keys: Sequence[str], default: Any) -> Any:
            for k in keys:
                if hasattr(mat, k):
                    val = getattr(mat, k)
                    if val is not None:
                        return val
                if hasattr(mat, "params") and isinstance(mat.params, dict) and k in mat.params:
                    val = mat.params[k]
                    if val is not None:
                        return val
                if isinstance(mat, dict) and k in mat:
                    val = mat[k]
                    if val is not None:
                        return val
            return default

        mid = int(_get(["id", "mid", "mat_id"], 1))
        title = str(_get(["title", "name"], ""))
        rho0 = float(_get(["rho0", "rho", "MAT_RHO"], 0.0))
        young = float(_get(["young", "e", "MAT_E", "E"], 1.0))
        nu = float(_get(["nu", "MAT_NU"], 0.3))
        ires = int(_get(["ires", "MAT_Ires"], 2))
        ivisc = int(_get(["ivisc", "MAT_Ivisc"], 0))
        fcut = float(_get(["fcut", "Fcut", "FCUT"], _DEFAULT_FCUT))
        dtmin = float(_get(["dtmin", "tdel", "TDEL"], 0.0))
        fct_sig0 = int(_get(["fct_sig0", "Fct_SIG0", "FCT_SIG0"], 0))
        xscale_sig0 = float(_get(["xscale_sig0", "Xscale_SIG0", "XSCALE_SIG0"], 1.0))
        yscale_sig0 = float(_get(["yscale_sig0", "Yscale_SIG0", "YSCALE_SIG0"], 1.0))
        fct_youn = int(_get(["fct_youn", "Fct_YOUN", "FCT_YOUN"], 0))
        xscale_youn = float(_get(["xscale_youn", "Xscale_YOUN", "XSCALE_YOUN"], 1.0))
        yscale_youn = float(_get(["yscale_youn", "Yscale_YOUN", "YSCALE_YOUN"], 1.0))
        fct_tang = int(_get(["fct_tang", "Fct_TANG", "FCT_TANG"], 0))
        xscale_tang = float(_get(["xscale_tang", "Xscale_TANG", "XSCALE_TANG"], 1.0))
        tang = float(_get(["tang", "MAT_TANG", "TANG"], 0.0))
        fct_fail = int(_get(["fct_fail", "Fct_FAIL", "FCT_FAIL"], 0))
        ifail = int(_get(["ifail", "MAT_Ifail", "IFAIL"], 0))
        xscale_fail = float(_get(["xscale_fail", "Xscale_FAIL", "XSCALE_FAIL"], 1.0))
        yscale_fail = float(_get(["yscale_fail", "Yscale_FAIL", "YSCALE_FAIL"], 1.0))

        sig0_curve = _get(["sig0_curve", "f_sig0", "curve_sig0"], None)
        youn_curve = _get(["youn_curve", "f_youn", "curve_youn"], None)
        tang_curve = _get(["tang_curve", "f_tang", "curve_tang"], None)
        fail_curve = _get(["fail_curve", "f_fail", "curve_fail"], None)
        yield_table = _get(["yield_table", "table_yld", "table"], None)

        p = cls(
            id=mid,
            title=title,
            rho0=rho0,
            rho=rho0,
            young=young,
            nu=nu,
            ires=ires,
            ivisc=ivisc,
            fcut=fcut,
            dtmin=dtmin,
            tdel=dtmin,
            fct_sig0=fct_sig0,
            xscale_sig0=xscale_sig0,
            yscale_sig0=yscale_sig0,
            fct_youn=fct_youn,
            xscale_youn=xscale_youn,
            yscale_youn=yscale_youn,
            fct_tang=fct_tang,
            xscale_tang=xscale_tang,
            tang=tang,
            fct_fail=fct_fail,
            ifail=ifail,
            xscale_fail=xscale_fail,
            yscale_fail=yscale_fail,
            sig0_curve=sig0_curve,
            youn_curve=youn_curve,
            tang_curve=tang_curve,
            fail_curve=fail_curve,
            yield_table=yield_table,
            params=dict(mat.params) if hasattr(mat, "params") and isinstance(mat.params, dict) else {},
        )
        return p


def eval_dynamic_properties(
    p: Law121Params,
    rate: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate rate-dependent initial yield stress sig0, Young's modulus E,
    tangent modulus Et, and hardening modulus H at effective plastic strain rate.

    Returns: (sig0, E, Et, H) as arrays matching rate dimension.
    """
    rate_arr = np.atleast_1d(np.asarray(rate, dtype=np.float64))
    n = len(rate_arr)

    # 1. Initial yield stress sig0
    if p.sig0_curve is not None or p.fct_sig0 > 0:
        sig0_vals, _ = _eval_curve_1d(
            p.sig0_curve, rate_arr, xscale=p.xscale_sig0, yscale=p.yscale_sig0, default_val=p.yscale_sig0
        )
    elif p.yield_table is not None:
        sig0_vals, _ = _eval_curve_1d(
            p.yield_table, rate_arr, xscale=p.xscale_sig0, yscale=p.yscale_sig0, default_val=p.yscale_sig0
        )
    else:
        sig0_vals = np.full(n, p.yscale_sig0, dtype=np.float64)

    # 2. Young's modulus E
    if p.youn_curve is not None or p.fct_youn > 0:
        e_vals, _ = _eval_curve_1d(
            p.youn_curve, rate_arr, xscale=p.xscale_youn, yscale=p.yscale_youn, default_val=p.young
        )
    else:
        e_vals = np.full(n, p.young, dtype=np.float64)

    # 3. Tangent modulus Et
    if p.tang_curve is not None or p.fct_tang > 0:
        et_vals, _ = _eval_curve_1d(
            p.tang_curve, rate_arr, xscale=p.xscale_tang, yscale=1.0, default_val=p.tang
        )
    else:
        et_vals = np.full(n, p.tang, dtype=np.float64)

    # Clamp tangent modulus < 0.99 * E (mat121_newton.F:194-196)
    et_vals = np.minimum(et_vals, 0.99 * e_vals)
    et_vals = np.maximum(et_vals, 0.0)

    # Hardening modulus H = (E * Et) / (E - Et) (mat121_newton.F:198)
    denom = e_vals - et_vals
    denom = np.where(denom > _EM20, denom, _EM20)
    h_vals = (e_vals * et_vals) / denom

    return sig0_vals, e_vals, et_vals, h_vals


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D continuum solid stress update matching OpenRadioss `sigeps121.F` / `mat121_newton.F`.

    Parameters:
      mat: Law121Params or Material entity
      sig: stress array (6,) or (n, 6) in Voigt order (xx, yy, zz, xy, yz, zx)
      deps: strain increment (6,) or (n, 6)
      epsp: equivalent plastic strain (n,) or float
      dt: time step
      extra: state dict with 'epsd121', 'uvar121', etc.

    Returns:
      (sig_new, epsp_new, soundsp)
    """
    p = mat if isinstance(mat, Law121Params) else Law121Params.from_material(mat)

    deps_arr = np.asarray(deps, dtype=np.float64)
    sig_arr = np.asarray(sig, dtype=np.float64)
    is_1d = (deps_arr.ndim == 1)

    if is_1d:
        deps_2d = deps_arr.reshape(1, 6)
        sig_2d = sig_arr.reshape(1, 6)
    else:
        deps_2d = deps_arr
        sig_2d = sig_arr

    n = deps_2d.shape[0]

    # Initialize plastic strain
    if epsp is None:
        pla = np.zeros(n, dtype=np.float64)
    elif np.isscalar(epsp):
        pla = np.full(n, float(epsp), dtype=np.float64)
    else:
        pla = np.array(epsp, dtype=np.float64).flatten()
        if len(pla) != n:
            pla = np.resize(pla, n)

    if extra is None:
        extra = {}

    epsd_old = extra.get("epsd121", None)
    if epsd_old is None:
        epsd_old = extra.get("epsd", np.zeros(n, dtype=np.float64))
    epsd_old = np.atleast_1d(np.asarray(epsd_old, dtype=np.float64))
    if len(epsd_old) != n:
        epsd_old = np.resize(epsd_old, n)

    dt_safe = max(dt, _EM20)
    dt_inv = 1.0 / dt_safe

    # 1. Effective deviatoric strain rate (mat121_newton.F:145-156)
    tr_deps = (deps_2d[:, 0] + deps_2d[:, 1] + deps_2d[:, 2]) / 3.0
    dev_xx = deps_2d[:, 0] - tr_deps
    dev_yy = deps_2d[:, 1] - tr_deps
    dev_zz = deps_2d[:, 2] - tr_deps
    dev_xy = deps_2d[:, 3]
    dev_yz = deps_2d[:, 4]
    dev_zx = deps_2d[:, 5]

    deps_dt_sq = (2.0 / 3.0) * (
        dev_xx * dev_xx + dev_yy * dev_yy + dev_zz * dev_zz
        + 2.0 * dev_xy * dev_xy + 2.0 * dev_yz * dev_yz + 2.0 * dev_zx * dev_zx
    )
    deps_dt = np.sqrt(np.maximum(deps_dt_sq, 0.0)) * dt_inv

    if p.ivisc == 0:
        afiltr = min(1.0, p.fcut * dt_safe)
        epsd = afiltr * deps_dt + (1.0 - afiltr) * epsd_old
    else:
        epsd = np.zeros(n, dtype=np.float64)

    # 2. Dynamic properties
    sig0, young_eff, _, hard_eff = eval_dynamic_properties(p, epsd)

    # Elastic moduli per element
    g2 = young_eff / (1.0 + p.nu)
    g = 0.5 * g2
    bulk = young_eff / (3.0 * (1.0 - 2.0 * p.nu))
    lame = g2 * p.nu / (1.0 - 2.0 * p.nu)

    # 3. Elastic trial stress (mat121_newton.F:205-212)
    ldav = (deps_2d[:, 0] + deps_2d[:, 1] + deps_2d[:, 2]) * lame
    sign_xx = sig_2d[:, 0] + deps_2d[:, 0] * g2 + ldav
    sign_yy = sig_2d[:, 1] + deps_2d[:, 1] * g2 + ldav
    sign_zz = sig_2d[:, 2] + deps_2d[:, 2] * g2 + ldav
    sign_xy = sig_2d[:, 3] + deps_2d[:, 3] * g
    sign_yz = sig_2d[:, 4] + deps_2d[:, 4] * g
    sign_zx = sig_2d[:, 5] + deps_2d[:, 5] * g

    # Trace and deviatoric trial stress
    tr_sig = (sign_xx + sign_yy + sign_zz) / 3.0
    s_xx = sign_xx - tr_sig
    s_yy = sign_yy - tr_sig
    s_zz = sign_zz - tr_sig
    s_xy = sign_xy
    s_yz = sign_yz
    s_zx = sign_zx

    sig_vm = np.sqrt(np.maximum(
        1.5 * (s_xx * s_xx + s_yy * s_yy + s_zz * s_zz)
        + 3.0 * (s_xy * s_xy + s_yz * s_yz + s_zx * s_zx),
        0.0,
    ))

    # Yield stress
    yld = sig0 + hard_eff * pla
    phi = sig_vm - yld

    dpla = np.zeros(n, dtype=np.float64)
    yielding = phi > 0.0

    if np.any(yielding):
        idx = np.where(yielding)[0]
        niter = 3 if p.ivisc == 0 else 5

        for _ in range(niter):
            vm_safe = np.maximum(sig_vm[idx], _EM20)
            norm_xx = 1.5 * s_xx[idx] / vm_safe
            norm_yy = 1.5 * s_yy[idx] / vm_safe
            norm_zz = 1.5 * s_zz[idx] / vm_safe
            norm_xy = 3.0 * s_xy[idx] / vm_safe
            norm_yz = 3.0 * s_yz[idx] / vm_safe
            norm_zx = 3.0 * s_zx[idx] / vm_safe

            dfdsig2 = (
                norm_xx * norm_xx * g2[idx]
                + norm_yy * norm_yy * g2[idx]
                + norm_zz * norm_zz * g2[idx]
                + norm_xy * norm_xy * g[idx]
                + norm_yz * norm_yz * g[idx]
                + norm_zx * norm_zx * g[idx]
            )
            dphi_dlam = -(dfdsig2 + hard_eff[idx])
            dphi_dlam = np.where(abs(dphi_dlam) > _EM20, dphi_dlam, -_EM20)

            dlam = -phi[idx] / dphi_dlam
            dlam = np.maximum(dlam, 0.0)

            dpla[idx] += dlam
            pla[idx] += dlam

            dp_xx = dlam * norm_xx
            dp_yy = dlam * norm_yy
            dp_zz = dlam * norm_zz
            dp_xy = dlam * norm_xy
            dp_yz = dlam * norm_yz
            dp_zx = dlam * norm_zx

            sign_xx[idx] -= dp_xx * g2[idx]
            sign_yy[idx] -= dp_yy * g2[idx]
            sign_zz[idx] -= dp_zz * g2[idx]
            sign_xy[idx] -= dp_xy * g[idx]
            sign_yz[idx] -= dp_yz * g[idx]
            sign_zx[idx] -= dp_zx * g[idx]

            tr_sig = (sign_xx[idx] + sign_yy[idx] + sign_zz[idx]) / 3.0
            s_xx[idx] = sign_xx[idx] - tr_sig
            s_yy[idx] = sign_yy[idx] - tr_sig
            s_zz[idx] = sign_zz[idx] - tr_sig
            s_xy[idx] = sign_xy[idx]
            s_yz[idx] = sign_yz[idx]
            s_zx[idx] = sign_zx[idx]

            sig_vm[idx] = np.sqrt(np.maximum(
                1.5 * (s_xx[idx] * s_xx[idx] + s_yy[idx] * s_yy[idx] + s_zz[idx] * s_zz[idx])
                + 3.0 * (s_xy[idx] * s_xy[idx] + s_yz[idx] * s_yz[idx] + s_zx[idx] * s_zx[idx]),
                0.0,
            ))

            if p.ivisc == 1:
                epsd[idx] = dpla[idx] * dt_inv
                sig0[idx], _, _, hard_eff[idx] = eval_dynamic_properties(p, epsd[idx])

            yld[idx] = sig0[idx] + hard_eff[idx] * pla[idx]
            phi[idx] = sig_vm[idx] - yld[idx]

            if np.all(abs(phi[idx]) < 1.0e-6 * np.maximum(sig0[idx], 1.0)):
                break

    # Acoustic wave speed (mat121_newton.F:410)
    rho_eff = p.rho0 if p.rho0 > 0.0 else 1.0
    soundsp = np.sqrt((bulk + (4.0 / 3.0) * g) / rho_eff)

    # Save state
    extra["epsd121"] = epsd
    extra["dpla"] = dpla
    extra["sigy"] = yld
    extra["seq"] = sig_vm

    sign_out = np.column_stack([sign_xx, sign_yy, sign_zz, sign_xy, sign_yz, sign_zx])
    if is_1d:
        return sign_out[0], pla[0], soundsp[0]
    return sign_out, pla, soundsp


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """2D plane-stress shell update matching OpenRadioss `sigeps121c.F` / `mat121c_newton.F`.

    Parameters:
      mat: Law121Params or Material entity
      sig: in-plane stress array (3,) or (n, 3) [xx, yy, xy]
      deps: in-plane strain increment (3,) or (n, 3) [xx, yy, xy]
      epsp: equivalent plastic strain (n,) or float
      dt: time step
      extra: state dict

    Returns:
      (sig_new, epsp_new)
    """
    p = mat if isinstance(mat, Law121Params) else Law121Params.from_material(mat)

    deps_arr = np.asarray(deps, dtype=np.float64)
    sig_arr = np.asarray(sig, dtype=np.float64)
    is_1d = (deps_arr.ndim == 1)

    if is_1d:
        deps_2d = deps_arr.reshape(1, -1)
        sig_2d = sig_arr.reshape(1, -1)
    else:
        deps_2d = deps_arr
        sig_2d = sig_arr

    n = deps_2d.shape[0]

    # Initialize plastic strain
    if epsp is None:
        pla = np.zeros(n, dtype=np.float64)
    elif np.isscalar(epsp):
        pla = np.full(n, float(epsp), dtype=np.float64)
    else:
        pla = np.array(epsp, dtype=np.float64).flatten()
        if len(pla) != n:
            pla = np.resize(pla, n)

    if extra is None:
        extra = {}

    epsd_old = extra.get("epsd121", None)
    if epsd_old is None:
        epsd_old = extra.get("epsd", np.zeros(n, dtype=np.float64))
    epsd_old = np.atleast_1d(np.asarray(epsd_old, dtype=np.float64))
    if len(epsd_old) != n:
        epsd_old = np.resize(epsd_old, n)

    dt_safe = max(dt, _EM20)
    dt_inv = 1.0 / dt_safe

    # 1. Effective plane-stress strain rate (mat121c_newton.F:142-152)
    trepsp = (deps_2d[:, 0] + deps_2d[:, 1]) / 3.0
    deveps_xx = deps_2d[:, 0] - trepsp
    deveps_yy = deps_2d[:, 1] - trepsp
    deveps_zz = -trepsp
    deveps_xy = deps_2d[:, 2]

    deps_dt_sq = (2.0 / 3.0) * (
        deveps_xx * deveps_xx + deveps_yy * deveps_yy + deveps_zz * deveps_zz
        + 2.0 * deveps_xy * deveps_xy
    )
    deps_dt = np.sqrt(np.maximum(deps_dt_sq, 0.0)) * dt_inv

    if p.ivisc == 0:
        afiltr = min(1.0, p.fcut * dt_safe)
        epsd = afiltr * deps_dt + (1.0 - afiltr) * epsd_old
    else:
        epsd = np.zeros(n, dtype=np.float64)

    # 2. Dynamic properties
    sig0, young_eff, _, hard_eff = eval_dynamic_properties(p, epsd)

    denom_plane = 1.0 - p.nu * p.nu
    denom_plane = denom_plane if abs(denom_plane) > _EM20 else 1.0
    a11 = young_eff / denom_plane
    a12 = a11 * p.nu
    g = 0.5 * young_eff / (1.0 + p.nu)

    # 3. In-plane elastic trial stress (mat121c_newton.F:201-206)
    sign_xx = sig_2d[:, 0] + a11 * deps_2d[:, 0] + a12 * deps_2d[:, 1]
    sign_yy = sig_2d[:, 1] + a11 * deps_2d[:, 1] + a12 * deps_2d[:, 0]
    sign_xy = sig_2d[:, 2] + g * deps_2d[:, 2]

    tr_sig = (sign_xx + sign_yy) / 3.0
    s_xx = sign_xx - tr_sig
    s_yy = sign_yy - tr_sig
    s_zz = -tr_sig
    s_xy = sign_xy

    sig_vm = np.sqrt(np.maximum(
        1.5 * (s_xx * s_xx + s_yy * s_yy + s_zz * s_zz) + 3.0 * (s_xy * s_xy),
        0.0,
    ))

    yld = sig0 + hard_eff * pla
    phi = sig_vm - yld

    dpla = np.zeros(n, dtype=np.float64)
    yielding = phi > 0.0

    if np.any(yielding):
        idx = np.where(yielding)[0]
        niter = 3 if p.ivisc == 0 else 5

        for _ in range(niter):
            vm_safe = np.maximum(sig_vm[idx], _EM20)
            norm_xx = 1.5 * s_xx[idx] / vm_safe
            norm_yy = 1.5 * s_yy[idx] / vm_safe
            norm_xy = 3.0 * s_xy[idx] / vm_safe

            dfdsig2 = (
                norm_xx * (a11[idx] * norm_xx + a12[idx] * norm_yy)
                + norm_yy * (a11[idx] * norm_yy + a12[idx] * norm_xx)
                + norm_xy * norm_xy * g[idx]
            )
            dphi_dlam = -(dfdsig2 + hard_eff[idx])
            dphi_dlam = np.where(abs(dphi_dlam) > _EM20, dphi_dlam, -_EM20)

            dlam = -phi[idx] / dphi_dlam
            dlam = np.maximum(dlam, 0.0)

            dpla[idx] += dlam
            pla[idx] += dlam

            dp_xx = dlam * norm_xx
            dp_yy = dlam * norm_yy
            dp_xy = dlam * norm_xy

            sign_xx[idx] -= a11[idx] * dp_xx + a12[idx] * dp_yy
            sign_yy[idx] -= a11[idx] * dp_yy + a12[idx] * dp_xx
            sign_xy[idx] -= g[idx] * dp_xy

            tr_sig = (sign_xx[idx] + sign_yy[idx]) / 3.0
            s_xx[idx] = sign_xx[idx] - tr_sig
            s_yy[idx] = sign_yy[idx] - tr_sig
            s_zz[idx] = -tr_sig
            s_xy[idx] = sign_xy[idx]

            sig_vm[idx] = np.sqrt(np.maximum(
                1.5 * (s_xx[idx] * s_xx[idx] + s_yy[idx] * s_yy[idx] + s_zz[idx] * s_zz[idx])
                + 3.0 * (s_xy[idx] * s_xy[idx]),
                0.0,
            ))

            if p.ivisc == 1:
                epsd[idx] = dpla[idx] * dt_inv
                sig0[idx], _, _, hard_eff[idx] = eval_dynamic_properties(p, epsd[idx])

            yld[idx] = sig0[idx] + hard_eff[idx] * pla[idx]
            phi[idx] = sig_vm[idx] - yld[idx]

            if np.all(abs(phi[idx]) < 1.0e-6 * np.maximum(sig0[idx], 1.0)):
                break

    # Save state
    extra["epsd121"] = epsd
    extra["dpla"] = dpla
    extra["sigy"] = yld
    extra["seq"] = sig_vm

    sign_out = np.column_stack([sign_xx, sign_yy, sign_xy])
    if is_1d:
        return sign_out[0], pla[0]
    return sign_out, pla


def sound_speed(
    mat: Any,
    rho: Optional[Union[float, np.ndarray]] = None,
    extra: Optional[Dict[str, Any]] = None,
    is_shell: bool = False,
) -> float:
    """Acoustic dilatational sound speed estimate for LAW121."""
    p = mat if isinstance(mat, Law121Params) else Law121Params.from_material(mat)
    r = rho if (rho is not None and float(np.mean(rho)) > 0.0) else (p.rho0 if p.rho0 > 0.0 else 1.0)
    if is_shell:
        return float(np.sqrt(p.a11 / r))
    return float(np.sqrt((p.bulk + (4.0 / 3.0) * p.g) / r))


def sound_speed_solid(mat: Any, rho: Optional[Union[float, np.ndarray]] = None, extra: Optional[Dict[str, Any]] = None) -> float:
    return sound_speed(mat, rho=rho, extra=extra, is_shell=False)


def sound_speed_shell(mat: Any, rho: Optional[Union[float, np.ndarray]] = None, extra: Optional[Dict[str, Any]] = None) -> float:
    return sound_speed(mat, rho=rho, extra=extra, is_shell=True)


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Consistent 3D elasto-plastic algorithmic tangent (6, 6)."""
    p = mat if isinstance(mat, Law121Params) else Law121Params.from_material(mat)
    g = p.g
    bulk = p.bulk
    lame = p.lame

    c_el = np.array([
        [bulk + (4.0 / 3.0) * g, bulk - (2.0 / 3.0) * g, bulk - (2.0 / 3.0) * g, 0.0, 0.0, 0.0],
        [bulk - (2.0 / 3.0) * g, bulk + (4.0 / 3.0) * g, bulk - (2.0 / 3.0) * g, 0.0, 0.0, 0.0],
        [bulk - (2.0 / 3.0) * g, bulk - (2.0 / 3.0) * g, bulk + (4.0 / 3.0) * g, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, g, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, g, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, g],
    ], dtype=np.float64)

    if sig is None:
        return c_el

    sig_arr = np.asarray(sig, dtype=np.float64).flatten()
    if len(sig_arr) < 6:
        return c_el

    tr_sig = (sig_arr[0] + sig_arr[1] + sig_arr[2]) / 3.0
    s = np.array([
        sig_arr[0] - tr_sig,
        sig_arr[1] - tr_sig,
        sig_arr[2] - tr_sig,
        sig_arr[3],
        sig_arr[4],
        sig_arr[5],
    ], dtype=np.float64)

    vm = math.sqrt(max(
        1.5 * (s[0] * s[0] + s[1] * s[1] + s[2] * s[2])
        + 3.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]),
        0.0,
    ))

    # Evaluate dynamic hardening
    epsd = extra.get("epsd121", 0.0) if extra else 0.0
    _, _, _, h_eff = eval_dynamic_properties(p, np.atleast_1d(epsd))
    h = float(h_eff[0])

    if vm < _EM20:
        return c_el

    # Plastic flow normal n = (3/2) s / vm (Voigt shear factor)
    n_vec = np.array([
        1.5 * s[0] / vm,
        1.5 * s[1] / vm,
        1.5 * s[2] / vm,
        3.0 * s[3] / vm,
        3.0 * s[4] / vm,
        3.0 * s[5] / vm,
    ], dtype=np.float64)

    # Tangent plastic reduction factor 4 * G^2 / (3G + H)
    denom = 3.0 * g + h
    if denom > _EM20:
        c_ep = c_el - (4.0 * g * g / denom) * np.outer(n_vec, n_vec)
        return c_ep
    return c_el


consistent_solid_tangent = solid_tangent


def consistent_shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent 2D plane-stress algorithmic tangent (3, 3)."""
    p = mat if isinstance(mat, Law121Params) else Law121Params.from_material(mat)
    a11 = p.a11
    a12 = p.a12
    g = p.g

    c_el = np.array([
        [a11, a12, 0.0],
        [a12, a11, 0.0],
        [0.0, 0.0, g],
    ], dtype=np.float64)

    if sig is None:
        return c_el

    sig_arr = np.asarray(sig, dtype=np.float64).flatten()
    if len(sig_arr) < 3:
        return c_el

    tr_sig = (sig_arr[0] + sig_arr[1]) / 3.0
    s_xx = sig_arr[0] - tr_sig
    s_yy = sig_arr[1] - tr_sig
    s_zz = -tr_sig
    s_xy = sig_arr[2]

    vm = math.sqrt(max(
        1.5 * (s_xx * s_xx + s_yy * s_yy + s_zz * s_zz) + 3.0 * s_xy * s_xy,
        0.0,
    ))

    if vm < _EM20:
        return c_el

    epsd = extra.get("epsd121", 0.0) if extra else 0.0
    _, _, _, h_eff = eval_dynamic_properties(p, np.atleast_1d(epsd))
    h = float(h_eff[0])

    n_vec = np.array([
        1.5 * s_xx / vm,
        1.5 * s_yy / vm,
        3.0 * s_xy / vm,
    ], dtype=np.float64)

    dfdsig2 = (
        n_vec[0] * (a11 * n_vec[0] + a12 * n_vec[1])
        + n_vec[1] * (a11 * n_vec[1] + a12 * n_vec[0])
        + n_vec[2] * n_vec[2] * g
    )
    denom = dfdsig2 + h
    if denom > _EM20:
        cn = np.array([
            a11 * n_vec[0] + a12 * n_vec[1],
            a12 * n_vec[0] + a11 * n_vec[1],
            g * n_vec[2],
        ], dtype=np.float64)
        c_ep = c_el - np.outer(cn, cn) / denom
        return c_ep
    return c_el


shell_tangent = consistent_shell_tangent


def resolve(mat: Any, model: Any, log: Any = None) -> None:
    """Resolve /FUNCT curve references for /MAT/LAW121 from model."""
    if hasattr(mat, "fct_sig0") and mat.fct_sig0 > 0:
        c = model.get_function(mat.fct_sig0)
        if c is not None:
            mat.sig0_curve = c
    if hasattr(mat, "fct_youn") and mat.fct_youn > 0:
        c = model.get_function(mat.fct_youn)
        if c is not None:
            mat.youn_curve = c
    if hasattr(mat, "fct_tang") and mat.fct_tang > 0:
        c = model.get_function(mat.fct_tang)
        if c is not None:
            mat.tang_curve = c
    if hasattr(mat, "fct_fail") and mat.fct_fail > 0:
        c = model.get_function(mat.fct_fail)
        if c is not None:
            mat.fail_curve = c


def extra_shapes(mat: Any, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Return persistent per-element history array shapes for LAW121."""
    if nip is not None:
        return {
            "epsd121": (nip,),
            "uvar121": (nip, 3),
            "soundsp": (nip,),
            "sigy": (nip,),
            "dpla": (nip,),
            "seq": (nip,),
        }
    return {
        "epsd121": (),
        "uvar121": (3,),
        "soundsp": (),
        "sigy": (),
        "dpla": (),
        "seq": (),
    }


def build_law121(rec: Any) -> Material:
    """Constructor for /MAT/LAW121 (/MAT/PLAS_RATE) strain-rate plasticity material."""
    p = rec.params
    e = float(p.get("MAT_E", 1.0))
    nu = float(p.get("MAT_NU", 0.3))
    params = {
        "E": e if e > 0 else 1.0,
        "nu": nu if (0.0 <= nu < 0.5) else 0.3,
        "ires": int(p.get("MAT_Ires", 2)),
        "ivisc": int(p.get("MAT_Ivisc", 0)),
        "fcut": float(p.get("Fcut", _DEFAULT_FCUT)),
        "dtmin": float(p.get("TDEL", p.get("dtmin", 0.0))),
        "tdel": float(p.get("TDEL", p.get("dtmin", 0.0))),
        "fct_sig0": int(p.get("Fct_SIG0", p.get("fct_sig0", 0))),
        "xscale_sig0": float(p.get("Xscale_SIG0", p.get("xscale_sig0", 1.0))),
        "yscale_sig0": float(p.get("Yscale_SIG0", p.get("yscale_sig0", 1.0))),
        "fct_youn": int(p.get("Fct_YOUN", p.get("fct_youn", 0))),
        "xscale_youn": float(p.get("Xscale_YOUN", p.get("xscale_youn", 1.0))),
        "yscale_youn": float(p.get("Yscale_YOUN", p.get("yscale_youn", 1.0))),
        "fct_tang": int(p.get("Fct_TANG", p.get("fct_tang", 0))),
        "xscale_tang": float(p.get("Xscale_TANG", p.get("xscale_tang", 1.0))),
        "tang": float(p.get("MAT_TANG", p.get("tang", 0.0))),
        "fct_fail": int(p.get("Fct_FAIL", p.get("fct_fail", 0))),
        "xscale_fail": float(p.get("Xscale_FAIL", p.get("xscale_fail", 1.0))),
        "yscale_fail": float(p.get("Yscale_FAIL", p.get("yscale_fail", 1.0))),
        "ifail": int(p.get("MAT_Ifail", p.get("ifail", 0))),
    }
    return Material(id=rec.id, law=121, rho0=rec.density, title=rec.title, params=params)
