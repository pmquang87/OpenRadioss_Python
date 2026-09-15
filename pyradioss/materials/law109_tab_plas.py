"""OpenRadioss /MAT/LAW109 (/MAT/TAB_PLAS / MLAW109).

Tabulated elastoplastic material law with strain rate, temperature, and
Taylor-Quinney adiabatic heating effects for 3D continuum solids and 2D shells.

Fortran source references:
- Starter input reader:
  `starter/source/materials/mat/mat109/hm_read_mat109.F`
- 3D continuum solid stress update:
  `engine/source/materials/mat/mat109/sigeps109.F`
- 2D plane-stress shell stress update:
  `engine/source/materials/mat/mat109/sigeps109c.F`
- 2D table rate-dependent interpolation:
  `engine/source/tools/curve/table2d_vinterp_log.F`
- Card layout configuration:
  `hm_cfg_files/config/CFG/radioss2021/MAT/mat109.cfg`
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

_DEFAULT_TREF = 293.0
_DEFAULT_ETA = 1.0
_DEFAULT_FCUT = 10000.0
_EPS20 = 1.0e-20
_EPS10 = 1.0e-10


@dataclass
class Law109Table:
    """Table container for rate-dependent yield curves."""
    rates: Sequence[float] = field(default_factory=list)
    curves: Sequence[Any] = field(default_factory=list)


@dataclass
class Law109Params:
    """Parameters for OpenRadioss /MAT/LAW109."""
    id: int = 1
    title: str = ""
    law: int = 109
    law_name: str = "LAW109"
    rho0: float = 0.0
    rhor: float = 0.0
    rho: float = 0.0
    refer_rho: float = 0.0
    young: float = 0.0
    nu: float = 0.0
    cp: float = 0.0
    eta: float = 1.0
    tref: float = 293.0
    tini: float = 293.0
    tab_yld: int = 0
    tab_temp: int = 0
    xscale: float = 1.0
    yscale: float = 1.0
    ismooth: int = 1
    tab_eta: int = 0
    xrate: float = 1.0
    fcut: float = 10000.0

    # Resolved curve/table objects or callables
    yield_table: Any = None
    temp_table: Any = None
    eta_table: Any = None
    params: Dict[str, Any] = field(default_factory=dict)

    # Derived elastic constants
    g: float = field(init=False)
    bulk: float = field(init=False)
    lame: float = field(init=False)
    a11: float = field(init=False)
    a12: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rho > 0.0 and self.rho0 <= 0.0:
            self.rho0 = self.rho
        if self.rho0 > 0.0 and self.rho <= 0.0:
            self.rho = self.rho0
        if self.refer_rho > 0.0 and self.rhor <= 0.0:
            self.rhor = self.refer_rho
        if self.rhor > 0.0 and self.refer_rho <= 0.0:
            self.refer_rho = self.rhor

        if self.ismooth == 0:
            self.ismooth = 1
        if self.tref <= 0.0:
            self.tref = _DEFAULT_TREF
        if self.tini <= 0.0:
            self.tini = self.tref
        if self.xscale <= 0.0:
            self.xscale = 1.0
        if self.yscale <= 0.0:
            self.yscale = 1.0
        if self.xrate <= 0.0:
            self.xrate = 1.0
        if self.eta <= 0.0:
            self.eta = 1.0
        if self.fcut <= 0.0:
            self.fcut = _DEFAULT_FCUT

        if self.young > 0.0:
            self.g = 0.5 * self.young / (1.0 + self.nu)
            denom = 1.0 - 2.0 * self.nu
            self.lame = (2.0 * self.g * self.nu / denom) if abs(denom) > 1.0e-12 else 0.0
            self.bulk = (self.young / (3.0 * denom)) if abs(denom) > 1.0e-12 else 0.0
            denom2d = 1.0 - self.nu * self.nu
            self.a11 = (self.young / denom2d) if abs(denom2d) > 1.0e-12 else self.young
            self.a12 = self.nu * self.a11
        else:
            self.g = 0.0
            self.bulk = 0.0
            self.lame = 0.0
            self.a11 = 0.0
            self.a12 = 0.0

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
    def from_material(cls, mat: Any) -> Law109Params:
        """Construct Law109Params from MaterialLaw109 or generic Material entity."""
        def _get(keys: Sequence[str], default: Any) -> Any:
            for k in keys:
                if hasattr(mat, k):
                    v = getattr(mat, k)
                    if v is not None:
                        return v
                if hasattr(mat, "params") and isinstance(mat.params, dict) and k in mat.params:
                    v = mat.params[k]
                    if v is not None:
                        return v
                if isinstance(mat, dict) and k in mat:
                    v = mat[k]
                    if v is not None:
                        return v
            return default

        mid = int(_get(["id", "mid", "mat_id"], 1))
        title = str(_get(["title", "name"], ""))
        rho0 = float(_get(["rho0", "rho", "MAT_RHO"], 0.0))
        rhor = float(_get(["rhor", "Refer_Rho"], rho0))
        young = float(_get(["young", "e", "MAT_E", "E"], 0.0))
        nu = float(_get(["nu", "MAT_NU"], 0.0))
        cp = float(_get(["cp", "MAT_SPHEAT", "spheat"], 0.0))
        eta = float(_get(["eta", "MAT_ETA"], 1.0))
        tref = float(_get(["tref", "WPREF", "T_ref"], 293.0))
        tini = float(_get(["tini", "T_Initial", "T_ini"], tref))
        tab_yld = int(_get(["tab_yld", "MAT_TAB_YLD", "tab_id_h"], 0))
        tab_temp = int(_get(["tab_temp", "MAT_TAB_TEMP", "tab_id_t"], 0))
        xscale = float(_get(["xscale", "MAT_Xscale", "xscale_h"], 1.0))
        yscale = float(_get(["yscale", "MAT_Yscale", "yscale_h"], 1.0))
        ismooth = int(_get(["ismooth", "MAT_Ismooth", "I_smooth"], 1))
        tab_eta = int(_get(["tab_eta", "TAB_ETA"], 0))
        xrate = float(_get(["xrate", "MAT_Xrate", "xscale_eta"], 1.0))
        fcut = float(_get(["fcut", "FCUT"], 10000.0))

        yield_table = _get(["yield_table", "tab_yld_obj", "fct_yld"], None)
        temp_table = _get(["temp_table", "tab_temp_obj", "fct_temp"], None)
        eta_table = _get(["eta_table", "tab_eta_obj", "fct_eta"], None)
        params_dict = getattr(mat, "params", {}) if hasattr(mat, "params") and isinstance(mat.params, dict) else {}

        return cls(
            id=mid, title=title,
            rho0=rho0, rhor=rhor, young=young, nu=nu, cp=cp, eta=eta,
            tref=tref, tini=tini, tab_yld=tab_yld, tab_temp=tab_temp,
            xscale=xscale, yscale=yscale, ismooth=ismooth, tab_eta=tab_eta,
            xrate=xrate, fcut=fcut,
            yield_table=yield_table, temp_table=temp_table, eta_table=eta_table,
            params=dict(params_dict),
        )


def _eval_curve_1d(curve: Any, x: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate 1D curve returning (value, slope)."""
    x_arr = np.asarray(x, dtype=np.float64)
    if curve is None:
        return np.zeros_like(x_arr), np.zeros_like(x_arr)
    if callable(curve):
        try:
            res = curve(x_arr)
            if isinstance(res, tuple) and len(res) == 2:
                return np.asarray(res[0], dtype=np.float64), np.asarray(res[1], dtype=np.float64)
            val = np.asarray(res, dtype=np.float64)
            # finite difference slope
            dx = 1.0e-6
            val_p = np.asarray(curve(x_arr + dx), dtype=np.float64)
            slope = (val_p - val) / dx
            return val, slope
        except Exception:
            pass

    cx, cy = None, None
    if hasattr(curve, "x") and hasattr(curve, "y"):
        cx = np.asarray(curve.x, dtype=np.float64)
        cy = np.asarray(curve.y, dtype=np.float64)
    elif (
        isinstance(curve, (tuple, list))
        and len(curve) == 2
        and isinstance(curve[0], (np.ndarray, list, tuple))
        and isinstance(curve[1], (np.ndarray, list, tuple))
    ):
        cx = np.asarray(curve[0], dtype=np.float64)
        cy = np.asarray(curve[1], dtype=np.float64)
    elif hasattr(curve, "data"):
        data = np.asarray(curve.data, dtype=np.float64)
        if data.ndim == 2 and data.shape[1] >= 2:
            cx = data[:, 0]
            cy = data[:, 1]
    elif isinstance(curve, (list, tuple, np.ndarray)):
        arr = np.asarray(curve, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            cx = arr[:, 0]
            cy = arr[:, 1]
    elif isinstance(curve, dict) and "x" in curve and "y" in curve:
        cx = np.asarray(curve["x"], dtype=np.float64)
        cy = np.asarray(curve["y"], dtype=np.float64)

    if cx is not None and cy is not None and len(cx) > 0:
        val = np.interp(x_arr, cx, cy)
        # slopes per interval
        if len(cx) > 1:
            dx = np.diff(cx)
            dy = np.diff(cy)
            slopes = np.zeros_like(cx)
            slopes[:-1] = np.where(dx > 1.0e-20, dy / dx, 0.0)
            slopes[-1] = slopes[-2]
            idx = np.searchsorted(cx, x_arr, side="right") - 1
            idx = np.clip(idx, 0, len(slopes) - 1)
            slope = slopes[idx]
        else:
            slope = np.zeros_like(val)
        return val, slope

    if isinstance(curve, (int, float)):
        c = float(curve)
        return np.full_like(x_arr, c), np.zeros_like(x_arr)

    return np.zeros_like(x_arr), np.zeros_like(x_arr)


def _is_table_active(tbl: Any) -> bool:
    """Check if table/curve object or id is provided and non-zero."""
    if tbl is None:
        return False
    if isinstance(tbl, (int, float)):
        return tbl != 0
    return True


def eval_table2d_log(
    table: Any,
    x1: np.ndarray,
    x2: np.ndarray,
    ismooth: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """2D table evaluation matching OpenRadioss `table2d_vinterp_log.F`.

    Parameters:
      table: 2D table or 1D curve or callable
      x1: first variable (effective plastic strain, epsp)
      x2: second variable (effective strain rate, epsd * xscale)
      ismooth: 1 = linear in x2, 2 = log10 in x2, 3 = ln in x2

    Returns:
      (yy, dydx1): interpolated value and partial derivative w.r.t. x1.
    """
    x1_arr = np.asarray(x1, dtype=np.float64)
    x2_arr = np.asarray(x2, dtype=np.float64)
    n = len(x1_arr)

    if not _is_table_active(table):
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)

    if callable(table):
        try:
            res = table(x1_arr, x2_arr)
            if isinstance(res, tuple) and len(res) == 2:
                return np.asarray(res[0], dtype=np.float64), np.asarray(res[1], dtype=np.float64)
            val = np.asarray(res, dtype=np.float64)
            dx = 1.0e-6
            val_p = np.asarray(table(x1_arr + dx, x2_arr), dtype=np.float64)
            dydx1 = (val_p - val) / dx
            return val, dydx1
        except TypeError:
            return _eval_curve_1d(table, x1_arr)

    # Check for 2D table structure: table with curves or 2D grid
    curves_dict = None
    if hasattr(table, "curves") and hasattr(table, "rates"):
        curves_dict = list(zip(table.rates, table.curves))
    elif isinstance(table, dict) and "rates" in table and "curves" in table:
        curves_dict = list(zip(table["rates"], table["curves"]))
    elif isinstance(table, (list, tuple)) and len(table) > 0 and isinstance(table[0], (tuple, list)) and len(table[0]) == 2:
        curves_dict = table

    if curves_dict is not None and len(curves_dict) > 0:
        curves_sorted = sorted(curves_dict, key=lambda pair: float(pair[0]))
        rates = np.array([float(p[0]) for p in curves_sorted], dtype=np.float64)
        c_list = [p[1] for p in curves_sorted]

        if len(rates) == 1:
            return _eval_curve_1d(c_list[0], x1_arr)

        yy = np.zeros(n, dtype=np.float64)
        dydx1 = np.zeros(n, dtype=np.float64)

        for i in range(n):
            val2 = x2_arr[i]
            val1 = x1_arr[i]

            if val2 <= rates[0]:
                y, d1 = _eval_curve_1d(c_list[0], val1)
                yy[i] = float(y)
                dydx1[i] = float(d1)
            elif val2 >= rates[-1]:
                y, d1 = _eval_curve_1d(c_list[-1], val1)
                yy[i] = float(y)
                dydx1[i] = float(d1)
            else:
                j = np.searchsorted(rates, val2) - 1
                j = max(0, min(j, len(rates) - 2))
                r_lo = rates[j]
                r_hi = rates[j + 1]

                y_lo, d_lo = _eval_curve_1d(c_list[j], val1)
                y_hi, d_hi = _eval_curve_1d(c_list[j + 1], val1)

                if ismooth == 2:  # log10
                    v2_clamped = max(val2, _EPS10)
                    r_lo_clamped = max(r_lo, _EPS10)
                    r_hi_clamped = max(r_hi, _EPS10)
                    denom = math.log10(r_hi_clamped) - math.log10(r_lo_clamped)
                    r2 = (math.log10(r_hi_clamped) - math.log10(v2_clamped)) / denom if abs(denom) > 1.0e-12 else 0.5
                elif ismooth == 3:  # ln
                    v2_clamped = max(val2, _EPS10)
                    r_lo_clamped = max(r_lo, _EPS10)
                    r_hi_clamped = max(r_hi, _EPS10)
                    denom = math.log(r_hi_clamped) - math.log(r_lo_clamped)
                    r2 = (math.log(r_hi_clamped) - math.log(v2_clamped)) / denom if abs(denom) > 1.0e-12 else 0.5
                else:  # linear
                    denom = r_hi - r_lo
                    r2 = (r_hi - val2) / denom if abs(denom) > 1.0e-12 else 0.5

                r2 = max(0.0, min(1.0, r2))
                unr2 = 1.0 - r2
                yy[i] = r2 * float(y_lo) + unr2 * float(y_hi)
                dydx1[i] = r2 * float(d_lo) + unr2 * float(d_hi)

        return yy, dydx1

    return _eval_curve_1d(table, x1_arr)


def eval_table_temp(
    table: Any,
    pla: np.ndarray,
    temp: np.ndarray,
    tref: float,
) -> np.ndarray:
    """Evaluate temperature scale factor TFAC = YLD_TEMP / YLD_TREF."""
    if not _is_table_active(table):
        return np.ones_like(pla, dtype=np.float64)

    pla_arr = np.asarray(pla, dtype=np.float64)
    temp_arr = np.asarray(temp, dtype=np.float64)
    tref_arr = np.full_like(temp_arr, tref)

    if callable(table):
        try:
            y_tref = np.asarray(table(pla_arr, tref_arr), dtype=np.float64)
            y_temp = np.asarray(table(pla_arr, temp_arr), dtype=np.float64)
            denom = np.where(abs(y_tref) > 1.0e-20, y_tref, 1.0)
            return y_temp / denom
        except TypeError:
            pass

    y_tref, _ = _eval_curve_1d(table, tref_arr)
    y_temp, _ = _eval_curve_1d(table, temp_arr)
    denom = np.where(abs(y_tref) > 1.0e-20, y_tref, 1.0)
    return y_temp / denom


def eval_table_eta(
    table: Any,
    rate_eta: np.ndarray,
    temp: np.ndarray,
    pla: np.ndarray,
    eta_base: float = 1.0,
    eta: Optional[float] = None,
) -> np.ndarray:
    """Evaluate Taylor-Quinney thermal conversion factor FTHERM."""
    if eta is not None:
        eta_base = eta
    if not _is_table_active(table):
        return np.full_like(pla, min(eta_base, 1.0), dtype=np.float64)

    rate_arr = np.asarray(rate_eta, dtype=np.float64)
    temp_arr = np.asarray(temp, dtype=np.float64)
    pla_arr = np.asarray(pla, dtype=np.float64)

    if callable(table):
        try:
            res = np.asarray(table(rate_arr, temp_arr, pla_arr), dtype=np.float64)
            return np.clip(eta_base * res, 0.0, 1.0)
        except TypeError:
            try:
                res = np.asarray(table(rate_arr, temp_arr), dtype=np.float64)
                return np.clip(eta_base * res, 0.0, 1.0)
            except TypeError:
                pass

    fact_eta, _ = _eval_curve_1d(table, rate_arr)
    return np.clip(eta_base * fact_eta, 0.0, 1.0)


# ============================================================================
# 3D Solid Continuum Stress Update (`sigeps109.F`)
# ============================================================================

def solid_update(
    mat: Any,
    deps: Any = None,
    sigo: Any = None,
    extra: Optional[Dict[str, Any]] = None,
    dt: float = 1.0e-6,
    rho: Optional[Union[float, np.ndarray]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """3D solid continuum stress update matching OpenRadioss `sigeps109.F`.

    Parameters:
      mat: Law109Params or MaterialLaw109 or generic material entity
      deps: strain increments (6,) or (n, 6) in Voigt order (xx, yy, zz, xy, yz, zx)
      sigo: old stresses (6,) or (n, 6)
      extra: state dict with 'pla', 'uvar' (epsd), 'temp', etc.
      dt: time step
      rho: density

    Returns:
      (sign, extra_new)
    """
    if sigo is None and "sig" in kwargs:
        sigo = kwargs.pop("sig")
    if deps is None and "deps" in kwargs:
        deps = kwargs.pop("deps")
    if extra is None and "extra" in kwargs:
        extra = kwargs.pop("extra")
    if isinstance(dt, dict) and extra is None:
        extra = dt
        dt = kwargs.pop("dt", 1.0e-6)
    if isinstance(extra, (int, float, np.number, np.ndarray)) or extra is None:
        if "extra" in kwargs and isinstance(kwargs["extra"], dict):
            epsp_in = extra
            extra = kwargs.pop("extra")
            deps, sigo = sigo, deps
            if epsp_in is not None and "pla" not in extra:
                extra["pla"] = epsp_in
    if extra is None:
        extra = {}
    if "epsp" in kwargs:
        epsp_val = kwargs.pop("epsp")
        if epsp_val is not None and "pla" not in extra:
            extra["pla"] = epsp_val

    p = mat if isinstance(mat, Law109Params) else Law109Params.from_material(mat)

    deps_arr = np.asarray(deps if deps is not None else np.zeros(6), dtype=np.float64)
    sigo_arr = np.asarray(sigo if sigo is not None else np.zeros(6), dtype=np.float64)
    is_1d = (deps_arr.ndim == 1)

    if is_1d:
        deps_2d = deps_arr.reshape(1, 6)
        sigo_2d = sigo_arr.reshape(1, 6)
    else:
        deps_2d = deps_arr
        sigo_2d = sigo_arr

    n = deps_2d.shape[0]

    # Density
    if rho is None:
        rho_arr = np.full(n, p.rho0 if p.rho0 > 0.0 else 1.0, dtype=np.float64)
    elif isinstance(rho, (int, float)):
        rho_arr = np.full(n, float(rho), dtype=np.float64)
    else:
        rho_arr = np.asarray(rho, dtype=np.float64).reshape(-1)
        if len(rho_arr) != n:
            rho_arr = np.full(n, rho_arr[0] if len(rho_arr) > 0 else p.rho0, dtype=np.float64)

    # State variables
    pla = extra.get("pla")
    if pla is None:
        pla_arr = np.zeros(n, dtype=np.float64)
    else:
        pla_arr = np.asarray(pla, dtype=np.float64).reshape(-1)
        if len(pla_arr) != n:
            pla_arr = np.zeros(n, dtype=np.float64)

    epsd = extra.get("epsd", extra.get("uvar"))
    if epsd is None:
        epsd_arr = np.zeros(n, dtype=np.float64)
    else:
        epsd_arr = np.asarray(epsd, dtype=np.float64).reshape(-1)
        if len(epsd_arr) != n:
            epsd_arr = np.zeros(n, dtype=np.float64)

    temp = extra.get("temp")
    if temp is None:
        temp_arr = np.full(n, p.tini, dtype=np.float64)
    else:
        temp_arr = np.asarray(temp, dtype=np.float64).reshape(-1)
        if len(temp_arr) != n:
            temp_arr = np.full(n, p.tini, dtype=np.float64)

    off = extra.get("off", np.ones(n, dtype=np.float64))
    off_arr = np.asarray(off, dtype=np.float64).reshape(-1)
    if len(off_arr) != n:
        off_arr = np.ones(n, dtype=np.float64)

    g = p.g
    g2 = 2.0 * g
    lame = p.lame
    young = p.young
    dt_safe = max(dt, _EPS20)
    dtinv = 1.0 / dt_safe

    # Strain rate filtering parameter ASRATE = 2*pi*FCUT
    asrate = 2.0 * math.pi * p.fcut
    alpha = min(asrate * dt_safe, 1.0)
    alphi = 1.0 - alpha

    pla0 = pla_arr.copy()
    dpla = np.zeros(n, dtype=np.float64)

    # 1. Thermal conversion factor
    ftherm = eval_table_eta(p.eta_table, epsd_arr * p.xrate, temp_arr, pla_arr, p.eta)

    # 2. Elastic trial stress
    ldav = (deps_2d[:, 0] + deps_2d[:, 1] + deps_2d[:, 2]) * lame
    sign = np.zeros_like(sigo_2d)
    sign[:, 0] = sigo_2d[:, 0] + deps_2d[:, 0] * g2 + ldav
    sign[:, 1] = sigo_2d[:, 1] + deps_2d[:, 1] * g2 + ldav
    sign[:, 2] = sigo_2d[:, 2] + deps_2d[:, 2] * g2 + ldav
    sign[:, 3] = sigo_2d[:, 3] + deps_2d[:, 3] * g
    sign[:, 4] = sigo_2d[:, 4] + deps_2d[:, 4] * g
    sign[:, 5] = sigo_2d[:, 5] + deps_2d[:, 5] * g

    # Trace and deviatoric stress
    sigm = (sign[:, 0] + sign[:, 1] + sign[:, 2]) / 3.0
    sxx = sign[:, 0] - sigm
    syy = sign[:, 1] - sigm
    szz = sign[:, 2] - sigm
    sxy = sign[:, 3]
    syz = sign[:, 4]
    szx = sign[:, 5]

    j2 = 0.5 * (sxx**2 + syy**2 + szz**2) + sxy**2 + syz**2 + szx**2
    svm = np.sqrt(3.0 * np.maximum(j2, 0.0))

    # Initial yield stress and hardening modulus from table
    yld_raw, hardp_raw = eval_table2d_log(
        p.yield_table, pla_arr, epsd_arr * p.xscale, ismooth=p.ismooth
    )
    yld = yld_raw * p.yscale
    hardp = hardp_raw * p.yscale

    # Temperature scaling
    if _is_table_active(p.temp_table):
        tfac = eval_table_temp(p.temp_table, pla_arr, temp_arr, p.tref)
        yld *= tfac
        hardp *= tfac
    else:
        tfac = np.ones(n, dtype=np.float64)

    phi = svm - yld
    yielding = (phi >= 0.0) & (off_arr == 1.0)

    # 3. Cutting plane semi-implicit plastic return (NITER = 3)
    niter = 3
    if np.any(yielding):
        yield_idx = np.where(yielding)[0]
        for _ in range(niter):
            svm_safe = np.maximum(svm[yield_idx], _EPS20)
            norm_xx = 1.5 * sxx[yield_idx] / svm_safe
            norm_yy = 1.5 * syy[yield_idx] / svm_safe
            norm_zz = 1.5 * szz[yield_idx] / svm_safe
            norm_xy = 3.0 * sxy[yield_idx] / svm_safe
            norm_yz = 3.0 * syz[yield_idx] / svm_safe
            norm_zx = 3.0 * szx[yield_idx] / svm_safe

            dfdsig2 = (
                norm_xx**2 * g2 + norm_yy**2 * g2 + norm_zz**2 * g2
                + norm_xy**2 * g + norm_yz**2 * g + norm_zx**2 * g
            )
            sig_dfdsig = (
                sign[yield_idx, 0] * norm_xx
                + sign[yield_idx, 1] * norm_yy
                + sign[yield_idx, 2] * norm_zz
                + sign[yield_idx, 3] * norm_xy
                + sign[yield_idx, 4] * norm_yz
                + sign[yield_idx, 5] * norm_zx
            )
            yld_safe = np.maximum(yld[yield_idx], _EPS20)
            dpla_dlam = sig_dfdsig / yld_safe

            dphi_dlam = -dfdsig2 - hardp[yield_idx] * dpla_dlam
            dphi_dlam = np.where(
                abs(dphi_dlam) < _EPS20,
                np.sign(dphi_dlam) * _EPS20 if np.all(dphi_dlam != 0) else -_EPS20,
                dphi_dlam,
            )

            dlam = -phi[yield_idx] / dphi_dlam
            ddep = dpla_dlam * dlam
            dpla[yield_idx] = np.maximum(dpla[yield_idx] + ddep, 0.0)
            pla_arr[yield_idx] = pla0[yield_idx] + dpla[yield_idx]

            dpxx = dlam * norm_xx
            dpyy = dlam * norm_yy
            dpzz = dlam * norm_zz
            dpxy = dlam * norm_xy
            dpyz = dlam * norm_yz
            dpzx = dlam * norm_zx

            # Update yield stress and derivative
            yld_i, hardp_i = eval_table2d_log(
                p.yield_table, pla_arr[yield_idx], epsd_arr[yield_idx] * p.xscale, ismooth=p.ismooth
            )
            yld[yield_idx] = yld_i * p.yscale * tfac[yield_idx]
            hardp[yield_idx] = hardp_i * p.yscale * tfac[yield_idx]

            # Update Cauchy stress
            sign[yield_idx, 0] -= dpxx * g2
            sign[yield_idx, 1] -= dpyy * g2
            sign[yield_idx, 2] -= dpzz * g2
            sign[yield_idx, 3] -= dpxy * g
            sign[yield_idx, 4] -= dpyz * g
            sign[yield_idx, 5] -= dpzx * g

            # Recompute invariants
            sigm[yield_idx] = (sign[yield_idx, 0] + sign[yield_idx, 1] + sign[yield_idx, 2]) / 3.0
            sxx[yield_idx] = sign[yield_idx, 0] - sigm[yield_idx]
            syy[yield_idx] = sign[yield_idx, 1] - sigm[yield_idx]
            szz[yield_idx] = sign[yield_idx, 2] - sigm[yield_idx]
            sxy[yield_idx] = sign[yield_idx, 3]
            syz[yield_idx] = sign[yield_idx, 4]
            szx[yield_idx] = sign[yield_idx, 5]

            j2_i = (
                0.5 * (sxx[yield_idx]**2 + syy[yield_idx]**2 + szz[yield_idx]**2)
                + sxy[yield_idx]**2 + syz[yield_idx]**2 + szx[yield_idx]**2
            )
            svm[yield_idx] = np.sqrt(3.0 * np.maximum(j2_i, 0.0))
            phi[yield_idx] = svm[yield_idx] - yld[yield_idx]

        # Adiabatic plastic work heating
        if p.cp > 0.0:
            temp_arr[yield_idx] += (
                ftherm[yield_idx] * yld[yield_idx] * dpla[yield_idx]
                / (p.cp * rho_arr[yield_idx])
            )

    # Plastic strain rate filtering
    epsd_arr = alpha * dpla * dtinv + alphi * epsd_arr

    # Hourglass stabilization variable
    et = hardp / (hardp + young) if young > 0.0 else np.ones(n, dtype=np.float64)

    # Sound speed
    soundsp = np.sqrt(np.maximum((p.bulk + 4.0 / 3.0 * g) / rho_arr, 0.0))

    extra_out = dict(extra)
    if is_1d:
        extra_out["pla"] = float(pla_arr[0])
        extra_out["epsd"] = float(epsd_arr[0])
        extra_out["uvar"] = float(epsd_arr[0])
        extra_out["temp"] = float(temp_arr[0])
        extra_out["seq"] = float(svm[0])
        extra_out["sigy"] = float(yld[0])
        extra_out["et"] = float(et[0])
        extra_out["soundsp"] = float(soundsp[0])
        return sign.reshape(6), extra_out
    else:
        extra_out["pla"] = pla_arr
        extra_out["epsd"] = epsd_arr
        extra_out["uvar"] = epsd_arr
        extra_out["temp"] = temp_arr
        extra_out["seq"] = svm
        extra_out["sigy"] = yld
        extra_out["et"] = et
        extra_out["soundsp"] = soundsp
        return sign, extra_out


# ============================================================================
# 2D Plane-Stress Shell Stress Update (`sigeps109c.F`)
# ============================================================================

def shell_update(
    mat: Any,
    deps: Any = None,
    sigo: Any = None,
    extra: Optional[Dict[str, Any]] = None,
    dt: float = 1.0e-6,
    rho: Optional[Union[float, np.ndarray]] = None,
    gs: Optional[Union[float, np.ndarray]] = None,
    thk: Optional[Union[float, np.ndarray]] = None,
    thkly: Optional[Union[float, np.ndarray]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """2D plane-stress shell stress update matching OpenRadioss `sigeps109c.F`.

    Parameters:
      mat: Law109Params or MaterialLaw109 or generic material entity
      deps: strain increments (5,) or (n, 5) (xx, yy, xy, yz, zx)
      sigo: old stresses (5,) or (n, 5)
      extra: state dict with 'pla', 'uvar', 'temp', 'thk', etc.
      dt: time step
      rho: density
      gs: transverse shear modulus
      thk: current thickness
      thkly: layer thickness ratio

    Returns:
      (sign, extra_new)
    """
    if sigo is None and "sig" in kwargs:
        sigo = kwargs.pop("sig")
    if deps is None and "deps" in kwargs:
        deps = kwargs.pop("deps")
    if extra is None and "extra" in kwargs:
        extra = kwargs.pop("extra")
    if isinstance(dt, dict) and extra is None:
        extra = dt
        dt = kwargs.pop("dt", 1.0e-6)
    if isinstance(extra, (int, float, np.number, np.ndarray)) or extra is None:
        if "extra" in kwargs and isinstance(kwargs["extra"], dict):
            epsp_in = extra
            extra = kwargs.pop("extra")
            deps, sigo = sigo, deps
            if epsp_in is not None and "pla" not in extra:
                extra["pla"] = epsp_in
    if extra is None:
        extra = {}
    if "epsp" in kwargs:
        epsp_val = kwargs.pop("epsp")
        if epsp_val is not None and "pla" not in extra:
            extra["pla"] = epsp_val

    p = mat if isinstance(mat, Law109Params) else Law109Params.from_material(mat)

    deps_arr = np.asarray(deps if deps is not None else np.zeros(5), dtype=np.float64)
    sigo_arr = np.asarray(sigo if sigo is not None else np.zeros(5), dtype=np.float64)
    is_1d = (deps_arr.ndim == 1)

    if is_1d:
        deps_2d = deps_arr.reshape(1, -1)
        sigo_2d = sigo_arr.reshape(1, -1)
    else:
        deps_2d = deps_arr
        sigo_2d = sigo_arr

    n = deps_2d.shape[0]

    # Density
    if rho is None:
        rho_arr = np.full(n, p.rho0 if p.rho0 > 0.0 else 1.0, dtype=np.float64)
    elif isinstance(rho, (int, float)):
        rho_arr = np.full(n, float(rho), dtype=np.float64)
    else:
        rho_arr = np.asarray(rho, dtype=np.float64).reshape(-1)
        if len(rho_arr) != n:
            rho_arr = np.full(n, rho_arr[0] if len(rho_arr) > 0 else p.rho0, dtype=np.float64)

    # Transverse shear modulus
    if gs is None:
        gs_arr = np.full(n, p.g, dtype=np.float64)
    elif isinstance(gs, (int, float)):
        gs_arr = np.full(n, float(gs), dtype=np.float64)
    else:
        gs_arr = np.asarray(gs, dtype=np.float64).reshape(-1)
        if len(gs_arr) != n:
            gs_arr = np.full(n, gs_arr[0] if len(gs_arr) > 0 else p.g, dtype=np.float64)

    # Thickness
    if thk is None:
        thk_arr = np.asarray(extra.get("thk", np.ones(n, dtype=np.float64)), dtype=np.float64).reshape(-1)
        if len(thk_arr) != n:
            thk_arr = np.ones(n, dtype=np.float64)
    elif isinstance(thk, (int, float)):
        thk_arr = np.full(n, float(thk), dtype=np.float64)
    else:
        thk_arr = np.asarray(thk, dtype=np.float64).reshape(-1)
        if len(thk_arr) != n:
            thk_arr = np.ones(n, dtype=np.float64)

    if thkly is None:
        thkly_arr = np.ones(n, dtype=np.float64)
    elif isinstance(thkly, (int, float)):
        thkly_arr = np.full(n, float(thkly), dtype=np.float64)
    else:
        thkly_arr = np.asarray(thkly, dtype=np.float64).reshape(-1)
        if len(thkly_arr) != n:
            thkly_arr = np.ones(n, dtype=np.float64)

    # State variables
    pla = extra.get("pla")
    if pla is None:
        pla_arr = np.zeros(n, dtype=np.float64)
    else:
        pla_arr = np.asarray(pla, dtype=np.float64).reshape(-1)
        if len(pla_arr) != n:
            pla_arr = np.zeros(n, dtype=np.float64)

    epsd = extra.get("epsd", extra.get("uvar"))
    if epsd is None:
        epsd_arr = np.zeros(n, dtype=np.float64)
    else:
        epsd_arr = np.asarray(epsd, dtype=np.float64).reshape(-1)
        if len(epsd_arr) != n:
            epsd_arr = np.zeros(n, dtype=np.float64)

    temp = extra.get("temp")
    if temp is None:
        temp_arr = np.full(n, p.tini, dtype=np.float64)
    else:
        temp_arr = np.asarray(temp, dtype=np.float64).reshape(-1)
        if len(temp_arr) != n:
            temp_arr = np.full(n, p.tini, dtype=np.float64)

    off = extra.get("off", np.ones(n, dtype=np.float64))
    off_arr = np.asarray(off, dtype=np.float64).reshape(-1)
    if len(off_arr) != n:
        off_arr = np.ones(n, dtype=np.float64)

    g = p.g
    a11 = p.a11
    a12 = p.a12
    young = p.young
    nu = p.nu
    dt_safe = max(dt, _EPS20)
    dtinv = 1.0 / dt_safe

    # Strain rate filtering parameter ASRATE = 2*pi*FCUT
    asrate = 2.0 * math.pi * p.fcut
    alpha = min(asrate * dt_safe, 1.0)
    alphi = 1.0 - alpha

    pla0 = pla_arr.copy()
    dpla = np.zeros(n, dtype=np.float64)
    dezz = np.zeros(n, dtype=np.float64)

    # 1. Thermal conversion factor
    ftherm = eval_table_eta(p.eta_table, epsd_arr * p.xrate, temp_arr, pla_arr, p.eta)

    # 2. Elastic trial stress
    sign = np.zeros_like(sigo_2d)
    sign[:, 0] = sigo_2d[:, 0] + a11 * deps_2d[:, 0] + a12 * deps_2d[:, 1]
    sign[:, 1] = sigo_2d[:, 1] + a11 * deps_2d[:, 1] + a12 * deps_2d[:, 0]
    sign[:, 2] = sigo_2d[:, 2] + g * deps_2d[:, 2]
    if deps_2d.shape[1] >= 5:
        sign[:, 3] = sigo_2d[:, 3] + gs_arr * deps_2d[:, 3]
        sign[:, 4] = sigo_2d[:, 4] + gs_arr * deps_2d[:, 4]

    # Trace and deviatoric stress
    sigm = (sign[:, 0] + sign[:, 1]) / 3.0
    sxx = sign[:, 0] - sigm
    syy = sign[:, 1] - sigm
    szz = -sigm
    sxy = sign[:, 2]

    j2 = 0.5 * (sxx**2 + syy**2 + szz**2 + 2.0 * sxy**2)
    svm = np.sqrt(3.0 * np.maximum(j2, 0.0))

    # Initial yield stress and hardening modulus from table
    yld_raw, hardp_raw = eval_table2d_log(
        p.yield_table, pla_arr, epsd_arr * p.xscale, ismooth=p.ismooth
    )
    yld = yld_raw * p.yscale
    hardp = hardp_raw * p.yscale

    # Temperature scaling
    if _is_table_active(p.temp_table):
        tfac = eval_table_temp(p.temp_table, pla_arr, temp_arr, p.tref)
        yld *= tfac
        hardp *= tfac
    else:
        tfac = np.ones(n, dtype=np.float64)

    phi = svm - yld
    yielding = (phi >= 0.0) & (off_arr == 1.0)

    # 3. Cutting plane semi-implicit plastic return (NITER = 3)
    niter = 3
    if np.any(yielding):
        yield_idx = np.where(yielding)[0]
        for _ in range(niter):
            svm_safe = np.maximum(svm[yield_idx], _EPS20)
            norm_xx = 1.5 * sxx[yield_idx] / svm_safe
            norm_yy = 1.5 * syy[yield_idx] / svm_safe
            norm_xy = 3.0 * sxy[yield_idx] / svm_safe

            dfdsig2 = (
                norm_xx * (a11 * norm_xx + a12 * norm_yy)
                + norm_yy * (a11 * norm_yy + a12 * norm_xx)
                + norm_xy**2 * g
            )
            sig_dfdsig = (
                sign[yield_idx, 0] * norm_xx
                + sign[yield_idx, 1] * norm_yy
                + sign[yield_idx, 2] * norm_xy
            )
            yld_safe = np.maximum(yld[yield_idx], _EPS20)
            dpla_dlam = sig_dfdsig / yld_safe

            dphi_dlam = -dfdsig2 - hardp[yield_idx] * dpla_dlam
            dphi_dlam = np.where(
                abs(dphi_dlam) < _EPS20,
                np.sign(dphi_dlam) * _EPS20 if np.all(dphi_dlam != 0) else -_EPS20,
                dphi_dlam,
            )

            dlam = -phi[yield_idx] / dphi_dlam
            ddep = dpla_dlam * dlam
            dpla[yield_idx] = np.maximum(dpla[yield_idx] + ddep, 0.0)
            pla_arr[yield_idx] = pla0[yield_idx] + dpla[yield_idx]

            dpxx = dlam * norm_xx
            dpyy = dlam * norm_yy
            dpxy = dlam * norm_xy

            # Update yield stress and derivative
            yld_i, hardp_i = eval_table2d_log(
                p.yield_table, pla_arr[yield_idx], epsd_arr[yield_idx] * p.xscale, ismooth=p.ismooth
            )
            yld[yield_idx] = yld_i * p.yscale * tfac[yield_idx]
            hardp[yield_idx] = hardp_i * p.yscale * tfac[yield_idx]

            # Update Cauchy stress
            sign[yield_idx, 0] -= a11 * dpxx + a12 * dpyy
            sign[yield_idx, 1] -= a12 * dpxx + a11 * dpyy
            sign[yield_idx, 2] -= g * dpxy

            # Accumulate plastic thickness strain
            dezz[yield_idx] = dezz[yield_idx] - dpxx - dpyy

            # Recompute invariants
            sigm[yield_idx] = (sign[yield_idx, 0] + sign[yield_idx, 1]) / 3.0
            sxx[yield_idx] = sign[yield_idx, 0] - sigm[yield_idx]
            syy[yield_idx] = sign[yield_idx, 1] - sigm[yield_idx]
            szz[yield_idx] = -sigm[yield_idx]
            sxy[yield_idx] = sign[yield_idx, 2]

            j2_i = 0.5 * (sxx[yield_idx]**2 + syy[yield_idx]**2 + szz[yield_idx]**2 + 2.0 * sxy[yield_idx]**2)
            svm[yield_idx] = np.sqrt(3.0 * np.maximum(j2_i, 0.0))
            phi[yield_idx] = svm[yield_idx] - yld[yield_idx]

        # Adiabatic plastic work heating
        if p.cp > 0.0:
            temp_arr[yield_idx] += (
                ftherm[yield_idx] * yld[yield_idx] * dpla[yield_idx]
                / (p.cp * rho_arr[yield_idx])
            )

    # Elastic thickness thinning and update
    if young > 0.0:
        dezz = -nu * (sign[:, 0] - sigo_2d[:, 0] + sign[:, 1] - sigo_2d[:, 1]) / young + dezz
    thk_arr = thk_arr + dezz * thkly_arr * off_arr

    # Plastic strain rate filtering
    epsd_arr = alpha * dpla * dtinv + alphi * epsd_arr

    # Hourglass stabilization variable
    et = hardp / (hardp + young) if young > 0.0 else np.ones(n, dtype=np.float64)

    # Sound speed
    soundsp = np.sqrt(np.maximum(a11 / rho_arr, 0.0))

    extra_out = dict(extra)
    if is_1d:
        extra_out["pla"] = float(pla_arr[0])
        extra_out["epsd"] = float(epsd_arr[0])
        extra_out["uvar"] = float(epsd_arr[0])
        extra_out["temp"] = float(temp_arr[0])
        extra_out["thk"] = float(thk_arr[0])
        extra_out["dezz"] = float(dezz[0])
        extra_out["seq"] = float(svm[0])
        extra_out["sigy"] = float(yld[0])
        extra_out["et"] = float(et[0])
        extra_out["soundsp"] = float(soundsp[0])
        return sign.reshape(-1), extra_out
    else:
        extra_out["pla"] = pla_arr
        extra_out["epsd"] = epsd_arr
        extra_out["uvar"] = epsd_arr
        extra_out["temp"] = temp_arr
        extra_out["thk"] = thk_arr
        extra_out["dezz"] = dezz
        extra_out["seq"] = svm
        extra_out["sigy"] = yld
        extra_out["et"] = et
        extra_out["soundsp"] = soundsp
        return sign, extra_out


# ============================================================================
# Wave Speeds & Tangents
# ============================================================================

def sound_speed(
    mat: Any,
    rho: Optional[Any] = None,
    is_shell: bool = False,
    **kwargs: Any,
) -> Any:
    """Acoustic wave speed for /MAT/LAW109."""
    p = mat if isinstance(mat, Law109Params) else Law109Params.from_material(mat)
    modulus = max(0.0, p.a11 if is_shell else (p.bulk + (4.0 / 3.0) * p.g))
    rho0 = p.rho0 if p.rho0 > 0.0 else 1.0
    if rho is not None:
        r = np.asarray(rho, dtype=np.float64)
        r_val = np.where(r > 0.0, r, rho0)
        c = np.sqrt(np.maximum(0.0, modulus / np.maximum(1e-20, r_val)))
        return float(c) if r.ndim == 0 else c
    return float(math.sqrt(np.maximum(0.0, modulus / max(1e-20, rho0))))


def solid_tangent(
    mat: Any,
    stress: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent 6x6 tangent modulus for 3D continuum solids."""
    if stress is None and "sig" in kwargs:
        stress = kwargs["sig"]
    p = mat if isinstance(mat, Law109Params) else Law109Params.from_material(mat)
    c_el = np.zeros((6, 6), dtype=np.float64)
    g2 = 2.0 * p.g
    lame = p.lame
    g = p.g

    c_el[0, 0] = lame + g2
    c_el[1, 1] = lame + g2
    c_el[2, 2] = lame + g2
    c_el[0, 1] = lame
    c_el[1, 0] = lame
    c_el[0, 2] = lame
    c_el[2, 0] = lame
    c_el[1, 2] = lame
    c_el[2, 1] = lame
    c_el[3, 3] = g
    c_el[4, 4] = g
    c_el[5, 5] = g

    if stress is None:
        return c_el

    sig_arr = np.asarray(stress, dtype=np.float64)
    if sig_arr.ndim == 2:
        nel = len(sig_arr)
        t_arr = np.zeros((nel, 6, 6), dtype=np.float64)
        for i in range(nel):
            t_arr[i] = solid_tangent(p, stress=sig_arr[i], extra=extra)
        return t_arr

    c = c_el.copy()
    if extra is not None:
        seq = extra.get("seq", 0.0)
        sigy = extra.get("sigy", 0.0)
        hardp = extra.get("hardp", 0.0)
        if seq >= sigy > 0.0 and hardp > 0.0:
            s = sig_arr[:6].copy()
            sigm = (s[0] + s[1] + s[2]) / 3.0
            s[0] -= sigm
            s[1] -= sigm
            s[2] -= sigm
            norm = np.array([1.5 * s[0] / seq, 1.5 * s[1] / seq, 1.5 * s[2] / seq,
                             3.0 * s[3] / seq, 3.0 * s[4] / seq, 3.0 * s[5] / seq])
            cn = c @ norm
            denom = norm @ cn + hardp
            if denom > 1.0e-12:
                c -= np.outer(cn, cn) / denom
    return c


def consistent_shell_tangent(
    mat: Any,
    stress: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent 3x3 plane-stress membrane tangent modulus."""
    if stress is None and "sig" in kwargs:
        stress = kwargs["sig"]
    p = mat if isinstance(mat, Law109Params) else Law109Params.from_material(mat)
    c_el = np.zeros((3, 3), dtype=np.float64)
    c_el[0, 0] = p.a11
    c_el[1, 1] = p.a11
    c_el[0, 1] = p.a12
    c_el[1, 0] = p.a12
    c_el[2, 2] = p.g

    if stress is None:
        return c_el

    sig_arr = np.asarray(stress, dtype=np.float64)
    if sig_arr.ndim == 2:
        nel = len(sig_arr)
        t_arr = np.zeros((nel, 3, 3), dtype=np.float64)
        for i in range(nel):
            t_arr[i] = consistent_shell_tangent(p, stress=sig_arr[i], extra=extra)
        return t_arr

    c = c_el.copy()
    if extra is not None:
        seq = extra.get("seq", 0.0)
        sigy = extra.get("sigy", 0.0)
        hardp = extra.get("hardp", 0.0)
        if seq >= sigy > 0.0 and hardp > 0.0:
            s = sig_arr[:3].copy()
            sigm = (s[0] + s[1]) / 3.0
            s[0] -= sigm
            s[1] -= sigm
            norm = np.array([1.5 * s[0] / seq, 1.5 * s[1] / seq, 3.0 * s[2] / seq])
            cn = c @ norm
            denom = norm @ cn + hardp
            if denom > 1.0e-12:
                c -= np.outer(cn, cn) / denom
    return c


def build_law109(mat: Any = None, **kwargs: Any) -> Law109Params:
    """Construct Law109Params from material entity or keyword arguments."""
    if mat is not None:
        p = Law109Params.from_material(mat)
    else:
        p = Law109Params()
    for k, v in kwargs.items():
        if hasattr(p, k):
            setattr(p, k, v)
    return p


def resolve(mat: Any, model: Any, log: Any = None) -> None:
    """Resolve curve and table references for /MAT/LAW109 from model."""
    if hasattr(mat, "tab_yld") and mat.tab_yld:
        curve = model.get_function(mat.tab_yld)
        if curve is not None:
            mat.yield_table = curve
    if hasattr(mat, "tab_temp") and mat.tab_temp:
        curve = model.get_function(mat.tab_temp)
        if curve is not None:
            mat.temp_table = curve
    if hasattr(mat, "tab_eta") and mat.tab_eta:
        curve = model.get_function(mat.tab_eta)
        if curve is not None:
            mat.eta_table = curve


def extra_shapes(mat: Any, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Return dictionary of extra history array shapes for LAW109."""
    if nip:
        return {
            "pla": (nip,),
            "epsd": (nip,),
            "uvar": (nip,),
            "temp": (nip,),
            "thk": (nip,),
            "seq": (nip,),
            "sigy": (nip,),
            "et": (nip,),
            "soundsp": (nip,),
        }
    return {
        "pla": (),
        "epsd": (),
        "uvar": (),
        "temp": (),
        "thk": (),
        "seq": (),
        "sigy": (),
        "et": (),
        "soundsp": (),
    }

