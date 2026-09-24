"""OpenRadioss /MAT/LAW106 (/MAT/JCOOK_ALM).

Johnson-Cook elastoplastic model for Additive Layer Manufacturing (ALM) with:
- Temperature-dependent Young's modulus E(T) during heating (fct_ID1) and cooling (fct_ID2)
- Temperature-dependent Poisson's ratio nu(T) (fct_ID3)
- Strain rate sensitivity (VP=1 viscoplastic, VP=2 total strain rate, VP=3 deviatoric strain rate)
- Temperature softening with melting temperature and optional linear exponent for T > T_MAX
- Taylor-Quinney adiabatic self-heating
- Cutting-plane iterative return mapping for 3D continuum solids and 2D plane-stress shells
- Through-thickness thinning for shells
- Ductile rupture and element deletion at eps_max

Fortran source references:
- Starter input reader:
  `starter/source/materials/mat/mat106/hm_read_mat106.F90`
- 3D continuum solid stress update:
  `engine/source/materials/mat/mat106/sigeps106.F90`
- 2D plane-stress shell stress update:
  `engine/source/materials/mat/mat106/sigeps106c.F90`
- Card layout configuration:
  `hm_cfg_files/config/CFG/radioss2026/MAT/mat_law106.cfg`
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

_DEFAULT_EPS_MAX = 1.0e30
_DEFAULT_SIGMA_MAX = 1.0e30
_DEFAULT_TMELT = 1.0e30
_DEFAULT_TMAX = 1.0e30
_DEFAULT_TREF = 300.0
_DEFAULT_FCUT = 10000.0
_DEFAULT_TOL = 1.0e-20


def _eval_curve_1d(curve: Any, x: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear evaluation of 1D curve returning (value, slope)."""
    x_arr = np.asarray(x, dtype=np.float64)
    if curve is None:
        return np.zeros_like(x_arr), np.zeros_like(x_arr)

    if hasattr(curve, "eval"):
        val = np.asarray(curve.eval(x_arr), dtype=np.float64)
        slope = getattr(curve, "slope", np.zeros_like(val))
        if isinstance(slope, np.ndarray) and slope.size > 0:
            slp = np.full_like(val, slope[-1] if len(slope) > 0 else 0.0)
        else:
            slp = np.zeros_like(val)
        return val, slp

    cx, cy = None, None
    if hasattr(curve, "x") and hasattr(curve, "y"):
        cx = np.asarray(curve.x, dtype=np.float64)
        cy = np.asarray(curve.y, dtype=np.float64)
    elif hasattr(curve, "data"):
        data = np.asarray(curve.data, dtype=np.float64)
        if data.ndim == 2 and data.shape[1] >= 2:
            cx, cy = data[:, 0], data[:, 1]
    elif isinstance(curve, dict):
        if "x" in curve and "y" in curve:
            cx = np.asarray(curve["x"], dtype=np.float64)
            cy = np.asarray(curve["y"], dtype=np.float64)
        else:
            try:
                sorted_items = sorted((float(k), float(v)) for k, v in curve.items())
                cx = np.array([k for k, _ in sorted_items], dtype=np.float64)
                cy = np.array([v for _, v in sorted_items], dtype=np.float64)
            except Exception:
                pass
    elif isinstance(curve, (list, tuple)):
        try:
            data = np.asarray(curve, dtype=np.float64)
            if data.ndim == 2 and data.shape[1] >= 2:
                cx, cy = data[:, 0], data[:, 1]
            elif data.ndim == 2 and data.shape[0] == 2:
                cx, cy = data[0], data[1]
        except Exception:
            pass

    if cx is not None and cy is not None and len(cx) >= 2:
        val = np.interp(x_arr, cx, cy)
        # End-slope extrapolation
        low_slp = (cy[1] - cy[0]) / max(1.0e-20, cx[1] - cx[0])
        high_slp = (cy[-1] - cy[-2]) / max(1.0e-20, cx[-1] - cx[-2])
        mask_low = x_arr < cx[0]
        mask_high = x_arr > cx[-1]
        if np.any(mask_low):
            val = np.where(mask_low, cy[0] + low_slp * (x_arr - cx[0]), val)
        if np.any(mask_high):
            val = np.where(mask_high, cy[-1] + high_slp * (x_arr - cx[-1]), val)
        slp = np.where(mask_low, low_slp, np.where(mask_high, high_slp, (cy[-1] - cy[0]) / max(1.0e-20, cx[-1] - cx[0])))
        return val, slp

    return np.zeros_like(x_arr), np.zeros_like(x_arr)


@dataclass
class JCookAlmParams:
    """Consolidated parameters for /MAT/LAW106 (/MAT/JCOOK_ALM)."""
    id: int = 1
    title: str = ""
    rho: float = 0.0
    refer_rho: float = 0.0
    young: float = 0.0
    nu: float = 0.0
    fct_id1: int = 0
    fct_id2: int = 0
    fct_id3: int = 0
    fct1: Any = None
    fct2: Any = None
    fct3: Any = None
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    eps_max: float = _DEFAULT_EPS_MAX
    sigma_max: float = _DEFAULT_SIGMA_MAX
    fcut: float = _DEFAULT_FCUT
    vp: int = 2
    nmax: int = 3
    tol: float = _DEFAULT_TOL
    cjc: float = 0.0
    deps0: float = 1.0
    m: float = 1.0
    tmelt: float = _DEFAULT_TMELT
    tmax: float = _DEFAULT_TMAX
    cs: float = 0.0
    eta: float = 1.0
    t0: float = _DEFAULT_TREF
    tref: float = _DEFAULT_TREF
    law: int = 106
    law_name: str = "LAW106"
    fail: Any = None
    eos: Any = None
    params: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.refer_rho <= 0.0 and self.rho > 0.0:
            self.refer_rho = self.rho
        if self.eps_max <= 0.0:
            self.eps_max = _DEFAULT_EPS_MAX
        if self.sigma_max <= 0.0:
            self.sigma_max = _DEFAULT_SIGMA_MAX
        if self.m <= 0.0:
            self.m = 1.0
        if self.tmelt <= 0.0:
            self.tmelt = _DEFAULT_TMELT
        if self.tref <= 0.0:
            self.tref = _DEFAULT_TREF
        if self.t0 <= 0.0:
            self.t0 = self.tref
        self.vp = min(max(self.vp, 0), 3)
        if self.vp == 0:
            self.vp = 2
        if self.nmax <= 0:
            self.nmax = 6 if self.vp == 1 else 3
        if self.tol <= 0.0:
            self.tol = _DEFAULT_TOL
        if self.deps0 <= 0.0:
            self.deps0 = 1.0
        if self.vp == 1:
            self.fcut = 0.0
        elif self.fcut <= 0.0:
            self.fcut = _DEFAULT_FCUT
        self.eta = min(max(self.eta, 0.0), 1.0)
        if self.eta == 0.0:
            self.eta = 1.0

    @property
    def e(self) -> float:
        return self.young

    @property
    def E(self) -> float:
        return self.young

    @property
    def G(self) -> float:
        return self.young / (2.0 * (1.0 + self.nu)) if (1.0 + self.nu) != 0.0 else 0.0

    @property
    def g(self) -> float:
        return self.G

    @property
    def K(self) -> float:
        return self.young / (3.0 * (1.0 - 2.0 * self.nu)) if (1.0 - 2.0 * self.nu) != 0.0 else 0.0

    @property
    def bulk(self) -> float:
        return self.K

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def sound_speed(self) -> float:
        r = self.refer_rho if self.refer_rho > 0.0 else (self.rho if self.rho > 0.0 else 1.0)
        dpdm = self.bulk + (4.0 / 3.0) * self.g
        return math.sqrt(max(0.0, dpdm) / r)

    @property
    def sound_speed_shell(self) -> float:
        r = self.refer_rho if self.refer_rho > 0.0 else (self.rho if self.rho > 0.0 else 1.0)
        c_sq = self.young / (r * (1.0 - self.nu * self.nu)) if (1.0 - self.nu * self.nu) > 0.0 else 0.0
        return math.sqrt(max(0.0, c_sq))


def build_law106(mat: Any = None, **kwargs: Any) -> JCookAlmParams:
    """Construct JCookAlmParams from material entity or keyword arguments."""
    p_dict: Dict[str, Any] = {}
    if mat is not None:
        for attr in (
            "id", "title", "rho", "rho0", "refer_rho", "rhor", "density", "young", "e", "nu",
            "fct_id1", "fct_id2", "fct_id3", "fct1", "fct2", "fct3",
            "a", "b", "n", "eps_max", "sigma_max", "fcut", "vp", "nmax", "tol",
            "cjc", "deps0", "m", "tmelt", "tmax", "cs", "eta", "t0", "tref", "params"
        ):
            if hasattr(mat, attr):
                p_dict[attr] = getattr(mat, attr)
        if hasattr(mat, "params") and isinstance(mat.params, dict):
            p_dict.update(mat.params)

    p_dict.update(kwargs)

    young_val = float(p_dict.get("young", p_dict.get("e", p_dict.get("E", p_dict.get("MAT_E", 0.0)))))
    nu_val = float(p_dict.get("nu", p_dict.get("Nu", p_dict.get("MAT_NU", 0.0))))
    rho_val = float(p_dict.get("rho", p_dict.get("rho0", p_dict.get("density", p_dict.get("MAT_RHO", 0.0)))))
    refer_rho = float(p_dict.get("refer_rho", p_dict.get("rhor", rho_val)))
    if refer_rho <= 0.0:
        refer_rho = rho_val

    a_val = float(p_dict.get("a", p_dict.get("sigy", p_dict.get("MAT_SIGY", 0.0))))
    b_val = float(p_dict.get("b", p_dict.get("beta", p_dict.get("MAT_BETA", 0.0))))
    n_val = float(p_dict.get("n", p_dict.get("hard", p_dict.get("MAT_HARD", 0.0))))
    eps_max_val = float(p_dict.get("eps_max", p_dict.get("epsm", p_dict.get("MLAW106_EP_MAX", _DEFAULT_EPS_MAX))))
    sig_max_val = float(p_dict.get("sigma_max", p_dict.get("sigm", p_dict.get("MLAW106_SIGMA_MAX", _DEFAULT_SIGMA_MAX))))

    fcut_val = float(p_dict.get("fcut", p_dict.get("MAT_FCUT", p_dict.get("MLAW106_FCUT", _DEFAULT_FCUT))))
    vp_val = int(p_dict.get("vp", p_dict.get("MLAW106_VP", 2)))
    nmax_val = int(p_dict.get("nmax", p_dict.get("MLAW106_NMAX", 0)))
    if nmax_val == 0:
        nmax_val = 6 if vp_val == 1 else 3
    tol_val = float(p_dict.get("tol", p_dict.get("MLAW106_TOL", _DEFAULT_TOL)))
    cjc_val = float(p_dict.get("cjc", p_dict.get("MLAW106_CJC", 0.0)))
    deps0_val = float(p_dict.get("deps0", p_dict.get("MLAW106_DEPS0", 1.0)))

    m_val = float(p_dict.get("m", p_dict.get("MAT_M", 1.0)))
    tmelt_val = float(p_dict.get("tmelt", p_dict.get("MAT_TMELT", _DEFAULT_TMELT)))
    tmax_val = float(p_dict.get("tmax", p_dict.get("MAT_TMAX", _DEFAULT_TMAX)))
    cs_val = float(p_dict.get("cs", p_dict.get("spheat", p_dict.get("rhocp", p_dict.get("MAT_SPHEAT", 0.0)))))
    eta_val = float(p_dict.get("eta", p_dict.get("MLAW106_ETA", 1.0)))
    t0_val = float(p_dict.get("t0", p_dict.get("MLAW106_T0", _DEFAULT_TREF)))
    tref_val = float(p_dict.get("tref", p_dict.get("MLAW106_TR", _DEFAULT_TREF)))

    fct_id1 = int(p_dict.get("fct_id1", p_dict.get("MLAW106_FCT_ID1", 0)))
    fct_id2 = int(p_dict.get("fct_id2", p_dict.get("MLAW106_FCT_ID2", 0)))
    fct_id3 = int(p_dict.get("fct_id3", p_dict.get("MLAW106_FCT_ID3", 0)))

    fct1 = p_dict.get("fct1")
    fct2 = p_dict.get("fct2")
    fct3 = p_dict.get("fct3")

    # If functions dict is passed in kwargs, resolve curves
    funcs = kwargs.get("funcs", kwargs.get("functions", {}))
    if isinstance(funcs, dict):
        if fct1 is None and fct_id1 in funcs:
            fct1 = funcs[fct_id1]
        if fct2 is None and fct_id2 in funcs:
            fct2 = funcs[fct_id2]
        if fct3 is None and fct_id3 in funcs:
            fct3 = funcs[fct_id3]

    res = JCookAlmParams(
        id=int(p_dict.get("id", 1)),
        title=str(p_dict.get("title", "")),
        rho=rho_val,
        refer_rho=refer_rho,
        young=young_val,
        nu=nu_val,
        fct_id1=fct_id1,
        fct_id2=fct_id2,
        fct_id3=fct_id3,
        fct1=fct1,
        fct2=fct2,
        fct3=fct3,
        a=a_val,
        b=b_val,
        n=n_val,
        eps_max=eps_max_val,
        sigma_max=sig_max_val,
        fcut=fcut_val,
        vp=vp_val,
        nmax=nmax_val,
        tol=tol_val,
        cjc=cjc_val,
        deps0=deps0_val,
        m=m_val,
        tmelt=tmelt_val,
        tmax=tmax_val,
        cs=cs_val,
        eta=eta_val,
        t0=t0_val,
        tref=tref_val,
        extra=p_dict,
    )
    res.params = p_dict
    return res


def resolve(mat: Any, model: Any, log: Any = None) -> None:
    """Resolve fct_id1, fct_id2, fct_id3 curves from model functions/curves."""
    p = getattr(mat, "params", {})
    if not isinstance(p, dict):
        p = {}
    for fid_key, fct_key in (("fct_id1", "fct1"), ("fct_id2", "fct2"), ("fct_id3", "fct3")):
        fid = p.get(fid_key, getattr(mat, fid_key, 0))
        if fid and fid != 0:
            curve = None
            if hasattr(model, "curves") and fid in model.curves:
                curve = model.curves[fid]
            elif hasattr(model, "functions") and fid in model.functions:
                curve = model.functions[fid]
            elif hasattr(model, "tables") and fid in model.tables:
                curve = model.tables[fid]
            if curve is not None:
                if hasattr(mat, "params") and isinstance(mat.params, dict):
                    mat.params[fct_key] = curve
                if hasattr(mat, fct_key):
                    setattr(mat, fct_key, curve)
            elif log is not None and hasattr(log, "warning"):
                log.warning(f"/MAT/LAW106/{getattr(mat, 'id', 0)}: function curve ID {fid} not found in model", "MAT CHECK")


def compute_temperature_elastic_moduli(
    params: JCookAlmParams,
    temp: Union[float, np.ndarray],
    temp_prev: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute temperature-dependent Young's modulus E(T), Poisson's ratio nu(T),

    shear modulus G(T), and bulk modulus K(T) matching sigeps106.F90:208-260.
    """
    t_arr = np.asarray(temp, dtype=np.float64)
    t_prev_arr = np.asarray(temp_prev, dtype=np.float64)
    is_scalar = (t_arr.ndim == 0)
    t_arr = np.atleast_1d(t_arr)
    t_prev_arr = np.atleast_1d(t_prev_arr)

    e_heat = np.full_like(t_arr, params.young)
    e_cool = np.full_like(t_arr, params.young)
    nu_arr = np.full_like(t_arr, params.nu)

    if params.fct1 is not None or params.fct_id1 > 0:
        val, _ = _eval_curve_1d(params.fct1, t_arr)
        if np.any(val > 0.0):
            e_heat = val

    if params.fct2 is not None or params.fct_id2 > 0:
        val, _ = _eval_curve_1d(params.fct2, t_arr)
        if np.any(val > 0.0):
            e_cool = val
    else:
        e_cool = e_heat

    if params.fct3 is not None or params.fct_id3 > 0:
        val, _ = _eval_curve_1d(params.fct3, t_arr)
        if np.any(val > 0.0):
            nu_arr = val

    # Heating vs cooling logic (sigeps106.F90:231-237)
    young_arr = np.where(
        (params.fct2 is None and params.fct_id2 == 0) | (t_arr > t_prev_arr),
        e_heat,
        e_cool,
    )
    nu_arr = np.clip(nu_arr, -0.999, 0.495)

    shear_arr = young_arr / (2.0 * (1.0 + nu_arr))
    bulk_arr = young_arr / (3.0 * (1.0 - 2.0 * nu_arr))

    if is_scalar:
        return young_arr[0], nu_arr[0], shear_arr[0], bulk_arr[0]
    return young_arr, nu_arr, shear_arr, bulk_arr


def compute_jcook_alm_yield_stress(
    params: JCookAlmParams,
    eps_p: float = 0.0,
    eps_dot: float = 0.0,
    temp: float = _DEFAULT_TREF,
    dt: float = 0.0,
    **kwargs: Any,
) -> Tuple[float, float]:
    """Compute Johnson-Cook flow stress and hardening modulus matching sigeps106.F90:297-311, 375-396."""
    if "epsp" in kwargs:
        eps_p = kwargs["epsp"]
    if "eps_rate" in kwargs:
        eps_dot = kwargs["eps_rate"]
    p_eff = max(0.0, float(eps_p))
    t_curr = float(temp)

    # 1. Plastic Hardening
    if params.b > 0.0 and params.n > 0.0:
        hard = params.a + params.b * math.exp(params.n * math.log(p_eff + 1.0e-20))
    else:
        hard = params.a
    hard = min(params.sigma_max, hard)

    # 2. Strain Rate Dependency
    srdep = 1.0
    if params.cjc > 0.0 and params.deps0 > 0.0:
        srdep = 1.0 + params.cjc * math.log(1.0 + max(0.0, eps_dot) / params.deps0)

    # 3. Thermal Softening
    tempr = max(params.tref, min(t_curr, params.tmelt))
    if params.tmelt > params.tref:
        t_star = (tempr - params.tref) / (params.tmelt - params.tref)
        m_eff = 1.0 if t_curr > params.tmax else params.m
        thsoft = max(0.0, 1.0 - (t_star ** m_eff))
    else:
        thsoft = 1.0

    sigy = max(1.0e-10, hard * srdep * thsoft)

    # Hardening modulus d(sigy)/d(eps_p)
    if hard < params.sigma_max and params.b > 0.0 and params.n > 0.0:
        hardp = params.n * params.b * math.exp((params.n - 1.0) * math.log(p_eff + 1.0e-20))
    else:
        hardp = 0.0
    hardp *= srdep

    if params.vp == 1 and dt > 0.0 and params.deps0 > 0.0:
        hardp += hard * (params.cjc / dt) / (params.deps0 + eps_dot)

    hardp *= thsoft

    return sigy, hardp


def solid_update(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    rho: Optional[Any] = None,
    rho0: Optional[Any] = None,
    off: Optional[Any] = None,
    pnew: Optional[Any] = None,
    psh: float = 0.0,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    ismstr: int = 0,
    *,
    epsp: Optional[np.ndarray] = None,
    return_tuple: bool = True,
    **kwargs: Any,
) -> Any:
    """3D solid continuum constitutive update matching sigeps106.F90."""
    params = mat if isinstance(mat, JCookAlmParams) else build_law106(mat, **kwargs)

    if sig is None:
        sig = np.zeros(6, dtype=np.float64)
    is_1d = (sig.ndim == 1)

    sig_arr = np.atleast_2d(sig).astype(np.float64)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(np.float64)
    nel = len(sig_arr)

    # History variables:
    # col 0: temp (temperature)
    # col 1: epsd (filtered plastic strain rate)
    # col 2: young0 (previous Young's modulus)
    # col 3: nu0 (previous Poisson's ratio)
    # col 4: epsp (accumulated plastic strain)
    hist_arr = np.zeros((nel, 5), dtype=np.float64)
    hist_arr[:, 0] = params.t0
    hist_arr[:, 2] = params.young
    hist_arr[:, 3] = params.nu

    if extra is not None and isinstance(extra, dict):
        if "uvar106" in extra:
            u = np.asarray(extra["uvar106"], dtype=np.float64)
            if u.ndim == 1:
                hist_arr[0, :min(5, len(u))] = u[:5]
            elif u.ndim == 2:
                hist_arr[:, :min(5, u.shape[1])] = u[:, :5]
        elif "uvar" in extra:
            u = np.asarray(extra["uvar"], dtype=np.float64)
            if u.ndim == 1:
                hist_arr[0, :min(5, len(u))] = u[:5]
            elif u.ndim == 2:
                hist_arr[:, :min(5, u.shape[1])] = u[:, :5]

    if epsp is not None:
        ep_arr = np.atleast_1d(epsp).astype(np.float64)
        hist_arr[:, 4] = ep_arr[:nel]

    rho_arr = np.full(nel, params.rho, dtype=np.float64)
    if rho is not None:
        r = np.atleast_1d(rho).astype(np.float64)
        rho_arr[:len(r)] = r
    rho_arr = np.where(rho_arr > 0.0, rho_arr, (params.refer_rho if params.refer_rho > 0.0 else 1.0))

    off_arr = np.ones(nel, dtype=np.float64)
    if off is not None:
        o = np.atleast_1d(off).astype(np.float64)
        off_arr[:min(nel, len(o))] = o[:nel]
    elif extra is not None and isinstance(extra, dict) and "off" in extra:
        o = np.atleast_1d(extra["off"]).astype(np.float64)
        off_arr[:min(nel, len(o))] = o[:nel]

    sig_out = np.zeros_like(sig_arr)
    epsp_out = np.zeros(nel, dtype=np.float64)
    soundsp_out = np.zeros(nel, dtype=np.float64)

    for i in range(nel):
        s_old = sig_arr[i].copy()
        d_e = deps_arr[i]
        o_val = off_arr[i]
        t_prev = hist_arr[i, 0]
        t_curr = t_prev
        epsd_val = hist_arr[i, 1]
        e0_val = hist_arr[i, 2] if hist_arr[i, 2] > 0.0 else params.young
        nu0_val = hist_arr[i, 3] if hist_arr[i, 3] > 0.0 else params.nu
        pla0 = hist_arr[i, 4]
        pla_curr = pla0

        # Element failure degradation matching sigeps106.F90:181-183
        if o_val < 0.1:
            o_val = 0.0
        elif o_val < 1.0:
            o_val *= 0.8

        if o_val <= 0.0:
            off_arr[i] = 0.0
            sig_out[i] = 0.0
            epsp_out[i] = pla0
            soundsp_out[i] = params.sound_speed
            hist_arr[i, 4] = pla0
            continue

        # 1. Temperature-dependent elastic properties
        e_curr, nu_curr, shear_curr, bulk_curr = compute_temperature_elastic_moduli(params, t_curr, t_prev)
        shear0 = e0_val / (2.0 * (1.0 + nu0_val))
        bulk0 = e0_val / (3.0 * (1.0 - 2.0 * nu0_val))

        # 2. Strain rate evaluation
        if params.vp > 1:
            if params.vp == 2:
                # Total strain rate
                eps_dot = math.sqrt(d_e[0]**2 + d_e[1]**2 + d_e[2]**2 + 2.0 * (0.5 * d_e[3])**2 + 2.0 * (0.5 * d_e[4])**2 + 2.0 * (0.5 * d_e[5])**2) / (dt if dt > 0.0 else 1.0)
            else:
                # Deviatoric strain rate
                tr_de = (d_e[0] + d_e[1] + d_e[2]) / 3.0
                de_dev0 = d_e[0] - tr_de
                de_dev1 = d_e[1] - tr_de
                de_dev2 = d_e[2] - tr_de
                eps_dot = math.sqrt(de_dev0**2 + de_dev1**2 + de_dev2**2 + 2.0 * (0.5 * d_e[3])**2 + 2.0 * (0.5 * d_e[4])**2 + 2.0 * (0.5 * d_e[5])**2) * math.sqrt(2.0 / 3.0) / (dt if dt > 0.0 else 1.0)

            if params.fcut > 0.0 and dt > 0.0:
                omega_dt = 2.0 * math.pi * params.fcut * dt
                alpha = omega_dt / (1.0 + omega_dt)
                epsd_val = alpha * eps_dot + (1.0 - alpha) * epsd_val
            else:
                epsd_val = eps_dot

        # 3. Trial stress tensor
        p_old = (s_old[0] + s_old[1] + s_old[2]) / 3.0
        s_dev0 = np.array([
            s_old[0] - p_old,
            s_old[1] - p_old,
            s_old[2] - p_old,
            s_old[3],
            s_old[4],
            s_old[5],
        ])

        tr_deps = d_e[0] + d_e[1] + d_e[2]
        dev_deps = np.array([
            d_e[0] - tr_deps / 3.0,
            d_e[1] - tr_deps / 3.0,
            d_e[2] - tr_deps / 3.0,
            d_e[3],
            d_e[4],
            d_e[5],
        ])

        mod_ratio = shear_curr / max(1.0e-20, shear0)
        s_dev = np.zeros(6, dtype=np.float64)
        s_dev[:3] = s_dev0[:3] * mod_ratio + 2.0 * shear_curr * dev_deps[:3]
        s_dev[3:] = s_dev0[3:] * mod_ratio + shear_curr * dev_deps[3:]

        p_new = p_old * (bulk_curr / max(1.0e-20, bulk0)) + bulk_curr * tr_deps

        # Von Mises stress
        seq2 = (s_dev[0]**2 + s_dev[1]**2 + s_dev[2]**2) + 2.0 * (s_dev[3]**2 + s_dev[4]**2 + s_dev[5]**2)
        seq = math.sqrt(max(0.0, 1.5 * seq2))

        # Flow stress
        sigy, hardp = compute_jcook_alm_yield_stress(params, eps_p=pla_curr, eps_dot=epsd_val, temp=t_curr, dt=dt)

        phi = seq - sigy
        dpla = 0.0

        # 4. Plastic return mapping (cutting-plane iterative method matching sigeps106.F90:334-451)
        if phi > 0.0 and o_val > 0.1:
            for _ in range(params.nmax):
                norm = np.zeros(6, dtype=np.float64)
                denom_seq = max(seq, 1.0e-20)
                norm[:3] = 1.5 * s_dev[:3] / denom_seq
                norm[3:] = 3.0 * s_dev[3:] / denom_seq

                dfdsig2 = (
                    norm[0]**2 * 2.0 * shear_curr
                    + norm[1]**2 * 2.0 * shear_curr
                    + norm[2]**2 * 2.0 * shear_curr
                    + norm[3]**2 * shear_curr
                    + norm[4]**2 * shear_curr
                    + norm[5]**2 * shear_curr
                )

                dpla_dlam = seq / max(sigy, 1.0e-20)
                dphi_dlam = -dfdsig2 - hardp * dpla_dlam
                dphi_dlam = math.copysign(max(abs(dphi_dlam), 1.0e-20), dphi_dlam)

                dlam = -phi / dphi_dlam
                dlam = max(0.0, dlam)

                ddep = dpla_dlam * dlam
                dpla += ddep
                pla_curr = pla0 + dpla

                if params.vp == 1 and dt > 0.0:
                    epsd_val = dpla / dt

                # Update deviatoric stress
                s_dev[:3] -= 2.0 * shear_curr * (dlam * norm[:3])
                s_dev[3:] -= shear_curr * (dlam * norm[3:])

                # Recompute seq and sigy
                seq2 = (s_dev[0]**2 + s_dev[1]**2 + s_dev[2]**2) + 2.0 * (s_dev[3]**2 + s_dev[4]**2 + s_dev[5]**2)
                seq = math.sqrt(max(0.0, 1.5 * seq2))
                sigy, hardp = compute_jcook_alm_yield_stress(params, eps_p=pla_curr, eps_dot=epsd_val, temp=t_curr, dt=dt)
                phi = seq - sigy
                if abs(phi) < params.tol * max(sigy, 1.0):
                    break

            # 5. Taylor-Quinney adiabatic self-heating
            if params.cs > 0.0:
                t_curr += params.eta * sigy * dpla / params.cs

        # 6. Ductile rupture test (sigeps106.F90:478-482)
        if o_val >= 1.0 and pla_curr >= params.eps_max:
            o_val = 0.8
        off_arr[i] = o_val

        # 7. Final stress assembly
        s_final = np.zeros(6, dtype=np.float64)
        s_final[:3] = (s_dev[:3] + p_new) * o_val
        s_final[3:] = s_dev[3:] * o_val

        # Sound speed
        c_solid = math.sqrt((bulk_curr + (4.0 / 3.0) * shear_curr) / rho_arr[i])

        sig_out[i] = s_final
        epsp_out[i] = pla_curr
        soundsp_out[i] = c_solid

        # Update history
        hist_arr[i, 0] = t_curr
        hist_arr[i, 1] = epsd_val
        hist_arr[i, 2] = e_curr
        hist_arr[i, 3] = nu_curr
        hist_arr[i, 4] = pla_curr

    if extra is not None and isinstance(extra, dict):
        if is_1d:
            extra["uvar106"] = hist_arr[0]
            extra["uvar"] = hist_arr[0]
            extra["temp"] = hist_arr[0, 0]
            if "off" in extra and isinstance(extra["off"], np.ndarray):
                extra["off"][0] = off_arr[0]
            else:
                extra["off"] = float(off_arr[0])
        else:
            extra["uvar106"] = hist_arr
            extra["uvar"] = hist_arr
            extra["temp"] = hist_arr[:, 0]
            if "off" in extra and isinstance(extra["off"], np.ndarray):
                extra["off"][:nel] = off_arr
            else:
                extra["off"] = off_arr

    if is_1d:
        sig_res = sig_out[0]
        epsp_res = float(epsp_out[0])
        ssp_res = float(soundsp_out[0])
    else:
        sig_res = sig_out
        epsp_res = epsp_out
        ssp_res = soundsp_out

    if return_tuple:
        return sig_res, epsp_res, ssp_res
    return sig_res, epsp_res


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    rho: Optional[Any] = None,
    thk: Optional[Any] = None,
    thkly: float = 1.0,
    off: Optional[Any] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    hist_old: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Any:
    """2D plane-stress shell constitutive update matching sigeps106c.F90."""
    params = mat if isinstance(mat, JCookAlmParams) else build_law106(mat, **kwargs)

    if thk is None and extra is not None and isinstance(extra, dict):
        thk = extra.get("thk", extra.get("thick"))
        thkly = float(extra.get("thkly", thkly))

    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(np.float64)
    deps_arr = np.atleast_2d(deps).astype(np.float64)
    nel = len(sig_arr)

    # History variables:
    # col 0: temp
    # col 1: epsd
    # col 2: young0
    # col 3: nu0
    # col 4: eplaxx
    # col 5: eplayy
    # col 6: pla
    hist_arr = np.zeros((nel, 7), dtype=np.float64)
    hist_arr[:, 0] = params.t0
    hist_arr[:, 2] = params.young
    hist_arr[:, 3] = params.nu

    if hist_old is not None:
        h = np.asarray(hist_old, dtype=np.float64)
        if h.ndim == 1:
            hist_arr[0, :min(7, len(h))] = h[:7]
        elif h.ndim == 2:
            hist_arr[:, :min(7, h.shape[1])] = h[:, :7]
    elif extra is not None and isinstance(extra, dict) and "uvar106" in extra:
        u = np.asarray(extra["uvar106"], dtype=np.float64)
        if u.ndim == 1:
            hist_arr[0, :min(7, len(u))] = u[:7]
        elif u.ndim == 2:
            hist_arr[:, :min(7, u.shape[1])] = u[:, :7]
    elif extra is not None and isinstance(extra, dict) and "uvar" in extra:
        u = np.asarray(extra["uvar"], dtype=np.float64)
        if u.ndim == 1:
            hist_arr[0, :min(7, len(u))] = u[:7]
        elif u.ndim == 2:
            hist_arr[:, :min(7, u.shape[1])] = u[:, :7]
    elif kwargs.get("epsp") is not None:
        ep_arr = np.atleast_1d(kwargs.get("epsp")).astype(np.float64)
        hist_arr[:, 6] = ep_arr[:nel]

    rho_arr = np.full(nel, params.rho, dtype=np.float64)
    if rho is not None:
        r = np.atleast_1d(rho).astype(np.float64)
        rho_arr[:len(r)] = r
    rho_arr = np.where(rho_arr > 0.0, rho_arr, (params.refer_rho if params.refer_rho > 0.0 else 1.0))

    off_arr = np.ones(nel, dtype=np.float64)
    if off is not None:
        o = np.atleast_1d(off).astype(np.float64)
        off_arr[:min(nel, len(o))] = o[:nel]
    elif extra is not None and isinstance(extra, dict) and "off" in extra:
        o = np.atleast_1d(extra["off"]).astype(np.float64)
        off_arr[:min(nel, len(o))] = o[:nel]

    sig_out = np.zeros_like(sig_arr)
    epsp_out = np.zeros(nel, dtype=np.float64)
    soundsp_out = np.zeros(nel, dtype=np.float64)

    for i in range(nel):
        s_old = sig_arr[i].copy()
        d_e = deps_arr[i]
        o_val = off_arr[i]

        t_prev = hist_arr[i, 0]
        t_curr = t_prev
        epsd_val = hist_arr[i, 1]
        e0_val = hist_arr[i, 2] if hist_arr[i, 2] > 0.0 else params.young
        nu0_val = hist_arr[i, 3] if hist_arr[i, 3] > 0.0 else params.nu
        eplaxx = hist_arr[i, 4]
        eplayy = hist_arr[i, 5]
        pla0 = hist_arr[i, 6]
        pla_curr = pla0

        # Element failure degradation matching sigeps106c.F90:173-175
        if o_val < 0.1:
            o_val = 0.0
        elif o_val < 1.0:
            o_val *= 0.8

        if o_val <= 0.0:
            off_arr[i] = 0.0
            sig_out[i] = 0.0
            epsp_out[i] = pla0
            soundsp_out[i] = params.sound_speed_shell
            hist_arr[i, 6] = pla0
            continue

        # 1. Temperature-dependent properties
        e_curr, nu_curr, shear_curr, _ = compute_temperature_elastic_moduli(params, t_curr, t_prev)
        shear0 = e0_val / (2.0 * (1.0 + nu0_val))

        aii = e_curr / (1.0 - nu_curr * nu_curr)
        aij = aii * nu_curr

        # 2. Strain rate evaluation
        if params.vp > 1:
            epspzz = -(nu_curr / (1.0 - nu_curr)) * (d_e[0] + d_e[1])
            if params.vp == 2:
                eps_dot = math.sqrt(d_e[0]**2 + d_e[1]**2 + epspzz**2 + 2.0 * (0.5 * d_e[2])**2) / (dt if dt > 0.0 else 1.0)
            else:
                dav = (d_e[0] + d_e[1] + epspzz) / 3.0
                deve1 = d_e[0] - dav
                deve2 = d_e[1] - dav
                deve3 = epspzz - dav
                deve4 = 0.5 * d_e[2]
                eps_dot = math.sqrt(0.5 * (deve1**2 + deve2**2 + deve3**2) + deve4**2) * math.sqrt(3.0) / 1.5 / (dt if dt > 0.0 else 1.0)

            if params.fcut > 0.0 and dt > 0.0:
                omega_dt = 2.0 * math.pi * params.fcut * dt
                alpha = omega_dt / (1.0 + omega_dt)
                epsd_val = alpha * eps_dot + (1.0 - alpha) * epsd_val
            else:
                epsd_val = eps_dot

        # 3. Trial stress tensor
        mod_ratio = shear_curr / max(1.0e-20, shear0)
        s_xx = aii * (d_e[0] + (s_old[0] - nu_curr * s_old[1]) / max(1.0e-20, e0_val)) + aij * (d_e[1] + (s_old[1] - nu_curr * s_old[0]) / max(1.0e-20, e0_val))
        # Incremental formula matching standard plane-stress
        s_xx = s_old[0] + aii * d_e[0] + aij * d_e[1]
        s_yy = s_old[1] + aij * d_e[0] + aii * d_e[1]
        s_xy = s_old[2] * mod_ratio + shear_curr * d_e[2]

        seq = math.sqrt(max(0.0, s_xx**2 + s_yy**2 - s_xx * s_yy + 3.0 * s_xy**2))
        sigy, hardp = compute_jcook_alm_yield_stress(params, eps_p=pla_curr, eps_dot=epsd_val, temp=t_curr, dt=dt)

        phi = seq - sigy
        dpla = 0.0
        dezz = 0.0

        # 4. Plane-stress iterative return mapping (sigeps106c.F90:346-458)
        if phi > 0.0 and o_val > 0.1:
            for _ in range(params.nmax):
                denom_seq = max(seq, 1.0e-20)
                norm_xx = (s_xx - 0.5 * s_yy) / denom_seq
                norm_yy = (s_yy - 0.5 * s_xx) / denom_seq
                norm_xy = 3.0 * s_xy / denom_seq

                dfdsig2 = (
                    norm_xx * (aii * norm_xx + aij * norm_yy)
                    + norm_yy * (aij * norm_xx + aii * norm_yy)
                    + norm_xy * norm_xy * shear_curr
                )

                dpla_dlam = seq / max(sigy, 1.0e-20)
                dphi_dlam = -dfdsig2 - hardp * dpla_dlam
                dphi_dlam = math.copysign(max(abs(dphi_dlam), 1.0e-20), dphi_dlam)

                dlam = -phi / dphi_dlam
                dlam = max(0.0, dlam)

                ddep = dpla_dlam * dlam
                dpla += ddep
                pla_curr = pla0 + dpla

                if params.vp == 1 and dt > 0.0:
                    epsd_val = dpla / dt

                dpxx = dlam * norm_xx
                dpyy = dlam * norm_yy
                dpxy = dlam * norm_xy

                eplaxx += dpxx
                eplayy += dpyy

                s_xx -= (aii * dpxx + aij * dpyy)
                s_yy -= (aij * dpxx + aii * dpyy)
                s_xy -= shear_curr * dpxy

                seq = math.sqrt(max(0.0, s_xx**2 + s_yy**2 - s_xx * s_yy + 3.0 * s_xy**2))
                sigy, hardp = compute_jcook_alm_yield_stress(params, eps_p=pla_curr, eps_dot=epsd_val, temp=t_curr, dt=dt)
                phi = seq - sigy

                dezz -= (dpxx + dpyy)
                if abs(phi) < params.tol * max(sigy, 1.0):
                    break

            if params.cs > 0.0:
                t_curr += params.eta * sigy * dpla / params.cs

        # Elastic + plastic thickness variation (sigeps106c.F90:512-515)
        dezz = -nu_curr * (s_xx - s_old[0] + s_yy - s_old[1]) / max(1.0e-20, e_curr) + dezz
        if thk is not None:
            if hasattr(thk, "__setitem__"):
                thk[i] *= (1.0 + dezz * thkly * o_val)
            elif isinstance(thk, (list, np.ndarray)):
                thk[i] = thk[i] * (1.0 + dezz * thkly * o_val)
            elif isinstance(thk, (int, float)):
                thk_new = float(thk) * (1.0 + dezz * thkly * o_val)
                if extra is not None and isinstance(extra, dict):
                    if "thk" in extra:
                        extra["thk"] = thk_new
                    if "thick" in extra:
                        extra["thick"] = thk_new

        if extra is not None and isinstance(extra, dict):
            extra["thickness_strain"] = dezz

        # Rupture test (sigeps106c.F90:485-489)
        if o_val >= 1.0 and pla_curr >= params.eps_max:
            o_val = 0.8
        off_arr[i] = o_val

        sig_final = np.array([s_xx, s_yy, s_xy], dtype=np.float64) * o_val
        c_shell = math.sqrt(aii / rho_arr[i])

        sig_out[i] = sig_final
        epsp_out[i] = pla_curr
        soundsp_out[i] = c_shell

        hist_arr[i, 0] = t_curr
        hist_arr[i, 1] = epsd_val
        hist_arr[i, 2] = e_curr
        hist_arr[i, 3] = nu_curr
        hist_arr[i, 4] = eplaxx
        hist_arr[i, 5] = eplayy
        hist_arr[i, 6] = pla_curr

    if extra is not None and isinstance(extra, dict):
        if is_1d:
            extra["uvar106"] = hist_arr[0]
            extra["uvar"] = hist_arr[0]
            extra["temp"] = hist_arr[0, 0]
            if "off" in extra and isinstance(extra["off"], np.ndarray):
                extra["off"][0] = off_arr[0]
            else:
                extra["off"] = float(off_arr[0])
        else:
            extra["uvar106"] = hist_arr
            extra["uvar"] = hist_arr
            extra["temp"] = hist_arr[:, 0]
            if "off" in extra and isinstance(extra["off"], np.ndarray):
                extra["off"][:nel] = off_arr
            else:
                extra["off"] = off_arr

    if is_1d:
        return sig_out[0], float(epsp_out[0]), float(soundsp_out[0])
    return sig_out, epsp_out, soundsp_out


def sound_speed(mat: Any = None, rho: Optional[Any] = None, is_shell: bool = False, **kwargs: Any) -> Any:
    """Compute acoustic wave speed for LAW106."""
    params = mat if isinstance(mat, JCookAlmParams) else build_law106(mat, **kwargs)
    if is_shell:
        return sound_speed_shell(params, rho=rho)
    return sound_speed_solid(params, rho=rho)


def sound_speed_solid(mat: Any = None, rho: Optional[Any] = None, **kwargs: Any) -> Any:
    """Compute 3D dilatational wave speed for solid elements."""
    params = mat if isinstance(mat, JCookAlmParams) else build_law106(mat, **kwargs)
    r = rho if (rho is not None and float(np.min(rho)) > 0.0) else (params.refer_rho if params.refer_rho > 0.0 else params.rho)
    r_val = max(1e-20, float(r)) if np.isscalar(r) else np.maximum(1e-20, np.asarray(r, dtype=np.float64))
    c_sq = (params.bulk + (4.0 / 3.0) * params.g) / r_val
    return math.sqrt(max(0.0, float(c_sq))) if np.isscalar(c_sq) else np.sqrt(np.maximum(0.0, c_sq))


def sound_speed_shell(mat: Any = None, rho: Optional[Any] = None, **kwargs: Any) -> Any:
    """Compute 2D acoustic wave speed for shell elements."""
    params = mat if isinstance(mat, JCookAlmParams) else build_law106(mat, **kwargs)
    r = rho if (rho is not None and float(np.min(rho)) > 0.0) else (params.refer_rho if params.refer_rho > 0.0 else params.rho)
    r_val = max(1e-20, float(r)) if np.isscalar(r) else np.maximum(1e-20, np.asarray(r, dtype=np.float64))
    denom = 1.0 - params.nu * params.nu
    c_sq = params.young / (r_val * denom) if denom > 0.0 else 0.0
    return math.sqrt(max(0.0, float(c_sq))) if np.isscalar(c_sq) else np.sqrt(np.maximum(0.0, c_sq))


def solid_tangent(mat: Any, sig: Optional[np.ndarray] = None, **kwargs: Any) -> np.ndarray:
    """Compute 6x6 algorithmic solid tangent stiffness matrix."""
    params = mat if isinstance(mat, JCookAlmParams) else build_law106(mat, **kwargs)
    young = params.young
    nu = params.nu
    g = params.G
    c11 = young * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c12 = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    c_el = np.zeros((6, 6), dtype=np.float64)
    c_el[0, 0] = c_el[1, 1] = c_el[2, 2] = c11
    c_el[0, 1] = c_el[1, 0] = c_el[0, 2] = c_el[2, 0] = c_el[1, 2] = c_el[2, 1] = c12
    c_el[3, 3] = c_el[4, 4] = c_el[5, 5] = g

    if sig is None:
        return c_el

    sig_arr = np.asarray(sig, dtype=np.float64)
    if sig_arr.ndim == 2:
        nel = len(sig_arr)
        t_arr = np.zeros((nel, 6, 6), dtype=np.float64)
        for i in range(nel):
            t_arr[i] = solid_tangent(params, sig=sig_arr[i])
        return t_arr

    p = (sig_arr[0] + sig_arr[1] + sig_arr[2]) / 3.0
    s = np.array([sig_arr[0] - p, sig_arr[1] - p, sig_arr[2] - p, sig_arr[3], sig_arr[4], sig_arr[5]])
    seq = math.sqrt(max(0.0, 1.5 * (s[0]**2 + s[1]**2 + s[2]**2 + 2.0 * (s[3]**2 + s[4]**2 + s[5]**2))))

    epsp = kwargs.get("epsp", 0.0)
    sigy, hp = compute_jcook_alm_yield_stress(params, eps_p=epsp, eps_dot=0.0, temp=params.t0)
    if seq < sigy or seq <= 1.0e-12:
        return c_el

    n = np.zeros(6, dtype=np.float64)
    n[:3] = 1.5 * s[:3] / seq
    n[3:] = 3.0 * s[3:] / seq

    denom = 3.0 * g + hp
    gamma = (3.0 * g) / max(1.0e-20, denom)
    c_tan = c_el - (2.0 * g * gamma) * np.outer(n, n)
    return c_tan


def consistent_shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Compute 3x3 plane-stress algorithmic tangent matrix for shells."""
    params = mat if isinstance(mat, JCookAlmParams) else build_law106(mat, **kwargs)
    young = params.young
    nu = params.nu
    q11 = young / (1.0 - nu * nu)
    q12 = q11 * nu
    q33 = params.G

    c_el = np.array([
        [q11, q12, 0.0],
        [q12, q11, 0.0],
        [0.0, 0.0, q33],
    ], dtype=np.float64)

    if sig is None:
        return c_el

    sig_arr = np.asarray(sig, dtype=np.float64)
    if sig_arr.ndim == 2:
        nel = len(sig_arr)
        t_arr = np.zeros((nel, 3, 3), dtype=np.float64)
        for i in range(nel):
            t_arr[i] = consistent_shell_tangent(params, sig=sig_arr[i])
        return t_arr

    sxx, syy, sxy = sig_arr[0], sig_arr[1], sig_arr[2]
    seq = math.sqrt(max(0.0, sxx**2 + syy**2 - sxx * syy + 3.0 * sxy**2))

    epsp = kwargs.get("epsp", 0.0)
    sigy, hp = compute_jcook_alm_yield_stress(params, eps_p=epsp, eps_dot=0.0, temp=params.t0)
    if seq < sigy or seq <= 1.0e-12:
        return c_el

    n_xx = (sxx - 0.5 * syy) / seq
    n_yy = (syy - 0.5 * sxx) / seq
    n_xy = 3.0 * sxy / seq
    n_vec = np.array([n_xx, n_yy, n_xy], dtype=np.float64)

    c_n = c_el @ n_vec
    denom = float(n_vec @ c_n) + hp
    if denom > 1.0e-20:
        return c_el - np.outer(c_n, c_n) / denom
    return c_el


def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Extra history shapes needed for LAW106."""
    return {"uvar106": (nip, 7) if nip else (5,)}

