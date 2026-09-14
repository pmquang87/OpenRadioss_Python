"""/MAT/LAW105 (/MAT/POWDER_BURN / /MAT/POWDERBURN) — Powder Burn Propellant Material Model (M576).

Constitutive model for powder burn deflagration / explosive propellants based on
the formulation by Atwood, Friis, and Moxnes.

Upstream reference:
  - Starter reader: ``starter/source/materials/mat/mat105/hm_read_mat105.F90``
  - Detonation / lighting time: ``starter/source/materials/mat/mat105/m105init.F``
  - Engine 3D solid continuum kernel: ``engine/source/materials/mat/mat105/sigeps105.F``
  - CFG configuration: ``radioss2026/MAT/matl105.cfg``, ``radioss2026/MAT/mat_EOS.cfg``
  - Alternative /EOS template: ``common_source/eos/powder_burn.F``

Key features:
  - Exponential gas equation of state:
      P_g = P_0 + rho_g * e_g * exp(rho_g / D)
  - Linear solid grain compressibility:
      rho_si = (rho0 / compac) * (P_g / Bulk + 1.0), where compac = 0.93
      P_s = P_0 + Bulk * (rho_si / rho0 - 1.0)
  - Burn rate pressure dependence:
      B_rate = scale_b * f_b(P_g / scale_p)
  - Grain growth burning kinetics:
      dF/dt = Gr * (1 - alpha * F)^C * B_rate
  - Ignition flame front velocity propagation:
      bfrac = min(1, max(0, (2/3) * C1 * (t - t_b) / L_char))
  - Total mixture burn fraction:
      TOTAL_BFRAC = bfrac * F(t)
  - Mixture pressure & sound speed:
      P = TOTAL_BFRAC * P_g + (1 - TOTAL_BFRAC) * P_s
      P_eff = (P - P_sh) * off
      c_mix = TOTAL_BFRAC * c_g + (1 - TOTAL_BFRAC) * c_s
  - Deviatoric stresses are identically zero (hydrodynamic fluid EOS behaviour).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# Threshold constants matching sigeps105.F and OpenRadioss constants
_COMPAC_DEFAULT: float = 0.93  # Initial solid packing fraction (1 - 0.07)
_EM04: float = 1.0e-4          # Minimum threshold for flame ignition fraction
_EM10: float = 1.0e-10         # Minimum divisor for solid density
_EM12: float = 1.0e-12         # Minimum volume / gas density divisor
_EM20: float = 1.0e-20         # Machine precision floor
_TWO_THIRD: float = 2.0 / 3.0  # 2/3 factor for flame front propagation


@dataclass
class PowderBurnParams:
    """Material parameters for /MAT/LAW105 (/MAT/POWDER_BURN)."""

    id: int = 0
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    bulk: float = 0.0          # UPARAM(1): Solid bulk modulus K
    p0: float = 0.0            # UPARAM(2): Initial pressure P0
    psh: float = 0.0           # UPARAM(3): Pressure shift PSH
    e0: float = 0.0            # UPARAM(4): Initial specific internal energy
    d: float = 0.0             # UPARAM(5): Gas EOS exponential parameter D
    eg: float = 0.0            # UPARAM(6): Specific gas detonation/reaction energy
    gas_d: float = 0.0         # Alias for d
    gas_eg: float = 0.0        # Alias for eg
    gas_gam: float = 1.3       # Gas adiabatic index gamma
    gr: float = 0.0            # UPARAM(7): Grain growth parameter Gr
    c: float = 0.0             # UPARAM(8): Grain reaction ratio exponent C
    alpha: float = 0.0         # UPARAM(9): Reaction ratio factor alpha
    c1: float = 0.0            # UPARAM(10): Burn front velocity parameter C1
    c2: float = 0.0            # UPARAM(11): Burn front velocity parameter C2
    func_b: int = 0            # IFUNC(1): POWDER_B_FUNC (burn rate vs pressure)
    scale_b: float = 1.0       # UPARAM(13): Burn rate ordinate scale factor
    scale_p: float = 1.0       # UPARAM(14): Pressure abscissa scale factor
    func_gam: int = 0          # IFUNC(2): POWDER_GAM_FUNC (ignition front function)
    scale_gam: float = 1.0     # UPARAM(12): Gamma ordinate scale factor
    scale_rho: float = 1.0     # UPARAM(15): Density abscissa scale factor
    compac: float = _COMPAC_DEFAULT
    curve_b: Any = None        # Resolved callable/curve for burn rate vs pressure
    curve_gam: Any = None      # Resolved callable/curve for ignition front
    law: int = 105
    law_name: str = "LAW105"
    params: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.refer_rho <= 0.0 and self.rho0 > 0.0:
            self.refer_rho = self.rho0
        elif self.rho0 <= 0.0 and self.refer_rho > 0.0:
            self.rho0 = self.refer_rho

        if self.scale_b == 0.0:
            self.scale_b = 1.0
        if self.scale_p == 0.0:
            self.scale_p = 1.0
        if self.scale_gam == 0.0:
            self.scale_gam = 1.0
        if self.scale_rho == 0.0:
            self.scale_rho = 1.0
        if self.compac <= 0.0:
            self.compac = _COMPAC_DEFAULT

        if self.d == 0.0 and self.gas_d != 0.0:
            self.d = self.gas_d
        elif self.gas_d == 0.0 and self.d != 0.0:
            self.gas_d = self.d

        if self.eg == 0.0 and self.gas_eg != 0.0:
            self.eg = self.gas_eg
        elif self.gas_eg == 0.0 and self.eg != 0.0:
            self.gas_eg = self.eg

        # Compute initial internal energy e0 = P0 / ((1 + mu0) * exp((1 + mu0) * rho0 / D))
        if self.e0 == 0.0 and self.d > 0.0 and self.rho0 > 0.0 and self.p0 != 0.0:
            rhor = self.refer_rho if self.refer_rho > 0.0 else self.rho0
            mu0 = (self.rho0 / rhor) - 1.0
            denom = (1.0 + mu0) * math.exp((1.0 + mu0) * self.rho0 / self.d)
            if abs(denom) > _EM20:
                self.e0 = self.p0 / denom

    @property
    def rho(self) -> float:
        return self.rho0

    @property
    def K(self) -> float:
        return self.bulk

    @property
    def young(self) -> float:
        """Equivalent Young's modulus with nominal Poisson's ratio 0.3: E = 3K(1 - 2*0.3) = 1.2K."""
        return 1.2 * self.bulk

    @property
    def nu(self) -> float:
        return 0.3

    @property
    def G(self) -> float:
        """Equivalent shear modulus: G = 1.2K / (2 * 1.3) = 6K / 13."""
        return (6.0 / 13.0) * self.bulk

    @property
    def sound_speed(self) -> float:
        r = self.refer_rho if self.refer_rho > 0.0 else (self.rho0 if self.rho0 > 0.0 else 1.0)
        return math.sqrt(max(0.0, max(0.0, self.bulk) / max(1e-20, r)))

    @property
    def sound_speed_solid(self) -> float:
        return self.sound_speed


def build_law105(mat: Any = None, **kwargs: Any) -> PowderBurnParams:
    """Construct PowderBurnParams from material entity, dict, or keyword arguments."""
    p_dict: Dict[str, Any] = {}
    if mat is not None:
        for attr in (
            "id", "title", "rho", "rho0", "refer_rho", "rhor", "density",
            "bulk", "p0", "psh", "e0", "gas_d", "gas_eg", "gr", "c", "alpha",
            "c1", "c2", "func_b", "scale_b", "scale_p", "func_gam", "scale_gam", "scale_rho",
            "compac", "curve_b", "curve_gam", "params"
        ):
            if hasattr(mat, attr):
                p_dict[attr] = getattr(mat, attr)
        if hasattr(mat, "params") and isinstance(mat.params, dict):
            p_dict.update(mat.params)

    p_dict.update(kwargs)

    rho0_val = float(p_dict.get("rho0", p_dict.get("rho", p_dict.get("density", p_dict.get("MAT_RHO", 0.0)))))
    refer_rho_val = float(p_dict.get("refer_rho", p_dict.get("rhor", rho0_val)))
    if refer_rho_val <= 0.0:
        refer_rho_val = rho0_val

    bulk_val = float(p_dict.get("bulk", p_dict.get("POWDER_BULK", 0.0)))
    p0_val = float(p_dict.get("p0", p_dict.get("POWDER_P0", 0.0)))
    psh_val = float(p_dict.get("psh", p_dict.get("MAT_PSH", 0.0)))
    e0_val = float(p_dict.get("e0", 0.0))

    d_val = float(p_dict.get("gas_d", p_dict.get("d", p_dict.get("DD", p_dict.get("GAS_D", 0.0)))))
    eg_val = float(p_dict.get("gas_eg", p_dict.get("eg", p_dict.get("EG", p_dict.get("GAS_EG", 0.0)))))

    gr_val = float(p_dict.get("gr", p_dict.get("POWDER_Gr", 0.0)))
    c_val = float(p_dict.get("c", p_dict.get("POWDER_C", 0.0)))
    alpha_val = float(p_dict.get("alpha", p_dict.get("Alpha", 0.0)))

    c1_val = float(p_dict.get("c1", p_dict.get("MAT_C1", 0.0)))
    c2_val = float(p_dict.get("c2", p_dict.get("MAT_C2", 0.0)))

    func_b_val = int(p_dict.get("func_b", p_dict.get("POWDER_B_FUNC", 0)))
    scale_b_val = float(p_dict.get("scale_b", p_dict.get("POWDER_SCALE_B", p_dict.get("scale_b_unit", 1.0))))
    scale_p_val = float(p_dict.get("scale_p", p_dict.get("POWDER_SCALE_P", p_dict.get("scale_p_unit", 1.0))))

    func_gam_val = int(p_dict.get("func_gam", p_dict.get("POWDER_GAM_FUNC", 0)))
    scale_gam_val = float(p_dict.get("scale_gam", p_dict.get("POWDER_SCALE_GAM", p_dict.get("scale_g_unit", 1.0))))
    scale_rho_val = float(p_dict.get("scale_rho", p_dict.get("POWDER_SCALE_RHO", p_dict.get("scale_rho_unit", 1.0))))

    compac_val = float(p_dict.get("compac", _COMPAC_DEFAULT))

    curve_b = p_dict.get("curve_b")
    curve_gam = p_dict.get("curve_gam")

    res = PowderBurnParams(
        id=int(p_dict.get("id", 0)),
        title=str(p_dict.get("title", "")),
        rho0=rho0_val,
        refer_rho=refer_rho_val,
        bulk=bulk_val,
        p0=p0_val,
        psh=psh_val,
        e0=e0_val,
        d=d_val,
        eg=eg_val,
        gr=gr_val,
        c=c_val,
        alpha=alpha_val,
        c1=c1_val,
        c2=c2_val,
        func_b=func_b_val,
        scale_b=scale_b_val,
        scale_p=scale_p_val,
        func_gam=func_gam_val,
        scale_gam=scale_gam_val,
        scale_rho=scale_rho_val,
        compac=compac_val,
        curve_b=curve_b,
        curve_gam=curve_gam,
        extra=p_dict,
    )
    res.params = p_dict
    return res


def resolve(mat: Any, model: Any, log: Any = None) -> None:
    """Resolve func_b and func_gam curve IDs against model functions/curves."""
    p = getattr(mat, "params", {})
    if not isinstance(p, dict):
        p = {}

    for fid_key, fct_key in (("func_b", "curve_b"), ("func_gam", "curve_gam")):
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
                log.warning(
                    f"/MAT/LAW105/{getattr(mat, 'id', 0)}: function curve ID {fid} for {fid_key} not found in model",
                    "MAT CHECK",
                )


def _eval_curve(curve: Any, x: float) -> float:
    """Evaluate a curve, function, callable, or tabulated lookup at abscissa x."""
    if curve is None:
        return 0.0
    if callable(curve):
        return float(curve(x))
    if hasattr(curve, "evaluate"):
        return float(curve.evaluate(x))
    if hasattr(curve, "interp"):
        return float(curve.interp(x))
    if hasattr(curve, "get_value"):
        return float(curve.get_value(x))
    if isinstance(curve, (tuple, list)) and len(curve) >= 2:
        xs, ys = curve[0], curve[1]
        return float(np.interp(x, xs, ys))
    if isinstance(curve, (int, float)):
        return float(curve)
    return 0.0


def solid_update_single(
    params: PowderBurnParams,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: float = 0.0,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, float, float]:
    """Single-element update for /MAT/LAW105 replicating sigeps105.F.

    Args:
        params: PowderBurnParams instance.
        sig: Stress tensor [sxx, syy, szz, sxy, syz, szx] (6,).
        deps: Strain increment [dexx, deyy, dezz, dexy, deyz, dezx] (6,).
        epsp: Effective plastic strain (unused for hydro EOS, preserved).
        dt: Timestep dt1.
        extra: Environmental dictionary containing state, history, time, etc.

    Returns:
        (sig_new, epsp_new, sound_speed)
    """
    if extra is None:
        extra = {}

    time = float(extra.get("time", extra.get("tt", 0.0)))
    rho0 = params.rho0 if params.rho0 > 0.0 else float(extra.get("rho0", 1.0))
    rhor = params.refer_rho if params.refer_rho > 0.0 else rho0

    # Volume and density handling
    dvol = float(np.sum(deps[:3]))  # Volumetric strain trace
    vol_old = float(extra.get("volume", extra.get("vol", extra.get("voln", 1.0))))
    vol_new = float(extra.get("voln", vol_old * (1.0 + dvol)))
    if vol_new <= 0.0:
        vol_new = max(_EM12, vol_old)

    rho_old = float(extra.get("rho", rho0))
    rho_new = rho_old / max(0.1, (1.0 + dvol)) if "rho" not in extra else float(extra["rho"])
    if rho_new <= 0.0:
        rho_new = rho0

    xl = float(extra.get("deltax", extra.get("xl", extra.get("dx", vol_new ** (1.0 / 3.0)))))
    if xl <= 0.0:
        xl = vol_new ** (1.0 / 3.0)

    off = float(extra.get("off", 1.0))
    psh = float(extra.get("psh", params.psh))

    # Retrieve or allocate UVAR(1:7)
    # UVAR(1): PS, UVAR(2): PG, UVAR(3): RHO_S, UVAR(4): RHO_G, UVAR(5): POLD, UVAR(6): F(t), UVAR(7): Mass0
    uvar = extra.get("uvar", extra.get("history"))
    if uvar is None or len(uvar) < 7:
        uvar = np.zeros(7, dtype=np.float64)
        extra["uvar"] = uvar

    # Lighting / detonation time: In Radioss, TBURN is negative (-tb)
    tb_val = float(extra.get("tburn", extra.get("t_burn", extra.get("tb", 0.0))))
    tb = -tb_val if tb_val < 0.0 else tb_val

    # Element ignition fraction BFRAC
    bfrac = float(extra.get("bfrac", 0.0))

    # Initial step dt == 0 (matching sigeps105.F lines 186-212)
    if dt <= 0.0 or extra.get("initial_step", False):
        p_init = params.p0 - psh
        uvar[0] = p_init
        uvar[1] = p_init
        uvar[2] = rho0 / params.compac
        uvar[3] = 0.0
        uvar[4] = p_init
        uvar[5] = 0.0
        uvar[6] = rho0 * vol_new

        ssp = math.sqrt(max(0.0, params.bulk / rho0))
        sig_out = np.array([-p_init, -p_init, -p_init, 0.0, 0.0, 0.0], dtype=np.float64) * off
        extra["bfrac"] = 0.0
        extra["sound_speed"] = ssp
        extra["uvar"] = uvar
        extra["initial_step"] = False
        return sig_out, epsp, ssp

    # Ensure Mass0 is initialized
    if uvar[6] <= 0.0:
        uvar[6] = rho0 * vol_new
    mass0 = uvar[6]

    # 1. Flame ignition fraction BFRAC (sigeps105.F lines 231-240)
    if bfrac < 1.0:
        bfrac = 0.0
        if time > tb and params.c1 > 0.0:
            bfrac = params.c1 * (time - tb) * _TWO_THIRD / max(_EM20, xl)
        if bfrac < _EM04:
            bfrac = 0.0
        elif bfrac > 1.0:
            bfrac = 1.0

    # 2. Grain burn fraction F(t) (sigeps105.F lines 247-261)
    f_burn = uvar[5]
    pg = uvar[1]  # Gas pressure from previous step

    if bfrac <= 0.0:
        total_bfrac = 0.0
        f_burn = 0.0
    elif f_burn < 1.0:
        # Pressure for burn rate evaluation
        pg_eval = pg
        curve_b = params.curve_b or extra.get("curve_b")
        if curve_b is not None:
            brate_raw = _eval_curve(curve_b, pg_eval / params.scale_p)
        else:
            # If no curve is provided, default to 1.0 so brate = scale_b
            brate_raw = 1.0

        brate = params.scale_b * brate_raw
        arg = max(_EM12, 1.0 - params.alpha * f_burn)
        growth_factor = arg ** params.c
        delta_bf = params.gr * growth_factor * brate * dt
        f_burn = min(1.0, max(0.0, f_burn + delta_bf))
        total_bfrac = bfrac * f_burn
    else:
        total_bfrac = bfrac * f_burn

    # 3. EOS solving (sigeps105.F lines 266-287)
    mass_s = (1.0 - f_burn) * mass0
    mass_g = mass0 - mass_s

    rho_s = (1.0 - f_burn) * rho_new

    # Compressible solid grain density: RHO_Si = RHO0 / compac * (PG / BULK + 1.0)
    rho_si = (rho0 / params.compac) * ((pg / max(_EM20, params.bulk)) + 1.0)
    rho_si_eff = max(_EM10, rho_si)

    # Available gas volume: V_gas = VOLN - MASS_S / RHO_Si
    vol_gas = vol_new - (mass_s / rho_si_eff)
    rho_g = 0.0
    if mass_g > 0.0 and vol_gas > 0.0:
        rho_g = mass_g / max(_EM12, vol_gas)

    # Gas pressure: PG = P0 + RHO_G * EG * EXP(RHO_G / D)
    pg_new = params.p0
    if rho_g > 0.0 and params.d > 0.0:
        pg_new += rho_g * params.eg * math.exp(rho_g / params.d)

    # Solid pressure: PS = P0 + BULK * (RHO_Si / RHO0 - 1.0)
    ps_new = params.p0 + params.bulk * ((rho_si / rho0) - 1.0)

    # Sound speeds
    ssp_s = math.sqrt(max(0.0, params.bulk / rho0))
    ssp_g = 0.0
    if rho_g > 0.0 and params.d > 0.0:
        term_g = (rho0 / params.d) * pg_new + math.exp(rho_g / params.d) * pg_new / max(_EM20, (rho_g / rho0) ** 2)
        ssp_g = math.sqrt(max(0.0, (1.0 / rho_g) * term_g))

    # Global pressure and sound speed
    ssp = total_bfrac * ssp_g + (1.0 - total_bfrac) * ssp_s
    pnew = total_bfrac * pg_new + (1.0 - total_bfrac) * ps_new
    pnew_eff = (pnew - psh) * off

    # Hydrodynamic stress: sigma = -pnew_eff * I (deviatoric stresses are zero)
    sig_out = np.zeros(6, dtype=np.float64)
    sig_out[:3] = -pnew_eff

    # State update
    uvar[0] = ps_new
    uvar[1] = pg_new
    uvar[2] = rho_s
    uvar[3] = rho_g
    uvar[4] = pnew_eff
    uvar[5] = f_burn
    uvar[6] = mass0

    extra["bfrac"] = bfrac
    extra["total_bfrac"] = total_bfrac
    extra["sound_speed"] = ssp
    extra["uvar"] = uvar
    extra["pressure"] = pnew_eff
    extra["rho_g"] = rho_g
    extra["rho_s"] = rho_s

    return sig_out, epsp, ssp


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = 0.0,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_tuple: bool = True,
) -> Any:
    """Vectorized or single-element solid update entry point for /MAT/LAW105.

    Dispatched by pyradioss.materials.solid_update.
    """
    params = build_law105(mat) if not isinstance(mat, PowderBurnParams) else mat
    if extra is None:
        extra = {}

    sig_arr = np.asarray(sig, dtype=np.float64)
    deps_arr = np.asarray(deps, dtype=np.float64)

    # 1D single-element call
    if sig_arr.ndim == 1:
        epsp_val = float(epsp) if epsp is not None and not isinstance(epsp, np.ndarray) else 0.0
        sig_new, epsp_out, ssp = solid_update_single(params, sig_arr, deps_arr, epsp_val, dt, extra)
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sig_new
            except Exception:
                pass
        if return_tuple:
            return sig_new, epsp_out, ssp
        return sig_new

    # Multi-element 2D call (nel, 6)
    nel = sig_arr.shape[0]
    sig_out = np.zeros_like(sig_arr)
    epsp_out = np.zeros(nel, dtype=np.float64)
    ssp_out = np.zeros(nel, dtype=np.float64)

    for i in range(nel):
        elem_extra: Dict[str, Any] = {}
        for k, v in extra.items():
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) == nel:
                elem_extra[k] = v[i]
            else:
                elem_extra[k] = v

        ep_in = float(epsp[i]) if (epsp is not None and isinstance(epsp, (list, tuple, np.ndarray))) else 0.0
        s_res, ep_res, ssp_res = solid_update_single(params, sig_arr[i], deps_arr[i], ep_in, dt, elem_extra)
        sig_out[i] = s_res
        epsp_out[i] = ep_res
        ssp_out[i] = ssp_res

    if hasattr(sig, "__setitem__"):
        try:
            sig[:] = sig_out
        except Exception:
            pass

    if return_tuple:
        return sig_out, epsp_out, ssp_out
    return sig_out


def sound_speed(
    mat: Any,
    rho: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
    is_shell: bool = False,
) -> Any:
    """Compute acoustic wave speed for /MAT/LAW105.

    If simulation is ongoing, returns mixture sound speed from extra if present,
    otherwise unburnt solid sound speed sqrt(Bulk / rho0).
    """
    params = build_law105(mat) if not isinstance(mat, PowderBurnParams) else mat
    if extra is not None and "sound_speed" in extra:
        return extra["sound_speed"]

    bulk = max(0.0, params.bulk)
    if rho is not None:
        r = np.asarray(rho, dtype=np.float64)
        r_val = np.where(r > 0.0, r, (params.rho0 if params.rho0 > 0.0 else 1.0))
        c = np.sqrt(np.maximum(0.0, bulk / np.maximum(1e-20, r_val)))
        return float(c) if r.ndim == 0 else c

    rho0 = params.rho0 if params.rho0 > 0.0 else 1.0
    return float(math.sqrt(max(0.0, bulk / max(1e-20, rho0))))


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Consistent 6x6 tangent stiffness matrix for /MAT/LAW105.

    For a hydrodynamic EOS material with effective bulk modulus K_eff = rho * c^2:
      C_11 = C_22 = C_33 = C_12 = C_13 = C_23 = K_eff
      All shear tangent moduli are zero (C_44 = C_55 = C_66 = 0).
    """
    params = build_law105(mat) if not isinstance(mat, PowderBurnParams) else mat
    c = sound_speed(params, extra=extra)
    rho_val = float(extra.get("rho", params.rho0)) if extra else params.rho0
    if rho_val <= 0.0:
        rho_val = params.rho0 if params.rho0 > 0.0 else 1.0

    k_eff = rho_val * (c ** 2) if c > 0.0 else params.bulk

    c_mat = np.zeros((6, 6), dtype=np.float64)
    # Hydrostatic pressure tangent
    c_mat[0:3, 0:3] = k_eff
    return c_mat


def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Persistent state arrays required by LAW105 solid elements."""
    return {
        "uvar": (nip, 7) if nip is not None else (7,),
        "bfrac": (nip,) if nip is not None else (),
        "total_bfrac": (nip,) if nip is not None else (),
    }

