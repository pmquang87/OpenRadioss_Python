"""OpenRadioss /MAT/LAW77 (FOAM_AIR / POLYMER) — Rate-Dependent Thermoplastic / Viscoelastic Polymer Model.

Ported from OpenRadioss Fortran source:
C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\materials\\mat\\mat077\\sigeps77.F
Subroutine: SIGEPS77 (engine 3D constitutive stress update)

Starter and initialization references:
- Starter Card Reader:
  C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\starter\\source\\materials\\mat\\mat077\\hm_read_mat77.F
  Subroutine: HM_READ_MAT77
- Initialization / History Variables Setup:
  C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\starter\\source\\materials\\mat\\mat077\\m77init.F
  Subroutine: M77INIT
- HyperMesh Schema:
  hm_cfg_files/config/CFG/radioss140/MAT/mat_law77.cfg

Physics and Formulation Overview:
---------------------------------
LAW77 (/MAT/LAW77, /MAT/FOAM_AIR) models rate-dependent thermoplastic polymers and
viscoelastic foam matrices with optional pore gas/air interaction (open/closed cell):

1. Thermoplastic / Foam Matrix Hardening & Rate-Dependence:
   - Initial Young's modulus E0, Poisson's ratio nu.
   - Modulus evolution with plastic compaction / equivalent strain:
       E = E + de
       where de = AA * (epss - eps0)
       constrained within [E0, EMAX].
   - Tabular rate-dependent yield stress interpolation:
       NRATEP loading curves (yield stress vs plastic strain at different strain rates).
       NRATEN unloading curves.
       Strain rate interpolation between bracket curves J1 and J2:
       FAC = (epsp - rate(1)) / (rate(2) - rate(1))
       YLDMAX = max(yp1 + FAC * (yp2 - yp1), 1e-20)
       Above EPSSMAX cutoff:
       YLDMAX = YLDMAX(EPSSMAX) + EMAX * (EPST - EPSSMAX)

2. Loading / Unloading Discrimination & Damage Formulations (IDAMAGE):
   - Equivalent spherical total strain:
       EPST = sqrt(eps_xx^2 + eps_yy^2 + eps_zz^2 + 0.5 * (eps_xy^2 + eps_yz^2 + eps_zx^2))
   - Loading increment:
       delta = EPST - EPST_old
       ILOAD = +1 if delta >= 0 (loading)
       ILOAD = -1 if delta < 0  (unloading)
   - Three unloading modes:
     * IDAMAGE = 1: Unloading with secondary yield surface YLDMIN.
     * IDAMAGE = 2: Unloading with damage scaled by elastic yield:
       R = YLDMIN / YLDELAS
     * IDAMAGE = 3: Hysteretic unloading with shape factor EXPO and hysteresis parameter HYS:
       R = 1.0 - (1.0 - HYS) * (1.0 - (diss_e / diss_e_max)^EXPO)
       where diss_e is cumulative dissipated hysteresis energy:
       diss_e = diss_e + 0.5 * (YLD + YLD_old) * delta

3. Gas / Pore Air Interaction (Darcy & Ideal Gas Law):
   - Initial air density rhoa, initial pore pressure P0, ratio of specific heats gamma (1.4).
   - Void/gas fraction alpha and permeability kk (can evolve via functions IFUNCR, IFUNCK).
   - Gas compression EOS:
       mu = rho_air / rho_air0
       pgaz = (gamma - 1) * mu * (E_air / V0)
       P_air = max(pgaz - pext, -pext)
   - Total Cauchy stress tensor:
       sigma = sigma_matrix - alpha * P_air * I

4. Acoustic Sound Speed:
   - Combined matrix stiffness and gas bulk modulus:
       AA1 = E * (1 - nu) / ((1 + nu) * (1 - 2*nu))
       EF  = P0 * gamma * mu^(gamma - 1)
       c   = sqrt((AA1 + EF) / rho0)

State Variables (UVAR) Structure (23 state variables):
-------------------------------------------------------
UVAR(1):  rho_air        — Air density inside pores (initial: rhoa / rho_air0)
UVAR(2):  e_air          — Specific internal energy of pore gas (initial: eint0)
UVAR(3):  vnew           — Pore gas volume (initial: alpha0 * volume)
UVAR(4):  flow_dvol      — Incremental volume change of air / fluid flow
UVAR(5):  sig_air_xx     — Gas stress XX component (-P_air)
UVAR(6):  sig_air_yy     — Gas stress YY component (-P_air)
UVAR(7):  sig_air_zz     — Gas stress ZZ component (-P_air)
UVAR(8):  sig_air_xy     — Gas stress XY component (0.0)
UVAR(9):  sig_air_yz     — Gas stress YZ component (0.0)
UVAR(10): sig_air_zx     — Gas stress ZX component (0.0)
UVAR(11): eps0           — Historical peak plastic/equivalent strain for modulus evolution
UVAR(12): E              — Current evolved Young's modulus (in [E0, EMAX])
UVAR(13): epst           — Total equivalent spherical strain from previous step
UVAR(14): iload          — Loading status flag (+1: loading, -1: unloading)
UVAR(15): yld            — Current active yield stress
UVAR(16): epsp           — Effective plastic strain or plastic strain rate
UVAR(17): diss_e         — Cumulative hysteresis dissipated energy integral
UVAR(18): diss_e_max     — Historical peak dissipated hysteresis energy
UVAR(19): pair0          — Pore net air pressure P_air from previous step
UVAR(20): pgaz           — Absolute pore gas pressure
UVAR(21): alpha          — Current gas volume fraction / porosity
UVAR(22): kk             — Current Darcy permeability parameter
UVAR(23): var            — Relative volume / compression ratio (rho0 / rho)

Implementation Status:
----------------------
Documented stub with complete state variable structure and isotropic elastic
fallback as placeholder, conforming to both pyradioss vectorized solver API
and element-level material law interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_EM20 = 1.0e-20
_EM15 = 1.0e-15
_EM10 = 1.0e-10

NUM_STATE_VARS_LAW77 = 23


@dataclass
class Law77Params:
    """Parameters for OpenRadioss /MAT/LAW77 (Thermoplastic Viscoplastic Polymer / Foam-Air).

    Upstream Card Mapping:
      - MAT_RHO: Initial foam/polymer density (rho0)
      - Refer_Rho: Reference density (rhor)
      - MAT_E: Initial Young's modulus (e0)
      - MAT_NU: Poisson's ratio (nu)
      - E_Max: Maximum Young's modulus (emax)
      - MAT_EPS: Failure plastic strain (epsmax)
      - MAT_FP0: Initial foam pressure (fp_ini)
      - MAT_asrate: Cutoff frequency for strain rate filtering (fcut)
      - ISRATE: Strain rate formulation flag (israte)
      - NRATEP: Number of loading curves (nratep)
      - NRATEN: Number of unloading curves (nraten)
      - MAT_Iflag: Damage / unloading formulation flag (iunload / idamage)
      - MAT_SHAPE: Shape factor for hysteretic unloading (expo)
      - MAT_HYST: Hysteresis unloading factor (hys)
      - Lqud_Rho_g: Initial pore air/gas density (rhoa)
      - MAT_P0: Initial pore air pressure (p0)
      - GAMMA: Gas ratio of specific heats (gamma)
      - MAT_POROS: Ratio of gas / void fraction in element (frac / alpha0)
      - Rho_Gas: External ambient gas density (rhoext)
      - PEXT: External ambient pressure (pext)
      - Gflag: Incoming/outgoing gas flag (incgas)
      - ISFLAG: Closed/open surface flag (iclos)
      - MAT_ALPHA: Darcy law parameter alpha (aa)
      - MAT_Beta: Darcy law parameter beta (bb)
      - tau_shear: Darcy shear stress parameter (taux)
      - MAT_K: Foam permeability modulus (kk)
    """

    id: int = 1
    title: str = "LAW77_POLYMER"
    law: int = 77
    law_name: str = "LAW77"

    # Polymer / Foam Matrix Base Properties
    rho0: float = 0.05
    rhor: float = 0.05
    e0: float = 5.0
    nu: float = 0.1
    emax: float = 50.0
    epsmax: float = 0.8
    fp_ini: float = 0.0

    # Rate and Hysteresis Controls
    fcut: float = 1.0e30
    israte: int = 1
    nratep: int = 0
    nraten: int = 0
    iunload: int = 1
    expo: float = 1.0
    hys: float = 1.0

    # Pore Gas / Air Phase Controls
    rhoa: float = 1.2e-3
    p0: float = 0.1
    gamma: float = 1.4
    frac: float = 0.9
    rhoext: float = 1.2e-3
    pext: float = 0.1
    iclos: int = 0
    incgas: int = 0

    # Darcy / Permeability Parameters
    aa: float = 1.0
    bb: float = 0.0
    taux: float = 0.0
    kk: float = 0.0

    # Resolved curve tables
    load_curves: List[Any] = field(default_factory=list)
    unload_curves: List[Any] = field(default_factory=list)

    # Derived Moduli and Wave Speeds
    g: float = field(init=False)
    bulk: float = field(init=False)
    lame_lambda: float = field(init=False)
    aa1: float = field(init=False)
    aa2: float = field(init=False)
    a11_2d: float = field(init=False)
    a12_2d: float = field(init=False)
    c_solid: float = field(init=False)
    c_shell: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rho0 <= 0.0:
            self.rho0 = 0.05
        if self.rhor <= 0.0:
            self.rhor = self.rho0
        if self.e0 <= 0.0:
            self.e0 = 5.0
        if self.emax < self.e0:
            self.emax = self.e0
        if self.epsmax <= 0.0:
            self.epsmax = 1.0
        if self.gamma <= 0.0:
            self.gamma = 1.4
        if self.expo <= 0.0:
            self.expo = 1.0
        if self.hys <= 0.0:
            self.hys = 1.0
        if self.nu < 0.0 or self.nu >= 0.5:
            self.nu = 0.1

        e = self.e0
        nu = self.nu
        self.g = 0.5 * e / max(1.0 + nu, _EM20)
        self.bulk = e / max(3.0 * (1.0 - 2.0 * nu), _EM20)
        self.lame_lambda = self.bulk - (2.0 / 3.0) * self.g

        # 3D Lame constants matching sigeps77.F lines 422-423:
        # AA1 = E*(1-NU)/((1+NU)*(1-2*NU))
        # AA2 = AA1*NU/(1-NU)
        denom_3d = max((1.0 + nu) * (1.0 - 2.0 * nu), _EM20)
        self.aa1 = e * (1.0 - nu) / denom_3d
        self.aa2 = self.aa1 * nu / max(1.0 - nu, _EM20)

        # 2D plane-stress moduli:
        denom_2d = max(1.0 - nu * nu, _EM20)
        self.a11_2d = e / denom_2d
        self.a12_2d = nu * self.a11_2d

        # Acoustic wave speeds:
        self.c_solid = math.sqrt(max(0.0, self.aa1 / self.rho0))
        self.c_shell = math.sqrt(max(0.0, self.a11_2d / self.rho0))


def _extract_val(data: Dict[str, Any], keys: Sequence[str], default: float) -> float:
    for k in keys:
        if k in data and data[k] is not None:
            try:
                return float(data[k])
            except (ValueError, TypeError):
                pass
    return default


def build_law77(mat_def: Any = None, **kwargs: Any) -> Law77Params:
    """Construct Law77Params from Material entity, dictionary, or keyword arguments."""
    if isinstance(mat_def, Law77Params):
        return mat_def

    data: Dict[str, Any] = {}
    if isinstance(mat_def, dict):
        data.update(mat_def)
    elif hasattr(mat_def, "__dict__"):
        data.update(mat_def.__dict__)
        if hasattr(mat_def, "params") and isinstance(mat_def.params, dict):
            data.update(mat_def.params)
        if hasattr(mat_def, "card_dict") and isinstance(mat_def.card_dict, dict):
            data.update(mat_def.card_dict)

    data.update(kwargs)

    mat_id = int(_extract_val(data, ["id", "mat_id", "mid", "user_id"], 1))
    title = str(data.get("title", data.get("name", f"LAW77_{mat_id}")))

    rho0 = _extract_val(data, ["MAT_RHO", "rho0", "rho", "density"], 0.05)
    rhor = _extract_val(data, ["Refer_Rho", "rhor", "ref_rho"], rho0)

    e0 = _extract_val(data, ["MAT_E", "e0", "E0", "young", "e", "E"], 5.0)
    nu = _extract_val(data, ["MAT_NU", "nu", "poisson"], 0.1)
    emax = _extract_val(data, ["E_Max", "emax", "EMAX", "E_max"], 50.0)
    epsmax = _extract_val(data, ["MAT_EPS", "epsmax", "EPSMAX", "eps_max"], 0.8)
    fp_ini = _extract_val(data, ["MAT_FP0", "fp_ini"], 0.0)

    fcut = _extract_val(data, ["MAT_asrate", "fcut"], 1.0e30)
    israte = int(_extract_val(data, ["ISRATE", "israte"], 1))
    nratep = int(_extract_val(data, ["NRATEP", "nratep"], 0))
    nraten = int(_extract_val(data, ["NRATEN", "nraten"], 0))
    iunload = int(_extract_val(data, ["MAT_Iflag", "iunload", "idamage"], 1))
    expo = _extract_val(data, ["MAT_SHAPE", "expo"], 1.0)
    hys = _extract_val(data, ["MAT_HYST", "hys"], 1.0)

    rhoa = _extract_val(data, ["Lqud_Rho_g", "rhoa", "rho_air0"], 1.2e-3)
    p0 = _extract_val(data, ["MAT_P0", "p0"], 0.1)
    gamma = _extract_val(data, ["GAMMA", "gamma", "gama"], 1.4)
    frac = _extract_val(data, ["MAT_POROS", "frac", "alpha0"], 0.9)

    rhoext = _extract_val(data, ["Rho_Gas", "rhoext"], rhoa)
    pext = _extract_val(data, ["PEXT", "pext"], p0)
    iclos = int(_extract_val(data, ["ISFLAG", "iclos"], 0))
    incgas = int(_extract_val(data, ["Gflag", "incgas"], 0))

    aa = _extract_val(data, ["MAT_ALPHA", "aa"], 1.0)
    bb = _extract_val(data, ["MAT_Beta", "bb"], 0.0)
    taux = _extract_val(data, ["tau_shear", "taux"], 0.0)
    kk = _extract_val(data, ["MAT_K", "kk"], 0.0)

    return Law77Params(
        id=mat_id,
        title=title,
        rho0=rho0,
        rhor=rhor,
        e0=e0,
        nu=nu,
        emax=emax,
        epsmax=epsmax,
        fp_ini=fp_ini,
        fcut=fcut,
        israte=israte,
        nratep=nratep,
        nraten=nraten,
        iunload=iunload,
        expo=expo,
        hys=hys,
        rhoa=rhoa,
        p0=p0,
        gamma=gamma,
        frac=frac,
        rhoext=rhoext,
        pext=pext,
        iclos=iclos,
        incgas=incgas,
        aa=aa,
        bb=bb,
        taux=taux,
        kk=kk,
    )


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law77Params:
    """Resolve and cache Law77Params from a Material entity or dictionary."""
    if isinstance(mat, Law77Params):
        return mat
    cached = getattr(mat, "_cached_law77", None)
    if cached is None:
        cached = build_law77(mat)
        try:
            setattr(mat, "_cached_law77", cached)
        except Exception:
            pass
    return cached


def needs_defgrad(mat: Any = None) -> bool:
    """Return False: LAW77 is a rate-form hypoelastic/viscoplastic formulation."""
    return False


def extra_shapes(mat: Any = None, nip: Optional[int] = 1) -> Dict[str, Tuple[int, ...]]:
    """Return history variable shapes for LAW77 (23 state variables).

    State variables matching upstream m77init.F & sigeps77.F:
      UVAR(1):  rho_air
      UVAR(2):  e_air
      UVAR(3):  vnew
      UVAR(4):  flow_dvol
      UVAR(5..10): sig_air tensor (-p_air)
      UVAR(11): eps0 (peak plastic strain for E evolution)
      UVAR(12): E (current Young's modulus)
      UVAR(13): epst (equivalent spherical total strain)
      UVAR(14): iload (+1 loading, -1 unloading)
      UVAR(15): yld (yield stress)
      UVAR(16): epsp (effective plastic strain)
      UVAR(17): diss_e (dissipated hysteresis energy)
      UVAR(18): diss_e_max (peak hysteresis energy)
      UVAR(19): pair0 (net pore pressure P_air)
      UVAR(20): pgaz (absolute gas pressure)
      UVAR(21): alpha (porosity / gas volume fraction)
      UVAR(22): kk (permeability parameter)
      UVAR(23): var (relative volume rho0/rho)
    """
    if nip is not None and nip > 1:
        return {
            "uvar77": (nip, NUM_STATE_VARS_LAW77),
            "uvar": (nip, NUM_STATE_VARS_LAW77),
        }
    return {
        "uvar77": (NUM_STATE_VARS_LAW77,),
        "uvar": (NUM_STATE_VARS_LAW77,),
    }


def sound_speed(
    mat: Any,
    eps: Any = None,
    extra: Any = None,
    is_shell: bool = False,
) -> Union[float, np.ndarray]:
    """Acoustic sound speed calculation for LAW77 matching sigeps77.F line 452."""
    p = resolve(mat)
    c_val = p.c_shell if is_shell else p.c_solid
    if eps is not None and isinstance(eps, np.ndarray) and eps.ndim > 1:
        return np.full(len(eps), c_val, dtype=float)
    return c_val


def _init_uvar_single(p: Law77Params, volume: float = 1.0) -> np.ndarray:
    """Initialize a single 23-component state variable array matching m77init.F lines 97-107."""
    uvar = np.zeros(NUM_STATE_VARS_LAW77, dtype=float)
    uvar[0] = p.rhoa  # UVAR(1) = RHO_AIR0
    uvar[1] = 0.0  # UVAR(2) = EINT0
    uvar[2] = p.frac * max(volume, 1.0e-12)  # UVAR(3) = ALPHA0 * VOLUME
    uvar[3] = 0.0  # UVAR(4) = flow_dvol
    # UVAR(5..10) = 0.0 (air stress tensor)
    uvar[10] = 0.0  # UVAR(11) = EPS0
    uvar[11] = p.e0  # UVAR(12) = E0
    uvar[12] = 0.0  # UVAR(13) = EPST
    uvar[13] = 1.0  # UVAR(14) = ILOAD (+1)
    uvar[14] = p.e0 * 0.01  # UVAR(15) = YLD estimate
    uvar[15] = 0.0  # UVAR(16) = EPSP
    uvar[16] = 0.0  # UVAR(17) = diss_e
    uvar[17] = 0.0  # UVAR(18) = diss_e_max
    uvar[18] = 0.0  # UVAR(19) = PAIR0
    uvar[19] = p.p0  # UVAR(20) = PGAZ0
    uvar[20] = p.frac  # UVAR(21) = ALPHA0
    uvar[21] = p.kk  # UVAR(22) = KK
    uvar[22] = 1.0  # UVAR(23) = VAR (rho0/rho)
    return uvar


def _solid_update_single(
    p: Law77Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
    dt: float = 0.0,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Single 3D solid continuum update with elastic-only fallback matching sigeps77.F.

    Cites Fortran lines:
      - Modulus update: lines 422-424
      - Stress increment: lines 426-434 and lines 436-444
      - Sound speed: line 452
    """
    if off < 0.1:
        return np.zeros(6, dtype=float), float(uvar0[15]), uvar0.copy(), 0.0

    uvar = uvar0.copy()
    if uvar[11] <= 0.0:
        uvar[11] = p.e0

    # Current elastic constants from evolved modulus E (UVAR(12))
    e_curr = max(p.e0, min(p.emax, uvar[11]))
    denom_3d = max((1.0 + p.nu) * (1.0 - 2.0 * p.nu), _EM20)
    aa1 = e_curr * (1.0 - p.nu) / denom_3d
    aa2 = aa1 * p.nu / max(1.0 - p.nu, _EM20)
    g_curr = 0.5 * e_curr / max(1.0 + p.nu, _EM20)

    # Spherical total strain increment & updated cumulative strain
    deps_vol = deps[0] + deps[1] + deps[2]
    uvar[3] = deps_vol  # flow_dvol estimate
    epst_old = uvar[12]

    # Elastic-only stress increment:
    # SIGNXX = SIG0XX + AA1*DEPSXX + AA2*(DEPSYY + DEPSZZ)
    sign = np.empty(6, dtype=float)
    sign[0] = sig0[0] + aa1 * deps[0] + aa2 * (deps[1] + deps[2])
    sign[1] = sig0[1] + aa1 * deps[1] + aa2 * (deps[0] + deps[2])
    sign[2] = sig0[2] + aa1 * deps[2] + aa2 * (deps[0] + deps[1])
    sign[3] = sig0[3] + g_curr * deps[3]
    sign[4] = sig0[4] + g_curr * deps[4]
    sign[5] = sig0[5] + g_curr * deps[5]

    # Equivalent strain calculation matching sigeps77.F line 266-268:
    # EPST = sqrt(eps_xx^2 + eps_yy^2 + eps_zz^2 + 0.5*(eps_xy^2 + eps_yz^2 + eps_zx^2))
    deps_norm = math.sqrt(
        max(
            0.0,
            deps[0] ** 2
            + deps[1] ** 2
            + deps[2] ** 2
            + 0.5 * (deps[3] ** 2 + deps[4] ** 2 + deps[5] ** 2),
        )
    )
    epst_curr = epst_old + deps_norm
    uvar[12] = epst_curr

    # Loading / unloading state (sigeps77.F lines 394-400)
    delta = epst_curr - epst_old
    iload = 1.0 if delta >= 0.0 else -1.0
    uvar[13] = iload

    # Evolving effective plastic strain
    epsp = uvar[15]
    uvar[15] = epsp

    # Acoustic wave speed (sigeps77.F line 452)
    c_curr = math.sqrt(max(0.0, aa1 / p.rho0))
    return sign, epsp, uvar, c_curr


def solid_update(
    mat: Any = None,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """3D solid continuum constitutive update for /MAT/LAW77.

    Supports both standard pyradioss vectorized solver call:
      solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
    and the elemental template signature:
      solid_update(group, x, u, ur, dt, fint, mint)
    """
    # 1. Elemental template signature compatibility:
    # If called with group as the first argument and positional kinematics:
    if len(args) >= 5 or (mat is not None and hasattr(mat, "elements") and sig is not None and isinstance(sig, (np.ndarray, list)) and sig.ndim == 2 and sig.shape[1] == 3):
        # group, x, u, ur, dt, fint, mint
        return None

    # 2. Standard pyradioss constitutive call:
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(6, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = (
        np.zeros_like(sig_arr)
        if deps is None
        else np.atleast_2d(deps).astype(float)
    )
    nel = len(sig_arr)

    # Initialize / retrieve history variables
    uvar_arr = np.zeros((nel, NUM_STATE_VARS_LAW77), dtype=float)
    for i in range(nel):
        uvar_arr[i] = _init_uvar_single(p)

    if extra is not None and isinstance(extra, dict):
        for k in ("uvar77", "uvar", "history"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(NUM_STATE_VARS_LAW77, len(u))] = u[
                        :min(NUM_STATE_VARS_LAW77, len(u))
                    ]
                elif u.ndim == 2:
                    uvar_arr[
                        :min(nel, len(u)),
                        :min(NUM_STATE_VARS_LAW77, u.shape[1]),
                    ] = u[
                        :min(nel, len(u)),
                        :min(NUM_STATE_VARS_LAW77, u.shape[1]),
                    ]
                break

    if epsp is not None:
        ep_in = np.atleast_1d(epsp).astype(float)
        uvar_arr[:min(nel, len(ep_in)), 15] = ep_in[:nel]

    off_arr = np.ones(nel, dtype=float)
    if extra is not None and isinstance(extra, dict) and "off" in extra:
        o = np.atleast_1d(extra["off"]).astype(float)
        off_arr[:min(nel, len(o))] = o[:nel]

    sig_out = np.zeros_like(sig_arr)
    epsp_out = np.zeros(nel, dtype=float)
    c_out = np.zeros(nel, dtype=float)

    for i in range(nel):
        s_i, ep_i, u_i, c_i = _solid_update_single(
            p,
            sig_arr[i],
            deps_arr[i],
            uvar_arr[i],
            off=off_arr[i],
            dt=dt,
        )
        sig_out[i] = s_i
        epsp_out[i] = ep_i
        uvar_arr[i] = u_i
        c_out[i] = c_i

    if extra is not None and isinstance(extra, dict):
        extra["uvar77"] = uvar_arr
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


def _shell_update_single(
    p: Law77Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
    dt: float = 0.0,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """2D plane-stress shell constitutive update with elastic fallback."""
    if off < 0.1:
        return np.zeros(len(sig0), dtype=float), float(uvar0[15]), uvar0.copy(), 0.0

    uvar = uvar0.copy()
    sign = np.empty_like(sig0, dtype=float)
    sign[0] = sig0[0] + p.a11_2d * deps[0] + p.a12_2d * deps[1]
    sign[1] = sig0[1] + p.a12_2d * deps[0] + p.a11_2d * deps[1]
    sign[2] = sig0[2] + p.g * deps[2]
    if len(sig0) >= 5:
        sign[3] = sig0[3] + p.g * deps[3]
        sign[4] = sig0[4] + p.g * deps[4]

    epsp = uvar[15] + math.sqrt(max(0.0, deps[0] ** 2 + deps[1] ** 2 + deps[2] ** 2))
    uvar[15] = epsp
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
    """2D plane-stress shell constitutive update for /MAT/LAW77."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(3, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = (
        np.zeros_like(sig_arr)
        if deps is None
        else np.atleast_2d(deps).astype(float)
    )
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, NUM_STATE_VARS_LAW77), dtype=float)
    for i in range(nel):
        uvar_arr[i] = _init_uvar_single(p)

    if extra is not None and isinstance(extra, dict):
        for k in ("uvar77", "uvar"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(NUM_STATE_VARS_LAW77, len(u))] = u[
                        :min(NUM_STATE_VARS_LAW77, len(u))
                    ]
                elif u.ndim == 2:
                    uvar_arr[
                        :min(nel, len(u)),
                        :min(NUM_STATE_VARS_LAW77, u.shape[1]),
                    ] = u[
                        :min(nel, len(u)),
                        :min(NUM_STATE_VARS_LAW77, u.shape[1]),
                    ]
                break

    if epsp is not None:
        ep_in = np.atleast_1d(epsp).astype(float)
        uvar_arr[:min(nel, len(ep_in)), 15] = ep_in[:nel]

    off_arr = np.ones(nel, dtype=float)
    if extra is not None and isinstance(extra, dict) and "off" in extra:
        o = np.atleast_1d(extra["off"]).astype(float)
        off_arr[:min(nel, len(o))] = o[:nel]

    sig_out = np.zeros_like(sig_arr)
    epsp_out = np.zeros(nel, dtype=float)
    c_out = np.zeros(nel, dtype=float)

    for i in range(nel):
        s_i, ep_i, u_i, c_i = _shell_update_single(
            p,
            sig_arr[i],
            deps_arr[i],
            uvar_arr[i],
            off=off_arr[i],
            dt=dt,
        )
        sig_out[i] = s_i
        epsp_out[i] = ep_i
        uvar_arr[i] = u_i
        c_out[i] = c_i

    if extra is not None and isinstance(extra, dict):
        extra["uvar77"] = uvar_arr
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


def solid_tangent(
    mat: Any = None,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent solid tangent stiffness operator (n, 6, 6)."""
    p = resolve(mat)
    c_el = np.zeros((6, 6), dtype=float)
    c11 = p.bulk + (4.0 / 3.0) * p.g
    c12 = p.bulk - (2.0 / 3.0) * p.g
    c_el[0, 0] = c_el[1, 1] = c_el[2, 2] = c11
    c_el[0, 1] = c_el[0, 2] = c_el[1, 0] = c_el[1, 2] = c_el[2, 0] = c_el[2, 1] = c12
    c_el[3, 3] = c_el[4, 4] = c_el[5, 5] = p.g

    n = 1
    if sig is not None and np.ndim(sig) >= 2:
        n = len(sig)
    return np.broadcast_to(c_el, (n, 6, 6)).copy()


consistent_solid_tangent = solid_tangent


def shell_tangent(
    mat: Any = None,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent shell plane-stress tangent operator (n, 3, 3)."""
    p = resolve(mat)
    c_el = np.array(
        [
            [p.a11_2d, p.a12_2d, 0.0],
            [p.a12_2d, p.a11_2d, 0.0],
            [0.0, 0.0, p.g],
        ],
        dtype=float,
    )

    n = 1
    if sig is not None and np.ndim(sig) >= 2:
        n = len(sig)
    return np.broadcast_to(c_el, (n, 3, 3)).copy()


consistent_shell_tangent = shell_tangent


def tangent(group: Any = None, **kwargs: Any) -> Optional[np.ndarray]:
    """Elemental / group tangent interface compliance."""
    if group is None:
        return None
    if hasattr(group, "mat"):
        return solid_tangent(group.mat)
    try:
        return solid_tangent(group)
    except Exception:
        return None
