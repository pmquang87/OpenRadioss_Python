"""pyradioss.engine.airbag_commu — Communicating airbag chambers (/MONVOL/COMMU1).

Upstream OpenRadioss Fortran reference:
- Card reader & parameters:
  `starter/source/airbag/hm_read_monvol_type9.F` (HM_READ_MONVOL_TYPE9, lines 308-323, 871-972)
- Engine mass & enthalpy balance:
  `engine/source/airbag/airbaga1.F` (lines 390-472, LEAKS AND COMMUNICATIONS)
- Engine inter-chamber orifice flow kinetics:
  `engine/source/airbag/airbag2.F` (lines 435-565, AIRBAG COMMUNIQUANTS)

Physics & Constitutive Theory:
------------------------------
1. Orifice Flow Regimes:
   For gas with heat capacity ratio gamma and specific gas constant R, the
   critical pressure ratio for sonic choking is:
       r_crit = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
   For gamma = 1.4: r_crit ~= 0.52828.

   Let pressure ratio r = P_down / P_up <= 1.0.
   - If r <= r_crit: Choked (sonic) regime at orifice throat:
         flow_function(r, gamma) = (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
   - If r > r_crit: Subsonic compressible regime:
         flow_function(r, gamma) = sqrt( 2 / (gamma - 1) * (r ** (2 / gamma) - r ** ((gamma + 1) / gamma)) )
   - If r >= 1.0: Zero flow.

2. Mass Flow Rate:
       mdot = Cd * A * P_up * sqrt(gamma / (R * T_up)) * flow_function(P_down / P_up, gamma)
   matching upstream Fortran `airbag2.F:511-520`.

3. Energy / Enthalpy Transfer:
   Mass leaving the upstream chamber carries specific enthalpy h_up = c_p * T_up:
       Hdot = mdot * c_p * T_up
   First Law of Thermodynamics for the composite 2-chamber system:
       dE_up = - mdot * c_p * T_up * dt
       dE_down = + mdot * c_p * T_up * dt
   Total internal energy across both communicating chambers is strictly conserved:
       d(E_up + E_down) / dt = 0.

4. Communication Triggering & Deflation Thresholds:
   - Time trigger: activates when t >= T_open (airbag2.F line 468).
   - Pressure differential trigger: activates when (P_up - P_down) >= DeltaP_def
     sustained for duration >= DeltatP_def (airbag2.F lines 455-458).
   - Direction modes:
     * "bidirectional": flow proceeds from higher to lower pressure chamber.
     * "1to2" / "forward": check-valve permitting flow only from chamber 1 to 2.
     * "2to1" / "reverse": check-valve permitting flow only from chamber 2 to 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

_EM20 = 1.0e-20
_EM10 = 1.0e-10


@dataclass
class MonvolCommu1:
    """Communicating chamber orifice specification (/MONVOL/COMMU1).

    Upstream Fortran reference:
    - Card Reader: `starter/source/airbag/hm_read_monvol_type9.F` (lines 308-323, 871-972)
    - Flow Engine: `engine/source/airbag/airbag2.F` (lines 435-565)
    """
    id: int = 1
    title: str = ""
    chamber1_id: int = 1
    chamber2_id: int = 2

    # Orifice geometry & discharge
    area: float = 0.0          # Orifice area A (Acom in hm_read_monvol_type9.F)
    cd: float = 1.0            # Discharge coefficient (0 < cd <= 1.0)

    # Opening thresholds
    t_open: float = 0.0        # Opening time threshold (Tcom in Fortran)
    dp_def: float = 0.0        # Pressure differential burst threshold (DeltaPCdef)
    dtp_def: float = 0.0       # Duration requirement for dp_def (DeltatPCdef)

    # Communication direction
    direction: str = "bidirectional"  # "bidirectional", "1to2" (forward/check_valve), "2to1" (reverse)

    # Optional function scalings
    surf_id: int = 0           # Communicating surface ID (surf_IDc)
    fct_id_t: int = 0          # Time-dependent area scale function (fct_IDCt)
    fscale_t: float = 1.0      # Scale factor for fct_id_t (FscaleCt)
    fct_id_p: int = 0          # Pressure-dependent area scale function (fct_IDCP)
    fscale_p: float = 1.0      # Scale factor for fct_id_p (FscaleCP)

    # Dynamic state tracking
    is_open: bool = False      # Open / deflated state (IDEF in airbag2.F)
    dp_timer: float = 0.0      # Accumulated time where dp >= dp_def
    total_mass_transferred: float = 0.0    # Net mass transferred from chamber 1 to chamber 2 (kg)
    total_energy_transferred: float = 0.0  # Net enthalpy transferred from chamber 1 to chamber 2 (J)
    last_mdot: float = 0.0                 # Last computed mass flow rate (kg/s)
    last_regime: str = "none"              # "choked", "subsonic", "closed", "zero"

    def __post_init__(self) -> None:
        if self.cd <= 0.0:
            self.cd = 1.0
        # If neither t_open nor dp_def is set (> 0), the orifice is open from t=0
        if self.t_open <= 0.0 and self.dp_def <= 0.0:
            self.is_open = True

    # Convenience aliases matching Fortran variable names
    @property
    def bag1_id(self) -> int:
        return self.chamber1_id

    @bag1_id.setter
    def bag1_id(self, val: int) -> None:
        self.chamber1_id = int(val)

    @property
    def bag2_id(self) -> int:
        return self.chamber2_id

    @bag2_id.setter
    def bag2_id(self, val: int) -> None:
        self.chamber2_id = int(val)

    @property
    def acom(self) -> float:
        return self.area

    @acom.setter
    def acom(self, val: float) -> None:
        self.area = float(val)

    @property
    def a_vent(self) -> float:
        return self.area

    @a_vent.setter
    def a_vent(self, val: float) -> None:
        self.area = float(val)

    @property
    def tcom(self) -> float:
        return self.t_open

    @tcom.setter
    def tcom(self, val: float) -> None:
        self.t_open = float(val)

    @property
    def deltap_cdef(self) -> float:
        return self.dp_def

    @deltap_cdef.setter
    def deltap_cdef(self, val: float) -> None:
        self.dp_def = float(val)

    @property
    def deltat_pcdef(self) -> float:
        return self.dtp_def

    @deltat_pcdef.setter
    def deltat_pcdef(self, val: float) -> None:
        self.dtp_def = float(val)


@dataclass
class CommuStepResult:
    """Result of a single communication time-step update."""
    commu_id: int
    is_open: bool
    direction: str          # "1to2", "2to1", or "none"
    regime: str             # "choked", "subsonic", "closed", or "zero"
    mass_flow_rate: float   # mdot (kg/s)
    mass_transferred: float # dm = mdot * dt (kg)
    enthalpy_rate: float    # Hdot (W)
    energy_transferred: float # dE = Hdot * dt (J)
    p_up: float             # Upstream pressure (Pa)
    p_down: float           # Downstream pressure (Pa)
    t_up: float             # Upstream temperature (K)
    p1: float               # Chamber 1 pressure after step (Pa)
    p2: float               # Chamber 2 pressure after step (Pa)


def critical_pressure_ratio(gamma: float = 1.4) -> float:
    """Critical sonic choking pressure ratio P_crit / P_up for an ideal gas.

    Upstream Fortran reference:
    `engine/source/airbag/airbag2.F` line 112:
        PCRIT = P * (TWO / (GAMA + ONE)) ** (GAMA / (GAMA - ONE))
    """
    if gamma <= 1.0:
        gamma = 1.4
    return (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))


def choked_flow_function(gamma: float = 1.4) -> float:
    """Value of the compressible flow function in the choked (sonic) regime.

    flow_function(r_crit, gamma) = (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
    """
    if gamma <= 1.0:
        gamma = 1.4
    return (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


def subsonic_flow_function(pressure_ratio: float, gamma: float = 1.4) -> float:
    """Value of the compressible flow function in the subsonic regime.

    flow_function(r, gamma) = sqrt( 2 / (gamma - 1) * (r ** (2/gamma) - r ** ((gamma+1)/gamma)) )
    """
    if gamma <= 1.0:
        gamma = 1.4
    r = max(0.0, min(1.0, float(pressure_ratio)))
    if r >= 1.0:
        return 0.0
    term1 = r ** (2.0 / gamma)
    term2 = r ** ((gamma + 1.0) / gamma)
    val = (2.0 / (gamma - 1.0)) * (term1 - term2)
    return math.sqrt(max(0.0, val))


def flow_function(pressure_ratio: float, gamma: float = 1.4) -> Tuple[float, str]:
    """Compute compressible orifice flow function and identify flow regime.

    Args:
        pressure_ratio: Downstream to upstream pressure ratio P_down / P_up in [0, 1].
        gamma: Ratio of specific heats (c_p / c_v).

    Returns:
        (psi, regime): Tuple of flow function value and regime name ("choked", "subsonic", "zero").
    """
    if pressure_ratio >= 1.0 - _EM10:
        return 0.0, "zero"

    r_crit = critical_pressure_ratio(gamma)
    if pressure_ratio <= r_crit:
        return choked_flow_function(gamma), "choked"
    else:
        return subsonic_flow_function(pressure_ratio, gamma), "subsonic"


def compute_orifice_flow(
    area: float,
    cd: float,
    p_up: float,
    t_up: float,
    p_down: float,
    gamma: float = 1.4,
    r_spec: float = 287.05,
) -> Tuple[float, str]:
    """Calculate compressible orifice mass flow rate (kg/s).

    Upstream Fortran:
    `engine/source/airbag/airbag2.F` lines 511-520:
        PVOIS = MAX(PVOIS, PCRIT)
        U = SQRT(TWO * GAMA / (GAMA - ONE) * P / RO * (ONE - (PVOIS / P) ** ((GAMA - ONE) / GAMA)))
        FLOUT = AOUT * U
        DMOUT = FLOUT * RO * (PVOIS / P) ** (ONE / GAMA)

    which is identically:
        mdot = Cd * A * P_up * sqrt(gamma / (R * T_up)) * flow_function(P_down / P_up, gamma)

    Returns:
        (mdot, regime): Mass flow rate in kg/s and regime name.
    """
    if area <= 0.0 or cd <= 0.0 or p_up <= _EM20 or t_up <= _EM20:
        return 0.0, "zero"

    if p_down >= p_up:
        return 0.0, "zero"

    r = max(0.0, p_down / p_up)
    psi, regime = flow_function(r, gamma)
    if psi <= 0.0:
        return 0.0, "zero"

    c_throat = math.sqrt(gamma / (r_spec * t_up))
    mdot = cd * area * p_up * c_throat * psi
    return mdot, regime


def _get_chamber_prop(ch: Any, name: str, default: Any = 0.0) -> Any:
    """Extract property from object or dict."""
    if isinstance(ch, dict):
        return ch.get(name, default)
    return getattr(ch, name, default)


def _set_chamber_prop(ch: Any, name: str, value: Any) -> None:
    """Set property on object or dict."""
    if isinstance(ch, dict):
        ch[name] = value
    else:
        setattr(ch, name, value)


def _get_chamber_thermo(ch: Any) -> Dict[str, float]:
    """Extract and reconcile thermodynamic properties of a chamber."""
    mass = float(_get_chamber_prop(ch, "mass", 0.0))
    vol = max(float(_get_chamber_prop(ch, "volume", 1e-9)), 1e-9)
    gamma = float(_get_chamber_prop(ch, "gamma", 1.4))
    if gamma <= 1.0:
        gamma = 1.4

    r_spec = float(_get_chamber_prop(ch, "r_spec", 287.05))
    if r_spec <= 0.0:
        r_spec = 287.05

    cv = float(_get_chamber_prop(ch, "cv", r_spec / (gamma - 1.0)))
    if cv <= 0.0:
        cv = r_spec / (gamma - 1.0)
    cp = gamma * cv

    temp = float(_get_chamber_prop(ch, "temperature", _get_chamber_prop(ch, "t_initial", 293.15)))
    if temp <= 0.0:
        temp = 293.15

    # Reconcile mass if not initialized
    if mass <= 0.0:
        pres_init = float(_get_chamber_prop(ch, "pressure", _get_chamber_prop(ch, "pext", 101325.0)))
        if pres_init > 0.0 and r_spec > 0.0 and temp > 0.0:
            mass = pres_init * vol / (r_spec * temp)
        else:
            mass = 1.0 * vol
        _set_chamber_prop(ch, "mass", mass)

    energy = _get_chamber_prop(ch, "energy", None)
    if energy is None or energy <= 0.0:
        energy = mass * cv * temp
        _set_chamber_prop(ch, "energy", energy)
    else:
        energy = float(energy)

    pres = float(_get_chamber_prop(ch, "pressure", (gamma - 1.0) * energy / vol))
    if pres <= 0.0:
        pres = (gamma - 1.0) * energy / vol
        _set_chamber_prop(ch, "pressure", pres)

    return {
        "mass": mass,
        "volume": vol,
        "gamma": gamma,
        "r_spec": r_spec,
        "cv": cv,
        "cp": cp,
        "temperature": temp,
        "pressure": pres,
        "energy": energy,
    }


def step_airbag_commu(
    commu: MonvolCommu1,
    chamber1: Any,
    chamber2: Any,
    dt: float,
    current_time: float,
    functions: Optional[Dict[int, Any]] = None,
) -> CommuStepResult:
    """Advance communicating airbag flow kinetics by time-step dt.

    Upstream Fortran reference:
    - `engine/source/airbag/airbaga1.F` (lines 438-472)
    - `engine/source/airbag/airbag2.F` (lines 438-565)

    Args:
        commu: MonvolCommu1 instance holding orifice parameters and state.
        chamber1: Upstream or downstream monitored volume / chamber.
        chamber2: The second communicating monitored volume / chamber.
        dt: Simulation time step (s).
        current_time: Current simulation time (s).
        functions: Optional mapping of function ID to Function object.

    Returns:
        CommuStepResult containing flow kinematics and updated chamber pressures.
    """
    dt = max(float(dt), 1.0e-12)
    th1 = _get_chamber_thermo(chamber1)
    th2 = _get_chamber_thermo(chamber2)

    p1 = th1["pressure"]
    p2 = th2["pressure"]

    # 1. Evaluate opening criteria
    if not commu.is_open:
        # Time threshold (airbag2.F line 468)
        if commu.t_open > 0.0 and current_time >= commu.t_open:
            commu.is_open = True

        # Pressure differential burst threshold (airbag2.F lines 455-458)
        if (not commu.is_open) and commu.dp_def > 0.0:
            dp_current = abs(p1 - p2)
            if dp_current >= commu.dp_def:
                commu.dp_timer += dt
                if commu.dp_timer >= commu.dtp_def:
                    commu.is_open = True
            else:
                commu.dp_timer = 0.0

    if not commu.is_open:
        commu.last_mdot = 0.0
        commu.last_regime = "closed"
        return CommuStepResult(
            commu_id=commu.id,
            is_open=False,
            direction="none",
            regime="closed",
            mass_flow_rate=0.0,
            mass_transferred=0.0,
            enthalpy_rate=0.0,
            energy_transferred=0.0,
            p_up=max(p1, p2),
            p_down=min(p1, p2),
            t_up=th1["temperature"] if p1 >= p2 else th2["temperature"],
            p1=p1,
            p2=p2,
        )

    # 2. Determine upstream and downstream chambers based on direction
    direction_mode = commu.direction.lower().strip()
    flow_dir: str = "none"

    if abs(p1 - p2) < _EM10:
        commu.last_mdot = 0.0
        commu.last_regime = "zero"
        return CommuStepResult(
            commu_id=commu.id,
            is_open=True,
            direction="none",
            regime="zero",
            mass_flow_rate=0.0,
            mass_transferred=0.0,
            enthalpy_rate=0.0,
            energy_transferred=0.0,
            p_up=p1,
            p_down=p2,
            t_up=th1["temperature"],
            p1=p1,
            p2=p2,
        )

    if p1 > p2:
        if direction_mode in ("bidirectional", "1to2", "forward", "one_way"):
            flow_dir = "1to2"
            ch_up = chamber1
            ch_down = chamber2
            th_up = th1
            th_down = th2
        else:
            # Check-valve blocks backflow
            flow_dir = "none"
    else:  # p2 > p1
        if direction_mode in ("bidirectional", "2to1", "reverse"):
            flow_dir = "2to1"
            ch_up = chamber2
            ch_down = chamber1
            th_up = th2
            th_down = th1
        else:
            flow_dir = "none"

    if flow_dir == "none":
        commu.last_mdot = 0.0
        commu.last_regime = "zero"
        return CommuStepResult(
            commu_id=commu.id,
            is_open=True,
            direction="none",
            regime="zero",
            mass_flow_rate=0.0,
            mass_transferred=0.0,
            enthalpy_rate=0.0,
            energy_transferred=0.0,
            p_up=max(p1, p2),
            p_down=min(p1, p2),
            t_up=th_up["temperature"] if "th_up" in locals() else th1["temperature"],
            p1=p1,
            p2=p2,
        )

    # 3. Calculate effective orifice area with scaling functions
    area_eff = commu.area * commu.cd
    if commu.fct_id_t > 0 and functions is not None and commu.fct_id_t in functions:
        scale_t = commu.fscale_t * float(functions[commu.fct_id_t].eval(current_time))
        area_eff *= max(0.0, scale_t)
    if commu.fct_id_p > 0 and functions is not None and commu.fct_id_p in functions:
        scale_p = commu.fscale_p * float(functions[commu.fct_id_p].eval(abs(p1 - p2)))
        area_eff *= max(0.0, scale_p)

    # 4. Compute mass flow rate
    p_up = th_up["pressure"]
    p_down = th_up["pressure"] if th_down["pressure"] < 0 else th_down["pressure"]
    t_up = th_up["temperature"]
    gamma = th_up["gamma"]
    r_spec = th_up["r_spec"]
    c_p = th_up["cp"]

    mdot, regime = compute_orifice_flow(
        area=area_eff,
        cd=1.0,  # cd already folded into area_eff
        p_up=p_up,
        t_up=t_up,
        p_down=p_down,
        gamma=gamma,
        r_spec=r_spec,
    )

    dm_requested = mdot * dt

    # 5. Stability & Courant mass limiters (airbag2.F lines 514-517)
    # Limiter A: Do not evacuate more than 50% of upstream mass in one step
    dm_max_mass = 0.5 * th_up["mass"]

    # Limiter B: Equilibrium pressure equalization limit (prevents overshoot/inversion)
    # dp_eq = P_up - P_down = (gamma - 1) * cp * T_up * (1/V_up + 1/V_down) * dm_eq
    v_up = th_up["volume"]
    v_down = th_down["volume"]
    denom_eq = (gamma - 1.0) * c_p * t_up * (1.0 / v_up + 1.0 / v_down)
    if denom_eq > _EM20:
        dm_max_pres = 0.999 * max(0.0, p_up - p_down) / denom_eq
    else:
        dm_max_pres = dm_max_mass

    dm = min(dm_requested, dm_max_mass, dm_max_pres)
    dm = max(0.0, dm)

    # 6. Enthalpy / Energy transfer
    de = dm * c_p * t_up
    hdot = mdot * c_p * t_up

    # 7. Update chamber thermodynamic states
    # Upstream chamber loses mass and enthalpy
    m_up_new = max(th_up["mass"] - dm, _EM20)
    e_up_new = max(th_up["energy"] - de, 0.0)
    t_up_new = e_up_new / (m_up_new * th_up["cv"])
    p_up_new = (gamma - 1.0) * e_up_new / v_up

    # Downstream chamber gains mass and enthalpy
    m_down_new = th_down["mass"] + dm
    e_down_new = th_down["energy"] + de
    t_down_new = e_down_new / (m_down_new * th_down["cv"])
    p_down_new = (gamma - 1.0) * e_down_new / v_down

    _set_chamber_prop(ch_up, "mass", m_up_new)
    _set_chamber_prop(ch_up, "energy", e_up_new)
    _set_chamber_prop(ch_up, "temperature", t_up_new)
    _set_chamber_prop(ch_up, "pressure", p_up_new)

    _set_chamber_prop(ch_down, "mass", m_down_new)
    _set_chamber_prop(ch_down, "energy", e_down_new)
    _set_chamber_prop(ch_down, "temperature", t_down_new)
    _set_chamber_prop(ch_down, "pressure", p_down_new)

    # Update cumulative statistics
    if flow_dir == "1to2":
        commu.total_mass_transferred += dm
        commu.total_energy_transferred += de
        p1_new = p_up_new
        p2_new = p_down_new
    else:
        commu.total_mass_transferred -= dm
        commu.total_energy_transferred -= de
        p1_new = p_down_new
        p2_new = p_up_new

    commu.last_mdot = mdot
    commu.last_regime = regime

    return CommuStepResult(
        commu_id=commu.id,
        is_open=True,
        direction=flow_dir,
        regime=regime,
        mass_flow_rate=mdot,
        mass_transferred=dm,
        enthalpy_rate=hdot,
        energy_transferred=de,
        p_up=p_up,
        p_down=p_down,
        t_up=t_up,
        p1=p1_new,
        p2=p2_new,
    )


def apply_airbag_communications(
    model: Any,
    dt: float,
    current_time: float,
) -> List[CommuStepResult]:
    """Process all communicating airbag chambers defined on a Model instance.

    Looks for communications in `model.monvol_commus`, `model.monvol_communications`,
    or `model.monvol_comms`.
    Resolves chamber 1 and chamber 2 from `model.monitored_volumes`, `model.monvol_gases`,
    or `model.monvol_airbags`.
    """
    commus: Sequence[Any] = []
    if hasattr(model, "monvol_commus") and model.monvol_commus:
        commus = list(model.monvol_commus.values()) if isinstance(model.monvol_commus, dict) else list(model.monvol_commus)
    elif hasattr(model, "monvol_communications") and model.monvol_communications:
        commus = list(model.monvol_communications.values()) if isinstance(model.monvol_communications, dict) else list(model.monvol_communications)
    elif hasattr(model, "monvol_comms") and model.monvol_comms:
        commus = list(model.monvol_comms.values()) if isinstance(model.monvol_comms, dict) else list(model.monvol_comms)

    if not commus:
        return []

    # Map available chambers
    chambers: Dict[int, Any] = {}
    for attr in ("monitored_volumes", "monvol_gases", "monvol_airbags", "airbags"):
        d = getattr(model, attr, None)
        if isinstance(d, dict):
            chambers.update(d)
        elif isinstance(d, (list, tuple)):
            for item in d:
                if hasattr(item, "id"):
                    chambers[item.id] = item

    functions = getattr(model, "functions", None)
    results: List[CommuStepResult] = []

    for c in commus:
        # Wrap or cast if needed
        if not isinstance(c, MonvolCommu1):
            c_obj = MonvolCommu1(
                id=getattr(c, "id", 1),
                title=getattr(c, "title", ""),
                chamber1_id=getattr(c, "chamber1_id", getattr(c, "monvol1_id", getattr(c, "bag1_id", 1))),
                chamber2_id=getattr(c, "chamber2_id", getattr(c, "monvol2_id", getattr(c, "bag2_id", 2))),
                area=getattr(c, "area", getattr(c, "a_vent", getattr(c, "acom", 0.0))),
                cd=getattr(c, "cd", 1.0 if getattr(c, "cd", 0.0) == 0.0 else getattr(c, "cd", 1.0)),
                t_open=getattr(c, "t_open", getattr(c, "tcom", 0.0)),
                dp_def=getattr(c, "dp_def", getattr(c, "deltap_cdef", 0.0)),
                dtp_def=getattr(c, "dtp_def", getattr(c, "deltat_pcdef", 0.0)),
                direction=getattr(c, "direction", "bidirectional"),
                surf_id=getattr(c, "surf_id", getattr(c, "surface_id", 0)),
                fct_id_t=getattr(c, "fct_id_t", getattr(c, "fct_id", 0)),
                fscale_t=getattr(c, "fscale_t", 1.0),
                fct_id_p=getattr(c, "fct_id_p", 0),
                fscale_p=getattr(c, "fscale_p", 1.0),
            )
        else:
            c_obj = c

        ch1 = chambers.get(c_obj.chamber1_id)
        ch2 = chambers.get(c_obj.chamber2_id)
        if ch1 is None or ch2 is None:
            continue

        res = step_airbag_commu(c_obj, ch1, ch2, dt, current_time, functions=functions)
        results.append(res)

    return results
