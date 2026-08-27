"""
The Model: everything the Starter builds and the Engine consumes.

Fortran origin: there is no single "model object" in OpenRadioss — the model
is the union of the big arrays (X, V, MS, IXS, IXC, PM, GEO, IPART ...) and
dozens of modules. This class is their sum, with the same information laid
out in named fields. The restart file written by the Starter is (a pickle
of) this object, playing the role of the binary ``*_0000.rst`` restart.

Element storage
---------------
Like the Fortran, elements are stored **by type** in homogeneous groups
(`bricks`, `tetras`, `shells`, `sh3n`, `trusses`, `springs`, `beams`),
each carrying:

* ``ids``   — user element IDs, shape (n,)
* ``conn``  — 0-based node indices, shape (n, nodes_per_elem)
* ``part``  — index into ``model.parts_list`` for each element
* ``state`` — dict of per-element arrays created at initialization time
  (stress tensors, plastic strain, hourglass state, volumes, masses...).
  This mirrors the Fortran *element buffer* (``ELBUF_TAB`` of
  ``elbufdef_mod.F``) which holds GBUF%SIG, GBUF%PLA, GBUF%VOL etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .entities import (
    AddedMass, BoundaryCondition, Box, ConcentratedLoad, Damping,
    CentrifugalLoad, EntityGroup, Gravity, ImposedAcceleration, ImposedDisplacement, ImposedTemperature, ImposedVelocity,
    AleBoundaryCondition,
    InitialVelocity, Interface, Line, Material, Mpc, NodeGroup, Part,
    PressureLoad, Property, Random, Rbe3, RigidBody, RigidWall, Section, Sensor,
    DetonatorPlane, DetonatorPoint,
    ConvectionLoad, InivolContainer, InitialVolume,
    RadiationLoad, ImposedFlux, InitialTemperature,
    InitialBrickState, InitialShellState,
    InitialTrussState, InitialBeamState, InitialSpringState,
    CyclicBoundaryCondition, SolidPartPerturbation, PBlastLoad,
    SubInterface, GuidedCable, ShellPartPerturbation, FailurePerturbation, SphGlobal, SmsGlobal,
    BcsNrf, BcsWall, RigidLink, CylJoint, GeneralJoint,
    MergeNode, MergeRbody, IniCrack, IniCrackSegment, LaserLoad,
    PcylLoad, PfluidLoad, Preload, PreloadAxial, DampInter, DampRange,
    AnalyGlobal, UpwindGlobal, CaaControl,
    Gauge, Cluster, ExtLink, FxBody, IniGrav, IniMap1D, IniMap2D, IniStateFile,
    MonvolPres, MonvolGas, MonvolCommu1, MonvolLFluid, LeakMat,
    Activ,
    MonvolAirbag, MonvolCommu, MonvolPart,
    AleGrid, AleLink, AleSolver, AleClose,
    Retractor, Slipring, Pretensioner, UserWindow,
    DetonationWave, ElementActivation, MonvolFvmBag2, Autoposition,
    LoadCentri, LoadPfluid, LoadPressure, LoadGravity, LoadBody, LoadTherm,
    EulerBcs, HeatBcs, InivelAxis, InivelFvm, InivelNode, InivelPart, InivelSph,
    ImpdispFgeo, ImpvelFgeo, RwallTherm, RwallBox, RwallCone, SphInOut,
    SphBcs, MadymoLink, MadymoExfem,
    AleGridDonea, AleGridSpring, AleGridStandard, AleGridDisp, AleGridLaplacian, AleGridVolume,
    AdmeshGlobal, StampingInit, RandomNoise, Accelerometer, Subset,
    FailComposite, EbcsPropellant, AdmasNonUniform, AdmasNonUniformItem,
    SectCircle, SectParal, SectBox, SectCut, DynainShell, MonvolArea, StateDt,
    SphReserve, MoveFunct,
    EigenMode, StressFile, MemoryRequest,
    FailFractal, TransformPosition, TransformProjection, TransformFrame,
    DampGlobal, DampPart, ExternalLink, ArchSpec,
    FunctPython, FrictionModel, FrictionPartPair, RefstaNode, ErefSpec,
    NbcsBlock, NbcsNode, AleMuscl, BemModel,
    GaugePoint, SphGlo, AnalyOptions, AleCfdSph,
    FailOrthBiquad, SlipringShell,
    EbcsNrf, EbcsPeriodic, EbcsCyclic, FailRtcl, FailGurson,
    FailPuck, FailSahraei, FailSyazwan, FailTab2, FailGene1,
    FailNxt, FailLadDama, FailInievo,
    FailHcDsse, FailMullins, FailSnconnect, FailSpalling,
    DfsDetcord, LoadPressure, AleMat, EulerMat,
    Stack, StackPly, SubLaminate, SubLaminatePly, InertiaPart, AleGridConstraint,
    LagmulGlobal, GearConstraint, RackConstraint, DiffConstraint,
    WaveShaper, DetLine, DetCirc, IniMap3D, SetGeneric,
    MaterialPlasZeril, MaterialPlasBodne, MaterialViscProny, MaterialThermStress, DampStiff,
    MaterialConc, MaterialBarlat, MaterialLaw83, MaterialLaw80,
    MaterialLaw117, MaterialLaw90, MaterialLaw33, MatHeatModifier, MatNonlocalModifier,
    MaterialLaw66, MaterialLaw35, MaterialLaw62, MaterialLaw28, MaterialLaw44,
    MaterialLaw88, MaterialLaw92, MaterialLaw94, MaterialLaw46, MaterialLaw69,
    MaterialLaw124, MaterialLaw126, MaterialLaw125, MaterialLaw127, MaterialLaw130,
    MaterialLaw128, MaterialLaw129, MaterialLaw123, MaterialLaw132, MaterialLaw134,
    MaterialLaw104, MaterialLaw105, MaterialLaw106, MaterialLaw107, MaterialLaw110, MaterialLaw115,
    MaterialLaw109, MaterialLaw111, MaterialLaw112, MaterialLaw116, MaterialLaw122, MaterialLaw158,
    AirbagInjector, AirbagVenthole,
    ErefElement, IniCrack, PropRivet, PropXelem, AdmeshControl,
    PreloadBolt, LoadHydro,
    Func2DTable, NonlocalModel, FricOrient, IniSphCel,
    MaterialViscPlas,
    EbcsPres, EbcsVel, EbcsInlet, EbcsFluxout, EbcsGradp0, EbcsNormv, EbcsValv, EbcsMonvol,
    BcsWall, SeatbeltSystem, AmsControl,
    BcsCyclic, PcylLoad, DampVrel, MatLaw113, MatLaw79, MatViscLprony,
    MatLaw190, MatLaw41, FailChang, PropType20, PropType21, PropType22, PropType22Layer,
    FailFabric, FailHoffman, FailMaxStrain, FailTsaiHill, FailTsaiWu, PropType6, LoadCload, LoadPload,
    MatLaw114, MatLaw117, MatLaw119, MatLaw120, MatLaw121, PropType26, PropType27,
    MatLaw50, MatLaw57, MatLaw87, MatLaw95, MatLaw163, MatLaw169,
    MatLaw49, MatLaw76, PropType11, PropSandwLayer, PropType16, PropFabricLayer, PropType17, PropType44,
    MatLaw60, MatLaw63, MatLaw48, MatLaw26, PropType12, PropType15, PropStrandLayer, PropType28,
    MatLaw6, MatLaw11, MatLaw77, MatLaw77Curve, MatLaw151, MatMultiFluidFraction, MatLaw187, MatLaw187Rate, PropType33, PropType46, PropType35,
    MatLaw3, MatLaw4, MatLaw5, MatLaw10, MatLaw14, MatLaw21, MatLaw32, MatLaw37, PropType45, PropType36,
    MatLaw12, MatLaw13, MatLaw15, MatLaw18, MatLaw22, MatLaw25, MatLaw28, PropType9, PropType10, PropType51, PropType5, PropType6, PropType20,
    MatLaw52, MatLaw16, MatLaw14, MatLaw59, MatLaw64, FailLadDama, FailPuck, FailWierzbicki, FailWilkins, FailSpalling,
    PropType14, PropType8, PropType25, PropType32, PropType43,
    MatLaw68, MatLaw72, MatLaw65, MatLaw58, MatLaw20, MatLaw38, MatLaw29, MatLaw34, MatLaw23, MatLaw78,
    FailHashin, FailTensstrain, FailEnergy, FailUser,
    PropType34, PropType29, PropType30, PropType31,
    MatLaw100, MatLaw97, MatLaw71, MatLaw73, MatLaw84, MatLaw93, MatLaw133, MatLaw101, MatLaw43,
    FailLemaitre, FailComposite, FailTab2, FailAlter, FailVisual, FailOrthstrain,
    EbcsPropellant, EbcsCyclic, Preload,
    FailEMC, FailNXT, FailTButcher, FailMullins, FailCockcroft, FailGene1, FailXFEM,
    MatLaw53, MatLaw54, MatLaw74, MatLaw82, PropIntBeamIP, PropType18,
    DefInterType11, DefInterType19, DefInterType25, StateDirective,
    SphFlow, MidDirective, PidDirective, SphParticle, FailTab1,
    MatLaw40, MatLaw80, MatLaw102, MatNLocal, PropType12, PropType13,
    DefInterType2, EngineTHRecord,
    MatLaw103, MatLaw108, MatPlasPredef, MatDPrag2, PropType23,
    DampFreqRange, DampFunct, FunctSmooth, IniStateTable,
    PblastLoad, Inivol, InigravLoad, Inista, BemControl, PerturbControl,
    EbcsInip, EbcsIniv, PropInject1, PropInject2, PropJoint, PropTorsion,
    PropSpringElasPlas, PropSpringBeam, PropSpotweld, PropBushing,
    MaterialSprSeatbelt, MaterialShSeatbelt, MaterialTapo, MaterialPlasRate, MaterialCdpm2,
    Subdomain, Submodel, Surface, Table, THRequest, Xref,
)
from ..common.tables import FunctTable
from .skew import SkewSet


@dataclass
class ElementGroup:
    """Homogeneous element block (see module docstring)."""

    ids: np.ndarray                    # (n,)   user ids
    conn: np.ndarray                   # (n, k) 0-based node indices
    part: np.ndarray                   # (n,)   index into model.parts_list
    state: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.ids)


@dataclass
class EngineControls:
    """Contents of the engine file ``*_0001.rad``.

    Fortran origin: engine/source/input read into the ``/RUN``, ``/DT``,
    ``/TFILE``, ``/ANIM`` options (module ``output_mod``, ``dt_mod`` ...).
    """

    run_name: str = "RUN"
    t_end: float = 0.0            # /RUN final time
    dt_scale: float = 0.9         # /DT  scale factor  (dt = k * dt_critical)
    dt_min: float = 0.0           # /DT  minimum dt: below this -> stop
    dt_noda: str = ""             # '' | 'NODA' | 'CST' (/DT/NODA[/CST], M6)
    dt_ams: bool = False          # /DT/AMS present (M61)
    dt_ams_igrp: int = 0          # AMS target part group (0 = all)
    dt_ams_tol: float = 1e-4      # AMS PCG tolerance
    dt_ams_itmax: int = 200       # AMS PCG max iterations
    dt_controls: Dict[str, Dict[str, Any]] = field(default_factory=dict) # /DT/<elem_type>[/<action>] (M121)
    th_dt: float = 0.0            # /TFILE time-history output period
    anim_dt: float = 0.0          # /ANIM/DT animation state period
    state_dt: float = 0.0         # /STATE/DT restart-snapshot period (M6)
    state_tstart: float = 0.0     # /STATE/DT first snapshot time
    print_cycles: int = 100       # /PRINT listing frequency (cycles)
    energy_error_stop: float = 15.0  # %, /STOP-like divergence guard
    anim_vect: List[str] = field(default_factory=lambda: ["VEL", "DIS"])
    anim_elem: List[str] = field(default_factory=lambda: ["VONM", "EPSP"])
    anim_tens: List[str] = field(default_factory=list) # /ANIM/ELEM/TENS, /ANIM/BRICK/TENS, /ANIM/SHELL/TENS (M122)

    # M120: Extended Engine control cards
    debug_flags: Dict[str, int] = field(default_factory=dict)  # /DEBUG options
    debug_acc_start: float = 0.0                               # /DEBUG/ACC start time
    debug_acc_freq: int = 1                                    # /DEBUG/ACC frequency
    bcs_active: Dict[int, bool] = field(default_factory=dict)  # /BCS/ON, /BCS/OFF
    rbody_active: Dict[int, bool] = field(default_factory=dict)# /RBODY/ON, /RBODY/OFF
    ale_active: Dict[int, bool] = field(default_factory=dict)  # /ALE/ON, /ALE/OFF
    noise_dt: float = 0.0                                      # /NOIS/DT period
    noise_tstart: float = 0.0                                  # /NOIS/DT start time
    noise_flags: Dict[str, bool] = field(default_factory=dict) # /NOIS flags (VEL, ACC, etc.)
    h3d_dt: float = 0.0                                        # /H3D/DT output period
    h3d_requests: List[str] = field(default_factory=list)      # /H3D channel requests
    flow_dt: float = 0.0                                       # /FLOW/DT output period
    upwind_active: bool = False                                # /UPWIND present
    upwind_mom: float = 1.0                                    # /UPWIND momentum coeff
    upwind_mass_eng: float = 1.0                               # /UPWIND mass & energy coeff
    upwind_wet_surf: float = 1.0                               # /UPWIND wet surface coeff
    eig_off: List[int] = field(default_factory=list)           # /EIG/OFF deactivated modes
    stop_nstep: int = 0                                        # /STOP/NSTEP (M123)
    stop_tstop: float = 0.0                                    # /STOP/TSTOP (M123)
    stop_timet: float = 0.0                                    # /STOP/TIMET (M123)

    # M146: Engine Dynamic Controls & Solver Directives
    inter_windows: Dict[int, tuple[int, float, float]] = field(default_factory=dict) # /INTER window (M146)
    inter_active: Dict[int, bool] = field(default_factory=dict)                      # /INTER/ON, /INTER/OFF (M146)
    del_elements: Dict[str, List[int]] = field(default_factory=dict)                 # /DEL/<type> (M146)
    dli7_controls: Dict[str, Any] = field(default_factory=dict)                     # /DLI7 (M146)
    kerel_active: bool = False                                                       # /KEREL (M146)
    kerel_tstart: float = 0.0
    kerel_tstop: float = 0.0
    kerel_istatg: int = 0
    dyrel_active: bool = False                                                       # /DYREL (M146)
    dyrel_beta: float = 1.0
    dyrel_period: float = 0.0
    dyrel_istatg: int = 0
    thermal_acc_fact: float = 1.0                                                    # /THERMAL (M146)
    thermal_dt: float = 0.0                                                          # /THERMAL/DT, /HEAT/DT (M146)
    thermal_tstart: float = 0.0
    abf_dt: float = 0.0                                                              # /ABF, /ABF/DT (M146)
    abf_dt_write: float = 0.0
    inivel_engine: Dict[str, Dict[int, float]] = field(default_factory=dict)         # /INIVEL (M146)

    # M148: Engine Directives Suite II
    damp_dt: float = 0.0                                                             # /DAMP/DT (M148)
    damp_alpha: float = 0.0                                                          # /DAMP mass damping factor (M148)
    damp_beta: float = 0.0                                                           # /DAMP stiffness damping factor (M148)
    damp_tstart: float = 0.0                                                         # /DAMP start time (M148)
    damp_tstop: float = 0.0                                                          # /DAMP stop time (M148)
    damp_grpart: int = 0                                                             # /DAMP target part group (M148)
    mass_reset: bool = False                                                         # /MASS/RESET (M148)
    sensor_reset: List[int] = field(default_factory=list)                           # /SENSOR/RESET, /SENS/RESET (M148)
    viper_active: bool = False                                                       # /VIPER, /VIPER/ON (M148)
    madymo_mode: str = ""                                                            # /MADYMO/ON, /MADYMO/ON2, /MADYMO/MPP (M148)
    rad2r_active: bool = False                                                       # /RAD2RAD/ON, /RAD2R/ON (M148)
    fvbag_remesh: bool = False                                                       # /FVMBAG/REMESH (M148)
    fvbag_modif: bool = False                                                        # /FVMBAG/MODIF (M148)
    perf_sort: int = 1                                                               # /PERF/SORT1 (1), /PERF/SORT2 (2), /PERF/SORT3 (0) (M148)
    dt1tet10: int = 0                                                                # /DT1TET10 iterative subcycling factor (M148)
    dttsh: bool = False                                                              # /DTTSH thick shell dt flag (M148)
    report_freq: int = 0                                                             # /REPORT listing cycles (M148)
    report_dt: float = 0.0                                                           # /REPORT/DT output period (M148)
    negvol_action: str = ""                                                          # /NEGVOL/STOP, /NEGVOL/DEL (M148)
    th_records: List[EngineTHRecord] = field(default_factory=list)                    # /TH engine time-history records (M194)
    python_functions: Dict[int, Any] = field(default_factory=dict)                   # /FUNCT_PYTHON, /PYTHON_FUNCT (M194)
    checksum_mode: str = ""                                                          # /CHECKSUM/START, /CHECKSUM/END (M195)
    dynain_dt: float = 0.0                                                           # /DYNAIN/DT, /ENG/DYNAIN/DT (M195)
    dynain_tstart: float = 0.0                                                       # /DYNAIN/DT start time (M195)
    dtix_tini: float = 0.0                                                           # /DTIX, /ENG/DTIX initial dt (M200)
    dtix_tmax: float = 0.0                                                           # /DTIX, /ENG/DTIX max dt (M200)
    parith: str = "ON"                                                               # /PARITH/ON, /PARITH/OFF (M200)
    heat_active: bool = False                                                        # /HEAT (M202)
    heat_flag: bool = False

    @property
    def tfile_dt(self) -> float:
        return self.th_dt

    @tfile_dt.setter
    def tfile_dt(self, val: float) -> None:
        self.th_dt = val

    @property
    def dtix(self):
        if self.dtix_tini != 0.0 or self.dtix_tmax != 0.0:
            from .entities import DtixControl
            return DtixControl(id=1, t_ini=self.dtix_tini, t_max=self.dtix_tmax)
        return None




    # ------------------------------------------------------------------
    # /IMPL implicit-static control (M8). ``implicit`` switches the run
    # from the explicit leap-frog loop to the Newton–Raphson static
    # driver (pyradioss/implicit/statics.py); the fields below mirror the
    # OpenRadioss /IMPL cards minimally (final load factor = t_end,
    # increment size, Newton tolerance + iteration cap, linear-solver
    # backend). Unused by explicit runs.
    # ------------------------------------------------------------------
    implicit: bool = False        # /IMPL present -> implicit static run
    impl_dt: float = 0.0          # load-factor increment (0 = single step)
    impl_tol: float = 1.0e-6      # Newton residual tolerance (relative)
    impl_max_iter: int = 25       # Newton iteration cap per increment
    impl_linsolve: str = ""       # '', 'superlu', 'cholmod', 'mumps'
    # -- M9 nonlinear geometry (/IMPL/NONLIN) and arc-length (/IMPL/ARCL) --
    impl_nlgeom: bool = False     # updated-Lagrangian frame + K_geo tangent
    impl_arc: bool = False        # Crisfield arc-length continuation
    impl_arc_dl: float = 0.0      # initial arc radius (0 = from 1st predictor)
    impl_arc_maxinc: int = 200    # arc increment cap (runaway-path guard)
    impl_arc_itdes: int = 5       # target Newton iterations per arc increment
    # -- M10 implicit DYNAMICS (/IMPL/DYNA) — Newmark/HHT time integration.
    # ``impl_dyna`` mirrors the original's IDYNA flag (input/freimpl.F):
    # 0 = off (implicit runs STATIC), 1 = HHT-alpha (/IMPL/DYNA/1 — the
    # alpha value is read; gamma/beta derived as 1/2-a and (1-a)^2/4,
    # exactly imp_dyna.F's DYNA_INI), 2 = plain Newmark (/IMPL/DYNA/2 —
    # gamma and beta read directly, in that order: DY_G = NM_A,
    # DY_B = NM_B). With /IMPL/DYNA the /RUN "time" is PHYSICAL time again
    # and /IMPL/DTINI the physical time step.
    impl_dyna: int = 0            # 0 = static, 1 = HHT-alpha, 2 = Newmark
    impl_dyna_alpha: float = 0.0  # HHT alpha (0 = trapezoidal; -1/3 <= a <= 0)
    impl_dyna_gamma: float = 0.5  # Newmark gamma (/IMPL/DYNA/2 field 1)
    impl_dyna_beta: float = 0.25  # Newmark beta  (/IMPL/DYNA/2 field 2)
    # -- M11 Rayleigh damping in the implicit system (/IMPL/DYNA/DAMP,
    # freimpl.F IDY_DAMP: card reads DAMPA_IMP then DAMPB_IMP). The damping
    # matrix is C = a*M + b*K with M the lumped mass and K the tangent at
    # the step start (imp_dyna.F IMP_DYKV). The card IMPLIES dynamics
    # (freimpl.F: IF (IDYNA==0) IDYNA=1).
    impl_dyna_damp: bool = False  # /IMPL/DYNA/DAMP present
    impl_dyna_dampa: float = 0.0  # DAMPA_IMP — mass-proportional a
    impl_dyna_dampb: float = 0.0  # DAMPB_IMP — stiffness-proportional b
    # -- M11 automatic implicit step control (/IMPL/DT/1 + /IMPL/DT/STOP,
    # imp_dt.F IMP_DTN with IDTC = 1): cut on non-convergence, grow back
    # toward /IMPL/DTINI on easy steps. Defaults are PORT choices where the
    # original reads them from the card (documented in statics.py).
    impl_dt_itw: int = 6          # NL_DTP - target iterations (grow below)
    impl_dt_scaleup: float = 1.1  # SCAL_DTP - growth factor per easy step
    impl_dt_scaledn: float = 0.5  # SCAL_DTN - cut factor on non-convergence
    impl_dt_min: float = 0.0      # /IMPL/DT/STOP dt_min (0 = dtini * 1e-4)
    impl_dt_max: float = 0.0      # /IMPL/DT/STOP dt_max (0 = dtini)
    impl_dt_fixp: list[float] = field(default_factory=list) # /IMPL/DT/FIXP sequence
    # -- M11 /IMPL/BUCKL (imp_buck.F): linearized buckling extraction after
    # the static prestress increments. 0 = off; 1|2 mirrors /IMPL/BUCKL/n.
    impl_buckl: int = 0           # /IMPL/BUCKL/n present
    impl_buckl_nmode: int = 4     # NBUCK — number of critical loads
    # -- M16 /IMPL/EIGV: modal (free-vibration) eigenvalue extraction. There
    # is no /IMPL/EIGV card in the open-source freimpl.F (only BUCKL and
    # DYNA); the port adds this minimal card to drive the consistent-mass
    # eigensolver (implicit/modal.py) the same way M11 added the thin
    # /IMPL/BUCKL card over the M9 library eigensolver. 0 = off.
    impl_eigv: bool = False       # /IMPL/EIGV present -> modal analysis
    impl_eigv_nmode: int = 6      # number of natural frequencies to extract
    impl_eigv_prestress: bool = False  # /IMPL/EIGV/STRS: K = K_mat + K_geo
    # -- M17 MODAL-SUPERPOSITION dynamics (PORT cards — freimpl.F has no
    # frequency-domain or mode-superposition path at all; the open-source
    # solver is time-domain only). These build on the M16 eigenpairs
    # (implicit/modal.py) and are library-first exactly like /IMPL/EIGV.
    # /IMPL/MODAL/DYNA: mode-superposition TRANSIENT response history.
    impl_modal_dyna: bool = False   # /IMPL/MODAL/DYNA present
    impl_modal_tend: float = 0.0    # transient end time (0 -> use t_end)
    impl_modal_dt: float = 0.0      # sampling step (0 -> use impl_dt)
    impl_modal_nmode: int = 6       # modes retained in the superposition
    impl_modal_macc: bool = False   # mode-acceleration static correction
    impl_modal_zeta: float = 0.0    # uniform modal damping ratio (MODAL/DAMP)
    impl_modal_prestress: bool = False  # extract modes on the prestressed K
    # /IMPL/FREQ: harmonic / steady-state frequency response (FRF sweep).
    impl_freq: bool = False         # /IMPL/FREQ present
    impl_freq_fmin: float = 0.0     # sweep start frequency (Hz)
    impl_freq_fmax: float = 0.0     # sweep end frequency (Hz)
    impl_freq_nf: int = 200         # number of sweep points
    impl_freq_zeta: float = 0.02    # uniform modal damping for the FRF
    # -- M18 /IMPL/CEIGV: COMPLEX / DAMPED eigenvalues (PORT card — freimpl.F
    # has no complex/damped eigensolver at all). Solves the QEP (l^2 M + l C
    # + K) phi = 0 via the state-space linearization; C = Rayleigh a M + b K
    # (from /IMPL/DYNA/DAMP) + the deck's discrete dashpots (/PROP/SPRING c,
    # /DAMP). Library-first exactly like /IMPL/EIGV. See complex_modal.py.
    impl_ceigv: bool = False        # /IMPL/CEIGV present -> complex modal
    impl_ceigv_nmode: int = 6       # number of complex modes to extract
    impl_ceigv_prestress: bool = False  # /IMPL/CEIGV/STRS: K = K_mat + K_geo
    # /IMPL/CEIGV/TRAN: complex-mode superposition TRANSIENT (card: t_end dt)
    impl_ceigv_tran: bool = False
    impl_ceigv_tend: float = 0.0
    impl_ceigv_dt: float = 0.0
    # /IMPL/CEIGV/FRF: complex (damped) FRF sweep (card: fmin fmax nf)
    impl_ceigv_frf: bool = False
    impl_ceigv_fmin: float = 0.0
    impl_ceigv_fmax: float = 0.0
    impl_ceigv_nf: int = 200
    # -- M19 /IMPL/PSD: RANDOM / SPECTRAL (PSD) response (PORT card — freimpl.F
    # has no random-vibration path). Response PSD S_uu = |H|^2 S_ff through the
    # M17 real-mode FRF (classical damping) or the M18 complex FRF
    # (/IMPL/PSD/CPLX, non-classical); reports the RMS, spectral moments and
    # crossing/peak rates. See implicit/random_response.py.
    impl_psd: bool = False          # /IMPL/PSD present -> random response
    impl_psd_funct: int = 0         # /FUNCT id of the input PSD S_ff(f)
    impl_psd_fmin: float = 0.0      # sweep start frequency (Hz)
    impl_psd_fmax: float = 0.0      # sweep end frequency (Hz)
    impl_psd_nf: int = 400          # number of sweep points
    impl_psd_nmode: int = 6         # modes retained in the FRF
    impl_psd_zeta: float = 0.02     # uniform modal damping for the FRF
    impl_psd_base: bool = False     # /IMPL/PSD/BASE: base-acceleration PSD
    impl_psd_dir: int = 0           # base-excitation rigid direction (0..5)
    impl_psd_cplx: bool = False     # /IMPL/PSD/CPLX: use the M18 complex FRF
    impl_psd_prestress: bool = False  # extract modes on the prestressed K
    # -- M19 /IMPL/RSPEC: RESPONSE SPECTRUM analysis (PORT card). Per-mode peak
    # r_i = Gamma_i Sa(omega_i)/omega_i^2 combined by SRSS and CQC (Der
    # Kiureghian 1981). See implicit/response_spectrum.py.
    impl_rspec: bool = False        # /IMPL/RSPEC present -> response spectrum
    impl_rspec_funct: int = 0       # /FUNCT id of the design spectrum Sa(f)
    impl_rspec_dir: int = 0         # excitation rigid direction (0..5)
    impl_rspec_zeta: float = 0.05   # spectrum / modal damping ratio
    impl_rspec_nmode: int = 6       # modes retained in the combination
    impl_rspec_prestress: bool = False  # extract modes on the prestressed K
    # -- M20 /IMPL/FATIG: RANDOM-VIBRATION (SPECTRAL) FATIGUE (PORT card —
    # freimpl.F has no spectral-fatigue solver). Recovers a STRESS PSD from the
    # M19 stress modes and evaluates the narrow-band (Bendat) / Dirlik /
    # Wirsching-Light / Tovo-Benasciutti damage of an S-N curve N = C*S^-m
    # under a Miner sum. See implicit/spectral_fatigue.py + random_response.py.
    impl_fatig: bool = False        # /IMPL/FATIG present -> spectral fatigue
    impl_fatig_funct: int = 0       # /FUNCT id of the input PSD S_ff(f)
    impl_fatig_fmin: float = 0.0    # FRF sweep start frequency (Hz)
    impl_fatig_fmax: float = 0.0    # FRF sweep end frequency (Hz)
    impl_fatig_nf: int = 800        # number of sweep points
    impl_fatig_nmode: int = 6       # modes retained in the FRF / stress modes
    impl_fatig_zeta: float = 0.02   # uniform modal damping for the FRF
    impl_fatig_base: bool = False   # /IMPL/FATIG/BASE: base-acceleration PSD
    impl_fatig_dir: int = 0         # base-excitation rigid direction (0..5)
    impl_fatig_prestress: bool = False  # extract modes on the prestressed K
    impl_fatig_snm: float = 0.0     # S-N slope m (N = C*S^-m)
    impl_fatig_snc: float = 0.0     # S-N coefficient C
    impl_fatig_mean: float = 0.0    # static mean stress (basic Goodman option)
    impl_fatig_ult: float = 0.0     # ultimate tensile strength S_u (Goodman)
    impl_fatig_mcdur: float = 0.0   # Monte-Carlo cross-check duration (0=off)
    impl_fatig_seed: int = 1        # Monte-Carlo random seed (reproducible)
    # M21 /IMPL/FATIG/MULT: MULTIAXIAL / critical-plane spectral fatigue — the
    # full stress-tensor cross-PSD reduced to an equivalent-stress PSD (von
    # Mises + max-normal / max-shear critical plane). See implicit/
    # multiaxial_fatigue.py.
    impl_fatig_mult: bool = False   # /IMPL/FATIG/MULT present -> multiaxial
    impl_fatig_nplane: int = 24     # candidate-plane azimuth divisions (15 deg)
    # M22 /IMPL/FATIG/MULT/NPROP: NON-PROPORTIONAL multiaxial fatigue — the
    # critical-plane TIME-DOMAIN path-counting damage of the rotating shear path
    # (MCC/MRH shear amplitude + Findley / Fatemi-Socie with the per-plane max
    # normal stress). Implies MULT. See implicit/nonproportional_fatigue.py.
    impl_fatig_nprop: bool = False  # /IMPL/FATIG/MULT/NPROP -> path-counting
    impl_fatig_k: float = 0.3       # Findley/Fatemi-Socie normal-sensitivity k
    impl_fatig_sigy: float = 1.0    # yield stress sigma_y (Fatemi-Socie)
    impl_fatig_amp: str = "mrh"     # shear-path amplitude: mrh / mcc / chord
    # M23 /IMPL/FATIG/MULT/NPROP/SPEC: SPECTRAL non-proportional multiaxial
    # fatigue — the frequency-domain F_np and critical-plane damage estimated
    # DIRECTLY from the cross-PSD moment matrices (no synthesised history), run
    # ALONGSIDE the M21 spectral + M22 time-domain answers. Implies NPROP (hence
    # MULT). See implicit/spectral_nonproportional_fatigue.py.
    impl_fatig_spec: bool = False   # /IMPL/FATIG/MULT/NPROP/SPEC -> spectral NP
    # M24 /IMPL/FATIG/NGAUSS: NON-GAUSSIAN / KURTOSIS correction — the Winterstein
    # Hermite-moment model + the Benasciutti-Braccesi / Rizzi-Kihm closed-form
    # correction factor lambda_ng that scales the Gaussian spectral damage for a
    # target kurtosis (and skewness), plus a non-Gaussian Monte-Carlo cross-check.
    # Composes with MULT / NPROP / SPEC (a scalar correction on the equivalent
    # stress). See implicit/nongaussian_fatigue.py.
    impl_fatig_ngauss: bool = False  # /IMPL/FATIG/NGAUSS -> non-Gaussian corr.
    impl_fatig_kurt: float = 3.0     # target kurtosis gamma_4 (3 = Gaussian)
    impl_fatig_skew: float = 0.0     # target skewness gamma_3 (0 = symmetric)
    impl_fatig_copula: str = "gaussian"  # target copula ("gaussian", "t")
    impl_fatig_copula_params: float = 4.0 # copula degrees of freedom (for t-copula)
    impl_fatig_bwcorr: bool = True   # Benasciutti-Tovo bandwidth attenuation
    # M32 /IMPL/FATIG/NGAUSS + /WVILLE: TIME-VARYING non-Gaussian instantaneous
    # spectrum — the target kurtosis gamma_4(t) (and end value for a linear sweep)
    # drifts ALONG the M31 continuous Wigner-Ville spectrum, re-computed per instant
    # from the instantaneous bandwidth alpha_2(t). A kurtosis-vs-time /FUNCT
    # (impl_fatig_kfunct) is sampled continuously onto the fine instant grid; else a
    # linear sweep kurt -> kurt1. See implicit/nongaussian_wigner_ville_fatigue.py.
    impl_fatig_kfunct: int = 0       # /FUNCT id: target kurtosis gamma_4-vs-time
    impl_fatig_kurt1: float = 0.0    # end kurtosis for a linear sweep (0 = constant)
    # M25 /IMPL/FATIG/NSTAT: NON-STATIONARY / EVOLUTIONARY-PSD fatigue — the M20
    # stationary estimators evaluated per stationary segment / time-window and
    # Miner-summed (the piecewise-stationary "mission profile" block model), plus
    # the amplitude-modulated (evolutionary S(w,t) = |A(t)|^2 S(w)) damage
    # integrated over the RMS distribution, with a non-stationary Monte-Carlo
    # cross-check and the M25<->M24 kurtosis bridge. Composes with MULT / NPROP /
    # SPEC / NGAUSS (a scaling on the equivalent-stress PSD). See implicit/
    # nonstationary_fatigue.py.
    impl_fatig_nstat: bool = False   # /IMPL/FATIG/NSTAT -> non-stationary path
    impl_fatig_modfunct: int = 0     # /FUNCT id: RMS scale-vs-time modulation
    impl_fatig_nstat_nseg: int = 0   # sample the modulation into nseg blocks
    #                                  (0 = use the function's own breakpoints)

    # M26 /IMPL/FATIG/EVOL: FULLY EVOLUTIONARY / NON-SEPARABLE-PSD fatigue — the
    # M20 estimators evaluated per time-WINDOW of a spectrogram whose spectral
    # SHAPE (not merely its RMS level) DRIFTS with time (a swept centre frequency /
    # broadening bandwidth — a "chirp-like" random process), each window carrying
    # its OWN full moment set, and Miner-summed. The general NON-SEPARABLE
    # extension of the M25 (separable |A(t)|^2 S(w)) block model — of which the M25
    # shared-shape scaling is the constant-shape reduction. The drifting-shape
    # window sweeps fc0 -> fc1 and broadens bw0 -> bw1 across nwin windows; the RMS
    # level schedule (and the mission time span) come from the shared modulation
    # /FUNCT (impl_fatig_modfunct) so EVOL composes with /NSTAT. Composes with
    # MULT / NPROP / SPEC / NGAUSS (a drifting window on the equivalent-stress
    # PSD). See implicit/evolutionary_fatigue.py.
    impl_fatig_evol: bool = False    # /IMPL/FATIG/EVOL -> evolutionary path
    impl_fatig_evol_fc0: float = 0.0  # window centre freq at the FIRST window (Hz)
    impl_fatig_evol_fc1: float = 0.0  # window centre freq at the LAST window (Hz)
    impl_fatig_evol_bw0: float = 0.0  # window bandwidth start (0 = flat/no drift)
    impl_fatig_evol_bw1: float = 0.0  # window bandwidth end (broadens/narrows)
    impl_fatig_evol_nwin: int = 12   # time-windows the spectrogram is sampled into

    # M27 /IMPL/FATIG/MULT/EVOL/JOINT: FULLY NON-STATIONARY / EVOLUTIONARY
    # MULTIAXIAL (JOINT-TENSOR) fatigue — the M21/M23 critical-plane reductions
    # evaluated PER TIME-WINDOW of the full 6x6 stress-TENSOR cross-PSD
    # S_sigmasigma(w, t), the critical-plane ORIENTATION and the non-proportionality
    # factor F_np RE-SEARCHED from the WINDOW's OWN tensor (so the plane may ROTATE
    # and F_np may DRIFT window to window), and the per-window multiaxial damages
    # Miner-summed — cross-validated against a non-stationary MULTIVARIATE
    # time-domain Monte-Carlo. Where M26 windowed the equivalent SCALAR of a FIXED
    # reduction, M27 lets the JOINT tensor evolve so the reduction itself drifts.
    # /JOINT implies MULT + EVOL and reuses the /EVOL drifting-shape schedule
    # (fc0 fc1 bw0 bw1 nwin) and the shared modulation /FUNCT; it composes with
    # /NPROP / /SPEC / /NGAUSS / /NSTAT. A PORT sub-flag (freimpl.F has no
    # joint-tensor evolutionary fatigue path). See
    # implicit/joint_evolutionary_fatigue.py.
    impl_fatig_joint: bool = False   # /IMPL/FATIG/MULT/EVOL/JOINT -> joint tensor
    # M33 /IMPL/FATIG/NGAUSS/JOINT/WVILLE: JOINT-TENSOR NON-GAUSSIAN distribution — a
    # VECTOR (component-wise) Winterstein-Hermite / translation-process transform of the
    # CORRELATED 6x6 stress tensor imposing a PER-COMPONENT target kurtosis (the Voigt
    # components xx yy zz xy yz zx) while PRESERVING the marginal variances / covariance,
    # so the resolved critical plane INHERITS the INDUCED kurtosis of the joint tensor
    # statistics (NOT the M24/M32 kurtosis imposed on the already-resolved scalar). The
    # per-component targets live on the trailing columns 4..9 of the M24 kurtosis line;
    # a NON-EMPTY list triggers the M33 joint path (composing with /JOINT + /WVILLE),
    # reported alongside the M32 equivalent-scalar and M31/M27 Gaussian numbers. A PORT
    # sub-flag (freimpl.F has no non-Gaussian / joint-tensor / Hermite path). See
    # implicit/joint_nongaussian_fatigue.py.
    impl_fatig_joint_kurt: tuple = ()  # per-component target kurtoses (empty = off)
    # M34 /IMPL/FATIG/NGAUSS/JOINT/WVILLE/EXACT (or /NORTA): the EXACT translation-process
    # CORRELATION-DISTORTION INVERSION — solve the underlying-Gaussian correlation rho^U
    # (the Grigoriu / Nataf / Cario-Nelson NORTA correlation matching) per component pair
    # so the component-wise Winterstein-Hermite transform of the correlated 6x6 tensor
    # reproduces the TARGET covariance EXACTLY (not merely to M33's leading order), driving
    # the covariance preservation error to ~0. A positive-definite Higham nearest-
    # correlation repair keeps the underlying Gaussian a valid covariance. Only meaningful
    # with the M33 /JOINT + /NGAUSS + /WVILLE joint path (a per-component kurtosis line);
    # reported as an ``exact_covariance`` sub-entry ALONGSIDE the M33 leading-order joint,
    # the M32 equivalent-scalar and the M31/M27 Gaussian numbers (all left byte-identical
    # — the EXACT path is a NEW path alongside them). A PORT sub-flag (freimpl.F has no
    # translation / copula / correlation solver). See implicit/joint_nongaussian_fatigue.py.
    impl_fatig_exact: bool = False   # /IMPL/FATIG/.../EXACT -> NORTA covariance-exact

    # M28 /IMPL/PSD/MULTI and /IMPL/FATIG/MINPUT: MULTI-INPUT / PARTIALLY-COHERENT
    # random-vibration response & fatigue — the stationary response (and
    # stress-tensor) cross-PSD driven by SEVERAL simultaneous random inputs with a
    # full Hermitian input cross-spectral matrix S_ff(w) = [sqrt(G_a G_b) gamma_ab
    # exp(i theta_ab)] (auto-PSDs on the diagonal, coherence gamma_ab and phase
    # theta_ab off-diagonal), propagated through the VECTOR FRF by the MIMO relation
    # S_uu = H S_ff H^H / S_sigmasigma = H_sigma S_ff H_sigma^H, then reduced by the
    # WHOLE M20-M27 estimator family UNCHANGED (the multi-input S_sigmasigma flows
    # straight into them). /MULTI (PSD) and /MINPUT (FATIG) read a table of input
    # load patterns + their auto-PSD /FUNCTs and a coherence/phase model, assemble
    # S_ff, recover S_uu / S_sigmasigma, run the reductions and the multi-input
    # Monte-Carlo, and report the multi-input answers ALONGSIDE the single-input
    # numbers (a "multi_input" sub-entry). The single scalar input is exactly the
    # 1x1 (diagonal / rank-1) special case (bit-identical). A PORT sub-flag
    # (freimpl.F has no multi-input / coherence path — the sole PSD token is the
    # MUMPS flag IMUMPSD). Composes with /MULT / /NPROP / /SPEC / /NGAUSS / /NSTAT /
    # /EVOL / /JOINT. See implicit/multi_input_response.py + multi_input_fatigue.py.
    impl_psd_multi: bool = False     # /IMPL/PSD/MULTI -> multi-input response
    impl_fatig_minput: bool = False  # /IMPL/FATIG/MINPUT -> multi-input fatigue
    # the input-pattern table (list of dicts: cload_funct, psd_funct, pos) + the
    # coherence model, filled by the reader; the driver assembles S_ff from them
    impl_mi_inputs: tuple = ()       # per-input (cload_funct, psd_funct, x,y,z)
    impl_mi_cohmodel: int = 0        # 0 = constant coherence, 1 = exponential/decay
    impl_mi_gamma: float = 0.0       # constant coherence gamma_ab in [0,1]
    impl_mi_phase: float = 0.0       # constant phase theta_ab (degrees)
    impl_mi_decay: float = 0.0       # exponential-coherence decay coefficient
    impl_mi_speed: float = 1.0       # exponential-coherence reference speed

    # M29 /IMPL/FATIG/MULT/MINPUT/EVOL and /IMPL/PSD/MULTI/EVOL: FULLY NON-STATIONARY
    # / EVOLUTIONARY MULTI-INPUT cross-PSD — the input coherence matrix S_ff(w, t)
    # itself DRIFTS with time (the coherence gamma_ab(t) and phase theta_ab(t)), so
    # the per-window multi-input stress-tensor cross-PSD S_sigmasigma(w, t_i) =
    # H_sigma S_ff(t_i) H_sigma^H drives a per-window critical-plane search whose
    # plane / F_np may DRIFT as the coherence evolves, reduced by the M20-M27
    # estimators + the M28 multi-input path and Miner-summed. Where M28 was
    # STATIONARY multi-input (a fixed S_ff modulated at most by a scalar RMS profile /
    # drifting-shape window, the coherence held stationary), M29 interpolates the
    # coherence across the M26/M27 windows from a START pair (impl_mi_gamma /
    # impl_mi_phase) to an END pair (impl_mi_gamma1 / impl_mi_phase1). It reuses the
    # /EVOL drifting-shape schedule (impl_fatig_evol_fc0 .. _nwin) and the shared
    # modulation /FUNCT; it composes with /JOINT / /NSTAT / /NGAUSS. The M28
    # stationary answer is exactly the single-window / constant-coherence special
    # case; the M27 single-input answer the ninput = 1 special case. A PORT sub-flag
    # (freimpl.F has no time-varying-coherence path — the sole PSD token is IMUMPSD).
    # See implicit/evolutionary_multi_input.py.
    impl_mi_gamma1: float = -1.0     # END coherence gamma_ab(t_1); < 0 -> no drift
    impl_mi_phase1: float = 0.0      # END phase theta_ab(t_1) (degrees)

    # M30 /IMPL/FATIG/MULT/MINPUT/EVOL/FCOH: FREQUENCY-DEPENDENT + TIME-VARYING
    # (EVOLUTIONARY) INPUT COHERENCE — a coherence matrix gamma_ab(f, t) varying with
    # BOTH FREQUENCY AND TIME. Where M29 drifted a per-window SCALAR gamma_ab(t_i)
    # (frequency-FLAT, constant across f within a window) and held the M28 exponential
    # coherence STATIONARY, M30 lets the coherence be a FULL (nf, ninput, ninput)
    # frequency-dependent stack gamma_ab(f) that ALSO drifts window to window: either
    # a per-pair MEASURED shape gamma(f) /FUNCT interpolated start (impl_mi_gfunct0)
    # -> end (impl_mi_gfunct1), or the M28 exponential/convection field with a
    # TIME-VARYING decay coefficient (impl_mi_decay -> impl_mi_decay1) / reference
    # speed (impl_mi_speed -> impl_mi_speed1) — a turbulence field whose
    # DECORRELATION FREQUENCY drifts through the mission. Each window's S_ff(w, t_i)
    # drives a per-window multi-input S_sigmasigma whose critical plane / F_np may
    # DRIFT as the coherence FREQUENCY-SHAPE evolves, reduced by the M20-M27 estimators
    # + the M28/M29 multi-input paths and Miner-summed, cross-validated by a
    # non-stationary multi-input Monte-Carlo whose measured coherence SPECTRUM (per
    # frequency band) tracks the target. M29 is EXACTLY the frequency-flat special
    # case; M28 (frequency-dependent) the single-window special case. A PORT sub-flag
    # (freimpl.F has no frequency-dependent-time-varying-coherence path — the sole PSD
    # token is IMUMPSD). Reported ALONGSIDE the M29 scalar-coherence + M28
    # frequency-dependent-stationary numbers (a "freq_evolutionary_multi_input"
    # sub-entry). See implicit/freq_evolutionary_multi_input.py.
    impl_mi_fcoh: bool = False       # /FCOH -> frequency-dependent evolutionary path
    impl_mi_decay1: float = -1.0     # END exponential decay coeff; < 0 -> no drift
    impl_mi_speed1: float = -1.0     # END exponential ref speed; < 0 -> no drift
    impl_mi_gfunct0: int = 0         # START measured coherence shape gamma(f) /FUNCT
    impl_mi_gfunct1: int = 0         # END measured coherence shape gamma(f) /FUNCT

    # M31 /IMPL/FATIG/.../WVILLE: CONTINUOUS WIGNER-VILLE / LOEVE INSTANTANEOUS
    # TIME-FREQUENCY SPECTRUM — a bilinear time-frequency distribution S_WV(omega, t)
    # of the scalar (M26) / 6x6 joint-tensor (M27) / multi-input coherence-matrix
    # (M29/M30) response process, replacing the M26-M30 SHORT-TIME WINDOWED
    # SPECTROGRAM (the piecewise-locally-stationary windows those milestones sample
    # the mission into) with a CONTINUOUS instantaneous spectrum evaluated at a fine
    # instant grid, reduced by the M20-M27 estimators AT EACH INSTANT (the critical
    # plane / F_np drifting CONTINUOUSLY) and Palmgren-Miner INTEGRATED over time (an
    # integral, not a per-window sum). The windowed spectrogram is EXACTLY the
    # long-window / coarsest-grid (refine = 1) / un-smoothed (smooth = 0) limit — the
    # continuous path DELEGATES to M26/M27/M29/M30 there byte-identically, so the
    # windowed numbers are reported ALONGSIDE unchanged (a "wigner_ville" sub-entry).
    # /WVILLE IMPLIES /EVOL (it needs the drifting-shape schedule) and composes with
    # /JOINT / /MINPUT / /FCOH (whichever tensor / multi-input path is active becomes
    # the continuous-instantaneous one) + /NSTAT. The grid-refinement factor `refine`
    # (fine instants per M26-M30 window) and the Cohen-class cross-term smoothing
    # width `smooth` live on the trailing columns of the /EVOL drifting-shape line
    # (fc0 fc1 bw0 bw1 nwin [refine smooth]). A PORT sub-flag (freimpl.F has no
    # time-frequency / Wigner-Ville / spectral solver of any kind — the sole PSD token
    # is IMUMPSD, a MUMPS flag). See implicit/wigner_ville_fatigue.py.
    impl_fatig_wville: bool = False  # /WVILLE -> continuous instantaneous spectrum
    impl_fatig_wv_refine: int = 8    # fine instants per M26-M30 window (grid refine)
    impl_fatig_wv_smooth: float = 0.0  # Cohen-class time-smoothing width (mission frac)


class Model:
    """The full model. Created empty, filled by the keyword parsers, then
    finalized (IDs → indices, mass init...) by the Starter."""

    def __init__(self):
        # ------------------------------------------------------------------
        # Nodal data (Fortran X(3,NUMNOD), V(3,NUMNOD), MS(NUMNOD), ITAB)
        # Stored as (N, 3) row-major — the transpose of the Fortran layout,
        # which is the natural NumPy orientation.
        # ------------------------------------------------------------------
        self.node_ids: np.ndarray = np.zeros(0, dtype=np.int64)   # ITAB
        self.x0: np.ndarray = np.zeros((0, 3))    # initial coordinates
        self.x: np.ndarray = np.zeros((0, 3))     # current coordinates
        self.v: np.ndarray = np.zeros((0, 3))     # velocities
        self.vr: np.ndarray = np.zeros((0, 3))    # rotational velocities (shells)
        self.mass: np.ndarray = np.zeros(0)       # lumped mass MS
        self.mass0: np.ndarray = np.zeros(0)      # physical MS before any
        # /DT/NODA/CST mass scaling (M6) — gravity and init-time bounds
        self.inertia: np.ndarray = np.zeros(0)    # lumped nodal inertia IN (shells)
        self._id2idx: Dict[int, int] = {}         # USR2SYS node map

        # ------------------------------------------------------------------
        # Elements by type
        # ------------------------------------------------------------------
        self.bricks: Optional[ElementGroup] = None    # /BRICK  (IXS)
        self.bricks_heph: Optional[ElementGroup] = None  # /BRICK (HEPH ISOLID=24)
        self.quads: Optional[ElementGroup] = None     # /QUAD   (IXQ)
        self.tetras: Optional[ElementGroup] = None    # /TETRA4 (IXS10 kin)
        self.tetra10s: Optional[ElementGroup] = None  # /TETRA10
        self.shells: Optional[ElementGroup] = None    # /SHELL  (IXC)
        self.shells_qbat: Optional[ElementGroup] = None  # /SHELL Ishell=12
        #                                       (QBAT split, M41 dispatch)
        self.shells_qeph: Optional[ElementGroup] = None  # /SHELL Ishell=24
        #                                       (QEPH split, M41 dispatch)
        self.sh3n: Optional[ElementGroup] = None      # /SH3N   (IXTG)
        self.sh3n_dkt18: Optional[ElementGroup] = None  # /SH3N Ish3n=2
        self.trusses: Optional[ElementGroup] = None   # /TRUSS  (IXT)
        self.springs: Optional[ElementGroup] = None   # /SPRING (IXR)
        self.beams: Optional[ElementGroup] = None     # /BEAM   (IXP)
        self.shel16s: Optional[ElementGroup] = None   # /SHEL16 (IXS16)
        self.bric20s: Optional[ElementGroup] = None   # /BRIC20 / /HEXA20 (M122)
        # raw (id, part_id, node ids...) tuples collected during parsing,
        # converted to ElementGroups in Starter finalization:
        self.raw_elems: Dict[str, list] = {
            "BRICK": [], "QUAD": [], "TETRA4": [], "TETRA10": [], "SHELL": [], "SH3N": [],
            "TRUSS": [], "SPRING": [], "BEAM": [], "SHEL16": [], "BRIC20": [], "HEXA20": [], "SPH": []}

        # ------------------------------------------------------------------
        # Definitions keyed by user id
        # ------------------------------------------------------------------
        self.materials: Dict[int, Material] = {}
        # /FAIL cards awaiting attachment to their material: parsed as
        # (mat_id, FailureModel, source) tuples, attached by the Starter
        # resolve step (deck order between /MAT and /FAIL is free).
        self.raw_fails: list = []
        # /EOS cards, same pattern (M6): (mat_id, EquationOfState, source)
        self.global_damping: Optional[Dict] = None

        self.n2d: int = 0  # 0: 3D, 1: axisymmetric, 2: plane strain (from /ANALY)
        self.has_ale: bool = False  # True if /ALE/DONE is present (M63)

        # Global element defaults (M68): /DEF_SHELL and /DEF_SOLID.
        # Mirrors Fortran DEFAULTS_SHELL / DEFAULTS_SOLID (defaults_mod.F90).
        # Values of 0 mean "use the Starter's init_def_elem fallback".
        self.def_shell: Dict[str, int] = {
            'ishell': 0, 'ismstr': 0, 'ithick': 0, 'iplas': 0,
            'istrain': 0, 'ish3n': 0, 'idrill': 0}
        self.def_solid: Dict[str, int] = {
            'isolid': 0, 'ismstr': 0, 'icpre': 0,
            'itetra4': 0, 'itetra10': 0, 'imas': 0, 'iframe': 0}

        self.raw_eos: list = []
        # /ALE/MAT, /EULER/MAT, /HEAT/MAT parse-only notes (M37): parsed
        # as (kind, mat_id, params, source), attached to the material's
        # params by the Starter resolve step — accepted, no physics.
        self.raw_mat_notes: list = []
        self.properties: Dict[int, Property] = {}
        self.parts: Dict[int, Part] = {}
        self.parts_list: List[Part] = []          # dense order for elements
        self.functions: Dict[int, FunctTable] = {}
        self.tables: Dict[int, Table] = {}
        self.randoms: Dict[int, Random] = {}
        self.node_groups: Dict[int, NodeGroup] = {}
        self.move_functs: List[Tuple[int, float, float, float, float]] = []
        self.surfaces: Dict[int, Surface] = {}
        self.monitored_volumes: Dict[int, 'MonitoredVolume'] = {}
        self.lines: Dict[int, Line] = {}
        self.boxes: Dict[int, Box] = {}
        # ELEMENT groups (M37): /GRSHEL, /GRSH3N, /GRBRIC, /GRQUAD,
        # /GRTRUS, /GRBEAM, /GRSPRI, /GRPART — one id namespace per
        # family (the Fortran IGRSH4N/IGRSH3N/IGRBRIC/... arrays are
        # separate), keyed family -> {id -> EntityGroup}.
        self.egroups: Dict[str, Dict[int, EntityGroup]] = {}

        # ------------------------------------------------------------------
        # Unit systems (M37): /BEGIN work/input units + /UNIT blocks.
        # ``unit_work``/``unit_input`` are (fac_m, fac_l, fac_t) SI-factor
        # triples or None (legacy deck without unit declarations);
        # ``units`` maps /UNIT ids to the same triples; ``raw_unit_refs``
        # collects (keyword, user_id, unit_id, source) for every block
        # whose header carried a LOCAL unit id — converted into the work
        # unit system by starter/initialization (see input/units.py).
        # ------------------------------------------------------------------
        self.unit_work = None
        self.unit_input = None
        self.units: Dict[int, tuple] = {}
        self.raw_unit_refs: list = []

        # Reference systems (M39): every /SKEW and /FRAME, resolved to
        # rotation matrices + origins by starter/initialization.py
        # (resolve_skews).  Row 0 is the GLOBAL system, so a consumer's
        # skew_ID/frame_ID = 0 needs no special case — see model/skew.py.
        self.skews = SkewSet()

        # SUBMODEL tracking (M42/M87)
        self.active_submodels: List[int] = []
        self.node_submodel: np.ndarray = np.zeros(0, dtype=np.int32)
        self.submodels: Dict[int, Submodel] = {}

        # SUBDOMAIN domain partitions (M88 — Rad2Rad coupling)
        self.subdomains: Dict[int, Subdomain] = {}

        # XREF reference geometry (M89)
        self.xrefs: Dict[int, Xref] = {}

        # Detonation ignition sources (M91)
        self.det_points: List[DetonatorPoint] = []
        self.det_planes: List[DetonatorPlane] = []

        # Loads / constraints / contacts
        self.bcs: List[BoundaryCondition] = []
        self.cyclic_bcs: Dict[int, CyclicBoundaryCondition] = {} # /BCS/CYCLIC (M99)
        self.ale_bcs: List[AleBoundaryCondition] = []
        self.inivel: List[InitialVelocity] = []
        self.ini_bricks: Dict[int, InitialBrickState] = {}  # /INIBRI (M96)
        self.ini_shells: Dict[int, InitialShellState] = {}  # /INISHE, /INISH3 (M96)
        self.ini_trusses: Dict[int, InitialTrussState] = {} # /INITRU (M97)
        self.ini_beams: Dict[int, InitialBeamState] = {}    # /INIBEA (M97)
        self.ini_springs: Dict[int, InitialSpringState] = {} # /INISPR (M97)
        self.perturbations: Dict[int, SolidPartPerturbation] = {} # /PERTURB/PART/SOLID (M99)
        self.gravity: List[Gravity] = []
        self.cloads: List[ConcentratedLoad] = []
        self.centri_loads: List[CentrifugalLoad] = []  # /LOAD/CENTRI (M93)
        self.impvel: List[ImposedVelocity] = []
        self.impdisp: List[ImposedDisplacement] = []   # /IMPDISP (M5)
        self.impacc: List[ImposedAcceleration] = []    # /IMPACC  (M92)
        self.imptemp: List[ImposedTemperature] = []    # /IMPTEMP (M93)
        self.convec_loads: List[ConvectionLoad] = []    # /CONVEC  (M94)
        self.radiation_loads: List[RadiationLoad] = []  # /RADIATION (M95)
        self.impflux_loads: List[ImposedFlux] = []      # /IMPFLUX (M95)
        self.impfluxes: Dict[int, ImposedFlux] = {}     # /IMPFLUX (M95/M152)
        self.initemp: List[InitialTemperature] = []    # /INITEMP (M95)
        self.inivol: List[InitialVolume] = []          # /INIVOL  (M94)
        self.ploads: List[PressureLoad] = []           # /PLOAD   (M5)
        self.pblast_loads: Dict[int, PBlastLoad] = {}  # /LOAD/PBLAST (M99)
        self.def_inter: Dict[str, Any] = {}            # /DEF_INTER (M99/M101)
        self.perturb_shells: Dict[int, ShellPartPerturbation] = {} # /PERTURB/PART/SHELL (M101)
        self.perturb_fails: Dict[int, FailurePerturbation] = {}    # /PERTURB/FAIL (M101)
        self.sph_global: Optional[SphGlobal] = None    # /SPHGLO (M101)
        self.sms_global: Optional[SmsGlobal] = None    # /SMS, /AMS (M101)
        self.admas: List[AddedMass] = []               # /ADMAS   (M5)
        self.rwalls: List[RigidWall] = []
        self.rbodies: List[RigidBody] = []             # /RBODY + /RBE2 (M5)
        self.rbe3: List[Rbe3] = []                     # /RBE3    (M5)
        self.sections: List[Section] = []              # /SECT    (M5)
        self.damps: List[Damping] = []                 # /DAMP    (M6)
        self.sensors: List[Sensor] = []                # /SENSOR  (M6)
        self.mpcs: List[Mpc] = []                      # /MPC     (M6)
        self.interfaces: List[Interface] = []
        self.sub_interfaces: List[SubInterface] = []   # /INTER/SUB (M100)
        self.guided_cables: Dict[int, GuidedCable] = {} # /INTER/GUIDED_CABLE (M149)
        self.plies: Dict[int, Ply] = {}                # /PLY (M100)
        self.laminates: Dict[int, Laminate] = {}       # /LAMINATE (M100)
        self.bcs_nrf: Dict[int, BcsNrf] = {}           # /BCS/NRF (M102)
        self.bcs_walls: Dict[int, BcsWall] = {}        # /BCS/WALL (M102)
        self.rlinks: Dict[int, RigidLink] = {}         # /RLINK (M102)
        self.cyl_joints: Dict[int, CylJoint] = {}      # /CYL_JOINT (M102)
        self.gjoints: Dict[int, GeneralJoint] = {}     # /GJOINT (M102)
        self.node_merges: Dict[int, MergeNode] = {}    # /MERGE/NODE (M102)
        self.rbody_merges: Dict[int, MergeRbody] = {}  # /MERGE/RBODY (M102)
        self.inicracks: Dict[int, IniCrack] = {}       # /INICRACK (M102)
        self.laser_loads: Dict[int, LaserLoad] = {}    # /LASER (M102)
        self.pcyl_loads: Dict[int, PcylLoad] = {}      # /LOAD/PCYL (M103)
        self.pfluid_loads: Dict[int, PfluidLoad] = {}  # /LOAD/PFLUID (M103)
        self.preloads: Dict[int, Preload] = {}         # /PRELOAD (M103)
        self.preload_axials: Dict[int, PreloadAxial] = {} # /PRELOAD/AXIAL (M103)
        self.damp_inters: Dict[int, DampInter] = {}    # /DAMP/INTER (M103)
        self.damp_ranges: Dict[int, DampRange] = {}    # /DAMP/RANGE (M103)
        self.analy_global: Optional[AnalyGlobal] = None # /ANALY (M103)
        self.upwind_global: Optional[UpwindGlobal] = None # /UPWIND (M103)
        self.caa_controls: Dict[int, CaaControl] = {}  # /CAA (M103)
        self.gauges: Dict[int, Gauge] = {}             # /GAUGE (M104)
        self.clusters: Dict[int, Cluster] = {}         # /CLUSTER (M104)
        self.ext_links: Dict[int, ExtLink] = {}        # /EXTLNK (M104)
        self.fxbodies: Dict[int, FxBody] = {}          # /FXBODY (M104)
        self.ini_gravs: Dict[int, IniGrav] = {}        # /INIGRAV (M104)
        self.ini_map1ds: Dict[int, IniMap1D] = {}      # /INIMAP1D (M104)
        self.ini_map2ds: Dict[int, IniMap2D] = {}      # /INIMAP2D (M104)
        self.ini_state_file: Optional[IniStateFile] = None # /INISTATE (M104)
        self.monvol_pres: Dict[int, MonvolPres] = {}   # /MONVOL/PRES (M105)
        self.monvol_gases: Dict[int, MonvolGas] = {}   # /MONVOL/GAS (M105)
        self.monvol_commus: Dict[int, MonvolCommu1] = {} # /MONVOL/COMMU1 (M105)
        self.monvol_lfluids: Dict[int, MonvolLFluid] = {} # /MONVOL/LFLUID (M105)
        self.monvol_airbags: Dict[int, MonvolAirbag] = {} # /MONVOL/AIRBAG (M133)
        self.monvol_commu_type5: Dict[int, MonvolCommu] = {} # /MONVOL/COMMU TYPE5 (M133)
        self.monvol_parts: Dict[int, MonvolPart] = {}   # /MONVOL/PART (M133)
        self.leak_mats: Dict[int, LeakMat] = {}        # /LEAK/MAT (M105, M133)
        self.leak_parts: Dict[int, LeakMat] = {}       # /LEAK/PART (M133)
        self.leak_areas: Dict[int, LeakMat] = {}       # /LEAK/AREA (M133)
        self.ale_grids: Dict[int, AleGrid] = {}        # /ALE/GRID (M105)
        self.ale_links: Dict[int, AleLink] = {}        # /ALE/LINK (M105)
        self.ale_solver: Optional[AleSolver] = None    # /ALE/SOLVER (M105)
        self.ale_close: Optional[AleClose] = None      # /ALE/CLOS (M105)
        self.retractors: Dict[int, Retractor] = {}     # /RETRACTOR (M106)
        self.sliprings: Dict[int, Slipring] = {}       # /SLIPRING (M106)
        self.pretensioners: Dict[int, Pretensioner] = {} # /PRETENSIONER (M197)
        self.user_windows: List[UserWindow] = []       # /USERWI (M106)
        self.drapes: Dict[int, Drape] = {}             # /DRAPE (M107)
        self.inibri_erefs: List[IniBriEref] = []       # /INIBRI/EREF (M107)
        self.dyna_includes: List[IncludeDyna] = []     # /INCLUDE_DYNA (M107)
        self.monvol_fvmbags: Dict[int, MonvolFvmBag1] = {} # /MONVOL/FVMBAG1 (M108)
        self.detonations: List[DetonationWave] = []    # /INIT/DET_* (M110)
        self.activations: List[ElementActivation] = [] # /ACTIV (M110)
        self.monvol_fvmbag2s: Dict[int, MonvolFvmBag2] = {} # /MONVOL/FVMBAG2 (M111)
        self.autopositions: List[Autoposition] = []    # /TRANSFORM/AUTOPOSITION (M111)
        self.transform_projections: List[TransformProjection] = [] # /TRANSFORM/PROJ (M134)
        self.transform_frames: List[TransformFrame] = [] # /TRANSFORM/FRAME (M134)
        self.damp_globals: List[DampGlobal] = []       # /DAMP/GLOBAL (M134)
        self.damp_parts: Dict[int, DampPart] = {}      # /DAMP/PART (M134)
        self.load_centris: Dict[int, LoadCentri] = {}       # /LOAD/CENTRI (M112)
        self.load_pfluids: Dict[int, LoadPfluid] = {}       # /LOAD/PFLUID (M112)
        self.load_pressures: Dict[int, LoadPressure] = {}   # /LOAD/PRESSURE (M112)
        self.load_gravities: Dict[int, LoadGravity] = {}   # /LOAD/GRAV (M135)
        self.load_bodies: Dict[int, LoadBody] = {}         # /LOAD/BODY (M135)
        self.load_therms: Dict[int, LoadTherm] = {}        # /LOAD/HEAT (M135)
        self.euler_bcs: Dict[int, EulerBcs] = {}           # /EULER/BCS (M135)
        self.heat_bcs: Dict[int, HeatBcs] = {}             # /HEAT/BCS (M135)
        self.inivel_axes: Dict[int, InivelAxis] = {}        # /INIVEL/AXIS (M112)
        self.inivel_fvms: Dict[int, InivelFvm] = {}         # /INIVEL/FVM (M112)
        self.inivel_nodes: Dict[int, InivelNode] = {}       # /INIVEL/NODE (M112)
        self.inivel_parts: Dict[int, InivelPart] = {}       # /INIVEL/PART (M137)
        self.inivel_sphs: Dict[int, InivelSph] = {}         # /INIVEL/SPH (M137)
        self.impdisp_fgeos: Dict[int, ImpdispFgeo] = {}     # /IMPDISP/FGEO (M112)
        self.impvel_fgeos: Dict[int, ImpvelFgeo] = {}       # /IMPVEL/FGEO (M112)
        self.rwall_therms: Dict[int, RwallTherm] = {}       # /RWALL/THERM (M112)
        self.rwall_boxes: Dict[int, RwallBox] = {}         # /RWALL/BOX (M136)
        self.rwall_cones: Dict[int, RwallCone] = {}         # /RWALL/CONE (M136)
        self.sect_boxes: Dict[int, SectBox] = {}           # /SECT/BOX (M136)
        self.sect_cuts: Dict[int, SectCut] = {}             # /SECT/CUT (M136)
        self.sph_inouts: Dict[int, SphInOut] = {}           # /SPH/INOUT (M112)
        self.sph_bcs: Dict[int, SphBcs] = {}                # /SPHBCS (M113)
        self.madymo_links: Dict[int, MadymoLink] = {}       # /MADYMO/LINK (M113)
        self.madymo_exfems: Dict[int, MadymoExfem] = {}     # /MADYMO/EXFEM (M113)
        self.ale_grid_donea: Optional[AleGridDonea] = None  # /ALE/GRID/DONEA (M113)
        self.ale_grid_spring: Optional[AleGridSpring] = None # /ALE/GRID/SPRING (M113)
        self.ale_grid_standard: Optional[AleGridStandard] = None # /ALE/GRID/STANDARD (M113)
        self.ale_grid_disp: Optional[AleGridDisp] = None    # /ALE/GRID/DISP (M113)
        self.ale_grid_laplacian: Optional[AleGridLaplacian] = None # /ALE/GRID/LAPLACIAN (M113)
        self.ale_grid_volume: Optional[AleGridVolume] = None # /ALE/GRID/VOLUME (M113)
        self.admesh_global: Optional[AdmeshGlobal] = None   # /ADMESH/GLOBAL (M113)
        self.stamping_inits: List[StampingInit] = []        # /STAMPING (M113)
        self.random_noises: List[RandomNoise] = []          # /RANDOM (M113)
        self.accelerometers: Dict[int, Accelerometer] = {}  # /ACCEL (M113)
        self.subsets: Dict[int, Subset] = {}                # /SUBSET (M113)
        self.fail_composites: Dict[int, FailComposite] = {} # /FAIL/COMPOSITE (M114)
        self.ebcs_propellants: Dict[int, EbcsPropellant] = {} # /EBCS/PROPELLANT (M114)
        self.admas_non_uniforms: Dict[int, AdmasNonUniform] = {} # /ADMAS/NON_UNIFORM (M114)
        self.sect_circles: Dict[int, SectCircle] = {}       # /SECT/CIRCLE (M114)
        self.sect_parals: Dict[int, SectParal] = {}         # /SECT/PARAL (M114)
        self.dynain_shells: List[DynainShell] = []          # /DYNAIN/SHELL (M114)
        self.monvol_areas: Dict[int, MonvolArea] = {}       # /MONVOL/AREA (M115)
        self.state_dts: List[StateDt] = []                  # /STATE/DT, /DYNAIN/DT (M115)
        self.th_titles: List[str] = []                      # /TH/TITLE (M115)
        self.sph_reserves: Dict[int, SphReserve] = {}       # /SPH/RESERVE (M116)
        self.eigen_modes: Dict[int, EigenMode] = {}         # /EIG (M117)
        self.stress_files: List[StressFile] = []            # /STATE/STR_FILE (M117)
        self.memory_requests: List[MemoryRequest] = []      # /MEMORY (M117)
        self.shfra_v4: bool = False                         # /SHFRA/V4 (M117)
        self.intthick_v5: bool = False                      # /INTTHICK/V5 (M117)
        self.fail_fractals: Dict[int, FailFractal] = {}     # /FAIL/FRACTAL (M118)
        self.transform_positions: Dict[int, TransformPosition] = {} # /TRANSFORM/POS (M118)
        self.external_links: Dict[int, ExternalLink] = {}   # /EXTERN/LINK (M118)
        self.subdomains: Dict[int, Subdomain] = {}          # /SUBDOMAIN (M118)
        self.arch_specs: List[ArchSpec] = []                # /ARCH (M118)
        self.altdoctags: List[str] = []                     # /ALTDOCTAG (M118)
        self.ale_grid_flow_tracking: Optional[Dict[str, Any]] = None # /ALE/GRID/FLOW-TRACKING (M118)
        self.ale_grid_lagrange: bool = False                # /ALE/GRID/LAGRANGE (M118)
        self.ale_zero: bool = False                         # /ALE/ZERO (M118)
        self.funct_pythons: Dict[int, FunctPython] = {}     # /FUNCT_PYTHON (M119)
        self.friction_models: Dict[int, FrictionModel] = {} # /FRICTION (M119)
        self.refsta_nodes: Dict[int, RefstaNode] = {}       # /REFSTA (M119)
        self.eref_specs: Dict[int, ErefSpec] = {}           # /EREF (M119)
        self.nbcs_blocks: Dict[int, NbcsBlock] = {}         # /NBCS (M119)
        self.ale_muscl: Optional[AleMuscl] = None           # /ALE/MUSCL (M119)
        self.bem_models: Dict[int, BemModel] = {}           # /BEM (M119)
        self.gauge_points: Dict[int, GaugePoint] = {}       # /GAUGE/POINT (M121)
        self.sphglo: Optional[SphGlo] = None                # /SPHGLO (M121)
        self.analy: Optional[AnalyOptions] = None           # /ANALY (M121)
        self.alecfdsph: Optional[AleCfdSph] = None          # /ALECFDSPH (M122)
        self.fail_orthbiquads: Dict[int, FailOrthBiquad] = {} # /FAIL/ORTHBIQUAD (M123)
        self.slipring_shells: Dict[int, SlipringShell] = {}   # /SLIPRING/SHELL (M123)
        self.ebcs_nrfs: Dict[int, EbcsNrf] = {}               # /EBCS/NRF (M125)
        self.fail_rtcls: Dict[int, FailRtcl] = {}             # /FAIL/RTCL (M125)
        self.fail_gursons: Dict[int, FailGurson] = {}         # /FAIL/GURSON (M125)
        self.fail_pucks: Dict[int, FailPuck] = {}             # /FAIL/PUCK (M126)
        self.fail_sahraeis: Dict[int, FailSahraei] = {}       # /FAIL/SAHRAEI (M126)
        self.fail_syazwans: Dict[int, FailSyazwan] = {}       # /FAIL/SYAZWAN (M126)
        self.fail_tab2s: Dict[int, FailTab2] = {}             # /FAIL/TAB2 (M126)
        self.fail_gene1s: Dict[int, FailGene1] = {}           # /FAIL/GENE1 (M126)
        self.fail_nxts: Dict[int, FailNxt] = {}               # /FAIL/NXT (M159)
        self.fail_laddamas: Dict[int, FailLadDama] = {}       # /FAIL/LAD_DAMA (M159)
        self.fail_inievos: Dict[int, FailInievo] = {}         # /FAIL/INIEVO (M159)
        self.stacks: Dict[int, Stack] = {}                   # /STACK (M127)
        self.lagmul_global: Optional[LagmulGlobal] = None     # /LAGMUL (M131)
        self.gears: Dict[int, GearConstraint] = {}            # /GEAR (M131)
        self.racks: Dict[int, RackConstraint] = {}            # /RACK (M131)
        self.diffs: Dict[int, DiffConstraint] = {}            # /DIFF (M131)
        self.diff_constraints = self.diffs
        self.wave_shapers: Dict[int, WaveShaper] = {}      # /DFS/WAVE_SHAPER (M132)
        self.det_lines: Dict[int, DetLine] = {}            # /DFS/DETLINE (M137)
        self.det_circs: Dict[int, DetCirc] = {}            # /DFS/DETCIRC (M137)
        self.ini_map3ds: Dict[int, IniMap3D] = {}          # /INIMAP/3D (M137)
        self.generic_sets: Dict[str, Dict[int, SetGeneric]] = {} # /SET (M137)
        self.ebcs_periodics: Dict[int, EbcsPeriodic] = {}  # /EBCS/PERIODIC (M138)
        self.ebcs_cyclics: Dict[int, EbcsCyclic] = {}      # /EBCS/CYCLIC (M138)
        self.mat_plas_zerils: Dict[int, MaterialPlasZeril] = {} # /MAT/PLAS_ZERIL (M141)
        self.mat_plas_bodnes: Dict[int, MaterialPlasBodne] = {} # /MAT/PLAS_BODNE (M141)
        self.mat_visc_pronys: Dict[int, MaterialViscProny] = {} # /MAT/VISC_PRONY (M141)
        self.mat_therm_stresses: Dict[int, MaterialThermStress] = {} # /MAT/THERM_STRESS (M141)
        self.damp_stiffs: Dict[int, DampStiff] = {}        # /DAMP/STIFF (M141)
        self.airbag_injectors: Dict[int, AirbagInjector] = {} # /AIRBAG/INJECTOR (M142)
        self.airbag_ventholes: Dict[int, AirbagVenthole] = {} # /AIRBAG/VENTHOLE (M142)
        self.eref_elements: Dict[int, ErefElement] = {}       # /EREF/* (M143)
        self.ini_cracks: Dict[int, IniCrack] = {}             # /INICRACK (M143)
        self.prop_rivets: Dict[int, PropRivet] = {}           # /PROP/RIVET (M143)
        self.prop_xelems: Dict[int, PropXelem] = {}           # /PROP/XELEM (M143)
        self.admesh_controls: Dict[int, AdmeshControl] = {}   # /ADMESH/* (M143)
        self.preload_bolts: Dict[int, PreloadBolt] = {}       # /PRELOAD/BOLT (M144)
        self.load_hydros: Dict[int, LoadHydro] = {}           # /LOAD/HYDRO (M144)
        self.func2d_tables: Dict[int, Func2DTable] = {}       # /FUNC_2D (M145)
        self.nonlocal_models: Dict[int, NonlocalModel] = {}   # /NONLOCAL (M145)
        self.fric_orients: Dict[int, FricOrient] = {}         # /FRIC_ORIENT (M145)
        self.ini_sphcels: Dict[int, IniSphCel] = {}           # /INISPHCEL (M145)
        self.implicit_flag: bool = False                      # /IMPLICIT (M145)
        self.visc_plas_models: Dict[int, MaterialViscPlas] = {} # /MAT/VISC_PLAS (M147)
        self.ebcs_pres: Dict[int, EbcsPres] = {}           # /EBCS/PRES (M150)
        self.ebcs_vel: Dict[int, EbcsVel] = {}             # /EBCS/VEL (M150)
        self.ebcs_inlets: Dict[int, EbcsInlet] = {}        # /EBCS/INLET (M150)
        self.ebcs_fluxouts: Dict[int, EbcsFluxout] = {}    # /EBCS/FLUXOUT (M150)
        self.ebcs_gradp0: Dict[int, EbcsGradp0] = {}       # /EBCS/GRADP0 (M150)
        self.ebcs_normv: Dict[int, EbcsNormv] = {}         # /EBCS/NORMV (M150)
        self.ebcs_valves: Dict[int, EbcsValv] = {}         # /EBCS/VALVIN, /EBCS/VALVOUT (M150)
        self.ebcs_monvols: Dict[int, EbcsMonvol] = {}      # /EBCS/MONVOL (M150)
        self.bcs_walls: Dict[int, BcsWall] = {}            # /BCS/WALL (M150)
        self.seatbelt_systems: Dict[int, SeatbeltSystem] = {} # /SEATBELT (M150)
        self.ams_control: Optional[AmsControl] = None      # /AMS (M150)
        self.pblast_loads: Dict[int, PblastLoad] = {}      # /LOAD/PBLAST, /PBLAST (M151)
        self.inivols: Dict[int, Inivol] = {}               # /INIVOL (M151)
        self.inigrav_loads: Dict[int, InigravLoad] = {}    # /INIGRAV (M151)
        self.inistas: Dict[int, Inista] = {}               # /INISTA, /INISTATE (M151)
        self.bem_controls: Dict[int, BemControl] = {}      # /BEM/FLOW, /BEM/DAA (M151)
        self.perturb_controls: Dict[int, PerturbControl] = {} # /PERTURB (M151)
        self.ebcs_inips: Dict[int, EbcsInip] = {}          # /EBCS/INIP (M152)
        self.ebcs_inivs: Dict[int, EbcsIniv] = {}          # /EBCS/INIV (M152)
        self.prop_inject1s: Dict[int, PropInject1] = {}    # /PROP/INJECT1 (M152)
        self.prop_inject2s: Dict[int, PropInject2] = {}    # /PROP/INJECT2 (M152)
        self.prop_joints: Dict[int, PropJoint] = {}        # /PROP/TYPE33 (M152)
        self.prop_torsions: Dict[int, PropTorsion] = {}    # /PROP/TYPE35 (M152)
        self.prop_spring_elas_plas: Dict[int, PropSpringElasPlas] = {} # /PROP/TYPE36 (M152)
        self.prop_spring_beams: Dict[int, PropSpringBeam] = {} # /PROP/TYPE44 (M152)
        self.prop_spotwelds: Dict[int, PropSpotweld] = {}  # /PROP/TYPE45 (M152)
        self.prop_bushings: Dict[int, PropBushing] = {}    # /PROP/TYPE46 (M152)
        self.mat_spr_seatbelts: Dict[int, MaterialSprSeatbelt] = {} # /MAT/LAW114 (M161)
        self.mat_sh_seatbelts: Dict[int, MaterialShSeatbelt] = {}   # /MAT/LAW119 (M161)
        self.mat_tapos: Dict[int, MaterialTapo] = {}                # /MAT/LAW120 (M161)
        self.mat_plas_rates: Dict[int, MaterialPlasRate] = {}       # /MAT/LAW121 (M161)
        self.mat_cdpm2s: Dict[int, MaterialCdpm2] = {}              # /MAT/LAW124 (M161)
        self.fail_hc_dsses: Dict[int, FailHcDsse] = {}              # /FAIL/HC_DSSE (M162)
        self.fail_mullins: Dict[int, FailMullins] = {}              # /FAIL/MULLINS (M162)
        self.fail_snconnects: Dict[int, FailSnconnect] = {}        # /FAIL/SNCONNECT (M162)
        self.fail_spallings: Dict[int, FailSpalling] = {}          # /FAIL/SPALLING (M162)
        self.dfs_detcords: Dict[int, DfsDetcord] = {}              # /DFS/DETCORD (M163)
        self.load_pressures: Dict[int, LoadPressure] = {}          # /LOAD/PRESSURE (M163)
        self.ale_mats: Dict[int, AleMat] = {}                      # /ALE/MAT (M163)
        self.euler_mats: Dict[int, EulerMat] = {}                  # /EULER/MAT (M163)
        self.ebcs_nrfs: Dict[int, EbcsNrf] = {}                    # /EBCS/NRF, /BCS/NRF (M164)
        self.bcs_walls: Dict[int, BcsWall] = {}                    # /BCS/WALL (M164)
        self.slipring_shells: Dict[int, SlipringShell] = {}        # /SLIPRING/SHELL (M164)
        self.sub_laminates: Dict[int, SubLaminate] = {}            # /SUBLAMINATE (M165)
        self.inertia_parts: Dict[int, InertiaPart] = {}            # /INERTIA/PART (M169)
        self.ale_grid_constraints: Dict[int, AleGridConstraint] = {}  # /ALE/GRID/DISP, /ALE/GRID/VEL (M169)
        self.mat_concs: Dict[int, MaterialConc] = {}               # /MAT/LAW24, /MAT/CONC (M170)
        self.mat_barlats: Dict[int, MaterialBarlat] = {}           # /MAT/LAW87, /MAT/BARLAT (M170)
        self.mat_law83s: Dict[int, MaterialLaw83] = {}             # /MAT/LAW83, /MAT/SPR_JOU (M170)
        self.mat_law80s: Dict[int, MaterialLaw80] = {}             # /MAT/LAW80, /MAT/TRANSFO (M170)
        self.mat_law117s: Dict[int, MaterialLaw117] = {}           # /MAT/LAW117, /MAT/COH_MC (M171)
        self.mat_law90s: Dict[int, MaterialLaw90] = {}             # /MAT/LAW90, /MAT/PLAS_TAB (M171)
        self.mat_law33s: Dict[int, MaterialLaw33] = {}             # /MAT/LAW33, /MAT/FOAM_PLAS (M171)
        self.mat_heat_modifiers: Dict[int, MatHeatModifier] = {}   # /MAT/HEAT, /HEAT/MAT (M171)
        self.mat_nonlocal_modifiers: Dict[int, MatNonlocalModifier] = {}  # /MAT/NONLOCAL, /NONLOCAL/MAT (M171)
        self.mat_law66s: Dict[int, MaterialLaw66] = {}             # /MAT/LAW66, /MAT/FOAM_TAB (M172)
        self.mat_law35s: Dict[int, MaterialLaw35] = {}             # /MAT/LAW35, /MAT/FOAM_VISC (M172)
        self.mat_law62s: Dict[int, MaterialLaw62] = {}             # /MAT/LAW62, /MAT/VISC_HYP (M172)
        self.mat_law28s: Dict[int, MaterialLaw28] = {}             # /MAT/LAW28, /MAT/HONEYCOMB (M172)
        self.mat_law44s: Dict[int, MaterialLaw44] = {}             # /MAT/LAW44, /MAT/COWPER_SYMONDS (M172)
        self.mat_law88s: Dict[int, MaterialLaw88] = {}             # /MAT/LAW88, /MAT/HYPER_ELAS (M173)
        self.mat_law92s: Dict[int, MaterialLaw92] = {}             # /MAT/LAW92, /MAT/ARRUDA_BOYCE (M173)
        self.mat_law94s: Dict[int, MaterialLaw94] = {}             # /MAT/LAW94, /MAT/YEOH (M173)
        self.mat_law46s: Dict[int, MaterialLaw46] = {}             # /MAT/LAW46, /MAT/HYD_VISC (M173)
        self.mat_law69s: Dict[int, MaterialLaw69] = {}             # /MAT/LAW69, /MAT/HYP_EXT_COMP (M173)
        self.mat_law124s: Dict[int, MaterialLaw124] = {}           # /MAT/LAW124, /MAT/CDPM2 (M174)
        self.mat_law126s: Dict[int, MaterialLaw126] = {}           # /MAT/LAW126, /MAT/JOHNSON_HOLMQUIST_CONCRETE (M174)
        self.mat_law125s: Dict[int, MaterialLaw125] = {}           # /MAT/LAW125, /MAT/LAMINATED_COMPOSITE (M174)
        self.mat_law127s: Dict[int, MaterialLaw127] = {}           # /MAT/LAW127, /MAT/ENHANCED_COMPOSITE (M174)
        self.mat_law130s: Dict[int, MaterialLaw130] = {}           # /MAT/LAW130, /MAT/MODIFIED_HONEYCOMB (M174)
        self.mat_law128s: Dict[int, MaterialLaw128] = {}           # /MAT/LAW128, /MAT/HILL_VISC_PLAST (M175)
        self.mat_law129s: Dict[int, MaterialLaw129] = {}           # /MAT/LAW129, /MAT/THERM_CREEP (M175)
        self.mat_law123s: Dict[int, MaterialLaw123] = {}           # /MAT/LAW123, /MAT/DAIMLER_PINHO (M175)
        self.mat_law132s: Dict[int, MaterialLaw132] = {}           # /MAT/LAW132, /MAT/DAIMLER_CAMANHO (M175)
        self.mat_law134s: Dict[int, MaterialLaw134] = {}           # /MAT/LAW134, /MAT/VISCOUS_FOAM (M175)
        self.mat_law104s: Dict[int, MaterialLaw104] = {}           # /MAT/LAW104, /MAT/JOHNS_VOCE_DRUCKER (M176)
        self.mat_law105s: Dict[int, MaterialLaw105] = {}           # /MAT/LAW105, /MAT/POWDER_BURN (M176)
        self.mat_law106s: Dict[int, MaterialLaw106] = {}           # /MAT/LAW106, /MAT/JCOOK_ALM (M176)
        self.mat_law107s: Dict[int, MaterialLaw107] = {}           # /MAT/LAW107, /MAT/PAPER_LIGHT (M176)
        self.mat_law110s: Dict[int, MaterialLaw110] = {}           # /MAT/LAW110, /MAT/VEGTER (M176)
        self.mat_law115s: Dict[int, MaterialLaw115] = {}           # /MAT/LAW115, /MAT/DESHPANDE_FLECK (M176)
        self.mat_law109s: Dict[int, MaterialLaw109] = {}           # /MAT/LAW109, /MAT/LAW109 (M177)
        self.mat_law111s: Dict[int, MaterialLaw111] = {}           # /MAT/LAW111, /MAT/MARLOW (M177)
        self.mat_law112s: Dict[int, MaterialLaw112] = {}           # /MAT/LAW112, /MAT/PAPER (M177)
        self.mat_law116s: Dict[int, MaterialLaw116] = {}           # /MAT/LAW116, /MAT/COH_HYST (M177)
        self.mat_law122s: Dict[int, MaterialLaw122] = {}           # /MAT/LAW122, /MAT/MODIFIED_LADEVEZE (M177)
        self.mat_law158s: Dict[int, MaterialLaw158] = {}           # /MAT/LAW158, /MAT/FABR_NL (M177)
        self.bcs_cyclics: Dict[int, BcsCyclic] = {}                 # /BCS/CYCLIC (M178)
        self.pcyl_loads: Dict[int, PcylLoad] = {}                   # /LOAD/PCYL (M178)
        self.damp_vrels: Dict[int, DampVrel] = {}                   # /DAMP/VREL (M179)
        self.fail_syazwans: Dict[int, FailSyazwan] = {}             # /FAIL/SYAZWAN (M179)
        self.mat_law113s: Dict[int, MatLaw113] = {}                 # /MAT/LAW113, /MAT/SPR_BEAM (M179)
        self.mat_law79s: Dict[int, MatLaw79] = {}                   # /MAT/LAW79, /MAT/JOHN_HOLM (M179)
        self.mat_visc_lpronys: Dict[int, MatViscLprony] = {}         # /MAT/VISC_LPRONY, /VISC/LPRONY (M179)
        self.mat_law190s: Dict[int, MatLaw190] = {}                 # /MAT/LAW190, /MAT/FOAM_DUBOIS (M180)
        self.mat_law41s: Dict[int, MatLaw41] = {}                   # /MAT/LAW41, /MAT/LEE_T (M180)
        self.fail_changs: Dict[int, FailChang] = {}                 # /FAIL/CHANG (M180)
        self.prop_tshells: Dict[int, PropType20] = {}               # /PROP/TYPE20, /PROP/TSHELL (M180)
        self.prop_tsh_orths: Dict[int, PropType21] = {}             # /PROP/TYPE21, /PROP/TSH_ORTH (M180)
        self.prop_tsh_comps: Dict[int, PropType22] = {}             # /PROP/TYPE22, /PROP/TSH_COMP (M180)
        self.fail_fabrics: Dict[int, FailFabric] = {}               # /FAIL/FABRIC (M181)
        self.fail_hoffmans: Dict[int, FailHoffman] = {}             # /FAIL/HOFFMAN (M181)
        self.fail_maxstrains: Dict[int, FailMaxStrain] = {}         # /FAIL/MAX_STRAIN (M181)
        self.fail_tsaihills: Dict[int, FailTsaiHill] = {}           # /FAIL/TSAI_HILL (M181)
        self.fail_tsaiwus: Dict[int, FailTsaiWu] = {}               # /FAIL/TSAI_WU (M181)
        self.prop_sol_orths: Dict[int, PropType6] = {}              # /PROP/TYPE6, /PROP/SOL_ORTH (M181)
        self.load_cloads: Dict[int, LoadCload] = {}                 # /LOAD/CLOAD, /CLOAD (M181)
        self.load_ploads: Dict[int, LoadPload] = {}                 # /LOAD/PLOAD, /PLOAD (M181)
        self.mat_law114s: Dict[int, MatLaw114] = {}                 # /MAT/LAW114, /MAT/SPR_SEATBELT (M182)
        self.mat_law117s: Dict[int, MatLaw117] = {}                 # /MAT/LAW117, /MAT/COH_TAB (M182)
        self.mat_law119s: Dict[int, MatLaw119] = {}                 # /MAT/LAW119, /MAT/SH_SEATBELT (M182)
        self.mat_law120s: Dict[int, MatLaw120] = {}                 # /MAT/LAW120, /MAT/TAPO (M182)
        self.mat_law121s: Dict[int, MatLaw121] = {}                 # /MAT/LAW121, /MAT/PLAS_RATE (M182)
        self.prop_spr_tabs: Dict[int, PropType26] = {}              # /PROP/TYPE26, /PROP/SPR_TAB (M182)
        self.prop_spr_bdamps: Dict[int, PropType27] = {}            # /PROP/TYPE27, /PROP/SPR_BDAMP (M182)
        self.mat_law50s: Dict[int, MatLaw50] = {}                   # /MAT/LAW50, /MAT/VISC_HONEY (M183)
        self.mat_law57s: Dict[int, MatLaw57] = {}                   # /MAT/LAW57, /MAT/BARLAT3 (M183)
        self.mat_law87s: Dict[int, MatLaw87] = {}                   # /MAT/LAW87, /MAT/BARLAT_YLD2000 (M183)
        self.mat_law95s: Dict[int, MatLaw95] = {}                   # /MAT/LAW95, /MAT/BERGSTROM_BOYCE (M183)
        self.mat_law163s: Dict[int, MatLaw163] = {}                 # /MAT/LAW163, /MAT/CRUSHABLE_FOAM (M183)
        self.mat_law169s: Dict[int, MatLaw169] = {}                 # /MAT/LAW169, /MAT/ARUP_ADHESIVE (M183)
        self.mat_law49s: Dict[int, MatLaw49] = {}                   # /MAT/LAW49, /MAT/STEINB (M184)
        self.mat_law76s: Dict[int, MatLaw76] = {}                   # /MAT/LAW76, /MAT/SAMP (M184)
        self.prop_type11s: Dict[int, PropType11] = {}               # /PROP/TYPE11, /PROP/SH_SANDW (M184)
        self.prop_type16s: Dict[int, PropType16] = {}               # /PROP/TYPE16, /PROP/SH_FABR (M184)
        self.prop_type17s: Dict[int, PropType17] = {}               # /PROP/TYPE17, /PROP/STACK (M184)
        self.prop_type44s: Dict[int, PropType44] = {}               # /PROP/TYPE44, /PROP/SPR_CRUS (M184)
        self.mat_law60s: Dict[int, MatLaw60] = {}                   # /MAT/LAW60, /MAT/PLAS_T3 (M185)
        self.mat_law63s: Dict[int, MatLaw63] = {}                   # /MAT/LAW63, /MAT/HANSEL (M185)
        self.mat_law48s: Dict[int, MatLaw48] = {}                   # /MAT/LAW48, /MAT/ZHAO (M185)
        self.mat_law26s: Dict[int, MatLaw26] = {}                   # /MAT/LAW26, /MAT/SESAM (M185)
        self.prop_type12s: Dict[int, PropType12] = {}               # /PROP/TYPE12, /PROP/SPR_PUL (M185)
        self.prop_type15s: Dict[int, PropType15] = {}               # /PROP/TYPE15, /PROP/POROUS (M185)
        self.prop_type28s: Dict[int, PropType28] = {}               # /PROP/TYPE28, /PROP/NSTRAND (M185)
        self.mat_law6s: Dict[int, MatLaw6] = {}                     # /MAT/LAW6, /MAT/VISC_FLUID (M186)
        self.mat_law11s: Dict[int, MatLaw11] = {}                   # /MAT/LAW11, /MAT/BOUND (M186)
        self.mat_law77s: Dict[int, MatLaw77] = {}                   # /MAT/LAW77, /MAT/FOAM_AIR (M186)
        self.mat_law151s: Dict[int, MatLaw151] = {}                 # /MAT/LAW151, /MAT/MULTIFLUID (M186)
        self.mat_law187s: Dict[int, MatLaw187] = {}                 # /MAT/LAW187, /MAT/BARLAT20003D (M186)
        self.prop_type33s: Dict[int, PropType33] = {}               # /PROP/TYPE33, /PROP/KJOINT (M186)
        self.prop_type46s: Dict[int, PropType46] = {}               # /PROP/TYPE46, /PROP/SPR_MUSCLE (M186)
        self.prop_type35s: Dict[int, PropType35] = {}               # /PROP/TYPE35, /PROP/STITCH (M186)
        self.mat_law3s: Dict[int, MatLaw3] = {}                     # /MAT/LAW3, /MAT/PLAS_BOST (M187)
        self.mat_law4s: Dict[int, MatLaw4] = {}                     # /MAT/LAW4, /MAT/HYD_JCOOK (M187)
        self.mat_hyd_jcooks = self.mat_law4s
        self.mat_law5s: Dict[int, MatLaw5] = {}                     # /MAT/LAW5, /MAT/JCOOK_TAB (M187)
        self.mat_law10s: Dict[int, MatLaw10] = {}                   # /MAT/LAW10, /MAT/SOIL (M187)
        self.mat_law14s: Dict[int, MatLaw14] = {}                   # /MAT/LAW14, /MAT/CAM_CLAY (M187)
        self.mat_law21s: Dict[int, MatLaw21] = {}                   # /MAT/LAW21, /MAT/DUCKHUB (M187)
        self.mat_law32s: Dict[int, MatLaw32] = {}                   # /MAT/LAW32, /MAT/HILL_TAB (M187)
        self.mat_law37s: Dict[int, MatLaw37] = {}                   # /MAT/LAW37, /MAT/BIQUAD (M187)
        self.prop_type45s: Dict[int, PropType45] = {}               # /PROP/TYPE45, /PROP/KJOINT2 (M187)
        self.prop_type36s: Dict[int, PropType36] = {}               # /PROP/TYPE36, /PROP/PREDIT (M187)
        self.mat_law12s: Dict[int, MatLaw12] = {}                   # /MAT/LAW12, /MAT/3PARBI (M188)
        self.mat_law13s: Dict[int, MatLaw13] = {}                   # /MAT/LAW13, /MAT/HONEYCOMB (M188)
        self.mat_law15s: Dict[int, MatLaw15] = {}                   # /MAT/LAW15, /MAT/CHANG (M188)
        self.mat_law18s: Dict[int, MatLaw18] = {}                   # /MAT/LAW18, /MAT/CONCR_DRA (M188)
        self.mat_law22s: Dict[int, MatLaw22] = {}                   # /MAT/LAW22, /MAT/TSAI_WU (M188)
        self.mat_law25s: Dict[int, MatLaw25] = {}                   # /MAT/LAW25, /MAT/COMP_PLAS (M188)
        self.mat_law28s: Dict[int, MatLaw28] = {}                   # /MAT/LAW28, /MAT/HONEYCOMB_SOL (M188)
        self.prop_type9s: Dict[int, PropType9] = {}                 # /PROP/TYPE9, /PROP/SH_ORTH (M188)
        self.prop_sh_orths = self.prop_type9s
        self.prop_type10s: Dict[int, PropType10] = {}               # /PROP/TYPE10, /PROP/SH_COMP (M188)
        self.prop_sh_comps = self.prop_type10s
        self.prop_type51s: Dict[int, PropType51] = {}               # /PROP/TYPE51, /PROP/SH_COH (M188)
        self.prop_sh_cohs = self.prop_type51s
        self.prop_type5s: Dict[int, PropType5] = {}                 # /PROP/TYPE5, /PROP/RIVET (M188)
        self.prop_rivets = self.prop_type5s
        self.prop_type6s: Dict[int, PropType6] = {}                 # /PROP/TYPE6, /PROP/SOL_ORTH (M188)
        self.prop_sol_orths = self.prop_type6s
        self.prop_type20s: Dict[int, PropType20] = {}               # /PROP/TYPE20, /PROP/TSHELL (M188)
        self.prop_tshells = self.prop_type20s
        # M189 containers and aliases
        self.mat_law52s: Dict[int, MatLaw52] = {}                   # /MAT/LAW52, /MAT/GURSON (M189)
        self.mat_gursons = self.mat_law52s
        self.mat_law16s: Dict[int, MatLaw16] = {}                   # /MAT/LAW16, /MAT/GRAY (M189)
        self.mat_grays = self.mat_law16s
        self.mat_camclays = self.mat_law14s
        self.mat_compsos = self.mat_law14s
        self.mat_law59s: Dict[int, MatLaw59] = {}                   # /MAT/LAW59, /MAT/CONNECT (M189)
        self.mat_connects = self.mat_law59s
        self.mat_law64s: Dict[int, MatLaw64] = {}                   # /MAT/LAW64, /MAT/TRANSFO_MART (M189)
        self.mat_transfo_marts = self.mat_law64s
        self.fail_lad_damas = self.fail_laddamas
        self.fail_ladevezes = self.fail_laddamas
        self.fail_wierzbickis: Dict[int, FailWierzbicki] = {}       # /FAIL/WIERZBICKI, /FAIL/MMC (M189)
        self.fail_mmcs = self.fail_wierzbickis
        self.fail_wilkinss: Dict[int, FailWilkins] = {}             # /FAIL/WILKINS (M189)
        self.fail_spallings: Dict[int, FailSpalling] = {}           # /FAIL/SPALLING, /FAIL/SPALL (M189)
        self.fail_spalls = self.fail_spallings
        self.prop_type14s: Dict[int, PropType14] = {}               # /PROP/TYPE14, /PROP/SOLID (M189)
        self.prop_solids = self.prop_type14s
        self.prop_sol_genes = self.prop_type14s
        self.prop_type8s: Dict[int, PropType8] = {}                 # /PROP/TYPE8, /PROP/SPR_GENE (M189)
        self.prop_spr_genes = self.prop_type8s
        self.prop_type25s: Dict[int, PropType25] = {}               # /PROP/TYPE25, /PROP/SPR_AXI (M189)
        self.prop_spr_axis = self.prop_type25s
        self.prop_type32s: Dict[int, PropType32] = {}               # /PROP/TYPE32, /PROP/SPR_PRE (M189)
        self.prop_spr_pres = self.prop_type32s
        self.prop_type43s: Dict[int, PropType43] = {}               # /PROP/TYPE43, /PROP/CONNECT (M189)
        self.prop_connects = self.prop_type43s
        # M190 containers and aliases
        self.mat_law68s: Dict[int, MatLaw68] = {}                   # /MAT/LAW68, /MAT/COSSER (M190)
        self.mat_cossers = self.mat_law68s
        self.mat_cosserats = self.mat_law68s
        self.mat_law72s: Dict[int, MatLaw72] = {}                   # /MAT/LAW72, /MAT/HILL_MMC (M190)
        self.mat_hill_mmcs = self.mat_law72s
        self.mat_law65s: Dict[int, MatLaw65] = {}                   # /MAT/LAW65, /MAT/ELASTOMER (M190)
        self.mat_elastomers = self.mat_law65s
        self.mat_law58s: Dict[int, MatLaw58] = {}                   # /MAT/LAW58, /MAT/FABR_A (M190)
        self.mat_fabr_as = self.mat_law58s
        self.mat_law20s: Dict[int, MatLaw20] = {}                   # /MAT/LAW20, /MAT/BIMAT (M190)
        self.mat_bimats = self.mat_law20s
        self.mat_law38s: Dict[int, MatLaw38] = {}                   # /MAT/LAW38, /MAT/VISC_TAB (M190)
        self.mat_visc_tabs = self.mat_law38s
        self.mat_law29s: Dict[int, MatLaw29] = {}                   # /MAT/LAW29, /MAT/FEM (M190)
        self.mat_fems = self.mat_law29s
        self.mat29_fems = self.mat_law29s
        self.mat_law34s: Dict[int, MatLaw34] = {}                   # /MAT/LAW34, /MAT/BOLTZMAN (M190)
        self.mat_boltzmans = self.mat_law34s
        self.mat_boltzmanns = self.mat_law34s
        self.mat_law23s: Dict[int, MatLaw23] = {}                   # /MAT/LAW23, /MAT/PLAS_DAMA (M190)
        self.mat_plas_damas = self.mat_law23s
        self.mat_law78s: Dict[int, MatLaw78] = {}                   # /MAT/LAW78 (M190)
        self.fail_hashins: Dict[int, FailHashin] = {}               # /FAIL/HASHIN (M190)
        self.fail_tensstrains: Dict[int, FailTensstrain] = {}       # /FAIL/TENSSTRAIN (M190)
        self.fail_tenstrains = self.fail_tensstrains
        self.fail_energys: Dict[int, FailEnergy] = {}               # /FAIL/ENERGY (M190)
        self.fail_users: Dict[int, FailUser] = {}                   # /FAIL/USER (M190)
        self.prop_type34s: Dict[int, PropType34] = {}               # /PROP/TYPE34, /PROP/SPH (M190)
        self.prop_sphs = self.prop_type34s
        self.prop_prop_sphs = self.prop_type34s
        self.prop_type29s: Dict[int, PropType29] = {}               # /PROP/TYPE29 (M190)
        self.prop_type30s: Dict[int, PropType30] = {}               # /PROP/TYPE30 (M190)
        self.prop_type31s: Dict[int, PropType31] = {}               # /PROP/TYPE31 (M190)
        # M191 containers and aliases
        self.mat_law100s: Dict[int, MatLaw100] = {}                 # /MAT/LAW100, /MAT/SPOTWELD (M191)
        self.mat_spotwelds = self.mat_law100s
        self.mat_structural_adhesives = self.mat_law100s
        self.mat_law97s: Dict[int, MatLaw97] = {}                   # /MAT/LAW97, /MAT/EXPLOSIVE_JWLS (M191)
        self.mat_explosive_jwlss = self.mat_law97s
        self.mat_jwlss = self.mat_law97s
        self.mat_law71s: Dict[int, MatLaw71] = {}                   # /MAT/LAW71, /MAT/SUPER_ELAS (M191)
        self.mat_super_elass = self.mat_law71s
        self.mat_nitinols = self.mat_law71s
        self.mat_law73s: Dict[int, MatLaw73] = {}                   # /MAT/LAW73, /MAT/THERM_HILL (M191)
        self.mat_therm_hills = self.mat_law73s
        self.mat_hill_therms = self.mat_law73s
        self.mat_law84s: Dict[int, MatLaw84] = {}                   # /MAT/LAW84, /MAT/SWIFT_VOCE (M191)
        self.mat_swift_voces = self.mat_law84s
        self.mat_plas_swift_voces = self.mat_law84s
        self.mat_law93s: Dict[int, MatLaw93] = {}                   # /MAT/LAW93, /MAT/ORTH_HILL (M191)
        self.mat_orth_hills = self.mat_law93s
        self.mat_law133s: Dict[int, MatLaw133] = {}                 # /MAT/LAW133, /MAT/GRANULAR (M191)
        self.mat_granulars = self.mat_law133s
        self.mat_law101s: Dict[int, MatLaw101] = {}                 # /MAT/LAW101, /MAT/PLAS_POLY (M191)
        self.mat_plas_polys = self.mat_law101s
        self.mat_law43s: Dict[int, MatLaw43] = {}                   # /MAT/LAW43, /MAT/HILL_TAB (M191)
        self.mat_hill_tabs = self.mat_law43s
        self.fail_lemaitres: Dict[int, FailLemaitre] = {}           # /FAIL/LEMAITRE (M191)
        self.fail_composites: Dict[int, FailComposite] = {}         # /FAIL/COMPOSITE (M191)
        self.fail_tab2s: Dict[int, FailTab2] = {}                   # /FAIL/TAB2, /FAIL/TABULATED2 (M191)
        self.fail_tabulated2s = self.fail_tab2s
        self.fail_alters: Dict[int, FailAlter] = {}                 # /FAIL/ALTER (M191)
        self.fail_visuals: Dict[int, FailVisual] = {}               # /FAIL/VISUAL (M191)
        self.fail_orthstrains: Dict[int, FailOrthstrain] = {}       # /FAIL/ORTHSTRAIN (M191)
        self.ebcs_propellants: Dict[int, EbcsPropellant] = {}       # /EBCS/PROPELLANT (M191)
        self.ebcs_cyclics: Dict[int, EbcsCyclic] = {}               # /EBCS/CYCLIC (M191)
        self.fail_emcs: Dict[int, FailEMC] = {}                     # /FAIL/EMC (M193)
        self.fail_nxts: Dict[int, FailNXT] = {}                     # /FAIL/NXT (M193)
        self.fail_tbutchers: Dict[int, FailTButcher] = {}           # /FAIL/TBUTCHER (M193)
        self.fail_mullins: Dict[int, FailMullins] = {}              # /FAIL/MULLINS, /FAIL/MULLINS_OR (M193)
        self.fail_mullins_ors = self.fail_mullins
        self.fail_cockcrofts: Dict[int, FailCockcroft] = {}         # /FAIL/COCKCROFT (M193)
        self.fail_gene1s: Dict[int, FailGene1] = {}                 # /FAIL/GENE1 (M193)
        self.fail_xfems: Dict[int, FailXFEM] = {}                   # /FAIL/XFEM/... (M193)
        self.mat_law53s: Dict[int, MatLaw53] = {}                   # /MAT/LAW53, /MAT/TSAI_TAB (M193)
        self.mat_tsai_tabs = self.mat_law53s
        self.mat_law54s: Dict[int, MatLaw54] = {}                   # /MAT/LAW54, /MAT/PREDIT (M193)
        self.mat_predits = self.mat_law54s
        self.mat_law74s: Dict[int, MatLaw74] = {}                   # /MAT/LAW74, /MAT/HILL_THERM (M193)
        self.mat_hill_therms = self.mat_law74s
        self.mat_law82s: Dict[int, MatLaw82] = {}                   # /MAT/LAW82, /MAT/OGDEN (M193)
        self.mat_ogdens = self.mat_law82s
        self.prop_int_beams: Dict[int, PropType18] = {}             # /PROP/TYPE18, /PROP/INT_BEAM (M193)
        self.prop_type18s = self.prop_int_beams
        self.def_inter_type11: Optional[DefInterType11] = None      # /DEF_INTER/TYPE11 (M193)
        self.def_inter_type19: Optional[DefInterType19] = None      # /DEF_INTER/TYPE19 (M193)
        self.def_inter_type25: Optional[DefInterType25] = None      # /DEF_INTER/TYPE25 (M193)
        self.state_directives: List[StateDirective] = []            # /STATE/... (M193)
        self.sph_flows: Dict[int, SphFlow] = {}                     # /SPH_FLOW (M194)
        self.mid_directives: Dict[int, MidDirective] = {}           # /MID (M194)
        self.pid_directives: Dict[int, PidDirective] = {}           # /PID (M194)
        self.sphs: Dict[int, SphParticle] = {}                      # /SPHCEL, /SPHCELL (M194)
        self.fail_tab1s: Dict[int, FailTab1] = {}                   # /FAIL/TAB1 (M194)
        self.mat_law40s: Dict[int, MatLaw40] = {}                   # /MAT/LAW40 (M194)
        self.mat_concr_subs = self.mat_law40s
        self.mat_law80s: Dict[int, MaterialLaw80] = {}              # /MAT/LAW80, /MAT/TRANSFO (M170/M194)
        self.mat_law102s: Dict[int, MatLaw102] = {}                 # /MAT/LAW102 (M194)
        self.mat_hill_48s = self.mat_law102s
        self.mat_nlocals: Dict[int, MatNLocal] = {}                 # /MAT/NLOCAL (M194)
        self.prop_spr_pulls: Dict[int, PropType13] = {}             # /PROP/TYPE13, /PROP/SPR_PULL (M194)
        self.prop_type13s = self.prop_spr_pulls
        self.def_inter_type2: Optional[DefInterType2] = None        # /DEF_INTER/TYPE2 (M194)
        self.mat_law103s: Dict[int, MatLaw103] = {}                 # /MAT/LAW103, /MAT/HENSEL_SPITTEL (M195)
        self.mat_hensel_spittels = self.mat_law103s
        self.mat_law108s: Dict[int, MatLaw108] = {}                 # /MAT/LAW108, /MAT/SPR_GENE (M195)
        self.mat_spr_genes = self.mat_law108s
        self.mat_plas_predefs: Dict[int, MatPlasPredef] = {}         # /MAT/PLAS_PREDEF (M195)
        self.mat_dprag2s: Dict[int, MatDPrag2] = {}                 # /MAT/DPRAG2 (M195)
        self.prop_type23s: Dict[int, PropType23] = {}               # /PROP/TYPE23, /PROP/SPR_MAT (M195)
        self.prop_spr_mats = self.prop_type23s
        self.damp_freq_ranges: Dict[int, DampFreqRange] = {}        # /DAMP/FREQUENCY_RANGE (M195)
        self.damp_frequency_ranges = self.damp_freq_ranges
        self.damp_functs: Dict[int, DampFunct] = {}                 # /DAMP/FUNCT (M195)
        self.funct_smooths: Dict[int, FunctSmooth] = {}             # /FUNCT_SMOOTH (M195)
        self.ini_state_tables: Dict[str, IniStateTable] = {}        # /INI... state tables (M195)
        self.checksum_directives: List[str] = []                    # /CHECKSUM directives (M195)

        # M196 additions
        self.mat_law113s: Dict[int, Any] = {}                       # /MAT/LAW113, /MAT/SPR_BEAM (M196)
        self.mat_spr_beams = self.mat_law113s
        self.mat_law95s: Dict[int, Any] = {}                        # /MAT/LAW95, /MAT/SEW (M196)
        self.mat_sews = self.mat_law95s
        self.mat_law48s: Dict[int, Any] = {}                        # /MAT/LAW48, /MAT/ZHAO (M196)
        self.mat_zhaos = self.mat_law48s
        self.mat_law49s: Dict[int, Any] = {}                        # /MAT/LAW49, /MAT/STEINB (M196)
        self.mat_steinbergs = self.mat_law49s
        self.mat_law106s: Dict[int, Any] = {}                       # /MAT/LAW106, /MAT/P_FOAM (M196)
        self.mat_poly_foams = self.mat_law106s
        self.mat_law77s: Dict[int, Any] = {}                        # /MAT/LAW77, /MAT/OGDEN_HYPO (M196)
        self.mat_ogden_hypos = self.mat_law77s
        self.mat_law63s: Dict[int, Any] = {}                        # /MAT/LAW63, /MAT/SOIL_DISC (M196)
        self.mat_soil_discs = self.mat_law63s
        self.mat_law92s: Dict[int, Any] = {}                        # /MAT/LAW92, /MAT/HILL_ORTH (M196)
        self.mat_hill_orths = self.mat_law92s
        self.prop_type26s: Dict[int, Any] = {}                      # /PROP/TYPE26, /PROP/SPR_TAB (M196)
        self.prop_spr_tabs = self.prop_type26s
        self.damp_inters: Dict[int, Any] = {}                       # /DAMP/INTER (M196)
        self.inter_type18s: Dict[int, Any] = {}                     # /INTER/TYPE18 (M196)
        self.frame_nods: Dict[int, Any] = {}                        # /FRAME/NOD, /FRAME/NODE (M196)
        self.ini_spr_tables: Dict[int, Any] = {}                    # /INISPR, /INISPRI (M196)
        self.table_blocks: Dict[int, Any] = {}                      # /TABLE, /TABLE/0, /TABLE/1 (M196)
        self.merge_rbodies: Dict[int, Any] = {}                     # /MERGE/RBODY (M196)

        self.th_requests: List[THRequest] = []

        self.title: str = "pyradioss model"
        self.subtitles: List[str] = []                              # /SUBTITLE (M198)
        self.subtitle: str = ""                                     # /SUBTITLE (M198)
        self.cnodes: Dict[int, Any] = {}                            # /CNODE (M198)
        self.upbeams: Dict[int, Any] = {}                           # /UPBEAM, /UPBEAM/INT_BEAM (M198)
        self.relax: bool = False                                    # /RELAX (M198)
        self.relax_systems: Dict[int, Any] = {}                     # /RELAX, /RELAX/SYSTEM, /RELAX/DYNA (M198)
        self.centris: Dict[int, Any] = {}                           # /CENTRI (M198)
        self.monvol_comms: Dict[int, Any] = {}                      # /MONVOL/COMM, /MONVOL/COMMUNICATION (M198)
        self.monvol_communications = self.monvol_comms
        self.dttsh: bool = False                                    # /DTTSH, /DT/TSH (M198)
        self.transform_matrices: Dict[int, Any] = {}                # /TRANSFORM/MATRIX, /MATRIX (M199)
        self.inter_type10s: Dict[int, Any] = {}                     # /INTER/TYPE10 (M199)
        self.inter_type12s: Dict[int, Any] = {}                     # /INTER/TYPE12 (M199)
        self.mat_law51s: Dict[int, Any] = {}                        # /MAT/LAW51, /MAT/DRUCKER_PRAGER, /MAT/MULTIFLUID (M199)
        self.mat_multifluids = self.mat_law51s
        self.mat_drucker_pragers = self.mat_law51s
        # M200 attributes
        self.bcs_nrfs: Dict[int, Any] = self.bcs_nrf                # /BCS/NRF (M200)
        self.ebcs_cyclics: Dict[int, Any] = {}                      # /EBCS/CYCLIC (M200)
        self.ebcs_propellants: Dict[int, Any] = {}                  # /EBCS/PROPELLANT (M200)
        self.detpoint_nodes: Dict[int, Any] = {}                    # /DFS/DETPOINT/NODE, /DETPOINT/NODE (M200)
        self.detpoint_sets: Dict[int, Any] = {}                     # /DFS/DETPOINT/SET, /DETPOINT/SET (M200)
        self.detpoints = self.detpoint_nodes
        self.dtix: Optional[Any] = None                             # /DTIX, /ENG/DTIX (M200)
        self.th_title: bool = False                                 # /TH/TITLE (M200)
        self.th_title_enabled: bool = False                         # /TH/TITLE (M200)
        self.parith: str = "ON"                                     # /PARITH (M200)
        self.dynain_shell_aux: str = ""                             # /DYNAIN/SHELL/AUX/FULL (M200)
        self.dynain_shell_stres: str = ""                           # /DYNAIN/SHELL/STRES/FULL (M200)
        self.dynain_shell_strain: str = ""                          # /DYNAIN/SHELL/STRAIN/FULL (M200)

        # M201 Entities
        self.th_subs: Dict[int, Any] = {}                            # /TH/SUBS (M201)
        self.th_part_groups: Dict[int, Any] = {}                     # /THPART/GR... (M201)
        self.dfs_wav_shas: Dict[int, Any] = {}                       # /DFS/WAV_SHA, /WAVE (M201)
        self.sensors_dist_surf: Dict[int, Any] = {}                  # /SENSOR/DIST_SURF (M201)
        self.sensors_sens_and_or: Dict[int, Any] = {}                # /SENSOR/SENS_AND_OR (M201)
        self.props_pcompp: Dict[int, Any] = {}                       # /PROP/PCOMPP (M201)
        self.props_type51: Dict[int, Any] = {}                       # /PROP/TYPE51, /PROP/P51 (M201)
        self.props_p51 = self.props_type51

        # M202 Entities
        self.heat_fluxes: Dict[int, Any] = {}                        # /HEAT/FLUX, /FLUX (M202)
        self.heat_convecs: Dict[int, Any] = {}                       # /HEAT/CONVEC, /CONVEC (M202)
        self.heat_radiations: Dict[int, Any] = {}                    # /HEAT/RADIATION, /RADIATION (M202)
        self.surfs_surf: Dict[int, Any] = {}                         # /SURF/SURF, /SURFSURF (M202)
        self.surf_surfs = self.surfs_surf
        self.bcs_lagmuls: Dict[int, Any] = {}                        # /BCS/LAGMUL (M202)
        self.spcnds: Dict[int, Any] = {}                             # /SPCND (M202)
        self.ddws: Dict[int, Any] = {}                               # /DDW (M202)
        self.ddw_points: Dict[int, Any] = {}                         # /DDW/POINT (M202)
        self.stampings: List[Any] = []                               # /STAMPING, /STAMP (M202)
        self.sensors_work: Dict[int, Any] = {}                       # /SENSOR/WORK (M201/M202)

        # M203 Entities
        self.lagmul_gears = self.gears
        self.lagmul_racks = self.racks
        self.lagmul_diffs = self.diffs
        self.inter_type26s = self.guided_cables
        self.sensors_python: Dict[int, Any] = {}                     # /SENSOR/PYTHON (M203)
        self.transforms_pos = self.transform_positions               # /TRANSFORM/POS, /POS (M203)
        self.pos_transforms = self.transform_positions
        self.props_inject1 = self.prop_inject1s
        self.props_type15 = self.prop_inject1s
        self.props_inject2 = self.prop_inject2s
        self.props_type16 = self.prop_inject2s
        self.checksums: List[Any] = []                               # /CHECKSUM/START, /CHECKSUM/END (M203)
        self.fails_johnson: Dict[int, Any] = {}                      # /FAIL/JOHNSON (M203)
        self.fail_johnsons = self.fails_johnson
        self.fails_biquad: Dict[int, Any] = {}                       # /FAIL/BIQUAD (M203)
        self.fail_biquads = self.fails_biquad
        self.fails_fld: Dict[int, Any] = {}                          # /FAIL/FLD (M203)
        self.fail_flds = self.fails_fld
        self.fails_connect: Dict[int, Any] = {}                      # /FAIL/CONNECT (M203)
        self.fail_connects = self.fails_connect
        self.fails_fractal_dmg: Dict[int, Any] = {}                  # /FAIL/FRACTAL_DMG (M203)
        self.fail_fractal_dmgs = self.fails_fractal_dmg
        self.fails_orthenerg: Dict[int, Any] = {}                    # /FAIL/ORTHENERG (M203)
        self.fail_orthenergs = self.fails_orthenerg
        self.admesh_sets: Dict[int, Any] = {}                        # /ADMESH/SET (M203)
        self.gauge_sphs: Dict[int, Any] = {}                         # /GAUGE/SPH (M203)
        self.boxes_cyl = self.boxes
        self.boxes_rect = self.boxes
        self.boxes_sphere = self.boxes
        self.pressure_loads = self.load_pressures

        # M204 Entities
        self.heat_solvers: Dict[int, Any] = {}                       # /HEAT/SOLVER, /HEAT/GLOBAL (M204)
        self.heat_globals = self.heat_solvers
        self.xfem_controls: Dict[int, Any] = {}                      # /XFEM (M204)
        self.sensors_airbag: Dict[int, Any] = {}                     # /SENSOR/AIRBAG, /SENSOR/MONVOL (M204)
        self.sensors_shell: Dict[int, Any] = {}                      # /SENSOR/SHELL (M204)
        self.sensors_solid: Dict[int, Any] = {}                      # /SENSOR/SOLID (M204)
        self.sensors_sph: Dict[int, Any] = {}                        # /SENSOR/SPH (M204)
        self.initemps: Dict[int, Any] = {}                           # /INITEMP dict map (M204)
        self.imptemps: Dict[int, Any] = {}                           # /IMPTEMP dict map (M204)
        self.initial_temperatures = self.initemps
        self.imposed_temperatures = self.imptemps
        self.load_heats = self.load_therms
        self.heat_loads = self.load_therms
        self.heat_mats = self.mat_heat_modifiers

        # M205 Entities
        self.flow_boundaries: Dict[int, Any] = {}                    # /FLOW, /ALE/FLOW (M205)
        self.flows = self.flow_boundaries
        self.heat_rad_cavs: Dict[int, Any] = {}                      # /HEAT/RAD_CAV (M205)
        self.props_type19: Dict[int, Any] = {}                       # /PROP/TYPE19, /PROP/SPR_TORS (M205)
        self.props_spr_tors = self.props_type19
        self.props_type20: Dict[int, Any] = {}                       # /PROP/TYPE20, /PROP/SPR_BEND (M205)
        self.props_spr_bend = self.props_type20
        self.pblasts = self.pblast_loads
        self.load_pblasts = self.pblast_loads
        self.pfluids = self.pfluid_loads
        self.load_pfluids = self.pfluid_loads
        self.lasers = self.laser_loads
        self.load_lasers = self.laser_loads

        # M206 Entities
        self.props_type21: Dict[int, Any] = {}                       # /PROP/TYPE21, /PROP/TSH_ORTH (M206)
        self.props_tsh_orth = self.props_type21
        self.props_type22: Dict[int, Any] = {}                       # /PROP/TYPE22, /PROP/TSH_COMP (M206)
        self.props_tsh_comp = self.props_type22
        self.sensors_geom: Dict[int, Any] = {}                       # /SENSOR/GEOM (M206)
        self.sensors_rel: Dict[int, Any] = {}                        # /SENSOR/REL (M206)
        self.ale_zero_pressure: bool = False                         # /ALE/ZERO_PRESSURE (M206)
        self.ale_zero_vel: bool = False                              # /ALE/ZERO_VEL (M206)
        self.inter_type25s: Dict[int, Any] = {}                      # /INTER/TYPE25, /INTER/TIED_BREAK (M206)
        self.inter_tied_breaks = self.inter_type25s

        # M207 Entities
        self.props_type47: Dict[int, Any] = {}                       # /PROP/TYPE47, /PROP/SPR_PULL (M207)
        self.props_spr_pull = self.props_type47
        self.props_type48: Dict[int, Any] = {}                       # /PROP/TYPE48, /PROP/SPR_PUSH (M207)
        self.props_spr_push = self.props_type48
        self.sensors_ratio: Dict[int, Any] = {}                      # /SENSOR/RATIO, /SENSOR/ENERGY_RATIO (M207)
        self.sensors_energy_ratio = self.sensors_ratio
        self.sensors_shear_lock: Dict[int, Any] = {}                  # /SENSOR/SHEAR_LOCK (M207)
        self.dt_inter_del: bool = False                              # /DT/INTER/DEL (M207)
        self.dt_inter_del_val: float = 0.0
        self.dt_noda_cfl: float = 0.9                                # /DT/NODA/CFL (M207)

        # M208 Entities
        self.ball_joints: Dict[int, Any] = {}                        # /LAGMUL/BALL_JOINT, /BALL_JOINT (M208)
        self.pin_joints: Dict[int, Any] = {}                         # /LAGMUL/PIN_JOINT, /PIN_JOINT (M208)
        self.props_type54: Dict[int, Any] = {}                       # /PROP/TYPE54, /PROP/TSH_P54 (M208)
        self.props_tsh_p54 = self.props_type54
        self.damp_alpha: float = 0.0                                 # /ENG/DAMP, /DAMP (M208)
        self.damp_beta: float = 0.0
        self.damp_tstart: float = 0.0
        self.damp_tstop: float = 1.0e30

        # M209 Entities
        self.slider_joints: Dict[int, Any] = {}                      # /LAGMUL/SLIDER, /SLIDER (M209)
        self.cyl_joints: Dict[int, Any] = {}                         # /LAGMUL/CYL_JOINT, /CYL_JOINT (M209)
        self.damp_parts: Dict[int, Any] = {}                         # /DAMP/PART (M209)
        self.sub_cycle_enabled: bool = False                         # /ENG/SUB_CYCLE, /SUB_CYCLE (M209)
        self.sub_cycle_ratio: int = 1
        self.sub_cycle_inter: bool = False

        # M210 Entities
        self.planar_joints: Dict[int, Any] = {}                      # /LAGMUL/PLANAR, /PLANAR (M210)
        self.cardan_joints: Dict[int, Any] = {}                      # /LAGMUL/CARDAN, /CARDAN (M210)
        self.fail_tbids: Dict[int, Any] = {}                         # /FAIL/TBID (M210)
        self.anim_dt: float = 0.0                                    # /ANIM/DT, /ENG/ANIM/DT (M210)
        self.anim_tstart: float = 0.0
        self.anim_sens_id: int = 0

        # M211 Entities
        self.rigid_joints: Dict[int, Any] = {}                       # /LAGMUL/RIGID, /RIGID_JOINT (M211)
        self.screw_joints: Dict[int, Any] = {}                       # /LAGMUL/SCREW, /SCREW (M211)
        self.fail_sn_curves: Dict[int, Any] = {}                     # /FAIL/SN_CURVE (M211)
        self.run_tstop: float = 0.0                                  # /RUN, /ENG/RUN (M211)
        self.run_title: str = ""
        self.run_cycle_max: int = 0

        # M212 Entities
        self.cv_joints: Dict[int, Any] = {}                          # /LAGMUL/CV_JOINT, /CV_JOINT (M212)
        self.inline_joints: Dict[int, Any] = {}                      # /LAGMUL/INLINE, /INLINE (M212)
        self.fail_hoops: Dict[int, Any] = {}                         # /FAIL/HOOP (M212)
        self.stop_sens_id: int = 0                                   # /STOP, /ENG/STOP (M212)
        self.stop_cycle_max: int = 0
        self.stop_time_max: float = 0.0
        self.tfile_dt: float = 0.0                                   # /TFILE, /ENG/TFILE (M212)
        self.tfile_sens_id: int = 0

        # M213 Entities
        self.parallel_joints: Dict[int, Any] = {}                   # /LAGMUL/PARALLEL, /PARALLEL (M213)
        self.perpendicular_joints: Dict[int, Any] = {}               # /LAGMUL/PERPENDICULAR, /PERPENDICULAR (M213)
        self.fail_spalling_cuts: Dict[int, Any] = {}                 # /FAIL/SPALLING_CUT (M213)
        self.rfile_dt: float = 0.0                                   # /RFILE, /ENG/RFILE (M213)
        self.rfile_ncycle: int = 0
        self.rfile_sens_id: int = 0

        # M214 Entities
        self.gimbal_joints: Dict[int, Any] = {}                      # /LAGMUL/GIMBAL, /GIMBAL (M214)
        self.distance_joints: Dict[int, Any] = {}                    # /LAGMUL/DISTANCE, /DISTANCE (M214)
        self.fail_voids: Dict[int, Any] = {}                         # /FAIL/VOIDS (M214)
        self.print_ncycle: int = 0                                   # /PRINT, /ENG/PRINT (M214)
        self.print_dt: float = 0.0
        self.print_sens_id: int = 0

        # M215 Entities
        self.fail_hcs: Dict[int, Any] = {}                           # /FAIL/HC, /FAIL/HOSFORD_COULOMB (M215)
        self.parith_enabled: bool = False                            # /PARITH, /ENG/PARITH (M215)
        self.eng_version: float = 0.0                                # /VERS, /ENG/VERS (M215)
        self.rbody_stops: Dict[int, Any] = {}                        # /RBODY/STOP, /ENG/RBODY/STOP (M215)

        # M216 Entities
        self.fail_lad_evrs: Dict[int, Any] = {}                      # /FAIL/LAD_EVR, /FAIL/LADEVEZE_EVR (M216)
        self.monitors: Dict[int, Any] = {}                           # /MONITOR, /ENG/MONITOR (M216)
        self.nois_freq: float = 0.0                                  # /NOIS, /ENG/NOIS (M216)
        self.nois_type: int = 0
        self.sensor_gaps: Dict[int, Any] = {}                        # /SENSOR/TIME_GAP, /SENSOR/GAP (M216)

        # M217 Entities
        self.slot_joints: Dict[int, Any] = {}                        # /LAGMUL/SLOT, /SLOT (M217)
        self.fail_orthos: Dict[int, Any] = {}                        # /FAIL/ORTHO, /FAIL/ORTHOTROPIC (M217)
        self.eng_fxfreqs: Dict[int, Any] = {}                        # /FXFREQ, /ENG/FXFREQ (M217)
        self.sensor_energy_ratios: Dict[int, Any] = {}               # /SENSOR/ENERGY_RATIO, /SENSOR/ENG_RATIO (M217)

        # M218 Entities
        self.fail_cohesives: Dict[int, Any] = {}                     # /FAIL/COHESIVE, /FAIL/COH (M218)
        self.eng_tracks: Dict[int, Any] = {}                         # /TRACK, /ENG/TRACK (M218)
        self.sensor_cross_sections: Dict[int, Any] = {}              # /SENSOR/CROSSSECTION, /SENSOR/SEC_FORCE (M218)

        # M219 Entities
        self.fail_maxstresses: Dict[int, Any] = {}                   # /FAIL/MAX_STRESS, /FAIL/MAXSTRESS (M219)
        self.eng_helms: Dict[int, Any] = {}                          # /HELM, /ENG/HELM (M219)
        self.sensor_ruptures: Dict[int, Any] = {}                    # /SENSOR/RUPT, /SENSOR/SHELL_FAIL (M219)

        # M220 Entities
        self.fail_snows: Dict[int, Any] = {}                         # /FAIL/SNOW (M220)
        self.eng_truncs: Dict[int, Any] = {}                         # /TRUNC, /ENG/TRUNC (M220)
        self.sensor_shears: Dict[int, Any] = {}                      # /SENSOR/SHEAR, /SENSOR/SHEAR_STRESS (M220)

        # M221 Entities
        self.fail_viscos: Dict[int, Any] = {}                        # /FAIL/VISCO (M221)
        self.eng_masses: Dict[int, Any] = {}                         # /MASS, /ENG/MASS (M221)
        self.sensor_pressures: Dict[int, Any] = {}                   # /SENSOR/PRESSURE, /SENSOR/PRESS (M221)

        # M222 Entities
        self.fail_bammans: Dict[int, Any] = {}                       # /FAIL/BAMMAN (M222)
        self.eng_energies: Dict[int, Any] = {}                       # /ENERGY, /ENG/ENERGY (M222)
        self.sensor_mass_ratios: Dict[int, Any] = {}                 # /SENSOR/MASS, /SENSOR/MASS_RATIO (M222)

        # M223 Entities
        self.fail_weibulls: Dict[int, Any] = {}                      # /FAIL/WEIBULL (M223)
        self.eng_moments: Dict[int, Any] = {}                        # /MOMENT, /ENG/MOMENT (M223)
        self.sensor_energy_errors: Dict[int, Any] = {}               # /SENSOR/ENERGY_ERROR (M223)

        # M224 Entities
        self.fail_pus: Dict[int, Any] = {}                           # /FAIL/PU (M224)
        self.eng_states: Dict[int, Any] = {}                         # /STATE, /ENG/STATE (M224)
        self.sensor_work_ratios: Dict[int, Any] = {}                 # /SENSOR/WORK_RATIO (M224)

        # M225 Entities
        self.fail_griffiths: Dict[int, Any] = {}                     # /FAIL/GRIFFITH (M225)
        self.eng_surfs: Dict[int, Any] = {}                          # /SURF, /ENG/SURF (M225)
        self.sensor_springs: Dict[int, Any] = {}                     # /SENSOR/SPRING (M225)

        # M226 Entities
        self.fail_druckers: Dict[int, Any] = {}                      # /FAIL/DRUCKER (M226)
        self.eng_ales: Dict[int, Any] = {}                           # /ALE, /ENG/ALE (M226)
        self.sensor_shell_strains: Dict[int, Any] = {}               # /SENSOR/SHELL_STRAIN (M226)

        # M227 Entities
        self.fail_woods: Dict[int, Any] = {}                         # /FAIL/WOOD (M227)
        self.eng_sh_thicks: Dict[int, Any] = {}                      # /SH_THICK, /ENG/SH_THICK (M227)
        self.sensor_solid_strains: Dict[int, Any] = {}               # /SENSOR/SOLID_STRAIN (M227)

        # M228 Entities
        self.fail_hills: Dict[int, Any] = {}                         # /FAIL/HILL (M228)
        self.eng_geos: Dict[int, Any] = {}                           # /GEO, /ENG/GEO (M228)
        self.sensor_beam_strains: Dict[int, Any] = {}                # /SENSOR/BEAM_STRAIN (M228)

        # M229 Entities
        self.fail_nortons: Dict[int, Any] = {}                       # /FAIL/NORTON (M229)
        self.eng_tenses: Dict[int, Any] = {}                         # /TENS, /ENG/TENS (M229)
        self.sensor_truss_strains: Dict[int, Any] = {}               # /SENSOR/TRUSS_STRAIN (M229)

        # M230 Entities
        self.fail_mohrs: Dict[int, Any] = {}                         # /FAIL/MOHR (M230)
        self.eng_stresses: Dict[int, Any] = {}                       # /STRESS, /ENG/STRESS (M230)
        self.sensor_shell_forces: Dict[int, Any] = {}                # /SENSOR/SHELL_FORCE (M230)

        # M231 Entities
        self.fail_lusases: Dict[int, Any] = {}                       # /FAIL/LUSAS (M231)
        self.eng_strains: Dict[int, Any] = {}                        # /STRAIN, /ENG/STRAIN (M231)
        self.sensor_solid_forces: Dict[int, Any] = {}                # /SENSOR/SOLID_FORCE (M231)

        # M232 Entities
        self.fail_gtns: Dict[int, Any] = {}                          # /FAIL/GTN (M232)
        self.eng_plastics: Dict[int, Any] = {}                       # /PLASTIC, /ENG/PLASTIC (M232)
        self.sensor_beam_forces: Dict[int, Any] = {}                 # /SENSOR/BEAM_FORCE (M232)

        # M233 Entities
        self.fail_tab3s: Dict[int, Any] = {}                         # /FAIL/TAB3 (M233)
        self.eng_velocities: Dict[int, Any] = {}                     # /VELOCITY, /ENG/VELOCITY (M233)
        self.sensor_truss_forces: Dict[int, Any] = {}                # /SENSOR/TRUSS_FORCE (M233)

        # M234 Entities
        self.fail_chaboches: Dict[int, Any] = {}                     # /FAIL/CHABOCHE (M234)
        self.eng_accels: Dict[int, Any] = {}                         # /ACCEL, /ENG/ACCEL (M234)
        self.sensor_spring_energies: Dict[int, Any] = {}             # /SENSOR/SPRING_ENERGY (M234)

        # M235 Entities
        self.fail_gursons: Dict[int, Any] = {}                       # /FAIL/GURSON (M235)
        self.eng_disps: Dict[int, Any] = {}                          # /DISP, /ENG/DISP (M235)
        self.sensor_spring_defls: Dict[int, Any] = {}                # /SENSOR/SPRING_DEFL (M235)

        # M236 Entities
        self.fail_tvergaards: Dict[int, Any] = {}                    # /FAIL/TVERGAARD (M236)
        self.eng_rotcs: Dict[int, Any] = {}                          # /ROTC, /ENG/ROTC (M236)
        self.sensor_spring_rots: Dict[int, Any] = {}                 # /SENSOR/SPRING_ROT (M236)

        # M237 Entities
        self.fail_henckys: Dict[int, Any] = {}                       # /FAIL/HENCKY (M237)
        self.eng_rotvs: Dict[int, Any] = {}                          # /ROTV, /ENG/ROTV (M237)
        self.sensor_spring_rotvs: Dict[int, Any] = {}                # /SENSOR/SPRING_ROTV (M237)

        # M238 Entities
        self.fail_energy_densitys: Dict[int, Any] = {}               # /FAIL/ENERGY_DENSITY (M238)
        self.eng_rotas: Dict[int, Any] = {}                          # /ROTA, /ENG/ROTA (M238)
        self.sensor_spring_rotas: Dict[int, Any] = {}                # /SENSOR/SPRING_ROTA (M238)

        # M239 Entities
        self.fail_energy_ratios: Dict[int, Any] = {}                 # /FAIL/ENERGY_RATIO (M239)
        self.eng_forces: Dict[int, Any] = {}                         # /FORCE, /ENG/FORCE (M239)
        self.sensor_spring_axials: Dict[int, Any] = {}               # /SENSOR/SPRING_AXIAL (M239)

        # M240 Entities
        self.fail_rice_traceys: Dict[int, Any] = {}                  # /FAIL/RICE_TRACEY (M240)
        self.eng_volumes: Dict[int, Any] = {}                        # /VOLUME, /ENG/VOLUME (M240)
        self.sensor_spring_shears: Dict[int, Any] = {}               # /SENSOR/SPRING_SHEAR (M240)

        # M241 Entities
        self.fail_bao_wierzbickis: Dict[int, Any] = {}               # /FAIL/BAO_WIERZBICKI (M241)
        self.eng_densities: Dict[int, Any] = {}                      # /DENSITY, /ENG/DENSITY (M241)
        self.sensor_spring_bends: Dict[int, Any] = {}                # /SENSOR/SPRING_BEND (M241)

        # M242 Entities
        self.fail_lou_huhns: Dict[int, Any] = {}                     # /FAIL/LOU_HUHN (M242)
        self.eng_internal_energies: Dict[int, Any] = {}              # /INTERNAL_ENERGY, /ENG/INTERNAL_ENERGY (M242)
        self.sensor_spring_torsions: Dict[int, Any] = {}             # /SENSOR/SPRING_TORSION (M242)

        # M243 Entities
        self.fail_hollomons: Dict[int, Any] = {}                     # /FAIL/HOLLOMON (M243)
        self.eng_kinetic_energies: Dict[int, Any] = {}               # /KINETIC_ENERGY, /ENG/KINETIC_ENERGY (M243)
        self.sensor_spring_strain_energies: Dict[int, Any] = {}      # /SENSOR/SPRING_STRAIN_ENERGY (M243)

        # M244 Entities
        self.fail_swifts: Dict[int, Any] = {}                        # /FAIL/SWIFT (M244)
        self.eng_total_energies: Dict[int, Any] = {}                 # /TOTAL_ENERGY, /ENG/TOTAL_ENERGY (M244)
        self.sensor_spring_kinetic_energies: Dict[int, Any] = {}     # /SENSOR/SPRING_KINETIC_ENERGY (M244)

        # M245 Entities
        self.fail_ludwiks: Dict[int, Any] = {}                       # /FAIL/LUDWIK (M245)
        self.eng_hourglass_energies: Dict[int, Any] = {}             # /HOURGLASS_ENERGY, /ENG/HOURGLASS_ENERGY (M245)
        self.sensor_spring_hourglass_energies: Dict[int, Any] = {}   # /SENSOR/SPRING_HOURGLASS_ENERGY (M245)

        # M246 Entities
        self.fail_voces: Dict[int, Any] = {}                         # /FAIL/VOCE (M246)
        self.eng_contact_energies: Dict[int, Any] = {}               # /CONTACT_ENERGY, /ENG/CONTACT_ENERGY (M246)
        self.sensor_spring_contact_energies: Dict[int, Any] = {}     # /SENSOR/SPRING_CONTACT_ENERGY (M246)

        # M247 Entities
        self.fail_ghoshs: Dict[int, Any] = {}                        # /FAIL/GHOSH (M247)
        self.eng_numerical_dissipations: Dict[int, Any] = {}         # /NUMERICAL_DISSIPATION, /ENG/NUMERICAL_DISSIPATION (M247)
        self.sensor_spring_numerical_dissipations: Dict[int, Any] = {} # /SENSOR/SPRING_NUMERICAL_DISSIPATION (M247)

        # M248 Entities
        self.fail_swift_voces: Dict[int, Any] = {}                   # /FAIL/SWIFT_VOCE (M248)
        self.eng_ext_works: Dict[int, Any] = {}                      # /EXT_WORK, /ENG/EXT_WORK (M248)
        self.sensor_spring_ext_works: Dict[int, Any] = {}            # /SENSOR/SPRING_EXT_WORK (M248)

        # M249 Entities
        self.fail_bonoras: Dict[int, Any] = {}                       # /FAIL/BONORA (M249)
        self.eng_tot_energies: Dict[int, Any] = {}                   # /TOT_ENERGY, /ENG/TOT_ENERGY (M249)
        self.sensor_spring_tot_energies: Dict[int, Any] = {}         # /SENSOR/SPRING_TOT_ENERGY (M249)

        # M250 Entities
        self.fail_ritchies: Dict[int, Any] = {}                      # /FAIL/RITCHIE_KNOTT_RICE (M250)
        self.eng_mass_energies: Dict[int, Any] = {}                  # /MASS_ENERGY, /ENG/MASS_ENERGY (M250)
        self.rack_pinion_joints: Dict[int, Any] = {}                 # /RACK_PINION, /LAGMUL/RACK_PINION (M250)
        self.sensor_spring_mass_energies: Dict[int, Any] = {}        # /SENSOR/SPRING_MASS_ENERGY (M250)

        # M251 Entities
        self.fail_gologanus: Dict[int, Any] = {}                     # /FAIL/GOLOGANU (M251)
        self.eng_mass_changes: Dict[int, Any] = {}                   # /MASS_CHANGE, /ENG/MASS_CHANGE (M251)
        self.belt_pulley_joints: Dict[int, Any] = {}                 # /BELT_PULLEY, /LAGMUL/BELT_PULLEY (M251)
        self.sensor_spring_mass_changes: Dict[int, Any] = {}         # /SENSOR/SPRING_MASS_CHANGE (M251)

        # M252 Entities
        self.fail_rousseliers: Dict[int, Any] = {}                   # /FAIL/ROUSSELIER (M252)
        self.eng_pressures: Dict[int, Any] = {}                      # /PRESSURE, /ENG/PRESSURE (M252)
        self.oldham_joints: Dict[int, Any] = {}                      # /OLDHAM, /LAGMUL/OLDHAM (M252)
        self.sensor_spring_pressures: Dict[int, Any] = {}            # /SENSOR/SPRING_PRESSURE (M252)

        # M253 Entities
        self.fail_hockett_sherbys: Dict[int, Any] = {}               # /FAIL/HOCKETT_SHERBY (M253)
        self.eng_temperatures: Dict[int, Any] = {}                   # /ENG/TEMPERATURE, /ENG/TEMP (M253)
        self.tripod_joints: Dict[int, Any] = {}                      # /TRIPOD, /LAGMUL/TRIPOD (M253)
        self.sensor_spring_temperatures: Dict[int, Any] = {}         # /SENSOR/SPRING_TEMPERATURE (M253)

        # M254 Entities
        self.fail_kim_baeks: Dict[int, Any] = {}                     # /FAIL/KIM_BAEK (M254)
        self.eng_volumes: Dict[int, Any] = {}                        # /ENG/VOLUME, /ENG/VOL (M254)
        self.bevel_gear_joints: Dict[int, Any] = {}                  # /BEVEL_GEAR, /LAGMUL/BEVEL_GEAR (M254)
        self.sensor_spring_volumes: Dict[int, Any] = {}              # /SENSOR/SPRING_VOLUME (M254)

        # M255 Entities
        self.fail_bai_wierzbickis: Dict[int, Any] = {}               # /FAIL/BAI_WIERZBICKI (M255)
        self.eng_densities: Dict[int, Any] = {}                      # /ENG/DENSITY, /ENG/RHO (M255)
        self.worm_gear_joints: Dict[int, Any] = {}                   # /WORM_GEAR, /LAGMUL/WORM_GEAR (M255)
        self.sensor_spring_densities: Dict[int, Any] = {}            # /SENSOR/SPRING_DENSITY (M255)

        # M256 Entities
        self.fail_jh2s: Dict[int, Any] = {}                          # /FAIL/JH2 (M256)
        self.eng_entropies: Dict[int, Any] = {}                      # /ENG/ENTROPY (M256)
        self.hypoid_gear_joints: Dict[int, Any] = {}                 # /HYPOID_GEAR, /LAGMUL/HYPOID_GEAR (M256)
        self.sensor_spring_entropies: Dict[int, Any] = {}            # /SENSOR/SPRING_ENTROPY (M256)

        # M257 Entities
        self.fail_rhts: Dict[int, Any] = {}                          # /FAIL/RHT (M257)
        self.eng_sound_speeds: Dict[int, Any] = {}                   # /ENG/SOUND_SPEED (M257)
        self.epicyclic_gear_joints: Dict[int, Any] = {}              # /EPICYCLIC_GEAR, /LAGMUL/EPICYCLIC_GEAR (M257)
        self.sensor_spring_sound_speeds: Dict[int, Any] = {}         # /SENSOR/SPRING_SOUND_SPEED (M257)

        # M259 Entities
        self.fail_rtcls: Dict[int, Any] = {}                         # /FAIL/RTCL (M259)
        self.eng_yield_stresses: Dict[int, Any] = {}                 # /ENG/YIELD_STRESS (M259)
        self.lagmul_harmonic_drives: Dict[int, Any] = {}             # /HARMONIC_DRIVE, /LAGMUL/HARMONIC_DRIVE (M259)
        self.sensor_spring_yield_stresses: Dict[int, Any] = {}       # /SENSOR/SPRING_YIELD_STRESS (M259)

        # M260 Entities
        self.fail_sahraeis: Dict[int, Any] = {}                      # /FAIL/SAHRAEI (M260)
        self.eng_plastic_works: Dict[int, Any] = {}                  # /ENG/PLASTIC_WORK (M260)
        self.lagmul_cycloidal_drives: Dict[int, Any] = {}            # /CYCLOIDAL_DRIVE, /LAGMUL/CYCLOIDAL_DRIVE (M260)
        self.sensor_spring_plastic_works: Dict[int, Any] = {}        # /SENSOR/SPRING_PLASTIC_WORK (M260)

        # M261 Entities
        self.fail_syazwans: Dict[int, Any] = {}                      # /FAIL/SYAZWAN (M261)
        self.eng_temperatures: Dict[int, Any] = {}                   # /ENG/TEMPERATURE (M261)
        self.lagmul_rack_pinions: Dict[int, Any] = {}                # /RACK_AND_PINION, /LAGMUL/RACK_AND_PINION (M261)
        self.sensor_spring_force_rates: Dict[int, Any] = {}          # /SENSOR/SPRING_FORCE_RATE (M261)

        # M262 Entities
        self.fail_pucks: Dict[int, Any] = {}                         # /FAIL/PUCK (M262)
        self.eng_stress_tris: Dict[int, Any] = {}                    # /ENG/STRESS_TRI (M262)
        self.lagmul_screw_joints: Dict[int, Any] = {}                # /SCREW_JOINT, /LAGMUL/SCREW_JOINT (M262)
        self.sensor_spring_force_impulses: Dict[int, Any] = {}       # /SENSOR/SPRING_FORCE_IMPULSE (M262)

        # M263 Entities
        self.fail_gursons: Dict[int, Any] = {}                       # /FAIL/GURSON (M263)
        self.eng_lode_angles: Dict[int, Any] = {}                    # /ENG/LODE_ANGLE (M263)
        self.lagmul_differential_gears: Dict[int, Any] = {}          # /DIFFERENTIAL_GEAR, /LAGMUL/DIFFERENTIAL_GEAR (M263)
        self.sensor_spring_moment_rates: Dict[int, Any] = {}         # /SENSOR/SPRING_MOMENT_RATE (M263)

        # M264 Entities
        self.fail_johnson_cooks: Dict[int, Any] = {}                 # /FAIL/JOHNSON_COOK (M264)
        self.eng_max_shears: Dict[int, Any] = {}                     # /ENG/MAX_SHEAR (M264)
        self.lagmul_transfer_cases: Dict[int, Any] = {}              # /TRANSFER_CASE, /LAGMUL/TRANSFER_CASE (M264)
        self.sensor_spring_moment_impulses: Dict[int, Any] = {}      # /SENSOR/SPRING_MOMENT_IMPULSE (M264)

        # M265 Entities
        self.fail_cockcroft_lathams: Dict[int, Any] = {}             # /FAIL/COCKCROFT_LATHAM (M265)
        self.eng_effective_stresses: Dict[int, Any] = {}             # /ENG/EFFECTIVE_STRESS (M265)
        self.lagmul_torque_split_gears: Dict[int, Any] = {}          # /TORQUE_SPLIT_GEAR, /LAGMUL/TORQUE_SPLIT_GEAR (M265)
        self.sensor_spring_torsional_energies: Dict[int, Any] = {}   # /SENSOR/SPRING_TORSIONAL_ENERGY (M265)

        # M266 Entities
        self.fail_lemaitre_damages: Dict[int, Any] = {}              # /FAIL/LEMAITRE_DAMAGE (M266)
        self.eng_hydrostatic_pressures: Dict[int, Any] = {}          # /ENG/HYDROSTATIC_PRESSURE (M266)
        self.lagmul_geneva_drives: Dict[int, Any] = {}               # /GENEVA_DRIVE, /LAGMUL/GENEVA_DRIVE (M266)
        self.sensor_spring_bending_energies: Dict[int, Any] = {}     # /SENSOR/SPRING_BENDING_ENERGY (M266)

        # M267 Entities
        self.fail_tabulated_plasticities: Dict[int, Any] = {}        # /FAIL/TABULATED_PLASTICITY (M267)
        self.eng_octahedral_shears: Dict[int, Any] = {}              # /ENG/OCTAHEDRAL_SHEAR (M267)
        self.lagmul_scotch_yokes: Dict[int, Any] = {}                # /SCOTCH_YOKE, /LAGMUL/SCOTCH_YOKE (M267)
        self.sensor_spring_total_strain_energies: Dict[int, Any] = {}# /SENSOR/SPRING_TOTAL_STRAIN_ENERGY (M267)

        # M268 Entities
        self.fail_mohr_coulombs: Dict[int, Any] = {}                 # /FAIL/MOHR_COULOMB (M268)
        self.eng_deviatoric_energies: Dict[int, Any] = {}            # /ENG/DEVIATORIC_ENERGY (M268)
        self.lagmul_oldham_couplings: Dict[int, Any] = {}            # /OLDHAM_COUPLING, /LAGMUL/OLDHAM_COUPLING (M268)
        self.sensor_spring_volumetric_energies: Dict[int, Any] = {}  # /SENSOR/SPRING_VOLUMETRIC_ENERGY (M268)

        # M269 Entities
        self.fail_drucker_pragers: Dict[int, Any] = {}               # /FAIL/DRUCKER_PRAGER (M269)
        self.eng_strain_rates: Dict[int, Any] = {}                   # /ENG/STRAIN_RATE (M269)
        self.lagmul_schmidt_couplings: Dict[int, Any] = {}           # /SCHMIDT_COUPLING, /LAGMUL/SCHMIDT_COUPLING (M269)
        self.sensor_spring_shear_energies: Dict[int, Any] = {}       # /SENSOR/SPRING_SHEAR_ENERGY (M269)

        # M270 Entities
        self.fail_hosford_coulombs: Dict[int, Any] = {}              # /FAIL/HOSFORD_COULOMB (M270)
        self.eng_bulk_viscosities: Dict[int, Any] = {}               # /ENG/BULK_VISCOSITY (M270)
        self.lagmul_rzeppa_joints: Dict[int, Any] = {}               # /RZEPPA_JOINT, /LAGMUL/RZEPPA_JOINT (M270)
        self.sensor_spring_axial_energies: Dict[int, Any] = {}       # /SENSOR/SPRING_AXIAL_ENERGY (M270)

        # M271 Entities
        self.fail_biquad_anisos: Dict[int, Any] = {}                 # /FAIL/BIQUAD_ANISO (M271)
        self.eng_hourglass_energies: Dict[int, Any] = {}             # /ENG/HOURGLASS_ENERGY (M271)
        self.lagmul_birfield_joints: Dict[int, Any] = {}             # /BIRFIELD_JOINT, /LAGMUL/BIRFIELD_JOINT (M271)
        self.sensor_spring_damping_energies: Dict[int, Any] = {}     # /SENSOR/SPRING_DAMPING_ENERGY (M271)

        # M272 Entities
        self.fail_wilkins_cumulatives: Dict[int, Any] = {}           # /FAIL/WILKINS_CUMULATIVE (M272)
        self.eng_contact_energies: Dict[int, Any] = {}               # /ENG/CONTACT_ENERGY (M272)
        self.lagmul_tripod_joints: Dict[int, Any] = {}               # /TRIPOD_JOINT, /LAGMUL/TRIPOD_JOINT (M272)
        self.sensor_spring_coupling_energies: Dict[int, Any] = {}    # /SENSOR/SPRING_COUPLING_ENERGY (M272)

        # M273 Entities
        self.fail_tuler_butchers: Dict[int, Any] = {}                # /FAIL/TULER_BUTCHER (M273)
        self.eng_spring_energies: Dict[int, Any] = {}                # /ENG/SPRING_ENERGY (M273)
        self.lagmul_hooke_joints: Dict[int, Any] = {}                # /HOOKE_JOINT, /LAGMUL/HOOKE_JOINT (M273)
        self.sensor_spring_torsional_energies: Dict[int, Any] = {}   # /SENSOR/SPRING_TORSIONAL_ENERGY (M273)

        # M274 Entities
        self.fail_extended_mohrs: Dict[int, Any] = {}                # /FAIL/EXTENDED_MOHR (M274)
        self.eng_joint_energies: Dict[int, Any] = {}                 # /ENG/JOINT_ENERGY (M274)
        self.lagmul_tracta_joints: Dict[int, Any] = {}               # /TRACTA_JOINT, /LAGMUL/TRACTA_JOINT (M274)
        self.sensor_spring_bending_energies: Dict[int, Any] = {}     # /SENSOR/SPRING_BENDING_ENERGY (M274)

        # M275 Entities
        self.fail_oyanes: Dict[int, Any] = {}                        # /FAIL/OYANE (M275)
        self.eng_rwall_energies: Dict[int, Any] = {}                 # /ENG/RWALL_ENERGY (M275)
        self.lagmul_thompson_couplings: Dict[int, Any] = {}          # /THOMPSON_COUPLING, /LAGMUL/THOMPSON_COUPLING (M275)
        self.sensor_spring_pinching_energies: Dict[int, Any] = {}    # /SENSOR/SPRING_PINCHING_ENERGY (M275)

        # M276 Entities
        self.fail_freudenthals: Dict[int, Any] = {}                  # /FAIL/FREUDENTHAL (M276)
        self.eng_surf_energies: Dict[int, Any] = {}                  # /ENG/SURF_ENERGY (M276)
        self.lagmul_weiss_joints: Dict[int, Any] = {}                # /WEISS_JOINT, /LAGMUL/WEISS_JOINT (M276)
        self.sensor_spring_friction_energies: Dict[int, Any] = {}    # /SENSOR/SPRING_FRICTION_ENERGY (M276)

        # M277 Entities
        self.fail_alters: Dict[int, Any] = {}                        # /FAIL/ALTER (M277)
        self.eng_heat_exchanges: Dict[int, Any] = {}                 # /ENG/HEAT_EXCHANGE (M277)
        self.lagmul_tripod_ball_joints: Dict[int, Any] = {}          # /TRIPOD_BALL_JOINT, /LAGMUL/TRIPOD_BALL_JOINT (M277)
        self.sensor_spring_thermal_dissipations: Dict[int, Any] = {} # /SENSOR/SPRING_THERMAL_DISSIPATION (M277)

        # M278 Entities
        self.eng_sph_energies: Dict[int, Any] = {}                   # /ENG/SPH_ENERGY, /ENG/SPH_WORK (M278)
        self.lagmul_clevis_joints: Dict[int, Any] = {}               # /CLEVIS_JOINT, /LAGMUL/CLEVIS_JOINT (M278)
        self.sensor_spring_total_works: Dict[int, Any] = {}          # /SENSOR/SPRING_TOTAL_WORK (M278)

        # M279 Entities
        self.eng_ale_energies: Dict[int, Any] = {}                   # /ENG/ALE_ENERGY, /ENG/ALE_WORK (M279)
        self.lagmul_pin_in_slot_joints: Dict[int, Any] = {}          # /PIN_IN_SLOT_JOINT, /LAGMUL/PIN_IN_SLOT_JOINT (M279)
        self.sensor_spring_rotational_works: Dict[int, Any] = {}     # /SENSOR/SPRING_ROTATIONAL_WORK (M279)

        # M280 Entities
        self.eng_fsi_energies: Dict[int, Any] = {}                   # /ENG/FSI_ENERGY, /ENG/FSI_WORK (M280)
        self.lagmul_slider_slot_joints: Dict[int, Any] = {}          # /SLIDER_SLOT_JOINT, /LAGMUL/SLIDER_SLOT_JOINT (M280)
        self.sensor_spring_translational_works: Dict[int, Any] = {}  # /SENSOR/SPRING_TRANSLATIONAL_WORK (M280)

        # M281 Entities
        self.eng_xfem_energies: Dict[int, Any] = {}                  # /ENG/XFEM_ENERGY, /ENG/XFEM_WORK (M281)
        self.lagmul_parallel_axis_joints: Dict[int, Any] = {}        # /PARALLEL_AXIS_JOINT, /LAGMUL/PARALLEL_AXIS_JOINT (M281)
        self.sensor_spring_shear_works: Dict[int, Any] = {}          # /SENSOR/SPRING_SHEAR_WORK (M281)

        # M282 Entities
        self.eng_helmholtz_energies: Dict[int, Any] = {}             # /ENG/HELMHOLTZ_ENERGY, /ENG/HELMHOLTZ_WORK (M282)
        self.lagmul_cam_follower_joints: Dict[int, Any] = {}         # /CAM_FOLLOWER_JOINT, /LAGMUL/CAM_FOLLOWER_JOINT (M282)
        self.sensor_spring_normal_works: Dict[int, Any] = {}         # /SENSOR/SPRING_NORMAL_WORK (M282)

        # M283 Entities
        self.eng_entropy_productions: Dict[int, Any] = {}            # /ENG/ENTROPY_PRODUCTION, /ENG/ENTROPY_PROD (M283)
        self.lagmul_screw_nut_joints: Dict[int, Any] = {}            # /SCREW_NUT_JOINT, /LAGMUL/SCREW_NUT_JOINT (M283)
        self.sensor_spring_total_forces: Dict[int, Any] = {}         # /SENSOR/SPRING_TOTAL_FORCE (M283)

        # M284 Entities
        self.eng_internal_pressures: Dict[int, Any] = {}             # /ENG/INTERNAL_PRESSURE, /ENG/INT_PRESSURE (M284)
        self.lagmul_geneva_joints: Dict[int, Any] = {}               # /GENEVA_JOINT, /LAGMUL/GENEVA_JOINT (M284)
        self.sensor_spring_total_moments: Dict[int, Any] = {}        # /SENSOR/SPRING_TOTAL_MOMENT (M284)

        # M285 Entities
        self.fail_louhuos: Dict[int, Any] = {}                       # /FAIL/LOU_HUO (M285)
        self.eng_coriolis_energies: Dict[int, Any] = {}              # /ENG/CORIOLIS_ENERGY (M285)
        self.lagmul_cable_pulley_joints: Dict[int, Any] = {}         # /CABLE_PULLEY_JOINT, /LAGMUL/CABLE_PULLEY_JOINT (M285)
        self.sensor_spring_angular_velocities: Dict[int, Any] = {}   # /SENSOR/SPRING_ANGULAR_VELOCITY (M285)

        # M286 Entities
        self.fail_ladstrs: Dict[int, Any] = {}                       # /FAIL/LAD_STR (M286)
        self.eng_magnetic_energies: Dict[int, Any] = {}              # /ENG/MAGNETIC_ENERGY (M286)
        self.lagmul_swash_plate_joints: Dict[int, Any] = {}          # /SWASH_PLATE_JOINT, /LAGMUL/SWASH_PLATE_JOINT (M286)
        self.sensor_spring_angular_accelerations: Dict[int, Any] = {} # /SENSOR/SPRING_ANGULAR_ACCELERATION (M286)

        # M287 Entities
        self.fail_ladviscs: Dict[int, Any] = {}                      # /FAIL/LAD_VISC (M287)
        self.eng_poynting_energies: Dict[int, Any] = {}              # /ENG/POYNTING_ENERGY (M287)
        self.lagmul_scissor_mechanism_joints: Dict[int, Any] = {}    # /SCISSOR_MECHANISM_JOINT, /LAGMUL/SCISSOR_MECHANISM_JOINT (M287)
        self.sensor_spring_torsional_rates: Dict[int, Any] = {}      # /SENSOR/SPRING_TORSIONAL_RATE (M287)

        # M288 Entities
        self.fail_ladinters: Dict[int, Any] = {}                     # /FAIL/LAD_INTER (M288)
        self.eng_maxwell_stress_energies: Dict[int, Any] = {}        # /ENG/MAXWELL_STRESS_ENERGY (M288)
        self.lagmul_parallelogram_joints: Dict[int, Any] = {}        # /PARALLELOGRAM_JOINT, /LAGMUL/PARALLELOGRAM_JOINT (M288)
        self.sensor_spring_normal_accelerations: Dict[int, Any] = {} # /SENSOR/SPRING_NORMAL_ACCELERATION (M288)

        # M289 Entities
        self.fail_ladfibs: Dict[int, Any] = {}                       # /FAIL/LAD_FIB (M289)
        self.eng_joule_heat_energies: Dict[int, Any] = {}            # /ENG/JOULE_HEAT_ENERGY (M289)
        self.lagmul_delta_robot_joints: Dict[int, Any] = {}          # /DELTA_ROBOT_JOINT, /LAGMUL/DELTA_ROBOT_JOINT (M289)
        self.sensor_spring_shear_accelerations: Dict[int, Any] = {}  # /SENSOR/SPRING_SHEAR_ACCELERATION (M289)

        # M290 Entities
        self.fail_ladmicros: Dict[int, Any] = {}                     # /FAIL/LAD_MICRO (M290)
        self.eng_lorentz_force_energies: Dict[int, Any] = {}         # /ENG/LORENTZ_FORCE_ENERGY (M290)
        self.lagmul_spherical_wrist_joints: Dict[int, Any] = {}      # /SPHERICAL_WRIST_JOINT, /LAGMUL/SPHERICAL_WRIST_JOINT (M290)
        self.sensor_spring_resultant_accelerations: Dict[int, Any] = {} # /SENSOR/SPRING_RESULTANT_ACCELERATION (M290)

        # M291 Entities
        self.fail_ladcouples: Dict[int, Any] = {}                    # /FAIL/LAD_COUPLE (M291)
        self.eng_plasmonic_energies: Dict[int, Any] = {}             # /ENG/PLASMONIC_ENERGY (M291)
        self.lagmul_lead_screw_joints: Dict[int, Any] = {}           # /LEAD_SCREW_JOINT, /LAGMUL/LEAD_SCREW_JOINT (M291)
        self.sensor_spring_torsional_accelerations: Dict[int, Any] = {} # /SENSOR/SPRING_TORSIONAL_ACCELERATION (M291)

        # M292 Entities
        self.fail_ladviscoplasts: Dict[int, Any] = {}                # /FAIL/LAD_VISCO_PLAST (M292)
        self.eng_dielectric_loss_energies: Dict[int, Any] = {}       # /ENG/DIELECTRIC_LOSS_ENERGY (M292)
        self.lagmul_hoeken_linkage_joints: Dict[int, Any] = {}       # /HOEKEN_LINKAGE_JOINT, /LAGMUL/HOEKEN_LINKAGE_JOINT (M292)
        self.sensor_spring_bending_accelerations: Dict[int, Any] = {} # /SENSOR/SPRING_BENDING_ACCELERATION (M292)

        # M293 Entities
        self.fail_ladcreeps: Dict[int, Any] = {}                     # /FAIL/LAD_CREEP (M293)
        self.eng_magnetic_hysteresis_energies: Dict[int, Any] = {}   # /ENG/MAGNETIC_HYSTERESIS_ENERGY (M293)
        self.lagmul_chebyshev_linkage_joints: Dict[int, Any] = {}    # /CHEBYSHEV_LINKAGE_JOINT, /LAGMUL/CHEBYSHEV_LINKAGE_JOINT (M293)
        self.sensor_spring_total_angular_accelerations: Dict[int, Any] = {} # /SENSOR/SPRING_TOTAL_ANGULAR_ACCELERATION (M293)

        # M294 Entities
        self.fail_ladtransisotropics: Dict[int, Any] = {}            # /FAIL/LAD_TRANS_ISOTROPIC (M294)
        self.eng_magnetostriction_energies: Dict[int, Any] = {}      # /ENG/MAGNETOSTRICTION_ENERGY (M294)
        self.lagmul_roberts_linkage_joints: Dict[int, Any] = {}      # /ROBERTS_LINKAGE_JOINT, /LAGMUL/ROBERTS_LINKAGE_JOINT (M294)
        self.sensor_spring_normal_jerks: Dict[int, Any] = {}         # /SENSOR/SPRING_NORMAL_JERK (M294)

        # M295 Entities
        self.fail_ladviscodamages: Dict[int, Any] = {}               # /FAIL/LAD_VISCO_DAMAGE (M295)
        self.eng_electrocaloric_energies: Dict[int, Any] = {}        # /ENG/ELECTROCALORIC_ENERGY (M295)
        self.lagmul_evans_linkage_joints: Dict[int, Any] = {}        # /EVANS_LINKAGE_JOINT, /LAGMUL/EVANS_LINKAGE_JOINT (M295)
        self.sensor_spring_shear_jerks: Dict[int, Any] = {}          # /SENSOR/SPRING_SHEAR_JERK (M295)

        # M296 Entities
        self.fail_laddelams: Dict[int, Any] = {}                     # /FAIL/LAD_DELAM (M296)
        self.eng_magnetocaloric_energies: Dict[int, Any] = {}        # /ENG/MAGNETOCALORIC_ENERGY (M296)
        self.lagmul_watt_linkage_joints: Dict[int, Any] = {}         # /WATT_LINKAGE_JOINT, /LAGMUL/WATT_LINKAGE_JOINT (M296)
        self.sensor_spring_resultant_jerks: Dict[int, Any] = {}      # /SENSOR/SPRING_RESULTANT_JERK (M296)

        # M297 Entities
        self.fail_ladtcasymmetries: Dict[int, Any] = {}              # /FAIL/LAD_TC_ASYMMETRY (M297)
        self.eng_thermoelectric_energies: Dict[int, Any] = {}        # /ENG/THERMOELECTRIC_ENERGY (M297)
        self.lagmul_hart_linkage_joints: Dict[int, Any] = {}         # /HART_LINKAGE_JOINT, /LAGMUL/HART_LINKAGE_JOINT (M297)
        self.sensor_spring_torsional_jerks: Dict[int, Any] = {}      # /SENSOR/SPRING_TORSIONAL_JERK (M297)

        # M298 Entities
        self.fail_ladanisos: Dict[int, Any] = {}                     # /FAIL/LAD_ANISO (M298)
        self.eng_pyroelectric_energies: Dict[int, Any] = {}          # /ENG/PYROELECTRIC_ENERGY (M298)
        self.lagmul_peaucellier_linkage_joints: Dict[int, Any] = {}  # /PEAUCELLIER_LINKAGE_JOINT, /LAGMUL/PEAUCELLIER_LINKAGE_JOINT (M298)
        self.sensor_spring_bending_jerks: Dict[int, Any] = {}        # /SENSOR/SPRING_BENDING_JERK (M298)

        # M299 Entities
        self.fail_ladfatigues: Dict[int, Any] = {}                   # /FAIL/LAD_FATIGUE (M299)
        self.eng_thermomagnetic_energies: Dict[int, Any] = {}        # /ENG/THERMOMAGNETIC_ENERGY (M299)
        self.lagmul_sarrus_linkage_joints: Dict[int, Any] = {}       # /SARRUS_LINKAGE_JOINT, /LAGMUL/SARRUS_LINKAGE_JOINT (M299)
        self.sensor_spring_total_angular_jerks: Dict[int, Any] = {}  # /SENSOR/SPRING_TOTAL_ANGULAR_JERK (M299)

        # M300 Entities
        self.fail_ladviscofatigues: Dict[int, Any] = {}              # /FAIL/LAD_VISCO_FATIGUE (M300)
        self.eng_thermogalvanic_energies: Dict[int, Any] = {}        # /ENG/THERMOGALVANIC_ENERGY (M300)
        self.lagmul_klann_linkage_joints: Dict[int, Any] = {}        # /KLANN_LINKAGE_JOINT, /LAGMUL/KLANN_LINKAGE_JOINT (M300)
        self.sensor_spring_total_acceleration_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TOTAL_ACCELERATION_RATE (M300)

        # M301 Entities
        self.fail_ladcoupledamages: Dict[int, Any] = {}              # /FAIL/LAD_COUPLE_DAMAGE (M301)
        self.eng_thermionic_energies: Dict[int, Any] = {}            # /ENG/THERMIONIC_ENERGY (M301)
        self.lagmul_jansen_linkage_joints: Dict[int, Any] = {}       # /JANSEN_LINKAGE_JOINT, /LAGMUL/JANSEN_LINKAGE_JOINT (M301)
        self.sensor_spring_normal_acceleration_rates: Dict[int, Any] = {} # /SENSOR/SPRING_NORMAL_ACCELERATION_RATE (M301)

        # M302 Entities
        self.fail_ladfractures: Dict[int, Any] = {}                  # /FAIL/LAD_FRACTURE (M302)
        self.eng_thermophotonic_energies: Dict[int, Any] = {}        # /ENG/THERMOPHOTONIC_ENERGY (M302)
        self.lagmul_kempe_linkage_joints: Dict[int, Any] = {}        # /KEMPE_LINKAGE_JOINT, /LAGMUL/KEMPE_LINKAGE_JOINT (M302)
        self.sensor_spring_transverse_acceleration_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TRANSVERSE_ACCELERATION_RATE (M302)

        # M303 Entities
        self.fail_ladfibermatrixdebondings: Dict[int, Any] = {}      # /FAIL/LAD_FIBER_MATRIX_DEBONDING (M303)
        self.eng_electrostrictive_energies: Dict[int, Any] = {}      # /ENG/ELECTROSTRICTIVE_ENERGY (M303)
        self.lagmul_sylvester_kempe_linkage_joints: Dict[int, Any] = {} # /SYLVESTER_KEMPE_LINKAGE_JOINT, /LAGMUL/SYLVESTER_KEMPE_LINKAGE_JOINT (M303)
        self.sensor_spring_torsional_acceleration_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TORSIONAL_ACCELERATION_RATE (M303)

        # M304 Entities
        self.fail_ladfiberkinkings: Dict[int, Any] = {}              # /FAIL/LAD_FIBER_KINKING (M304)
        self.eng_photomagnetic_energies: Dict[int, Any] = {}         # /ENG/PHOTOMAGNETIC_ENERGY (M304)
        self.lagmul_wobble_yoke_joints: Dict[int, Any] = {}          # /WOBBLE_YOKE_JOINT, /LAGMUL/WOBBLE_YOKE_JOINT (M304)
        self.sensor_spring_bending_acceleration_rates: Dict[int, Any] = {} # /SENSOR/SPRING_BENDING_ACCELERATION_RATE (M304)

        # M305 Entities
        self.fail_laddiffusedamages: Dict[int, Any] = {}             # /FAIL/LAD_DIFFUSE_DAMAGE (M305)
        self.eng_thermophotonic_emission_energies: Dict[int, Any] = {} # /ENG/THERMOPHOTONIC_EMISSION_ENERGY (M305)
        self.lagmul_hypocyclic_linkage_joints: Dict[int, Any] = {}   # /HYPOCYCLIC_LINKAGE_JOINT, /LAGMUL/HYPOCYCLIC_LINKAGE_JOINT (M305)
        self.sensor_spring_total_angular_acceleration_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TOTAL_ANGULAR_ACCELERATION_RATE (M305)

        # M306 Entities
        self.fail_ladinterlaminarshears: Dict[int, Any] = {}         # /FAIL/LAD_INTERLAMINAR_SHEAR (M306)
        self.eng_phonon_polariton_energies: Dict[int, Any] = {}      # /ENG/PHONON_POLARITON_ENERGY (M306)
        self.lagmul_pantograph_linkage_joints: Dict[int, Any] = {}   # /PANTOGRAPH_LINKAGE_JOINT, /LAGMUL/PANTOGRAPH_LINKAGE_JOINT (M306)
        self.sensor_spring_total_acceleration_jerks: Dict[int, Any] = {} # /SENSOR/SPRING_TOTAL_ACCELERATION_JERK (M306)

        # M307 Entities
        self.fail_ladfibermatrixinteractions: Dict[int, Any] = {}    # /FAIL/LAD_FIBER_MATRIX_INTERACTION (M307)
        self.eng_exciton_polariton_energies: Dict[int, Any] = {}     # /ENG/EXCITON_POLARITON_ENERGY (M307)
        self.lagmul_watt_parallel_motion_joints: Dict[int, Any] = {} # /WATT_PARALLEL_MOTION_JOINT, /LAGMUL/WATT_PARALLEL_MOTION_JOINT (M307)
        self.sensor_spring_normal_acceleration_jerks: Dict[int, Any] = {} # /SENSOR/SPRING_NORMAL_ACCELERATION_JERK (M307)

        # M308 Entities
        self.fail_ladtransversetensions: Dict[int, Any] = {}         # /FAIL/LAD_TRANSVERSE_TENSION (M308)
        self.eng_magnon_polariton_energies: Dict[int, Any] = {}      # /ENG/MAGNON_POLARITON_ENERGY (M308)
        self.lagmul_scott_russell_linkage_joints: Dict[int, Any] = {} # /SCOTT_RUSSELL_LINKAGE_JOINT, /LAGMUL/SCOTT_RUSSELL_LINKAGE_JOINT (M308)
        self.sensor_spring_transverse_acceleration_jerks: Dict[int, Any] = {} # /SENSOR/SPRING_TRANSVERSE_ACCELERATION_JERK (M308)

        # M309 Entities
        self.fail_ladtransversecompressions: Dict[int, Any] = {}    # /FAIL/LAD_TRANSVERSE_COMPRESSION (M309)
        self.eng_piezomagnetic_energies: Dict[int, Any] = {}        # /ENG/PIEZOMAGNETIC_ENERGY (M309)
        self.lagmul_watt_beam_engine_joints: Dict[int, Any] = {}    # /WATT_BEAM_ENGINE_JOINT, /LAGMUL/WATT_BEAM_ENGINE_JOINT (M309)
        self.sensor_spring_torsional_jerk_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TORSIONAL_JERK_RATE (M309)

        # M310 Entities
        self.fail_ladinplaneshears: Dict[int, Any] = {}             # /FAIL/LAD_INPLANE_SHEAR (M310)
        self.eng_barocaloric_energies: Dict[int, Any] = {}          # /ENG/BAROCALORIC_ENERGY (M310)
        self.lagmul_stephenson_linkage_joints: Dict[int, Any] = {}  # /STEPHENSON_LINKAGE_JOINT, /LAGMUL/STEPHENSON_LINKAGE_JOINT (M310)
        self.sensor_spring_bending_jerk_rates: Dict[int, Any] = {}  # /SENSOR/SPRING_BENDING_JERK_RATE (M310)

        # M311 Entities
        self.fail_ladinterfacialdelaminations: Dict[int, Any] = {} # /FAIL/LAD_INTERFACIAL_DELAMINATION (M311)
        self.eng_thermomagnetoelectric_energies: Dict[int, Any] = {} # /ENG/THERMOMAGNETOELECTRIC_ENERGY (M311)
        self.lagmul_wobble_plate_mechanism_joints: Dict[int, Any] = {} # /WOBBLE_PLATE_MECHANISM_JOINT, /LAGMUL/WOBBLE_PLATE_MECHANISM_JOINT (M311)
        self.sensor_spring_total_jerk_rates: Dict[int, Any] = {}    # /SENSOR/SPRING_TOTAL_JERK_RATE (M311)

        # M312 Entities
        self.fail_ladtransverseshearinteractions: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_SHEAR_INTERACTION (M312)
        self.eng_electrohydrodynamic_energies: Dict[int, Any] = {}    # /ENG/ELECTROHYDRODYNAMIC_ENERGY (M312)
        self.lagmul_whitworth_quick_return_joints: Dict[int, Any] = {} # /WHITWORTH_QUICK_RETURN_JOINT, /LAGMUL/WHITWORTH_QUICK_RETURN_JOINT (M312)
        self.sensor_spring_normal_jerk_rates: Dict[int, Any] = {}     # /SENSOR/SPRING_NORMAL_JERK_RATE (M312)

        # M313 Entities
        self.fail_ladfibermatrixdebondrates: Dict[int, Any] = {}    # /FAIL/LAD_FIBER_MATRIX_DEBOND_RATE (M313)
        self.eng_magnetogalvanic_energies: Dict[int, Any] = {}      # /ENG/MAGNETOGALVANIC_ENERGY (M313)
        self.lagmul_chebyshev_lambda_linkage_joints: Dict[int, Any] = {} # /CHEBYSHEV_LAMBDA_LINKAGE_JOINT, /LAGMUL/CHEBYSHEV_LAMBDA_LINKAGE_JOINT (M313)
        self.sensor_spring_transverse_jerk_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TRANSVERSE_JERK_RATE (M313)

        # M314 Entities
        self.fail_ladinplaneshearrates: Dict[int, Any] = {}         # /FAIL/LAD_INPLANE_SHEAR_RATE (M314)
        self.eng_elastocaloric_energies: Dict[int, Any] = {}        # /ENG/ELASTOCALORIC_ENERGY (M314)
        self.lagmul_four_bar_crank_rocker_joints: Dict[int, Any] = {} # /FOUR_BAR_CRANK_ROCKER_JOINT, /LAGMUL/FOUR_BAR_CRANK_ROCKER_JOINT (M314)
        self.sensor_spring_torsional_snap_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TORSIONAL_SNAP_RATE (M314)

        # M315 Entities
        self.fail_ladtransversecompressionrates: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_COMPRESSION_RATE (M315)
        self.eng_thermophononic_energies: Dict[int, Any] = {}       # /ENG/THERMOPHONONIC_ENERGY (M315)
        self.lagmul_four_bar_double_crank_joints: Dict[int, Any] = {} # /FOUR_BAR_DOUBLE_CRANK_JOINT, /LAGMUL/FOUR_BAR_DOUBLE_CRANK_JOINT (M315)
        self.sensor_spring_bending_snap_rates: Dict[int, Any] = {}  # /SENSOR/SPRING_BENDING_SNAP_RATE (M315)

        # M316 Entities
        self.fail_ladtransversetensionrates: Dict[int, Any] = {}    # /FAIL/LAD_TRANSVERSE_TENSION_RATE (M316)
        self.eng_thermoplasmonic_energies: Dict[int, Any] = {}      # /ENG/THERMOPLASMONIC_ENERGY (M316)
        self.lagmul_four_bar_double_rocker_joints: Dict[int, Any] = {} # /FOUR_BAR_DOUBLE_ROCKER_JOINT, /LAGMUL/FOUR_BAR_DOUBLE_ROCKER_JOINT (M316)
        self.sensor_spring_normal_snap_rates: Dict[int, Any] = {}   # /SENSOR/SPRING_NORMAL_SNAP_RATE (M316)

        # M317 Entities
        self.fail_ladinterfacialdelaminationrates: Dict[int, Any] = {} # /FAIL/LAD_INTERFACIAL_DELAMINATION_RATE (M317)
        self.eng_thermomagnetic_generator_energies: Dict[int, Any] = {} # /ENG/THERMOMAGNETIC_GENERATOR_ENERGY (M317)
        self.lagmul_slider_rocker_inversion_joints: Dict[int, Any] = {} # /SLIDER_ROCKER_INVERSION_JOINT, /LAGMUL/SLIDER_ROCKER_INVERSION_JOINT (M317)
        self.sensor_spring_transverse_snap_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TRANSVERSE_SNAP_RATE (M317)

        # M318 Entities
        self.fail_ladtransverseshearinteractionrates: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_SHEAR_INTERACTION_RATE (M318)
        self.eng_magnetorheological_energies: Dict[int, Any] = {}     # /ENG/MAGNETORHEOLOGICAL_ENERGY (M318)
        self.lagmul_scotch_yoke_mechanism_joints: Dict[int, Any] = {} # /SCOTCH_YOKE_MECHANISM_JOINT, /LAGMUL/SCOTCH_YOKE_MECHANISM_JOINT (M318)
        self.sensor_spring_total_snap_rates: Dict[int, Any] = {}      # /SENSOR/SPRING_TOTAL_SNAP_RATE (M318)

        # M319 Entities
        self.fail_ladnonlocalgradients: Dict[int, Any] = {}          # /FAIL/LAD_NONLOCAL_GRADIENT (M319)
        self.eng_electrorheological_energies: Dict[int, Any] = {}    # /ENG/ELECTRORHEOLOGICAL_ENERGY (M319)
        self.lagmul_geneva_drive_mechanism_joints: Dict[int, Any] = {} # /GENEVA_DRIVE_MECHANISM_JOINT, /LAGMUL/GENEVA_DRIVE_MECHANISM_JOINT (M319)
        self.sensor_spring_normal_crackle_rates: Dict[int, Any] = {}  # /SENSOR/SPRING_NORMAL_CRACKLE_RATE (M319)

        # M320 Entities
        self.fail_ladanisotropicplasticities: Dict[int, Any] = {}      # /FAIL/LAD_ANISOTROPIC_PLASTICITY (M320)
        self.eng_thermoacoustic_energies: Dict[int, Any] = {}          # /ENG/THERMOACOUSTIC_ENERGY (M320)
        self.lagmul_double_cardan_joints: Dict[int, Any] = {}          # /DOUBLE_CARDAN_JOINT, /LAGMUL/DOUBLE_CARDAN_JOINT (M320)
        self.sensor_spring_transverse_crackle_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE (M320)

        # M321 Entities
        self.fail_ladnonlocalgradientrates: Dict[int, Any] = {}       # /FAIL/LAD_NONLOCAL_GRADIENT_RATE (M321)
        self.eng_ferroelectric_energies: Dict[int, Any] = {}          # /ENG/FERROELECTRIC_ENERGY (M321)
        self.lagmul_bennett_linkage_joints: Dict[int, Any] = {}       # /BENNETT_LINKAGE_JOINT, /LAGMUL/BENNETT_LINKAGE_JOINT (M321)
        self.sensor_spring_total_crackle_rates: Dict[int, Any] = {}   # /SENSOR/SPRING_TOTAL_CRACKLE_RATE (M321)

        # M322 Entities
        self.fail_ladfiberkinkingrates: Dict[int, Any] = {}          # /FAIL/LAD_FIBER_KINKING_RATE (M322)
        self.eng_flexoelectric_energies: Dict[int, Any] = {}         # /ENG/FLEXOELECTRIC_ENERGY (M322)
        self.lagmul_bricard_linkage_joints: Dict[int, Any] = {}      # /BRICARD_LINKAGE_JOINT, /LAGMUL/BRICARD_LINKAGE_JOINT (M322)
        self.sensor_spring_torsional_crackle_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TORSIONAL_CRACKLE_RATE (M322)

        # M323 Entities
        self.fail_ladfibertensionrates: Dict[int, Any] = {}          # /FAIL/LAD_FIBER_TENSION_RATE (M323)
        self.eng_pyromagnetic_energies: Dict[int, Any] = {}          # /ENG/PYROMAGNETIC_ENERGY (M323)
        self.lagmul_myard_linkage_joints: Dict[int, Any] = {}        # /MYARD_LINKAGE_JOINT, /LAGMUL/MYARD_LINKAGE_JOINT (M323)
        self.sensor_spring_bending_crackle_rates: Dict[int, Any] = {} # /SENSOR/SPRING_BENDING_CRACKLE_RATE (M323)

        # M324 Entities
        self.fail_ladfibercompressionrates: Dict[int, Any] = {}      # /FAIL/LAD_FIBER_COMPRESSION_RATE (M324)
        self.eng_piezothermal_energies: Dict[int, Any] = {}          # /ENG/PIEZOTHERMAL_ENERGY (M324)
        self.lagmul_goldberg_linkage_joints: Dict[int, Any] = {}     # /GOLDBERG_LINKAGE_JOINT, /LAGMUL/GOLDBERG_LINKAGE_JOINT (M324)
        self.sensor_spring_total_angular_crackle_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE (M324)

        # M325 Entities
        self.fail_ladhygrothermals: Dict[int, Any] = {}              # /FAIL/LAD_HYGROTHERMAL (M325)
        self.eng_thermoflexoelectric_energies: Dict[int, Any] = {}   # /ENG/THERMOFLEXOELECTRIC_ENERGY (M325)
        self.lagmul_waldron_linkage_joints: Dict[int, Any] = {}      # /WALDRON_LINKAGE_JOINT, /LAGMUL/WALDRON_LINKAGE_JOINT (M325)
        self.sensor_spring_normal_pop_rates: Dict[int, Any] = {}     # /SENSOR/SPRING_NORMAL_POP_RATE (M325)

        # M326 Entities
        self.fail_ladcoupleplasticitys: Dict[int, Any] = {}          # /FAIL/LAD_COUPLE_PLASTICITY (M326)
        self.eng_flexomagnetic_energies: Dict[int, Any] = {}         # /ENG/FLEXOMAGNETIC_ENERGY (M326)
        self.lagmul_dietmaier_linkage_joints: Dict[int, Any] = {}    # /DIETMAIER_LINKAGE_JOINT, /LAGMUL/DIETMAIER_LINKAGE_JOINT (M326)
        self.sensor_spring_transverse_pop_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TRANSVERSE_POP_RATE (M326)

        # M327 Entities
        self.fail_ladcouplecreeps: Dict[int, Any] = {}               # /FAIL/LAD_COUPLE_CREEP (M327)
        self.eng_pyroelectric_resonance_energies: Dict[int, Any] = {} # /ENG/PYROELECTRIC_RESONANCE_ENERGY (M327)
        self.lagmul_baker_linkage_joints: Dict[int, Any] = {}        # /BAKER_LINKAGE_JOINT, /LAGMUL/BAKER_LINKAGE_JOINT (M327)
        self.sensor_spring_total_pop_rates: Dict[int, Any] = {}      # /SENSOR/SPRING_TOTAL_POP_RATE (M327)

        # M328 Entities
        self.fail_ladcoupleviscoplasticitys: Dict[int, Any] = {}     # /FAIL/LAD_COUPLE_VISCOPLASTICITY (M328)
        self.eng_thermomagnetic_resonance_energies: Dict[int, Any] = {} # /ENG/THERMOMAGNETIC_RESONANCE_ENERGY (M328)
        self.lagmul_wohlhart_linkage_joints: Dict[int, Any] = {}     # /WOHLHART_LINKAGE_JOINT, /LAGMUL/WOHLHART_LINKAGE_JOINT (M328)
        self.sensor_spring_torsional_pop_rates: Dict[int, Any] = {}  # /SENSOR/SPRING_TORSIONAL_POP_RATE (M328)

        # M329 Entities
        self.fail_ladmicrodelaminationrates: Dict[int, Any] = {}     # /FAIL/LAD_MICRO_DELAMINATION_RATE (M329)
        self.eng_flexothermal_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMAL_RESONANCE_ENERGY (M329)
        self.lagmul_altmann_linkage_joints: Dict[int, Any] = {}      # /ALTMANN_LINKAGE_JOINT, /LAGMUL/ALTMANN_LINKAGE_JOINT (M329)
        self.sensor_spring_bending_pop_rates: Dict[int, Any] = {}    # /SENSOR/SPRING_BENDING_POP_RATE (M329)

        # M330 Entities
        self.fail_ladcoupledamageviscoelasticitys: Dict[int, Any] = {} # /FAIL/LAD_COUPLE_DAMAGE_VISCOELASTICITY (M330)
        self.eng_electromagnetomechanical_resonance_energies: Dict[int, Any] = {} # /ENG/ELECTROMAGNETOMECHANICAL_RESONANCE_ENERGY (M330)
        self.lagmul_wunderlich_linkage_joints: Dict[int, Any] = {}   # /WUNDERLICH_LINKAGE_JOINT, /LAGMUL/WUNDERLICH_LINKAGE_JOINT (M330)
        self.sensor_spring_total_angular_pop_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE (M330)

        # M331 Entities
        self.fail_laddynamiccrushrates: Dict[int, Any] = {}          # /FAIL/LAD_DYNAMIC_CRUSH_RATE (M331)
        self.eng_flexomagnetoelectric_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOELECTRIC_RESONANCE_ENERGY (M331)
        self.lagmul_delassus_linkage_joints: Dict[int, Any] = {}     # /DELASSUS_LINKAGE_JOINT, /LAGMUL/DELASSUS_LINKAGE_JOINT (M331)
        self.sensor_spring_normal_lock_rates: Dict[int, Any] = {}    # /SENSOR/SPRING_NORMAL_LOCK_RATE (M331)

        # M332 Entities
        self.fail_ladtransversecrushrates: Dict[int, Any] = {}       # /FAIL/LAD_TRANSVERSE_CRUSH_RATE (M332)
        self.eng_flexothermomagnetic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOMAGNETIC_RESONANCE_ENERGY (M332)
        self.lagmul_schatz_linkage_joints: Dict[int, Any] = {}       # /SCHATZ_LINKAGE_JOINT, /LAGMUL/SCHATZ_LINKAGE_JOINT (M332)
        self.sensor_spring_transverse_lock_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TRANSVERSE_LOCK_RATE (M332)

        # M333 Entities
        self.fail_ladcoupledynamiccrushs: Dict[int, Any] = {}        # /FAIL/LAD_COUPLE_DYNAMIC_CRUSH (M333)
        self.eng_flexothermoelectric_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOELECTRIC_RESONANCE_ENERGY (M333)
        self.lagmul_franke_linkage_joints: Dict[int, Any] = {}       # /FRANKE_LINKAGE_JOINT, /LAGMUL/FRANKE_LINKAGE_JOINT (M333)
        self.sensor_spring_total_lock_rates: Dict[int, Any] = {}     # /SENSOR/SPRING_TOTAL_LOCK_RATE (M333)

        # M334 Entities
        self.fail_ladcouplecrushrates: Dict[int, Any] = {}           # /FAIL/LAD_COUPLE_CRUSH_RATE (M334)
        self.eng_flexothermoacoustic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOACOUSTIC_RESONANCE_ENERGY (M334)
        self.lagmul_krames_linkage_joints: Dict[int, Any] = {}       # /KRAMES_LINKAGE_JOINT, /LAGMUL/KRAMES_LINKAGE_JOINT (M334)
        self.sensor_spring_torsional_lock_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TORSIONAL_LOCK_RATE (M334)

        # M335 Entities
        self.fail_laddynamicdelaminationrates: Dict[int, Any] = {}   # /FAIL/LAD_DYNAMIC_DELAMINATION_RATE (M335)
        self.eng_flexoelectromagnetic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOELECTROMAGNETIC_RESONANCE_ENERGY (M335)
        self.lagmul_borel_linkage_joints: Dict[int, Any] = {}        # /BOREL_LINKAGE_JOINT, /LAGMUL/BOREL_LINKAGE_JOINT (M335)
        self.sensor_spring_bending_lock_rates: Dict[int, Any] = {}   # /SENSOR/SPRING_BENDING_LOCK_RATE (M335)

        # M336 Entities
        self.fail_ladtransversedelaminationrates: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_DELAMINATION_RATE (M336)
        self.eng_flexoelectroacoustic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOELECTROACOUSTIC_RESONANCE_ENERGY (M336)
        self.lagmul_herve_linkage_joints: Dict[int, Any] = {}        # /HERVE_LINKAGE_JOINT, /LAGMUL/HERVE_LINKAGE_JOINT (M336)
        self.sensor_spring_total_angular_lock_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE (M336)

        # M337 Entities
        self.fail_ladcoupledelaminationrates: Dict[int, Any] = {}    # /FAIL/LAD_COUPLE_DELAMINATION_RATE (M337)
        self.eng_flexomagnetoacoustic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOACOUSTIC_RESONANCE_ENERGY (M337)
        self.lagmul_kong_linkage_joints: Dict[int, Any] = {}         # /KONG_LINKAGE_JOINT, /LAGMUL/KONG_LINKAGE_JOINT (M337)
        self.sensor_spring_normal_drop_rates: Dict[int, Any] = {}    # /SENSOR/SPRING_NORMAL_DROP_RATE (M337)

        # M338 Entities
        self.fail_laddynamicmicrobucklingrates: Dict[int, Any] = {}  # /FAIL/LAD_DYNAMIC_MICROBUCKLING_RATE (M338)
        self.eng_flexothermoelectromagnetic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOELECTROMAGNETIC_RESONANCE_ENERGY (M338)
        self.lagmul_hunt_linkage_joints: Dict[int, Any] = {}         # /HUNT_LINKAGE_JOINT, /LAGMUL/HUNT_LINKAGE_JOINT (M338)
        self.sensor_spring_transverse_drop_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TRANSVERSE_DROP_RATE (M338)

        # M339 Entities
        self.fail_ladtransversemicrobucklingrates: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_MICROBUCKLING_RATE (M339)
        self.eng_flexothermoelectroacoustic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOELECTROACOUSTIC_RESONANCE_ENERGY (M339)
        self.lagmul_baker_line_linkage_joints: Dict[int, Any] = {}   # /BAKER_LINE_LINKAGE_JOINT, /LAGMUL/BAKER_LINE_LINKAGE_JOINT (M339)
        self.sensor_spring_total_drop_rates: Dict[int, Any] = {}     # /SENSOR/SPRING_TOTAL_DROP_RATE (M339)

        # M340 Entities
        self.fail_ladcouplemicrobucklingrates: Dict[int, Any] = {}   # /FAIL/LAD_COUPLE_MICROBUCKLING_RATE (M340)
        self.eng_flexothermomagnetoacoustic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOMAGNETOACOUSTIC_RESONANCE_ENERGY (M340)
        self.lagmul_baker_plane_linkage_joints: Dict[int, Any] = {}  # /BAKER_PLANE_LINKAGE_JOINT, /LAGMUL/BAKER_PLANE_LINKAGE_JOINT (M340)
        self.sensor_spring_torsional_drop_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TORSIONAL_DROP_RATE (M340)

        # M341 Entities
        self.fail_laddynamicfibersplittingrates: Dict[int, Any] = {} # /FAIL/LAD_DYNAMIC_FIBER_SPLITTING_RATE (M341)
        self.eng_flexothermoelectromagnetoacoustic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOELECTROMAGNETOACOUSTIC_RESONANCE_ENERGY (M341)
        self.lagmul_wohlhart_hybrid_linkage_joints: Dict[int, Any] = {} # /WOHLHART_HYBRID_LINKAGE_JOINT, /LAGMUL/WOHLHART_HYBRID_LINKAGE_JOINT (M341)
        self.sensor_spring_bending_drop_rates: Dict[int, Any] = {}   # /SENSOR/SPRING_BENDING_DROP_RATE (M341)

        # M342 Entities
        self.fail_ladtransversefibersplittingrates: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_FIBER_SPLITTING_RATE (M342)
        self.eng_flexothermophotonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOPHOTONIC_RESONANCE_ENERGY (M342)
        self.lagmul_chen_linkage_joints: Dict[int, Any] = {}         # /CHEN_LINKAGE_JOINT, /LAGMUL/CHEN_LINKAGE_JOINT (M342)
        self.sensor_spring_total_angular_drop_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TOTAL_ANGULAR_DROP_RATE (M342)

        # M343 Entities
        self.fail_ladcouplefibersplittingrates: Dict[int, Any] = {}   # /FAIL/LAD_COUPLE_FIBER_SPLITTING_RATE (M343)
        self.eng_flexothermoplasmonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOPLASMONIC_RESONANCE_ENERGY (M343)
        self.lagmul_baker_symmetric_linkage_joints: Dict[int, Any] = {} # /BAKER_SYMMETRIC_LINKAGE_JOINT, /LAGMUL/BAKER_SYMMETRIC_LINKAGE_JOINT (M343)
        self.sensor_spring_normal_drift_rates: Dict[int, Any] = {}   # /SENSOR/SPRING_NORMAL_DRIFT_RATE (M343)

        # M344 Entities
        self.fail_laddynamicfibercrushingrates: Dict[int, Any] = {}  # /FAIL/LAD_DYNAMIC_FIBER_CRUSHING_RATE (M344)
        self.eng_flexothermoexcitonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOEXCITONIC_RESONANCE_ENERGY (M344)
        self.lagmul_altmann_spatial_linkage_joints: Dict[int, Any] = {} # /ALTMANN_SPATIAL_LINKAGE_JOINT, /LAGMUL/ALTMANN_SPATIAL_LINKAGE_JOINT (M344)
        self.sensor_spring_transverse_drift_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TRANSVERSE_DRIFT_RATE (M344)

        # M345 Entities
        self.fail_ladtransversefibercrushingrates: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_FIBER_CRUSHING_RATE (M345)
        self.eng_flexothermomagnonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOMAGNONIC_RESONANCE_ENERGY (M345)
        self.lagmul_dietmaier_spatial_linkage_joints: Dict[int, Any] = {} # /DIETMAIER_SPATIAL_LINKAGE_JOINT, /LAGMUL/DIETMAIER_SPATIAL_LINKAGE_JOINT (M345)
        self.sensor_spring_total_drift_rates: Dict[int, Any] = {}    # /SENSOR/SPRING_TOTAL_DRIFT_RATE (M345)

        # M346 Entities
        self.fail_ladcouplefibercrushingrates: Dict[int, Any] = {}   # /FAIL/LAD_COUPLE_FIBER_CRUSHING_RATE (M346)
        self.eng_flexothermomagnonpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOMAGNONPOLARITONIC_RESONANCE_ENERGY (M346)
        self.lagmul_wohlhart_spatial_linkage_joints: Dict[int, Any] = {} # /WOHLHART_SPATIAL_LINKAGE_JOINT, /LAGMUL/WOHLHART_SPATIAL_LINKAGE_JOINT (M346)
        self.sensor_spring_torsional_drift_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TORSIONAL_DRIFT_RATE (M346)

        # M347 Entities
        self.fail_laddynamicinterlaminarshearrates: Dict[int, Any] = {} # /FAIL/LAD_DYNAMIC_INTERLAMINAR_SHEAR_RATE (M347)
        self.eng_flexothermoplasmonpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOPLASMONPOLARITONIC_RESONANCE_ENERGY (M347)
        self.lagmul_hunt_spatial_linkage_joints: Dict[int, Any] = {} # /HUNT_SPATIAL_LINKAGE_JOINT, /LAGMUL/HUNT_SPATIAL_LINKAGE_JOINT (M347)
        self.sensor_spring_bending_drift_rates: Dict[int, Any] = {}  # /SENSOR/SPRING_BENDING_DRIFT_RATE (M347)

        # M348 Entities
        self.fail_ladtransverseinterlaminarshearrates: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_INTERLAMINAR_SHEAR_RATE (M348)
        self.eng_flexothermoexcitonpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOEXCITONPOLARITONIC_RESONANCE_ENERGY (M348)
        self.lagmul_chen_spatial_linkage_joints: Dict[int, Any] = {} # /CHEN_SPATIAL_LINKAGE_JOINT, /LAGMUL/CHEN_SPATIAL_LINKAGE_JOINT (M348)
        self.sensor_spring_total_angular_drift_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TOTAL_ANGULAR_DRIFT_RATE (M348)

        # M349 Entities
        self.fail_ladcoupleinterlaminarshearrates: Dict[int, Any] = {} # /FAIL/LAD_COUPLE_INTERLAMINAR_SHEAR_RATE (M349)
        self.eng_flexothermophononpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOPHONONPOLARITONIC_RESONANCE_ENERGY (M349)
        self.lagmul_baker_spatial_linkage_joints: Dict[int, Any] = {} # /BAKER_SPATIAL_LINKAGE_JOINT, /LAGMUL/BAKER_SPATIAL_LINKAGE_JOINT (M349)
        self.sensor_spring_normal_surge_rates: Dict[int, Any] = {}   # /SENSOR/SPRING_NORMAL_SURGE_RATE (M349)

        # M350 Entities
        self.fail_laddynamicinterlaminartensionrates: Dict[int, Any] = {} # /FAIL/LAD_DYNAMIC_INTERLAMINAR_TENSION_RATE (M350)
        self.eng_flexothermoplasmonphononpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOPLASMONPHONONPOLARITONIC_RESONANCE_ENERGY (M350)
        self.lagmul_waldron_spatial_linkage_joints: Dict[int, Any] = {} # /WALDRON_SPATIAL_LINKAGE_JOINT, /LAGMUL/WALDRON_SPATIAL_LINKAGE_JOINT (M350)
        self.sensor_spring_transverse_surge_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TRANSVERSE_SURGE_RATE (M350)

        # M351 Entities
        self.fail_ladtransverseinterlaminartensionrates: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_INTERLAMINAR_TENSION_RATE (M351)
        self.eng_flexothermoplasmonexcitonpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOPLASMONEXCITONPOLARITONIC_RESONANCE_ENERGY (M351)
        self.lagmul_bricard_spatial_linkage_joints: Dict[int, Any] = {} # /BRICARD_SPATIAL_LINKAGE_JOINT, /LAGMUL/BRICARD_SPATIAL_LINKAGE_JOINT (M351)
        self.sensor_spring_total_surge_rates: Dict[int, Any] = {}     # /SENSOR/SPRING_TOTAL_SURGE_RATE (M351)

        # M352 Entities
        self.fail_ladcoupleinterlaminartensionrates: Dict[int, Any] = {} # /FAIL/LAD_COUPLE_INTERLAMINAR_TENSION_RATE (M352)
        self.eng_flexothermoplasmonmagnonpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOPLASMONMAGNONPOLARITONIC_RESONANCE_ENERGY (M352)
        self.lagmul_bennett_spatial_linkage_joints: Dict[int, Any] = {} # /BENNETT_SPATIAL_LINKAGE_JOINT, /LAGMUL/BENNETT_SPATIAL_LINKAGE_JOINT (M352)
        self.sensor_spring_torsional_surge_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TORSIONAL_SURGE_RATE (M352)

        # M353 Entities
        self.fail_laddynamicmatrixmicrocrackingrates: Dict[int, Any] = {} # /FAIL/LAD_DYNAMIC_MATRIX_MICROCRACKING_RATE (M353)
        self.eng_flexothermoexcitonphononpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOEXCITONPHONONPOLARITONIC_RESONANCE_ENERGY (M353)
        self.lagmul_myard_spatial_linkage_joints: Dict[int, Any] = {} # /MYARD_SPATIAL_LINKAGE_JOINT, /LAGMUL/MYARD_SPATIAL_LINKAGE_JOINT (M353)
        self.sensor_spring_bending_surge_rates: Dict[int, Any] = {}   # /SENSOR/SPRING_BENDING_SURGE_RATE (M353)

        # M354 Entities
        self.fail_ladtransversematrixmicrocrackingrates: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_MATRIX_MICROCRACKING_RATE (M354)
        self.eng_flexothermoexcitonmagnonpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOEXCITONMAGNONPOLARITONIC_RESONANCE_ENERGY (M354)
        self.lagmul_goldberg_spatial_linkage_joints: Dict[int, Any] = {} # /GOLDBERG_SPATIAL_LINKAGE_JOINT, /LAGMUL/GOLDBERG_SPATIAL_LINKAGE_JOINT (M354)
        self.sensor_spring_total_angular_surge_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TOTAL_ANGULAR_SURGE_RATE (M354)

        # M355 Entities
        self.fail_ladcouplematrixmicrocrackingrates: Dict[int, Any] = {} # /FAIL/LAD_COUPLE_MATRIX_MICROCRACKING_RATE (M355)
        self.eng_flexothermophononmagnonpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY (M355)
        self.lagmul_sarrus_spatial_linkage_joints: Dict[int, Any] = {} # /SARRUS_SPATIAL_LINKAGE_JOINT, /LAGMUL/SARRUS_SPATIAL_LINKAGE_JOINT (M355)
        self.sensor_spring_normal_pop_rates: Dict[int, Any] = {}     # /SENSOR/SPRING_NORMAL_POP_RATE (M355)

        # M356 Entities
        self.fail_laddynamicfibercompressionkinkingrates: Dict[int, Any] = {} # /FAIL/LAD_DYNAMIC_FIBER_COMPRESSION_KINKING_RATE (M356)
        self.eng_flexothermoplasmonexcitonphononpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOPLASMONEXCITONPHONONPOLARITONIC_RESONANCE_ENERGY (M356)
        self.lagmul_delassus_spatial_linkage_joints: Dict[int, Any] = {} # /DELASSUS_SPATIAL_LINKAGE_JOINT, /LAGMUL/DELASSUS_SPATIAL_LINKAGE_JOINT (M356)
        self.sensor_spring_transverse_pop_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TRANSVERSE_POP_RATE (M356)

        # M357 Entities
        self.fail_ladtransversefibercompressionkinkingrates: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_FIBER_COMPRESSION_KINKING_RATE (M357)
        self.eng_flexothermoplasmonexcitonmagnonpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOPLASMONEXCITONMAGNONPOLARITONIC_RESONANCE_ENERGY (M357)
        self.lagmul_wohlhart_spatial_linkage_joints: Dict[int, Any] = {} # /WOHLHART_SPATIAL_LINKAGE_JOINT, /LAGMUL/WOHLHART_SPATIAL_LINKAGE_JOINT (M357)
        self.sensor_spring_total_pop_rates: Dict[int, Any] = {}      # /SENSOR/SPRING_TOTAL_POP_RATE (M357)

        # M358 Entities
        self.fail_ladcouplefibercompressionkinkingrates: Dict[int, Any] = {} # /FAIL/LAD_COUPLE_FIBER_COMPRESSION_KINKING_RATE (M358)
        self.eng_flexothermoplasmonphononmagnonpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOPLASMONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY (M358)
        self.lagmul_altmann_spatial_linkage_joints: Dict[int, Any] = {} # /ALTMANN_SPATIAL_LINKAGE_JOINT, /LAGMUL/ALTMANN_SPATIAL_LINKAGE_JOINT (M358)
        self.sensor_spring_torsional_pop_rates: Dict[int, Any] = {}  # /SENSOR/SPRING_TORSIONAL_POP_RATE (M358)

        # M359 Entities
        self.fail_laddynamicdelaminationmicrodebondingrates: Dict[int, Any] = {} # /FAIL/LAD_DYNAMIC_DELAMINATION_MICRODEBONDING_RATE (M359)
        self.eng_flexothermoexcitonphononmagnonpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOEXCITONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY (M359)
        self.lagmul_baker_spatial_linkage_joints: Dict[int, Any] = {} # /BAKER_SPATIAL_LINKAGE_JOINT, /LAGMUL/BAKER_SPATIAL_LINKAGE_JOINT (M359)
        self.sensor_spring_bending_pop_rates: Dict[int, Any] = {}    # /SENSOR/SPRING_BENDING_POP_RATE (M359)

        # M360 Entities
        self.fail_ladtransversedelaminationmicrodebondingrates: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_DELAMINATION_MICRODEBONDING_RATE (M360)
        self.eng_flexothermoplasmonexcitonphononmagnonpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOTHERMOPLASMONEXCITONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY (M360)
        self.lagmul_dietmaier_spatial_linkage_joints: Dict[int, Any] = {} # /DIETMAIER_SPATIAL_LINKAGE_JOINT, /LAGMUL/DIETMAIER_SPATIAL_LINKAGE_JOINT (M360)
        self.sensor_spring_total_angular_pop_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE (M360)

        # M361 Entities
        self.fail_ladcoupledelaminationmicrodebondingrates: Dict[int, Any] = {} # /FAIL/LAD_COUPLE_DELAMINATION_MICRODEBONDING_RATE (M361)
        self.eng_flexomagnetoplasmonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPLASMONIC_RESONANCE_ENERGY (M361)
        self.lagmul_waldron_spatial_linkage_joints: Dict[int, Any] = {} # /WALDRON_SPATIAL_LINKAGE_JOINT, /LAGMUL/WALDRON_SPATIAL_LINKAGE_JOINT (M361)
        self.sensor_spring_normal_crackle_rates: Dict[int, Any] = {} # /SENSOR/SPRING_NORMAL_CRACKLE_RATE (M361)

        # M362 Entities
        self.fail_laddynamicmatrixsheardegradationrates: Dict[int, Any] = {} # /FAIL/LAD_DYNAMIC_MATRIX_SHEAR_DEGRADATION_RATE (M362)
        self.eng_flexomagnetophononic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPHONONIC_RESONANCE_ENERGY (M362)
        self.lagmul_hunt_spatial_linkage_joints: Dict[int, Any] = {} # /HUNT_SPATIAL_LINKAGE_JOINT, /LAGMUL/HUNT_SPATIAL_LINKAGE_JOINT (M362)
        self.sensor_spring_transverse_crackle_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE (M362)

        # M363 Entities
        self.fail_ladtransversematrixsheardegradationrates: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_MATRIX_SHEAR_DEGRADATION_RATE (M363)
        self.eng_flexomagnetoexcitonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOEXCITONIC_RESONANCE_ENERGY (M363)
        self.lagmul_chen_spatial_linkage_joints: Dict[int, Any] = {} # /CHEN_SPATIAL_LINKAGE_JOINT, /LAGMUL/CHEN_SPATIAL_LINKAGE_JOINT (M363)
        self.sensor_spring_total_crackle_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TOTAL_CRACKLE_RATE (M363)

        # M364 Entities
        self.fail_ladcouplematrixsheardegradationrates: Dict[int, Any] = {} # /FAIL/LAD_COUPLE_MATRIX_SHEAR_DEGRADATION_RATE (M364)
        self.eng_flexomagnetopolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPOLARITONIC_RESONANCE_ENERGY (M364)
        self.lagmul_wunderlich_spatial_linkage_joints: Dict[int, Any] = {} # /WUNDERLICH_SPATIAL_LINKAGE_JOINT, /LAGMUL/WUNDERLICH_SPATIAL_LINKAGE_JOINT (M364)
        self.sensor_spring_torsional_crackle_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TORSIONAL_CRACKLE_RATE (M364)

        # M365 Entities
        self.fail_laddynamicfibertensionrupturerates: Dict[int, Any] = {} # /FAIL/LAD_DYNAMIC_FIBER_TENSION_RUPTURE_RATE (M365)
        self.eng_flexomagnetoplasmonicphonon_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPLASMONICPHONON_RESONANCE_ENERGY (M365)
        self.lagmul_konnok_spatial_linkage_joints: Dict[int, Any] = {} # /KONNOK_SPATIAL_LINKAGE_JOINT, /LAGMUL/KONNOK_SPATIAL_LINKAGE_JOINT (M365)
        self.sensor_spring_bending_crackle_rates: Dict[int, Any] = {} # /SENSOR/SPRING_BENDING_CRACKLE_RATE (M365)

        # M366 Entities
        self.fail_ladtransversefibertensionrupturerates: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_FIBER_TENSION_RUPTURE_RATE (M366)
        self.eng_flexomagnetoplasmonicexciton_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPLASMONICEXCITON_RESONANCE_ENERGY (M366)
        self.lagmul_pfurner_spatial_linkage_joints: Dict[int, Any] = {} # /PFURNER_SPATIAL_LINKAGE_JOINT, /LAGMUL/PFURNER_SPATIAL_LINKAGE_JOINT (M366)
        self.sensor_spring_total_angular_crackle_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE (M366)

        # M367 Entities
        self.fail_ladcouplefibertensionrupturerates: Dict[int, Any] = {} # /FAIL/LAD_COUPLE_FIBER_TENSION_RUPTURE_RATE (M367)
        self.eng_flexomagnetoplasmonicmagnon_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPLASMONICMAGNON_RESONANCE_ENERGY (M367)
        self.lagmul_phillips_spatial_linkage_joints: Dict[int, Any] = {} # /PHILLIPS_SPATIAL_LINKAGE_JOINT, /LAGMUL/PHILLIPS_SPATIAL_LINKAGE_JOINT (M367)
        self.sensor_spring_normal_snap_rates: Dict[int, Any] = {} # /SENSOR/SPRING_NORMAL_SNAP_RATE (M367)

        # M368 Entities
        self.fail_laddynamicfibercompressioncrushingrates: Dict[int, Any] = {} # /FAIL/LAD_DYNAMIC_FIBER_COMPRESSION_CRUSHING_RATE (M368)
        self.eng_flexomagnetoplasmonicpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE_ENERGY (M368)
        self.lagmul_stevens_spatial_linkage_joints: Dict[int, Any] = {} # /STEVENS_SPATIAL_LINKAGE_JOINT, /LAGMUL/STEVENS_SPATIAL_LINKAGE_JOINT (M368)
        self.sensor_spring_transverse_snap_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TRANSVERSE_SNAP_RATE (M368)

        # M369 Entities
        self.fail_ladtransversefibercompressioncrushingrates: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_FIBER_COMPRESSION_CRUSHING_RATE (M369)
        self.eng_flexomagnetophononicexcitonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPHONONICEXCITONIC_RESONANCE_ENERGY (M369)
        self.lagmul_baker_hybrid_spatial_linkage_joints: Dict[int, Any] = {} # /BAKER_HYBRID_SPATIAL_LINKAGE_JOINT, /LAGMUL/BAKER_HYBRID_SPATIAL_LINKAGE_JOINT (M369)
        self.sensor_spring_total_snap_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TOTAL_SNAP_RATE (M369)

        # M370 Entities
        self.fail_ladcouplefibercompressioncrushingrates: Dict[int, Any] = {} # /FAIL/LAD_COUPLE_FIBER_COMPRESSION_CRUSHING_RATE (M370)
        self.eng_flexomagnetophononicmagnonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPHONONICMAGNONIC_RESONANCE_ENERGY (M370)
        self.lagmul_waldron_hybrid_spatial_linkage_joints: Dict[int, Any] = {} # /WALDRON_HYBRID_SPATIAL_LINKAGE_JOINT, /LAGMUL/WALDRON_HYBRID_SPATIAL_LINKAGE_JOINT (M370)
        self.sensor_spring_torsional_snap_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TORSIONAL_SNAP_RATE (M370)

        # M371 Entities
        self.fail_laddynamicinterlaminarsheardelaminationrates: Dict[int, Any] = {} # /FAIL/LAD_DYNAMIC_INTERLAMINAR_SHEAR_DELAMINATION_RATE (M371)
        self.eng_flexomagnetophononicpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPHONONICPOLARITONIC_RESONANCE_ENERGY (M371)
        self.lagmul_chen_hybrid_spatial_linkage_joints: Dict[int, Any] = {} # /CHEN_HYBRID_SPATIAL_LINKAGE_JOINT, /LAGMUL/CHEN_HYBRID_SPATIAL_LINKAGE_JOINT (M371)
        self.sensor_spring_bending_snap_rates: Dict[int, Any] = {} # /SENSOR/SPRING_BENDING_SNAP_RATE (M371)

        # M372 Entities
        self.fail_ladtransverseinterlaminarsheardelaminationrates: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_INTERLAMINAR_SHEAR_DELAMINATION_RATE (M372)
        self.eng_flexomagnetoexcitonicmagnonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOEXCITONICMAGNONIC_RESONANCE_ENERGY (M372)
        self.lagmul_wohlhart_hybrid_spatial_linkage_joints: Dict[int, Any] = {} # /WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT, /LAGMUL/WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT (M372)
        self.sensor_spring_total_angular_snap_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TOTAL_ANGULAR_SNAP_RATE (M372)

        # M373 Entities
        self.fail_ladcoupleinterlaminarsheardelaminationrates: Dict[int, Any] = {} # /FAIL/LAD_COUPLE_INTERLAMINAR_SHEAR_DELAMINATION_RATE (M373)
        self.eng_flexomagnetoexcitonicpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE_ENERGY (M373)
        self.lagmul_maverick_hybrid_spatial_linkage_joints: Dict[int, Any] = {} # /MAVERICK_HYBRID_SPATIAL_LINKAGE_JOINT, /LAGMUL/MAVERICK_HYBRID_SPATIAL_LINKAGE_JOINT (M373)
        self.sensor_spring_normal_pop_rates: Dict[int, Any] = {} # /SENSOR/SPRING_NORMAL_POP_RATE (M373)

        # M374 Entities
        self.fail_laddynamicinterlaminarnormalpeelingrates: Dict[int, Any] = {} # /FAIL/LAD_DYNAMIC_INTERLAMINAR_NORMAL_PEELING_RATE (M374)
        self.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONIC_RESONANCE_ENERGY (M374)
        self.lagmul_krause_hybrid_spatial_linkage_joints: Dict[int, Any] = {} # /KRAUSE_HYBRID_SPATIAL_LINKAGE_JOINT, /LAGMUL/KRAUSE_HYBRID_SPATIAL_LINKAGE_JOINT (M374)
        self.sensor_spring_transverse_pop_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TRANSVERSE_POP_RATE (M374)

        # M375 Entities
        self.fail_ladtransverseinterlaminarnormalpeelingrates: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_INTERLAMINAR_NORMAL_PEELING_RATE (M375)
        self.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY (M375)
        self.lagmul_sturgess_hybrid_spatial_linkage_joints: Dict[int, Any] = {} # /STURGESS_HYBRID_SPATIAL_LINKAGE_JOINT, /LAGMUL/STURGESS_HYBRID_SPATIAL_LINKAGE_JOINT (M375)
        self.sensor_spring_total_pop_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TOTAL_POP_RATE (M375)

        # M376 Entities
        self.fail_ladcoupleinterlaminarnormalpeelingrates: Dict[int, Any] = {} # /FAIL/LAD_COUPLE_INTERLAMINAR_NORMAL_PEELING_RATE (M376)
        self.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY (M376)
        self.lagmul_bevan_hybrid_spatial_linkage_joints: Dict[int, Any] = {} # /BEVAN_HYBRID_SPATIAL_LINKAGE_JOINT, /LAGMUL/BEVAN_HYBRID_SPATIAL_LINKAGE_JOINT (M376)
        self.sensor_spring_torsional_pop_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TORSIONAL_POP_RATE (M376)

        # M377 Entities
        self.fail_laddynamicfibermatrixdebondingrates: Dict[int, Any] = {} # /FAIL/LAD_DYNAMIC_FIBER_MATRIX_DEBONDING_RATE (M377)
        self.eng_flexomagnetophononicexcitonicmagnonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPHONONICEXCITONICMAGNONIC_RESONANCE_ENERGY (M377)
        self.lagmul_heinrichs_hybrid_spatial_linkage_joints: Dict[int, Any] = {} # /HEINRICHS_HYBRID_SPATIAL_LINKAGE_JOINT, /LAGMUL/HEINRICHS_HYBRID_SPATIAL_LINKAGE_JOINT (M377)
        self.sensor_spring_bending_pop_rates: Dict[int, Any] = {} # /SENSOR/SPRING_BENDING_POP_RATE (M377)

        # M378 Entities
        self.fail_ladtransversefibermatrixdebondingrates: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_FIBER_MATRIX_DEBONDING_RATE (M378)
        self.eng_flexomagnetophononicexcitonicpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPHONONICEXCITONICPOLARITONIC_RESONANCE_ENERGY (M378)
        self.lagmul_altmann_hybrid_spatial_linkage_joints: Dict[int, Any] = {} # /ALTMANN_HYBRID_SPATIAL_LINKAGE_JOINT, /LAGMUL/ALTMANN_HYBRID_SPATIAL_LINKAGE_JOINT (M378)
        self.sensor_spring_total_angular_pop_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE (M378)

        # M379 Entities
        self.fail_ladcouplefibermatrixdebondingrates: Dict[int, Any] = {} # /FAIL/LAD_COUPLE_FIBER_MATRIX_DEBONDING_RATE (M379)
        self.eng_flexomagnetophononicmagnonicpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY (M379)
        self.lagmul_kirkpatrick_hybrid_spatial_linkage_joints: Dict[int, Any] = {} # /KIRKPATRICK_HYBRID_SPATIAL_LINKAGE_JOINT, /LAGMUL/KIRKPATRICK_HYBRID_SPATIAL_LINKAGE_JOINT (M379)
        self.sensor_spring_normal_lock_rates: Dict[int, Any] = {} # /SENSOR/SPRING_NORMAL_LOCK_RATE (M379)

        # M380 Entities
        self.fail_laddynamicplymicrocrackingrates: Dict[int, Any] = {} # /FAIL/LAD_DYNAMIC_PLY_MICRO_CRACKING_RATE (M380)
        self.eng_flexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY (M380)
        self.lagmul_alexander_hybrid_spatial_linkage_joints: Dict[int, Any] = {} # /ALEXANDER_HYBRID_SPATIAL_LINKAGE_JOINT, /LAGMUL/ALEXANDER_HYBRID_SPATIAL_LINKAGE_JOINT (M380)
        self.sensor_spring_transverse_lock_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TRANSVERSE_LOCK_RATE (M380)

        # M381 Entities
        self.fail_ladtransverseplymicrocrackingrates: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_PLY_MICRO_CRACKING_RATE (M381)
        self.eng_flexomagnetophononicplasmonicmagnonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPHONONICPLASMONICMAGNONIC_RESONANCE_ENERGY (M381)
        self.lagmul_chung_hybrid_spatial_linkage_joints: Dict[int, Any] = {} # /CHUNG_HYBRID_SPATIAL_LINKAGE_JOINT, /LAGMUL/CHUNG_HYBRID_SPATIAL_LINKAGE_JOINT (M381)
        self.sensor_spring_total_lock_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TOTAL_LOCK_RATE (M381)

        # M382 Entities
        self.fail_ladcoupleplymicrocrackingrates: Dict[int, Any] = {} # /FAIL/LAD_COUPLE_PLY_MICRO_CRACKING_RATE (M382)
        self.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY (M382)
        self.lagmul_stevens_hybrid_spatial_linkage_joints: Dict[int, Any] = {} # /STEVENS_HYBRID_SPATIAL_LINKAGE_JOINT, /LAGMUL/STEVENS_HYBRID_SPATIAL_LINKAGE_JOINT (M382)
        self.sensor_spring_torsional_lock_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TORSIONAL_LOCK_RATE (M382)

        # M383 Entities
        self.fail_laddynamicmatrixmicrofissuringrates: Dict[int, Any] = {} # /FAIL/LAD_DYNAMIC_MATRIX_MICRO_FISSURING_RATE (M383)
        self.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONIC_RESONANCE_ENERGY (M383)
        self.lagmul_baker_spatial_linkage_joints: Dict[int, Any] = {} # /BAKER_SPATIAL_LINKAGE_JOINT, /LAGMUL/BAKER_SPATIAL_LINKAGE_JOINT (M383)
        self.sensor_spring_bending_lock_rates: Dict[int, Any] = {} # /SENSOR/SPRING_BENDING_LOCK_RATE (M383)

        # M384 Entities
        self.fail_ladtransversematrixmicrofissuringrates: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_MATRIX_MICRO_FISSURING_RATE (M384)
        self.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY (M384)
        self.lagmul_dietmeier_spatial_linkage_joints: Dict[int, Any] = {} # /DIETMEIER_SPATIAL_LINKAGE_JOINT, /LAGMUL/DIETMEIER_SPATIAL_LINKAGE_JOINT (M384)
        self.sensor_spring_total_angular_lock_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE (M384)

        # M385 Entities
        self.fail_ladcouplematrixmicrofissuringrates: Dict[int, Any] = {} # /FAIL/LAD_COUPLE_MATRIX_MICRO_FISSURING_RATE (M385)
        self.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY (M385)
        self.lagmul_hunt_spatial_linkage_joints: Dict[int, Any] = {} # /HUNT_SPATIAL_LINKAGE_JOINT, /LAGMUL/HUNT_SPATIAL_LINKAGE_JOINT (M385)
        self.sensor_spring_normal_drop_rates: Dict[int, Any] = {} # /SENSOR/SPRING_NORMAL_DROP_RATE (M385)

        # M386 Entities
        self.fail_laddynamicmatrixmicrocrushingrates: Dict[int, Any] = {} # /FAIL/LAD_DYNAMIC_MATRIX_MICRO_CRUSHING_RATE (M386)
        self.eng_flexomagnetophononicexcitonicmagnonicpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPHONONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY (M386)
        self.lagmul_pfurner_spatial_linkage_joints: Dict[int, Any] = {} # /PFURNER_SPATIAL_LINKAGE_JOINT, /LAGMUL/PFURNER_SPATIAL_LINKAGE_JOINT (M386)
        self.sensor_spring_transverse_drop_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TRANSVERSE_DROP_RATE (M386)

        # M387 Entities
        self.fail_ladtransversematrixmicrocrushingrates: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_MATRIX_MICRO_CRUSHING_RATE (M387)
        self.eng_flexomagnetophononicplasmonicexcitonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONIC_RESONANCE_ENERGY (M387)
        self.lagmul_phillips_spatial_linkage_joints: Dict[int, Any] = {} # /PHILLIPS_SPATIAL_LINKAGE_JOINT, /LAGMUL/PHILLIPS_SPATIAL_LINKAGE_JOINT (M387)
        self.sensor_spring_total_drop_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TOTAL_DROP_RATE (M387)

        # M388 Entities
        self.fail_ladcouplematrixmicrocrushingrates: Dict[int, Any] = {} # /FAIL/LAD_COUPLE_MATRIX_MICRO_CRUSHING_RATE (M388)
        self.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY (M388)
        self.lagmul_konnok_spatial_linkage_joints: Dict[int, Any] = {} # /KONNOK_SPATIAL_LINKAGE_JOINT, /LAGMUL/KONNOK_SPATIAL_LINKAGE_JOINT (M388)
        self.sensor_spring_torsional_drop_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TORSIONAL_DROP_RATE (M388)

        # M389 Entities
        self.fail_laddynamicfibermicrobucklingrates: Dict[int, Any] = {} # /FAIL/LAD_DYNAMIC_FIBER_MICRO_BUCKLING_RATE (M389)
        self.eng_flexomagnetophononicplasmonicmagnonicpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPHONONICPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY (M389)
        self.lagmul_wohlhart_hybrid_spatial_linkage_joints: Dict[int, Any] = {} # /WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT, /LAGMUL/WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT (M389)
        self.sensor_spring_bending_drop_rates: Dict[int, Any] = {} # /SENSOR/SPRING_BENDING_DROP_RATE (M389)

        # M390 Entities
        self.fail_ladtransversefibermicrobucklingrates: Dict[int, Any] = {} # /FAIL/LAD_TRANSVERSE_FIBER_MICRO_BUCKLING_RATE (M390)
        self.eng_flexomagnetophononicplasmonicexcitonicpolaritonic_resonance_energies: Dict[int, Any] = {} # /ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY (M390)
        self.lagmul_maverick_spatial_linkage_joints: Dict[int, Any] = {} # /MAVERICK_SPATIAL_LINKAGE_JOINT, /LAGMUL/MAVERICK_SPATIAL_LINKAGE_JOINT (M390)
        self.sensor_spring_total_angular_drop_rates: Dict[int, Any] = {} # /SENSOR/SPRING_TOTAL_ANGULAR_DROP_RATE (M390)






































































































    # ----------------------------------------------------------------------
    # Node handling
    # ----------------------------------------------------------------------
    def add_nodes(self, ids: np.ndarray, xyz: np.ndarray) -> None:
        """Append /NODE data (may be called several times: multiple /NODE
        blocks and #include'd mesh files are the norm in real decks)."""
        start = len(self.node_ids)
        self.node_ids = np.concatenate([self.node_ids, ids.astype(np.int64)])
        self.x0 = np.vstack([self.x0, xyz])
        
        sub_id = self.active_submodels[-1] if self.active_submodels else 0
        self.node_submodel = np.concatenate([
            self.node_submodel, 
            np.full(len(ids), sub_id, dtype=np.int32)
        ])
        
        for k, nid in enumerate(ids):
            self._id2idx[int(nid)] = start + k

    def node_index(self, user_id: int) -> int:
        """USR2SYS: user node ID -> dense 0-based index (KeyError if absent)."""
        return self._id2idx[user_id]

    def node_indices(self, user_ids) -> np.ndarray:
        return np.array([self._id2idx[int(u)] for u in user_ids], dtype=np.int64)

    @property
    def numnod(self) -> int:
        return len(self.node_ids)

    # ----------------------------------------------------------------------
    def element_groups(self):
        """Iterate (name, group) over the non-empty element groups."""
        for name in ("bricks", "bricks_heph", "bric20s", "quads", "tetras", "tetra10s", "shel16s", "shells", "shells_qbat",
                     "shells_qeph", "sh3n", "sh3n_dkt18", "trusses", "springs", "beams"):
            g = getattr(self, name)
            if g is not None and g.n:
                yield name, g

    @property
    def frames(self) -> dict:
        return {sf.id: sf for sf in self.skews.entries if sf.kind == "FRAME"}


