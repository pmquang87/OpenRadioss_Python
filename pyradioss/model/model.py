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
    ShellPartPerturbation, FailurePerturbation, SphGlobal, SmsGlobal,
    BcsNrf, BcsWall, RigidLink, CylJoint, GeneralJoint,
    MergeNode, MergeRbody, IniCrack, IniCrackSegment, LaserLoad,
    PcylLoad, PfluidLoad, Preload, PreloadAxial, DampInter, DampRange,
    AnalyGlobal, UpwindGlobal, CaaControl,
    Gauge, Cluster, ExtLink, FxBody, IniGrav, IniMap1D, IniMap2D, IniStateFile,
    MonvolPres, MonvolGas, MonvolCommu1, MonvolLFluid, LeakMat,
    AleGrid, AleLink, AleSolver, AleClose,
    Retractor, Slipring, UserWindow,
    DetonationWave, ElementActivation, MonvolFvmBag2, Autoposition,
    LoadCentri, LoadPfluid, LoadPressure, InivelAxis, InivelFvm, InivelNode,
    ImpdispFgeo, ImpvelFgeo, RwallTherm, SphInOut,
    SphBcs, MadymoLink, MadymoExfem,
    AleGridDonea, AleGridSpring, AleGridStandard, AleGridDisp, AleGridLaplacian, AleGridVolume,
    AdmeshGlobal, StampingInit, RandomNoise, Accelerometer, Subset,
    FailComposite, EbcsPropellant, AdmasNonUniform, AdmasNonUniformItem,
    SectCircle, SectParal, DynainShell, MonvolArea, StateDt,
    SphReserve, MoveFunct,
    EigenMode, StressFile, MemoryRequest,
    FailFractal, TransformPosition, ExternalLink, ArchSpec,
    FunctPython, FrictionModel, FrictionPartPair, RefstaNode, ErefSpec,
    NbcsBlock, NbcsNode, AleMuscl, BemModel,
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
    th_dt: float = 0.0            # /TFILE time-history output period
    anim_dt: float = 0.0          # /ANIM/DT animation state period
    state_dt: float = 0.0         # /STATE/DT restart-snapshot period (M6)
    state_tstart: float = 0.0     # /STATE/DT first snapshot time
    print_cycles: int = 100       # /PRINT listing frequency (cycles)
    energy_error_stop: float = 15.0  # %, /STOP-like divergence guard
    anim_vect: List[str] = field(default_factory=lambda: ["VEL", "DIS"])
    anim_elem: List[str] = field(default_factory=lambda: ["VONM", "EPSP"])

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
        # raw (id, part_id, node ids...) tuples collected during parsing,
        # converted to ElementGroups in Starter finalization:
        self.raw_elems: Dict[str, list] = {
            "BRICK": [], "QUAD": [], "TETRA4": [], "TETRA10": [], "SHELL": [], "SH3N": [],
            "TRUSS": [], "SPRING": [], "BEAM": [], "SHEL16": []}

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
        self.leak_mats: Dict[int, LeakMat] = {}        # /LEAK (M105)
        self.ale_grids: Dict[int, AleGrid] = {}        # /ALE/GRID (M105)
        self.ale_links: Dict[int, AleLink] = {}        # /ALE/LINK (M105)
        self.ale_solver: Optional[AleSolver] = None    # /ALE/SOLVER (M105)
        self.ale_close: Optional[AleClose] = None      # /ALE/CLOS (M105)
        self.retractors: Dict[int, Retractor] = {}     # /RETRACTOR (M106)
        self.sliprings: Dict[int, Slipring] = {}       # /SLIPRING (M106)
        self.user_windows: List[UserWindow] = []       # /USERWI (M106)
        self.drapes: Dict[int, Drape] = {}             # /DRAPE (M107)
        self.inibri_erefs: List[IniBriEref] = []       # /INIBRI/EREF (M107)
        self.dyna_includes: List[IncludeDyna] = []     # /INCLUDE_DYNA (M107)
        self.monvol_fvmbags: Dict[int, MonvolFvmBag1] = {} # /MONVOL/FVMBAG1 (M108)
        self.detonations: List[DetonationWave] = []    # /INIT/DET_* (M110)
        self.activations: List[ElementActivation] = [] # /ACTIV (M110)
        self.monvol_fvmbag2s: Dict[int, MonvolFvmBag2] = {} # /MONVOL/FVMBAG2 (M111)
        self.autopositions: List[Autoposition] = []    # /TRANSFORM/AUTOPOSITION (M111)
        self.load_centris: Dict[int, LoadCentri] = {}       # /LOAD/CENTRI (M112)
        self.load_pfluids: Dict[int, LoadPfluid] = {}       # /LOAD/PFLUID (M112)
        self.load_pressures: Dict[int, LoadPressure] = {}   # /LOAD/PRESSURE (M112)
        self.inivel_axes: Dict[int, InivelAxis] = {}        # /INIVEL/AXIS (M112)
        self.inivel_fvms: Dict[int, InivelFvm] = {}         # /INIVEL/FVM (M112)
        self.inivel_nodes: Dict[int, InivelNode] = {}       # /INIVEL/NODE (M112)
        self.impdisp_fgeos: Dict[int, ImpdispFgeo] = {}     # /IMPDISP/FGEO (M112)
        self.impvel_fgeos: Dict[int, ImpvelFgeo] = {}       # /IMPVEL/FGEO (M112)
        self.rwall_therms: Dict[int, RwallTherm] = {}       # /RWALL/THERM (M112)
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
        self.th_requests: List[THRequest] = []

        self.title: str = "pyradioss model"

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
        for name in ("bricks", "bricks_heph", "quads", "tetras", "tetra10s", "shel16s", "shells", "shells_qbat",
                     "shells_qeph", "sh3n", "sh3n_dkt18", "trusses", "springs", "beams"):
            g = getattr(self, name)
            if g is not None and g.n:
                yield name, g
