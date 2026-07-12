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
    AddedMass, BoundaryCondition, Box, ConcentratedLoad, Damping, Gravity,
    ImposedDisplacement, ImposedVelocity, InitialVelocity, Interface, Line,
    Material, Mpc, NodeGroup, Part, PressureLoad, Property, Rbe3, RigidBody,
    RigidWall, Section, Sensor, Surface, THRequest,
)
from ..common.tables import FunctTable


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
    th_dt: float = 0.0            # /TFILE time-history output period
    anim_dt: float = 0.0          # /ANIM/DT animation state period
    state_dt: float = 0.0         # /STATE/DT restart-snapshot period (M6)
    state_tstart: float = 0.0     # /STATE/DT first snapshot time
    print_cycles: int = 100       # /PRINT listing frequency (cycles)
    energy_error_stop: float = 15.0  # %, /STOP-like divergence guard
    anim_vect: List[str] = field(default_factory=lambda: ["VEL", "DIS"])
    anim_elem: List[str] = field(default_factory=lambda: ["VONM", "EPSP"])

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
    impl_dt_itw: int = 6          # NL_DTP — target iterations (grow below)
    impl_dt_scaleup: float = 1.1  # SCAL_DTP — growth factor per easy step
    impl_dt_scaledn: float = 0.5  # SCAL_DTN — cut factor on non-convergence
    impl_dt_min: float = 0.0      # /IMPL/DT/STOP dt_min (0 = dtini * 1e-4)
    impl_dt_max: float = 0.0      # /IMPL/DT/STOP dt_max (0 = dtini)
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
    impl_fatig_bwcorr: bool = True   # Benasciutti-Tovo bandwidth attenuation
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
        self.tetras: Optional[ElementGroup] = None    # /TETRA4 (IXS10 kin)
        self.shells: Optional[ElementGroup] = None    # /SHELL  (IXC)
        self.sh3n: Optional[ElementGroup] = None      # /SH3N   (IXTG)
        self.trusses: Optional[ElementGroup] = None   # /TRUSS  (IXT)
        self.springs: Optional[ElementGroup] = None   # /SPRING (IXR)
        self.beams: Optional[ElementGroup] = None     # /BEAM   (IXP)
        # raw (id, part_id, node ids...) tuples collected during parsing,
        # converted to ElementGroups in Starter finalization:
        self.raw_elems: Dict[str, list] = {
            "BRICK": [], "TETRA4": [], "SHELL": [], "SH3N": [],
            "TRUSS": [], "SPRING": [], "BEAM": []}

        # ------------------------------------------------------------------
        # Definitions keyed by user id
        # ------------------------------------------------------------------
        self.materials: Dict[int, Material] = {}
        # /FAIL cards awaiting attachment to their material: parsed as
        # (mat_id, FailureModel, source) tuples, attached by the Starter
        # resolve step (deck order between /MAT and /FAIL is free).
        self.raw_fails: list = []
        # /EOS cards, same pattern (M6): (mat_id, EquationOfState, source)
        self.raw_eos: list = []
        self.properties: Dict[int, Property] = {}
        self.parts: Dict[int, Part] = {}
        self.parts_list: List[Part] = []          # dense order for elements
        self.functions: Dict[int, FunctTable] = {}
        self.node_groups: Dict[int, NodeGroup] = {}
        self.surfaces: Dict[int, Surface] = {}
        self.lines: Dict[int, Line] = {}
        self.boxes: Dict[int, Box] = {}

        # Loads / constraints / contacts
        self.bcs: List[BoundaryCondition] = []
        self.inivel: List[InitialVelocity] = []
        self.gravity: List[Gravity] = []
        self.cloads: List[ConcentratedLoad] = []
        self.impvel: List[ImposedVelocity] = []
        self.impdisp: List[ImposedDisplacement] = []   # /IMPDISP (M5)
        self.ploads: List[PressureLoad] = []           # /PLOAD   (M5)
        self.admas: List[AddedMass] = []               # /ADMAS   (M5)
        self.rwalls: List[RigidWall] = []
        self.rbodies: List[RigidBody] = []             # /RBODY + /RBE2 (M5)
        self.rbe3: List[Rbe3] = []                     # /RBE3    (M5)
        self.sections: List[Section] = []              # /SECT    (M5)
        self.damps: List[Damping] = []                 # /DAMP    (M6)
        self.sensors: List[Sensor] = []                # /SENSOR  (M6)
        self.mpcs: List[Mpc] = []                      # /MPC     (M6)
        self.interfaces: List[Interface] = []
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
        for name in ("bricks", "tetras", "shells", "sh3n",
                     "trusses", "springs", "beams"):
            g = getattr(self, name)
            if g is not None and g.n:
                yield name, g
