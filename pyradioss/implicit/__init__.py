"""
pyradioss.implicit — implicit (Newton–Raphson) analysis (M8).

Fortran origin: ``engine/source/implicit/`` — the whole implicit branch of
OpenRadioss, which lives entirely apart from the explicit ``resol.F`` loop:

    engine/source/implicit/imp_solv.F      the implicit driver / load steps
                                           (+ its /IMPL/NONLIN updated-
                                           Lagrangian and arc-length branch)
    engine/source/implicit/imp_buck.F      buckling eigen-extraction
                                           (/IMPL/BUCKL) + Newton bookkeeping
    engine/source/implicit/ind_glob_k.F    global equation numbering (DOFs)
    engine/source/implicit/imp_glob_k.F    sparse tangent assembly (material
                                           KE + the imp_kgeo geometric-
                                           stiffness branch)
    engine/source/implicit/imp_dsolv*.F    the direct linear-solver interface
                                           (Radioss wraps MUMPS / a built-in
                                           sparse LDL^T here)
    engine/source/implicit/imp_dyna.F      implicit DYNAMICS (/IMPL/DYNA):
                                           Newmark/HHT setup, dynamic
                                           residual + effective stiffness,
                                           a/v recovery (M10 →
                                           ``dynamics.py``)

The whole port was EXPLICIT before this milestone: leap-frog central
differences, a lumped (diagonal) mass, and NO global matrix ever assembled
(``pyradioss/engine/engine.py``). M8 adds a *parallel* implicit driver — it
does not touch the explicit loop. It REUSES the element force kernels for the
internal-force vector (the residual) and adds one NEW piece per element: the
element TANGENT STIFFNESS (``tangent()`` alongside ``forces()``), assembled
into a global sparse matrix and factorized by a direct solver.

What M8 ports (implicit **statics** first)
------------------------------------------
* **Equation numbering** (``dofmap.py``): every free nodal DOF gets an
  index; /BCS-fixed DOFs are *condensed out* of the system (removed, not
  penalized) — the implicit analogue of the model's dense index bookkeeping.
  Shell rotations are numbered where they carry stiffness; solid nodes get
  translations only.
* **Sparse tangent assembly** (``assembly.py``): each element returns its
  element tangent as COO triplets in global-DOF space; they scatter into a
  ``scipy.sparse`` CSR matrix. scipy is imported **inside** this package
  (see ``require_scipy`` below) so the base explicit install stays
  NumPy-only — scipy is required for implicit, optional for everything else.
* **Consistent tangents** (in the element/material modules): LAW1 elastic
  (the material tangent is just C) for hexa8 and BT4, and the LAW2 radial
  return's CONSISTENT (algorithmic) elastoplastic tangent — the one that
  gives Newton its quadratic convergence (see ``materials.law02`` and the
  derivation there).
* **Newton–Raphson** (``statics.py``): load stepping, residual
  R = f_ext - f_int (f_int from the EXISTING kernels), K Δu = R via the
  direct solver, displacement/residual convergence norms and a hard cap on
  iterations with a clear non-convergence stop.
* **Direct linear solver** (``linsolve.py``): a small ``solve(K, R)``
  interface behind the exact same backend-selection pattern as the M7
  compute backends — ``scipy.sparse.linalg.splu`` (SuperLU) is the default
  (no extra dependency); optional CHOLMOD (scikit-sparse) and MUMPS
  (python-mumps) are selected by env var / CLI flag and fall back to SuperLU
  with a warning when their library is absent.

What M9 adds (implicit NONLINEAR GEOMETRY — see statics.py for the theory)
--------------------------------------------------------------------------
* **Geometric (initial-stress) stiffness K_geo** per element (``kgeo()`` in
  hexa8 / BT4 / truss): the stress-dependent tangent term that carries
  stress stiffening, compression softening and buckling. Added to the
  material+hourglass tangent when ``/IMPL/NONLIN`` is active; identically
  zero at zero stress, so the M8 small-strain results are untouched.
* **Updated-Lagrangian reference frame**: the committed geometry ADVANCES to
  the deformed configuration between increments; within an increment the
  stress integrates at the midpoint geometry (Hughes–Winget — exact for
  finite rigid rotations) and the force is assembled on the end geometry
  (each element's ``static_internal_forces``).
* **Arc-length continuation** (``/IMPL/ARCL``, Crisfield's cylindrical
  method): the load factor becomes an unknown constrained by the step
  length, so the driver traces THROUGH limit points (snap-through) where
  load control necessarily fails.
* **Linearized buckling** (``buckling.py``, the imp_buck.F analogue): the
  (K_mat + mu*K_geo) phi = 0 eigenproblem on a pre-stressed state — the
  Euler-column validation path.
* **Truss implicit tangent** (exact corotational, LAW1): the textbook
  geometric-nonlinearity element, used by the snap-through validation.

What M10 adds (implicit DYNAMICS — see dynamics.py for the theory)
------------------------------------------------------------------
* **Newmark-beta / HHT-alpha time integration** (``dynamics.py``, the
  imp_dyna.F analogue, ``/IMPL/DYNA/1|2``): the lumped starter mass /
  inertia condensed to equation space, the HHT-weighted dynamic residual
  R = (1+a)(f_ext + f_int)_{n+1} - a(...)_n - M a_{n+1}, the effective
  tangent K_eff = (1+a) K_T + M/(beta dt^2), and the Newmark a/v recovery
  from each converged displacement increment — on top of the statics
  residual/commit machinery, under BOTH geometry modes (M8 linear and M9
  /IMPL/NONLIN). /RUN's time is physical again; rate devices (bulk
  viscosity, LAW2 strain-rate term) are disabled explicitly rather than
  fed the pseudo-velocity (see the dynamics.py docstring).

What M11 adds (implicit COMPLETENESS)
-------------------------------------
* **Element tangent completeness**: tangent()/kgeo()/
  static_internal_forces() for tetra4, sh3n, beam and spring — every
  element family of the port is implicit-capable and the assembler gate
  admits full mixed models. The spring (total-form) and the LAW2 truss
  (whose explicit one-step return cannot serve implicit increment sizes)
  carry their OWN implicit residual, ``implicit_internal_forces``,
  dispatched by ``statics._internal_forces`` instead of forces().
* **LAW2 consistent tangents for shells and the truss**: the plane-stress
  Iplas=2 radial projection's algorithmic tangent
  (``law02.consistent_shell_tangent``), integrated per layer with the
  force path's own quadrature; the truss's E·H/(E+H) modulus.
* **/IMPL/DYNA/DAMP Rayleigh damping** (imp_dyna.F IDY_DAMP): C = a·M +
  b·K(step start) in the HHT residual and the effective tangent, with the
  DY_EDAMP trapezoidal dissipation ledger (see dynamics.py).
* **Automatic implicit step control** (imp_dt.F — ``statics.StepControl``):
  cut-and-retry on non-convergence, growth back toward /IMPL/DTINI after
  easy steps, for statics increments AND dynamics time steps.
* **The /IMPL/BUCKL engine card**: prestress increments + the M9
  eigensolver, factors/modes reported in the listing and on the result.

What M12 adds (implicit CONSTRAINTS & CONTACT)
----------------------------------------------
* **Kinematic constraints by CONDENSATION** (``constraints.py`` — the
  rby_imp0.F / rbe2_imp0.F / rbe3_imp0.F / i2_imp1.F block condensations
  as one sparse transform): /RBODY, /RBE2, /INTER/TYPE2 tied, /RBE3 and
  /MPC in the implicit system — dependent DOFs eliminated through
  K_red = T^T K T, R_red = T^T R (never penalized), in BOTH geometry
  modes (T rebuilt per committed frame + exact rigid re-placement under
  /IMPL/NONLIN) and under /IMPL/DYNA (T^T M T is the exact rigid-body
  6-DOF mass at the master; initial velocities projected onto the
  constraint manifold).
* **Penalty contact in the Newton loop** (``contact.py`` — i7ke3.F /
  i7keg3.F): /INTER/TYPE7 frictionless contact force in the residual at
  the trial configuration + the exact gap tangent K g g^T in K(_eff),
  with the ACTIVE SET re-evaluated every iteration and non-convergence
  handed to the M11 StepControl cut. The i7sti3 stiffness/gap machinery
  of contact/stiffness.py is reused unchanged.

What M13 adds (implicit FRICTION, TYPE11, FOLLOWER LOADS, LAW36)
----------------------------------------------------------------
* **/INTER/TYPE7 Coulomb friction in the Newton loop** (``contact.py`` —
  the FRIC blocks of i7keg3.F, checked: I7KFOR3's incremental branch is
  a genuine return mapping): stick = tangential penalty spring K_t = K
  on the slip increment, slip = radial return to the cone with the
  CONSISTENT nonsymmetric tangent (the original's I7KEG3 assembles a
  mu-scaled always-stick spring instead); anchors committed per
  converged increment, slip work in the ``efric`` dynamics ledger
  channel; mu = 0 bit-identical to M12.
* **/INTER/TYPE11 edge-to-edge under implicit** (``ImplicitContact11``):
  the same penalty-in-residual + gap-tangent pattern on the
  segment-segment closest points, with the EXACT edge-edge closest-point
  curvature (2x2 optimality-system linearization, every projection
  region) and a two-point overlap quadrature for near-parallel pairs
  (the single closest point of parallel edges flips ends — a period-2
  Newton cycle otherwise). TYPE11 friction warns and runs frictionless.
* **/PLOAD follower-load stiffness under /IMPL/NONLIN**
  (``followerload.py`` — the IMP_KPRES analogue, documented deviation):
  trial-configuration pressure residual + the exact nonsymmetric
  -d f_ext/d x in K; /PLOAD + /IMPL/ARCL refused.
* **LAW36 consistent tangents** (solids + shells): the LAW2 algorithmic
  algebra with H from the table's local slope; the piecewise-linear
  return is already exact at implicit increments (measured); rate
  families truncated to the static curve (warned).
* Two solver lessons: the imp_solv.F-style backtracking LINE SEARCH in
  the statics Newton loop (engages only when the residual grows — smooth
  runs bit-identical; the dynamic loop's M/(beta dt^2) diagonal already
  regularizes the cycling)
  and the PERSISTENT implicit hourglass state ``hgq`` (the incremental
  static stabilization ratcheted across commits — latent since M8).

Explicitly DEFERRED (documented in PORTING_GUIDE.md, not half-implemented):

* thermal contact; TYPE19/24/25; Inacti/Igap 2/3; /RWALL under implicit
  (refused); /IMPDISP on constraint nodes; /PLOAD with /IMPL/ARCL
  (refused); the IFQ >= 10 / MODFR = 2 explicit tangential formulation
  (the implicit return mapping IS that formulation — M15 note in
  contact.py);
* **consistent (element) mass**, modal/eigenvalue dynamics,
  implicit-explicit switching mid-run, /IMPVEL under implicit dynamics
  (use /IMPDISP);
* rate devices under implicit — the LAW2 strain-rate term, the LAW36
  rate-curve family, the bulk viscosity, the spring dashpot, the MFROT
  friction-model VELOCITY terms (M15: reduced to their static limit
  mu(p, v=0), warned) and the IFQ force filter (a time device — its DC
  limit is the unfiltered force, warned) are disabled LOUDLY, never fed
  the pseudo-velocity;
* the IDTC = 2/3 step controls and /IMPL/DT/FIXP;
  the exact plane-stress (Iplas=1) LAW2 return.

M15 removed the last MATERIAL/friction refusals: Ifric > 0 friction
MODELS run in the implicit loop with the pressure-dependent Coulomb cone
mu(p) f_n and the consistent mu'(p) coupling tangent (contact.py);
LAW27 shells carry the damaged fixed-crack unilateral tangent
(materials/law27_brittle.py); LAW2 BEAMS carry the algorithmic tangent
of the global resultant-plasticity return with an ITERATED implicit
consistency solve (elements/beam_type3.py).

Why scipy is guarded
--------------------
The explicit solver depends on NumPy alone (``pyproject`` base deps). scipy
is only needed for the sparse assembly and the SuperLU factorization of the
implicit branch, so importing it at module top-level would make every base
install carry scipy. Instead the sparse-matrix imports live behind
``require_scipy()``: an explicit run never imports scipy, and an implicit run
without scipy gets one clear error telling the user to install it.
"""

from __future__ import annotations


def require_scipy():
    """Return the ``scipy.sparse`` and ``scipy.sparse.linalg`` modules, or
    raise a clear ImportError naming the optional dependency.

    Every implicit module obtains its scipy handles through this function
    (never a top-level ``import scipy``), so the base explicit install stays
    NumPy-only and an implicit run without scipy fails with one actionable
    message instead of a bare ModuleNotFoundError deep in the assembly."""
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except ImportError as exc:  # pragma: no cover - exercised without scipy
        raise ImportError(
            "the implicit solver (M8) requires SciPy for sparse assembly and "
            "the direct linear solve, but SciPy is not installed. Install it "
            "with `pip install scipy` (or `pip install -e \".[implicit]\"`). "
            "SciPy is NOT needed for the explicit solver."
        ) from exc
    return sp, spla
