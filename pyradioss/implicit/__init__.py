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

Explicitly DEFERRED (documented in PORTING_GUIDE.md, not half-implemented):

* **consistent (element) mass**, Rayleigh damping in the implicit system
  (/IMPL/DYNA/DAMP), modal/eigenvalue dynamics, implicit-explicit
  switching mid-run, automatic implicit time-step control (imp_dt.F),
  /IMPVEL under implicit dynamics (use /IMPDISP), rate-dependent
  plasticity under implicit dynamics;
* **contact and general constraints in the tangent system** (/INTER, /RBODY,
  /MPC): the explicit interfaces are kinematic/penalty and do not contribute
  to K here;
* **follower-load (pressure) stiffness**: /PLOAD is evaluated at the
  committed frame; its configuration-dependence is not linearized into K;
* LAW2 shell / LAW2 truss consistent tangents; tetra4 / sh3n / beam /
  spring element tangents; the /IMPL/BUCKL engine card (the buckling
  eigensolver is a library function).

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
