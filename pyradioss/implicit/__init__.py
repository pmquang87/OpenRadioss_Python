"""
pyradioss.implicit — implicit (Newton–Raphson) analysis (M8).

Fortran origin: ``engine/source/implicit/`` — the whole implicit branch of
OpenRadioss, which lives entirely apart from the explicit ``resol.F`` loop:

    engine/source/implicit/imp_solv.F      the implicit driver / load steps
    engine/source/implicit/imp_buck.F      Newton iteration bookkeeping
    engine/source/implicit/ind_glob_k.F    global equation numbering (DOFs)
    engine/source/implicit/imp_glob_k.F    sparse tangent assembly
    engine/source/implicit/imp_dsolv*.F    the direct linear-solver interface
                                           (Radioss wraps MUMPS / a built-in
                                           sparse LDL^T here)

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

Explicitly DEFERRED out of M8 (documented in PORTING_GUIDE.md, not
half-implemented):

* **geometric / initial-stress stiffness** (large-displacement K_geo): M8
  lands small-strain *linear* geometry only. The residual still uses the
  full corotational kernels, so moderate rotations are handled in f_int, but
  the tangent omits the stress-dependent geometric term;
* **implicit DYNAMICS** (Newmark / HHT / generalized-α) — statics only;
* **contact and general constraints in the tangent system** (/INTER, /RBODY,
  /MPC): the explicit interfaces are kinematic/penalty and do not contribute
  to K here;
* **arc-length / snap-through** continuation — plain load control only.

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
