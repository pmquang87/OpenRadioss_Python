"""
Direct linear solver behind a small ``solve(K, R)`` interface (M8).

Fortran origin: ``engine/source/implicit/imp_dsolv*.F`` and the
``imp_solv.F`` driver's linear-solve call — where OpenRadioss dispatches the
factorization of the assembled tangent to a direct sparse solver (its built
-in LDL^T, or MUMPS when linked). The port keeps that same shape: a single
``solve(K, R)`` entry point with a *selectable, wrappable* backend.

Backend selection — mirrors the M7 compute-backend pattern EXACTLY
------------------------------------------------------------------
M7 introduced an optional numba compute backend selected by
``PYRADIOSS_BACKEND`` / ``-backend`` with a NumPy fallback and a warning when
the library is absent (see ``pyradioss/accel/__init__.py``). The implicit
linear solver reuses that pattern verbatim for the FACTORIZATION library:

* ``scipy.sparse.linalg.splu`` (SuperLU) — the DEFAULT. It ships with SciPy,
  which the implicit branch already requires, so the default adds no extra
  dependency. Unsymmetric LU; robust for every statics case here.
* ``CHOLMOD`` (scikit-sparse) — OPTIONAL. A sparse Cholesky for the
  Symmetric-Positive-Definite statics case (linear elastic K, and plastic K
  while it stays SPD); typically the fastest and lowest-memory choice.
* ``MUMPS`` (python-mumps) — OPTIONAL. A multifrontal solver for large /
  indefinite systems. We DO NOT port MUMPS — we wrap the existing library,
  exactly as the original links against it.

Selection is by env var ``PYRADIOSS_LINSOLVE=superlu|cholmod|mumps`` or the
engine CLI ``-linsolve``. A requested-but-missing optional solver warns and
falls back to SuperLU — never breaking a run — and the optional solvers stay
OPTIONAL dependencies (the base install keeps working with SuperLU alone).

The ``solve(K, R)`` contract
----------------------------
``K`` is a SciPy CSR matrix (from ``assembly.assemble``), ``R`` a dense
equation vector; ``solve`` returns ``Δu`` with ``K Δu = R``. Each backend
factorizes ``K`` and back-substitutes ``R``. A LinearSolver object caches
nothing between Newton iterations (K changes every iteration in general), but
the interface leaves room for a "same-pattern refactorize" optimization,
noted where it would go.
"""

from __future__ import annotations

import os
import warnings

#: resolved backend, resolved lazily like accel._state: {"name", "mod"?}.
_state = {"name": None}


def select_linsolve(name=None, log=None) -> str:
    """Select the direct-solver backend; returns the name actually active.

    ``name=None`` reads ``PYRADIOSS_LINSOLVE`` (default "superlu"). An
    unknown name, or a missing optional library, warns and falls back to
    SuperLU — requesting a fancier solver must never break a run (the exact
    contract of ``accel.select_backend`` for the compute backends)."""
    if name is None:
        name = os.environ.get("PYRADIOSS_LINSOLVE", "superlu")
    name = (name or "superlu").strip().lower()

    if name == "cholmod":
        try:
            import sksparse.cholmod  # noqa: F401 - probe availability
            _state["name"] = "cholmod"
        except ImportError:
            _fallback("CHOLMOD backend requested but scikit-sparse is not "
                      "installed — falling back to SuperLU "
                      "(pip install scikit-sparse)", log)
    elif name == "mumps":
        try:
            import mumps  # noqa: F401 - python-mumps, probe availability
            _state["name"] = "mumps"
        except ImportError:
            _fallback("MUMPS backend requested but python-mumps is not "
                      "installed — falling back to SuperLU "
                      "(pip install python-mumps)", log)
    elif name == "superlu":
        _state["name"] = "superlu"
    else:
        _fallback(f"unknown PYRADIOSS_LINSOLVE '{name}' — using SuperLU", log)
    return _state["name"]


def _fallback(msg, log):
    warnings.warn(msg, stacklevel=3)
    if log is not None:
        log.warning(msg, "LINSOLVE")
    _state["name"] = "superlu"


def linsolve_name(log=None) -> str:
    """The active backend name, resolving lazily (``log`` routes a fallback
    warning into the engine listing — same as ``accel.backend_name``)."""
    if _state["name"] is None:
        select_linsolve(log=log)
    return _state["name"]


class LinearSolver:
    """A ``solve(K, R)`` front-end over the selected direct backend.

    One instance per implicit run; ``solve`` factorizes the CSR ``K`` and
    back-substitutes ``R`` each call (K generally changes every Newton
    iteration). The backend is resolved once at construction."""

    def __init__(self, name=None, log=None):
        self.name = select_linsolve(name, log) if name is not None \
            else linsolve_name(log)

    # ----------------------------------------------------------------------
    def solve(self, K, R):
        """Return Δu solving K Δu = R for a CSR ``K`` and dense ``R``."""
        if self.name == "cholmod":
            return self._solve_cholmod(K, R)
        if self.name == "mumps":
            return self._solve_mumps(K, R)
        return self._solve_superlu(K, R)

    # ---- SuperLU (default, ships with SciPy) -----------------------------
    def _solve_superlu(self, K, R):
        from . import require_scipy
        _, spla = require_scipy()
        # CSC is SuperLU's native layout; splu does the LU factorization and
        # .solve back-substitutes. (A same-sparsity-pattern run could reuse
        # the column permutation via permc_spec — deferred: statics here is
        # a few iterations, factorization dominated by the numeric phase.)
        lu = spla.splu(K.tocsc())
        return lu.solve(R)

    # ---- CHOLMOD (optional, SPD statics) ---------------------------------
    def _solve_cholmod(self, K, R):  # pragma: no cover - optional dep
        from sksparse.cholmod import cholesky, CholmodNotPositiveDefiniteError
        try:
            factor = cholesky(K.tocsc())
            return factor(R)
        except CholmodNotPositiveDefiniteError:
            # K lost positive-definiteness (a limit point / softening plastic
            # tangent): CHOLMOD cannot factor it — fall back to SuperLU for
            # this solve rather than aborting the Newton step.
            warnings.warn("CHOLMOD: tangent not positive definite — using "
                          "SuperLU for this solve", stacklevel=2)
            return self._solve_superlu(K, R)

    # ---- MUMPS (optional, wrapped — NOT ported) --------------------------
    def _solve_mumps(self, K, R):  # pragma: no cover - optional dep
        import mumps
        # python-mumps exposes a one-shot spsolve for CSR/COO systems
        return mumps.spsolve(K.tocsc(), R)
