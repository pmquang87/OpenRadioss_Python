"""
Direct and iterative linear solvers behind a small ``solve(K, R)`` interface (M8, M619).

Fortran origins:
- ``engine/source/implicit/imp_dsolv*.F`` and ``imp_solv.F`` (direct solver dispatch)
- ``engine/source/implicit/imp_pcg.F`` (PCG solver: IMP_PCGH lines 227-750, CRIT_STOP lines 25-57)
- ``engine/source/implicit/imp_fsa_inv.F`` (SP_STATIC lines 60-100)
- ``engine/source/implicit/imp_fac_ic.F`` (IMP_FAC_ICJ lines 31-100)

Backend selection — mirrors the M7 compute-backend pattern EXACTLY
------------------------------------------------------------------
M7 introduced an optional numba compute backend selected by
``PYRADIOSS_BACKEND`` / ``-backend`` with a NumPy fallback and a warning when
the library is absent (see ``pyradioss/accel/__init__.py``). The implicit
linear solver reuses that pattern verbatim for the FACTORIZATION / SOLVER library:

* ``scipy.sparse.linalg.splu`` (SuperLU) — the DEFAULT direct solver. It ships with SciPy,
  which the implicit branch already requires, so the default adds no extra
  dependency. Unsymmetric LU; robust for every statics case here.
* ``PCG`` (Preconditioned Conjugate Gradient) — the ITERATIVE solver ported from
  OpenRadioss ``imp_pcg.F``. Ideal for large Symmetric-Positive-Definite (SPD)
  linear elasticity and structural problems. Supports Jacobi (diagonal), ILU
  (incomplete LU), and SSOR (symmetric successive over-relaxation) preconditioners.
* ``CHOLMOD`` (scikit-sparse) — OPTIONAL. A sparse Cholesky for the
  Symmetric-Positive-Definite statics case (linear elastic K, and plastic K
  while it stays SPD); typically the fastest and lowest-memory direct choice.
* ``MUMPS`` (python-mumps) — OPTIONAL. A multifrontal solver for large /
  indefinite systems. We DO NOT port MUMPS — we wrap the existing library,
  exactly as the original links against it.

Selection is by env var ``PYRADIOSS_LINSOLVE=superlu|pcg|cholmod|mumps`` or the
engine CLI ``-linsolve``. A requested-but-missing optional solver warns and
falls back to SuperLU — never breaking a run — and the optional solvers stay
OPTIONAL dependencies (the base install keeps working with SuperLU alone).

The ``solve(K, R)`` contract
----------------------------
``K`` is a SciPy CSR matrix (from ``assembly.assemble``), ``R`` a dense
equation vector; ``solve`` returns ``Δu`` with ``K Δu = R``. Each backend
factorizes or iteratively solves ``K`` with ``R``. A LinearSolver object caches
nothing between Newton iterations (K changes every iteration in general), but
the interface leaves room for a "same-pattern refactorize" optimization,
noted where it would go.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

#: resolved backend, resolved lazily like accel._state: {"name", "mod"?}.
_state = {"name": None}


def select_linsolve(name=None, log=None) -> str:
    """Select the linear solver backend; returns the name actually active.

    ``name=None`` reads ``PYRADIOSS_LINSOLVE`` (default "superlu"). An
    unknown name, or a missing optional library, warns and falls back to
    SuperLU — requesting a fancier solver must never break a run (the exact
    contract of ``accel.select_backend`` for the compute backends)."""
    if name is None:
        name = os.environ.get("PYRADIOSS_LINSOLVE", "superlu")
    name = (name or "superlu").strip().lower()

    if name.startswith("pcg") or name == "1":
        # Supports "pcg", "pcg_jacobi", "pcg:ilu", "pcg_ssor", or OpenRadioss ISOLV=1
        _state["name"] = "pcg" if name == "1" else name
    elif name == "cholmod":
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


def pcg_solve(
    K,
    R: np.ndarray,
    precond: Union[str, int] = "jacobi",
    tol: float = 1e-6,
    maxiter: Optional[int] = None,
    x0: Optional[np.ndarray] = None,
    log: Optional[Any] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Preconditioned Conjugate Gradient (PCG) iterative linear solver.

    Ported from OpenRadioss Fortran:
      - ``engine/source/implicit/imp_pcg.F`` (subroutine IMP_PCGH, lines 227-750;
        stopping criterion CRIT_STOP, lines 31-57)
      - ``engine/source/implicit/imp_fsa_inv.F`` (SP_STATIC, lines 60-100)
      - ``engine/source/implicit/imp_fac_ic.F`` (IMP_FAC_ICJ, lines 31-100)

    Solves the linear system K x = R for a symmetric positive definite (SPD)
    stiffness matrix K and right-hand side residual vector R.

    Parameters
    ----------
    K : scipy.sparse.spmatrix or array-like
        Symmetric positive-definite stiffness matrix (typically CSR).
    R : np.ndarray
        Right-hand side equation vector.
    precond : str or int, optional
        Preconditioner type:
        - "diag" or "jacobi" (IPREC=1 in imp_pcg.F): M_inv = 1.0 / diag(K)
        - "ilu" (IPREC=5 in imp_pcg.F): incomplete LU factorization via scipy.sparse.linalg.spilu
        - "ssor" (IPREC=2 in imp_pcg.F): symmetric successive over-relaxation
        - "none" or "identity": unpreconditioned CG (M_inv = I)
    tol : float, optional
        Relative convergence tolerance ||r_k||_2 / ||R||_2 <= tol. Default is 1e-6.
    maxiter : int, optional
        Maximum number of iterations. Default is min(1000, max(20, 10 * n)).
    x0 : np.ndarray, optional
        Initial solution guess. Default is zeros.
    log : optional
        Logger instance.

    Returns
    -------
    x : np.ndarray
        Solution vector.
    info : dict
        Convergence metadata: "converged", "iterations", "residual", "rel_residual", "reason".
    """
    if hasattr(K, "tocsr"):
        K_csr = K.tocsr()
    else:
        import scipy.sparse as sp
        K_csr = sp.csr_matrix(K)

    n = K_csr.shape[0]
    R_vec = np.asarray(R, dtype=np.float64).ravel()
    norm_R = float(np.linalg.norm(R_vec))

    if n == 0 or len(R_vec) == 0:
        return np.zeros(0, dtype=np.float64), {
            "converged": True,
            "iterations": 0,
            "residual": 0.0,
            "rel_residual": 0.0,
            "reason": "empty_system",
        }

    if norm_R == 0.0:
        return np.zeros(n, dtype=np.float64), {
            "converged": True,
            "iterations": 0,
            "residual": 0.0,
            "rel_residual": 0.0,
            "reason": "zero_rhs",
        }

    if maxiter is None or maxiter <= 0:
        maxiter = min(1000, max(20, 10 * n))

    # Map integer IPREC from Fortran card
    if isinstance(precond, int):
        iprec_map = {1: "jacobi", 2: "ssor", 5: "ilu"}
        p_type = iprec_map.get(precond, "jacobi")
    else:
        p_type = (precond or "jacobi").strip().lower()

    if p_type == "diag":
        p_type = "jacobi"

    # ------------------------------------------------------------------
    # Preconditioner setup (M_inv)
    # ------------------------------------------------------------------
    if p_type == "ilu":
        from . import require_scipy
        _, spla = require_scipy()
        try:
            ilu_obj = spla.spilu(K_csr.tocsc())
            apply_precond = lambda v: ilu_obj.solve(v)
        except Exception as e:
            if log is not None:
                log.warning(f"ILU preconditioning failed: {e}; falling back to Jacobi", "LINSOLVE")
            diag_K = np.array(K_csr.diagonal(), dtype=np.float64)
            safe_diag = np.where(np.abs(diag_K) > 1e-14, diag_K, 1.0)
            apply_precond = lambda v: v / safe_diag
    elif p_type == "ssor":
        import scipy.sparse as sp
        from . import require_scipy
        _, spla = require_scipy()
        try:
            diag_K = np.array(K_csr.diagonal(), dtype=np.float64)
            safe_diag = np.where(np.abs(diag_K) > 1e-14, diag_K, 1.0)
            L_csr = sp.tril(K_csr, format="csr")
            U_csr = sp.triu(K_csr, format="csr")

            def ssor_apply(v):
                y = spla.spsolve_triangular(L_csr, v, lower=True)
                return spla.spsolve_triangular(U_csr, safe_diag * y, lower=False)

            apply_precond = ssor_apply
        except Exception as e:
            if log is not None:
                log.warning(f"SSOR preconditioning failed: {e}; falling back to Jacobi", "LINSOLVE")
            diag_K = np.array(K_csr.diagonal(), dtype=np.float64)
            safe_diag = np.where(np.abs(diag_K) > 1e-14, diag_K, 1.0)
            apply_precond = lambda v: v / safe_diag
    elif p_type in ("none", "identity"):
        apply_precond = lambda v: np.copy(v)
    else:  # "jacobi" / "diag" (IPREC=1)
        diag_K = np.array(K_csr.diagonal(), dtype=np.float64)
        safe_diag = np.where(np.abs(diag_K) > 1e-14, diag_K, 1.0)
        apply_precond = lambda v: v / safe_diag

    # ------------------------------------------------------------------
    # Conjugate Gradient Algorithm (imp_pcg.F)
    # ------------------------------------------------------------------
    # r_0 = R - K x_0
    if x0 is not None:
        x = np.array(x0, dtype=np.float64, copy=True).ravel()
        r = R_vec - K_csr @ x
    else:
        x = np.zeros(n, dtype=np.float64)
        r = np.array(R_vec, copy=True)

    norm_r = float(np.linalg.norm(r))
    rel_res = norm_r / norm_R
    if rel_res <= tol:
        return x, {
            "converged": True,
            "iterations": 0,
            "residual": norm_r,
            "rel_residual": rel_res,
            "reason": "initial_converged",
        }

    # z_0 = M_inv r_0, p_0 = z_0
    z = apply_precond(r)
    p = np.copy(z)
    rho = float(np.dot(r, z))

    if rho <= 0.0 or not np.isfinite(rho):
        return x, {
            "converged": False,
            "iterations": 0,
            "residual": norm_r,
            "rel_residual": rel_res,
            "reason": "indefinite_preconditioner",
        }

    converged = False
    it = 0
    reason = "max_iterations"
    for it in range(1, maxiter + 1):
        Kp = K_csr @ p
        p_Kp = float(np.dot(p, Kp))
        if p_Kp <= 0.0 or not np.isfinite(p_Kp):
            # Matrix is not positive-definite -> curvature breakdown
            reason = "non_positive_curvature"
            break

        # alpha_k = (r_k, z_k) / (p_k, K p_k)
        alpha = rho / p_Kp

        # x_{k+1} = x_k + alpha_k p_k
        x += alpha * p

        # r_{k+1} = r_k - alpha_k K p_k
        r -= alpha * Kp
        norm_r = float(np.linalg.norm(r))
        rel_res = norm_r / norm_R

        # check convergence: norm(r_{k+1}) / norm(R) <= tol
        if rel_res <= tol:
            converged = True
            reason = "converged"
            break

        # z_{k+1} = M_inv r_{k+1}
        z = apply_precond(r)
        rho_new = float(np.dot(r, z))
        if rho_new <= 0.0 or not np.isfinite(rho_new):
            reason = "loss_of_orthogonality"
            break

        # beta_k = (r_{k+1}, z_{k+1}) / (r_k, z_k)
        beta = rho_new / rho

        # p_{k+1} = z_{k+1} + beta_k p_k
        p = z + beta * p
        rho = rho_new

    info = {
        "converged": converged,
        "iterations": it,
        "residual": norm_r,
        "rel_residual": rel_res,
        "reason": reason,
    }
    return x, info


class LinearSolver:
    """A ``solve(K, R)`` front-end over the selected direct or iterative backend.

    One instance per implicit run; ``solve`` factorizes the CSR ``K`` (or applies
    PCG iterative solver) and back-substitutes ``R`` each call (K generally changes
    every Newton iteration). The backend is resolved once at construction."""

    def __init__(self, name=None, log=None, precond=None, tol=1e-6, maxiter=None):
        self.name = select_linsolve(name, log) if name is not None \
            else linsolve_name(log)
        self.log = log

        detected_precond = None
        if self.name.startswith("pcg"):
            parts = self.name.replace(":", "_").split("_", 1)
            if len(parts) > 1 and parts[1] in ("jacobi", "diag", "ilu", "ssor", "none"):
                detected_precond = parts[1]

        if isinstance(precond, int):
            iprec_map = {1: "jacobi", 2: "ssor", 5: "ilu"}
            precond = iprec_map.get(precond, "jacobi")

        if precond is not None:
            self.precond = str(precond).lower()
        elif detected_precond is not None:
            self.precond = detected_precond
        else:
            env_precond = os.environ.get(
                "PYRADIOSS_PCG_PRECOND",
                os.environ.get("PYRADIOSS_PRECOND", "jacobi")
            )
            self.precond = (env_precond or "jacobi").strip().lower()

        if self.precond == "diag":
            self.precond = "jacobi"

        self.tol = tol
        self.maxiter = maxiter

    # ----------------------------------------------------------------------
    def solve(self, K, R):
        """Return Δu solving K Δu = R for a CSR ``K`` and dense ``R``."""
        if self.name.startswith("pcg"):
            return self._solve_pcg(K, R)
        if self.name == "cholmod":
            return self._solve_cholmod(K, R)
        if self.name == "mumps":
            return self._solve_mumps(K, R)
        return self._solve_superlu(K, R)

    # ---- PCG (iterative solver from imp_pcg.F) ---------------------------
    def _solve_pcg(self, K, R):
        """Solve K Δu = R using Preconditioned Conjugate Gradient (imp_pcg.F)."""
        x, info = pcg_solve(
            K, R,
            precond=self.precond,
            tol=self.tol,
            maxiter=self.maxiter,
            log=self.log,
        )
        if info["converged"]:
            return x

        # Fallback to SuperLU direct solver (indefinite or non-converging iterative solve)
        msg = (
            f"PCG ({self.precond}) solver did not converge (reason: {info.get('reason')}, "
            f"rel_res={info.get('rel_residual', 0.0):.2e}) — falling back to SuperLU"
        )
        warnings.warn(msg, stacklevel=2)
        if self.log is not None:
            self.log.warning(msg, "LINSOLVE")
        try:
            return self._solve_superlu(K, R)
        except Exception:
            from . import require_scipy
            _, spla = require_scipy()
            M_op = self._build_scipy_precond_operator(K)
            max_it = self.maxiter if self.maxiter is not None else min(1000, max(20, 10 * K.shape[0]))
            x_cg, exit_code = spla.cg(K, R, rtol=self.tol, maxiter=max_it, M=M_op)
            if exit_code == 0 and np.all(np.isfinite(x_cg)):
                return x_cg
            raise

    def _build_scipy_precond_operator(self, K):
        """Construct a scipy LinearOperator matching self.precond for scipy.sparse.linalg.cg."""
        from . import require_scipy
        _, spla = require_scipy()
        n = K.shape[0]
        p_type = self.precond
        if p_type == "ilu":
            try:
                ilu = spla.spilu(K.tocsc())
                return spla.LinearOperator((n, n), matvec=ilu.solve)
            except Exception:
                pass
        elif p_type == "ssor":
            try:
                import scipy.sparse as sp
                diag_K = np.array(K.diagonal(), dtype=np.float64)
                safe_diag = np.where(np.abs(diag_K) > 1e-14, diag_K, 1.0)
                L_csr = sp.tril(K, format="csr")
                U_csr = sp.triu(K, format="csr")

                def ssor_apply(v):
                    y = spla.spsolve_triangular(L_csr, v, lower=True)
                    return spla.spsolve_triangular(U_csr, safe_diag * y, lower=False)

                return spla.LinearOperator((n, n), matvec=ssor_apply)
            except Exception:
                pass
        # default to jacobi / diagonal
        diag_K = np.array(K.diagonal(), dtype=np.float64)
        safe_diag = np.where(np.abs(diag_K) > 1e-14, diag_K, 1.0)
        return spla.LinearOperator((n, n), matvec=lambda v: v / safe_diag)

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
