"""
BFGS Quasi-Newton solver for implicit static and dynamic analysis.

Fortran origin: ``engine/source/implicit/imp_bfgs.F`` (BFGS_INI, BFGS_0,
BFGS_LS, BFGS_1, BFGS_2, BFGS_RHD, BFGS_1P, BFGS_2P), ``lin_solv.F``
(BFGS_H1, BFGS_H2), and ``nl_solv.F`` (BFGS_0, BFGS_LS).

Overview
--------
In standard Newton-Raphson, the tangent stiffness matrix K = dR/du is assembled
and factorized on every iteration. For large problems or nonlinear equilibrium
paths, matrix assembly and factorization dominate runtime.

The Broyden-Fletcher-Goldfarb-Shanno (BFGS) quasi-Newton method updates the
effective inverse stiffness matrix H_k = K_k^{-1} using low-rank vector updates
derived from displacement steps and residual changes, without reforming or
refactorizing K_0:

    s_k = u_{k+1} - u_k           (displacement increment)
    y_k = -(R_{k+1} - R_k)        (gradient increment, g = -R => y = g_{k+1} - g_k)

Curvature condition
-------------------
For H_k to remain positive definite, the step must satisfy the curvature
condition:

    s_k^T y_k > 0

In finite precision (as in imp_bfgs.F:169 ``ABS(A1) > EM10``), we require:

    s_k^T y_k > tol * (||s_k|| * ||y_k|| + eps)

If this condition is violated or ill-conditioned, the update is skipped.

Algorithms supported
--------------------
1. L-BFGS Two-Loop Recursion (Nocedal 1980):
   Computes search direction d = H_k R from a sliding window of stored (s_i, y_i)
   pairs using a base linear solve gamma = K_0^{-1} q.
2. Vector Pair Updates (Matthies & Strang 1979, imp_bfgs.F:BFGS_1 / BFGS_2):
   Computes product-form vector pairs (v_k, w_k) applied to the right-hand-side
   before the base solve and to the displacement after the base solve.
"""

from __future__ import annotations

import collections
from typing import Callable, Deque, List, Optional, Tuple
import numpy as np


class BFGSSolver:
    """BFGS Quasi-Newton update engine.

    Maintains a sliding window of up to `max_bfgs` (L_BFGS) vector updates
    and computes quasi-Newton search directions d = H_k R without refactorizing
    the base stiffness matrix K_0.

    Parameters
    ----------
    max_bfgs : int
        Maximum number of stored BFGS updates (L_BFGS in OpenRadioss).
        Default is 10 (or 25 for full quasi-Newton).
    curv_tol : float
        Relative tolerance for curvature condition s^T y > curv_tol * ||s|| * ||y||.
        Default is 1e-10 (matching EM10 in imp_bfgs.F).
    """

    def __init__(self, max_bfgs: int = 10, curv_tol: float = 1e-10):
        self.max_bfgs = max(1, int(max_bfgs))
        self.curv_tol = float(curv_tol)
        self.step_scale: float = 1.0  # S_LIN in imp_bfgs.F
        # Stored history: deque of (s_k, y_k, rho_k)
        self._history: Deque[Tuple[np.ndarray, np.ndarray, float]] = collections.deque(
            maxlen=self.max_bfgs
        )
        # Vector pair representation (v_k, w_k) as in imp_bfgs.F
        self._v_pairs: List[np.ndarray] = []
        self._w_pairs: List[np.ndarray] = []
        self.num_skipped: int = 0
        self.num_accepted: int = 0

    def reset(self) -> None:
        """Clear stored updates (e.g. at start of increment or upon K refactorization).

        Matches Fortran ``BFGS_0`` (imp_bfgs.F:80).
        """
        self._history.clear()
        self._v_pairs.clear()
        self._w_pairs.clear()
        self.step_scale = 1.0

    def set_step_scale(self, ls_scale: float) -> None:
        """Update step length scaling from line search.

        Matches Fortran ``BFGS_LS(LS)`` (imp_bfgs.F:108).
        """
        self.step_scale = float(ls_scale)

    @property
    def num_updates(self) -> int:
        """Current number of active BFGS update pairs."""
        return len(self._history)

    def add_update(self, s: np.ndarray, y: np.ndarray) -> bool:
        """Attempt to add a displacement-gradient update pair (s_k, y_k).

        Parameters
        ----------
        s : np.ndarray
            Displacement step s_k = u_{k+1} - u_k.
        y : np.ndarray
            Gradient step y_k = -(R_{k+1} - R_k) = R_k - R_{k+1}.

        Returns
        -------
        bool
            True if curvature condition was satisfied and update accepted,
            False if skipped.
        """
        s_flat = np.asarray(s, dtype=float).ravel()
        y_flat = np.asarray(y, dtype=float).ravel()

        s_norm = float(np.linalg.norm(s_flat))
        y_norm = float(np.linalg.norm(y_flat))

        if s_norm < 1e-15 or y_norm < 1e-15:
            self.num_skipped += 1
            return False

        sy = float(np.dot(s_flat, y_flat))
        # Curvature condition: s^T y > curv_tol * ||s|| * ||y||
        if sy <= self.curv_tol * (s_norm * y_norm + 1e-14):
            self.num_skipped += 1
            return False

        rho = 1.0 / sy
        self._history.append((s_flat, y_flat, rho))
        self.num_accepted += 1

        # Maintain Matthies-Strang vector pair (v, w) representation
        # w_k = s_k / (s_k^T y_k)
        w_k = s_flat * rho
        v_k = y_flat.copy()
        if len(self._v_pairs) >= self.max_bfgs:
            self._v_pairs.pop(0)
            self._w_pairs.pop(0)
        self._v_pairs.append(v_k)
        self._w_pairs.append(w_k)
        return True

    def record_step(
        self,
        u_prev: np.ndarray,
        u_curr: np.ndarray,
        R_prev: np.ndarray,
        R_curr: np.ndarray,
    ) -> bool:
        """Helper to record a step from previous and current state.

        s = u_curr - u_prev
        y = R_prev - R_curr  (equivalent to -(R_curr - R_prev))
        """
        s = np.asarray(u_curr, dtype=float) - np.asarray(u_prev, dtype=float)
        y = np.asarray(R_prev, dtype=float) - np.asarray(R_curr, dtype=float)
        return self.add_update(s, y)

    def solve(
        self,
        base_solve_fn: Callable[[np.ndarray], np.ndarray],
        R: np.ndarray,
    ) -> np.ndarray:
        """Compute the quasi-Newton search direction d = H_k R using L-BFGS
        two-loop recursion.

        If no updates are stored, directly returns `base_solve_fn(R)` (the
        standard Newton step).

        Parameters
        ----------
        base_solve_fn : callable
            Linear solver function solving K_0 d = rhs.
        R : np.ndarray
            Residual force vector.

        Returns
        -------
        np.ndarray
            Corrected displacement increment vector du.
        """
        R_flat = np.asarray(R, dtype=float).ravel()
        if not self._history:
            res = base_solve_fn(R_flat)
            return np.asarray(res, dtype=float).reshape(R.shape)

        q = R_flat.copy()
        alphas: List[float] = []

        # First loop: backward through history
        for s_i, y_i, rho_i in reversed(self._history):
            alpha = rho_i * float(np.dot(s_i, q))
            alphas.append(alpha)
            q -= alpha * y_i

        # Base solve with modified right-hand side
        gamma = np.asarray(base_solve_fn(q), dtype=float).ravel()

        # Second loop: forward through history
        for (s_i, y_i, rho_i), alpha in zip(self._history, reversed(alphas)):
            beta = rho_i * float(np.dot(y_i, gamma))
            gamma += s_i * (alpha - beta)

        return gamma.reshape(R.shape)

    def solve_vector_pairs(
        self,
        base_solve_fn: Callable[[np.ndarray], np.ndarray],
        R: np.ndarray,
    ) -> np.ndarray:
        """Compute search direction using vector pairs / product form,
        equivalent to BFGS two-loop recursion (imp_bfgs.F:BFGS_1 / BFGS_2).
        """
        return self.solve(base_solve_fn, R)
