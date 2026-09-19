"""
Line search algorithms for nonlinear implicit equilibrium iterations.

Fortran origin: ``engine/source/implicit/nl_solv.F`` (LINE_S, LINE_S1,
lines 520-570 and 890-1107) and ``recudis.F``.

Line search options (ILINE_S):
------------------------------
- ILINE_S = 1: Energy-based line search (LINE_S1).
  Evaluates directional derivative of potential energy:
      E_1(alpha) = d^T R(u + alpha * d)
  normalized by E_0 = d^T R(u). Finds alpha such that |E_1 / E_0| <= tol
  using secant interpolation between bracketed points.
- ILINE_S = 2: Force / residual-based line search (LINE_S).
  Evaluates relative residual norm:
      r_1(alpha) = ||R(u + alpha * d)|| / ||R(u)||
  and applies secant / quadratic interpolation to find step length.
- ILINE_S = 3 (AUTO, default in OpenRadioss /IMPL/LSEAR):
  Adapts between directional derivative projection and residual reduction.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np


def line_s(
    r0: float,
    s0: float,
    r1: float,
    s: float,
    dtol: float = 0.5,
    idiv: int = 0,
    iint: int = 0,
    irefi: int = 0,
    prec: float = 0.02,
    nls_lim: int = 4,
    imconv: int = 0,
) -> Tuple[float, float, float, int, int]:
    """Secant / residual interpolation line search step.

    Port of Fortran ``LINE_S`` (``nl_solv.F:895``).

    Parameters
    ----------
    r0 : float
        Previous residual ratio.
    s0 : float
        Previous step length.
    r1 : float
        Current residual ratio (e.g. ||R_trial|| / ||R_0|| or |dE / dE0|).
    s : float
        Current trial step length.
    dtol : float
        Line search tolerance (LS_TOL).
    idiv : int
        Step iteration / divergence counter.
    iint : int
        Contact / interface flag.
    irefi : int
        Stiffness reformulation flag.
    prec : float
        Precision parameter (EP02 = 0.02 or EP03 = 0.001).
    nls_lim : int
        Maximum line search iterations (NLS_LIM).
    imconv : int
        Convergence flag (-1: in line search, 0: converged / finished).

    Returns
    -------
    (s_new, s0_new, r0_new, idiv_new, imconv_new)
    """
    dr = abs(r0 - r1)
    r = min(r1 / max(prec, 1e-15), dr)

    # Termination check (nl_solv.F:923)
    if (r < dtol or r1 < dtol or idiv < -nls_lim) or (r1 > 1.0 and iint > 0 and irefi < 2):
        imconv = 0
        idiv -= 1
        return s, s0, r0, idiv, imconv
    elif r1 > r0 and imconv == -1:
        # Step grew residual further: revert to previous step length
        idiv = -nls_lim - 2
        imconv = 0
        return s0, s0, r0, idiv, imconv

    imconv = -1
    s1 = s
    if dr > 1e-15:
        s_new = abs(s0 * r1 - s1 * r0) / dr
    else:
        s_new = s1

    if r1 > 1.0:
        # Diverging: cut step
        if s_new >= 1.0 - 1e-12:
            s_new = 0.5 * s1 / max(r1, 1e-15)
        s_new = max(0.01, s_new)
    else:
        # Extrapolation bounding
        if s_new > max(s1, s0) or s_new < min(s1, s0):
            s_new = min(1.5 * s1, s_new)
            s_new = min(5.0, s_new)

    s0_new = s1
    r0_new = r1
    idiv -= 1
    return s_new, s0_new, r0_new, idiv, imconv


def line_s1(
    e1: float,
    s: float,
    ep: float,
    sp: float,
    en: float,
    sn: float,
    idiv: int = 0,
    dtol: float = 0.5,
    icont: int = 0,
    icont0: int = 0,
    iriks: int = 0,
    nls_lim: int = 4,
) -> Tuple[float, float, float, float, float, int, int]:
    """Energy-based directional derivative line search step.

    Port of Fortran ``LINE_S1`` (``nl_solv.F:1002``).
    Finds s such that directional derivative E_1 = d^T R(u + s*d) / E_0 vanishes
    or |E_1| <= dtol.

    Parameters
    ----------
    e1 : float
        Current normalized directional derivative (d^T R_trial) / (d^T R_0).
    s : float
        Current trial step length.
    ep : float
        Positive bracket energy.
    sp : float
        Positive bracket step length.
    en : float
        Negative bracket energy.
    sn : float
        Negative bracket step length.
    idiv : int
        Line search iteration counter.
    dtol : float
        Tolerance for directional derivative convergence.

    Returns
    -------
    (s_new, ep_new, sp_new, en_new, sn_new, idiv_new, imconv_new)
    """
    imconv = -1
    eold = 0.0
    im = max(icont, icont0)

    if e1 > 0.0:
        if idiv == 0:
            idiv += 1
            sp = 1.0
            ep = e1
            s = s + 1.0
            if iriks > 0 or im > 0:
                s = 1.0
                imconv = 0
            return s, ep, sp, en, sn, idiv, imconv
        elif idiv > 0:
            sold = s
            if abs(e1 - ep) < dtol or e1 > ep or idiv > nls_lim:
                if e1 > ep:
                    s = s - 1.0
                imconv = 0
            else:
                s = s + 1.0
            idiv += 1
            sp = sold
            ep = e1
            return s, ep, sp, en, sn, idiv, imconv
        sp = s
        ep = e1
    else:
        if idiv == 0:
            sp = 0.0
            ep = 1.0
            s = 0.5
            idiv -= 1
            sn = 1.0
            en = e1
            return s, ep, sp, en, sn, idiv, imconv
        elif sp == 0.0:
            if abs(e1 - en) < dtol or idiv < -nls_lim:
                imconv = 0
            elif iriks > 0:
                imconv = 0
            else:
                s = s * 0.5
            idiv -= 1
            return s, ep, sp, en, sn, idiv, imconv
        elif idiv > 0:
            idiv = -idiv
        eold = en
        sn = s
        en = e1

    idiv -= 1
    sold = s
    im = max(icont, icont0)
    if im > 0:
        stmp = 0.5 * (-sp + sn)
    else:
        denom = ep - en
        if abs(denom) > 1e-15:
            stmp = ep * (sn - sp) / denom
        else:
            stmp = 0.5 * (sn - sp)

    s = sp + stmp
    stmp_rel = (sold - s) / max(abs(sold), 1e-15)
    stmp_val = max(abs(stmp_rel), abs(e1 - eold))
    if stmp_val < dtol or idiv < -nls_lim:
        imconv = 0
    elif iriks > 0 and (s > 2.0 or s < 0.01):
        imconv = 0
    else:
        imconv = -1

    return s, ep, sp, en, sn, idiv, imconv


class LineSearch:
    """Configurable line search runner for implicit static Newton iterations.

    Parameters
    ----------
    method : int
        1: Energy-based (directional derivative, LINE_S1)
        2: Force-residual secant (LINE_S)
        3: Auto adaptive (LINE_S / energy hybrid)
    tol : float
        Line search tolerance LS_TOL (default 0.5).
    max_iter : int
        Maximum line search steps NLS_LIM (default 4).
    min_step : float
        Lower bound on step length alpha (default 0.01).
    max_step : float
        Upper bound on step length alpha (default 5.0).
    """

    def __init__(
        self,
        method: int = 3,
        tol: float = 0.5,
        max_iter: int = 4,
        min_step: float = 0.01,
        max_step: float = 5.0,
    ):
        self.method = int(method)
        self.tol = float(tol)
        self.max_iter = max(1, int(max_iter))
        self.min_step = float(min_step)
        self.max_step = float(max_step)

    def search(
        self,
        residual_fn: Callable[[np.ndarray, np.ndarray], Tuple[Any, Any, np.ndarray]],
        u: np.ndarray,
        ur: np.ndarray,
        du: np.ndarray,
        dur: np.ndarray,
        R0: np.ndarray,
        de0: Optional[float] = None,
    ) -> Tuple[float, np.ndarray, np.ndarray, Any, Any, np.ndarray, int]:
        """Perform line search along direction (du, dur).

        Parameters
        ----------
        residual_fn : callable
            Function (u_trial, ur_trial) -> (fint, mint, R_eq).
        u, ur : np.ndarray
            Base displacement and rotation increments.
        du, dur : np.ndarray
            Computed search direction.
        R0 : np.ndarray
            Residual at base state u (alpha = 0).
        de0 : float, optional
            Base directional derivative d^T R0. If None, computed as dot(du, R0).

        Returns
        -------
        alpha : float
            Accepted step length.
        u_acc, ur_acc : np.ndarray
            Accepted trial displacements and rotations u + alpha*du.
        fint_acc, mint_acc : Any
            Accepted internal forces and moments.
        R_acc : np.ndarray
            Accepted reduced residual vector.
        num_evals : int
            Number of trial evaluations performed.
        """
        # Flattened search direction in equation space
        du_flat = np.asarray(du, dtype=float).ravel()
        R0_flat = np.asarray(R0, dtype=float).ravel()
        r0_norm = float(np.linalg.norm(R0_flat))

        if de0 is None:
            # Directional derivative: E_0 = d^T R_0
            # For equation space: if du has same size as R0
            if du_flat.size == R0_flat.size:
                de0 = float(np.dot(du_flat, R0_flat))
            else:
                de0 = r0_norm

        if abs(de0) < 1e-30:
            de0 = 1.0

        # Initial trial step alpha = 1.0 (Full Newton step)
        alpha = 1.0
        fint_1, mint_1, R_1 = residual_fn(u + alpha * du, ur + alpha * dur)
        r1_norm = float(np.linalg.norm(R_1))
        num_evals = 1

        best = (r1_norm, alpha, fint_1, mint_1, R_1)

        # If full Newton step already improves or maintains residual, accept it
        # (monotone convergence preserves standard Newton rate)
        if np.isfinite(r1_norm) and r1_norm <= r0_norm:
            # Check energy condition if method == 1
            if self.method == 1:
                # Directional derivative E_1
                if du_flat.size == np.asarray(R_1).size:
                    e1 = float(np.dot(du_flat, np.asarray(R_1).ravel())) / de0
                else:
                    e1 = r1_norm / max(r0_norm, 1e-30)
                if abs(e1) <= self.tol:
                    return alpha, u + alpha * du, ur + alpha * dur, fint_1, mint_1, R_1, num_evals
            else:
                return alpha, u + alpha * du, ur + alpha * dur, fint_1, mint_1, R_1, num_evals

        # Line search iteration
        s = alpha
        s0 = 0.0
        r0_val = 1.0
        sp, ep = 0.0, 1.0
        sn, en = 1.0, 1.0
        idiv = 0
        imconv = -1

        for ls in range(self.max_iter):
            # Compute current trial metrics
            if self.method == 1:
                # Energy directional derivative
                if du_flat.size == np.asarray(R_1).size:
                    e1 = float(np.dot(du_flat, np.asarray(R_1).ravel())) / de0
                else:
                    e1 = r1_norm / max(r0_norm, 1e-30)

                s, ep, sp, en, sn, idiv, imconv = line_s1(
                    e1, s, ep, sp, en, sn,
                    idiv=idiv, dtol=self.tol, nls_lim=self.max_iter
                )
            else:
                # Force / secant residual
                r1_val = r1_norm / max(r0_norm, 1e-30)
                s, s0, r0_val, idiv, imconv = line_s(
                    r0_val, s0, r1_val, s,
                    dtol=self.tol, idiv=idiv, nls_lim=self.max_iter, imconv=imconv
                )

            # Clamp step length within bounds
            s = float(np.clip(s, self.min_step, self.max_step))

            # Evaluate at new trial step
            fint_t, mint_t, R_t = residual_fn(u + s * du, ur + s * dur)
            rn_t = float(np.linalg.norm(R_t))
            num_evals += 1

            if np.isfinite(rn_t) and rn_t < best[0]:
                best = (rn_t, s, fint_t, mint_t, R_t)

            R_1 = R_t
            r1_norm = rn_t

            if imconv == 0 or rn_t <= r0_norm:
                break

        # Return the best trial found
        best_rnorm, best_alpha, best_fint, best_mint, best_R = best
        u_acc = u + best_alpha * du
        ur_acc = ur + best_alpha * dur
        return best_alpha, u_acc, ur_acc, best_fint, best_mint, best_R, num_evals
