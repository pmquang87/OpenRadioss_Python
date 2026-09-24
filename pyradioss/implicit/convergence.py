"""
Multi-criterion convergence and divergence check for implicit nonlinear equilibrium iterations.

Fortran origin: ``engine/source/implicit/nl_solv.F`` (CRIT_ITE lines 805-885).

Convergence criteria (NITOL):
-----------------------------
- NITOL = 1: Energy tolerance (N_TOLE).
  Relative energy error |dE / E_cum| <= N_TOLE.
  Guarded against premature convergence at it=1 or when force residual has not decreased.
- NITOL = 2: Force tolerance (N_TOLF).
  Relative residual ||R|| / ||R_0|| <= N_TOLF (or relative to reference reaction).
- NITOL = 3: Displacement tolerance (N_TOLU).
  Relative displacement ||du|| / ||u|| <= N_TOLU.
- NITOL = 12: Combined Energy (1) and Force (2).
  max(RR / N_TOLF, |ER| / N_TOLE) <= 1.0.
- NITOL = 13: Combined Energy (1) and Displacement (3).
  max(UR / N_TOLU, |ER| / N_TOLE) <= 1.0.
- NITOL = 23: Combined Force (2) and Displacement (3).
  max(UR / N_TOLU, RR / N_TOLF) <= 1.0.
- NITOL = 123: Combined Energy (1), Force (2), and Displacement (3).
  max(UR / N_TOLU, RR / N_TOLF, |ER| / N_TOLE) <= 1.0.

Divergence criteria (/IMPL/DIVER):
----------------------------------
- TOL_DIV: maximum allowed residual growth ratio before aborting (default 1e4).
- NDIVER: maximum consecutive divergence iterations allowed (default 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ConvergenceStatus:
    """Status of convergence check at current Newton iteration."""
    converged: bool
    diverged: bool
    err: float
    ru: float   # relative displacement
    rr: float   # relative force residual
    er: float   # relative energy increment


def crit_ite(
    it: int,
    ur: float,
    rr: float,
    er: float,
    ndiv: int,
    tol: float,
    nitol: int = 2,
    n_tole: float = 1e-4,
    n_tolf: float = 1e-3,
    n_tolu: float = 1e-3,
    tol_div: float = 1e4,
    ndiver: int = 3,
) -> Tuple[int, int, float]:
    """Port of Fortran ``CRIT_ITE`` (``nl_solv.F:809-885``).

    Parameters
    ----------
    it : int
        Current Newton iteration index (1-based, or 0-based + 1).
    ur : float
        Relative displacement norm ||du|| / ||u||.
    rr : float
        Relative force residual norm ||R|| / ||R_0||.
    er : float
        Relative energy increment |dE| / |E_cum|.
    ndiv : int
        Consecutive divergence count.
    tol : float
        Base tolerance.
    nitol : int
        Convergence criterion mode (1, 2, 3, 12, 13, 23, 123).
    n_tole, n_tolf, n_tolu : float
        Specific tolerances for energy, force, and displacement.
    tol_div : float
        Divergence threshold ratio (TOL_DIV).
    ndiver : int
        Maximum permitted consecutive divergence iterations (NDIVER).

    Returns
    -------
    imconv : int
        1: converged, 0: continue, -2: diverged, -3: early divergence warning
    ndiv : int
        Updated divergence count.
    err : float
        Calculated normalized error metric.
    """
    told = tol_div
    tolr = 1.0 + tol

    # Evaluate error metric according to NITOL
    if nitol == 1:
        err = abs(er) / tol
        if it == 1 or rr >= 1.0:
            err += tolr
    elif nitol == 2:
        err = rr / tol
    elif nitol == 3:
        err = ur / tol
        if rr >= 1.0:
            err += tolr
    elif nitol == 12:
        err = max(rr / n_tolf, abs(er) / n_tole)
    elif nitol == 13:
        err = max(ur / n_tolu, abs(er) / n_tole)
    elif nitol == 23:
        err = max(ur / n_tolu, rr / n_tolf)
    elif nitol == 123:
        err = max(ur / n_tolu, rr / n_tolf, abs(er) / n_tole)
    else:
        # Fallback to force
        err = rr / tol

    # Convergence check
    if err <= 1.0:
        imconv = 1
    elif ur < min(1e-5, tol) and rr < 1e-1:
        imconv = 1
    else:
        imconv = 0
        # Divergence check (nl_solv.F:874-883)
        if rr > tolr:
            ndiv += 1
            if it == 1:
                imconv = -3
                if ndiv < ndiver:
                    imconv = 0
            else:
                imconv = -2
                if ndiv < ndiver and rr < told:
                    imconv = 0
        else:
            ndiv = 0

    return imconv, ndiv, err


class ConvergenceChecker:
    """Stateful convergence and divergence checker for implicit static iterations."""

    def __init__(
        self,
        tol: float = 1e-6,
        nitol: int = 2,
        n_tole: float = 1e-4,
        n_tolf: float = 1e-3,
        n_tolu: float = 1e-3,
        tol_div: float = 1e4,
        ndiver: int = 3,
    ):
        self.tol = float(tol)
        self.nitol = int(nitol)
        self.n_tole = float(n_tole)
        self.n_tolf = float(n_tolf)
        self.n_tolu = float(n_tolu)
        self.tol_div = float(tol_div)
        self.ndiver = int(ndiver)

        self.ndiv: int = 0
        self.e_cum: float = 0.0

    def reset(self) -> None:
        """Reset divergence count and accumulated energy for a new increment."""
        self.ndiv = 0
        self.e_cum = 0.0

    def check(
        self,
        it: int,
        du_norm: float,
        u_norm: float,
        r_norm: float,
        r0_norm: float,
        dE: float,
    ) -> ConvergenceStatus:
        """Evaluate convergence and divergence at iteration `it` (0-indexed)."""
        # Relative displacement
        if it == 0 or u_norm < 1e-30:
            ur = 1.0
        else:
            ur = du_norm / max(u_norm, 1e-30)

        # Relative force
        rr = r_norm / max(r0_norm, 1e-30)

        # Relative energy
        self.e_cum += abs(dE)
        if self.e_cum < 1e-30:
            er = 1.0
        else:
            er = abs(dE) / max(self.e_cum, 1e-30)

        imconv, self.ndiv, err = crit_ite(
            it=it + 1,
            ur=ur,
            rr=rr,
            er=er,
            ndiv=self.ndiv,
            tol=self.tol,
            nitol=self.nitol,
            n_tole=self.n_tole,
            n_tolf=self.n_tolf,
            n_tolu=self.n_tolu,
            tol_div=self.tol_div,
            ndiver=self.ndiver,
        )

        converged = (imconv == 1)
        diverged = (imconv in (-2, -3))

        return ConvergenceStatus(
            converged=converged,
            diverged=diverged,
            err=err,
            ru=ur,
            rr=rr,
            er=er,
        )
