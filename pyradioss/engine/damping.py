"""
/DAMP — Rayleigh mass damping (M6).

Fortran origin: ``engine/source/assembly/damping.F``, subroutine
``damping51`` (lines 100–170 for the mass-proportional branch in global
coordinates, lines 175–228 for the rotational-DOF branch).  The Fortran
uses an implicit-trapezoidal acceleration correction::

    OMEGA = 1/(1 + 0.5*DAMP_A*DT1)
    DA = (A - DAMP_A*V - BETASDT*(A - A_old)) * OMEGA - A
    A = A + DA

with energy booked as  DW += m * DA * (V + 0.5*A*DT1) * DT12.  The port
replaces this with the exact integrating factor (see below), which gives
identical physics (both are first-order-accurate mass damping) but is
unconditionally stable and books energy exactly from the KE identity.

A mass-proportional damping force  f_i = -alpha * m_i * v_i  acts on
the nodes of a group, optionally windowed in time (Tstart/Tstop). The
stiffness-proportional (beta) branch of full Rayleigh damping needs K*v
products the explicit port does not assemble — not ported (documented).

Implementation — the exact integrating factor
---------------------------------------------
Mass damping decouples per node:  m dv/dt = -alpha m v  has the exact
solution  v(t+dt) = v(t) * exp(-alpha dt).  Instead of adding the force
-alpha m v^{n-1/2} explicitly (which is half-step LAGGED — exactly the
phase defect the M6 bulk-viscosity investigation showed feeding phantom
energy channels at marginal time steps, and which would also erode the
stable dt for large alpha), the port applies the integrating factor to
the freshly updated velocities:

    v <- v * exp(-alpha dt)          (translations, and rotations where
                                      the node carries inertia)

This is unconditionally stable for ANY alpha*dt, exact for the damping
ODE, claims no time step, and its dissipation is booked EXACTLY from the
kinetic-energy identity:

    E_damp += sum 1/2 m (|v_before|^2 - |v_after|^2)

so the constraint of the milestone — "book its dissipation, the balance
must stay honest" — holds to round-off by construction. The dissipation
goes to its own ledger (DE in the T01, 'DAMPING DISSIPATION' in the
listing) and enters the energy balance.

The damper runs right after the acceleration update, BEFORE the
kinematic conditions: a /BCS, /IMPVEL or rigid-body enforcement then
overwrites the damped velocity like it overwrites the accelerated one
(damping never fights a kinematic condition; on such nodes it
effectively does nothing — the booking uses the velocities it actually
changed, so no phantom energy is booked either way... with one bounded
exception: a node corrected by a WALL in the same cycle books the
damping share of its arrest twice, once here and once in the wall's
injection identity; the overlap is one force phase of the few cycles a
node spends being arrested and vanishes with dt).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..model.model import Model


class Dampers:
    """Engine-side /DAMP list."""

    def __init__(self, model: Model, log):
        self.items = []          # (idx, alpha, tstart, tstop)
        self.all_idx = np.zeros(0, dtype=np.int64)   # union, for the
        # attribution correction the Engine books (see engine step 4b)
        # Build rigid-body slave set once — Fortran damping.F:150 excludes
        # nodes with TAGSLV_RBY != 0 because rigid_body.advance() will
        # overwrite their velocities, making any damping booking phantom.
        rb_slaves = set()
        for rb in getattr(model, 'rbodies', []):
            slaves = getattr(rb, 'slaves', None)
            if slaves is not None:
                rb_slaves.update(int(s) for s in slaves)
        if rb_slaves:
            rb_arr = np.array(sorted(rb_slaves), dtype=np.int64)
        else:
            rb_arr = np.zeros(0, dtype=np.int64)
        for dp in model.damps:
            g = model.node_groups.get(dp.grnod_id)
            if g is None or g.node_idx is None or g.node_idx.size == 0:
                log.error(f"/DAMP/{dp.id}: node group {dp.grnod_id} is "
                          f"missing or empty", "DAMP CHECK")
                continue
            idx = g.node_idx[model.mass[g.node_idx] < 1e29]
            # Exclude rigid-body slave nodes (damping.F:150 TAGSLV_RBY check)
            if len(rb_arr):
                idx = idx[~np.isin(idx, rb_arr)]
            if len(idx) == 0:
                continue
            self.items.append((idx, dp.alpha, dp.tstart, dp.tstop))
            self.all_idx = np.unique(np.concatenate([self.all_idx, idx]))
            log.info(f"     /DAMP/{dp.id}: ALPHA = {dp.alpha:12.5E} ON "
                     f"{len(idx)} NODE(S)"
                     + (f", ACTIVE {dp.tstart:g} TO {dp.tstop:g}"
                        if dp.tstop < 1e30 or dp.tstart > 0 else ""))

    def __len__(self):
        return len(self.items)

    # ------------------------------------------------------------------
    def apply(self, t: float, dt: float, v: np.ndarray, vr: np.ndarray,
              mass: np.ndarray, inertia: np.ndarray) -> float:
        """Damp the group velocities (exact integrating factor) and
        return the kinetic energy removed this cycle.

        Fortran: ``damping.F`` lines 129–171 (translational, ISK<=1, global
        coords) and lines 175–228 (rotational DOFs, IRODDL branch).

        The Fortran applies  DA = (A - alpha*V) * omega - A  per axis and
        books  DW += m * DA * (V + 0.5*A*DT1) * DT12.  The port replaces
        this with the exact ODE solution  v *= exp(-alpha*dt)  and books
        the KE drop  de += 0.5 * m * (|v_old|^2 - |v_new|^2)  — identical
        to roundoff for the damping ODE and unconditionally stable.
        """
        de = 0.0
        for idx, alpha, tstart, tstop in self.items:
            if dt <= 0.0 or t < tstart or t > tstop:
                continue
            fac = np.exp(-alpha * dt)
            de += float(0.5 * (1.0 - fac * fac)
                        * (mass[idx, None] * v[idx] ** 2).sum())
            v[idx] *= fac
            has_in = inertia[idx] > 0.0
            if np.any(has_in):
                ir = idx[has_in]
                de += float(0.5 * (1.0 - fac * fac)
                            * (inertia[ir, None] * vr[ir] ** 2).sum())
                vr[ir] *= fac
        return de


class DynamicRelaxation:
    """Dynamic Relaxation solver controls (/DYREL, /KEREL).

    Fortran origin: ``engine/source/general_controls/damping/static.F`` (subroutine
    ``STATIC`` called from ``resol.F:7321``).

    Supports:
    1. /DYREL (ISTAT=1): Viscous relaxation damping on velocities matching static.F:
           omega = betate * dt
           omega2 = max(0.0, (1.0 - 2.0 * omega) ** 2)
           v *= (1.0 - 2.0 * omega)
       Dissipated energy is booked exactly into DE (damping dissipation).
    2. /KEREL (ISTAT=2): Kinetic energy peak zeroing. Monitors the system kinetic
       energy on the relaxation node group; when a peak is detected (ke < ke_prev),
       all velocities on the group are reset to zero, freezing the structure at
       the static equilibrium point.
    """

    def __init__(self, model: Model, controls, log=None):
        self.model = model
        self.controls = controls
        self.log = log

        self.dyrel_active = bool(getattr(controls, "dyrel_active", False))
        self.dyrel_beta = float(getattr(controls, "dyrel_beta", 1.0))
        self.dyrel_period = float(getattr(controls, "dyrel_period", 0.0))
        self.dyrel_istatg = int(getattr(controls, "dyrel_istatg", 0))

        if self.dyrel_period > 0.0:
            self.betate = self.dyrel_beta / self.dyrel_period
        else:
            self.betate = self.dyrel_beta

        self.kerel_active = bool(getattr(controls, "kerel_active", False))
        self.kerel_tstart = float(getattr(controls, "kerel_tstart", 0.0))
        self.kerel_tstop = float(getattr(controls, "kerel_tstop", 0.0))
        if self.kerel_tstop <= 0.0:
            self.kerel_tstop = float(getattr(controls, "t_end", 0.0) or 1e30)
        self.kerel_istatg = int(getattr(controls, "kerel_istatg", 0))

        self.ke_prev = 0.0

        self.dyrel_idx = self._resolve_nodes(self.dyrel_istatg)
        self.kerel_idx = self._resolve_nodes(self.kerel_istatg)

        if log is not None:
            if self.dyrel_active:
                log.info(f"     /DYREL: BETA = {self.dyrel_beta:g}, "
                         f"PERIOD = {self.dyrel_period:g} (BETATE = {self.betate:12.5E}) "
                         f"ON {len(self.dyrel_idx)} NODE(S)")
            if self.kerel_active:
                log.info(f"     /KEREL: ACTIVE {self.kerel_tstart:g} TO {self.kerel_tstop:g} "
                         f"ON {len(self.kerel_idx)} NODE(S)")

    def _resolve_nodes(self, istatg: int) -> np.ndarray:
        if istatg != 0 and hasattr(self.model, "node_groups"):
            target_id = abs(istatg)
            g = self.model.node_groups.get(target_id) or self.model.node_groups.get(istatg)
            if g is not None and getattr(g, "node_idx", None) is not None and len(g.node_idx) > 0:
                idx = g.node_idx[self.model.mass[g.node_idx] < 1e29]
                return idx
        if hasattr(self.model, "mass") and self.model.mass is not None:
            return np.where(self.model.mass < 1e29)[0]
        n = getattr(self.model, "numnod", 0)
        return np.arange(n, dtype=np.int64)

    @property
    def active(self) -> bool:
        return bool(self.dyrel_active or self.kerel_active)

    def __len__(self) -> int:
        return 1 if self.active else 0

    def apply(self, t: float, dt: float, v: np.ndarray, vr: Optional[np.ndarray],
              mass: np.ndarray, inertia: Optional[np.ndarray]) -> float:
        """Apply dynamic relaxation (DYREL and/or KEREL) and return dissipated energy."""
        de = 0.0
        if dt <= 0.0 or not self.active:
            return de

        # 1. /DYREL viscous relaxation damping
        if self.dyrel_active and len(self.dyrel_idx) > 0:
            idx = self.dyrel_idx
            omega = self.betate * dt
            fac = max(0.0, 1.0 - 2.0 * omega)
            omega2 = fac * fac
            de += float(0.5 * (1.0 - omega2) * np.sum(mass[idx, None] * (v[idx] ** 2)))
            v[idx] *= fac
            if vr is not None and inertia is not None:
                has_in = inertia[idx] > 0.0
                if np.any(has_in):
                    ir = idx[has_in]
                    de += float(0.5 * (1.0 - omega2) * np.sum(inertia[ir, None] * (vr[ir] ** 2)))
                    vr[ir] *= fac

        # 2. /KEREL kinetic energy peak zeroing
        if self.kerel_active and (self.kerel_tstart <= t <= self.kerel_tstop) and len(self.kerel_idx) > 0:
            idx = self.kerel_idx
            ke = float(0.5 * np.sum(mass[idx, None] * (v[idx] ** 2)))
            has_in = inertia is not None and vr is not None and (inertia[idx] > 0.0)
            if vr is not None and inertia is not None and np.any(has_in):
                ir = idx[has_in]
                ke += float(0.5 * np.sum(inertia[ir, None] * (vr[ir] ** 2)))

            if ke < self.ke_prev and self.ke_prev > 0.0:
                # Kinetic energy peak detected!
                v[idx] = 0.0
                if vr is not None and inertia is not None and np.any(has_in):
                    vr[ir] = 0.0
                de += self.ke_prev
                self.ke_prev = 0.0
            else:
                self.ke_prev = ke

        return de

