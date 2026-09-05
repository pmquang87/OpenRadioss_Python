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

from typing import List

import numpy as np

from ..model.model import Model


class Dampers:
    """Engine-side /DAMP list."""

    def __init__(self, model: Model, log):
        self.items = []          # (idx, alpha, tstart, tstop)
        self.all_idx = np.zeros(0, dtype=np.int64)   # union, for the
        # attribution correction the Engine books (see engine step 4b)
        for dp in model.damps:
            g = model.node_groups.get(dp.grnod_id)
            if g is None or g.node_idx is None or g.node_idx.size == 0:
                log.error(f"/DAMP/{dp.id}: node group {dp.grnod_id} is "
                          f"missing or empty", "DAMP CHECK")
                continue
            idx = g.node_idx[model.mass[g.node_idx] < 1e29]
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
