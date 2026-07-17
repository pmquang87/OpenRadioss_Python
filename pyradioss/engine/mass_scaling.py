"""
/DT/NODA and /DT/NODA/CST — nodal time step and mass scaling (M6).

Fortran origin: ``engine/source/time_step/`` — the nodal time-step option
of the /DT engine card family:

    dtnoda.F    nodal time step from the assembled nodal stiffness STIFN
                and the nodal mass:  dt_i = sqrt(2 M_i / K_i)
    (the element routines accumulate their stiffness into STIFN, the
    interfaces add their penalty-spring stiffness — the same accumulation
    this port already performs for the interface dt, see the M4 lesson in
    contact/inter_type7.py)

Theory — the nodal time step
----------------------------
The element time step dt_e = 2/omega_max(e) bounds the step by the WORST
single element. The nodal view assembles, per node, an equivalent
stiffness from every element (and contact spring) loading it and bounds
the step by the worst node-on-spring system:

    K_i = sum_e k_i^e ,      dt_i = sqrt(2 M_i / K_i)

The port derives the element's nodal-stiffness contribution from the SAME
quantity the element kernels already compute — their critical time step:
an element whose lumped nodal mass share is m_i^e and whose critical step
is dt_e behaves, seen from one of its nodes, like a spring of stiffness

    k_i^e = 2 m_i^e / dt_e^2         (so that sqrt(2 m/k) = dt_e)

For a uniform mesh this makes the nodal dt EQUAL to the element dt (each
node's mass and stiffness scale together); where a small stiff element
borders heavy nodes the nodal dt is larger — the classic reason /DT/NODA
alone already buys a few percent of step. Deleted elements return
dt_e = 1e30, so their stiffness contribution vanishes automatically.

Mass scaling (/DT/NODA/CST)
---------------------------
With the CST ("constant") option the Engine no longer lets dt fall below
the target: whenever a node's dt_i drops under dt_min, MASS IS ADDED to
that node until it holds,

    M_i^needed = K_i * (dt_min / dt_sca)^2 / 2 ,   dm_i = M^needed - M_i

(mass is only ever added, never removed — the Radioss DMAS convention).
Added mass CHANGES THE PHYSICS: it increases inertia, momentum and
kinetic energy. The port keeps the balance honest:

* dm is added to the PHYSICAL nodal mass (the KE/momentum ledgers see
  it) and to the effective mass used for accelerations;
* the kinetic energy the addition creates at the node's current
  velocity, 0.5 dm |v|^2, is booked into a dedicated counter reported in
  the listing and INCLUDED in the energy balance (like external work);
* the total added mass and its fraction of the initial mass are printed
  as the run progresses and in the termination summary — a mass-scaled
  run with dM/M beyond a few percent is a different structure, and the
  listing must say so (the original prints its MAS.ERR column for the
  same reason).

Nodes whose motion is PRESCRIBED (rigid-body members, tied /INTER/TYPE2
secondaries, /RBE3 dependents, frozen massless placeholders) carry no
stability constraint of their own — their velocities are overwritten by
the constraint — so they are excluded both from the nodal-dt minimum and
from mass addition (adding mass there could not change dt anyway, but it
WOULD silently alter the constraint inertia the Starter assembled).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..common.constants import EM20, EP30
from ..model.model import Model


class NodalTimeStep:
    """Engine-side /DT/NODA[/CST] machinery (see module docstring)."""

    def __init__(self, model: Model, controls, log):
        self.model = model
        self.cst = controls.dt_noda == "CST"
        self.dt_sca = controls.dt_scale
        self.dt_min = controls.dt_min
        n = model.numnod

        # per-group (node-index matrix, nodal mass-share matrix): the same
        # lumping the Starter used to build the nodal mass (1/8 per brick
        # corner, 1/4 per shell/tetra corner, 1/3 per triangle corner, 1/2
        # per truss/spring/beam end — the beam's 3rd node is orientation
        # only and carries nothing)
        self._shares: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for name, group in model.element_groups():
            conn = group.state.get("mass_conn", group.conn)
            share = group.state["mass"][:, None] / conn.shape[1]
            self._shares.append((name, conn, np.broadcast_to(
                share, conn.shape).copy()))

        # nodes excluded from the nodal dt / mass addition (prescribed
        # motion — see module docstring); filled by ``set_prescribed``
        self.free = model.mass < 1e29
        self.stifn = np.zeros(n)         # assembled nodal stiffness
        # rigid bodies whose member stiffness is transported to the master
        # (rgbodfp.F + dtnoda.F — see ``add_rigid_body``/``_rigid_body_dt``):
        # (member node indices, DD=|x-x_master|^2, body mass M, min inertia)
        self._rbodies: List[Tuple[np.ndarray, np.ndarray, float, float]] = []
        self.mass_added = 0.0            # cumulative added mass
        self.e_madd = 0.0                # cumulative 0.5 dm v^2
        self.mom_added = np.zeros(3)     # cumulative dm * v
        self.mass0 = float(model.mass[model.mass < 1e29].sum())
        self._reported = 0.0             # last dM/M milestone printed
        self.log = log
        if self.cst and self.dt_min <= 0.0:
            log.warning("/DT/NODA/CST: dT_min is zero — no mass will ever "
                        "be added (give a positive target step)",
                        "DT NODA")

    # ------------------------------------------------------------------
    def set_prescribed(self, idx: np.ndarray) -> None:
        """Mark nodes whose motion a constraint prescribes (rigid bodies,
        tied secondaries, RBE3 dependents): no dt claim, no mass add."""
        if len(idx):
            self.free[idx] = False

    # ------------------------------------------------------------------
    def add_rigid_body(self, nodes: np.ndarray, master: int, mass: float,
                       inertia: np.ndarray, x0: np.ndarray) -> None:
        """Register a rigid body so its member stiffness is TRANSPORTED to
        the master node and gives the body its own nodal time step — the
        thing that replaces the member nodes dropped by ``set_prescribed``.

        Fortran origin: ``rgbodfp.F`` (IFLAG=1 gather, called from
        ``rbyfor.F``) sums every slave's nodal stiffness onto the master —
        translational ``STIFN(M) += Sum STIFN(slave)`` and rotational
        ``STIFR(M) += Sum (STIFR(slave) + DD*STIFN(slave))`` with
        ``DD = |x_slave - x_master|^2`` the Huygens-Steiner parallel-axis
        transport of the slave spring to the master — then zeroes the
        slaves (rgbodfp.F 741-755).  ``dtnoda.F`` then bounds the step with
        the master's two nodal dts, ``sqrt(2 MS(M)/STIFN(M))`` (line 257)
        and ``sqrt(2 IN(M)/STIFR(M))`` (line 469), where the Starter set
        ``MS(M)`` = the body mass and ``IN(M) = MIN`` principal moment of
        the body inertia (``inirby.F`` line 838).

        Without this the port marks every member node prescribed and drops
        its stiffness, so a stiff shell welded into a rigid body never
        constrains dt (the M39 RD-E-1000 rolling bug: the port ran the
        clamped strip 2.6x too fast).

        ``DD`` is taken from the initial positions ``x0`` — a rigid body
        preserves its inter-node distances, so ``|x-x_master|`` is constant
        and equals its t=0 value (rgbodfp recomputes it from the current X
        each cycle only because the generic gather cannot assume rigidity).

        NOTE — the port's nodal-dt machinery accumulates only the
        TRANSLATIONAL stiffness ``stifn`` (``sqrt(2 M/K)``); it has no
        rotational-stiffness (``STIFR``) accumulator, so the transported
        rotational stiffness here carries only rgbodfp's ``DD*STIFN`` term,
        not the members' own drilling/bending ``STIFR(slave)``.  The body
        rotational dt is therefore a mild OVER-estimate (on the RD-E-1000
        BATOZ roll: 2.07e-2 vs the Fortran 1.64e-2 — both far below the
        4.31e-2 the un-transported port used); the mechanism and the
        governing entity (the master node) are exact.
        """
        nodes = np.asarray(nodes)
        dd = ((x0[nodes] - x0[master]) ** 2).sum(axis=1)
        in_min = float(np.linalg.eigvalsh(inertia)[0])
        self._rbodies.append((nodes, dd, float(mass), in_min))

    # ------------------------------------------------------------------
    def _rigid_body_dt(self) -> float:
        """The transported master nodal dt of every registered rigid body
        (see ``add_rigid_body``): min over bodies of the translational
        ``sqrt(2 M/K_tra)`` and rotational ``sqrt(2 IN/K_rot)`` steps, with
        ``K_tra = Sum stifn`` and ``K_rot = Sum DD*stifn`` over the member
        nodes.  Read from ``self.stifn`` BEFORE ``apply`` resets it."""
        dt = EP30
        for nodes, dd, mass, in_min in self._rbodies:
            st = self.stifn[nodes]
            k_tra = float(st.sum())
            if k_tra > 0.0 and mass > 0.0:
                dt = min(dt, float(np.sqrt(2.0 * mass / k_tra)))
            k_rot = float((dd * st).sum())
            if k_rot > 0.0 and in_min > 0.0:
                dt = min(dt, float(np.sqrt(2.0 * in_min / k_rot)))
        return dt

    # ------------------------------------------------------------------
    def assemble(self, dt_claims) -> None:
        """Assemble the nodal stiffness for this cycle.

        ``dt_claims`` — list of per-element critical-dt arrays, one per
        element group in ``model.element_groups()`` order (the arrays the
        kernels return each cycle). The contact contribution is
        accumulated directly into ``self.stifn`` by the interfaces (the
        Engine passes it to their ``forces``)."""
        # (self.stifn already holds this cycle's contact springs)
        for (name, conn, share), dt_e in zip(self._shares, dt_claims):
            k = 2.0 * share / np.maximum(dt_e, EM20)[:, None] ** 2
            # a deleted element's claim is 1e30 -> its k underflows to 0
            np.add.at(self.stifn, conn.reshape(-1), k.reshape(-1))

    # ------------------------------------------------------------------
    def apply(self, mass_eff: np.ndarray, inv_mass: np.ndarray,
              v: np.ndarray, t: float) -> float:
        """Mass scaling + nodal dt for this cycle. Must run BEFORE the
        acceleration update (the added mass stabilizes the very cycle
        that needed it). Returns the nodal critical time step.

        The stiffness accumulator is consumed and reset here."""
        model = self.model
        loaded = (self.stifn > 0.0) & self.free
        if not np.any(loaded):
            # no FREE node claims a step, but a rigid body's transported
            # master dt still can (the RD-E-1000 case: every stiff shell is
            # welded into the body) — never silently drop it
            dt_rb = self._rigid_body_dt()
            self.stifn[:] = 0.0
            return dt_rb

        if self.cst and self.dt_min > 0.0:
            # mass needed so that dt_sca * sqrt(2 M / K) >= dt_min
            m_req = self.stifn[loaded] * (self.dt_min / self.dt_sca) ** 2 \
                / 2.0
            dm = m_req - mass_eff[loaded]
            add = dm > 0.0
            if np.any(add):
                idx = np.where(loaded)[0][add]
                dm = dm[add]
                # physical mass: the KE/momentum ledgers must see the new
                # inertia (that is the honest part of mass scaling)
                model.mass[idx] += dm
                mass_eff[idx] += dm
                inv_mass[idx] = 1.0 / mass_eff[idx]
                self.mass_added += float(dm.sum())
                # the addition creates kinetic energy and momentum at the
                # node's current velocity — booked and reported
                self.e_madd += float(
                    0.5 * (dm[:, None] * v[idx] ** 2).sum())
                self.mom_added += (dm[:, None] * v[idx]).sum(axis=0)
                frac = self.mass_added / max(self.mass0, EM20)
                if frac >= self._reported + 0.01:  # 1%-step announcements
                    self.log.info(
                        f" -- /DT/NODA/CST: ADDED MASS {self.mass_added:.5E}"
                        f" ({100.0 * frac:.2f}% OF THE INITIAL MASS)"
                        f" AT TIME {t:.5E}")
                    self._reported = frac

        dt_i = np.sqrt(2.0 * mass_eff[loaded] / self.stifn[loaded])
        # the rigid bodies' transported master dts join the free-node min
        # (rgbodfp.F/dtnoda.F — see add_rigid_body); a no-op with no /RBODY
        dt = min(float(dt_i.min()), self._rigid_body_dt())
        self.stifn[:] = 0.0
        return dt

    # ------------------------------------------------------------------
    def summary(self, log) -> None:
        """Termination-page report (the honesty contract)."""
        if not self.cst:
            return
        frac = 100.0 * self.mass_added / max(self.mass0, EM20)
        log.info(f"     ADDED MASS (/DT/NODA/CST) : {self.mass_added:14.7E}"
                 f"  ({frac:.3f}% OF INITIAL MASS)")
        log.info(f"     ENERGY FROM ADDED MASS  . : {self.e_madd:14.7E}")
        log.info(f"     MOMENTUM FROM ADDED MASS  : "
                 f"{self.mom_added[0]:12.5E} {self.mom_added[1]:12.5E} "
                 f"{self.mom_added[2]:12.5E}")
