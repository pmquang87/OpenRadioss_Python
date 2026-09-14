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

Rotational nodal time step (STIFR) — M40
----------------------------------------
``dtnoda.F`` bounds the step with a SECOND, rotational nodal dt whenever
the model carries rotational DOFs (``IRODDL /= 0``): for EVERY node with
``IN(N) > 0`` and an assembled rotational stiffness ``STIFR(N) > 0``,

    dt_i^rot = sqrt(2 IN_i / STIFR_i)          (dtnoda.F lines 452-471)

joins the same minimum as the translational one.  The element claims
mirror the translational ones — the shell dt routines set
``STIR = STI * (t^2 + A)/12`` (cndt3.F lines 209-218, the BATOZ/QEPH/DKT
family; chvis3/chsti3 use t^2/12 + A/9 for BT) and the assembly adds the
FULL element STIR to each of its nodes (cupdt3.F, pmcum3.F), exactly as
STI feeds STIFN.  The Starter lumps the nodal inertia with the MATCHING
factor (cinmas.F ~line 920: ``XI = m/4 (AREA/FAC + t^2/12)`` with
FAC = 12 for IHBE >= 11, 9 for BT) — so on an element-lumped free node
the rotational dt EQUALS the translational one by construction and the
claim never binds there.  It bites only where IN is decoupled from the
element lumping: above all the /RBODY master, whose transported
``STIFR(M) += Sum(STIFR(s) + DD*STIFN(s))`` (rgbodfp.F) grows with the
parallel-axis distance while ``IN(M)`` stays the body's minimum
principal moment — the RD-E-1000 rolling floor (M39 residual: the port's
translational-only transport floored c04 at 2.07e-2 vs the Fortran
1.64e-2 that includes the members' own STIFR).

The port derives the rotational claim exactly like the translational
one, from the element dt: a per-node lumped-inertia share ``I_i^e`` (the
SAME array the Starter assembled into ``model.inertia`` — kernels store
it as ``group.state['dt_iner']``) behaves like a torsion spring of

    kr_i^e = 2 I_i^e / dt_e^2       (so that sqrt(2 I/kr) = dt_e)

which reproduces upstream's ``STIR = STI * fac`` identically (both sides
carry the same factor, k_i^e * I/m) and keeps the free-node invariant
dt_rot == dt_tra to round-off.  With /DT/NODA/CST the rotational branch
scales INERTIA exactly as the translational one scales mass
(dtnoda.F lines 482-516: ``IN(N) = MAX(INER, IN(N))``, the DINERT
counter): only ever added, reported in the summary; no energy booking is
needed because the port's kinetic-energy ledger is translational (the
original books none for DINERT either).  Rotational SPRING stiffness
(torsional /PROP/TYPE8 etc.) is NOT claimed — those kernels currently
make no rotational dt claim at all; their translational claim flows
unchanged.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

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

        # per-group (node-index matrix, nodal mass-share matrix, nodal
        # inertia-share matrix or None): the same lumping the Starter used
        # to build the nodal mass (1/8 per brick corner, 1/4 per
        # shell/tetra corner, 1/3 per triangle corner, 1/2 per truss/
        # spring/beam end — the beam's 3rd node is orientation only and
        # carries nothing).  The inertia share ``dt_iner`` is the per-NODE
        # lumped rotational inertia the kernel's init_group assembled into
        # ``model.inertia`` — the rotational claim base (module docstring);
        # groups without rotational DOFs (solids, trusses, springs) leave
        # it unset and claim no STIFR, like the original's IRODDL gating.
        self._shares: List[Tuple[str, np.ndarray, np.ndarray,
                                 Optional[np.ndarray]]] = []
        for name, group in model.element_groups():
            conn = group.state.get("mass_conn", group.conn)
            share = group.state["mass"][:, None] / conn.shape[1]
            iner = group.state.get("dt_iner")
            if iner is not None:
                iner = np.broadcast_to(iner[:, None], conn.shape).copy()
            self._shares.append((name, conn, np.broadcast_to(
                share, conn.shape).copy(), iner))
        #: any rotational claims at all? (gates every STIFR op so a
        #: solids-only deck pays nothing — dtnoda.F's IRODDL == 0 branch)
        self._rot = any(s[3] is not None for s in self._shares)

        # nodes excluded from the nodal dt / mass addition (prescribed
        # motion — see module docstring); filled by ``set_prescribed``
        self.free = model.mass < 1e29
        self.stifn = np.zeros(n)         # assembled nodal stiffness
        self.stifr = np.zeros(n)         # assembled rotational stiffness
        # rigid bodies whose member stiffness is transported to the master
        # (rgbodfp.F + dtnoda.F — see ``add_rigid_body``/``_rigid_body_dt``):
        # (member node indices, DD=|x-x_master|^2, body mass M, min inertia)
        self._rbodies: List[Tuple[np.ndarray, np.ndarray, float, float]] = []
        self.mass_added = 0.0            # cumulative added mass
        self.iner_added = 0.0            # cumulative added inertia (DINERT)
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

        ``nodes`` includes the master itself (its ``DD`` is zero), so the
        sums pick up any element stiffness claimed AT the master — exactly
        rgbodfp's ``+=`` onto the master's pre-existing STIFN/STIFR.

        M40 completes the M39 transport with the members' own rotational
        stiffness: the port now assembles ``stifr`` (the shell bending/
        drilling and beam STIR claims, see the module docstring), and
        ``_rigid_body_dt`` sums the full rgbodfp gather
        ``F2 = STIFR(s) + DD*STIFN(s)`` (rgbodfp.F lines 116-118).  On the
        RD-E-1000 BATOZ roll this moves the c04 floor from the M39
        translational-only 2.07e-2 onto the Fortran 1.64e-2.
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
        ``K_tra = Sum stifn`` and ``K_rot = Sum (stifr + DD*stifn)`` over
        the member nodes (rgbodfp.F IFLAG=1: F1 = STIFN(s),
        F2 = STIFR(s) + DD*STIFN(s); the STIFR term is the M40
        completion).  Read from the accumulators BEFORE ``apply`` resets
        them."""
        dt = EP30
        for nodes, dd, mass, in_min in self._rbodies:
            st = self.stifn[nodes]
            k_tra = float(st.sum())
            if k_tra > 0.0 and mass > 0.0:
                dt = min(dt, float(np.sqrt(2.0 * mass / k_tra)))
            k_rot = float((self.stifr[nodes] + dd * st).sum())
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
        Engine passes it to their ``forces``).  Groups with rotational
        DOFs also claim ``stifr`` from their lumped-inertia share
        (kr = 2 I / dt_e^2 — the module docstring; the STIR analogue of
        cupdt3.F/pmcum3.F's per-node accumulation)."""
        # (self.stifn already holds this cycle's contact springs)
        for (name, conn, share, iner), dt_e in zip(self._shares, dt_claims):
            k = 2.0 * share / np.maximum(dt_e, EM20)[:, None] ** 2
            # a deleted element's claim is 1e30 -> its k underflows to 0
            np.add.at(self.stifn, conn.reshape(-1), k.reshape(-1))
            if iner is not None:
                kr = 2.0 * iner / np.maximum(dt_e, EM20)[:, None] ** 2
                np.add.at(self.stifr, conn.reshape(-1), kr.reshape(-1))

    # ------------------------------------------------------------------
    def apply(self, mass_eff: np.ndarray, inv_mass: np.ndarray,
              v: np.ndarray, t: float,
              inertia: Optional[np.ndarray] = None,
              inv_inertia: Optional[np.ndarray] = None,
              ams_nodes: Optional[np.ndarray] = None) -> float:
        """Mass scaling + nodal dt for this cycle. Must run BEFORE the
        acceleration update (the added mass stabilizes the very cycle
        that needed it). Returns the nodal critical time step.

        ``ams_nodes`` — boolean array of nodes active in AMS (which are 
        excluded from driving the explicit dt).
        
        ``inertia``/``inv_inertia`` — the physical nodal rotational
        inertia (``model.inertia``) and its inverse: enables the
        ROTATIONAL nodal dt ``sqrt(2 IN/STIFR)`` over every free node
        with IN > 0 (dtnoda.F lines 452-471 — ALL nodes, not only rigid
        bodies; see the module docstring for why it only ever binds
        through /RBODY-like constructs) and, with CST, the matching
        inertia scaling (dtnoda.F lines 482-516).

        The stiffness accumulators are consumed and reset here."""
        model = self.model
        loaded = (self.stifn > 0.0) & self.free
        if ams_nodes is not None:
            loaded &= ~ams_nodes
        # rotational claims: dtnoda.F's IRODDL/IN(N)>0 gating.  stifr > 0
        # implies the node took a shell/beam claim, which also fed stifn,
        # so rot is a subset of loaded (contact springs feed only stifn).
        rot = None
        if self._rot and inertia is not None:
            rot = (self.stifr > 0.0) & self.free & (inertia > 0.0)
            if ams_nodes is not None:
                rot &= ~ams_nodes
            if not np.any(rot):
                rot = None
        if not np.any(loaded):
            # no FREE node claims a step, but a rigid body's transported
            # master dt still can (the RD-E-1000 case: every stiff shell is
            # welded into the body) — never silently drop it
            dt_rb = self._rigid_body_dt()
            self.stifn[:] = 0.0
            if self._rot:
                self.stifr[:] = 0.0
            return dt_rb

        if self.cst and self.dt_min > 0.0 and self.dt_sca > 0.0:
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
            if rot is not None:
                # rotational CST: inertia needed so that the rotational
                # nodal dt holds the target too (dtnoda.F 482-516,
                # IN(N) = MAX(INER, IN(N)) and the DINERT counter — only
                # ever added; no energy booking, the KE ledger is
                # translational, matching the original which books none)
                i_req = self.stifr[rot] * (self.dt_min / self.dt_sca) ** 2 \
                    / 2.0
                di = i_req - inertia[rot]
                addr = di > 0.0
                if np.any(addr):
                    idx = np.where(rot)[0][addr]
                    di = di[addr]
                    inertia[idx] += di          # model.inertia (physical)
                    if inv_inertia is not None:
                        inv_inertia[idx] = 1.0 / inertia[idx]
                    self.iner_added += float(di.sum())

        dt_i = np.sqrt(2.0 * mass_eff[loaded] / self.stifn[loaded])
        dt = float(dt_i.min())
        if rot is not None:
            # the rotational nodal dt joins the same minimum (dtnoda.F
            # line 469: DTN = DTFAC*sqrt(2 IN/STIFR); the shared DTFAC —
            # the port's dt_scale — is applied by the caller)
            dt_r = np.sqrt(2.0 * inertia[rot] / self.stifr[rot])
            dt = min(dt, float(dt_r.min()))
        # the rigid bodies' transported master dts join the free-node min
        # (rgbodfp.F/dtnoda.F — see add_rigid_body); a no-op with no /RBODY
        dt = min(dt, self._rigid_body_dt())
        self.stifn[:] = 0.0
        if self._rot:
            self.stifr[:] = 0.0
        return dt

    # ------------------------------------------------------------------
    def summary(self, log) -> None:
        """Termination-page report (the honesty contract)."""
        if not self.cst:
            return
        frac = 100.0 * self.mass_added / max(self.mass0, EM20)
        log.info(f"     ADDED MASS (/DT/NODA/CST) : {self.mass_added:14.7E}"
                 f"  ({frac:.3f}% OF INITIAL MASS)")
        if self.iner_added > 0.0:
            # rotational analogue of DMAST: dtnoda.F's DINERT counter
            log.info(f"     ADDED INERTIA (DINERT). . : "
                     f"{self.iner_added:14.7E}")
        log.info(f"     ENERGY FROM ADDED MASS  . : {self.e_madd:14.7E}")
        log.info(f"     MOMENTUM FROM ADDED MASS  : "
                 f"{self.mom_added[0]:12.5E} {self.mom_added[1]:12.5E} "
                 f"{self.mom_added[2]:12.5E}")
