"""
Kinematic conditions and external loads.

Fortran origin: ``engine/source/constraints/general/bcs/bcs*.F`` (fixed
DOFs), ``.../impvel`` (imposed velocities AND displacements — fixvel.F
serves both through IFLAG), ``engine/source/loads``
(``force.F`` concentrated loads, ``gravit.F`` gravity,
``general/pload/pload.F`` follower pressure).

Two very different mechanisms, kept distinct exactly like the original:

* **Loads** (gravity, concentrated forces, pressure) ADD to the external
  force vector; the node still obeys F = m a.
* **Kinematic conditions** (BCS, imposed velocity/displacement, rigid
  walls) OVERRIDE the computed velocity after the acceleration update.
  Their energy contribution is booked as external work through the
  constraint impulse times the imposed velocity (see apply_kinematic) so
  the global energy balance stays consistent — the original recovers the
  same work from the constraint reactions.

/PLOAD (M5) is a *follower* pressure: it acts along the CURRENT segment
normal, i.e. it turns with the structure. The segment normal and area
come from the cross product of the diagonals (the standard bilinear-quad
area vector, which is also exact for the degenerate triangle segments of
the Radioss n4 = n3 convention); p*A is lumped to the corners with weight
1/4 (1/3 for triangles). Segments whose parent element was /FAIL-deleted
stop carrying pressure — a torn face is an open boundary (same provenance
machinery as contact, see contact/tracking.py).
"""

from __future__ import annotations

import numpy as np

from ..contact import tracking
from ..model.model import Model


class LoadsAndConstraints:
    """Pre-resolved (index-based) loads + constraints for the Engine."""

    def __init__(self, model: Model, log):
        self.model = model
        # BCS -> per-dof boolean masks
        self.fix_tra = np.zeros((model.numnod, 3), dtype=bool)
        self.fix_rot = np.zeros((model.numnod, 3), dtype=bool)
        for bc in model.bcs:
            idx = model.node_groups[bc.grnod_id].node_idx
            for d in range(3):
                if bc.fix_tra[d]:
                    self.fix_tra[idx, d] = True
                if bc.fix_rot[d]:
                    self.fix_rot[idx, d] = True

        # frozen (massless) nodes are fully fixed
        frozen = model.mass >= 1e29
        self.fix_tra[frozen, :] = True
        self.fix_rot[frozen, :] = True
        # gravity mass: a frozen node carries NO physical mass — its 1e30
        # placeholder must not turn into a 1e30*g force (harmless while
        # the node is BCS-fixed, catastrophic once a rigid body gathers
        # its force rows). The PHYSICAL (pre-mass-scaling) mass is used:
        # /DT/NODA/CST additions are numerical and must not weigh (which
        # also keeps a chained restart identical to the unchained run).
        m_phys = getattr(model, "mass0", model.mass)
        self._m_grav = np.where(frozen, 0.0, m_phys)

        # resolved loads: (node_idx, direction, funct, scale)
        def _grp(gid):
            if gid in (None, 0):
                return np.arange(model.numnod)
            return model.node_groups[gid].node_idx

        self.gravity = [(_grp(g.grnod_id), g.direction, model.functions[g.funct_id],
                         g.scale) for g in model.gravity]
        # /CLOAD entries carry their /SENSOR id (M6): 0 = always active
        self.cloads = [(_grp(c.grnod_id), c.direction, model.functions[c.funct_id],
                        c.scale, c.sens_id) for c in model.cloads]
        # /IMPVEL entries: (node_idx, dof, funct, Fscale_Y, 1/Ascale_x,
        # Tstart, Tstop) — the curve is evaluated at t/Ascale_x and the
        # condition only holds inside [Tstart, Tstop] (fixvel.F: FACX,
        # STARTT/STOPT).
        self.impvel = [(_grp(i.grnod_id), i.dof, model.functions[i.funct_id],
                        i.scale, 1.0 / i.xscale, i.tstart, i.tstop)
                       for i in model.impvel]
        # /IMPDISP: like /IMPVEL, plus the base coordinate of each node so
        # the target position x0 + d(t) is exact (no velocity-integration
        # drift). Entries: (node_idx, dof, funct, scale, facx, tstart,
        # tstop, x0_dof).
        self.impdisp = []
        for i in model.impdisp:
            idx = _grp(i.grnod_id)
            self.impdisp.append((idx, i.dof, model.functions[i.funct_id],
                                 i.scale, 1.0 / i.xscale, i.tstart, i.tstop,
                                 model.x0[idx, i.dof].copy()))
        # a FROZEN node under an imposed velocity/displacement is a
        # legitimate massless kinematic carrier (the standard way to drive
        # a moving /RWALL): release its auto-fix on the driven DOF, and
        # remember that its 1e30 placeholder mass must never enter the
        # constraint-work booking (it carries no physical inertia — the
        # reaction it transmits is booked where it acts, e.g. by the
        # moving-wall term-2 booking).
        self._frozen = frozen
        for entry in self.impvel + self.impdisp:
            idx, dof = entry[0], entry[1]
            self.fix_tra[idx[frozen[idx]], dof] = False
        for i in model.impdisp:
            f0 = model.functions[i.funct_id].eval(0.0) * i.scale
            if abs(f0) > 0.0:
                log.warning(f"/IMPDISP/{i.id}: curve starts at "
                            f"d(0) = {f0:.4g} != 0 — the nodes will JUMP "
                            f"there in the first cycle", "IMPDISP INIT")

        # /PLOAD: resolved segments + triangle lumping weights + deletion
        # provenance (a torn face stops carrying pressure)
        self.ploads = []
        for pl in model.ploads:
            surf = model.surfaces[pl.surf_id]
            segs = surf.segments if surf.segments is not None else \
                np.zeros((0, 4), dtype=np.int64)
            if len(segs) == 0:
                log.warning(f"/PLOAD/{pl.id}: surface {pl.surf_id} has no "
                            f"segments — load inactive", "PLOAD INIT")
            tri = segs[:, 3] == segs[:, 2] if len(segs) else \
                np.zeros(0, dtype=bool)
            # corner lumping weights: 1/4 per quad corner; triangles put
            # 1/3 on each distinct corner and 0 on the repeated slot
            wgt = np.full((len(segs), 4), 0.25)
            wgt[tri] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0]
            gtype = (surf.seg_gtype if surf.seg_gtype is not None
                     else np.zeros(len(segs), dtype="<U8"))
            elem = (surf.seg_elem if surf.seg_elem is not None
                    else np.full(len(segs), -1, dtype=np.int64))
            deletable = tracking.any_deletable(model, gtype)
            self.ploads.append((segs, wgt, model.functions[pl.funct_id],
                                pl.scale, gtype, elem, deletable,
                                pl.sens_id))

    # ------------------------------------------------------------------
    def external_forces(self, t: float, fext: np.ndarray,
                        x: np.ndarray, sensors=None) -> None:
        """Accumulate gravity + concentrated loads + follower pressure at
        time t (gravit.F, force.F, pload.F). Gravity is an acceleration
        -> F = m * a per node; /CLOAD applies the full F(t) to every node
        of its group; /PLOAD integrates p(t) over the current surface.

        ``sensors`` (M6): /SENSOR-gated loads evaluate their curve at the
        SHIFTED time t - t_fire once their sensor fired, and are silent
        before (see engine/sensors.py)."""
        m = self._m_grav
        for idx, direction, fct, scale in self.gravity:
            acc = scale * fct.eval(t)
            fext[idx] += (m[idx, None] * acc) * direction[None, :]
        for idx, direction, fct, scale, sens in self.cloads:
            te = t if sensors is None else sensors.shifted_time(sens, t)
            if te is None:
                continue                      # sensor has not fired yet
            F = scale * fct.eval(te)
            fext[idx] += F * direction[None, :]
        for segs, wgt, fct, scale, gtype, elem, deletable, sens \
                in self.ploads:
            if len(segs) == 0:
                continue
            te = t if sensors is None else sensors.shifted_time(sens, t)
            if te is None:
                continue
            p = scale * fct.eval(te)
            if p == 0.0:
                continue
            xs = x[segs]                                  # (nseg, 4, 3)
            # area vector = 1/2 (d13 x d24): exact for the bilinear quad
            # AND for the degenerate (n4 = n3) triangle segment
            av = 0.5 * np.cross(xs[:, 2] - xs[:, 0], xs[:, 3] - xs[:, 1])
            if deletable:
                alive = tracking.alive_segment_mask(self.model, gtype, elem)
                av[~alive] = 0.0
            fseg = p * av                                 # (nseg, 3)
            for k in range(4):
                np.add.at(fext, segs[:, k], wgt[:, k, None] * fseg)

    # ------------------------------------------------------------------
    def apply_kinematic(self, t: float, v: np.ndarray, vr: np.ndarray,
                        mass: np.ndarray, x: np.ndarray,
                        dt: float, v_old: np.ndarray = None) -> float:
        """Apply /IMPVEL, /IMPDISP and /BCS to the freshly updated
        velocities (``t`` is the END of the step, t_n + dt).

        Returns the external work done by the constraints this cycle.

        Work accounting (must be consistent with the element ledger, which
        books internal work at the leapfrog MIDSTEP velocity, and with the
        contact-work booking of engine step 5b):

        * /IMPVEL: the constraint applies the impulse J = m (v_imp - v_free)
          that overwrites the free velocity with v_imp. In leap-frog an
          impulse changes the kinetic energy by exactly  J . (v^{n-1/2} +
          v^{n+1/2}) / 2  — the MIDSTEP average of the velocity BEFORE the
          cycle (v_old) and the enforced value (v_imp), NOT J . v_imp.
          Booking J . v_imp is correct only in the steady state where the
          node already moves at v_imp (v_old == v_imp); at an IMPULSIVE
          start (a curve that is non-zero at t=0, so the node jumps 0 ->
          v_imp in one cycle) it DOUBLES the work — the exact source of the
          -50% cycle-1 energy error on the RD-V-0700 imposed-velocity decks,
          because the reference is then the (over-booked) external work and
          KE = 1/2 m v_imp^2 sits at exactly half of it. This is the same
          midstep identity the original books in ``fixvel.F`` (the DW term
          ``1/4 MS (A*DT12 + 2 V)(A-AOLD)`` expands to
          ``1/2 J (v_old + v_imp)``). ``v_old`` is the velocity at the
          START of the cycle (v^{n-1/2}); the caller (engine.py) passes it.
          When it is omitted the pre-enforcement free velocity is used as a
          surrogate (their difference is O(dt^2) per cycle).
        * /IMPDISP: identical, with the imposed velocity derived from the
          exact landing condition x + v dt = x0 + d(t+dt).
        * /BCS: a permanently fixed node never moves — the trial velocity
          m a dt it briefly acquires is a phantom (it is zeroed before it
          moves anything and before any element sees it), so a fixed DOF
          contributes exactly ZERO work.
        """
        w = 0.0
        # imposed velocities first (a BCS on the same dof wins, as in the
        # original where BCS is the strongest condition). Outside the
        # [Tstart, Tstop] window the condition is simply not applied —
        # the node is free that cycle (fixvel.F CYCLEs the entry).
        for idx, dof, fct, scale, facx, tstart, tstop in self.impvel:
            if len(idx) == 0 or t < tstart or t > tstop:
                continue
            vimp = scale * fct.eval(t * facx)
            dv = vimp - v[idx, dof]              # J/m: impulse over free vel
            m = np.where(self._frozen[idx], 0.0, mass[idx])
            # midstep velocity (v^{n-1/2} + v^{n+1/2})/2 — the leapfrog work
            v_mid = 0.5 * ((v_old[idx, dof] if v_old is not None
                            else v[idx, dof]) + vimp)
            w += float(np.dot(m * dv, v_mid))
            v[idx, dof] = vimp
        # imposed displacements: land exactly at x0 + d(t_end)
        for idx, dof, fct, scale, facx, tstart, tstop, x0d in self.impdisp:
            if len(idx) == 0 or dt <= 0.0 or t < tstart or t > tstop:
                continue
            target = x0d + scale * fct.eval(t * facx)
            vimp = (target - x[idx, dof]) / dt
            dv = vimp - v[idx, dof]
            m = np.where(self._frozen[idx], 0.0, mass[idx])
            v_mid = 0.5 * ((v_old[idx, dof] if v_old is not None
                            else v[idx, dof]) + vimp)
            w += float(np.dot(m * dv, v_mid))
            v[idx, dof] = vimp
        # fixed DOFs: zero velocity (no work — see docstring)
        v[self.fix_tra] = 0.0
        vr[self.fix_rot] = 0.0
        return w
