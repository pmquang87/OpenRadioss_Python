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
from .centri import CentrifugalLoadEngine
from .impacc import ImposedAccelerationEngine


class LoadsAndConstraints:
    """Pre-resolved (index-based) loads + constraints for the Engine."""

    def __init__(self, model: Model, log, controls=None):
        self.model = model
        # BCS -> per-dof boolean masks (the GLOBAL-system conditions)
        self.fix_tra = np.zeros((model.numnod, 3), dtype=bool)
        self.fix_rot = np.zeros((model.numnod, 3), dtype=bool)
        # /BCS in a /SKEW (M39): (skew_row, node_idx, tra_mask, rot_mask)
        # per skewed condition — the constraint is a PROJECTION along the
        # skew's axes, not a mask on the global components (bcs1.F). Kept
        # apart from the mask so the (overwhelmingly common) unskewed model
        # pays nothing.
        self.skew_bcs = []
        for bc in getattr(model, "bcs", []):
            if controls is not None and not controls.bcs_active.get(bc.id, True):
                continue
            if bc.grnod_id not in getattr(model, "node_groups", {}):
                if log is not None:
                    log.warning(f"/BCS/{bc.id}: node group {bc.grnod_id} not found in model", "BCS INIT")
                continue
            idx = model.node_groups[bc.grnod_id].node_idx
            if idx is not None:
                idx = idx[(idx >= 0) & (idx < model.numnod)]
            if idx is None or len(idx) == 0:
                continue
            row = getattr(bc, "skew_row", 0)
            if row:
                self.skew_bcs.append((int(row), idx,
                                      np.asarray(bc.fix_tra, dtype=bool),
                                      np.asarray(bc.fix_rot, dtype=bool)))
                continue
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
        m_phys = getattr(model, "mass0", None)
        if m_phys is None or len(m_phys) != len(model.mass):
            m_phys = model.mass
        self._m_grav = np.where(frozen, 0.0, m_phys)

        class _ConstantFunc:
            def __init__(self, val: float = 1.0):
                self.val = float(val)
            def eval(self, t: float) -> float:
                return self.val

        def _get_func(fid: int | None):
            if fid in (None, 0):
                return _ConstantFunc(1.0)
            funcs = getattr(model, "functions", {})
            if fid in funcs:
                return funcs[fid]
            if log is not None:
                log.warning(f"Function {fid} not found in model — defaulting to constant 1.0", "LOADS INIT")
            return _ConstantFunc(1.0)

        # resolved loads: (node_idx, direction, funct, scale)
        def _grp(gid: int | None, allow_all: bool = False) -> np.ndarray:
            if gid in (None, 0):
                return np.arange(model.numnod) if allow_all else np.zeros(0, dtype=np.int64)
            ngroups = getattr(model, "node_groups", {})
            if gid in ngroups:
                idx = ngroups[gid].node_idx
                if idx is not None:
                    idx = idx[(idx >= 0) & (idx < model.numnod)]
                return idx if idx is not None else np.zeros(0, dtype=np.int64)
            if log is not None:
                log.warning(f"Node group {gid} not found in model", "LOADS INIT")
            return np.zeros(0, dtype=np.int64)

        self.gravity = []
        for g in getattr(model, "gravity", []):
            idx = _grp(g.grnod_id, allow_all=True)
            if idx is None or len(idx) == 0:
                continue
            self.gravity.append((idx, g.direction, _get_func(g.funct_id), g.scale))

        # /CLOAD entries carry their /SENSOR id (M6): 0 = always active
        self.cloads = []
        for c in getattr(model, "cloads", []):
            idx = _grp(c.grnod_id)
            if idx is None or len(idx) == 0:
                continue
            self.cloads.append((idx, c.direction, _get_func(c.funct_id),
                                c.scale, c.sens_id, c.time_scale))

        # /IMPVEL entries: (node_idx, dof, funct, Fscale_Y, 1/Ascale_x,
        # Tstart, Tstop) — the curve is evaluated at t/Ascale_x and the
        # condition only holds inside [Tstart, Tstop] (fixvel.F: FACX,
        # STARTT/STOPT). dof 0..2 drive the translational velocity, 3..5
        # (XX/YY/ZZ) the ANGULAR velocity (M39; see apply_kinematic).
        # A condition that names a /SKEW is NOT here: it
        # goes to skew_impvel/skew_impdisp below and is applied along the
        # skew axis instead — one condition, ONE application (listing it
        # in both would impose the curve twice, on the global column AND
        # on the skew axis).
        def _skewed(i):
            return bool(getattr(i, "skew_row", 0))

        self.impvel = []
        for i in getattr(model, "impvel", []):
            if _skewed(i):
                continue
            idx = _grp(i.grnod_id)
            if idx is None or len(idx) == 0:
                continue
            sens_id = int(getattr(i, "sens_id", 0) or 0)
            self.impvel.append((idx, i.dof, _get_func(i.funct_id),
                                i.scale, 1.0 / i.xscale if getattr(i, "xscale", 1.0) not in (0.0, None) else 1.0,
                                i.tstart, i.tstop, sens_id))

        # /IMPDISP: like /IMPVEL, plus the base coordinate of each node so
        # the target position x0 + d(t) is exact (no velocity-integration
        # drift). Entries: (node_idx, dof, funct, scale, facx, tstart,
        # tstop, x0_dof, sens_id).
        self.impdisp = []
        for i in getattr(model, "impdisp", []):
            if _skewed(i):
                continue
            idx = _grp(i.grnod_id)
            if idx is None or len(idx) == 0:
                continue
            # a ROTATIONAL /IMPDISP (dof 3..5, M39) has no base position to
            # correct against — x0d is left zero and the imposed ANGLE is
            # enforced as the finite-difference angular velocity in
            # apply_kinematic (exact for a DOF driven from d(tstart)=0).
            x0d = (model.x0[idx, i.dof].copy() if i.dof < 3 and len(idx) > 0
                   else np.zeros(len(idx)))
            xscale = getattr(i, "xscale", 1.0)
            facx = 1.0 / xscale if xscale not in (0.0, None) else 1.0
            sens_id = int(getattr(i, "sens_id", 0) or 0)
            self.impdisp.append((idx, i.dof, _get_func(i.funct_id),
                                 i.scale, facx, i.tstart, i.tstop,
                                 x0d, sens_id))
        # /IMPVEL + /IMPDISP in a /SKEW (M39): the imposed component is the
        # one along the skew's Dir axis (fixvel.F 390-418), so the base
        # coordinate an /IMPDISP lands against is the skew PROJECTION of
        # x0, not one Cartesian column. Split out of the two lists above so
        # the unskewed fast path is untouched; each entry carries the skew
        # row (the axes are re-read every cycle — a /SKEW/MOV turns).
        self.skew_impvel, self.skew_impdisp = [], []
        for i in getattr(model, "impvel", []):
            row = getattr(i, "skew_row", 0)
            if row:
                idx = _grp(i.grnod_id)
                if idx is None or len(idx) == 0:
                    continue
                xscale = getattr(i, "xscale", 1.0)
                facx = 1.0 / xscale if xscale not in (0.0, None) else 1.0
                self.skew_impvel.append(
                    (int(row), idx, i.dof,
                     _get_func(i.funct_id), i.scale, facx,
                     i.tstart, i.tstop, None))
        for i in getattr(model, "impdisp", []):
            row = getattr(i, "skew_row", 0)
            if row:
                idx = _grp(i.grnod_id)
                if idx is None or len(idx) == 0:
                    continue
                xscale = getattr(i, "xscale", 1.0)
                facx = 1.0 / xscale if xscale not in (0.0, None) else 1.0
                x0_sub = model.x0[idx].copy() if len(idx) > 0 else np.zeros((0, 3))
                self.skew_impdisp.append(
                    (int(row), idx, i.dof, _get_func(i.funct_id),
                     i.scale, facx, i.tstart, i.tstop,
                     x0_sub))
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
            if len(idx) == 0:
                continue
            fzn = idx[frozen[idx]]
            if dof < 3:
                self.fix_tra[fzn, dof] = False
            else:
                self.fix_rot[fzn, dof - 3] = False
        for entry in self.skew_impvel + self.skew_impdisp:
            # a skewed condition drives a DIRECTION, not one global column:
            # release the frozen carrier on all three (the skew axis is a
            # combination of them and the other two stay free anyway —
            # nothing else fixes them).  dof 0..2 drive translation (release
            # the translational auto-fix), dof 3..5 drive rotation about the
            # skew axis (release the rotational auto-fix instead) — M40.
            idx, dof = entry[1], entry[2]
            if len(idx) == 0:
                continue
            fzn = idx[frozen[idx]]
            if dof >= 3:
                self.fix_rot[fzn, :] = False
            else:
                self.fix_tra[fzn, :] = False
        for i in getattr(model, "impdisp", []):
            f0 = _get_func(i.funct_id).eval(0.0) * i.scale
            if abs(f0) > 0.0 and log is not None:
                log.warning(f"/IMPDISP/{i.id}: curve starts at "
                            f"d(0) = {f0:.4g} != 0 — the nodes will JUMP "
                            f"there in the first cycle", "IMPDISP INIT")

        # /PLOAD: resolved segments + triangle lumping weights + deletion
        # provenance (a torn face stops carrying pressure)
        self.ploads = []
        for pl in getattr(model, "ploads", []):
            surfs = getattr(model, "surfaces", {})
            if pl.surf_id not in surfs:
                if log is not None:
                    log.warning(f"/PLOAD/{pl.id}: surface {pl.surf_id} not found in model — load inactive", "PLOAD INIT")
                continue
            surf = surfs[pl.surf_id]
            segs = surf.segments if surf.segments is not None else \
                np.zeros((0, 4), dtype=np.int64)
            if len(segs) == 0 and log is not None:
                log.warning(f"/PLOAD/{pl.id}: surface {pl.surf_id} has no "
                            f"segments — load inactive", "PLOAD INIT")
            tri = ((segs[:, 3] == segs[:, 2]) | (segs[:, 3] < 0)) if len(segs) else \
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
            self.ploads.append((segs, wgt, _get_func(pl.funct_id),
                                pl.scale, gtype, elem, deletable,
                                pl.sens_id))

        # Centrifugal loads and Imposed accelerations (M595)
        self.centri_engine = CentrifugalLoadEngine(model, log)
        self.impacc_engine = ImposedAccelerationEngine(model, log)
        self.impacc = self.impacc_engine.entries

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
        for idx, direction, fct, scale, sens, t_scale in self.cloads:
            te = t if sensors is None else sensors.shifted_time(sens, t)
            if te is None:
                continue                      # sensor has not fired yet
            if t_scale != 1.0 and t_scale != 0.0:
                te = te / t_scale
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
            xs = x[segs].copy()                                  # (nseg, 4, 3)
            degen = (segs[:, 3] == segs[:, 2]) | (segs[:, 3] < 0)
            if np.any(degen):
                xs[degen, 3] = xs[degen, 2]
            # area vector = 1/2 (d13 x d24): exact for the bilinear quad
            # AND for the degenerate (n4 = n3) triangle segment
            av = 0.5 * np.cross(xs[:, 2] - xs[:, 0], xs[:, 3] - xs[:, 1])
            if deletable:
                alive = tracking.alive_segment_mask(self.model, gtype, elem)
                av[~alive] = 0.0
            fseg = p * av                                 # (nseg, 3)
            for k in range(4):
                valid_k = wgt[:, k] > 0.0
                if np.any(valid_k):
                    nodes_k = segs[valid_k, k]
                    valid_nodes = (nodes_k >= 0) & (nodes_k < len(fext))
                    if np.any(valid_nodes):
                        np.add.at(fext, nodes_k[valid_nodes], (wgt[valid_k, k, None] * fseg[valid_k])[valid_nodes])

        # Centrifugal body forces (/LOAD/CENTRI and /CENTRI, M595)
        self.centri_engine.compute_forces(t, x, fext, sensors)

    # ------------------------------------------------------------------
    @property
    def impacc_reactions(self):
        """Reaction forces from /IMPACC imposed accelerations."""
        return self.impacc_engine.reactions

    def apply_acceleration(self, t: float, dt: float,
                           acc: np.ndarray, ar: Optional[np.ndarray],
                           mass: np.ndarray, inertia: Optional[np.ndarray],
                           v: np.ndarray, vr: Optional[np.ndarray],
                           v_old: np.ndarray, vr_old: Optional[np.ndarray],
                           sensors=None) -> float:
        """Enforce /IMPACC conditions during acceleration update (Step 4, M595),
        tracking reaction forces and returning external work."""
        return self.impacc_engine.apply(
            t, dt, acc, ar, mass, inertia, v, vr, v_old, vr_old, sensors
        )

    # ------------------------------------------------------------------
    def apply_kinematic(self, t: float, v: np.ndarray, vr: np.ndarray,
                        mass: np.ndarray, x: np.ndarray,
                        dt: float, v_old: np.ndarray = None,
                        inertia: np.ndarray = None,
                        vr_old: np.ndarray = None,
                        sensors=None) -> float:
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

        Rotational conditions (dof 3..5 = XX/YY/ZZ, M39) are the exact
        angular analogue: they overwrite the ANGULAR velocity ``vr`` (not
        ``v``), the impulse is the angular impulse J_rot (vimp - vr_free)
        with the nodal rotational INERTIA in place of the mass, and the
        midstep uses ``vr_old``. ``inertia``/``vr_old`` are the engine's
        rotational counterparts to ``mass``/``v_old`` (omitted by the pure
        translational unit-test callers; a rotational entry then falls back
        to ``mass``/``vr`` — never exercised in practice). When the driven
        group is an /RBODY master the entry has already been consumed by
        rigid_body.RigidBodyEngine (its idx is empty here), which drives
        the body spin so the constraint propagates to the slave nodes.

        /SKEW (M39): a condition that names a skew acts along that skew's
        AXES.  The skew rows are read fresh every cycle, so a /SKEW/MOV
        constraint turns with its nodes (the Engine rebuilds the moving
        rows once per cycle — see engine.py's NEWSKW step).
        """
        w = 0.0
        skews = getattr(self.model, "skews", None)

        def _book(vel, gen, gold, d, vimp, idx):
            # shared midstep booking for one imposed DOF ``d`` of the
            # velocity array ``vel`` (v or vr) against the generalized
            # inertia ``gen`` (mass or rotational inertia): the leapfrog
            # impulse J = gen (vimp - vel_free) changes the energy by
            # J . (gold + vimp)/2 — see the docstring's midstep identity.
            dv = vimp - vel[idx, d]
            g = np.where(self._frozen[idx], 0.0, gen[idx])
            v_mid = 0.5 * ((gold[idx, d] if gold is not None
                            else vel[idx, d]) + vimp)
            vel[idx, d] = vimp
            return float(np.dot(g * dv, v_mid))

        rot_gen = inertia if inertia is not None else mass
        # imposed velocities first (a BCS on the same dof wins, as in the
        # original where BCS is the strongest condition). Outside the
        # [Tstart, Tstop] window the condition is simply not applied —
        # the node is free that cycle (fixvel.F CYCLEs the entry).
        # dof 0..2 overwrite the translational velocity, 3..5 the ANGULAR
        # velocity ``vr`` against the rotational inertia (M39).
        for entry in self.impvel:
            idx, dof, fct, scale, facx, tstart, tstop = entry[:7]
            sens_id = entry[7] if len(entry) > 7 else 0
            if len(idx) == 0:
                continue
            if sensors is not None and not sensors.active(sens_id):
                continue
            te = t if (sensors is None or sens_id == 0) else sensors.shifted_time(sens_id, t)
            if te is None or te < tstart or te > tstop:
                continue
            t_mid = te - 0.5 * dt
            vimp = scale * fct.eval(t_mid * facx)
            if dof < 3:
                w += _book(v, mass, v_old, dof, vimp, idx)
            else:
                w += _book(vr, rot_gen, vr_old, dof - 3, vimp, idx)
        # imposed displacements: land exactly at x0 + d(t_end)
        for entry in self.impdisp:
            idx, dof, fct, scale, facx, tstart, tstop, x0d = entry[:8]
            sens_id = entry[8] if len(entry) > 8 else 0
            if len(idx) == 0 or dt <= 0.0:
                continue
            if sensors is not None and not sensors.active(sens_id):
                continue
            te = t if (sensors is None or sens_id == 0) else sensors.shifted_time(sens_id, t)
            if te is None or te < tstart or te > tstop:
                continue
            if dof < 3:
                target = x0d + scale * fct.eval(te * facx)
                vimp = (target - x[idx, dof]) / dt
                w += _book(v, mass, v_old, dof, vimp, idx)
            else:
                # rotational /IMPDISP: advance the imposed ANGLE d(t) at the
                # finite-difference rate over this step — there is no stored
                # nodal angle to read back (the translational branch reads
                # x[idx,dof]); exact for a DOF driven from d(tstart)=0.
                vimp = scale * (fct.eval(te * facx)
                                - fct.eval((te - dt) * facx)) / dt
                w += _book(vr, rot_gen, vr_old, dof - 3, vimp, idx)
        # ---- /IMPVEL + /IMPDISP in a /SKEW (fixvel.F 390-418) ------------
        # The curve is imposed on the component ALONG the skew's Dir axis;
        # the two transverse components stay free.  The reference projects
        # the current velocity onto the axis (VV), turns the curve into the
        # needed acceleration and adds the correction back along the SAME
        # axis: A += e * (YC - A0).  At the velocity level (where the port
        # applies its kinematic conditions) that is exactly
        # v <- v + e * (v_imp - e.v).
        for row, idx, dof, fct, scale, facx, tstart, tstop, x0 in \
                self.skew_impvel + self.skew_impdisp:
            if len(idx) == 0 or t < tstart or t > tstop:
                continue
            # dof 0..2 drive the TRANSLATIONAL velocity ``v`` along the
            # skew's (dof)-th axis; dof 3..5 drive the ANGULAR velocity
            # ``vr`` about the skew's (dof-3)-th axis — the exact rotational
            # analogue (fixvel.F runs the SAME SKEW projection VV/A0/AA on
            # the VR/AR arrays for a rotational DOF, driving rotation about
            # skew axis J).  Before M40 a rotational /IMPVEL naming a skew
            # indexed ``axes[row][dof]`` (dof 3..5) out of the (3,3) axes and
            # raised IndexError; now it selects axis (dof-3) and drives ``vr``
            # against the rotational inertia, matching the unskewed rotational
            # branch above.
            rot = dof >= 3
            e = skews.axes[row][dof - 3 if rot else dof]   # the driven axis
            vel = vr if rot else v
            gen = rot_gen if rot else mass
            gold = vr_old if rot else v_old
            vn = vel[idx] @ e                         # VV: current comp along e
            if x0 is None:                            # /IMPVEL: v(t) given
                t_mid = t - 0.5 * dt
                vimp = scale * fct.eval(t_mid * facx)
            else:                                     # /IMPDISP: d(t) given
                if dt <= 0.0:
                    continue
                if rot:
                    # rotational /IMPDISP: no stored nodal angle to project
                    # back — finite-difference the imposed angle over the
                    # step, exactly as the unskewed rotational branch above
                    vimp = scale * (fct.eval(t * facx)
                                    - fct.eval((t - dt) * facx)) / dt
                else:
                    # land on the imposed displacement measured along the
                    # axis from the original position (fixvel.F's DD=SKEW.D)
                    target = (x0 @ e) + scale * fct.eval(t * facx)
                    vimp = (target - (x[idx] @ e)) / dt
            dv = vimp - vn                            # the impulse / gen
            g = np.where(self._frozen[idx], 0.0, gen[idx])
            v_mid = 0.5 * ((gold[idx] @ e if gold is not None else vn)
                           + vimp)
            w += float(np.dot(g * dv, v_mid))
            vel[idx] += np.outer(dv, e)
        # ---- fixed DOFs: zero velocity (no work — see docstring) ---------
        v[self.fix_tra] = 0.0
        vr[self.fix_rot] = 0.0
        if getattr(self.model, "a", None) is not None:
            self.model.a[self.fix_tra] = 0.0
        # ---- /BCS in a /SKEW (bcs1v, the "USER SYSTEM" branch) -----------
        # Each constrained skew axis has its component PROJECTED OUT of the
        # velocity: VV = e.V ; V -= e VV.  The reference does the same to
        # the acceleration and applies one axis after another (LCOD 3/5/6
        # are two sequential projections) — legitimate because the axes are
        # orthonormal, so the order does not matter and the result is the
        # projection onto the free subspace.  Zero work: the removed
        # component is a phantom the node never moves with.
        for row, idx, ftra, frot in self.skew_bcs:
            if len(idx) == 0:
                continue
            axes = skews.axes[row]
            for d in range(3):
                if ftra[d]:
                    e = axes[d]
                    v[idx] -= np.outer(v[idx] @ e, e)
                    if getattr(self.model, "a", None) is not None:
                        self.model.a[idx] -= np.outer(self.model.a[idx] @ e, e)
                if frot[d]:
                    e = axes[d]
                    vr[idx] -= np.outer(vr[idx] @ e, e)
        return w
