"""
/RBODY and /RBE2 — rigid bodies and rigid links, engine side (M5).

Fortran origin: ``engine/source/constraints/general/rbody/`` —

    rbyfor.F   gather of the slave forces/moments into the body resultant
    rbycor.F   the 6-DOF rigid equation of motion + velocity scatter
    rbyact.F   activation bookkeeping
    (and ``constraints/general/rbe2/`` for the rigid links, which share
    the same mechanics in this port — see the RigidBody entity docstring)

Theory — the constrained rigid body in an explicit code
-------------------------------------------------------
A rigid body replaces the N slave nodes' 3N equations of motion by six:
the Newton-Euler equations of the assembly,

    M  dv_g/dt = F        F = sum of all forces on the body's nodes
    dL/dt      = T        T = sum of moments about the reference point

with L the angular momentum, L = J(t) w, and J(t) = R J0 R^T the inertia
tensor co-rotating with the body. The port integrates these the way the
gyroscopically *stable* explicit schemes do:

1.  **Gather** (rbyfor): the assembled nodal forces (internal + external +
    contact) and shell moments of every body node are reduced to F and T
    about the reference point, using the CURRENT node positions for the
    moment arms.

2.  **Angular momentum update**:  L <- L + T dt.  Updating L itself (not
    w) makes conservation exact by construction: a torque-free body keeps
    L to round-off forever, whatever the tumbling does — this is the
    rotational analogue of updating momentum instead of velocity, and it
    sidesteps the explicit-Euler instability of integrating the Euler
    equations  J dw/dt = T - w x Jw  directly (the gyroscopic term is
    handled *implicitly* by re-solving w from the rotated inertia).

3.  **Spin from momentum**:  w = (R J0 R^T)^{-1} L  — a 3x3 solve.

4.  **Scatter**: every body node gets the rigid velocity field
    v_s = v_ref + w x r_s (same arms r_s as the gather, which makes the
    constraint *do no work by construction*: sum f_s . v_s = F . v_ref +
    T . w exactly — the discrete power identity of the force/velocity
    dual pair). Rotational DOFs get vr_s = w.

5.  **Finite rotation** (after the position update): the body frame is
    advanced with the exponential map, R <- exp([w dt]) R, the Rodrigues
    closed form — an orthogonal update for ANY step size, unlike adding
    w dt x r to the positions (which spirals outward). The node positions
    are then *placed*: x_s = x_ref + R r0_s. Drift is impossible: the
    body's shape is reconstructed from the initial offsets every cycle.

Boundary conditions on the body
-------------------------------
A /BCS on the *master node* is honored at the body level (the node-level
treatment is disabled for it):

* translations fixed on a component: that component of v_ref is zeroed
  (fixing all three of an RBE2 master turns the link into a PIVOT: the
  body then rotates about the master point, with the inertia tensor
  transported there by the parallel-axis theorem — the physical-pendulum
  configuration);
* rotations fixed on a component: that component of w is zeroed and L is
  reset to J w so the removed spin does not come back.

An /IMPVEL whose group contains the master drives that component of the
body velocity (a moving rigid die); its work is booked exactly like the
nodal /IMPVEL, J . v_imp with J = M (v_imp - v_free). A ROTATIONAL
/IMPVEL or /IMPDISP (dir XX/YY/ZZ, M39) instead drives that component of
the body SPIN — the way the RD-E-1000 Bending decks roll a strip into a
circle by spinning a rigid body about X — booked with the rotational
midstep identity dL . (w_old + w_imp)/2 (see advance). BCS or /IMPVEL on
*slave* nodes clash with the rigidity and are warned about (the body
wins). Rigid walls do not know about bodies (the wall would fight the
enforcement) — use contact interfaces against rigid bodies instead.

Interaction with the rest of the solver (the M4 lessons applied)
----------------------------------------------------------------
* **Contact**: slave nodes remain ordinary contact secondaries/mains —
  their fcont rows are gathered like any force. Because the body velocity
  update runs BEFORE the Engine books the contact work, the booking sees
  the actual (rigid) velocities the contact forces act through — the
  midstep-velocity lesson holds unchanged.
* **Effective mass** (/INTER/TYPE2): a tied secondary hanging on a body
  node transfers its force to that node — so its *inertia* must also be
  in the body's mass/inertia, or the body would out-accelerate the
  physics. ``finalize_mass`` folds the (mass_eff - mass) increments of
  the body nodes into M and J.
* **Element deletion**: nodal masses never change on deletion, so the
  Starter-computed M, xg, J stay exact (documented in rbyini's port).
* **Time step**: the body is kinematic — it adds no stiffness and claims
  no dt. Elements fully inside a body still claim theirs (they never
  deform, so the claim is a constant, conservative bound; the original
  removes them from the loop — a pure optimization this port skips).

Energy: the constraint books NOTHING (see step 4 — zero work by
construction, asserted by the M5 tests); only the master /IMPVEL drive
books its external work, exactly like kinematics.py does.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..common.fastmath import cross3
from ..model.model import Model


def _exp_rotation(w: np.ndarray, dt: float) -> np.ndarray:
    """Rodrigues closed form of exp([w dt]): the rotation by angle |w| dt
    about axis w/|w|. Exactly orthogonal for any step size."""
    th = float(np.linalg.norm(w)) * dt
    if th < 1e-12:
        return np.eye(3) + dt * _skew(w)      # first order is exact enough
    k = w / np.linalg.norm(w)
    K = _skew(k)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    """Project R back onto SO(3) (polar decomposition via SVD): the
    Rodrigues update is orthogonal in exact arithmetic, but round-off
    accumulates over ~1e5 cycles; this keeps the frame clean."""
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


class RigidBodyEngine:
    """One /RBODY or /RBE2, engine-side.

    ``saved`` (M6 restart chaining): the dynamic state written by a
    previous engine run — when given, the initial-velocity projection is
    SKIPPED (the nodal velocities in the restart are already the exact
    rigid field of the saved state, and the body frame R cannot be
    reconstructed from positions) and the saved (R, L, v_ref, w, x_ref,
    xg) are restored instead. Everything static (masses, offsets, BCS
    flags, drives) is reconstructed deterministically from the model.
    """

    def __init__(self, rb, model: Model, loads, log, saved=None):
        self.rb = rb
        self.model = model
        who = f"/{getattr(rb, 'kind', 'RBODY')}/{getattr(rb, 'id', 1)}"
        self.master = int(getattr(rb, "master", 0))
        slaves = getattr(rb, "slaves", None)
        if slaves is None:
            self.slaves = np.zeros(0, dtype=np.int64)
        else:
            self.slaves = np.asarray(slaves, dtype=np.int64)
            self.slaves = self.slaves[self.slaves != self.master]
        self.nodes = np.concatenate([[self.master], self.slaves])

        mass_tot = float(getattr(rb, "mass_total", 0.0))
        self.M = max(mass_tot, 1e-20)
        j_val = getattr(rb, "J", None)
        self.J0 = j_val.copy() if j_val is not None else np.eye(3) * 1e-20
        xg_val = getattr(rb, "xg", None)
        self.xg = xg_val.copy() if xg_val is not None else (
            model.x0[self.master].copy() if len(model.x0) > self.master else np.zeros(3)
        )
        self.x_cg0 = self.xg.copy()    # initial COG position
        self.f_res = np.zeros(3)       # resultant force
        self.m_res = np.zeros(3)       # resultant moment
        self.R = np.eye(3)

        # ---- body-level boundary conditions from the MASTER's /BCS ------
        # (the node-level flags for the whole body are cleared: the body
        # supersedes both real BCS on its nodes and the automatic fixing
        # of frozen massless nodes — a massless master/slave is not dead,
        # it is carried by the body)
        self.fix_tra = np.zeros(3, dtype=bool)
        self.fix_rot = np.zeros(3, dtype=bool)
        # /BCS in a /SKEW on the MASTER (M39): the body's translation /
        # rotation is constrained along the SKEW's axes, so it cannot be
        # folded into the global fix_tra/fix_rot masks — it is a projection
        # applied to v_ref / w (bcs1v's USER SYSTEM branch, lifted to the
        # body's 6-DOF reference kinematics).  Entries: (skew_row, tra, rot).
        # RD-V-0220 (foam LAW70) is the corpus deck that does this: its
        # /BCS/1+/BCS/2 name /SKEW/MOV/1 and sit on the two /RBODY masters.
        self.bc_skew = []
        self.skews = getattr(model, "skews", None)
        bcs = getattr(model, "bcs", [])
        node_groups = getattr(model, "node_groups", {})
        for bc in bcs:
            grp = node_groups.get(bc.grnod_id)
            if grp is None or getattr(grp, "node_idx", None) is None:
                continue
            if self.master in grp.node_idx:
                row = int(getattr(bc, "skew_row", 0) or 0)
                ftra = np.asarray(bc.fix_tra, dtype=bool)
                frot = np.asarray(bc.fix_rot, dtype=bool)
                if row and not (ftra.all() and frot.all()):
                    # a FULL clamp is skew-invariant (zeroing the vector is
                    # the same in any orthonormal basis) — keep it on the
                    # cheap global mask; anything partial must project.
                    self.bc_skew.append((row, ftra, frot))
                    continue
                self.fix_tra |= ftra
                self.fix_rot |= frot
            if len(self.slaves) > 0 and np.isin(self.slaves, grp.node_idx).any():
                if log is not None:
                    log.warning(f"{who}: /BCS/{bc.id} touches slave node(s) — "
                                f"the rigid body wins (kinematic clash); put "
                                f"the BCS on the master node instead",
                                "RBODY INIT")
        if hasattr(loads, "fix_tra") and loads.fix_tra is not None and len(loads.fix_tra) > 0:
            loads.fix_tra[self.nodes] = False
        if hasattr(loads, "fix_rot") and loads.fix_rot is not None and len(loads.fix_rot) > 0:
            loads.fix_rot[self.nodes] = False

        # ---- pivot mode: master translations fully fixed -----------------
        # The body then rotates about the (fixed) master point; transport
        # the inertia tensor there (parallel-axis / Huygens-Steiner).
        self.pivot = bool(self.fix_tra.all())
        if self.pivot:
            self.x_ref = model.x0[self.master].copy() if len(model.x0) > self.master else np.zeros(3)
            c = self.xg - self.x_ref
            self.J0 = self.J0 + self.M * (np.eye(3) * float(c @ c)
                                          - np.outer(c, c))
            self.c0 = c.copy()         # COG offset in the (t=0) body frame
        else:
            self.x_ref = self.xg.copy()

        # initial node offsets from the reference point (body frame = t0)
        self.r0 = (model.x0[self.nodes] - self.x_ref) if len(model.x0) > 0 else np.zeros((len(self.nodes), 3))

        # ---- /IMPVEL driving the master: body-velocity drive --------------
        # (removed from the nodal treatment so its huge-frozen-mass work
        # booking cannot fire; slaves under /IMPVEL are a clash: warn and
        # remove them there too — the body wins)
        #
        # M39: a ROTATIONAL /IMPVEL (dof 3..5 = XX/YY/ZZ) on the master
        # drives that component of the body SPIN instead of v_ref — the
        # RD-E-1000 Bending decks spin a rigid body about X to roll a strip
        # into a circle. It is applied after the angular-momentum update
        # (advance), the exact rotational analogue of the translational
        # drive here.
        self.drives = []               # translational: (dof, funct, scale)
        self.rot_drives = []           # rotational: (rdof, fct, scale,
        #                                             facx, tstart, tstop)
        impvel_list = getattr(loads, "impvel", [])
        for k, entry in enumerate(impvel_list):
            idx, dof, fct, scale = entry[0], entry[1], entry[2], entry[3]
            if self.master in idx:
                if dof < 3:
                    self.drives.append((dof, fct, scale) + entry[4:7])
                else:
                    self.rot_drives.append((dof - 3, fct, scale) + entry[4:7])
            hit = np.isin(idx, self.nodes)
            if np.any(hit):
                if len(self.slaves) > 0 and np.isin(self.slaves, idx).any() and log is not None:
                    log.warning(f"{who}: /IMPVEL drives slave node(s) — "
                                f"the rigid body wins (kinematic clash)",
                                "RBODY INIT")
                impvel_list[k] = (idx[~hit],) + entry[1:]
        # ---- /IMPDISP driving the master: body-displacement drive ---------
        # (M37: the standard way the official decks move a rigid platen —
        # RD-V-0220 drives the /RBODY master with a /FUNCT_SMOOTH ramp.
        # The master dof must land on x0 + d(t) exactly, so the body
        # velocity is set from the CURRENT master position each cycle,
        # like the nodal /IMPDISP treatment in kinematics.apply.)
        # Slaves under /IMPDISP remain a clash: warn, the body wins.
        self.disp_drives = []          # (dof, fct, scale, facx, t0, t1, x0)
        # M39: rotational /IMPDISP master drive — (rdof, fct, scale, facx,
        # t0, t1); the imposed ANGLE is enforced as a finite-difference
        # spin (no base angle to correct against, like the nodal branch in
        # kinematics.apply_kinematic).
        self.rot_disp_drives = []
        impdisp_list = getattr(loads, "impdisp", [])
        for k, entry in enumerate(impdisp_list):
            idx, dof, x0d = entry[0], entry[1], entry[-1]
            if self.master in idx:
                if dof < 3:
                    pos = int(np.where(idx == self.master)[0][0])
                    self.disp_drives.append(entry[1:-1] + (float(x0d[pos]),))
                else:
                    self.rot_disp_drives.append((dof - 3,) + entry[2:-1])
            hit = np.isin(idx, self.nodes)
            if np.any(hit):
                if len(self.slaves) > 0 and np.isin(self.slaves, idx).any() and log is not None:
                    log.warning(f"{who}: /IMPDISP drives slave node(s) — "
                                f"the rigid body wins (kinematic clash)",
                                "RBODY INIT")
                impdisp_list[k] = (idx[~hit],) + entry[1:-1] + (
                    x0d[~hit] if x0d is not None else None,)
        # a pivoted (fully translation-clamped) master cannot TRANSLATE, so
        # translational drives are moot; a ROTATIONAL drive about the pivot
        # is perfectly valid, so it is NOT warned away.
        if self.pivot and self.disp_drives and log is not None:
            log.warning(f"{who}: /IMPDISP on a pivoted (fully clamped) "
                        f"master is ignored", "RBODY INIT")
        if self.pivot and self.drives and log is not None:
            log.warning(f"{who}: /IMPVEL on a pivoted (fully clamped) "
                        f"master is ignored", "RBODY INIT")

        # ---- /IMPVEL + /IMPDISP in a /SKEW driving the master (M39) -------
        # The skewed conditions live in their own lists (kinematics.py), so
        # they must be picked up here too — otherwise the nodal path would
        # drive the master while the body ALSO places it (a silent clash).
        # A skew drive imposes the curve on the master's velocity component
        # along the skew axis; the transverse components stay with the body.
        # Entries: (skew_row, dof, fct, scale, facx, t0, t1, x0_vec|None).
        self.skew_drives = []
        for name in ("skew_impvel", "skew_impdisp"):
            lst = getattr(loads, name, None)
            if not lst:
                continue
            for k, entry in enumerate(lst):
                row, idx = entry[0], entry[1]
                x0 = entry[-1]
                if self.master in idx:
                    pos = int(np.where(idx == self.master)[0][0])
                    self.skew_drives.append(
                        entry[:1] + entry[2:-1]
                        + (x0[pos].copy() if x0 is not None else None,))
                hit = np.isin(idx, self.nodes)
                if np.any(hit):
                    if len(self.slaves) > 0 and np.isin(self.slaves, idx).any() and log is not None:
                        log.warning(f"{who}: a skewed /IMP* drives slave "
                                    f"node(s) — the rigid body wins "
                                    f"(kinematic clash)", "RBODY INIT")
                    lst[k] = (entry[0], idx[~hit]) + entry[2:-1] + (
                        x0[~hit] if x0 is not None else None,)
        if self.pivot and self.skew_drives and log is not None:
            log.warning(f"{who}: a skewed /IMP* on a pivoted (fully "
                        f"clamped) master is ignored", "RBODY INIT")

        # ---- restart resume (M6): restore the dynamic state ---------------
        if saved is not None:
            self.R = saved["R"].copy()
            self.L = saved["L"].copy()
            self.v_ref = saved["v_ref"].copy()
            self.w = saved["w"].copy()
            self.x_ref = saved["x_ref"].copy()
            self.xg = saved["xg"].copy()
            if log is not None:
                log.info(f"     {who}: RESUMED (RESTART)")
            return

        # ---- initial state: project the nodal velocities ------------------
        # (an /INIVEL field on the slaves may not be exactly rigid; the
        # body can only carry its rigid part: v_g and L are the momenta of
        # the initial field, and the scatter below makes the nodal
        # velocities consistent with them)
        m = model.mass[self.nodes].copy() if len(model.mass) > 0 else np.zeros(len(self.nodes))
        m[m >= 1e29] = 0.0             # frozen = massless placeholder
        v0 = model.v[self.nodes] if len(model.v) > 0 else np.zeros((len(self.nodes), 3))
        msum = float(m.sum())
        self.v_ref = ((m[:, None] * v0).sum(axis=0) / msum if msum > 0
                      else np.zeros(3))
        r = (model.x0[self.nodes] - self.x_ref) if len(model.x0) > 0 else np.zeros((len(self.nodes), 3))
        self.L = np.cross(r, m[:, None] * v0).sum(axis=0)
        inertia = getattr(model, "inertia", None)
        if inertia is not None and len(inertia) == len(model.mass):
            vr = getattr(model, "vr", None)
            if vr is not None and len(vr) == len(model.mass):
                self.L += (inertia[self.nodes, None] * vr[self.nodes]).sum(axis=0)
        if self.pivot:
            self.v_ref = np.zeros(3)
        self.v_ref[self.fix_tra] = 0.0
        try:
            w = np.linalg.solve(self.J0, self.L)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(self.J0) @ self.L
        w[self.fix_rot] = 0.0
        self._apply_skew_bcs(self.v_ref, w)
        self.L = self.J0 @ w
        self.w = w
        if len(model.v) > 0 and len(model.x) > 0:
            model.v[self.nodes] = self._rigid_field(model.x[self.nodes])
        if len(getattr(model, "vr", [])) > 0:
            model.vr[self.nodes] = w

        if log is not None:
            log.info(f"     {who}: {len(self.slaves)} SLAVE NODE(S), MASS = "
                     f"{self.M:12.5E}" + ("  [PIVOTED AT MASTER]"
                                          if self.pivot else ""))

    # ------------------------------------------------------------------
    def _apply_skew_bcs(self, v_ref: np.ndarray, w: np.ndarray) -> None:
        """Project the /BCS-constrained SKEW axes out of the body's
        reference velocity and spin, IN PLACE (M39).

        The body lift of bcs1v's USER SYSTEM branch: each constrained skew
        axis has its component removed (``VV = e.V ; V -= e VV``), one
        orthonormal axis after another.  The rows are read fresh, so a
        /SKEW/MOV constraint turns with its nodes.
        """
        if not self.bc_skew:
            return
        for row, ftra, frot in self.bc_skew:
            axes = self.skews.axes[row]
            for d in range(3):
                if ftra[d]:
                    e = axes[d]
                    v_ref -= e * float(v_ref @ e)
                if frot[d]:
                    e = axes[d]
                    w -= e * float(w @ e)

    def _apply_skew_drives(self, x: np.ndarray, t: float, dt: float,
                           v_ref_old: Optional[np.ndarray] = None) -> float:
        """Impose a skewed /IMPVEL or /IMPDISP that drives the master, on
        the body's reference velocity (M39); returns its external work.

        Only the component along the skew's Dir axis is imposed — the body
        keeps its own motion transverse to it (fixvel.F 390-418).  The
        /IMPDISP form lands the MASTER on x0 + d(t) along that axis, the
        same exact-landing rule the nodal and global-dof body paths use;
        the spin transport w x (x_m - x_ref) is removed so it is the
        REFERENCE point's velocity that gets prescribed.  Work is booked
        J . v_imp, the convention of the global-dof drives above.
        """
        if not self.skew_drives:
            return 0.0
        wext = 0.0
        Jsp = self.R @ self.J0 @ self.R.T
        for row, dof, fct, scale, facx, t0, t1, x0 in self.skew_drives:
            if t < t0 or t > t1:
                continue
            axis_idx = dof - 3 if dof >= 3 else dof
            e = self.skews.axes[row][axis_idx]
            if dof < 3:
                if self.pivot:
                    continue
                if x0 is None:                                   # /IMPVEL
                    vimp = scale * fct.eval((t - 0.5 * dt) * facx)
                else:                                            # /IMPDISP
                    if dt <= 0.0:
                        continue
                    # land the MASTER on x0 + d(t) along the skew axis
                    target = float(x0 @ e) + scale * fct.eval(t * facx)
                    vimp = (target - float(x[self.master] @ e)) / dt
                vref_new = vimp - float(
                    cross3(self.w, x[self.master] - self.x_ref) @ e)
                dv = vref_new - float(self.v_ref @ e)
                v_old_e = float(v_ref_old @ e) if v_ref_old is not None else float(self.v_ref @ e)
                wext += self.M * dv * 0.5 * (v_old_e + vref_new)
                self.v_ref += e * dv
            else:
                if x0 is None:                                   # /IMPVEL
                    wimp = scale * fct.eval((t - 0.5 * dt) * facx)
                else:                                            # /IMPDISP
                    if dt <= 0.0:
                        continue
                    wimp = scale * (fct.eval(t * facx) - fct.eval((t - dt) * facx)) / dt
                dw = wimp - float(self.w @ e)
                w_new = self.w + e * dw
                dL = Jsp @ (e * dw)
                w_mid = 0.5 * (self.w + w_new)
                wext += float(dL @ w_mid)
                self.w = w_new
                self.L = Jsp @ self.w
        return wext

    # ------------------------------------------------------------------
    def finalize_mass(self, mass_eff: np.ndarray) -> None:
        """Fold /INTER/TYPE2 effective-mass increments on body nodes into
        the body mass and inertia (see module docstring): the tied
        secondaries' inertia rides the body."""
        dm = mass_eff[self.nodes] - self.model.mass[self.nodes]
        dm[self.model.mass[self.nodes] >= 1e29] = 0.0
        if not np.any(dm > 0.0):
            return
        self.M += float(dm.sum())
        r = self.r0
        r2 = np.einsum("nb,nb->n", r, r)
        self.J0 += (np.eye(3) * float((dm * r2).sum())
                    - np.einsum("n,nb,nc->bc", dm, r, r))

    # ------------------------------------------------------------------
    def _rigid_field(self, x_nodes: np.ndarray) -> np.ndarray:
        """v_ref + w x (x - x_ref) for the given node positions."""
        return self.v_ref + cross3(self.w, x_nodes - self.x_ref)

    # ------------------------------------------------------------------
    def advance(self, fint: np.ndarray, fext: np.ndarray, fcont: np.ndarray,
                mint: np.ndarray, v: np.ndarray, vr: np.ndarray,
                x: np.ndarray, dt: float, t_next: float) -> float:
        """One cycle of the 6-DOF EOM (rbyfor + rbycor): gather, update
        (v_ref, L), scatter the rigid velocity field. Runs after the
        generic acceleration update (whose trial velocities on body nodes
        are phantoms — overwritten here before anything moves) and BEFORE
        the contact-work booking, so the booking sees these velocities.

        Returns the external work of the master /IMPVEL drive (0 without
        one)."""
        nodes = self.nodes
        w_old = self.w.copy()          # start-of-cycle spin w^{n-1/2}
        f = fint[nodes] + fext[nodes] + fcont[nodes]
        F = f.sum(axis=0)
        r = x[nodes] - self.x_ref
        T = cross3(r, f).sum(axis=0) + mint[nodes].sum(axis=0)
        self.f_res = F.copy()
        self.m_res = T.copy()

        wext = 0.0
        v_ref_old = self.v_ref.copy()
        if not self.pivot:
            a = F / self.M
            self.v_ref = self.v_ref + a * dt
            t_mid = t_next - 0.5 * dt
            for dof, fct, scale, facx, t0, t1 in self.drives:      # moving rigid die
                if t_next < t0 or t_next > t1:
                    continue
                vimp = scale * fct.eval(t_mid * facx)
                vref_new = vimp - cross3(self.w, x[self.master] - self.x_ref)[dof]
                dv = vref_new - self.v_ref[dof]
                wext += self.M * dv * 0.5 * (v_ref_old[dof] + vref_new)           # J . v_imp, as /IMPVEL
                self.v_ref[dof] = vref_new
            # master /IMPDISP (M37): the master dof lands on x0 + d(t)
            # exactly — velocity from the CURRENT master position, with
            # the spin transport w x (x_m - x_ref) removed so it is the
            # REFERENCE point's velocity that gets prescribed
            for dof, fct, scale, facx, t0, t1, x0m in self.disp_drives:
                if dt <= 0.0 or t_next < t0 or t_next > t1:
                    continue
                target = x0m + scale * fct.eval(t_next * facx)
                vimp = (target - x[self.master, dof]) / dt
                vref_new = vimp - cross3(
                    self.w, x[self.master] - self.x_ref)[dof]
                dv = vref_new - self.v_ref[dof]
                wext += self.M * dv * 0.5 * (v_ref_old[dof] + vref_new)       # J . v_imp booking
                self.v_ref[dof] = vref_new
            # the same drives named in a /SKEW (M39) — imposed along the
            # skew axis, before /BCS (which wins, as in the reference where
            # bcs10 runs after fixvel)
            wext += self._apply_skew_drives(x, t_next, dt, v_ref_old)
            self.v_ref[self.fix_tra] = 0.0
        else:
            wext += self._apply_skew_drives(x, t_next, dt, v_ref_old)

        # angular momentum update + spin from the co-rotated inertia
        self.L = self.L + T * dt
        Jsp = self.R @ self.J0 @ self.R.T
        try:
            w = np.linalg.solve(Jsp, self.L)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(Jsp) @ self.L
        if np.any(self.fix_rot):
            w[self.fix_rot] = 0.0
            self.L = Jsp @ w

        # ---- rotational /IMPVEL // /IMPDISP master drive (M39) -------------
        # overwrite the driven spin component with the imposed angular
        # velocity, exactly as the translational drive overwrites v_ref.
        # Work booking is the rotational leapfrog identity: the angular
        # impulse dL = Jsp (w_end - w_free) does dL . (w_old + w_end)/2 —
        # the midstep average of the start-of-cycle spin and the enforced
        # value (the same fixvel.F identity kinematics.apply_kinematic
        # books for the translational nodal drive; w_end differs from the
        # free spin only on the driven component). Applied whether or not
        # the master is pivoted (a pivoted master still spins about the
        # pivot) and BEFORE the skew /BCS below, which wins as in the
        # translational path (bcs10 runs after fixvel in the reference).
        def _spin_drive(rdof, wimp):
            nonlocal w
            w_end = w.copy()
            w_end[rdof] = wimp
            dL = Jsp @ (w_end - w)               # angular impulse on rdof
            w_mid = 0.5 * (w_old + w_end)
            w[rdof] = wimp
            self.L = Jsp @ w                     # keep L consistent w/ spin
            return float(dL @ w_mid)

        for rdof, fct, scale, facx, t0, t1 in self.rot_drives:
            if t_next < t0 or t_next > t1:
                continue
            wext += _spin_drive(rdof, scale * fct.eval(t_next * facx))
        for rdof, fct, scale, facx, t0, t1 in self.rot_disp_drives:
            if dt <= 0.0 or t_next < t0 or t_next > t1:
                continue
            wimp = scale * (fct.eval(t_next * facx)
                            - fct.eval((t_next - dt) * facx)) / dt
            wext += _spin_drive(rdof, wimp)
        for row, dof, fct, scale, facx, t0, t1, _ in self.skew_drives:
            if dof >= 3 and t_next >= t0 and t_next <= t1:
                axis_idx = dof - 3
                e = self.skews.axes[row][axis_idx]
                w += e * (float(self.w @ e) - float(w @ e))
                self.L = Jsp @ w
        # /BCS in a /SKEW on the master (M39): project the constrained skew
        # axes out of BOTH the reference velocity and the spin
        if self.bc_skew:
            self._apply_skew_bcs(self.v_ref, w)
            self.L = Jsp @ w
        self.w = w

        v[nodes] = self._rigid_field(x[nodes])
        vr[nodes] = w
        return wext

    # ------------------------------------------------------------------
    def enforce(self, x: np.ndarray, v: np.ndarray, dt: float) -> None:
        """After the position update: advance the body frame with the
        exponential map and PLACE the nodes — the rigid shape is
        reconstructed from the initial offsets, so drift cannot
        accumulate (the i2vit3-style enforcement of this constraint).

        The nodal velocities are then RE-scattered as the instantaneous
        rigid field at the NEW positions. This is what the element
        kernels see at the next force computation, and it matters: every
        ported kernel measures an exactly ZERO strain rate on a linear
        velocity field v_ref + w x (x - x_ref) evaluated at its own
        current configuration, so elements interior to the body build up
        no stress at all. Leaving the step-5 velocities (the field at the
        PRE-update positions) instead feeds the kernels an O(w^2 r dt)
        spurious strain rate that RATCHETS — a hypoelastic stress inside
        the body grows without bound at high spin rates (the original
        avoids the issue by deactivating interior elements; the port
        keeps them alive at zero strain, which also keeps their faces
        available to contact). The step-5 scatter remains the one the
        energy bookings use — the two fields differ by O(w^2 r dt),
        second-order in the work."""
        self.R = _orthonormalize(_exp_rotation(self.w, dt) @ self.R)
        if self.pivot:
            self.xg = self.x_ref + self.R @ self.c0
        else:
            self.x_ref = self.x_ref + self.v_ref * dt
            self.xg = self.x_ref
        x[self.nodes] = self.x_ref + self.r0 @ self.R.T
        v[self.nodes] = self._rigid_field(x[self.nodes])

    @property
    def x_cg(self) -> np.ndarray:
        return self.xg

    @property
    def v_cg(self) -> np.ndarray:
        return self.v_ref


def build_rigid_bodies(model: Model, loads, log,
                       saved_map=None) -> List[RigidBodyEngine]:
    """Instantiate the engine-side rigid bodies (/RBODY + /RBE2).
    ``saved_map`` (M6): {body id: state dict} from an engine restart.

    Rigid-body CHAINS (a master of one body a slave of another — allowed
    through the Starter since M14 for the IMPLICIT condensation) are
    REFUSED here loudly: this explicit integrator advances each body's
    6-DOF EOM independently and scatters the rigid field per body — a
    chained master would be written by its parent and read by its child
    with no nesting order, silently corrupting both. (The original
    resolves chains in the Starter — rbody_part_modif.F90 — so its engine
    never sees one.)"""
    saved_map = saved_map or {}
    bodies = [rb for rb in getattr(model, "rbodies", []) if getattr(rb, "slaves", None) is not None]
    if not bodies:
        return []
    numnod = getattr(model, "numnod", len(getattr(model, "node_ids", [])))
    is_master = np.zeros(numnod, dtype=bool)
    for rb in bodies:
        if 0 <= rb.master < numnod:
            is_master[rb.master] = True
    for rb in bodies:
        valid_slaves = rb.slaves[(rb.slaves >= 0) & (rb.slaves < numnod)]
        if is_master[valid_slaves].any():
            raise NotImplementedError(
                f"/{rb.kind}/{rb.id}: rigid-body CHAIN (a slave of this "
                f"body is the master of another) — supported by the "
                f"IMPLICIT solver only (PORTING_GUIDE M14); the explicit "
                f"engine refuses it rather than integrate the bodies in "
                f"an undefined order.")
    return [RigidBodyEngine(rb, model, loads, log, saved_map.get(rb.id))
            for rb in bodies]
