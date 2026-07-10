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
nodal /IMPVEL, J . v_imp with J = M (v_imp - v_free). BCS or /IMPVEL on
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
    """One /RBODY or /RBE2, engine-side."""

    def __init__(self, rb, model: Model, loads, log):
        self.rb = rb
        self.model = model
        who = f"/{rb.kind}/{rb.id}"
        self.master = rb.master
        self.slaves = rb.slaves
        self.nodes = np.concatenate([[self.master], self.slaves])

        self.M = float(rb.mass_total)
        self.J0 = rb.J.copy()          # about the COG, global axes at t=0
        self.xg = rb.xg.copy()         # current COG position
        self.R = np.eye(3)

        # ---- body-level boundary conditions from the MASTER's /BCS ------
        # (the node-level flags for the whole body are cleared: the body
        # supersedes both real BCS on its nodes and the automatic fixing
        # of frozen massless nodes — a massless master/slave is not dead,
        # it is carried by the body)
        self.fix_tra = np.zeros(3, dtype=bool)
        self.fix_rot = np.zeros(3, dtype=bool)
        for bc in model.bcs:
            grp = model.node_groups.get(bc.grnod_id)
            if grp is None or grp.node_idx is None:
                continue
            if self.master in grp.node_idx:
                self.fix_tra |= bc.fix_tra.astype(bool)
                self.fix_rot |= bc.fix_rot.astype(bool)
            if np.isin(self.slaves, grp.node_idx).any():
                log.warning(f"{who}: /BCS/{bc.id} touches slave node(s) — "
                            f"the rigid body wins (kinematic clash); put "
                            f"the BCS on the master node instead",
                            "RBODY INIT")
        loads.fix_tra[self.nodes] = False
        loads.fix_rot[self.nodes] = False

        # ---- pivot mode: master translations fully fixed -----------------
        # The body then rotates about the (fixed) master point; transport
        # the inertia tensor there (parallel-axis / Huygens-Steiner).
        self.pivot = bool(self.fix_tra.all())
        if self.pivot:
            self.x_ref = model.x0[self.master].copy()
            c = self.xg - self.x_ref
            self.J0 = self.J0 + self.M * (np.eye(3) * float(c @ c)
                                          - np.outer(c, c))
            self.c0 = c.copy()         # COG offset in the (t=0) body frame
        else:
            self.x_ref = self.xg.copy()

        # initial node offsets from the reference point (body frame = t0)
        self.r0 = model.x0[self.nodes] - self.x_ref

        # ---- /IMPVEL driving the master: body-velocity drive --------------
        # (removed from the nodal treatment so its huge-frozen-mass work
        # booking cannot fire; slaves under /IMPVEL are a clash: warn and
        # remove them there too — the body wins)
        self.drives = []               # (dof, funct, scale)
        for k, (idx, dof, fct, scale) in enumerate(loads.impvel):
            if self.master in idx:
                self.drives.append((dof, fct, scale))
            hit = np.isin(idx, self.nodes)
            if np.any(hit):
                if np.isin(self.slaves, idx).any():
                    log.warning(f"{who}: /IMPVEL drives slave node(s) — "
                                f"the rigid body wins (kinematic clash)",
                                "RBODY INIT")
                loads.impvel[k] = (idx[~hit], dof, fct, scale)
        for k, (idx, dof, fct, scale, x0d) in enumerate(loads.impdisp):
            hit = np.isin(idx, self.nodes)
            if np.any(hit):
                log.warning(f"{who}: /IMPDISP drives body node(s) — the "
                            f"rigid body wins (kinematic clash)",
                            "RBODY INIT")
                loads.impdisp[k] = (idx[~hit], dof, fct, scale,
                                    x0d[~hit] if x0d is not None else None)
        if self.pivot and self.drives:
            log.warning(f"{who}: /IMPVEL on a pivoted (fully clamped) "
                        f"master is ignored", "RBODY INIT")

        # ---- initial state: project the nodal velocities ------------------
        # (an /INIVEL field on the slaves may not be exactly rigid; the
        # body can only carry its rigid part: v_g and L are the momenta of
        # the initial field, and the scatter below makes the nodal
        # velocities consistent with them)
        m = model.mass[self.nodes].copy()
        m[m >= 1e29] = 0.0             # frozen = massless placeholder
        v0 = model.v[self.nodes]
        msum = float(m.sum())
        self.v_ref = ((m[:, None] * v0).sum(axis=0) / msum if msum > 0
                      else np.zeros(3))
        r = model.x0[self.nodes] - self.x_ref
        self.L = np.cross(r, m[:, None] * v0).sum(axis=0)
        self.L += (model.inertia[self.nodes, None]
                   * model.vr[self.nodes]).sum(axis=0)
        if self.pivot:
            self.v_ref = np.zeros(3)
        self.v_ref[self.fix_tra] = 0.0
        w = np.linalg.solve(self.J0, self.L)
        w[self.fix_rot] = 0.0
        self.L = self.J0 @ w
        self.w = w
        model.v[self.nodes] = self._rigid_field(model.x[self.nodes])
        model.vr[self.nodes] = w

        log.info(f"     {who}: {len(self.slaves)} SLAVE NODE(S), MASS = "
                 f"{self.M:12.5E}" + ("  [PIVOTED AT MASTER]"
                                      if self.pivot else ""))

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
        return self.v_ref + np.cross(self.w, x_nodes - self.x_ref)

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
        f = fint[nodes] + fext[nodes] + fcont[nodes]
        F = f.sum(axis=0)
        r = x[nodes] - self.x_ref
        T = np.cross(r, f).sum(axis=0) + mint[nodes].sum(axis=0)

        wext = 0.0
        if not self.pivot:
            a = F / self.M
            self.v_ref = self.v_ref + a * dt
            for dof, fct, scale in self.drives:      # moving rigid die
                vimp = scale * fct.eval(t_next)
                dv = vimp - self.v_ref[dof]
                wext += self.M * dv * vimp           # J . v_imp, as /IMPVEL
                self.v_ref[dof] = vimp
            self.v_ref[self.fix_tra] = 0.0

        # angular momentum update + spin from the co-rotated inertia
        self.L = self.L + T * dt
        Jsp = self.R @ self.J0 @ self.R.T
        w = np.linalg.solve(Jsp, self.L)
        if np.any(self.fix_rot):
            w[self.fix_rot] = 0.0
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


def build_rigid_bodies(model: Model, loads, log) -> List[RigidBodyEngine]:
    """Instantiate the engine-side rigid bodies (/RBODY + /RBE2)."""
    return [RigidBodyEngine(rb, model, loads, log)
            for rb in model.rbodies if rb.slaves is not None]
