"""
Kinematic constraints in the implicit system by CONDENSATION (M12).

Fortran origin
--------------
``engine/source/constraints/general/rbody/rby_imp0.F`` (/RBODY),
``engine/source/constraints/general/rbe2/rbe2_imp0.F`` (/RBE2),
``engine/source/constraints/general/rbe3/rbe3_imp0.F`` (/RBE3) and
``engine/source/interfaces/interf/i2_imp1.F`` (/INTER/TYPE2 tied) — the
constraint-condensation family the implicit driver (``imp_solv.F`` /
``imp_dyna.F``'s IMP_DYKV) calls around every assembly:

* ``RBY_IMP1`` / ``RBE2_IMP1``: transform every slave DOF block of the
  assembled K through the rigid map and ACCUMULATE it on the master —
  ``UPDKB_RB`` computes exactly ``K' = CDI^T K CDI`` with (their comment)
  ``CDI = [[I][R]; [0][I]]``, R the moment-arm operator built from the
  CURRENT arm ``x_s - x_M``, and the slave equations are marked condensed
  (``IKC = 7``: they never enter the solve);
* ``RBY_IMPR1`` / ``RBE2_IMPB0``: the matching residual condensation
  ``B_M += CDI^T B_s`` (``UPDB_RB``: force summed, moment = arm x force);
* ``RBY_IMPR2``: the recovery — after the solve the slave motion is
  rebuilt from the master's (the transpose direction);
* ``I2UPDK0``: the tied interface does the same with the projection
  weights H of the Starter search (the secondary block scattered to the
  main corners, ``IKC = 5``);
* ``RBE3_IMP1`` / ``RBE3_IMPR1``: the interpolation constraint eliminates
  the DEPENDENT (reference) node through the weighted fit and its
  virtual-work dual.

The port expresses all of these as ONE global sparse transformation

    u_full = T u_red        (u_full: every numbered equation,
                             u_red : the independent ones)

so the reduced system of a Newton iteration is the classic master-slave
elimination

    K_red = T^T K T,     R_red = T^T R,     K_red du_red = R_red,
    du_full = T du_red

— algebraically identical to the original's in-place block condensations
(each *_IMP1 call IS one block row/column of T^T K T), and exactly the
same "condensed, NOT penalized" treatment the /BCS conditions already get
in the DofMap: a dependent DOF contributes its element stiffness and its
loads (through T^T) but is no unknown.

The constraint rows (all linearized at the geometry the increment is
linearized at, exactly like the original's use of the CURRENT X for the
arms — under /IMPL/NONLIN they are REBUILT on every committed frame):

* **/RBODY, /RBE2** (rigid): u_s = u_M + theta_M x r_s,  theta_s = theta_M,
  with r_s = x_s - x_M. T^T then carries F_M += F_s, T_M += r_s x F_s + M_s
  — the rbyfor gather — and T^T M T puts the EXACT rigid-body 6x6 mass at
  the master: total mass on translations, the parallel-axis inertia tensor
  Sum m(|r|^2 I - r r^T) + Sum I_s on rotations, and the m*skew(r)
  COG-offset coupling blocks (what /IMPL/DYNA needs — nothing extra to
  code, the congruence transform IS the mass condensation).
* **/INTER/TYPE2** (tied): u_s = Sum_k w_k u_k with the STARTER projection
  weights (the port reuses the explicit ``ContactType2`` search, so the
  ties are bit-identical to the explicit run's). Like the explicit port,
  the rotational tie and the offset moment redistribution (the UPDKB_RB
  arm branch of I2UPDK0 for 6-DOF mains) are NOT ported: exact for
  on-surface ties, second-order in the usual spot-weld offsets — the same
  documented simplification, so implicit and explicit answers agree.
* **/RBE3**: the reference node's six DOFs are eliminated through the
  weighted least-squares rigid fit of the master cloud (engine/rbe3.py's
  math, reused):

      theta_ref = J_w^-1 Sum w_i r_i x u_i,     u_ref = u_G + theta_ref x rho

  which is linear in the master displacements; T^T reproduces exactly the
  rbe3f dual force distribution (total force and moment transmitted, no
  stiffening — the fit and its dual are transposes of each other, so the
  virtual-work exactness of the explicit module carries over verbatim).
* **/MPC**: the general homogeneous rows G u = 0 are eliminated by
  Gaussian elimination with column pivoting (QR): the best-conditioned
  nc columns become dependent, u_d = -G_d^-1 G_f u_f. /BCS-fixed DOFs in
  a row read as ground (their column is condensed to zero motion),
  matching the explicit module's infinite-mass convention. Redundant
  (rank-deficient) row sets drop the dependent rows beyond the rank with
  a warning — the constraint set is enforced, the multipliers were never
  unique.

Equation numbering hooks (used by DofMap): a constraint master that no
element touches — the classic standalone /RBODY master node, frozen by
the Starter's mass check (mass = 1e30) — must get equations even though
the default rules would fix or skip it, and every dependent DOF must be
numbered so its element stiffness and loads are captured before being
condensed. ``ImplicitConstraints`` therefore publishes ``extra`` (slots
to force-number), ``unfreeze`` (nodes whose frozen-placeholder fixity is
overridden — a real /BCS still applies: /BCS on a rigid MASTER is the
body-level condition, translations-all-fixed = the PIVOT, exactly the
explicit convention) and ``bcs_ignore`` (dependent nodes whose own /BCS
clashes with the constraint — warned, the constraint wins, mirroring the
explicit "the body wins" treatment).

Geometry modes
--------------
* LINEAR (M8): T is built once at x0 — the constraint linearized where
  everything else is.
* NONLINEAR (/IMPL/NONLIN, M9): T is REBUILT at every committed frame
  (arms/fits from the updated geometry — the original reads the current X
  every time). Within an increment the map is the linearized one (the
  original's too); at commit the rigid bodies are RE-PLACED exactly —
  x_s = x_M + exp(skew(theta_M)) r_s, the Rodrigues map of the increment
  rotation — so the body cannot stretch by the O(theta^2) of the
  linearized map, increment after increment (the same "placement, not
  integration" philosophy as the explicit rigid_body.enforce, and the
  implicit-pendulum validation depends on it); tied nodes are re-placed
  on their segment with the co-rotated offset (the explicit i2vit3
  placement). The O(theta^2) mismatch this creates against the committed
  element state is picked up by the next increment's Newton — the same
  order as the Hughes–Winget incremental-objectivity error already
  accepted per increment.

Deferred loudly (PORTING_GUIDE M12): constraint CHAINS (a dependent DOF of
one constraint appearing in another — e.g. an /MPC row on a rigid-body
slave, an /RBE3 master inside an /RBODY: refused with a clear error, the
original resolves some of these orderings), /IMPDISP or /IMPVEL on
constraint nodes (drive a free master with forces, or the structure), the
TYPE2 rotational tie / offset-moment branch, RBE3 per-set weights.
"""

from __future__ import annotations

import numpy as np

from . import require_scipy
from .dofmap import DOFS_PER_NODE


def _eqof(dof, nodes, comp):
    """Equation index of (node, component 0..5); -1 = condensed/absent."""
    return dof.eq[np.asarray(nodes) * DOFS_PER_NODE + comp]


class ImplicitConstraints:
    """All kinematic constraints of one implicit run, as a condensation
    transform (see module docstring). Two-phase:

    1. ``ImplicitConstraints(model, log)`` scans the model, resolves every
       constraint (reusing the explicit modules' searches/fits so the
       kinematics are identical to the explicit run), publishes the
       DofMap hooks (``extra``, ``unfreeze``, ``bcs_ignore``) and checks
       the clashes/chains it can see at node level.
    2. ``build(dof, x_ref)`` assembles the sparse T at a given geometry
       (called once under linear geometry, per committed frame under
       /IMPL/NONLIN).
    """

    def __init__(self, model, log):
        self.model = model
        n = model.numnod
        self.extra = np.zeros((n, DOFS_PER_NODE), dtype=bool)
        self.unfreeze = np.zeros(n, dtype=bool)
        self.bcs_ignore = np.zeros(n, dtype=bool)
        #: nodes carrying a DEPENDENT DOF (for clash checks + /IMPDISP veto)
        self.dep_nodes = np.zeros(n, dtype=bool)
        #: nodes serving as constraint MASTERS (for the chain check message)
        self.master_nodes = np.zeros(n, dtype=bool)

        # ---- /RBODY + /RBE2 (rby_imp0.F / rbe2_imp0.F) --------------------
        self.rigid = []
        for rb in model.rbodies:
            if rb.slaves is None:
                continue
            m, slaves = rb.master, rb.slaves
            who = f"/{rb.kind}/{rb.id}"
            if self.dep_nodes[m] or np.any(self.dep_nodes[slaves]):
                raise NotImplementedError(
                    f"{who}: node(s) already dependent in another "
                    f"constraint — chained kinematic constraints are "
                    f"DEFERRED under the implicit solver (PORTING_GUIDE "
                    f"M12)")
            self.rigid.append(rb)
            self.dep_nodes[slaves] = True
            self.master_nodes[m] = True
            # the master needs all 6 equations (the condensed body block is
            # 6x6 — RBY_IMP1's ND = 6), even when frozen/massless or free
            # of rotational stiffness; a real /BCS on it is the body-level
            # condition and still condenses (the pivot).
            self.extra[m, :] = True
            self.unfreeze[m] = True
            # dependent slave slots must be numbered so their element
            # stiffness/loads are captured, then condensed — including
            # massless carried slaves; a slave /BCS clashes (body wins,
            # the explicit convention) and is ignored with a warning.
            self.extra[slaves, :3] = True
            self.unfreeze[slaves] = True
            if _nodes_have_bcs(model, slaves):
                log.warning(f"{who}: /BCS on slave node(s) is ignored under "
                            f"the implicit solver — the rigid body wins "
                            f"(kinematic clash; put the BCS on the master)",
                            "IMPL CONSTR")
                self.bcs_ignore[slaves] = True
            log.info(f"     {who}: CONDENSED — {len(slaves)} slave node(s) "
                     f"-> master 6-DOF block (rby_imp0.F)")

        # ---- /INTER/TYPE2 tied (i2_imp1.F I2UPDK0) -------------------------
        # Reuse the EXPLICIT projection search verbatim (same ties, same
        # weights — the cross-validation against the explicit run depends
        # on it). The instance is discarded; only the resolved arrays stay.
        self.tied = []
        for itf in model.interfaces:
            if itf.type != 2:
                continue
            from ..contact.inter_type2 import ContactType2
            t2 = ContactType2(itf, model, log)
            act = t2.active
            snode, seg, w = t2.snode[act], t2.seg[act], t2.w[act]
            off = t2.off_loc[act]
            if len(snode) == 0:
                continue
            if np.any(self.dep_nodes[snode]) or \
                    np.any(self.master_nodes[snode]):
                raise NotImplementedError(
                    f"/INTER/TYPE2/{itf.id}: tied secondary node(s) already "
                    f"belong to another constraint — chained kinematic "
                    f"constraints are DEFERRED (PORTING_GUIDE M12)")
            self.tied.append((snode, seg, w, off))
            self.dep_nodes[snode] = True
            self.master_nodes[np.unique(seg)] = True
            self.extra[snode, :3] = True
            self.unfreeze[snode] = True
            if _nodes_have_bcs(model, snode):
                log.warning(f"/INTER/TYPE2/{itf.id}: /BCS on tied secondary "
                            f"node(s) is ignored — the tie wins (kinematic "
                            f"clash)", "IMPL CONSTR")
                self.bcs_ignore[snode] = True
            log.info(f"     /INTER/TYPE2/{itf.id}: CONDENSED — "
                     f"{len(snode)} tie(s) -> main corners (i2_imp1.F)")

        # ---- /RBE3 (rbe3_imp0.F) -------------------------------------------
        # Reuse the explicit module's fit geometry (identical dual).
        self.rbe3 = []
        for r3 in model.rbe3:
            from ..engine.rbe3 import Rbe3Constraint
            c = Rbe3Constraint(r3, model, log)
            if self.dep_nodes[c.ref] or self.master_nodes[c.ref] or \
                    np.any(self.dep_nodes[c.masters]):
                raise NotImplementedError(
                    f"/RBE3/{r3.id}: node(s) already belong to another "
                    f"constraint — chained kinematic constraints are "
                    f"DEFERRED (PORTING_GUIDE M12)")
            self.rbe3.append(c)
            self.dep_nodes[c.ref] = True
            self.master_nodes[c.masters] = True
            self.extra[c.ref, :] = True     # ref carries force AND moment
            self.unfreeze[c.ref] = True
            if _nodes_have_bcs(model, np.array([c.ref])):
                log.warning(f"/RBE3/{r3.id}: /BCS on the reference node is "
                            f"ignored — the interpolation constraint wins",
                            "IMPL CONSTR")
                self.bcs_ignore[c.ref] = True
            log.info(f"     /RBE3/{r3.id}: CONDENSED — reference node -> "
                     f"{len(c.masters)} master(s) (rbe3_imp0.F)")

        # ---- /MPC ------------------------------------------------------------
        # columns resolved at build() (they depend on the equation
        # numbering); here only the node bookkeeping + rotation check.
        self.mpc_rows = []
        for mpc in model.mpcs:
            idx = model.node_indices(mpc.node_ids)
            dofs = np.asarray(mpc.dofs, dtype=int) - 1        # 0..5
            coefs = np.asarray(mpc.coefs, dtype=float)
            self.mpc_rows.append((mpc.id, idx, dofs, coefs))
            log.info(f"     /MPC/{mpc.id}: {len(idx)} TERM(S) — eliminated "
                     f"by column-pivoted condensation")
        self._log = log

        self.n_constraints = (len(self.rigid) + len(self.tied)
                              + len(self.rbe3) + len(self.mpc_rows))

    def __bool__(self):
        return self.n_constraints > 0

    # ------------------------------------------------------------------
    def veto_imposed(self, imposed):
        """/IMPDISP on a constraint node is a kinematic clash the implicit
        solver refuses (DEFERRED — drive a free master DOF with forces
        instead; the original orders some of these cases)."""
        touched = self.dep_nodes | self.master_nodes
        for idx, d, fct, scale in imposed:
            if np.any(touched[idx]):
                raise NotImplementedError(
                    "/IMPDISP on nodes of a rigid body / tie / RBE3 / MPC "
                    "is DEFERRED under the implicit solver (PORTING_GUIDE "
                    "M12): the prescribed DOF and the constraint would "
                    "fight over the same unknown. Load the structure or a "
                    "free master DOF instead.")

    # ------------------------------------------------------------------
    def build(self, dof, x_ref):
        """Assemble the sparse transformation T (ndof x nred) at geometry
        ``x_ref``: every independent equation maps 1:1, every dependent
        equation carries its constraint row (see module docstring). Sets
        ``self.T`` (CSR) and ``self.nred``."""
        sp, _ = require_scipy()
        ndof = dof.ndof
        dep = np.zeros(ndof, dtype=bool)
        rows = []          # (dependent eq, [(master eq, coef), ...])

        def add_row(e, terms):
            if e < 0:
                return               # slot condensed by a (master) BCS
            dep[e] = True
            rows.append((e, [(me, c) for (me, c) in terms
                             if me >= 0 and c != 0.0]))

        # ---- rigid bodies: u_s = u_M + theta_M x r_s,  theta_s = theta_M --
        for rb in self.rigid:
            m, slaves = rb.master, rb.slaves
            me = [_eqof(dof, np.array([m]), c)[0] for c in range(6)]
            r = x_ref[slaves] - x_ref[m]
            for k, s in enumerate(slaves):
                rx, ry, rz = r[k]
                # (theta x r)_x = ty*rz - tz*ry, etc. (UPDKB_RB's R block)
                add_row(_eqof(dof, np.array([s]), 0)[0],
                        [(me[0], 1.0), (me[4], rz), (me[5], -ry)])
                add_row(_eqof(dof, np.array([s]), 1)[0],
                        [(me[1], 1.0), (me[3], -rz), (me[5], rx)])
                add_row(_eqof(dof, np.array([s]), 2)[0],
                        [(me[2], 1.0), (me[3], ry), (me[4], -rx)])
                for c in range(3):                     # theta_s = theta_M
                    add_row(_eqof(dof, np.array([s]), 3 + c)[0],
                            [(me[3 + c], 1.0)])

        # ---- tied: u_s = sum_k w_k u_k (constant Starter weights) ----------
        for snode, seg, w, off in self.tied:
            for k in range(len(snode)):
                corners = seg[k]
                for c in range(3):
                    ce = _eqof(dof, corners, c)
                    add_row(_eqof(dof, np.array([snode[k]]), c)[0],
                            list(zip(ce.tolist(), w[k].tolist())))

        # ---- RBE3: the weighted least-squares fit, linearized ---------------
        # theta_ref = Jinv sum w_i skew(r_i) u_i
        # u_ref     = sum (w_i/W) u_i + theta_ref x rho
        #           = sum [ (w_i/W) I - skew(rho) Jinv w_i skew(r_i) ] u_i
        for c3 in self.rbe3:
            xg, r, Jinv = c3._geometry(x_ref)
            rho = x_ref[c3.ref] - xg
            Srho = _skew(rho)
            terms_t = [[] for _ in range(3)]     # u_ref components
            terms_r = [[] for _ in range(3)]     # theta_ref components
            for i, mnode in enumerate(c3.masters):
                Ai_rot = c3.w[i] * (Jinv @ _skew(r[i]))
                Ai_tra = (c3.w[i] / c3.W) * np.eye(3) - Srho @ Ai_rot
                ce = _eqof(dof, np.full(3, mnode), np.arange(3))
                for c in range(3):
                    terms_t[c].extend(zip(ce.tolist(), Ai_tra[c].tolist()))
                    terms_r[c].extend(zip(ce.tolist(), Ai_rot[c].tolist()))
            for c in range(3):
                add_row(_eqof(dof, np.array([c3.ref]), c)[0], terms_t[c])
                add_row(_eqof(dof, np.array([c3.ref]), 3 + c)[0], terms_r[c])

        # ---- MPC: eliminate the best-pivot column of each row ---------------
        if self.mpc_rows:
            self._build_mpc_rows(dof, dep, rows, add_row)

        # ---- chain check in equation space -----------------------------------
        for e, terms in rows:
            for me, c in terms:
                if dep[me]:
                    raise NotImplementedError(
                        "chained kinematic constraints (a master DOF of one "
                        "constraint is dependent in another) are DEFERRED "
                        "under the implicit solver (PORTING_GUIDE M12)")

        # ---- assemble T --------------------------------------------------------
        red = np.cumsum(~dep) - 1                 # independent eq -> column
        self.nred = int((~dep).sum())
        ind = np.where(~dep)[0]
        rr, cc, vv = list(ind), list(red[ind]), [1.0] * len(ind)
        for e, terms in rows:
            for me, c in terms:
                rr.append(e)
                cc.append(red[me])
                vv.append(c)
        self.T = sp.coo_matrix(
            (np.asarray(vv, dtype=float),
             (np.asarray(rr, dtype=np.int64), np.asarray(cc, dtype=np.int64))),
            shape=(ndof, self.nred)).tocsr()
        self.Tt = self.T.T.tocsr()
        self.dep = dep
        #: independent equation indices in reduced-coordinate order (red is
        #: a cumsum over ~dep, so np.where preserves the ordering) — the
        #: SELECTION that reads reduced coordinates off a full vector
        self.ind = ind
        return self

    # ------------------------------------------------------------------
    def make_consistent(self, dof, u, ur):
        """Project a nodal displacement field onto the constraint manifold:
        keep the INDEPENDENT (master) components and recompute every
        dependent one through T.

        This matters for the DYNAMICS predictor (an M12 lesson, found by a
        pendulum that spun over the top): the constant-acceleration
        predictor u = dt v + dt^2/2 a extrapolates every node ALONG ITS
        TANGENT, which violates the constraint by O(theta^2) x arm — and
        Newton can never remove that violation, because its corrections
        live in range(T) while the reduced residual T^T R is blind to any
        component the transpose annihilates. Left alone, the dependent
        nodes walk off the manifold by a first-order-in-dt step error that
        the commit placement then silently converts into energy (the
        pendulum gained 20x its drop energy and circulated). Projecting
        the predictor restores the invariant the whole reduction rests on:
        u = T u_red, always."""
        u_eq = dof.gather_residual(u, ur)
        return dof.scatter_solution(self.T @ u_eq[self.ind])

    # ------------------------------------------------------------------
    def _build_mpc_rows(self, dof, dep, rows, add_row):
        """Eliminate the /MPC rows: G over the numbered columns (fixed
        DOFs = ground, dropped), QR with column pivoting picks the
        dependent columns, u_d = -G_d^-1 G_f u_f. Redundant rows beyond
        the rank are dropped with a warning (see module docstring)."""
        from scipy.linalg import qr
        # unique (eq) columns over all rows
        col_eqs, G_rows = [], []
        for mid, idx, dofs, coefs in self.mpc_rows:
            terms = {}
            for ni, d, c in zip(idx, dofs, coefs):
                if d >= 3 and not (dof.has_rot[ni]
                                   or self.extra[ni, d]):
                    raise ValueError(
                        f"/MPC/{mid}: rotational dof on node "
                        f"{self.model.node_ids[ni]} which carries no "
                        f"rotational stiffness — no such equation exists "
                        f"in the implicit system")
                e = _eqof(dof, np.array([ni]), d)[0]
                if e < 0:
                    continue                      # fixed = ground
                terms[e] = terms.get(e, 0.0) + c
            if terms:
                G_rows.append(terms)
            else:
                self._log.warning(f"/MPC/{mid}: every term is on a fixed "
                                  f"DOF — row dropped", "IMPL CONSTR")
        if not G_rows:
            return
        col_eqs = sorted({e for t in G_rows for e in t})
        colmap = {e: j for j, e in enumerate(col_eqs)}
        G = np.zeros((len(G_rows), len(col_eqs)))
        for i, t in enumerate(G_rows):
            for e, c in t.items():
                G[i, colmap[e]] = c
        Q, R, piv = qr(G, pivoting=True)
        diag = np.abs(np.diag(R))
        tol = max(G.shape) * np.finfo(float).eps * (diag[0] if len(diag)
                                                    else 0.0)
        rank = int((diag > max(tol, 1e-12 * (diag[0] if len(diag) else 1.0)
                               )).sum())
        if rank < len(G_rows):
            self._log.warning(f"/MPC: {len(G_rows) - rank} redundant "
                              f"constraint row(s) dropped (rank "
                              f"deficiency)", "IMPL CONSTR")
        if rank == 0:
            return
        # u_d = C u_f with C = -R_dd^-1 R_df  (columns in pivot order)
        C = -np.linalg.solve(R[:rank, :rank], R[:rank, rank:]) \
            if rank < len(col_eqs) else np.zeros((rank, 0))
        for i in range(rank):
            e_dep = col_eqs[piv[i]]
            if dep[e_dep]:
                raise NotImplementedError(
                    "/MPC row eliminates a DOF already dependent in "
                    "another constraint — chained kinematic constraints "
                    "are DEFERRED (PORTING_GUIDE M12)")
            terms = [(col_eqs[piv[rank + j]], C[i, j])
                     for j in range(C.shape[1])]
            add_row(e_dep, terms)

    # ------------------------------------------------------------------
    def reduce_matrix(self, K):
        """K_red = T^T K T — the *_IMP1 block condensations in one product."""
        return (self.Tt @ K @ self.T).tocsr()

    def reduce_vector(self, r):
        """R_red = T^T R — the *_IMPR1 residual condensation."""
        return self.Tt @ r

    def expand(self, du_red):
        """du_full = T du_red — the RBY_IMPR2-style recovery of the
        dependent motion from the masters'."""
        return self.T @ du_red

    # ------------------------------------------------------------------
    def project_velocity(self, dof, M_eq, v, vr, solver):
        """Project nodal velocities onto the constraint manifold, mass
        weighted: solve (T^T M T) v_red = T^T M v and re-expand. For a
        rigid body this IS the explicit port's momentum projection
        (v_ref = total momentum / mass, w = J^-1 L); MPC/tie/RBE3
        residuals are similarly removed with the minimum kinetic-energy
        change. Zero-mass reduced rows (massless rotations) keep zero
        velocity. Returns (v, vr) new arrays."""
        sp, _ = require_scipy()
        v_eq = dof.gather_residual(v, vr)
        Mred = (self.Tt @ sp.diags(M_eq) @ self.T).tocsr()
        rhs = self.Tt @ (M_eq * v_eq)
        v_red = _solve_semidefinite(Mred, rhs, solver)
        v2, vr2 = dof.scatter_solution(self.T @ v_red)
        # keep whatever lives on slots outside the equation system
        # (nothing does — scatter covers every numbered slot)
        return v2, vr2

    # ------------------------------------------------------------------
    def commit_placement(self, model, u, ur):
        """After a converged /IMPL/NONLIN increment: re-place the
        dependent nodes EXACTLY (see module docstring — the linearized
        map stretches a rigid body by O(theta^2) per increment; the
        explicit modules place, never integrate). ``u``/``ur`` are the
        increment just committed (model.x already holds x_old + u)."""
        from ..engine.rigid_body import _exp_rotation
        for rb in self.rigid:
            m, slaves = rb.master, rb.slaves
            th = ur[m]
            R = _exp_rotation(th, 1.0)          # exp(skew(theta)), dt = 1
            x_m_old = model.x[m] - u[m]
            arms_old = (model.x[slaves] - u[slaves]) \
                - x_m_old                        # committed-frame arms
            model.x[slaves] = model.x[m] + arms_old @ R.T
        from ..contact.inter_type2 import _segment_frames
        for snode, seg, w, off in self.tied:
            xs = model.x[seg]
            t1, t2, nn = _segment_frames(xs)
            model.x[snode] = (np.einsum("nk,nkb->nb", w, xs)
                              + off[:, 0:1] * t1 + off[:, 1:2] * t2
                              + off[:, 2:3] * nn)


def _skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def _nodes_have_bcs(model, nodes):
    for bc in model.bcs:
        grp = model.node_groups.get(bc.grnod_id)
        if grp is None or grp.node_idx is None:
            continue
        if np.isin(nodes, grp.node_idx).any():
            return True
    return False


def _solve_semidefinite(M, rhs, solver):
    """Solve a symmetric positive SEMI-definite reduced-mass system:
    zero-diagonal rows (equations no mass reaches — massless rotations)
    are replaced by identity rows with zero right-hand side (their
    velocity/acceleration is quasi-static, exactly the convention of the
    unconstrained diagonal path)."""
    sp, _ = require_scipy()
    d = M.diagonal()
    zero = d <= 0.0
    if np.any(zero):
        fix = sp.diags(zero.astype(float))
        M = (M + fix).tocsr()
        rhs = np.where(zero, 0.0, rhs)
    return solver.solve(M, rhs)


def build_constraints(model, log):
    """Scan the model; return an ImplicitConstraints or None when the deck
    carries no /RBODY, /RBE2, /INTER/TYPE2, /RBE3 or /MPC — the drivers
    keep the exact pre-M12 code path in that case."""
    if not (any(rb.slaves is not None for rb in model.rbodies)
            or any(i.type == 2 for i in model.interfaces)
            or model.rbe3 or model.mpcs):
        return None
    return ImplicitConstraints(model, log)
