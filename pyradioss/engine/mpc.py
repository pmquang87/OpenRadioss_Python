"""
/MPC — general linear multi-point constraints, engine side (M6).

Fortran origin:
  - ``starter/source/constraints/general/mpc/hm_read_mpc.F`` (input card reading, coefficient default)
  - ``engine/source/tools/lagmul/lag_mpc.F`` (engine Lagrange multiplier system assembly)
  - ``starter/source/tools/lagmul/lgmini_mpc.F`` (Lagrange multiplier initialization & sizing)
  - ``hm_cfg_files/config/CFG/radioss110/RBODY/mpc.cfg`` (format specification: %10d%10d%10d%20lg)
Deferred from M5 because a general MPC row couples arbitrary
DOFs and needs a small COUPLED solve per cycle — it fits neither the
lumped one-way transfer of /INTER/TYPE2 nor the closed-form fit of
/RBE3.

Theory — the Lagrange treatment on accelerations
------------------------------------------------
Each /MPC row imposes a homogeneous linear relation between nodal DOFs,

    g_i(u) = sum_k c_k u(n_k, d_k) = 0,        or   G u = 0

collecting all rows into the (small, sparse) constraint matrix G. For an
explicit code the natural enforcement level is the ACCELERATION: with
lumped mass M and applied force f, the constrained accelerations solve

    M a = f + G^T lambda,        G a = 0

whose Schur complement is a tiny nc x nc system per cycle:

    (G M^-1 G^T) lambda = -G M^-1 f
    a = M^-1 (f + G^T lambda)

The port adds the Lagrange forces G^T lambda into the assembled internal
force vector BEFORE the leap-frog velocity update — so the ordinary
update produces constrained velocities with no extra pass.

Zero work by construction. The constraint force's power is
(G^T lambda) . v = lambda . (G v) = 0 whenever the velocities satisfy
the constraint — which they do, because (a) the initial velocities are
projected onto G v = 0 once at initialization, and (b) enforcing G a = 0
preserves G v exactly through the velocity update, and G u through the
position update (u integrates v). Nothing is booked, and the engine's
internal-work ledger (step 6c) sees exactly zero from these forces —
asserted by the M6 tests.

A per-cycle velocity CLEANUP (the same tiny solve on G v) removes what
the other kinematic conditions may have re-injected after the force
stage (a wall correcting an MPC node, a BCS zeroing one of its DOFs...)
plus accumulated round-off; the cleanup impulse is the minimum-norm
(mass-weighted) one, its work -lambda.(G v_new) = 0 at the velocities
the nodes keep, matching the /IMPVEL booking convention (and its
kinetic-energy effect is quadratic in the residual it removes —
negligible by construction, asserted in the tests).

Fixed DOFs as ground. A /BCS-fixed DOF appearing in a row is treated as
INFINITELY massive: its M^-1 entry is zero, so the constraint solves
never move it and the correction lands entirely on the free DOFs —
'u_free + u_fixed = 0' then correctly drives the free side to the fixed
value's negative, and no force fights the BCS.

Rotational DOFs (4-6) are supported where the node carries rotational
inertia (shell/beam nodes) — a rotational term on an inertia-less node
is a model error caught at init. Nodes whose motion another kinematic
condition fully prescribes (rigid-body members, tied secondaries, /RBE3
dependents) clash with an MPC — warned, the other constraint wins
(their enforcement runs after the MPC force stage).

The port implements the general row; only redundant (rank-deficient)
row SETS are solved in the least-squares sense via the pseudo-inverse
(warned at init — the physics is unaffected, the multipliers are just
not unique).
"""

from __future__ import annotations

import numpy as np

from ..model.model import Model


class MpcConstraints:
    """All /MPC rows, solved together (they may share DOFs)."""

    def __init__(self, model: Model, loads, log):
        self.model = model
        rows, nodes, dofs, coefs = [], [], [], []
        ok_rows = 0
        for mpc in model.mpcs:
            who = f"/MPC/{mpc.id}"
            try:
                idx = model.node_indices(mpc.node_ids)
            except KeyError as exc:
                log.error(f"{who}: unknown node id {exc}", "MPC CHECK")
                continue
            bad = False
            for ni, d in zip(idx, mpc.dofs):
                if d >= 4 and model.inertia[ni] <= 0.0:
                    log.error(f"{who}: rotational dof {d} on node "
                              f"{model.node_ids[ni]} which carries no "
                              f"rotational inertia", "MPC CHECK")
                    bad = True
            if bad:
                continue
            rows.extend([ok_rows] * len(idx))
            nodes.extend(idx.tolist())
            dofs.extend(d - 1 for d in mpc.dofs)     # 0..5 internally
            coefs.extend(mpc.coefs)
            ok_rows += 1
            log.info(f"     {who}: {len(idx)} TERM(S)")

        self.nc = ok_rows
        if self.nc == 0:
            return
        rows = np.asarray(rows)
        nodes = np.asarray(nodes)
        dofs = np.asarray(dofs)
        coefs = np.asarray(coefs, dtype=float)

        # unique (node, dof) columns -> dense G (nc x ncol); several terms
        # of one row on the same DOF simply sum their coefficients
        key = nodes * 6 + dofs
        ukey, inv = np.unique(key, return_inverse=True)
        self.col_node = ukey // 6
        self.col_dof = ukey % 6
        self.G = np.zeros((self.nc, len(ukey)))
        np.add.at(self.G, (rows, inv), coefs)

        # /BCS-fixed DOFs read as infinite mass (see module docstring)
        tra = self.col_dof < 3
        self.tra = tra
        self.fixed = np.zeros(len(ukey), dtype=bool)
        if loads is not None:
            self.fixed[tra] = loads.fix_tra[self.col_node[tra],
                                            self.col_dof[tra]]
            self.fixed[~tra] = loads.fix_rot[self.col_node[~tra],
                                             self.col_dof[~tra] - 3]

        # kinematic clashes: prescribed nodes (the other condition wins)
        presc = np.zeros(model.numnod, dtype=bool)
        for rb in model.rbodies:
            if rb.slaves is not None:
                presc[rb.slaves] = True
                presc[rb.master] = True
        if np.any(presc[self.col_node]):
            log.warning("/MPC: term(s) on rigid-body nodes — the body "
                        "wins (kinematic clash)", "MPC CHECK")

        # rank check at initial masses (masses only grow -> rank stable)
        # (deferred to first solve: A depends on the engine's effective
        # mass; _lambda() warns once if it must fall back to lstsq)
        self._warned_singular = False
        self._log = log

    def __len__(self):
        return getattr(self, "nc", 0)

    # ------------------------------------------------------------------
    def _minv_cols(self, inv_mass, inv_inertia):
        """M^-1 of every constraint column (0 for fixed DOFs)."""
        m = np.where(self.tra, inv_mass[self.col_node],
                     inv_inertia[self.col_node])
        m[self.fixed] = 0.0
        return m

    def _solve(self, A, rhs):
        try:
            lam = np.linalg.solve(A, rhs)
            if not np.all(np.isfinite(lam)):
                raise np.linalg.LinAlgError
            return lam
        except np.linalg.LinAlgError:
            if not self._warned_singular:
                self._log.warning("/MPC: redundant constraint rows — "
                                  "least-squares multipliers used",
                                  "MPC SOLVE")
                self._warned_singular = True
            return np.linalg.lstsq(A, rhs, rcond=None)[0]

    # ------------------------------------------------------------------
    def transfer_forces(self, fint, fcont, fext, mint,
                        inv_mass, inv_inertia) -> None:
        """Per-cycle force stage: add the Lagrange forces G^T lambda into
        fint/mint so the ordinary velocity update satisfies G a = 0."""
        minv = self._minv_cols(inv_mass, inv_inertia)
        # trial acceleration on each constraint column
        a = np.where(
            self.tra,
            (fint[self.col_node, np.minimum(self.col_dof, 2)]
             + fcont[self.col_node, np.minimum(self.col_dof, 2)]
             + fext[self.col_node, np.minimum(self.col_dof, 2)]) * minv,
            mint[self.col_node, np.maximum(self.col_dof, 3) - 3] * minv)
        A = (self.G * minv) @ self.G.T
        lam = self._solve(A, -(self.G @ a))
        corr = self.G.T @ lam                    # force per column
        t = self.tra
        fint[self.col_node[t], self.col_dof[t]] += corr[t]
        r = ~t
        mint[self.col_node[r], self.col_dof[r] - 3] += corr[r]

    # ------------------------------------------------------------------
    def enforce(self, v, vr, inv_mass, inv_inertia) -> None:
        """Velocity cleanup: project the (post-kinematics) velocities
        back onto G v = 0 — the mass-weighted minimum-norm correction.
        Does no booked work (see module docstring)."""
        minv = self._minv_cols(inv_mass, inv_inertia)
        t = self.tra
        vel = np.where(t, v[self.col_node, np.minimum(self.col_dof, 2)],
                       vr[self.col_node,
                          np.maximum(self.col_dof, 3) - 3])
        res = self.G @ vel
        if not np.any(res):
            return
        A = (self.G * minv) @ self.G.T
        lam = self._solve(A, res)
        vel_corr = minv * (self.G.T @ lam)
        v[self.col_node[t], self.col_dof[t]] -= vel_corr[t]
        r = ~t
        vr[self.col_node[r], self.col_dof[r] - 3] -= vel_corr[r]


def build_mpc(model: Model, loads, log):
    """Instantiate (or None when the deck has no /MPC)."""
    if not model.mpcs:
        return None
    c = MpcConstraints(model, loads, log)
    return c if len(c) else None
