"""
/RBE3 — interpolation constraint, engine side (M5).

Fortran origin: ``engine/source/constraints/general/rbe3/`` —

    rbe3f.F    distribution of the dependent node's forces to the masters
    rbe3v.F    kinematic update of the dependent node from the masters
    starter/source/constraints/general/rbe3/hm_read_rbe3.F (input)

Theory — the weighted least-squares rigid fit and its dual
----------------------------------------------------------
An RBE3 makes ONE dependent (reference) node follow the *average* motion
of a set of independent (master) nodes — crucially WITHOUT rigidifying
the masters against each other (the classic use: hang a point mass or
introduce a load on a flexible patch without stiffening it; compare
/RBE2, which freezes the patch).

Kinematics. With weights w_i (uniform here), master positions x_i, the
weighted centroid G and the fit's moment tensor are

    x_G = sum w_i x_i / W,      r_i = x_i - x_G
    J_w = sum w_i (|r_i|^2 E - r_i r_i^T)

The best-fit rigid motion of the master cloud (weighted least squares on
the velocity field) is the translation rate v_G = sum w_i v_i / W and the
rotation rate

    w_fit = J_w^{-1} sum w_i r_i x v_i

(the normal equations of min sum w_i |v_i - v_G - w x r_i|^2 — J_w is
exactly the "inertia" of the cloud with masses w_i). The dependent node,
offset rho = x_ref - x_G, then moves with

    v_ref = v_G + w_fit x rho,        vr_ref = w_fit.

Force distribution. The dual (virtual-work-consistent) map of that
interpolation distributes a force F applied at the reference node as

    f_i = (w_i / W) F + w_i m x r_i,      m = J_w^{-1} (rho x F)

which is *exact*: sum f_i = F (total force), sum r_i x f_i = rho x F
(total moment about G — the identity sum w_i r_i x (m x r_i) = J_w m).
A moment M at the reference node distributes as the couple field
f_i = w_i (J_w^{-1} M) x r_i alone. Because distribution and
interpolation are dual pairs evaluated on the SAME current geometry, the
constraint transmits power exactly (sum f_i . v_i = F . v_ref + M . w_fit)
— it does no spurious work, and books nothing (asserted by the M5 tests).

Implementation = the ContactType2 lumped pattern, hook for hook:

1. ``augment_mass``  — the dependent node's mass is carried by the
   masters in the Engine's EFFECTIVE mass, M_i += (w_i/W) m_ref
   (translation weights only — the rotational coupling of the mass is
   second order in the offset, the same documented simplification as the
   TYPE2 offset ties);
2. ``transfer_forces`` — per cycle, the dependent rows of fint/fext/fcont
   (and its mint moment) are distributed to the masters and zeroed;
3. ``enforce``       — after the masters moved, the dependent node is
   *placed*: velocity from the fit, position integrated with it.

Degenerate master sets (a single node, or all masters collinear) make
J_w singular: the fit's rotation about the deficient direction is
undetermined, so the pseudo-inverse is used — it returns the minimum-norm
rotation (zero about the deficient axis), and the force distribution
loses nothing because the corresponding moment component cannot exist on
such a cloud anyway (warned at init).
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..model.model import Model


class Rbe3Constraint:
    """One /RBE3, engine-side."""

    def __init__(self, r3, model: Model, log):
        self.model = model
        who = f"/RBE3/{r3.id}"
        self.ref = model.node_index(r3.ref_id)
        g = model.node_groups[r3.grnod_id]
        masters = g.node_idx[g.node_idx != self.ref]
        if len(masters) < len(g.node_idx):
            log.warning(f"{who}: the reference node is in the master "
                        f"group — removed from it", "RBE3 INIT")
        if len(masters) == 0:
            raise ValueError(f"{who}: no master nodes")
        self.masters = masters
        self.w = np.ones(len(masters))          # uniform unit weights
        self.W = float(self.w.sum())

        # rank check (init-time only, geometry barely changes the rank)
        xg = (self.w[:, None] * model.x0[masters]).sum(axis=0) / self.W
        r = model.x0[masters] - xg
        r2 = np.einsum("nb,nb->n", r, r)
        Jw = (np.eye(3) * float((self.w * r2).sum())
              - np.einsum("n,nb,nc->bc", self.w, r, r))
        lam = np.linalg.eigvalsh(Jw)
        if lam[0] < 1e-8 * max(lam[2], 1e-30):
            log.warning(f"{who}: master nodes are collinear or a single "
                        f"point — the rotation fit about the deficient "
                        f"direction is dropped (pseudo-inverse)",
                        "RBE3 INIT")
        log.info(f"     {who}: NODE {r3.ref_id} INTERPOLATED FROM "
                 f"{len(masters)} MASTER NODE(S)")

        self.x_prev = model.x0[self.ref].copy()

    # ------------------------------------------------------------------
    def _geometry(self, x: np.ndarray):
        """Current centroid, arms, offset and (pseudo-)inverted moment
        tensor of the master cloud."""
        xg = (self.w[:, None] * x[self.masters]).sum(axis=0) / self.W
        r = x[self.masters] - xg
        r2 = np.einsum("nb,nb->n", r, r)
        Jw = (np.eye(3) * float((self.w * r2).sum())
              - np.einsum("n,nb,nc->bc", self.w, r, r))
        return xg, r, np.linalg.pinv(Jw, rcond=1e-8)

    # ------------------------------------------------------------------
    def augment_mass(self, mass_eff: np.ndarray) -> None:
        """M_i += (w_i/W) m_ref (once, engine init) — the masters carry
        the dependent node's inertia for the acceleration solve; the
        physical mass stays at the node (energies, momentum)."""
        m_ref = self.model.mass[self.ref]
        if m_ref < 1e29:
            np.add.at(mass_eff, self.masters, self.w / self.W * m_ref)

    # ------------------------------------------------------------------
    def transfer_forces(self, fint: np.ndarray, fext: np.ndarray,
                        fcont: np.ndarray, mint: np.ndarray,
                        x: np.ndarray) -> None:
        """Per-cycle step (rbe3f): distribute the dependent node's
        assembled forces and moment to the masters (see the module
        docstring for the exact map) and zero its rows."""
        xg, r, Jinv = self._geometry(x)
        rho = x[self.ref] - xg
        for arr in (fint, fext, fcont):
            F = arr[self.ref]
            m = Jinv @ np.cross(rho, F)
            arr[self.masters] += (self.w[:, None] / self.W) * F \
                + self.w[:, None] * np.cross(m, r)
            arr[self.ref] = 0.0
        M = mint[self.ref]
        if M @ M > 0.0:
            m = Jinv @ M
            # a couple has no net force: pure w_i (m x r_i) field
            fint[self.masters] += self.w[:, None] * np.cross(m, r)
            mint[self.ref] = 0.0

    # ------------------------------------------------------------------
    def enforce(self, x: np.ndarray, v: np.ndarray, vr: np.ndarray,
                dt: float) -> None:
        """Per-cycle step (rbe3v): fit the rigid motion of the masters'
        (post-update) velocities and place the dependent node."""
        xg, r, Jinv = self._geometry(x)
        vm = v[self.masters]
        v_g = (self.w[:, None] * vm).sum(axis=0) / self.W
        w_fit = Jinv @ np.cross(r, self.w[:, None] * vm).sum(axis=0)
        rho = self.x_prev - xg
        v_ref = v_g + np.cross(w_fit, rho)
        v[self.ref] = v_ref
        vr[self.ref] = w_fit
        x[self.ref] = self.x_prev + v_ref * dt
        self.x_prev = x[self.ref].copy()


def build_rbe3(model: Model, log) -> List[Rbe3Constraint]:
    return [Rbe3Constraint(r3, model, log) for r3 in model.rbe3]
