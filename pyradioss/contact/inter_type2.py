"""
/INTER/TYPE2 — tied contact (kinematic secondary-to-main gluing).

Fortran origin: ``engine/source/interfaces/int02/`` —

    i2for3.F   transfer of the secondary nodal forces to the main segment
    i2vit3.F   kinematic update of the secondary velocities from the mains
    i2curv.F   segment local frame (for offset secondary nodes)
    starter/source/interfaces/inter3d1/i2buc1.F, i2dst3.F
               (Starter) projection search: each secondary node onto its
               closest main segment, isoparametric weights + offset

Theory — the constrained (kinematic) tied formulation
-----------------------------------------------------
TYPE2 is not a force model: it is a *constraint*. Each secondary node s is
glued, for the whole run, to the material point of its main segment it was
projected on at time 0:

    x_s(t) = sum_k w_k x_k(t) + offset(t),         sum_k w_k = 1

with w_k the (constant) interpolation weights of the projection and
``offset`` the initial normal/tangential misfit expressed in the segment's
co-rotating local frame (so a spot-weld between two shell mid-surfaces,
which are half a thickness apart, rotates rigidly with the segment).

The explicit implementation is the classic lumped constraint method (the
same one the original and LS-DYNA's constrained tied contacts use):

1. **Force transfer** (i2for3): the secondary node's assembled force is
   distributed to the segment corners with the weights,
   f_k += w_k f_s, and removed from the secondary node.
2. **Mass transfer** (Starter i2 initialization): likewise, once, for the
   lumped masses: M_k += w_k m_s. Steps 1+2 make the main nodes carry the
   secondary's inertia and loading; total force and total mass are
   conserved by construction (sum w_k = 1).
3. **Kinematic update** (i2vit3): after the main nodes moved, the
   secondary node is *placed* — not integrated:
   x_s = sum w_k x_k + offset, and its velocity is set to the consistent
   v_s = (x_s^{n+1} - x_s^n) / dt.

Momentum is conserved *exactly*: d/dt(m_s v_s + sum M_k v_k) =
sum_k (M_k + w_k m_s) a_k = sum_k (f_k + w_k f_s) = total applied force.
And the tie does **no work by construction** — it books nothing into the
contact energy, and the global balance must close without a CE term
(asserted by the M4 tests).

Element deletion (M3<->M4): a tie whose main segment's parent element is
deleted is *released* (the crack must not keep carrying load through the
glue); its transferred mass is handed back so the released node resumes
free flight with its own inertia. A secondary node whose own elements all
died is released the same way.

Port simplifications (documented): rotational DOFs of shell secondary
nodes are left free (Radioss Spotflag=1-style rotation tying not ported);
the moment of the transferred force about an OFFSET projection point is
not redistributed (exact for zero offset; for the usual spot-weld offsets
— a fraction of the element size — the angular-momentum error is second
order and the validation tests confirm the balance stays tight).
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20
from ..model.model import Model
from . import tracking
from .inter_type7 import _closest_point_on_triangle
from .stiffness import _segment_areas


def _segment_frames(xs: np.ndarray):
    """Co-rotating orthonormal frame (t1, t2, n) of 3/4-node segments
    (i2curv flavour). ``xs`` is (n, 4, 3); triangles repeat node 3.

    Built from the covariant mid-edge vectors r = (x2+x3)-(x1+x4) and
    s = (x3+x4)-(x1+x2): n = r x s (the average normal), t1 = r direction.
    This frame rotates rigidly with the segment, which is exactly what an
    offset tied node must follow.
    """
    r = xs[:, 1] + xs[:, 2] - xs[:, 0] - xs[:, 3]
    s = xs[:, 2] + xs[:, 3] - xs[:, 0] - xs[:, 1]
    n = np.cross(r, s)
    n /= np.maximum(np.linalg.norm(n, axis=1), EM20)[:, None]
    t1 = r / np.maximum(np.linalg.norm(r, axis=1), EM20)[:, None]
    t2 = np.cross(n, t1)
    return t1, t2, n


class ContactType2:
    """One /INTER/TYPE2 tied interface, engine-side."""

    def __init__(self, itf, model: Model, log):
        self.itf = itf
        self.model = model
        surf = model.surfaces[itf.surf_id]
        segs = surf.segments if surf.segments is not None else \
            np.zeros((0, 4), dtype=np.int64)
        cand = np.asarray(model.node_groups[itf.grnod_id].node_idx)

        # a node that is itself a corner of the main surface cannot be
        # tied to it (it would be glued to itself); frozen (massless,
        # m = 1e30) nodes are structurally meaningless to tie
        surf_nodes = np.unique(segs)
        bad = np.isin(cand, surf_nodes) | (model.mass[cand] >= 1e29)
        if np.any(bad):
            log.warning(f"/INTER/TYPE2/{itf.id}: {int(bad.sum())} secondary "
                        f"node(s) excluded (main-surface corners or "
                        f"massless)", "TIED INIT")
        cand = cand[~bad]

        # ---- projection search (Starter i2buc1/i2dst3) --------------------
        # closest point of every candidate on every segment, chunked to
        # bound memory; keep the best segment within the search distance
        area = _segment_areas(model.x0, segs)
        lc = float(np.sqrt(area.mean())) if len(area) else 1.0
        dsearch = itf.dsearch if itf.dsearch > 0 else lc

        nbest = len(cand)
        best_d = np.full(nbest, np.inf)
        best_seg = np.full(nbest, -1, dtype=np.int64)
        best_w = np.zeros((nbest, 4))
        x0 = model.x0
        chunk = max(1, 2 ** 22 // max(nbest, 1))     # ~4M pairs per chunk
        for s0 in range(0, len(segs), chunk):
            sc = segs[s0:s0 + chunk]
            # all (node, segment-of-chunk) pairs
            pi = np.repeat(np.arange(nbest), len(sc))
            sj = np.tile(np.arange(len(sc)), nbest)
            p = x0[cand[pi]]
            dmin = np.full(len(pi), np.inf)
            wq = np.zeros((len(pi), 4))
            for cols in ((0, 1, 2), (0, 2, 3)):
                a, b, c = (x0[sc[sj, cols[0]]], x0[sc[sj, cols[1]]],
                           x0[sc[sj, cols[2]]])
                pt, u, vv, w = _closest_point_on_triangle(p, a, b, c)
                d = np.linalg.norm(p - pt, axis=1)
                better = d < dmin
                dmin = np.where(better, d, dmin)
                wtmp = np.zeros((len(pi), 4))
                wtmp[:, cols[0]], wtmp[:, cols[1]], wtmp[:, cols[2]] = \
                    u, vv, w
                wq[better] = wtmp[better]
            # reduce over the chunk: best segment per node
            dmat = dmin.reshape(nbest, len(sc))
            jbest = dmat.argmin(axis=1)
            dbest = dmat[np.arange(nbest), jbest]
            upd = dbest < best_d
            best_d = np.where(upd, dbest, best_d)
            best_seg[upd] = s0 + jbest[upd]
            best_w[upd] = wq.reshape(nbest, len(sc), 4)[
                np.arange(nbest), jbest][upd]

        found = (best_d <= dsearch) & (best_seg >= 0)
        if np.any(~found):
            log.warning(f"/INTER/TYPE2/{itf.id}: {int((~found).sum())} "
                        f"secondary node(s) farther than the search "
                        f"distance {dsearch:.4g} — left free", "TIED INIT")

        self.snode = cand[found]                       # tied secondary nodes
        self.seg = segs[best_seg[found]]               # (nt, 4) node indices
        self.w = best_w[found]                         # (nt, 4) weights
        seg_rows = best_seg[found]
        self.seg_gtype = (surf.seg_gtype[seg_rows]
                          if surf.seg_gtype is not None
                          else np.zeros(len(seg_rows), dtype="<U8"))
        self.seg_elem = (surf.seg_elem[seg_rows]
                         if surf.seg_elem is not None
                         else np.full(len(seg_rows), -1, dtype=np.int64))

        # initial offset, expressed in the segment's local frame so it
        # co-rotates (see module docstring)
        xs = x0[self.seg]
        t1, t2, n = _segment_frames(xs)
        xc = np.einsum("nk,nkb->nb", self.w, xs)
        dvec = x0[self.snode] - xc
        self.off_loc = np.column_stack([
            np.einsum("nb,nb->n", dvec, t1),
            np.einsum("nb,nb->n", dvec, t2),
            np.einsum("nb,nb->n", dvec, n)])
        off_mag = np.linalg.norm(dvec, axis=1)
        if len(off_mag) and off_mag.max() > 0.5 * lc:
            log.warning(f"/INTER/TYPE2/{itf.id}: largest tie offset "
                        f"{off_mag.max():.4g} exceeds half the segment size "
                        f"— check the projection", "TIED INIT")
        log.info(f"     /INTER/TYPE2/{itf.id}: {len(self.snode)} NODE(S) "
                 f"TIED TO SURFACE {itf.surf_id}")

        # a tied node's motion is fully prescribed by the constraint: an
        # /IMPVEL on it would fight the tie (two kinematic conditions on
        # one DOF — the original Starter errors out on such clashes)
        for imp in model.impvel:
            g = model.node_groups.get(imp.grnod_id)
            if g is not None and g.node_idx is not None and \
                    np.isin(self.snode, g.node_idx).any():
                log.warning(f"/INTER/TYPE2/{itf.id}: /IMPVEL/{imp.id} "
                            f"drives tied secondary node(s) — the tie "
                            f"wins (kinematic condition clash)",
                            "TIED INIT")

        # tracked position of the tied nodes (see enforce) — from the
        # CURRENT coordinates, not x0: on a restart-chained run (M6) the
        # ties resume where the saved model left them (at a fresh start
        # model.x == x0, so nothing changes)
        self.x_prev = model.x[self.snode].copy()
        self.active = np.ones(len(self.snode), dtype=bool)

        # deletion bookkeeping (release, not force filtering)
        self.deletable = tracking.any_deletable(model, self.seg_gtype)
        if self.deletable:
            self.ref_total = tracking.node_reference_counts(
                model, alive_only=False)
            # ties already released by /FAIL deletion in a PREVIOUS run
            # (M6 restart): the deletion state lives in the model, so the
            # release set is reconstructed here instead of persisted
            seg_dead = ~tracking.alive_segment_mask(
                model, self.seg_gtype, self.seg_elem)
            node_dead = ~tracking.tracked_node_mask(
                model, self.ref_total)[self.snode]
            self.active &= ~(seg_dead | node_dead)

    # ------------------------------------------------------------------
    def augment_mass(self, mass_eff: np.ndarray) -> None:
        """Mass transfer M_k += w_k m_s (once, engine init). ``mass_eff``
        is the Engine's EFFECTIVE mass used for accelerations only — the
        physical ``model.mass`` (energies, momentum) is untouched.
        Released ties (reconstructed at init after a chained restart)
        transfer nothing — their nodes fly with their own inertia."""
        act = self.active
        m_s = self.model.mass[self.snode[act]]
        for k in range(4):
            np.add.at(mass_eff, self.seg[act, k], self.w[act, k] * m_s)

    # ------------------------------------------------------------------
    def _release(self, dead: np.ndarray, mass_eff: np.ndarray,
                 inv_mass_eff: np.ndarray) -> None:
        """Release ties (deletion): hand the transferred mass back and
        deactivate. ``dead`` is a mask over the tie arrays."""
        m_s = self.model.mass[self.snode[dead]]
        seg = self.seg[dead]
        w = self.w[dead]
        for k in range(4):
            np.add.at(mass_eff, seg[:, k], -w[:, k] * m_s)
        touched = np.unique(seg)
        inv_mass_eff[touched] = 1.0 / mass_eff[touched]
        self.active[dead] = False

    # ------------------------------------------------------------------
    def transfer_forces(self, fint: np.ndarray, fext: np.ndarray,
                        fcont: np.ndarray, mass_eff: np.ndarray,
                        inv_mass_eff: np.ndarray, cycle: int) -> None:
        """Per-cycle step 1 (i2for3): move the tied nodes' assembled
        forces (internal, external AND contact — a tied node can also be
        a penalty secondary) to their main segments. Also polls the
        deletion release (cheap, only for models that can actually delete
        elements)."""
        if self.deletable and cycle % 8 == 0:      # poll every few cycles
            seg_dead = ~tracking.alive_segment_mask(
                self.model, self.seg_gtype, self.seg_elem)
            node_dead = ~tracking.tracked_node_mask(
                self.model, self.ref_total)[self.snode]
            dead = self.active & (seg_dead | node_dead)
            if np.any(dead):
                self._release(dead, mass_eff, inv_mass_eff)

        act = self.active
        if not np.any(act):
            return
        sn = self.snode[act]
        seg = self.seg[act]
        w = self.w[act]
        for arr in (fint, fext, fcont):
            F = arr[sn]
            for k in range(4):
                np.add.at(arr, seg[:, k], w[:, k, None] * F)
            arr[sn] = 0.0

    # ------------------------------------------------------------------
    def enforce(self, x: np.ndarray, v: np.ndarray, dt: float) -> None:
        """Per-cycle step 3 (i2vit3): place the tied nodes on their
        segments (weights + co-rotated offset) and set the consistent
        velocity. Called after the main nodes' position update."""
        act = self.active
        if not np.any(act):
            return
        sn = self.snode[act]
        xs = x[self.seg[act]]
        t1, t2, n = _segment_frames(xs)
        loc = self.off_loc[act]
        x_new = (np.einsum("nk,nkb->nb", self.w[act], xs)
                 + loc[:, 0:1] * t1 + loc[:, 1:2] * t2 + loc[:, 2:3] * n)
        if dt > EM20:
            v[sn] = (x_new - self.x_prev[act]) / dt
        x[sn] = x_new
        self.x_prev[act] = x_new
