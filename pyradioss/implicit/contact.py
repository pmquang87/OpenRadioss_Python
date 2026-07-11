"""
Penalty contact in the implicit Newton loop: /INTER/TYPE7 (M12).

Fortran origin
--------------
``engine/source/interfaces/int07/i7ke3.F`` (I7KE3, the per-interface
implicit driver called from ``engine/source/implicit/imp_int_k.F``'s
IMP_INT_K) and ``i7keg3.F`` (I7KEG3, the per-pair stiffness blocks),
with ``imp_solv.F``'s ICONTA bookkeeping deciding when the contact set
forces extra Newton work. Checked against the source, not docs:

* IMP_INT_K forces ``IMP_INT7 = 3`` around the assembly, and in that mode
  I7KEG3 keeps the pair stiffness CONSTANT (``STIF = HALF*STIF``, no
  GAP/(GAP-PENE) stiffening factor) — i.e. the assembled tangent is the
  plain linear penalty spring, the very same law the explicit port's
  i7for3 already integrates (the near-crossing stiffening guard is a
  documented M4 simplification on the explicit side too, so residual and
  tangent here are mutually consistent AND consistent with the explicit
  solver the cross-validations compare against);
* the blocks it assembles are the spring along the CONTACT NORMAL:
  KI11 = K*H1*(n n^T) etc. — the projection weights H distributing the
  node-side spring onto the segment corners, with KI12 = -KI11 the
  node-corner coupling. The port assembles the sharper EXACT gap
  linearization K * g g^T with g = [n, -H1 n, ..., -H4 n] (the
  original's corner blocks use H_j where the exact form has H_i*H_j — a
  diagonal-boosted modified-Newton variant) PLUS the closest-point
  curvature term the original omits entirely (see "Theory" below — the
  press example's grid-aligned punch corners forced the issue);
* friction: I7KEG3 adds a crude tangential-plane term scaled by FRIC —
  DEFERRED here (frictionless first, warned loudly), see PORTING_GUIDE
  M12.

Theory — the frictionless penalty contact residual and its tangent
-------------------------------------------------------------------
At the TRIAL configuration x of every Newton iteration, each candidate
(secondary node, main segment) pair is re-projected exactly like the
explicit narrow phase (i7dst3: closest point on the two triangles of the
quad, barycentric weights H). With penetration

    p = gap - d,      d = |x_node - x_proj|,      n = (x_node - x_proj)/d

an ACTIVE pair (p > 0) contributes the penalty force

    f_node    = +K p n            (push-out on the secondary node)
    f_corner_k = -H_k K p n       (the equal-and-opposite reaction,
                                   distributed with the projection weights
                                   — momentum conservation, as i7for3)

to the residual, and the exact linearization of that force at FROZEN
projection weights to the tangent:

    dp = -dd = -n . (du_node - Sum_k H_k du_k)  =  -g . du,
    K_pair = K (g g^T)  -  K p (P P^T) (x) Hess(d),
    g = [n, -H_1 n, ..., -H_4 n],      P = [1, -H_1, ..., -H_4]

The first, rank-one positive-semidefinite block is the penalty spring
along the normal — the whole tangent i7keg3.F assembles. The second is
the CURVATURE of the closest-point distance (K_c = grad^2 of the stored
energy 1/2 K p^2 = K grad-d grad-d^T - K p grad^2-d), and it depends on
WHERE the projection lands (the distance function's exact Hessian):

* FACE interior: d is the distance to a plane — grad^2 d = 0. Nothing to
  add: the g g^T block alone is the exact frozen-weight linearization
  (flat segments; the residual-exactness argument of the weights is the
  standard node-to-segment simplification, O(p/L)).
* VERTEX region: d = |x_node - x_corner| — grad^2 d = (I - n n^T)/d, the
  point-tie lateral term. It is NOT small when the node sits close to a
  corner (K p/d — an M12 lesson: a punch whose corner nodes land exactly
  on the pad's mesh grid left Newton in a LIMIT CYCLE at ~1e-5 residual,
  every load increment cut to the floor, because the missing lateral
  term was 23% of K). Negative semidefinite — the physical
  "snap-sideways" softening of a compressed point spring — and EXACT
  here including the corner-motion coupling (the [[H,-H],[-H,H]] block
  the P-pattern reproduces with the vertex weight 1).
* EDGE region (a real quad boundary edge): d is the distance to a line —
  grad^2 d = (I - n n^T - t t^T)/d with t the edge direction; the
  variation of the foot along the edge is the remaining (small) omitted
  piece. The two-triangle split's DIAGONAL is face interior of a planar
  quad, not an edge — classified as face.

The active set is decided on p > 0 exactly as the force; the region is
classified from the projection weights of the trial configuration.

ACTIVE-SET Newton: the active set is simply re-evaluated at every residual
and tangent call (pairs enter and leave between iterations as p crosses
zero); no separate bookkeeping is needed because the penalty force is
CONTINUOUS at p = 0 (force and energy both vanish). An increment whose
active set refuses to settle fails its Newton budget and is handed to the
M11 StepControl, which cuts the increment and retries — the imp_dt.F
coupling the original reaches through its ICONTA/IMCONV machinery.

The pair stiffness K reuses ``contact/stiffness.py`` unchanged (the
i7sti3 element formulas + the Istf combination), and the gap logic (Igap
0/1, Gap_min/Gap_max) mirrors the explicit ContactType7 line for line.
The candidate search here is a plain vectorized box overlap test per
call (implicit models are orders of magnitude below the explicit crash
meshes; the voxel broad phase of inter_type7.py exists for THOSE — no
shared state is touched, per the M7 parity contract).

Rate devices stay off, consistent with the whole implicit branch: the
explicit normal VISCOUS damper (C*vn) is a rate device and does not
exist here, and Coulomb friction (velocity-driven) is deferred loudly.

Contact is evaluated at the CURRENT trial configuration (model.x + u) in
BOTH geometry modes — contact is inherently geometric; under the M8
small-strain element path the elements still linearize at x0 while the
gaps close on the accumulated displacement (documented port choice; the
original's implicit contact also walks the current X).

A DOCUMENTED limitation (not a bug to fix silently): when the converged
state parks a secondary node EXACTLY on a non-smooth point of the
closest-point map — laterally on a main-mesh grid line (a vertex
projection whose region flips between neighbouring segments) or on the
crease of a warped quad's two-triangle split (the projection is then
genuinely discontinuous: the node sits on the dented surface's medial
axis) — Newton can end in a small limit cycle (~1e-5 relative residual
here) that no tangent can remove, and the StepControl cuts the increment
to its floor. This takes deliberately grid-aligned meshes (the press
example documents the geometry that provoked it); the remedy is to break
the alignment (any realistic mesh does) or loosen /IMPL/NEWTON's
tolerance above the cycle amplitude. Node-to-segment contact in the
original has the same non-smooth set — its looser default convergence
tolerance simply steps over it.

DEFERRED loudly (PORTING_GUIDE M12): friction in residual/tangent,
/INTER/TYPE11 edge-to-edge (a different narrow phase — nothing here
reuses), Inacti initial-penetration treatments, Igap 2/3, sensor gating
(TSTART/TSTOP) under the implicit clock, the I7KEG3 stiffening modes
IMP_INT7 = 0/1.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20
from ..contact.inter_type7 import _narrow
from ..contact.stiffness import (combine_stiffness, node_stiffness_gap,
                                 segment_stiffness_gap, _segment_areas)
from .dofmap import DOFS_PER_NODE


class ImplicitContact7:
    """One /INTER/TYPE7 interface in the implicit system (see module
    docstring). Owns copies of everything it needs — it never mutates the
    model, the interface record or the explicit contact modules."""

    def __init__(self, itf, model, log):
        self.itf = itf
        surf = model.surfaces[itf.surf_id]
        self.segs = (surf.segments if surf.segments is not None
                     else np.zeros((0, 4), dtype=np.int64))
        seg_gtype = (surf.seg_gtype if surf.seg_gtype is not None
                     else np.zeros(len(self.segs), dtype="<U8"))
        seg_elem = (surf.seg_elem if surf.seg_elem is not None
                    else np.full(len(self.segs), -1, dtype=np.int64))

        # secondary nodes: group, or the surface's own nodes (self-impact)
        if itf.grnod_id == 0:
            self.nodes = np.unique(self.segs)
        else:
            self.nodes = np.sort(model.node_groups[itf.grnod_id].node_idx)

        # penalty stiffness + gaps: the i7sti3 element formulas, reused
        # from contact/stiffness.py exactly like the explicit interface
        scale = itf.stfac if itf.istf != 1 else 1.0
        Km, gm = segment_stiffness_gap(model, self.segs, seg_gtype,
                                       seg_elem, scale)
        Ks_all, gs_all = node_stiffness_gap(model, scale)
        self.Km = Km
        self.Ks = Ks_all[self.nodes] if len(self.nodes) else np.zeros(0)

        area = _segment_areas(model.x0, self.segs)
        lc = float(np.sqrt(area.mean())) if len(area) else 1.0
        gap_floor = itf.gap if itf.gap > 0 else (
            float(gm.mean()) if len(gm) and gm.max() > 0 else 0.02 * lc)
        if itf.igap == 1:
            self.gap_m = gm
            self.gap_s = (gs_all[self.nodes] if len(self.nodes)
                          else np.zeros(0))
            self.gap_min = gap_floor
            self.gap_max = itf.gap_max if itf.gap_max > 0 else np.inf
            gap_hi = ((self.gap_s.max() if len(self.gap_s) else 0.0)
                      + (gm.max() if len(gm) else 0.0))
            self.gap_bound = float(np.clip(gap_hi, self.gap_min,
                                           self.gap_max))
        else:
            self.gap_const = gap_floor
            self.gap_bound = gap_floor

        if itf.fric > 0.0:
            log.warning(
                f"/INTER/TYPE7/{itf.id}: Coulomb friction (fric = "
                f"{itf.fric:g}) is DEFERRED under the implicit solver — "
                f"the interface runs FRICTIONLESS (see PORTING_GUIDE M12)",
                "IMPL CONTACT")
        if getattr(itf, "sens_id", 0):
            log.warning(
                f"/INTER/TYPE7/{itf.id}: /SENSOR gating is not evaluated "
                f"under the implicit solver — the interface is active for "
                f"the whole run", "IMPL CONTACT")
        log.info(f"     /INTER/TYPE7/{itf.id}: IMPLICIT PENALTY CONTACT — "
                 f"{len(self.nodes)} secondary node(s) vs "
                 f"{len(self.segs)} segment(s) (i7ke3.F, frictionless)")

    # ------------------------------------------------------------------
    def _active_pairs(self, x):
        """Re-project every box-overlap candidate at the trial geometry x
        and keep the penetrating pairs. Returns
        (ni, seg, w, nvec, pen, K, d) — the full active-set state of this
        configuration (``d`` the closest-point distance, needed by the
        curvature term of the tangent)."""
        empty = (np.zeros(0, dtype=np.int64),
                 np.zeros((0, 4), dtype=np.int64), np.zeros((0, 4)),
                 np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0))
        if len(self.segs) == 0 or len(self.nodes) == 0:
            return empty
        # vectorized box-overlap candidates (see module docstring: the
        # implicit models are small — no voxel machinery needed)
        margin = 1.001 * self.gap_bound
        xs = x[self.segs]                                # (nseg, 4, 3)
        lo = xs.min(axis=1) - margin                     # (nseg, 3)
        hi = xs.max(axis=1) + margin
        xn = x[self.nodes]                               # (nn, 3)
        inside = np.all((xn[:, None, :] >= lo[None, :, :])
                        & (xn[:, None, :] <= hi[None, :, :]), axis=2)
        ii, jj = np.nonzero(inside)
        if len(ii) == 0:
            return empty
        ni = self.nodes[ii]
        seg = self.segs[jj]
        # self-exclusion: a node never contacts a segment it is a corner of
        keep = np.ones(len(ni), dtype=bool)
        for k in range(4):
            keep &= seg[:, k] != ni
        ni, seg, jj, ii = ni[keep], seg[keep], jj[keep], ii[keep]
        if len(ni) == 0:
            return empty

        # narrow phase: the explicit i7dst3 port, called read-only
        best_d, best_pt, best_w = _narrow(x, ni, seg)

        if self.itf.igap == 1:
            gap = np.clip(self.gap_s[ii] + self.gap_m[jj],
                          self.gap_min, self.gap_max)
        else:
            gap = np.full(len(ni), self.gap_const)
        pen = gap - best_d
        act = pen > 0.0
        if not np.any(act):
            return empty
        ni, seg, ii, jj = ni[act], seg[act], ii[act], jj[act]
        pen = pen[act]
        d = np.maximum(best_d[act], EM20)
        nvec = (x[ni] - best_pt[act]) / d[:, None]
        w = best_w[act]
        K = combine_stiffness(self.itf.istf, self.itf.stfac,
                              self.Km[jj], self.Ks[ii])
        return ni, seg, w, nvec, pen, K, d

    # ------------------------------------------------------------------
    def forces(self, x, fcont):
        """Penalty force of the active set at trial geometry ``x``,
        scattered into ``fcont`` (node space) — the residual contribution
        (i7for3's spring term, no rate damper). Returns the number of
        active pairs (the listing's active-set trace)."""
        ni, seg, w, nvec, pen, K, _ = self._active_pairs(x)
        if len(ni) == 0:
            return 0
        Fvec = (K * pen)[:, None] * nvec
        np.add.at(fcont, ni, Fvec)
        np.add.at(fcont, seg.reshape(-1),
                  (-w[:, :, None] * Fvec[:, None, :]).reshape(-1, 3))
        return len(ni)

    # ------------------------------------------------------------------
    def energy(self, x):
        """Stored penalty-spring energy 1/2 K p^2 of the active set — a
        state function, booked in the implicit-dynamics ledger so a
        contact transient balances (the explicit CE analogue)."""
        _, _, _, _, pen, K, _ = self._active_pairs(x)
        return 0.5 * float((K * pen * pen).sum())

    # ------------------------------------------------------------------
    def triplets(self, x, dof):
        """COO triplets of the active-set tangent in EQUATION space at
        trial geometry ``x``: per pair the rank-one gap block K * g g^T
        MINUS the closest-point curvature K p (P P^T) (x) Hess(d) — the
        exact tangent per projection region (see module docstring) —
        over [node, corner1..4], mapped through the DofMap exactly like
        assembly.element_triplets."""
        ni, seg, w, nvec, pen, K, d = self._active_pairs(x)
        z = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
             np.zeros(0))
        if len(ni) == 0:
            return z
        npair = len(ni)
        # g (npair, 15): the gap gradient over the 5 nodes' translations
        g = np.zeros((npair, 15))
        g[:, 0:3] = nvec
        for k in range(4):
            g[:, 3 + 3 * k: 6 + 3 * k] = -w[:, k, None] * nvec
        ke = K[:, None, None] * g[:, :, None] * g[:, None, :]
        ke -= self._curvature_blocks(x, ni, seg, w, nvec, pen, K, d)
        # global scalar DOF slots of the 5 nodes' translations
        edofs = np.zeros((npair, 15), dtype=np.int64)
        for c in range(3):
            edofs[:, c] = ni * DOFS_PER_NODE + c
            for k in range(4):
                edofs[:, 3 + 3 * k + c] = seg[:, k] * DOFS_PER_NODE + c
        eq = dof.eq[edofs]
        row_eq = np.repeat(eq[:, :, None], 15, axis=2)
        col_eq = np.repeat(eq[:, None, :], 15, axis=1)
        keep = (row_eq >= 0) & (col_eq >= 0)
        return row_eq[keep].ravel(), col_eq[keep].ravel(), ke[keep].ravel()

    # ------------------------------------------------------------------
    @staticmethod
    def _curvature_blocks(x, ni, seg, w, nvec, pen, K, d):
        """The closest-point curvature K p (P P^T) (x) Hess(d) of every
        active pair (npair, 15, 15) — see the module docstring for the
        region-wise Hessian. Zero for face-interior projections (the
        common case), so the loop below only visits vertex/edge pairs.
        ``d`` is floored at 5% of the pair gap so a node sitting exactly
        ON a corner cannot blow the 1/d factor up (the capped term still
        breaks the limit cycle; the residual stays exact)."""
        npair = len(ni)
        out = np.zeros((npair, 15, 15))
        tol = 1e-8
        # triangle segments repeat corner 3: merge its weight so the
        # classification sees the real corner count
        tri = seg[:, 2] == seg[:, 3]
        wc = w.copy()
        wc[tri, 2] += wc[tri, 3]
        wc[tri, 3] = 0.0
        active = wc > tol
        cnt = active.sum(axis=1)
        boundary_edges = {(0, 1), (1, 2), (2, 3), (0, 3)}
        for p in np.where(cnt <= 2)[0]:
            n = nvec[p]
            H = np.eye(3) - np.outer(n, n)
            if cnt[p] == 2:
                ca, cb = np.where(active[p])[0]
                # the quad's split diagonal (0,2)/(1,3) is face interior;
                # a triangle segment's every corner pair is a real edge
                if not tri[p] and (ca, cb) not in boundary_edges:
                    continue
                t = x[seg[p, cb]] - x[seg[p, ca]]
                tn = np.linalg.norm(t)
                if tn > 0.0:
                    t = t / tn
                    H = H - np.outer(t, t)
            dcl = max(d[p], 0.05 * (d[p] + pen[p]))
            P = np.empty(5)
            P[0] = 1.0
            P[1:] = -w[p]
            out[p] = (K[p] * pen[p] / dcl) * np.einsum(
                "i,j,ab->iajb", P, P, H).reshape(15, 15)
        return out


# ----------------------------------------------------------------------------
# builders + assembly helper
# ----------------------------------------------------------------------------

def build_implicit_contacts(model, log):
    """Instantiate the implicit contact treatments: TYPE7 here, TYPE2 is a
    CONSTRAINT (implicit/constraints.py — the i2_imp1.F condensation),
    TYPE11 is refused loudly (deferred: its edge-edge narrow phase shares
    nothing with this module — PORTING_GUIDE M12)."""
    out = []
    for itf in model.interfaces:
        if itf.type == 7:
            out.append(ImplicitContact7(itf, model, log))
        elif itf.type == 11:
            raise NotImplementedError(
                f"/INTER/TYPE11/{itf.id} is DEFERRED under the implicit "
                f"solver (PORTING_GUIDE M12) — remove it or run the "
                f"explicit solver.")
        # TYPE2 handled by the constraint layer
    return out


def contact_forces(contacts, x, n):
    """Assembled contact force of every interface at trial geometry ``x``
    (node space, (n,3)) + the total active-pair count."""
    fcont = np.zeros((n, 3))
    nact = 0
    for c in contacts:
        nact += c.forces(x, fcont)
    return fcont, nact


def contact_tangent(contacts, x, dof):
    """Assembled active-set tangent (CSR, ndof x ndof) at trial geometry
    ``x`` — added to the element tangent in the drivers (the IMP_INT_K
    assembly step)."""
    from . import require_scipy
    sp, _ = require_scipy()
    rows, cols, vals = [], [], []
    for c in contacts:
        r, cc, v = c.triplets(x, dof)
        rows.append(r)
        cols.append(cc)
        vals.append(v)
    rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    vals = np.concatenate(vals) if vals else np.zeros(0)
    return sp.coo_matrix((vals, (rows, cols)),
                         shape=(dof.ndof, dof.ndof)).tocsr()
