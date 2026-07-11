"""
Penalty contact in the implicit Newton loop: /INTER/TYPE7 (M12, friction
M13) and /INTER/TYPE11 edge-to-edge (M13).

Fortran origin
--------------
``engine/source/interfaces/int07/i7ke3.F`` (I7KE3, the per-interface
implicit driver called from ``engine/source/implicit/imp_int_k.F``'s
IMP_INT_K) and ``i7keg3.F`` (I7KEG3, the per-pair stiffness blocks; the
same file holds I7FRF3 and I7KFOR3 — the implicit contact FORCE paths),
plus — M13 — ``engine/source/interfaces/int11/i11ke3.F`` / ``i11keg3.F``
(I11KE3/I11KEG3, the TYPE11 edge-to-edge implicit stiffness), with
``imp_solv.F``'s ICONTA bookkeeping deciding when the contact set forces
extra Newton work. Checked against the source, not docs:

* IMP_INT_K forces ``IMP_INT7 = 3`` around the assembly (for BOTH the
  I7KE3 and I11KE3 calls — verified in imp_int_k.F), and in that mode
  I7KEG3/I11KEG3 keep the pair stiffness CONSTANT (``STIF = HALF*STIF``,
  no GAP/(GAP-PENE) stiffening factor) — i.e. the assembled tangent is
  the plain linear penalty spring, the very same law the explicit port's
  i7for3/i11for3 already integrate (the near-crossing stiffening guard is
  a documented M4 simplification on the explicit side too, so residual
  and tangent here are mutually consistent AND consistent with the
  explicit solver the cross-validations compare against);
* the blocks it assembles are the spring along the CONTACT NORMAL:
  KI11 = K*H1*(n n^T) etc. — the projection weights H distributing the
  node-side spring onto the segment corners, with KI12 = -KI11 the
  node-corner coupling (I11KEG3: the same blocks with the edge weights
  H1 = HS1*HM1 ... — exactly the cross products the exact gap gradient
  produces). The port assembles the sharper EXACT gap linearization
  K * g g^T with g = [n, -H1 n, ..., -H4 n] (the original's corner blocks
  use H_j where the exact form has H_i*H_j — a diagonal-boosted
  modified-Newton variant) PLUS the closest-point curvature term the
  original omits entirely (see "Theory" below — the press example's
  grid-aligned punch corners forced the issue);
* FRICTION (M13 — this removes the M12 deferral). What the original's
  implicit branch actually does, from the fetched source:

  - the TANGENT (I7KEG3 / I11KEG3 "with friction" blocks): a tangential-
    plane spring FRIC*STIF*(I - n n^T) scaled by the corner weights — the
    plane projector is built as Q(1,:)Q(1,:)^T + Q(2,:)Q(2,:)^T from a
    Q-frame whose first axis is the tangential relative-velocity
    direction. No stick/slip decision: the SAME mu-scaled spring is
    assembled whether the pair sticks or slides (a modified Newton);
  - the FORCE (I7KFOR3, the "INCREMENTAL (STIFFNESS) FORMULATION" branch,
    IFQ >= 10): a genuine incremental RETURN MAPPING — the stored
    tangential force CAND_FX/Y/Z plus the tangential-stiffness increment
    STIF0*(DX,DY,DZ), projected off the normal, then radially returned to
    the Coulomb cone by ``BETA = MIN(ONE, XMU*SQRT(FN/FT))``;
  - I7FRF3 (the matrix-free re-evaluation path) applies the plain
    tangential spring FT = -FACT*DT on the step's tangential displacement
    with no cone cap.

  The port implements the RETURN MAPPING (the I7KFOR3 incremental branch,
  which is also the textbook static stick/slip algorithm — Simo &
  Laursen's penalty-regularized Coulomb law; Wriggers, *Computational
  Contact Mechanics* §5) and pairs it with its CONSISTENT tangent instead
  of I7KEG3's mu-scaled always-stick spring:

  - STICK (|f_t^tr| <= mu f_n): tangential spring K_t on the slip
    increment; tangent block K_t (I - n n^T);
  - SLIP: radial return f_t = mu f_n t, t = f_t^tr/|f_t^tr|, with the
    exact NONSYMMETRIC tangent mu K t n^T + (mu f_n K_t/|f_t^tr|)
    (I - n n^T - t t^T) (the derivative of the return map at frozen
    projection). The direct solver is LU; nonsymmetry costs nothing.

  K_t = K (the pair's normal penalty stiffness), exactly the original's
  choice (STIF0/FACT are the SAME stiffness that loads the normal). The
  anchored tangential force is COMMITTED once per converged increment /
  time step (the CAND_F storage of the original) and every Newton
  residual is a pure function of the trial displacement from that
  committed anchor — the same snapshot discipline as the element buffers.
  The anchor is projected onto the current tangential plane before the
  trial update (the original's FTN removal), so a rotating surface never
  accumulates a normal ghost component. The explicit port's friction is a
  velocity-regularized KINETIC law — a rate device; feeding it the
  pseudo-velocity du/1 would make the force step-size-dependent, so the
  static algorithm above replaces it (they agree where they must: in
  steady sliding both give |f_t| = mu f_n exactly, the cross-validation).

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

The friction force adds its stick/slip tangent blocks (module head) at
the same frozen projection; the friction-force curvature (the rotation
of n and the weight variation under corner motion) is omitted like the
original — the same O(p/L) class as the frozen-weight normal terms.

The active set is decided on p > 0 exactly as the force; the region is
classified from the projection weights of the trial configuration.

ACTIVE-SET Newton: the active set is simply re-evaluated at every residual
and tangent call (pairs enter and leave between iterations as p crosses
zero); no separate bookkeeping is needed because the penalty force is
CONTINUOUS at p = 0 (force and energy both vanish) and the friction
return map is CONTINUOUS across the stick/slip boundary (the derivative
jumps — the usual non-smooth Newton, which the M11 StepControl backstops).
An increment whose active set refuses to settle fails its Newton budget
and is handed to the StepControl, which cuts the increment and retries —
the imp_dt.F coupling the original reaches through its ICONTA/IMCONV
machinery.

Theory — TYPE11 edge-to-edge (M13)
----------------------------------
The same penalty-in-residual + gap-tangent pattern with segment-segment
closest-point kinematics (the explicit i11dst3 port,
``inter_type11._closest_points_on_segments``, called read-only): with
closest points cA = (1-s) x1 + s x2 on the secondary edge and
cB = (1-t) x3 + t x4 on the main edge,

    d = |cA - cB|,   n = (cA - cB)/d,   p = gap - d,
    f = K p:  +(1-s) f n, +s f n on the secondary ends,
              -(1-t) f n, -t f n on the main ends
    grad d = g = [(1-s)n, s n, -(1-t)n, -t n]        (12 dofs)

(the envelope theorem kills the ds/dt terms in grad d exactly as the
barycentric weights drop out of the TYPE7 gradient — so K g g^T is again
the exact frozen-parameter tangent, and the original's I11KEG3 blocks
K*HS_i*HM_j*(n n^T) are its diagonal-boosted variant). The curvature term
-K p grad^2 d here is EXACT in every projection region — the M12 TYPE7
lesson worked out for the edge-edge closest-point map instead of bounded:
differentiate the interior optimality conditions u.(cA-cB) = 0,
v.(cA-cB) = 0 (u, v the edge vectors),

    [ |u|^2   -u.v  ] [ds]   [ -u.P dx - d n.(dx2-dx1) ]
    [  u.v   -|v|^2 ] [dt] = [ -v.P dx - d n.(dx4-dx3) ]

(P dx the frozen-parameter relative motion), then
dn = [(I - n n^T) P dx + u ds - v dt]/d and the Hessian rows follow from
g. A CLAMPED parameter (closest point at an edge end) simply drops its
row (ds = 0), which reproduces the exact point-segment Hessian
(I - n n^T - v v^T/|v|^2)/d including the foot variation — sharper than
the TYPE7 edge-region term, which omits the foot motion. The 1/d factor
is floored at 5% of (d + p) exactly like the TYPE7 vertex term.

NEAR-PARALLEL edges (sin^2 of the crossing angle = |u x v|^2/(|u|^2|v|^2)
below 1e-4) get a special OVERLAP treatment, an M13 lesson learned the
same way M12 learned the curvature term: the closest-point pair of
parallel overlapping edges is NON-UNIQUE, the clamped solver picks an
END of the overlap, and which end it picks flips with the tilt SIGN as
the structure deforms — a genuinely DISCONTINUOUS single-point residual
that left Newton in an exact period-2 cycle on two parallel edges
pressed together (the force hopped between the edge ends every
iteration). The port therefore splits a near-parallel pair with a
genuine overlap into TWO sub-pairs at the overlap-interval ends, each
carrying HALF the pair stiffness with its own penetration and normal —
the two-point trapezoid quadrature of the line contact (exact resultant
for a linearly varying penetration; the explicit port's single
closest-point convention is a documented M4 simplification that
dynamics dithers through, but a static residual must be continuous).
Sub-pair parameters are OVERLAP constructions, not distance minimizers,
so their curvature keeps only the clamped-parameter terms (the s-drift
terms are O(theta) below the parallel threshold), and crossing the
threshold itself redistributes the force between two nearby points
(resultant-continuous; the moment jump is O(K L theta_c) — documented,
not hidden).

Theory — TYPE11 COULOMB FRICTION (M14)
--------------------------------------
The TYPE7 return mapping generalized to edge pairs. What the ORIGINAL's
implicit branch does for TYPE11 friction, from the fetched source
(i11keg3.F): the force path I11KFOR3 applies a plain tangential spring
``FTN = -FRIC*STIF*DXT`` on the step's tangential relative displacement
at the closest-point weights — with NO Coulomb cone cap and NO stored
anchor (unlike I7KFOR3's CAND_F incremental return) — and the tangent
path I11KEG3 assembles the same mu-scaled ALWAYS-STICK tangential-plane
spring FRIC*STIF*(Q1 Q1^T + Q2 Q2^T) as I7KEG3 (a modified Newton, no
stick/slip decision). The port deviates exactly as M13 did for TYPE7,
and for the same reason (Newton needs a residual with a bounded force
and a tangent consistent with it): the INCREMENTAL RETURN MAPPING on the
anchored tangential force, capped on the cone, with the consistent
stick/slip tangent per regime.

Kinematics: the slip increment of a pair is the relative motion of the
two closest MATERIAL points at frozen parameters (the TYPE7 frozen
weights, verbatim),

    delta = [cA(x) - cB(x)] - [cA(x_com) - cB(x_com)]   at fixed (s, t),

projected onto the tangential plane I - n n^T. Note what that plane IS
for edge-to-edge contact: at an interior-interior solution n is parallel
to u x v, so the plane orthogonal to n CONTAINS both edge directions —
axial sliding of either edge across the other is genuine rubbing and
enters the return map (only the normal approach is excluded). The
anchored force is projected onto the current tangential plane before the
trial update (the FTN removal), the trial force f_t^tr = f0 - K_t*dt is
radially returned to the cone mu*K*p, and the consistent tangent blocks
are (P P^T) (x) M over the four end nodes with P = [(1-s), s, -(1-t), -t]
and M the TYPE7 stick/slip matrices (K_t(I - n n^T); the nonsymmetric
slip derivative). K_t = K, the pair's normal penalty stiffness — the
original's own choice (FACT = FRIC*STIF).

ANCHOR KEYING (the near-parallel sub-pairs need a decision — here it
is): anchors are stored per (secondary edge row, main edge row, END
index) as key = 2*(i*n_main + j) + k. A generic single-closest-point
pair uses k = 0. The two overlap sub-pairs of a near-parallel pair use
k = 0 for the LOW overlap end and k = 1 for the HIGH end (ends ordered
by the secondary-edge parameter s — a stable labeling). Crossing INTO
the overlap regime the low-end sub-pair therefore INHERITS the
single-point anchor and the high end starts fresh at zero; crossing out,
the high-end anchor is dropped. The stored tangential resultant is
continuous to the same order as the frictionless force redistribution
already accepted at that threshold (module section above); the
alternative — one shared anchor per edge pair — would smear one slip
history over two points with different normals and break the per-point
return map.

The friction slip work lands in the same ``efric`` ledger channel as
TYPE7 (mu f_n dgamma per commit), the stick spring store joins
``econt``, and mu = 0 leaves every M13 path bit-identical (guarded at
each branch — asserted by the validations).

The pair stiffness K reuses ``contact/stiffness.py`` unchanged (the
i7sti3/i11sti3 element formulas + the Istf combination), and the gap
logic (Igap 0/1, Gap_min/Gap_max) mirrors the explicit interfaces line
for line. The candidate search here is a plain vectorized box overlap
test per call (implicit models are orders of magnitude below the
explicit crash meshes; the voxel broad phase of inter_type7/11 exists
for THOSE — no shared state is touched, per the M7 parity contract).

Rate devices stay off, consistent with the whole implicit branch: the
explicit normal VISCOUS damper (C*vn) is a rate device and does not
exist here, and the explicit velocity-regularized kinetic friction is
replaced by the static return mapping above.

Contact is evaluated at the CURRENT trial configuration (model.x + u) in
BOTH geometry modes — contact is inherently geometric; under the M8
small-strain element path the elements still linearize at x0 while the
gaps close on the accumulated displacement (documented port choice; the
original's implicit contact also walks the current X).

Energy bookkeeping (implicit dynamics): the stored normal spring energy
1/2 K p^2 plus the stored tangential (stick) spring energy
1/2 |f_t|^2 / K_t are state functions booked in the ``econt`` ledger
channel; frictional SLIP dissipates mu f_n dgamma per commit into the
``efric`` channel (dgamma = (|f_t^tr| - mu f_n)/K_t, the return map's
plastic slip — the exact analogue of plastic work). A pair that SEPARATES
while carrying a stored tangential force drops that energy from the
ledger without a matching work term — an inherent artifact of penalty
friction (the original's incremental CAND_F storage has it too),
documented here rather than hidden; it is bounded by 1/2 (mu f_n)^2/K_t
per released pair.

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
original has the same non-smooth set behind a looser default convergence
tolerance.

DEFERRED loudly (PORTING_GUIDE M13/M14): Ifric > 0 friction models
(MFROT 1/2/3 — viscous/Darmstadt/Renard — and the IFQ friction
filtering), Inacti initial-penetration treatments, Igap 2/3, sensor
gating (TSTART/TSTOP) under the implicit clock, the I7KEG3/I11KEG3
stiffening modes IMP_INT7 = 0/1.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20
from ..contact.inter_type7 import _narrow
from ..contact.inter_type11 import _closest_points_on_segments
from ..contact.stiffness import (combine_stiffness, edge_stiffness_gap,
                                 node_stiffness_gap, segment_stiffness_gap,
                                 _segment_areas)
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

        # ---- Coulomb friction state (M13) --------------------------------
        # mu > 0 switches the incremental return mapping on (module
        # docstring). The COMMITTED state is the anchor configuration
        # ``x_com`` (the geometry of the last converged increment) and the
        # anchored tangential force per pair (the original's CAND_FX/Y/Z),
        # stored as sorted integer keys node*nseg + segrow -> (m, 3)
        # vectors. mu = 0 leaves every friction path untouched (the M12
        # results stay bit-identical — asserted by the validations).
        self.mu = float(itf.fric)
        self.x_com = model.x.copy()
        self.ft_keys = np.zeros(0, dtype=np.int64)
        self.ft_vals = np.zeros((0, 3))

        if getattr(itf, "sens_id", 0):
            log.warning(
                f"/INTER/TYPE7/{itf.id}: /SENSOR gating is not evaluated "
                f"under the implicit solver — the interface is active for "
                f"the whole run", "IMPL CONTACT")
        fric_txt = (f"COULOMB FRICTION mu = {self.mu:g} (i7kfor3 return "
                    f"mapping)" if self.mu > 0.0 else "frictionless")
        log.info(f"     /INTER/TYPE7/{itf.id}: IMPLICIT PENALTY CONTACT — "
                 f"{len(self.nodes)} secondary node(s) vs "
                 f"{len(self.segs)} segment(s) (i7ke3.F, {fric_txt})")

    # ------------------------------------------------------------------
    def _active_pairs(self, x):
        """Re-project every box-overlap candidate at the trial geometry x
        and keep the penetrating pairs. Returns
        (ni, seg, w, nvec, pen, K, d, srow) — the full active-set state of
        this configuration (``d`` the closest-point distance, needed by
        the curvature term of the tangent; ``srow`` the segment row, the
        friction anchor key)."""
        empty = (np.zeros(0, dtype=np.int64),
                 np.zeros((0, 4), dtype=np.int64), np.zeros((0, 4)),
                 np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0),
                 np.zeros(0, dtype=np.int64))
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
        return ni, seg, w, nvec, pen, K, d, jj

    # ------------------------------------------------------------------
    def _friction_state(self, x, ni, seg, srow, w, nvec, pen, K):
        """The incremental return mapping of every active pair at trial
        geometry ``x`` from the COMMITTED anchors (module docstring —
        the I7KFOR3 incremental branch):

            delta   = (x - x_com) relative motion at the CURRENT weights
            f_t^tr  = P_T(anchor) - K_t * P_T(delta)     (K_t = K)
            stick   : f_t = f_t^tr                (|f_t^tr| <= mu K p)
            slip    : f_t = mu K p * f_t^tr/|f_t^tr|

        Returns (ft, stick, tnorm, ttr) — the tangential force (m, 3),
        the stick mask, |f_t^tr| and its unit direction (slip rows only
        meaningful) — everything forces()/triplets()/commit() need.
        Never called when mu = 0."""
        m = len(ni)
        # slip increment at frozen (current) weights, tangential part
        delta = ((x[ni] - self.x_com[ni])
                 - np.einsum("mk,mkb->mb", w, x[seg] - self.x_com[seg]))
        dn = np.einsum("mb,mb->m", delta, nvec)
        dt_vec = delta - dn[:, None] * nvec
        # committed anchor lookup (sorted-key searchsorted; missing = 0)
        f0 = np.zeros((m, 3))
        if len(self.ft_keys):
            key = ni * len(self.segs) + srow
            pos = np.searchsorted(self.ft_keys, key)
            pos = np.minimum(pos, len(self.ft_keys) - 1)
            hit = self.ft_keys[pos] == key
            f0[hit] = self.ft_vals[pos[hit]]
        # project the anchor onto the CURRENT tangential plane (the
        # original's FTN removal — a rotated surface must not leave a
        # normal ghost in the anchor)
        f0 -= np.einsum("mb,mb->m", f0, nvec)[:, None] * nvec
        ftr = f0 - K[:, None] * dt_vec
        tnorm = np.sqrt(np.einsum("mb,mb->m", ftr, ftr))
        fcap = self.mu * K * pen                       # the Coulomb cone
        stick = tnorm <= fcap
        ttr = ftr / np.maximum(tnorm, EM20)[:, None]
        ft = np.where(stick[:, None], ftr, fcap[:, None] * ttr)
        return ft, stick, tnorm, ttr

    # ------------------------------------------------------------------
    def forces(self, x, fcont):
        """Penalty (+ friction, M13) force of the active set at trial
        geometry ``x``, scattered into ``fcont`` (node space) — the
        residual contribution (i7for3's spring term, no rate damper; the
        I7KFOR3 return-mapped tangential force). Returns the number of
        active pairs (the listing's active-set trace)."""
        ni, seg, w, nvec, pen, K, _, srow = self._active_pairs(x)
        if len(ni) == 0:
            return 0
        Fvec = (K * pen)[:, None] * nvec
        if self.mu > 0.0:
            ft, _, _, _ = self._friction_state(x, ni, seg, srow, w, nvec,
                                               pen, K)
            Fvec = Fvec + ft
        np.add.at(fcont, ni, Fvec)
        np.add.at(fcont, seg.reshape(-1),
                  (-w[:, :, None] * Fvec[:, None, :]).reshape(-1, 3))
        return len(ni)

    # ------------------------------------------------------------------
    def energy(self, x):
        """Stored penalty-spring energy of the active set — normal
        1/2 K p^2 plus (M13) the stick spring's tangential store
        1/2 |f_t|^2/K_t. A state function, booked in the implicit-dynamics
        ledger so a contact transient balances (the explicit CE
        analogue)."""
        ni, seg, w, nvec, pen, K, _, srow = self._active_pairs(x)
        if len(ni) == 0:
            return 0.0
        e = 0.5 * float((K * pen * pen).sum())
        if self.mu > 0.0:
            ft, _, _, _ = self._friction_state(x, ni, seg, srow, w, nvec,
                                               pen, K)
            e += 0.5 * float((np.einsum("mb,mb->m", ft, ft) / K).sum())
        return e

    # ------------------------------------------------------------------
    def commit(self, x):
        """Re-base the friction anchors on a CONVERGED configuration ``x``
        (the original's CAND_F save — called by the drivers exactly where
        the element buffers commit; a failed increment never reaches
        here). Returns the frictional SLIP dissipation of the committed
        increment, Sum mu f_n dgamma with dgamma = (|f_t^tr| - mu f_n)/K_t
        — the return map's plastic-slip work, booked in the dynamics
        ``efric`` ledger channel (statics ignores the return value)."""
        if self.mu <= 0.0:
            return 0.0
        ni, seg, w, nvec, pen, K, _, srow = self._active_pairs(x)
        if len(ni) == 0:
            self.x_com = x.copy()
            self.ft_keys = np.zeros(0, dtype=np.int64)
            self.ft_vals = np.zeros((0, 3))
            return 0.0
        # evaluate the converged tangential forces BEFORE re-basing the
        # anchor configuration (the slip increment is measured from the
        # OLD x_com — re-basing first would store the stale anchors and
        # silently drop the increment's tangential update)
        ft, stick, tnorm, _ = self._friction_state(x, ni, seg, srow, w,
                                                   nvec, pen, K)
        self.x_com = x.copy()
        fcap = self.mu * K * pen
        slip_g = np.where(stick, 0.0, (tnorm - fcap) / K)   # plastic slip
        diss = float((fcap * slip_g).sum())
        key = ni * len(self.segs) + srow
        order = np.argsort(key)
        self.ft_keys = key[order]
        self.ft_vals = ft[order]
        return diss

    # ------------------------------------------------------------------
    def triplets(self, x, dof):
        """COO triplets of the active-set tangent in EQUATION space at
        trial geometry ``x``: per pair the rank-one gap block K * g g^T
        MINUS the closest-point curvature K p (P P^T) (x) Hess(d) — the
        exact tangent per projection region (see module docstring) —
        PLUS (M13) the friction stick/slip blocks (P P^T) (x) M with
        M = K_t (I - n n^T) for stick and the nonsymmetric return-map
        derivative mu K t n^T + (mu f_n K_t/|f_t^tr|)(I - n n^T - t t^T)
        for slip — over [node, corner1..4], mapped through the DofMap
        exactly like assembly.element_triplets."""
        ni, seg, w, nvec, pen, K, d, srow = self._active_pairs(x)
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
        if self.mu > 0.0:
            # friction blocks (P P^T) (x) M — see the docstring above for
            # M per regime; P = [1, -H1..-H4] is the same pair pattern the
            # curvature term uses, so the assembled rows/columns match the
            # residual's force distribution exactly.
            ft, stick, tnorm, ttr = self._friction_state(
                x, ni, seg, srow, w, nvec, pen, K)
            eye = np.eye(3)
            T = eye[None] - np.einsum("mi,mj->mij", nvec, nvec)
            M = np.where(stick[:, None, None],
                         K[:, None, None] * T,
                         # slip: mu K t n^T + (mu fn K/|ftr|)(T - t t^T)
                         (self.mu * K)[:, None, None]
                         * np.einsum("mi,mj->mij", ttr, nvec)
                         + (self.mu * K * pen * K
                            / np.maximum(tnorm, EM20))[:, None, None]
                         * (T - np.einsum("mi,mj->mij", ttr, ttr)))
            P = np.empty((npair, 5))
            P[:, 0] = 1.0
            P[:, 1:] = -w
            ke += np.einsum("ma,mb,mij->maibj", P, P, M).reshape(
                npair, 15, 15)
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
# /INTER/TYPE11 edge-to-edge (M13)
# ----------------------------------------------------------------------------

class ImplicitContact11:
    """One /INTER/TYPE11 edge-to-edge interface in the implicit system
    (module docstring "Theory — TYPE11"). Same ownership contract as
    ImplicitContact7: copies everything, mutates nothing shared."""

    def __init__(self, itf, model, log):
        self.itf = itf

        def _line(lid, side):
            ln = model.lines[lid]
            if ln.segments is None or len(ln.segments) == 0:
                log.warning(f"/INTER/TYPE11/{itf.id}: {side} line {lid} is "
                            f"empty — interface inactive", "IMPL CONTACT")
                return (np.zeros((0, 2), dtype=np.int64),
                        np.zeros(0, dtype="<U8"),
                        np.zeros(0, dtype=np.int64))
            return ln.segments, ln.seg_gtype, ln.seg_elem

        self.es, es_gtype, es_elem = _line(itf.line_id1, "secondary")
        self.em, em_gtype, em_elem = _line(itf.line_id2, "main")

        # per-edge stiffness and gap: the i11sti3 formulas, reused from
        # contact/stiffness.py exactly like the explicit interface
        scale = itf.stfac if itf.istf != 1 else 1.0
        self.Ks, gs = edge_stiffness_gap(model, self.es, es_gtype,
                                         es_elem, scale)
        self.Km, gm = edge_stiffness_gap(model, self.em, em_gtype,
                                         em_elem, scale)

        # gap policy: mirror of the explicit ContactType11 __init__
        if len(self.em):
            lc = float(np.linalg.norm(model.x0[self.em[:, 1]]
                                      - model.x0[self.em[:, 0]],
                                      axis=1).mean())
        else:
            lc = 1.0
        both = (gm.mean() if len(gm) else 0.0) + (gs.mean() if len(gs)
                                                  else 0.0)
        gap_floor = itf.gap if itf.gap > 0 else (
            both if both > 0 else 0.02 * lc)
        if itf.igap == 1:
            self.gap_s = gs
            self.gap_m = gm
            self.gap_min = gap_floor
            self.gap_max = itf.gap_max if itf.gap_max > 0 else np.inf
            hi = (gs.max() if len(gs) else 0.0) + (gm.max() if len(gm)
                                                   else 0.0)
            self.gap_bound = float(np.clip(hi, self.gap_min, self.gap_max))
        else:
            self.gap_const = gap_floor
            self.gap_bound = gap_floor

        # ---- Coulomb friction state (M14) --------------------------------
        # the TYPE7 return mapping generalized to edge pairs (module
        # docstring "TYPE11 COULOMB FRICTION"): committed anchor geometry
        # x_com + anchored tangential forces keyed 2*(i*n_main + j) + k
        # (k the overlap-end index — the documented keying decision).
        # mu = 0 leaves every friction branch untouched (M13 bit-identical).
        self.mu = float(itf.fric)
        self.x_com = model.x.copy()
        self.ft_keys = np.zeros(0, dtype=np.int64)
        self.ft_vals = np.zeros((0, 3))

        if getattr(itf, "sens_id", 0):
            log.warning(
                f"/INTER/TYPE11/{itf.id}: /SENSOR gating is not evaluated "
                f"under the implicit solver — the interface is active for "
                f"the whole run", "IMPL CONTACT")
        fric_txt = (f"COULOMB FRICTION mu = {self.mu:g} (M14 edge-pair "
                    f"return mapping)" if self.mu > 0.0 else "frictionless")
        log.info(f"     /INTER/TYPE11/{itf.id}: IMPLICIT PENALTY CONTACT — "
                 f"{len(self.es)} secondary edge(s) vs "
                 f"{len(self.em)} main edge(s) (i11ke3.F, {fric_txt})")

    #: near-parallel threshold on sin^2 of the edge crossing angle — below
    #: it the closest-point pair is treated as non-unique and the pair
    #: goes through the two-point overlap quadrature (module docstring).
    _PAR_TOL = 1e-4

    # ------------------------------------------------------------------
    def _active_pairs(self, x):
        """Box-overlap candidates re-projected at trial geometry ``x``
        (the explicit i11dst3 closest points, read-only); returns the
        penetrating pairs: (ea, eb, s, t, nvec, pen, K, d, frozen, key)
        with ``ea`` the secondary edge ends (m, 2), ``eb`` the main ends,
        s/t the closest-point parameters, ``frozen`` True on the
        near-parallel OVERLAP sub-pairs, whose parameters are overlap
        constructions rather than distance minimizers (module docstring),
        and ``key`` the M14 friction-anchor key 2*(i*n_main + j) + k
        (the documented sub-pair keying)."""
        empty = (np.zeros((0, 2), dtype=np.int64),
                 np.zeros((0, 2), dtype=np.int64), np.zeros(0),
                 np.zeros(0), np.zeros((0, 3)), np.zeros(0), np.zeros(0),
                 np.zeros(0), np.zeros(0, dtype=bool),
                 np.zeros(0, dtype=np.int64))
        if len(self.es) == 0 or len(self.em) == 0:
            return empty
        margin = 1.001 * self.gap_bound
        xa = x[self.es]                                  # (ns, 2, 3)
        xb = x[self.em]                                  # (nm, 2, 3)
        lo_a, hi_a = xa.min(axis=1), xa.max(axis=1)
        lo_b = xb.min(axis=1) - margin
        hi_b = xb.max(axis=1) + margin
        inside = np.all((hi_a[:, None, :] >= lo_b[None, :, :])
                        & (lo_a[:, None, :] <= hi_b[None, :, :]), axis=2)
        ii, jj = np.nonzero(inside)
        if len(ii) == 0:
            return empty
        ea = self.es[ii]
        eb = self.em[jj]
        # edges sharing a node never contact each other (adjacency
        # exclusion — the explicit broad phase does the same)
        keep = np.ones(len(ii), dtype=bool)
        for a in range(2):
            for b in range(2):
                keep &= ea[:, a] != eb[:, b]
        ea, eb, ii, jj = ea[keep], eb[keep], ii[keep], jj[keep]
        if len(ii) == 0:
            return empty

        # friction-anchor base key of each candidate PAIR (M14): the
        # sub-index k is appended below (2*base + k)
        base_key = ii * max(len(self.em), 1) + jj

        s, t, cA, cB = _closest_points_on_segments(
            x[ea[:, 0]], x[ea[:, 1]], x[eb[:, 0]], x[eb[:, 1]])
        if self.itf.igap == 1:
            gap = np.clip(self.gap_s[ii] + self.gap_m[jj],
                          self.gap_min, self.gap_max)
        else:
            gap = np.full(len(ii), self.gap_const)
        K = combine_stiffness(self.itf.istf, self.itf.stfac,
                              self.Km[jj], self.Ks[ii])

        # ---- near-parallel overlap split (module docstring: the M13
        # period-2 lesson) — a parallel pair with a genuine overlap is
        # replaced by TWO sub-pairs at the overlap-interval ends, each
        # with half the stiffness and its own closest point on the main
        # edge; everything else flows through the same pair pipeline.
        u = x[ea[:, 1]] - x[ea[:, 0]]
        v = x[eb[:, 1]] - x[eb[:, 0]]
        uu = np.maximum(np.einsum("mb,mb->m", u, u), EM20)
        vv = np.maximum(np.einsum("mb,mb->m", v, v), EM20)
        uv = np.einsum("mb,mb->m", u, v)
        par = (uu * vv - uv * uv) < self._PAR_TOL * uu * vv
        if np.any(par):
            w = x[eb[par, 0]] - x[ea[par, 0]]
            sb0 = np.einsum("mb,mb->m", w, u[par]) / uu[par]
            sb1 = sb0 + uv[par] / uu[par]
            lo = np.clip(np.minimum(sb0, sb1), 0.0, 1.0)
            hi = np.clip(np.maximum(sb0, sb1), 0.0, 1.0)
            ov = hi - lo > 1e-6           # no overlap -> the generic
            if np.any(ov):                # (endpoint) solution is smooth
                rows = np.where(par)[0][ov]
                subs = []
                for send in (lo[ov], hi[ov]):
                    cAs = x[ea[rows, 0]] + send[:, None] * u[rows]
                    ts = np.clip(np.einsum(
                        "mb,mb->m", cAs - x[eb[rows, 0]], v[rows])
                        / vv[rows], 0.0, 1.0)
                    subs.append((send, ts))
                keep = np.ones(len(ea), dtype=bool)
                keep[rows] = False
                ea_s = np.concatenate([ea[keep]] + [ea[rows]] * 2)
                eb_s = np.concatenate([eb[keep]] + [eb[rows]] * 2)
                s = np.concatenate([s[keep], subs[0][0], subs[1][0]])
                t = np.concatenate([t[keep], subs[0][1], subs[1][1]])
                gap = np.concatenate([gap[keep]] + [gap[rows]] * 2)
                K = np.concatenate([K[keep]] + [0.5 * K[rows]] * 2)
                frozen = np.concatenate([np.zeros(int(keep.sum()),
                                                  dtype=bool),
                                         np.ones(2 * len(rows),
                                                 dtype=bool)])
                # anchor keys: generic pairs k = 0; overlap sub-pairs
                # k = 0 (LOW s end, inheriting the single-point anchor)
                # and k = 1 (HIGH end) — the documented keying decision
                key = np.concatenate([2 * base_key[keep],
                                      2 * base_key[rows],
                                      2 * base_key[rows] + 1])
                ea, eb = ea_s, eb_s
                cA = ((1.0 - s)[:, None] * x[ea[:, 0]]
                      + s[:, None] * x[ea[:, 1]])
                cB = ((1.0 - t)[:, None] * x[eb[:, 0]]
                      + t[:, None] * x[eb[:, 1]])
            else:
                frozen = np.zeros(len(ea), dtype=bool)
                key = 2 * base_key
        else:
            frozen = np.zeros(len(ea), dtype=bool)
            key = 2 * base_key

        dvec = cA - cB
        dd = np.sqrt(np.einsum("mb,mb->m", dvec, dvec))
        pen = gap - dd
        act = pen > 0.0
        if not np.any(act):
            return empty
        ea, eb, s, t = ea[act], eb[act], s[act], t[act]
        pen = pen[act]
        d = np.maximum(dd[act], EM20)
        nvec = dvec[act] / d[:, None]        # pushes the secondary edge out
        return ea, eb, s, t, nvec, pen, K[act], d, frozen[act], key[act]

    # ------------------------------------------------------------------
    @staticmethod
    def _gvec(s, t, nvec):
        """The exact gap gradient over the 12 end-node translations,
        g = [(1-s)n, s n, -(1-t)n, -t n] (envelope theorem — the
        closest-point parameter variations drop out)."""
        m = len(s)
        g = np.zeros((m, 12))
        g[:, 0:3] = (1.0 - s)[:, None] * nvec
        g[:, 3:6] = s[:, None] * nvec
        g[:, 6:9] = -(1.0 - t)[:, None] * nvec
        g[:, 9:12] = -t[:, None] * nvec
        return g

    # ------------------------------------------------------------------
    def _friction_state(self, x, ea, eb, s, t, key, nvec, pen, K):
        """The M14 edge-pair return mapping at trial geometry ``x`` from
        the committed anchors (module docstring "TYPE11 COULOMB
        FRICTION"): the slip increment is the relative motion of the two
        closest MATERIAL points at frozen parameters, tangentially
        projected (the plane orthogonal to n — which CONTAINS both edge
        directions at a crossing: axial edge sliding is genuine slip);
        stick keeps the trial spring force, slip radially returns it to
        the cone mu K p. Returns (ft, stick, tnorm, ttr) like the TYPE7
        twin. Never called when mu = 0."""
        m = len(s)
        # relative closest-point motion at frozen (s, t): trial - committed
        dx = x - self.x_com
        delta = ((1.0 - s)[:, None] * dx[ea[:, 0]]
                 + s[:, None] * dx[ea[:, 1]]
                 - (1.0 - t)[:, None] * dx[eb[:, 0]]
                 - t[:, None] * dx[eb[:, 1]])
        dn = np.einsum("mb,mb->m", delta, nvec)
        dt_vec = delta - dn[:, None] * nvec
        # committed anchor lookup (sorted-key searchsorted; missing = 0)
        f0 = np.zeros((m, 3))
        if len(self.ft_keys):
            pos = np.searchsorted(self.ft_keys, key)
            pos = np.minimum(pos, len(self.ft_keys) - 1)
            hit = self.ft_keys[pos] == key
            f0[hit] = self.ft_vals[pos[hit]]
        # project the anchor onto the CURRENT tangential plane (the FTN
        # removal — a rotating pair must not keep a normal ghost)
        f0 -= np.einsum("mb,mb->m", f0, nvec)[:, None] * nvec
        ftr = f0 - K[:, None] * dt_vec
        tnorm = np.sqrt(np.einsum("mb,mb->m", ftr, ftr))
        fcap = self.mu * K * pen                       # the Coulomb cone
        stick = tnorm <= fcap
        ttr = ftr / np.maximum(tnorm, EM20)[:, None]
        ft = np.where(stick[:, None], ftr, fcap[:, None] * ttr)
        return ft, stick, tnorm, ttr

    # ------------------------------------------------------------------
    def forces(self, x, fcont):
        """Penalty (+ friction, M14) force of the active set at trial
        geometry ``x``, scattered into ``fcont`` — i11for3's spring term
        (no rate damper) plus the M14 return-mapped tangential force. The
        force is distributed with the closest-point parameters — the
        normal parts are collinear equal/opposite point forces (exact
        linear AND angular momentum balance); the tangential parts are
        equal/opposite at the two closest points, whose normal offset d
        leaves the same O(f_t*d) moment the TYPE7 node-vs-projection
        transfer carries."""
        ea, eb, s, t, nvec, pen, K, _, _, key = self._active_pairs(x)
        if len(s) == 0:
            return 0
        F = (K * pen)[:, None] * nvec
        if self.mu > 0.0:
            ft, _, _, _ = self._friction_state(x, ea, eb, s, t, key, nvec,
                                               pen, K)
            F = F + ft
        va = np.empty((len(s), 2, 3))
        va[:, 0, :] = (1.0 - s)[:, None] * F
        va[:, 1, :] = s[:, None] * F
        np.add.at(fcont, ea.reshape(-1), va.reshape(-1, 3))
        vb = np.empty((len(t), 2, 3))
        vb[:, 0, :] = -(1.0 - t)[:, None] * F
        vb[:, 1, :] = -t[:, None] * F
        np.add.at(fcont, eb.reshape(-1), vb.reshape(-1, 3))
        return len(s)

    # ------------------------------------------------------------------
    def energy(self, x):
        """Stored penalty-spring energy 1/2 K p^2 of the active set,
        plus (M14) the stick spring's tangential store 1/2 |f_t|^2/K_t."""
        ea, eb, s, t, nvec, pen, K, _, _, key = self._active_pairs(x)
        if len(s) == 0:
            return 0.0
        e = 0.5 * float((K * pen * pen).sum())
        if self.mu > 0.0:
            ft, _, _, _ = self._friction_state(x, ea, eb, s, t, key, nvec,
                                               pen, K)
            e += 0.5 * float((np.einsum("mb,mb->m", ft, ft) / K).sum())
        return e

    # ------------------------------------------------------------------
    def commit(self, x):
        """Re-base the friction anchors on a CONVERGED configuration
        (M14 — the TYPE7 commit discipline verbatim: forces evaluated
        from the OLD anchors first, then the anchor geometry re-based).
        Returns the frictional slip dissipation of the increment for the
        ``efric`` ledger channel."""
        if self.mu <= 0.0:
            return 0.0
        ea, eb, s, t, nvec, pen, K, _, _, key = self._active_pairs(x)
        if len(s) == 0:
            self.x_com = x.copy()
            self.ft_keys = np.zeros(0, dtype=np.int64)
            self.ft_vals = np.zeros((0, 3))
            return 0.0
        ft, stick, tnorm, _ = self._friction_state(x, ea, eb, s, t, key,
                                                   nvec, pen, K)
        self.x_com = x.copy()
        fcap = self.mu * K * pen
        slip_g = np.where(stick, 0.0, (tnorm - fcap) / K)   # plastic slip
        diss = float((fcap * slip_g).sum())
        # duplicate keys cannot happen: a pair contributes one sub-index
        # each (the near-parallel split emits k = 0 and k = 1 once)
        order = np.argsort(key)
        self.ft_keys = key[order]
        self.ft_vals = ft[order]
        return diss

    # ------------------------------------------------------------------
    def triplets(self, x, dof):
        """COO triplets of the active-set tangent in EQUATION space:
        K g g^T minus the EXACT edge-edge closest-point curvature
        K p Hess(d) (module docstring "Theory — TYPE11": the 2x2
        optimality-system linearization, exact in every projection
        region, with the near-parallel guard), PLUS (M14) the friction
        stick/slip blocks (P P^T) (x) M with P = [(1-s), s, -(1-t), -t]
        and M the TYPE7 consistent regime matrices."""
        ea, eb, s, t, nvec, pen, K, d, frozen, key = self._active_pairs(x)
        z = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
             np.zeros(0))
        m = len(s)
        if m == 0:
            return z
        g = self._gvec(s, t, nvec)
        ke = K[:, None, None] * g[:, :, None] * g[:, None, :]
        ke -= self._curvature_blocks(x, ea, eb, s, t, nvec, pen, K, d,
                                     frozen)
        if self.mu > 0.0:
            # friction blocks (P P^T) (x) M — stick K_t(I - n n^T), slip
            # the nonsymmetric return-map derivative (module docstring);
            # P matches the force distribution rows exactly.
            ft, stick, tnorm, ttr = self._friction_state(
                x, ea, eb, s, t, key, nvec, pen, K)
            eye = np.eye(3)
            T = eye[None] - np.einsum("mi,mj->mij", nvec, nvec)
            M = np.where(stick[:, None, None],
                         K[:, None, None] * T,
                         (self.mu * K)[:, None, None]
                         * np.einsum("mi,mj->mij", ttr, nvec)
                         + (self.mu * K * pen * K
                            / np.maximum(tnorm, EM20))[:, None, None]
                         * (T - np.einsum("mi,mj->mij", ttr, ttr)))
            P = np.empty((m, 4))
            P[:, 0] = 1.0 - s
            P[:, 1] = s
            P[:, 2] = -(1.0 - t)
            P[:, 3] = -t
            ke += np.einsum("ma,mb,mij->maibj", P, P, M).reshape(m, 12, 12)
        edofs = np.zeros((m, 12), dtype=np.int64)
        ends = np.concatenate([ea, eb], axis=1)          # (m, 4)
        for k in range(4):
            for c in range(3):
                edofs[:, 3 * k + c] = ends[:, k] * DOFS_PER_NODE + c
        eq = dof.eq[edofs]
        row_eq = np.repeat(eq[:, :, None], 12, axis=2)
        col_eq = np.repeat(eq[:, None, :], 12, axis=1)
        keep = (row_eq >= 0) & (col_eq >= 0)
        return row_eq[keep].ravel(), col_eq[keep].ravel(), ke[keep].ravel()

    # ------------------------------------------------------------------
    @staticmethod
    def _curvature_blocks(x, ea, eb, s, t, nvec, pen, K, d, frozen):
        """K p Hess(d) of every active pair, (m, 12, 12) — the EXACT
        Hessian of the segment-segment closest-point distance (module
        docstring): differentiate the interior optimality conditions for
        (ds, dt), then dn, then the gradient rows. CLAMPED parameters
        (s or t at an edge end) drop their optimality row — this exactly
        reproduces the point-segment / point-point Hessians. FROZEN
        pairs (the near-parallel overlap sub-pairs) treat s as clamped
        too — their s is an overlap construction, not a minimizer, and
        the singular 2x2 system must never be solved for them. 1/d
        floored at 5% of (d + p) like the TYPE7 vertex term."""
        m = len(s)
        out = np.zeros((m, 12, 12))
        eye = np.eye(3)
        ptol = 1e-9
        for p in range(m):
            u = x[ea[p, 1]] - x[ea[p, 0]]
            v = x[eb[p, 1]] - x[eb[p, 0]]
            n = nvec[p]
            dcl = max(d[p], 0.05 * (d[p] + pen[p]))
            # frozen-parameter relative-motion map P3 (3 x 12) and the
            # end-difference maps Du, Dv (3 x 12)
            P3 = np.zeros((3, 12))
            P3[:, 0:3] = (1.0 - s[p]) * eye
            P3[:, 3:6] = s[p] * eye
            P3[:, 6:9] = -(1.0 - t[p]) * eye
            P3[:, 9:12] = -t[p] * eye
            Du = np.zeros((3, 12))
            Du[:, 0:3] = -eye
            Du[:, 3:6] = eye
            Dv = np.zeros((3, 12))
            Dv[:, 6:9] = -eye
            Dv[:, 9:12] = eye
            # optimality-row right-hand sides (12,)
            r1 = -(u @ P3) - d[p] * (n @ Du)
            r2 = -(v @ P3) - d[p] * (n @ Dv)
            uu, vv, uv = float(u @ u), float(v @ v), float(u @ v)
            int_s = (not frozen[p]) and ptol < s[p] < 1.0 - ptol
            int_t = ptol < t[p] < 1.0 - ptol
            gs = np.zeros(12)                            # grad s (12,)
            gt = np.zeros(12)                            # grad t (12,)
            if int_s and int_t:
                det = -(uu * vv - uv * uv)               # = -|u x v|^2
                if abs(det) > 1e-8 * uu * vv:
                    # [uu -uv; uv -vv] [gs; gt] = [r1; r2]
                    gs = (-vv * r1 + uv * r2) / det
                    gt = (-uv * r1 + uu * r2) / det
                # else: near-parallel — drop the parameter response
                # (non-unique closest points; guard, see docstring)
            elif int_s:                                  # t clamped
                gs = r1 / uu if uu > 0.0 else gs
            elif int_t:                                  # s clamped
                gt = -r2 / vv if vv > 0.0 else gt
            # dn = [(I - n n^T) P3 + u gs^T - v gt^T]/d  (3 x 12)
            N3 = ((eye - np.outer(n, n)) @ P3
                  + np.outer(u, gs) - np.outer(v, gt)) / dcl
            # gradient rows: g1 = (1-s)n, g2 = s n, g3 = -(1-t)n, g4 = -t n
            H = np.empty((12, 12))
            H[0:3] = -np.outer(n, gs) + (1.0 - s[p]) * N3
            H[3:6] = np.outer(n, gs) + s[p] * N3
            H[6:9] = np.outer(n, gt) - (1.0 - t[p]) * N3
            H[9:12] = -np.outer(n, gt) - t[p] * N3
            out[p] = (K[p] * pen[p]) * H
        return out


# ----------------------------------------------------------------------------
# builders + assembly helpers
# ----------------------------------------------------------------------------

def build_implicit_contacts(model, log):
    """Instantiate the implicit contact treatments: TYPE7 (friction since
    M13) and TYPE11 (M13) here; TYPE2 is a CONSTRAINT
    (implicit/constraints.py — the i2_imp1.F condensation)."""
    out = []
    for itf in model.interfaces:
        if itf.type == 7:
            out.append(ImplicitContact7(itf, model, log))
        elif itf.type == 11:
            out.append(ImplicitContact11(itf, model, log))
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


def commit_contacts(contacts, x):
    """Commit hook the drivers call once per CONVERGED increment/step:
    re-bases the friction anchors on the converged configuration and
    returns the total frictional slip dissipation of the increment (the
    ``efric`` ledger channel — see ImplicitContact7.commit)."""
    return sum(c.commit(x) for c in contacts)


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
