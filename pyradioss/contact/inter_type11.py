"""
/INTER/TYPE11 — penalty edge-to-edge contact.

Fortran origin: ``engine/source/interfaces/int11/`` —

    intsort/i11main_tri.F / i11buce.F   candidate edge pairs (bucket sort)
    int11/i11dst3.F                     exact edge-edge distance/projection
    int11/i11for3.F                     penalty force + friction
    inter3d1/i11sti3.F                  (Starter) edge stiffness and gap
                                        — ported in contact/stiffness.py

Why a separate interface exists at all: node-to-surface contact (TYPE7)
cannot see two edges crossing *between* their nodes — think of two shell
plates meeting like crossed swords, or a beam sliding along another
beam's flank. The closest points then lie in the interior of both edges
and no node penetrates any segment; only an edge-edge measure detects it.
The standard crash-model practice is a TYPE7 plus a TYPE11 on the same
parts, which the port supports (the two interfaces just add forces).

Algorithm — the same skeleton as the ported TYPE7:

1. **Broad phase** (every ``refresh`` cycles): voxel binning of the main
   edges' bounding boxes, inflated by gap + travel margin (cell size =
   largest box extent, so a box covers at most 8 cells); secondary edges
   are binned the same way; candidates = box pairs sharing a cell,
   deduplicated, minus pairs sharing a node.

2. **Narrow phase** (every cycle): exact closest points between the two
   segments (Ericson, 'Real-Time Collision Detection' §5.1.9 — the
   clamped two-parameter minimization), giving the points
   cS = a1 + s (a2-a1) on the secondary edge and cM = b1 + t (b2-b1) on
   the main edge. Penetration p = gap_pair - |cS - cM|.

3. **Penalty force** (i11for3): a spring along the connecting direction
   pushes the edges apart, plus the same normal damper and regularized
   Coulomb friction as TYPE7. The force is split onto the edge end nodes
   with the closest-point parameters — (1-s, s) on the secondary edge,
   -(1-t, t) on the main edge — which conserves BOTH linear and angular
   momentum (the two point forces are equal, opposite and collinear).

4. **Element deletion (M3<->M4)**: edges inherit their parent element
   from the /LINE (Starter provenance); edges of deleted elements are
   masked out every cycle, exactly like TYPE7 segments.

The interface time step is the node-on-spring bound dt = sqrt(2 m / K)
for the worst (mass, combined stiffness) on either side, like TYPE7.

Port simplifications: no Inacti / Tstart / sensors; the gap is constant
per pair (Igap=1 uses the two edges' half-thicknesses, no mesh-size
scaling variants); parallel-edge contact acts at the single closest-point
pair (the original distributes along the overlap; the resultant is the
same, the distribution slightly different).
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20
from ..model.model import Model
from . import tracking
from .inter_type7 import _expand_matches
from .stiffness import combine_stiffness, edge_stiffness_gap

_VISC = 0.05  # normal damping ratio, as TYPE7


def _closest_points_on_segments(p1, q1, p2, q2):
    """Vectorized exact closest points between segments [p1,q1] and
    [p2,q2] (Ericson §5.1.9). All args (n,3). Returns (s, t, cA, cB):
    parameters in [0,1] and the closest points on each segment."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = np.einsum("nk,nk->n", d1, d1)
    e = np.einsum("nk,nk->n", d2, d2)
    f = np.einsum("nk,nk->n", d2, r)
    c = np.einsum("nk,nk->n", d1, r)
    b = np.einsum("nk,nk->n", d1, d2)
    denom = a * e - b * b

    # general (non-parallel) case; parallel edges pick s = 0 and let the
    # t-clamp below find the right point on the other edge
    s = np.where(denom > EM20,
                 np.clip((b * f - c * e) / np.maximum(denom, EM20), 0., 1.),
                 0.0)
    t = (b * s + f) / np.maximum(e, EM20)

    # clamp t, then recompute s for the clamped t (still exact)
    tlo = t < 0.0
    thi = t > 1.0
    t = np.clip(t, 0.0, 1.0)
    s = np.where(tlo, np.clip(-c / np.maximum(a, EM20), 0.0, 1.0), s)
    s = np.where(thi, np.clip((b - c) / np.maximum(a, EM20), 0.0, 1.0), s)

    cA = p1 + s[:, None] * d1
    cB = p2 + t[:, None] * d2
    return s, t, cA, cB


class ContactType11:
    """One /INTER/TYPE11 interface, engine-side."""

    def __init__(self, itf, model: Model, log):
        self.itf = itf
        self.model = model

        def _line(lid, side):
            ln = model.lines[lid]
            if ln.segments is None or len(ln.segments) == 0:
                log.warning(f"/INTER/TYPE11/{itf.id}: {side} line {lid} is "
                            f"empty — interface inactive", "CONTACT INIT")
                return (np.zeros((0, 2), dtype=np.int64),
                        np.zeros(0, dtype="<U8"), np.zeros(0, dtype=np.int64))
            return ln.segments, ln.seg_gtype, ln.seg_elem

        self.es, self.es_gtype, self.es_elem = _line(itf.line_id1,
                                                     "secondary")
        self.em, self.em_gtype, self.em_elem = _line(itf.line_id2, "main")

        # ---- per-edge stiffness and gap (i11sti3) --------------------------
        scale = itf.stfac if itf.istf != 1 else 1.0
        self.Ks, gs = edge_stiffness_gap(model, self.es, self.es_gtype,
                                         self.es_elem, scale)
        self.Km, gm = edge_stiffness_gap(model, self.em, self.em_gtype,
                                         self.em_elem, scale)

        # ---- gap (same default policy as TYPE7) ----------------------------
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
        self.fric = itf.fric

        # ---- interface time step bound -------------------------------------
        self.dt_bound = self._compute_dt_bound(model.mass)

        # ---- deletion bookkeeping ------------------------------------------
        self.deletable = (tracking.any_deletable(model, self.es_gtype)
                          or tracking.any_deletable(model, self.em_gtype))
        self.es_alive = np.ones(len(self.es), dtype=bool)
        self.em_alive = np.ones(len(self.em), dtype=bool)

        self.pairs_s = np.zeros(0, dtype=np.int64)   # secondary edge rows
        self.pairs_m = np.zeros(0, dtype=np.int64)   # main edge rows
        self._last_refresh = -10**9
        self.refresh = 20

    # ------------------------------------------------------------------
    def _compute_dt_bound(self, mass) -> float:
        """Worst node-on-spring bound over both edge sets (see TYPE7)."""
        if len(self.es) == 0 or len(self.em) == 0:
            return np.inf
        itf = self.itf
        K_s = combine_stiffness(itf.istf, itf.stfac,
                                np.full(len(self.es), self.Km.max()),
                                self.Ks)
        dt_s = np.sqrt(2.0 * mass[self.es].min(axis=1)
                       / np.maximum(K_s, EM20)).min()
        K_m = combine_stiffness(itf.istf, itf.stfac, self.Km,
                                np.full(len(self.em),
                                        self.Ks.max() if len(self.Ks)
                                        else 0.0))
        dt_m = np.sqrt(2.0 * mass[self.em].min(axis=1)
                       / np.maximum(K_m, EM20)).min()
        return float(min(dt_s, dt_m))

    # ------------------------------------------------------------------
    def _broad_phase(self, x, v, dt):
        """Voxel candidate search over edge bounding boxes (i11buce)."""
        es = self.es[self.es_alive]
        em = self.em[self.em_alive]
        rows_s = np.where(self.es_alive)[0]
        rows_m = np.where(self.em_alive)[0]
        if len(es) == 0 or len(em) == 0:
            self.pairs_s = np.zeros(0, dtype=np.int64)
            self.pairs_m = np.zeros(0, dtype=np.int64)
            return
        margin = self.gap_bound + 2.0 * self.refresh * dt * \
            (np.abs(v).max() if len(v) else 0.0)

        def boxes(edges, infl):
            xe = x[edges]                       # (n, 2, 3)
            return xe.min(axis=1) - infl, xe.max(axis=1) + infl

        lo_m, hi_m = boxes(em, margin)          # inflate the main side only
        lo_s, hi_s = boxes(es, 0.0)
        h = max(float((hi_m - lo_m).max()), float((hi_s - lo_s).max()), EM20)
        origin = np.minimum(lo_m.min(axis=0), lo_s.min(axis=0))

        def bins(lo, hi):
            ilo = np.floor((lo - origin) / h).astype(np.int64)
            ihi = np.floor((hi - origin) / h).astype(np.int64)
            return ilo, ihi

        ilo_m, ihi_m = bins(lo_m, hi_m)
        ilo_s, ihi_s = bins(lo_s, hi_s)
        dims = np.maximum(ihi_m.max(axis=0), ihi_s.max(axis=0)) + 2

        def key(ijk):
            return (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]

        def bin_boxes(ilo, ihi):
            ids, keys = [], []
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        ijk = ilo + np.array([dx, dy, dz])
                        inside = np.all(ijk <= ihi, axis=1)
                        ids.append(np.where(inside)[0])
                        keys.append(key(ijk[inside]))
            return np.concatenate(ids), np.concatenate(keys)

        ids_s, keys_s = bin_boxes(ilo_s, ihi_s)
        ids_m, keys_m = bin_boxes(ilo_m, ihi_m)
        ii, jj = _expand_matches(keys_s, keys_m)
        ps = rows_s[ids_s[ii]]
        pm = rows_m[ids_m[jj]]
        if len(ps):
            pair_key = ps * len(self.em) + pm
            _, first = np.unique(pair_key, return_index=True)
            ps, pm = ps[first], pm[first]
            # edges sharing a node never contact each other (adjacent
            # edges of one shell mesh, and self-exclusion within one line)
            keep = np.ones(len(ps), dtype=bool)
            for a in range(2):
                for b in range(2):
                    keep &= self.es[ps, a] != self.em[pm, b]
            ps, pm = ps[keep], pm[keep]
        self.pairs_s = ps
        self.pairs_m = pm

    # ------------------------------------------------------------------
    def forces(self, x, v, mass, dt, fcont, cycle):
        """Penalty forces for one cycle, scattered into ``fcont``.
        Returns (contact_work_increment, dt_interface) — the same
        contract as ContactType7.forces (the Engine books the exact
        midstep contact energy from ``fcont``, see engine.py)."""
        if len(self.es) == 0 or len(self.em) == 0:
            return 0.0, np.inf

        if self.deletable:
            self.es_alive = tracking.alive_segment_mask(
                self.model, self.es_gtype, self.es_elem)
            self.em_alive = tracking.alive_segment_mask(
                self.model, self.em_gtype, self.em_elem)

        if cycle - self._last_refresh >= self.refresh:
            self._broad_phase(x, v, dt)
            self._last_refresh = cycle
        if len(self.pairs_s) == 0:
            return 0.0, self.dt_bound

        live = self.es_alive[self.pairs_s] & self.em_alive[self.pairs_m]
        ps = self.pairs_s[live]
        pm = self.pairs_m[live]
        if len(ps) == 0:
            return 0.0, self.dt_bound
        ea = self.es[ps]                                  # (np, 2)
        eb = self.em[pm]

        # ---- narrow phase: exact closest points (i11dst3) ------------------
        s, t, cA, cB = _closest_points_on_segments(
            x[ea[:, 0]], x[ea[:, 1]], x[eb[:, 0]], x[eb[:, 1]])
        dvec = cA - cB
        d = np.linalg.norm(dvec, axis=1)

        if self.itf.igap == 1:
            gap = np.clip(self.gap_s[ps] + self.gap_m[pm],
                          self.gap_min, self.gap_max)
        else:
            gap = np.full(len(ps), self.gap_const)

        # ---- interface dt: NEAR-pair stiffness accumulation ----------------
        # springs that may close within the next cycles enter the nodal
        # bound with their stiffnesses SUMMED on shared nodes — see the
        # matching block in inter_type7.py for the full reasoning
        vmax = float(np.abs(v).max()) if len(v) else 0.0
        near = d < gap + np.maximum(gap, 2.0 * vmax * dt)
        if not np.any(near):
            return 0.0, self.dt_bound
        ea, eb = ea[near], eb[near]
        s, t = s[near], t[near]
        d, gap, dvec = d[near], gap[near], dvec[near]
        ps, pm = ps[near], pm[near]

        K = combine_stiffness(self.itf.istf, self.itf.stfac,
                              self.Km[pm], self.Ks[ps])
        Knode = np.zeros(len(fcont))
        for cols in (ea, eb):
            np.add.at(Knode, cols[:, 0], K)
            np.add.at(Knode, cols[:, 1], K)
        loaded = Knode > 0.0
        dt_int = min(self.dt_bound, float(
            np.sqrt(2.0 * mass[loaded] / Knode[loaded]).min()))

        pen = gap - d
        active = pen > 0.0
        if not np.any(active):
            return 0.0, dt_int

        ea, eb = ea[active], eb[active]
        K = K[active]                          # per-pair stiffness (Istf)
        s, t = s[active], t[active]
        gap = gap[active]
        pen = pen[active]
        d = np.maximum(d[active], EM20)
        nvec = dvec[active] / d[:, None]      # pushes the secondary edge out

        # relative velocity of the two closest points
        vA = (1 - s)[:, None] * v[ea[:, 0]] + s[:, None] * v[ea[:, 1]]
        vB = (1 - t)[:, None] * v[eb[:, 0]] + t[:, None] * v[eb[:, 1]]
        vrel = vA - vB
        vn = np.einsum("nb,nb->n", vrel, nvec)

        # normal force: spring + damper (mass of the lighter secondary end
        # node sizes the damper, as the node mass does in TYPE7)
        m_ref = mass[ea].min(axis=1)
        C = 2.0 * _VISC * np.sqrt(K * m_ref)
        Fn = K * pen - C * np.minimum(vn, 0.0)
        Fvec = Fn[:, None] * nvec

        # Coulomb friction, regularized around zero slip
        if self.fric > 0.0:
            gap_ref = float(np.mean(gap))
            vt = vrel - vn[:, None] * nvec
            vt_mag = np.linalg.norm(vt, axis=1)
            Ft = self.fric * Fn * vt_mag / (
                vt_mag + 1e-3 * gap_ref / max(dt, EM20))
            Fvec -= (Ft / np.maximum(vt_mag, EM20))[:, None] * vt

        # scatter with the closest-point parameters: +F on the secondary
        # edge ends, -F on the main edge ends (collinear equal/opposite
        # forces: linear AND angular momentum conserved)
        np.add.at(fcont, ea[:, 0], (1 - s)[:, None] * Fvec)
        np.add.at(fcont, ea[:, 1], s[:, None] * Fvec)
        np.add.at(fcont, eb[:, 0], -(1 - t)[:, None] * Fvec)
        np.add.at(fcont, eb[:, 1], -t[:, None] * Fvec)

        wrk = float(np.einsum("nb,nb->", Fvec, vrel)) * dt
        return -wrk, dt_int
