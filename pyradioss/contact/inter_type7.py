"""
/INTER/TYPE7 — penalty node-to-surface contact.

Fortran origin: ``engine/source/interfaces/inter3d/`` — the cycle path is

    i7main_tri.F  candidate search bookkeeping (re-sort when needed)
    i7buce.F      bucket (voxel) search: nodes near segments
    i7dst3.F      exact node-segment distance/projection
    i7for3.F      penalty force + friction, scatter to nodes

Algorithm ported here (with the simplifications listed at the end):

1. **Broad phase** (every ``refresh`` cycles): candidate (node, segment)
   pairs are those whose bounding boxes, inflated by the gap plus a
   travel margin, overlap. The margin is sized from the current maximum
   nodal speed so no pair can penetrate undetected between refreshes —
   the same safety logic as the original's bucket sort criterion.

2. **Narrow phase** (every cycle): each candidate quad segment is split
   into two triangles; the closest point on each triangle to the node is
   found exactly (barycentric clamping). Penetration is

       p = gap - d,     d = |x_node - x_closest|

3. **Penalty force** (i7for3): if p > 0, a spring pushes the node back
   along the closest-point direction, plus a small viscous damper on the
   normal relative velocity (stabilizes the bounce):

       Fn = K * p + C * vn_rel,    C = 2 * visc * sqrt(K * m_node)

   The reaction is distributed to the segment's corner nodes with the
   barycentric weights of the closest point (equal and opposite total
   force — momentum is conserved exactly).

4. **Coulomb friction**: the tangential relative velocity direction gets
   a force min(mu * Fn, ...) opposing it (kinetic friction, smoothly
   regularized near zero slip velocity).

Port simplifications vs the full TYPE7 (roadmap M4): constant penalty
stiffness (the original stiffens as p -> gap, guaranteeing non-crossing);
no Igap variants; no self-impact; secondary side is a node group only.
The penalty stiffness enters the interface time step as
dt_i = sqrt(2 m_min / K) — included in the Engine's dt minimum, as in the
original.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20
from ..model.model import Model

_VISC = 0.05  # normal damping ratio (Radioss VIS_S default 5%)


def _closest_point_on_triangle(p, a, b, c):
    """Vectorized exact closest point on triangles (Ericson, 'Real-Time
    Collision Detection' §5.1.5). All args (n,3). Returns (point, wa, wb, wc)
    barycentric weights."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("nk,nk->n", ab, ap)
    d2 = np.einsum("nk,nk->n", ac, ap)
    bp = p - b
    d3 = np.einsum("nk,nk->n", ab, bp)
    d4 = np.einsum("nk,nk->n", ac, bp)
    cp = p - c
    d5 = np.einsum("nk,nk->n", ab, cp)
    d6 = np.einsum("nk,nk->n", ac, cp)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    # start from the inside-face solution and overwrite region by region
    denom = np.maximum(va + vb + vc, EM20)
    v = vb / denom
    w = vc / denom

    # edge AB region
    onAB = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1 / np.maximum(d1 - d3, EM20)
    v = np.where(onAB, t, v)
    w = np.where(onAB, 0.0, w)
    # edge AC region
    onAC = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = d2 / np.maximum(d2 - d6, EM20)
    v = np.where(onAC, 0.0, v)
    w = np.where(onAC, t, w)
    # edge BC region
    onBC = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
    t = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), EM20)
    v = np.where(onBC, 1.0 - t, v)
    w = np.where(onBC, t, w)
    # vertex regions
    atA = (d1 <= 0) & (d2 <= 0)
    v = np.where(atA, 0.0, v)
    w = np.where(atA, 0.0, w)
    atB = (d3 >= 0) & (d4 <= d3)
    v = np.where(atB, 1.0, v)
    w = np.where(atB, 0.0, w)
    atC = (d6 >= 0) & (d5 <= d6)
    v = np.where(atC, 0.0, v)
    w = np.where(atC, 1.0, w)

    u = 1.0 - v - w
    point = u[:, None] * a + v[:, None] * b + w[:, None] * c
    return point, u, v, w


class ContactType7:
    """One /INTER/TYPE7 interface, engine-side."""

    def __init__(self, itf, model: Model, log):
        self.itf = itf
        self.nodes = model.node_groups[itf.grnod_id].node_idx
        self.segs = model.surfaces[itf.surf_id].segments
        if self.segs is None or len(self.segs) == 0:
            log.warning(f"/INTER/TYPE7/{itf.id}: main surface is empty — "
                        f"interface inactive", "CONTACT INIT")
            self.segs = np.zeros((0, 4), dtype=np.int64)

        # --- penalty stiffness (i7sti3 flavour, simplified) -------------
        # K ~ Stfac * E_ref * L_seg : dimensionally a spring per unit
        # penetration; E_ref = stiffest material in the model (safe upper
        # bound; the original uses the main segment's own material).
        E_ref = max((m.E for m in model.materials.values()), default=1.0)
        xs = model.x0[self.segs] if len(self.segs) else np.zeros((0, 4, 3))
        if len(self.segs):
            d1 = xs[:, 2] - xs[:, 0]
            d2 = xs[:, 3] - xs[:, 1]
            self.seg_area = 0.5 * np.linalg.norm(np.cross(d1, d2), axis=1)
            lc = float(np.sqrt(self.seg_area.mean()))
        else:
            self.seg_area = np.zeros(0)
            lc = 1.0
        self.K = itf.stfac * E_ref * lc

        # --- contact gap -------------------------------------------------
        # gap = 0 in the deck -> automatic: a few % of the segment size
        # (the original derives it from shell thickness / brick size).
        self.gap = itf.gap if itf.gap > 0 else 0.02 * lc
        self.fric = itf.fric

        # broad-phase bookkeeping
        self.pairs_node = np.zeros(0, dtype=np.int64)   # candidate node rows
        self.pairs_seg = np.zeros(0, dtype=np.int64)    # candidate seg rows
        self._last_refresh = -10**9
        self.refresh = 20                                # cycles

    # ------------------------------------------------------------------
    def _broad_phase(self, x: np.ndarray, v: np.ndarray, dt: float):
        """Candidate pairs by inflated bounding-box overlap (i7buce)."""
        margin = self.gap + 2.0 * self.refresh * dt * \
            (np.abs(v).max() if len(v) else 0.0)
        xs = x[self.segs]                                # (nseg, 4, 3)
        lo = xs.min(axis=1) - margin                     # (nseg, 3)
        hi = xs.max(axis=1) + margin
        xn = x[self.nodes]                               # (nn, 3)
        # (nn, nseg) box test — fine for the model sizes this port targets;
        # the original's voxel sort is the M4 roadmap replacement.
        inside = np.all((xn[:, None, :] >= lo[None, :, :])
                        & (xn[:, None, :] <= hi[None, :, :]), axis=2)
        # a secondary node that is also a corner of the segment is never
        # a candidate against it (self-contact exclusion)
        for k in range(4):
            inside &= self.nodes[:, None] != self.segs[None, :, k]
        ii, jj = np.where(inside)
        self.pairs_node, self.pairs_seg = ii, jj

    # ------------------------------------------------------------------
    def forces(self, x, v, mass, dt, fcont, cycle):
        """Penalty forces for one cycle, scattered into ``fcont``.
        Returns (contact_work_increment, dt_interface)."""
        if len(self.segs) == 0 or len(self.nodes) == 0:
            return 0.0, np.inf
        if cycle - self._last_refresh >= self.refresh:
            self._broad_phase(x, v, dt)
            self._last_refresh = cycle
        if len(self.pairs_node) == 0:
            return 0.0, self._dt_interface(mass)

        ni = self.nodes[self.pairs_node]                 # global node idx
        seg = self.segs[self.pairs_seg]                  # (np, 4)
        p = x[ni]

        best_d = np.full(len(ni), np.inf)
        best_pt = np.zeros((len(ni), 3))
        best_w = np.zeros((len(ni), 4))
        # quad = triangles (0,1,2) and (0,2,3)
        for tri, cols in (((0, 1, 2), (0, 1, 2)), ((0, 2, 3), (0, 2, 3))):
            a, b, c = (x[seg[:, tri[0]]], x[seg[:, tri[1]]],
                       x[seg[:, tri[2]]])
            pt, u, vv, w = _closest_point_on_triangle(p, a, b, c)
            d = np.linalg.norm(p - pt, axis=1)
            better = d < best_d
            best_d = np.where(better, d, best_d)
            best_pt[better] = pt[better]
            wq = np.zeros((len(ni), 4))
            wq[:, cols[0]], wq[:, cols[1]], wq[:, cols[2]] = u, vv, w
            best_w[better] = wq[better]

        pen = self.gap - best_d
        active = pen > 0.0
        if not np.any(active):
            return 0.0, self._dt_interface(mass)

        ni = ni[active]
        seg = seg[active]
        pen = pen[active]
        d = np.maximum(best_d[active], EM20)
        nvec = (x[ni] - best_pt[active]) / d[:, None]    # push-out direction
        wseg = best_w[active]

        # relative velocity node vs interpolated segment point
        vseg = np.einsum("nk,nkb->nb", wseg, v[seg])
        vrel = v[ni] - vseg
        vn = np.einsum("nb,nb->n", vrel, nvec)

        # normal force: spring + damper (only damp approaching motion)
        C = 2.0 * _VISC * np.sqrt(self.K * mass[ni])
        Fn = self.K * pen - C * np.minimum(vn, 0.0)
        Fvec = Fn[:, None] * nvec

        # Coulomb friction, regularized around zero slip
        if self.fric > 0.0:
            vt = vrel - vn[:, None] * nvec
            vt_mag = np.linalg.norm(vt, axis=1)
            Ft = self.fric * Fn * vt_mag / (vt_mag + 1e-3 * self.gap / max(dt, EM20))
            Fvec -= (Ft / np.maximum(vt_mag, EM20))[:, None] * vt

        # scatter: action on the node, exact opposite reaction on the
        # segment corners (momentum conservation)
        np.add.at(fcont, ni, Fvec)
        for k in range(4):
            np.add.at(fcont, seg[:, k], -wseg[:, k, None] * Fvec)

        # contact work this cycle (stored elastic + dissipated), for the
        # energy balance
        wrk = float(np.einsum("nb,nb->", Fvec, vrel)) * dt
        return -wrk, self._dt_interface(mass)

    # ------------------------------------------------------------------
    def _dt_interface(self, mass) -> float:
        """Penalty springs add stiffness: their stability limit is
        dt = sqrt(2 m_min / K) (original: i7 stiffness in the nodal dt)."""
        m_min = float(mass[self.nodes].min()) if len(self.nodes) else np.inf
        return np.sqrt(2.0 * m_min / max(self.K, EM20))
