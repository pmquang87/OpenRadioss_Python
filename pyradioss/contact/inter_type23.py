"""
/INTER/TYPE23 — Mortar segment-to-segment penalty contact interface.

Fortran origin: ``engine/source/interfaces/int23/``
  - ``i23mainf.F``: interface driver, time gating, and candidate dispatch
  - ``i23dst3.F``: 4-triangle centroid subdivision of master quad segments,
    smoothed facet normals, exact projection, and variationally consistent
    mortar shape function distribution H1..H4
  - ``i23for3.F``: penalty stiffness, normal contact force, viscous damping,
    Coulomb stick-slip friction, and energy accumulation
  - ``starter/source/interfaces/int23/hm_read_inter_type23.F``: input reader

Mortar Formulation:
-------------------
Standard node-to-segment interfaces can suffer from artificial stress
concentrations and non-uniform contact pressures on non-matching meshes.
TYPE23 resolves this via a variationally consistent mortar projection:
1. Each master quad segment is subdivided into 4 triangles by connecting the
   4 corner nodes to the facet centroid X0 = 0.25 * (X1 + X2 + X3 + X4).
2. For each triangle k in 1..4, the slave node is projected onto the triangle
   plane, determining barycentric coordinates (LAk, LBk, LCk).
3. The penetration and facet normal are computed for each sub-triangle.
4. Mortar shape functions H1..H4 are evaluated by integrating the centroid
   subdivision weights back to the 4 corner nodes:
       H0 = 0.25 * sum_{k=1..4} (P_k * LA_k)
       H1 = H0 + P1*LB1 + P4*LC4
       H2 = H0 + P2*LB2 + P1*LC1
       H3 = H0 + P3*LB3 + P2*LC2
       H4 = H0 + P4*LB4 + P3*LC3
   Normalized so sum(H_i) = 1.0 (partition of unity).
5. The contact force on the slave node is F_s.
   The master corner nodes receive reactive forces F_{m,i} = -H_i * F_s.
   This guarantees exact momentum conservation and smooth, uniform interface
   pressure transfer across non-matching meshes without spurious nodal peaks.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20, EM30
from ..common.fastmath import cross3, norm3
from ..model.model import Model
from .stiffness import combine_stiffness, segment_stiffness_gap, node_stiffness_gap


def _closest_point_on_triangle(p, a, b, c):
    """
    Vectorized exact closest point on triangle (a, b, c) from point p.
    Returns (closest_pt, la, lb, lc) where closest_pt = la*a + lb*b + lc*c,
    with la + lb + lc = 1.0 and la, lb, lc in [0, 1].
    """
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

    denom = np.maximum(va + vb + vc, EM20)
    v = vb / denom
    w = vc / denom

    # edge AB region
    onAB = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    t_ab = d1 / np.maximum(d1 - d3, EM20)
    v = np.where(onAB, t_ab, v)
    w = np.where(onAB, 0.0, w)

    # edge AC region
    onAC = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    t_ac = d2 / np.maximum(d2 - d6, EM20)
    v = np.where(onAC, 0.0, v)
    w = np.where(onAC, t_ac, w)

    # edge BC region
    onBC = (va <= 0.0) & (d4 - d3 >= 0.0) & (d5 - d6 >= 0.0)
    t_bc = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), EM20)
    v = np.where(onBC, 1.0 - t_bc, v)
    w = np.where(onBC, t_bc, w)

    # vertex regions
    atA = (d1 <= 0.0) & (d2 <= 0.0)
    v = np.where(atA, 0.0, v)
    w = np.where(atA, 0.0, w)

    atB = (d3 >= 0.0) & (d4 <= d3)
    v = np.where(atB, 1.0, v)
    w = np.where(atB, 0.0, w)

    atC = (d6 >= 0.0) & (d5 <= d6)
    v = np.where(atC, 0.0, v)
    w = np.where(atC, 1.0, w)

    u = 1.0 - v - w
    # Clamp weights
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    w = np.clip(w, 0.0, 1.0)
    sw = np.maximum(u + v + w, EM20)
    u /= sw
    v /= sw
    w /= sw

    pt = u[:, None] * a + v[:, None] * b + w[:, None] * c
    return pt, u, v, w


class ContactType23:
    """One /INTER/TYPE23 Mortar segment-to-segment interface, engine-side."""

    def __init__(self, itf, model: Model, log):
        self.itf = itf
        self.model = model
        self.log = log

        # Master surface (surf_id or mainentityids)
        master_surf_id = getattr(itf, "surf_id", 0) or getattr(itf, "main_id", 0)
        surf_m = model.surfaces.get(master_surf_id)

        # Secondary surface (surf_id1 or secondaryentityids)
        slave_surf_id = getattr(itf, "surf_id1", 0) or getattr(itf, "sec_id", 0)
        surf_s = model.surfaces.get(slave_surf_id) if slave_surf_id > 0 else None

        if surf_m is None or surf_m.segments is None or len(surf_m.segments) == 0:
            log.warning(f"/INTER/TYPE23/{itf.id}: main surface {master_surf_id} empty or missing", "CONTACT INIT")
            self._init_empty()
            return

        segs_m = np.asarray(surf_m.segments, dtype=np.int64)
        if segs_m.ndim == 2 and segs_m.shape[1] == 3:
            # Triangles: duplicate 3rd node to indicate tri (IX3 == IX4 in Fortran)
            segs_m = np.column_stack([segs_m, segs_m[:, 2]])
        elif segs_m.ndim == 2 and segs_m.shape[1] == 4:
            segs_m = segs_m.copy()
            neg = segs_m[:, 3] < 0
            segs_m[neg, 3] = segs_m[neg, 2]

        self.segs = segs_m
        self.seg_gtype = surf_m.seg_gtype
        self.seg_elem = surf_m.seg_elem

        # Secondary nodes
        if surf_s is not None and surf_s.segments is not None and len(surf_s.segments) > 0:
            self.nodes = np.unique(np.asarray(surf_s.segments, dtype=np.int64))
        elif getattr(itf, "grnod_id", 0) > 0:
            grn = model.node_groups.get(itf.grnod_id)
            if grn is not None:
                if getattr(grn, "node_idx", None) is not None and len(grn.node_idx) > 0:
                    self.nodes = np.asarray(grn.node_idx, dtype=np.int64)
                else:
                    self.nodes = np.asarray(getattr(grn, "nodes", []), dtype=np.int64)
            else:
                self.nodes = np.unique(self.segs)
        else:
            # Self-contact: secondary nodes are the nodes of the master surface
            self.nodes = np.unique(self.segs)

        self.nodes.sort()

        # Interface parameters
        self.istf = int(getattr(itf, "istf", 0))
        self.stfac = float(getattr(itf, "stfac", 1.0) or 1.0)
        self.stmin = float(getattr(itf, "stmin", 0.0) or 0.0)
        self.stmax = float(getattr(itf, "stmax", 1e30) or 1e30)
        self.fric = float(getattr(itf, "fric", 0.0) or 0.0)
        self.gap_min = float(getattr(itf, "gap", getattr(itf, "gap_min", 0.0)) or 0.0)
        self.gap_max = float(getattr(itf, "gap_max", 1e30) or 1e30)
        if self.gap_max <= 0.0:
            self.gap_max = 1e30
        self.fscale_gap = float(getattr(itf, "fscale_gap", 1.0) or 1.0)
        self.igap = int(getattr(itf, "igap", 0))
        self.inactiv = int(getattr(itf, "inactiv", 0))
        self.visc = float(getattr(itf, "viss", getattr(itf, "stiff_dc", 0.05)) or 0.05)
        if self.visc <= 0.0:
            self.visc = 0.05
        self.tstart = float(getattr(itf, "tstart", 0.0) or 0.0)
        self.tstop = float(getattr(itf, "tstop", 1e30) or 1e30)

        # Stiffness and gap evaluation
        scale = self.stfac if self.istf != 1 else 1.0
        self.Km, self.gap_m = segment_stiffness_gap(
            model, self.segs, self.seg_gtype, self.seg_elem, scale, fscale_gap=self.fscale_gap
        )
        Ks_all, gs_all = node_stiffness_gap(model, scale, fscale_gap=self.fscale_gap)
        if len(self.nodes) and len(Ks_all):
            valid_k = (self.nodes >= 0) & (self.nodes < len(Ks_all))
            self.Ks = np.zeros(len(self.nodes), dtype=float)
            self.Ks[valid_k] = Ks_all[self.nodes[valid_k]]
            self.gap_s = np.zeros(len(self.nodes), dtype=float)
            self.gap_s[valid_k] = gs_all[self.nodes[valid_k]]
        else:
            self.Ks = np.zeros(len(self.nodes), dtype=float)
            self.gap_s = np.zeros(len(self.nodes), dtype=float)

        self.gap_const = max(self.gap_min, 1e-4) if self.gap_min > 0.0 else 0.001
        self.gap_bound = max(
            float(self.gap_m.max() if len(self.gap_m) else self.gap_const),
            float(self.gap_s.max() if len(self.gap_s) else self.gap_const),
            self.gap_const,
        )

        # Tangential force history storage for friction (stick-slip)
        self.ft_hist = {}  # key: (node_id, seg_idx) -> np.ndarray (3,)

        self.dt_bound = self._compute_dt_bound(model.mass)

        # Broad phase bookkeeping
        self.pairs_node = np.zeros(0, dtype=np.int64)
        self.pairs_seg = np.zeros(0, dtype=np.int64)
        self._last_refresh = -10**9
        self.refresh = 20

    def _init_empty(self):
        self.segs = np.zeros((0, 4), dtype=np.int64)
        self.nodes = np.zeros(0, dtype=np.int64)
        self.Km = np.zeros(0, dtype=float)
        self.Ks = np.zeros(0, dtype=float)
        self.gap_m = np.zeros(0, dtype=float)
        self.gap_s = np.zeros(0, dtype=float)
        self.gap_const = 0.0
        self.gap_bound = 0.0
        self.gap_min = 0.0
        self.gap_max = 1e30
        self.stmin = 0.0
        self.stmax = 1e30
        self.fric = 0.0
        self.visc = 0.05
        self.tstart = 0.0
        self.tstop = 1e30
        self.ft_hist = {}
        self.dt_bound = np.inf
        self.pairs_node = np.zeros(0, dtype=np.int64)
        self.pairs_seg = np.zeros(0, dtype=np.int64)
        self._last_refresh = -10**9
        self.refresh = 20

    def _compute_dt_bound(self, mass: np.ndarray) -> float:
        if len(self.segs) == 0 or len(self.nodes) == 0:
            return np.inf
        valid_nodes = self.nodes[(self.nodes >= 0) & (self.nodes < len(mass))]
        if len(valid_nodes) == 0:
            return np.inf
        Km_max = np.full(len(valid_nodes), self.Km.max() if len(self.Km) else 0.0)
        loc = np.searchsorted(self.nodes, valid_nodes)
        Ks_sub = self.Ks[loc]
        K_sec = combine_stiffness(self.istf, self.stfac, Km_max, Ks_sub)
        if self.stmin > 0.0:
            K_sec = np.maximum(K_sec, self.stmin)
        if self.stmax > 0.0:
            K_sec = np.minimum(K_sec, self.stmax)
        m_sec = mass[valid_nodes]
        pos = (m_sec > 0.0) & (K_sec > 0.0)
        dt_sec = np.sqrt(2.0 * m_sec[pos] / K_sec[pos]).min() if np.any(pos) else np.inf
        return float(dt_sec)

    def _broad_phase(self, x: np.ndarray, v: np.ndarray, dt: float):
        """Bounding box candidate search for mortar contact."""
        segs = self.segs
        nodes = self.nodes
        if len(segs) == 0 or len(nodes) == 0:
            self.pairs_node = np.zeros(0, dtype=np.int64)
            self.pairs_seg = np.zeros(0, dtype=np.int64)
            return

        vmax = float(np.abs(v).max()) if len(v) else 0.0
        margin = self.gap_bound * self.fscale_gap + 2.0 * self.refresh * dt * vmax

        xs = x[segs]  # (nseg, 4, 3)
        seg_min = xs.min(axis=1) - margin  # (nseg, 3)
        seg_max = xs.max(axis=1) + margin  # (nseg, 3)

        xn = x[nodes]  # (nnod, 3)

        # Vectorized box check
        in_x = (xn[:, 0, None] >= seg_min[:, 0]) & (xn[:, 0, None] <= seg_max[:, 0])
        in_y = (xn[:, 1, None] >= seg_min[:, 1]) & (xn[:, 1, None] <= seg_max[:, 1])
        in_z = (xn[:, 2, None] >= seg_min[:, 2]) & (xn[:, 2, None] <= seg_max[:, 2])
        cand = in_x & in_y & in_z

        nid_idx, srow = np.where(cand)
        if len(nid_idx) == 0:
            self.pairs_node = np.zeros(0, dtype=np.int64)
            self.pairs_seg = np.zeros(0, dtype=np.int64)
            return

        ni = nodes[nid_idx]

        # Self-exclusion: a node that belongs to the segment cannot impact it
        corner_0 = segs[srow, 0]
        corner_1 = segs[srow, 1]
        corner_2 = segs[srow, 2]
        corner_3 = segs[srow, 3]
        not_corner = (ni != corner_0) & (ni != corner_1) & (ni != corner_2) & (ni != corner_3)

        self.pairs_node = ni[not_corner]
        self.pairs_seg = srow[not_corner]

    def forces(
        self,
        x: np.ndarray,
        v: np.ndarray,
        mass: np.ndarray,
        dt: float,
        fcont: np.ndarray,
        cycle: int = 0,
        stifn: np.ndarray = None,
        t: float = None,
    ) -> tuple[float, float]:
        """
        Evaluate Mortar contact forces and assemble into fcont.
        Returns (econt_est, dt_bound).
        """
        if len(self.segs) == 0 or len(self.nodes) == 0 or dt <= 0.0:
            return 0.0, self.dt_bound

        if t is not None:
            if t < self.tstart or t > self.tstop:
                return 0.0, self.dt_bound

        # Refresh candidate list
        if cycle - self._last_refresh >= self.refresh:
            self._broad_phase(x, v, dt)
            self._last_refresh = cycle

        if len(self.pairs_node) == 0:
            return 0.0, self.dt_bound

        ni = self.pairs_node
        srow = self.pairs_seg
        seg = self.segs[srow]

        # Coordinates of corners
        x1 = x[seg[:, 0]]
        x2 = x[seg[:, 1]]
        x3 = x[seg[:, 2]]
        x4 = x[seg[:, 3]]
        xp = x[ni]

        is_quad = seg[:, 2] != seg[:, 3]
        x0 = np.where(is_quad[:, None], 0.25 * (x1 + x2 + x3 + x4), x3)

        # Gap calculation
        loc = np.searchsorted(self.nodes, ni)
        loc = np.clip(loc, 0, len(self.nodes) - 1)
        if self.igap in (1, 2, 3):
            gap = (self.gap_s[loc] + self.gap_m[srow]) * self.fscale_gap
            gap = np.clip(gap, self.gap_min, self.gap_max)
        else:
            gap = np.full(len(ni), max(self.gap_min, self.gap_const))

        gap2 = gap * gap

        # Triangle 1: (x0, x1, x2)
        pt1, la1, lb1, lc1 = _closest_point_on_triangle(xp, x0, x1, x2)
        n1_raw = cross3(x1 - x0, x2 - x0)
        norm_n1 = norm3(n1_raw)
        n1 = np.where((norm_n1 > EM30)[:, None], n1_raw / np.maximum(norm_n1, EM30)[:, None], np.array([0.0, 0.0, 1.0]))
        d1 = xp - pt1
        h1 = np.einsum("nk,nk->n", d1, n1)
        p1 = np.maximum(0.0, gap - np.abs(h1))
        # Lateral distance cutoff
        lat1 = np.sum(d1 * d1, axis=1) - h1 * h1
        p1 = np.where(lat1 < gap2, p1, 0.0)

        # Triangle 2: (x0, x2, x3)
        pt2, la2, lb2, lc2 = _closest_point_on_triangle(xp, x0, x2, x3)
        n2_raw = cross3(x2 - x0, x3 - x0)
        norm_n2 = norm3(n2_raw)
        n2 = np.where((norm_n2 > EM30)[:, None], n2_raw / np.maximum(norm_n2, EM30)[:, None], np.array([0.0, 0.0, 1.0]))
        d2 = xp - pt2
        h2 = np.einsum("nk,nk->n", d2, n2)
        p2 = np.maximum(0.0, gap - np.abs(h2))
        lat2 = np.sum(d2 * d2, axis=1) - h2 * h2
        p2 = np.where((lat2 < gap2) & is_quad, p2, 0.0)

        # Triangle 3: (x0, x3, x4)
        pt3, la3, lb3, lc3 = _closest_point_on_triangle(xp, x0, x3, x4)
        n3_raw = cross3(x3 - x0, x4 - x0)
        norm_n3 = norm3(n3_raw)
        n3 = np.where((norm_n3 > EM30)[:, None], n3_raw / np.maximum(norm_n3, EM30)[:, None], np.array([0.0, 0.0, 1.0]))
        d3 = xp - pt3
        h3 = np.einsum("nk,nk->n", d3, n3)
        p3 = np.maximum(0.0, gap - np.abs(h3))
        lat3 = np.sum(d3 * d3, axis=1) - h3 * h3
        p3 = np.where((lat3 < gap2) & is_quad, p3, 0.0)

        # Triangle 4: (x0, x4, x1)
        pt4, la4, lb4, lc4 = _closest_point_on_triangle(xp, x0, x4, x1)
        n4_raw = cross3(x4 - x0, x1 - x0)
        norm_n4 = norm3(n4_raw)
        n4 = np.where((norm_n4 > EM30)[:, None], n4_raw / np.maximum(norm_n4, EM30)[:, None], np.array([0.0, 0.0, 1.0]))
        d4 = xp - pt4
        h4 = np.einsum("nk,nk->n", d4, n4)
        p4 = np.maximum(0.0, gap - np.abs(h4))
        lat4 = np.sum(d4 * d4, axis=1) - h4 * h4
        p4 = np.where((lat4 < gap2) & is_quad, p4, 0.0)

        # Total penetration
        pene = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))

        # Stiffness
        K = combine_stiffness(self.istf, self.stfac, self.Km[srow], self.Ks[loc])
        if self.stmin > 0.0:
            K = np.maximum(K, self.stmin)
        if self.stmax > 0.0:
            K = np.minimum(K, self.stmax)

        # Time step accumulation
        dt_int = self.dt_bound
        near = pene > 0.0
        if np.any(near):
            n_nod = len(fcont)
            Knode = np.bincount(ni[near], weights=K[near], minlength=n_nod)
            for c_idx in range(4):
                Knode += np.bincount(seg[near, c_idx], weights=0.25 * K[near], minlength=n_nod)
            loaded = Knode > 0.0
            if np.any(loaded):
                m_loaded = np.maximum(mass[loaded], EM20)
                dt_int = min(self.dt_bound, float(np.sqrt(2.0 * m_loaded / Knode[loaded]).min()))
            if stifn is not None:
                stifn[loaded] += Knode[loaded]

        # Filter active penetrations
        active = pene > 0.0
        if not np.any(active):
            return 0.0, dt_int

        ni = ni[active]
        srow = srow[active]
        seg = seg[active]
        is_quad = is_quad[active]
        pene = pene[active]
        p1 = p1[active]
        p2 = p2[active]
        p3 = p3[active]
        p4 = p4[active]
        la1 = la1[active]
        lb1 = lb1[active]
        lc1 = lc1[active]
        la2 = la2[active]
        lb2 = lb2[active]
        lc2 = lc2[active]
        la3 = la3[active]
        lb3 = lb3[active]
        lc3 = lc3[active]
        la4 = la4[active]
        lb4 = lb4[active]
        lc4 = lc4[active]
        n1 = n1[active]
        n2 = n2[active]
        n3 = n3[active]
        n4 = n4[active]
        K = K[active]

        # Mortar shape functions H1..H4
        # Quad case (i23dst3.F lines 500-510):
        h0 = 0.25 * (p1 * la1 + p2 * la2 + p3 * la3 + p4 * la4)
        H1 = h0 + p1 * lb1 + p4 * lc4
        H2 = h0 + p2 * lb2 + p1 * lc1
        H3 = h0 + p3 * lb3 + p2 * lc2
        H4 = h0 + p4 * lb4 + p3 * lc3

        # Tri case: H1=lb1, H2=lc1, H3=la1, H4=0.0
        H1 = np.where(is_quad, H1, lb1)
        H2 = np.where(is_quad, H2, lc1)
        H3 = np.where(is_quad, H3, la1)
        H4 = np.where(is_quad, H4, 0.0)

        H_sum = np.maximum(H1 + H2 + H3 + H4, EM20)
        H1 /= H_sum
        H2 /= H_sum
        H3 /= H_sum
        H4 /= H_sum

        # Weighted normal (i23dst3.F)
        N_weighted = (
            p1[:, None] * n1
            + p2[:, None] * n2
            + p3[:, None] * n3
            + p4[:, None] * n4
        )
        norm_N = norm3(N_weighted)
        facet_norm = np.where(
            (norm_N > EM20)[:, None],
            N_weighted / np.maximum(norm_N, EM20)[:, None],
            n1,
        )

        # Oriented normal direction (opposing slave penetration)
        xm = (
            H1[:, None] * x[seg[:, 0]]
            + H2[:, None] * x[seg[:, 1]]
            + H3[:, None] * x[seg[:, 2]]
            + H4[:, None] * x[seg[:, 3]]
        )
        dx_sm = x[ni] - xm
        side = np.einsum("nk,nk->n", dx_sm, facet_norm)
        normal = np.where((side >= 0.0)[:, None], facet_norm, -facet_norm)

        # Master velocity at contact point
        vm = (
            H1[:, None] * v[seg[:, 0]]
            + H2[:, None] * v[seg[:, 1]]
            + H3[:, None] * v[seg[:, 2]]
            + H4[:, None] * v[seg[:, 3]]
        )
        v_rel = v[ni] - vm
        vn = np.einsum("nk,nk->n", v_rel, normal)

        # Normal penalty force + damping (only damp approaching motion vn < 0)
        fn_scalar = K * pene
        c_damp = 2.0 * self.visc * np.sqrt(K * np.maximum(mass[ni], EM20))
        fn_total = fn_scalar - c_damp * np.minimum(vn, 0.0)

        fn_vec = fn_total[:, None] * normal

        # Friction (Coulomb stick-slip)
        vt_vec = v_rel - vn[:, None] * normal

        ft_vec = np.zeros_like(fn_vec)
        if self.fric > 0.0:
            mu = self.fric
            ft_limit = mu * np.abs(fn_scalar)

            ft_prev = np.zeros_like(fn_vec)
            for idx, (node_idx, s_idx) in enumerate(zip(ni, srow)):
                key = (int(node_idx), int(s_idx))
                if key in self.ft_hist:
                    ft_prev[idx] = self.ft_hist[key]

            ft_trial = ft_prev - K[:, None] * vt_vec * dt
            ft_trial -= np.einsum("nk,nk->n", ft_trial, normal)[:, None] * normal
            ft_trial_norm = norm3(ft_trial)

            scale = np.where(
                ft_trial_norm > ft_limit,
                ft_limit / np.maximum(ft_trial_norm, EM30),
                1.0,
            )
            ft_vec = ft_trial * scale[:, None]

            for idx, (node_idx, s_idx) in enumerate(zip(ni, srow)):
                key = (int(node_idx), int(s_idx))
                self.ft_hist[key] = ft_vec[idx]

        # Total force on slave node
        f_slave = fn_vec + ft_vec

        # Reactive forces on master corner nodes weighted by mortar shape functions H1..H4
        f_m1 = -H1[:, None] * f_slave
        f_m2 = -H2[:, None] * f_slave
        f_m3 = -H3[:, None] * f_slave
        f_m4 = -H4[:, None] * f_slave

        # Assemble into fcont using in-place scatter add
        np.add.at(fcont, ni, f_slave)
        np.add.at(fcont, seg[:, 0], f_m1)
        np.add.at(fcont, seg[:, 1], f_m2)
        np.add.at(fcont, seg[:, 2], f_m3)
        np.add.at(fcont, seg[:, 3], f_m4)

        # Estimate contact elastic energy
        econt_est = float(0.5 * np.sum(K * pene * pene))

        return econt_est, dt_int
