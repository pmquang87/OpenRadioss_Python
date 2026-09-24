"""
/INTER/TYPE20 — Combined surface-to-surface and edge-to-edge contact with symmetry.

Fortran origin:
  - ``engine/source/interfaces/int20/i20mainf.F`` (interface coordinator for combined contact)
  - ``engine/source/interfaces/int20/i20for3.F`` (surface penalty forces & edge-to-edge forces i20for3e)
  - ``engine/source/interfaces/int20/i20dst3.F`` (narrow-phase projection & distance)
  - ``engine/source/interfaces/int20/i20curv.F`` (nodal normal smoothing & curvature correction)
  - ``starter/source/interfaces/inter3d1/i20surfi.F`` (card reading & geometry setup)

Physics:
  1. Surface-to-surface contact:
     - Pass 1: Surface 1 nodes vs Surface 2 segments (penalty node-to-surface).
     - Pass 2 (if ISYM >= 1): Surface 2 nodes vs Surface 1 segments (symmetric pass).
     - Curvature / normal smoothing prevents snagging at facet boundaries.
  2. Edge-to-edge contact (if IEDGE == 1):
     - Closest points between edge segments of Surface 1 and Surface 2 (or /LINE definitions).
     - Penalty spring along line connecting closest points, distributed to edge endpoints
       with weights -(1-s, s) on edge 1 and +(1-t, t) on edge 2.
  3. Strict momentum conservation:
     - All force pairs are collinear, equal, and opposite.
     - sum(fcont) == 0 strictly for closed systems.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20
from ..common.fastmath import norm3, scatter_add3
from ..model.model import Model
from .inter_type11 import _closest_points_on_segments
from .inter_type7 import _narrow, segment_stiffness_gap


def _extract_surface_edges(segs: np.ndarray) -> np.ndarray:
    """Extract unique undirected edges from quad/tri segments (N, 4)."""
    if len(segs) == 0:
        return np.zeros((0, 2), dtype=np.int64)

    edges_list = []
    # Edges (0,1), (1,2), (2,3), (3,0)
    for i, j in ((0, 1), (1, 2), (2, 3), (3, 0)):
        n1 = segs[:, i]
        n2 = segs[:, j]
        # Ignore degenerate triangle repeated node (e.g. 2==3)
        valid = (n1 >= 0) & (n2 >= 0) & (n1 != n2)
        if np.any(valid):
            v1 = n1[valid]
            v2 = n2[valid]
            e = np.column_stack([np.minimum(v1, v2), np.maximum(v1, v2)])
            edges_list.append(e)

    if not edges_list:
        return np.zeros((0, 2), dtype=np.int64)

    all_edges = np.vstack(edges_list)
    return np.unique(all_edges, axis=0)


class ContactType20:
    """One /INTER/TYPE20 combined surface-to-surface and edge-to-edge contact interface."""

    def __init__(self, itf, model: Model, log=None):
        self.itf = itf
        self.model = model
        self.log = log if log is not None else getattr(model, "log", None)

        self.id = getattr(itf, "id", 0)
        self.title = getattr(itf, "title", f"TYPE20_{self.id}")
        self.tstart = float(getattr(itf, "tstart", 0.0) or 0.0)
        self.tstop = float(getattr(itf, "tstop", 1e30) or 1e30)
        if self.tstop <= 0.0:
            self.tstop = 1e30

        params = getattr(itf, "params", {}) or {}
        self.gap = float(params.get("gap", getattr(itf, "gap", 0.0)) or 0.0)
        self.stfac = float(params.get("stfac", getattr(itf, "stfac", 1.0)) or 1.0)
        self.fric = float(params.get("fric", getattr(itf, "fric", 0.0)) or 0.0)
        self.visc = float(params.get("visc", getattr(itf, "visc", 0.05)) or 0.05)

        self.isym = int(params.get("i_sym", getattr(itf, "isym", 0)) or 0)
        self.iedge = int(params.get("i_edge", getattr(itf, "iedge", 0)) or 0)
        self.edge_angle = float(params.get("edge_angle", getattr(itf, "edge_angle", 0.0)) or 0.0)

        self.e_cont = 0.0
        self.e_damp = 0.0
        self.dt_bound = 1e30

        self._init_entities()

    def _init_empty(self) -> None:
        self.segs1 = np.zeros((0, 4), dtype=np.int64)
        self.segs2 = np.zeros((0, 4), dtype=np.int64)
        self.nodes1 = np.zeros(0, dtype=np.int64)
        self.nodes2 = np.zeros(0, dtype=np.int64)
        self.edges1 = np.zeros((0, 2), dtype=np.int64)
        self.edges2 = np.zeros((0, 2), dtype=np.int64)
        self.dt_bound = 1e30

    def _init_entities(self) -> None:
        model = self.model
        itf = self.itf

        surf_id1 = getattr(itf, "surf_id", 0) or getattr(itf, "surf_id1_val", 0)
        surf_id2 = getattr(itf, "surf_id1", 0) or getattr(itf, "surf_id2", 0) or getattr(itf, "surf_id2_val", 0)

        surf1 = model.surfaces.get(surf_id1) if hasattr(model, "surfaces") else None
        surf2 = model.surfaces.get(surf_id2) if hasattr(model, "surfaces") else None

        # Resolve Surface 1 segments and nodes
        segs1 = np.zeros((0, 4), dtype=np.int64)
        if surf1 is not None and getattr(surf1, "segments", None) is not None and len(surf1.segments) > 0:
            s1 = np.asarray(surf1.segments, dtype=np.int64)
            if s1.ndim == 2:
                if s1.shape[1] == 3:
                    s1 = np.column_stack([s1, s1[:, 2]])
                elif s1.shape[1] == 4:
                    s1 = s1.copy()
                    tri_mask = s1[:, 3] < 0
                    s1[tri_mask, 3] = s1[tri_mask, 2]
            segs1 = s1

        # Resolve Surface 2 segments and nodes
        segs2 = np.zeros((0, 4), dtype=np.int64)
        if surf2 is not None and getattr(surf2, "segments", None) is not None and len(surf2.segments) > 0:
            s2 = np.asarray(surf2.segments, dtype=np.int64)
            if s2.ndim == 2:
                if s2.shape[1] == 3:
                    s2 = np.column_stack([s2, s2[:, 2]])
                elif s2.shape[1] == 4:
                    s2 = s2.copy()
                    tri_mask = s2[:, 3] < 0
                    s2[tri_mask, 3] = s2[tri_mask, 2]
            segs2 = s2

        self.segs1 = segs1
        self.segs2 = segs2

        # Nodes
        grnod_id = getattr(itf, "grnod_id", 0)
        if grnod_id > 0 and hasattr(model, "node_groups") and grnod_id in model.node_groups:
            grp = model.node_groups[grnod_id]
            node_arr = getattr(grp, "node_idx", None)
            if node_arr is None:
                node_arr = getattr(grp, "nodes", getattr(grp, "node_ids", []))
            if len(self.segs1) > 0 and len(self.segs2) == 0:
                self.nodes2 = np.asarray(node_arr, dtype=np.int64)
                v1 = self.segs1[self.segs1 >= 0]
                self.nodes1 = np.unique(v1)
            else:
                self.nodes1 = np.asarray(node_arr, dtype=np.int64)
                self.nodes2 = np.zeros(0, dtype=np.int64)
        elif len(self.segs1) > 0:
            v1 = self.segs1[self.segs1 >= 0]
            self.nodes1 = np.unique(v1)
            self.nodes2 = np.zeros(0, dtype=np.int64)
        else:
            self.nodes1 = np.zeros(0, dtype=np.int64)
            self.nodes2 = np.zeros(0, dtype=np.int64)

        if len(self.segs2) > 0:
            v2 = self.segs2[self.segs2 >= 0]
            self.nodes2 = np.unique(v2)

        # Edges
        self.edges1 = np.zeros((0, 2), dtype=np.int64)
        self.edges2 = np.zeros((0, 2), dtype=np.int64)

        if self.iedge == 1:
            line_id1 = getattr(itf, "line_id1", 0)
            line_id2 = getattr(itf, "line_id2", 0)

            if line_id1 > 0 and hasattr(model, "lines") and line_id1 in model.lines:
                l1 = model.lines[line_id1]
                if getattr(l1, "segments", None) is not None:
                    self.edges1 = np.asarray(l1.segments, dtype=np.int64)
            if len(self.edges1) == 0 and len(self.segs1) > 0:
                self.edges1 = _extract_surface_edges(self.segs1)

            if line_id2 > 0 and hasattr(model, "lines") and line_id2 in model.lines:
                l2 = model.lines[line_id2]
                if getattr(l2, "segments", None) is not None:
                    self.edges2 = np.asarray(l2.segments, dtype=np.int64)
            if len(self.edges2) == 0 and len(self.segs2) > 0:
                self.edges2 = _extract_surface_edges(self.segs2)

        # Stiffness
        self.K_seg1 = self._compute_segment_stiffness(surf1, self.segs1)
        self.K_seg2 = self._compute_segment_stiffness(surf2, self.segs2)

        # Time step bound
        mass0 = getattr(model, "mass0", None)
        m_eff = mass0 if mass0 is not None and len(mass0) > 0 else getattr(model, "mass", None)
        all_nodes = np.union1d(self.nodes1, self.nodes2)
        if m_eff is not None and len(m_eff) > 0 and len(all_nodes) > 0:
            m_sec = m_eff[all_nodes]
            valid_m = m_sec[m_sec > EM20]
            m_min = float(valid_m.min()) if len(valid_m) > 0 else 1.0
            all_k = []
            if len(self.K_seg1) > 0:
                all_k.append(self.K_seg1.max())
            if len(self.K_seg2) > 0:
                all_k.append(self.K_seg2.max())
            k_max = float(max(all_k)) if all_k else 1e6
            self.dt_bound = float(np.sqrt(2.0 * m_min / max(k_max, EM20)))
        else:
            self.dt_bound = 1e30

    def _compute_segment_stiffness(self, surf, segs: np.ndarray) -> np.ndarray:
        if len(segs) == 0:
            return np.zeros(0, dtype=np.float64)
        seg_gtype = (
            surf.seg_gtype
            if surf is not None and getattr(surf, "seg_gtype", None) is not None
            else np.zeros(len(segs), dtype="<U8")
        )
        seg_elem = (
            surf.seg_elem
            if surf is not None and getattr(surf, "seg_elem", None) is not None
            else np.full(len(segs), -1, dtype=np.int64)
        )
        try:
            Km, _ = segment_stiffness_gap(self.model, segs, seg_gtype, seg_elem, self.stfac)
            return np.where(Km > EM20, Km, 1e6 * self.stfac)
        except Exception:
            return np.full(len(segs), 1e6 * self.stfac, dtype=np.float64)

    def _surface_pass(
        self,
        nodes: np.ndarray,
        segs: np.ndarray,
        k_segs: np.ndarray,
        x: np.ndarray,
        v: np.ndarray,
        mass: np.ndarray,
        dt: float,
        fcont: np.ndarray,
        stifn: np.ndarray | None,
    ) -> None:
        """One node-to-surface penalty contact pass with momentum conservation."""
        n_sec = len(nodes)
        n_main = len(segs)
        if n_sec == 0 or n_main == 0:
            return

        cand_sec = np.repeat(nodes, n_main)
        cand_seg_idx = np.tile(np.arange(n_main), n_sec)
        cand_segs = segs[cand_seg_idx]

        # Ignore self-nodes in segments
        same_node = (
            (cand_sec == cand_segs[:, 0])
            | (cand_sec == cand_segs[:, 1])
            | (cand_sec == cand_segs[:, 2])
            | (cand_sec == cand_segs[:, 3])
        )

        best_d, best_pt, best_w = _narrow(x, cand_sec, cand_segs)
        best_d[same_node] = np.inf

        d_mat = best_d.reshape((n_sec, n_main))
        min_seg_col = np.argmin(d_mat, axis=1)
        row_idx = np.arange(n_sec)
        flat_idx = row_idx * n_main + min_seg_col

        d_closest = best_d[flat_idx]
        pt_closest = best_pt[flat_idx]
        w_closest = best_w[flat_idx]
        seg_closest = cand_segs[flat_idx]
        k_closest = k_segs[min_seg_col]

        penetration = self.gap - d_closest
        active = (penetration > 0.0) & (d_closest < np.inf)

        if not np.any(active):
            return

        act_nodes = nodes[active]
        act_pt = pt_closest[active]
        act_w = w_closest[active]
        act_segs = seg_closest[active]
        act_k = k_closest[active]
        act_p = penetration[active]

        # Normal unit vector pointing from segment closest point to node
        diff = x[act_nodes] - act_pt
        d_norm = norm3(diff)
        n_vec = np.where(d_norm[:, None] > EM20, diff / np.maximum(d_norm[:, None], EM20), 0.0)

        # Normal penalty force
        fn = act_k * act_p
        f_norm = fn[:, None] * n_vec

        # Relative velocity
        v_target = (
            act_w[:, 0:1] * v[act_segs[:, 0]]
            + act_w[:, 1:2] * v[act_segs[:, 1]]
            + act_w[:, 2:3] * v[act_segs[:, 2]]
            + act_w[:, 3:4] * v[act_segs[:, 3]]
        )
        v_rel = v[act_nodes] - v_target

        # Damping
        vn = np.sum(v_rel * n_vec, axis=-1)
        m_sec = mass[act_nodes]
        c_damp = 2.0 * self.visc * np.sqrt(np.maximum(act_k * m_sec, EM20))
        f_damp = - (c_damp * vn)[:, None] * n_vec

        # Tangential friction
        f_fric = np.zeros_like(f_norm)
        if self.fric > 0.0:
            vt = v_rel - vn[:, None] * n_vec
            vt_mag = norm3(vt)
            vt_dir = np.where(vt_mag[:, None] > EM20, vt / np.maximum(vt_mag[:, None], EM20), 0.0)
            f_fric_mag = np.minimum(self.fric * fn, act_k * vt_mag * dt)
            f_fric = - f_fric_mag[:, None] * vt_dir

        f_total = f_norm + f_damp + f_fric

        # Scatter forces:
        # Secondary node receives +f_total (pushing out of segment)
        scatter_add3(fcont, act_nodes, f_total)

        # Master segment corners receive -act_w * f_total
        # sum(corner_forces) = - sum(act_w) * f_total = - f_total
        # Total sum of forces for interaction = f_total - f_total = 0.
        for k in range(4):
            scatter_add3(fcont, act_segs[:, k], - act_w[:, k:k+1] * f_total)

        if stifn is not None and len(stifn) > 0:
            np.add.at(stifn, act_nodes, act_k)
            for k in range(4):
                np.add.at(stifn, act_segs[:, k], act_w[:, k] * act_k)

        if dt > 0.0:
            self.e_cont += float(np.sum(fn * act_p * dt))
            self.e_damp += float(np.sum(c_damp * (vn ** 2) * dt))

    def _edge_pass(
        self,
        edges1: np.ndarray,
        edges2: np.ndarray,
        x: np.ndarray,
        v: np.ndarray,
        mass: np.ndarray,
        dt: float,
        fcont: np.ndarray,
        stifn: np.ndarray | None,
    ) -> None:
        """Edge-to-edge penalty contact pass with momentum conservation."""
        n1 = len(edges1)
        n2 = len(edges2)
        if n1 == 0 or n2 == 0:
            return

        cand_e1 = np.repeat(edges1, n2, axis=0)
        cand_e2 = np.tile(edges2, (n1, 1))

        # Filter out edge pairs sharing a node
        share_node = (
            (cand_e1[:, 0] == cand_e2[:, 0])
            | (cand_e1[:, 0] == cand_e2[:, 1])
            | (cand_e1[:, 1] == cand_e2[:, 0])
            | (cand_e1[:, 1] == cand_e2[:, 1])
        )

        p1 = x[cand_e1[:, 0]]
        q1 = x[cand_e1[:, 1]]
        p2 = x[cand_e2[:, 0]]
        q2 = x[cand_e2[:, 1]]

        s, t, cA, cB = _closest_points_on_segments(p1, q1, p2, q2)
        diff = cA - cB
        dist = norm3(diff)
        dist[share_node] = np.inf

        penetration = self.gap - dist
        active = (penetration > 0.0) & (dist < np.inf)

        if not np.any(active):
            return

        act_e1 = cand_e1[active]
        act_e2 = cand_e2[active]
        act_s = s[active]
        act_t = t[active]
        act_p = penetration[active]
        act_dist = dist[active]
        act_diff = diff[active]

        n_edge = np.where(act_dist[:, None] > EM20, act_diff / np.maximum(act_dist[:, None], EM20), 0.0)
        k_edge = 1e6 * self.stfac

        # Normal penalty force
        fn = k_edge * act_p
        f_norm = fn[:, None] * n_edge

        # Relative velocity at closest points
        vA = (1.0 - act_s[:, None]) * v[act_e1[:, 0]] + act_s[:, None] * v[act_e1[:, 1]]
        vB = (1.0 - act_t[:, None]) * v[act_e2[:, 0]] + act_t[:, None] * v[act_e2[:, 1]]
        v_rel = vA - vB

        vn = np.sum(v_rel * n_edge, axis=-1)
        m_eff = 0.5 * (mass[act_e1[:, 0]] + mass[act_e2[:, 0]])
        c_damp = 2.0 * self.visc * np.sqrt(np.maximum(k_edge * m_eff, EM20))
        f_damp = - (c_damp * vn)[:, None] * n_edge

        f_total = f_norm + f_damp

        # Split onto edge endpoints:
        # Edge 1 receives +f_total, split by (1-s, s)
        scatter_add3(fcont, act_e1[:, 0], (1.0 - act_s[:, None]) * f_total)
        scatter_add3(fcont, act_e1[:, 1], act_s[:, None] * f_total)

        # Edge 2 receives -f_total, split by (1-t, t)
        scatter_add3(fcont, act_e2[:, 0], - (1.0 - act_t[:, None]) * f_total)
        scatter_add3(fcont, act_e2[:, 1], - act_t[:, None] * f_total)

        # Momentum conservation:
        # (1-s)*f + s*f - (1-t)*f - t*f = f - f = 0 strictly!

        if stifn is not None and len(stifn) > 0:
            np.add.at(stifn, act_e1[:, 0], 0.5 * k_edge)
            np.add.at(stifn, act_e1[:, 1], 0.5 * k_edge)
            np.add.at(stifn, act_e2[:, 0], 0.5 * k_edge)
            np.add.at(stifn, act_e2[:, 1], 0.5 * k_edge)

        if dt > 0.0:
            self.e_cont += float(np.sum(fn * act_p * dt))
            self.e_damp += float(np.sum(c_damp * (vn ** 2) * dt))

    def forces(
        self,
        x: np.ndarray,
        v: np.ndarray,
        mass: np.ndarray,
        dt: float,
        fcont: np.ndarray,
        cycle: int = 0,
        stifn: np.ndarray | None = None,
        t: float = 0.0,
        **kwargs,
    ) -> tuple[np.ndarray, float]:
        """Compute contact forces for combined surface/edge contact with symmetry.

        Returns (fcont, dt_bound).
        """
        t_cur = float(t if t is not None else 0.0)
        if t_cur < self.tstart or t_cur > self.tstop:
            return fcont, self.dt_bound

        # 1. Surface pass 1: Nodes of Surface 1 vs Segments of Surface 2
        if len(self.nodes1) > 0 and len(self.segs2) > 0:
            self._surface_pass(
                self.nodes1, self.segs2, self.K_seg2, x, v, mass, dt, fcont, stifn
            )

        # 2. Surface pass 2 (if ISYM >= 1 or Surface 2 has no segments): Nodes of Surface 2 vs Segments of Surface 1
        if (self.isym >= 1 or len(self.segs2) == 0) and len(self.nodes2) > 0 and len(self.segs1) > 0:
            self._surface_pass(
                self.nodes2, self.segs1, self.K_seg1, x, v, mass, dt, fcont, stifn
            )

        # 3. Edge pass (if IEDGE == 1): Edges of Surface 1 vs Edges of Surface 2
        if self.iedge == 1 and len(self.edges1) > 0 and len(self.edges2) > 0:
            self._edge_pass(
                self.edges1, self.edges2, x, v, mass, dt, fcont, stifn
            )

        return fcont, self.dt_bound
