"""
/INTER/TYPE1 — ALE/Eulerian to Lagrangian Fluid-Structure Coupling Interface.

Fortran origin:
  - ``starter/source/interfaces/int01/hm_read_inter_type01.F`` (card reading)
  - ``engine/source/interfaces/interf/intti1.F`` (tied/coupling interface coordinator)
  - ``engine/source/ale/inter/intal1.F``, ``intal2.F``, ``intal3.F`` (ALE interface coordinator)
  - ``engine/source/ale/inter/iqela1.F``, ``iqela2.F`` (coupling acceleration & force computation)
  - ``engine/source/ale/ale3d/iqel03.F`` (3D segment geometry & normal calculation)

Physics:
Couples secondary fluid nodes (ALE or Eulerian formulation) to Lagrangian structural
main faces (shells or solid faces: 4-node quads or 3-node triangles) using a penalty
coupling formulation:
  1. Main face projection: 4-node quads are decomposed into sub-triangles meeting at
     face centroid xc = 0.25 * (x1 + x2 + x3 + x4). 3-node triangles project directly.
     The shape functions H1..H4 form an exact partition of unity (sum(Hi) = 1.0).
  2. Penetration check: dist = |x_sec - pt_closest|. If dist <= gap, the pair is active
     and normal penetration is pene = gap - dist.
  3. Contact stiffness: K is scaled from master segment/node stiffness or stfac.
  4. Repulsive penalty reaction: F_n = K * pene + C * (-vn) for approaching velocity (vn < 0).
  5. Coulomb friction: tangential velocity opposing force regularized by slip speed.
  6. Momentum conservation: secondary node receives +F, master face corners receive -Hi * F.
     Linear and angular momentum are conserved to machine precision.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20
from ..common.fastmath import cross3, norm3
from ..model.model import Model
from . import tracking
from .inter_type7 import (
    _closest_point_on_triangle,
    combine_stiffness,
    node_stiffness_gap,
    segment_stiffness_gap,
)


def _t1_project(xs, main_nodes, x):
    """Vectorized projection of secondary points onto master face segments.

    Parameters
    ----------
    xs : np.ndarray (N_pairs, 3)
        Secondary node coordinates.
    main_nodes : np.ndarray (N_pairs, 4)
        Master face corner node indices.
    x : np.ndarray (N_nodes, 3)
        All nodal coordinates.

    Returns
    -------
    dist : np.ndarray (N_pairs,)
        Distance from xs to closest point on segment.
    pt : np.ndarray (N_pairs, 3)
        Coordinates of the closest point on segment.
    H : np.ndarray (N_pairs, 4)
        Shape function interpolation weights at closest point.
    n_face : np.ndarray (N_pairs, 3)
        Outward unit normal of the segment.
    """
    n_pairs = len(xs)
    dist = np.zeros(n_pairs, dtype=float)
    pt = np.zeros((n_pairs, 3), dtype=float)
    H = np.zeros((n_pairs, 4), dtype=float)
    n_face = np.zeros((n_pairs, 3), dtype=float)

    if n_pairs == 0:
        return dist, pt, H, n_face

    mn1 = main_nodes[:, 0]
    mn2 = main_nodes[:, 1]
    mn3 = main_nodes[:, 2]
    mn4 = main_nodes[:, 3]
    mn4_safe = np.maximum(mn4, 0)

    x1 = x[mn1]
    x2 = x[mn2]
    x3 = x[mn3]
    x4 = x[mn4_safe]

    # Detect 3-node triangles (n4 < 0 or n4 == n3 or n4 == n2)
    is_tri = (mn4 < 0) | (mn4 == mn3) | (mn4 == mn2)
    is_quad = ~is_tri

    # --- Triangles branch ---
    if np.any(is_tri):
        pt_tri, wa, wb, wc = _closest_point_on_triangle(
            xs[is_tri], x1[is_tri], x2[is_tri], x3[is_tri]
        )
        pt[is_tri] = pt_tri
        H[is_tri, 0] = wa
        H[is_tri, 1] = wb
        H[is_tri, 2] = wc
        H[is_tri, 3] = 0.0
        dist[is_tri] = norm3(xs[is_tri] - pt_tri)

        e12 = x2[is_tri] - x1[is_tri]
        e13 = x3[is_tri] - x1[is_tri]
        nf = cross3(e12, e13)
        nf_norm = norm3(nf)
        n_face[is_tri] = np.where(
            nf_norm[:, None] > EM20,
            nf / np.maximum(nf_norm, EM20)[:, None],
            np.array([0.0, 0.0, 1.0]),
        )

    # --- Quads branch (4-subtriangle decomposition around centroid) ---
    if np.any(is_quad):
        q_idx = np.where(is_quad)[0]
        xq1, xq2, xq3, xq4 = x1[q_idx], x2[q_idx], x3[q_idx], x4[q_idx]
        xsq = xs[q_idx]
        xc = 0.25 * (xq1 + xq2 + xq3 + xq4)

        pt0, wa0, wb0, wc0 = _closest_point_on_triangle(xsq, xq1, xq2, xc)
        pt1, wa1, wb1, wc1 = _closest_point_on_triangle(xsq, xq2, xq3, xc)
        pt2, wa2, wb2, wc2 = _closest_point_on_triangle(xsq, xq3, xq4, xc)
        pt3, wa3, wb3, wc3 = _closest_point_on_triangle(xsq, xq4, xq1, xc)

        d0 = np.sum((xsq - pt0) ** 2, axis=1)
        d1 = np.sum((xsq - pt1) ** 2, axis=1)
        d2 = np.sum((xsq - pt2) ** 2, axis=1)
        d3 = np.sum((xsq - pt3) ** 2, axis=1)

        dists = np.column_stack((d0, d1, d2, d3))
        min_idx = np.argmin(dists, axis=1)
        dist[q_idx] = np.sqrt(np.take_along_axis(dists, min_idx[:, None], axis=1).flatten())

        m0 = min_idx == 0
        m1 = min_idx == 1
        m2 = min_idx == 2
        m3 = min_idx == 3

        if np.any(m0):
            idx0 = q_idx[m0]
            H[idx0, 0] = wa0[m0] + 0.25 * wc0[m0]
            H[idx0, 1] = wb0[m0] + 0.25 * wc0[m0]
            H[idx0, 2] = 0.25 * wc0[m0]
            H[idx0, 3] = 0.25 * wc0[m0]
            pt[idx0] = pt0[m0]

        if np.any(m1):
            idx1 = q_idx[m1]
            H[idx1, 0] = 0.25 * wc1[m1]
            H[idx1, 1] = wa1[m1] + 0.25 * wc1[m1]
            H[idx1, 2] = wb1[m1] + 0.25 * wc1[m1]
            H[idx1, 3] = 0.25 * wc1[m1]
            pt[idx1] = pt1[m1]

        if np.any(m2):
            idx2 = q_idx[m2]
            H[idx2, 0] = 0.25 * wc2[m2]
            H[idx2, 1] = 0.25 * wc2[m2]
            H[idx2, 2] = wa2[m2] + 0.25 * wc2[m2]
            H[idx2, 3] = wb2[m2] + 0.25 * wc2[m2]
            pt[idx2] = pt2[m2]

        if np.any(m3):
            idx3 = q_idx[m3]
            H[idx3, 0] = wb3[m3] + 0.25 * wc3[m3]
            H[idx3, 1] = 0.25 * wc3[m3]
            H[idx3, 2] = 0.25 * wc3[m3]
            H[idx3, 3] = wa3[m3] + 0.25 * wc3[m3]
            pt[idx3] = pt3[m3]

        d13 = xq3 - xq1
        d24 = xq4 - xq2
        nf_q = cross3(d13, d24)
        nf_q_norm = norm3(nf_q)
        n_face[q_idx] = np.where(
            nf_q_norm[:, None] > EM20,
            nf_q / np.maximum(nf_q_norm, EM20)[:, None],
            np.array([0.0, 0.0, 1.0]),
        )

    return dist, pt, H, n_face


def _t1_forces(
    x: np.ndarray,
    v: np.ndarray,
    mass: np.ndarray,
    sec_nodes: np.ndarray,
    main_nodes: np.ndarray,
    stfac: float,
    gap: float,
    stiff_dc: float,
    fric: float,
    dt: float,
    K_pair: np.ndarray | None = None,
):
    """Vectorized calculation of TYPE1 penalty forces and momentum-conserving distribution.

    Returns
    -------
    forces_sec : np.ndarray (N_pairs, 3)
    forces_main : np.ndarray (N_pairs, 4, 3)
    active : np.ndarray (N_pairs,) bool
    stif : np.ndarray (N_active,) float
    H : np.ndarray (N_pairs, 4) float
    dt_bound : float
    """
    n_pairs = len(sec_nodes)
    forces_sec = np.zeros((n_pairs, 3), dtype=float)
    forces_main = np.zeros((n_pairs, 4, 3), dtype=float)
    H = np.zeros((n_pairs, 4), dtype=float)

    if n_pairs == 0 or dt <= 0.0:
        return forces_sec, forces_main, np.zeros(0, dtype=bool), np.zeros(0), H, np.inf

    xs = x[sec_nodes]
    vs = v[sec_nodes]

    # Exclude self-impact pairs where secondary node is a corner of the master segment
    self_pair = (
        (sec_nodes[:, None] == main_nodes[:, :3]).any(axis=1)
        | ((main_nodes[:, 3] >= 0) & (sec_nodes == main_nodes[:, 3]))
    )

    dist, pt, H, n_face = _t1_project(xs, main_nodes, x)

    # Penetration condition: dist <= gap (and not a self-corner pair)
    gap_val = gap if gap > 0.0 else 0.01
    active = (dist <= gap_val) & (~self_pair)

    if not np.any(active):
        return forces_sec, forces_main, active, np.zeros(0), H, np.inf

    act_idx = np.where(active)[0]
    pene = np.maximum(0.0, gap_val - dist[active])
    norm_d = dist[active]

    # Outward unit normal: points from master face towards secondary fluid node
    dvec = xs[active] - pt[active]
    deg = norm_d <= EM20
    d_safe = np.maximum(norm_d, EM20)
    nvec = np.where(deg[:, None], n_face[active], dvec / d_safe[:, None])

    # Contact stiffness
    if K_pair is not None and len(K_pair) == n_pairs:
        stif = K_pair[active]
    else:
        stif = np.full(len(act_idx), stfac if stfac > 0.0 else 1000.0, dtype=float)

    # Relative velocity at contact point
    mn = main_nodes[active]
    mn4_safe = np.maximum(mn[:, 3], 0)
    v_main = (
        H[active, 0:1] * v[mn[:, 0]]
        + H[active, 1:2] * v[mn[:, 1]]
        + H[active, 2:3] * v[mn[:, 2]]
        + H[active, 3:4] * v[mn4_safe]
    )
    vrel = vs[active] - v_main
    vn = np.einsum("nb,nb->n", vrel, nvec)

    # Normal penalty force (repulsive only)
    fn = stif * pene
    if stiff_dc > 0.0:
        # Viscous damping opposes approaching motion (vn < 0)
        m_eff = mass[sec_nodes[active]]
        C_damp = 2.0 * stiff_dc * np.sqrt(np.maximum(stif * m_eff, 0.0))
        damp = np.where(vn < 0.0, C_damp * (-vn), 0.0)
        fn += damp
    fn = np.maximum(fn, 0.0)

    f_total_sec = fn[:, None] * nvec

    # Coulomb friction
    if fric > 0.0:
        vt = vrel - vn[:, None] * nvec
        vt_mag = norm3(vt)
        v_ref = np.maximum(1e-3 * gap_val / max(dt, EM20), EM20)
        ft_mag = fric * fn * vt_mag / (vt_mag + v_ref)
        ft_vec = - (ft_mag / np.maximum(vt_mag, EM20))[:, None] * vt
        f_total_sec += ft_vec

    # Force distribution: secondary fluid node receives +F, master corners receive -H_k * F
    forces_sec[active] = f_total_sec
    for k in range(4):
        forces_main[active, k] = - H[active, k:k+1] * f_total_sec

    # Local stability timestep bound
    m_sec = np.maximum(mass[sec_nodes[active]], EM20)
    dt_bound = float(np.min(np.sqrt(2.0 * m_sec / np.maximum(stif, EM20))))

    return forces_sec, forces_main, active, stif, H, dt_bound


class ContactType1:
    """One /INTER/TYPE1 ALE-Lagrangian fluid-structure coupling interface, engine-side."""

    def __init__(self, itf, model: Model, log):
        self.itf = itf
        self.model = model
        self.id = getattr(itf, "id", 0)
        self.title = getattr(itf, "title", f"TYPE1_{self.id}")
        params = getattr(itf, "params", {}) or {}

        self.tstart = float(params.get("tstart", getattr(itf, "tstart", 0.0)) or 0.0)
        self.tstop = float(params.get("tstop", getattr(itf, "tstop", 1e30)) or 1e30)
        self.stfac = float(params.get("stfac", getattr(itf, "stfac", 1.0)) or 1.0)
        self.gap = float(params.get("gap", getattr(itf, "gap", 0.0)) or 0.0)
        self.fric = float(params.get("fric", getattr(itf, "fric", 0.0)) or 0.0)
        self.stiff_dc = float(params.get("stiff_dc", getattr(itf, "stiff_dc", 0.05)) or 0.05)

        # 1. Resolve master Lagrangian surface segments
        surfaces = getattr(model, "surfaces", {}) or {}
        master_surf_id = (
            params.get("surf_idl")
            or params.get("surf_id")
            or params.get("main_id")
            or getattr(itf, "surf_id", 0)
            or getattr(itf, "surf_id1", 0)
        )
        surf = surfaces.get(master_surf_id) if hasattr(surfaces, "get") else None

        # If surf_id didn't match, check if surf_id1 has valid segments
        if (surf is None or getattr(surf, "segments", None) is None or len(surf.segments) == 0) and getattr(itf, "surf_id1", 0) > 0:
            alt_id = getattr(itf, "surf_id1", 0)
            alt_surf = surfaces.get(alt_id)
            if alt_surf is not None and getattr(alt_surf, "segments", None) is not None and len(alt_surf.segments) > 0:
                surf = alt_surf
                master_surf_id = alt_id

        if surf is None or getattr(surf, "segments", None) is None or len(surf.segments) == 0:
            log.warning(
                f"/INTER/TYPE1/{self.id}: master Lagrangian surface {master_surf_id} "
                f"is missing or empty — interface inactive",
                "CONTACT INIT",
            )
            self._init_empty()
            return

        self.segs = np.asarray(surf.segments, dtype=np.int64)
        if self.segs.ndim == 2:
            if self.segs.shape[1] == 3:
                self.segs = np.column_stack([self.segs, np.full(len(self.segs), -1, dtype=np.int64)])
        self.seg_gtype = (
            surf.seg_gtype
            if getattr(surf, "seg_gtype", None) is not None
            else np.zeros(len(self.segs), dtype="<U8")
        )
        self.seg_elem = (
            surf.seg_elem
            if getattr(surf, "seg_elem", None) is not None
            else np.full(len(self.segs), -1, dtype=np.int64)
        )

        # 2. Resolve secondary fluid nodes (ALE)
        sec_nodes = np.zeros(0, dtype=np.int64)
        node_groups = getattr(model, "node_groups", {}) or {}

        grnod_id = params.get("grnod_id") or getattr(itf, "grnod_id", 0)
        if grnod_id > 0 and grnod_id in node_groups:
            grp = node_groups[grnod_id]
            if getattr(grp, "node_idx", None) is not None and len(grp.node_idx) > 0:
                sec_nodes = np.asarray(grp.node_idx, dtype=np.int64)

        if len(sec_nodes) == 0:
            # Check secondary surface
            sec_surf_id = (
                params.get("surf_ida")
                or params.get("surf_id2")
                or params.get("sec_id")
                or (getattr(itf, "surf_id1", 0) if getattr(itf, "surf_id1", 0) != master_surf_id else getattr(itf, "surf_id", 0))
            )
            sec_surf = surfaces.get(sec_surf_id) if hasattr(surfaces, "get") else None
            if sec_surf is not None and getattr(sec_surf, "segments", None) is not None and len(sec_surf.segments) > 0:
                s_segs = np.asarray(sec_surf.segments, dtype=np.int64)
                sec_nodes = np.unique(s_segs[s_segs >= 0])

        if len(sec_nodes) == 0:
            log.warning(
                f"/INTER/TYPE1/{self.id}: secondary fluid nodes empty or missing — interface inactive",
                "CONTACT INIT",
            )
            self._init_empty()
            return

        self.nodes = np.sort(np.unique(sec_nodes))

        # Stiffness calculation
        scale = self.stfac if self.stfac > 0.0 else 1.0
        Km, gm = segment_stiffness_gap(model, self.segs, self.seg_gtype, self.seg_elem, scale)
        self.Km = Km
        Ks_all, _ = node_stiffness_gap(model, scale)
        self.Ks = np.zeros(len(self.nodes), dtype=float)
        if len(Ks_all) > 0 and len(self.nodes) > 0:
            valid_ks = (self.nodes >= 0) & (self.nodes < len(Ks_all))
            self.Ks[valid_ks] = Ks_all[self.nodes[valid_ks]]

        self.idel = int(params.get("idel", getattr(itf, "idel", 0)) or 0)
        self.deletable = self.idel >= 1 and tracking.any_deletable(
            model, self.seg_gtype, sec_nodes=self.nodes
        )
        self.seg_alive = np.ones(len(self.segs), dtype=bool)

        mass0 = getattr(model, "mass0", None)
        m_eff = mass0 if mass0 is not None and len(mass0) > 0 else getattr(model, "mass", np.ones(len(self.nodes)))
        self.dt_bound = self._compute_dt_bound(m_eff)

        log.info(
            f"Initialized /INTER/TYPE1/{self.id} '{self.title}' "
            f"({len(self.nodes)} secondary nodes vs {len(self.segs)} master segments)"
        )

    def _init_empty(self) -> None:
        """Initialize empty inactive state."""
        self.nodes = np.zeros(0, dtype=np.int64)
        self.segs = np.zeros((0, 4), dtype=np.int64)
        self.seg_gtype = np.zeros(0, dtype="<U8")
        self.seg_elem = np.zeros(0, dtype=np.int64)
        self.seg_alive = np.zeros(0, dtype=bool)
        self.Km = np.zeros(0, dtype=float)
        self.Ks = np.zeros(0, dtype=float)
        self.deletable = False
        self.dt_bound = np.inf

    def _compute_dt_bound(self, mass: np.ndarray) -> float:
        """Calculate stability time step limit for the interface."""
        if len(self.segs) == 0 or len(self.nodes) == 0 or len(mass) == 0:
            return np.inf
        valid_nodes = self.nodes[(self.nodes >= 0) & (self.nodes < len(mass))]
        if len(valid_nodes) == 0:
            return np.inf
        Km_max = self.Km.max() if len(self.Km) and self.Km.max() > 0.0 else self.stfac
        m_min = np.min(mass[valid_nodes])
        k_eff = max(Km_max, EM20)
        return float(np.sqrt(2.0 * m_min / k_eff))

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
        """Compute fluid-structure penalty coupling forces.

        Updates fcont in-place and returns (fcont, dt_bound).
        """
        n_sec = len(self.nodes)
        n_main = len(self.segs)
        if n_sec == 0 or n_main == 0 or dt <= 0.0:
            return fcont, self.dt_bound

        if t is not None:
            if t < self.tstart or t > self.tstop:
                return fcont, self.dt_bound

        if self.deletable:
            self.seg_alive = tracking.alive_segment_mask(
                self.model, self.seg_gtype, self.seg_elem
            )

        live_mask = self.seg_alive
        if not np.any(live_mask):
            return fcont, self.dt_bound

        live_segs = self.segs[live_mask]
        n_live = len(live_segs)

        # Compute segment centroids for candidate filtering
        mn1 = live_segs[:, 0]
        mn2 = live_segs[:, 1]
        mn3 = live_segs[:, 2]
        mn4 = live_segs[:, 3]
        mn4_safe = np.maximum(mn4, 0)
        is_tri = (mn4 < 0) | (mn4 == mn3) | (mn4 == mn2)

        xc_quad = 0.25 * (x[mn1] + x[mn2] + x[mn3] + x[mn4_safe])
        xc_tri = (x[mn1] + x[mn2] + x[mn3]) / 3.0
        xc = np.where(is_tri[:, None], xc_tri, xc_quad)

        xs = x[self.nodes]

        # Chunked closest segment matching to support large clouds efficiently
        closest_idx = np.empty(n_sec, dtype=np.int64)
        chunk_size = max(1, 250000 // max(1, n_live))
        for start in range(0, n_sec, chunk_size):
            end = min(start + chunk_size, n_sec)
            diff = xs[start:end, None, :] - xc[None, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            closest_idx[start:end] = np.argmin(dist_sq, axis=1)

        matched_segs = live_segs[closest_idx]

        # Stiffness per pair
        live_km = self.Km[live_mask] if len(self.Km) == len(self.segs) else np.full(n_live, self.stfac)
        K_pair = combine_stiffness(0, self.stfac, live_km[closest_idx], self.Ks)

        fsec, fmain, active, stif, H, dt_step = _t1_forces(
            x,
            v,
            mass,
            self.nodes,
            matched_segs,
            self.stfac,
            self.gap,
            self.stiff_dc,
            self.fric,
            dt,
            K_pair=K_pair,
        )

        if not np.any(active):
            return fcont, self.dt_bound

        act_sec = self.nodes[active]
        act_segs = matched_segs[active]
        act_fsec = fsec[active]
        act_fmain = fmain[active]
        act_stif = stif

        # Scatter forces in-place into fcont
        np.add.at(fcont, act_sec, act_fsec)
        np.add.at(fcont, act_segs[:, 0], act_fmain[:, 0])
        np.add.at(fcont, act_segs[:, 1], act_fmain[:, 1])
        np.add.at(fcont, act_segs[:, 2], act_fmain[:, 2])
        valid4 = act_segs[:, 3] >= 0
        if np.any(valid4):
            np.add.at(fcont, act_segs[valid4, 3], act_fmain[valid4, 3])

        # Accumulate stiffness in stifn for /DT/NODA if requested
        if stifn is not None:
            np.add.at(stifn, act_sec, act_stif)
            for k in range(3):
                np.add.at(stifn, act_segs[:, k], H[active, k] * act_stif)
            if np.any(valid4):
                np.add.at(stifn, act_segs[valid4, 3], H[active][valid4, 3] * act_stif[valid4])

        dt_bound = min(self.dt_bound, dt_step)
        return fcont, dt_bound
