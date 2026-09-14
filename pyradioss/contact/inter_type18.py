"""
/INTER/TYPE18 — Fluid-Structure Penalty Contact Interface (M60/M491).

Fortran origin:
  - ``starter/source/interfaces/int18/hm_read_inter_type18.F`` (card reading)
  - ``engine/source/interfaces/int18/i18dst3.F`` (gap, penetration, closest-point projection)
  - ``engine/source/interfaces/int18/i18for3.F`` (penalty reaction forces, sub-triangle partition of unity, damping)
  - ``engine/source/interfaces/int18/i18main_kine.F`` (kinematic coordinator)
  - ``engine/source/interfaces/int18/i18tri.F`` (triangle projection kernel)
  - ``engine/source/interfaces/int18/multi_i18_force_pon.F`` (momentum-conserving force distribution)

Physics:
Couples secondary fluid nodes (typically ALE or SPH) to structural main faces
(shells or solid surfaces: 4-node quads or 3-node triangles) using a dynamic
penetration penalty formulation:
  1. Main face projection: 4-node quads are decomposed into 4 sub-triangles meeting
     at face centroid xc = (x1 + x2 + x3 + x4) / 4. 3-node triangles project directly.
     The shape functions H1..H4 form a partition of unity (sum(Hi) = 1.0).
  2. Penetration check: dist = |x_sec - pt_closest|. If dist <= gap, the pair is active
     and normal penetration is pene = gap - dist.
  3. Dynamic stiffness: K = Stfval * (pene / gap) if gap > EM20 else Stfval.
  4. Relative motion: normal velocity vn = (v_sec - v_main) . n. Normal displacement
     accumulates dynamically in cand_p += vn * dt while in contact; separation resets cand_p to 0.
  5. Reaction force: F_n = K * cand_p + C * vn (C = Stiff_dc * (pene/gap) for vn > 0).
  6. Momentum conservation: F_sec = -F_n * n, F_main,i = Hi * F_n * n. Linear and angular
     momentum are conserved to machine precision.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20
from ..common.fastmath import cross3, norm3
from ..model.model import Model
from .inter_type7 import _closest_point_on_triangle


def _t18_forces(x, v, mass, sec_nodes, main_nodes, stfval, gap, stiff_dc, cand_p, dt):
    """Vectorized numpy kernel for TYPE18 penalty forces (i18for3.F).

    Parameters
    ----------
    x : np.ndarray (N_nodes, 3)
        Nodal coordinates.
    v : np.ndarray (N_nodes, 3)
        Nodal velocities.
    mass : np.ndarray (N_nodes,)
        Nodal masses.
    sec_nodes : np.ndarray (N_pairs,)
        Secondary fluid node indices.
    main_nodes : np.ndarray (N_pairs, 4)
        Main structure face corner node indices (quads or triangles).
    stfval : float
        Penalty stiffness parameter.
    gap : float
        Interface gap.
    stiff_dc : float
        Viscous damping coefficient.
    cand_p : np.ndarray (N_pairs,)
        Accumulated normal displacement tracker (modified in place).
    dt : float
        Current engine time step.

    Returns
    -------
    forces_sec : np.ndarray (N_pairs, 3)
        Forces on secondary nodes.
    forces_main : np.ndarray (N_pairs, 4, 3)
        Forces on main face corners.
    active : np.ndarray (N_pairs,) bool
        Active contact mask.
    stif : np.ndarray (N_active,) float
        Contact stiffness per active pair.
    H : np.ndarray (N_pairs, 4) float
        Shape function weights for the main face corners.
    econt_inc : float
        Total contact energy increment for the step.
    """
    n_pairs = len(sec_nodes)
    forces_sec = np.zeros((n_pairs, 3))
    forces_main = np.zeros((n_pairs, 4, 3))
    H = np.zeros((n_pairs, 4))

    if n_pairs == 0:
        return forces_sec, forces_main, np.zeros(0, dtype=bool), np.zeros(0), H, 0.0

    xs = x[sec_nodes]
    vs = v[sec_nodes]

    mn1 = main_nodes[:, 0]
    mn2 = main_nodes[:, 1]
    mn3 = main_nodes[:, 2]
    mn4 = main_nodes[:, 3]
    mn4_safe = np.maximum(mn4, 0)

    x1, x2, x3 = x[mn1], x[mn2], x[mn3]
    x4 = x[mn4_safe]
    v1, v2, v3 = v[mn1], v[mn2], v[mn3]
    v4 = v[mn4_safe]

    # Detect 3-node triangular segments (n4 == n3 or n4 < 0) per i18for3.F line 228
    is_tri = (mn4 == mn3) | (mn4 < 0)
    is_quad = ~is_tri

    dist = np.zeros(n_pairs)
    pt = np.zeros((n_pairs, 3))
    n_face = np.zeros((n_pairs, 3))

    # --- Triangles branch (i18for3.F lines 228-238) ---
    if np.any(is_tri):
        pt_tri, wa, wb, wc = _closest_point_on_triangle(
            xs[is_tri], x1[is_tri], x2[is_tri], x3[is_tri])
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
            np.array([0.0, 0.0, 1.0])
        )

    # --- Quads branch (i18for3.F lines 200-226) ---
    if np.any(is_quad):
        q_idx = np.where(is_quad)[0]
        xq1, xq2, xq3, xq4 = x1[q_idx], x2[q_idx], x3[q_idx], x4[q_idx]
        xsq = xs[q_idx]
        xc = 0.25 * (xq1 + xq2 + xq3 + xq4)

        # 4 sub-triangles: T0=(x1, x2, xc), T1=(x2, x3, xc), T2=(x3, x4, xc), T3=(x4, x1, xc)
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

        e12 = xq2 - xq1
        e14 = xq4 - xq1
        nf = cross3(e12, e14)
        nf_norm = norm3(nf)
        n_face[q_idx] = np.where(
            nf_norm[:, None] > EM20,
            nf / np.maximum(nf_norm, EM20)[:, None],
            np.array([0.0, 0.0, 1.0])
        )

    # Penetration check (i18for3.F line 273)
    active = dist <= gap
    cand_p[~active] = 0.0

    if not np.any(active):
        return forces_sec, forces_main, active, np.zeros(0), H, 0.0

    pene = np.maximum(0.0, gap - dist[active])
    norm_d = dist[active]
    deg = norm_d <= EM20

    # Normal vector pointing outward from master surface towards fluid
    dvec = xs[active] - pt[active]
    d_safe = np.maximum(norm_d, EM20)
    nvec = np.where(deg[:, None], n_face[active], dvec / d_safe[:, None])

    # Dynamic stiffness (i18for3.F line 277)
    stif = np.where(gap > EM20, stfval * (pene / gap), stfval)

    # Relative velocity of closest points (i18for3.F line 254-257)
    v_main = (
        H[active, 0:1] * v1[active]
        + H[active, 1:2] * v2[active]
        + H[active, 2:3] * v3[active]
        + H[active, 3:4] * v4[active]
    )
    vrel = vs[active] - v_main
    vn = np.einsum("nb,nb->n", vrel, nvec)

    # Accumulate normal displacement (i18for3.F line 275)
    cand_p[active] += vn * dt

    # Normal reaction force (i18for3.F line 290)
    fni = stif * cand_p[active]

    # Viscous damping (i18for3.F line 327)
    if stiff_dc > 0.0:
        damp = np.where(
            vn > 0.0,
            np.where(gap > EM20, stiff_dc * (pene / gap) * vn, stiff_dc * vn),
            0.0
        )
        fni += damp

    fvec = fni[:, None] * nvec

    # Force distribution: secondary node gets -fvec, main face corners get Hi * fvec
    forces_sec[active] = -fvec
    forces_main[active, 0] = H[active, 0:1] * fvec
    forces_main[active, 1] = H[active, 1:2] * fvec
    forces_main[active, 2] = H[active, 2:3] * fvec
    forces_main[active, 3] = H[active, 3:4] * fvec

    # Contact energy increment
    econt_inc = float(np.sum(fni * vn * dt))

    return forces_sec, forces_main, active, stif, H, econt_inc


class ContactType18:
    """Penalty fluid-structure contact interface (/INTER/TYPE18).

    Engine interface implementation compliant with resol.F contact coordinator contract.
    """

    def __init__(self, itf, model: Model, log):
        self.itf = itf
        self.id = getattr(itf, "id", 0)
        self.title = getattr(itf, "title", f"TYPE18_{self.id}")

        self.grnod = getattr(itf, "grnod_id", 0)
        self.surf = getattr(itf, "surf_id", 0)

        self.stfac = float(getattr(itf, "stfac", 1.0) or 1.0)
        self.gap = float(getattr(itf, "gap", 0.0) or 0.0)
        self.stiff_dc = float(getattr(itf, "stiff_dc", 0.0) or 0.0)
        self.sort_fact = float(getattr(itf, "sort_fact", 0.2) or 0.2)
        self.dt_bound = np.inf
        self.tstart = float(getattr(itf, "tstart", 0.0) or 0.0)
        self.tstop = float(getattr(itf, "tstop", np.inf) or np.inf)
        if self.tstop <= 0.0:
            self.tstop = np.inf

        # Secondary nodes (fluid)
        g = model.node_groups.get(self.grnod)
        if g is None or g.node_idx is None:
            log.error(f"/INTER/TYPE18/{self.id}: secondary node group {self.grnod} "
                      f"not found in model", "CONTACT INIT")
            self.sec_nodes = np.zeros(0, dtype=np.int64)
        else:
            self.sec_nodes = np.array(g.node_idx, dtype=np.int64)

        # Main faces (structure)
        surf = model.surfaces.get(self.surf)
        if surf is None or surf.segments is None or len(surf.segments) == 0:
            log.error(f"/INTER/TYPE18/{self.id}: main surface {self.surf} "
                      f"not found or empty in model", "CONTACT INIT")
            self.main_faces = np.zeros((0, 4), dtype=np.int64)
        else:
            self.main_faces = np.array(surf.segments, dtype=np.int64)

        if len(self.sec_nodes) == 0 or len(self.main_faces) == 0:
            self._init_empty()
            return

        self.cand_p = np.zeros(len(self.sec_nodes), dtype=float)

        log.info(f"Initialized /INTER/TYPE18/{self.id} '{self.title}' "
                 f"({len(self.sec_nodes)} secondary nodes vs {len(self.main_faces)} main segments)")

    def _init_empty(self) -> None:
        """Initialize empty state for inactive or missing interfaces."""
        self.sec_nodes = np.zeros(0, dtype=np.int64)
        self.main_faces = np.zeros((0, 4), dtype=np.int64)
        self.cand_p = np.zeros(0, dtype=float)
        self.dt_bound = np.inf
        self.tstart = float(getattr(self.itf, "tstart", 0.0) or 0.0)
        self.tstop = float(getattr(self.itf, "tstop", np.inf) or np.inf)
        if self.tstop <= 0.0:
            self.tstop = np.inf

    def forces(self, x: np.ndarray, v: np.ndarray, mass: np.ndarray, dt: float,
               fcont: np.ndarray, cycle: int = 0, stifn: np.ndarray | None = None,
               t: float | None = None, **kwargs) -> tuple[float, float]:
        """Penalty forces for one cycle, scattered into ``fcont``.

        Parameters
        ----------
        x : np.ndarray (N, 3)
            Nodal coordinates.
        v : np.ndarray (N, 3)
            Nodal velocities.
        mass : np.ndarray (N,)
            Nodal masses.
        dt : float
            Current time step.
        fcont : np.ndarray (N, 3)
            Global contact force accumulator array.
        cycle : int
            Current cycle counter.
        stifn : np.ndarray (N,) | None
            Nodal stiffness accumulator for /DT/NODA.
        t : float | None
            Current simulation time.

        Returns
        -------
        contact_work : float
            Contact energy increment booked into energy ledger.
        dt_interface : float
            Time step bound for explicit stability.
        """
        n_sec = len(self.sec_nodes)
        n_main = len(self.main_faces)
        if n_sec == 0 or n_main == 0 or dt <= 0.0:
            return 0.0, self.dt_bound

        if t is not None:
            if t < self.tstart or t > self.tstop:
                return 0.0, self.dt_bound

        xs = x[self.sec_nodes]

        # Centers of all main faces
        mn1 = self.main_faces[:, 0]
        mn2 = self.main_faces[:, 1]
        mn3 = self.main_faces[:, 2]
        mn4 = self.main_faces[:, 3]
        mn4_safe = np.maximum(mn4, 0)
        is_tri = (mn4 == mn3) | (mn4 < 0)

        # Centroid xc: for triangles (x1+x2+x3)/3, for quads (x1+x2+x3+x4)/4
        xc_quad = 0.25 * (x[mn1] + x[mn2] + x[mn3] + x[mn4_safe])
        xc_tri = (x[mn1] + x[mn2] + x[mn3]) / 3.0
        xc = np.where(is_tri[:, None], xc_tri, xc_quad)

        # Chunked closest face search to prevent high memory usage
        closest_idx = np.empty(n_sec, dtype=np.int64)
        chunk_size = max(1, 250000 // max(1, n_main))
        for start in range(0, n_sec, chunk_size):
            end = min(start + chunk_size, n_sec)
            diff = xs[start:end, None, :] - xc[None, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            closest_idx[start:end] = np.argmin(dist_sq, axis=1)

        closest_faces = self.main_faces[closest_idx]

        fsec, fmain, active, stif, H, econt = _t18_forces(
            x, v, mass, self.sec_nodes, closest_faces,
            self.stfac, self.gap, self.stiff_dc,
            self.cand_p, dt
        )

        if not np.any(active):
            return 0.0, self.dt_bound

        act_sec = self.sec_nodes[active]
        act_faces = closest_faces[active]

        # Scatter contact forces into fcont
        np.add.at(fcont, act_sec, fsec[active])
        np.add.at(fcont, act_faces[:, 0], fmain[active, 0])
        np.add.at(fcont, act_faces[:, 1], fmain[active, 1])
        np.add.at(fcont, act_faces[:, 2], fmain[active, 2])
        has_node4 = act_faces[:, 3] >= 0
        if np.any(has_node4):
            np.add.at(fcont, act_faces[has_node4, 3], fmain[active][has_node4, 3])

        # Stiffness accumulation for /DT/NODA and interface time step bound
        n_nod = len(fcont)
        K_node = np.zeros(n_nod, dtype=float)
        np.add.at(K_node, act_sec, stif)
        np.add.at(K_node, act_faces[:, 0], H[active, 0] * stif)
        np.add.at(K_node, act_faces[:, 1], H[active, 1] * stif)
        np.add.at(K_node, act_faces[:, 2], H[active, 2] * stif)
        if np.any(has_node4):
            np.add.at(K_node, act_faces[has_node4, 3], H[active][has_node4, 3] * stif[has_node4])

        if stifn is not None:
            stifn += K_node

        loaded = K_node > 0.0
        if np.any(loaded):
            dt_int = float(np.min(np.sqrt(2.0 * mass[loaded] / K_node[loaded])))
        else:
            dt_int = np.inf

        dt_bound = min(self.dt_bound, dt_int)
        return econt, dt_bound
