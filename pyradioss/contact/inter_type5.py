"""
/INTER/TYPE5 — Node-to-Surface Penalty Contact Interface with Nonlinear & Velocity-Dependent Friction.

Fortran origin:
  - ``starter/source/interfaces/int05/hm_read_inter_type05.F`` (card reading)
  - ``engine/source/interfaces/inter3d/i5for3.F`` (penalty forces, linear momentum conservation)
  - ``engine/source/interfaces/inter3d/i5cork3.F`` (closest point projection & coordinates)
  - ``engine/source/interfaces/inter3d/i5keg3.F`` (friction models MFROT 1..4 & tangential filter IFQ)
  - ``engine/source/interfaces/interf/ibcoff.F`` (Cartesian DOF deactivation)

Physics:
Node-to-surface penalty contact interface:
  1. Broad phase / closest point projection: each secondary node is projected onto
     candidate master face segments (4-node quads or 3-node triangles).
  2. Partition of unity: shape function weights H1..H4 satisfy sum(Hi) = 1.0.
  3. Penetration check: dist = |x_sec - pt_closest|. If dist <= gap, pair is active with
     pene = gap - dist.
  4. Penalty forces: normal repulsive force Fn = K * pene + C * (-vn) for approaching velocity (vn < 0).
  5. Nonlinear and velocity-dependent Coulomb friction (MFROT 1..4):
     - MFROT = 0: constant Coulomb mu = Fric
     - MFROT = 1: generalized viscous polynomial mu(p, v)
     - MFROT = 2: Darmstadt exponential law
     - MFROT = 3: Renard piecewise law
     - MFROT = 4: exponential static-to-dynamic decay
     Contact pressure p = Fn / area_segment, slip speed v = |v_tan|.
  6. Tangential force filtering (IFQ 1..3): first-order exponential smoothing.
  7. Cartesian DOF deactivation: ibc1, ibc2, ibc3 deactivates X, Y, Z forces on both sides.
  8. Momentum conservation: secondary node receives +F, master face corners receive -Hi * F.
     Total linear momentum sum(fcont) is zero to machine precision (<= 1e-12).
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20
from ..common.fastmath import cross3, norm3
from ..model.model import Model
from . import friction, tracking
from .inter_type1 import _t1_project
from .inter_type7 import (
    combine_stiffness,
    node_stiffness_gap,
    segment_stiffness_gap,
)


def _segment_areas(main_nodes: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Calculate surface area of master segments (quads or triangles)."""
    n_segs = len(main_nodes)
    if n_segs == 0:
        return np.zeros(0, dtype=float)

    mn1 = main_nodes[:, 0]
    mn2 = main_nodes[:, 1]
    mn3 = main_nodes[:, 2]
    mn4 = main_nodes[:, 3]
    mn4_safe = np.maximum(mn4, 0)
    is_tri = (mn4 < 0) | (mn4 == mn3) | (mn4 == mn2)

    # Triangles: area = 0.5 * |(x2 - x1) x (x3 - x1)|
    e12 = x[mn2] - x[mn1]
    e13 = x[mn3] - x[mn1]
    tri_areas = 0.5 * norm3(cross3(e12, e13))

    # Quads: area = 0.5 * |(x3 - x1) x (x4 - x2)|
    d13 = x[mn3] - x[mn1]
    d24 = x[mn4_safe] - x[mn2]
    quad_areas = 0.5 * norm3(cross3(d13, d24))

    return np.where(is_tri, tri_areas, quad_areas)


def _t5_forces(
    x: np.ndarray,
    v: np.ndarray,
    mass: np.ndarray,
    sec_nodes: np.ndarray,
    main_nodes: np.ndarray,
    stfac: float,
    gap: float,
    stiff_dc: float,
    fric: float,
    mfrot: int,
    fric_c: np.ndarray,
    ibc1: bool,
    ibc2: bool,
    ibc3: bool,
    dt: float,
    K_pair: np.ndarray | None = None,
):
    """Compute TYPE5 contact forces with nonlinear friction and DOF deactivation.

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

    # Exclude self-impact pairs
    self_pair = (
        (sec_nodes[:, None] == main_nodes[:, :3]).any(axis=1)
        | ((main_nodes[:, 3] >= 0) & (sec_nodes == main_nodes[:, 3]))
    )

    dist, pt, H, n_face = _t1_project(xs, main_nodes, x)

    gap_val = gap if gap > 0.0 else 0.01
    active = (dist <= gap_val) & (~self_pair)

    if not np.any(active):
        return forces_sec, forces_main, active, np.zeros(0), H, np.inf

    act_idx = np.where(active)[0]
    pene = np.maximum(0.0, gap_val - dist[active])
    norm_d = dist[active]

    dvec = xs[active] - pt[active]
    deg = norm_d <= EM20
    d_safe = np.maximum(norm_d, EM20)
    nvec = np.where(deg[:, None], n_face[active], dvec / d_safe[:, None])

    if K_pair is not None and len(K_pair) == n_pairs:
        stif = K_pair[active]
    else:
        stif = np.full(len(act_idx), stfac if stfac > 0.0 else 1000.0, dtype=float)

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

    fn = stif * pene
    if stiff_dc > 0.0:
        m_eff = mass[sec_nodes[active]]
        C_damp = 2.0 * stiff_dc * np.sqrt(np.maximum(stif * m_eff, 0.0))
        damp = np.where(vn < 0.0, C_damp * (-vn), 0.0)
        fn += damp
    fn = np.maximum(fn, 0.0)

    f_total_sec = fn[:, None] * nvec

    # Nonlinear & velocity-dependent Coulomb friction
    if fric > 0.0 or mfrot > 0:
        vt = vrel - vn[:, None] * nvec
        vt_mag = norm3(vt)

        # Contact pressure p = Fn / Area
        areas = _segment_areas(mn, x)
        p = fn / np.maximum(areas, 1e-12)

        if mfrot == 0:
            mu = np.full(len(fn), fric, dtype=float)
        else:
            mu = friction.mu_kinetic(mfrot, fric, fric_c, p, vt_mag)

        v_ref = np.maximum(1e-3 * gap_val / max(dt, EM20), EM20)
        ft_mag = mu * fn * vt_mag / (vt_mag + v_ref)
        ft_vec = - (ft_mag / np.maximum(vt_mag, EM20))[:, None] * vt
        f_total_sec += ft_vec

    # Cartesian DOF deactivation
    if ibc1:
        f_total_sec[:, 0] = 0.0
    if ibc2:
        f_total_sec[:, 1] = 0.0
    if ibc3:
        f_total_sec[:, 2] = 0.0

    forces_sec[active] = f_total_sec
    for k in range(4):
        forces_main[active, k] = - H[active, k:k+1] * f_total_sec

    m_sec = np.maximum(mass[sec_nodes[active]], EM20)
    dt_bound = float(np.min(np.sqrt(2.0 * m_sec / np.maximum(stif, EM20))))

    return forces_sec, forces_main, active, stif, H, dt_bound


class ContactType5:
    """One /INTER/TYPE5 node-to-surface penalty contact interface with nonlinear friction, engine-side."""

    def __init__(self, itf, model: Model, log):
        self.itf = itf
        self.model = model
        self.id = getattr(itf, "id", 0)
        self.title = getattr(itf, "title", f"TYPE5_{self.id}")
        params = getattr(itf, "params", {}) or {}

        self.tstart = float(params.get("tstart", getattr(itf, "tstart", 0.0)) or 0.0)
        self.tstop = float(params.get("tstop", getattr(itf, "tstop", 1e30)) or 1e30)
        self.stfac = float(params.get("stfac", getattr(itf, "stfac", 1.0)) or 1.0)
        self.gap = float(params.get("gap", getattr(itf, "gap", 0.0)) or 0.0)
        self.fric = float(params.get("fric", getattr(itf, "fric", 0.0)) or 0.0)
        self.stiff_dc = float(params.get("stiff_dc", getattr(itf, "stiff_dc", 0.05)) or 0.05)

        # Nonlinear friction options (MFROT, IFQ, XFILTR, C1..C6)
        self.mfrot = int(params.get("mfrot", getattr(itf, "mfrot", getattr(itf, "ifric", 0))) or 0)
        self.ifq = int(params.get("ifq", getattr(itf, "ifq", getattr(itf, "ifiltr", 0))) or 0)
        self.xfiltr = float(params.get("xfiltr", getattr(itf, "xfiltr", getattr(itf, "xfreq", 0.0))) or 0.0)

        fric_c = params.get("fric_c", getattr(itf, "fric_c", None))
        if fric_c is not None and np.any(np.asarray(fric_c) != 0.0):
            self.fric_c = np.asarray(fric_c, dtype=float)
        else:
            c1 = float(params.get("c1", getattr(itf, "c1", 0.0)) or 0.0)
            c2 = float(params.get("c2", getattr(itf, "c2", 0.0)) or 0.0)
            c3 = float(params.get("c3", getattr(itf, "c3", 0.0)) or 0.0)
            c4 = float(params.get("c4", getattr(itf, "c4", 0.0)) or 0.0)
            c5 = float(params.get("c5", getattr(itf, "c5", 0.0)) or 0.0)
            c6 = float(params.get("c6", getattr(itf, "c6", 0.0)) or 0.0)
            self.fric_c = np.array([c1, c2, c3, c4, c5, c6], dtype=float)

        self._filt_keys = np.zeros(0, dtype=np.int64)
        self._filt_vals = np.zeros((0, 3), dtype=float)

        # Cartesian DOF deactivation (ibc1, ibc2, ibc3)
        ibc_val = int(params.get("ibc", getattr(itf, "ibc", 0)) or 0)
        self.ibc1 = bool(
            params.get("ibc1", getattr(itf, "ibc1", (ibc_val & 4) != 0))
            or params.get("Deactivate_X_BC", False)
        )
        self.ibc2 = bool(
            params.get("ibc2", getattr(itf, "ibc2", (ibc_val & 2) != 0))
            or params.get("Deactivate_Y_BC", False)
        )
        self.ibc3 = bool(
            params.get("ibc3", getattr(itf, "ibc3", (ibc_val & 1) != 0))
            or params.get("Deactivate_Z_BC", False)
        )

        # 1. Master surface lookup
        surfaces = getattr(model, "surfaces", {}) or {}
        master_surf_id = (
            params.get("mainentityids")
            or params.get("surf_id")
            or getattr(itf, "surf_id", 0)
        )
        surf = surfaces.get(master_surf_id) if hasattr(surfaces, "get") else None

        if surf is None or getattr(surf, "segments", None) is None or len(surf.segments) == 0:
            log.warning(
                f"/INTER/TYPE5/{self.id}: master surface {master_surf_id} "
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

        # 2. Secondary nodes lookup
        sec_nodes = np.zeros(0, dtype=np.int64)
        node_groups = getattr(model, "node_groups", {}) or {}

        grnod_id = params.get("grnod_id") or getattr(itf, "grnod_id", 0)
        if grnod_id > 0 and grnod_id in node_groups:
            grp = node_groups[grnod_id]
            if getattr(grp, "node_idx", None) is not None and len(grp.node_idx) > 0:
                sec_nodes = np.asarray(grp.node_idx, dtype=np.int64)

        if len(sec_nodes) == 0:
            sec_surf_id = (
                params.get("secondaryentityids")
                or params.get("surf_id1")
                or params.get("surf_id2")
                or getattr(itf, "surf_id1", 0)
                or getattr(itf, "surf_id2", 0)
            )
            sec_surf = surfaces.get(sec_surf_id) if hasattr(surfaces, "get") else None
            if sec_surf is not None and getattr(sec_surf, "segments", None) is not None and len(sec_surf.segments) > 0:
                s_segs = np.asarray(sec_surf.segments, dtype=np.int64)
                sec_nodes = np.unique(s_segs[s_segs >= 0])

        if len(sec_nodes) == 0:
            log.warning(
                f"/INTER/TYPE5/{self.id}: secondary node group {grnod_id} "
                f"is missing or empty — interface inactive",
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
            f"Initialized /INTER/TYPE5/{self.id} '{self.title}' "
            f"({len(self.nodes)} secondary nodes vs {len(self.segs)} master segments, "
            f"MFROT={self.mfrot}, IFQ={self.ifq})"
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
        """Compute node-to-surface penalty contact forces with nonlinear friction.

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

        closest_idx = np.empty(n_sec, dtype=np.int64)
        chunk_size = max(1, 250000 // max(1, n_live))
        for start in range(0, n_sec, chunk_size):
            end = min(start + chunk_size, n_sec)
            diff = xs[start:end, None, :] - xc[None, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            closest_idx[start:end] = np.argmin(dist_sq, axis=1)

        matched_segs = live_segs[closest_idx]

        live_km = self.Km[live_mask] if len(self.Km) == len(self.segs) else np.full(n_live, self.stfac)
        K_pair = combine_stiffness(0, self.stfac, live_km[closest_idx], self.Ks)

        fsec, fmain, active, stif, H, dt_step = _t5_forces(
            x,
            v,
            mass,
            self.nodes,
            matched_segs,
            self.stfac,
            self.gap,
            self.stiff_dc,
            self.fric,
            self.mfrot,
            self.fric_c,
            self.ibc1,
            self.ibc2,
            self.ibc3,
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

        # Optional tangential force filter (IFQ > 0)
        if self.ifq > 0 and self.xfiltr > 0.0:
            alpha = friction.filter_alpha(self.ifq, self.xfiltr, dt)
            keys = act_sec * max(len(self.segs), 1) + closest_idx[active]
            act_fsec, self._filt_keys, self._filt_vals = friction.apply_filter(
                keys, act_fsec, alpha, self._filt_keys, self._filt_vals
            )
            for k in range(4):
                act_fmain[:, k] = - H[active, k:k+1] * act_fsec

        np.add.at(fcont, act_sec, act_fsec)
        np.add.at(fcont, act_segs[:, 0], act_fmain[:, 0])
        np.add.at(fcont, act_segs[:, 1], act_fmain[:, 1])
        np.add.at(fcont, act_segs[:, 2], act_fmain[:, 2])
        valid4 = act_segs[:, 3] >= 0
        if np.any(valid4):
            np.add.at(fcont, act_segs[valid4, 3], act_fmain[valid4, 3])

        if stifn is not None:
            np.add.at(stifn, act_sec, act_stif)
            for k in range(3):
                np.add.at(stifn, act_segs[:, k], H[active, k] * act_stif)
            if np.any(valid4):
                np.add.at(stifn, act_segs[valid4, 3], H[active][valid4, 3] * act_stif[valid4])

        dt_bound = min(self.dt_bound, dt_step)
        return fcont, dt_bound
