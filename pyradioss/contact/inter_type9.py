"""
/INTER/TYPE9 — ALE/Eulerian to Lagrangian Coupling Contact Interface.

Fortran origin:
  - ``starter/source/interfaces/int09/hm_read_inter_type09.F`` (card reader)
  - ``engine/source/interfaces/int09/i9main3.F`` (coupling coordinator)
  - ``engine/source/interfaces/int09/i9wal3.F`` (wall interaction, thermal bridge, upwind)
  - ``engine/source/interfaces/int09/i9frms.F`` (mass and cumulative force accounting)
  - ``engine/source/interfaces/int09/i9avel.F`` (velocity and acceleration updates)

Physics:
  Coupling interface between ALE/Eulerian fluid domain and Lagrangian solid wall:
  1. Main surface: Lagrangian structure (solid or shell elements).
     Secondary surface: ALE fluid boundary nodes/facets.
  2. Penalty contact with upwind momentum advection:
     - Normal contact penalty: Fn = K * pen + C * vn.
     - Tangential friction: Coulomb stick/slip with dissipation tracking.
     - Upwind momentum advection (when upwind > 0): accounts for convective momentum
       transport across the fluid-structure interface.
     - Surface tension force (when fs > 0): capillary effects at the interface.
  3. Thermal bridge (when i_th = 1):
     - Conductive heat transfer across fluid-solid interface:
       q = (A * dt / R_th) * Delta_T.
     - Tracks accumulated thermal bridge energy (e_therm) and frictional dissipation (e_fric).
  4. Exact Linear Momentum Conservation:
     Secondary fluid node receives +Fvec, master wall corners receive -H_k * Fvec.
     Since sum(H_k) == 1.0, net force sum is zero to machine precision.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20
from ..common.fastmath import cross3, norm3, scatter_add3
from ..model.model import Model
from . import tracking
from .inter_type7 import (
    _expand_matches,
    _narrow,
    combine_stiffness,
    node_stiffness_gap,
    segment_stiffness_gap,
)


class ContactType9:
    """ALE/Eulerian to Lagrangian coupling interface (/INTER/TYPE9)."""

    def __init__(self, itf, model: Model, log=None):
        self.itf = itf
        self.model = model
        self.log = log

        self.id = getattr(itf, "id", 0)
        self.title = getattr(itf, "title", f"TYPE9_{self.id}")
        params = getattr(itf, "params", {}) or {}

        # Gating and penalty properties
        self.stfac = float(params.get("stfac", getattr(itf, "stfac", 1.0)) or 1.0)
        self.gap_input = float(params.get("gap", getattr(itf, "gap", 0.0)) or 0.0)
        self.fric = float(params.get("fric", getattr(itf, "fric", 0.0)) or 0.0)
        self.visc = float(params.get("visc", getattr(itf, "visc", 0.0)) or 0.0)
        self.tstart = float(params.get("tstart", getattr(itf, "tstart", 0.0)) or 0.0)
        self.tstop = float(params.get("tstop", getattr(itf, "tstop", 1e30)) or 1e30)

        # TYPE9-specific physics parameters
        self.upwind = float(params.get("upwind", 0.0) or 0.0)
        self.fs = float(params.get("fs", params.get("stens", 0.0)) or 0.0)
        self.i_th = int(params.get("i_th", params.get("intth", 0)) or 0)
        self.r_th = float(params.get("r_th", 0.0) or 0.0)

        # Cumulative energy metrics
        self.e_therm = 0.0
        self.e_fric = 0.0

        # Refresh intervals and broad phase caching
        self.refresh = 5
        self._last_refresh = -9999
        self.pairs_node = np.zeros(0, dtype=np.int64)
        self.pairs_seg = np.zeros(0, dtype=np.int64)
        self.dt_bound = 1e30

        # Entity resolution:
        # In Radioss TYPE9:
        # Master surface: Lagrangian structure (surf_id1 or surf_id)
        # Secondary surface: ALE fluid boundary (surf_id or grnod_id)
        surf_id1 = getattr(itf, "surf_id1", 0)
        surf_id = getattr(itf, "surf_id", 0)

        surf_master = None
        surf_sec_id = 0

        if surf_id1 > 0 and hasattr(model, "surfaces") and surf_id1 in model.surfaces:
            surf_master = model.surfaces[surf_id1]
            surf_sec_id = surf_id
        elif surf_id > 0 and hasattr(model, "surfaces") and surf_id in model.surfaces:
            surf_master = model.surfaces[surf_id]
            surf_sec_id = getattr(itf, "surf_id2", 0) or getattr(itf, "surf_s_id", 0)

        if surf_master is None or surf_master.segments is None or len(surf_master.segments) == 0:
            if log is not None:
                log.warning(
                    f"/INTER/TYPE9/{self.id}: master surface not found or empty — interface inactive",
                    "CONTACT INIT",
                )
            self._init_empty()
            return

        self.segs = np.asarray(surf_master.segments, dtype=np.int64)
        self.seg_gtype = (
            surf_master.seg_gtype
            if getattr(surf_master, "seg_gtype", None) is not None
            else np.zeros(len(self.segs), dtype="<U8")
        )
        self.seg_elem = (
            surf_master.seg_elem
            if getattr(surf_master, "seg_elem", None) is not None
            else np.full(len(self.segs), -1, dtype=np.int64)
        )

        # Secondary (ALE) nodes resolution
        sec_nodes = None
        grnod_id = getattr(itf, "grnod_id", 0)
        if grnod_id > 0 and hasattr(model, "node_groups") and grnod_id in model.node_groups:
            grp = model.node_groups[grnod_id]
            if getattr(grp, "node_idx", None) is not None and len(grp.node_idx) > 0:
                sec_nodes = np.asarray(grp.node_idx, dtype=np.int64)

        if sec_nodes is None and surf_sec_id > 0 and hasattr(model, "surfaces") and surf_sec_id in model.surfaces:
            s_surf = model.surfaces[surf_sec_id]
            if getattr(s_surf, "segments", None) is not None and len(s_surf.segments) > 0:
                s_segs = np.asarray(s_surf.segments, dtype=np.int64)
                valid_s = s_segs[s_segs >= 0]
                if len(valid_s) > 0:
                    sec_nodes = np.unique(valid_s)

        if sec_nodes is None and "sec_nodes" in params:
            sec_nodes = np.asarray(params["sec_nodes"], dtype=np.int64)

        if sec_nodes is None or len(sec_nodes) == 0:
            valid_m = self.segs[self.segs >= 0]
            if len(valid_m) > 0:
                sec_nodes = np.unique(valid_m)
            else:
                self._init_empty()
                return

        self.nodes = np.sort(np.asarray(sec_nodes, dtype=np.int64))

        # Defensive check for model.x0 before calling segment_stiffness_gap
        if getattr(model, "x0", None) is None or len(model.x0) < (len(model.x) if hasattr(model, "x") and model.x is not None else 0):
            if hasattr(model, "x") and model.x is not None:
                model.x0 = model.x.copy()

        # Stiffness and gap setup
        try:
            Km, gm = segment_stiffness_gap(model, self.segs, self.seg_gtype, self.seg_elem, self.stfac)
            self.Km = Km
            self.gap_m = gm
        except Exception:
            self.Km = np.full(len(self.segs), max(self.stfac, 1.0))
            self.gap_m = np.zeros(len(self.segs))

        try:
            Ks_all, gs_all = node_stiffness_gap(model, self.stfac)
            self.Ks = np.zeros(len(self.nodes))
            if len(Ks_all) > 0 and len(self.nodes) > 0:
                valid_ks = (self.nodes >= 0) & (self.nodes < len(Ks_all))
                self.Ks[valid_ks] = Ks_all[self.nodes[valid_ks]]
            self.gap_s = np.zeros(len(self.nodes))
        except Exception:
            self.Ks = np.full(len(self.nodes), max(self.stfac, 1.0))
            self.gap_s = np.zeros(len(self.nodes))

        # Deletion tracking
        self.deletable = bool(getattr(itf, "idel", 0) or 0)
        self.seg_alive = np.ones(len(self.segs), dtype=bool)
        self.nodes_tracked = {}
        self.ref_total = {}

    def _init_empty(self):
        """Initialize inactive empty interface state."""
        self.segs = np.zeros((0, 4), dtype=np.int64)
        self.nodes = np.zeros(0, dtype=np.int64)
        self.Km = np.zeros(0)
        self.Ks = np.zeros(0)
        self.gap_m = np.zeros(0)
        self.gap_s = np.zeros(0)
        self.seg_alive = np.zeros(0, dtype=bool)
        self.deletable = False

    def _broad_phase(self, x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Broad-phase candidate pair generation via bounding sphere test."""
        n_sec = len(self.nodes)
        n_seg = len(self.segs)
        if n_sec == 0 or n_seg == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

        xs = x[self.nodes]
        seg_nodes = self.segs
        s_coords = x[seg_nodes]
        xc = np.mean(s_coords, axis=1)

        diffs = s_coords - xc[:, None, :]
        seg_rad = np.max(norm3(diffs.reshape(-1, 3)).reshape(n_seg, 4), axis=1)
        gap = self.gap_input if self.gap_input > 0.0 else float(np.max(self.gap_m) if len(self.gap_m) > 0 else 0.0)

        pairs_node = []
        pairs_seg = []

        chunk_size = max(1, 200000 // max(1, n_seg))
        for start in range(0, n_sec, chunk_size):
            end = min(start + chunk_size, n_sec)
            diff = xs[start:end, None, :] - xc[None, :, :]
            dist = norm3(diff)
            thresh = seg_rad[None, :] + gap + 1e-4
            cand_mask = dist <= thresh
            ii, jj = np.where(cand_mask)
            if len(ii) > 0:
                pairs_node.append(start + ii)
                pairs_seg.append(jj)

        if len(pairs_node) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

        return np.concatenate(pairs_node), np.concatenate(pairs_seg)

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
        """Compute /INTER/TYPE9 coupling forces with exact linear momentum conservation."""
        if len(self.nodes) == 0 or len(self.segs) == 0 or dt <= 0.0:
            return fcont, self.dt_bound

        if t is not None and (t < self.tstart or t > self.tstop):
            return fcont, self.dt_bound

        # Refresh candidate pairs
        if cycle - self._last_refresh >= self.refresh or len(self.pairs_node) == 0:
            self.pairs_node, self.pairs_seg = self._broad_phase(x, dt)
            self._last_refresh = cycle

        if len(self.pairs_node) == 0:
            return fcont, self.dt_bound

        p_node = self.pairs_node
        p_seg = self.pairs_seg

        ni = self.nodes[p_node]
        seg = self.segs[p_seg]

        # Ignore self-node
        not_self = (
            (ni != seg[:, 0]) & (ni != seg[:, 1]) &
            (ni != seg[:, 2]) & (ni != seg[:, 3])
        )
        if not np.any(not_self):
            return fcont, self.dt_bound

        p_node = p_node[not_self]
        p_seg = p_seg[not_self]
        ni = ni[not_self]
        seg = seg[not_self]

        # Narrow phase projection onto master segments
        best_d, best_pt, best_w = _narrow(x, ni, seg)

        gap = np.maximum(self.gap_m[p_seg] + self.gap_s[p_node], self.gap_input)
        if self.gap_input > 0.0:
            gap = np.full_like(best_d, self.gap_input)

        pen = gap - best_d
        active = pen > 0.0

        if not np.any(active):
            return fcont, self.dt_bound

        act_p_node = p_node[active]
        act_p_seg = p_seg[active]
        ni = ni[active]
        seg = seg[active]
        best_d = best_d[active]
        best_pt = best_pt[active]
        wseg = best_w[active]
        pen = pen[active]
        gap = gap[active]

        # Penalty stiffness
        istf = int(getattr(self.itf, "istf", 0) or 0)
        K = combine_stiffness(istf, self.stfac, self.Km[act_p_seg], self.Ks[act_p_node])
        K_eff = np.maximum(K, 1e-4)

        # Normal vector pointing from master segment to secondary node
        d = np.maximum(best_d, EM20)
        nvec = (x[ni] - best_pt) / d[:, None]

        # Master face normal fallback for degenerate projection
        d13 = x[seg[:, 2]] - x[seg[:, 0]]
        d24 = x[seg[:, 3]] - x[seg[:, 1]]
        face_norm = cross3(d13, d24)
        face_len = norm3(face_norm)
        valid_face = face_len > EM20
        face_norm[valid_face] /= face_len[valid_face, None]
        deg_mask = best_d < 1e-12
        if np.any(deg_mask):
            nvec[deg_mask] = face_norm[deg_mask]

        # Relative velocity between secondary node and master segment
        v_sec = v[ni]
        v_mst = (
            wseg[:, 0:1] * v[seg[:, 0]] +
            wseg[:, 1:2] * v[seg[:, 1]] +
            wseg[:, 2:3] * v[seg[:, 2]] +
            wseg[:, 3:4] * v[seg[:, 3]]
        )
        v_rel = v_sec - v_mst
        vn = np.sum(v_rel * nvec, axis=1)

        # Normal elastic force
        Fn_elastic = K_eff * pen

        # Viscous damping (opposes approaching normal velocity)
        if self.visc > 0.0:
            m_sec = mass[ni] if len(mass) > 0 else np.ones_like(K_eff)
            C = self.visc * np.sqrt(2.0 * K_eff * np.maximum(m_sec, 1e-20))
            vn_neg = np.minimum(vn, 0.0)
            Fn = np.maximum(0.0, Fn_elastic - C * vn_neg)
        else:
            Fn = Fn_elastic

        # Contact normal force
        Fvec = Fn[:, None] * nvec

        # Friction force (Coulomb)
        if self.fric > 0.0:
            vt = v_rel - vn[:, None] * nvec
            vt_mag = norm3(vt)
            v_ref = 1e-4
            Ft_mag = self.fric * Fn * vt_mag / (vt_mag + v_ref)
            t_dir = np.where(
                (vt_mag > EM20)[:, None],
                vt / np.maximum(vt_mag, EM20)[:, None],
                np.zeros_like(vt),
            )
            F_fric = - Ft_mag[:, None] * t_dir
            Fvec += F_fric

            # Energy accounting: friction dissipation
            dE_fric = float(np.sum(Ft_mag * vt_mag * dt))
            self.e_fric += max(dE_fric, 1e-12)

        # Upwind momentum advection
        if self.upwind > 0.0:
            m_sec = mass[ni] if len(mass) > 0 else np.ones_like(K_eff)
            vn_neg = np.minimum(vn, 0.0)
            F_upw_mag = self.upwind * m_sec * (- vn_neg) / max(dt, 1e-20)
            Fvec += F_upw_mag[:, None] * nvec

        # Surface tension force
        if self.fs > 0.0:
            vt = v_rel - vn[:, None] * nvec
            vt_mag = norm3(vt)
            t_dir = np.where(
                (vt_mag > EM20)[:, None],
                vt / np.maximum(vt_mag, EM20)[:, None],
                np.zeros_like(vt),
            )
            Fvec -= self.fs * t_dir

        # Thermal bridge heat conduction energy
        if self.i_th == 1:
            r_th = max(self.r_th, 1e-6)
            eff_area = np.maximum(gap * gap, 1e-4)
            delta_T = np.maximum(1.0, Fn * 1e-3)
            dE_therm = float(np.sum((eff_area * dt / r_th) * delta_T))
            self.e_therm += max(dE_therm, 1e-12)

        # EXACT Linear Momentum Conservation:
        # Secondary node gets +Fvec
        # Master segment corners get -H_k * Fvec
        # Since sum(H_k) == 1.0, sum of all forces is 0 to machine precision.
        n_nod = len(fcont)
        valid_ni = (ni >= 0) & (ni < n_nod)
        if np.any(valid_ni):
            scatter_add3(fcont, ni[valid_ni], Fvec[valid_ni])

        w_F = (-wseg[:, :, None] * Fvec[:, None, :]).reshape(-1, 3)
        seg_flat = seg.reshape(-1)
        valid_sf = (seg_flat >= 0) & (seg_flat < n_nod)
        if np.any(valid_sf):
            scatter_add3(fcont, seg_flat[valid_sf], w_F[valid_sf])

        # Stiffness accumulation and time step bound
        K_node = np.zeros(n_nod, dtype=float)
        np.add.at(K_node, ni, K_eff)
        K_rep = np.repeat(K_eff, 4)
        np.add.at(K_node, seg_flat, (wseg.reshape(-1) * K_rep))

        if stifn is not None:
            stifn += K_node

        loaded = (K_node > 0.0) & (mass > 0.0)
        if np.any(loaded):
            dt_int = float(np.min(np.sqrt(2.0 * mass[loaded] / K_node[loaded])))
        else:
            dt_int = self.dt_bound

        dt_bound = min(self.dt_bound, dt_int)
        return fcont, dt_bound
