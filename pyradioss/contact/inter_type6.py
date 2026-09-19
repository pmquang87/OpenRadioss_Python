"""
/INTER/TYPE6 — Nonlinear Surface-to-Surface Contact Interface.

Fortran origin:
  - ``engine/source/interfaces/inter3d/i6main.F`` (interface coordinator)
  - ``engine/source/interfaces/inter3d/i6for3.F`` (contact forces and penetration check)
  - ``engine/source/interfaces/inter3d/i6ini3.F`` (initialization)
  - ``starter/source/interfaces/inter3d1/i6pen3.F`` (initial penetration check)
  - ``starter/source/interfaces/inter3d1/i6sti3.F`` (interface stiffness calculation)
  - ``hm_cfg_files/config/CFG/radioss2017/INTER/inter_type6.cfg`` (card reading)

Physics:
  Nonlinear penalty surface-to-surface contact interface with user-defined
  force vs penetration curve:
  1. Main and secondary surfaces (or secondary node groups):
     Segments (quads and triangles) are defined on the surfaces.
  2. Narrow phase computes closest projection point of secondary nodes on
     master segments with partition-of-unity barycentric weights (H1..H4).
  3. When penetration pen = gap - dist > 0:
     - User curve force/penetration function lookup:
       If funct_id > 0 is defined in model.functions:
         pen_scaled = pen * facx
         Fn_elastic = fac * func.eval(pen_scaled)
       Otherwise:
         Fn_elastic = K * pen
     - Viscous damping: opposes approaching relative velocity (vn < 0)
       C = visc * sqrt(2.0 * K * m_sec)
       Fn = max(0.0, Fn_elastic - C * min(vn, 0.0))
     - Friction: Coulomb friction Ft <= fric * Fn opposing sliding.
  4. Exact linear momentum conservation:
     Secondary node receives +Fvec, master corners receive -H_k * Fvec.
     Since sum(H_k) == 1.0, sum(F) == 0 to machine precision.
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


class ContactType6:
    """One /INTER/TYPE6 nonlinear surface-to-surface interface, engine-side."""

    def __init__(self, itf, model: Model, log):
        self.itf = itf
        self.model = model
        self.log = log
        self.id = getattr(itf, "id", 0)
        self.title = getattr(itf, "title", f"TYPE6_{self.id}")
        self.tstart = float(getattr(itf, "tstart", 0.0) or 0.0)
        self.tstop = float(getattr(itf, "tstop", 1e30) or 1e30)
        params = getattr(itf, "params", {}) or {}
        self.gap_input = float(params.get("gap", getattr(itf, "gap", 0.0)) or 0.0)
        self.fric = float(params.get("fric", getattr(itf, "fric", 0.0)) or 0.0)
        self.visc = float(params.get("visc", getattr(itf, "visc", 0.0)) or 0.0)
        self.stfac = float(params.get("stfac", getattr(itf, "stfac", 1.0)) or 1.0)
        if self.stfac <= 0.0:
            self.stfac = 1.0

        # User curve function lookup parameters (fct_IDId, Ascalex, Fscaleid)
        self.funct_id = (
            getattr(itf, "funct_id", None)
            or params.get("funct_id")
            or params.get("fct_idld")
            or params.get("fun_a1")
            or getattr(itf, "fun_a1", None)
            or getattr(itf, "fct_idld", None)
            or 0
        )
        self.facx = float(params.get("facx", getattr(itf, "facx", 1.0)) or 1.0)
        self.fac = float(params.get("fac", getattr(itf, "fac", 1.0)) or 1.0)
        self.iform = int(params.get("iform", getattr(itf, "iform", 0)) or 0)
        self.stiff = float(params.get("stiff", getattr(itf, "stiff", 0.0)) or 0.0)

        # Master surface lookup
        surf_m_id = getattr(itf, "surf_id1", 0) or getattr(itf, "surf_id", 0) or getattr(itf, "main_id", 0)
        surf_s_id = getattr(itf, "surf_id", 0) or getattr(itf, "sec_id", 0)
        if hasattr(itf, "surf_id1") and getattr(itf, "surf_id1", 0) > 0:
            surf_m_id = getattr(itf, "surf_id1", 0)
            surf_s_id = getattr(itf, "surf_id", 0)
        elif hasattr(itf, "surf_id") and getattr(itf, "surf_id", 0) > 0:
            surf_m_id = getattr(itf, "surf_id", 0)

        surf = model.surfaces.get(surf_m_id)
        if surf is None or surf.segments is None or len(surf.segments) == 0:
            surf_alt = model.surfaces.get(surf_s_id)
            if surf_alt is not None and surf_alt.segments is not None and len(surf_alt.segments) > 0:
                surf = surf_alt
                surf_s_id = surf_m_id
            else:
                log.warning(
                    f"/INTER/TYPE6/{self.id}: main surface {surf_m_id} "
                    f"is missing or empty — interface inactive",
                    "CONTACT INIT",
                )
                self._init_empty()
                return

        self.segs = np.asarray(surf.segments, dtype=np.int64)
        if self.segs.ndim == 2:
            if self.segs.shape[1] == 3:
                self.segs = np.column_stack([self.segs, self.segs[:, 2]])
            elif self.segs.shape[1] == 4:
                tri_mask = (self.segs[:, 3] < 0) | (self.segs[:, 3] == self.segs[:, 2])
                if np.any(tri_mask):
                    self.segs = self.segs.copy()
                    self.segs[tri_mask, 3] = self.segs[tri_mask, 2]

        self.seg_gtype = (
            surf.seg_gtype
            if surf.seg_gtype is not None
            else np.zeros(len(self.segs), dtype="<U8")
        )
        self.seg_elem = (
            surf.seg_elem
            if surf.seg_elem is not None
            else np.full(len(self.segs), -1, dtype=np.int64)
        )

        # Secondary node lookup: from node group or secondary surface
        sec_nodes = None
        grnod_id = getattr(itf, "grnod_id", 0)
        if grnod_id > 0 and grnod_id in model.node_groups:
            grp = model.node_groups[grnod_id]
            if grp.node_idx is not None and len(grp.node_idx) > 0:
                sec_nodes = np.asarray(grp.node_idx, dtype=np.int64)

        if sec_nodes is None and surf_s_id > 0 and surf_s_id in model.surfaces:
            surf_sec = model.surfaces[surf_s_id]
            if surf_sec.segments is not None and len(surf_sec.segments) > 0:
                s_segs = np.asarray(surf_sec.segments, dtype=np.int64)
                valid_s = s_segs[s_segs >= 0]
                if len(valid_s) > 0:
                    sec_nodes = np.unique(valid_s)

        if sec_nodes is None or len(sec_nodes) == 0:
            # Self-contact fallback: unique nodes of master surface
            valid_m = self.segs[self.segs >= 0]
            if len(valid_m) > 0:
                sec_nodes = np.unique(valid_m)
            else:
                log.warning(
                    f"/INTER/TYPE6/{self.id}: no secondary nodes found — interface inactive",
                    "CONTACT INIT",
                )
                self._init_empty()
                return

        self.nodes = np.sort(np.asarray(sec_nodes, dtype=np.int64))

        # Defensive model.x0 population
        if getattr(model, "x0", None) is None or len(model.x0) < (len(model.x) if hasattr(model, "x") and model.x is not None else 0):
            if hasattr(model, "x") and model.x is not None:
                model.x0 = model.x.copy()

        try:
            Km, gm = segment_stiffness_gap(model, self.segs, self.seg_gtype, self.seg_elem, self.stfac)
            self.Km = Km
            if len(self.Km) > 0 and np.all(self.Km <= 0.0):
                self.Km = np.full(len(self.segs), max(self.stfac, 1.0))
        except Exception:
            self.Km = np.full(len(self.segs), max(self.stfac, 1.0))
            gm = np.zeros(len(self.segs))

        try:
            Ks_all, gs_all = node_stiffness_gap(model, self.stfac)
            self.Ks = np.zeros(len(self.nodes))
            if len(Ks_all) > 0 and len(self.nodes) > 0:
                valid_ks = (self.nodes >= 0) & (self.nodes < len(Ks_all))
                self.Ks[valid_ks] = Ks_all[self.nodes[valid_ks]]
            if len(self.Ks) > 0 and np.all(self.Ks <= 0.0):
                self.Ks = np.full(len(self.nodes), max(self.stfac, 1.0))
        except Exception:
            self.Ks = np.full(len(self.nodes), max(self.stfac, 1.0))
            gs_all = np.zeros(len(model.x) if hasattr(model, "x") and model.x is not None else len(self.nodes))

        self.gap_m = gm
        self.gap_s = np.zeros(len(self.nodes))
        if len(gs_all) > 0 and len(self.nodes) > 0:
            valid_gs = (self.nodes >= 0) & (self.nodes < len(gs_all))
            self.gap_s[valid_gs] = gs_all[self.nodes[valid_gs]]

        self.gap_bound = self.gap_input
        if self.gap_bound <= 0.0:
            if len(gm) > 0:
                self.gap_bound = float(np.mean(gm))
            elif len(gs_all) > 0:
                self.gap_bound = float(np.mean(gs_all))
            else:
                self.gap_bound = 1e-3

        # Broad phase candidate caching
        self.pairs_node = np.zeros(0, dtype=np.int64)
        self.pairs_seg = np.zeros(0, dtype=np.int64)
        self._last_refresh = -10**9
        self.refresh = 20

        # Deletion tracking
        self.idel = int(getattr(itf, "idel", 0) or 0)
        self.deletable = self.idel >= 1 and tracking.any_deletable(
            model, self.seg_gtype, sec_nodes=self.nodes
        )
        if self.deletable:
            self.ref_total = tracking.node_reference_counts(model, alive_only=False)
        self.seg_alive = np.ones(len(self.segs), dtype=bool)
        self.nodes_tracked = self.nodes

        mass0 = getattr(model, "mass0", None)
        m_eff = mass0 if mass0 is not None and len(mass0) > 0 else model.mass
        self.dt_bound = self._compute_dt_bound(m_eff)

    def _init_empty(self):
        self.segs = np.zeros((0, 4), dtype=np.int64)
        self.seg_gtype = np.zeros(0, dtype="<U8")
        self.seg_elem = np.zeros(0, dtype=np.int64)
        self.nodes = np.zeros(0, dtype=np.int64)
        self.Km = np.zeros(0, dtype=float)
        self.Ks = np.zeros(0, dtype=float)
        self.gap_bound = 0.0
        self.dt_bound = np.inf
        self.pairs_node = np.zeros(0, dtype=np.int64)
        self.pairs_seg = np.zeros(0, dtype=np.int64)
        self._last_refresh = -10**9
        self.refresh = 20
        self.seg_alive = np.zeros(0, dtype=bool)
        self.nodes_tracked = np.zeros(0, dtype=np.int64)
        self.deletable = False

    def _compute_dt_bound(self, mass):
        if len(self.nodes) == 0 or len(self.Km) == 0:
            return np.inf
        K_ref = float(np.max(self.Km)) if len(self.Km) > 0 else 1.0
        m_valid = mass[self.nodes[self.nodes < len(mass)]]
        m_min = float(np.min(m_valid[m_valid > 0])) if np.any(m_valid > 0) else 1.0
        return float(np.sqrt(2.0 * m_min / max(K_ref, EM20)))

    def _broad_phase(self, x, dt):
        if len(self.nodes) == 0 or len(self.segs) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

        n_sec = len(self.nodes)
        n_seg = len(self.segs)

        # Segment centroids
        x_seg = x[self.segs]
        xc = np.mean(x_seg, axis=1)
        # Approximate segment bounding radii
        seg_rad = np.max(norm3(x_seg - xc[:, None, :]), axis=1)

        xs = x[self.nodes]
        gap = self.gap_bound

        # Distance matrix / chunked search
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

    def forces(self, x, v, mass, dt, fcont, cycle, stifn=None, t=0.0):
        """Compute /INTER/TYPE6 nonlinear contact forces.

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
        t : float
            Current simulation time.

        Returns
        -------
        fcont : np.ndarray (N, 3)
            The modified contact force array.
        dt_bound : float
            Interface stability time step bound.
        """
        if len(self.nodes) == 0 or len(self.segs) == 0 or dt <= 0.0:
            return fcont, self.dt_bound

        if t is not None and (t < self.tstart or t > self.tstop):
            return fcont, self.dt_bound

        # Deletion update
        if self.deletable:
            tracking.update_contact_alive(
                self.model, self.seg_elem, self.seg_gtype, self.seg_alive,
                self.nodes_tracked, self.ref_total,
            )
            if not np.any(self.seg_alive):
                return fcont, self.dt_bound

        # Refresh candidate pairs
        if cycle - self._last_refresh >= self.refresh or len(self.pairs_node) == 0:
            self.pairs_node, self.pairs_seg = self._broad_phase(x, dt)
            self._last_refresh = cycle

        if len(self.pairs_node) == 0:
            return fcont, self.dt_bound

        p_node = self.pairs_node
        p_seg = self.pairs_seg

        if self.deletable:
            alive_mask = self.seg_alive[p_seg]
            p_node = p_node[alive_mask]
            p_seg = p_seg[alive_mask]
            if len(p_node) == 0:
                return fcont, self.dt_bound

        ni = self.nodes[p_node]
        seg = self.segs[p_seg]

        # Ignore self-node on segment corner
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

        # Gap calculation
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

        # Per-pair baseline stiffness
        if self.stiff > 0.0:
            K = np.full(len(act_p_seg), self.stiff, dtype=float)
        else:
            istf = int(getattr(self.itf, "istf", 0) or 0)
            K = combine_stiffness(istf, self.stfac, self.Km[act_p_seg], self.Ks[act_p_node])
            K = np.where(K > 0.0, K, max(self.stfac, 1.0))

        # Normal vector pointing from master to secondary node
        d = np.maximum(best_d, EM20)
        nvec = (x[ni] - best_pt) / d[:, None]

        # Master face normal fallback for degenerate/exact zero distance
        d13 = x[seg[:, 2]] - x[seg[:, 0]]
        d24 = x[seg[:, 3]] - x[seg[:, 1]]
        n_raw = cross3(d13, d24)
        n_norm = norm3(n_raw)
        n_seg = np.where(
            (n_norm > EM20)[:, None],
            n_raw / np.maximum(n_norm, EM20)[:, None],
            np.array([0.0, 0.0, 1.0]),
        )
        nvec = np.where((best_d <= EM20)[:, None], n_seg, nvec)

        # User curve function lookup for normal force vs penetration
        func = None
        if self.funct_id and hasattr(self.model, "functions"):
            func = self.model.functions.get(self.funct_id)

        if func is not None:
            # Curve lookup: f(penetration * facx) * fac
            p_eval = pen * self.facx
            if hasattr(func, "eval"):
                f_val = np.asarray(func.eval(p_eval), dtype=float)
            elif callable(func):
                f_val = np.asarray(func(p_eval), dtype=float)
            else:
                f_val = K * pen
            Fn_elastic = np.maximum(0.0, f_val * self.fac)
            # Effective secant stiffness for damping and dt bound
            K_eff = np.where(pen > EM20, Fn_elastic / np.maximum(pen, EM20), K)
        else:
            Fn_elastic = K * pen
            K_eff = K

        # Relative velocity: v_sec - v_master_proj
        vseg = np.einsum("nk,nkb->nb", wseg, v[seg])
        vrel = v[ni] - vseg
        vn = np.einsum("nb,nb->n", vrel, nvec)

        # Viscous damping: opposes approach (vn < 0)
        if self.visc > 0.0:
            C = self.visc * np.sqrt(2.0 * K_eff * mass[ni])
            Fn_damp = -C * np.minimum(vn, 0.0)
            Fn = np.maximum(0.0, Fn_elastic + Fn_damp)
        else:
            Fn = Fn_elastic

        # Tangential friction
        Fvec = Fn[:, None] * nvec
        if self.fric > 0.0:
            vt = vrel - vn[:, None] * nvec
            vt_mag = norm3(vt)
            v_ref = np.maximum(1e-4 * gap / max(dt, EM20), EM20)
            Ft_mag = self.fric * Fn * vt_mag / (vt_mag + v_ref)
            t_dir = np.where(
                (vt_mag > EM20)[:, None],
                vt / np.maximum(vt_mag, EM20)[:, None],
                np.zeros_like(vt),
            )
            Fvec -= Ft_mag[:, None] * t_dir

        # EXACT Linear Momentum Conservation:
        # Secondary node gets +Fvec
        # Master segment corners get -H_k * Fvec
        # Since sum(H_k) == 1.0, sum of all distributed forces is 0 to machine precision.
        n_nod = len(fcont)
        valid_ni = (ni >= 0) & (ni < n_nod)
        if np.any(valid_ni):
            scatter_add3(fcont, ni[valid_ni], Fvec[valid_ni])

        w_F = (-wseg[:, :, None] * Fvec[:, None, :]).reshape(-1, 3)
        seg_flat = seg.reshape(-1)
        valid_sf = (seg_flat >= 0) & (seg_flat < n_nod)
        if np.any(valid_sf):
            scatter_add3(fcont, seg_flat[valid_sf], w_F[valid_sf])

        # Stiffness accumulation for /DT/NODA and time step bound
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
