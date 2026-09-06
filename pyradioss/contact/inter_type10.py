"""
/INTER/TYPE10 — Auto-Impacting Penalty Tied Contact Interface (M199/M492).

Fortran origin:
  - ``engine/source/interfaces/int10/i10mainf.F`` (interface coordinator)
  - ``engine/source/interfaces/int10/i10dst3.F`` (penetration check & candidate filtering)
  - ``engine/source/interfaces/int10/i10for3.F`` (incremental penalty forces, frozen anchor coordinates, tied/rebound logic, damping)
  - ``engine/source/interfaces/int07/i7ass3.F`` (force assembly and momentum conservation)
  - ``starter/source/interfaces/int10/hm_read_inter_type10.F`` (card reading)

Physics:
Auto-impacting penalty tied contact interface:
  1. Broad phase (voxel sort) finds candidate secondary nodes near master segments.
  2. Narrow phase computes closest projection point on master segments.
  3. When impact occurs (penetration > 0 within gap):
     - Local shape functions (H1, H2, H3, H4) at the initial contact point are stored and FROZEN in history.
     - The anchor point x_anchor = sum(H_k * x_master,k) remains fixed to the material coordinates of the master segment.
  4. Local orthonormal frame (T1, T2, N) is constructed from master segment diagonals and normal.
  5. Incremental spring forces accumulate in normal and tangential directions:
     - F_n = F_n_old + v_n * dt * K
     - F_t1 = F_t1_old + v_t1 * dt * K
     - F_t2 = F_t2_old + v_t2 * dt * K
  6. Tied vs Rebound (Itied):
     - Itied = 0: Rebound permitted. If normal force changes sign (compression -> tension) and node is outside gap,
       the tie separates and force history resets to zero.
     - Itied = 1: Tied permanently. Once impacted, the tie persists indefinitely under both compression and tension.
  7. Viscous damping: C = stiff_dc * sqrt(2 * K * m_sec) opposes relative velocity, booking positive dissipation.
  8. Exact linear and angular momentum conservation:
     - Secondary node receives -F_total, master segment corners receive +H_k * F_total.
"""

from __future__ import annotations

import numpy as np

from ..accel import get as accel_get
from ..common.constants import EM20
from ..common.fastmath import scatter_add3
from ..model.model import Model
from . import tracking
from .inter_type7 import (
    _expand_matches,
    _narrow,
    combine_stiffness,
    node_stiffness_gap,
    segment_stiffness_gap,
)


class ContactType10:
    """One /INTER/TYPE10 auto-impacting penalty tied interface, engine-side."""

    def __init__(self, itf, model: Model, log):
        self.itf = itf
        self.model = model
        self.id = getattr(itf, "id", 0)
        self.title = getattr(itf, "title", f"TYPE10_{self.id}")
        self.tstart = float(getattr(itf, "tstart", 0.0) or 0.0)
        self.tstop = float(getattr(itf, "tstop", 1e30) or 1e30)
        self.itied = int(getattr(itf, "itied", 0) or 0)
        self.stiff_dc = float(getattr(itf, "stiff_dc", 0.0) or 0.0)
        self.gap_bound = float(getattr(itf, "gap", 0.0) or 0.0)
        self.time = 0.0

        # Master surface lookup with defensive handling
        surf = model.surfaces.get(getattr(itf, "surf_id", 0))
        if surf is None or surf.segments is None or len(surf.segments) == 0:
            log.warning(
                f"/INTER/TYPE10/{self.id}: main surface {getattr(itf, 'surf_id', 0)} "
                f"is missing or empty — interface inactive",
                "CONTACT INIT",
            )
            self._init_empty()
            return

        self.segs = np.asarray(surf.segments, dtype=np.int64)
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

        # Secondary node group lookup with defensive handling
        grp = model.node_groups.get(getattr(itf, "grnod_id", 0))
        if grp is None or grp.node_idx is None or len(grp.node_idx) == 0:
            log.warning(
                f"/INTER/TYPE10/{self.id}: secondary node group {getattr(itf, 'grnod_id', 0)} "
                f"is missing or empty — interface inactive",
                "CONTACT INIT",
            )
            self._init_empty()
            return

        self.nodes = np.sort(np.asarray(grp.node_idx, dtype=np.int64))

        # Stiffness calculation
        scale = float(itf.stfac) if getattr(itf, "stfac", 1.0) > 0 else 1.0
        Km, gm = segment_stiffness_gap(
            model, self.segs, self.seg_gtype, self.seg_elem, scale
        )
        self.Km = Km
        Ks_all, _ = node_stiffness_gap(model, scale)
        if len(Ks_all) > 0 and len(self.nodes) > 0 and self.nodes.max() < len(Ks_all):
            self.Ks = Ks_all[self.nodes]
        else:
            self.Ks = np.zeros(len(self.nodes))

        # Deletion tracking — Fortran chkstfn3.F:1352 gates on IDEL >= 1
        self.idel = int(getattr(itf, "idel10", 0) or getattr(itf, "idel", 0) or 0)
        self.deletable = self.idel >= 1 and tracking.any_deletable(
            model, self.seg_gtype, sec_nodes=self.nodes
        )
        if self.deletable:
            self.ref_total = tracking.node_reference_counts(model, alive_only=False)
        self.seg_alive = np.ones(len(self.segs), dtype=bool)
        self.nodes_tracked = self.nodes

        # Broad-phase bookkeeping
        self.pairs_node = np.zeros(0, dtype=np.int64)
        self.pairs_seg = np.zeros(0, dtype=np.int64)
        self._last_refresh = -10**9
        self.refresh = 20

        # History arrays for incremental forces & frozen anchor coordinates
        self._hist_keys = np.zeros(0, dtype=np.int64)
        self._hist_fn = np.zeros(0, dtype=np.float64)       # Normal force component
        self._hist_ft1 = np.zeros(0, dtype=np.float64)      # Tangential 1 force component
        self._hist_ft2 = np.zeros(0, dtype=np.float64)      # Tangential 2 force component
        self._hist_w = np.zeros((0, 4), dtype=np.float64)   # Frozen shape functions H1..H4
        self._hist_fvec = np.zeros((0, 3), dtype=np.float64)# 3D force vector

        self.e_cont = 0.0
        self.e_damp = 0.0
        mass0 = getattr(model, "mass0", None)
        m_eff = mass0 if mass0 is not None and len(mass0) > 0 else model.mass
        self.dt_bound = self._compute_dt_bound(m_eff)

        log.info(
            f"Initialized /INTER/TYPE10/{self.id} '{self.title}' "
            f"({len(self.nodes)} secondary nodes vs {len(self.segs)} main segments)"
        )

    def _init_empty(self) -> None:
        """Initialize empty state for inactive or missing interfaces."""
        self.segs = np.zeros((0, 4), dtype=np.int64)
        self.seg_gtype = np.zeros(0, dtype="<U8")
        self.seg_elem = np.zeros(0, dtype=np.int64)
        self.nodes = np.zeros(0, dtype=np.int64)
        self.Km = np.zeros(0, dtype=float)
        self.Ks = np.zeros(0, dtype=float)
        self.deletable = False
        self.seg_alive = np.zeros(0, dtype=bool)
        self.nodes_tracked = np.zeros(0, dtype=np.int64)
        self.pairs_node = np.zeros(0, dtype=np.int64)
        self.pairs_seg = np.zeros(0, dtype=np.int64)
        self._hist_keys = np.zeros(0, dtype=np.int64)
        self._hist_fn = np.zeros(0, dtype=float)
        self._hist_ft1 = np.zeros(0, dtype=float)
        self._hist_ft2 = np.zeros(0, dtype=float)
        self._hist_w = np.zeros((0, 4), dtype=float)
        self._hist_fvec = np.zeros((0, 3), dtype=float)
        self.e_cont = 0.0
        self.e_damp = 0.0
        self.dt_bound = np.inf

    def _compute_dt_bound(self, mass: np.ndarray) -> float:
        """Compute explicit time step bound based on interface stiffness."""
        if len(self.segs) == 0 or len(self.nodes) == 0:
            return np.inf
        Km_max = np.full(len(self.nodes), self.Km.max()) if len(self.Km) else np.zeros(len(self.nodes))
        K_sec = combine_stiffness(0, self.itf.stfac, Km_max, self.Ks)
        dt_sec = np.sqrt(2.0 * mass[self.nodes] / np.maximum(K_sec, EM20)).min()

        Ks_max = np.full(len(self.segs), self.Ks.max() if len(self.Ks) else 0.0)
        K_main = combine_stiffness(0, self.itf.stfac, self.Km, Ks_max)
        valid_segs = np.maximum(self.segs, 0)
        m_corner = mass[valid_segs].min(axis=1)
        dt_main = np.sqrt(2.0 * m_corner / np.maximum(K_main, EM20)).min()
        return float(min(dt_sec, dt_main))

    def _broad_phase(self, x: np.ndarray, v: np.ndarray, dt: float) -> None:
        """Voxel bucket search to identify candidate secondary-node / master-segment pairs."""
        segs = self.segs[self.seg_alive]
        seg_rows = np.where(self.seg_alive)[0]
        nodes = self.nodes_tracked
        if len(segs) == 0 or len(nodes) == 0:
            self.pairs_node = np.zeros(0, dtype=np.int64)
            self.pairs_seg = np.zeros(0, dtype=np.int64)
            return

        margin = self.gap_bound + 2.0 * self.refresh * dt * (
            np.abs(v).max() if len(v) else 0.0
        )

        valid_segs = np.maximum(segs, 0)
        xs = x[valid_segs]
        lo = xs.min(axis=1) - margin
        hi = xs.max(axis=1) + margin
        h = max(float((hi - lo).max()), EM20)
        origin = np.minimum(lo.min(axis=0), x[nodes].min(axis=0))

        ilo = np.floor((lo - origin) / h).astype(np.int64)
        ihi = np.floor((hi - origin) / h).astype(np.int64)
        inode = np.floor((x[nodes] - origin) / h).astype(np.int64)
        dims = np.maximum(np.maximum(ihi.max(axis=0), inode.max(axis=0)), 0) + 2

        def key(ijk):
            return (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]

        seg_ids, seg_keys = [], []
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ijk = ilo + np.array([dx, dy, dz])
                    inside = np.all(ijk <= ihi, axis=1)
                    seg_ids.append(np.where(inside)[0])
                    seg_keys.append(key(ijk[inside]))
        seg_ids = np.concatenate(seg_ids)
        seg_keys = np.concatenate(seg_keys)

        ii, jj = _expand_matches(key(inode), seg_keys)
        ni = nodes[ii]
        sj = seg_rows[seg_ids[jj]]

        # Retain tied pairs even if they move out of broad phase margin
        if len(self._hist_keys) > 0:
            n_segs = max(len(self.segs), 1)
            tied_ni = self._hist_keys // n_segs
            tied_sj = self._hist_keys % n_segs

            all_ni = np.concatenate([ni, tied_ni])
            all_sj = np.concatenate([sj, tied_sj])

            pair_keys = all_ni * n_segs + all_sj
            _, uniq_idx = np.unique(pair_keys, return_index=True)
            self.pairs_node = all_ni[uniq_idx]
            self.pairs_seg = all_sj[uniq_idx]
        else:
            self.pairs_node = ni
            self.pairs_seg = sj

    def forces(
        self,
        x: np.ndarray,
        v: np.ndarray,
        mass: np.ndarray,
        dt_or_fcont: float | np.ndarray = 0.0,
        fcont_or_cycle: np.ndarray | int | None = None,
        cycle_or_dt: int | float = 0,
        stifn: np.ndarray | None = None,
        t: float | None = None,
        dt: float | None = None,
        cycle: int | None = None,
        fcont: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[float, float]:
        """Penalty tied contact forces for one cycle, scattered into ``fcont``.

        Supports both standard Engine coordinator signature:
            ``forces(x, v, mass, dt, fcont, cycle, stifn=None, t=None)``
        and legacy test signature:
            ``forces(x, v, mass, fcont, cycle=0, dt=1e-4, t=0.0)``
        """
        if dt is not None:
            _dt = float(dt)
        elif isinstance(dt_or_fcont, (float, int, np.floating, np.integer)) and not isinstance(dt_or_fcont, np.ndarray):
            _dt = float(dt_or_fcont)
        else:
            _dt = float(cycle_or_dt)

        if fcont is not None:
            _fcont = fcont
        elif isinstance(dt_or_fcont, np.ndarray):
            _fcont = dt_or_fcont
        elif isinstance(fcont_or_cycle, np.ndarray):
            _fcont = fcont_or_cycle
        else:
            _fcont = None

        if cycle is not None:
            _cycle = int(cycle)
        elif isinstance(fcont_or_cycle, (int, np.integer)):
            _cycle = int(fcont_or_cycle)
        elif isinstance(cycle_or_dt, (int, np.integer)):
            _cycle = int(cycle_or_dt)
        else:
            _cycle = 0

        _t = kwargs.get("t", t)

        dt = _dt
        fcont = _fcont
        cycle = _cycle
        t = _t

        if fcont is None:
            return 0.0, self.dt_bound

        if len(self.segs) == 0 or len(self.nodes) == 0 or dt <= 0.0:
            return 0.0, self.dt_bound

        current_t = self.time if t is None else t
        self.time = current_t + dt
        if current_t < self.tstart or current_t > self.tstop:
            return 0.0, self.dt_bound

        if self.deletable:
            self.seg_alive = tracking.alive_segment_mask(
                self.model, self.seg_gtype, self.seg_elem
            )

        if cycle - self._last_refresh >= self.refresh:
            if self.deletable:
                mask = tracking.tracked_node_mask(self.model, self.ref_total)
                if len(mask) > 0 and len(self.nodes) > 0 and self.nodes.max() < len(mask):
                    self.nodes_tracked = self.nodes[mask[self.nodes]]
                else:
                    self.nodes_tracked = self.nodes
            self._broad_phase(x, v, dt)
            self._last_refresh = cycle

        if len(self.pairs_node) == 0:
            return 0.0, self.dt_bound

        live = self.seg_alive[self.pairs_seg]
        ni = self.pairs_node[live]
        srow = self.pairs_seg[live]
        if len(ni) == 0:
            return 0.0, self.dt_bound
        seg = self.segs[srow]

        n_segs = max(len(self.segs), 1)
        keys = ni * n_segs + srow

        # Separate candidates into already tied (in history) vs new candidates
        is_hist = np.isin(keys, self._hist_keys)

        n_pairs = len(ni)
        best_d = np.zeros(n_pairs)
        best_pt = np.zeros((n_pairs, 3))
        best_w = np.zeros((n_pairs, 4))
        pen = np.zeros(n_pairs)
        fn_old = np.zeros(n_pairs)
        ft1_old = np.zeros(n_pairs)
        ft2_old = np.zeros(n_pairs)

        # 1. Existing tied pairs: retrieve frozen anchor coordinates from history (i10for3.F lines 250-261)
        if np.any(is_hist):
            hist_sorter = np.argsort(self._hist_keys)
            match_idx = np.searchsorted(self._hist_keys, keys[is_hist], sorter=hist_sorter)
            hist_loc = hist_sorter[match_idx]

            best_w[is_hist] = self._hist_w[hist_loc]
            fn_old[is_hist] = self._hist_fn[hist_loc]
            ft1_old[is_hist] = self._hist_ft1[hist_loc]
            ft2_old[is_hist] = self._hist_ft2[hist_loc]

            # Compute anchor point using frozen shape functions
            seg_hist = seg[is_hist]
            valid_hist = np.maximum(seg_hist, 0)
            x_corners = x[valid_hist]
            pt_frozen = np.einsum("nk,nkb->nb", best_w[is_hist], x_corners)
            best_pt[is_hist] = pt_frozen
            d_hist = np.linalg.norm(x[ni[is_hist]] - pt_frozen, axis=1)
            best_d[is_hist] = d_hist
            pen[is_hist] = self.gap_bound - d_hist

        # 2. New candidates: run narrow phase projection
        if np.any(~is_hist):
            new_idx = np.where(~is_hist)[0]
            jit = accel_get("t7_narrow")
            if jit is not None:
                d_new, pt_new, w_new = jit(x, ni[new_idx], seg[new_idx])
            else:
                d_new, pt_new, w_new = _narrow(x, ni[new_idx], seg[new_idx])

            best_d[new_idx] = d_new
            best_pt[new_idx] = pt_new
            best_w[new_idx] = w_new
            pen[new_idx] = self.gap_bound - d_new

        # Active pairs: existing tied pairs OR new pairs with penetration > 0
        active = is_hist | (pen > 0.0)
        if not np.any(active):
            # If pairs dropped out
            self._hist_keys = np.zeros(0, dtype=np.int64)
            self._hist_fn = np.zeros(0, dtype=float)
            self._hist_ft1 = np.zeros(0, dtype=float)
            self._hist_ft2 = np.zeros(0, dtype=float)
            self._hist_w = np.zeros((0, 4), dtype=float)
            self._hist_fvec = np.zeros((0, 3), dtype=float)
            return 0.0, self.dt_bound

        ni = ni[active]
        seg = seg[active]
        srow = srow[active]
        best_pt = best_pt[active]
        best_w = best_w[active]
        keys = keys[active]
        best_d = best_d[active]
        pen = pen[active]
        fn_old = fn_old[active]
        ft1_old = ft1_old[active]
        ft2_old = ft2_old[active]

        loc = np.searchsorted(self.nodes, ni)
        K = combine_stiffness(0, self.itf.stfac, self.Km[srow], self.Ks[loc])

        # Local orthonormal frame (T1, T2, N) per i10for3.F lines 274-302
        valid_seg = np.maximum(seg, 0)
        x1 = x[valid_seg[:, 0]]
        x2 = x[valid_seg[:, 1]]
        x3 = x[valid_seg[:, 2]]
        is_tri = (seg[:, 3] == seg[:, 2]) | (seg[:, 3] < 0)
        x4 = np.where(is_tri[:, None], x3, x[valid_seg[:, 3]])

        # T1 = normalize(x3 - x1)
        t1_raw = x3 - x1
        t1_norm = np.linalg.norm(t1_raw, axis=1)
        t1 = np.where(
            t1_norm[:, None] > EM20,
            t1_raw / np.maximum(t1_norm, EM20)[:, None],
            np.array([1.0, 0.0, 0.0]),
        )

        # T2_diag = x4 - x2 (quads) or x3 - x2 (triangles)
        t2_diag = np.where(is_tri[:, None], x3 - x2, x4 - x2)
        n_raw = np.cross(t1, t2_diag)
        n_norm = np.linalg.norm(n_raw, axis=1)
        nvec = np.where(
            n_norm[:, None] > EM20,
            n_raw / np.maximum(n_norm, EM20)[:, None],
            np.array([0.0, 0.0, 1.0]),
        )

        # T2 = normalize(N x T1)
        t2_raw = np.cross(nvec, t1)
        t2_norm = np.linalg.norm(t2_raw, axis=1)
        t2 = np.where(
            t2_norm[:, None] > EM20,
            t2_raw / np.maximum(t2_norm, EM20)[:, None],
            np.array([0.0, 1.0, 0.0]),
        )

        # Relative velocity of closest/anchor points (i10for3.F line 265-270)
        v_seg = np.einsum("nk,nkb->nb", best_w, v[valid_seg])
        vrel = v[ni] - v_seg
        vn = np.einsum("nb,nb->n", vrel, nvec)
        vt1 = np.einsum("nb,nb->n", vrel, t1)
        vt2 = np.einsum("nb,nb->n", vrel, t2)

        # Incremental force update (i10for3.F line 320-322)
        # Force opposes relative velocity: fn accumulates vn * dt * K
        fn_new = fn_old + vn * dt * K
        ft1_new = ft1_old + vt1 * dt * K
        ft2_new = ft2_old + vt2 * dt * K

        # Tied / Rebound condition (i10for3.F line 326-350)
        keep = np.ones(len(ni), dtype=bool)
        itied = int(getattr(self.itf, "itied", self.itied) or 0)
        if itied == 0:
            # Rebound permitted: if force changes sign from compression to tension and unpenetrated
            # In i10for3.F line 327: CAND_F(1) * FNI < 0 and PENE == 0
            rebound = (fn_old * fn_new < 0.0) & (pen <= 0.0)
            fn_new[rebound] = 0.0
            ft1_new[rebound] = 0.0
            ft2_new[rebound] = 0.0
            keep[rebound] = False

        # Viscous Damping (i10for3.F lines 360-368)
        if self.stiff_dc > 0.0:
            C = self.stiff_dc * np.sqrt(2.0 * K * mass[ni])
            fn_damp = vn * C
            ft1_damp = vt1 * C
            ft2_damp = vt2 * C
            dE_damp = float(np.sum(C * (vn**2 + vt1**2 + vt2**2) * dt))
        else:
            fn_damp = np.zeros(len(ni))
            ft1_damp = np.zeros(len(ni))
            ft2_damp = np.zeros(len(ni))
            dE_damp = 0.0

        f_normal = (fn_new + fn_damp)[:, None] * nvec
        f_tangent = (ft1_new + ft1_damp)[:, None] * t1 + (ft2_new + ft2_damp)[:, None] * t2
        fvec_total = -(f_normal + f_tangent)

        # Elastic energy increment (i10for3.F line 317-319)
        dE_elastic = float(
            np.sum((fn_old * vn + ft1_old * vt1 + ft2_old * vt2) * (dt * 0.5))
        )
        dE_step = dE_elastic + dE_damp
        self.e_cont += dE_elastic
        self.e_damp += dE_damp

        # Update History (only for kept active pairs)
        self._hist_keys = keys[keep]
        self._hist_fn = fn_new[keep]
        self._hist_ft1 = ft1_new[keep]
        self._hist_ft2 = ft2_new[keep]
        self._hist_w = best_w[keep]
        self._hist_fvec = fvec_total[keep]

        if not np.any(keep):
            return dE_step, self.dt_bound

        # Scatter contact forces into fcont: secondary gets +fvec_total, master gets -best_w * fvec_total
        act_ni = ni[keep]
        act_seg = valid_seg[keep]
        act_w = best_w[keep]
        act_fvec = fvec_total[keep]

        scatter_add3(fcont, act_ni, act_fvec)
        scatter_add3(fcont, act_seg[:, 0], -act_w[:, 0:1] * act_fvec)
        scatter_add3(fcont, act_seg[:, 1], -act_w[:, 1:2] * act_fvec)
        scatter_add3(fcont, act_seg[:, 2], -act_w[:, 2:3] * act_fvec)
        valid4 = (seg[keep, 3] >= 0) & (act_w[:, 3] > 0.0)
        if np.any(valid4):
            scatter_add3(
                fcont,
                act_seg[valid4, 3],
                -act_w[valid4, 3:4] * act_fvec[valid4],
            )

        # Stiffness accumulation for /DT/NODA and interface stability bound
        n_nod = len(fcont)
        K_node = np.zeros(n_nod, dtype=float)
        act_K = K[keep]

        np.add.at(K_node, act_ni, act_K)
        np.add.at(K_node, act_seg[:, 0], act_w[:, 0] * act_K)
        np.add.at(K_node, act_seg[:, 1], act_w[:, 1] * act_K)
        np.add.at(K_node, act_seg[:, 2], act_w[:, 2] * act_K)
        if np.any(valid4):
            np.add.at(K_node, act_seg[valid4, 3], act_w[valid4, 3] * act_K[valid4])

        if stifn is not None:
            stifn += K_node

        loaded = K_node > 0.0
        if np.any(loaded):
            dt_int = float(np.min(np.sqrt(2.0 * mass[loaded] / K_node[loaded])))
        else:
            dt_int = np.inf

        dt_bound = min(self.dt_bound, dt_int)
        return dE_step, dt_bound
