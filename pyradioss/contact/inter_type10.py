"""
/INTER/TYPE10 — penalty tied contact.

Fortran origin: ``engine/source/interfaces/int10/``
- ``i10mainf.F``: interface coordinator
- ``i10dst3.F``: penetration check & candidate filtering
- ``i10for3.F``: incremental penalty force, tied logic, damping

Physics:
- An auto-impacting tied contact using penalty springs (unlike TYPE2 which is kinematic).
- Active between Tstart and Tstop.
- Broad phase (voxel sort) finds candidates.
- Narrow phase computes exact closest point (using TYPE7's projection).
- Tied/Rebound (Itied):
  - Itied = 0: Rebound permitted. If normal force changes sign and node is not penetrated, it separates.
  - Itied = 1: Tied permanently once impacted.
"""

from __future__ import annotations

import numpy as np

from ..accel import get as accel_get
from ..common.constants import EM20
from ..common.fastmath import scatter_add3
from ..model.model import Model
from . import tracking
from .inter_type7 import _narrow, _segment_areas, combine_stiffness, node_stiffness_gap, segment_stiffness_gap, _expand_matches


class ContactType10:
    """One /INTER/TYPE10 interface, engine-side."""

    def __init__(self, itf, model: Model, log):
        self.itf = itf
        self.model = model

        surf = model.surfaces[itf.surf_id]
        self.segs = surf.segments
        if self.segs is None or len(self.segs) == 0:
            log.warning(f"/INTER/TYPE10/{itf.id}: main surface is empty — "
                        f"interface inactive", "CONTACT INIT")
            self.segs = np.zeros((0, 4), dtype=np.int64)
            self.seg_gtype = np.zeros(0, dtype="<U8")
            self.seg_elem = np.zeros(0, dtype=np.int64)
        else:
            self.seg_gtype = surf.seg_gtype
            self.seg_elem = surf.seg_elem

        # secondary nodes
        self.nodes = np.sort(model.node_groups[itf.grnod_id].node_idx)

        # stiffness
        scale = itf.stfac if itf.stfac > 0 else 1.0
        Km, gm = segment_stiffness_gap(model, self.segs, self.seg_gtype, self.seg_elem, scale)
        Ks_all, _ = node_stiffness_gap(model, scale)
        self.Km = Km
        self.Ks = Ks_all[self.nodes] if len(self.nodes) else np.zeros(0)
        self.gap_bound = itf.gap if itf.gap > 0 else 0.0

        # tracking
        self.deletable = tracking.any_deletable(model, self.seg_gtype)
        if self.deletable:
            self.ref_total = tracking.node_reference_counts(model, alive_only=False)
        self.seg_alive = np.ones(len(self.segs), dtype=bool)
        self.nodes_tracked = self.nodes

        # broad-phase bookkeeping
        self.pairs_node = np.zeros(0, dtype=np.int64)
        self.pairs_seg = np.zeros(0, dtype=np.int64)
        self._last_refresh = -10**9
        self.refresh = 20

        # history arrays for incremental force
        self._hist_keys = np.zeros(0, dtype=np.int64)
        self._hist_fvec = np.zeros((0, 3), dtype=np.float64)
        self._hist_fn = np.zeros(0, dtype=np.float64)

        self.e_cont = 0.0
        self.e_damp = 0.0

    def _compute_dt_bound(self, mass) -> float:
        if len(self.segs) == 0 or len(self.nodes) == 0:
            return np.inf
        Km_max = np.full(len(self.nodes), self.Km.max())
        # Type 10 doesn't have istf, it just scales. Using istf=0 default logic
        K_sec = combine_stiffness(0, self.itf.stfac, Km_max, self.Ks)
        dt_sec = np.sqrt(2.0 * mass[self.nodes] / np.maximum(K_sec, EM20)).min()
        Ks_max = np.full(len(self.segs), self.Ks.max() if len(self.Ks) else 0.0)
        K_main = combine_stiffness(0, self.itf.stfac, self.Km, Ks_max)
        m_corner = mass[self.segs].min(axis=1)
        dt_main = np.sqrt(2.0 * m_corner / np.maximum(K_main, EM20)).min()
        return float(min(dt_sec, dt_main))

    def _broad_phase(self, x: np.ndarray, v: np.ndarray, dt: float):
        segs = self.segs[self.seg_alive]
        seg_rows = np.where(self.seg_alive)[0]
        nodes = self.nodes_tracked
        if len(segs) == 0 or len(nodes) == 0:
            self.pairs_node = np.zeros(0, dtype=np.int64)
            self.pairs_seg = np.zeros(0, dtype=np.int64)
            return
        
        margin = self.gap_bound + 2.0 * self.refresh * dt * (np.abs(v).max() if len(v) else 0.0)
        
        xs = x[segs]
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
        
        # Keep tied pairs even if they fall out of broad phase margin
        if len(self._hist_keys) > 0:
            n_segs = max(len(self.segs), 1)
            tied_ni = self._hist_keys // n_segs
            tied_sj = self._hist_keys % n_segs
            
            # Combine broad phase found pairs with tied pairs
            all_ni = np.concatenate([ni, tied_ni])
            all_sj = np.concatenate([sj, tied_sj])
            
            # Deduplicate
            pair_keys = all_ni * n_segs + all_sj
            _, uniq_idx = np.unique(pair_keys, return_index=True)
            self.pairs_node = all_ni[uniq_idx]
            self.pairs_seg = all_sj[uniq_idx]
        else:
            self.pairs_node = ni
            self.pairs_seg = sj

    def forces(self, x: np.ndarray, v: np.ndarray, mass: np.ndarray, 
               fcont: np.ndarray, cycle: int, dt: float, t: float, stifn=None):
        if len(self.segs) == 0 or len(self.nodes) == 0:
            return 0.0, np.inf
            
        if t < self.itf.tstart or t > self.itf.tstop:
            return 0.0, np.inf

        if self.deletable:
            self.seg_alive = tracking.alive_segment_mask(self.model, self.seg_gtype, self.seg_elem)

        if cycle - self._last_refresh >= self.refresh:
            if self.deletable:
                mask = tracking.tracked_node_mask(self.model, self.ref_total)
                self.nodes_tracked = self.nodes[mask[self.nodes]]
            self._broad_phase(x, v, dt)
            self._last_refresh = cycle
            
        if len(self.pairs_node) == 0:
            return 0.0, np.inf

        live = self.seg_alive[self.pairs_seg]
        ni = self.pairs_node[live]
        srow = self.pairs_seg[live]
        if len(ni) == 0:
            return 0.0, np.inf
        seg = self.segs[srow]

        jit = accel_get("t7_narrow")
        if jit is not None:
            best_d, best_pt, best_w = jit(x, ni, seg)
        else:
            best_d, best_pt, best_w = _narrow(x, ni, seg)

        n_segs = max(len(self.segs), 1)
        keys = ni * n_segs + srow
        
        # Find which pairs are already active (in history)
        is_hist = np.isin(keys, self._hist_keys)
        
        # New pairs check penetration
        gap = np.full(len(ni), self.gap_bound)
        pen = gap - best_d
        
        # A pair is active if it's already in history OR (pen > 0 and inactiv allows it)
        # Note: if inactiv == 1, initial penetration might deactivate it, but we follow standard penalty
        active = is_hist | (pen > 0.0)
        
        if not np.any(active):
            return 0.0, np.inf

        ni = ni[active]
        seg = seg[active]
        srow = srow[active]
        best_pt = best_pt[active]
        best_w = best_w[active]
        keys = keys[active]
        best_d = best_d[active]
        pen = pen[active]
        
        loc = np.searchsorted(self.nodes, ni)
        K = combine_stiffness(0, self.itf.stfac, self.Km[srow], self.Ks[loc])
        
        d_vec = x[ni] - best_pt
        d_norm = np.maximum(best_d, EM20)
        nvec = d_vec / d_norm[:, None]
        
        vseg = np.einsum("nk,nkb->nb", best_w, v[seg])
        vrel = v[ni] - vseg
        vn = np.einsum("nb,nb->n", vrel, nvec)
        
        # Fetch history
        fvec_old = np.zeros((len(ni), 3))
        fn_old = np.zeros(len(ni))
        
        # Vectorized lookup for history
        if len(self._hist_keys) > 0:
            sorter = np.argsort(self._hist_keys)
            idx = np.searchsorted(self._hist_keys, keys, sorter=sorter)
            valid = (idx < len(self._hist_keys))
            idx_valid = idx[valid]
            match = self._hist_keys[sorter[idx_valid]] == keys[valid]
            
            final_idx = sorter[idx_valid[match]]
            fvec_old[valid.nonzero()[0][match]] = self._hist_fvec[final_idx]
            fn_old[valid.nonzero()[0][match]] = self._hist_fn[final_idx]

        # Incremental Force Update
        fvec_new = fvec_old - vrel * dt * K[:, None]  # Force on slave node opposes vrel
        fn_new = np.einsum("nb,nb->n", fvec_new, nvec)
        
        # Tied / Rebound condition
        keep = np.ones(len(ni), dtype=bool)
        if self.itf.itied == 0:
            # Rebound permitted: if force changes sign and unpenetrated
            rebound = (fn_old * fn_new < 0.0) & (pen <= 0.0)
            fvec_new[rebound] = 0.0
            keep[rebound] = False
            
        # Viscous Damping
        C = self.itf.stiff_dc * np.sqrt(2.0 * K * mass[ni])
        fdamp = -vrel * C[:, None]
        fvec_total = fvec_new + fdamp
        
        # Update History (only for kept pairs)
        self._hist_keys = keys[keep]
        self._hist_fvec = fvec_new[keep]
        self._hist_fn = fn_new[keep]
        
        # Scatter forces
        scatter_add3(fcont, ni, fvec_total)
        scatter_add3(fcont, seg[:, 0], -best_w[:, 0:1] * fvec_total)
        scatter_add3(fcont, seg[:, 1], -best_w[:, 1:2] * fvec_total)
        scatter_add3(fcont, seg[:, 2], -best_w[:, 2:3] * fvec_total)
        scatter_add3(fcont, seg[:, 3], -best_w[:, 3:4] * fvec_total)
        
        # Energy accounting
        vrel_mag2 = np.einsum("nb,nb->n", vrel, vrel)
        # Elastic work is int(F_elastic * dx), which is roughly F_total * V_rel * dt / 2
        dE_elastic = np.sum(np.einsum("nb,nb->n", -fvec_total, vrel) * (dt / 2.0))
        dE_damp = np.sum(C * vrel_mag2 * dt)
        
        self.e_cont += dE_elastic
        self.e_damp += dE_damp
        
        # Time step calculation
        n_nod = len(fcont)
        Knode = np.bincount(ni, weights=K, minlength=n_nod)
        Knode += np.bincount(seg.reshape(-1), weights=np.repeat(K, 4), minlength=n_nod)
        loaded = Knode > 0.0
        dt_int = min(1e30, float(np.sqrt(2.0 * mass[loaded] / Knode[loaded]).min()))
        
        if stifn is not None:
            stifn[loaded] += Knode[loaded]
            
        return dE_elastic + dE_damp, dt_int
