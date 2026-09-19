"""
/INTER/TYPE12 — ALE mesh motion / grid velocity coupling interface.

Fortran origin:
  - ``engine/source/interfaces/inter3d/i12cor3.F`` (ALE surface-to-surface projection & coordinates)
  - ``engine/source/interfaces/inter3d/i12loc3.F`` (nearest neighbor localization)
  - ``engine/source/interfaces/inter3d/i12msr3.F`` (main segment identification)
  - ``engine/source/interfaces/inter3d/i12dis3.F`` (tolerance & penetration check)
  - ``engine/source/interfaces/interf/i12for3.F``  (interface forces & momentum transfer)
  - ``engine/source/interfaces/interf/i12vit3.F``  (grid velocity coupling)
  - ``engine/source/interfaces/interf/intti12.F``  (coordinator for ALE sliding/tied interface)
  - ``starter/source/interfaces/int12/hm_read_inter_type12.F`` (card reading)

Physics:
  Couples secondary nodes (from secondary ALE or Lagrangian surface) to master segments:
  1. For each secondary node, projects onto the closest master surface segment to obtain
     isoparametric coordinates (s, t) and shape function weights H_1..H_4.
  2. Evaluates displacement offset:
       delta_x = x_sec - sum(H_k * x_master,k)
     and relative velocity:
       delta_v = v_sec - sum(H_k * v_master,k)
  3. When node is within tolerance / contact zone (or tied):
       F_sec = - (K * delta_x + C * delta_v)
       F_master,k = H_k * (K * delta_x + C * delta_v) = - H_k * F_sec
  4. Momentum conservation:
       sum(F_master,k) + F_sec = sum(H_k) * (-F_sec) + F_sec = 0
     Linear and angular momentum are strictly conserved for closed systems.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20
from ..common.fastmath import norm3, scatter_add3
from ..model.model import Model
from .inter_type7 import _narrow, segment_stiffness_gap


class ContactType12:
    """One /INTER/TYPE12 ALE mesh motion / grid velocity coupling interface."""

    def __init__(self, itf, model: Model, log=None):
        self.itf = itf
        self.model = model
        self.log = log if log is not None else getattr(model, "log", None)

        self.id = getattr(itf, "id", 0)
        self.title = getattr(itf, "title", f"TYPE12_{self.id}")
        self.tstart = float(getattr(itf, "tstart", 0.0) or 0.0)
        self.tstop = float(getattr(itf, "tstop", 1e30) or 1e30)
        if self.tstop <= 0.0:
            self.tstop = 1e30

        self.tol = float(getattr(itf, "tol", 0.0) or getattr(itf, "gap", 0.0) or 0.02)
        self.gap = self.tol
        self.itied = int(getattr(itf, "itied", 0) or 0)
        self.interpol = int(getattr(itf, "interpol", 0) or 0)
        self.bcopt = int(getattr(itf, "bcopt", 0) or 0)
        self.stfac = float(getattr(itf, "stfac", 1.0) or 1.0)
        self.visc = float(getattr(itf, "visc", 0.05) or 0.05)

        self.e_cont = 0.0
        self.e_damp = 0.0
        self.dt_bound = 1e30

        self._init_entities()

    def _init_empty(self) -> None:
        self.sec_nodes = np.zeros(0, dtype=np.int64)
        self.main_segs = np.zeros((0, 4), dtype=np.int64)
        self.K_seg = np.zeros(0, dtype=np.float64)
        self.dt_bound = 1e30

    def _init_entities(self) -> None:
        model = self.model
        itf = self.itf

        surf_ids = (
            getattr(itf, "surf_ids", 0)
            or getattr(itf, "surf_id", 0)
            or getattr(itf, "params", {}).get("surf_ids", 0)
        )
        surf_idm = (
            getattr(itf, "surf_idm", 0)
            or getattr(itf, "surf_id1", 0)
            or getattr(itf, "params", {}).get("surf_idm", 0)
        )
        grnod_id = getattr(itf, "grnod_id", 0)

        # Resolve secondary nodes
        sec_nodes = []
        if grnod_id > 0 and hasattr(model, "node_groups") and grnod_id in model.node_groups:
            grp = model.node_groups[grnod_id]
            if getattr(grp, "node_idx", None) is not None:
                sec_nodes = list(grp.node_idx)
            elif getattr(grp, "nodes", None) is not None:
                sec_nodes = list(grp.nodes)

        if len(sec_nodes) == 0 and surf_ids > 0 and hasattr(model, "surfaces") and surf_ids in model.surfaces:
            surf_s = model.surfaces[surf_ids]
            if getattr(surf_s, "segments", None) is not None and len(surf_s.segments) > 0:
                s_arr = np.asarray(surf_s.segments, dtype=np.int64)
                sec_nodes = list(np.unique(s_arr[s_arr >= 0]))
            elif getattr(surf_s, "seg_nodes", None) and len(surf_s.seg_nodes) > 0:
                nodes_flat = [nid for sub in surf_s.seg_nodes for nid in sub]
                if hasattr(model, "node_id_to_idx"):
                    sec_nodes = [model.node_id_to_idx[nid] for nid in nodes_flat if nid in model.node_id_to_idx]
                else:
                    sec_nodes = nodes_flat

        self.sec_nodes = np.unique(np.asarray(sec_nodes, dtype=np.int64))

        # Resolve master segments
        main_segs = np.zeros((0, 4), dtype=np.int64)
        if surf_idm > 0 and hasattr(model, "surfaces") and surf_idm in model.surfaces:
            surf_m = model.surfaces[surf_idm]
            if getattr(surf_m, "segments", None) is not None and len(surf_m.segments) > 0:
                segs = np.asarray(surf_m.segments, dtype=np.int64)
                if segs.ndim == 2:
                    if segs.shape[1] == 3:
                        segs = np.column_stack([segs, segs[:, 2]])
                    elif segs.shape[1] == 4:
                        segs = segs.copy()
                        tri_mask = segs[:, 3] < 0
                        segs[tri_mask, 3] = segs[tri_mask, 2]
                main_segs = segs

        self.main_segs = main_segs

        if len(self.sec_nodes) == 0 or len(self.main_segs) == 0:
            if self.log:
                self.log.info(
                    f"/INTER/TYPE12/{self.id}: empty secondary nodes ({len(self.sec_nodes)}) "
                    f"or main segments ({len(self.main_segs)}) — interface inactive",
                    "CONTACT INIT",
                )
            self._init_empty()
            return

        # Estimate segment stiffness
        seg_gtype = (
            surf_m.seg_gtype
            if getattr(surf_m, "seg_gtype", None) is not None
            else np.zeros(len(self.main_segs), dtype="<U8")
        )
        seg_elem = (
            surf_m.seg_elem
            if getattr(surf_m, "seg_elem", None) is not None
            else np.full(len(self.main_segs), -1, dtype=np.int64)
        )

        try:
            Km, _ = segment_stiffness_gap(model, self.main_segs, seg_gtype, seg_elem, self.stfac)
            self.K_seg = np.where(Km > EM20, Km, 1e6 * self.stfac)
        except Exception:
            self.K_seg = np.full(len(self.main_segs), 1e6 * self.stfac, dtype=np.float64)

        # Time step estimate
        mass0 = getattr(model, "mass0", None)
        m_eff = mass0 if mass0 is not None and len(mass0) > 0 else getattr(model, "mass", None)
        if m_eff is not None and len(m_eff) > 0 and len(self.sec_nodes) > 0:
            m_sec = m_eff[self.sec_nodes]
            valid_m = m_sec[m_sec > EM20]
            m_min = float(valid_m.min()) if len(valid_m) > 0 else 1.0
            k_max = float(self.K_seg.max()) if len(self.K_seg) > 0 else 1e6
            self.dt_bound = float(np.sqrt(2.0 * m_min / max(k_max, EM20)))
        else:
            self.dt_bound = 1e30

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
        """Compute interface forces for one explicit cycle, scattered into ``fcont``.

        Returns (fcont, dt_bound).
        """
        if len(self.sec_nodes) == 0 or len(self.main_segs) == 0:
            return fcont, self.dt_bound

        t_cur = float(t if t is not None else 0.0)
        if t_cur < self.tstart or t_cur > self.tstop:
            return fcont, self.dt_bound

        n_sec = len(self.sec_nodes)
        n_main = len(self.main_segs)

        # For each secondary node, find the closest main segment
        # Using vectorized narrow-phase evaluation across all main segments for small/moderate sizes
        # or best candidate per secondary node
        node_coords = x[self.sec_nodes]  # (n_sec, 3)

        # Expand all secondary nodes against all main segments
        # segs shape: (n_main, 4)
        cand_sec = np.repeat(self.sec_nodes, n_main)
        cand_seg_idx = np.tile(np.arange(n_main), n_sec)
        cand_segs = self.main_segs[cand_seg_idx]

        best_d, best_pt, best_w = _narrow(x, cand_sec, cand_segs)

        # Reshape to (n_sec, n_main) to find the minimum distance segment for each secondary node
        d_mat = best_d.reshape((n_sec, n_main))
        min_seg_col = np.argmin(d_mat, axis=1)  # (n_sec,)
        row_idx = np.arange(n_sec)
        flat_idx = row_idx * n_main + min_seg_col

        d_closest = best_d[flat_idx]
        pt_closest = best_pt[flat_idx]
        w_closest = best_w[flat_idx]
        seg_closest = cand_segs[flat_idx]
        k_closest = self.K_seg[min_seg_col]

        # Active check: within tolerance, or tied
        if self.itied >= 1:
            active = np.ones(n_sec, dtype=bool)
        else:
            active = d_closest <= self.tol

        if not np.any(active):
            return fcont, self.dt_bound

        act_sec_nodes = self.sec_nodes[active]
        act_pt = pt_closest[active]
        act_w = w_closest[active]
        act_segs = seg_closest[active]
        act_k = k_closest[active]

        # Position offset: secondary node position minus projected target position
        delta_x = node_coords[active] - act_pt  # (N_act, 3)

        # Velocity offset: secondary node velocity minus projected target velocity
        # v_target = sum(H_k * v_master,k)
        v_target = (
            act_w[:, 0:1] * v[act_segs[:, 0]]
            + act_w[:, 1:2] * v[act_segs[:, 1]]
            + act_w[:, 2:3] * v[act_segs[:, 2]]
            + act_w[:, 3:4] * v[act_segs[:, 3]]
        )
        delta_v = v[act_sec_nodes] - v_target  # (N_act, 3)

        # Damping coefficient: C = 2 * visc * sqrt(K * m_sec)
        m_sec = mass[act_sec_nodes]
        c_damp = 2.0 * self.visc * np.sqrt(np.maximum(act_k * m_sec, EM20))  # (N_act,)

        # Restoring / coupling force on secondary node
        # F_sec = - (K * delta_x + C * delta_v)
        f_sec = - (act_k[:, None] * delta_x + c_damp[:, None] * delta_v)  # (N_act, 3)

        # Scatter to secondary node: fcont += f_sec
        scatter_add3(fcont, act_sec_nodes, f_sec)

        # Scatter equal and opposite reaction to master segment corners with shape weights H_k:
        # sum(H_k * (-f_sec)) = -f_sec * sum(H_k) = -f_sec
        # so total sum of forces across secondary node + master corners is identically ZERO.
        for k in range(4):
            scatter_add3(fcont, act_segs[:, k], - act_w[:, k:k+1] * f_sec)

        # Accumulate contact stiffness if requested
        if stifn is not None and len(stifn) > 0:
            np.add.at(stifn, act_sec_nodes, act_k)
            for k in range(4):
                np.add.at(stifn, act_segs[:, k], act_w[:, k] * act_k)

        # Energy tracking
        if dt > 0.0:
            e_inc = float(np.sum(act_k[:, None] * delta_x * delta_v * dt))
            damp_inc = float(np.sum(c_damp[:, None] * (delta_v ** 2) * dt))
            self.e_cont += e_inc
            self.e_damp += damp_inc

        return fcont, self.dt_bound
