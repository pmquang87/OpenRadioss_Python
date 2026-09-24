"""
/INTER/GUIDED_CABLE — 1D cable sliding through guide eyelets.

Fortran origin:
  OpenRadioss starter/source/tools/seatbelts/hm_read_guided_cable.F90
  OpenRadioss engine/source/tools/seatbelts/guided_cable_force.F90
  OpenRadioss engine/source/tools/seatbelts/compute_contact_force_guide.F90

Mechanism:
1. Cable elements (1D truss/beam/springs) pass through guide eyelet nodes (anchor nodes).
2. The anchor node is projected onto the active cable segment.
3. Transverse deviation generates a centering spring and damping force holding the cable inside the eyelet.
4. Relative axial motion generates Coulomb friction resisting cable sliding.
5. All forces are scattered conservatively onto the anchor node (-F) and the segment end nodes
   ((1-alpha)*F and alpha*F), conserving linear momentum exactly.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
import numpy as np

from ..common.constants import EM20
from ..common.fastmath import norm3
from ..model.model import Model


class ContactGuidedCable:
    """1D cable sliding through guide eyelets with Coulomb friction and transverse centering.

    Fortran reference:
      ``guided_cable_force_mod`` in ``engine/source/tools/seatbelts/guided_cable_force.F90``
      ``compute_contact_force_guide_mod`` in ``engine/source/tools/seatbelts/compute_contact_force_guide.F90``
    """

    def __init__(self, itf: Any, model: Model, log: Any = None):
        self.itf = itf
        self.model = model
        self.log = log if log is not None else getattr(model, "log", None)

        self.id = getattr(itf, "id", 1)
        self.grnod_id = getattr(itf, "grnod_id", getattr(itf, "grnod_main", 0))
        self.grpart_id = getattr(itf, "grpart_id", getattr(itf, "grnod_sub", 0))
        params = getattr(itf, "params", {}) or {}
        self.k_trans = float(params.get("k_trans", 0.0) or 0.0)
        self.istiff = int(getattr(itf, "istiff", 1) or 1)
        self.stfac = float(params.get("stfac", getattr(itf, "stfac", 1.0)) or 1.0)
        self.fric = float(params.get("mu", params.get("fric", getattr(itf, "fric", 0.0))) or 0.0)
        self.tstart = float(getattr(itf, "tstart", 0.0) or 0.0)
        self.tstop = float(getattr(itf, "tstop", np.inf) or np.inf)
        if self.tstop <= 0.0:
            self.tstop = np.inf
        self.dt_bound = np.inf

        self.anchor_nodes = np.zeros(0, dtype=np.int64)
        self.cable_segments = np.zeros((0, 2), dtype=np.int64)
        self.active_segment_idx = np.zeros(0, dtype=np.int64)

        self._resolve_entities()

    def _resolve_entities(self) -> None:
        """Resolve anchor nodes and cable 1D elements from the model."""
        params = getattr(self.itf, "params", {}) or {}
        # 1. Resolve anchor nodes (eyelets)
        if "guide_nodes" in params and params["guide_nodes"] is not None:
            self.anchor_nodes = np.asarray(params["guide_nodes"], dtype=np.int64)
        elif hasattr(self.itf, "anchor_nodes") and self.itf.anchor_nodes is not None:
            self.anchor_nodes = np.asarray(self.itf.anchor_nodes, dtype=np.int64)
        elif self.grnod_id > 0:
            g = getattr(self.model, "node_groups", {}).get(self.grnod_id)
            if g is not None:
                if getattr(g, "node_idx", None) is not None:
                    self.anchor_nodes = np.asarray(g.node_idx, dtype=np.int64)
                elif getattr(g, "node_ids", None) and hasattr(self.model, "node_id_to_idx"):
                    self.anchor_nodes = np.array([
                        self.model.node_id_to_idx[nid]
                        for nid in g.node_ids
                        if nid in self.model.node_id_to_idx
                    ], dtype=np.int64)
                elif getattr(g, "nodes", None) is not None:
                    self.anchor_nodes = np.asarray(g.nodes, dtype=np.int64)

        # 2. Resolve cable 1D segments (trusses, beams, springs)
        if "cable_nodes" in params and params["cable_nodes"] is not None:
            s = np.asarray(params["cable_nodes"], dtype=np.int64)
            if s.ndim == 2 and s.shape[1] >= 2:
                self.cable_segments = s[:, :2]
                self.active_segment_idx = np.zeros(len(self.anchor_nodes), dtype=np.int64)
                return

        segs: List[Tuple[int, int]] = []
        if hasattr(self.itf, "segments") and self.itf.segments is not None:
            s = np.asarray(self.itf.segments, dtype=np.int64)
            if s.ndim == 2 and s.shape[1] >= 2:
                self.cable_segments = s[:, :2]
                self.active_segment_idx = np.zeros(len(self.anchor_nodes), dtype=np.int64)
                return

        cable_part_ids = set()
        if self.grpart_id > 0:
            pgroups = getattr(self.model, "part_groups", {})
            pg = pgroups.get(self.grpart_id)
            if pg is not None:
                if hasattr(pg, "part_ids"):
                    cable_part_ids.update(pg.part_ids)
                elif hasattr(pg, "members"):
                    for _, pids in pg.members:
                        cable_part_ids.update(pids)
            else:
                cable_part_ids.add(self.grpart_id)

        # Search 1D element collections in model
        for elem_attr in ("trusses", "beams", "springs"):
            elements = getattr(self.model, elem_attr, None)
            if elements is None:
                continue
            conn = getattr(elements, "conn", None)
            if conn is None:
                conn = getattr(elements, "ixs", None)
            if conn is None and isinstance(elements, np.ndarray):
                conn = elements
            if conn is None:
                continue

            part_ids = getattr(elements, "part_id", getattr(elements, "pid", None))
            n_elem = len(conn)
            for i in range(n_elem):
                if cable_part_ids and part_ids is not None and len(part_ids) > i:
                    if part_ids[i] not in cable_part_ids:
                        continue
                if conn.shape[1] >= 2:
                    segs.append((int(conn[i, 0]), int(conn[i, 1])))

        if segs:
            self.cable_segments = np.asarray(segs, dtype=np.int64)
            self.active_segment_idx = np.zeros(len(self.anchor_nodes), dtype=np.int64)

    def forces(self, x: np.ndarray, v: np.ndarray, mass: np.ndarray, dt: float,
               fcont: np.ndarray, cycle: int = 0, stifn: Optional[np.ndarray] = None,
               t: Optional[float] = None) -> Tuple[float, float]:
        """Compute guided cable forces and reactions.

        Fortran reference:
          guided_cable_force.F90 lines 110-291.
          compute_contact_force_guide.F90 lines 88-115.

        Returns (-work, dt_contact).
        """
        n_anchors = len(self.anchor_nodes)
        n_segs = len(self.cable_segments)
        if n_anchors == 0 or n_segs == 0 or dt <= 0.0:
            return 0.0, self.dt_bound

        if t is not None and (t < self.tstart or t > self.tstop):
            return 0.0, self.dt_bound

        n_coords = len(x)
        total_work = 0.0
        dt_min = self.dt_bound
        adamp = 0.05

        for j in range(n_anchors):
            anchor = self.anchor_nodes[j]
            if anchor < 0 or anchor >= n_coords:
                continue

            xa = x[anchor]
            va = v[anchor]
            ma = float(mass[anchor]) if len(mass) > anchor else 1.0

            # Find closest cable segment to anchor node
            best_dist_sq = np.inf
            best_seg_idx = self.active_segment_idx[j] if j < len(self.active_segment_idx) else 0
            best_alpha = 0.0
            best_t_vec = np.zeros(3, dtype=float)
            best_xproj = np.copy(xa)

            # Check candidate segments (start near previously active segment, or check all)
            for s_idx in range(n_segs):
                n1, n2 = self.cable_segments[s_idx]
                if n1 < 0 or n1 >= n_coords or n2 < 0 or n2 >= n_coords:
                    continue

                x1 = x[n1]
                x2 = x[n2]
                s_vec = x2 - x1
                s_len2 = float(np.dot(s_vec, s_vec))
                if s_len2 <= EM20:
                    continue

                # Projection parameter alpha (Fortran: guided_cable_force.F90 line 141)
                alpha = float(np.dot(xa - x1, s_vec)) / s_len2
                alpha_clamped = max(0.0, min(1.0, alpha))

                x_proj = x1 + alpha_clamped * s_vec
                diff = x_proj - xa
                dist_sq = float(np.dot(diff, diff))

                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_seg_idx = s_idx
                    best_alpha = alpha_clamped
                    best_t_vec = s_vec / np.sqrt(s_len2)
                    best_xproj = x_proj

            if best_dist_sq >= np.inf:
                continue

            if j < len(self.active_segment_idx):
                self.active_segment_idx[j] = best_seg_idx

            n1, n2 = self.cable_segments[best_seg_idx]
            m1 = float(mass[n1]) if len(mass) > n1 else 1.0
            m2 = float(mass[n2]) if len(mass) > n2 else 1.0

            # Stiffness calculation (Fortran lines 212-235)
            if self.k_trans > 0.0:
                stiff = self.k_trans
            elif self.istiff == 1:
                # Istiff=1: stiffness computed from time step
                stiff = 0.1 * min(m1, m2, ma) / (dt * dt)
            elif self.istiff == 2:
                stiff = 1.0
            else:
                stiff = 0.1 * min(m1, m2, ma) / (dt * dt)
            stiff *= self.stfac

            # Harmonic mass and damping viscosity (Fortran lines 237-239)
            mass_harm = (m1 * m2) / (m1 + m2 + EM20)
            visc = adamp * np.sqrt(stiff * mass_harm)

            # Transverse centering and damping force (compute_contact_force_guide.F90 lines 90-108)
            v_cable = (1.0 - best_alpha) * v[n1] + best_alpha * v[n2]
            v_rel = v_cable - va

            v_axial = float(np.dot(v_rel, best_t_vec))
            v_perp = v_rel - v_axial * best_t_vec

            # Transverse centering spring: pulls eyelet toward cable projection point
            delta_x = best_xproj - xa
            f_trans = stiff * delta_x + visc * v_perp
            f_norm = float(np.linalg.norm(f_trans))

            # Axial Coulomb friction force: opposes eyelet sliding along cable
            # v_slide is eyelet motion relative to cable
            v_slide = float(np.dot(va - v_cable, best_t_vec))
            f_axial = 0.0
            if self.fric > 0.0 and f_norm > 0.0:
                v_ref = 1e-4
                f_axial = -self.fric * f_norm * (v_slide / (abs(v_slide) + v_ref))

            f_total = f_trans + f_axial * best_t_vec

            # Force distribution (guided_cable_force.F90 lines 282-286):
            # Eyelet receives +f_total (centering toward cable + friction along cable)
            # Cable node1 receives -(1-alpha)*f_total
            # Cable node2 receives -alpha*f_total
            fcont[anchor] += f_total
            fcont[n1] -= (1.0 - best_alpha) * f_total
            fcont[n2] -= best_alpha * f_total

            # Stability stiffness accumulation in stifn (lines 287-290)
            stiff_stab = stiff * (adamp + np.sqrt(1.0 + adamp * adamp)) ** 2
            if stifn is not None:
                stifn[anchor] += stiff_stab
                stifn[n1] += stiff_stab * (1.0 - best_alpha)
                stifn[n2] += stiff_stab * best_alpha

            # Time step bound
            m_loaded = min(ma, m1, m2)
            dt_cand = np.sqrt(2.0 * m_loaded / max(stiff, EM20))
            dt_min = min(dt_min, float(dt_cand))

            # Energy work done
            total_work += float(np.dot(f_total, v_rel)) * dt

        return -total_work, dt_min
