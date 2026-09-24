"""
/INTER/TYPE8 — Drawbead Contact Interface for Sheet Metal Forming.

Fortran origin:
  - ``starter/source/interfaces/int08/hm_read_inter_type08.F`` (card reader)
  - ``engine/source/interfaces/inter3d/i8for3.F`` (drawbead restraining and normal forces)
  - ``engine/source/interfaces/inter3d/i8cst3.F`` (drawbead surface normal / geometry)
  - ``engine/source/interfaces/inter3d/i8gap3.F`` (gap / penetration check)
  - ``engine/source/interfaces/inter3d/i8dis3.F`` (distance / projection)

Physics:
  Drawbead line restraining force acting on sheet metal blank nodes:
  1. Bead line is defined by an ordered polyline of bead nodes [B_0, B_1, ..., B_M].
     For consecutive bead nodes [B_k, B_{k+1}], segment tangent L = x(B_{k+1}) - x(B_k).
  2. For each blank node P:
     - Project x_P onto bead segment [B_k, B_{k+1}]:
       xi = dot(x_P - x(B_k), L) / dot(L, L).
       If 0 <= xi <= 1, closest point is x_proj = (1 - xi)*x(B_k) + xi*x(B_{k+1}).
     - Normal distance d = ||x_P - x_proj||.
     - If depth > 0 and d > depth, blank node is outside bead capture zone.
  3. Kinematics & Restraining Force:
     - Velocity of blank node relative to bead segment:
       v_rel = v_P - ((1 - xi)*v(B_k) + xi*v(B_{k+1})).
     - Component along bead tangent: v_along = dot(v_rel, t_bead) * t_bead.
     - Transverse velocity across bead: v_trans = v_rel - v_along.
     - Restraining force opposes transverse motion:
       F_restr = - dbead_force * tanh(||v_trans|| / v_ref) * (v_trans / ||v_trans||).
  4. Clamping / Normal Force (when mu > 0, depth > 0, pen = depth - d > 0):
     - Sheet normal n_sheet = cross(t_bead, u_trans).
     - Clamping normal force along n_sheet:
       F_norm = mu * ||F_restr|| * (pen / depth).
  5. Exact Linear Momentum Conservation:
     Blank node receives F_blank = F_restr + F_norm.
     Bead segment nodes receive reaction forces:
       -(1 - xi) * F_blank on B_k
       - xi * F_blank on B_{k+1}
     Total force sum: F_blank - (1 - xi)*F_blank - xi*F_blank = 0 to machine precision.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20
from ..model.model import Model


class ContactType8:
    """Drawbead restraining force contact interface (/INTER/TYPE8)."""

    def __init__(self, itf, model: Model, log=None):
        self.itf = itf
        self.model = model
        self.log = log

        self.id = getattr(itf, "id", 0)
        self.title = getattr(itf, "title", f"TYPE8_{self.id}")
        params = getattr(itf, "params", {}) or {}

        # Drawbead parameters
        # In Radioss: DBEAD_FORCE is restraining force, MU is normal/friction parameter
        self.dbead_force = float(
            params.get("dbead_force",
                       getattr(itf, "dbead_force",
                               getattr(itf, "stfac", 1000.0))) or 1000.0
        )
        self.mu = float(
            params.get("mu",
                       getattr(itf, "mu",
                               getattr(itf, "fric", 0.0))) or 0.0
        )
        self.depth = float(
            params.get("depth",
                       getattr(itf, "depth",
                               getattr(itf, "gap",
                                       getattr(itf, "pext", 0.0)))) or 0.0
        )
        self.tstart = float(params.get("tstart", getattr(itf, "tstart", 0.0)) or 0.0)
        self.tstop = float(params.get("tstop", getattr(itf, "tstop", 1e30)) or 1e30)

        # 1. Resolve bead line nodes (ordered polyline)
        bead_nodes = []
        grnod_id = getattr(itf, "grnod_id", 0)
        if grnod_id > 0 and hasattr(model, "node_groups") and grnod_id in model.node_groups:
            g = model.node_groups[grnod_id]
            if getattr(g, "node_idx", None) is not None and len(g.node_idx) >= 2:
                bead_nodes = list(g.node_idx)
            elif getattr(g, "node_ids", None) is not None and hasattr(model, "node_id_to_idx"):
                bead_nodes = [
                    model.node_id_to_idx[nid]
                    for nid in g.node_ids
                    if nid in model.node_id_to_idx
                ]

        if len(bead_nodes) < 2 and "bead_nodes" in params:
            bead_nodes = list(params["bead_nodes"])

        self.bead_nodes = (
            np.asarray(bead_nodes, dtype=np.int64)
            if len(bead_nodes) >= 2
            else np.zeros(0, dtype=np.int64)
        )

        # 2. Resolve blank nodes (secondary side)
        blank_nodes = []
        surf_id = getattr(itf, "surf_id", 0)
        if surf_id > 0 and hasattr(model, "surfaces") and surf_id in model.surfaces:
            surf = model.surfaces[surf_id]
            if getattr(surf, "segments", None) is not None and len(surf.segments) > 0:
                segs = np.asarray(surf.segments, dtype=np.int64)
                valid = segs[segs >= 0]
                if len(valid) > 0:
                    blank_nodes = np.unique(valid)

        if len(blank_nodes) == 0:
            surf_id2 = getattr(itf, "surf_id1", 0) or getattr(itf, "surf_s_id", 0)
            if surf_id2 > 0 and hasattr(model, "surfaces") and surf_id2 in model.surfaces:
                surf = model.surfaces[surf_id2]
                if getattr(surf, "segments", None) is not None and len(surf.segments) > 0:
                    segs = np.asarray(surf.segments, dtype=np.int64)
                    valid = segs[segs >= 0]
                    if len(valid) > 0:
                        blank_nodes = np.unique(valid)

        if len(blank_nodes) == 0 and "blank_nodes" in params:
            blank_nodes = list(params["blank_nodes"])

        if len(blank_nodes) == 0 and len(self.bead_nodes) >= 2:
            # Fallback: all nodes not in bead line
            if hasattr(model, "x") and model.x is not None and len(model.x) > 0:
                blank_nodes = np.setdiff1d(np.arange(len(model.x)), self.bead_nodes)

        self.blank_nodes = (
            np.asarray(blank_nodes, dtype=np.int64)
            if len(blank_nodes) > 0
            else np.zeros(0, dtype=np.int64)
        )

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
        """Compute drawbead restraining forces with exact linear momentum conservation."""
        if t < self.tstart or t > self.tstop or dt <= 0.0:
            return fcont, 1e30
        if len(self.bead_nodes) < 2 or len(self.blank_nodes) == 0:
            return fcont, 1e30

        n_bead_segs = len(self.bead_nodes) - 1
        n_nod = len(fcont)

        # Process each blank node against bead segments
        for p_idx in self.blank_nodes:
            if p_idx < 0 or p_idx >= n_nod:
                continue
            xp = x[p_idx]
            vp = v[p_idx]

            # Find closest bead segment with projection xi in [0, 1]
            best_seg = -1
            best_xi = 0.0
            best_dist = 1e30
            best_dvec = None
            best_tbead = None

            for k in range(n_bead_segs):
                b0 = self.bead_nodes[k]
                b1 = self.bead_nodes[k + 1]
                if b0 < 0 or b0 >= n_nod or b1 < 0 or b1 >= n_nod:
                    continue

                p0 = x[b0]
                p1 = x[b1]
                seg_vec = p1 - p0
                seg_len_sq = float(np.dot(seg_vec, seg_vec))
                if seg_len_sq < 1e-20:
                    continue

                seg_len = np.sqrt(seg_len_sq)
                t_bead = seg_vec / seg_len

                # Barycentric projection parameter along bead segment
                r = xp - p0
                xi = float(np.dot(r, seg_vec) / seg_len_sq)
                if xi < 0.0 or xi > 1.0:
                    continue

                # Projected point and perpendicular displacement
                x_proj = p0 + xi * seg_vec
                d_vec = xp - x_proj
                dist = float(np.linalg.norm(d_vec))

                if dist < best_dist:
                    best_dist = dist
                    best_seg = k
                    best_xi = xi
                    best_dvec = d_vec
                    best_tbead = t_bead

            if best_seg < 0:
                continue

            # Check bead capture depth
            if self.depth > 0.0 and best_dist > self.depth:
                continue

            pen = max(0.0, self.depth - best_dist) if self.depth > 0.0 else 1.0

            b0 = self.bead_nodes[best_seg]
            b1 = self.bead_nodes[best_seg + 1]
            v_proj = (1.0 - best_xi) * v[b0] + best_xi * v[b1]
            v_rel = vp - v_proj

            # Transverse velocity across bead (in sheet plane, orthogonal to bead line)
            v_along = float(np.dot(v_rel, best_tbead)) * best_tbead
            v_trans = v_rel - v_along
            spd_trans = float(np.linalg.norm(v_trans))

            if spd_trans < 1e-12:
                continue

            u_trans = v_trans / spd_trans

            # Restraining force opposing transverse sliding across bead
            v_ref = 1e-4
            reg_scale = np.tanh(spd_trans / v_ref)
            f_restr_mag = self.dbead_force * reg_scale
            f_restr = - f_restr_mag * u_trans

            # Clamping / normal force along sheet normal
            # Sheet normal is perpendicular to bead tangent and transverse direction
            n_sheet = np.cross(best_tbead, u_trans)
            norm_n = float(np.linalg.norm(n_sheet))
            if norm_n > 1e-12:
                n_sheet /= norm_n
            else:
                n_sheet = np.array([0.0, 0.0, 1.0])

            f_norm = np.zeros(3)
            if self.mu > 0.0 and self.depth > 0.0 and pen > 0.0:
                fn_mag = self.mu * f_restr_mag * (pen / self.depth)
                d_normal = float(np.dot(best_dvec, n_sheet))
                sign_d = 1.0 if d_normal >= 0.0 else -1.0
                f_norm = - fn_mag * sign_d * n_sheet

            f_blank = f_restr + f_norm

            # Apply force to blank node
            fcont[p_idx] += f_blank

            # Distribute reaction force to bead segment endpoints (exact momentum conservation)
            fcont[b0] -= (1.0 - best_xi) * f_blank
            fcont[b1] -= best_xi * f_blank

        # Explicit stability time step bound
        dt_bound = 1e30
        if self.dbead_force > 0.0 and len(self.blank_nodes) > 0:
            eff_depth = max(self.depth, 1e-3)
            k_eff = max(self.dbead_force / eff_depth, 1.0)
            m_min = float(np.min(mass[self.blank_nodes])) if len(mass) > 0 else 1.0
            if m_min > 0.0:
                dt_bound = 0.9 * 2.0 * np.sqrt(m_min / k_eff)

        return fcont, dt_bound
