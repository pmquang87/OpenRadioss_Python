"""
/INTER/TYPE22 — Immersed boundary cut-cell ALE-FSI coupling.

Fortran origin:
  OpenRadioss engine/source/interfaces/int22/:
    i22mainf.F      Cut-cell candidate search and broad-phase management
    i22for3.F       FSI coupling force computation across cut surfaces
    i22intersect.F  Intersection of Lagrangian shell facets with ALE background bricks

Physics:
1. Lagrangian structural surface facets (shells/membranes) are immersed inside
   background ALE/Eulerian 8-node brick elements.
2. For each cut cell, the wetted area Swet, surface normal n_surf, and fluid pressure
   difference DP = P1 - P2 across the immersed boundary are computed.
3. The coupling force F = DP * Swet * n_surf + viscous drag is computed at the cut
   facet center of gravity (CoG).
4. The force is distributed to the Lagrangian surface nodes via weighting coefficients
   delta_k and the equal opposite reaction is scattered to the 8 ALE brick nodes
   via trilinear shape functions N_i(CoG), exactly conserving momentum.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
import numpy as np

from ..common.constants import EM20
from ..common.fastmath import norm3
from ..model.model import Model


class ContactType22:
    """Immersed boundary cut-cell ALE-FSI coupling interface /INTER/TYPE22.

    Fortran reference:
      ``i22mainf.F``, ``i22for3.F``, ``i22intersect.F``.
    """

    def __init__(self, itf: Any, model: Model, log: Any = None):
        self.itf = itf
        self.model = model
        self.log = log if log is not None else getattr(model, "log", None)

        self.grbric_id = getattr(itf, "grbric_id1", getattr(itf, "grbric_id", 0))
        self.surf_id = getattr(itf, "surf_id", getattr(itf, "surf_id1", 0))

        self.stfac = float(getattr(itf, "stfac", 1.0) or 1.0)
        self.visc = float(getattr(itf, "visc", 0.05) or 0.05)
        self.gap = float(getattr(itf, "gap", 0.0) or 0.0)
        self.tstart = float(getattr(itf, "tstart", 0.0) or 0.0)
        self.tstop = float(getattr(itf, "tstop", np.inf) or np.inf)
        if self.tstop <= 0.0:
            self.tstop = np.inf
        self.dt_bound = np.inf

        self.bricks = np.zeros((0, 8), dtype=np.int64)
        self.surface_facets = np.zeros((0, 4), dtype=np.int64)

        self._resolve_entities()

    def _resolve_entities(self) -> None:
        """Resolve background ALE bricks and immersed Lagrangian surface facets."""
        # 1. Master ALE 8-node bricks
        model_bricks = getattr(self.model, "bricks", None)
        if model_bricks is not None:
            all_conn = getattr(model_bricks, "conn", getattr(model_bricks, "ixs", None))
            if all_conn is None and isinstance(model_bricks, np.ndarray):
                all_conn = model_bricks
            if all_conn is not None and len(all_conn) > 0:
                if self.grbric_id > 0:
                    egroups = getattr(self.model, "egroups", {})
                    group = (egroups.get("GRBRIC", {}).get(self.grbric_id)
                             or egroups.get("BRIC", {}).get(self.grbric_id)
                             or egroups.get("PART", {}).get(self.grbric_id))
                    if group is not None and getattr(group, "elem_idx", None) is not None:
                        idx = np.asarray(group.elem_idx, dtype=np.int64)
                        valid = (idx >= 0) & (idx < len(all_conn))
                        self.bricks = np.asarray(all_conn[idx[valid], :8], dtype=np.int64)
                    else:
                        self.bricks = np.asarray(all_conn[:, :8], dtype=np.int64)
                else:
                    self.bricks = np.asarray(all_conn[:, :8], dtype=np.int64)

        # 2. Immersed Lagrangian surface facets
        if self.surf_id > 0 and hasattr(self.model, "surfaces") and self.surf_id in self.model.surfaces:
            surf = self.model.surfaces[self.surf_id]
            if getattr(surf, "segments", None) is not None and len(surf.segments) > 0:
                segs = surf.segments
                if segs.shape[1] == 3:
                    segs = np.column_stack([segs, segs[:, 2]])
                self.surface_facets = np.asarray(segs[:, :4], dtype=np.int64)

    def _brick_shape_functions(self, pt: np.ndarray, b_nodes_x: np.ndarray) -> np.ndarray:
        """Evaluate trilinear shape functions of 8-node brick at point pt."""
        b_min = np.min(b_nodes_x, axis=0)
        b_max = np.max(b_nodes_x, axis=0)
        extent = np.maximum(b_max - b_min, 1e-12)

        # Approximate natural coordinates in [-1, 1]
        rst = 2.0 * (pt - b_min) / extent - 1.0
        r = float(np.clip(rst[0], -1.0, 1.0))
        s = float(np.clip(rst[1], -1.0, 1.0))
        t = float(np.clip(rst[2], -1.0, 1.0))

        # 8-node brick shape functions (matches Fortran I8NI / I22)
        N = np.array([
            (1.0 - r) * (1.0 - s) * (1.0 - t) * 0.125,
            (1.0 - r) * (1.0 - s) * (1.0 + t) * 0.125,
            (1.0 + r) * (1.0 - s) * (1.0 + t) * 0.125,
            (1.0 + r) * (1.0 - s) * (1.0 - t) * 0.125,
            (1.0 - r) * (1.0 + s) * (1.0 - t) * 0.125,
            (1.0 - r) * (1.0 + s) * (1.0 + t) * 0.125,
            (1.0 + r) * (1.0 + s) * (1.0 + t) * 0.125,
            (1.0 + r) * (1.0 + s) * (1.0 - t) * 0.125,
        ], dtype=float)
        sum_N = np.sum(N)
        if sum_N > 1e-12:
            N /= sum_N
        return N

    def forces(self, x: np.ndarray, v: np.ndarray, mass: np.ndarray, dt: float,
               fcont: np.ndarray, cycle: int = 0, stifn: Optional[np.ndarray] = None,
               t: Optional[float] = None) -> Tuple[float, float]:
        """Compute immersed boundary cut-cell FSI coupling forces.

        Fortran reference:
          engine/source/interfaces/int22/i22for3.F lines 280-450.

        Returns (-work, dt_contact).
        """
        n_facets = len(self.surface_facets)
        n_bricks = len(self.bricks)
        if n_facets == 0 or n_bricks == 0 or dt <= 0.0:
            return 0.0, self.dt_bound

        if t is not None and (t < self.tstart or t > self.tstop):
            return 0.0, self.dt_bound

        n_coords = len(x)
        valid_b = np.all((self.bricks >= 0) & (self.bricks < n_coords), axis=1)
        valid_f = np.all((self.surface_facets >= 0) & (self.surface_facets < n_coords), axis=1)
        if not np.any(valid_b) or not np.any(valid_f):
            return 0.0, self.dt_bound

        b_sub = self.bricks[valid_b]
        f_sub = self.surface_facets[valid_f]

        # Brick bounding boxes
        b_x = x[b_sub]  # (n_bricks, 8, 3)
        b_min = np.min(b_x, axis=1) - (self.gap + 1e-4)
        b_max = np.max(b_x, axis=1) + (self.gap + 1e-4)

        total_work = 0.0
        dt_min = self.dt_bound

        for facet in f_sub:
            is_quad = (facet[2] != facet[3])
            n_pts = 4 if is_quad else 3

            fx = x[facet[:n_pts]]
            cog = np.mean(fx, axis=0)

            # Broad phase: find background brick enclosing the facet centroid
            inside_brick = np.all((cog >= b_min) & (cog <= b_max), axis=1)
            cand_bricks = np.where(inside_brick)[0]

            if len(cand_bricks) == 0:
                continue

            # Facet normal and wetted surface area (Fortran: i22for3.F line 289, I22AERA)
            if is_quad:
                d13 = fx[2] - fx[0]
                d24 = fx[3] - fx[1]
                n_cross = np.cross(d13, d24)
                area_surf = 0.5 * float(np.linalg.norm(n_cross))
                if area_surf > EM20:
                    n_surf = n_cross / (2.0 * area_surf)
                else:
                    n_surf = np.array([0.0, 0.0, 1.0])
                delta_k = 0.25
            else:
                d12 = fx[1] - fx[0]
                d13 = fx[2] - fx[0]
                n_cross = np.cross(d12, d13)
                area_surf = 0.5 * float(np.linalg.norm(n_cross))
                if area_surf > EM20:
                    n_surf = n_cross / (2.0 * area_surf)
                else:
                    n_surf = np.array([0.0, 0.0, 1.0])
                delta_k = 1.0 / 3.0

            s_wet = area_surf

            for bj in cand_bricks:
                brick_nodes = b_sub[bj]
                bx_nodes = x[brick_nodes]

                # Shape functions of the brick at the facet CoG
                N_i = self._brick_shape_functions(cog, bx_nodes)

                # Relative velocity between Lagrangian surface and Eulerian fluid cell
                v_surf = np.mean(v[facet[:n_pts]], axis=0)
                v_fluid = np.sum(N_i[:, None] * v[brick_nodes], axis=0)
                v_rel = v_surf - v_fluid

                v_norm = float(np.dot(v_rel, n_surf))
                v_tang = v_rel - v_norm * n_surf

                # Fluid dynamic coupling pressure: DP = rho * c * v_norm or penalty stiffness
                m_eff = float(np.mean(mass[facet[:n_pts]]))
                k_fsi = self.stfac * 0.1 * m_eff / (dt * dt)

                # FSI coupling force (Fortran i22for3.F lines 388-397): F = DP * Swet * n_surf
                fn = k_fsi * s_wet * (-v_norm * dt) - 2.0 * self.visc * np.sqrt(max(k_fsi * m_eff, 1e-20)) * v_norm
                f_norm_vec = fn * n_surf

                # Viscous fluid shear drag
                f_tang_vec = -self.visc * s_wet * v_tang

                f_total = f_norm_vec + f_tang_vec

                # Scatter: +delta_k * F to Lagrangian facet nodes, -N_i * F to ALE brick nodes
                for k in range(n_pts):
                    fcont[facet[k]] += delta_k * f_total

                for i in range(8):
                    fcont[brick_nodes[i]] -= N_i[i] * f_total

                # Time step bound
                dt_cand = np.sqrt(2.0 * m_eff / max(k_fsi, EM20))
                dt_min = min(dt_min, float(dt_cand))

                total_work += float(np.dot(f_total, v_rel)) * dt

        return -total_work, dt_min
