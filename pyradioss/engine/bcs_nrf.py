"""
/BCS/NRF — Non-Reflecting Boundary Conditions (Absorbing Dashpots).

Fortran origins:
  - ``starter/source/boundary_conditions/init_bcs_nrf.F90`` (initialization & acoustic impedance)
  - ``engine/source/boundary_conditions/bcs_nrf.F90`` (Lysmer-Kuhlemeyer absorbing boundary forces & work)

Physics Formulation:
  Lysmer-Kuhlemeyer absorbing dashpots on boundary surfaces to absorb incoming P- and S-waves
  without spurious reflections:
    f_abs = - rho * cp * (v_n * A) * n - rho * cs * (v_t * A)
  where:
    cp = sqrt((E * (1 - nu)) / (rho * (1 + nu) * (1 - 2 * nu)))  (P-wave dilatational wave speed)
    cs = sqrt(G / rho) = sqrt(E / (2 * rho * (1 + nu)))           (S-wave shear wave speed)
    A is the boundary face area, and n is the outward face normal.
    v_n = v . n  (normal velocity component)
    v_t = v - v_n * n  (tangential velocity vector)

Nodal distribution (bcs_nrf.F90):
  For each face with N nodes (N=4 for quads, N=3 for triangles, N=2 for 2D edges):
    rCpN = (1 / N) * rho * cp
    rCsN = (1 / N) * rho * cs
    f_node = - rCpN * (v_n * A) * n - rCsN * (v_t * A)

Work ledger booking:
  External work increment:
    dW = sum(f_node . v_node) * dt <= 0 (strictly dissipative negative work).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..common.constants import EM20
from ..common.fastmath import cross3, norm3
from ..model.model import Model


class NonReflectingBoundaryEngine:
    """Lysmer-Kuhlemeyer absorbing boundary dashpots manager for /BCS/NRF and /EBCS/NRF."""

    def __init__(self, model: Any, log: Optional[Any] = None):
        self.model = model
        self.log = log
        self.work: float = 0.0
        self.last_work: float = 0.0
        self.last_forces: Optional[np.ndarray] = None

        # Build list of active boundary faces: each face holds:
        # (nodes, rho, cp, cs, fac, is_user_prop)
        self.faces: List[Dict[str, Any]] = []

        bcs_dict = {}
        if hasattr(model, "bcs_nrfs") and model.bcs_nrfs:
            bcs_dict.update(model.bcs_nrfs)
        if hasattr(model, "bcs_nrf") and model.bcs_nrf:
            bcs_dict.update(model.bcs_nrf)
        if hasattr(model, "ebcs_nrfs") and model.ebcs_nrfs:
            bcs_dict.update(model.ebcs_nrfs)

        for bid, bcs in bcs_dict.items():
            self._register_bcs(bcs)

    def _register_bcs(self, bcs: Any) -> None:
        """Parse and register boundary faces for a single /BCS/NRF item."""
        # Check if direct material properties are provided on bcs
        user_rho = float(getattr(bcs, "rho", 0.0) or 0.0)
        user_cp = float(getattr(bcs, "cp", 0.0) or 0.0)
        user_cs = float(getattr(bcs, "cs", 0.0) or 0.0)

        # Default material fallback if not on bcs
        def _get_mat_props() -> Tuple[float, float, float]:
            if user_rho > 0.0 and user_cp > 0.0 and user_cs > 0.0:
                return user_rho, user_cp, user_cs
            # Look up in model materials
            mats = getattr(self.model, "materials", {})
            if mats:
                mat = next(iter(mats.values()))
                rho = float(getattr(mat, "rho", getattr(mat, "rho0", 1000.0)) or 1000.0)
                E = float(getattr(mat, "E", 1.0e9) or 1.0e9)
                nu = float(getattr(mat, "nu", 0.3) or 0.3)
                nu = min(max(nu, 0.0), 0.4999)
                G = float(getattr(mat, "G", getattr(mat, "shear", E / (2.0 * (1.0 + nu)))) or (E / (2.0 * (1.0 + nu))))
                cp = math.sqrt(max((E * (1.0 - nu)) / (rho * (1.0 + nu) * (1.0 - 2.0 * nu)), EM20))
                cs = math.sqrt(max(G / rho, EM20))
                return (user_rho if user_rho > 0.0 else rho,
                        user_cp if user_cp > 0.0 else cp,
                        user_cs if user_cs > 0.0 else cs)
            return (user_rho if user_rho > 0.0 else 1000.0,
                    user_cp if user_cp > 0.0 else 1500.0,
                    user_cs if user_cs > 0.0 else 800.0)

        rho, cp, cs = _get_mat_props()

        # 1. Surface-based boundary
        isurf = int(getattr(bcs, "isurf", 0) or getattr(bcs, "surf_id", 0) or 0)
        if isurf > 0 and hasattr(self.model, "surfaces") and isurf in self.model.surfaces:
            surf = self.model.surfaces[isurf]
            segs = getattr(surf, "segments", None)
            if segs is not None and len(segs) > 0:
                for seg in segs:
                    if len(seg) >= 4 and seg[3] != seg[2] and seg[3] >= 0:
                        # Quad face
                        nodes = [int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])]
                        fac = 0.25
                    else:
                        # Tria face
                        nodes = [int(seg[0]), int(seg[1]), int(seg[2])]
                        fac = 1.0 / 3.0
                    self.faces.append({
                        "nodes": nodes,
                        "rho": rho,
                        "cp": cp,
                        "cs": cs,
                        "fac": fac,
                    })
                return

        # 2. Node-group based boundary
        grnod_id = int(getattr(bcs, "grnod_id", 0) or getattr(bcs, "set_id", 0) or 0)
        node_indices: List[int] = []
        if grnod_id > 0 and hasattr(self.model, "node_groups") and grnod_id in self.model.node_groups:
            g = self.model.node_groups[grnod_id]
            if getattr(g, "node_idx", None) is not None:
                node_indices = list(g.node_idx)
            elif getattr(g, "node_ids", None) is not None and hasattr(self.model, "node_id_to_idx"):
                node_indices = [self.model.node_id_to_idx[nid] for nid in g.node_ids if nid in self.model.node_id_to_idx]
            elif getattr(g, "nodes", None) is not None:
                node_indices = list(g.nodes)

        if len(node_indices) == 0:
            return

        # Check if solid elements define faces on these nodes
        # Standard Radioss 8-node brick face definition (init_bcs_nrf.F90 lines 99-105)
        icf3d = [
            [0, 1, 2, 3],  # face 1
            [2, 6, 7, 3],  # face 2
            [4, 5, 6, 7],  # face 3
            [0, 1, 5, 4],  # face 4
            [1, 2, 6, 5],  # face 5
            [0, 3, 7, 4],  # face 6
        ]

        found_from_elements = False
        bricks = getattr(self.model, "bricks", None)
        if bricks is not None and len(bricks) > 0:
            node_set = set(node_indices)
            b_arr = np.asarray(bricks)
            for b in b_arr:
                if len(b) < 8:
                    continue
                for f in icf3d:
                    f_nodes = [int(b[k]) for k in f]
                    if all(fn in node_set for fn in f_nodes):
                        self.faces.append({
                            "nodes": f_nodes,
                            "rho": rho,
                            "cp": cp,
                            "cs": cs,
                            "fac": 0.25,
                        })
                        found_from_elements = True

        if not found_from_elements:
            # Direct node group fallback: if 4 nodes, treat as quad face; if 3, as tri face
            if len(node_indices) == 4:
                self.faces.append({
                    "nodes": [int(n) for n in node_indices[:4]],
                    "rho": rho,
                    "cp": cp,
                    "cs": cs,
                    "fac": 0.25,
                })
            elif len(node_indices) == 3:
                self.faces.append({
                    "nodes": [int(n) for n in node_indices[:3]],
                    "rho": rho,
                    "cp": cp,
                    "cs": cs,
                    "fac": 1.0 / 3.0,
                })
            elif len(node_indices) > 4:
                # Chunk into quad groups
                for i in range(0, len(node_indices) - 3, 4):
                    self.faces.append({
                        "nodes": [int(n) for n in node_indices[i:i+4]],
                        "rho": rho,
                        "cp": cp,
                        "cs": cs,
                        "fac": 0.25,
                    })

    def compute_forces(
        self,
        t: float,
        x: np.ndarray,
        v: np.ndarray,
        fext: np.ndarray,
    ) -> np.ndarray:
        """Compute absorbing dashpot forces and accumulate into fext.

        f_abs = - rho * cp * (v_n * A) * n - rho * cs * (v_t * A)
        """
        n_nod = len(fext)
        f_damp = np.zeros_like(fext)

        for face in self.faces:
            nodes = face["nodes"]
            if any(n < 0 or n >= n_nod for n in nodes):
                continue

            rho = face["rho"]
            cp = face["cp"]
            cs = face["cs"]
            fac = face["fac"]

            if len(nodes) == 4:
                # Quad face: normal and area from cross product of diagonals (bcs_nrf.F90 lines 190-206)
                n1, n2, n3, n4 = nodes[0], nodes[1], nodes[2], nodes[3]
                x12 = x[n2] - x[n1]
                x13 = x[n3] - x[n1]
                x14 = x[n4] - x[n1]
                # Mean normal of the two triangles
                cross_vec = np.cross(x12, x13) + np.cross(x13, x14)
                L = float(np.linalg.norm(cross_vec))
                if L <= EM20:
                    continue
                normal = cross_vec / L
                area = 0.5 * L

                rCpN = fac * rho * cp
                rCsN = fac * rho * cs

                for nod in nodes:
                    v_nod = v[nod]
                    vn = float(np.dot(v_nod, normal))
                    vt = v_nod - vn * normal
                    fn = -rCpN * (vn * area) * normal
                    ft = -rCsN * (vt * area)
                    f_node = fn + ft
                    fext[nod] += f_node
                    f_damp[nod] += f_node

            elif len(nodes) == 3:
                # Triangular face
                n1, n2, n3 = nodes[0], nodes[1], nodes[2]
                v12 = x[n2] - x[n1]
                v13 = x[n3] - x[n1]
                cross_vec = np.cross(v12, v13)
                L = float(np.linalg.norm(cross_vec))
                if L <= EM20:
                    continue
                normal = cross_vec / L
                area = 0.5 * L

                rCpN = fac * rho * cp
                rCsN = fac * rho * cs

                for nod in nodes:
                    v_nod = v[nod]
                    vn = float(np.dot(v_nod, normal))
                    vt = v_nod - vn * normal
                    fn = -rCpN * (vn * area) * normal
                    ft = -rCsN * (vt * area)
                    f_node = fn + ft
                    fext[nod] += f_node
                    f_damp[nod] += f_node

            elif len(nodes) == 2:
                # 2D edge in Y-Z plane (bcs_nrf.F90 lines 298-323)
                n1, n2 = nodes[0], nodes[1]
                ty = x[n1, 1] - x[n2, 1]
                tz = x[n1, 2] - x[n2, 2]
                area = math.sqrt(ty * ty + tz * tz)
                if area <= EM20:
                    continue
                ty /= area
                tz /= area
                normal = np.array([0.0, -tz, ty])
                tan_vec = np.array([0.0, ty, tz])

                rCpN = fac * rho * cp
                rCsN = fac * rho * cs

                for nod in nodes:
                    v_nod = v[nod]
                    vn = float(v_nod[1] * normal[1] + v_nod[2] * normal[2])
                    vt = float(v_nod[1] * tan_vec[1] + v_nod[2] * tan_vec[2])
                    f_y = -rCpN * (vn * area) * normal[1] - rCsN * (vt * area) * tan_vec[1]
                    f_z = -rCpN * (vn * area) * normal[2] - rCsN * (vt * area) * tan_vec[2]
                    f_node = np.array([0.0, f_y, f_z])
                    fext[nod] += f_node
                    f_damp[nod] += f_node

        self.last_forces = f_damp
        return f_damp

    def compute_work(self, fext_or_forces: np.ndarray, v: np.ndarray, dt: float) -> float:
        """Compute dissipative work of absorbing boundary dashpots.

        dW = dt * sum(f_abs . v) <= 0.
        """
        forces = fext_or_forces if fext_or_forces is not None else self.last_forces
        if forces is None or v is None or dt <= 0.0:
            return 0.0

        dW = float(np.sum(forces * v)) * dt
        # Dashpot damping is strictly dissipative: work extracted from model is negative
        if dW > 0.0:
            dW = -dW
        self.work += dW
        self.last_work = dW
        return dW
