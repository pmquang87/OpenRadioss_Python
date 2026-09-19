"""
/BCS/CYCLIC — Cyclic Sector Symmetry Boundary Condition.

Fortran origins:
  - ``engine/source/constraints/general/bcs/bcscyc.F`` (BCSCYC, ACC_CYCON, V_C2CYLIN, V_CYC2C, CHKV0_CY)
  - ``engine/source/ams/sms_bcscyc.F`` (AMS cyclic symmetry solver)

Physics & Kinematics Formulation:
  Lagrangian cyclic sector symmetry constraint coupling paired boundary node groups
  across rotational sector cut planes around a skew coordinate rotation axis:
  1. Coordinate transform to skew coordinate system:
       x_local = S @ (x - x_orig)
     where S is the 3x3 skew transformation matrix and x_orig is the skew origin.
  2. Cylindrical components (r, theta, z):
       r = sqrt(x_local_x^2 + x_local_y^2)
       cos_theta = x_local_x / r, sin_theta = x_local_y / r
       z = x_local_z
  3. Transform vector field (velocity, acceleration, position offset) to cylindrical:
       w = S @ v
       v_r = w_x * cos_theta + w_y * sin_theta
       v_theta = w_y * cos_theta - w_x * sin_theta
       v_z = w_z
  4. Average cylindrical components between paired sector nodes (n1, n2):
       v_r_avg = 0.5 * (v_r(n1) + v_r(n2))
       v_theta_avg = 0.5 * (v_theta(n1) + v_theta(n2))
       v_z_avg = 0.5 * (v_z(n1) + v_z(n2))
  5. Transform back to Cartesian at each node's own local azimuth:
       w_new(ni) = [v_r_avg * cos(theta_i) - v_theta_avg * sin(theta_i),
                    v_r_avg * sin(theta_i) + v_theta_avg * cos(theta_i),
                    v_z_avg]
       v_new(ni) = S.T @ w_new(ni)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..common.constants import EM20
from ..model.model import Model


class CyclicBoundaryEngine:
    """Cyclic sector symmetry kinematic boundary condition coordinator for /BCS/CYCLIC."""

    def __init__(self, model: Any, log: Optional[Any] = None):
        self.model = model
        self.log = log
        # Each paired set: {"skew_id": int, "pairs": [(idx1, idx2), ...]}
        self.sectors: List[Dict[str, Any]] = []

        raw_bcs = []
        if hasattr(model, "bcs_cyclics") and model.bcs_cyclics:
            raw_bcs.extend(model.bcs_cyclics.values())
        if hasattr(model, "cyclic_bcs") and model.cyclic_bcs:
            for cb in model.cyclic_bcs.values():
                if cb not in raw_bcs:
                    raw_bcs.append(cb)

        for cb in raw_bcs:
            self._register_cyclic_bcs(cb)

    def _resolve_nodes(self, grnd_id: int) -> List[int]:
        """Resolve node group ID to 0-based node indices."""
        if grnd_id <= 0 or not hasattr(self.model, "node_groups"):
            return []
        if grnd_id not in self.model.node_groups:
            return []
        g = self.model.node_groups[grnd_id]
        if getattr(g, "node_idx", None) is not None:
            return [int(idx) for idx in g.node_idx if 0 <= idx < self.model.numnod]
        if getattr(g, "node_ids", None) is not None and hasattr(self.model, "node_id_to_idx"):
            return [int(self.model.node_id_to_idx[nid]) for nid in g.node_ids
                    if nid in self.model.node_id_to_idx and 0 <= self.model.node_id_to_idx[nid] < self.model.numnod]
        if getattr(g, "nodes", None) is not None:
            return [int(idx) for idx in g.nodes if 0 <= idx < self.model.numnod]
        return []

    def _get_skew_matrix_and_orig(self, skew_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve 3x3 rotation matrix and origin for a skew system."""
        if skew_id <= 0 or not hasattr(self.model, "skews") or not self.model.skews:
            return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

        skews = self.model.skews
        orig = np.zeros(3, dtype=np.float64)
        if hasattr(skews, "orig") and skew_id in skews.orig:
            orig = np.asarray(skews.orig[skew_id], dtype=np.float64)

        axes = np.eye(3, dtype=np.float64)
        if hasattr(skews, "axes") and skew_id in skews.axes:
            axes = np.asarray(skews.axes[skew_id], dtype=np.float64)
        elif hasattr(skews, "get_axes"):
            axes = np.asarray(skews.get_axes(skew_id), dtype=np.float64)

        return axes, orig

    def _register_cyclic_bcs(self, cb: Any) -> None:
        """Pair nodes from sector 1 and sector 2."""
        skew_id = int(getattr(cb, "skew_id", 0) or 0)
        grnd1 = int(getattr(cb, "grnd_id1", 0) or getattr(cb, "grnod1_id", 0) or 0)
        grnd2 = int(getattr(cb, "grnd_id2", 0) or getattr(cb, "grnod2_id", 0) or 0)

        nodes1 = self._resolve_nodes(grnd1)
        nodes2 = self._resolve_nodes(grnd2)
        if not nodes1 or not nodes2:
            return

        axes, orig = self._get_skew_matrix_and_orig(skew_id)
        x_all = getattr(self.model, "x", getattr(self.model, "x0", None))

        pairs: List[Tuple[int, int]] = []
        if len(nodes1) == len(nodes2) and x_all is not None and len(x_all) > max(max(nodes1), max(nodes2)):
            # Geometric matching by cylindrical (r, z)
            def _cyl(idx_list: List[int]) -> np.ndarray:
                coords = x_all[idx_list] - orig
                local = np.einsum("ij,nj->ni", axes, coords)
                r = np.sqrt(local[:, 0]**2 + local[:, 1]**2)
                z = local[:, 2]
                return np.column_stack([r, z])

            rz1 = _cyl(nodes1)
            rz2 = _cyl(nodes2)

            matched_2 = set()
            for i, n1 in enumerate(nodes1):
                best_j = -1
                best_dist = float("inf")
                for j, n2 in enumerate(nodes2):
                    if j in matched_2:
                        continue
                    d = math.hypot(rz1[i, 0] - rz2[j, 0], rz1[i, 1] - rz2[j, 1])
                    if d < best_dist:
                        best_dist = d
                        best_j = j
                if best_j >= 0:
                    matched_2.add(best_j)
                    pairs.append((n1, nodes2[best_j]))
                else:
                    pairs.append((n1, nodes2[i]))
        else:
            # Direct 1-to-1 index matching
            for n1, n2 in zip(nodes1, nodes2):
                pairs.append((n1, n2))

        self.sectors.append({
            "skew_id": skew_id,
            "pairs": pairs,
        })

    def apply(
        self,
        v: np.ndarray,
        vr: Optional[np.ndarray] = None,
        a: Optional[np.ndarray] = None,
        x: Optional[np.ndarray] = None,
    ) -> None:
        """Apply cyclic sector symmetry averaging to velocities, angular velocities, and accelerations."""
        if not self.sectors or v is None:
            return

        coords = x if x is not None else getattr(self.model, "x", getattr(self.model, "x0", None))
        if coords is None:
            return

        for sector in self.sectors:
            skew_id = sector["skew_id"]
            pairs = sector["pairs"]
            axes, orig = self._get_skew_matrix_and_orig(skew_id)

            for n1, n2 in pairs:
                if n1 >= len(v) or n2 >= len(v):
                    continue

                # Cylindrical azimuth for each node
                p1 = coords[n1] - orig
                p2 = coords[n2] - orig
                q1 = axes @ p1
                q2 = axes @ p2

                r1 = math.sqrt(q1[0] * q1[0] + q1[1] * q1[1])
                r2 = math.sqrt(q2[0] * q2[0] + q2[1] * q2[1])

                cos1 = q1[0] / r1 if r1 > EM20 else 1.0
                sin1 = q1[1] / r1 if r1 > EM20 else 0.0
                cos2 = q2[0] / r2 if r2 > EM20 else 1.0
                sin2 = q2[1] / r2 if r2 > EM20 else 0.0

                # 1. Linear velocity v (bcscyc.F V_C2CYLIN & V_CYC2C)
                w1 = axes @ v[n1]
                w2 = axes @ v[n2]

                vr1 = w1[0] * cos1 + w1[1] * sin1
                vt1 = w1[1] * cos1 - w1[0] * sin1
                vz1 = w1[2]

                vr2 = w2[0] * cos2 + w2[1] * sin2
                vt2 = w2[1] * cos2 - w2[0] * sin2
                vz2 = w2[2]

                vr_avg = 0.5 * (vr1 + vr2)
                vt_avg = 0.5 * (vt1 + vt2)
                vz_avg = 0.5 * (vz1 + vz2)

                # Back to Cartesian at node 1 local azimuth
                w1_avg = np.array([
                    vr_avg * cos1 - vt_avg * sin1,
                    vr_avg * sin1 + vt_avg * cos1,
                    vz_avg,
                ])
                v[n1] = axes.T @ w1_avg

                # Back to Cartesian at node 2 local azimuth
                w2_avg = np.array([
                    vr_avg * cos2 - vt_avg * sin2,
                    vr_avg * sin2 + vt_avg * cos2,
                    vz_avg,
                ])
                v[n2] = axes.T @ w2_avg

                # 2. Acceleration a (if provided)
                if a is not None and n1 < len(a) and n2 < len(a):
                    aw1 = axes @ a[n1]
                    aw2 = axes @ a[n2]
                    ar1 = aw1[0] * cos1 + aw1[1] * sin1
                    at1 = aw1[1] * cos1 - aw1[0] * sin1
                    az1 = aw1[2]
                    ar2 = aw2[0] * cos2 + aw2[1] * sin2
                    at2 = aw2[1] * cos2 - aw2[0] * sin2
                    az2 = aw2[2]
                    ar_avg = 0.5 * (ar1 + ar2)
                    at_avg = 0.5 * (at1 + at2)
                    az_avg = 0.5 * (az1 + az2)
                    aw1_avg = np.array([ar_avg * cos1 - at_avg * sin1, ar_avg * sin1 + at_avg * cos1, az_avg])
                    aw2_avg = np.array([ar_avg * cos2 - at_avg * sin2, ar_avg * sin2 + at_avg * cos2, az_avg])
                    a[n1] = axes.T @ aw1_avg
                    a[n2] = axes.T @ aw2_avg

                # 3. Rotational velocity vr (if provided)
                if vr is not None and n1 < len(vr) and n2 < len(vr):
                    rw1 = axes @ vr[n1]
                    rw2 = axes @ vr[n2]
                    rr1 = rw1[0] * cos1 + rw1[1] * sin1
                    rt1 = rw1[1] * cos1 - rw1[0] * sin1
                    rz1 = rw1[2]
                    rr2 = rw2[0] * cos2 + rw2[1] * sin2
                    rt2 = rw2[1] * cos2 - rw2[0] * sin2
                    rz2 = rw2[2]
                    rr_avg = 0.5 * (rr1 + rr2)
                    rt_avg = 0.5 * (rt1 + rt2)
                    rz_avg = 0.5 * (rz1 + rz2)
                    rw1_avg = np.array([rr_avg * cos1 - rt_avg * sin1, rr_avg * sin1 + rt_avg * cos1, rz_avg])
                    rw2_avg = np.array([rr_avg * cos2 - rt_avg * sin2, rr_avg * sin2 + rt_avg * cos2, rz_avg])
                    vr[n1] = axes.T @ rw1_avg
                    vr[n2] = axes.T @ rw2_avg

                # 4. Position averaging (if x provided explicitly)
                if x is not None and n1 < len(x) and n2 < len(x):
                    r_avg = 0.5 * (r1 + r2)
                    z_avg = 0.5 * (q1[2] + q2[2])
                    q1_avg = np.array([r_avg * cos1, r_avg * sin1, z_avg])
                    q2_avg = np.array([r_avg * cos2, r_avg * sin2, z_avg])
                    x[n1] = orig + axes.T @ q1_avg
                    x[n2] = orig + axes.T @ q2_avg

    def enforce(
        self,
        x: np.ndarray,
        v: np.ndarray,
        a: Optional[np.ndarray] = None,
    ) -> None:
        """Enforce cyclic sector symmetry on x and v (alias conforming to engine contract)."""
        self.apply(v=v, a=a, x=x)
