"""
/GJOINT — General Mechanism Joints, engine side (M594).

Fortran origin:
  - ``starter/source/constraints/general/gjoint/hm_read_gjoint.F``
  - ``engine/source/tools/lagmul/lag_gjnt.F`` (manager, carrier frame actualisation)
  - ``engine/source/tools/lagmul/gjnt_gear.F`` (11-DOF gear mechanism)
  - ``engine/source/tools/lagmul/gjnt_diff.F`` (13-DOF automotive differential)
  - ``engine/source/tools/lagmul/gjnt_rack.F`` (12-DOF rack-and-pinion)
  - ``engine/source/tools/skew/rotbmr.F`` (carrier frame finite rotation)
  - ``hm_cfg_files/config/CFG/radioss110/RBODY/gjoint.cfg``

Kinematic Formulation & Mechanism Types:
----------------------------------------
1. Type 1 (GEAR): Transmission ratio alpha (gjnt_gear.F).
   - Gear transmission along shaft axes a1, a2 relative to carrier N0:
       (omega_2 - omega_0) . a2 = -alpha * [(omega_1 - omega_0) . a1]
   - Transverse shaft locking: gears cannot wobble off shaft axes:
       (omega_1 - omega_0) . y_T1 = 0,  (omega_1 - omega_0) . z_T1 = 0
       (omega_2 - omega_0) . y_T2 = 0,  (omega_2 - omega_0) . z_T2 = 0
   - Total constraints: 11 kinematic conditions.
   - Reactions: T1 = -alpha * T2, T0 = -(T1 + T2) [conserving angular momentum].

2. Type 2 (DIFF): Automotive differential mechanism (gjnt_diff.F).
   - Speed averaging: 2 * (omega_0 . a0) = (omega_1 . a1) + (omega_2 . a2)
     or 4-node: (omega_2 - omega_0) . a2 + (omega_3 - omega_0) . a3 = 2 * alpha * (omega_1 - omega_0) . a1
   - Transverse shaft locking for all axles.
   - Total constraints: 13 kinematic conditions.
   - Reactions: T0 = -(T1 + T2 [+ T3]).

3. Type 3 (RACK): Rack and pinion mechanism (gjnt_rack.F).
   - Kinematic rack transmission: (v_2 - v_0) . a2 = alpha * [(omega_1 - omega_0) . a1]
   - Pinion transverse rotation tied to carrier: (omega_1 - omega_0) . y_T1 = 0, (omega_1 - omega_0) . z_T1 = 0
   - Rack rotation tied to carrier: omega_2 - omega_0 = 0
   - Rack transverse translation locked: (v_2 - v_0) . y_T2 = 0, (v_2 - v_0) . z_T2 = 0
   - Total constraints: 12 kinematic conditions.
   - Reactions: F0 = -F2, T1 = -alpha * F2, T0 = -T1 - (x2 - x0) x F2.

4. Type 4 (CV): Constant Velocity joint.
   - 1:1 angular speed: (omega_2 - omega_0) . a2 = (omega_1 - omega_0) . a1
   - Total constraints: 7 kinematic conditions.
   - Reactions: T1 = -T2, T0 = -(T1 + T2).
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, List
import numpy as np

from ..common.messages import MessageLog
from ..model.entities import GJoint
from ..model.model import Model


def _normalize(vec: Tuple[float, float, float] | np.ndarray, default: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    """Normalize 3D vector to unit length."""
    v = np.array(vec, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm > 1e-12:
        return v / norm
    return np.array(default, dtype=np.float64)


def _compute_transverse_basis(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute orthonormal transverse vectors (y, z) perpendicular to shaft axis u.

    Fortran origin: ``engine/source/tools/lagmul/gjnt_gear.F`` lines 40-75.
    """
    u_norm = _normalize(u)
    if abs(u_norm[0]) > 1e-6:
        z = np.array([-u_norm[2], 0.0, u_norm[0]], dtype=np.float64)
    else:
        z = np.array([0.0, -u_norm[2], u_norm[1]], dtype=np.float64)
    z = _normalize(z, default=(0.0, 1.0, 0.0))
    y = np.cross(z, u_norm)
    y = _normalize(y, default=(0.0, 0.0, 1.0))
    return y, z


def _exp_rodrigues(w: np.ndarray, dt: float) -> np.ndarray:
    """Rodrigues formula for finite rotation matrix exp([w dt]).

    Fortran origin: ``engine/source/tools/skew/rotbmr.F``.
    """
    th = float(np.linalg.norm(w)) * dt
    if th < 1e-12:
        K = np.array([[0.0, -w[2], w[1]],
                      [w[2], 0.0, -w[0]],
                      [-w[1], w[0], 0.0]], dtype=np.float64)
        return np.eye(3, dtype=np.float64) + dt * K
    k = w / np.linalg.norm(w)
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


class GJointEngine:
    """General mechanism joint engine constraint (GEAR, DIFF, RACK, CV)."""

    def __init__(self, joint: GJoint, model: Model, log: MessageLog):
        self.joint = joint
        self.id = joint.id
        self.model = model
        self.log = log

        sub = (joint.subtype or "DEFAULT").upper()
        if sub in ("GEAR", "DEFAULT"):
            self.joint_type = 1
            self.subtype = "GEAR"
        elif sub == "DIFF":
            self.joint_type = 2
            self.subtype = "DIFF"
        elif sub == "RACK":
            self.joint_type = 3
            self.subtype = "RACK"
        elif sub == "CV":
            self.joint_type = 4
            self.subtype = "CV"
        else:
            self.joint_type = 1
            self.subtype = "GEAR"

        # Transmission ratio alpha (fscale in Radioss deck)
        self.alpha: float = float(joint.fscale if joint.fscale != 0.0 else 1.0)

        # Node index resolution
        self.idx0: Optional[int] = model.node_index(joint.node_id0) if joint.node_id0 and joint.node_id0 in model._id2idx else None
        self.idx1: Optional[int] = model.node_index(joint.node_id1) if joint.node_id1 and joint.node_id1 in model._id2idx else None
        self.idx2: Optional[int] = model.node_index(joint.node_id2) if joint.node_id2 and joint.node_id2 in model._id2idx else None
        self.idx3: Optional[int] = model.node_index(joint.node_id3) if getattr(joint, "node_id3", 0) and joint.node_id3 in model._id2idx else None

        self.is_valid: bool = (self.idx1 is not None and self.idx2 is not None)
        if not self.is_valid:
            return

        # Local unit axes
        self.a1: np.ndarray = _normalize(joint.r1, default=(1.0, 0.0, 0.0))
        self.a2: np.ndarray = _normalize(joint.r2, default=(1.0, 0.0, 0.0))
        self.a3: np.ndarray = _normalize(getattr(joint, "r3", (1.0, 0.0, 0.0)), default=(1.0, 0.0, 0.0))
        self.a0: np.ndarray = self.a1.copy()

        # Carrier frame tracking (ROTBMR in lag_gjnt.F)
        self.has_skew: bool = bool(getattr(joint, "skew_id", 0))
        self.R_carrier: np.ndarray = np.eye(3, dtype=np.float64)
        self.a1_init: np.ndarray = self.a1.copy()
        self.a2_init: np.ndarray = self.a2.copy()
        self.a3_init: np.ndarray = self.a3.copy()
        self.a0_init: np.ndarray = self.a0.copy()

        # Initial reference positions for drift correction
        self._x1_init = model.x[self.idx1].copy() if self.idx1 is not None else np.zeros(3)
        self._x2_init = model.x[self.idx2].copy() if self.idx2 is not None else np.zeros(3)
        self._x0_init = model.x[self.idx0].copy() if self.idx0 is not None else np.zeros(3)

    def update_carrier_frame(self, vr0: Optional[np.ndarray], dt: float) -> None:
        """Update carrier local frame orientation using ROTBMR (lag_gjnt.F lines 50-80)."""
        if dt > 0.0 and vr0 is not None:
            norm_w = float(np.linalg.norm(vr0))
            if norm_w > 1e-12:
                dR = _exp_rodrigues(vr0, dt)
                self.R_carrier = dR @ self.R_carrier
                self.a1 = _normalize(self.R_carrier @ self.a1_init)
                self.a2 = _normalize(self.R_carrier @ self.a2_init)
                self.a3 = _normalize(self.R_carrier @ self.a3_init)
                self.a0 = _normalize(self.R_carrier @ self.a0_init)

    def _is_node_fixed(self, idx: Optional[int]) -> bool:
        """Check if node has boundary condition fixing all degrees of freedom."""
        if idx is None:
            return True
        nid = self.model.nodes[idx] if hasattr(self.model, "nodes") and idx < len(self.model.nodes) else 0
        if hasattr(self.model, "bcs") and self.model.bcs:
            bcs_list = self.model.bcs.values() if isinstance(self.model.bcs, dict) else self.model.bcs
            for bcs in bcs_list:
                if nid in getattr(bcs, "node_ids", []):
                    if all(getattr(bcs, "tra", (False, False, False))) and all(getattr(bcs, "rot", (False, False, False))):
                        return True
        return False

    def _get_node_inertia(self, idx: Optional[int], inertia: Optional[np.ndarray], mass: Optional[np.ndarray]) -> float:
        """Get effective rotational inertia of node."""
        if idx is None:
            return 1e30
        if self._is_node_fixed(idx):
            return 1e30

        iner = float(inertia[idx]) if inertia is not None and idx < len(inertia) else 0.0
        if iner <= 0.0 or iner >= 1e29:
            # Check added inertia from joint
            if idx == self.idx0 and self.joint.inertia0 > 0.0:
                iner = self.joint.inertia0
            elif idx == self.idx1 and self.joint.inertia1 > 0.0:
                iner = self.joint.inertia1
            elif idx == self.idx2 and self.joint.inertia2 > 0.0:
                iner = self.joint.inertia2
            elif idx == self.idx3 and getattr(self.joint, "inertia3", 0.0) > 0.0:
                iner = self.joint.inertia3
            elif mass is not None and idx < len(mass) and 0.0 < mass[idx] < 1e28:
                iner = float(mass[idx]) * 1.0  # nominal radius r=1.0
            else:
                iner = 1.0
        return max(iner, 1e-12)

    def _get_node_mass(self, idx: Optional[int], mass: Optional[np.ndarray]) -> float:
        """Get effective translational mass of node."""
        if idx is None:
            return 1e30
        if self._is_node_fixed(idx):
            return 1e30

        m = float(mass[idx]) if mass is not None and idx < len(mass) else 0.0
        if m <= 0.0 or m >= 1e29:
            if idx == self.idx0 and self.joint.mass0 > 0.0:
                m = self.joint.mass0
            elif idx == self.idx1 and self.joint.mass1 > 0.0:
                m = self.joint.mass1
            elif idx == self.idx2 and self.joint.mass2 > 0.0:
                m = self.joint.mass2
            elif idx == self.idx3 and getattr(self.joint, "mass3", 0.0) > 0.0:
                m = self.joint.mass3
            else:
                m = 1.0
        return max(m, 1e-12)

    def get_constraint_equation(self, v: np.ndarray, vr: Optional[np.ndarray], x: np.ndarray) -> Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """Evaluate primary constraint residual C and gradient vectors g_v, g_omega.

        Returns:
            C: scalar constraint violation
            g_v: dict mapping node index to translational gradient vector (3,)
            g_omega: dict mapping node index to rotational gradient vector (3,)
        """
        g_v: Dict[int, np.ndarray] = {}
        g_omega: Dict[int, np.ndarray] = {}

        v0 = v[self.idx0] if self.idx0 is not None else np.zeros(3)
        vr0 = vr[self.idx0] if (vr is not None and self.idx0 is not None) else np.zeros(3)
        v1 = v[self.idx1] if self.idx1 is not None else np.zeros(3)
        vr1 = vr[self.idx1] if (vr is not None and self.idx1 is not None) else np.zeros(3)
        v2 = v[self.idx2] if self.idx2 is not None else np.zeros(3)
        vr2 = vr[self.idx2] if (vr is not None and self.idx2 is not None) else np.zeros(3)

        if self.joint_type == 1:  # GEAR
            # (omega_2 - omega_0) . a2 = -alpha * [(omega_1 - omega_0) . a1]
            # Constraint: [(omega_2 - omega_0) . a2] + alpha * [(omega_1 - omega_0) . a1] = 0
            w1_rel = float(np.dot(vr1 - vr0, self.a1))
            w2_rel = float(np.dot(vr2 - vr0, self.a2))
            C = w2_rel + self.alpha * w1_rel

            g_omega[self.idx1] = self.alpha * self.a1
            g_omega[self.idx2] = self.a2.copy()
            if self.idx0 is not None:
                g_omega[self.idx0] = -(self.alpha * self.a1 + self.a2)

        elif self.joint_type == 2:  # DIFF
            if self.idx3 is not None:
                # 4-node differential with input shaft N1, output axles N2, N3
                vr3 = vr[self.idx3] if vr is not None else np.zeros(3)
                w1_rel = float(np.dot(vr1 - vr0, self.a1))
                w2_rel = float(np.dot(vr2 - vr0, self.a2))
                w3_rel = float(np.dot(vr3 - vr0, self.a3))
                # 2 * alpha * w1 = w2 + w3
                C = w2_rel + w3_rel - 2.0 * self.alpha * w1_rel

                g_omega[self.idx1] = -2.0 * self.alpha * self.a1
                g_omega[self.idx2] = self.a2.copy()
                g_omega[self.idx3] = self.a3.copy()
                if self.idx0 is not None:
                    g_omega[self.idx0] = 2.0 * self.alpha * self.a1 - self.a2 - self.a3
            else:
                # 3-node differential: 2 * (omega_0 . a0) = (omega_1 . a1) + (omega_2 . a2)
                # C = (omega_1 . a1) + (omega_2 . a2) - 2 * (omega_0 . a0)
                w0 = float(np.dot(vr0, self.a0))
                w1 = float(np.dot(vr1, self.a1))
                w2 = float(np.dot(vr2, self.a2))
                C = w1 + w2 - 2.0 * w0

                g_omega[self.idx1] = self.a1.copy()
                g_omega[self.idx2] = self.a2.copy()
                if self.idx0 is not None:
                    g_omega[self.idx0] = -2.0 * self.a0

        elif self.joint_type == 3:  # RACK
            # (v_2 - v_0) . a2 = alpha * [(omega_1 - omega_0) . a1]
            # C = (v_2 - v_0) . a2 - alpha * (omega_1 - omega_0) . a1 = 0
            v_rel = float(np.dot(v2 - v0, self.a2))
            w_rel = float(np.dot(vr1 - vr0, self.a1))
            C = v_rel - self.alpha * w_rel

            g_v[self.idx2] = self.a2.copy()
            g_omega[self.idx1] = -self.alpha * self.a1
            if self.idx0 is not None:
                g_v[self.idx0] = -self.a2.copy()
                g_omega[self.idx0] = self.alpha * self.a1

        elif self.joint_type == 4:  # CV
            # (omega_2 - omega_0) . a2 = (omega_1 - omega_0) . a1
            # C = (omega_2 - omega_0) . a2 - (omega_1 - omega_0) . a1 = 0
            w1_rel = float(np.dot(vr1 - vr0, self.a1))
            w2_rel = float(np.dot(vr2 - vr0, self.a2))
            C = w2_rel - w1_rel

            g_omega[self.idx1] = -self.a1.copy()
            g_omega[self.idx2] = self.a2.copy()
            if self.idx0 is not None:
                g_omega[self.idx0] = self.a1 - self.a2

        return C, g_v, g_omega

    def get_all_constraint_equations(
        self, v: np.ndarray, vr: Optional[np.ndarray], x: np.ndarray
    ) -> List[Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]]:
        """Evaluate all mechanism kinematic constraints (transmission + transverse locking + translations).

        Fortran origin:
          - ``gjnt_gear.F`` (11 constraints)
          - ``gjnt_diff.F`` (13 constraints)
          - ``gjnt_rack.F`` (12 constraints)
          - ``lag_gjnt.F`` (manager)
        """
        eqs: List[Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]] = []
        if not self.is_valid:
            return eqs

        # 1. Primary transmission constraint (index 0)
        eqs.append(self.get_constraint_equation(v, vr, x))

        v0 = v[self.idx0] if self.idx0 is not None else np.zeros(3)
        vr0 = vr[self.idx0] if (vr is not None and self.idx0 is not None) else np.zeros(3)
        v1 = v[self.idx1] if self.idx1 is not None else np.zeros(3)
        vr1 = vr[self.idx1] if (vr is not None and self.idx1 is not None) else np.zeros(3)
        v2 = v[self.idx2] if self.idx2 is not None else np.zeros(3)
        vr2 = vr[self.idx2] if (vr is not None and self.idx2 is not None) else np.zeros(3)

        if self.joint_type == 1:  # GEAR (gjnt_gear.F)
            # Transverse rotational locking: shafts cannot wobble off axes a1 and a2
            y1, z1 = _compute_transverse_basis(self.a1)
            y2, z2 = _compute_transverse_basis(self.a2)
            if vr is not None:
                w1_rel = vr1 - vr0
                C_y1 = float(np.dot(w1_rel, y1))
                g_o_y1 = {self.idx1: y1.copy()}
                if self.idx0 is not None:
                    g_o_y1[self.idx0] = -y1.copy()
                eqs.append((C_y1, {}, g_o_y1))

                C_z1 = float(np.dot(w1_rel, z1))
                g_o_z1 = {self.idx1: z1.copy()}
                if self.idx0 is not None:
                    g_o_z1[self.idx0] = -z1.copy()
                eqs.append((C_z1, {}, g_o_z1))

                w2_rel = vr2 - vr0
                C_y2 = float(np.dot(w2_rel, y2))
                g_o_y2 = {self.idx2: y2.copy()}
                if self.idx0 is not None:
                    g_o_y2[self.idx0] = -y2.copy()
                eqs.append((C_y2, {}, g_o_y2))

                C_z2 = float(np.dot(w2_rel, z2))
                g_o_z2 = {self.idx2: z2.copy()}
                if self.idx0 is not None:
                    g_o_z2[self.idx0] = -z2.copy()
                eqs.append((C_z2, {}, g_o_z2))

        elif self.joint_type == 2:  # DIFF (gjnt_diff.F)
            y1, z1 = _compute_transverse_basis(self.a1)
            y2, z2 = _compute_transverse_basis(self.a2)
            if vr is not None:
                w1_rel = vr1 - vr0
                w2_rel = vr2 - vr0
                for idx, w_rel, (y_ax, z_ax) in [(self.idx1, w1_rel, (y1, z1)), (self.idx2, w2_rel, (y2, z2))]:
                    C_y = float(np.dot(w_rel, y_ax))
                    g_o_y = {idx: y_ax.copy()}
                    if self.idx0 is not None:
                        g_o_y[self.idx0] = -y_ax.copy()
                    eqs.append((C_y, {}, g_o_y))

                    C_z = float(np.dot(w_rel, z_ax))
                    g_o_z = {idx: z_ax.copy()}
                    if self.idx0 is not None:
                        g_o_z[self.idx0] = -z_ax.copy()
                    eqs.append((C_z, {}, g_o_z))

                if self.idx3 is not None:
                    vr3 = vr[self.idx3] if vr is not None else np.zeros(3)
                    y3, z3 = _compute_transverse_basis(self.a3)
                    w3_rel = vr3 - vr0
                    C_y3 = float(np.dot(w3_rel, y3))
                    g_o_y3 = {self.idx3: y3.copy()}
                    if self.idx0 is not None:
                        g_o_y3[self.idx0] = -y3.copy()
                    eqs.append((C_y3, {}, g_o_y3))

                    C_z3 = float(np.dot(w3_rel, z3))
                    g_o_z3 = {self.idx3: z3.copy()}
                    if self.idx0 is not None:
                        g_o_z3[self.idx0] = -z3.copy()
                    eqs.append((C_z3, {}, g_o_z3))

        elif self.joint_type == 3:  # RACK (gjnt_rack.F)
            y1, z1 = _compute_transverse_basis(self.a1)
            y2, z2 = _compute_transverse_basis(self.a2)
            if vr is not None:
                # Pinion transverse rotation locked
                w1_rel = vr1 - vr0
                C_y1 = float(np.dot(w1_rel, y1))
                g_o_y1 = {self.idx1: y1.copy()}
                if self.idx0 is not None:
                    g_o_y1[self.idx0] = -y1.copy()
                eqs.append((C_y1, {}, g_o_y1))

                C_z1 = float(np.dot(w1_rel, z1))
                g_o_z1 = {self.idx1: z1.copy()}
                if self.idx0 is not None:
                    g_o_z1[self.idx0] = -z1.copy()
                eqs.append((C_z1, {}, g_o_z1))

        return eqs

    def _apply_velocity_fallback(self, v: np.ndarray, vr: Optional[np.ndarray]) -> None:
        """Fallback when all connected nodes have infinite effective inertia."""
        if self.joint_type == 1 and vr is not None:  # GEAR: w2 = w0 - alpha*(w1 - w0)
            vr0 = vr[self.idx0] if self.idx0 is not None else np.zeros(3)
            w1_rel = float(np.dot(vr[self.idx1] - vr0, self.a1))
            w2_rel_old = float(np.dot(vr[self.idx2] - vr0, self.a2))
            w2_rel_new = -self.alpha * w1_rel
            vr[self.idx2] += (w2_rel_new - w2_rel_old) * self.a2
        elif self.joint_type == 2 and vr is not None:  # DIFF: w2 = 2*w0 - w1
            vr0 = vr[self.idx0] if self.idx0 is not None else np.zeros(3)
            w0 = float(np.dot(vr0, self.a0))
            w1 = float(np.dot(vr[self.idx1], self.a1))
            w2_old = float(np.dot(vr[self.idx2], self.a2))
            w2_new = 2.0 * w0 - w1
            vr[self.idx2] += (w2_new - w2_old) * self.a2
        elif self.joint_type == 3:  # RACK: v2 = v0 + alpha*(w1 - w0)
            v0 = v[self.idx0] if self.idx0 is not None else np.zeros(3)
            vr0 = vr[self.idx0] if (vr is not None and self.idx0 is not None) else np.zeros(3)
            w_rel = float(np.dot(vr[self.idx1] - vr0, self.a1)) if vr is not None else 0.0
            v2_rel_old = float(np.dot(v[self.idx2] - v0, self.a2))
            v2_rel_new = self.alpha * w_rel
            v[self.idx2] += (v2_rel_new - v2_rel_old) * self.a2
        elif self.joint_type == 4 and vr is not None:  # CV: w2 = w0 + (w1 - w0)
            vr0 = vr[self.idx0] if self.idx0 is not None else np.zeros(3)
            w1_rel = float(np.dot(vr[self.idx1] - vr0, self.a1))
            w2_rel_old = float(np.dot(vr[self.idx2] - vr0, self.a2))
            w2_rel_new = w1_rel
            vr[self.idx2] += (w2_rel_new - w2_rel_old) * self.a2

    def apply_velocity(self, v: np.ndarray, vr: Optional[np.ndarray], x: np.ndarray,
                       mass: Optional[np.ndarray] = None,
                       inertia: Optional[np.ndarray] = None) -> None:
        """Project velocities onto kinematic constraint manifold, satisfying
        transmission constraint identically and conserving momentum.
        """
        if not self.is_valid:
            return

        C, g_v, g_omega = self.get_constraint_equation(v, vr, x)
        if abs(C) < 1e-14:
            return

        D = 0.0
        for idx, gv in g_v.items():
            m = self._get_node_mass(idx, mass)
            if m < 1e28:
                D += float(np.dot(gv, gv)) / m
        for idx, go in g_omega.items():
            iner = self._get_node_inertia(idx, inertia, mass)
            if iner < 1e28:
                D += float(np.dot(go, go)) / iner

        if D < 1e-20:
            self._apply_velocity_fallback(v, vr)
            return

        lam = -C / D
        for idx, gv in g_v.items():
            m = self._get_node_mass(idx, mass)
            if m < 1e28:
                v[idx] += (lam / m) * gv
        if vr is not None:
            for idx, go in g_omega.items():
                iner = self._get_node_inertia(idx, inertia, mass)
                if iner < 1e28:
                    vr[idx] += (lam / iner) * go

    def _apply_acceleration_fallback(self, a: np.ndarray, ar: Optional[np.ndarray]) -> None:
        """Fallback when all connected nodes have infinite effective inertia."""
        if self.joint_type == 1 and ar is not None:
            ar0 = ar[self.idx0] if self.idx0 is not None else np.zeros(3)
            a1_rel = float(np.dot(ar[self.idx1] - ar0, self.a1))
            a2_rel_old = float(np.dot(ar[self.idx2] - ar0, self.a2))
            a2_rel_new = -self.alpha * a1_rel
            ar[self.idx2] += (a2_rel_new - a2_rel_old) * self.a2
        elif self.joint_type == 2 and ar is not None:
            ar0 = ar[self.idx0] if self.idx0 is not None else np.zeros(3)
            a0 = float(np.dot(ar0, self.a0))
            a1 = float(np.dot(ar[self.idx1], self.a1))
            a2_old = float(np.dot(ar[self.idx2], self.a2))
            a2_new = 2.0 * a0 - a1
            ar[self.idx2] += (a2_new - a2_old) * self.a2
        elif self.joint_type == 3:
            a0 = a[self.idx0] if self.idx0 is not None else np.zeros(3)
            ar0 = ar[self.idx0] if (ar is not None and self.idx0 is not None) else np.zeros(3)
            a_rel = float(np.dot(ar[self.idx1] - ar0, self.a1)) if ar is not None else 0.0
            a2_rel_old = float(np.dot(a[self.idx2] - a0, self.a2))
            a2_rel_new = self.alpha * a_rel
            a[self.idx2] += (a2_rel_new - a2_rel_old) * self.a2
        elif self.joint_type == 4 and ar is not None:
            ar0 = ar[self.idx0] if self.idx0 is not None else np.zeros(3)
            a1_rel = float(np.dot(ar[self.idx1] - ar0, self.a1))
            a2_rel_old = float(np.dot(ar[self.idx2] - ar0, self.a2))
            a2_rel_new = a1_rel
            ar[self.idx2] += (a2_rel_new - a2_rel_old) * self.a2

    def apply_acceleration(self, a: np.ndarray, ar: Optional[np.ndarray], x: np.ndarray,
                           mass: Optional[np.ndarray] = None,
                           inertia: Optional[np.ndarray] = None) -> None:
        """Project accelerations onto kinematic constraint manifold, ensuring
        transmission constraint acceleration is identically zero.
        """
        if not self.is_valid:
            return

        C_a, g_v, g_omega = self.get_constraint_equation(a, ar, x)
        if abs(C_a) < 1e-14:
            return

        D = 0.0
        for idx, gv in g_v.items():
            m = self._get_node_mass(idx, mass)
            if m < 1e28:
                D += float(np.dot(gv, gv)) / m
        for idx, go in g_omega.items():
            iner = self._get_node_inertia(idx, inertia, mass)
            if iner < 1e28:
                D += float(np.dot(go, go)) / iner

        if D < 1e-20:
            self._apply_acceleration_fallback(a, ar)
            return

        lam = -C_a / D
        for idx, gv in g_v.items():
            m = self._get_node_mass(idx, mass)
            if m < 1e28:
                a[idx] += (lam / m) * gv
        if ar is not None:
            for idx, go in g_omega.items():
                iner = self._get_node_inertia(idx, inertia, mass)
                if iner < 1e28:
                    ar[idx] += (lam / iner) * go

    def transfer_forces(self, fint: np.ndarray, fext: np.ndarray, fcont: np.ndarray,
                        mint: np.ndarray, x: np.ndarray,
                        mass: Optional[np.ndarray] = None,
                        inv_mass: Optional[np.ndarray] = None,
                        inv_inertia: Optional[np.ndarray] = None) -> None:
        """Transfer reaction forces and torques between carrier and gear/axle nodes,
        conserving total linear and angular momentum identically.

        Fortran origin: ``engine/source/tools/lagmul/lag_gjnt.F`` lines 100-240.
        For GEAR: T1 = -alpha * T2, T0 = -(T1 + T2).
        For DIFF: T0 = -(T1 + T2 [+ T3]).
        For RACK: F0 = -F2, T1 = -alpha * F2, T0 = -T1 - (x2 - x0) x F2.
        For CV:   T1 = -T2, T0 = -(T1 + T2).
        """
        if not self.is_valid:
            return

        f_tot = fint + fext + fcont
        m_tot = mint

        if self.joint_type == 1:  # GEAR: T1 = -alpha * T2, T0 = -(T1 + T2)
            T2 = float(np.dot(m_tot[self.idx2], self.a2))
            T1 = float(np.dot(m_tot[self.idx1], self.a1))
            if abs(T2) > 1e-14 and abs(T1) < 1e-14:
                T1_reac = -self.alpha * T2
                mint[self.idx1] += T1_reac * self.a1
                if self.idx0 is not None:
                    mint[self.idx0] += -(T1_reac * self.a1 + T2 * self.a2)
            elif abs(T1) > 1e-14 and abs(T2) < 1e-14:
                T2_reac = -(1.0 / self.alpha) * T1
                mint[self.idx2] += T2_reac * self.a2
                if self.idx0 is not None:
                    mint[self.idx0] += -(T1 * self.a1 + T2_reac * self.a2)
            elif self.idx0 is not None:
                mint[self.idx0] -= (T1 * self.a1 + T2 * self.a2)

        elif self.joint_type == 2:  # DIFF: T0 = -(T1 + T2)
            T1 = float(np.dot(m_tot[self.idx1], self.a1))
            T2 = float(np.dot(m_tot[self.idx2], self.a2))
            if self.idx3 is not None:
                T3 = float(np.dot(m_tot[self.idx3], self.a3))
                if self.idx0 is not None:
                    mint[self.idx0] -= (T1 * self.a1 + T2 * self.a2 + T3 * self.a3)
            else:
                if self.idx0 is not None:
                    mint[self.idx0] -= (T1 * self.a1 + T2 * self.a2)

        elif self.joint_type == 3:  # RACK: F2 -> T1 = -alpha*F2, F0 = -F2
            F2 = float(np.dot(f_tot[self.idx2], self.a2))
            if abs(F2) > 1e-14:
                T1 = -self.alpha * F2
                mint[self.idx1] += T1 * self.a1
                if self.idx0 is not None:
                    fint[self.idx0] -= F2 * self.a2
                    r_offset = x[self.idx2] - x[self.idx0]
                    mint[self.idx0] -= (T1 * self.a1 + np.cross(r_offset, F2 * self.a2))

        elif self.joint_type == 4:  # CV: T1 = -T2, T0 = -(T1 + T2)
            T2 = float(np.dot(m_tot[self.idx2], self.a2))
            T1 = float(np.dot(m_tot[self.idx1], self.a1))
            if abs(T2) > 1e-14 and abs(T1) < 1e-14:
                mint[self.idx1] += (-T2) * self.a1
                if self.idx0 is not None:
                    mint[self.idx0] += (T2 * self.a1 - T2 * self.a2)
            elif abs(T1) > 1e-14 and abs(T2) < 1e-14:
                mint[self.idx2] += (-T1) * self.a2
                if self.idx0 is not None:
                    mint[self.idx0] += (T1 * self.a2 - T1 * self.a1)
            elif self.idx0 is not None:
                mint[self.idx0] -= (T1 * self.a1 + T2 * self.a2)

    def enforce(self, x: np.ndarray, v: Optional[np.ndarray] = None,
                vr: Optional[np.ndarray] = None, dt: float = 0.0) -> None:
        """Enforce geometric placement, correct drift, and update carrier frame (lag_gjnt.F)."""
        if not self.is_valid:
            return

        # Update carrier local frame orientation (ROTBMR in lag_gjnt.F)
        if self.has_skew and dt > 0.0 and vr is not None and self.idx0 is not None:
            self.update_carrier_frame(vr[self.idx0], dt)

        if self.joint_type == 3 and self.idx2 is not None:
            x2 = x[self.idx2]
            x_ref = x[self.idx0] if self.idx0 is not None else self._x0_init
            offset = x2 - x_ref
            par_dist = float(np.dot(offset, self.a2))
            x_proj = x_ref + par_dist * self.a2
            x[self.idx2] = x_proj


def build_gjoints(model: Model, log: MessageLog) -> list[GJointEngine]:
    """Instantiate GJointEngine for all /GJOINT entities in the model."""
    gjoints: list[GJointEngine] = []
    if hasattr(model, "gjoints") and model.gjoints:
        for joint in model.gjoints.values():
            try:
                gj = GJointEngine(joint, model, log)
                if gj.is_valid:
                    gjoints.append(gj)
                else:
                    log.warning(f"/GJOINT/{joint.id}: incomplete node definition (N1={joint.node_id1}, N2={joint.node_id2})")
            except Exception as exc:
                log.warning(f"Failed to initialize /GJOINT/{joint.id}: {exc}")
    return gjoints
