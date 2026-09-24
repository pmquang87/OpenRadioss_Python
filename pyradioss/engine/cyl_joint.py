"""
/CYL_JOINT — Cylindrical kinematic joints, engine side (M586).

Fortran origin:
  - ``starter/source/constraints/general/cyl_joint/hm_read_cyljoint.F``
  - ``engine/source/constraints/general/cyl_joint/cjoint.F``
  - ``engine/source/constraints/general/cyl_joint/telesc.F``
  - ``hm_cfg_files/config/CFG/radioss110/RBODY/cyl_jont.cfg``

Kinematic Formulation:
----------------------
A cylindrical joint defines an axis of translation and rotation between
two main nodes NA and NB:
    n = (x_B - x_A) / S,   where S = ||x_B - x_A||

For each secondary node s (from grnod_id or node lists):
  1. Projection along axis:
     xi = (x_s - x_A) . n
     w_B = xi / S,   w_A = 1 - xi / S

  2. Axis velocity interpolation at projection point:
     v_axis = w_A * v_A + w_B * v_B

  3. Preserves parallel axial velocity component:
     v_parallel = (v_s . n) * n

  4. Constrains perpendicular velocity to axis perpendicular velocity:
     v_axis_perp = v_axis - (v_axis . n) * n
     v_s_new = v_parallel + v_axis_perp

  5. Reaction forces / impulses transferred back to main nodes NA and NB:
     Conserves linear momentum:
       Delta p_s = m_s * (v_axis_perp - v_s_perp)
       Delta p_A = -w_A * Delta p_s
       Delta p_B = -w_B * Delta p_s
     and angular momentum:
       x_s x Delta p_s + x_A x Delta p_A + x_B x Delta p_B = 0
     identically.
"""

from __future__ import annotations

import numpy as np

from ..common.messages import MessageLog
from ..model.entities import CylJoint
from ..model.model import Model


class CylJointEngine:
    """Cylindrical joint engine constraint between two main nodes and secondary nodes."""

    def __init__(self, joint: CylJoint, model: Model, log: MessageLog):
        self.joint = joint
        self.id = joint.id
        self.model = model
        self.log = log

        n1 = joint.node_id1 or joint.node1
        n2 = joint.node_id2 or joint.node2
        if n1 not in model._id2idx or n2 not in model._id2idx:
            raise ValueError(f"/CYL_JOINT/{joint.id}: main nodes {n1} and/or {n2} not found in model")

        self.idx_A: int = model.node_index(n1)
        self.idx_B: int = model.node_index(n2)

        # Collect secondary node indices
        sec_indices: list[int] = []
        if joint.grnod_id > 0 and hasattr(model, "node_groups") and joint.grnod_id in model.node_groups:
            grp = model.node_groups[joint.grnod_id]
            if getattr(grp, "node_idx", None) is not None:
                sec_indices.extend(int(idx) for idx in grp.node_idx)
            elif getattr(grp, "node_ids", None):
                for nid in grp.node_ids:
                    if nid in model._id2idx:
                        sec_indices.append(model.node_index(nid))

        if hasattr(joint, "secondary_nodes") and joint.secondary_nodes:
            for nid in joint.secondary_nodes:
                if nid in model._id2idx:
                    sec_indices.append(model.node_index(nid))

        # Filter out main nodes and remove duplicates
        filtered = [idx for idx in sec_indices if idx != self.idx_A and idx != self.idx_B]
        self.sec_idx: np.ndarray = np.unique(np.array(filtered, dtype=np.int64)) if filtered else np.empty(0, dtype=np.int64)
        self.is_valid: bool = len(self.sec_idx) > 0
        self._perp_offsets: dict[int, np.ndarray] = {}

    def get_axis(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Compute axis position, length, and unit direction vector n."""
        xA = x[self.idx_A]
        xB = x[self.idx_B]
        vec = xB - xA
        S = float(np.linalg.norm(vec))
        if S < 1e-14:
            return xA, 0.0, np.array([1.0, 0.0, 0.0], dtype=np.float64)
        n = vec / S
        return xA, S, n

    def transfer_forces(self, fint: np.ndarray, fext: np.ndarray, fcont: np.ndarray,
                        mint: np.ndarray, x: np.ndarray) -> None:
        """Transfer perpendicular reaction forces from secondary nodes to main nodes NA and NB,
        conserving linear and angular momentum identically (cjoint.F, telesc.F).
        """
        if not self.is_valid or len(self.sec_idx) == 0:
            return
        xA, S, n = self.get_axis(x)
        if S < 1e-14:
            return

        for s in self.sec_idx:
            xs = x[s]
            xi = float(np.dot(xs - xA, n))
            wB = xi / S
            wA = 1.0 - wB

            r_perp = (xs - xA) - xi * n

            for arr in (fint, fext, fcont):
                Fs = arr[s].copy()
                F_par = float(np.dot(Fs, n)) * n
                F_perp = Fs - F_par

                arr[self.idx_A] += wA * F_perp
                arr[self.idx_B] += wB * F_perp
                arr[s] = F_par

                if np.dot(r_perp, r_perp) > 1e-20:
                    M_perp = np.cross(r_perp, F_perp)
                    mint[self.idx_A] += wA * M_perp
                    mint[self.idx_B] += wB * M_perp

    def apply_acceleration(self, a: np.ndarray, x: np.ndarray,
                           mass: np.ndarray | None = None) -> None:
        """Constrain secondary node accelerations along perpendicular direction
        while preserving parallel axial acceleration and transferring reactions
        back to main nodes NA and NB (telesc.F).
        """
        if not self.is_valid or len(self.sec_idx) == 0:
            return
        xA, S, n = self.get_axis(x)
        if S < 1e-14:
            return

        aA = a[self.idx_A]
        aB = a[self.idx_B]

        for s in self.sec_idx:
            xs = x[s]
            a_s = a[s]
            xi = float(np.dot(xs - xA, n))
            wB = xi / S
            wA = 1.0 - wB

            a_axis = wA * aA + wB * aB
            a_axis_par = float(np.dot(a_axis, n)) * n
            a_axis_perp = a_axis - a_axis_par

            a_s_par = float(np.dot(a_s, n)) * n
            a_s_perp = a_s - a_s_par

            if mass is not None:
                ms = mass[s]
                mA = mass[self.idx_A]
                mB = mass[self.idx_B]
                if ms > 0.0:
                    inv_m = 1.0 / ms
                    if mA > 0.0:
                        inv_m += (wA * wA) / mA
                    if mB > 0.0:
                        inv_m += (wB * wB) / mB
                    Ja_perp = (a_axis_perp - a_s_perp) / inv_m
                    a[s] += Ja_perp / ms
                    if mA > 0.0:
                        a[self.idx_A] -= wA * Ja_perp / mA
                    if mB > 0.0:
                        a[self.idx_B] -= wB * Ja_perp / mB
                else:
                    a[s] = a_s_par + a_axis_perp
            else:
                a[s] = a_s_par + a_axis_perp

    def apply_velocity(self, v: np.ndarray, x: np.ndarray,
                       mass: np.ndarray | None = None) -> None:
        """Constrain secondary node velocities along perpendicular direction
        while preserving parallel axial velocity and transferring reactions
        back to main nodes NA and NB (telesc.F).
        """
        if not self.is_valid or len(self.sec_idx) == 0:
            return
        xA, S, n = self.get_axis(x)
        if S < 1e-14:
            return

        vA = v[self.idx_A]
        vB = v[self.idx_B]

        for s in self.sec_idx:
            xs = x[s]
            v_s = v[s]
            xi = float(np.dot(xs - xA, n))
            wB = xi / S
            wA = 1.0 - wB

            v_axis = wA * vA + wB * vB
            v_axis_par = float(np.dot(v_axis, n)) * n
            v_axis_perp = v_axis - v_axis_par

            v_s_par = float(np.dot(v_s, n)) * n
            v_s_perp = v_s - v_s_par

            if mass is not None:
                ms = mass[s]
                mA = mass[self.idx_A]
                mB = mass[self.idx_B]
                if ms > 0.0:
                    inv_m = 1.0 / ms
                    if mA > 0.0:
                        inv_m += (wA * wA) / mA
                    if mB > 0.0:
                        inv_m += (wB * wB) / mB
                    J_perp = (v_axis_perp - v_s_perp) / inv_m
                    v[s] += J_perp / ms
                    if mA > 0.0:
                        v[self.idx_A] -= wA * J_perp / mA
                    if mB > 0.0:
                        v[self.idx_B] -= wB * J_perp / mB
                else:
                    v[s] = v_s_par + v_axis_perp
            else:
                v[s] = v_s_par + v_axis_perp

    def enforce(self, x: np.ndarray, v: np.ndarray | None = None, dt: float = 0.0) -> None:
        """Enforce geometric placement of secondary nodes on the joint axis,
        eliminating perpendicular numerical drift (telesc.F).
        """
        if not self.is_valid or len(self.sec_idx) == 0:
            return
        xA, S, n = self.get_axis(x)
        if S < 1e-14:
            return

        for s in self.sec_idx:
            xs = x[s]
            xi = float(np.dot(xs - xA, n))
            x_proj = xA + xi * n
            if s not in self._perp_offsets:
                self._perp_offsets[s] = xs - x_proj
            x[s] = x_proj + self._perp_offsets[s]


def build_cyl_joints(model: Model, log: MessageLog) -> list[CylJointEngine]:
    """Instantiate CylJointEngine for all /CYL_JOINT entities in the model."""
    cyl_joints: list[CylJointEngine] = []
    for jid, joint in getattr(model, "cyl_joints", {}).items():
        try:
            cj = CylJointEngine(joint, model, log)
            if cj.is_valid:
                cyl_joints.append(cj)
                log.info(f" /CYL_JOINT/{jid}: {len(cj.sec_idx)} secondary node(s) on axis {joint.node_id1 or joint.node1}-{joint.node_id2 or joint.node2}")
        except Exception as exc:
            log.warning(f"/CYL_JOINT/{jid}: initialization skipped: {exc}", "CYL_JOINT")
    return cyl_joints
