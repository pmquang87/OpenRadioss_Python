"""
Kinematic Mechanism Joints (/PROP/TYPE33 and /PROP/TYPE45) — M602.

Fortran origin:
  - ``starter/source/properties/spring/hm_read_prop33.F``
  - ``starter/source/properties/spring/hm_read_prop45.F``
  - ``starter/source/elements/joint/rjoint/rini33.F``
  - ``starter/source/elements/joint/rjoint/rini45.F``
  - ``engine/source/elements/joint/rgjoint.F``
  - ``engine/source/elements/joint/ruser33.F``
  - ``engine/source/elements/joint/rcum33.F``
  - ``engine/source/elements/joint/rskew33.F``
  - ``engine/source/elements/spring/rforc3.F`` (IGTYP == 33 .OR. IGTYP == 45)

Kinematic Formulations and Joint Types:
----------------------------------------
1. Revolute (hinge): 1 rotational DOF free (Rx about local axis 1),
   3 translations (Tx, Ty, Tz) and 2 rotations (Ry, Rz) locked.
2. Spherical (ball-and-socket): 3 rotations free (Rx, Ry, Rz),
   3 translations locked (Tx, Ty, Tz).
3. Cylindrical: 1 translation (Tx) and 1 coaxial rotation (Rx) free,
   2 translations (Ty, Tz) and 2 rotations (Ry, Rz) locked.
4. Planar: 2 in-plane translations (Ty, Tz) and 1 normal rotation (Rx) free,
   1 translation (Tx) and 2 rotations (Ry, Rz) locked.
5. Universal / Cardan: 2 rotation DOFs free (Ry, Rz),
   3 translations (Tx, Ty, Tz) and 1 rotation (Rx) locked.
6. Slider / Prismatic: 1 translation free (Tx),
   2 translations (Ty, Tz) and 3 rotations (Rx, Ry, Rz) locked.
7. Oldham: 2 in-plane translations (Ty, Tz) free,
   1 translation (Tx) and 3 rotations (Rx, Ry, Rz) locked.
8. Fixed (rigid): all 6 DOFs locked.
9. Free: all 6 DOFs free.

Enforcement:
  - Linear penalty and Lagrange multiplier enforcement of locked DOFs.
  - Elastic stiffness and damping on free and locked DOFs.
  - Momentum-conserving reaction force and torque distribution:
    sum(F) = 0, sum(M) = 0 to machine precision (< 1e-12).
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..common.constants import EM20, EP30
from ..common.messages import MessageLog
from ..model.entities import KJoint, PropType33, PropType45, Property
from ..model.model import Model


class JointType(IntEnum):
    """Kinematic joint types supported in M602."""
    REVOLUTE = 1      # Hinge: 1 rot free (Rx), 3 trans + 2 rot locked
    SPHERICAL = 2     # Ball-and-socket: 3 rot free, 3 trans locked
    CYLINDRICAL = 3   # 1 trans (Tx) + 1 rot (Rx) free, 2 trans + 2 rot locked
    PLANAR = 4        # 2 trans (Ty, Tz) + 1 rot (Rx) free, 1 trans + 2 rot locked
    UNIVERSAL = 5     # 2 rot free (Ry, Rz), 3 trans + 1 rot (Rx) locked
    SLIDER = 6        # Prismatic slider: 1 trans (Tx) free, 2 trans + 3 rot locked
    PRISMATIC = 6     # Alias for SLIDER
    OLDHAM = 7        # 2 trans (Ty, Tz) free, 1 trans + 3 rot locked
    FIXED = 8         # 6 DOFs locked
    FREE = 9          # 6 DOFs free


def resolve_joint_type(val: int | str | JointType, convention: str = "standard") -> JointType:
    """Resolve joint type from integer code, enum, or name."""
    if isinstance(val, JointType):
        return val
    if isinstance(val, str):
        v = val.strip().upper()
        if "REV" in v or "HINGE" in v:
            return JointType.REVOLUTE
        if "SPH" in v or "BALL" in v:
            return JointType.SPHERICAL
        if "CYL" in v:
            return JointType.CYLINDRICAL
        if "PLAN" in v:
            return JointType.PLANAR
        if "UNIV" in v or "CARDAN" in v:
            return JointType.UNIVERSAL
        if "SLID" in v or "PRISM" in v or "TRANS" in v:
            return JointType.SLIDER
        if "OLD" in v:
            return JointType.OLDHAM
        if "FIX" in v or "RIGID" in v:
            return JointType.FIXED
        if "FREE" in v:
            return JointType.FREE
        try:
            val = int(v)
        except ValueError:
            return JointType.REVOLUTE

    val_int = int(val)
    if convention == "radioss":
        # Pure Radioss /PROP/TYPE33 CFG convention: 1=Spherical, 2=Revolute
        radioss_map = {
            1: JointType.SPHERICAL,
            2: JointType.REVOLUTE,
            3: JointType.CYLINDRICAL,
            4: JointType.PLANAR,
            5: JointType.UNIVERSAL,
            6: JointType.SLIDER,
            7: JointType.OLDHAM,
            8: JointType.FIXED,
            9: JointType.FREE,
        }
        return radioss_map.get(val_int, JointType.SPHERICAL)
    else:
        # Standard M602 convention: 1=Revolute, 2=Spherical, 3=Cylindrical, 4=Planar, 5=Universal, 6=Slider
        std_map = {
            1: JointType.REVOLUTE,
            2: JointType.SPHERICAL,
            3: JointType.CYLINDRICAL,
            4: JointType.PLANAR,
            5: JointType.UNIVERSAL,
            6: JointType.SLIDER,
            7: JointType.OLDHAM,
            8: JointType.FIXED,
            9: JointType.FREE,
        }
        return std_map.get(val_int, JointType.REVOLUTE)


class KJointEngine:
    """Kinematic mechanism joint solver engine instance for a single joint."""

    def __init__(
        self,
        joint: Any,
        model: Model,
        log: Optional[MessageLog] = None,
        convention: str = "standard",
        idx1: Optional[int] = None,
        idx2: Optional[int] = None,
    ):
        self.joint = joint
        self.id = getattr(joint, "id", 1)
        self.model = model
        self.log = log

        # Node index resolution
        if idx1 is not None and idx2 is not None:
            self.idx1 = int(idx1)
            self.idx2 = int(idx2)
        else:
            n1 = getattr(joint, "node1", 0) or getattr(joint, "node_id1", 0)
            n2 = getattr(joint, "node2", 0) or getattr(joint, "node_id2", 0)
            self.idx1 = model.node_index(n1) if hasattr(model, "_id2idx") and n1 in model._id2idx else -1
            self.idx2 = model.node_index(n2) if hasattr(model, "_id2idx") and n2 in model._id2idx else -1

        self.is_valid: bool = (self.idx1 >= 0 and self.idx2 >= 0 and self.idx1 != self.idx2)
        if not self.is_valid:
            return

        # Joint type
        raw_type = getattr(joint, "joint_type", None)
        if raw_type is None:
            raw_type = getattr(joint, "type", 1)
        self.joint_type = resolve_joint_type(raw_type, convention=convention)

        # Skew frame ID
        self.skew_id: int = int(getattr(joint, "skew_id", 0) or getattr(joint, "id_sk1", 0) or getattr(joint, "skew1", 0) or 0)

        # Penalty stiffness and damping
        self.kn: float = float(getattr(joint, "kn", 0.0) or getattr(joint, "xk", 0.0) or 0.0)
        self.cr: float = float(getattr(joint, "cr", 0.0) or 0.0)
        self.scale: float = float(getattr(joint, "scale", 1.0) or 1.0)

        # Elastic stiffness on DOFs [Tx, Ty, Tz, Rx, Ry, Rz]
        self.free_stiff = np.array([
            float(getattr(joint, "ktx", 0.0) or 0.0),
            float(getattr(joint, "kty", 0.0) or 0.0),
            float(getattr(joint, "ktz", 0.0) or 0.0),
            float(getattr(joint, "krx", 0.0) or 0.0),
            float(getattr(joint, "kry", 0.0) or 0.0),
            float(getattr(joint, "krz", 0.0) or 0.0),
        ], dtype=np.float64)

        # Damping coefficients on DOFs [Tx, Ty, Tz, Rx, Ry, Rz]
        self.free_damp = np.array([
            float(getattr(joint, "ctx", 0.0) or 0.0),
            float(getattr(joint, "cty", 0.0) or 0.0),
            float(getattr(joint, "ctz", 0.0) or 0.0),
            float(getattr(joint, "crx", 0.0) or 0.0),
            float(getattr(joint, "cry", 0.0) or 0.0),
            float(getattr(joint, "crz", 0.0) or 0.0),
        ], dtype=np.float64)

        # Determine locked vs free DOFs
        self._setup_dof_masks()

        # Local coordinate triad (e1, e2, e3)
        self.e1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.e2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self.e3 = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # Prefer current coords model.x, fallback to initial coords model.x0
        x_init = getattr(model, "x", None)
        if x_init is None or len(x_init) <= max(self.idx1, self.idx2):
            x_init = getattr(model, "x0", None)

        if x_init is not None and len(x_init) > max(self.idx1, self.idx2):
            self._init_frame(x_init)
            x1_0 = x_init[self.idx1]
            x2_0 = x_init[self.idx2]
            r0_glob = x2_0 - x1_0
            self.r0 = np.array([
                float(np.dot(r0_glob, self.e1)),
                float(np.dot(r0_glob, self.e2)),
                float(np.dot(r0_glob, self.e3)),
            ], dtype=np.float64)
            self.len0 = float(np.linalg.norm(r0_glob))
        else:
            self.r0 = np.zeros(3, dtype=np.float64)
            self.len0 = 0.0

        # Internal state tracking
        self.theta = np.zeros(3, dtype=np.float64)
        self.theta_prev = np.zeros(3, dtype=np.float64)
        self.f_prev = np.zeros(3, dtype=np.float64)
        self.m_prev = np.zeros(3, dtype=np.float64)
        self.eint: float = 0.0

    @classmethod
    def from_prop(
        cls,
        eid: int,
        idx1: int,
        idx2: int,
        prop: Any,
        model: Model,
        log: Optional[MessageLog] = None,
    ) -> KJointEngine:
        """Construct KJointEngine from a /PROP/TYPE33 or /PROP/TYPE45 property."""
        p = getattr(prop, "params", {}) or {}
        convention = "radioss" if getattr(prop, "type", 0) in (33, 45) else "standard"
        joint = KJoint(
            id=eid,
            node1=model.node_ids[idx1] if hasattr(model, "node_ids") and idx1 < len(model.node_ids) else idx1 + 1,
            node2=model.node_ids[idx2] if hasattr(model, "node_ids") and idx2 < len(model.node_ids) else idx2 + 1,
            prop_id=getattr(prop, "id", 0),
            joint_type=p.get("joint_type", p.get("type", 1)),
            title=getattr(prop, "title", ""),
            skew_id=int(p.get("id_sk1", p.get("skew1", 0)) or 0),
            kn=float(p.get("kn", p.get("xk", 0.0)) or 0.0),
            cr=float(p.get("cr", 0.0) or 0.0),
            scale=float(p.get("scale", 1.0) or 1.0),
            ktx=float(p.get("ktx", 0.0) or 0.0),
            kty=float(p.get("kty", 0.0) or 0.0),
            ktz=float(p.get("ktz", 0.0) or 0.0),
            krx=float(p.get("krx", 0.0) or 0.0),
            kry=float(p.get("kry", 0.0) or 0.0),
            krz=float(p.get("krz", 0.0) or 0.0),
            ctx=float(p.get("ctx", 0.0) or 0.0),
            cty=float(p.get("cty", 0.0) or 0.0),
            ctz=float(p.get("ctz", 0.0) or 0.0),
            crx=float(p.get("crx", 0.0) or 0.0),
            cry=float(p.get("cry", 0.0) or 0.0),
            crz=float(p.get("crz", 0.0) or 0.0),
            prop=prop,
        )
        return cls(joint, model, log, convention=convention, idx1=idx1, idx2=idx2)

    def _setup_dof_masks(self) -> None:
        """Assign free and locked DOFs (0..2 translation, 3..5 rotation)."""
        jt = self.joint_type
        if jt == JointType.REVOLUTE:
            # 1 rotational DOF free (Rx), 3 translations and 2 rotations locked
            self.free_dofs = {3}
            self.locked_dofs = {0, 1, 2, 4, 5}
        elif jt == JointType.SPHERICAL:
            # 3 rotations free, 3 translations locked
            self.free_dofs = {3, 4, 5}
            self.locked_dofs = {0, 1, 2}
        elif jt == JointType.CYLINDRICAL:
            # 1 translation (Tx) and 1 coaxial rotation (Rx) free, 2 trans + 2 rot locked
            self.free_dofs = {0, 3}
            self.locked_dofs = {1, 2, 4, 5}
        elif jt == JointType.PLANAR:
            # 2 in-plane translations (Ty, Tz) and 1 normal rotation (Rx) free
            self.free_dofs = {1, 2, 3}
            self.locked_dofs = {0, 4, 5}
        elif jt == JointType.UNIVERSAL:
            # 2 rotation DOFs free (Ry, Rz), 3 translations and 1 rotation locked
            self.free_dofs = {4, 5}
            self.locked_dofs = {0, 1, 2, 3}
        elif jt in (JointType.SLIDER, JointType.PRISMATIC):
            # 1 translation free (Tx), 2 translations and 3 rotations locked
            self.free_dofs = {0}
            self.locked_dofs = {1, 2, 3, 4, 5}
        elif jt == JointType.OLDHAM:
            self.free_dofs = {1, 2}
            self.locked_dofs = {0, 3, 4, 5}
        elif jt == JointType.FIXED:
            self.free_dofs = set()
            self.locked_dofs = {0, 1, 2, 3, 4, 5}
        elif jt == JointType.FREE:
            self.free_dofs = {0, 1, 2, 3, 4, 5}
            self.locked_dofs = set()
        else:
            self.free_dofs = {3}
            self.locked_dofs = {0, 1, 2, 4, 5}

    def _init_frame(self, x: np.ndarray) -> None:
        """Initialize orthonormal triad e1, e2, e3 from skew or initial geometry."""
        # 1. Skew lookup
        if self.skew_id > 0 and hasattr(self.model, "skews") and self.model.skews is not None:
            if hasattr(self.model.skews, "axes") and 0 <= self.skew_id < len(self.model.skews.axes):
                sk_axes = self.model.skews.axes[self.skew_id]
                self.e1 = np.array(sk_axes[0], dtype=np.float64)
                self.e2 = np.array(sk_axes[1], dtype=np.float64)
                self.e3 = np.array(sk_axes[2], dtype=np.float64)
                return

        # 2. Geometry-based frame
        if x is not None and len(x) > max(self.idx1, self.idx2):
            dx = x[self.idx2] - x[self.idx1]
            dist = float(np.linalg.norm(dx))
            if dist > 1e-12:
                self.e1 = dx / dist
                # Select perpendicular axis
                aux = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                if abs(self.e1[2]) > 0.9:
                    aux = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                e2_proj = aux - np.dot(aux, self.e1) * self.e1
                norm_e2 = float(np.linalg.norm(e2_proj))
                self.e2 = e2_proj / norm_e2 if norm_e2 > 1e-12 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
                self.e3 = np.cross(self.e1, self.e2)
                return
        self.e1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.e2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self.e3 = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def update_frame(self, x: np.ndarray, vr: Optional[np.ndarray] = None, dt: float = 0.0) -> None:
        """Update local frame tracking rigid rotation."""
        if self.skew_id > 0 and hasattr(self.model, "skews") and self.model.skews is not None:
            if hasattr(self.model.skews, "axes") and 0 <= self.skew_id < len(self.model.skews.axes):
                sk_axes = self.model.skews.axes[self.skew_id]
                self.e1 = np.array(sk_axes[0], dtype=np.float64)
                self.e2 = np.array(sk_axes[1], dtype=np.float64)
                self.e3 = np.array(sk_axes[2], dtype=np.float64)
                return

        # Mean rotation of the two nodes (rskew33.F)
        if vr is not None and dt > 0.0:
            w_mean = 0.5 * (vr[self.idx1] + vr[self.idx2])
            dth = w_mean * dt
            angle = float(np.linalg.norm(dth))
            if angle > 1e-14:
                # Rodrigues update for e1, e2, e3
                axis = dth / angle
                c = np.cos(angle)
                s = np.sin(angle)
                for vec in (self.e1, self.e2, self.e3):
                    v_rot = vec * c + np.cross(axis, vec) * s + axis * np.dot(axis, vec) * (1.0 - c)
                    vec[:] = v_rot

            # Re-orthogonalize via Gram-Schmidt
            self.e1 /= np.linalg.norm(self.e1)
            self.e2 -= np.dot(self.e2, self.e1) * self.e1
            self.e2 /= np.linalg.norm(self.e2)
            self.e3 = np.cross(self.e1, self.e2)
            self.e3 /= np.linalg.norm(self.e3)

    def _get_masses(self, mass: Optional[np.ndarray]) -> tuple[float, float]:
        """Get effective masses of nodes 1 and 2."""
        m1 = float(mass[self.idx1]) if mass is not None and self.idx1 < len(mass) else 1.0
        m2 = float(mass[self.idx2]) if mass is not None and self.idx2 < len(mass) else 1.0
        m1 = 1e30 if m1 >= 1e29 else max(m1, 1e-12)
        m2 = 1e30 if m2 >= 1e29 else max(m2, 1e-12)
        return m1, m2

    def _get_inertias(self, inertia: Optional[np.ndarray], mass: Optional[np.ndarray]) -> tuple[float, float]:
        """Get effective rotational inertias of nodes 1 and 2."""
        m1, m2 = self._get_masses(mass)
        i1 = float(inertia[self.idx1]) if inertia is not None and self.idx1 < len(inertia) else 0.0
        i2 = float(inertia[self.idx2]) if inertia is not None and self.idx2 < len(inertia) else 0.0
        if i1 <= 0.0 or i1 >= 1e29:
            i1 = m1 if m1 < 1e28 else 1.0
        if i2 <= 0.0 or i2 >= 1e29:
            i2 = m2 if m2 < 1e28 else 1.0
        return max(i1, 1e-12), max(i2, 1e-12)

    def compute_internal_forces(
        self,
        x: np.ndarray,
        v: np.ndarray,
        vr: Optional[np.ndarray] = None,
        dt: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate linear penalty forces and elastic stiffness/damping on nodes.

        Returns:
            F1, F2: (3,) global force vectors on node 1 and node 2.
            M1, M2: (3,) global moment vectors on node 1 and node 2.
            Guarantees:
              F1 + F2 = 0
              M1 + M2 + x1 x F1 + x2 x F2 = 0 (momentum conserving).
        """
        if not self.is_valid or x is None or len(x) <= max(self.idx1, self.idx2):
            return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

        self.update_frame(x, vr, dt)

        # Current relative displacement vector in global frame
        r_glob = x[self.idx2] - x[self.idx1]
        v_rel = v[self.idx2] - v[self.idx1]
        w_rel = (vr[self.idx2] - vr[self.idx1]) if vr is not None else np.zeros(3)

        # Transform to local frame
        axes = [self.e1, self.e2, self.e3]
        d_loc = np.array([float(np.dot(r_glob, ax)) - self.r0[i] for i, ax in enumerate(axes)], dtype=np.float64)
        v_loc = np.array([float(np.dot(v_rel, ax)) for ax in axes], dtype=np.float64)
        w_loc = np.array([float(np.dot(w_rel, ax)) for ax in axes], dtype=np.float64)

        # Advance accumulated local rotation
        if dt > 0.0:
            self.theta += w_loc * dt

        # Penalty parameters (rini33.F, ruser33.F)
        m1, m2 = self._get_masses(getattr(self.model, "mass", None))
        m_eff = (m1 * m2) / (m1 + m2) if (m1 + m2) > 0 and max(m1, m2) < 1e28 else min(m1, m2)
        i1, i2 = self._get_inertias(getattr(self.model, "inertia", None), getattr(self.model, "mass", None))
        i_eff = (i1 * i2) / (i1 + i2) if (i1 + i2) > 0 and max(i1, i2) < 1e28 else min(i1, i2)

        k_pen = float(self.kn if self.kn > 0.0 else 1e7 * self.scale)
        # OpenRadioss rini33.F: KR = KNN * MAX(ONE, LEN2)
        k_pen_rot = k_pen * max(1.0, self.len0 ** 2)

        xi = max(self.cr, 0.1)
        c_pen = 2.0 * xi * np.sqrt(k_pen * m_eff)
        c_pen_rot = 2.0 * xi * np.sqrt(k_pen_rot * i_eff)

        # Compute local forces and torques
        F_loc = np.zeros(3, dtype=np.float64)
        M_loc = np.zeros(3, dtype=np.float64)

        for i in range(3):
            # Translations: DOF 0, 1, 2
            if i in self.locked_dofs:
                F_loc[i] = k_pen * d_loc[i] + c_pen * v_loc[i]
            elif i in self.free_dofs:
                F_loc[i] = self.free_stiff[i] * d_loc[i] + self.free_damp[i] * v_loc[i]

            # Rotations: DOF 3, 4, 5
            dof_r = i + 3
            if dof_r in self.locked_dofs:
                M_loc[i] = k_pen_rot * self.theta[i] + c_pen_rot * w_loc[i]
            elif dof_r in self.free_dofs:
                M_loc[i] = self.free_stiff[dof_r] * self.theta[i] + self.free_damp[dof_r] * w_loc[i]

        # Convert to global vectors
        F_glob = F_loc[0] * self.e1 + F_loc[1] * self.e2 + F_loc[2] * self.e3
        M_joint = M_loc[0] * self.e1 + M_loc[1] * self.e2 + M_loc[2] * self.e3

        # Moment arm compensation (rcum33.F lines 104-124)
        # M_arm = 0.5 * (x2 - x1) x F_glob
        M_arm = 0.5 * np.cross(r_glob, F_glob)

        F1 = F_glob.copy()
        F2 = -F_glob.copy()
        M1 = M_joint + M_arm
        M2 = -M_joint + M_arm

        # Internal energy increment (ruser33.F line 324)
        if dt > 0.0:
            d_disp = d_loc - getattr(self, "_d_prev", d_loc)
            d_rot = self.theta - self.theta_prev
            self.eint += 0.5 * float(np.dot(F_loc + self.f_prev, d_disp) + np.dot(M_loc + self.m_prev, d_rot))

        self.f_prev = F_loc.copy()
        self.m_prev = M_loc.copy()
        self.theta_prev = self.theta.copy()
        self._d_prev = d_loc.copy()

        return F1, F2, M1, M2

    def transfer_forces(
        self,
        fint: np.ndarray,
        fext: np.ndarray,
        fcont: np.ndarray,
        mint: np.ndarray,
        x: np.ndarray,
        mass: Optional[np.ndarray] = None,
        inv_mass: Optional[np.ndarray] = None,
        inv_inertia: Optional[np.ndarray] = None,
    ) -> None:
        """Accumulate joint internal forces and moments into global arrays."""
        if not self.is_valid or x is None or len(x) <= max(self.idx1, self.idx2):
            return
        v = getattr(self.model, "v", np.zeros_like(x))
        vr = getattr(self.model, "vr", None)
        dt = getattr(self.model, "dt", 0.0)
        F1, F2, M1, M2 = self.compute_internal_forces(x, v, vr, dt)

        fint[self.idx1] += F1
        fint[self.idx2] += F2
        mint[self.idx1] += M1
        mint[self.idx2] += M2

    def apply_velocity(
        self,
        v: np.ndarray,
        vr: Optional[np.ndarray],
        x: np.ndarray,
        mass: Optional[np.ndarray] = None,
        inertia: Optional[np.ndarray] = None,
    ) -> None:
        """Lagrange multiplier velocity projection for locked DOFs.

        Enforces:
          Relative velocity along locked translation axes = 0
          Relative angular velocity around locked rotation axes = 0
        Conserves linear and angular momentum identically (< 1e-12).
        """
        if not self.is_valid or v is None or len(v) <= max(self.idx1, self.idx2) or x is None or len(x) <= max(self.idx1, self.idx2):
            return

        m1, m2 = self._get_masses(mass)
        i1, i2 = self._get_inertias(inertia, mass)
        axes = [self.e1, self.e2, self.e3]
        r_glob = x[self.idx2] - x[self.idx1]

        # 1. Locked Translations
        w_m = (1.0 / m1) + (1.0 / m2)
        if w_m > 1e-30:
            for i in range(3):
                if i in self.locked_dofs:
                    ax = axes[i]
                    c_v = float(np.dot(v[self.idx2] - v[self.idx1], ax))
                    if abs(c_v) > 1e-14:
                        lam = c_v / w_m
                        # Impulse along ax
                        dp = lam * ax
                        if m1 < 1e28:
                            v[self.idx1] += dp / m1
                        if m2 < 1e28:
                            v[self.idx2] -= dp / m2

                        # Moment arm conservation for non-coincident nodes
                        if vr is not None and (i1 + i2) > 0 and np.linalg.norm(r_glob) > 1e-12:
                            dL = np.cross(r_glob, dp)
                            d_omega = dL / (i1 + i2)
                            if i1 < 1e28:
                                vr[self.idx1] += d_omega
                            if i2 < 1e28:
                                vr[self.idx2] += d_omega

        # 2. Locked Rotations
        if vr is not None:
            w_i = (1.0 / i1) + (1.0 / i2)
            if w_i > 1e-30:
                for i in range(3):
                    dof_r = i + 3
                    if dof_r in self.locked_dofs:
                        ax = axes[i]
                        c_w = float(np.dot(vr[self.idx2] - vr[self.idx1], ax))
                        if abs(c_w) > 1e-14:
                            lam_w = c_w / w_i
                            d_ang = lam_w * ax
                            if i1 < 1e28:
                                vr[self.idx1] += d_ang / i1
                            if i2 < 1e28:
                                vr[self.idx2] -= d_ang / i2

    def apply_acceleration(
        self,
        a: np.ndarray,
        ar: Optional[np.ndarray],
        x: np.ndarray,
        mass: Optional[np.ndarray] = None,
        inertia: Optional[np.ndarray] = None,
    ) -> None:
        """Lagrange multiplier acceleration projection for locked DOFs.

        Enforces zero relative acceleration along locked DOFs while conserving momentum.
        """
        if not self.is_valid or a is None or len(a) <= max(self.idx1, self.idx2) or x is None or len(x) <= max(self.idx1, self.idx2):
            return

        m1, m2 = self._get_masses(mass)
        i1, i2 = self._get_inertias(inertia, mass)
        axes = [self.e1, self.e2, self.e3]
        r_glob = x[self.idx2] - x[self.idx1]

        # 1. Locked Translations
        w_m = (1.0 / m1) + (1.0 / m2)
        if w_m > 1e-30:
            for i in range(3):
                if i in self.locked_dofs:
                    ax = axes[i]
                    c_a = float(np.dot(a[self.idx2] - a[self.idx1], ax))
                    if abs(c_a) > 1e-14:
                        lam = c_a / w_m
                        dp = lam * ax
                        if m1 < 1e28:
                            a[self.idx1] += dp / m1
                        if m2 < 1e28:
                            a[self.idx2] -= dp / m2

                        if ar is not None and (i1 + i2) > 0 and np.linalg.norm(r_glob) > 1e-12:
                            dL = np.cross(r_glob, dp)
                            d_ar = dL / (i1 + i2)
                            if i1 < 1e28:
                                ar[self.idx1] += d_ar
                            if i2 < 1e28:
                                ar[self.idx2] += d_ar

        # 2. Locked Rotations
        if ar is not None:
            w_i = (1.0 / i1) + (1.0 / i2)
            if w_i > 1e-30:
                for i in range(3):
                    dof_r = i + 3
                    if dof_r in self.locked_dofs:
                        ax = axes[i]
                        c_ar = float(np.dot(ar[self.idx2] - ar[self.idx1], ax))
                        if abs(c_ar) > 1e-14:
                            lam_w = c_ar / w_i
                            d_ang = lam_w * ax
                            if i1 < 1e28:
                                ar[self.idx1] += d_ang / i1
                            if i2 < 1e28:
                                ar[self.idx2] -= d_ang / i2

    def enforce(
        self,
        x: np.ndarray,
        v: np.ndarray,
        vr: Optional[np.ndarray] = None,
        dt: float = 0.0,
    ) -> None:
        """Kinematic position correction on locked DOFs, preserving center of mass."""
        if not self.is_valid or x is None or len(x) <= max(self.idx1, self.idx2):
            return

        m1, m2 = self._get_masses(getattr(self.model, "mass", None))
        w_tot = m1 + m2
        if w_tot <= 0.0:
            return

        axes = [self.e1, self.e2, self.e3]
        r_glob = x[self.idx2] - x[self.idx1]

        # Correct relative position along locked translation axes
        for i in range(3):
            if i in self.locked_dofs:
                ax = axes[i]
                cur_dist = float(np.dot(r_glob, ax))
                c_x = cur_dist - self.r0[i]
                if abs(c_x) > 1e-14:
                    corr = c_x * ax
                    if m1 < 1e28 and m2 < 1e28:
                        x[self.idx1] += (m2 / w_tot) * corr
                        x[self.idx2] -= (m1 / w_tot) * corr
                    elif m1 < 1e28:
                        x[self.idx1] += corr
                    elif m2 < 1e28:
                        x[self.idx2] -= corr

        # Reset accumulated rotation drift on locked rotation axes
        for i in range(3):
            dof_r = i + 3
            if dof_r in self.locked_dofs:
                self.theta[i] = 0.0


def build_kjoints(model: Model, log: Optional[MessageLog] = None) -> List[KJointEngine]:
    """Scan model for kinematic mechanism joints (/PROP/TYPE33 and /PROP/TYPE45)."""
    kjoints: List[KJointEngine] = []

    # 1. From model.kjoints dict or list
    if hasattr(model, "kjoints") and model.kjoints:
        items = model.kjoints.values() if isinstance(model.kjoints, dict) else model.kjoints
        for j in items:
            kj_eng = KJointEngine(j, model, log)
            if kj_eng.is_valid:
                kjoints.append(kj_eng)

    # 2. From spring element groups referencing /PROP/TYPE33 or /PROP/TYPE45
    for name, group in model.element_groups():
        if "spring" not in name.lower():
            continue
        slices = group.state.get("slices", [])
        for sl, mat, prop in slices:
            pt = getattr(prop, "type", 0)
            if pt in (33, 45) or isinstance(prop, (PropType33, PropType45)):
                conn = group.conn[sl]
                eids = group.ids[sl] if hasattr(group, "ids") else range(len(conn))
                for i in range(len(conn)):
                    idx1 = int(conn[i, 0])
                    idx2 = int(conn[i, 1])
                    eid = int(eids[i])
                    kj_eng = KJointEngine.from_prop(eid, idx1, idx2, prop, model, log)
                    if kj_eng.is_valid:
                        kjoints.append(kj_eng)

    # 3. From model.raw_elems["SPRING"] if not already built from element groups
    if not kjoints and hasattr(model, "raw_elems") and "SPRING" in model.raw_elems:
        for item in model.raw_elems["SPRING"]:
            eid, pid, nodes = item[0], item[1], item[2]
            part = model.parts.get(pid) if hasattr(model, "parts") else None
            prop = None
            if part is not None and hasattr(model, "properties"):
                prop = model.properties.get(part.prop_id)
            if prop is None and hasattr(model, "properties"):
                prop = model.properties.get(pid)
            if prop is not None:
                pt = getattr(prop, "type", 0)
                if pt in (33, 45) or isinstance(prop, (PropType33, PropType45)):
                    n1, n2 = int(nodes[0]), int(nodes[1])
                    idx1 = model.node_index(n1) if hasattr(model, "_id2idx") and n1 in model._id2idx else -1
                    idx2 = model.node_index(n2) if hasattr(model, "_id2idx") and n2 in model._id2idx else -1
                    if idx1 >= 0 and idx2 >= 0:
                        kj_eng = KJointEngine.from_prop(eid, idx1, idx2, prop, model, log)
                        if kj_eng.is_valid:
                            kjoints.append(kj_eng)

    return kjoints
