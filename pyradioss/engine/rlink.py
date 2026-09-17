"""
/RLINK — Standard rigid link constraints, engine side (M586).

Fortran origin:
  - ``starter/source/constraints/rigidlink/hm_read_rlink.F``
  - ``engine/source/constraints/general/rlink/rlink1.F``
  - ``engine/source/constraints/general/rlink/rlink10.F``
  - ``hm_cfg_files/config/CFG/radioss110/RBODY/rlink.cfg``

Kinematic Formulation:
----------------------
A rigid link ties a group of nodes rigidly along specified translational (Tx, Ty, Tz)
and rotational (Rx, Ry, Rz) degrees of freedom:
  1. Center-of-mass translational acceleration and velocity:
     a_cm = sum(m_i * a_i) / sum(m_i)
     v_cm = sum(m_i * v_i) / sum(m_i)

  2. Center-of-mass rotational acceleration and velocity:
     ar_cm = sum(I_i * ar_i) / sum(I_i)
     vr_cm = sum(I_i * vr_i) / sum(I_i)

  3. Kinematic projection (matching rlink1.F):
     Overwrites the active DOFs of each tied node with the center-of-mass values,
     conserving linear and angular momentum identically.

  4. Force transfer:
     Distributes total force along tied DOFs according to nodal mass fractions:
     F_i = (m_i / M_tot) * F_tot
"""

from __future__ import annotations

import numpy as np

from ..common.messages import MessageLog
from ..model.entities import RigidLink
from ..model.model import Model


class RigidLinkEngine:
    """Rigid link constraint tying a group of nodes along active DOFs (rlink1.F)."""

    def __init__(self, rlink: RigidLink, model: Model, log: MessageLog):
        self.rlink = rlink
        self.id = rlink.id
        self.model = model
        self.log = log

        indices: list[int] = []
        if rlink.grnod_id > 0 and hasattr(model, "node_groups") and rlink.grnod_id in model.node_groups:
            grp = model.node_groups[rlink.grnod_id]
            if getattr(grp, "node_idx", None) is not None:
                indices.extend(int(idx) for idx in grp.node_idx)
            elif getattr(grp, "node_ids", None):
                for nid in grp.node_ids:
                    if nid in model._id2idx:
                        indices.append(model.node_index(nid))

        if hasattr(rlink, "node_ids") and rlink.node_ids:
            for nid in rlink.node_ids:
                if nid in model._id2idx:
                    indices.append(model.node_index(nid))

        self.node_idx: np.ndarray = np.unique(np.array(indices, dtype=np.int64)) if indices else np.empty(0, dtype=np.int64)
        self.is_valid: bool = len(self.node_idx) > 0
        self._ref_offsets: np.ndarray | None = None

    def transfer_forces(self, fint: np.ndarray, fext: np.ndarray, fcont: np.ndarray,
                        mint: np.ndarray, mass: np.ndarray) -> None:
        """Distribute total forces/moments on tied DOFs across tied nodes proportional
        to mass/inertia, conserving total force (rlink1.F).
        """
        if not self.is_valid or len(self.node_idx) < 2:
            return
        idx = self.node_idx
        m = mass[idx]
        M = float(np.sum(m))
        if M <= 0.0:
            return

        tx, ty, tz = self.rlink.tx, self.rlink.ty, self.rlink.tz
        weights = (m / M)[:, None]

        for d, active in enumerate((tx, ty, tz)):
            if active:
                for arr in (fint, fext, fcont):
                    F_tot = float(np.sum(arr[idx, d]))
                    arr[idx, d] = weights[:, 0] * F_tot

    def apply_acceleration(self, a: np.ndarray, mass: np.ndarray,
                           ar: np.ndarray | None = None,
                           inertia: np.ndarray | None = None) -> None:
        """Overwrite accelerations along tied DOFs with center-of-mass acceleration (rlink1.F)."""
        if not self.is_valid or len(self.node_idx) < 2:
            return
        idx = self.node_idx

        tx, ty, tz = self.rlink.tx, self.rlink.ty, self.rlink.tz
        if tx or ty or tz:
            m = mass[idx]
            M = float(np.sum(m))
            if M > 0.0:
                a_cm = np.sum(m[:, None] * a[idx], axis=0) / M
                if tx:
                    a[idx, 0] = a_cm[0]
                if ty:
                    a[idx, 1] = a_cm[1]
                if tz:
                    a[idx, 2] = a_cm[2]

        rx, ry, rz = self.rlink.rx, self.rlink.ry, self.rlink.rz
        if (rx or ry or rz) and ar is not None and inertia is not None:
            iner = inertia[idx]
            I_tot = float(np.sum(iner))
            if I_tot > 0.0:
                ar_cm = np.sum(iner[:, None] * ar[idx], axis=0) / I_tot
                if rx:
                    ar[idx, 0] = ar_cm[0]
                if ry:
                    ar[idx, 1] = ar_cm[1]
                if rz:
                    ar[idx, 2] = ar_cm[2]

    def apply_velocity(self, v: np.ndarray, mass: np.ndarray,
                       vr: np.ndarray | None = None,
                       inertia: np.ndarray | None = None) -> None:
        """Overwrite velocities along tied DOFs with center-of-mass velocity (rlink1.F)."""
        if not self.is_valid or len(self.node_idx) < 2:
            return
        idx = self.node_idx

        tx, ty, tz = self.rlink.tx, self.rlink.ty, self.rlink.tz
        if tx or ty or tz:
            m = mass[idx]
            M = float(np.sum(m))
            if M > 0.0:
                v_cm = np.sum(m[:, None] * v[idx], axis=0) / M
                if tx:
                    v[idx, 0] = v_cm[0]
                if ty:
                    v[idx, 1] = v_cm[1]
                if tz:
                    v[idx, 2] = v_cm[2]

        rx, ry, rz = self.rlink.rx, self.rlink.ry, self.rlink.rz
        if (rx or ry or rz) and vr is not None and inertia is not None:
            iner = inertia[idx]
            I_tot = float(np.sum(iner))
            if I_tot > 0.0:
                vr_cm = np.sum(iner[:, None] * vr[idx], axis=0) / I_tot
                if rx:
                    vr[idx, 0] = vr_cm[0]
                if ry:
                    vr[idx, 1] = vr_cm[1]
                if rz:
                    vr[idx, 2] = vr_cm[2]

    def enforce(self, x: np.ndarray, v: np.ndarray | None = None, dt: float = 0.0) -> None:
        """Lock relative nodal positions along tied translational DOFs, preventing drift."""
        if not self.is_valid or len(self.node_idx) < 2:
            return
        idx = self.node_idx
        m = self.model.mass[idx]
        M = float(np.sum(m))
        if M <= 0.0:
            return

        x_cm = np.sum(m[:, None] * x[idx], axis=0) / M
        if self._ref_offsets is None:
            self._ref_offsets = x[idx] - x_cm
        else:
            if self.rlink.tx:
                x[idx, 0] = x_cm[0] + self._ref_offsets[:, 0]
            if self.rlink.ty:
                x[idx, 1] = x_cm[1] + self._ref_offsets[:, 1]
            if self.rlink.tz:
                x[idx, 2] = x_cm[2] + self._ref_offsets[:, 2]


def build_rlinks(model: Model, log: MessageLog) -> list[RigidLinkEngine]:
    """Instantiate RigidLinkEngine for all /RLINK entities in the model."""
    rlink_engines: list[RigidLinkEngine] = []
    for rid, rlink in getattr(model, "rlinks", {}).items():
        try:
            rl = RigidLinkEngine(rlink, model, log)
            if rl.is_valid:
                rlink_engines.append(rl)
                log.info(f" /RLINK/{rid}: {len(rl.node_idx)} node(s) tied with DOFs={rlink.dofs}")
        except Exception as exc:
            log.warning(f"/RLINK/{rid}: initialization skipped: {exc}", "RLINK")
    return rlink_engines
