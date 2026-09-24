"""
/NBCS — Non-Linear Boundary Conditions Engine.

Fortran origins:
  - ``engine/source/constraints/general/bcs/bcsn.F`` (BCSN: pack/unpack non-linear condition codes)
  - ``engine/source/constraints/general/bcs/bcs1.F`` (BCS1V: DOF zeroing and skew projection)
  - ``starter/source/constraints/general/bcs/hm_read_nbcs.F`` (card reading)

Physics & Kinematics:
  Dynamic non-linear kinematic constraints on individual nodal DOFs:
  1. For each constrained node in model.nbcs_blocks:
  2. If condition is active (node_cond.active is True):
  3. In global Cartesian system:
       Zero active translational DOFs in v and a.
       Zero active rotational DOFs in vr.
  4. In a local skew coordinate system:
       Project out constrained components along skew axes (bcs1.F USER SYSTEM):
         v_d = dot(v, axis_d)
         v -= v_d * axis_d
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from ..model.model import Model


class NonLinearBcsEngine:
    """Dynamic non-linear kinematic boundary condition manager for /NBCS."""

    def __init__(self, model: Any, log: Optional[Any] = None):
        self.model = model
        self.log = log

    def apply(
        self,
        v: np.ndarray,
        vr: Optional[np.ndarray] = None,
        a: Optional[np.ndarray] = None,
    ) -> None:
        """Enforce non-linear kinematic boundary constraints on velocities and accelerations."""
        if v is None:
            return

        blocks = getattr(self.model, "nbcs_blocks", {})
        if not blocks:
            return

        skews = getattr(self.model, "skews", None)
        n_nod = len(v)

        for block in blocks.values():
            nodes = getattr(block, "nodes", [])
            for cond in nodes:
                if not getattr(cond, "active", True):
                    continue

                nid = getattr(cond, "node_id", 0)
                # Resolve 0-based node index
                if hasattr(self.model, "_id2idx") and nid in self.model._id2idx:
                    idx = int(self.model._id2idx[nid])
                elif hasattr(self.model, "node_index"):
                    try:
                        idx = int(self.model.node_index(nid))
                    except (KeyError, IndexError):
                        idx = nid - 1 if nid > 0 else 0
                elif hasattr(self.model, "node_ids"):
                    m = np.where(self.model.node_ids == nid)[0]
                    idx = int(m[0]) if len(m) > 0 else (nid - 1 if nid > 0 else 0)
                else:
                    idx = nid - 1 if nid > 0 else 0

                if idx < 0 or idx >= n_nod:
                    continue

                # Determine active translation and rotation DOFs
                tra = getattr(cond, "tra", None)
                if tra is not None and len(tra) >= 3:
                    tx, ty, tz = bool(tra[0]), bool(tra[1]), bool(tra[2])
                else:
                    tx = bool(getattr(cond, "tx", 0))
                    ty = bool(getattr(cond, "ty", 0))
                    tz = bool(getattr(cond, "tz", 0))

                rot = getattr(cond, "rot", None)
                if rot is not None and len(rot) >= 3:
                    wx, wy, wz = bool(rot[0]), bool(rot[1]), bool(rot[2])
                else:
                    wx = bool(getattr(cond, "wx", 0))
                    wy = bool(getattr(cond, "wy", 0))
                    wz = bool(getattr(cond, "wz", 0))

                skew_id = int(getattr(cond, "skew_id", 0) or 0)
                if skew_id > 0 and skews is not None:
                    # Skew projection (bcs1.F)
                    axes = None
                    if hasattr(skews, "axes") and skew_id in skews.axes:
                        axes = skews.axes[skew_id]
                    elif hasattr(skews, "get_axes"):
                        axes = skews.get_axes(skew_id)

                    if axes is not None:
                        for d, constrained in enumerate((tx, ty, tz)):
                            if constrained:
                                e = axes[d]
                                comp_v = float(np.dot(v[idx], e))
                                v[idx] -= comp_v * e
                                if a is not None and idx < len(a):
                                    comp_a = float(np.dot(a[idx], e))
                                    a[idx] -= comp_a * e

                        if vr is not None and idx < len(vr):
                            for d, constrained in enumerate((wx, wy, wz)):
                                if constrained:
                                    e = axes[d]
                                    comp_vr = float(np.dot(vr[idx], e))
                                    vr[idx] -= comp_vr * e
                        continue

                # Global Cartesian system
                if tx:
                    v[idx, 0] = 0.0
                    if a is not None and idx < len(a):
                        a[idx, 0] = 0.0
                if ty:
                    v[idx, 1] = 0.0
                    if a is not None and idx < len(a):
                        a[idx, 1] = 0.0
                if tz:
                    v[idx, 2] = 0.0
                    if a is not None and idx < len(a):
                        a[idx, 2] = 0.0

                if vr is not None and idx < len(vr):
                    if wx:
                        vr[idx, 0] = 0.0
                    if wy:
                        vr[idx, 1] = 0.0
                    if wz:
                        vr[idx, 2] = 0.0
