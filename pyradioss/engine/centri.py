"""Centrifugal load engine: body force from rotational velocity and acceleration.

Fortran origins:
- ``starter/source/loads/general/load_centri/hm_read_load_centri.F``
- ``engine/source/loads/general/load_centri/cfield.F``
- ``engine/source/loads/general/load_centri/cfield_imp.F``

Physics formulation:
Given a rotation axis passing through origin x0 with unit direction vector n,
and time-dependent angular velocity omega(t) and angular acceleration alpha(t) = domega/dt:
For each node i in the specified node group / part:
  Delta_x = x_i - x_0
  r_i = Delta_x - (Delta_x . n) n       (perpendicular vector from rotation axis)
  a_centri = omega^2 * r_i              (radial outward centrifugal acceleration)
  a_tan = alpha * (n x Delta_x)          (tangential acceleration from angular acceleration)
  a_total = a_centri + a_tan
  F_i = m_i * a_total                   (nodal force added to fext)

Work ledger booking:
  dW = sum(F_i . v_{i, mid}) * dt is booked into external work (state.wext / state.e_ext).
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple
import numpy as np

from ..common.messages import MessageLog
from ..model.entities import CentrifugalLoad


class CentrifugalLoadEngine:
    """Computes nodal centrifugal and angular acceleration forces for /LOAD/CENTRI and /CENTRI."""

    def __init__(self, model: Any, log: Optional[MessageLog] = None):
        self.model = model
        self.log = log
        self.loads: List[dict] = []

        def _node_idx(nid: int) -> int:
            if nid <= 0:
                return -1
            if hasattr(model, "_id2idx") and nid in model._id2idx:
                return model._id2idx[nid]
            if hasattr(model, "node_ids"):
                m = np.where(model.node_ids == nid)[0]
                if len(m) > 0:
                    return int(m[0])
            return -1

        # Helper to resolve node group to node indices
        def _grp(gid: int) -> np.ndarray:
            if gid == 0:
                return np.arange(model.numnod, dtype=np.int64)
            if hasattr(model, "node_groups") and gid in model.node_groups:
                g = model.node_groups[gid]
                nids = g.node_ids if hasattr(g, "node_ids") else g
                res = []
                for nid in nids:
                    idx = _node_idx(int(nid))
                    if 0 <= idx < model.numnod:
                        res.append(idx)
                return np.array(res, dtype=np.int64)
            return np.zeros(0, dtype=np.int64)

        # Helper to resolve function
        def _get_func(fid: int):
            if fid == 0 or fid is None:
                return None
            funcs = getattr(model, "functions", {})
            return funcs.get(fid)

        # Gather all centrifugal loads from model.centri_loads and model.centris
        raw_loads: List[CentrifugalLoad] = []
        seen_ids = set()

        for cl in getattr(model, "centri_loads", []):
            if cl.id not in seen_ids:
                seen_ids.add(cl.id)
                raw_loads.append(cl)

        for cid, c in getattr(model, "centris", {}).items():
            if cid not in seen_ids:
                seen_ids.add(cid)
                raw_loads.append(c)

        for cl in raw_loads:
            gid = getattr(cl, "grnod_id", 0) or getattr(cl, "grnd_id", 0)
            idx = _grp(gid)
            if len(idx) == 0:
                continue

            fid = getattr(cl, "funct_id", 0) or getattr(cl, "fct_id", 0)
            fct = _get_func(fid)

            scale_x = getattr(cl, "scale_x", 1.0)
            if scale_x in (0.0, None):
                scale_x = 1.0

            scale_y = getattr(cl, "scale_y", 1.0)
            if scale_y in (0.0, None):
                scale_y = 1.0

            omega_base = float(getattr(cl, "omega", 0.0) or 0.0)
            ivar = int(getattr(cl, "ivar", 1) or 1)
            sens_id = int(getattr(cl, "sens_id", 0) or 0)

            # Determine rotation axis origin x0 and unit direction vector n
            node_orig = int(getattr(cl, "node_orig", 0) or 0)
            node_axis = int(getattr(cl, "node_axis", 0) or 0)
            frame_id = int(getattr(cl, "frame_id", 0) or 0)
            dir_str = str(getattr(cl, "dir", "ZZ") or "ZZ").upper()

            x0 = np.zeros(3, dtype=np.float64)
            n = np.zeros(3, dtype=np.float64)

            # Center node
            if node_orig > 0:
                n_idx = _node_idx(node_orig)
                if 0 <= n_idx < model.numnod:
                    x0 = model.x0[n_idx].copy()

            # Axis direction
            if node_axis > 0:
                a_idx = _node_idx(node_axis)
                if 0 <= a_idx < model.numnod:
                    ax_pt = model.x0[a_idx].copy()
                    axis_vec = ax_pt - x0
                    norm = np.linalg.norm(axis_vec)
                    if norm > 1e-12:
                        n = axis_vec / norm

            if np.linalg.norm(n) < 1e-6:
                # Use dir_str and frame
                if dir_str in ("X", "XX"):
                    n = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                elif dir_str in ("Y", "YY"):
                    n = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                elif dir_str in ("Z", "ZZ"):
                    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                else:
                    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)

                # If frame is defined, rotate n and shift x0 if frame has origin
                if frame_id > 0:
                    frames = getattr(model, "frames", {})
                    skews = getattr(model, "skews", {})
                    if frame_id in frames:
                        fr = frames[frame_id]
                        if hasattr(fr, "origin"):
                            x0 = np.array(fr.origin, dtype=np.float64)
                        if hasattr(fr, "matrix"):
                            mat = np.array(fr.matrix, dtype=np.float64).reshape(3, 3)
                            n = mat @ n
                    elif frame_id in skews:
                        sk = skews[frame_id]
                        if hasattr(sk, "origin"):
                            x0 = np.array(sk.origin, dtype=np.float64)
                        if hasattr(sk, "matrix"):
                            mat = np.array(sk.matrix, dtype=np.float64).reshape(3, 3)
                            n = mat @ n

            norm_n = np.linalg.norm(n)
            if norm_n > 1e-12:
                n = n / norm_n
            else:
                n = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            self.loads.append({
                "id": getattr(cl, "id", 0),
                "idx": idx,
                "fct": fct,
                "scale_x": scale_x,
                "scale_y": scale_y,
                "omega_base": omega_base,
                "ivar": ivar,
                "sens_id": sens_id,
                "x0": x0,
                "n": n,
                "node_orig": node_orig,
            })

    def compute_forces(self, t: float, x: np.ndarray, fext: np.ndarray,
                       sensors: Any = None) -> None:
        """Compute centrifugal and angular acceleration forces at time t and add to fext."""
        if not self.loads:
            return

        # Physical mass excluding frozen dummy mass (1e30)
        m_all = self.model.mass
        if hasattr(self.model, "frozen"):
            m_eff = np.where(self.model.frozen, 0.0, m_all)
        else:
            m_eff = m_all

        for ld in self.loads:
            idx = ld["idx"]
            if len(idx) == 0:
                continue

            sens_id = ld["sens_id"]
            if sens_id > 0:
                if sensors is not None:
                    if not sensors.active(sens_id):
                        continue
                    ts = sensors.shifted_time(sens_id, t)
                    if ts is None or ts < 0.0:
                        continue
                else:
                    ts = t
            else:
                ts = t

            fct = ld["fct"]
            sx = ld["scale_x"]
            sy = ld["scale_y"]
            omega_base = ld["omega_base"]
            ivar = ld["ivar"]
            n = ld["n"]

            # Dynamic center position if node_orig is specified
            node_orig = ld["node_orig"]
            if node_orig > 0:
                n_idx = -1
                if hasattr(self.model, "_id2idx") and node_orig in self.model._id2idx:
                    n_idx = self.model._id2idx[node_orig]
                elif hasattr(self.model, "node_ids"):
                    m = np.where(self.model.node_ids == node_orig)[0]
                    if len(m) > 0:
                        n_idx = int(m[0])
                if 0 <= n_idx < len(x):
                    x0 = x[n_idx]
                else:
                    x0 = ld["x0"]
            else:
                x0 = ld["x0"]

            def _eval_curve(c, t_val):
                if hasattr(c, "eval"):
                    return float(c.eval(t_val))
                return float(c(t_val))

            # Evaluate angular velocity omega(t)
            if fct is not None:
                eval_t = ts / sx
                fval = _eval_curve(fct, eval_t)
                if omega_base != 0.0:
                    omega = omega_base * sy * fval
                else:
                    omega = sy * fval
            else:
                omega = omega_base * sy

            # Evaluate angular acceleration alpha(t) = domega/dt if ivar == 2
            alpha = 0.0
            if ivar == 2 and fct is not None:
                # Finite-difference derivative of the velocity curve
                dt_probe = max(1e-7, 1e-5 * sx)
                f1 = _eval_curve(fct, (ts + dt_probe) / sx)
                f0 = _eval_curve(fct, (ts - dt_probe) / sx)
                dfdt = (f1 - f0) / (2.0 * dt_probe)
                if omega_base != 0.0:
                    alpha = omega_base * sy * dfdt
                else:
                    alpha = sy * dfdt

            # Geometry for nodes:
            # dx = x_i - x0
            # r_i = dx - (dx . n) * n
            dx = x[idx] - x0  # (N, 3)
            dot_n = np.sum(dx * n, axis=1, keepdims=True)  # (N, 1)
            r = dx - dot_n * n  # (N, 3)

            # a_centri = omega^2 * r
            omega2 = omega * omega
            a_centri = omega2 * r

            # a_tan = alpha * (n x dx)
            if abs(alpha) > 1e-15:
                n_cross_dx = np.cross(n, dx)  # (N, 3)
                a_tan = alpha * n_cross_dx
            else:
                a_tan = 0.0

            a_total = a_centri + a_tan
            m_sub = m_eff[idx, None]  # (N, 1)
            F = m_sub * a_total  # (N, 3)

            fext[idx] += F
