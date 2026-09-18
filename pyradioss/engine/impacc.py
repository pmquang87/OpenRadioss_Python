"""Imposed Acceleration Engine (/IMPACC).

Fortran origins:
- ``starter/source/constraints/general/impvel/hm_read_impacc.F``
- ``engine/source/constraints/general/impvel/fixvel.F`` (IBFV(7, N) == 0 branch)

Physics & Kinematics:
Prescribes time-dependent acceleration a_imp(t) = scale * funct(t / xscale)
on specified DOF of a node group inside activation window [tstart, tstop].

Enforcement:
Enforced during acceleration update (Step 4 of explicit cycle):
  a_free = acc[idx, dof]
  acc[idx, dof] = a_imp
  Reaction force: R_i = mass_i * (a_imp - a_free)
  Work ledger booking: dW = sum(R_i . v_{i, mid}) * dt is booked into external work (state.wext / state.e_ext).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..common.messages import MessageLog
from ..model.entities import ImposedAcceleration


class ImposedAccelerationEngine:
    """Manages and enforces /IMPACC kinematic acceleration conditions."""

    def __init__(self, model: Any, log: Optional[MessageLog] = None):
        self.model = model
        self.log = log
        self.entries: List[dict] = []
        self.reactions: Dict[int, np.ndarray] = {}  # {impacc_id: reaction_forces}
        self.total_reaction_tra = np.zeros((model.numnod, 3), dtype=np.float64)
        self.total_reaction_rot = np.zeros((model.numnod, 3), dtype=np.float64)

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

        def _get_func(fid: int):
            if fid == 0 or fid is None:
                return None
            funcs = getattr(model, "functions", {})
            return funcs.get(fid)

        frozen = getattr(model, "frozen", np.zeros(model.numnod, dtype=bool))
        self._frozen = frozen

        for ia in getattr(model, "impacc", []):
            idx = _grp(ia.grnod_id)
            if len(idx) == 0:
                continue

            fct = _get_func(ia.funct_id)
            scale = float(getattr(ia, "scale", 1.0) or 1.0)
            xscale = float(getattr(ia, "xscale", 1.0) or 1.0)
            facx = 1.0 / xscale if xscale not in (0.0, None) else 1.0
            tstart = float(getattr(ia, "tstart", 0.0) or 0.0)
            tstop = float(getattr(ia, "tstop", 1.0e30) or 1.0e30)
            sens_id = int(getattr(ia, "sens_id", 0) or 0)
            skew_row = int(getattr(ia, "skew_row", 0) or getattr(ia, "skew_id", 0) or 0)
            dof = int(ia.dof)

            # Release frozen placeholder for driven DOFs
            fzn = idx[frozen[idx]]
            if len(fzn) > 0:
                if hasattr(model, "fix_tra") and dof < 3:
                    model.fix_tra[fzn, dof] = False
                elif hasattr(model, "fix_rot") and dof >= 3:
                    model.fix_rot[fzn, dof - 3] = False

            self.entries.append({
                "id": ia.id,
                "idx": idx,
                "dof": dof,
                "fct": fct,
                "scale": scale,
                "facx": facx,
                "tstart": tstart,
                "tstop": tstop,
                "sens_id": sens_id,
                "skew_row": skew_row,
            })

    def apply(self, t: float, dt: float,
              acc: np.ndarray, ar: Optional[np.ndarray],
              mass: np.ndarray, inertia: Optional[np.ndarray],
              v: np.ndarray, vr: Optional[np.ndarray],
              v_old: np.ndarray, vr_old: Optional[np.ndarray],
              sensors: Any = None) -> float:
        """Enforce imposed acceleration at time t on acc/ar, compute reaction forces and work.

        Returns:
            dW: external work done by the constraint forces over this step.
        """
        if not self.entries:
            return 0.0

        w = 0.0
        self.total_reaction_tra[:] = 0.0
        self.total_reaction_rot[:] = 0.0

        rot_gen = inertia if inertia is not None else mass

        for entry in self.entries:
            idx = entry["idx"]
            if len(idx) == 0:
                continue

            sens_id = entry["sens_id"]
            if sens_id > 0:
                if sensors is not None:
                    if not sensors.active(sens_id):
                        continue
                    te = sensors.shifted_time(sens_id, t)
                    if te is None:
                        continue
                else:
                    te = t
            else:
                te = t

            tstart = entry["tstart"]
            tstop = entry["tstop"]
            if te < tstart or te > tstop:
                continue

            fct = entry["fct"]
            scale = entry["scale"]
            facx = entry["facx"]
            dof = entry["dof"]
            skew_row = entry["skew_row"]

            # Prescribed acceleration
            if fct is not None:
                aimp = scale * (float(fct.eval(te * facx)) if hasattr(fct, "eval") else float(fct(te * facx)))
            else:
                aimp = scale

            # Mass excluding frozen placeholder
            g = np.where(self._frozen[idx], 0.0, mass[idx])

            if skew_row > 0 and hasattr(self.model, "skews") and skew_row in self.model.skews:
                # Skew coordinate system
                sk = self.model.skews[skew_row]
                mat = np.array(sk.matrix, dtype=np.float64).reshape(3, 3) if hasattr(sk, "matrix") else np.eye(3)
                axis_idx = dof % 3
                e = mat[axis_idx]  # unit axis vector

                if dof < 3:
                    # Translational along skew axis
                    a0 = np.sum(acc[idx] * e, axis=1)  # projection
                    da = aimp - a0
                    acc[idx] += da[:, None] * e
                    R = (g * da)[:, None] * e  # (N, 3)
                    self.total_reaction_tra[idx] += R
                    self.reactions[entry["id"]] = R

                    # Work booking: R . v_mid * dt
                    # v_mid = v_old + 0.5 * acc * dt
                    v_mid = 0.5 * (v_old[idx] + (v_old[idx] + acc[idx] * dt))
                    w += float(np.sum(R * v_mid)) * dt
                else:
                    # Rotational about skew axis
                    if ar is not None:
                        g_rot = np.where(self._frozen[idx], 0.0, rot_gen[idx])
                        a0 = np.sum(ar[idx] * e, axis=1)
                        da = aimp - a0
                        ar[idx] += da[:, None] * e
                        R = (g_rot * da)[:, None] * e
                        self.total_reaction_rot[idx] += R
                        self.reactions[entry["id"]] = R
                        vr_base = vr_old[idx] if vr_old is not None else (vr[idx] if vr is not None else 0.0)
                        vr_mid = 0.5 * (vr_base + (vr_base + ar[idx] * dt))
                        w += float(np.sum(R * vr_mid)) * dt
            else:
                # Cartesian global DOF
                if dof < 3:
                    a_old = acc[idx, dof].copy()
                    acc[idx, dof] = aimp
                    da = aimp - a_old
                    R = g * da  # (N,)
                    self.total_reaction_tra[idx, dof] += R
                    self.reactions[entry["id"]] = R

                    # Midstep velocity for leapfrog work: v_mid = 0.5 * (v_old + v_new)
                    v_base = v_old[idx, dof]
                    v_new = v_base + aimp * dt
                    v_mid = 0.5 * (v_base + v_new)
                    w += float(np.dot(R, v_mid)) * dt
                else:
                    dof_rot = dof - 3
                    if ar is not None:
                        g_rot = np.where(self._frozen[idx], 0.0, rot_gen[idx])
                        a_old = ar[idx, dof_rot].copy()
                        ar[idx, dof_rot] = aimp
                        da = aimp - a_old
                        R = g_rot * da
                        self.total_reaction_rot[idx, dof_rot] += R
                        self.reactions[entry["id"]] = R

                        vr_base = vr_old[idx, dof_rot] if vr_old is not None else (vr[idx, dof_rot] if vr is not None else 0.0)
                        vr_new = vr_base + aimp * dt
                        vr_mid = 0.5 * (vr_base + vr_new)
                        w += float(np.dot(R, vr_mid)) * dt

        # Expose reactions on model for post-processing / interrogation
        self.model.impacc_reactions = self.reactions
        return w
