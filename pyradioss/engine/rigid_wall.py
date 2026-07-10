"""
Rigid wall (/RWALL/PLANE) — kinematic treatment.

Fortran origin: ``engine/source/constraints/general/rwall/rgwal*.F``.

The infinite plane is defined by point M and outward normal n (nodes are
expected on the +n side). Each cycle, AFTER the velocity update but BEFORE
the position update, every candidate node is tested: if its end-of-step
position would be behind the wall,

    s_new = (x - M).n + v.n * dt < 0

its normal velocity is replaced so the node lands exactly ON the wall:

    v.n  <-  -s/dt        (s = current signed distance)

* slide = 0 : tangential velocity kept (frictionless sliding)
* slide = 1 : tied — the full velocity is zeroed once in contact
* slide = 2 : Coulomb friction — the tangential velocity is reduced by
              the friction impulse  |dv_t| <= mu * |dv_n|

Energy accounting
-----------------
A kinematic wall arresting a node destroys the node's incoming normal
kinetic energy (a perfectly inelastic impact for that nodal mass — the
well-known kinematic-wall dissipation, reported by the original as
rigid-wall energy). But the accounting must NOT charge the *trial*
velocity a resting node re-acquires every cycle from the element forces
(that velocity is a phantom: it is cancelled before the node moves and
before any element sees it — same argument as for /BCS, see
kinematics.py). Hence the dissipation is measured against the REAL
carried-in velocity ``v_old`` (the velocity with which the node entered
this cycle):

    D = 1/2 m (v_old.n)^2 - 1/2 m (v_new.n)^2        (normal arrest)
      + friction / tied tangential terms

For a node resting on the wall v_old.n ~ v_new.n ~ 0, so persistent
contact correctly contributes nothing. The returned D goes to the
Engine's contact-energy counter.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..common.constants import EM20
from ..model.model import Model


class RigidWalls:
    def __init__(self, model: Model, log):
        self.walls = []
        for rw in model.rwalls:
            if rw.grnod_id in (None, 0):
                idx = np.arange(model.numnod)
            else:
                idx = model.node_groups[rw.grnod_id].node_idx
            self.walls.append((rw, idx))

    def apply(self, x: np.ndarray, v: np.ndarray, v_old: np.ndarray,
              mass: np.ndarray, dt: float) -> float:
        """Correct velocities against every wall.

        v      : velocities after this cycle's acceleration update (modified
                 in place)
        v_old  : velocities the nodes ENTERED the cycle with (for the
                 dissipation bookkeeping, see module docstring)

        Returns the dissipated energy to book as contact energy."""
        if not self.walls or dt <= 0.0:
            return 0.0
        removed = 0.0
        for rw, idx in self.walls:
            n = rw.normal
            s = (x[idx] - rw.point) @ n              # signed distance now
            vn = v[idx] @ n
            hit = s + vn * dt < 0.0                  # would end behind wall
            if not np.any(hit):
                continue
            i = idx[hit]
            m = mass[i]
            vn_old = v_old[i] @ n                    # real carried-in normal v
            vn_new = -s[hit] / dt                    # land exactly on wall
            dvn = vn_new - vn[hit]                   # normal velocity change
            v[i] += dvn[:, None] * n[None, :]
            # normal arrest dissipation vs the carried-in velocity
            removed += float((0.5 * m * (np.minimum(vn_old, 0.0) ** 2
                                         - vn_new ** 2)).sum())
            if rw.slide == 1:                        # tied: kill everything
                vt_old = v_old[i] - vn_old[:, None] * n[None, :]
                removed += float((0.5 * m * np.einsum(
                    "nb,nb->n", vt_old, vt_old)).sum())
                v[i] = vn_new[:, None] * n[None, :]
            elif rw.slide == 2 and rw.fric > 0.0:    # Coulomb friction
                vt = v[i] - (v[i] @ n)[:, None] * n[None, :]
                vt_mag = np.linalg.norm(vt, axis=1)
                dv_fric = np.minimum(vt_mag, rw.fric * np.abs(dvn))
                scale = 1.0 - dv_fric / np.maximum(vt_mag, EM20)
                removed += float((0.5 * m * (vt_mag ** 2
                                             - (vt_mag * scale) ** 2)).sum())
                v[i] = (v[i] @ n)[:, None] * n[None, :] + vt * scale[:, None]
        return removed
