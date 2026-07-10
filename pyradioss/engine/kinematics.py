"""
Kinematic conditions and external loads.

Fortran origin: ``engine/source/constraints/general/bcs/bcs*.F`` (fixed
DOFs), ``.../impvel`` (imposed velocities), ``engine/source/loads``
(``force.F`` concentrated loads, ``gravit.F`` gravity).

Two very different mechanisms, kept distinct exactly like the original:

* **Loads** (gravity, concentrated forces) ADD to the external force
  vector; the node still obeys F = m a.
* **Kinematic conditions** (BCS, imposed velocity, rigid walls) OVERRIDE
  the computed velocity after the acceleration update. Their energy
  contribution is booked as external work through the kinetic-energy
  change they cause (see _apply_velocity_override) so the global energy
  balance stays consistent — the original recovers the same work from the
  constraint reactions.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..model.model import Model


class LoadsAndConstraints:
    """Pre-resolved (index-based) loads + constraints for the Engine."""

    def __init__(self, model: Model, log):
        self.model = model
        # BCS -> per-dof boolean masks
        self.fix_tra = np.zeros((model.numnod, 3), dtype=bool)
        self.fix_rot = np.zeros((model.numnod, 3), dtype=bool)
        for bc in model.bcs:
            idx = model.node_groups[bc.grnod_id].node_idx
            for d in range(3):
                if bc.fix_tra[d]:
                    self.fix_tra[idx, d] = True
                if bc.fix_rot[d]:
                    self.fix_rot[idx, d] = True

        # frozen (massless) nodes are fully fixed
        frozen = model.mass >= 1e29
        self.fix_tra[frozen, :] = True
        self.fix_rot[frozen, :] = True

        # resolved loads: (node_idx, direction, funct, scale)
        def _grp(gid):
            if gid in (None, 0):
                return np.arange(model.numnod)
            return model.node_groups[gid].node_idx

        self.gravity = [(_grp(g.grnod_id), g.direction, model.functions[g.funct_id],
                         g.scale) for g in model.gravity]
        self.cloads = [(_grp(c.grnod_id), c.direction, model.functions[c.funct_id],
                        c.scale) for c in model.cloads]
        self.impvel = [(_grp(i.grnod_id), i.dof, model.functions[i.funct_id],
                        i.scale) for i in model.impvel]

    # ------------------------------------------------------------------
    def external_forces(self, t: float, fext: np.ndarray) -> None:
        """Accumulate gravity + concentrated loads at time t (gravit.F,
        force.F). Gravity is an acceleration -> F = m * a per node;
        /CLOAD applies the full F(t) to every node of its group."""
        m = self.model.mass
        for idx, direction, fct, scale in self.gravity:
            acc = scale * fct.eval(t)
            fext[idx] += (m[idx, None] * acc) * direction[None, :]
        for idx, direction, fct, scale in self.cloads:
            F = scale * fct.eval(t)
            fext[idx] += F * direction[None, :]

    # ------------------------------------------------------------------
    def apply_kinematic(self, t: float, v: np.ndarray, vr: np.ndarray,
                        mass: np.ndarray) -> float:
        """Apply BCS + /IMPVEL to the freshly updated velocities.

        Returns the external work done by the constraints this cycle.

        Work accounting (must be consistent with the element ledger, which
        measures internal work with the POST-enforcement velocities):

        * /IMPVEL: the constraint applies the impulse J = m (v_imp - v_free)
          and the node then MOVES with v_imp, so the constraint's mechanical
          work over the coming interval is  J . v_imp  (reaction force times
          actual displacement). This is how the original recovers /IMPVEL
          work from the constraint reactions.
        * /BCS: a permanently fixed node never moves — the trial velocity
          m a dt it briefly acquires is a phantom (it is zeroed before it
          moves anything and before any element sees it), so a fixed DOF
          contributes exactly ZERO work.
        """
        w = 0.0
        # imposed velocities first (a BCS on the same dof wins, as in the
        # original where BCS is the strongest condition)
        for idx, dof, fct, scale in self.impvel:
            vimp = scale * fct.eval(t)
            dv = vimp - v[idx, dof]
            w += float(np.dot(mass[idx], dv)) * vimp
            v[idx, dof] = vimp
        # fixed DOFs: zero velocity (no work — see docstring)
        v[self.fix_tra] = 0.0
        vr[self.fix_rot] = 0.0
        return w
