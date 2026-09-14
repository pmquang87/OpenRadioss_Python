"""
Global equation numbering for the implicit solve (M8).

Fortran origin: ``engine/source/implicit/ind_glob_k.F`` (and ``imp_glob*``)
— the routine that walks the nodes, decides which nodal DOFs are *active*
(free and stiffness-carrying), and assigns each one a global equation index
so the sparse tangent and the residual can be addressed by equation number.

The idea
--------
Explicit dynamics never needs this: it works node-by-node with a diagonal
(lumped) mass, so a "solve" is a per-DOF division and fixed DOFs are handled
by simply zeroing the velocity after the update. Implicit statics assembles a
GLOBAL stiffness matrix K and solves K Δu = R, so every unknown must map to a
row/column of K. This module builds that map.

Each node owns six scalar DOF slots, ordered exactly like the rest of the
port's kinematics (translations first, then rotations):

    slot = node_index * 6 + component,   component 0,1,2 = ux,uy,uz
                                                   3,4,5 = rx,ry,rz

``global_dof(node, comp)`` returns that scalar slot id; the element tangents
address themselves in this space (see ``assembly.py``). The map then assigns
each ACTIVE slot a dense equation index and marks every other slot ``-1``:

* **Translations** are active on every node that carries mass and is not
  /BCS-fixed. (A free translational DOF with no element attached would make K
  singular; a well-posed static model has none, exactly as the original
  assumes.) A node with exactly ZERO mass carries no element — the only such
  node a valid deck produces is a standalone /BEAM orientation node N3, which
  receives neither mass nor force — so it gets no equations either (M11:
  numbering it would put zero rows in K; the beam frame reads its
  coordinates, never its motion).
* **Rotations** are active only on nodes that carry ROTATIONAL stiffness —
  nodes attached to a shell (BT4), a 3-node shell, or (M11) the two
  force-carrying nodes N1/N2 of a beam — and are not /BCS-rotation-fixed.
  Solid/truss/spring nodes get no rotational equations (their rotational
  DOFs are massless and stiffness-less: numbering them would put a zero row
  in K). This is the "rotations where they carry stiffness" rule of the M8
  task.

/BCS-fixed DOFs are **condensed** (removed from the system), not penalized:
a fixed slot simply never receives an equation index, so it contributes no
row/column to K. A prescribed non-zero value would enter the residual as a
known term (not used by the M8 validations, which fix DOFs to zero and load
the free ones); the machinery to support it is a one-liner in the driver and
is noted there.

Fully fixed (``mass >= 1e29``) placeholder nodes are treated as /BCS-fixed on
every DOF, matching ``engine/kinematics.py``'s frozen-node convention.
"""

from __future__ import annotations

import numpy as np

from ..model.model import Model

#: DOF slots per node: 3 translations + 3 rotations (see module docstring).
DOFS_PER_NODE = 6


class DofMap:
    """Equation numbering for one implicit model (see module docstring).

    Attributes
    ----------
    ndof   : number of active equations (the size of K / R)
    eq     : (numnod*6,) int — equation index of each scalar DOF slot, or
             -1 if the slot is fixed/inactive (condensed out)
    has_rot: (numnod,) bool — node carries rotational stiffness (shell node)
    fix_tra, fix_rot : (numnod, 3) bool — the /BCS + frozen fixity masks
    """

    def __init__(self, model: Model, log=None, prescribed=None,
                 constraints=None):
        """``constraints`` (M12, optional): an
        ``implicit.constraints.ImplicitConstraints`` whose numbering hooks
        apply — ``extra`` slots are force-numbered (constraint masters need
        their full 6-DOF block even when massless/rotation-free, and every
        DEPENDENT slot must be numbered so its element stiffness and loads
        are captured before condensation), ``unfreeze`` overrides the
        frozen-placeholder fixity of standalone master nodes (a REAL /BCS
        on a master still applies: it is the body-level condition — the
        rigid-body PIVOT when all translations are fixed), and
        ``bcs_ignore`` drops the /BCS of clash nodes the constraint wins
        over (warned at the constraint scan)."""
        self.model = model
        n = model.numnod
        extra = (constraints.extra if constraints
                 else np.zeros((n, DOFS_PER_NODE), dtype=bool))
        unfreeze = (constraints.unfreeze if constraints
                    else np.zeros(n, dtype=bool))
        bcs_ignore = (constraints.bcs_ignore if constraints
                      else np.zeros(n, dtype=bool))

        # ---- which nodes carry rotational stiffness ----------------------
        # shell families and (M11) beams put bending/twist stiffness on
        # their nodes; solids/trusses/springs contribute translational
        # stiffness only, so numbering their rotational slots would leave
        # zero rows in K. Beam connectivity is (N1, N2, N3) with N3 the
        # force-free ORIENTATION node — only N1/N2 carry stiffness.
        has_rot = np.zeros(n, dtype=bool)
        for name in ("shells", "sh3n", "shells_qbat", "shells_qeph", "sh3n_dkt18"):
            g = getattr(model, name, None)
            if g is not None and g.n:
                has_rot[g.conn.reshape(-1)] = True
        g = getattr(model, "beams", None)
        if g is not None and g.n:
            has_rot[g.conn[:, :2].reshape(-1)] = True
        self.has_rot = has_rot

        # ---- /BCS + frozen fixity masks (same convention as kinematics) ---
        fix_tra = np.zeros((n, 3), dtype=bool)
        fix_rot = np.zeros((n, 3), dtype=bool)
        for bc in model.bcs:
            grp = model.node_groups.get(bc.grnod_id)
            if grp is None or grp.node_idx is None:
                continue
            # M12: nodes whose /BCS clashes with a kinematic constraint
            # (rigid slaves, tied secondaries, RBE3 references) — the
            # constraint wins, exactly the explicit convention (warned)
            idx = grp.node_idx[~bcs_ignore[grp.node_idx]]
            for d in range(3):
                if bc.fix_tra[d]:
                    fix_tra[idx, d] = True
                if bc.fix_rot[d]:
                    fix_rot[idx, d] = True
        # frozen placeholder nodes (mass ~ 1e30) are fully fixed — except
        # constraint masters/dependents (M12): a standalone /RBODY master
        # is exactly such a placeholder, yet it must carry the body's six
        # condensed equations
        frozen = (model.mass >= 1e29) & ~unfreeze
        fix_tra[frozen, :] = True
        fix_rot[frozen, :] = True
        # PRESCRIBED (imposed-displacement) DOFs are condensed exactly like
        # /BCS-fixed ones — they are known, not unknown. Their known VALUE is
        # carried by the driver in the displacement vector (it enters the
        # residual through f_int); only their equation row/column is removed.
        # ``prescribed`` is an optional (numnod, 6) bool mask.
        if prescribed is not None:
            fix_tra |= prescribed[:, :3]
            fix_rot |= prescribed[:, 3:]
        self.fix_tra = fix_tra
        self.fix_rot = fix_rot

        # ---- assign equation indices -------------------------------------
        # a node with no mass and no rotational stiffness contributes nothing
        # (its translations would be zero rows) — but a massed, unfixed node
        # is assumed element-attached (well-posed static model), so every
        # such translational slot is active. Exactly-zero-mass nodes (a
        # standalone beam orientation node N3 — the module docstring) carry
        # no element and are excluded (M11).
        massed = (model.mass > 0.0) & (model.mass < 1e29)
        eq = np.full(n * DOFS_PER_NODE, -1, dtype=np.int64)
        counter = 0
        for i in range(n):
            base = i * DOFS_PER_NODE
            # translations (M12: constraint masters/dependents force-
            # numbered through ``extra`` — see the constructor docstring)
            if massed[i] or extra[i, :3].any():
                for c in range(3):
                    if not fix_tra[i, c] and (massed[i] or extra[i, c]):
                        eq[base + c] = counter
                        counter += 1
            # rotations (shell/beam nodes, + forced constraint slots)
            if has_rot[i] or extra[i, 3:].any():
                for c in range(3):
                    if not fix_rot[i, c] and (has_rot[i]
                                              or extra[i, 3 + c]):
                        eq[base + 3 + c] = counter
                        counter += 1
        self.eq = eq
        self.ndof = counter

        if log is not None:
            log.info(f" IMPLICIT EQUATIONS (FREE DOFS) . . . : {self.ndof}")

    # ----------------------------------------------------------------------
    @staticmethod
    def global_dof(node: np.ndarray, comp: int) -> np.ndarray:
        """Scalar DOF slot id of (node, component). Vectorized over node."""
        return node * DOFS_PER_NODE + comp

    def equations(self, slots: np.ndarray) -> np.ndarray:
        """Equation index of each scalar DOF slot (-1 where condensed)."""
        return self.eq[slots]

    def gather_residual(self, fnod: np.ndarray, mnod: np.ndarray) -> np.ndarray:
        """Pack per-node force/moment vectors (numnod,3) into an equation
        vector of length ndof, dropping the condensed (fixed) rows.

        ``fnod`` are translational components, ``mnod`` the rotational
        (moment) components — exactly the ``fint``/``mint`` split the
        element kernels already produce."""
        out = np.zeros(self.ndof)
        eq = self.eq
        for c in range(3):
            e = eq[np.arange(self.model.numnod) * DOFS_PER_NODE + c]
            act = e >= 0
            out[e[act]] = fnod[act, c]
        for c in range(3):
            e = eq[np.arange(self.model.numnod) * DOFS_PER_NODE + 3 + c]
            act = e >= 0
            out[e[act]] = mnod[act, c]
        return out

    def scatter_solution(self, du_eq: np.ndarray):
        """Expand an equation-space solution vector back to per-node
        (numnod,3) translation and rotation increments (fixed DOFs = 0)."""
        n = self.model.numnod
        du = np.zeros((n, 3))
        dur = np.zeros((n, 3))
        eq = self.eq
        node = np.arange(n)
        for c in range(3):
            e = eq[node * DOFS_PER_NODE + c]
            act = e >= 0
            du[act, c] = du_eq[e[act]]
        for c in range(3):
            e = eq[node * DOFS_PER_NODE + 3 + c]
            act = e >= 0
            dur[act, c] = du_eq[e[act]]
        return du, dur
