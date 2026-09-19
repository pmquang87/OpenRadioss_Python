"""
/INTER/LAGMUL — Global Sparse Lagrange Multipliers Solver.

OpenRadioss uses a global sparse matrix formulation (L M^-1 L^T lambda = b)
solved via a Preconditioned Conjugate Gradient (PCG) solver for all
/INTER/LAGMUL interfaces (TYPE16, TYPE17, TYPE7, TYPE2), rather than a
local block-diagonal approach.

This module provides the central LagmulSolver that aggregates the kinematic
constraint rows (L) from all active LAGMUL interfaces, constructs the global
sparse H matrix, solves the PCG system, and applies the constraint forces
to the nodal accelerations.
"""

from __future__ import annotations

import numpy as np
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    HAS_SCIPY = True
except ImportError:
    sp = None
    spla = None
    HAS_SCIPY = False

from ..model.model import Model


class LagmulBcs:
    """Lagrange multiplier constraint handler for /BCS/LAGMUL boundary conditions.

    Fortran origin: ``engine/source/tools/lagmul/lag_bcs.F``.
    Constrains nodal translational and rotational DOFs via global constraint matrix L.
    """

    def __init__(self, bcs, model: Model, log=None):
        self.bcs = bcs
        self.model = model
        self.log = log if log is not None else getattr(model, "log", None)

    def generate_l_constraint_rows(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.generate_l_matrix(1e-6)

    def generate_l_matrix(self, dt=None):
        """Yield (data, node_indices, dof_indices, eq_indices, n_rows) for global LagmulSolver.
        If dt is provided, returns (L_data, L_row, L_col).
        """
        nodes_list = []
        grnod_id = getattr(self.bcs, "grnod_id", 0)
        node_groups = getattr(self.model, "node_groups", {})
        if grnod_id in node_groups:
            idx = node_groups[grnod_id].node_idx
            if idx is not None:
                nodes_list.extend(int(n) for n in idx if 0 <= n < self.model.numnod)
        node_id = getattr(self.bcs, "node_id", 0)
        if node_id != 0:
            node_map = getattr(self.model, "node_id_to_idx", {})
            if node_id in node_map:
                nodes_list.append(int(node_map[node_id]))

        if not nodes_list:
            if dt is not None:
                return (np.zeros(0, dtype=float), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
            return (np.zeros(0, dtype=float), np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), 0)

        tra = getattr(self.bcs, "tra", None)
        if isinstance(tra, str):
            tra_flags = [c == "1" for c in tra.ljust(3, "0")[:3]]
        elif isinstance(tra, (list, tuple, np.ndarray)):
            tra_flags = [bool(x) for x in tra[:3]]
        else:
            tra_flags = [False, False, False]

        rot = getattr(self.bcs, "rot", None)
        if isinstance(rot, str):
            rot_flags = [c == "1" for c in rot.ljust(3, "0")[:3]]
        elif isinstance(rot, (list, tuple, np.ndarray)):
            rot_flags = [bool(x) for x in rot[:3]]
        else:
            rot_flags = [False, False, False]

        data: list[float] = []
        nodes: list[int] = []
        dofs: list[int] = []
        eq_ids: list[int] = []
        eq_id = 0

        for nod in nodes_list:
            for d in range(3):
                if tra_flags[d]:
                    data.append(1.0)
                    nodes.append(nod)
                    dofs.append(d)
                    eq_ids.append(eq_id)
                    eq_id += 1
            for d in range(3):
                if rot_flags[d]:
                    data.append(1.0)
                    nodes.append(nod)
                    dofs.append(d)
                    eq_ids.append(eq_id)
                    eq_id += 1

        n_rows = eq_id
        if dt is not None:
            L_data = np.asarray(data, dtype=float)
            L_row = np.asarray(eq_ids, dtype=np.int64)
            L_col = np.asarray(nodes, dtype=np.int64) * 3 + np.asarray(dofs, dtype=np.int64)
            return L_data, L_row, L_col

        return (
            np.asarray(data, dtype=float),
            np.asarray(nodes, dtype=np.int64),
            np.asarray(dofs, dtype=np.int64),
            np.asarray(eq_ids, dtype=np.int64),
            n_rows,
        )


class LagmulSolver:
    """Global sparse PCG solver for /INTER/LAGMUL constraints (M6/Lagmul)."""

    def __init__(self, model: Model, loads, log):
        self.model = model
        self.log = log
        self._warned = False
        from ..contact.inter_type16 import LagmulType16
        from ..contact.inter_type17 import LagmulType17
        from ..contact.inter_type7 import LagmulType7
        from ..contact.inter_type2 import LagmulType2
        from ..contact.inter_type11 import LagmulType11
        
        # Instantiate the specific handlers which generate the L rows.
        self.interfaces = []
        for itf in model.interfaces:
            if not getattr(itf, "lagmul", False):
                continue
            if not HAS_SCIPY:
                raise ImportError("LagmulSolver requires scipy to be installed.")
            if itf.type == 16:
                self.interfaces.append(LagmulType16(itf, model, log))
            elif itf.type == 17:
                self.interfaces.append(LagmulType17(itf, model, log))
            elif itf.type == 7:
                self.interfaces.append(LagmulType7(itf, model, log))
            elif itf.type == 2:
                self.interfaces.append(LagmulType2(itf, model, log))
            elif itf.type == 11:
                self.interfaces.append(LagmulType11(itf, model, log))
            else:
                log.warning(f"/INTER/LAGMUL/TYPE{itf.type} Engine constraint builder not implemented.", "LAGMUL")

        from .rigid_wall import LagmulRWall
        for rw in getattr(model, "rwalls", []):
            if not getattr(rw, "lagmul", False):
                continue
            if not HAS_SCIPY:
                raise ImportError("LagmulSolver requires scipy to be installed.")
            self.interfaces.append(LagmulRWall(rw, model, log))

        # Wire /BCS/LAGMUL
        bcs_lagmuls = getattr(model, "bcs_lagmuls", {})
        if isinstance(bcs_lagmuls, dict):
            bcs_list = list(bcs_lagmuls.values())
        elif isinstance(bcs_lagmuls, (list, tuple)):
            bcs_list = list(bcs_lagmuls)
        else:
            bcs_list = []
        for bcs in bcs_list:
            if not HAS_SCIPY:
                raise ImportError("LagmulSolver requires scipy to be installed.")
            self.interfaces.append(LagmulBcs(bcs, model, log))

        self.nc = 0  # Total number of constraint rows

    def __len__(self):
        return len(self.interfaces)

    def transfer_forces(self, fint, fcont, fext, mint,
                        inv_mass, inv_inertia, dt) -> None:
        """Per-cycle force stage: solve the global sparse system and inject
        Lagrange constraint forces."""
        if not self.interfaces:
            return
            
        all_data = []
        all_col = []
        all_row = []
        
        row_offset = 0
        for itf in self.interfaces:
            data, nodes, dofs, eq_ids, n_rows = itf.generate_l_matrix()
            if n_rows == 0:
                continue
            all_data.append(data)
            # col index is (node * 3) + dof
            col = nodes * 3 + dofs
            all_col.append(col)
            all_row.append(eq_ids + row_offset)
            row_offset += n_rows
            
        self.nc = row_offset
        if self.nc == 0:
            return
            
        data = np.concatenate(all_data)
        row = np.concatenate(all_row)
        col = np.concatenate(all_col)
        
        n_dofs = len(inv_mass) * 3
        L = sp.coo_matrix((data, (row, col)), shape=(self.nc, n_dofs)).tocsr()
        
        # 3. a0 = (fint + fcont + fext) * inv_mass
        a0 = ((fint + fcont + fext) * inv_mass[:, None]).flatten()
        
        # 4. rhs = -L @ a0
        rhs = -L @ a0
        
        # 5. H = L @ M_inv @ L.T
        # We can form M_inv as a diagonal sparse matrix
        M_inv_diag = np.repeat(inv_mass, 3)
        M_inv = sp.diags(M_inv_diag, format="csr")
        H = L @ M_inv @ L.T
        
        # 6. lambda_vec, info = spla.cg(H, rhs, rtol=1e-5)
        lambda_vec, info = spla.cg(H, rhs, rtol=1e-5)
        if info != 0:
            self.log.warning(f"LAGMUL PCG solve failed with info={info}", "LAGMUL")
            
        # 7. forces = L.T @ lambda_vec
        forces = L.T @ lambda_vec
        
        # 8. fint += forces
        fint += forces.reshape(-1, 3)

    def enforce(self, v, vr, inv_mass, inv_inertia) -> None:
        """Velocity cleanup stage: project post-kinematic velocities back
        onto L v = 0 using mass-weighted minimum-norm correction."""
        if not self.interfaces:
            return
            
        # Velocity projection: L v = 0
        all_data = []
        all_col = []
        all_row = []
        
        row_offset = 0
        for itf in self.interfaces:
            data, nodes, dofs, eq_ids, n_rows = itf.generate_l_matrix()
            if n_rows == 0:
                continue
            all_data.append(data)
            col = nodes * 3 + dofs
            all_col.append(col)
            all_row.append(eq_ids + row_offset)
            row_offset += n_rows
            
        self.nc = row_offset
        if self.nc == 0:
            return
            
        data = np.concatenate(all_data)
        row = np.concatenate(all_row)
        col = np.concatenate(all_col)
        
        n_dofs = len(inv_mass) * 3
        L = sp.coo_matrix((data, (row, col)), shape=(self.nc, n_dofs)).tocsr()
        
        v_flat = v.flatten()
        rhs = -L @ v_flat
        
        M_inv_diag = np.repeat(inv_mass, 3)
        M_inv = sp.diags(M_inv_diag, format="csr")
        H = L @ M_inv @ L.T
        
        lambda_vec, info = spla.cg(H, rhs, rtol=1e-5)
        
        dv = M_inv @ (L.T @ lambda_vec)
        v += dv.reshape(-1, 3)
