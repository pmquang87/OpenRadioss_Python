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
            else:
                log.warning(f"/INTER/LAGMUL/TYPE{itf.type} Engine constraint builder not implemented.", "LAGMUL")
        
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
