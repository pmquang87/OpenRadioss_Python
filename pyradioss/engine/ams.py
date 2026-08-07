from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np
from numba import njit
from ..common.constants import EM20
from ..model.model import Model
import scipy.sparse as sp

class AMSManager:
    """Advanced Mass Scaling logic and state."""
    def __init__(self, model: Model, controls):
        self.model = model
        self.controls = controls
        self.dt_ams_tol = controls.dt_ams_tol
        self.dt_ams_itmax = controls.dt_ams_itmax
        self.dt_ams_igrp = controls.dt_ams_igrp
        
        # Determine AMS active parts
        self.active_parts = np.ones(len(model.parts_list), dtype=bool)
        if self.dt_ams_igrp > 0:
            self.active_parts[:] = False
            grpart = model.egroups.get("PART", {}).get(self.dt_ams_igrp)
            if grpart is not None:
                for pid in grpart.members:
                    p = model.parts.get(pid)
                    if p is not None:
                        try:
                            idx = model.parts_list.index(p)
                            self.active_parts[idx] = True
                        except ValueError:
                            pass
        
        self.dt_target = controls.dt_min
        if controls.dt_scale > 0.0:
            self.dt_target /= controls.dt_scale

    def build_ams_matrix(self, dt_claims: List[np.ndarray]) -> Tuple[np.ndarray, Optional[sp.csr_matrix], float]:
        """
        Build the AMS added mass arrays.
        Returns:
            diag_added: Array of shape (N,) containing the diagonal added mass.
            M_offdiag: CSR matrix containing the off-diagonal added mass.
            max_dmels: The maximum element added mass (for logging/debug).
        """
        n = self.model.numnod
        rows = []
        cols = []
        vals = []
        
        # Accumulate the diagonal added mass using np.add.at
        diag_added = np.zeros(n, dtype=np.float64)
        max_dmels = 0.0
        
        for (name, group), dt_e in zip(self.model.element_groups(), dt_claims):
            elem_active = self.active_parts[group.part]
            needs_ams = elem_active & (dt_e < self.dt_target)
            if not np.any(needs_ams):
                continue
                
            conn = group.state.get("mass_conn", group.conn)[needs_ams]
            mass = group.state["mass"][needs_ams]
            dt = dt_e[needs_ams]
            
            dmels = 2.0 * mass * ((self.dt_target / dt)**2 - 1.0)
            if len(dmels) > 0:
                max_dmels = max(max_dmels, float(np.max(dmels)))
            
            xnod = conn.shape[1]
            
            # Delegate to numba kernel to extract the unique node pairs and values
            r, c, v, d_add = _build_group_ams(conn, dmels, xnod)
            
            rows.append(r)
            cols.append(c)
            vals.append(v)
            np.add.at(diag_added, np.arange(n), d_add) # d_add is shape (n,)
            
        if len(rows) > 0:
            rows = np.concatenate(rows)
            cols = np.concatenate(cols)
            vals = np.concatenate(vals)
            
            # Coalesce duplicates by converting to CSR
            M_offdiag = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        else:
            M_offdiag = None
            
        return diag_added, M_offdiag, max_dmels

    def solve(self, acc: np.ndarray, f: np.ndarray, M_diag: np.ndarray, M_offdiag: sp.csr_matrix) -> Tuple[int, float]:
        """
        Solve (M_diag + M_offdiag) * acc = f.
        Updates acc in-place.
        """
        # Preconditioner is 1 / M_diag (since diagonal is strictly positive due to rigid-body preservation)
        precond = np.zeros_like(acc)
        mask = M_diag > 0
        precond[mask, 0] = 1.0 / M_diag[mask]
        precond[mask, 1] = 1.0 / M_diag[mask]
        precond[mask, 2] = 1.0 / M_diag[mask]
        
        _, iters, rel_res = ams_pcg(
            acc, f, M_diag, 
            M_offdiag.data, M_offdiag.indices, M_offdiag.indptr,
            precond, self.dt_ams_tol, self.dt_ams_itmax
        )
        return iters, rel_res

@njit(cache=True)
def _build_group_ams(conn: np.ndarray, dmels: np.ndarray, xnod: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract off-diagonal entries and diagonal accumulations for a group of elements.
    Returns:
        rows, cols, vals: 1D arrays for the COO matrix.
        diag_add: Array of shape (max_node_id + 1,) to be added to the diagonal.
    """
    n_elems = conn.shape[0]
    
    # Pre-allocate for the maximum possible number of pairs
    # For each element, max pairs is xnod * (xnod - 1) / 2
    # And we add both (i, j) and (j, i) so xnod * (xnod - 1)
    max_entries = n_elems * xnod * (xnod - 1)
    
    rows = np.empty(max_entries, dtype=np.int32)
    cols = np.empty(max_entries, dtype=np.int32)
    vals = np.empty(max_entries, dtype=np.float64)
    
    max_node = np.max(conn)
    diag_add = np.zeros(max_node + 1, dtype=np.float64)
    
    idx = 0
    for e in range(n_elems):
        c = conn[e]
        
        # Find unique nodes and max multiplicity (KMULT)
        # Since xnod is small (e.g. <= 8), we can just do a simple nested loop
        unq = np.empty(xnod, dtype=np.int32)
        counts = np.zeros(xnod, dtype=np.int32)
        n_unq = 0
        
        for i in range(xnod):
            node = c[i]
            found = False
            for j in range(n_unq):
                if unq[j] == node:
                    counts[j] += 1
                    found = True
                    break
            if not found:
                unq[n_unq] = node
                counts[n_unq] = 1
                n_unq += 1
                
        kmult = np.max(counts[:n_unq])
        mele12 = (kmult / xnod) * 0.5 * dmels[e]
        
        # Add off-diagonal entries for every distinct pair
        for i in range(n_unq):
            n1 = unq[i]
            for j in range(i + 1, n_unq):
                n2 = unq[j]
                
                rows[idx] = n1
                cols[idx] = n2
                vals[idx] = -mele12
                idx += 1
                
                rows[idx] = n2
                cols[idx] = n1
                vals[idx] = -mele12
                idx += 1
                
            # Diagonal gets the positive sum of off-diagonals for this node
            # Node n1 connects to (n_unq - 1) other distinct nodes
            diag_add[n1] += (n_unq - 1) * mele12
            
    return rows[:idx], cols[:idx], vals[:idx], diag_add

@njit(cache=True)
def ams_pcg(x: np.ndarray, b: np.ndarray, M_diag: np.ndarray, 
            M_offdiag_val: np.ndarray, M_offdiag_col: np.ndarray, M_offdiag_ptr: np.ndarray, 
            precond: np.ndarray, tol: float, maxiter: int) -> Tuple[np.ndarray, int, float]:
    """
    Preconditioned Conjugate Gradient (PCG) solver for (M_diag + M_offdiag) * x = b.
    M_offdiag is provided in CSR format.
    
    Returns:
        x: The solution vector.
        iters: Number of iterations performed.
        rel_res: Final relative residual.
    """
    n = x.shape[0]
    
    # Initial matrix-vector product: q = A * x
    q = np.empty_like(x)
    for i in range(n):
        row_start = M_offdiag_ptr[i]
        row_end = M_offdiag_ptr[i+1]
        
        # Diagonal part
        val0 = M_diag[i] * x[i, 0]
        val1 = M_diag[i] * x[i, 1]
        val2 = M_diag[i] * x[i, 2]
        
        # Off-diagonal part
        for j in range(row_start, row_end):
            col = M_offdiag_col[j]
            v = M_offdiag_val[j]
            val0 += v * x[col, 0]
            val1 += v * x[col, 1]
            val2 += v * x[col, 2]
            
        q[i, 0] = val0
        q[i, 1] = val1
        q[i, 2] = val2

    # Initial residual r = b - A*x
    r = b - q
    
    # Preconditioned residual z = M^-1 * r
    z = r * precond
    
    # Initial search direction p = z
    p = z.copy()
    
    # Initial residual norm squared: rz = r^T * z
    rz = 0.0
    for i in range(n):
        rz += r[i, 0]*z[i, 0] + r[i, 1]*z[i, 1] + r[i, 2]*z[i, 2]
        
    b_norm2 = 0.0
    for i in range(n):
        b_norm2 += b[i, 0]*b[i, 0] + b[i, 1]*b[i, 1] + b[i, 2]*b[i, 2]
        
    if b_norm2 == 0.0:
        return x, 0, 0.0
        
    rel_tol2 = tol * tol * b_norm2
    
    iters = 0
    while iters < maxiter and rz > rel_tol2:
        # q = A * p
        for i in range(n):
            row_start = M_offdiag_ptr[i]
            row_end = M_offdiag_ptr[i+1]
            
            val0 = M_diag[i] * p[i, 0]
            val1 = M_diag[i] * p[i, 1]
            val2 = M_diag[i] * p[i, 2]
            
            for j in range(row_start, row_end):
                col = M_offdiag_col[j]
                v = M_offdiag_val[j]
                val0 += v * p[col, 0]
                val1 += v * p[col, 1]
                val2 += v * p[col, 2]
                
            q[i, 0] = val0
            q[i, 1] = val1
            q[i, 2] = val2
            
        # alpha = rz / (p^T * q)
        pq = 0.0
        for i in range(n):
            pq += p[i, 0]*q[i, 0] + p[i, 1]*q[i, 1] + p[i, 2]*q[i, 2]
            
        alpha = rz / max(pq, EM20)
        
        # x = x + alpha * p
        # r = r - alpha * q
        for i in range(n):
            x[i, 0] += alpha * p[i, 0]
            x[i, 1] += alpha * p[i, 1]
            x[i, 2] += alpha * p[i, 2]
            
            r[i, 0] -= alpha * q[i, 0]
            r[i, 1] -= alpha * q[i, 1]
            r[i, 2] -= alpha * q[i, 2]
            
        # z = M^-1 * r
        for i in range(n):
            z[i, 0] = r[i, 0] * precond[i, 0]
            z[i, 1] = r[i, 1] * precond[i, 1]
            z[i, 2] = r[i, 2] * precond[i, 2]
            
        # rz_new = r^T * z
        rz_new = 0.0
        for i in range(n):
            rz_new += r[i, 0]*z[i, 0] + r[i, 1]*z[i, 1] + r[i, 2]*z[i, 2]
            
        beta = rz_new / max(rz, EM20)
        
        # p = z + beta * p
        for i in range(n):
            p[i, 0] = z[i, 0] + beta * p[i, 0]
            p[i, 1] = z[i, 1] + beta * p[i, 1]
            p[i, 2] = z[i, 2] + beta * p[i, 2]
            
        rz = rz_new
        iters += 1
        
    return x, iters, np.sqrt(rz / b_norm2)
