"""
/DT/AMS — Advanced Mass Scaling (Selective Mass Scaling) (M61, M501).

Fortran origin: ``engine/source/ams/``:
* ``sms_build_mat_2.F`` — element added mass and off-diagonal coupling assembly
* ``sms_build_diag.F``  — diagonal added mass accumulation
* ``sms_pcg.F``         — Preconditioned Conjugate Gradient (PCG) linear solver
* ``sms_mass_scale_2.F``— mass scale factor determination
* ``sms_init.F``        — active parts and initial bounds
* ``time_step/dtnodams.F`` — time step evaluation under AMS
Deck reading in ``pyradioss/input/engine_keywords.py`` (/DT/AMS, /DT/AMS/igrp).

Theory:
-------
Conventional mass scaling (/DT/NODA/CST) increases diagonal lumped masses
m_i += dm_i, which introduces artificial inertia and alters the low-frequency
and rigid-body dynamics (sum F != M_orig * a_cm).

Selective Mass Scaling (SMS / AMS) adds a non-diagonal mass perturbation
Delta M = M_diag_added + M_offdiag at the element level such that:
    sum_j Delta M_{ij} = 0   for every row i (zero row-sum property).

Consequently:
1. Rigid-Body Invariance:
   For any rigid-body translation v_rigid = [c, c, c]^T, Delta M * v_rigid = 0.
   The total linear momentum and center-of-mass kinematics are strictly preserved
   without spurious inertia forces: sum F_ext = M_orig * a_cm.
2. High-Frequency Filtering:
   High-frequency internal deformation modes have positive eigenvalues under
   Delta M, shifting the maximum eigenfrequency omega_max downward and
   permitting a larger explicit time step dt_target = dt_min / dt_scale.
3. Linear System Solve:
   Accelerations are obtained each cycle by solving:
       (M_diag + M_offdiag) * acc = F_total
   via a Preconditioned Conjugate Gradient (PCG) solver with Jacobi
   preconditioning P = 1 / M_diag.
"""

from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def dec(fn):
            return fn
        return dec

from ..common.constants import EM20
from ..model.model import Model

try:
    import scipy.sparse as sp
except ImportError:
    sp = None


class AMSManager:
    """Advanced Mass Scaling logic and state."""

    def __init__(self, model: Model, controls):
        self.model = model
        self.controls = controls
        self.dt_ams_tol = getattr(controls, "dt_ams_tol", 1e-4)
        self.dt_ams_itmax = getattr(controls, "dt_ams_itmax", 200)
        self.dt_ams_igrp = getattr(controls, "dt_ams_igrp", 0)

        # Determine AMS active parts
        n_parts = len(model.parts_list) if hasattr(model, "parts_list") else 0
        self.active_parts = np.ones(n_parts, dtype=bool)
        if self.dt_ams_igrp > 0:
            self.active_parts[:] = False
            grpart = model.egroups.get("PART", {}).get(self.dt_ams_igrp) if hasattr(model, "egroups") else None
            if grpart is not None:
                pids = getattr(grpart, "part_ids_resolved", None)
                if pids is None:
                    pids = getattr(grpart, "members", [])
                for pid in pids:
                    p = model.parts.get(pid) if hasattr(model, "parts") else None
                    if p is not None:
                        try:
                            idx = model.parts_list.index(p)
                            if idx < len(self.active_parts):
                                self.active_parts[idx] = True
                        except (ValueError, AttributeError):
                            pass

        dt_min = getattr(controls, "dt_min", 0.0)
        dt_scale = getattr(controls, "dt_scale", 0.9)
        self.dt_target = dt_min
        if dt_scale > 0.0:
            self.dt_target /= dt_scale

    def tag_nodes(self, dt_claims: List[np.ndarray]) -> np.ndarray:
        """Tag nodes that belong to active AMS elements requiring mass scaling."""
        n = self.model.numnod
        tagged = np.zeros(n, dtype=bool)
        if not hasattr(self.model, "element_groups"):
            return tagged
        for (name, group), dt_e in zip(self.model.element_groups(), dt_claims):
            if hasattr(group, "part") and group.part is not None and 0 <= group.part < len(self.active_parts):
                elem_active = self.active_parts[group.part]
            else:
                elem_active = True
            dt_arr = np.asarray(dt_e, dtype=np.float64)
            needs_ams = elem_active & (dt_arr < self.dt_target)
            if not np.any(needs_ams):
                continue
            conn = group.state.get("mass_conn", group.conn)[needs_ams]
            valid = conn[(conn >= 0) & (conn < n)]
            if len(valid) > 0:
                tagged[valid] = True
        return tagged

    def build_ams_matrix(self, dt_claims: List[np.ndarray]) -> Tuple[np.ndarray, Optional[sp.csr_matrix], float]:
        """
        Build the AMS added mass arrays.
        (sms_build_mat_2.F, sms_build_diag.F)

        Returns:
            diag_added: Array of shape (N,) containing the diagonal added mass.
            M_offdiag: CSR matrix containing the off-diagonal added mass (or None).
            max_dmels: The maximum element added mass (for logging/debug).
        """
        if sp is None:
            raise ImportError("AMS (/DT/AMS) requires scipy. Please install scipy.")

        n = self.model.numnod
        rows = []
        cols = []
        vals = []

        diag_added = np.zeros(n, dtype=np.float64)
        max_dmels = 0.0

        if not hasattr(self.model, "element_groups") or self.dt_target <= 0.0:
            return diag_added, None, max_dmels

        for (name, group), dt_e in zip(self.model.element_groups(), dt_claims):
            if hasattr(group, "part") and group.part is not None and 0 <= group.part < len(self.active_parts):
                elem_active = self.active_parts[group.part]
            else:
                elem_active = True
            dt_arr = np.asarray(dt_e, dtype=np.float64)
            needs_ams = elem_active & (dt_arr < self.dt_target)
            if not np.any(needs_ams):
                continue

            conn = group.state.get("mass_conn", group.conn)[needs_ams]
            if conn.ndim != 2 or conn.shape[0] == 0:
                continue
            mass = group.state.get("mass", np.ones(conn.shape[0]))[needs_ams]
            dt = np.maximum(dt_arr[needs_ams], 1e-30)

            # sms_mass_scale_2.F: dmels = 2.0 * mass * ((dt_target / dt)**2 - 1.0)
            ratio = self.dt_target / dt
            dmels = np.maximum(2.0 * mass * (ratio * ratio - 1.0), 0.0)
            if len(dmels) > 0:
                max_dmels = max(max_dmels, float(np.max(dmels)))

            xnod = conn.shape[1]
            if xnod <= 1:
                continue

            # Delegate to numba kernel to extract the unique node pairs and values
            r, c, v, d_add = _build_group_ams(conn, dmels, xnod, n)

            if len(r) > 0:
                rows.append(r)
                cols.append(c)
                vals.append(v)
            if len(d_add) > 0:
                diag_added[:min(n, len(d_add))] += d_add[:min(n, len(d_add))]

        if len(rows) > 0:
            rows = np.concatenate(rows)
            cols = np.concatenate(cols)
            vals = np.concatenate(vals)

            # Coalesce duplicates by converting to CSR
            M_offdiag = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        else:
            M_offdiag = None

        return diag_added, M_offdiag, max_dmels

    def solve(self, acc: np.ndarray, f: np.ndarray, M_diag: np.ndarray, M_offdiag: Optional[sp.csr_matrix]) -> Tuple[int, float]:
        """
        Solve (M_diag + M_offdiag) * acc = f.
        Updates acc in-place.
        (sms_pcg.F)

        Returns:
            iters: Number of iterations performed.
            rel_res: Final relative residual.
        """
        mask = M_diag > 0
        if M_offdiag is None or getattr(M_offdiag, "nnz", 0) == 0:
            acc.fill(0.0)
            acc[mask, 0] = f[mask, 0] / M_diag[mask]
            acc[mask, 1] = f[mask, 1] / M_diag[mask]
            acc[mask, 2] = f[mask, 2] / M_diag[mask]
            return 0, 0.0

        # Preconditioner is 1 / M_diag (since diagonal is strictly positive due to rigid-body preservation)
        precond = np.zeros_like(acc)
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
def _build_group_ams(conn: np.ndarray, dmels: np.ndarray, xnod: int, n_nodes: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract off-diagonal entries and diagonal accumulations for a group of elements.
    (sms_build_mat_2.F lines 240-350)

    Returns:
        rows, cols, vals: 1D arrays for the COO matrix.
        diag_add: Array of shape (n_nodes,) to be added to the diagonal.
    """
    n_elems = conn.shape[0]

    # Pre-allocate for the maximum possible number of pairs
    # For each element, max pairs is xnod * (xnod - 1)
    max_entries = n_elems * xnod * (xnod - 1)

    rows = np.empty(max_entries, dtype=np.int32)
    cols = np.empty(max_entries, dtype=np.int32)
    vals = np.empty(max_entries, dtype=np.float64)

    size_diag = n_nodes if n_nodes > 0 else (np.max(conn) + 1 if n_elems > 0 else 0)
    diag_add = np.zeros(size_diag, dtype=np.float64)

    idx = 0
    for e in range(n_elems):
        c = conn[e]

        # Find unique non-negative nodes and count multiplicity (KMULT)
        unq = np.empty(xnod, dtype=np.int32)
        counts = np.zeros(xnod, dtype=np.int32)
        n_unq = 0

        for i in range(xnod):
            node = c[i]
            if node < 0:
                continue
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

        if n_unq <= 1:
            continue

        kmult = 1
        for j in range(n_unq):
            if counts[j] > kmult:
                kmult = counts[j]

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
            if n1 < size_diag:
                diag_add[n1] += (n_unq - 1) * mele12

    return rows[:idx], cols[:idx], vals[:idx], diag_add


@njit(cache=True)
def ams_pcg(x: np.ndarray, b: np.ndarray, M_diag: np.ndarray,
            M_offdiag_val: np.ndarray, M_offdiag_col: np.ndarray, M_offdiag_ptr: np.ndarray,
            precond: np.ndarray, tol: float, maxiter: int) -> Tuple[np.ndarray, int, float]:
    """
    Preconditioned Conjugate Gradient (PCG) solver for (M_diag + M_offdiag) * x = b.
    (sms_pcg.F)
    M_offdiag is provided in CSR format.

    Returns:
        x: The solution vector.
        iters: Number of iterations performed.
        rel_res: Final relative residual.
    """
    n = x.shape[0]

    b_norm2 = 0.0
    for i in range(n):
        b_norm2 += b[i, 0]*b[i, 0] + b[i, 1]*b[i, 1] + b[i, 2]*b[i, 2]

    if b_norm2 == 0.0:
        x.fill(0.0)
        return x, 0, 0.0

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

    return x, iters, float(np.sqrt(max(0.0, rz) / max(b_norm2, 1e-30)))
