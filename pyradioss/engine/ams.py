"""
/DT/AMS — Advanced Mass Scaling (Selective Mass Scaling) (M61, M501, M613).

Fortran origin: ``engine/source/ams/``:
* ``sms_build_mat_2.F`` — element added mass and off-diagonal coupling assembly
* ``sms_build_diag.F``  — diagonal added mass accumulation
* ``sms_pcg.F``         — Preconditioned Conjugate Gradient (PCG) linear solver
* ``sms_mass_scale_2.F``— mass scale factor determination
* ``sms_encin_2.F``     — kinetic energy calculation under AMS
* ``sms_bcs.F``         — boundary condition projection
* ``sms_fixvel.F``      — prescribed velocity projection
* ``sms_rbe2.F``        — rigid body & RBE2 remontee / descente
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
   preconditioning P = 1 / M_diag, Krylov subspace projection of boundary
   conditions, and rigid body condensation.
"""

from __future__ import annotations
from typing import Tuple, List, Optional, Dict, Any
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


def name_to_ityp(name: str) -> int:
    """Map pyradioss element group name to OpenRadioss ITY type code."""
    if name in ("shells", "shells_qbat", "shells_qeph", "quads", "quads_full", "shel16s"):
        return 3   # 2D 4-node quad shell
    elif name in ("sh3n", "sh3n_dkt18", "shells_dkt6", "trias"):
        return 7   # 2D 3-node tri shell
    elif name == "trusses":
        return 4   # 1D truss
    elif name in ("beams", "beams_fiber"):
        return 5   # 1D beam
    elif name == "springs":
        return 6   # 1D spring
    else:
        return 1   # 3D solid / thick shell


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

    def _get_elem_active(self, group, n_elem: int) -> np.ndarray:
        if hasattr(group, "part") and group.part is not None:
            p_arr = np.asarray(group.part)
            if p_arr.ndim == 0:
                val = int(p_arr)
                active = bool(self.active_parts[val]) if 0 <= val < len(self.active_parts) else True
                return np.full(n_elem, active, dtype=bool)
            else:
                valid_idx = (p_arr >= 0) & (p_arr < len(self.active_parts))
                active = np.ones(len(p_arr), dtype=bool)
                active[valid_idx] = self.active_parts[p_arr[valid_idx]]
                return active
        return np.ones(n_elem, dtype=bool)

    def tag_nodes(self, dt_claims: List[np.ndarray]) -> np.ndarray:
        """Tag nodes that belong to active AMS elements requiring mass scaling."""
        n = self.model.numnod
        tagged = np.zeros(n, dtype=bool)
        if not hasattr(self.model, "element_groups"):
            return tagged
        for (name, group), dt_e in zip(self.model.element_groups(), dt_claims):
            dt_arr = np.asarray(dt_e, dtype=np.float64)
            elem_active = self._get_elem_active(group, len(dt_arr))
            needs_ams = elem_active & (dt_arr < self.dt_target)
            if not np.any(needs_ams):
                continue
            conn = group.state.get("mass_conn", group.conn)[needs_ams]
            valid = conn[(conn >= 0) & (conn < n)]
            if len(valid) > 0:
                tagged[valid] = True
        return tagged

    def build_ams_matrix(self, dt_claims: List[np.ndarray],
                         tagslv_rby: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[sp.csr_matrix], float]:
        """
        Build the AMS added mass arrays.
        (sms_build_mat_2.F, sms_build_diag.F)

        Parameters:
            dt_claims: Per-element time step claims.
            tagslv_rby: Optional rigid-body tag array (shape numnod) where non-zero
                        identifies rigid body ID. Nodes in the same rigid body have
                        internal coupling filtered out (sms_build_mat_2.F:1413).

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

        tag_arr = np.zeros(0, dtype=np.int32)
        if tagslv_rby is not None and len(tagslv_rby) > 0:
            tag_arr = np.asarray(tagslv_rby, dtype=np.int32)

        for (name, group), dt_e in zip(self.model.element_groups(), dt_claims):
            dt_arr = np.asarray(dt_e, dtype=np.float64)
            elem_active = self._get_elem_active(group, len(dt_arr))
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

            ityp = name_to_ityp(name)

            # Delegate to numba kernel to extract the unique node pairs and values
            r, c, v, d_add = _build_group_ams(conn, dmels, xnod, n, ityp, tag_arr)

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

    def solve(self, acc: np.ndarray, f: np.ndarray, M_diag: np.ndarray,
              M_offdiag: Optional[sp.csr_matrix],
              fix_tra: Optional[np.ndarray] = None,
              rb_masters: Optional[np.ndarray] = None,
              rb_slaves: Optional[np.ndarray] = None) -> Tuple[int, float]:
        """
        Solve (M_diag + M_offdiag) * acc = f.
        Updates acc in-place.
        (sms_pcg.F, sms_bcs.F, sms_rbe2.F)

        Parameters:
            acc: Acceleration vector (N, 3) to be updated in-place.
            f: Total force vector (N, 3).
            M_diag: Diagonal mass vector (N,).
            M_offdiag: CSR off-diagonal added mass matrix (N, N).
            fix_tra: Optional boolean mask (N, 3) indicating constrained translational DOFs.
            rb_masters: Optional int array of master node indices for rigid body slaves.
            rb_slaves: Optional int array of slave node indices for rigid bodies.

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
            if fix_tra is not None and fix_tra.shape[0] > 0:
                _project_bcs(acc, fix_tra)
            return 0, 0.0

        # Preconditioner is 1 / M_diag
        M_diag_eff = np.where(M_diag > 1e-20, M_diag, 1.0)
        precond = np.zeros_like(acc)
        inv_m = 1.0 / M_diag_eff
        precond[:, 0] = inv_m
        precond[:, 1] = inv_m
        precond[:, 2] = inv_m

        fix_arr = np.zeros((0, 3), dtype=np.bool_)
        if fix_tra is not None and fix_tra.shape[0] > 0:
            fix_arr = np.asarray(fix_tra, dtype=np.bool_)

        rb_m_arr = np.zeros(0, dtype=np.int32)
        rb_s_arr = np.zeros(0, dtype=np.int32)
        if rb_masters is not None and rb_slaves is not None and len(rb_masters) > 0:
            rb_m_arr = np.asarray(rb_masters, dtype=np.int32)
            rb_s_arr = np.asarray(rb_slaves, dtype=np.int32)

        _, iters, rel_res = ams_pcg(
            acc, f, M_diag,
            M_offdiag.data, M_offdiag.indices, M_offdiag.indptr,
            precond, self.dt_ams_tol, self.dt_ams_itmax,
            fix_arr, rb_m_arr, rb_s_arr
        )
        return iters, rel_res

    def compute_kinetic_energy(self, v: np.ndarray, a: np.ndarray, dt12: float,
                               mass_orig: np.ndarray, M_diag: np.ndarray,
                               M_offdiag: Optional[sp.csr_matrix]) -> Tuple[float, float, np.ndarray]:
        """
        Compute generalized AMS kinetic energy, physical kinetic energy, and linear momentum.
        (sms_encin_2.F, ecrit.F)

        Returns:
            e_kin_ams: Generalized discrete kinetic energy 0.5 * wv^T * (M_orig + Delta_M) * wv
            e_kin_phys: Physical unscaled kinetic energy 0.5 * sum(m_orig * wv^2)
            momentum: Total linear momentum vector (3,)
        """
        wv = v + 0.5 * dt12 * a
        p_base = mass_orig[:, None] * wv

        wmv = np.zeros_like(wv)
        if M_offdiag is not None and getattr(M_offdiag, "nnz", 0) > 0:
            diag_add = np.maximum(0.0, M_diag - mass_orig)
            wmv = diag_add[:, None] * wv + M_offdiag.dot(wv)

        xmom_sms = p_base + wmv
        e_kin_ams = float(0.5 * np.sum(wv * xmom_sms))
        e_kin_phys = float(0.5 * np.sum(mass_orig[:, None] * (wv ** 2)))
        momentum = np.sum(xmom_sms, axis=0)
        return e_kin_ams, e_kin_phys, momentum


@njit(cache=True)
def _build_group_ams(conn: np.ndarray, dmels: np.ndarray, xnod: int, n_nodes: int = 0,
                     ityp: int = 0, tagslv_rby: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract off-diagonal entries and diagonal accumulations for a group of elements.
    (sms_build_mat_2.F lines 240-820)

    Parameters:
        conn: Element connectivity (n_elems, xnod)
        dmels: Element added mass (n_elems,)
        xnod: Number of nodes per element
        n_nodes: Total number of nodes in model
        ityp: OpenRadioss element type:
              1 = 3D Solid / Thick Shell (Hexa8 1/16, Tetra4 1/8, Penta6 1/12, Pyra5 0.4, Tetra10 1/640)
              3 = 2D Quad Shell (1/6)
              7 = 2D Tri Shell (1/6)
              4, 5, 6 = 1D Truss, Beam, Spring (0.5)
              0 = generic fallback: (kmult / xnod) * 0.5 * dmels
        tagslv_rby: Optional rigid-body tag array (shape n_nodes). If two nodes share the
                    same non-zero rigid body ID, their off-diagonal coupling is skipped (sms_build_mat_2.F:1413).

    Returns:
        rows, cols, vals: 1D arrays for the COO matrix.
        diag_add: Array of shape (n_nodes,) to be added to the diagonal.
    """
    n_elems = conn.shape[0]

    # Pre-allocate for the maximum possible number of pairs
    max_entries = n_elems * xnod * (xnod - 1)

    rows = np.empty(max_entries, dtype=np.int32)
    cols = np.empty(max_entries, dtype=np.int32)
    vals = np.empty(max_entries, dtype=np.float64)

    size_diag = n_nodes if n_nodes > 0 else (np.max(conn) + 1 if n_elems > 0 else 0)
    diag_add = np.zeros(size_diag, dtype=np.float64)

    has_rby = False
    if tagslv_rby is not None and tagslv_rby.shape[0] > 0:
        has_rby = True

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

        # Element-specific coupling factor mele12 (sms_build_mat_2.F)
        dm = dmels[e]
        if ityp == 1:
            # 3D Solids / Thick Shells
            if xnod == 8 and kmult == 1:
                mele12 = dm / 16.0         # Standard Hexa8 (sms_build_mat_2.F:315)
            elif xnod == 4:
                mele12 = dm / 8.0          # Tetra4 (sms_build_mat_2.F:245)
            elif xnod == 6 or (xnod == 8 and kmult == 2):
                mele12 = dm / 12.0         # Penta6 wedge (sms_build_mat_2.F:276)
            elif xnod == 5 or (xnod == 8 and kmult == 4):
                mele12 = 0.4 * dm          # Pyra5 collapsed pyramid
            elif xnod == 10:
                mele12 = dm / 640.0        # Tetra10 quadratic tetra (sms_build_mat_2.F:400)
            else:
                mele12 = (kmult / float(xnod)) * 0.5 * dm
        elif ityp == 3:
            # 2D 4-node Quad Shells (BT4, QEPH, QBAT, BATOZ) (sms_build_mat_2.F:642)
            mele12 = dm / 6.0
        elif ityp == 7:
            # 2D 3-node Tri Shells (Tri3, DKT18, COQUE 3N) (sms_build_mat_2.F:797)
            mele12 = dm / 6.0
        elif ityp == 4 or ityp == 5 or ityp == 6:
            # 1D Trusses, Beams, Springs (sms_build_mat_2.F:667, 691, 719)
            mele12 = 0.5 * dm
        else:
            # Generic fallback
            mele12 = (kmult / float(xnod)) * 0.5 * dm

        # Add off-diagonal entries for every distinct pair
        for i in range(n_unq):
            n1 = unq[i]
            for j in range(i + 1, n_unq):
                n2 = unq[j]

                # Filter out pairs belonging to the same rigid body (sms_build_mat_2.F:1413)
                if has_rby and n1 < tagslv_rby.shape[0] and n2 < tagslv_rby.shape[0]:
                    rb1 = tagslv_rby[n1]
                    rb2 = tagslv_rby[n2]
                    if rb1 != 0 and rb1 == rb2:
                        continue

                rows[idx] = n1
                cols[idx] = n2
                vals[idx] = -mele12
                idx += 1

                rows[idx] = n2
                cols[idx] = n1
                vals[idx] = -mele12
                idx += 1

                # Diagonal gets the positive sum of off-diagonals for this node pair
                if n1 < size_diag:
                    diag_add[n1] += mele12
                if n2 < size_diag:
                    diag_add[n2] += mele12

    return rows[:idx], cols[:idx], vals[:idx], diag_add


@njit(cache=True)
def _project_bcs(v: np.ndarray, fix_tra: np.ndarray):
    """Zero out constrained translational degrees of freedom (sms_bcs.F)."""
    n = v.shape[0]
    n_fix = fix_tra.shape[0]
    if n_fix == 0:
        return
    for i in range(n):
        if i < n_fix:
            if fix_tra[i, 0]:
                v[i, 0] = 0.0
            if fix_tra[i, 1]:
                v[i, 1] = 0.0
            if fix_tra[i, 2]:
                v[i, 2] = 0.0


@njit(cache=True)
def _rbody_remontee(r: np.ndarray, rb_masters: np.ndarray, rb_slaves: np.ndarray):
    """Condense slave residuals onto master nodes (sms_rbe2.F)."""
    num_slaves = rb_slaves.shape[0]
    if num_slaves == 0:
        return
    n = r.shape[0]
    for s_idx in range(num_slaves):
        s = rb_slaves[s_idx]
        m = rb_masters[s_idx]
        if s >= 0 and m >= 0 and s < n and m < n:
            r[m, 0] += r[s, 0]
            r[m, 1] += r[s, 1]
            r[m, 2] += r[s, 2]
            r[s, 0] = 0.0
            r[s, 1] = 0.0
            r[s, 2] = 0.0


@njit(cache=True)
def _rbody_descente(x: np.ndarray, rb_masters: np.ndarray, rb_slaves: np.ndarray):
    """Broadcast master vector to slave nodes (sms_rbe2.F)."""
    num_slaves = rb_slaves.shape[0]
    if num_slaves == 0:
        return
    n = x.shape[0]
    for s_idx in range(num_slaves):
        s = rb_slaves[s_idx]
        m = rb_masters[s_idx]
        if s >= 0 and m >= 0 and s < n and m < n:
            x[s, 0] = x[m, 0]
            x[s, 1] = x[m, 1]
            x[s, 2] = x[m, 2]


def ams_pcg(x: np.ndarray, b: np.ndarray, M_diag: np.ndarray,
            M_offdiag_val: np.ndarray, M_offdiag_col: np.ndarray, M_offdiag_ptr: np.ndarray,
            precond: np.ndarray, tol: float, maxiter: int,
            fix_tra: Optional[np.ndarray] = None,
            rb_masters: Optional[np.ndarray] = None,
            rb_slaves: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int, float]:
    """
    Preconditioned Conjugate Gradient (PCG) solver for (M_diag + M_offdiag) * x = b
    with Krylov subspace boundary condition projection and rigid body condensation.
    (sms_pcg.F, sms_bcs.F, sms_rbe2.F)
    """
    fix_arr = np.zeros((0, 3), dtype=np.bool_) if fix_tra is None else np.asarray(fix_tra, dtype=np.bool_)
    rb_m_arr = np.zeros(0, dtype=np.int32) if rb_masters is None else np.asarray(rb_masters, dtype=np.int32)
    rb_s_arr = np.zeros(0, dtype=np.int32) if rb_slaves is None else np.asarray(rb_slaves, dtype=np.int32)
    return _ams_pcg_jit(
        x, b, M_diag, M_offdiag_val, M_offdiag_col, M_offdiag_ptr,
        precond, tol, maxiter, fix_arr, rb_m_arr, rb_s_arr
    )


@njit(cache=True)
def _ams_pcg_jit(x: np.ndarray, b: np.ndarray, M_diag: np.ndarray,
                 M_offdiag_val: np.ndarray, M_offdiag_col: np.ndarray, M_offdiag_ptr: np.ndarray,
                 precond: np.ndarray, tol: float, maxiter: int,
                 fix_tra: np.ndarray, rb_masters: np.ndarray, rb_slaves: np.ndarray) -> Tuple[np.ndarray, int, float]:

    n = x.shape[0]

    b_norm2 = 0.0
    for i in range(n):
        b_norm2 += b[i, 0]*b[i, 0] + b[i, 1]*b[i, 1] + b[i, 2]*b[i, 2]

    if b_norm2 == 0.0:
        x.fill(0.0)
        return x, 0, 0.0

    # Project initial guess
    _project_bcs(x, fix_tra)
    _rbody_descente(x, rb_masters, rb_slaves)

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
    _project_bcs(r, fix_tra)
    _rbody_remontee(r, rb_masters, rb_slaves)

    # Preconditioned residual z = M^-1 * r
    z = np.empty_like(r)
    if precond.ndim == 1:
        for i in range(n):
            pv = precond[i] if abs(precond[i]) > 1e-20 else 1.0
            z[i, 0] = r[i, 0] * pv
            z[i, 1] = r[i, 1] * pv
            z[i, 2] = r[i, 2] * pv
    else:
        for i in range(n):
            z[i, 0] = r[i, 0] * (precond[i, 0] if abs(precond[i, 0]) > 1e-20 else 1.0)
            z[i, 1] = r[i, 1] * (precond[i, 1] if abs(precond[i, 1]) > 1e-20 else 1.0)
            z[i, 2] = r[i, 2] * (precond[i, 2] if abs(precond[i, 2]) > 1e-20 else 1.0)

    _project_bcs(z, fix_tra)
    _rbody_descente(z, rb_masters, rb_slaves)

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

        _project_bcs(q, fix_tra)

        # alpha = rz / (p^T * q)
        pq = 0.0
        for i in range(n):
            pq += p[i, 0]*q[i, 0] + p[i, 1]*q[i, 1] + p[i, 2]*q[i, 2]

        if pq <= 1e-20:
            break

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

        _project_bcs(x, fix_tra)
        _rbody_descente(x, rb_masters, rb_slaves)

        _project_bcs(r, fix_tra)
        _rbody_remontee(r, rb_masters, rb_slaves)

        # z = M^-1 * r
        if precond.ndim == 1:
            for i in range(n):
                pv = precond[i] if abs(precond[i]) > 1e-20 else 1.0
                z[i, 0] = r[i, 0] * pv
                z[i, 1] = r[i, 1] * pv
                z[i, 2] = r[i, 2] * pv
        else:
            for i in range(n):
                z[i, 0] = r[i, 0] * (precond[i, 0] if abs(precond[i, 0]) > 1e-20 else 1.0)
                z[i, 1] = r[i, 1] * (precond[i, 1] if abs(precond[i, 1]) > 1e-20 else 1.0)
                z[i, 2] = r[i, 2] * (precond[i, 2] if abs(precond[i, 2]) > 1e-20 else 1.0)

        _project_bcs(z, fix_tra)
        _rbody_descente(z, rb_masters, rb_slaves)

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

        _project_bcs(p, fix_tra)
        _rbody_descente(p, rb_masters, rb_slaves)

        rz = rz_new
        iters += 1

    return x, iters, float(np.sqrt(max(0.0, rz) / max(b_norm2, 1e-30)))
