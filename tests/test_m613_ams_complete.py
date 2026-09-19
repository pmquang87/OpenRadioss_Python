"""
Milestone M613: Comprehensive AMS (Advanced Mass Scaling) Fortran Parity Suite.
Tests:
1. Element coupling dispatch for all element families (Hexa8, Tetra4, Penta6, Pyra5, Tetra10, Quad, Tri, 1D).
2. Zero row-sum invariance across all element families.
3. Rigid body internal coupling filtering (sms_build_mat_2.F:1413).
4. Boundary condition projection in ams_pcg (sms_bcs.F).
5. Rigid body master-slave condensation in ams_pcg (sms_rbe2.F).
6. Generalized AMS kinetic energy and momentum conservation (sms_encin_2.F, ecrit.F).
"""

import numpy as np
import pytest
import scipy.sparse as sp

from pyradioss.engine.ams import (
    AMSManager, _build_group_ams, ams_pcg, name_to_ityp
)
from pyradioss.model.model import Model


class DummyControls:
    def __init__(self, dt_min=1e-5, dt_scale=0.9, dt_ams_tol=1e-6, dt_ams_itmax=300, dt_ams_igrp=0):
        self.dt_min = dt_min
        self.dt_scale = dt_scale
        self.dt_ams = True
        self.dt_ams_tol = dt_ams_tol
        self.dt_ams_itmax = dt_ams_itmax
        self.dt_ams_igrp = dt_ams_igrp
        self.dt_noda = "NODA"


# ==============================================================================
# 1. Element Coupling Factors & Zero Row-Sum Invariance (sms_build_mat_2.F)
# ==============================================================================

def test_hexa8_coupling_factor_and_zero_sum():
    """Hexa8: mele12 = dmels / 16 (sms_build_mat_2.F:315)."""
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
    dmels = np.array([32.0], dtype=np.float64)
    rows, cols, vals, diag_add = _build_group_ams(conn, dmels, xnod=8, n_nodes=8, ityp=1)

    expected_mele12 = 32.0 / 16.0  # 2.0
    assert np.allclose(vals, -expected_mele12)
    # Each node connected to 7 other nodes -> diag_add = 7 * 2.0 = 14.0
    assert np.allclose(diag_add, 7 * expected_mele12)

    # Row sum verification
    delta_M = np.zeros((8, 8), dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        delta_M[r, c] += v
    for i in range(8):
        delta_M[i, i] += diag_add[i]
    np.testing.assert_allclose(delta_M.sum(axis=1), 0.0, atol=1e-14)


def test_tetra4_coupling_factor_and_zero_sum():
    """Tetra4: mele12 = dmels / 8 (sms_build_mat_2.F:245)."""
    conn = np.array([[0, 1, 2, 3]], dtype=np.int32)
    dmels = np.array([16.0], dtype=np.float64)
    rows, cols, vals, diag_add = _build_group_ams(conn, dmels, xnod=4, n_nodes=4, ityp=1)

    expected_mele12 = 16.0 / 8.0  # 2.0
    assert np.allclose(vals, -expected_mele12)
    assert np.allclose(diag_add, 3 * expected_mele12)

    delta_M = np.zeros((4, 4), dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        delta_M[r, c] += v
    for i in range(4):
        delta_M[i, i] += diag_add[i]
    np.testing.assert_allclose(delta_M.sum(axis=1), 0.0, atol=1e-14)


def test_penta6_coupling_factor_and_zero_sum():
    """Penta6: mele12 = dmels / 12 (sms_build_mat_2.F:276)."""
    conn = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int32)
    dmels = np.array([24.0], dtype=np.float64)
    rows, cols, vals, diag_add = _build_group_ams(conn, dmels, xnod=6, n_nodes=6, ityp=1)

    expected_mele12 = 24.0 / 12.0  # 2.0
    assert np.allclose(vals, -expected_mele12)
    assert np.allclose(diag_add, 5 * expected_mele12)

    delta_M = np.zeros((6, 6), dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        delta_M[r, c] += v
    for i in range(6):
        delta_M[i, i] += diag_add[i]
    np.testing.assert_allclose(delta_M.sum(axis=1), 0.0, atol=1e-14)


def test_quad_shell_coupling_factor_and_zero_sum():
    """Quad shell (BT4, QEPH, QBAT): mele12 = dmelc / 6 (sms_build_mat_2.F:642)."""
    conn = np.array([[0, 1, 2, 3]], dtype=np.int32)
    dmels = np.array([18.0], dtype=np.float64)
    rows, cols, vals, diag_add = _build_group_ams(conn, dmels, xnod=4, n_nodes=4, ityp=3)

    expected_mele12 = 18.0 / 6.0  # 3.0
    assert np.allclose(vals, -expected_mele12)
    assert np.allclose(diag_add, 3 * expected_mele12)

    delta_M = np.zeros((4, 4), dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        delta_M[r, c] += v
    for i in range(4):
        delta_M[i, i] += diag_add[i]
    np.testing.assert_allclose(delta_M.sum(axis=1), 0.0, atol=1e-14)


def test_tri_shell_coupling_factor_and_zero_sum():
    """Tri shell (Tri3, DKT18): mele12 = dmeltg / 6 (sms_build_mat_2.F:797)."""
    conn = np.array([[0, 1, 2]], dtype=np.int32)
    dmels = np.array([12.0], dtype=np.float64)
    rows, cols, vals, diag_add = _build_group_ams(conn, dmels, xnod=3, n_nodes=3, ityp=7)

    expected_mele12 = 12.0 / 6.0  # 2.0
    assert np.allclose(vals, -expected_mele12)
    assert np.allclose(diag_add, 2 * expected_mele12)

    delta_M = np.zeros((3, 3), dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        delta_M[r, c] += v
    for i in range(3):
        delta_M[i, i] += diag_add[i]
    np.testing.assert_allclose(delta_M.sum(axis=1), 0.0, atol=1e-14)


def test_1d_element_coupling_factor_and_zero_sum():
    """1D elements (Truss, Beam, Spring): mele12 = 0.5 * dmels (sms_build_mat_2.F:667, 691)."""
    conn = np.array([[0, 1]], dtype=np.int32)
    dmels = np.array([10.0], dtype=np.float64)
    rows, cols, vals, diag_add = _build_group_ams(conn, dmels, xnod=2, n_nodes=2, ityp=5)

    expected_mele12 = 0.5 * 10.0  # 5.0
    assert np.allclose(vals, -expected_mele12)
    assert np.allclose(diag_add, expected_mele12)

    delta_M = np.zeros((2, 2), dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        delta_M[r, c] += v
    for i in range(2):
        delta_M[i, i] += diag_add[i]
    np.testing.assert_allclose(delta_M.sum(axis=1), 0.0, atol=1e-14)


def test_name_to_ityp_mapping():
    """Verify mapping from element group names to OpenRadioss ITY codes."""
    assert name_to_ityp("bricks") == 1
    assert name_to_ityp("tetras") == 1
    assert name_to_ityp("shells") == 3
    assert name_to_ityp("shells_qeph") == 3
    assert name_to_ityp("shells_qbat") == 3
    assert name_to_ityp("sh3n") == 7
    assert name_to_ityp("sh3n_dkt18") == 7
    assert name_to_ityp("trusses") == 4
    assert name_to_ityp("beams") == 5
    assert name_to_ityp("springs") == 6


# ==============================================================================
# 2. Rigid Body Filtering in AMS Assembly (sms_build_mat_2.F:1413)
# ==============================================================================

def test_rigid_body_same_body_filtering():
    """
    Verify that if two nodes share the same rigid body ID, their off-diagonal
    coupling is skipped, maintaining zero row-sum among uncoupled components.
    """
    conn = np.array([[0, 1, 2, 3]], dtype=np.int32)
    dmels = np.array([24.0], dtype=np.float64)

    # Nodes 0 and 1 belong to rigid body 1; nodes 2 and 3 are free (0)
    tagslv_rby = np.array([1, 1, 0, 0], dtype=np.int32)

    rows, cols, vals, diag_add = _build_group_ams(conn, dmels, xnod=4, n_nodes=4, ityp=3, tagslv_rby=tagslv_rby)

    # Check that pair (0, 1) and (1, 0) is absent
    for r, c in zip(rows, cols):
        assert not (r == 0 and c == 1)
        assert not (r == 1 and c == 0)

    # But pairs (0, 2), (0, 3), (1, 2), (1, 3), (2, 3) must be present
    pairs = set(zip(rows, cols))
    assert (0, 2) in pairs
    assert (0, 3) in pairs
    assert (1, 2) in pairs
    assert (1, 3) in pairs
    assert (2, 3) in pairs

    # Zero row sum must still hold exactly
    delta_M = np.zeros((4, 4), dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        delta_M[r, c] += v
    for i in range(4):
        delta_M[i, i] += diag_add[i]
    np.testing.assert_allclose(delta_M.sum(axis=1), 0.0, atol=1e-14)


# ==============================================================================
# 3. Boundary Condition Projection in PCG (sms_bcs.F, sms_pcg.F)
# ==============================================================================

def test_pcg_bcs_projection():
    """Verify that constrained DOFs maintain strictly zero acceleration under PCG solve."""
    n = 4
    # Symmetric positive-definite coupled system
    A_off = np.array([
        [ 0.0, -2.0, -1.0,  0.0],
        [-2.0,  0.0, -3.0, -1.0],
        [-1.0, -3.0,  0.0, -2.0],
        [ 0.0, -1.0, -2.0,  0.0]
    ], dtype=np.float64)
    M_diag = np.array([10.0, 12.0, 15.0, 8.0], dtype=np.float64) - A_off.sum(axis=1)
    csr_off = sp.csr_matrix(A_off)

    precond = np.zeros((n, 3), dtype=np.float64)
    for c in range(3):
        precond[:, c] = 1.0 / M_diag

    # Large applied forces on all nodes and all DOFs
    f = np.full((n, 3), 100.0, dtype=np.float64)

    # Node 0 fixed in X, Y, Z; Node 2 fixed in Y
    fix_tra = np.zeros((n, 3), dtype=bool)
    fix_tra[0, :] = True
    fix_tra[2, 1] = True

    acc = np.zeros((n, 3), dtype=np.float64)
    sol, iters, rel_res = ams_pcg(
        acc, f, M_diag,
        csr_off.data, csr_off.indices, csr_off.indptr,
        precond, tol=1e-8, maxiter=200,
        fix_tra=fix_tra
    )

    # Node 0 accelerations must be exactly 0
    assert np.allclose(sol[0, :], 0.0, atol=1e-15)
    # Node 2 Y-acceleration must be exactly 0
    assert np.isclose(sol[2, 1], 0.0, atol=1e-15)
    # Unconstrained nodes must have non-zero acceleration
    assert np.all(np.abs(sol[1, :]) > 1.0)
    assert np.abs(sol[2, 0]) > 1.0
    assert np.abs(sol[2, 2]) > 1.0
    assert np.all(np.abs(sol[3, :]) > 1.0)


# ==============================================================================
# 4. Rigid Body Master-Slave Condensation in PCG (sms_rbe2.F)
# ==============================================================================

def test_pcg_rigid_body_condensation():
    """Verify that slave nodes mirror master acceleration under PCG solve."""
    n = 4
    A_off = np.array([
        [ 0.0, -1.0, -1.0, -1.0],
        [-1.0,  0.0, -1.0, -1.0],
        [-1.0, -1.0,  0.0, -1.0],
        [-1.0, -1.0, -1.0,  0.0]
    ], dtype=np.float64)
    M_diag = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64) - A_off.sum(axis=1)
    csr_off = sp.csr_matrix(A_off)

    precond = np.zeros((n, 3), dtype=np.float64)
    for c in range(3):
        precond[:, c] = 1.0 / M_diag

    f = np.array([
        [10.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
        [30.0, 0.0, 0.0],
        [40.0, 0.0, 0.0]
    ], dtype=np.float64)

    # Master: node 0, Slaves: node 1 and node 2
    rb_masters = np.array([0, 0], dtype=np.int32)
    rb_slaves = np.array([1, 2], dtype=np.int32)

    acc = np.zeros((n, 3), dtype=np.float64)
    sol, iters, rel_res = ams_pcg(
        acc, f, M_diag,
        csr_off.data, csr_off.indices, csr_off.indptr,
        precond, tol=1e-8, maxiter=200,
        rb_masters=rb_masters, rb_slaves=rb_slaves
    )

    # Slaves 1 and 2 must have identical acceleration to master 0
    np.testing.assert_allclose(sol[1, :], sol[0, :], atol=1e-12)
    np.testing.assert_allclose(sol[2, :], sol[0, :], atol=1e-12)


# ==============================================================================
# 5. Generalized Kinetic Energy & Momentum Conservation (sms_encin_2.F)
# ==============================================================================

def test_ams_kinetic_energy_and_momentum_conservation():
    """
    Verify that compute_kinetic_energy computes both E_k_ams and physical E_k,
    and strictly preserves total linear momentum for rigid motion.
    """
    n = 3
    conn = np.array([[0, 1, 2]], dtype=np.int32)
    dmels = np.array([12.0], dtype=np.float64)
    rows, cols, vals, diag_add = _build_group_ams(conn, dmels, xnod=3, n_nodes=n, ityp=7)

    M_offdiag = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    mass_orig = np.array([2.0, 3.0, 5.0], dtype=np.float64)
    M_diag = mass_orig + diag_add

    model = Model()
    model.numnod = n
    controls = DummyControls()
    manager = AMSManager(model, controls)

    # Rigid translation velocity v = (4.0, -2.0, 1.0)
    v_rigid_vec = np.array([4.0, -2.0, 1.0], dtype=np.float64)
    v = np.tile(v_rigid_vec, (n, 1))
    a = np.zeros((n, 3), dtype=np.float64)
    dt12 = 0.001

    e_ams, e_phys, p_tot = manager.compute_kinetic_energy(v, a, dt12, mass_orig, M_diag, M_offdiag)

    # For rigid translation, Delta M * v == 0, so E_ams MUST exactly equal E_phys!
    expected_e_phys = 0.5 * np.sum(mass_orig) * np.sum(v_rigid_vec ** 2)
    assert np.isclose(e_phys, expected_e_phys, rtol=1e-12)
    assert np.isclose(e_ams, expected_e_phys, rtol=1e-12)

    # Momentum must be M_total * v_rigid
    expected_p = np.sum(mass_orig) * v_rigid_vec
    np.testing.assert_allclose(p_tot, expected_p, atol=1e-12)
