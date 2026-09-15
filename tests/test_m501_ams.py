"""
Milestone M501: /DT/AMS Advanced Mass Scaling (Selective Mass Scaling)
Hardening, PCG Solver, Row-Sum Zero Invariance & Comprehensive Unit Tests.

Fortran origin: ``engine/source/ams/``:
* ``sms_build_mat_2.F`` — element added mass and off-diagonal coupling assembly
* ``sms_build_diag.F``  — diagonal added mass accumulation
* ``sms_pcg.F``         — Preconditioned Conjugate Gradient (PCG) linear solver
* ``sms_mass_scale_2.F``— mass scale factor determination
* ``sms_init.F``        — active parts and initial bounds
* ``time_step/dtnodams.F`` — time step evaluation under AMS
Deck reading in ``pyradioss/input/engine_keywords.py`` (/DT/AMS, /DT/AMS/igrp).
"""

from pathlib import Path
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.linalg import eigh

from pyradioss.engine.ams import AMSManager, _build_group_ams, ams_pcg
from pyradioss.model.model import Model
from pyradioss.engine.mass_scaling import NodalTimeStep
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.engine_keywords import parse_engine_deck


class DummyControls:
    def __init__(self, dt_min=1e-5, dt_scale=0.9, dt_ams_tol=1e-4, dt_ams_itmax=200, dt_ams_igrp=0):
        self.dt_min = dt_min
        self.dt_scale = dt_scale
        self.dt_ams = True
        self.dt_ams_tol = dt_ams_tol
        self.dt_ams_itmax = dt_ams_itmax
        self.dt_ams_igrp = dt_ams_igrp
        self.dt_noda = "NODA"


class DummyElementGroup:
    def __init__(self, conn, mass, part=0):
        self.conn = np.asarray(conn, dtype=np.int32)
        self.part = part
        self.n = len(self.conn)
        self.state = {
            "mass": np.asarray(mass, dtype=np.float64),
            "mass_conn": self.conn
        }


# ==============================================================================
# 1. Zero Row-Sum Invariance & Element-Level Coupling
# ==============================================================================

def test_single_element_zero_row_sum():
    """Verify that every row sum of Delta M for a single 4-node element is 0."""
    conn = np.array([[0, 1, 2, 3]], dtype=np.int32)
    dmels = np.array([12.0], dtype=np.float64)
    xnod = 4
    n_nodes = 4

    rows, cols, vals, diag_add = _build_group_ams(conn, dmels, xnod, n_nodes)

    # Reconstruct dense Delta M = M_offdiag + diag(diag_add)
    delta_M = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        delta_M[r, c] += v
    for i in range(n_nodes):
        delta_M[i, i] += diag_add[i]

    # Check that each row sums to zero to machine precision
    row_sums = delta_M.sum(axis=1)
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-14)
    # Check that diagonal is strictly positive
    assert np.all(diag_add > 0.0)
    # Check that off-diagonals are non-positive
    assert np.all(vals <= 0.0)


def test_8node_brick_zero_row_sum():
    """Verify that every row sum of Delta M for an 8-node brick element is 0."""
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
    dmels = np.array([40.0], dtype=np.float64)
    xnod = 8
    n_nodes = 8

    rows, cols, vals, diag_add = _build_group_ams(conn, dmels, xnod, n_nodes)

    delta_M = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        delta_M[r, c] += v
    for i in range(n_nodes):
        delta_M[i, i] += diag_add[i]

    row_sums = delta_M.sum(axis=1)
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-14)


def test_assembled_mesh_zero_row_sum():
    """Verify that multi-element assembled Delta M satisfies sum_j Delta M_{ij} = 0."""
    # 2 quads sharing nodes 1 and 2:
    # Quad 0: [0, 1, 2, 3], Quad 1: [1, 4, 5, 2]
    conn = np.array([
        [0, 1, 2, 3],
        [1, 4, 5, 2]
    ], dtype=np.int32)
    dmels = np.array([10.0, 20.0], dtype=np.float64)
    xnod = 4
    n_nodes = 6

    rows, cols, vals, diag_add = _build_group_ams(conn, dmels, xnod, n_nodes)

    M_offdiag = sp.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    off_row_sums = np.asarray(M_offdiag.sum(axis=1)).ravel()

    total_row_sums = off_row_sums + diag_add
    np.testing.assert_allclose(total_row_sums, 0.0, atol=1e-14)


# ==============================================================================
# 2. Rigid-Body Translational Invariance & Momentum Conservation
# ==============================================================================

def test_rigid_body_translational_invariance():
    """
    Verify that (M_diag + M_offdiag) * v_rigid == M_diag * v_rigid.
    Rigid body motion experiences zero net inertial perturbation from AMS.
    """
    conn = np.array([
        [0, 1, 2, 3],
        [1, 4, 5, 2],
        [3, 2, 5, 6]
    ], dtype=np.int32)
    dmels = np.array([8.0, 12.0, 16.0], dtype=np.float64)
    n_nodes = 7

    rows, cols, vals, diag_add = _build_group_ams(conn, dmels, 4, n_nodes)
    M_offdiag = sp.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()

    # Original diagonal masses
    m_orig = np.full(n_nodes, 5.0, dtype=np.float64)
    M_diag = m_orig + diag_add

    # Arbitrary 3D rigid translation
    v_rigid_vec = np.array([3.5, -2.1, 7.8], dtype=np.float64)
    v_rigid = np.tile(v_rigid_vec, (n_nodes, 1))  # (n_nodes, 3)

    # 1. Original momentum: M_orig * v_rigid
    orig_f = m_orig[:, None] * v_rigid

    # 2. AMS momentum: (M_diag + M_offdiag) * v_rigid
    ams_f = M_diag[:, None] * v_rigid + M_offdiag.dot(v_rigid)

    # Must be identical to machine precision!
    np.testing.assert_allclose(ams_f, orig_f, atol=1e-13)

    # Sum of nodal forces across the structure:
    # sum_i F_ams = sum_i F_orig = M_total * a_cm
    assert abs(np.sum(ams_f) - np.sum(orig_f)) < 1e-12


def test_total_mass_conservation():
    """Verify that total structural mass is completely unaffected by AMS."""
    conn = np.array([[0, 1, 2, 3]], dtype=np.int32)
    dmels = np.array([10.0], dtype=np.float64)
    rows, cols, vals, diag_add = _build_group_ams(conn, dmels, 4, 4)

    # Off-diagonal sum
    off_sum = np.sum(vals)
    # Diagonal added sum
    diag_sum = np.sum(diag_add)

    # The sum of all elements in Delta M is identically zero!
    assert abs(off_sum + diag_sum) < 1e-14


# ==============================================================================
# 3. Degenerate & Negative Node Filtering
# ==============================================================================

def test_triangular_degenerate_shell_negative_node():
    """Verify 3-node triangular shell with placeholder -1 node is handled safely."""
    # Triangular shell: [0, 1, 2, -1]
    conn = np.array([[0, 1, 2, -1]], dtype=np.int32)
    dmels = np.array([9.0], dtype=np.float64)

    rows, cols, vals, diag_add = _build_group_ams(conn, dmels, 4, 4)

    # Verify no negative indices in rows or cols
    assert len(rows) > 0
    assert np.all(rows >= 0)
    assert np.all(cols >= 0)
    # Only nodes 0, 1, 2 should be involved (3 pairs * 2 = 6 entries)
    assert len(rows) == 6
    assert set(rows) == {0, 1, 2}
    assert set(cols) == {0, 1, 2}

    # Verify row sums on active nodes
    for i in range(3):
        row_mask = (rows == i)
        assert abs(np.sum(vals[row_mask]) + diag_add[i]) < 1e-14
    assert diag_add[3] == 0.0


def test_triangular_shell_repeated_node():
    """Verify 3-node triangular shell with repeated node [0, 1, 2, 2] is handled."""
    conn = np.array([[0, 1, 2, 2]], dtype=np.int32)
    dmels = np.array([9.0], dtype=np.float64)

    rows, cols, vals, diag_add = _build_group_ams(conn, dmels, 4, 4)

    assert len(rows) == 6
    assert set(rows) == {0, 1, 2}
    for i in range(3):
        row_mask = (rows == i)
        assert abs(np.sum(vals[row_mask]) + diag_add[i]) < 1e-14


def test_single_node_degenerate_element():
    """Degenerate element with all identical nodes [0, 0, 0, 0] should produce no pairs."""
    conn = np.array([[0, 0, 0, 0]], dtype=np.int32)
    dmels = np.array([10.0], dtype=np.float64)

    rows, cols, vals, diag_add = _build_group_ams(conn, dmels, 4, 4)
    assert len(rows) == 0
    assert len(cols) == 0
    assert len(vals) == 0
    assert np.all(diag_add == 0.0)


# ==============================================================================
# 4. PCG Linear Solver Accuracy & Convergence
# ==============================================================================

def test_ams_pcg_accuracy():
    """Verify ams_pcg matches scipy direct solver for (M_diag + M_offdiag) * x = b."""
    np.random.seed(42)
    n = 10

    # Create synthetic connected graph with positive diagonal and zero row-sums
    A_off = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        v = np.random.uniform(0.5, 2.0)
        A_off[i, i + 1] = -v
        A_off[i + 1, i] = -v

    diag_add = -A_off.sum(axis=1)
    m_base = np.random.uniform(2.0, 5.0, size=n)
    M_diag = m_base + diag_add

    csr_off = sp.csr_matrix(A_off)

    # Full symmetric positive-definite matrix
    A_full = np.diag(M_diag) + A_off

    # Right hand side (forces)
    b = np.random.randn(n, 3)

    # Exact direct solution
    x_exact = np.linalg.solve(A_full, b)

    # PCG solution
    precond = np.zeros((n, 3), dtype=np.float64)
    for c in range(3):
        precond[:, c] = 1.0 / M_diag

    x_init = np.zeros((n, 3), dtype=np.float64)
    x_sol, iters, rel_res = ams_pcg(
        x_init, b, M_diag,
        csr_off.data, csr_off.indices, csr_off.indptr,
        precond, tol=1e-8, maxiter=500
    )

    np.testing.assert_allclose(x_sol, x_exact, rtol=1e-4, atol=1e-5)
    assert iters > 0
    assert rel_res < 1e-6


def test_ams_pcg_zero_force_early_exit():
    """Verify ams_pcg exits immediately with zero acceleration when b is zero."""
    n = 5
    M_diag = np.ones(n, dtype=np.float64)
    M_off = sp.csr_matrix((n, n), dtype=np.float64)
    precond = np.ones((n, 3), dtype=np.float64)

    # Pass non-zero initial guess; it must be zeroed out
    x = np.ones((n, 3), dtype=np.float64) * 99.0
    b = np.zeros((n, 3), dtype=np.float64)

    x_sol, iters, rel_res = ams_pcg(
        x, b, M_diag,
        M_off.data, M_off.indices, M_off.indptr,
        precond, tol=1e-5, maxiter=100
    )

    assert iters == 0
    assert rel_res == 0.0
    np.testing.assert_array_equal(x_sol, 0.0)


# ==============================================================================
# 5. AMSManager API & Selective Part Gating
# ==============================================================================

def test_ams_manager_tag_nodes():
    """Verify AMSManager.tag_nodes correctly flags nodes needing scaling."""
    model = Model()
    model.node_ids = np.array([1, 2, 3, 4, 5, 6])
    model.parts_list = ["P1"]

    # E1 (nodes 0,1,2,3) dt=0.5e-5 (needs AMS)
    # E2 (nodes 3,4,5) dt=2.0e-5 (does NOT need AMS)
    g1 = DummyElementGroup([[0, 1, 2, 3]], [10.0], part=0)
    g2 = DummyElementGroup([[3, 4, 5]], [10.0], part=0)
    model.element_groups = lambda: [("shells", g1), ("shells", g2)]

    controls = DummyControls(dt_min=1.0e-5, dt_scale=1.0)  # dt_target = 1.0e-5
    mgr = AMSManager(model, controls)

    dt_claims = [np.array([0.5e-5]), np.array([2.0e-5])]
    tagged = mgr.tag_nodes(dt_claims)

    # Nodes 0, 1, 2, 3 should be tagged; 4, 5 should not
    assert np.all(tagged[:4] == True)
    assert np.all(tagged[4:] == False)


def test_ams_manager_part_selectivity():
    """Verify /DT/AMS/igrp selectively scales only elements belonging to designated part."""
    model = Model()
    model.node_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    model.parts_list = ["P1", "P2"]
    model.parts = {1: "P1", 2: "P2"}

    # Part 1 has group 1; Part 2 has group 2
    class PartGroup:
        def __init__(self, members):
            self.members = members

    g1 = DummyElementGroup([[0, 1, 2, 3]], [10.0], part=0)
    g2 = DummyElementGroup([[4, 5, 6, 7]], [10.0], part=1)

    model.egroups = {"PART": {1: PartGroup([1])}}
    model.element_groups = lambda: [("shells", g1), ("quads", g2)]

    # Only scale Part 1 (igrp=1)
    controls = DummyControls(dt_min=1.0e-5, dt_scale=1.0, dt_ams_igrp=1)
    mgr = AMSManager(model, controls)

    assert mgr.active_parts[0] == True
    assert mgr.active_parts[1] == False

    # Both elements have low dt claims
    dt_claims = [np.array([0.5e-5]), np.array([0.5e-5])]
    tagged = mgr.tag_nodes(dt_claims)

    # Only nodes 0..3 of Part 1 tagged!
    assert np.all(tagged[:4] == True)
    assert np.all(tagged[4:] == False)


def test_ams_manager_build_ams_matrix():
    """Verify AMSManager.build_ams_matrix builds valid arrays and bounds."""
    model = Model()
    model.node_ids = np.array([1, 2, 3, 4])
    model.parts_list = ["P1"]

    g1 = DummyElementGroup([[0, 1, 2, 3]], [10.0], part=0)
    model.element_groups = lambda: [("shells", g1)]

    # dt_min = 1.0e-5, dt_scale = 0.5 -> dt_target = 2.0e-5
    controls = DummyControls(dt_min=1.0e-5, dt_scale=0.5)
    mgr = AMSManager(model, controls)
    assert abs(mgr.dt_target - 2.0e-5) < 1e-12

    # dt_claim = 1.0e-5 -> ratio = 2.0 -> dmels = 2 * 10 * (4 - 1) = 60
    dt_claims = [np.array([1.0e-5])]
    diag_added, M_offdiag, max_dmels = mgr.build_ams_matrix(dt_claims)

    assert abs(max_dmels - 60.0) < 1e-10
    assert M_offdiag is not None
    assert M_offdiag.shape == (4, 4)
    # Zero row sum
    np.testing.assert_allclose(diag_added + np.asarray(M_offdiag.sum(axis=1)).ravel(), 0.0, atol=1e-13)


def test_ams_manager_solve_no_ams_needed():
    """Verify AMSManager.solve handles empty off-diagonal matrix (M_offdiag is None)."""
    model = Model()
    model.node_ids = np.array([1, 2, 3])
    model.parts_list = ["P1"]
    controls = DummyControls()
    mgr = AMSManager(model, controls)

    acc = np.empty((3, 3), dtype=np.float64)
    f = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 20.0, 0.0],
        [0.0, 0.0, 30.0]
    ], dtype=np.float64)
    M_diag = np.array([2.0, 4.0, 5.0], dtype=np.float64)

    iters, rel_res = mgr.solve(acc, f, M_diag, None)

    assert iters == 0
    assert rel_res == 0.0
    np.testing.assert_allclose(acc[:, 0], [5.0, 0.0, 0.0])
    np.testing.assert_allclose(acc[:, 1], [0.0, 5.0, 0.0])
    np.testing.assert_allclose(acc[:, 2], [0.0, 0.0, 6.0])


# ==============================================================================
# 6. Physical Eigenvalue Shift & Critical Time Step
# ==============================================================================

def test_high_frequency_eigenvalue_attenuation():
    """
    Verify the fundamental physics of Selective Mass Scaling:
    For a discrete system (K, M), M_ams = M + Delta M:
    - The zero (rigid-body) eigenfrequency is invariant (omega_0 = 0).
    - The maximum internal eigenfrequency is reduced: omega_max,ams < omega_max,orig.
    """
    # 2-node 1D bar element:
    # K = k * [[1, -1], [-1, 1]]
    # M = m * [[1, 0], [0, 1]]
    k = 1000.0
    m = 2.0
    K = np.array([[k, -k], [-k, k]], dtype=np.float64)
    M_orig = np.array([[m, 0.0], [0.0, m]], dtype=np.float64)

    # Conventional generalized eigenvalues: det(K - omega^2 M) = 0
    w2_orig, _ = eigh(K, M_orig)
    # Roots: omega_0^2 = 0, omega_max^2 = 2k/m = 1000
    assert abs(w2_orig[0]) < 1e-12
    assert abs(w2_orig[1] - 1000.0) < 1e-10

    # Add AMS mass: Delta M = mu * [[1, -1], [-1, 1]]
    mu = 1.0  # added off-diagonal mass
    Delta_M = np.array([[mu, -mu], [-mu, mu]], dtype=np.float64)
    M_ams = M_orig + Delta_M

    w2_ams, _ = eigh(K, M_ams)

    # 1. Rigid mode is still 0!
    assert abs(w2_ams[0]) < 1e-12
    # 2. Maximum frequency is reduced from 2k/m to 2k/(m + 2mu) = 2000 / 4 = 500!
    expected_w2_ams = (2.0 * k) / (m + 2.0 * mu)
    assert abs(w2_ams[1] - expected_w2_ams) < 1e-10
    assert w2_ams[1] < w2_orig[1]

    # Explicit critical time step dt_crit = 2 / omega_max:
    dt_orig = 2.0 / np.sqrt(w2_orig[1])
    dt_ams = 2.0 / np.sqrt(w2_ams[1])
    assert dt_ams > dt_orig
    # With mu=1, dt_ams is sqrt(2) larger!
    assert abs(dt_ams / dt_orig - np.sqrt(2.0)) < 1e-10


# ==============================================================================
# 7. Engine Integration & Keyword Parsing
# ==============================================================================

def test_engine_keyword_parsing_dt_ams(tmp_path: Path):
    """Verify /DT/AMS card parsing in pyradioss/input/engine_keywords.py."""
    deck = """\
# Radioss Engine Deck
/DT/AMS/2
 0.9 1.5E-5
 1.0E-4
 150
/RUN/TEST/1
 0.01
"""
    p = tmp_path / "TEST_0001.rad"
    p.write_text(deck, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    engine_controls = parse_engine_deck(blocks, log)

    assert engine_controls.dt_ams == True
    assert engine_controls.dt_ams_igrp == 2
    assert abs(engine_controls.dt_scale - 0.9) < 1e-6
    assert abs(engine_controls.dt_min - 1.5e-5) < 1e-10
    assert abs(engine_controls.dt_ams_tol - 1.0e-4) < 1e-8
    assert engine_controls.dt_ams_itmax == 150


def test_mass_scaling_coordination():
    """Verify NodalTimeStep excludes AMS-tagged nodes from CST mass scaling."""
    model = Model()
    model.node_ids = np.array([1, 2])
    model.mass = np.array([1.0, 1.0])
    model.mass0 = model.mass.copy()
    model.inv_mass = 1.0 / model.mass
    model.inertia = np.zeros(2)

    class DummyControlsCST:
        dt_noda = "CST"
        dt_scale = 1.0
        dt_min = 1.0e-4

    class DummyLog:
        def warning(self, *args): pass
        def info(self, *args): pass

    v = np.zeros((2, 3), dtype=np.float64)

    # Case A: No AMS nodes -> both nodes scaled
    nts = NodalTimeStep(model, DummyControlsCST(), DummyLog())
    nts.stifn[:] = 4.0e8  # m_req = 4e8 * (1e-4)^2 / 2 = 2.0 > 1.0
    m_eff = model.mass.copy()
    inv_m = 1.0 / m_eff
    nts.apply(m_eff, inv_m, v, 0.0, ams_nodes=None)
    assert m_eff[0] > 1.0
    assert m_eff[1] > 1.0

    # Case B: Node 0 is an AMS node -> Node 0 is NOT scaled by CST, Node 1 IS scaled
    model.mass[:] = 1.0
    nts_b = NodalTimeStep(model, DummyControlsCST(), DummyLog())
    nts_b.stifn[:] = 4.0e8
    m_eff_b = model.mass.copy()
    inv_m_b = 1.0 / m_eff_b
    ams_nodes = np.array([True, False])
    nts_b.apply(m_eff_b, inv_m_b, v, 0.0, ams_nodes=ams_nodes)
    assert m_eff_b[0] == 1.0  # protected by AMS!
    assert m_eff_b[1] > 1.0   # scaled by CST


# ==============================================================================
# 8. Multi-Cycle, Extreme Mass Ratio & Edge Cases
# ==============================================================================

def test_multi_cycle_acceleration_update():
    """Verify AMS solve behaves consistently across consecutive dynamic time steps."""
    model = Model()
    model.node_ids = np.array([1, 2, 3, 4])
    model.parts_list = ["P1"]
    g1 = DummyElementGroup([[0, 1, 2, 3]], [10.0], part=0)
    model.element_groups = lambda: [("shells", g1)]

    controls = DummyControls(dt_min=1.0e-5, dt_scale=0.5, dt_ams_tol=1e-6, dt_ams_itmax=100)
    mgr = AMSManager(model, controls)

    dt_claims = [np.array([1.0e-5])]
    diag_added, M_offdiag, max_dmels = mgr.build_ams_matrix(dt_claims)

    m_base = np.array([2.5, 2.5, 2.5, 2.5], dtype=np.float64)
    M_diag = m_base + diag_added

    acc = np.zeros((4, 3), dtype=np.float64)

    # 3 consecutive cycles with time-varying sinusoidal forces
    for step in range(3):
        t = step * 1e-5
        f = np.zeros((4, 3), dtype=np.float64)
        f[0, 0] = 100.0 * np.cos(1000.0 * t)
        f[1, 1] = 50.0 * np.sin(1000.0 * t)
        f[2, 2] = -75.0

        iters, rel_res = mgr.solve(acc, f, M_diag, M_offdiag)
        assert iters > 0
        assert rel_res < 1e-5

        # Check residual: (M_diag + M_offdiag) * acc - f
        res = M_diag[:, None] * acc + M_offdiag.dot(acc) - f
        np.testing.assert_allclose(res, 0.0, atol=1e-3)


def test_extreme_mass_ratio_stability():
    """Verify PCG converges accurately even with 1000x ratio (dt_target / dt = 10)."""
    conn = np.array([[0, 1, 2, 3]], dtype=np.int32)
    # dt_target = 1e-5, dt = 1e-6 -> ratio = 10 -> ratio^2 = 100 -> dmels = 2 * 1.0 * 99 = 198
    dmels = np.array([198.0], dtype=np.float64)
    n_nodes = 4
    m_orig = np.full(n_nodes, 0.25, dtype=np.float64)

    rows, cols, vals, diag_add = _build_group_ams(conn, dmels, 4, n_nodes)
    M_offdiag = sp.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    M_diag = m_orig + diag_add

    # Even with high scaling ratio, row sums must be exactly 0
    np.testing.assert_allclose(diag_add + np.asarray(M_offdiag.sum(axis=1)).ravel(), 0.0, atol=1e-13)

    acc = np.zeros((4, 3), dtype=np.float64)
    f = np.array([
        [10.0, 0.0, 0.0],
        [-10.0, 5.0, 0.0],
        [0.0, -5.0, 2.0],
        [0.0, 0.0, -2.0]
    ], dtype=np.float64)

    controls = DummyControls(dt_ams_tol=1e-7, dt_ams_itmax=300)
    mgr = AMSManager(Model(), controls)
    iters, rel_res = mgr.solve(acc, f, M_diag, M_offdiag)

    assert iters > 0
    assert rel_res < 1e-6
    res = M_diag[:, None] * acc + M_offdiag.dot(acc) - f
    np.testing.assert_allclose(res, 0.0, atol=1e-4)


def test_pcg_maxiter_exhaustion_safety():
    """Verify PCG returns gracefully with max iterations when tolerance is impossible."""
    n = 4
    # Coupled off-diagonal matrix
    A_off = np.array([
        [0.0, -5.0, 0.0, 0.0],
        [-5.0, 0.0, -5.0, 0.0],
        [0.0, -5.0, 0.0, -5.0],
        [0.0, 0.0, -5.0, 0.0]
    ], dtype=np.float64)
    M_diag = np.array([20.0, 30.0, 30.0, 20.0], dtype=np.float64)
    M_off = sp.csr_matrix(A_off)
    precond = np.zeros((n, 3), dtype=np.float64)
    for c in range(3):
        precond[:, c] = 1.0 / M_diag

    x = np.zeros((n, 3), dtype=np.float64)
    b = np.array([[100.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 25.0], [10.0, 20.0, 30.0]], dtype=np.float64)

    # Maxiter = 1: should perform 1 iteration and return without error
    x_sol, iters, rel_res = ams_pcg(
        x, b, M_diag,
        M_off.data, M_off.indices, M_off.indptr,
        precond, tol=1e-15, maxiter=1
    )

    assert iters == 1
    assert rel_res > 0.0


def test_empty_model_ams_safety():
    """Verify AMSManager handles empty model and zero dt_target cleanly."""
    model = Model()
    model.node_ids = np.array([], dtype=int)
    controls = DummyControls(dt_min=0.0)  # dt_target = 0
    mgr = AMSManager(model, controls)

    tagged = mgr.tag_nodes([])
    assert len(tagged) == 0

    diag_added, M_offdiag, max_dmels = mgr.build_ams_matrix([])
    assert len(diag_added) == 0
    assert M_offdiag is None
    assert max_dmels == 0.0
