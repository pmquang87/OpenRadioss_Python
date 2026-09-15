# pyradioss - Milestone M498: /RWALL/LAGMUL Rigid Wall Lagrange Multiplier Tests
import pytest
import numpy as np

from pyradioss.model.model import Model
from pyradioss.model.entities import RigidWall, NodeGroup
from pyradioss.common.messages import MessageLog
from pyradioss.engine.rigid_wall import RigidWalls, LagmulRWall
from pyradioss.engine.lagmul import LagmulSolver
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_rwall


def test_rwall_lagmul_parsing_fixed(tmp_path):
    """Verify /RWALL/LAGMUL fixed format parsing with title, cards, and parameters."""
    deck_fixed = """/BEGIN
Fixed Deck
2022  0
/RWALL/LAGMUL/10
Rigid Wall Lagmul Fixed
        99         0        10         0
            0.500000            0.200000            1.000000            0.100000         1
           10.000000            0.000000            0.000000          -15.000000
            0.000000            0.000000            1.000000
"""
    p = tmp_path / "fixed_rwall_lagmul.rad"
    p.write_text(deck_fixed)
    model = Model()
    log = MessageLog()
    blocks = list(read_deck(str(p)))
    # block 0 is /BEGIN, block 1 is /RWALL/LAGMUL/10
    read_rwall(blocks[1], model, log)

    assert len(model.rwalls) == 1
    rw = model.rwalls[0]
    assert rw.id == 10
    assert rw.lagmul is True
    assert rw.geom == "PLANE"
    assert rw.node_id == 99
    assert rw.slide == 0
    assert rw.grnod_id == 10
    assert np.isclose(rw.dist, 0.5)
    assert np.isclose(rw.fric, 0.2)
    assert np.isclose(rw.mass, 10.0)
    assert np.isclose(rw.vz, -15.0)
    assert rw.ifq == 1
    assert np.isclose(rw.freq, 0.1)
    assert np.isclose(rw.alpha, 0.1)
    assert rw.title == "Rigid Wall Lagmul Fixed"


def test_rwall_lagmul_parsing_free(tmp_path):
    """Verify /RWALL/LAGMUL free format parsing with geometry and moving carrier parameters."""
    deck_free = """# Free format /RWALL/LAGMUL/PLANE
/RWALL/LAGMUL/PLANE/25
Rigid Wall Lagmul Free
88 1 50 0
0.2 0.0 0.0 0.0 0
5.0 1.0 0.0 0.0
0.0 0.0 1.0
"""
    p = tmp_path / "free_rwall_lagmul.rad"
    p.write_text(deck_free)
    model = Model()
    model.x0 = np.array([[0.0, 0.0, 0.0]])
    model.mass = np.array([10.0])
    model.v = np.zeros((1, 3))
    model._id2idx = {88: 0}
    log = MessageLog()
    blocks = list(read_deck(str(p)))
    read_rwall(blocks[0], model, log)

    assert len(model.rwalls) == 1
    rw = model.rwalls[0]
    assert rw.id == 25
    assert rw.lagmul is True
    assert rw.geom == "PLANE"
    assert rw.node_id == 88
    assert rw.slide == 1
    assert rw.grnod_id == 50
    assert np.isclose(rw.dist, 0.2)
    assert np.isclose(rw.mass, 5.0)
    # Carrier node mass increased by wall mass
    assert np.isclose(model.mass[0], 15.0)
    # Carrier node initial velocity set from card
    assert np.allclose(model.v[0], [1.0, 0.0, 0.0])


def test_rwall_lagmul_entity_resolution():
    """Verify candidate secondary node resolution, group exclusions, and massless node filtering."""
    model = Model()
    model.x0 = np.array([
        [0.0, 0.0, 0.0],  # 0: carrier node (wnode)
        [0.0, 0.0, 0.1],  # 1: normal candidate
        [0.0, 0.0, 0.2],  # 2: excluded candidate
        [0.0, 0.0, 0.3],  # 3: candidate
        [0.0, 0.0, 0.4],  # 4: massless/frozen node
        [0.0, 0.0, 0.5],  # 5: normal candidate
    ], dtype=np.float64)
    model.mass = np.array([100.0, 1.0, 1.0, 1.0, 1e30, 1.0], dtype=np.float64)
    model._id2idx = {99: 0}

    # Group 1 has nodes 1, 2, 3, 4, 5
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([1, 2, 3, 4, 5], dtype=np.int64))
    # Group 2 (exclusion) has node 2
    model.node_groups[2] = NodeGroup(id=2, node_idx=np.array([2], dtype=np.int64))

    rw = RigidWall(
        id=1,
        point=np.array([0.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        slide=0,
        grnod_id=1,
        grnod_id2=2,
        node_id=99,
        lagmul=True
    )

    lag_rw = LagmulRWall(rw, model)
    # Candidate list must contain [1, 3, 5] (carrier 0, excluded 2, massless 4 omitted)
    assert set(lag_rw.cand.tolist()) == {1, 3, 5}
    assert lag_rw.wnode == 0


def test_rwall_lagmul_fixed_plane_sliding():
    """Verify fixed plane wall sliding constraint generation (slide=0, 1 normal DOF per node)."""
    model = Model()
    model.x0 = np.array([
        [0.0, 0.0, -0.05],  # 0: penetrating (Z < 0)
        [0.0, 0.0, 0.10],   # 1: outside (Z > 0)
        [0.0, 0.0, 0.02],   # 2: outside but predicted penetrating with Vz < 0
    ], dtype=np.float64)
    model.x = model.x0.copy()
    model.v = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],    # Moving away (+Z)
        [0.0, 0.0, -10.0],  # Moving inward (-Z)
    ], dtype=np.float64)
    model.mass = np.ones(3)

    rw = RigidWall(
        id=1,
        point=np.array([0.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        slide=0,
        node_id=0,  # Fixed wall
        lagmul=True
    )
    lag_rw = LagmulRWall(rw, model)

    # dt = 0.01: predicted position of node 2 is 0.02 + (-10)*(0.005) = -0.03 <= 0
    data, nodes, dofs, eq_ids, n_rows = lag_rw.generate_l_matrix(dt=0.01)

    # Nodes 0 and 2 should be in contact (n_rows = 2)
    assert n_rows == 2
    # Normal is [0, 0, 1], so coefficients on secondary nodes are [0, 0, 1] on DOFs (0, 1, 2)
    # Equation 0 (node 0):
    mask_eq0 = (eq_ids == 0)
    assert np.array_equal(nodes[mask_eq0], [0, 0, 0])
    assert np.allclose(data[mask_eq0], [0.0, 0.0, 1.0])
    # Equation 1 (node 2):
    mask_eq1 = (eq_ids == 1)
    assert np.array_equal(nodes[mask_eq1], [2, 2, 2])
    assert np.allclose(data[mask_eq1], [0.0, 0.0, 1.0])


def test_rwall_lagmul_fixed_plane_tied():
    """Verify fixed plane wall tied constraint generation (slide=1, 3 orthogonal DOFs per node)."""
    model = Model()
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float64)
    model.x = model.x0.copy()
    model.v = np.zeros((2, 3))
    model.mass = np.ones(2)

    rw = RigidWall(
        id=2,
        point=np.array([0.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        slide=1,  # Tied
        node_id=0,
        lagmul=True
    )
    lag_rw = LagmulRWall(rw, model)

    data, nodes, dofs, eq_ids, n_rows = lag_rw.generate_l_matrix(dt=0.001)

    # 2 nodes tied -> 6 equations (3 DOFs each)
    assert n_rows == 6
    for r in range(6):
        node_expected = r // 3
        dof_expected = r % 3
        mask = (eq_ids == r)
        assert len(data[mask]) == 1
        assert data[mask][0] == 1.0
        assert nodes[mask][0] == node_expected
        assert dofs[mask][0] == dof_expected


def test_rwall_lagmul_moving_plane_sliding_momentum_conservation():
    """Verify moving plane wall sliding constraint with exact momentum conservation (sum L_row = 0)."""
    model = Model()
    model.x0 = np.array([
        [0.0, 0.0, 0.0],   # 0: Carrier node (wnode)
        [0.0, 0.0, -0.01], # 1: Contacting node 1
        [1.0, 0.0, -0.02], # 2: Contacting node 2
    ], dtype=np.float64)
    model.x = model.x0.copy()
    model.v = np.zeros((3, 3))
    model.mass = np.array([50.0, 1.0, 1.0])
    model._id2idx = {77: 0}

    # Angled wall normal [0, 1/sqrt(2), 1/sqrt(2)]
    normal = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
    rw = RigidWall(
        id=3,
        point=np.array([0.0, 0.0, 0.0]),
        normal=normal,
        slide=0,
        node_id=77,  # Moving wall tied to node 77 (idx 0)
        lagmul=True
    )
    lag_rw = LagmulRWall(rw, model)

    data, nodes, dofs, eq_ids, n_rows = lag_rw.generate_l_matrix(dt=0.001)

    assert n_rows == 2
    # Verify exact linear momentum conservation for every row:
    # Secondary node gets +n, carrier node gets -n -> sum = 0.0 identically
    for r in range(n_rows):
        mask = (eq_ids == r)
        # Sum of row entries must be 0
        assert np.isclose(np.sum(data[mask]), 0.0, atol=1e-15)
        # Check per-DOF cancellation: L(dof) on secondary + L(dof) on carrier = 0
        for dof in range(3):
            dof_mask = mask & (dofs == dof)
            assert np.isclose(np.sum(data[dof_mask]), 0.0, atol=1e-15)


def test_rwall_lagmul_moving_plane_tied_momentum_conservation():
    """Verify moving plane wall tied constraint with exact momentum conservation (sum L_row = 0)."""
    model = Model()
    model.x0 = np.array([
        [0.0, 0.0, 0.0],  # 0: Carrier node
        [0.5, 0.5, 0.0],  # 1: Secondary tied node
    ], dtype=np.float64)
    model.x = model.x0.copy()
    model.v = np.zeros((2, 3))
    model.mass = np.array([25.0, 2.0])
    model._id2idx = {88: 0}

    rw = RigidWall(
        id=4,
        point=np.array([0.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        slide=1,  # Tied
        node_id=88,
        lagmul=True
    )
    lag_rw = LagmulRWall(rw, model)

    data, nodes, dofs, eq_ids, n_rows = lag_rw.generate_l_matrix(dt=0.001)

    assert n_rows == 3
    for r in range(n_rows):
        mask = (eq_ids == r)
        assert len(data[mask]) == 2
        # Secondary node +1.0, carrier node -1.0
        assert np.isclose(np.sum(data[mask]), 0.0, atol=1e-15)
        assert 1.0 in data[mask]
        assert -1.0 in data[mask]


def test_rwall_lagmul_cylinder_geometry():
    """Verify cylindrical rigid wall geometry and radial normal constraint generation."""
    model = Model()
    # Cylinder axis along Z through (0, 0, 0), radius R = 2.0
    # Candidate 0 is at (1.5, 0.0, 10.0) -> r = 1.5 < R=2.0 (penetrating)
    # Candidate 1 is at (3.0, 0.0, 5.0) -> r = 3.0 > R=2.0 (outside)
    model.x0 = np.array([
        [1.5, 0.0, 10.0],
        [3.0, 0.0, 5.0],
    ], dtype=np.float64)
    model.x = model.x0.copy()
    model.v = np.zeros((2, 3))
    model.mass = np.ones(2)

    rw = RigidWall(
        id=5,
        point=np.array([0.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),  # Cylinder axis
        geom="CYL",
        radius=2.0,
        slide=0,
        node_id=0,
        lagmul=True
    )
    lag_rw = LagmulRWall(rw, model)

    data, nodes, dofs, eq_ids, n_rows = lag_rw.generate_l_matrix(dt=0.001)

    assert n_rows == 1
    assert nodes[0] == 0
    # Radial unit normal for node at (1.5, 0, 10) is [1, 0, 0]
    assert np.allclose(data, [1.0, 0.0, 0.0])


def test_rwall_lagmul_sphere_geometry():
    """Verify spherical rigid wall geometry and radial normal constraint generation."""
    model = Model()
    # Sphere centered at (1.0, 1.0, 1.0), radius R = 3.0
    # Candidate 0 at (1.0, 3.0, 1.0) -> r = 2.0 < R=3.0 (penetrating along Y)
    # Candidate 1 at (1.0, 5.0, 1.0) -> r = 4.0 > R=3.0 (outside)
    model.x0 = np.array([
        [1.0, 3.0, 1.0],
        [1.0, 5.0, 1.0],
    ], dtype=np.float64)
    model.x = model.x0.copy()
    model.v = np.zeros((2, 3))
    model.mass = np.ones(2)

    rw = RigidWall(
        id=6,
        point=np.array([1.0, 1.0, 1.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        geom="SPHER",
        radius=3.0,
        slide=0,
        node_id=0,
        lagmul=True
    )
    lag_rw = LagmulRWall(rw, model)

    data, nodes, dofs, eq_ids, n_rows = lag_rw.generate_l_matrix(dt=0.001)

    assert n_rows == 1
    assert nodes[0] == 0
    # Radial unit normal for node at (1, 3, 1) relative to center (1, 1, 1) is [0, 1, 0]
    assert np.allclose(data, [0.0, 1.0, 0.0])


def test_rwall_lagmul_search_distance_filtering():
    """Verify search distance (dist > 0) selects only nearby nodes at initialization."""
    model = Model()
    model.x0 = np.array([
        [0.0, 0.0, 0.05],  # within dist = 0.1
        [0.0, 0.0, 0.08],  # within dist = 0.1
        [0.0, 0.0, 0.50],  # farther than dist = 0.1
    ], dtype=np.float64)
    model.mass = np.ones(3)

    rw = RigidWall(
        id=7,
        point=np.array([0.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        dist=0.1,
        slide=0,
        node_id=0,
        lagmul=True
    )
    lag_rw = LagmulRWall(rw, model)
    assert set(lag_rw.cand.tolist()) == {0, 1}


def test_rwall_lagmul_solver_transfer_forces():
    """Verify force transfer and equilibrium in LagmulSolver for moving wall impact."""
    model = Model()
    model.x0 = np.array([
        [0.0, 0.0, 0.0],    # 0: Moving wall carrier node
        [0.0, 0.0, -0.01],  # 1: Penetrating secondary node
    ], dtype=np.float64)
    model.x = model.x0.copy()
    model.mass = np.array([100.0, 2.0], dtype=np.float64)
    model.v = np.zeros((2, 3))
    model._id2idx = {100: 0}

    rw = RigidWall(
        id=8,
        point=np.array([0.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        slide=0,
        node_id=100,
        lagmul=True
    )
    model.rwalls.append(rw)

    log = MessageLog()
    solver = LagmulSolver(model, None, log)
    assert len(solver) == 1

    fint = np.zeros((2, 3))
    fcont = np.zeros((2, 3))
    fext = np.zeros((2, 3))
    # Push secondary node into wall with external force (-Z)
    fext[1] = np.array([0.0, 0.0, -50.0])
    mint = np.zeros((2, 3))
    inv_mass = 1.0 / model.mass
    inv_inertia = np.ones((2, 3))
    dt = 1e-4

    solver.transfer_forces(fint, fcont, fext, mint, inv_mass, inv_inertia, dt)

    # Net constraint forces must sum to [0, 0, 0] to machine precision (Newton's 3rd law!)
    sum_fint = np.sum(fint, axis=0)
    assert np.allclose(sum_fint, 0.0, atol=1e-10)
    # Secondary node receives outward force (+Z)
    assert fint[1, 2] > 0.0
    # Carrier node receives recoil force (-Z)
    assert fint[0, 2] < 0.0
    assert np.isclose(fint[1, 2], -fint[0, 2], atol=1e-10)


def test_rwall_lagmul_solver_enforce_velocity_projection():
    """Verify velocity projection in LagmulSolver.enforce with exact linear momentum conservation."""
    model = Model()
    model.x0 = np.array([
        [0.0, 0.0, 0.0],    # 0: Carrier node
        [0.0, 0.0, -0.01],  # 1: Penetrating node
    ], dtype=np.float64)
    model.x = model.x0.copy()
    model.mass = np.array([50.0, 5.0], dtype=np.float64)
    model._id2idx = {200: 0}

    rw = RigidWall(
        id=9,
        point=np.array([0.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        slide=0,
        node_id=200,
        lagmul=True
    )
    model.rwalls.append(rw)

    log = MessageLog()
    solver = LagmulSolver(model, None, log)

    v = np.zeros((2, 3))
    # Secondary node moving into the wall
    v[1] = np.array([0.0, 0.0, -20.0])
    model.v = v.copy()
    v_init = v.copy()
    vr = np.zeros((2, 3))
    inv_mass = 1.0 / model.mass
    inv_inertia = np.ones((2, 3))

    # Initial linear momentum
    p_init = np.sum(model.mass[:, None] * v_init, axis=0)

    solver.enforce(v, vr, inv_mass, inv_inertia)

    # Final linear momentum must equal initial linear momentum to machine precision
    p_final = np.sum(model.mass[:, None] * v, axis=0)
    assert np.allclose(p_final, p_init, atol=1e-10)

    # Constraint equation L v = 0 must be satisfied:
    # v_normal(secondary) - v_normal(carrier) = 0
    assert np.isclose(v[1, 2], v[0, 2], atol=1e-6)


def test_rwall_lagmul_skipped_by_kinematic_walls():
    """Verify that kinematic RigidWalls solver skips walls designated as lagmul."""
    model = Model()
    model.x0 = np.zeros((2, 3))
    model.mass = np.ones(2)

    rw_kin = RigidWall(id=1, point=np.zeros(3), normal=np.array([0, 0, 1]), lagmul=False)
    rw_lag = RigidWall(id=2, point=np.zeros(3), normal=np.array([0, 0, 1]), lagmul=True)
    model.rwalls = [rw_kin, rw_lag]

    log = MessageLog()
    kin_walls = RigidWalls(model, log)
    # Only the non-lagmul wall should be registered in kin_walls
    assert len(kin_walls.walls) == 1
    assert kin_walls.walls[0][0].id == 1
