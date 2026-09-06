# pyradioss - Milestone M496: /INTER/LAGMUL/TYPE17 Interface Hardening & Tests
import pytest
import numpy as np

from pyradioss.model.model import Model, ElementGroup
from pyradioss.model.entities import Interface, EntityGroup
from pyradioss.common.messages import MessageLog
from pyradioss.contact.inter_type17 import LagmulType17
from pyradioss.engine.lagmul import LagmulSolver
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_inter


def test_inter_type17_keyword_parsing_fixed(tmp_path):
    """Verify /INTER/LAGMUL/TYPE17 parsing in fixed format with title and cards."""
    deck_fixed = """/FORMAT/1
/INTER/LAGMUL/TYPE17/10
Fixed Tied Brick Interface
        12        34
                    1
"""
    p = tmp_path / "fixed.rad"
    p.write_text(deck_fixed)
    model = Model()
    log = MessageLog()
    blocks = list(read_deck(str(p)))
    read_inter(blocks[1], model, log)
    assert len(model.interfaces) == 1
    itf = model.interfaces[0]
    assert itf.id == 10
    assert itf.type == 17
    assert itf.grbric_id1 == 12
    assert itf.grbric_id2 == 34
    assert itf.itied == 1
    assert itf.lagmul is True
    assert itf.title == "Fixed Tied Brick Interface"


def test_inter_type17_keyword_parsing_free(tmp_path):
    """Verify /INTER/LAGMUL/TYPE17 parsing in free format."""
    deck_free = """# Free format
/INTER/LAGMUL/TYPE17/20
Free Sliding Brick Interface
55 66
0
"""
    p = tmp_path / "free.rad"
    p.write_text(deck_free)
    model = Model()
    log = MessageLog()
    blocks = list(read_deck(str(p)))
    read_inter(blocks[0], model, log)
    assert len(model.interfaces) == 1
    itf = model.interfaces[0]
    assert itf.id == 20
    assert itf.type == 17
    assert itf.grbric_id1 == 55
    assert itf.grbric_id2 == 66
    assert itf.itied == 0
    assert itf.lagmul is True


def _make_two_brick_mesh():
    """Helper creating two adjacent 8-node bricks.
    Brick 0 (Master): [0, 1] x [0, 1] x [0, 1]
    Brick 1 (Secondary): [1, 2] x [0, 1] x [0, 1]
    Nodes 0..7 for Brick 0, Nodes 8..15 for Brick 1.
    Note: Nodes 8, 9, 12, 13 of Brick 1 are colocated with nodes 3, 2, 7, 6 of Brick 0 at x=1.
    """
    coords = np.array([
        # Brick 0: [0, 1]^3
        [0.0, 0.0, 0.0],  # 0
        [0.0, 0.0, 1.0],  # 1
        [1.0, 0.0, 1.0],  # 2
        [1.0, 0.0, 0.0],  # 3
        [0.0, 1.0, 0.0],  # 4
        [0.0, 1.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [1.0, 1.0, 0.0],  # 7
        # Brick 1: [1, 2] x [0, 1] x [0, 1]
        [1.0, 0.0, 0.0],  # 8  (colocated with 3)
        [1.0, 0.0, 1.0],  # 9  (colocated with 2)
        [2.0, 0.0, 1.0],  # 10
        [2.0, 0.0, 0.0],  # 11
        [1.0, 1.0, 0.0],  # 12 (colocated with 7)
        [1.0, 1.0, 1.0],  # 13 (colocated with 6)
        [2.0, 1.0, 1.0],  # 14
        [2.0, 1.0, 0.0],  # 15
    ], dtype=np.float64)

    conn = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],         # Brick 0
        [8, 9, 10, 11, 12, 13, 14, 15],   # Brick 1
    ], dtype=np.int64)

    return coords, conn


def test_inter_type17_entity_resolution_element_group():
    """Verify entity resolution using modern ElementGroup and egroups."""
    coords, conn = _make_two_brick_mesh()
    model = Model()
    model.x0 = coords.copy()

    model.bricks = ElementGroup(
        ids=np.array([1, 2], dtype=np.int64),
        conn=conn,
        part=np.array([1, 2], dtype=np.int64),
    )

    sec_grp = EntityGroup(id=1, family="BRIC", title="SecBricks")
    sec_grp.elem_idx = np.array([1], dtype=np.int64)

    mas_grp = EntityGroup(id=2, family="BRIC", title="MasBricks")
    mas_grp.elem_idx = np.array([0], dtype=np.int64)

    model.egroups = {
        "GRBRIC": {
            1: sec_grp,
            2: mas_grp,
        }
    }

    itf = Interface(id=1, type=17, grbric_id1=1, grbric_id2=2, itied=1, lagmul=True)
    lag = LagmulType17(itf, model)

    assert len(lag.sec_bricks) == 1
    assert len(lag.mas_bricks) == 1
    np.testing.assert_array_equal(lag.sec_bricks[0], conn[1])
    np.testing.assert_array_equal(lag.mas_bricks[0], conn[0])
    assert len(lag.active_mas_bricks) > 0


def test_inter_type17_entity_resolution_legacy_ixs():
    """Verify entity resolution with legacy .ixs and raw numpy arrays."""
    coords, conn = _make_two_brick_mesh()
    model = Model()
    model.x0 = coords.copy()

    class LegacyBricks:
        pass

    lb = LegacyBricks()
    lb.ixs = conn.copy()
    lb.n = 2
    model.bricks = lb

    model.egroups = {
        "BRIC": {
            10: type("Grp", (), {"elem_idx": np.array([1], dtype=np.int64)})(),
            20: type("Grp", (), {"elem_idx": np.array([0], dtype=np.int64)})(),
        }
    }

    itf = Interface(id=2, type=17, grbric_id1=10, grbric_id2=20, itied=1, lagmul=True)
    lag = LagmulType17(itf, model)

    assert len(lag.sec_bricks) == 1
    assert len(lag.mas_bricks) == 1
    np.testing.assert_array_equal(lag.sec_bricks[0], conn[1])
    np.testing.assert_array_equal(lag.mas_bricks[0], conn[0])


def test_inter_type17_entity_resolution_part_members():
    """Verify entity resolution with egroups['PART'] and .members."""
    coords, conn = _make_two_brick_mesh()
    model = Model()
    model.x0 = coords.copy()
    model.bricks = np.array(conn, dtype=np.int64)

    class PartGrp:
        def __init__(self, idxs):
            self.members = [("bricks", idxs)]

    model.egroups = {
        "PART": {
            100: PartGrp(np.array([1], dtype=np.int64)),
            200: PartGrp(np.array([0], dtype=np.int64)),
        }
    }

    itf = Interface(id=3, type=17, grbric_id1=100, grbric_id2=200, itied=1, lagmul=True)
    lag = LagmulType17(itf, model)

    assert len(lag.sec_bricks) == 1
    assert len(lag.mas_bricks) == 1
    np.testing.assert_array_equal(lag.sec_bricks[0], conn[1])
    np.testing.assert_array_equal(lag.mas_bricks[0], conn[0])


def test_inter_type17_entity_resolution_default_all_bricks():
    """Verify default resolution when grbric_id is 0."""
    coords, conn = _make_two_brick_mesh()
    model = Model()
    model.x0 = coords.copy()
    model.bricks = conn.copy()

    itf = Interface(id=4, type=17, grbric_id1=0, grbric_id2=0, itied=1, lagmul=True)
    lag = LagmulType17(itf, model)

    assert len(lag.sec_bricks) == 2
    assert len(lag.mas_bricks) == 2


def test_inter_type17_direct_overrides():
    """Verify direct interface attribute overrides (secondary_bricks, master_bricks)."""
    coords, conn = _make_two_brick_mesh()
    model = Model()
    model.x0 = coords.copy()

    itf = Interface(id=5, type=17, itied=1, lagmul=True)
    itf.secondary_bricks = conn[1:2]
    itf.master_bricks = conn[0:1]

    lag = LagmulType17(itf, model)
    assert len(lag.sec_bricks) == 1
    assert len(lag.mas_bricks) == 1
    np.testing.assert_array_equal(lag.sec_bricks[0], conn[1])
    np.testing.assert_array_equal(lag.mas_bricks[0], conn[0])


def test_inter_type17_tied_projection_shape_functions():
    """Verify tied mode (ITIED=1) natural coordinate projection, partition of unity,
    and spatial coordinate reconstruction."""
    coords, conn = _make_two_brick_mesh()
    model = Model()
    model.x0 = coords.copy()

    itf = Interface(id=6, type=17, itied=1, lagmul=True)
    itf.secondary_bricks = conn[1:2]
    itf.master_bricks = conn[0:1]

    lag = LagmulType17(itf, model)

    # Secondary brick nodes 8, 9, 12, 13 lie on the interface x=1.0 with master brick
    assert len(lag.active_mas_bricks) == 4
    for i in range(len(lag.active_mas_bricks)):
        # Check master shape function partition of unity
        N_m = lag.active_mas_N[i]
        assert abs(np.sum(N_m) - 1.0) < 1e-6
        # Check secondary shape function partition of unity
        N_s = lag.active_sec_N[i]
        assert abs(np.sum(N_s) - 1.0) < 1e-6

        # Reconstruct spatial position from master shape functions
        m_brick = lag.active_mas_bricks[i]
        m_pts = coords[m_brick]
        recon_pt = np.sum(N_m[:, None] * m_pts, axis=0)

        # Expected secondary node position
        s_brick = lag.active_sec_bricks[i]
        s_pts = coords[s_brick]
        sec_pt = np.sum(N_s[:, None] * s_pts, axis=0)

        np.testing.assert_allclose(recon_pt, sec_pt, atol=1e-4)


def test_inter_type17_tied_l_matrix_conservation():
    """Verify tied mode (ITIED=1) constraint matrix L generation and exact momentum conservation."""
    coords, conn = _make_two_brick_mesh()
    model = Model()
    model.x0 = coords.copy()

    itf = Interface(id=7, type=17, itied=1, lagmul=True)
    itf.secondary_bricks = conn[1:2]
    itf.master_bricks = conn[0:1]

    lag = LagmulType17(itf, model)
    data, nodes, dofs, eq_ids, n_rows = lag.generate_l_matrix()

    # 4 active contact points * 3 DOFs = 12 constraint rows
    assert n_rows == 12
    assert len(data) == 12 * 16  # 16 nodes per row (8 master, 8 secondary)

    # Verify each equation row has row sum == 0 (exact linear momentum conservation)
    for eq_i in range(n_rows):
        row_mask = (eq_ids == eq_i)
        row_data = data[row_mask]
        assert abs(np.sum(row_data)) < 1e-7


def test_inter_type17_sliding_normal_and_velocity_condition():
    """Verify sliding mode (ITIED=0) outward normal and velocity penetration condition."""
    coords, conn = _make_two_brick_mesh()
    model = Model()
    model.x = coords.copy()

    itf = Interface(id=8, type=17, itied=0, lagmul=True)
    itf.secondary_bricks = conn[1:2]
    itf.master_bricks = conn[0:1]

    lag = LagmulType17(itf, model)

    # Set incoming velocities: secondary moving in -X toward master (penetration)
    v = np.zeros((16, 3), dtype=np.float64)
    v[8:16, 0] = -1.0  # moving left into master
    model.v = v

    data, nodes, dofs, eq_ids, n_rows = lag.generate_l_matrix()
    # Outward normal should be unit length and point in +X (from master toward secondary)
    for norm in lag.active_normals:
        assert abs(np.linalg.norm(norm) - 1.0) < 1e-6
        assert norm[0] > 0.9

    # Under penetrating velocity, active constraint rows are generated
    assert n_rows > 0

    # Under separating velocities: secondary moving +X away from master
    v_sep = np.zeros((16, 3), dtype=np.float64)
    v_sep[8:16, 0] = 5.0  # moving right away from master
    model.v = v_sep

    data_sep, nodes_sep, dofs_sep, eq_ids_sep, n_rows_sep = lag.generate_l_matrix()
    assert n_rows_sep == 0


def test_inter_type17_sliding_l_matrix_conservation():
    """Verify sliding mode (ITIED=0) constraint row entries and exact momentum conservation."""
    coords, conn = _make_two_brick_mesh()
    model = Model()
    model.x = coords.copy()

    itf = Interface(id=9, type=17, itied=0, lagmul=True)
    itf.secondary_bricks = conn[1:2]
    itf.master_bricks = conn[0:1]

    lag = LagmulType17(itf, model)

    # Penetrating velocity
    v = np.zeros((16, 3), dtype=np.float64)
    v[8:16, 0] = -2.0
    model.v = v

    data, nodes, dofs, eq_ids, n_rows = lag.generate_l_matrix()
    assert n_rows > 0

    # For each sliding constraint row:
    # 48 non-zero entries (16 nodes * 3 DOFs)
    # Sum of entries across each row must equal 0.0
    for eq_i in range(n_rows):
        row_mask = (eq_ids == eq_i)
        row_data = data[row_mask]
        assert len(row_data) == 48
        assert abs(np.sum(row_data)) < 1e-7


def test_inter_type17_disjoint_bricks_broad_phase():
    """Verify disjoint bricks broad-phase rejection and safe empty return."""
    coords, conn = _make_two_brick_mesh()
    # Move secondary brick far away
    coords[8:16] += 100.0

    model = Model()
    model.x0 = coords.copy()

    itf = Interface(id=10, type=17, itied=1, lagmul=True)
    itf.secondary_bricks = conn[1:2]
    itf.master_bricks = conn[0:1]

    log = MessageLog()
    lag = LagmulType17(itf, model, log)
    data, nodes, dofs, eq_ids, n_rows = lag.generate_l_matrix()

    assert n_rows == 0
    assert len(data) == 0


def test_inter_type17_degenerate_brick_resilience():
    """Verify resilience on degenerate / flat elements via pseudo-inverse."""
    coords, conn = _make_two_brick_mesh()
    # Flatten master brick in Z
    coords[0:8, 2] = 0.0

    model = Model()
    model.x0 = coords.copy()

    itf = Interface(id=11, type=17, itied=1, lagmul=True)
    itf.secondary_bricks = conn[1:2]
    itf.master_bricks = conn[0:1]

    # Should not crash with LinAlgError
    lag = LagmulType17(itf, model)
    data, nodes, dofs, eq_ids, n_rows = lag.generate_l_matrix()
    assert isinstance(n_rows, int)


def test_inter_type17_empty_fallback():
    """Verify empty model and missing entity graceful fallback."""
    model = Model()
    itf = Interface(id=12, type=17, itied=1, lagmul=True)

    lag = LagmulType17(itf, model)
    data, nodes, dofs, eq_ids, n_rows = lag.generate_l_matrix()

    assert n_rows == 0
    assert len(data) == 0
    assert len(nodes) == 0


def test_inter_type17_lagmul_solver_integration_tied():
    """Verify full end-to-end integration with LagmulSolver for tied interface."""
    coords, conn = _make_two_brick_mesh()
    model = Model()
    model.x = coords.copy()
    model.x0 = coords.copy()
    model.mass = np.ones(len(coords))

    itf = Interface(id=13, type=17, itied=1, lagmul=True)
    itf.secondary_bricks = conn[1:2]
    itf.master_bricks = conn[0:1]
    model.interfaces.append(itf)

    log = MessageLog()
    solver = LagmulSolver(model, None, log)
    assert len(solver) == 1
    assert isinstance(solver.interfaces[0], LagmulType17)

    # Force transfer stage
    n_nodes = len(coords)
    fint = np.zeros((n_nodes, 3), dtype=np.float64)
    fcont = np.zeros((n_nodes, 3), dtype=np.float64)
    fext = np.zeros((n_nodes, 3), dtype=np.float64)
    mint = np.zeros((n_nodes, 3), dtype=np.float64)
    inv_mass = 1.0 / model.mass
    inv_inertia = np.ones((n_nodes, 3), dtype=np.float64)

    # Apply pulling force on contacting secondary node 8
    fext[8, 0] = 50.0

    solver.transfer_forces(fint, fcont, fext, mint, inv_mass, inv_inertia, dt=1e-4)

    # Constraint forces are injected into fint
    assert np.any(np.abs(fint) > 0.0)
    # Global linear momentum of constraint forces must balance: sum(fint) == 0
    np.testing.assert_allclose(np.sum(fint, axis=0), np.zeros(3), atol=1e-4)

    # Secondary node should get negative force balancing fext
    assert fint[8, 0] < 0.0
    # Master nodes should get positive force
    assert np.sum(fint[0:8, 0]) > 0.0

    # Velocity cleanup stage
    model.v = np.zeros_like(coords)
    model.v[8] = np.array([4.0, 0.0, 0.0])
    vr = np.zeros_like(coords)
    solver.enforce(model.v, vr, inv_mass, inv_inertia)
    # Secondary node velocity should be projected to match colocated master node 3
    np.testing.assert_allclose(model.v[8], model.v[3], atol=1e-4)
    assert abs(model.v[8, 0] - 2.0) < 1e-4


def test_inter_type17_lagmul_solver_integration_sliding():
    """Verify full end-to-end integration with LagmulSolver for sliding interface."""
    coords, conn = _make_two_brick_mesh()
    model = Model()
    model.x = coords.copy()
    model.x0 = coords.copy()
    model.mass = np.ones(len(coords))

    itf = Interface(id=14, type=17, itied=0, lagmul=True)
    itf.secondary_bricks = conn[1:2]
    itf.master_bricks = conn[0:1]
    model.interfaces.append(itf)

    log = MessageLog()
    solver = LagmulSolver(model, None, log)
    assert len(solver) == 1

    n_nodes = len(coords)
    v = np.zeros((n_nodes, 3), dtype=np.float64)
    # Secondary pushing into master (-X)
    v[8:16, 0] = -5.0
    model.v = v

    fint = np.zeros((n_nodes, 3), dtype=np.float64)
    fcont = np.zeros((n_nodes, 3), dtype=np.float64)
    fext = np.zeros((n_nodes, 3), dtype=np.float64)
    mint = np.zeros((n_nodes, 3), dtype=np.float64)
    inv_mass = 1.0 / model.mass
    inv_inertia = np.ones((n_nodes, 3), dtype=np.float64)

    # Non-zero acceleration that penetrates
    fext[8, 0] = -20.0
    solver.transfer_forces(fint, fcont, fext, mint, inv_mass, inv_inertia, dt=1e-4)
    assert np.any(np.abs(fint) > 0.0)
    np.testing.assert_allclose(np.sum(fint, axis=0), np.zeros(3), atol=1e-4)

    # Velocity projection enforce()
    vr = np.zeros_like(coords)
    solver.enforce(model.v, vr, inv_mass, inv_inertia)
