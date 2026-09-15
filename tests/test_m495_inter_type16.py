# pyradioss - Milestone M495: /INTER/LAGMUL/TYPE16 Interface Hardening & Tests
import pytest
import numpy as np

from pyradioss.model.model import Model, ElementGroup
from pyradioss.model.entities import Interface, NodeGroup, EntityGroup
from pyradioss.common.messages import MessageLog
from pyradioss.contact.inter_type16 import LagmulType16
from pyradioss.engine.lagmul import LagmulSolver
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_inter


def test_inter_type16_keyword_parsing_fixed(tmp_path):
    """Verify /INTER/LAGMUL/TYPE16 parsing in fixed format with title and cards."""
    deck_fixed = """/FORMAT/1
/INTER/LAGMUL/TYPE16/10
Fixed Tied Interface
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
    assert itf.type == 16
    assert itf.grnod_id == 12
    assert itf.grbric_id1 == 34
    assert itf.itied == 1
    assert itf.lagmul is True
    assert itf.title == "Fixed Tied Interface"


def test_inter_type16_keyword_parsing_free(tmp_path):
    """Verify /INTER/LAGMUL/TYPE16 parsing in free format."""
    deck_free = """# Free format
/INTER/LAGMUL/TYPE16/20
Free Sliding Interface
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
    assert itf.type == 16
    assert itf.grnod_id == 55
    assert itf.grbric_id1 == 66
    assert itf.itied == 0
    assert itf.lagmul is True


def _make_unit_cube_mesh():
    """Helper creating a single 8-node unit cube brick mesh [0, 1]^3.
    Nodes:
      0: (0, 0, 0)
      1: (0, 0, 1)
      2: (1, 0, 1)
      3: (1, 0, 0)
      4: (0, 1, 0)
      5: (0, 1, 1)
      6: (1, 1, 1)
      7: (1, 1, 0)
      8: secondary node at (0.5, 0.5, 0.5) inside
      9: secondary node at (0.25, 0.75, 0.5) inside
      10: secondary node outside at (2.0, 3.0, 2.0)
    """
    coords = np.array([
        [0.0, 0.0, 0.0],   # 0
        [0.0, 0.0, 1.0],   # 1
        [1.0, 0.0, 1.0],   # 2
        [1.0, 0.0, 0.0],   # 3
        [0.0, 1.0, 0.0],   # 4
        [0.0, 1.0, 1.0],   # 5
        [1.0, 1.0, 1.0],   # 6
        [1.0, 1.0, 0.0],   # 7
        [0.5, 0.5, 0.5],   # 8 (center)
        [0.25, 0.75, 0.5],  # 9 (inside)
        [2.0, 3.0, 2.0],   # 10 (outside)
    ], dtype=np.float64)

    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    return coords, conn


def test_inter_type16_entity_resolution_element_group():
    """Verify entity resolution using modern ElementGroup, node_groups, egroups."""
    coords, conn = _make_unit_cube_mesh()
    model = Model()
    model.x0 = coords.copy()
    model.x = coords.copy()
    model.v = np.zeros_like(coords)
    model.bricks = ElementGroup(
        ids=np.array([101], dtype=np.int64),
        conn=conn,
        part=np.array([1], dtype=np.int64)
    )

    # Group of secondary nodes (nodes 8 and 9)
    ng = NodeGroup(id=5, title="SecondaryNodes")
    ng.node_idx = np.array([8, 9], dtype=np.int64)
    model.node_groups[5] = ng

    # Group of master bricks (brick 0) via members
    eg = EntityGroup(id=15, family="BRIC", title="MasterBricks")
    eg.members = [("bricks", np.array([0], dtype=np.int64))]
    model.egroups["GRBRIC"] = {15: eg}

    itf = Interface(id=1, type=16, grnod_id=5, grbric_id1=15, itied=1, lagmul=True)
    log = MessageLog()
    c16 = LagmulType16(itf, model, log)

    assert len(c16.snode) == 2
    assert np.array_equal(c16.snode, [8, 9])
    assert len(c16.bricks) == 1
    assert np.array_equal(c16.bricks[0], [0, 1, 2, 3, 4, 5, 6, 7])
    assert c16.tied is True
    assert len(c16.active_nodes) == 2


def test_inter_type16_entity_resolution_legacy_ixs():
    """Verify entity resolution with legacy object having .ixs attribute."""
    coords, conn = _make_unit_cube_mesh()
    model = Model()
    model.x0 = coords.copy()

    class LegacyBricks:
        def __init__(self, ixs):
            self.ixs = ixs
            self.n = len(ixs)

    model.bricks = LegacyBricks(conn)
    ng = NodeGroup(id=7)
    ng.node_idx = np.array([8])
    model.node_groups[7] = ng

    itf = Interface(id=1, type=16, grnod_id=7, grbric_id1=0, itied=1, lagmul=True)
    c16 = LagmulType16(itf, model)
    assert len(c16.bricks) == 1
    assert len(c16.active_nodes) == 1


def test_inter_type16_entity_resolution_ndarray():
    """Verify entity resolution with model.bricks as direct ndarray."""
    coords, conn = _make_unit_cube_mesh()
    model = Model()
    model.x0 = coords.copy()
    model.bricks = conn

    ng = NodeGroup(id=3)
    ng.node_ids = [999]  # mapped via node_id_to_idx
    model.node_id_to_idx = {999: 8}
    model.node_groups[3] = ng

    itf = Interface(id=1, type=16, grnod_id=3, grbric_id1=0, itied=1, lagmul=True)
    c16 = LagmulType16(itf, model)
    assert len(c16.snode) == 1
    assert c16.snode[0] == 8
    assert len(c16.bricks) == 1
    assert len(c16.active_nodes) == 1


def test_inter_type16_entity_resolution_overrides():
    """Verify entity resolution with direct overrides on Interface."""
    coords, conn = _make_unit_cube_mesh()
    model = Model()
    model.x0 = coords.copy()

    itf = Interface(id=1, type=16, itied=1, lagmul=True)
    itf.secondary_nodes = [8]
    itf.master_bricks = conn

    c16 = LagmulType16(itf, model)
    assert len(c16.snode) == 1
    assert c16.snode[0] == 8
    assert len(c16.bricks) == 1
    assert len(c16.active_nodes) == 1


def test_inter_type16_tied_projection_center():
    """Verify tied mode projection of secondary node at center (0.5, 0.5, 0.5)."""
    coords, conn = _make_unit_cube_mesh()
    model = Model()
    model.x0 = coords.copy()
    model.bricks = ElementGroup(
        ids=np.array([101]),
        conn=conn,
        part=np.array([1])
    )
    ng = NodeGroup(id=1)
    ng.node_idx = np.array([8])  # node 8 is at (0.5, 0.5, 0.5)
    model.node_groups[1] = ng

    itf = Interface(id=1, type=16, grnod_id=1, grbric_id1=0, itied=1, lagmul=True)
    c16 = LagmulType16(itf, model)

    assert len(c16.active_nodes) == 1
    assert c16.active_nodes[0] == 8
    # At center of unit cube, natural coords should be (0, 0, 0)
    assert np.allclose(c16.active_rst[0], [0.0, 0.0, 0.0], atol=1e-4)
    # Shape functions at center should all be 1/8 = 0.125
    assert np.allclose(c16.active_N[0], np.full(8, 0.125), atol=1e-4)
    assert np.isclose(np.sum(c16.active_N[0]), 1.0)


def test_inter_type16_tied_projection_arbitrary():
    """Verify tied projection of secondary node at (0.25, 0.75, 0.5)."""
    coords, conn = _make_unit_cube_mesh()
    model = Model()
    model.x0 = coords.copy()
    model.bricks = ElementGroup(
        ids=np.array([101]),
        conn=conn,
        part=np.array([1])
    )
    ng = NodeGroup(id=1)
    ng.node_idx = np.array([9])  # node 9 is at (0.25, 0.75, 0.5)
    model.node_groups[1] = ng

    itf = Interface(id=1, type=16, grnod_id=1, grbric_id1=0, itied=1, lagmul=True)
    c16 = LagmulType16(itf, model)

    assert len(c16.active_nodes) == 1
    # x = 0.25 -> r = -0.5; y = 0.75 -> s = +0.5; z = 0.5 -> t = 0.0
    assert np.allclose(c16.active_rst[0], [-0.5, 0.5, 0.0], atol=1e-4)
    assert np.isclose(np.sum(c16.active_N[0]), 1.0)

    # Reconstructed position using shape functions must match node 9 coordinate
    interp_pos = np.sum(c16.active_N[0][:, None] * coords[conn[0]], axis=0)
    assert np.allclose(interp_pos, coords[9], atol=1e-5)


def test_inter_type16_multi_brick_selection():
    """Verify projection selects the correct brick in a multi-brick mesh."""
    coords, conn = _make_unit_cube_mesh()
    # Add a second brick [1, 2] x [0, 1] x [0, 1]
    # Shift X by 1.0 for new 8 nodes:
    brick2_nodes = np.arange(len(coords), len(coords) + 8)
    coords2 = coords[:8] + np.array([1.0, 0.0, 0.0])
    # Add a secondary node at (1.5, 0.5, 0.5) inside brick 2
    sec2_node = len(coords) + 8
    coords_sec2 = np.array([[1.5, 0.5, 0.5]])

    all_coords = np.vstack([coords, coords2, coords_sec2])
    all_conn = np.vstack([conn, brick2_nodes])

    model = Model()
    model.x0 = all_coords
    model.bricks = ElementGroup(
        ids=np.array([1, 2]),
        conn=all_conn,
        part=np.array([1, 1])
    )
    ng = NodeGroup(id=1)
    ng.node_idx = np.array([sec2_node])
    model.node_groups[1] = ng

    itf = Interface(id=1, type=16, grnod_id=1, grbric_id1=0, itied=1, lagmul=True)
    c16 = LagmulType16(itf, model)

    assert len(c16.active_nodes) == 1
    assert c16.active_nodes[0] == sec2_node
    # Should project to brick 1 (the second brick)
    assert np.array_equal(c16.active_bricks[0], brick2_nodes)
    assert np.allclose(c16.active_rst[0], [0.0, 0.0, 0.0], atol=1e-4)


def test_inter_type16_tied_l_matrix_conservation():
    """Verify tied mode L matrix rows: 3 equations per node, exact partition of unity."""
    coords, conn = _make_unit_cube_mesh()
    model = Model()
    model.x0 = coords.copy()
    model.bricks = ElementGroup(
        ids=np.array([101]),
        conn=conn,
        part=np.array([1])
    )
    ng = NodeGroup(id=1)
    ng.node_idx = np.array([8, 9])
    model.node_groups[1] = ng

    itf = Interface(id=1, type=16, grnod_id=1, grbric_id1=0, itied=1, lagmul=True)
    c16 = LagmulType16(itf, model)

    data, nodes, dofs, eq_ids, n_rows = c16.generate_l_matrix()
    assert n_rows == 6  # 2 nodes * 3 DOFs
    assert len(data) == 6 * 9  # 9 entries per row (8 master + 1 secondary)

    # For each equation, verify sum of coefficients is 0: sum(N_k) - 1.0 == 0
    for eq in range(n_rows):
        mask = (eq_ids == eq)
        eq_data = data[mask]
        eq_nodes = nodes[mask]
        eq_dofs = dofs[mask]
        # All entries in equation must share the same DOF
        assert len(np.unique(eq_dofs)) == 1
        # Row sum must be 0 (linear momentum balance)
        assert np.isclose(np.sum(eq_data), 0.0, atol=1e-6)
        # Secondary node entry must be -1.0
        sec_node = c16.active_nodes[eq // 3]
        sec_mask = (eq_nodes == sec_node)
        assert np.sum(sec_mask) == 1
        assert np.isclose(eq_data[sec_mask][0], -1.0)


def test_inter_type16_sliding_mode_normal_and_velocity():
    """Verify sliding mode (ITIED=0): normal calculation and S * VN <= 0 contact condition."""
    coords, conn = _make_unit_cube_mesh()
    # Secondary node 11 near top face (y = 1, s = +1)
    sec_coord = np.array([0.5, 0.99, 0.5])
    coords = np.vstack([coords, sec_coord])
    sec_idx = len(coords) - 1

    model = Model()
    model.x0 = coords.copy()
    model.x = coords.copy()
    model.v = np.zeros_like(coords)
    model.bricks = ElementGroup(
        ids=np.array([101]),
        conn=conn,
        part=np.array([1])
    )
    ng = NodeGroup(id=1)
    ng.node_idx = np.array([sec_idx])
    model.node_groups[1] = ng

    itf = Interface(id=1, type=16, grnod_id=1, grbric_id1=0, itied=0, lagmul=True)
    c16 = LagmulType16(itf, model)
    assert c16.tied is False

    # Case A: Secondary node moving inward along -Y (penetrating):
    # s ~ +0.98 > 0. Normal at face s=+1 points along +Y.
    # v_sec = (0, -1, 0), v_master = 0 -> v_rel = (0, -1, 0).
    # VN = n . v_rel = -1.0. S * VN = 0.98 * (-1.0) <= 0 -> Penetration!
    model.v[sec_idx] = np.array([0.0, -1.0, 0.0])
    data, nodes, dofs, eq_ids, n_rows = c16.generate_l_matrix()
    assert n_rows == 1
    assert len(data) == 27  # 9 nodes * 3 DOFs

    # Verify normal points along +Y
    assert np.allclose(c16.active_normals[0], [0.0, 1.0, 0.0], atol=1e-3)

    # Momentum conservation in each DOF:
    for d in range(3):
        dof_mask = (dofs == d)
        assert np.isclose(np.sum(data[dof_mask]), 0.0, atol=1e-6)

    # Case B: Secondary node moving outward along +Y (separating):
    # VN = +1.0. S * VN > 0 -> No contact constraint!
    model.v[sec_idx] = np.array([0.0, 1.0, 0.0])
    data2, nodes2, dofs2, eq_ids2, n_rows2 = c16.generate_l_matrix()
    assert n_rows2 == 0
    assert len(data2) == 0


def test_inter_type16_outside_node_warning():
    """Verify warning logged when tied secondary node fails to project."""
    coords, conn = _make_unit_cube_mesh()
    model = Model()
    model.x0 = coords.copy()
    model.bricks = ElementGroup(
        ids=np.array([101]),
        conn=conn,
        part=np.array([1])
    )
    ng = NodeGroup(id=1)
    ng.node_idx = np.array([10])  # node 10 is at (2.0, 3.0, 2.0)
    model.node_groups[1] = ng

    log = MessageLog()
    itf = Interface(id=42, type=16, grnod_id=1, grbric_id1=0, itied=1, lagmul=True)
    c16 = LagmulType16(itf, model, log)

    assert len(c16.active_nodes) == 0
    # Warning should have been recorded in log
    assert any("failed to project" in w for w in log.warnings)


def test_inter_type16_degenerate_brick_protection():
    """Verify degenerate (zero volume / flat) brick does not cause LinAlgError."""
    # Degenerate brick with all nodes at origin
    coords = np.zeros((10, 3), dtype=np.float64)
    coords[8] = np.array([0.0, 0.0, 0.0])
    conn = np.zeros((1, 8), dtype=np.int64)

    model = Model()
    model.x0 = coords
    model.bricks = ElementGroup(
        ids=np.array([1]),
        conn=conn,
        part=np.array([1])
    )
    ng = NodeGroup(id=1)
    ng.node_idx = np.array([8])
    model.node_groups[1] = ng

    itf = Interface(id=1, type=16, grnod_id=1, grbric_id1=0, itied=1, lagmul=True)
    # Should not raise LinAlgError
    c16 = LagmulType16(itf, model)
    data, nodes, dofs, eq_ids, n_rows = c16.generate_l_matrix()
    # Gracefully executes
    assert isinstance(n_rows, int)


def test_inter_type16_empty_fallbacks():
    """Verify graceful degradation when nodes or bricks are empty."""
    model = Model()
    itf = Interface(id=1, type=16, grnod_id=99, grbric_id1=99, itied=1, lagmul=True)
    c16 = LagmulType16(itf, model)
    assert len(c16.snode) == 0
    assert len(c16.bricks) == 0
    data, nodes, dofs, eq_ids, n_rows = c16.generate_l_matrix()
    assert n_rows == 0
    assert len(data) == 0


def test_inter_type16_lagmul_solver_integration():
    """Verify end-to-end integration with LagmulSolver for both force transfer and velocity cleanup."""
    coords, conn = _make_unit_cube_mesh()
    model = Model()
    model.x0 = coords.copy()
    model.x = coords.copy()
    model.v = np.zeros_like(coords)
    model.mass = np.ones(len(coords))
    model.bricks = ElementGroup(
        ids=np.array([101]),
        conn=conn,
        part=np.array([1])
    )
    ng = NodeGroup(id=1)
    ng.node_idx = np.array([8])  # center node
    model.node_groups[1] = ng

    itf = Interface(id=1, type=16, grnod_id=1, grbric_id1=0, itied=1, lagmul=True)
    model.interfaces.append(itf)

    log = MessageLog()
    solver = LagmulSolver(model, None, log)
    assert len(solver) == 1

    # Stage 1: transfer_forces
    fint = np.zeros_like(coords)
    fcont = np.zeros_like(coords)
    fext = np.zeros_like(coords)
    mint = np.zeros_like(coords)
    inv_mass = 1.0 / model.mass
    inv_inertia = np.zeros_like(coords)
    fext[8] = np.array([10.0, 0.0, 0.0])

    solver.transfer_forces(fint, fcont, fext, mint, inv_mass, inv_inertia, dt=1e-4)

    # Constraint forces should balance the relative motion:
    # fint[8] should have negative force balancing fext[8], and master nodes should share +10.0
    total_force = np.sum(fint, axis=0)
    assert np.allclose(total_force, [0.0, 0.0, 0.0], atol=1e-4)
    assert fint[8, 0] < 0.0
    assert np.sum(fint[0:8, 0]) > 0.0

    # Stage 2: enforce velocity projection
    model.v[8] = np.array([5.0, 0.0, 0.0])
    model.v[0:8] = 0.0
    vr = np.zeros_like(coords)
    solver.enforce(model.v, vr, inv_mass, inv_inertia)

    # After enforce, relative velocity at center should be projected to 0
    # v_center = sum(N_k * v_master_k) should match v_sec
    v_mas_center = np.mean(model.v[0:8], axis=0)
    assert np.allclose(model.v[8], v_mas_center, atol=1e-4)
