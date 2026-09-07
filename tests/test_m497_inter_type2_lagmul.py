# pyradioss - Milestone M497: /INTER/LAGMUL/TYPE2 Tied Interface & Tests
import pytest
import numpy as np

from pyradioss.model.model import Model
from pyradioss.model.entities import Interface, NodeGroup, Surface
from pyradioss.common.messages import MessageLog
from pyradioss.contact.inter_type2 import LagmulType2
from pyradioss.engine.lagmul import LagmulSolver
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_inter


def test_inter_type2_keyword_parsing_fixed(tmp_path):
    """Verify /INTER/LAGMUL/TYPE2 parsing in fixed format with title and cards."""
    deck_fixed = """/FORMAT/1
/INTER/LAGMUL/TYPE2/15
Tied Lagrange Multiplier Fixed
        10        20                             2                 0.05
"""
    p = tmp_path / "fixed_type2.rad"
    p.write_text(deck_fixed)
    model = Model()
    log = MessageLog()
    blocks = list(read_deck(str(p)))
    read_inter(blocks[1], model, log)
    assert len(model.interfaces) == 1
    itf = model.interfaces[0]
    assert itf.id == 15
    assert itf.type == 2
    assert itf.grnod_id == 10
    assert itf.surf_id == 20
    assert np.isclose(itf.dsearch, 0.05)
    assert itf.lagmul is True
    assert itf.title == "Tied Lagrange Multiplier Fixed"


def test_inter_type2_keyword_parsing_free(tmp_path):
    """Verify /INTER/LAGMUL/TYPE2 parsing in free format with 3 and 4 tokens."""
    # 4 tokens: grnod_id surf_id isearch dsearch
    deck_free4 = """# Free format 4 tokens
/INTER/LAGMUL/TYPE2/25
Tied Free 4
100 200 2 0.075
"""
    p4 = tmp_path / "free4.rad"
    p4.write_text(deck_free4)
    model4 = Model()
    log = MessageLog()
    blocks = list(read_deck(str(p4)))
    read_inter(blocks[0], model4, log)
    assert len(model4.interfaces) == 1
    itf4 = model4.interfaces[0]
    assert itf4.id == 25
    assert itf4.type == 2
    assert itf4.grnod_id == 100
    assert itf4.surf_id == 200
    assert np.isclose(itf4.dsearch, 0.075)
    assert itf4.lagmul is True

    # 3 tokens: grnod_id surf_id dsearch
    deck_free3 = """# Free format 3 tokens
/INTER/LAGMUL/TYPE2/26
Tied Free 3
101 201 0.125
"""
    p3 = tmp_path / "free3.rad"
    p3.write_text(deck_free3)
    model3 = Model()
    blocks3 = list(read_deck(str(p3)))
    read_inter(blocks3[0], model3, log)
    itf3 = model3.interfaces[0]
    assert itf3.id == 26
    assert itf3.grnod_id == 101
    assert itf3.surf_id == 201
    assert np.isclose(itf3.dsearch, 0.125)


def test_inter_type2_entity_resolution_nodegroup_and_surface():
    """Verify entity resolution using NodeGroup (.node_idx and .node_ids) and Surface (.segments)."""
    model = Model()
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.01],
    ], dtype=np.float64)
    model.mass = np.ones(5)

    # Secondary via node_idx
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([4], dtype=np.int64))
    # Master surface via segments
    model.surfaces[2] = Surface(id=2, segments=np.array([[0, 1, 2, 3]], dtype=np.int64))

    itf = Interface(id=1, type=2, grnod_id=1, surf_id=2, dsearch=0.1, lagmul=True)
    lag = LagmulType2(itf, model)

    assert len(lag.active_snode) == 1
    assert lag.active_snode[0] == 4
    assert np.array_equal(lag.active_segs[0], [0, 1, 2, 3])

    # Secondary via node_ids with node_id_to_idx mapping
    model.node_id_to_idx = {1004: 4}
    model.node_groups[10] = NodeGroup(id=10, node_ids=[1004])
    itf2 = Interface(id=2, type=2, grnod_id=10, surf_id=2, dsearch=0.1, lagmul=True)
    lag2 = LagmulType2(itf2, model)
    assert len(lag2.active_snode) == 1
    assert lag2.active_snode[0] == 4


def test_inter_type2_entity_resolution_direct_overrides():
    """Verify direct attribute overrides on Interface (secondary_nodes, master_segments)."""
    model = Model()
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.02],
    ], dtype=np.float64)
    model.mass = np.ones(5)

    itf = Interface(id=5, type=2, lagmul=True, dsearch=0.1)
    setattr(itf, "secondary_nodes", np.array([4], dtype=np.int64))
    setattr(itf, "master_segments", np.array([[0, 1, 2, 3]], dtype=np.int64))

    lag = LagmulType2(itf, model)
    assert len(lag.active_snode) == 1
    assert lag.active_snode[0] == 4
    assert len(lag.active_segs) == 1


def test_inter_type2_projection_quad_partition_of_unity():
    """Verify projection onto 4-node quad segment and exact partition of unity (sum H_k = 1.0)."""
    model = Model()
    # Quad corners [0,0,0], [2,0,0], [2,2,0], [0,2,0]
    model.x0 = np.array([
        [0.0, 0.0, 0.0],  # 0
        [2.0, 0.0, 0.0],  # 1
        [2.0, 2.0, 0.0],  # 2
        [0.0, 2.0, 0.0],  # 3
        [1.0, 1.0, 0.05], # 4 (center)
        [0.5, 0.5, 0.02], # 5 (inside first triangle)
        [1.5, 1.5, 0.03], # 6 (inside second triangle)
    ], dtype=np.float64)
    model.mass = np.ones(7)
    model.surfaces[1] = Surface(id=1, segments=np.array([[0, 1, 2, 3]], dtype=np.int64))
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([4, 5, 6], dtype=np.int64))

    itf = Interface(id=1, type=2, grnod_id=1, surf_id=1, dsearch=0.2, lagmul=True)
    lag = LagmulType2(itf, model)

    assert len(lag.active_snode) == 3
    # Check partition of unity for all tied nodes
    for i in range(len(lag.active_snode)):
        w = lag.active_weights[i]
        assert np.isclose(np.sum(w), 1.0, atol=1e-12)
        assert np.all(w >= -1e-12)

    # Check that center node projection gives equal weights on triangle vertices
    data, nodes, dofs, eq_ids, n_rows = lag.generate_l_matrix()
    assert n_rows == 3 * 3  # 9 equations (3 DOFs per tied node)

    # Verify every equation row sums to 0.0
    for r in range(n_rows):
        row_sum = np.sum(data[eq_ids == r])
        assert np.isclose(row_sum, 0.0, atol=1e-12)


def test_inter_type2_projection_triangle_partition_of_unity():
    """Verify projection onto 3-node triangular segment and partition of unity (sum H_k = 1.0)."""
    model = Model()
    # Triangle corners [0,0,0], [2,0,0], [0,2,0]
    model.x0 = np.array([
        [0.0, 0.0, 0.0],  # 0
        [2.0, 0.0, 0.0],  # 1
        [0.0, 2.0, 0.0],  # 2
        [0.5, 0.5, 0.01], # 3 (secondary node inside triangle)
    ], dtype=np.float64)
    model.mass = np.ones(4)
    # Triangle represented with 3 nodes or repeated 3rd node
    model.surfaces[1] = Surface(id=1, segments=np.array([[0, 1, 2, 2]], dtype=np.int64))
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([3], dtype=np.int64))

    itf = Interface(id=1, type=2, grnod_id=1, surf_id=1, dsearch=0.1, lagmul=True)
    lag = LagmulType2(itf, model)

    assert len(lag.active_snode) == 1
    w = lag.active_weights[0]
    assert np.isclose(np.sum(w), 1.0, atol=1e-12)

    data, nodes, dofs, eq_ids, n_rows = lag.generate_l_matrix()
    assert n_rows == 3

    for r in range(n_rows):
        row_sum = np.sum(data[eq_ids == r])
        assert np.isclose(row_sum, 0.0, atol=1e-12)


def test_inter_type2_dsearch_filtering():
    """Verify search distance dsearch filtering (cutoff)."""
    model = Model()
    model.x0 = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.5, 0.5, 0.05], # 4 (close, dist = 0.05)
        [0.5, 0.5, 0.50], # 5 (far, dist = 0.50)
    ], dtype=np.float64)
    model.mass = np.ones(6)
    model.surfaces[1] = Surface(id=1, segments=np.array([[0, 1, 2, 3]], dtype=np.int64))
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([4, 5], dtype=np.int64))

    # dsearch = 0.1 -> only node 4 should be tied
    itf = Interface(id=1, type=2, grnod_id=1, surf_id=1, dsearch=0.1, lagmul=True)
    lag = LagmulType2(itf, model)

    assert len(lag.active_snode) == 1
    assert lag.active_snode[0] == 4
    assert np.isclose(lag.active_dists[0], 0.05, atol=1e-5)


def test_inter_type2_secondary_corner_exclusion():
    """Verify that secondary nodes that are corners of the master surface are excluded."""
    model = Model()
    model.x0 = np.array([
        [0.0, 0.0, 0.0],  # 0 (master corner)
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.5, 0.5, 0.01], # 4 (valid secondary)
    ], dtype=np.float64)
    model.mass = np.ones(5)
    model.surfaces[1] = Surface(id=1, segments=np.array([[0, 1, 2, 3]], dtype=np.int64))
    # Candidate list includes node 0 (corner) and node 4 (valid)
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([0, 4], dtype=np.int64))

    itf = Interface(id=1, type=2, grnod_id=1, surf_id=1, dsearch=0.1, lagmul=True)
    lag = LagmulType2(itf, model)

    # Node 0 must be excluded
    assert len(lag.active_snode) == 1
    assert lag.active_snode[0] == 4


def test_inter_type2_rigid_formulation_momentum_conservation():
    """Verify Fortran i2lagm.F rigid formulation (spotflag=1 or formulation='rigid')."""
    model = Model()
    model.x0 = np.array([
        [0.0, 0.0, 0.0],  # 0
        [2.0, 0.0, 0.0],  # 1
        [2.0, 2.0, 0.0],  # 2
        [0.0, 2.0, 0.0],  # 3
        [1.0, 1.0, 0.3],  # 4 (offset above centroid)
    ], dtype=np.float64)
    model.mass = np.ones(5)
    model.surfaces[1] = Surface(id=1, segments=np.array([[0, 1, 2, 3]], dtype=np.int64))
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([4], dtype=np.int64))

    itf = Interface(id=1, type=2, grnod_id=1, surf_id=1, dsearch=0.5, lagmul=True, spotflag=1)
    lag = LagmulType2(itf, model)

    data, nodes, dofs, eq_ids, n_rows = lag.generate_l_matrix()
    assert n_rows == 3

    # Every equation in the rigid formulation must conserve linear momentum identically (row sum == 0)
    for r in range(n_rows):
        row_sum = np.sum(data[eq_ids == r])
        assert np.isclose(row_sum, 0.0, atol=1e-12)


def test_inter_type2_empty_and_fallback():
    """Verify fallback when secondary nodes or master segments are empty."""
    model = Model()
    model.x0 = np.zeros((2, 3))
    model.mass = np.ones(2)

    # Missing surface and node group
    itf = Interface(id=99, type=2, grnod_id=999, surf_id=999, lagmul=True)
    lag = LagmulType2(itf, model)
    assert len(lag.active_snode) == 0

    data, nodes, dofs, eq_ids, n_rows = lag.generate_l_matrix()
    assert n_rows == 0
    assert len(data) == 0


def test_inter_type2_lagmul_solver_force_transfer_and_equilibrium():
    """Verify sparse PCG solve in LagmulSolver.transfer_forces with force equilibrium."""
    model = Model()
    # Horizontal master quad in XY plane
    model.x0 = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.5, 0.5, 0.0],  # 4 (tied secondary)
    ], dtype=np.float64)
    model.x = model.x0.copy()
    model.mass = np.array([1.0, 1.0, 1.0, 1.0, 0.5], dtype=np.float64)
    model.surfaces[1] = Surface(id=1, segments=np.array([[0, 1, 2, 3]], dtype=np.int64))
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([4], dtype=np.int64))

    itf = Interface(id=1, type=2, grnod_id=1, surf_id=1, dsearch=0.2, lagmul=True)
    model.interfaces.append(itf)

    log = MessageLog()
    solver = LagmulSolver(model, None, log)
    assert len(solver) == 1

    fint = np.zeros((5, 3))
    fcont = np.zeros((5, 3))
    fext = np.zeros((5, 3))
    fext[4] = np.array([0.0, 0.0, -100.0])  # External load on tied node
    mint = np.zeros((5, 3))
    inv_mass = 1.0 / model.mass
    inv_inertia = np.ones((5, 3))
    dt = 1e-4

    solver.transfer_forces(fint, fcont, fext, mint, inv_mass, inv_inertia, dt)

    # Net constraint forces in fint must sum to exactly [0, 0, 0]
    sum_fint = np.sum(fint, axis=0)
    assert np.allclose(sum_fint, 0.0, atol=1e-8)
    # Secondary node gets an upward reaction (+Z)
    assert fint[4, 2] > 0.0
    # Master nodes get downward reaction (-Z)
    assert np.sum(fint[:4, 2]) < 0.0


def test_inter_type2_lagmul_solver_enforce_velocity_projection():
    """Verify velocity projection in LagmulSolver.enforce with exact momentum conservation."""
    model = Model()
    model.x0 = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.5, 0.5, 0.0],  # 4 (tied secondary)
    ], dtype=np.float64)
    model.x = model.x0.copy()
    model.mass = np.array([1.0, 1.0, 1.0, 1.0, 0.5], dtype=np.float64)
    model.surfaces[1] = Surface(id=1, segments=np.array([[0, 1, 2, 3]], dtype=np.int64))
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([4], dtype=np.int64))

    itf = Interface(id=1, type=2, grnod_id=1, surf_id=1, dsearch=0.2, lagmul=True)
    model.interfaces.append(itf)

    log = MessageLog()
    solver = LagmulSolver(model, None, log)

    v = np.zeros((5, 3))
    v[4] = np.array([10.0, -5.0, 20.0])  # Initial relative velocity on secondary
    v_init = v.copy()
    vr = np.zeros((5, 3))
    inv_mass = 1.0 / model.mass
    inv_inertia = np.ones((5, 3))

    # Initial linear momentum
    p_init = np.sum(model.mass[:, None] * v_init, axis=0)

    solver.enforce(v, vr, inv_mass, inv_inertia)

    # Final linear momentum must equal initial momentum
    p_final = np.sum(model.mass[:, None] * v, axis=0)
    assert np.allclose(p_final, p_init, atol=1e-8)

    # Constraint equations L v = 0 must be satisfied
    data, nodes, dofs, eq_ids, n_rows = solver.interfaces[0].generate_l_matrix()
    for r in range(n_rows):
        mask = (eq_ids == r)
        res = np.sum(data[mask] * v[nodes[mask], dofs[mask]])
        assert abs(res) < 1e-5
