"""
Comprehensive test suite for Advanced 3D Solid Finite Element Technology:
- solid_hexa8_full (Isolid=2: 8-node 2x2x2 Gauss full integration)
- solid_hexa8_eas (Isolid=17/18: Enhanced Assumed Strain hex)
- solid_shell_ha8 (8-node HA8 solid shell with ANS)
- solid_cohesive (8-node cohesive zone interface solid)
- solid_tetra4_sfem (Itetra4=3: 4-node smoothed FEM tetra)
- solid_penta6_heph (6-node HEPH wedge with physical stabilization)
- solid_pyra5 (5-node degenerate pyramid solid with FAC=1/9 scaling)
"""

import numpy as np
import pytest

from pyradioss.elements import (
    solid_cohesive,
    solid_hexa8_eas,
    solid_hexa8_full,
    solid_penta6_heph,
    solid_pyra5,
    solid_shell_ha8,
    solid_tetra4_sfem,
)
from pyradioss.model.entities import Material, Property
from pyradioss.model.model import ElementGroup, Model


class MockLog:
    def info(self, msg, cat=""): pass
    def error(self, msg, cat=""): pass
    def warning(self, msg, cat=""): pass


def _make_cube_hex_model():
    """Unit cube [0, 1]^3 with 8 nodes."""
    model = Model()
    model.node_ids = np.arange(1, 9, dtype=np.int64)
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float64)
    model.x = model.x0.copy()
    mat = Material(id=1, law=1, rho0=7800.0, title="Steel", params={"E": 2.1e11, "nu": 0.3})
    prop = Property(id=1, type=14, title="SolidProp", params={"qa": 1.1, "qb": 0.05})
    return model, mat, prop


# ============================================================================
# 1. 8-node Fully-Integrated Hexahedron (solid_hexa8_full, Isolid=2)
# ============================================================================

def test_hexa8_full_init_and_rigid_body():
    model, mat, prop = _make_cube_hex_model()
    conn = np.arange(8, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state["slices"] = [(slice(0, 1), mat, prop)]

    node_idx, mass_c, _ = solid_hexa8_full.init_group(group, model, MockLog())
    assert len(node_idx) == 8
    # Mass = 7800 * 1.0 = 7800. Each node gets 7800 / 8 = 975
    assert np.isclose(group.state["vol0"][0], 1.0)
    assert np.isclose(group.state["mass"][0], 7800.0)
    assert np.allclose(mass_c, 975.0)

    # Rigid body translation test: fint must be zero
    v = np.ones((8, 3)) * 50.0
    fint = np.zeros((8, 3))
    dt_crit = solid_hexa8_full.forces(group, model.x, v, None, 1.0e-6, fint, None)
    assert np.allclose(fint, 0.0, atol=1e-6)
    assert dt_crit[0] > 0.0


def test_hexa8_full_pure_tension():
    model, mat, prop = _make_cube_hex_model()
    conn = np.arange(8, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_hexa8_full.init_group(group, model, MockLog())

    # Velocity field stretching cube along X: vx(x=1) = +1.0, vx(x=0) = 0.0
    v = np.zeros((8, 3))
    v[[1, 2, 5, 6], 0] = 1.0  # right face moving right
    fint = np.zeros((8, 3))
    dt = 1.0e-6

    dt_crit = solid_hexa8_full.forces(group, model.x, v, None, dt, fint, None)
    assert dt_crit[0] > 0.0

    # Total internal forces must be in self-equilibrium (sum of all nodal forces is zero)
    assert np.allclose(np.sum(fint, axis=0), 0.0, atol=1e-4)
    # Right face must feel negative X force (resisting tensile expansion)
    assert np.all(fint[[1, 2, 5, 6], 0] < 0.0)
    # Left face must feel positive X force
    assert np.all(fint[[0, 3, 4, 7], 0] > 0.0)
    # Internal energy must be positive
    assert group.state["eint"][0] > 0.0


# ============================================================================
# 2. 8-node Enhanced Assumed Strain Hexahedron (solid_hexa8_eas, Isolid=17/18)
# ============================================================================

def test_hexa8_eas_init_and_tension():
    model, mat, prop = _make_cube_hex_model()
    conn = np.arange(8, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state["slices"] = [(slice(0, 1), mat, prop)]

    node_idx, mass_c, _ = solid_hexa8_eas.init_group(group, model, MockLog())
    assert len(node_idx) == 8
    assert "pij" in group.state
    assert group.state["pij"].shape == (1, 9)

    v = np.zeros((8, 3))
    v[[1, 2, 5, 6], 0] = 1.0
    fint = np.zeros((8, 3))
    dt = 1.0e-6

    dt_crit = solid_hexa8_eas.forces(group, model.x, v, None, dt, fint, None)
    assert dt_crit[0] > 0.0
    assert np.allclose(np.sum(fint, axis=0), 0.0, atol=1e-4)
    assert np.all(fint[[1, 2, 5, 6], 0] < 0.0)
    assert group.state["eint"][0] > 0.0


# ============================================================================
# 3. 8-node Solid-Shell HA8 with ANS (solid_shell_ha8)
# ============================================================================

def test_solid_shell_ha8_ans():
    model, mat, prop = _make_cube_hex_model()
    conn = np.arange(8, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state["slices"] = [(slice(0, 1), mat, prop)]

    solid_shell_ha8.init_group(group, model, MockLog())
    v = np.zeros((8, 3))
    # In-plane shear vy along X
    v[[1, 2, 5, 6], 1] = 1.0
    fint = np.zeros((8, 3))
    dt = 1.0e-6

    dt_crit = solid_shell_ha8.forces(group, model.x, v, None, dt, fint, None)
    assert dt_crit[0] > 0.0
    assert np.allclose(np.sum(fint, axis=0), 0.0, atol=1e-4)
    assert group.state["eint"][0] > 0.0


# ============================================================================
# 4. 8-node Cohesive Zone Element (solid_cohesive)
# ============================================================================

def test_cohesive_traction_separation():
    model = Model()
    model.node_ids = np.arange(1, 9, dtype=np.int64)
    # Zero initial thickness: nodes 1-4 and 5-8 at z = 0
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    model.x = model.x0.copy()

    mat = Material(id=1, law=1, rho0=1000.0, title="InterfaceMat")
    prop = Property(id=1, type=21, title="CohesiveProp", params={
        "kn": 1.0e8, "kt": 1.0e8, "sigma_max": 5.0e6, "delta_max": 1.0e-3
    })
    prop.kn = 1.0e8
    prop.kt = 1.0e8
    prop.sigma_max = 5.0e6
    prop.delta_max = 1.0e-3

    conn = np.arange(8, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state["slices"] = [(slice(0, 1), mat, prop)]

    node_idx, mass_c, _ = solid_cohesive.init_group(group, model, MockLog())
    assert len(node_idx) == 8
    # Even with zero initial physical thickness, mass and area must be non-zero
    assert group.state["area0"][0] == 1.0

    # Normal opening: pull top nodes (4..7) in +Z direction
    model.x[4:8, 2] = 1.0e-5
    fint = np.zeros((8, 3))
    dt_crit = solid_cohesive.forces(group, model.x, np.zeros((8, 3)), None, 1.0e-6, fint, None)

    # Time step must be finite and non-zero (sz_dt1 eigenvalue bound!)
    assert dt_crit[0] > 0.0
    assert not np.isinf(dt_crit[0])

    # Equilibrium: top force + bottom force == 0
    assert np.allclose(np.sum(fint, axis=0), 0.0, atol=1e-5)
    # Bottom nodes feel +Z pull, top nodes feel -Z restoring force
    assert np.all(fint[0:4, 2] > 0.0)
    assert np.all(fint[4:8, 2] < 0.0)


# ============================================================================
# 5. 4-node Smoothed FEM Tetra (solid_tetra4_sfem, Itetra4=3)
# ============================================================================

def test_tetra4_sfem_smoothing():
    model = Model()
    model.node_ids = np.arange(1, 5, dtype=np.int64)
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    model.x = model.x0.copy()

    mat = Material(id=1, law=1, rho0=7800.0, title="Steel", params={"E": 2.1e11, "nu": 0.3})
    prop = Property(id=1, type=14, title="TetraProp", params={"qa": 1.1, "qb": 0.05, "itetra4": 3})


    conn = np.arange(4, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state["slices"] = [(slice(0, 1), mat, prop)]

    node_idx, mass_c, _ = solid_tetra4_sfem.init_group(group, model, MockLog())
    assert len(node_idx) == 4
    # Tetra volume = 1/6
    assert np.isclose(group.state["vol0"][0], 1.0 / 6.0)

    # Uniform expansion
    v = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    fint = np.zeros((4, 3))
    dt_crit = solid_tetra4_sfem.forces(group, model.x, v, None, 1.0e-6, fint, None)
    assert dt_crit[0] > 0.0
    assert group.state["eint"][0] > 0.0


# ============================================================================
# 6. 6-node HEPH Wedge (solid_penta6_heph)
# ============================================================================

def test_penta6_heph_stabilization():
    model = Model()
    model.node_ids = np.arange(1, 7, dtype=np.int64)
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float64)
    model.x = model.x0.copy()

    mat = Material(id=1, law=1, rho0=7800.0, title="Steel", params={"E": 2.1e11, "nu": 0.3})
    prop = Property(id=1, type=14, title="PentaProp", params={"isolid": 24})

    conn = np.arange(6, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state["slices"] = [(slice(0, 1), mat, prop)]

    node_idx, mass_c, _ = solid_penta6_heph.init_group(group, model, MockLog())
    assert len(node_idx) == 6
    assert np.isclose(group.state["vol0"][0], 0.5)

    # Hourglass bending mode velocity: h = [1, -1, 0, -1, 1, 0]
    v = np.zeros((6, 3))
    v[:, 0] = [1.0, -1.0, 0.0, -1.0, 1.0, 0.0]
    fint = np.zeros((6, 3))
    dt = 1.0e-6

    dt_crit = solid_penta6_heph.forces(group, model.x, v, None, dt, fint, None)
    assert dt_crit[0] > 0.0
    # Hourglass stabilization must generate non-zero resisting forces
    assert not np.allclose(fint, 0.0)
    assert np.allclose(np.sum(fint, axis=0), 0.0, atol=1e-5)


# ============================================================================
# 7. 5-node Pyramid Solid (solid_pyra5, IDEGE=6)
# ============================================================================

def test_pyramid5_volume_and_timestep():
    model = Model()
    model.node_ids = np.arange(1, 6, dtype=np.int64)
    # Unit square base [0, 1]x[0, 1] at z=0, apex at (0.5, 0.5, 1.0)
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ], dtype=np.float64)
    model.x = model.x0.copy()

    mat = Material(id=1, law=1, rho0=7800.0, title="Steel", params={"E": 2.1e11, "nu": 0.3})
    prop = Property(id=1, type=14, title="PyraProp")


    conn = np.arange(5, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state["slices"] = [(slice(0, 1), mat, prop)]

    node_idx, mass_c, _ = solid_pyra5.init_group(group, model, MockLog())
    assert len(node_idx) == 5
    # Pyramid volume V = 1/3 * 1.0 * 1.0 = 1/3
    assert np.isclose(group.state["vol0"][0], 1.0 / 3.0)
    # Mass lumping: 1/8 to each of 4 base nodes, 1/2 to apex
    total_mass = 7800.0 / 3.0
    assert np.isclose(mass_c[0], total_mass / 8.0)
    assert np.isclose(mass_c[4], total_mass * 0.5)

    v = np.zeros((5, 3))
    v[4, 2] = 1.0  # apex moving upwards
    fint = np.zeros((5, 3))
    dt_crit = solid_pyra5.forces(group, model.x, v, None, 1.0e-6, fint, None)
    assert dt_crit[0] > 0.0
    assert np.allclose(np.sum(fint, axis=0), 0.0, atol=1e-5)
