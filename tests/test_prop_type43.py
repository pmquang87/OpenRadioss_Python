"""
Unit tests for /PROP/TYPE43 (/PROP/CONNECT: Solid Connector Element).

Upstream Fortran reference:
  - C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\starter\\source\\properties\\solid\\hm_read_prop43.F
  - C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\elements\\solid\\sconnect\\scoor43.F
  - C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\elements\\solid\\sconnect\\sdef43.F
  - C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\elements\\solid\\sconnect\\sfint43.F
  - C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\elements\\solid\\sconnect\\suser43.F
  - C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\elements\\solid\\sconnect\\smom43.F
  - C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\elements\\solid\\solide\\srrota3.F
  - C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\materials\\fail\\connect\\fail_connect.F

Tests:
  1. Geometry and mass initialization (8-node and 4-node).
  2. Normal and shear elastic response with damping.
  3. Combined normal-shear failure ellipsoid criterion and deactivation.
  4. Energy conservation under elastic deformation.
  5. Static equilibrium: exact linear momentum balance sum(F) = 0.
  6. Critical time step calculation.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.common.constants import EM20, EP30
from pyradioss.elements.solid_connect import (
    init_connect_type43,
    forces_connect_type43,
    init_group,
    forces,
    _compute_local_frame_8node,
    _compute_local_frame_4node,
)
from pyradioss.model.entities import PropType43, Material


class MockModel:
    """Mock model containing nodal coordinates and properties."""
    def __init__(self, x0: np.ndarray):
        self.x0 = np.asarray(x0, dtype=float)
        self.numnod = len(self.x0)
        self.prop_type43s = {}
        self.properties = {}
        self.materials = {}


class MockElementGroup:
    """Mock element group for connector elements."""
    def __init__(self, conn: np.ndarray, prop=None, mat=None):
        self.conn = np.asarray(conn, dtype=np.int64)
        self.n = len(self.conn)
        self.state = {}
        self.prop = prop
        self.mat = mat


# ============================================================================
# Helpers to build test connector geometries
# ============================================================================

def make_unit_8node_connector(origin=(0.0, 0.0, 0.0), h=0.1, length=1.0, width=1.0):
    """Create an 8-node rectangular solid connector.

    Nodes 0..3: bottom patch at z = origin[2]
    Nodes 4..7: top patch at z = origin[2] + h
    """
    ox, oy, oz = origin
    nodes = np.array([
        # Bottom face (z = oz)
        [ox, oy, oz],
        [ox + length, oy, oz],
        [ox + length, oy + width, oz],
        [ox, oy + width, oz],
        # Top face (z = oz + h)
        [ox, oy, oz + h],
        [ox + length, oy, oz + h],
        [ox + length, oy + width, oz + h],
        [ox, oy + width, oz + h],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    return nodes, conn


def make_unit_4node_connector(origin=(0.0, 0.0, 0.0), h=0.1, length=1.0):
    """Create a 4-node planar/line solid connector.

    Nodes 0..1: bottom edge at z = origin[2]
    Nodes 2..3: top edge at z = origin[2] + h
    """
    ox, oy, oz = origin
    nodes = np.array([
        # Bottom edge
        [ox, oy, oz],
        [ox + length, oy, oz],
        # Top edge
        [ox, oy, oz + h],
        [ox + length, oy, oz + h],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)
    return nodes, conn


# ============================================================================
# 1. Geometry and Mass Initialization
# ============================================================================

def test_init_8node_geometry_and_mass():
    """Verify 8-node local frame, mid-surface area, orientation, and lumped mass."""
    x0, conn = make_unit_8node_connector(origin=(0.0, 0.0, 0.0), h=0.2, length=2.0, width=3.0)
    model = MockModel(x0)
    prop = PropType43(id=1, ismstr=1, thick=0.2)
    setattr(prop, "kn", 5.0e5)
    setattr(prop, "ks", 2.0e5)
    mat = Material(id=1, law=83, rho0=7800.0)

    group = MockElementGroup(conn, prop=prop, mat=mat)
    massn = np.zeros(len(x0))
    inertn = np.zeros((len(x0), 3))

    node_idx, mass_c, inert_c = init_connect_type43(group, model, None, idx43=None, massn=massn, inertn=inertn)

    # 1. Check mid-surface area: length * width = 2.0 * 3.0 = 6.0
    st = group.state
    assert st["area0"][0] == pytest.approx(6.0, rel=1e-6)
    assert st["area"][0] == pytest.approx(6.0, rel=1e-6)

    # 2. Check local triad (E1, E2, E3):
    # E3 should be normal pointing along +Z: (0, 0, 1)
    # E1 should be in-plane along +X: (1, 0, 0)
    # E2 should be in-plane along +Y: (0, 1, 0)
    q0 = st["q"][0]
    e1, e2, e3 = q0[:, 0], q0[:, 1], q0[:, 2]
    np.testing.assert_allclose(e3, [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(e1, [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(e2, [0.0, 1.0, 0.0], atol=1e-6)

    # 3. Check mass calculation: volume = area * thick = 6.0 * 0.2 = 1.2
    # Total mass = rho0 * volume = 7800.0 * 1.2 = 9360.0
    expected_mass = 7800.0 * 6.0 * 0.2
    assert st["mass"][0] == pytest.approx(expected_mass, rel=1e-6)

    # 4. Check lumped mass distribution: 8 nodes, each receiving 1/8 of total mass
    assert len(mass_c) == 8
    np.testing.assert_allclose(mass_c, expected_mass / 8.0, rtol=1e-6)
    np.testing.assert_allclose(massn, expected_mass / 8.0, rtol=1e-6)
    assert inert_c is None

    # 5. Check property transfer
    assert st["kn"][0] == pytest.approx(5.0e5)
    assert st["ks"][0] == pytest.approx(2.0e5)
    assert st["ismstr"][0] == 1
    assert st["thick"][0] == pytest.approx(0.2)
    assert st["off"][0] == 1.0
    assert st["soft"][0] == 1.0
    assert not st["failed"][0]


def test_init_4node_geometry_and_mass():
    """Verify 4-node local frame, length/area, and lumped mass."""
    x0, conn = make_unit_4node_connector(origin=(1.0, 2.0, 3.0), h=0.05, length=4.0)
    model = MockModel(x0)
    prop = PropType43(id=2, ismstr=1, thick=0.5)
    setattr(prop, "kn", 1.0e6)
    setattr(prop, "ks", 5.0e5)
    mat = Material(id=2, law=83, rho0=2700.0)

    group = MockElementGroup(conn, prop=prop, mat=mat)
    massn = np.zeros(len(x0))

    node_idx, mass_c, _ = init_connect_type43(group, model, None, idx43=None, massn=massn)

    st = group.state
    # Length = 4.0, thick = 0.5 -> Area = 2.0
    assert st["area0"][0] == pytest.approx(4.0 * 0.5, rel=1e-6)

    # Mass = rho0 * Area * h = 2700 * 2.0 * 0.5 = 2700.0
    expected_mass = 2700.0 * (4.0 * 0.5) * 0.5
    assert st["mass"][0] == pytest.approx(expected_mass, rel=1e-6)

    # 4 nodes, each receives 1/4 of total mass
    assert len(mass_c) == 4
    np.testing.assert_allclose(mass_c, expected_mass / 4.0, rtol=1e-6)
    np.testing.assert_allclose(massn, expected_mass / 4.0, rtol=1e-6)


def test_init_empty_group_defensive():
    """Verify robust handling of empty element group."""
    conn = np.zeros((0, 8), dtype=np.int64)
    model = MockModel(np.zeros((0, 3)))
    group = MockElementGroup(conn)
    node_idx, mass_c, inert_c = init_connect_type43(group, model, None)
    assert len(node_idx) == 0
    assert len(mass_c) == 0
    assert inert_c is None


# ============================================================================
# 2. Normal and Shear Elastic Response & Damping
# ============================================================================

def test_pure_normal_elastic_tension():
    """Verify pure normal tensile opening produces exact normal force Fn = Kn * delta * Area."""
    x0, conn = make_unit_8node_connector(origin=(0.0, 0.0, 0.0), h=0.1, length=1.0, width=1.0)
    model = MockModel(x0)
    kn_val = 1.0e7
    ks_val = 5.0e6
    prop = PropType43(id=1, ismstr=1)
    setattr(prop, "kn", kn_val)
    setattr(prop, "ks", ks_val)
    group = MockElementGroup(conn, prop=prop)
    init_connect_type43(group, model, None)

    # Apply normal opening displacement delta_z = 0.01 to top nodes 4..7
    delta_z = 0.01
    x_curr = x0.copy()
    x_curr[4:8, 2] += delta_z

    # Velocity = delta_z / dt
    dt = 1.0e-3
    v_curr = np.zeros_like(x0)
    v_curr[4:8, 2] = delta_z / dt

    fint = np.zeros_like(x0)
    dt_crit = forces_connect_type43(group, x_curr, v_curr, None, dt, fint)

    st = group.state
    # Area = 1.0
    # Expected normal force Fn = Kn * delta_z * Area = 1.0e7 * 0.01 * 1.0 = 1.0e5
    expected_fn = kn_val * delta_z * 1.0
    assert st["fn"][0] == pytest.approx(expected_fn, rel=1e-5)
    assert st["fs"][0] == pytest.approx(0.0, abs=1e-6)

    # Resisting internal force on top nodes is downward (-Z)
    # Resisting force on bottom nodes is upward (+Z)
    # Divided equally among 4 nodes per face
    f_node_z = expected_fn / 4.0
    np.testing.assert_allclose(fint[0:4, 2], f_node_z, rtol=1e-5)
    np.testing.assert_allclose(fint[4:8, 2], -f_node_z, rtol=1e-5)

    # In-plane forces should be 0
    np.testing.assert_allclose(fint[:, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(fint[:, 1], 0.0, atol=1e-6)


def test_pure_shear_elastic_response():
    """Verify pure shear displacement produces exact shear force Fs = Ks * delta_s * Area."""
    x0, conn = make_unit_8node_connector(origin=(0.0, 0.0, 0.0), h=0.1, length=1.0, width=1.0)
    model = MockModel(x0)
    kn_val = 1.0e7
    ks_val = 4.0e6
    prop = PropType43(id=1, ismstr=1)
    setattr(prop, "kn", kn_val)
    setattr(prop, "ks", ks_val)
    group = MockElementGroup(conn, prop=prop)
    init_connect_type43(group, model, None)

    # Apply shear displacement delta_x = 0.005 to top nodes 4..7
    delta_x = 0.005
    x_curr = x0.copy()
    x_curr[4:8, 0] += delta_x

    dt = 1.0e-3
    v_curr = np.zeros_like(x0)
    v_curr[4:8, 0] = delta_x / dt

    fint = np.zeros_like(x0)
    forces_connect_type43(group, x_curr, v_curr, None, dt, fint)

    st = group.state
    # Expected shear force Fs = Ks * delta_x * Area = 4.0e6 * 0.005 * 1.0 = 20000.0
    expected_fs = ks_val * delta_x * 1.0
    assert st["fs"][0] == pytest.approx(expected_fs, rel=1e-5)
    assert st["fn"][0] == pytest.approx(0.0, abs=1e-6)

    # Action-reaction: top nodes pulled in -X, bottom nodes pulled in +X
    np.testing.assert_allclose(np.sum(fint[0:4, 0]), expected_fs, rtol=1e-5)
    np.testing.assert_allclose(np.sum(fint[4:8, 0]), -expected_fs, rtol=1e-5)


def test_asymmetric_compression_stiffness():
    """Verify asymmetric compression stiffness Ecomp is applied in compression."""
    x0, conn = make_unit_8node_connector(origin=(0.0, 0.0, 0.0), h=0.1, length=1.0, width=1.0)
    model = MockModel(x0)
    kn_val = 1.0e6
    ecomp_val = 5.0e6  # 5x higher in compression
    prop = PropType43(id=1, ismstr=1)
    setattr(prop, "kn", kn_val)
    setattr(prop, "ecomp", ecomp_val)
    group = MockElementGroup(conn, prop=prop)
    init_connect_type43(group, model, None)

    # Push top nodes down by 0.002
    delta_z = -0.002
    x_curr = x0.copy()
    x_curr[4:8, 2] += delta_z

    dt = 1.0e-3
    v_curr = np.zeros_like(x0)
    v_curr[4:8, 2] = delta_z / dt

    fint = np.zeros_like(x0)
    forces_connect_type43(group, x_curr, v_curr, None, dt, fint)

    st = group.state
    # Fn should be negative (compressive) and governed by Ecomp
    expected_fn = ecomp_val * delta_z * 1.0  # -10000.0
    assert st["fn"][0] == pytest.approx(expected_fn, rel=1e-5)


def test_viscous_damping():
    """Verify viscous damping force adds to stiffness force."""
    x0, conn = make_unit_8node_connector(origin=(0.0, 0.0, 0.0), h=0.1, length=1.0, width=1.0)
    model = MockModel(x0)
    kn_val = 1.0e6
    cn_val = 2.0e4
    prop = PropType43(id=1)
    setattr(prop, "kn", kn_val)
    setattr(prop, "cn", cn_val)
    group = MockElementGroup(conn, prop=prop)
    init_connect_type43(group, model, None)

    delta_z = 0.01
    dt = 1.0e-3
    vel_z = delta_z / dt  # 10.0 m/s

    x_curr = x0.copy()
    x_curr[4:8, 2] += delta_z
    v_curr = np.zeros_like(x0)
    v_curr[4:8, 2] = vel_z

    fint = np.zeros_like(x0)
    forces_connect_type43(group, x_curr, v_curr, None, dt, fint)

    st = group.state
    # Expected Fn = (Kn * delta_z + Cn * vel_z) * Area
    expected_fn = (kn_val * delta_z + cn_val * vel_z) * 1.0
    assert st["fn"][0] == pytest.approx(expected_fn, rel=1e-5)


# ============================================================================
# 3. Combined Failure Criteria and Deactivation (fail_connect.F)
# ============================================================================

def test_failure_criterion_subcritical():
    """Verify element remains active when load is below failure ellipsoid."""
    x0, conn = make_unit_8node_connector(h=0.1, length=1.0, width=1.0)
    model = MockModel(x0)
    prop = PropType43(id=1)
    setattr(prop, "kn", 1.0e6)
    setattr(prop, "ks", 1.0e6)
    setattr(prop, "fn_max", 50000.0)
    setattr(prop, "fs_max", 30000.0)
    setattr(prop, "alpha", 2.0)
    setattr(prop, "beta", 2.0)
    group = MockElementGroup(conn, prop=prop)
    init_connect_type43(group, model, None)

    # Fn = 25000 (ratio 0.5), Fs = 15000 (ratio 0.5)
    # Phi = 0.5^2 + 0.5^2 = 0.50 < 1.0
    x_curr = x0.copy()
    x_curr[4:8, 2] += 0.025  # Fn = 25000
    x_curr[4:8, 0] += 0.015  # Fs = 15000
    dt = 1.0e-3
    v_curr = np.zeros_like(x0)

    fint = np.zeros_like(x0)
    forces_connect_type43(group, x_curr, v_curr, None, dt, fint)

    st = group.state
    assert not st["failed"][0]
    assert st["off"][0] == 1.0
    assert np.linalg.norm(fint) > 0.0


def test_failure_criterion_pure_normal():
    """Verify pure normal load exceeding Fn_max triggers rupture and deletion."""
    x0, conn = make_unit_8node_connector(h=0.1, length=1.0, width=1.0)
    model = MockModel(x0)
    prop = PropType43(id=1)
    setattr(prop, "kn", 1.0e6)
    setattr(prop, "fn_max", 20000.0)
    group = MockElementGroup(conn, prop=prop)
    init_connect_type43(group, model, None)

    # Fn = 25000 > Fn_max
    x_curr = x0.copy()
    x_curr[4:8, 2] += 0.025
    dt = 1.0e-3

    fint = np.zeros_like(x0)
    forces_connect_type43(group, x_curr, None, None, dt, fint)

    st = group.state
    assert st["failed"][0]
    assert st["off"][0] == 0.0
    assert st["soft"][0] == 0.0
    # Internal forces must drop to zero upon element rupture
    np.testing.assert_allclose(fint, 0.0, atol=1e-8)


def test_failure_criterion_pure_shear():
    """Verify pure shear load exceeding Fs_max triggers rupture and deletion."""
    x0, conn = make_unit_8node_connector(h=0.1, length=1.0, width=1.0)
    model = MockModel(x0)
    prop = PropType43(id=1)
    setattr(prop, "ks", 1.0e6)
    setattr(prop, "fs_max", 15000.0)
    group = MockElementGroup(conn, prop=prop)
    init_connect_type43(group, model, None)

    # Fs = 20000 > Fs_max
    x_curr = x0.copy()
    x_curr[4:8, 0] += 0.02
    dt = 1.0e-3

    fint = np.zeros_like(x0)
    forces_connect_type43(group, x_curr, None, None, dt, fint)

    st = group.state
    assert st["failed"][0]
    assert st["off"][0] == 0.0
    np.testing.assert_allclose(fint, 0.0, atol=1e-8)


def test_failure_criterion_mixed_mode_interaction():
    """Verify mixed-mode interaction triggers failure even when individual components are below limits."""
    x0, conn = make_unit_8node_connector(h=0.1, length=1.0, width=1.0)
    model = MockModel(x0)
    prop = PropType43(id=1)
    setattr(prop, "kn", 1.0e6)
    setattr(prop, "ks", 1.0e6)
    setattr(prop, "fn_max", 10000.0)
    setattr(prop, "fs_max", 10000.0)
    setattr(prop, "alpha", 2.0)
    setattr(prop, "beta", 2.0)
    group = MockElementGroup(conn, prop=prop)
    init_connect_type43(group, model, None)

    # Fn = 8000 (0.8 of max), Fs = 8000 (0.8 of max)
    # (0.8)^2 + (0.8)^2 = 0.64 + 0.64 = 1.28 >= 1.0 -> FAILS!
    x_curr = x0.copy()
    x_curr[4:8, 2] += 0.008
    x_curr[4:8, 0] += 0.008
    dt = 1.0e-3

    fint = np.zeros_like(x0)
    forces_connect_type43(group, x_curr, None, None, dt, fint)

    st = group.state
    assert st["failed"][0]
    assert st["off"][0] == 0.0


def test_compression_does_not_cause_tensile_failure():
    """Verify compressive normal force (Fn < 0) does not trigger tensile failure."""
    x0, conn = make_unit_8node_connector(h=0.1, length=1.0, width=1.0)
    model = MockModel(x0)
    prop = PropType43(id=1)
    setattr(prop, "kn", 1.0e6)
    setattr(prop, "fn_max", 5000.0)
    group = MockElementGroup(conn, prop=prop)
    init_connect_type43(group, model, None)

    # Large compression: delta_z = -0.05 -> Fn = -50000
    x_curr = x0.copy()
    x_curr[4:8, 2] -= 0.05
    dt = 1.0e-3

    fint = np.zeros_like(x0)
    forces_connect_type43(group, x_curr, None, None, dt, fint)

    st = group.state
    # Should NOT fail from compression
    assert not st["failed"][0]
    assert st["off"][0] == 1.0


# ============================================================================
# 4. Energy Conservation
# ============================================================================

def test_energy_conservation_elastic_cycle():
    """Verify internal energy matches theoretical strain energy 0.5 * K * delta^2 * Area."""
    x0, conn = make_unit_8node_connector(h=0.1, length=2.0, width=2.0)
    model = MockModel(x0)
    kn_val = 2.0e6
    ks_val = 1.0e6
    prop = PropType43(id=1, ismstr=1)
    setattr(prop, "kn", kn_val)
    setattr(prop, "ks", ks_val)
    group = MockElementGroup(conn, prop=prop)
    init_connect_type43(group, model, None)

    # Area = 4.0
    area = 4.0
    dt = 1.0e-4
    n_steps = 10
    total_dz = 0.005
    total_dx = 0.003
    step_dz = total_dz / n_steps
    step_dx = total_dx / n_steps

    x_curr = x0.copy()
    fint = np.zeros_like(x0)

    for _ in range(n_steps):
        x_curr[4:8, 2] += step_dz
        x_curr[4:8, 0] += step_dx
        v_step = np.zeros_like(x0)
        v_step[4:8, 2] = step_dz / dt
        v_step[4:8, 0] = step_dx / dt
        fint.fill(0.0)
        forces_connect_type43(group, x_curr, v_step, None, dt, fint)

    st = group.state
    # Theoretical elastic energy:
    # E_normal = 0.5 * Kn * total_dz^2 * Area
    # E_shear  = 0.5 * Ks * total_dx^2 * Area
    expected_en = 0.5 * kn_val * (total_dz**2) * area
    expected_es = 0.5 * ks_val * (total_dx**2) * area
    expected_eint = expected_en + expected_es

    assert st["eint"][0] == pytest.approx(expected_eint, rel=1e-3)


# ============================================================================
# 5. Static Equilibrium: Exact sum(F) = 0
# ============================================================================

def test_static_equilibrium_8node_arbitrary_pose():
    """Verify sum(F) = 0 for an 8-node connector in arbitrary 3D rotated orientation."""
    x0, conn = make_unit_8node_connector(origin=(1.5, -2.0, 0.5), h=0.2, length=1.5, width=2.5)

    # Apply 3D rotation to test frame invariance
    theta = math.radians(35.0)
    phi = math.radians(20.0)
    c_t, s_t = math.cos(theta), math.sin(theta)
    c_p, s_p = math.cos(phi), math.sin(phi)
    rot_matrix = np.array([
        [c_t, -s_t * c_p, s_t * s_p],
        [s_t, c_t * c_p, -c_t * s_p],
        [0.0, s_p, c_p],
    ])
    x0_rot = np.dot(x0, rot_matrix.T)

    model = MockModel(x0_rot)
    prop = PropType43(id=1, ismstr=4)
    setattr(prop, "kn", 5.0e6)
    setattr(prop, "ks", 3.0e6)
    setattr(prop, "cn", 1.0e3)
    setattr(prop, "cs", 1.0e3)
    group = MockElementGroup(conn, prop=prop)
    init_connect_type43(group, model, None)

    # Displace and deform top nodes arbitrarily
    x_curr = x0_rot.copy()
    displacements = np.array([0.003, -0.004, 0.006])
    x_curr[4:8] += displacements

    dt = 1.0e-3
    v_curr = np.zeros_like(x_curr)
    v_curr[4:8] = displacements / dt

    fint = np.zeros_like(x_curr)
    forces_connect_type43(group, x_curr, v_curr, None, dt, fint)

    # Static equilibrium: total resultant force must be strictly ZERO
    sum_f = np.sum(fint, axis=0)
    np.testing.assert_allclose(sum_f, [0.0, 0.0, 0.0], atol=1e-10)


def test_static_equilibrium_4node():
    """Verify sum(F) = 0 for a 4-node connector."""
    x0, conn = make_unit_4node_connector(origin=(0.0, 0.0, 0.0), h=0.1, length=2.0)
    model = MockModel(x0)
    prop = PropType43(id=1, thick=0.5)
    setattr(prop, "kn", 2.0e6)
    setattr(prop, "ks", 1.0e6)
    group = MockElementGroup(conn, prop=prop)
    init_connect_type43(group, model, None)

    x_curr = x0.copy()
    x_curr[2:4] += [0.005, 0.002, 0.008]
    dt = 1.0e-3
    v_curr = np.zeros_like(x_curr)
    v_curr[2:4] = [5.0, 2.0, 8.0]

    fint = np.zeros_like(x_curr)
    forces_connect_type43(group, x_curr, v_curr, None, dt, fint)

    sum_f = np.sum(fint, axis=0)
    np.testing.assert_allclose(sum_f, [0.0, 0.0, 0.0], atol=1e-10)


# ============================================================================
# 6. Critical Time Step
# ============================================================================

def test_critical_time_step():
    """Verify dt_crit calculation matches dt = 2 * sqrt(m_node / K_eff)."""
    x0, conn = make_unit_8node_connector(h=0.1, length=1.0, width=1.0)
    model = MockModel(x0)
    kn_val = 4.0e6
    ks_val = 1.0e6
    prop = PropType43(id=1)
    setattr(prop, "kn", kn_val)
    setattr(prop, "ks", ks_val)
    mat = Material(id=1, law=83, rho0=8000.0)
    group = MockElementGroup(conn, prop=prop, mat=mat)
    init_connect_type43(group, model, None)

    fint = np.zeros_like(x0)
    dt_crit = forces_connect_type43(group, x0, None, None, 1.0e-4, fint)

    st = group.state
    mass_node = st["mass"][0] / 8.0
    k_eff = kn_val * 1.0  # max(Kn, Ks) * Area
    expected_dt = 2.0 * math.sqrt(mass_node / k_eff)

    assert dt_crit[0] == pytest.approx(expected_dt, rel=1e-5)


# ============================================================================
# 7. pyradioss Standard Interfaces (init_group and forces)
# ============================================================================

def test_standard_interfaces_dispatch():
    """Verify init_group and forces wrappers correctly delegate."""
    x0, conn = make_unit_8node_connector(h=0.1, length=1.0, width=1.0)
    model = MockModel(x0)
    prop = PropType43(id=1)
    setattr(prop, "kn", 1.0e6)
    group = MockElementGroup(conn, prop=prop)

    # Test init_group
    node_idx, mass_c, inert_c = init_group(group, model, None)
    assert len(node_idx) == 8
    assert len(mass_c) == 8

    # Test forces
    fint = np.zeros_like(x0)
    dt_crit = forces(group, x0, None, None, 1.0e-4, fint)
    assert len(dt_crit) == 1
    assert dt_crit[0] > 0.0
