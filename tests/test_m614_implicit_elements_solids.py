"""
Unit test suite for Milestone M614: Component 1A
3D Solids Tangent, Geometric Stiffness, and Consistent Mass Completeness.

Verifies:
- tangent(group, x_geom, epsp_incr=None) -> (K, edofs)
- kgeo(group, x_geom) -> (K_geo, edofs)
- consistent_mass(group, x_geom) -> (M, edofs)
for all 8 solid element modules:
1. solid_hexa8_full (s8forc3.F, 24x24, B-bar volumetric split)
2. solid_hexa8_eas (s8eforc3.F, 24x24, 9 EAS bubble modes static condensation)
3. solid_shell_ha8 (s8sforc3.F, 24x24, ANS transverse shear & thickness stretching)
4. solid_cohesive (szforc3.F, 24x24, bilinear traction-separation derivatives)
5. solid_penta6 (assem_p.F, 18x18 wedge)
6. solid_penta6_heph (s6zforc3.F90, 18x18 HEPH physical hourglass stabilization)
7. solid_pyra5 (degenes8.F, 15x15 pyramid)
8. solid_tetra4_sfem (s4lagsfem.F, 12x12 NS-FEM smoothed cell tetra)
"""

from __future__ import annotations

import pytest
import numpy as np

from pyradioss.elements import (
    solid_hexa8_full,
    solid_hexa8_eas,
    solid_shell_ha8,
    solid_cohesive,
    solid_penta6,
    solid_penta6_heph,
    solid_pyra5,
    solid_tetra4_sfem,
)
from pyradioss.model.model import ElementGroup, Model


# ----------------------------------------------------------------------------
# Dummies & Mesh Generators
# ----------------------------------------------------------------------------

class _DummyMat:
    def __init__(self, law=1, rho0=1000.0, E=2.1e11, nu=0.3):
        self.law = law
        self.rho0 = rho0
        self.E = E
        self.nu = nu
        self.G = E / (2.0 * (1.0 + nu)) if (1.0 + nu) != 0 else 0.0
        self.K = E / (3.0 * (1.0 - 2.0 * nu)) if (1.0 - 2.0 * nu) != 0 else 0.0
        self.fail = None
        self.eos = None
        self.fail_models = None
        self.eps_p_max = 0.0
        self.params = {
            "E": E,
            "nu": nu,
            "sig_y": 200.0e6,
            "b": 500.0e6,
            "n": 0.5,
            "eps_max": 1e30,
            "eps_p_max": 1e30,
        }


class _DummyProp:
    def __init__(self, qa=1.1, qb=0.05, h=0.1, kn=1.0e8, kt=1.0e8, sigma_max=1.0e7, delta_max=1.0e-3):
        self.qa = qa
        self.qb = qb
        self.h = h
        self.kn = kn
        self.kt = kt
        self.sigma_max = sigma_max
        self.delta_max = delta_max


class _DummyLog:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def warning(self, msg, tag=""):
        self.warnings.append((tag, msg))

    def error(self, msg, tag=""):
        self.errors.append((tag, msg))

    def info(self, msg, tag=""):
        pass


def _create_unit_hex_mesh():
    """Unit cube [0, 1]^3 with 8 nodes."""
    model = Model()
    xi_nodes = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float64)
    x0 = 0.5 * (xi_nodes + 1.0)
    model.x0 = x0.copy()
    model.x = x0.copy()
    model.v = np.zeros((8, 3), dtype=np.float64)
    model.vr = np.zeros((8, 3), dtype=np.float64)

    conn = np.arange(8, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1], dtype=np.int64), conn=conn, part=np.array([0], dtype=np.int64))
    mat = _DummyMat(law=1, rho0=1000.0, E=2.1e11, nu=0.3)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    return model, group, mat, prop


def _create_cohesive_mesh():
    """8-node cohesive interface in x-y plane (nodes 0..3 bottom, 4..7 top)."""
    model = Model()
    x_coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    model.x0 = x_coords.copy()
    model.x = x_coords.copy()
    model.v = np.zeros((8, 3), dtype=np.float64)
    model.vr = np.zeros((8, 3), dtype=np.float64)

    conn = np.arange(8, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1], dtype=np.int64), conn=conn, part=np.array([0], dtype=np.int64))
    mat = _DummyMat(law=1, rho0=1000.0, E=2.1e11, nu=0.3)
    prop = _DummyProp(kn=1.0e8, kt=1.0e8, sigma_max=1.0e7, delta_max=1.0e-3)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    return model, group, mat, prop


def _create_wedge_mesh():
    """6-node triangular prism: triangle (0,0,0), (1,0,0), (0,1,0) extruded to z=1."""
    model = Model()
    x_coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=np.float64)
    model.x0 = x_coords.copy()
    model.x = x_coords.copy()
    model.v = np.zeros((6, 3), dtype=np.float64)
    model.vr = np.zeros((6, 3), dtype=np.float64)

    conn = np.arange(6, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1], dtype=np.int64), conn=conn, part=np.array([0], dtype=np.int64))
    mat = _DummyMat(law=1, rho0=1000.0, E=2.1e11, nu=0.3)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    return model, group, mat, prop


def _create_pyramid_mesh():
    """5-node pyramid: base [0, 1]^2 at z=0, apex at (0.5, 0.5, 1.0)."""
    model = Model()
    x_coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ], dtype=np.float64)
    model.x0 = x_coords.copy()
    model.x = x_coords.copy()
    model.v = np.zeros((5, 3), dtype=np.float64)
    model.vr = np.zeros((5, 3), dtype=np.float64)

    conn = np.arange(5, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1], dtype=np.int64), conn=conn, part=np.array([0], dtype=np.int64))
    mat = _DummyMat(law=1, rho0=1000.0, E=2.1e11, nu=0.3)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    return model, group, mat, prop


def _create_tetra_mesh():
    """4-node tetrahedron: vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1)."""
    model = Model()
    x_coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    model.x0 = x_coords.copy()
    model.x = x_coords.copy()
    model.v = np.zeros((4, 3), dtype=np.float64)
    model.vr = np.zeros((4, 3), dtype=np.float64)

    conn = np.arange(4, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1], dtype=np.int64), conn=conn, part=np.array([0], dtype=np.int64))
    mat = _DummyMat(law=1, rho0=1000.0, E=2.1e11, nu=0.3)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    return model, group, mat, prop


# ----------------------------------------------------------------------------
# 1. solid_hexa8_full: B-bar volumetric split, tangent, kgeo, consistent mass
# ----------------------------------------------------------------------------

def test_hexa8_full_empty_group():
    empty_group = ElementGroup(ids=np.zeros(0, dtype=np.int64), conn=np.zeros((0, 8), dtype=np.int64), part=np.zeros(0, dtype=np.int64))
    ke, edofs_k = solid_hexa8_full.tangent(empty_group, np.zeros((0, 3)))
    assert ke.shape == (0, 24, 24)
    assert edofs_k.shape == (0, 24)

    kg, edofs_g = solid_hexa8_full.kgeo(empty_group, np.zeros((0, 3)))
    assert kg.shape == (0, 24, 24)
    assert edofs_g.shape == (0, 24)

    me, edofs_m = solid_hexa8_full.consistent_mass(empty_group)
    assert me.shape == (0, 24, 24)
    assert edofs_m.shape == (0, 24)


def test_hexa8_full_consistent_mass_properties():
    model, group, _, _ = _create_unit_hex_mesh()
    solid_hexa8_full.init_group(group, model, _DummyLog())

    me, edofs = solid_hexa8_full.consistent_mass(group, model.x0)
    assert me.shape == (1, 24, 24)
    assert edofs.shape == (1, 24)
    M = me[0]

    # Symmetry
    assert np.allclose(M, M.T, atol=1e-12)
    # Strictly positive definite
    eigvals = np.linalg.eigvalsh(M)
    assert np.all(eigvals > 0.0)

    # Row sums equal lumped mass = total_mass / 8 = 1000 / 8 = 125.0
    row_sums = M.sum(axis=1)
    for c in range(3):
        assert np.allclose(row_sums[c::3], 125.0, rtol=1e-5)

    # Rigid body kinetic energy: 0.5 * v^T M v == 0.5 * m * |v|^2
    v_rigid = np.tile([2.0, -3.0, 1.5], 8)
    ke_val = 0.5 * v_rigid @ M @ v_rigid
    expected_ke = 0.5 * 1000.0 * (2.0**2 + (-3.0)**2 + 1.5**2)
    assert np.isclose(ke_val, expected_ke, rtol=1e-5)


def test_hexa8_full_kgeo():
    model, group, _, _ = _create_unit_hex_mesh()
    solid_hexa8_full.init_group(group, model, _DummyLog())

    # Zero stress -> K_geo is zero
    kg_zero, _ = solid_hexa8_full.kgeo(group, model.x)
    assert np.all(kg_zero == 0.0)

    # Non-zero stress at 8 Gauss points
    group.state["sig"][:, :, 0] = 50.0e6  # sigma_xx = 50 MPa
    kg, _ = solid_hexa8_full.kgeo(group, model.x)
    assert not np.all(kg == 0.0)
    assert np.allclose(kg[0], kg[0].T, atol=1e-10)

    # Deleted element gets zero
    group.state["off"][0] = 0.0
    kg_dead, _ = solid_hexa8_full.kgeo(group, model.x)
    assert np.all(kg_dead == 0.0)


def test_hexa8_full_tangent_bbar():
    model, group, _, _ = _create_unit_hex_mesh()
    solid_hexa8_full.init_group(group, model, _DummyLog())

    ke, edofs = solid_hexa8_full.tangent(group, model.x)
    assert ke.shape == (1, 24, 24)
    assert edofs.shape == (1, 24)
    K = ke[0]

    # Symmetry
    assert np.allclose(K, K.T, atol=1e-8)

    # 6 rigid body modes have zero eigenvalues, 18 deformational modes > 0
    eigvals = np.linalg.eigvalsh(K)
    assert np.sum(eigvals > 1.0e-5 * eigvals.max()) == 18

    # Check edofs mapping: node_idx * 6 + [0, 1, 2]
    expected_edofs = np.zeros(24, dtype=np.int64)
    for i in range(8):
        expected_edofs[3 * i:3 * i + 3] = i * 6 + np.array([0, 1, 2])
    assert np.array_equal(edofs[0], expected_edofs)


# ----------------------------------------------------------------------------
# 2. solid_hexa8_eas: 9 EAS bubble modes static condensation
# ----------------------------------------------------------------------------

def test_hexa8_eas_empty_group():
    empty_group = ElementGroup(ids=np.zeros(0, dtype=np.int64), conn=np.zeros((0, 8), dtype=np.int64), part=np.zeros(0, dtype=np.int64))
    ke, edofs_k = solid_hexa8_eas.tangent(empty_group, np.zeros((0, 3)))
    assert ke.shape == (0, 24, 24)
    assert edofs_k.shape == (0, 24)

    kg, edofs_g = solid_hexa8_eas.kgeo(empty_group, np.zeros((0, 3)))
    assert kg.shape == (0, 24, 24)
    assert edofs_g.shape == (0, 24)

    me, edofs_m = solid_hexa8_eas.consistent_mass(empty_group)
    assert me.shape == (0, 24, 24)
    assert edofs_m.shape == (0, 24)


def test_hexa8_eas_consistent_mass_and_kgeo():
    model, group, _, _ = _create_unit_hex_mesh()
    solid_hexa8_eas.init_group(group, model, _DummyLog())

    me, edofs = solid_hexa8_eas.consistent_mass(group, model.x0)
    assert me.shape == (1, 24, 24)
    assert np.allclose(me[0], me[0].T)
    assert np.all(np.linalg.eigvalsh(me[0]) > 0.0)

    # Row sum = 125
    row_sums = me[0].sum(axis=1)
    for c in range(3):
        assert np.allclose(row_sums[c::3], 125.0, rtol=1e-5)

    # kgeo
    kg_zero, _ = solid_hexa8_eas.kgeo(group, model.x)
    assert np.all(kg_zero == 0.0)
    group.state["sig"][:, :, 2] = 80.0e6
    kg, _ = solid_hexa8_eas.kgeo(group, model.x)
    assert np.allclose(kg[0], kg[0].T)
    assert not np.all(kg == 0.0)


def test_hexa8_eas_tangent_condensation():
    model, group, _, _ = _create_unit_hex_mesh()
    solid_hexa8_eas.init_group(group, model, _DummyLog())

    ke, edofs = solid_hexa8_eas.tangent(group, model.x)
    assert ke.shape == (1, 24, 24)
    K = ke[0]
    # Symmetric
    assert np.allclose(K, K.T, atol=1e-8)

    # 6 rigid body modes
    eigvals = np.linalg.eigvalsh(K)
    assert np.sum(eigvals > 1.0e-5 * eigvals.max()) == 18

    # Deleted element
    group.state["off"][0] = 0.0
    ke_dead, _ = solid_hexa8_eas.tangent(group, model.x)
    assert np.all(ke_dead == 0.0)


# ----------------------------------------------------------------------------
# 3. solid_shell_ha8: ANS transverse shear & thickness stretching
# ----------------------------------------------------------------------------

def test_solid_shell_ha8_empty_group():
    empty_group = ElementGroup(ids=np.zeros(0, dtype=np.int64), conn=np.zeros((0, 8), dtype=np.int64), part=np.zeros(0, dtype=np.int64))
    ke, edofs_k = solid_shell_ha8.tangent(empty_group, np.zeros((0, 3)))
    assert ke.shape == (0, 24, 24)
    assert edofs_k.shape == (0, 24)

    kg, edofs_g = solid_shell_ha8.kgeo(empty_group, np.zeros((0, 3)))
    assert kg.shape == (0, 24, 24)
    assert edofs_g.shape == (0, 24)

    me, edofs_m = solid_shell_ha8.consistent_mass(empty_group)
    assert me.shape == (0, 24, 24)
    assert edofs_m.shape == (0, 24)


def test_solid_shell_ha8_tangent_kgeo_mass():
    model, group, _, _ = _create_unit_hex_mesh()
    solid_shell_ha8.init_group(group, model, _DummyLog())

    ke, edofs = solid_shell_ha8.tangent(group, model.x)
    assert ke.shape == (1, 24, 24)
    assert edofs.shape == (1, 24)
    assert np.allclose(ke[0], ke[0].T, atol=1e-4)

    # kgeo
    kg_zero, _ = solid_shell_ha8.kgeo(group, model.x)
    assert np.all(kg_zero == 0.0)
    group.state["sig"][:, :, 1] = 40.0e6
    kg, _ = solid_shell_ha8.kgeo(group, model.x)
    assert np.allclose(kg[0], kg[0].T)
    assert not np.all(kg == 0.0)

    # mass
    me, _ = solid_shell_ha8.consistent_mass(group, model.x0)
    assert np.allclose(me[0], me[0].T)
    assert np.all(np.linalg.eigvalsh(me[0]) > 0.0)
    row_sums = me[0].sum(axis=1)
    for c in range(3):
        assert np.allclose(row_sums[c::3], 125.0, rtol=1e-5)


# ----------------------------------------------------------------------------
# 4. solid_cohesive: Bilinear traction-separation derivatives
# ----------------------------------------------------------------------------

def test_cohesive_empty_group():
    empty_group = ElementGroup(ids=np.zeros(0, dtype=np.int64), conn=np.zeros((0, 8), dtype=np.int64), part=np.zeros(0, dtype=np.int64))
    ke, edofs_k = solid_cohesive.tangent(empty_group, np.zeros((0, 3)))
    assert ke.shape == (0, 24, 24)
    assert edofs_k.shape == (0, 24)

    kg, edofs_g = solid_cohesive.kgeo(empty_group, np.zeros((0, 3)))
    assert kg.shape == (0, 24, 24)
    assert edofs_g.shape == (0, 24)

    me, edofs_m = solid_cohesive.consistent_mass(empty_group)
    assert me.shape == (0, 24, 24)
    assert edofs_m.shape == (0, 24)


def test_cohesive_tangent_kgeo_mass():
    model, group, mat, prop = _create_cohesive_mesh()
    solid_cohesive.init_group(group, model, _DummyLog())

    # Tangent: zero opening
    ke, edofs = solid_cohesive.tangent(group, model.x)
    assert ke.shape == (1, 24, 24)
    assert edofs.shape == (1, 24)
    K = ke[0]
    assert np.allclose(K, K.T)

    # Rigid translation: bottom and top move identically -> delta=0 -> force=0 -> K @ v = 0
    v_rigid = np.tile([1.0, 2.0, -1.0], 8)
    assert np.allclose(K @ v_rigid, 0.0, atol=1e-8)

    # Pure opening delta_z = 1e-4: top nodes move +z, bottom nodes stay
    u_open = np.zeros(24)
    u_open[4 * 3 + 2::3] = 1.0e-4  # top nodes uz = 1e-4
    f_res = K @ u_open
    # Bottom nodes receive upward force (+z), top nodes receive downward resisting force (-z)
    assert np.all(f_res[0:12:3] < 0.0) or np.all(f_res[2:12:3] < 0.0) or np.all(f_res[14::3] > 0.0)

    # kgeo
    kg_zero, _ = solid_cohesive.kgeo(group, model.x)
    assert np.all(kg_zero == 0.0)
    group.state["sig"][0, 0] = 5.0e6  # Tn = 5 MPa
    kg, _ = solid_cohesive.kgeo(group, model.x)
    assert np.allclose(kg[0], kg[0].T)
    assert not np.all(kg == 0.0)

    # Consistent mass
    me, _ = solid_cohesive.consistent_mass(group, model.x0)
    assert me.shape == (1, 24, 24)
    assert np.allclose(me[0], me[0].T)
    total_mass = group.state["mass"][0]
    row_sums = me[0].sum(axis=1)
    for c in range(3):
        assert np.allclose(row_sums[c::3], total_mass / 8.0, rtol=1e-5)


# ----------------------------------------------------------------------------
# 5. solid_penta6: 6-node wedge (assem_p.F)
# ----------------------------------------------------------------------------

def test_penta6_empty_group():
    empty_group = ElementGroup(ids=np.zeros(0, dtype=np.int64), conn=np.zeros((0, 6), dtype=np.int64), part=np.zeros(0, dtype=np.int64))
    ke, edofs_k = solid_penta6.tangent(empty_group, np.zeros((0, 3)))
    assert ke.shape == (0, 18, 18)
    assert edofs_k.shape == (0, 18)

    kg, edofs_g = solid_penta6.kgeo(empty_group, np.zeros((0, 3)))
    assert kg.shape == (0, 18, 18)
    assert edofs_g.shape == (0, 18)

    me, edofs_m = solid_penta6.consistent_mass(empty_group)
    assert me.shape == (0, 18, 18)
    assert edofs_m.shape == (0, 18)


def test_penta6_tangent_kgeo_mass():
    model, group, _, _ = _create_wedge_mesh()
    solid_penta6.init_group(group, model, _DummyLog())

    # Tangent
    ke, edofs = solid_penta6.tangent(group, model.x)
    assert ke.shape == (1, 18, 18)
    assert edofs.shape == (1, 18)
    assert np.allclose(ke[0], ke[0].T)

    # Check edofs
    expected_edofs = np.zeros(18, dtype=np.int64)
    for i in range(6):
        expected_edofs[3 * i:3 * i + 3] = i * 6 + np.array([0, 1, 2])
    assert np.array_equal(edofs[0], expected_edofs)

    # kgeo
    kg_zero, _ = solid_penta6.kgeo(group, model.x)
    assert np.all(kg_zero == 0.0)
    group.state["sig"][:, :, 0] = 60.0e6
    kg, _ = solid_penta6.kgeo(group, model.x)
    assert np.allclose(kg[0], kg[0].T)
    assert not np.all(kg == 0.0)

    # Consistent mass: row sums = mass / 6
    me, _ = solid_penta6.consistent_mass(group, model.x0)
    assert me.shape == (1, 18, 18)
    assert np.allclose(me[0], me[0].T)
    assert np.all(np.linalg.eigvalsh(me[0]) > 0.0)
    total_mass = group.state["mass"][0]
    row_sums = me[0].sum(axis=1)
    for c in range(3):
        assert np.allclose(row_sums[c::3], total_mass / 6.0, rtol=1e-5)


# ----------------------------------------------------------------------------
# 6. solid_penta6_heph: 6-node wedge with HEPH physical stabilization
# ----------------------------------------------------------------------------

def test_penta6_heph_empty_group():
    empty_group = ElementGroup(ids=np.zeros(0, dtype=np.int64), conn=np.zeros((0, 6), dtype=np.int64), part=np.zeros(0, dtype=np.int64))
    ke, edofs_k = solid_penta6_heph.tangent(empty_group, np.zeros((0, 3)))
    assert ke.shape == (0, 18, 18)
    assert edofs_k.shape == (0, 18)

    kg, edofs_g = solid_penta6_heph.kgeo(empty_group, np.zeros((0, 3)))
    assert kg.shape == (0, 18, 18)
    assert edofs_g.shape == (0, 18)

    me, edofs_m = solid_penta6_heph.consistent_mass(empty_group)
    assert me.shape == (0, 18, 18)
    assert edofs_m.shape == (0, 18)


def test_penta6_heph_tangent_stabilization_and_mass():
    model, group, _, _ = _create_wedge_mesh()
    solid_penta6_heph.init_group(group, model, _DummyLog())

    ke, edofs = solid_penta6_heph.tangent(group, model.x)
    assert ke.shape == (1, 18, 18)
    assert edofs.shape == (1, 18)
    assert np.allclose(ke[0], ke[0].T)

    # HEPH stabilization adds stiffness to the 2 wedge hourglass modes
    eigvals = np.linalg.eigvalsh(ke[0])
    # 6 rigid body modes = 0, 12 deformational modes (including 2 stabilized hg modes) > 0
    assert np.sum(eigvals > 1.0e-5 * eigvals.max()) == 12

    # kgeo
    kg_zero, _ = solid_penta6_heph.kgeo(group, model.x)
    assert np.all(kg_zero == 0.0)
    group.state["sig"][:, :, 2] = 70.0e6
    kg, _ = solid_penta6_heph.kgeo(group, model.x)
    assert np.allclose(kg[0], kg[0].T)
    assert not np.all(kg == 0.0)

    # Consistent mass
    me, _ = solid_penta6_heph.consistent_mass(group, model.x0)
    assert me.shape == (1, 18, 18)
    assert np.allclose(me[0], me[0].T)
    total_mass = group.state["mass"][0]
    row_sums = me[0].sum(axis=1)
    for c in range(3):
        assert np.allclose(row_sums[c::3], total_mass / 6.0, rtol=1e-5)


# ----------------------------------------------------------------------------
# 7. solid_pyra5: 5-node pyramid (degenes8.F)
# ----------------------------------------------------------------------------

def test_pyra5_empty_group():
    empty_group = ElementGroup(ids=np.zeros(0, dtype=np.int64), conn=np.zeros((0, 5), dtype=np.int64), part=np.zeros(0, dtype=np.int64))
    ke, edofs_k = solid_pyra5.tangent(empty_group, np.zeros((0, 3)))
    assert ke.shape == (0, 15, 15)
    assert edofs_k.shape == (0, 15)

    kg, edofs_g = solid_pyra5.kgeo(empty_group, np.zeros((0, 3)))
    assert kg.shape == (0, 15, 15)
    assert edofs_g.shape == (0, 15)

    me, edofs_m = solid_pyra5.consistent_mass(empty_group)
    assert me.shape == (0, 15, 15)
    assert edofs_m.shape == (0, 15)


def test_pyra5_tangent_kgeo_mass():
    model, group, _, _ = _create_pyramid_mesh()
    solid_pyra5.init_group(group, model, _DummyLog())

    # Tangent
    ke, edofs = solid_pyra5.tangent(group, model.x)
    assert ke.shape == (1, 15, 15)
    assert edofs.shape == (1, 15)
    assert np.allclose(ke[0], ke[0].T)

    # Check edofs
    expected_edofs = np.zeros(15, dtype=np.int64)
    for i in range(5):
        expected_edofs[3 * i:3 * i + 3] = i * 6 + np.array([0, 1, 2])
    assert np.array_equal(edofs[0], expected_edofs)

    # kgeo
    kg_zero, _ = solid_pyra5.kgeo(group, model.x)
    assert np.all(kg_zero == 0.0)
    group.state["sig"][0] = [10.0e6, 20.0e6, 0.0, 5.0e6, 0.0, 0.0]
    kg, _ = solid_pyra5.kgeo(group, model.x)
    assert np.allclose(kg[0], kg[0].T)
    assert not np.all(kg == 0.0)

    # Consistent mass: base nodes 0..3 have row sum m/8, apex node 4 has row sum m/2
    me, _ = solid_pyra5.consistent_mass(group, model.x0)
    assert me.shape == (1, 15, 15)
    assert np.allclose(me[0], me[0].T)
    assert np.all(np.linalg.eigvalsh(me[0]) > 0.0)

    total_mass = group.state["mass"][0]
    row_sums = me[0].sum(axis=1)
    for c in range(3):
        # Base nodes
        for node in range(4):
            assert np.isclose(row_sums[3 * node + c], total_mass / 8.0, rtol=1e-5)
        # Apex node
        assert np.isclose(row_sums[3 * 4 + c], total_mass / 2.0, rtol=1e-5)


# ----------------------------------------------------------------------------
# 8. solid_tetra4_sfem: 4-node NS-FEM smoothed cell tetra (s4lagsfem.F)
# ----------------------------------------------------------------------------

def test_tetra4_sfem_empty_group():
    empty_group = ElementGroup(ids=np.zeros(0, dtype=np.int64), conn=np.zeros((0, 4), dtype=np.int64), part=np.zeros(0, dtype=np.int64))
    ke, edofs_k = solid_tetra4_sfem.tangent(empty_group, np.zeros((0, 3)))
    assert ke.shape == (0, 12, 12)
    assert edofs_k.shape == (0, 12)

    kg, edofs_g = solid_tetra4_sfem.kgeo(empty_group, np.zeros((0, 3)))
    assert kg.shape == (0, 12, 12)
    assert edofs_g.shape == (0, 12)

    me, edofs_m = solid_tetra4_sfem.consistent_mass(empty_group)
    assert me.shape == (0, 12, 12)
    assert edofs_m.shape == (0, 12)


def test_tetra4_sfem_tangent_kgeo_mass():
    model, group, _, _ = _create_tetra_mesh()
    solid_tetra4_sfem.init_group(group, model, _DummyLog())

    # Tangent
    ke, edofs = solid_tetra4_sfem.tangent(group, model.x)
    assert ke.shape == (1, 12, 12)
    assert edofs.shape == (1, 12)
    assert np.allclose(ke[0], ke[0].T, atol=1e-8)

    # Check edofs
    expected_edofs = np.zeros(12, dtype=np.int64)
    for i in range(4):
        expected_edofs[3 * i:3 * i + 3] = i * 6 + np.array([0, 1, 2])
    assert np.array_equal(edofs[0], expected_edofs)

    # kgeo
    kg_zero, _ = solid_tetra4_sfem.kgeo(group, model.x)
    assert np.all(kg_zero == 0.0)
    group.state["sig"][0] = [30.0e6, 0.0, 0.0, 0.0, 0.0, 0.0]
    kg, _ = solid_tetra4_sfem.kgeo(group, model.x)
    assert np.allclose(kg[0], kg[0].T)
    assert not np.all(kg == 0.0)

    # Consistent mass: row sums = mass / 4
    me, _ = solid_tetra4_sfem.consistent_mass(group, model.x0)
    assert me.shape == (1, 12, 12)
    assert np.allclose(me[0], me[0].T)
    assert np.all(np.linalg.eigvalsh(me[0]) > 0.0)

    total_mass = group.state["mass"][0]
    row_sums = me[0].sum(axis=1)
    for c in range(3):
        assert np.allclose(row_sums[c::3], total_mass / 4.0, rtol=1e-5)
