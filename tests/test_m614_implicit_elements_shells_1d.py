"""
Milestone M614 Component 1B Verification Suite:
Implicit Tangent Stiffness, Geometric Stiffness, and Consistent Mass Completeness
for Thick Shells, Thin Shells, 2D Continuum Solids, and 1D Elements.

Formulations verified:
1. solid_tshell8.py: 8-node thick shell (24 x 24 translational DOFs).
2. thickshell_wedge6.py: 6-node thick shell wedge (18 x 18).
3. thickshell_composite.py: 8-node composite thick shell (24 x 24).
4. shell_dkt6.py: 6-node rotation-free DKT macro-patch shell (18 x 18 translational DOFs).
5. solid_quad4_full.py: 4-node 2D full 2x2 Gauss quad with B-bar (8 x 8 in plane).
6. solid_tria3.py: 3-node 2D CST triangle (6 x 6 in plane).
7. spring_advanced.py: advanced spring formulations (/PROP/TYPE26, /PROP/TYPE27, /PROP/SPR_MAT).
8. beam_fiber.py: integrated fiber beam (/PROP/TYPE18) (12 x 12).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import (
    beam_fiber,
    shell_bt4,
    shell_dkt6,
    solid_quad4_full,
    solid_tria3,
    solid_tshell8,
    spring,
    spring_advanced,
    thickshell_composite,
    thickshell_wedge6,
    truss,
)
from pyradioss.model.entities import Material, Property, PropType26, PropType27
from pyradioss.model.model import ElementGroup, Model


# ============================================================================
# 1. 8-node Thick Shell (solid_tshell8.py)
# ============================================================================

def test_solid_tshell8_tangent_kgeo_mass():
    """Verify solid_tshell8 tangent (24x24), kgeo (24x24), consistent_mass (24x24)."""
    # 1. Empty group
    empty = ElementGroup(ids=np.array([]), conn=np.zeros((0, 8), dtype=int), part=np.array([]))
    empty.state = {"slices": []}
    ke_emp, edofs_emp = solid_tshell8.tangent(empty, np.zeros((0, 3)))
    assert ke_emp.shape == (0, 24, 24)
    assert edofs_emp.shape == (0, 24)
    kg_emp, _ = solid_tshell8.kgeo(empty, np.zeros((0, 3)))
    assert kg_emp.shape == (0, 24, 24)
    me_emp, _ = solid_tshell8.consistent_mass(empty, np.zeros((0, 3)))
    assert me_emp.shape == (0, 24, 24)

    # 2. Unit thick shell element
    # Nodes 0..3: bottom (-z), 4..7: top (+z)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.2],
        [1.0, 0.0, 0.2],
        [1.0, 1.0, 0.2],
        [0.0, 1.0, 0.2],
    ], dtype=np.float64)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    model = Model()
    model.node_ids = np.arange(8)
    model.x0 = coords.copy()

    mat = Material(id=1, law=1, rho0=7800.0, params={"E": 2.1e11, "nu": 0.3})
    prop = Property(id=1, type=20, title="TSHELL", params={"inpts": 3, "qa": 1.1})
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state = {"slices": [(slice(0, 1), mat, prop)]}

    log = MessageLog()
    solid_tshell8.init_group(group, model, log)
    assert not log.errors

    # Tangent matrix
    ke, edofs = solid_tshell8.tangent(group, coords)
    assert ke.shape == (1, 24, 24)
    assert edofs.shape == (1, 24)
    ke0 = ke[0]
    # Symmetry
    assert np.allclose(ke0, ke0.T, atol=1e-5)
    # Rigid body translation modes: translation along X, Y, Z
    for axis in range(3):
        v_rigid = np.zeros(24)
        v_rigid[axis::3] = 1.0
        f_res = ke0 @ v_rigid
        assert np.allclose(f_res, 0.0, atol=1e-3), f"TSHELL rigid translation axis {axis} failed!"

    # Geometric stiffness kgeo
    # With zero stress: identically zero
    kg_zero, _ = solid_tshell8.kgeo(group, coords)
    assert np.allclose(kg_zero, 0.0)

    # With tensile stress in-plane: sigma_xx = 1e8 Pa
    group.state["sig"][:, :, 0] = 1.0e8
    kg, _ = solid_tshell8.kgeo(group, coords)
    assert kg.shape == (1, 24, 24)
    kg0 = kg[0]
    assert np.allclose(kg0, kg0.T, atol=1e-5)
    assert np.any(np.abs(kg0) > 0.0)

    # Consistent mass matrix
    me, me_dofs = solid_tshell8.consistent_mass(group, coords)
    assert me.shape == (1, 24, 24)
    assert me_dofs.shape == (1, 24)
    me0 = me[0]
    assert np.allclose(me0, me0.T, atol=1e-10)
    # Positive definiteness: all eigenvalues > 0
    eigvals = np.linalg.eigvalsh(me0)
    assert np.all(eigvals > 0.0)
    # Row sum of each translation direction sums to m/8 per node (partition of unity)
    total_m = group.state["mass"][0]
    for i in range(8):
        for c in range(3):
            row_sum = np.sum(me0[3 * i + c, c::3])
            assert row_sum == pytest.approx(total_m / 8.0, rel=1e-4)


# ============================================================================
# 2. 6-node Thick Shell Wedge (thickshell_wedge6.py)
# ============================================================================

def test_thickshell_wedge6_tangent_kgeo_mass():
    """Verify thickshell_wedge6 tangent (18x18), kgeo (18x18), consistent_mass (18x18)."""
    # 1. Empty group
    empty = ElementGroup(ids=np.array([]), conn=np.zeros((0, 6), dtype=int), part=np.array([]))
    empty.state = {"slices": []}
    ke_emp, edofs_emp = thickshell_wedge6.tangent(empty, np.zeros((0, 3)))
    assert ke_emp.shape == (0, 18, 18)
    assert edofs_emp.shape == (0, 18)
    kg_emp, _ = thickshell_wedge6.kgeo(empty, np.zeros((0, 3)))
    assert kg_emp.shape == (0, 18, 18)
    me_emp, _ = thickshell_wedge6.consistent_mass(empty, np.zeros((0, 3)))
    assert me_emp.shape == (0, 18, 18)

    # 2. Wedge geometry (3 bottom vertices, 3 top vertices)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.5],
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.5],
    ], dtype=np.float64)
    conn = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64)

    model = Model()
    model.node_ids = np.arange(6)
    model.x0 = coords.copy()

    mat = Material(id=1, law=1, rho0=2700.0, params={"E": 7.0e10, "nu": 0.33})
    prop = Property(id=1, type=21, title="WEDGE6", params={"qa": 1.1})
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state = {"slices": [(slice(0, 1), mat, prop)]}

    log = MessageLog()
    thickshell_wedge6.init_group(group, model, log)
    assert not log.errors

    # Tangent matrix
    ke, edofs = thickshell_wedge6.tangent(group, coords)
    assert ke.shape == (1, 18, 18)
    assert edofs.shape == (1, 18)
    ke0 = ke[0]
    assert np.allclose(ke0, ke0.T, atol=1e-5)
    # Rigid body translation modes in nullspace
    for axis in range(3):
        v_rigid = np.zeros(18)
        v_rigid[axis::3] = 1.0
        f_res = ke0 @ v_rigid
        assert np.allclose(f_res, 0.0, atol=1e-3), f"WEDGE6 rigid translation axis {axis} failed!"

    # Geometric stiffness kgeo
    kg_zero, _ = thickshell_wedge6.kgeo(group, coords)
    assert np.allclose(kg_zero, 0.0)

    # With nonzero stress
    group.state["sig"][:, :, 0] = 5.0e7
    kg, _ = thickshell_wedge6.kgeo(group, coords)
    assert kg.shape == (1, 18, 18)
    kg0 = kg[0]
    assert np.allclose(kg0, kg0.T, atol=1e-5)
    assert np.any(np.abs(kg0) > 0.0)

    # Consistent mass matrix
    me, me_dofs = thickshell_wedge6.consistent_mass(group, coords)
    assert me.shape == (1, 18, 18)
    assert me_dofs.shape == (1, 18)
    me0 = me[0]
    assert np.allclose(me0, me0.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(me0)
    assert np.all(eigvals > 0.0)
    # Row sums equal m/6 per node
    total_m = group.state["mass"][0]
    for i in range(6):
        for c in range(3):
            row_sum = np.sum(me0[3 * i + c, c::3])
            assert row_sum == pytest.approx(total_m / 6.0, rel=1e-5)


# ============================================================================
# 3. 8-node Composite Thick Shell (thickshell_composite.py)
# ============================================================================

def test_thickshell_composite_tangent_kgeo_mass():
    """Verify thickshell_composite tangent (24x24), kgeo (24x24), consistent_mass (24x24)."""
    # 1. Empty group
    empty = ElementGroup(ids=np.array([]), conn=np.zeros((0, 8), dtype=int), part=np.array([]))
    empty.state = {"slices": []}
    ke_emp, edofs_emp = thickshell_composite.tangent(empty, np.zeros((0, 3)))
    assert ke_emp.shape == (0, 24, 24)
    assert edofs_emp.shape == (0, 24)
    kg_emp, _ = thickshell_composite.kgeo(empty, np.zeros((0, 3)))
    assert kg_emp.shape == (0, 24, 24)
    me_emp, _ = thickshell_composite.consistent_mass(empty, np.zeros((0, 3)))
    assert me_emp.shape == (0, 24, 24)

    # 2. Composite brick element
    coords = np.array([
        [-0.5, -0.5, -0.1],
        [ 0.5, -0.5, -0.1],
        [ 0.5,  0.5, -0.1],
        [-0.5,  0.5, -0.1],
        [-0.5, -0.5,  0.1],
        [ 0.5, -0.5,  0.1],
        [ 0.5,  0.5,  0.1],
        [-0.5,  0.5,  0.1],
    ], dtype=np.float64)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    model = Model()
    model.node_ids = np.arange(8)
    model.x0 = coords.copy()

    mat = Material(id=1, law=1, rho0=1600.0, params={"E": 1.5e11, "nu": 0.25})
    prop = Property(id=1, type=20, title="COMPOSITE", params={"qa": 1.1})
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state = {"slices": [(slice(0, 1), mat, prop)]}

    log = MessageLog()
    thickshell_composite.init_group(group, model, log)
    assert not log.errors

    # Tangent matrix integrating through composite plies
    ke, edofs = thickshell_composite.tangent(group, coords)
    assert ke.shape == (1, 24, 24)
    assert edofs.shape == (1, 24)
    ke0 = ke[0]
    assert np.allclose(ke0, ke0.T, atol=1e-5)
    for axis in range(3):
        v_rigid = np.zeros(24)
        v_rigid[axis::3] = 1.0
        f_res = ke0 @ v_rigid
        assert np.allclose(f_res, 0.0, atol=1e-3), f"Composite rigid translation axis {axis} failed!"

    # Geometric stiffness kgeo
    kg_zero, _ = thickshell_composite.kgeo(group, coords)
    assert np.allclose(kg_zero, 0.0)

    group.state["sig"][:, :, 0] = 2.0e8
    kg, _ = thickshell_composite.kgeo(group, coords)
    assert kg.shape == (1, 24, 24)
    kg0 = kg[0]
    assert np.allclose(kg0, kg0.T, atol=1e-5)
    assert np.any(np.abs(kg0) > 0.0)

    # Consistent mass matrix
    me, me_dofs = thickshell_composite.consistent_mass(group, coords)
    assert me.shape == (1, 24, 24)
    assert me_dofs.shape == (1, 24)
    me0 = me[0]
    assert np.allclose(me0, me0.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(me0)
    assert np.all(eigvals > 0.0)
    total_m = group.state["mass"][0]
    for i in range(8):
        for c in range(3):
            row_sum = np.sum(me0[3 * i + c, c::3])
            assert row_sum == pytest.approx(total_m / 8.0, rel=1e-4)


# ============================================================================
# 4. 6-node Rotation-Free DKT Shell (shell_dkt6.py)
# ============================================================================

def test_shell_dkt6_tangent_kgeo_mass():
    """Verify shell_dkt6 tangent (18x18), kgeo (18x18), consistent_mass (18x18)."""
    # 1. Empty group
    empty = ElementGroup(ids=np.array([]), conn=np.zeros((0, 6), dtype=int), part=np.array([]))
    empty.state = {"slices": []}
    ke_emp, edofs_emp = shell_dkt6.tangent(empty, np.zeros((0, 3)))
    assert ke_emp.shape == (0, 18, 18)
    assert edofs_emp.shape == (0, 18)
    kg_emp, _ = shell_dkt6.kgeo(empty, np.zeros((0, 3)))
    assert kg_emp.shape == (0, 18, 18)
    me_emp, _ = shell_dkt6.consistent_mass(empty, np.zeros((0, 3)))
    assert me_emp.shape == (0, 18, 18)

    # 2. Macro-patch triangle with 3 central vertices + 3 neighbor vertices
    coords = np.array([
        [0.0, 0.0, 0.0],   # Node 0 (primary)
        [1.0, 0.0, 0.0],   # Node 1 (primary)
        [0.0, 1.0, 0.0],   # Node 2 (primary)
        [1.0, 1.0, 0.0],   # Node 3 (neighbor to edge 1-2)
        [-1.0, 1.0, 0.0],  # Node 4 (neighbor to edge 2-0)
        [0.5, -0.5, 0.0],  # Node 5 (neighbor to edge 0-1)
    ], dtype=np.float64)
    conn = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64)

    model = Model()
    model.node_ids = np.arange(6)
    model.x0 = coords.copy()

    mat = Material(id=1, law=1, rho0=7850.0, params={"E": 2.1e11, "nu": 0.3})
    prop = Property(id=1, type=1, title="DKT6", params={"thick": 0.002})
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state = {"slices": [(slice(0, 1), mat, prop)]}

    log = MessageLog()
    shell_dkt6.init_group(group, model, log)
    assert not log.errors

    # Tangent matrix (18 translational DOFs)
    ke, edofs = shell_dkt6.tangent(group, coords)
    assert ke.shape == (1, 18, 18)
    assert edofs.shape == (1, 18)
    ke0 = ke[0]
    assert np.allclose(ke0, ke0.T, atol=1e-5)
    # Rigid body translation modes: translation in X, Y, Z
    for axis in range(3):
        v_rigid = np.zeros(18)
        v_rigid[axis::3] = 1.0
        f_res = ke0 @ v_rigid
        assert np.allclose(f_res, 0.0, atol=1e-3), f"DKT6 rigid translation axis {axis} failed!"

    # Geometric stiffness kgeo
    kg_zero, _ = shell_dkt6.kgeo(group, coords)
    assert np.allclose(kg_zero, 0.0)

    # Non-zero membrane stress
    group.state["sig"][0, 0] = 1.0e6  # Nxx
    kg, _ = shell_dkt6.kgeo(group, coords)
    assert kg.shape == (1, 18, 18)
    kg0 = kg[0]
    assert np.allclose(kg0, kg0.T, atol=1e-5)
    assert np.any(np.abs(kg0) > 0.0)

    # Consistent mass matrix (mass on primary 3 nodes)
    me, me_dofs = shell_dkt6.consistent_mass(group, coords)
    assert me.shape == (1, 18, 18)
    assert me_dofs.shape == (1, 18)
    me0 = me[0]
    assert np.allclose(me0, me0.T, atol=1e-10)
    total_m = group.state["mass"][0]
    for i in range(3):
        for c in range(3):
            row_sum = np.sum(me0[3 * i + c, c:9:3])
            assert row_sum == pytest.approx(total_m / 3.0, rel=1e-5)


# ============================================================================
# 5. 4-node 2D Full 2x2 Gauss Quad with B-bar (solid_quad4_full.py)
# ============================================================================

def test_solid_quad4_full_tangent_kgeo_mass():
    """Verify solid_quad4_full tangent (8x8), kgeo (8x8), consistent_mass (8x8 in plane)."""
    # 1. Empty group
    empty = ElementGroup(ids=np.array([]), conn=np.zeros((0, 4), dtype=int), part=np.array([]))
    empty.state = {"slices": []}
    ke_emp, edofs_emp = solid_quad4_full.tangent(empty, np.zeros((0, 3)))
    assert ke_emp.shape == (0, 8, 8)
    assert edofs_emp.shape == (0, 8)
    kg_emp, _ = solid_quad4_full.kgeo(empty, np.zeros((0, 3)))
    assert kg_emp.shape == (0, 8, 8)
    me_emp, _ = solid_quad4_full.consistent_mass(empty, np.zeros((0, 3)))
    assert me_emp.shape == (0, 8, 8)

    # 2. Unit quad in (Y, Z) plane: X=0, Y in [0, 1], Z in [0, 1]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)

    model = Model()
    model.node_ids = np.arange(4)
    model.x0 = coords.copy()

    mat = Material(id=1, law=1, rho0=1000.0, params={"E": 2.0e7, "nu": 0.4})
    prop = Property(id=1, type=14, title="QUAD4", params={"thick": 1.0})
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state = {"slices": [(slice(0, 1), mat, prop)], "n2d": np.array([2])}

    log = MessageLog()
    solid_quad4_full.init_group(group, model, log)
    assert not log.errors

    # Tangent matrix (8 x 8 in plane)
    ke, edofs = solid_quad4_full.tangent(group, coords)
    assert ke.shape == (1, 8, 8)
    assert edofs.shape == (1, 8)
    ke0 = ke[0]
    assert np.allclose(ke0, ke0.T, atol=1e-5)
    # Rigid body in-plane translations (Y translation, Z translation)
    v_y = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    v_z = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    assert np.allclose(ke0 @ v_y, 0.0, atol=1e-3)
    assert np.allclose(ke0 @ v_z, 0.0, atol=1e-3)

    # Geometric stiffness kgeo
    kg_zero, _ = solid_quad4_full.kgeo(group, coords)
    assert np.allclose(kg_zero, 0.0)

    # With in-plane stress syy = 1e5 Pa
    group.state["sig"][:, :, 1] = 1.0e5
    kg, _ = solid_quad4_full.kgeo(group, coords)
    assert kg.shape == (1, 8, 8)
    kg0 = kg[0]
    assert np.allclose(kg0, kg0.T, atol=1e-5)
    assert np.any(np.abs(kg0) > 0.0)

    # Consistent mass matrix (8 x 8)
    me, me_dofs = solid_quad4_full.consistent_mass(group, coords)
    assert me.shape == (1, 8, 8)
    assert me_dofs.shape == (1, 8)
    me0 = me[0]
    assert np.allclose(me0, me0.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(me0)
    assert np.all(eigvals > 0.0)
    total_m = group.state["mass"][0]
    for i in range(4):
        for c in range(2):
            row_sum = np.sum(me0[2 * i + c, c::2])
            assert row_sum == pytest.approx(total_m / 4.0, rel=1e-5)


# ============================================================================
# 6. 3-node 2D CST Triangle (solid_tria3.py)
# ============================================================================

def test_solid_tria3_tangent_kgeo_mass():
    """Verify solid_tria3 tangent (6x6), kgeo (6x6), consistent_mass (6x6 in plane)."""
    # 1. Empty group
    empty = ElementGroup(ids=np.array([]), conn=np.zeros((0, 3), dtype=int), part=np.array([]))
    empty.state = {"slices": []}
    ke_emp, edofs_emp = solid_tria3.tangent(empty, np.zeros((0, 3)))
    assert ke_emp.shape == (0, 6, 6)
    assert edofs_emp.shape == (0, 6)
    kg_emp, _ = solid_tria3.kgeo(empty, np.zeros((0, 3)))
    assert kg_emp.shape == (0, 6, 6)
    me_emp, _ = solid_tria3.consistent_mass(empty, np.zeros((0, 3)))
    assert me_emp.shape == (0, 6, 6)

    # 2. Right triangle in (Y, Z) plane
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    conn = np.array([[0, 1, 2]], dtype=np.int64)

    model = Model()
    model.node_ids = np.arange(3)
    model.x0 = coords.copy()

    mat = Material(id=1, law=1, rho0=2000.0, params={"E": 5.0e7, "nu": 0.3})
    prop = Property(id=1, type=15, title="TRIA3", params={"thick": 1.0})
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state = {"slices": [(slice(0, 1), mat, prop)], "n2d": np.array([2])}

    log = MessageLog()
    solid_tria3.init_group(group, model, log)
    assert not log.errors

    # Tangent matrix (6 x 6)
    ke, edofs = solid_tria3.tangent(group, coords)
    assert ke.shape == (1, 6, 6)
    assert edofs.shape == (1, 6)
    ke0 = ke[0]
    assert np.allclose(ke0, ke0.T, atol=1e-5)
    # Rigid body translation in Y and Z
    v_y = np.array([1, 0, 1, 0, 1, 0], dtype=float)
    v_z = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    assert np.allclose(ke0 @ v_y, 0.0, atol=1e-3)
    assert np.allclose(ke0 @ v_z, 0.0, atol=1e-3)

    # Geometric stiffness kgeo
    kg_zero, _ = solid_tria3.kgeo(group, coords)
    assert np.allclose(kg_zero, 0.0)

    # Non-zero in-plane stress
    group.state["sig"][0, 1] = 2.0e5
    kg, _ = solid_tria3.kgeo(group, coords)
    assert kg.shape == (1, 6, 6)
    kg0 = kg[0]
    assert np.allclose(kg0, kg0.T, atol=1e-5)
    assert np.any(np.abs(kg0) > 0.0)

    # Consistent mass matrix (6 x 6)
    me, me_dofs = solid_tria3.consistent_mass(group, coords)
    assert me.shape == (1, 6, 6)
    assert me_dofs.shape == (1, 6)
    me0 = me[0]
    assert np.allclose(me0, me0.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(me0)
    assert np.all(eigvals > 0.0)
    total_m = group.state["mass"][0]
    for i in range(3):
        for c in range(2):
            row_sum = np.sum(me0[2 * i + c, c::2])
            assert row_sum == pytest.approx(total_m / 3.0, rel=1e-5)


# ============================================================================
# 7. Advanced Springs (spring_advanced.py)
# ============================================================================

def test_spring_advanced_tangent_kgeo_mass():
    """Verify spring_advanced tangent, kgeo, consistent_mass for TYPE26, TYPE27, SPR_MAT, TYPE19."""
    # 1. Empty group
    empty = ElementGroup(ids=np.array([]), conn=np.zeros((0, 2), dtype=int), part=np.array([]))
    empty.state = {"slices": []}
    ke_emp, edofs_emp = spring_advanced.tangent(empty, np.zeros((0, 3)))
    assert ke_emp.shape == (0, 6, 6)
    assert edofs_emp.shape == (0, 6)
    kg_emp, _ = spring_advanced.kgeo(empty, np.zeros((0, 3)))
    assert kg_emp.shape == (0, 6, 6)
    me_emp, _ = spring_advanced.consistent_mass(empty, np.zeros((0, 3)))
    assert me_emp.shape == (0, 6, 6)

    # 2. Translational spring with /PROP/TYPE26 (SPR_TAB)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float64)
    conn = np.array([[0, 1]], dtype=np.int64)

    prop26 = PropType26(id=1, kmax=5000.0, mass=10.0)
    group26 = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group26.state = {
        "slices": [(slice(0, 1), None, prop26)],
        "L0": np.array([2.0]),
        "mass": np.array([10.0]),
        "force": np.array([0.0]),
        "off": np.array([1.0]),
    }

    # Tabular dF/du stiffness
    k_ax, k_tor = spring_advanced.local_stiffness(group26, coords)
    assert k_ax[0] == pytest.approx(5000.0)
    assert k_tor[0] == 0.0

    ke26, edofs26 = spring_advanced.tangent(group26, coords)
    assert ke26.shape == (1, 6, 6)
    assert edofs26.shape == (1, 6)
    # Pure X extension: node 0 gets +5000, node 1 gets +5000, cross terms -5000
    assert ke26[0, 0, 0] == pytest.approx(5000.0)
    assert ke26[0, 3, 3] == pytest.approx(5000.0)
    assert ke26[0, 0, 3] == pytest.approx(-5000.0)
    assert np.allclose(ke26[0, 1:3, :], 0.0)

    # Geometric stiffness: zero at zero force
    kg_zero, _ = spring_advanced.kgeo(group26, coords)
    assert np.allclose(kg_zero, 0.0)

    # Taut-string geometric stiffness under tension F = 1000 N
    group26.state["force"][0] = 1000.0
    kg, _ = spring_advanced.kgeo(group26, coords)
    assert kg.shape == (1, 6, 6)
    # Along transverse directions Y, Z: F/L = 1000 / 2 = 500 N/m
    assert kg[0, 1, 1] == pytest.approx(500.0)
    assert kg[0, 2, 2] == pytest.approx(500.0)
    assert kg[0, 0, 0] == pytest.approx(0.0)  # Axial direction has 0 geometric stiffness

    # Consistent mass
    me, _ = spring_advanced.consistent_mass(group26, coords)
    assert me.shape == (1, 6, 6)
    assert np.allclose(np.diag(me[0]), 5.0)  # 10 / 2 on each node translation

    # 3. Bilinear / damped spring (/PROP/TYPE27, SPR_BDAMP)
    prop27 = PropType27(id=2, stiff=8000.0, damp=100.0, mass=4.0)
    group27 = ElementGroup(ids=np.array([2]), conn=conn, part=np.array([0]))
    group27.state = {
        "slices": [(slice(0, 1), None, prop27)],
        "L0": np.array([2.0]),
        "mass": np.array([4.0]),
        "force": np.array([0.0]),
        "off": np.array([1.0]),
    }
    ke27, _ = spring_advanced.tangent(group27, coords)
    assert ke27[0, 0, 0] == pytest.approx(8000.0)

    # 4. Material spring (/PROP/SPR_MAT)
    mat_spr = Material(id=3, law=1, params={"E": 2.0e11})
    prop_mat = Property(id=3, type=23, title="SPR_MAT", params={"area": 1e-4})
    group_mat = ElementGroup(ids=np.array([3]), conn=conn, part=np.array([0]))
    group_mat.state = {
        "slices": [(slice(0, 1), mat_spr, prop_mat)],
        "L0": np.array([2.0]),
        "mass": np.array([2.0]),
        "force": np.array([0.0]),
        "off": np.array([1.0]),
    }
    # k = E * A / L = 2e11 * 1e-4 / 2 = 1e7 N/m
    ke_mat, _ = spring_advanced.tangent(group_mat, coords)
    assert ke_mat[0, 0, 0] == pytest.approx(1.0e7)

    # 5. Torsional spring (/PROP/TYPE19) (12 x 12 stiffness)
    prop19 = Property(id=4, type=19, title="TORSION", params={"k_theta": 1200.0, "c_theta": 20.0, "inertia": 6.0, "mass": 2.0})
    group19 = ElementGroup(ids=np.array([4]), conn=conn, part=np.array([0]))
    group19.state = {
        "slices": [(slice(0, 1), None, prop19)],
        "L0": np.array([2.0]),
        "mass": np.array([2.0]),
        "t19_k_theta": np.array([1200.0]),
        "t19_inertia": np.array([6.0]),
        "force": np.array([0.0]),
        "off": np.array([1.0]),
    }
    ke19, edofs19 = spring_advanced.tangent(group19, coords)
    assert ke19.shape == (1, 12, 12)
    assert edofs19.shape == (1, 12)
    # Torsional twist about X (DOF index 3 for node 0, index 9 for node 1)
    assert ke19[0, 3, 3] == pytest.approx(1200.0)
    assert ke19[0, 9, 9] == pytest.approx(1200.0)
    assert ke19[0, 3, 9] == pytest.approx(-1200.0)


# ============================================================================
# 8. Integrated Fiber Beam (beam_fiber.py)
# ============================================================================

def test_beam_fiber_tangent_kgeo_mass():
    """Verify beam_fiber tangent (12x12), kgeo (12x12), consistent_mass (12x12)."""
    # 1. Empty group
    empty = ElementGroup(ids=np.array([]), conn=np.zeros((0, 3), dtype=int), part=np.array([]))
    empty.state = {"slices": []}
    ke_emp, edofs_emp = beam_fiber.tangent(empty, np.zeros((0, 3)))
    assert ke_emp.shape == (0, 12, 12)
    assert edofs_emp.shape == (0, 12)
    kg_emp, _ = beam_fiber.kgeo(empty, np.zeros((0, 3)))
    assert kg_emp.shape == (0, 12, 12)
    me_emp, _ = beam_fiber.consistent_mass(empty, np.zeros((0, 3)))
    assert me_emp.shape == (0, 12, 12)

    # 2. 3D Fiber beam along X axis
    coords = np.array([
        [0.0, 0.0, 0.0],   # N1
        [10.0, 0.0, 0.0],  # N2
        [0.0, 1.0, 0.0],   # N3 (orientation node)
    ], dtype=np.float64)
    conn = np.array([[0, 1, 2]], dtype=np.int64)

    model = Model()
    model.node_ids = np.arange(3)
    model.x0 = coords.copy()

    mat = Material(id=1, law=1, rho0=7800.0, params={"E": 2.1e11, "nu": 0.3})
    prop = Property(id=1, type=18, title="INT_BEAM", params={
        "area": 0.01, "iyy": 8.33e-6, "izz": 8.33e-6, "ixx": 1.66e-5,
    })
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state = {
        "slices": [(slice(0, 1), mat, prop)],
        "L0": np.array([10.0]),
        "fres": np.zeros((1, 3)),
        "mres": np.zeros((1, 3)),
        "mass": np.array([780.0]),  # rho0 * area * L = 7800 * 0.01 * 10 = 780 kg
    }

    # Tangent matrix (12 x 12)
    ke, edofs = beam_fiber.tangent(group, coords)
    assert ke.shape == (1, 12, 12)
    assert edofs.shape == (1, 12)
    ke0 = ke[0]
    assert np.allclose(ke0, ke0.T, atol=1e-5)
    eigvals = np.linalg.eigvalsh(ke0)
    # First 6 eigenvalues should be 0 (rigid body modes)
    assert np.all(np.abs(eigvals[:6]) < 1e-4)
    assert np.all(eigvals[6:] > 0.0)

    # Geometric stiffness kgeo
    kg_zero, _ = beam_fiber.kgeo(group, coords)
    assert np.allclose(kg_zero, 0.0)

    # With axial tension N = 1e6 N and bending moment My = 5e4 N*m
    group.state["fres"][0, 0] = 1.0e6
    group.state["mres"][0, 1] = 5.0e4
    kg, kg_dofs = beam_fiber.kgeo(group, coords)
    assert kg.shape == (1, 12, 12)
    assert kg_dofs.shape == (1, 12)
    kg0 = kg[0]
    assert np.allclose(kg0, kg0.T, atol=1e-5)
    assert np.any(np.abs(kg0) > 0.0)

    # Consistent mass matrix (12 x 12)
    me, me_dofs = beam_fiber.consistent_mass(group, coords)
    assert me.shape == (1, 12, 12)
    assert me_dofs.shape == (1, 12)
    me0 = me[0]
    assert np.allclose(me0, me0.T, atol=1e-10)
    # Mass conservation: pure translation in X excites row sums to m/2 on each node
    total_m = 780.0
    for c in range(3):
        # Node 1 translation DOF c
        row_n1 = np.sum(me0[c, [0, 1, 2, 6, 7, 8]])
        assert row_n1 == pytest.approx(total_m / 2.0, rel=1e-4)


# ============================================================================
# 9. Uniform Signature Verification Across All 8 Formulations
# ============================================================================

def test_all_8_element_signatures_match_standard():
    """Verify all 8 element modules adhere to the exact required signatures:
    - tangent(group, x_geom, epsp_incr=None) -> tuple[np.ndarray, np.ndarray]
    - kgeo(group, x_geom) -> tuple[np.ndarray, np.ndarray]
    - consistent_mass(group, x_geom) -> tuple[np.ndarray, np.ndarray]
    """
    kernels = [
        (solid_tshell8, 8, 24),
        (thickshell_wedge6, 6, 18),
        (thickshell_composite, 8, 24),
        (shell_dkt6, 6, 18),
        (solid_quad4_full, 4, 8),
        (solid_tria3, 3, 6),
        (spring_advanced, 2, 6),
        (beam_fiber, 3, 12),
    ]

    for kernel, n_nodes, n_dofs in kernels:
        name = kernel.__name__
        assert hasattr(kernel, "tangent"), f"{name} lacks tangent()"
        assert hasattr(kernel, "kgeo"), f"{name} lacks kgeo()"
        assert hasattr(kernel, "consistent_mass"), f"{name} lacks consistent_mass()"

        # Empty group invocation
        empty = ElementGroup(
            ids=np.array([], dtype=int),
            conn=np.zeros((0, n_nodes), dtype=int),
            part=np.array([], dtype=int),
        )
        empty.state = {"slices": []}
        x_dummy = np.zeros((0, 3))

        # Check tangent return
        res_t = kernel.tangent(empty, x_dummy, None)
        assert isinstance(res_t, tuple)
        assert len(res_t) == 2
        ke, edofs_t = res_t
        assert ke.shape == (0, n_dofs, n_dofs), f"{name}.tangent ke shape {ke.shape} != (0, {n_dofs}, {n_dofs})"
        assert edofs_t.shape == (0, n_dofs), f"{name}.tangent edofs shape {edofs_t.shape} != (0, {n_dofs})"

        # Check kgeo return
        res_kg = kernel.kgeo(empty, x_dummy)
        assert isinstance(res_kg, tuple)
        assert len(res_kg) == 2
        kg, edofs_kg = res_kg
        assert kg.shape == (0, n_dofs, n_dofs), f"{name}.kgeo kg shape {kg.shape} != (0, {n_dofs}, {n_dofs})"
        assert edofs_kg.shape == (0, n_dofs), f"{name}.kgeo edofs shape {edofs_kg.shape} != (0, {n_dofs})"

        # Check consistent_mass return
        res_m = kernel.consistent_mass(empty, x_dummy)
        assert isinstance(res_m, tuple)
        assert len(res_m) == 2
        me, edofs_m = res_m
        assert me.shape == (0, n_dofs, n_dofs), f"{name}.consistent_mass me shape {me.shape} != (0, {n_dofs}, {n_dofs})"
        assert edofs_m.shape == (0, n_dofs), f"{name}.consistent_mass edofs shape {edofs_m.shape} != (0, {n_dofs})"
