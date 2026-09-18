"""
Comprehensive test suite for Thick Shells and Advanced Thin Shells:
- thickshell_wedge6 (6-node thick shell wedge)
- thickshell_composite (8-node composite thick shell)
- shell_dkt6 (6-node rotation-free Discrete Kirchhoff Triangle shell)
"""

import numpy as np
import pytest

from pyradioss.elements import shell_dkt6, thickshell_composite, thickshell_wedge6
from pyradioss.model.entities import Material, Property
from pyradioss.model.model import ElementGroup, Model


class MockLog:
    def info(self, msg, cat=""): pass
    def error(self, msg, cat=""): pass
    def warning(self, msg, cat=""): pass


# ============================================================================
# 1. 6-node Thick Shell Wedge (thickshell_wedge6)
# ============================================================================

def test_thickshell_wedge6_kinematics_and_forces():
    model = Model()
    model.node_ids = np.arange(1, 7, dtype=np.int64)
    # Triangular base in (X, Y) with thickness in Z
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.1],
        [1.0, 0.0, 0.1],
        [0.0, 1.0, 0.1],
    ], dtype=np.float64)
    model.x = model.x0.copy()

    mat = Material(id=1, law=1, rho0=2700.0, title="Alu", params={"E": 7.0e10, "nu": 0.33})
    prop = Property(id=1, type=20, title="ThickProp")

    conn = np.arange(6, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state["slices"] = [(slice(0, 1), mat, prop)]

    node_idx, mass_c, _ = thickshell_wedge6.init_group(group, model, MockLog())
    assert len(node_idx) == 6
    # Volume = 0.5 * 1.0 * 1.0 * 0.1 = 0.05
    assert np.isclose(group.state["vol0"][0], 0.05)

    # Pure thickness tension: top triangle moving upwards (+Z)
    v = np.zeros((6, 3))
    v[3:6, 2] = 1.0
    fint = np.zeros((6, 3))
    dt_crit = thickshell_wedge6.forces(group, model.x, v, None, 1.0e-6, fint, None)

    assert dt_crit[0] > 0.0
    assert np.allclose(np.sum(fint, axis=0), 0.0, atol=1e-4)
    assert np.all(fint[3:6, 2] < 0.0)  # top nodes resist upward movement
    assert np.all(fint[0:3, 2] > 0.0)  # bottom nodes pulled upward
    assert group.state["eint"][0] > 0.0


# ============================================================================
# 2. 8-node Composite Thick Shell (thickshell_composite)
# ============================================================================

def test_thickshell_composite_layering():
    model = Model()
    model.node_ids = np.arange(1, 9, dtype=np.int64)
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.05],
        [1.0, 0.0, 0.05],
        [1.0, 1.0, 0.05],
        [0.0, 1.0, 0.05],
    ], dtype=np.float64)
    model.x = model.x0.copy()

    mat = Material(id=1, law=1, rho0=1600.0, title="CFRP", params={"E": 1.4e11, "nu": 0.28})
    prop = Property(id=1, type=20, title="CompProp")


    conn = np.arange(8, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state["slices"] = [(slice(0, 1), mat, prop)]

    node_idx, mass_c, _ = thickshell_composite.init_group(group, model, MockLog())
    assert len(node_idx) == 8
    assert np.isclose(group.state["vol0"][0], 0.05)

    v = np.zeros((8, 3))
    v[[1, 2, 5, 6], 0] = 1.0  # in-plane extension
    fint = np.zeros((8, 3))
    dt_crit = thickshell_composite.forces(group, model.x, v, None, 1.0e-6, fint, None)

    assert dt_crit[0] > 0.0
    assert np.allclose(np.sum(fint, axis=0), 0.0, atol=1e-4)
    assert group.state["eint"][0] > 0.0


# ============================================================================
# 3. 6-node Rotation-Free DKT Shell (shell_dkt6)
# ============================================================================

def test_shell_dkt6_rotation_free():
    model = Model()
    model.node_ids = np.arange(1, 7, dtype=np.int64)
    # 3 primary nodes (0, 1, 2) + 3 neighbor nodes (3, 4, 5)
    model.x0 = np.array([
        [0.0, 0.0, 0.0],   # node 1
        [1.0, 0.0, 0.0],   # node 2
        [0.0, 1.0, 0.0],   # node 3
        [1.0, 1.0, 0.0],   # neighbor across edge 2-3
        [-0.5, 0.5, 0.0],  # neighbor across edge 3-1
        [0.5, -0.5, 0.0],  # neighbor across edge 1-2
    ], dtype=np.float64)
    model.x = model.x0.copy()

    mat = Material(id=1, law=1, rho0=7800.0, title="Steel", params={"E": 2.1e11, "nu": 0.3})
    prop = Property(id=1, type=1, title="ShellProp", params={"thick": 1.0e-3, "ish3n": 3})
    prop.thick = 1.0e-3

    conn = np.arange(6, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state["slices"] = [(slice(0, 1), mat, prop)]

    node_idx, mass_c, _ = shell_dkt6.init_group(group, model, MockLog())
    # Primary 3 nodes carry the mass
    assert len(node_idx) == 3
    assert np.isclose(group.state["area0"][0], 0.5)

    # Pure bending motion: transverse deflection of neighbors
    v = np.zeros((6, 3))
    v[3:6, 2] = 1.0  # bending curvature
    fint = np.zeros((6, 3))
    mint = np.zeros((6, 3))

    dt_crit = shell_dkt6.forces(group, model.x, v, None, 1.0e-6, fint, mint)

    assert dt_crit[0] > 0.0
    # Rotation-free: mint MUST remain strictly zero!
    assert np.allclose(mint, 0.0)
    # Bending couples are self-equilibrating
    assert np.allclose(np.sum(fint, axis=0), 0.0, atol=1e-5)
