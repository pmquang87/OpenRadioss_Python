"""
Comprehensive test suite for 2D Continuum Solid Elements:
- solid_quad4_full (4-node full 2x2 Gauss quad with B-bar, plane strain & axisymmetric)
- solid_tria3 (3-node CST triangle, plane strain & axisymmetric)
"""

import numpy as np
import pytest

from pyradioss.elements import solid_quad4_full, solid_tria3
from pyradioss.model.entities import Material, Property
from pyradioss.model.model import ElementGroup, Model


class MockLog:
    def info(self, msg, cat=""): pass
    def error(self, msg, cat=""): pass
    def warning(self, msg, cat=""): pass


# ============================================================================
# 1. 2D 4-node Full Quadrilateral (solid_quad4_full, /QUAD4)
# ============================================================================

def test_quad4_full_plane_strain():
    model = Model()
    model.n2d = 2  # Plane strain
    model.node_ids = np.arange(1, 5, dtype=np.int64)
    # Unit quad in (Y, Z): Y is axis 1, Z is axis 2
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    model.x = model.x0.copy()

    mat = Material(id=1, law=1, rho0=7800.0, title="Steel", params={"E": 2.1e11, "nu": 0.3})
    prop = Property(id=1, type=14, title="QuadProp", params={"iquad": 2})

    conn = np.arange(4, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state["slices"] = [(slice(0, 1), mat, prop)]

    node_idx, mass_c, _ = solid_quad4_full.init_group(group, model, MockLog())
    assert len(node_idx) == 4
    # Area = 1.0
    assert np.isclose(group.state["vol0"][0], 1.0)
    assert np.allclose(mass_c, 7800.0 / 4.0)

    # In-plane tension in Y: nodes 1, 2 moving right (+Y)
    v = np.zeros((4, 3))
    v[[1, 2], 1] = 1.0
    fint = np.zeros((4, 3))
    dt_crit = solid_quad4_full.forces(group, model.x, v, None, 1.0e-6, fint, None)

    assert dt_crit[0] > 0.0
    assert np.allclose(np.sum(fint, axis=0), 0.0, atol=1e-4)
    assert np.all(fint[[1, 2], 1] < 0.0)  # right nodes resist expansion
    assert np.all(fint[[0, 3], 1] > 0.0)  # left nodes pulled right
    assert group.state["eint"][0] > 0.0


def test_quad4_full_axisymmetric():
    model = Model()
    model.n2d = 1  # Axisymmetric
    model.node_ids = np.arange(1, 5, dtype=np.int64)
    # Ring element at radius Y in [10.0, 11.0], height Z in [0.0, 1.0]
    model.x0 = np.array([
        [0.0, 10.0, 0.0],
        [0.0, 11.0, 0.0],
        [0.0, 11.0, 1.0],
        [0.0, 10.0, 1.0],
    ], dtype=np.float64)
    model.x = model.x0.copy()

    mat = Material(id=1, law=1, rho0=7800.0, title="Steel", params={"E": 2.1e11, "nu": 0.3})
    prop = Property(id=1, type=14, title="QuadProp", params={"iquad": 2})

    conn = np.arange(4, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state["slices"] = [(slice(0, 1), mat, prop)]

    node_idx, mass_c, _ = solid_quad4_full.init_group(group, model, MockLog())
    # 1-radian volume = area * r_mean = 1.0 * 10.5 = 10.5
    assert np.isclose(group.state["vol0"][0], 10.5)

    # Radial expansion: all nodes move radially outwards (+Y)
    v = np.zeros((4, 3))
    v[:, 1] = 1.0  # vy = +1.0 (radial velocity)
    fint = np.zeros((4, 3))
    dt_crit = solid_quad4_full.forces(group, model.x, v, None, 1.0e-6, fint, None)

    assert dt_crit[0] > 0.0
    # Net hoop stress creates inward radial restoring force
    assert np.sum(fint[:, 1]) < 0.0
    assert group.state["eint"][0] > 0.0



# ============================================================================
# 2. 2D 3-node Constant Strain Triangle (solid_tria3, /TRIA3)
# ============================================================================

def test_tria3_plane_strain():
    model = Model()
    model.n2d = 2  # Plane strain
    model.node_ids = np.arange(1, 4, dtype=np.int64)
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    model.x = model.x0.copy()

    mat = Material(id=1, law=1, rho0=7800.0, title="Steel", params={"E": 2.1e11, "nu": 0.3})
    prop = Property(id=1, type=14, title="TriaProp")

    conn = np.arange(3, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state["slices"] = [(slice(0, 1), mat, prop)]

    node_idx, mass_c, _ = solid_tria3.init_group(group, model, MockLog())
    assert len(node_idx) == 3
    # Triangle area = 0.5
    assert np.isclose(group.state["vol0"][0], 0.5)

    v = np.zeros((3, 3))
    v[1, 1] = 1.0  # node 2 moving in Y
    fint = np.zeros((3, 3))
    dt_crit = solid_tria3.forces(group, model.x, v, None, 1.0e-6, fint, None)

    assert dt_crit[0] > 0.0
    assert np.allclose(np.sum(fint, axis=0), 0.0, atol=1e-4)
    assert group.state["eint"][0] > 0.0


def test_tria3_axisymmetric():
    model = Model()
    model.n2d = 1  # Axisymmetric
    model.node_ids = np.arange(1, 4, dtype=np.int64)
    model.x0 = np.array([
        [0.0, 5.0, 0.0],
        [0.0, 6.0, 0.0],
        [0.0, 5.0, 1.0],
    ], dtype=np.float64)
    model.x = model.x0.copy()

    mat = Material(id=1, law=1, rho0=7800.0, title="Steel", params={"E": 2.1e11, "nu": 0.3})
    prop = Property(id=1, type=14, title="TriaProp")


    conn = np.arange(3, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state["slices"] = [(slice(0, 1), mat, prop)]

    node_idx, mass_c, _ = solid_tria3.init_group(group, model, MockLog())
    # r_mean = (5 + 6 + 5)/3 = 16/3. Area = 0.5. Vol = 0.5 * 16/3 = 8/3 ~ 2.66667
    assert np.isclose(group.state["vol0"][0], 8.0 / 3.0)

    # Radial expansion
    v = np.zeros((3, 3))
    v[:, 1] = 1.0
    fint = np.zeros((3, 3))
    dt_crit = solid_tria3.forces(group, model.x, v, None, 1.0e-6, fint, None)

    assert dt_crit[0] > 0.0
    assert np.sum(fint[:, 1]) < 0.0  # net hoop stress opposes radial expansion
    assert group.state["eint"][0] > 0.0

