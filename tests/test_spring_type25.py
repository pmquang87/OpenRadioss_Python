"""
Test suite for /PROP/TYPE25 (SPR_AXI) and enhanced 6-DOF /PROP/TYPE44 (SPR_CRUS).

Fortran references:
- ``starter/source/properties/spring/hm_read_prop25.F``
- ``starter/source/properties/spring/hm_read_prop44.F``
- ``engine/source/elements/spring/r6def3.F``
- ``engine/source/elements/spring/redef3.F90``
- ``engine/source/elements/spring/ruser44.F``
- ``engine/source/elements/spring/rforc3.F``
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.model.entities import Property, PropType25, PropType44
from pyradioss.model.model import ElementGroup, Model
from pyradioss.elements import spring, spring_advanced


class MockFunction:
    """Mock Radioss function curve."""
    def __init__(self, fid: int, x_pts, y_pts):
        self.id = fid
        self.x = np.array(x_pts, dtype=np.float64)
        self.y = np.array(y_pts, dtype=np.float64)

    def eval(self, val: float) -> float:
        return float(np.interp(val, self.x, self.y))


# ============================================================================
# 1. /PROP/TYPE25 Axisymmetric Nonlinear Spring Tests
# ============================================================================

def test_type25_linear_axial_and_shear_forces():
    """Verify TYPE25 linear response:
    - Axial force F_ax = K_ax * Delta L + C_ax * v_ax
    - Transverse shear force F_sh = K_sh * delta_sh + C_sh * v_sh
    - Exact equilibrium: F_node1 + F_node2 = 0
    - Energy accounting in group.state['eint']
    """
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    # Spring along X: (0,0,0) to (1,0,0), L0 = 1.0
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    prop = Property(
        id=1, type=25, title="Axi_Linear",
        params={
            "tension": {"stiff": 1000.0, "damp": 20.0},
            "shear": {"stiff": 400.0, "damp": 10.0},
            "mass": 2.0,
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )

    log = MessageLog()
    node_idx, massn, inertn = spring.init_group(group, model, log)
    assert not log.errors
    assert np.allclose(massn, [1.0, 1.0])

    # Elongate by 0.1 along X, and displace node 2 transversely by 0.05 along Y
    x = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=np.float64)
    v = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float64)  # v_ax = 2.0, v_sh = [0, 1, 0]
    dt = 0.05

    fint = np.zeros((2, 3))
    spring.forces(group, x, v, None, dt, fint, None)

    # After dt=0.05 with v_sh=[0, 1, 0], accumulated shear displacement is 0.05 along Y
    # Delta L = 0.1
    # Axial force: F_ax = 1000 * 0.1 + 20 * 2.0 = 100 + 40 = 140 N
    # Shear force: F_sh_y = 400 * 0.05 + 10 * 1.0 = 20 + 10 = 30 N
    # Total force on node 1: [+140, +30, 0]
    # Total force on node 2: [-140, -30, 0]
    assert fint[0, 0] == pytest.approx(140.0)
    assert fint[0, 1] == pytest.approx(30.0)
    assert fint[0, 2] == pytest.approx(0.0)

    assert fint[1, 0] == pytest.approx(-140.0)
    assert fint[1, 1] == pytest.approx(-30.0)
    assert fint[1, 2] == pytest.approx(0.0)

    # Total equilibrium
    assert np.allclose(fint[0] + fint[1], 0.0)
    assert group.state["eint"][0] > 0.0


def test_type25_nonlinear_curve_evaluation():
    """Verify TYPE25 nonlinear force vs displacement curve (hflag=0)."""
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)

    # Nonlinear curve: fun_id=10
    # delta -> force
    curve = MockFunction(10, [0.0, 0.1, 0.2, 0.5], [0.0, 150.0, 500.0, 800.0])
    model.functions = {10: curve}

    prop = Property(
        id=1, type=25, title="Axi_Curve",
        params={
            "tension": {"stiff": 1000.0, "damp": 0.0, "fun_a": 10, "hflag": 0},
            "shear": {"stiff": 0.0, "damp": 0.0},
            "mass": 1.0,
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )

    spring.init_group(group, model, MessageLog())

    # Displace to Delta L = 0.2 -> force should be 500.0 N from curve
    x = np.array([[0.0, 0.0, 0.0], [2.2, 0.0, 0.0]], dtype=np.float64)
    fint = np.zeros((2, 3))
    spring.forces(group, x, None, None, 0.001, fint, None)

    assert group.state["force"][0] == pytest.approx(500.0)
    assert fint[0, 0] == pytest.approx(500.0)
    assert fint[1, 0] == pytest.approx(-500.0)


def test_type25_rate_dependency_logarithmic():
    """Verify TYPE25 Cowper-Symonds / logarithmic rate scaling dfac = A + B * ln(max(1, |v/D|))."""
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    prop = Property(
        id=1, type=25, title="Axi_Rate",
        params={
            "tension": {"stiff": 1000.0, "damp": 0.0, "a": 1.0, "b": 0.5, "d": 10.0},
            "shear": {"stiff": 0.0, "damp": 0.0},
            "mass": 1.0,
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    # Delta L = 0.1 -> quasi-static F = 1000 * 0.1 = 100 N
    # v_ax = 100.0 -> dvv = 100 / 10 = 10.0
    # dfac = 1.0 + 0.5 * ln(10.0) = 1.0 + 0.5 * 2.302585 = 2.15129
    # Expected dynamic F = 100 * 2.15129 = 215.129 N
    x = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=np.float64)
    v = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float64)
    fint = np.zeros((2, 3))
    spring.forces(group, x, v, None, 0.001, fint, None)

    expected_dfac = 1.0 + 0.5 * math.log(10.0)
    expected_f = 100.0 * expected_dfac
    assert group.state["force"][0] == pytest.approx(expected_f, rel=1e-4)


def test_type25_uniaxial_displacement_rupture():
    """Verify TYPE25 uniaxial rupture in tension (max_rup) and compression (min_rup)."""
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    prop = Property(
        id=1, type=25, title="Axi_Rup",
        params={
            "tension": {"stiff": 1000.0, "min_rup": -0.2, "max_rup": 0.3},
            "shear": {"stiff": 200.0, "max_rup": 0.4},
            "mass": 1.0,
            "ifail": 0,
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    # 1. Extension below rupture: Delta L = 0.25 < 0.3
    x = np.array([[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]], dtype=np.float64)
    fint = np.zeros((2, 3))
    spring.forces(group, x, None, None, 0.001, fint, None)
    assert group.state["off"][0] == 1.0
    assert group.state["force"][0] == pytest.approx(250.0)

    # 2. Extension exceeds rupture: Delta L = 0.35 >= 0.3 -> ruptures!
    x = np.array([[0.0, 0.0, 0.0], [1.35, 0.0, 0.0]], dtype=np.float64)
    fint[:] = 0.0
    spring.forces(group, x, None, None, 0.001, fint, None)
    assert group.state["off"][0] == 0.0
    assert group.state["force"][0] == 0.0
    assert np.allclose(fint, 0.0)


def test_type25_multiaxial_rupture_interaction():
    """Verify TYPE25 multiaxial rupture criterion (u_ax / max_rup_ax)^2 + (u_sh / max_rup_sh)^2 >= 1."""
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    prop = Property(
        id=1, type=25, title="Axi_MultiRup",
        params={
            "tension": {"stiff": 1000.0, "max_rup": 1.0},
            "shear": {"stiff": 500.0, "max_rup": 1.0},
            "ifail": 1,  # multiaxial
            "mass": 1.0,
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    # Step 1: u_ax = 0.6, u_sh = 0.6 -> (0.6)^2 + (0.6)^2 = 0.72 < 1.0 -> not ruptured
    x = np.array([[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]], dtype=np.float64)
    v = np.array([[0.0, 0.0, 0.0], [0.0, 6.0, 0.0]], dtype=np.float64)
    fint = np.zeros((2, 3))
    spring.forces(group, x, v, None, 0.1, fint, None)  # delta_sh = 6.0 * 0.1 = 0.6
    assert group.state["off"][0] == 1.0

    # Step 2: increase u_ax to 0.8, u_sh reaches 0.8 -> (0.8)^2 + (0.8)^2 = 1.28 >= 1.0 -> ruptures!
    x = np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]], dtype=np.float64)
    v = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)  # delta_sh becomes 0.6 + 0.2 = 0.8
    fint[:] = 0.0
    spring.forces(group, x, v, None, 0.1, fint, None)
    assert group.state["off"][0] == 0.0
    assert group.state["force"][0] == 0.0
    assert np.allclose(fint, 0.0)


# ============================================================================
# 2. Enhanced 6-DOF /PROP/TYPE44 (SPR_CRUS) Tests
# ============================================================================

def test_type44_6dof_bending_and_moment_equilibrium():
    """Verify enhanced 6-DOF TYPE44 crushing frame spring:
    - Torsional moment M_x = K_44 * rot_x
    - Bending moments M_y1, M_y2, M_z1, M_z2
    - Equilibrium shear forces F_z = (M_y1 + M_y2) / L, F_y = -(M_z1 + M_z2) / L
    - Exact sum(F) = 0 and sum(M) + r x F = 0
    """
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    # Spring along X: (0,0,0) to (2,0,0), L = 2.0
    model.x0 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)

    prop = Property(
        id=1, type=44, title="Crush_6DOF",
        params={
            "k_unload": 5000.0,
            "f_yield": 1000.0,
            "k44": 2000.0,   # torsion stiffness
            "k55": 4000.0,   # bending Y stiffness
            "k66": 6000.0,   # bending Z stiffness
            "mass": 2.0,
            "inertia": 1.0,
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    x = model.x0.copy()
    v = np.zeros_like(x)
    # Relative angular velocities:
    # Node 1: omega = [0, 1.0, 2.0]
    # Node 2: omega = [5.0, 1.0, 2.0]  -> relative twist d_omega_x = 5.0
    vr = np.array([
        [0.0, 1.0, 2.0],
        [5.0, 1.0, 2.0],
    ], dtype=np.float64)
    dt = 0.02

    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    spring.forces(group, x, v, vr, dt, fint, mint)

    # Twist angle: rot_x = 5.0 * 0.02 = 0.1 rad
    # Torsion moment: M_x = 2000 * 0.1 = 200.0
    # Node 1 gets +200 on X, Node 2 gets -200 on X
    assert mint[0, 0] == pytest.approx(200.0)
    assert mint[1, 0] == pytest.approx(-200.0)

    # Bending Y: ry1 = 1.0 * 0.02 = 0.02 rad, ry2 = 1.0 * 0.02 = 0.02 rad
    # MY1 = 4000 * 0.02 = 80.0, MY2 = 4000 * 0.02 = 80.0
    # Equilibrium shear force F_z2 = (MY1 + MY2) / L = (80 + 80) / 2.0 = 80.0 N, F_z1 = -80.0 N
    assert fint[0, 2] == pytest.approx(-80.0)
    assert fint[1, 2] == pytest.approx(80.0)

    # Bending Z: rz1 = 2.0 * 0.02 = 0.04 rad, rz2 = 2.0 * 0.02 = 0.04 rad
    # MZ1 = 6000 * 0.04 = 240.0, MZ2 = 6000 * 0.04 = 240.0
    # Equilibrium shear force F_y2 = -(MZ1 + MZ2) / L = -(240 + 240) / 2.0 = -240.0 N, F_y1 = +240.0 N
    assert fint[0, 1] == pytest.approx(240.0)
    assert fint[1, 1] == pytest.approx(-240.0)

    # Conservation of linear and angular momentum:
    # 1. Total force = 0
    assert np.allclose(fint[0] + fint[1], 0.0)
    # 2. Total moment about origin: M1 + M2 + r1 x F1 + r2 x F2 = 0
    r1 = x[0]
    r2 = x[1]
    total_moment = mint[0] + mint[1] + np.cross(r1, fint[0]) + np.cross(r2, fint[1])
    assert np.allclose(total_moment, 0.0, atol=1e-6)


def test_type44_6dof_yield_curve_clamping():
    """Verify TYPE44 6-DOF yield curve clamping for torsion and bending."""
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    # Curve 20: Torsion yield curve rot_x -> M_yield
    curve_torsion = MockFunction(20, [0.0, 0.1, 0.5], [0.0, 100.0, 150.0])
    model.functions = {20: curve_torsion}

    prop = Property(
        id=1, type=44, title="Crush_TorsionYield",
        params={
            "k_unload": 1000.0,
            "f_yield": 500.0,
            "k44": 5000.0,
            "fun_b2": 20,  # positive torsion yield curve
            "mass": 1.0,
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    # Twist rate = 10 rad/s, dt = 0.05 -> rot_x = 0.5 rad
    # Elastic trial moment = 5000 * 0.5 = 2500 N*m
    # Clamped by curve 20 at rot_x=0.5: M_yield = 150.0 N*m
    x = model.x0.copy()
    v = np.zeros_like(x)
    vr = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)

    mint = np.zeros((2, 3))
    spring.forces(group, x, v, vr, 0.05, None, mint)

    assert mint[0, 0] == pytest.approx(150.0)
    assert mint[1, 0] == pytest.approx(-150.0)
