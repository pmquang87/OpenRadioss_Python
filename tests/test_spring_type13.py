"""
Unit tests for OpenRadioss /PROP/TYPE13 (/PROP/SPR_BEAM: 6-DOF nonlinear spring-beam element).

Fortran references:
  - starter/source/properties/spring/hm_read_prop13.F
  - starter/source/elements/spring/r4buf3.F (lines 146-244)
  - engine/source/elements/spring/r4evec3.F (lines 119-255)
  - engine/source/elements/spring/r4def3.F (lines 218-289, 336-377, 799-826)
  - engine/source/elements/spring/redef3.F90 (lines 736-1158)
  - engine/source/elements/spring/r4cum3.F (lines 80-138)
  - engine/source/elements/spring/r2len3.F (lines 180-186)
  - engine/source/elements/spring/r13ke3.F & r13sumg3.F (lines 56-150)
"""

import math
from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import spring, spring_beam
from pyradioss.input.deck_reader import read_deck
from pyradioss.model.entities import Property
from pyradioss.model.model import ElementGroup, Model
from pyradioss.input.starter_keywords import parse_starter_deck


class MockFunction:
    """Mock Radioss curve function."""
    def __init__(self, fid: int, x_pts, y_pts):
        self.id = fid
        self.x = np.array(x_pts, dtype=np.float64)
        self.y = np.array(y_pts, dtype=np.float64)

    def eval(self, val: float) -> float:
        return float(np.interp(val, self.x, self.y))

    def evaluate(self, val: float) -> float:
        return self.eval(val)


def _make_model_and_group(n1_xyz, n2_xyz, prop_params, n3_xyz=None):
    """Helper to construct a simple 1-element spring-beam model."""
    m = Model()
    if n3_xyz is not None:
        nodes = np.array([n1_xyz, n2_xyz, n3_xyz], dtype=np.float64)
        m.add_nodes(np.array([1, 2, 3]), nodes)
        conn = np.array([[0, 1, 2]], dtype=np.int64)
    else:
        nodes = np.array([n1_xyz, n2_xyz], dtype=np.float64)
        m.add_nodes(np.array([1, 2]), nodes)
        conn = np.array([[0, 1]], dtype=np.int64)

    prop = Property(id=1, type=13, params=prop_params)
    m.properties[1] = prop

    g = ElementGroup(
        ids=np.array([1], dtype=np.int64),
        conn=conn,
        part=np.array([0], dtype=np.int64)
    )
    g.state["slices"] = [(slice(0, 1), None, prop)]
    return m, g


def test_type13_initialization_triad_mass_inertia():
    """Verify TYPE13 triad computation (N1->N2 and optional N3) and mass/inertia distribution.

    Fortran origin: starter/source/elements/spring/r4buf3.F lines 146-244, hm_read_prop13.F.
    """
    params = {
        "mass": 4.0,
        "inertia": 1.2,
        "k1": 100.0, "k2": 100.0, "k3": 100.0,
        "k4": 50.0, "k5": 50.0, "k6": 50.0,
    }
    # Element along X axis with N3 in XY plane -> e1=[1,0,0], e2=[0,1,0], e3=[0,0,1]
    m, g = _make_model_and_group([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], params, n3_xyz=[1.0, 1.0, 0.0])
    log = MessageLog()
    node_idx, massn, inertn = spring.init_group(g, m, log)
    assert not log.errors

    # Check mass lumped half to node 1 and node 2 (4.0 / 2 = 2.0 each)
    np.testing.assert_allclose(massn[:2], [2.0, 2.0])
    # Check inertia lumped half to node 1 and node 2 (1.2 / 2 = 0.6 each)
    np.testing.assert_allclose(inertn[:2], [0.6, 0.6])

    b = g.state["spr_beam13"]
    np.testing.assert_allclose(b["e1"][0], [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(b["e2"][0], [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(b["e3"][0], [0.0, 0.0, 1.0], atol=1e-6)
    assert b["L0"][0] == pytest.approx(2.0)


def test_type13_multi_dof_linear_elastic():
    """Verify uncoupled 6-DOF elastic and viscous response (3 translations, 3 rotations).

    Fortran origin: engine/source/elements/spring/r4def3.F lines 218-289, redef3.F90.
    """
    params = {
        "mass": 2.0, "inertia": 0.5,
        "k1": 1000.0, "c1": 10.0,
        "k2": 2000.0, "c2": 20.0,
        "k3": 3000.0, "c3": 30.0,
        "k4": 400.0,  "c4": 4.0,
        "k5": 500.0,  "c5": 5.0,
        "k6": 600.0,  "c6": 6.0,
    }
    m, g = _make_model_and_group([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], params)
    spring.init_group(g, m, MessageLog())

    dt = 1e-3
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.05, 0.02, 0.03]  # delta = [0.05, 0.02, 0.03]
    ])
    v = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0]     # dv = [1.0, 2.0, 3.0]
    ])
    vr = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.6, 0.7]     # dw = [0.5, 0.6, 0.7]
    ])

    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    dtc = spring.forces(g, x, v, vr, dt, fint, mint)
    assert len(dtc) == 1
    assert dtc[0] > 0.0

    b = g.state["spr_beam13"]
    # F1 = 1000 * 0.05 + 10 * 1.0 = 50 + 10 = 60
    # F2 = 2000 * 0.02 + 20 * 2.0 = 40 + 40 = 80
    # F3 = 3000 * 0.03 + 30 * 3.0 = 90 + 90 = 180
    np.testing.assert_allclose(b["force"][0], [60.0, 80.0, 180.0], rtol=1e-5)

    # Rotations (rate-integrated: theta = dw * dt)
    # theta = [0.5, 0.6, 0.7] * 1e-3 = [0.0005, 0.0006, 0.0007]
    # M1 = 400 * 0.0005 + 4 * 0.5 = 0.2 + 2.0 = 2.2
    # M2 = 500 * 0.0006 + 5 * 0.6 = 0.3 + 3.0 = 3.3
    # M3 = 600 * 0.0007 + 6 * 0.7 = 0.42 + 4.2 = 4.62
    np.testing.assert_allclose(b["moment"][0], [2.2, 3.3, 4.62], rtol=1e-5)


def test_type13_nonlinear_function_and_hardening_models():
    """Verify nonlinear curve evaluation (FUN_A) and hardening models (HFLAG=0, 1, 2, 4, 7).

    Fortran origin: engine/source/elements/spring/redef3.F90 lines 736-1158.
    """
    # Define a piecewise curve for tension: (0, 0), (0.1, 100), (0.2, 150)
    curve1 = MockFunction(1, [0.0, 0.1, 0.2, 0.5], [0.0, 100.0, 150.0, 200.0])

    # 1. HFLAG = 0 (Nonlinear elastic)
    params = {
        "mass": 1.0, "inertia": 0.1,
        "k1": 1000.0, "fun_a1": 1, "hflag1": 0, "scale1": 1.0,
    }
    m, g = _make_model_and_group([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], params)
    m.functions = {1: curve1}
    spring.init_group(g, m, MessageLog())

    # Load to delta = 0.15 -> f(0.15) = 125
    x = np.array([[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]])
    spring.forces(g, x, None, None, 1e-4, np.zeros((2, 3)), np.zeros((2, 3)))
    b = g.state["spr_beam13"]
    assert b["force"][0, 0] == pytest.approx(125.0, rel=1e-4)

    # Unload to delta = 0.05 -> nonlinear elastic returns to f(0.05) = 50
    x[1, 0] = 1.05
    spring.forces(g, x, None, None, 1e-4, np.zeros((2, 3)), np.zeros((2, 3)))
    assert b["force"][0, 0] == pytest.approx(50.0, rel=1e-4)

    # 2. HFLAG = 1 (Isotropic hardening)
    params["hflag1"] = 1
    m, g = _make_model_and_group([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], params)
    m.functions = {1: curve1}
    spring.init_group(g, m, MessageLog())

    # Load to delta = 0.1 -> f = 100
    x[1, 0] = 1.1
    spring.forces(g, x, None, None, 1e-4, np.zeros((2, 3)), np.zeros((2, 3)))
    b = g.state["spr_beam13"]
    assert b["force"][0, 0] == pytest.approx(100.0, rel=1e-4)

    # Unload by d_delta = -0.05 with initial stiffness K1 = 1000 -> F = 100 - 1000 * 0.05 = 50
    x[1, 0] = 1.05
    spring.forces(g, x, None, None, 1e-4, np.zeros((2, 3)), np.zeros((2, 3)))
    assert b["force"][0, 0] == pytest.approx(50.0, rel=1e-4)

    # Unload further to delta = 1.0 (zero displacement) -> F = 100 - 1000 * 0.1 = 0
    x[1, 0] = 1.0
    spring.forces(g, x, None, None, 1e-4, np.zeros((2, 3)), np.zeros((2, 3)))
    assert b["force"][0, 0] == pytest.approx(0.0, abs=1e-5)


def test_type13_strain_rate_dependency_and_nonlinear_damping():
    """Verify strain rate factor (A + B ln(dvv) + E gx) and damping (C v + H gx2).

    Fortran origin: engine/source/elements/spring/redef3.F90 lines 751-820.
    """
    curve_f = MockFunction(1, [0.0, 1.0], [0.0, 100.0])  # FUN_A (yield/elastic curve)
    curve_g = MockFunction(2, [0.0, 1.0], [0.0, 2.0])    # FUN_B
    curve_h = MockFunction(3, [0.0, 1.0], [0.0, 5.0])    # FUN_D

    params = {
        "mass": 1.0, "inertia": 0.1,
        "k1": 100.0, "c1": 10.0,
        "a1": 1.5, "b1": 0.2, "d1": 0.5,
        "fun_a1": 1, "fun_b1": 2, "fun_d1": 3,
        "e1": 0.1, "h1": 2.0, "f1": 1.0,
    }
    m, g = _make_model_and_group([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], params)
    m.functions = {1: curve_f, 2: curve_g, 3: curve_h}
    spring.init_group(g, m, MessageLog())

    # delta = 0.5, dv = 2.0
    x = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    v = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    spring.forces(g, x, v, None, 1e-4, np.zeros((2, 3)), np.zeros((2, 3)))
    b = g.state["spr_beam13"]

    # dvv = |dv| / D = 2.0 / 0.5 = 4.0
    # gx = FUN_B(v) = 2.0 (from curve_g at v=2.0)
    # dfac = A + B * ln(dvv) + E * gx = 1.5 + 0.2 * ln(4.0) + 0.1 * 2.0 = 1.97725887
    # F_elastic = K * delta * dfac = 100 * 0.5 * 1.97725887 = 98.86294
    # gx2 = FUN_D(v) = 5.0 (from curve_h at v=2.0)
    # F_damp = (C * dv + H * gx2) = 10 * 2.0 + 2.0 * 5.0 = 20.0 + 10.0 = 30.0
    # Total F = 98.86294 + 30.0 = 128.86294
    expected_f = 100.0 * 0.5 * (1.5 + 0.2 * math.log(4.0) + 0.1 * 2.0) + (10.0 * 2.0 + 2.0 * 5.0)
    assert b["force"][0, 0] == pytest.approx(expected_f, rel=1e-4)


def test_type13_rupture_and_element_deactivation():
    """Verify displacement, force, and energy rupture limits and deactivation.

    Fortran origin: engine/source/elements/spring/r4def3.F lines 336-377, 819-826.
    """
    # 1. Uniaxial displacement limit: MIN_RUP = -0.1, MAX_RUP = 0.2
    params = {
        "mass": 1.0, "inertia": 0.1,
        "k1": 100.0,
        "min_rup1": -0.1, "max_rup1": 0.2,
        "ifail": 0, "ifail2": 0,  # Uniaxial, Displacement
    }
    m, g = _make_model_and_group([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], params)
    spring.init_group(g, m, MessageLog())

    # Step 1: delta = 0.15 < 0.2 -> element remains ACTIVE
    x = np.array([[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]])
    spring.forces(g, x, None, None, 1e-4, np.zeros((2, 3)), np.zeros((2, 3)))
    b = g.state["spr_beam13"]
    assert b["off"][0] == 1.0
    assert b["force"][0, 0] == pytest.approx(15.0)

    # Step 2: delta = 0.25 > 0.2 -> element RUPTURES
    x[1, 0] = 1.25
    fint = np.zeros((2, 3))
    spring.forces(g, x, None, None, 1e-4, fint, np.zeros((2, 3)))
    assert b["off"][0] == 0.0
    assert b["force"][0, 0] == 0.0
    np.testing.assert_allclose(fint, 0.0)

    # 2. Multiaxial rupture interaction (Ifail = 1): (dx/dmax)^2 + (dy/dmax)^2 >= 1.0
    params2 = {
        "mass": 1.0, "inertia": 0.1,
        "k1": 100.0, "max_rup1": 1.0,
        "k2": 100.0, "max_rup2": 1.0,
        "ifail": 1, "ifail2": 0,  # Multiaxial interaction
    }
    m2, g2 = _make_model_and_group([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], params2)
    spring.init_group(g2, m2, MessageLog())

    # dx = 0.8, dy = 0.8 -> crit = 0.8^2 + 0.8^2 = 1.28 >= 1.0 -> RUPTURE
    x2 = np.array([[0.0, 0.0, 0.0], [1.8, 0.8, 0.0]])
    spring.forces(g2, x2, None, None, 1e-4, np.zeros((2, 3)), np.zeros((2, 3)))
    b2 = g2.state["spr_beam13"]
    assert b2["off"][0] == 0.0
    assert b2["crit"][0] >= 1.0


def test_type13_static_equilibrium_and_moment_arm():
    """Verify translational equilibrium (f1 + f2 = 0) and rotational equilibrium (m1 + m2 + r x f2 = 0).

    Fortran origin: engine/source/elements/spring/r4cum3.F lines 80-138.
    """
    params = {
        "mass": 2.0, "inertia": 0.5,
        "k1": 100.0, "k2": 200.0, "k3": 300.0,
        "k4": 50.0,  "k5": 60.0,  "k6": 70.0,
    }
    # Initial beam length L0 = 2.0 along X
    m, g = _make_model_and_group([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], params)
    spring.init_group(g, m, MessageLog())

    # Apply both axial and shear displacements and rotation
    x = np.array([
        [0.0, 0.0, 0.0],
        [2.1, 0.05, 0.08]
    ])
    vr = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3]
    ])
    dt = 1e-3
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    spring.forces(g, x, None, vr, dt, fint, mint)

    f1, f2 = fint[0], fint[1]
    m1, m2 = mint[0], mint[1]

    # 1. Translational equilibrium: f1 + f2 = 0
    np.testing.assert_allclose(f1 + f2, [0.0, 0.0, 0.0], atol=1e-6)

    # 2. Rotational equilibrium: m1 + m2 + r x f2 = 0 where r = x2 - x1
    r = x[1] - x[0]
    moment_sum = m1 + m2 + np.cross(r, f2)
    np.testing.assert_allclose(moment_sum, [0.0, 0.0, 0.0], atol=1e-6)


def test_type13_energy_conservation():
    """Verify trapezoidal internal energy accounting matching analytical 0.5 * K * delta^2.

    Fortran origin: engine/source/elements/spring/redef3.F90 lines 810-820.
    """
    params = {
        "mass": 1.0, "inertia": 0.1,
        "k1": 1000.0, "k4": 500.0,
    }
    m, g = _make_model_and_group([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], params)
    spring.init_group(g, m, MessageLog())

    dt = 1e-4
    n_steps = 10
    total_dx = 0.05
    total_rot = 0.02
    dx_step = total_dx / n_steps
    wz = total_rot / (n_steps * dt)

    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vr = np.array([[0.0, 0.0, 0.0], [wz, 0.0, 0.0]])

    for step in range(n_steps):
        x[1, 0] += dx_step
        spring.forces(g, x, None, vr, dt, np.zeros((2, 3)), np.zeros((2, 3)))

    b = g.state["spr_beam13"]
    # Analytical energy: 0.5 * K1 * dx^2 + 0.5 * K4 * theta^2
    expected_e = 0.5 * 1000.0 * (total_dx ** 2) + 0.5 * 500.0 * (total_rot ** 2)
    assert b["eint"][0] == pytest.approx(expected_e, rel=1e-4)
    assert g.state["eint"][0] == pytest.approx(expected_e, rel=1e-4)


def test_type13_tangent_stiffness_matrix_numerical_perturbation():
    """Verify 12x12 tangent stiffness matrix against finite-difference numerical perturbations.

    Fortran origin: engine/source/elements/spring/r13ke3.F & r13sumg3.F lines 56-150.
    """
    params = {
        "mass": 2.0, "inertia": 0.5,
        "k1": 100.0, "k2": 200.0, "k3": 300.0,
        "k4": 40.0,  "k5": 50.0,  "k6": 60.0,
    }
    m, g = _make_model_and_group([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], params)
    spring.init_group(g, m, MessageLog())

    x0 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    ke, edofs = spring_beam.ke_spring_beam_type13(g, x0)

    assert ke.shape == (1, 12, 12)
    assert edofs.shape == (1, 12)

    k_mat = ke[0]
    # Check that translational stiffness along X is exactly K1
    # DOF 0 (u1x) and DOF 6 (u2x): K_0,0 = K1, K_0,6 = -K1, K_6,6 = K1
    assert k_mat[0, 0] == pytest.approx(100.0)
    assert k_mat[0, 6] == pytest.approx(-100.0)
    assert k_mat[6, 6] == pytest.approx(100.0)

    # Check torsion stiffness around X (DOF 3 and DOF 9) is K4
    assert k_mat[3, 3] == pytest.approx(40.0)
    assert k_mat[3, 9] == pytest.approx(-40.0)
    assert k_mat[9, 9] == pytest.approx(40.0)

    # Check shear-moment coupling: M_z = 0.5 * L * F_y -> K_1,11 and K_7,11
    # Fortran r13sumg3.F: transverse force arm is L/2 = 1.0
    L = 2.0
    expected_coupling = 0.5 * L * 200.0  # 100.0
    assert abs(k_mat[1, 5]) == pytest.approx(expected_coupling)
    assert abs(k_mat[1, 11]) == pytest.approx(expected_coupling)


def test_type13_starter_deck_full_cards_parsing(tmp_path: Path):
    """Verify parsing full /PROP/SPR_BEAM deck with all cards and options."""
    def _c1(k, c, a, b, d):
        return f"{k:>20.1f}{c:>20.1f}{a:>20.1f}{b:>20.1f}{d:>20.1f}"

    def _c2(fa, hf, fb, fc, fd, dmin, dmax):
        return f"{fa:>10}{hf:>10}{fb:>10}{fc:>10}{fd:>10}{' ':>10}{dmin:>20.1f}{dmax:>20.1f}"

    def _c3(f, e, sc, h):
        return f"{f:>20.1f}{e:>20.1f}{sc:>20.1f}{h:>20.1f}"

    lines = [
        "# OpenRadioss Starter Deck",
        "/BEGIN",
        "Test Deck Full SPR_BEAM",
        "      2021         0",
        "                  kg                  mm                  ms",
        "                  kg                  mm                  ms",
        "/NODE",
        "         1                 0.0                 0.0                 0.0",
        "         2                 1.0                 0.0                 0.0",
        "/PART/1",
        "Spring Beam Part 1",
        "         1         0         0",
        "/PROP/SPR_BEAM/1",
        "Spring Beam Full Property",
        f"{5.0:>20.1f}{1.5:>20.1f}{0:>10}{0:>10}{0:>10}{1:>10}{0:>10}{0:>10}",
        _c1(100.0, 10.0, 1.2, 0.1, 0.8),
        _c2(1, 0, 0, 0, 0, -0.5, 0.5),
        _c3(1.0, 0.0, 1.0, 1.0),
        _c1(200.0, 20.0, 1.0, 0.0, 1.0),
        _c2(0, 0, 0, 0, 0, -1.0, 1.0),
        _c3(1.0, 0.0, 1.0, 1.0),
        _c1(300.0, 30.0, 1.0, 0.0, 1.0),
        _c2(0, 0, 0, 0, 0, -1.0, 1.0),
        _c3(1.0, 0.0, 1.0, 1.0),
        _c1(400.0, 40.0, 1.0, 0.0, 1.0),
        _c2(0, 0, 0, 0, 0, -1.0, 1.0),
        _c3(1.0, 0.0, 1.0, 1.0),
        _c1(500.0, 50.0, 1.0, 0.0, 1.0),
        _c2(0, 0, 0, 0, 0, -1.0, 1.0),
        _c3(1.0, 0.0, 1.0, 1.0),
        _c1(600.0, 60.0, 1.0, 0.0, 1.0),
        _c2(0, 0, 0, 0, 0, -1.0, 1.0),
        _c3(1.0, 0.0, 1.0, 1.0),
        "/SPRING/1",
        "         1         1         2",
        "/END",
    ]
    deck_text = "\n".join(lines) + "\n"
    p = tmp_path / "TEST_0000.rad"
    p.write_text(deck_text, encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert not log.errors

    prop = model.properties[1]
    assert prop.type == 13
    assert prop.params["mass"] == pytest.approx(5.0)
    assert prop.params["inertia"] == pytest.approx(1.5)
    assert prop.params["ifail"] == 1

    assert prop.params["k1"] == pytest.approx(100.0)
    assert prop.params["c1"] == pytest.approx(10.0)
    assert prop.params["a1"] == pytest.approx(1.2)
    assert prop.params["b1"] == pytest.approx(0.1)
    assert prop.params["d1"] == pytest.approx(0.8)
    assert prop.params["fun_a1"] == 1
    assert prop.params["min_rup1"] == pytest.approx(-0.5)
    assert prop.params["max_rup1"] == pytest.approx(0.5)

    assert prop.params["k6"] == pytest.approx(600.0)
    assert prop.params["c6"] == pytest.approx(60.0)
