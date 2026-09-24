"""Unit tests for /PROP/TYPE36 (/PROP/PREDIT) progressive damage interface spring.

Fortran reference:
  - starter/source/properties/spring/hm_read_prop36.F (HM_READ_PROP36, RINI36)
  - engine/source/elements/spring/ruser36.F
  - engine/source/elements/spring/r5evec3.F
  - engine/source/elements/spring/r5def3.F
  - engine/source/elements/spring/r5cum3.F
  - config/CFG/radioss110/PROP/prop_p36_predit.cfg
  - config/CFG/radioss110/MAT/matl54_54.cfg
"""

import math
import numpy as np
import pytest

from pyradioss.model.model import Model
from pyradioss.model.entities import PropType36, PropPredit, Property, MatLaw54, Material
from pyradioss.input.deck_reader import Card, KeywordBlock
from pyradioss.input.starter_keywords import read_prop_type36
from pyradioss.input.prop_reader import parse_predit
from pyradioss.elements import spring, spring_advanced


class MockMessageLog:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, msg, context=""):
        self.errors.append((msg, context))

    def warning(self, msg, context=""):
        self.warnings.append((msg, context))


class MockElementGroup:
    def __init__(self, conn, slices=None):
        self.conn = np.array(conn, dtype=np.int64)
        self.n = len(self.conn)
        self.ids = np.arange(1, self.n + 1, dtype=np.int64)
        self.state = {
            "slices": slices or [(slice(0, self.n), None, Property(id=1, type=36, params={"area": 0.01, "e": 2.1e11}))],
            "off": np.ones(self.n, dtype=float),
            "force": np.zeros(self.n),
            "eint": np.zeros(self.n),
        }


# ============================================================================
# Test 1: Keyword Card Parsing (Iutype=1 and Iutype=2, Fixed and Free format)
# ============================================================================

def test_type36_parsing_iutype1_fixed():
    """Test fixed format parsing of 3-card /PROP/TYPE36 with Iutype=1."""
    cards = [
        Card("Interface Spring 1"),
        Card("         1"),                  # Card 1: Iutype = 1
        Card("         0        11        12"), # Card 2: skew_ID, prop_ID1, prop_ID2
        Card("               5.5e4"),        # Card 3: Xk
    ]
    block = KeywordBlock(
        keyword="/PROP/TYPE36",
        parts=["PROP", "TYPE36", "101"],
        user_id=101,
        cards=cards,
        fixed=True,
    )
    model = Model()
    log = MockMessageLog()

    read_prop_type36(block, model, log)
    assert len(log.errors) == 0
    assert 101 in model.prop_type36s
    p36 = model.prop_type36s[101]
    assert p36.lutype == 1
    assert p36.prop_id1 == 11
    assert p36.prop_id2 == 12
    assert pytest.approx(p36.xk) == 5.5e4


def test_type36_parsing_iutype2_free():
    """Test free format parsing of 3-card /PROP/PREDIT with Iutype=2."""
    cards = [
        Card("Interface End Property 2"),
        Card("2"),                                     # Card 1: Iutype = 2
        Card("501"),                                   # Card 2: MAT_ID = 501
        Card("0.02 1.5e-6 8.0e-7 8.0e-7 0.05"),         # Card 3: Area, Ixx, Iyy, Izz, Ray
    ]
    block = KeywordBlock(
        keyword="/PROP/PREDIT",
        parts=["PROP", "PREDIT", "102"],
        user_id=102,
        cards=cards,
        fixed=False,
    )
    model = Model()
    log = MockMessageLog()

    read_prop_type36(block, model, log)
    assert len(log.errors) == 0
    assert 102 in model.prop_type36s
    p36 = model.prop_type36s[102]
    assert p36.lutype == 2
    assert p36.mat_id == 501
    assert pytest.approx(p36.area) == 0.02
    assert pytest.approx(p36.ixx) == 1.5e-6
    assert pytest.approx(p36.iyy) == 8.0e-7
    assert pytest.approx(p36.izz) == 8.0e-7
    assert pytest.approx(p36.ray) == 0.05


def test_type36_parse_predit_helper():
    """Test parse_predit helper from prop_reader."""
    cards = [
        Card("Predit test"),
        Card("1"),
        Card("0 1 2"),
        Card("1.2e5"),
    ]
    block = KeywordBlock(
        keyword="/PROP/PREDIT",
        parts=["PROP", "PREDIT", "103"],
        user_id=103,
        cards=cards,
        fixed=False,
    )
    log = MockMessageLog()
    prop = parse_predit(block, log)
    assert prop.type == 36
    assert prop.params["lutype"] == 1
    assert prop.params["prop_id1"] == 1
    assert prop.params["prop_id2"] == 2
    assert pytest.approx(prop.params["xk"]) == 1.2e5


# ============================================================================
# Test 2: Mass and Rotational Inertia Initialization (RINI36)
# ============================================================================

def test_type36_mass_and_inertia_rini36():
    """Test element lumped mass M = L0 * A * rho and rotational inertia per RINI36."""
    model = Model()
    # 2 nodes along X axis, L0 = 2.0
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])

    area = 0.01
    rho = 7800.0
    ixx = 1.0e-6
    iyy = 5.0e-7
    izz = 5.0e-7
    e_mod = 2.1e11
    nu = 0.3

    prop = PropType36(
        id=1, lutype=2, area=area, ixx=ixx, iyy=iyy, izz=izz,
        rho=rho, e=e_mod, nu=nu
    )
    group = MockElementGroup([[0, 1]], [(slice(0, 1), None, prop)])
    log = MockMessageLog()

    node_idx, massn, inertn = spring.init_group(group, model, log)

    # L0 = 2.0, Area = 0.01, Rho = 7800
    # Expected mass: M = 2.0 * 0.01 * 7800 = 156.0 kg
    # Nodal mass (half and half): 78.0 kg at each node
    expected_m_elem = 2.0 * area * rho
    assert pytest.approx(group.state["t36_mass"][0]) == expected_m_elem
    assert pytest.approx(massn[0]) == expected_m_elem / 2.0
    assert pytest.approx(massn[1]) == expected_m_elem / 2.0

    # Rotational inertia per RINI36:
    # XINER = L0 * rho * max(Ixx, imyz + Area * L0^2 / 12)
    imyz = max(iyy, izz)
    expected_iner_elem = 2.0 * rho * max(ixx, imyz + area * (2.0 ** 2) / 12.0)
    assert pytest.approx(group.state["t36_xiner"][0]) == expected_iner_elem
    assert inertn is not None
    assert pytest.approx(inertn[0]) == expected_iner_elem / 2.0
    assert pytest.approx(inertn[1]) == expected_iner_elem / 2.0


# ============================================================================
# Test 3: Elastic 6-DOF Axial and Transverse Shear Response
# ============================================================================

def test_type36_elastic_axial_and_shear():
    """Verify elastic axial force and transverse shear with Timoshenko correction."""
    model = Model()
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    area = 0.005
    rho = 7800.0
    e_mod = 2.0e11
    nu = 0.3
    g_mod = e_mod / (2.0 * (1.0 + nu))
    ixx = 2.0e-6
    iyy = 1.0e-6
    izz = 1.0e-6

    prop = PropType36(
        id=1, lutype=2, area=area, rho=rho, e=e_mod, nu=nu, g=g_mod,
        ixx=ixx, iyy=iyy, izz=izz, sig0=1.0e30  # high yield stress (pure elastic)
    )
    group = MockElementGroup([[0, 1]], [(slice(0, 1), None, prop)])
    log = MockMessageLog()
    spring.init_group(group, model, log)

    dt = 1e-4
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    x = model.x0.copy()

    # Step 1: Pure axial elongation (node 2 moves with vx = 1.0 m/s)
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vr = np.zeros((2, 3))

    spring_advanced.forces_predit_type36(group, x, v, vr, dt, fint, mint, np.array([0]))

    # Axial strain rate = vx / L = 1.0 / 1.0 = 1.0 s^-1
    # Elastic axial stress increment = E * eps_rate * dt = 2.0e11 * 1.0 * 1e-4 = 2.0e7 Pa
    # Force Fx = stress * Area = 2.0e7 * 0.005 = 1.0e5 N
    expected_fx = e_mod * area * 1.0 * dt / 1.0
    assert pytest.approx(fint[0, 0]) == expected_fx
    assert pytest.approx(fint[1, 0]) == -expected_fx

    # Step 2: Transverse shear check with Timoshenko shear area
    fint.fill(0.0)
    mint.fill(0.0)
    # Relative angular velocity causing shear rate (omega_z1 = 0, omega_z2 = 0, but transverse vy = 10.0)
    v_shear = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    spring_advanced.forces_predit_type36(group, x, v_shear, vr, dt, fint, mint, np.array([0]))

    # Static equilibrium: nodal shear forces must be equal and opposite
    assert pytest.approx(fint[0, 1]) == -fint[1, 1]


# ============================================================================
# Test 4: Elastic Torsion and Bending Moments
# ============================================================================

def test_type36_elastic_torsion_and_bending():
    """Verify elastic torsional and bending moment responses."""
    model = Model()
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    area = 0.01
    rho = 7800.0
    e_mod = 2.0e11
    nu = 0.25
    g_mod = e_mod / (2.0 * (1.0 + nu))
    ixx = 1.0e-5
    iyy = 5.0e-6
    izz = 5.0e-6

    prop = PropType36(
        id=1, lutype=2, area=area, rho=rho, e=e_mod, nu=nu, g=g_mod,
        ixx=ixx, iyy=iyy, izz=izz, sig0=1.0e30
    )
    group = MockElementGroup([[0, 1]], [(slice(0, 1), None, prop)])
    log = MockMessageLog()
    spring.init_group(group, model, log)

    dt = 1e-4
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    x = model.x0.copy()
    v = np.zeros((2, 3))

    # Apply pure twist rate about X (omega_x2 - omega_x1 = 2.0 rad/s)
    vr = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    spring_advanced.forces_predit_type36(group, x, v, vr, dt, fint, mint, np.array([0]))

    # Torsional moment = G * Ixx / L0 * dtheta = (8.0e10 * 1.0e-5 / 1.0) * (2.0 * 1e-4) = 160.0 N*m
    expected_mx = (g_mod * ixx / 1.0) * 2.0 * dt
    assert pytest.approx(mint[0, 0], rel=1e-3) == expected_mx
    assert pytest.approx(mint[1, 0], rel=1e-3) == -expected_mx


# ============================================================================
# Test 5: Exact Nodal Force and Moment Equilibrium (R5CUM3)
# ============================================================================

def test_type36_static_and_moment_equilibrium():
    """Verify exact static force balance f1 + f2 = 0 and moment balance m1 + m2 + r12 x (-f2) = 0."""
    model = Model()
    model.x0 = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 6.0, 3.0],
    ])
    prop = PropType36(
        id=1, lutype=2, area=0.02, rho=7850.0, e=2.1e11, nu=0.3,
        ixx=2.0e-6, iyy=1.0e-6, izz=1.0e-6, sig0=1.0e30
    )
    group = MockElementGroup([[0, 1]], [(slice(0, 1), None, prop)])
    log = MockMessageLog()
    spring.init_group(group, model, log)

    # General arbitrary 3D velocities and rotations
    v = np.array([[1.5, -2.0, 0.5], [-0.5, 3.0, 1.2]])
    vr = np.array([[0.2, 0.4, -0.1], [-0.3, 0.1, 0.5]])
    dt = 1e-4
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))

    spring_advanced.forces_predit_type36(group, model.x0, v, vr, dt, fint, mint, np.array([0]))

    # Force equilibrium: sum F = f1 + f2 = 0
    np.testing.assert_allclose(fint[0] + fint[1], 0.0, atol=1e-9)

    # Moment equilibrium about node 1:
    # Sum M = m1 + m2 + (x2 - x1) x (-f2) = 0 (since force on node 2 is -f_elem, internal force opposing)
    dx = model.x0[1] - model.x0[0]
    total_moment = mint[0] + mint[1] + np.cross(dx, fint[1])
    np.testing.assert_allclose(total_moment, 0.0, atol=1e-8)


# ============================================================================
# Test 6: Plastic Radial Return Mapping (RUSER36)
# ============================================================================

def test_type36_plasticity_radial_return():
    """Verify onset of plasticity when SVM > sig0 and radial return onto yield surface."""
    model = Model()
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    sig0 = 2.5e8  # 250 MPa yield stress
    hpla = 1.0e9  # 1 GPa linear hardening
    area = 0.01

    prop = PropType36(
        id=1, lutype=2, area=area, rho=7800.0, e=2.0e11, nu=0.3,
        ixx=1e-6, iyy=5e-7, izz=5e-7, sig0=sig0, hpla=hpla, m=1.0,
        pr=1.0e30, ps=1.0e30
    )
    group = MockElementGroup([[0, 1]], [(slice(0, 1), None, prop)])
    log = MockMessageLog()
    spring.init_group(group, model, log)

    dt = 1e-3
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))

    # Pull fast enough to exceed yield: vx = 2.0 m/s -> elastic trial stress = 2.0e11 * (2.0 / 1.0) * 1e-3 = 4.0e8 Pa > 2.5e8 Pa
    v = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    vr = np.zeros((2, 3))

    spring_advanced.forces_predit_type36(group, model.x0, v, vr, dt, fint, mint, np.array([0]))

    # Check that plastic strain accumulated in UVAR(11)
    eps_p = group.state["t36_uvar1"][0]
    assert eps_p > 0.0

    # Check that the axial force is bounded by the yielded force (sig_y * Area)
    expected_yield_stress = sig0 + hpla * eps_p
    actual_stress = fint[0, 0] / area
    assert pytest.approx(actual_stress, rel=1e-2) == expected_yield_stress


# ============================================================================
# Test 7: Progressive Damage Evolution
# ============================================================================

def test_type36_progressive_damage():
    """Verify damage accumulation when eps_p > Ps and degradation of elastic modulus."""
    model = Model()
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    sig0 = 1.0e8
    hpla = 0.0
    ps = 1e-4   # damage initiation strain
    pr = 5e-3   # rupture strain
    dc = 0.8    # max critical damage

    prop = PropType36(
        id=1, lutype=2, area=0.01, rho=7800.0, e=2.0e11, nu=0.3,
        ixx=1e-6, iyy=5e-7, izz=5e-7, sig0=sig0, hpla=hpla,
        ps=ps, pr=pr, dc=dc
    )
    group = MockElementGroup([[0, 1]], [(slice(0, 1), None, prop)])
    log = MockMessageLog()
    spring.init_group(group, model, log)

    dt = 1e-3
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    v = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    vr = np.zeros((2, 3))

    # Pull past Ps without exceeding rupture strain Pr
    for _ in range(2):
        fint.fill(0.0)
        mint.fill(0.0)
        spring_advanced.forces_predit_type36(group, model.x0, v, vr, dt, fint, mint, np.array([0]))

    d_val = group.state["t36_uvar5"][0]
    eps_p = group.state["t36_uvar1"][0]
    assert eps_p > ps
    assert d_val > 0.0
    # In ruser36.F, once eps_p > ps, dD = dc * dpla / (pr - ps)
    expected_d = dc * eps_p / (pr - ps)
    assert pytest.approx(d_val, rel=1e-2) == expected_d


# ============================================================================
# Test 8: Rupture and Element Deactivation (OFF = 0)
# ============================================================================

def test_type36_rupture_deactivation():
    """Verify that when eps_p >= Pr, the element deactivates (off=0) and forces drop to zero."""
    model = Model()
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    sig0 = 1.0e8
    pr = 1e-3  # small failure strain
    ps = 1e-4

    prop = PropType36(
        id=1, lutype=2, area=0.01, rho=7800.0, e=2.0e11, nu=0.3,
        ixx=1e-6, iyy=5e-7, izz=5e-7, sig0=sig0, hpla=0.0,
        ps=ps, pr=pr, dc=1.0
    )
    group = MockElementGroup([[0, 1]], [(slice(0, 1), None, prop)])
    log = MockMessageLog()
    spring.init_group(group, model, log)

    dt = 1e-3
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    v = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])  # large pull velocity
    vr = np.zeros((2, 3))

    # Step to cause rupture
    spring_advanced.forces_predit_type36(group, model.x0, v, vr, dt, fint, mint, np.array([0]))

    assert group.state["off"][0] == 0.0

    # Next step after rupture: internal forces and moments must be exactly zero
    fint.fill(0.0)
    mint.fill(0.0)
    spring_advanced.forces_predit_type36(group, model.x0, v, vr, dt, fint, mint, np.array([0]))
    np.testing.assert_allclose(fint, 0.0)
    np.testing.assert_allclose(mint, 0.0)


# ============================================================================
# Test 9: Internal Energy Balance (R5DEF3)
# ============================================================================

def test_type36_energy_accounting():
    """Verify that internal energy in st['eint'] correctly accumulates mechanical work without leaks."""
    model = Model()
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    area = 0.01
    prop = PropType36(
        id=1, lutype=2, area=area, rho=7800.0, e=2.0e11, nu=0.3,
        ixx=1e-6, iyy=5e-7, izz=5e-7, sig0=1.0e30
    )
    group = MockElementGroup([[0, 1]], [(slice(0, 1), None, prop)])
    log = MockMessageLog()
    spring.init_group(group, model, log)

    dt = 1e-4
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vr = np.zeros((2, 3))

    total_ext_work = 0.0
    for step in range(10):
        fint.fill(0.0)
        mint.fill(0.0)
        f_before = group.state["t36_forces"][0, 0]
        spring_advanced.forces_predit_type36(group, model.x0, v, vr, dt, fint, mint, np.array([0]))
        f_after = group.state["t36_forces"][0, 0]
        # Trapezoidal external work done on node 2: 0.5 * (F_old + F_new) * v * dt
        total_ext_work += 0.5 * (f_before + f_after) * 1.0 * dt

    eint = group.state["eint"][0]
    assert eint > 0.0
    assert pytest.approx(eint, rel=1e-4) == total_ext_work
