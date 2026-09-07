"""
Comprehensive unit test suite for Milestone M509:
4-Node 2D Quadrilateral Solid Element (/QUAD + /PROP/SOLID, /PROP/TYPE15, /PROP/QUAD)
Axisymmetric (N2D=1) & Plane Strain (N2D=2) Kinematics, Flanagan-Belytschko 1-Point
Integration, Hourglass Control (qhvis2.F), Jaumann Stress Rate (qrota2.F), 12-DOF
Tangent/Geometric Stiffness, Consistent Mass, Failure Handling (AUD-010) & Momentum/Energy Conservation.
"""

from __future__ import annotations

import pytest
import numpy as np

from pyradioss.elements import solid_quad
from pyradioss.common.constants import EM20, EP30
from pyradioss.model.model import ElementGroup, Model


# ----------------------------------------------------------------------------
# Test Helpers & Dummy Objects
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
        self.params = {
            "E": E,
            "nu": nu,
            "sig_y": 200.0e6,
            "b": 500.0e6,
            "n": 0.5,
            "c": 0.0,
            "eps_max": 1e30,
            "eps_p_max": 1e30,
        }


class _DummyProp:
    def __init__(self, qa=1.1, qb=0.05, h=0.1):
        self.qa = qa
        self.qb = qb
        self.h = h
        self.params = {"qa": qa, "qb": qb, "h": h}


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


def _create_unit_quad(n2d=2, y0=0.0, z0=0.0, size=1.0):
    """Create a single 2D quad in the (Y, Z) plane."""
    model = Model()
    model.n2d = n2d
    # 4 nodes in counter-clockwise order:
    # 0: (y0, z0)
    # 1: (y0 + size, z0)
    # 2: (y0 + size, z0 + size)
    # 3: (y0, z0 + size)
    x0 = np.array([
        [0.0, y0, z0],
        [0.0, y0 + size, z0],
        [0.0, y0 + size, z0 + size],
        [0.0, y0, z0 + size],
    ], dtype=np.float64)
    model.x0 = x0.copy()
    model.x = x0.copy()
    model.v = np.zeros((4, 3), dtype=np.float64)
    model.vr = np.zeros((4, 3), dtype=np.float64)

    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)
    group = ElementGroup(ids=np.array([1], dtype=np.int64), conn=conn, part=np.array([0], dtype=np.int64))
    return model, group


# ----------------------------------------------------------------------------
# 1. Geometry and volume in plane strain (N2D=2)
# ----------------------------------------------------------------------------
def test_quad_geometry_plane_strain_unit_square():
    model, group = _create_unit_quad(n2d=2, size=1.0)
    PY1, PY2, PZ1, PZ2, dndx, area, vol = solid_quad._geometry(model.x0[None, ...], n2d=2)

    assert area[0] == pytest.approx(1.0, rel=1e-12)
    assert vol[0] == pytest.approx(1.0, rel=1e-12)
    assert dndx.shape == (1, 4, 2)

    # Partition of unity: sum_i dN_i / dy = 0, sum_i dN_i / dz = 0
    assert np.allclose(dndx[0].sum(axis=0), 0.0, atol=1e-14)

    # Linear coordinate reproduction
    Y = model.x0[:, 1]
    Z = model.x0[:, 2]
    assert np.isclose(np.sum(Y * dndx[0, :, 0]), 1.0, atol=1e-14)
    assert np.isclose(np.sum(Z * dndx[0, :, 1]), 1.0, atol=1e-14)
    assert np.isclose(np.sum(Y * dndx[0, :, 1]), 0.0, atol=1e-14)
    assert np.isclose(np.sum(Z * dndx[0, :, 0]), 0.0, atol=1e-14)


# ----------------------------------------------------------------------------
# 2. Geometry and revolution volume in axisymmetric mode (N2D=1)
# ----------------------------------------------------------------------------
def test_quad_geometry_axisymmetric_revolution_volume():
    # Ring from Y=10 to 11, Z=0 to 1 -> Area = 1, Yavg = 10.5
    model, group = _create_unit_quad(n2d=1, y0=10.0, z0=0.0, size=1.0)
    _, _, _, _, _, area, vol = solid_quad._geometry(model.x0[None, ...], n2d=1)

    assert area[0] == pytest.approx(1.0, rel=1e-12)
    # Fortran qvolu2.F exact 1-radian volume is Yavg * Area = 10.5
    assert vol[0] == pytest.approx(10.5, rel=1e-12)


# ----------------------------------------------------------------------------
# 3. Degenerate and collinear geometry error resilience
# ----------------------------------------------------------------------------
def test_quad_degenerate_and_collinear_safety():
    # 4 collapsed points
    xe_deg = np.zeros((1, 4, 3))
    PY1, PY2, PZ1, PZ2, dndx, area, vol = solid_quad._geometry(xe_deg, n2d=2)
    assert area[0] == 0.0
    assert vol[0] == 0.0
    assert np.all(dndx == 0.0)
    assert not np.isnan(dndx).any()

    # 4 collinear points along Y axis
    xe_col = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]])
    _, _, _, _, dndx_col, area_col, vol_col = solid_quad._geometry(xe_col, n2d=2)
    assert area_col[0] == pytest.approx(0.0, abs=1e-12)
    assert vol_col[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(dndx_col == 0.0)
    assert not np.isnan(dndx_col).any()


# ----------------------------------------------------------------------------
# 4. Constant strain gradient operator on arbitrary convex quad
# ----------------------------------------------------------------------------
def test_quad_constant_strain_gradient_operator():
    # General non-rectangular convex quad
    x0 = np.array([
        [0.0, 1.0, 1.0],
        [0.0, 4.0, 1.5],
        [0.0, 3.5, 3.8],
        [0.0, 0.8, 3.0],
    ])
    _, _, _, _, dndx, area, vol = solid_quad._geometry(x0[None, ...], n2d=2)
    assert area[0] > 0.0

    # 1) Partition of unity
    assert np.allclose(dndx[0].sum(axis=0), 0.0, atol=1e-14)

    # 2) Linear coordinate reproduction
    Y = x0[:, 1]
    Z = x0[:, 2]
    assert np.isclose(np.sum(Y * dndx[0, :, 0]), 1.0, atol=1e-14)
    assert np.isclose(np.sum(Z * dndx[0, :, 1]), 1.0, atol=1e-14)
    assert np.isclose(np.sum(Y * dndx[0, :, 1]), 0.0, atol=1e-14)
    assert np.isclose(np.sum(Z * dndx[0, :, 0]), 0.0, atol=1e-14)


# ----------------------------------------------------------------------------
# 5. Characteristic length and Courant time step
# ----------------------------------------------------------------------------
def test_quad_characteristic_length_and_courant_step():
    model, group = _create_unit_quad(n2d=2, size=1.0)
    mat = _DummyMat(law=1, rho0=1000.0, E=1.0e8, nu=0.0)  # c = sqrt(E/rho) = sqrt(1e5) = 316.2277
    prop = _DummyProp(qa=0.0, qb=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_quad.init_group(group, model, _DummyLog())

    # Unit square: area = 1.0, max diagonal = sqrt(2), lc = 1 / sqrt(2) ~ 0.70710678
    lc = solid_quad._char_length(model.x0[None, ...], np.array([1.0]))
    expected_lc = 1.0 / np.sqrt(2.0)
    assert lc[0] == pytest.approx(expected_lc, rel=1e-12)

    c_sound = np.sqrt(1.0e8 / 1000.0)
    dt_crit = solid_quad.forces(group, model.x0, model.v, model.vr, 0.0, None, None)
    expected_dt = 1.0 * expected_lc / c_sound
    assert dt_crit[0] == pytest.approx(expected_dt, rel=1e-6)


# ----------------------------------------------------------------------------
# 6. Empty group handling across all element APIs
# ----------------------------------------------------------------------------
def test_quad_empty_group_handling():
    empty_group = ElementGroup(ids=np.zeros(0, dtype=np.int64),
                               conn=np.zeros((0, 4), dtype=np.int64),
                               part=np.zeros(0, dtype=np.int64))
    log = _DummyLog()
    model = Model()

    node_idx, mass_c, _ = solid_quad.init_group(empty_group, model, log)
    assert len(node_idx) == 0
    assert len(mass_c) == 0

    dt_crit = solid_quad.forces(empty_group, model.x0, model.v, model.vr, 0.01, None, None)
    assert len(dt_crit) == 0

    ke, edofs = solid_quad.tangent(empty_group, model.x0)
    assert ke.shape == (0, 12, 12)
    assert edofs.shape == (0, 12)

    kg, edofs_g = solid_quad.kgeo(empty_group, model.x0)
    assert kg.shape == (0, 12, 12)

    me, edofs_m = solid_quad.consistent_mass(empty_group)
    assert me.shape == (0, 12, 12)

    solid_quad.static_internal_forces(empty_group, model.x0, None, None, None, None)
    solid_quad.implicit_internal_forces(empty_group, model.x0, None, None, None, None, nlgeom=False)
    solid_quad.implicit_internal_forces(empty_group, model.x0, None, None, None, None, nlgeom=True)


# ----------------------------------------------------------------------------
# 7. Axisymmetric radial expansion and hoop restoring force AX1
# ----------------------------------------------------------------------------
def test_quad_axisymmetric_radial_expansion_hoop_stress():
    model, group = _create_unit_quad(n2d=1, y0=10.0, z0=0.0, size=1.0)
    mat = _DummyMat(law=1, rho0=1.0, E=2.1e5, nu=0.0)
    prop = _DummyProp(qa=0.0, qb=0.0, h=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_quad.init_group(group, model, _DummyLog())

    dt = 1.0
    v = np.zeros((4, 3))
    v[:, 1] = 1.0  # pure radial velocity
    fint = np.zeros((4, 3))
    solid_quad.forces(group, model.x0, v, model.vr, dt, fint, None)

    # DYY = 0, DZZ = 0, DTT = v_y / Yavg = 1.0 / 10.5
    # sig_xx (hoop) = E * DTT = 210000 / 10.5 = 20000.0
    sig = group.state["sig"][0]
    assert sig[0] == pytest.approx(210000.0 / 10.5, rel=1e-5)
    assert sig[1] == pytest.approx(0.0, abs=1e-8)
    assert sig[2] == pytest.approx(0.0, abs=1e-8)

    # AX1 = sig_xx * Area / (4 * Yavg) = 20000.0 * 1.0 / (4 * 10.5)
    # Resisting force pulls inward (-Y)
    expected_f_radial = -210000.0 / 10.5 / (4.0 * 10.5)
    assert np.allclose(fint[:, 1], expected_f_radial, rtol=1e-4)
    assert np.allclose(fint[:, 2], 0.0, atol=1e-10)


# ----------------------------------------------------------------------------
# 8. Plane strain uniaxial compression constrained modulus
# ----------------------------------------------------------------------------
def test_quad_plane_strain_uniaxial_compression():
    model, group = _create_unit_quad(n2d=2, size=1.0)
    mat = _DummyMat(law=1, rho0=1000.0, E=2.1e5, nu=0.3)
    prop = _DummyProp(qa=0.0, qb=0.0, h=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_quad.init_group(group, model, _DummyLog())

    dt = 1.0
    v = np.zeros((4, 3))
    # Compressing top edge (nodes 2 and 3) downward in Z by 0.001
    v[2, 2] = -0.001
    v[3, 2] = -0.001

    fint = np.zeros((4, 3))
    solid_quad.forces(group, model.x0, v, model.vr, dt, fint, None)

    # eps_zz = -0.001, nu = 0.3
    # K = E / (3*(1 - 2nu)) = 210000 / 1.2 = 175000
    # G = E / (2*(1 + nu)) = 210000 / 2.6 = 80769.23077
    # P-wave modulus = K + 4/3 G = 175000 + 107692.3077 = 282692.3077
    # sig_zz = P * eps_zz = -282.6923
    # sig_yy = (K - 2/3 G) * eps_zz = (175000 - 53846.15) * -0.001 = -121.1538
    sig = group.state["sig"][0]
    assert sig[2] == pytest.approx(-282.6923, rel=1e-4)
    assert sig[1] == pytest.approx(-121.1538, rel=1e-4)

    # Resisting force on top nodes pushes back UP (+Z)
    expected_f_top = 0.5 * 282.6923
    assert fint[2, 2] == pytest.approx(expected_f_top, rel=1e-4)
    assert fint[3, 2] == pytest.approx(expected_f_top, rel=1e-4)
    # Bottom nodes push DOWN (-Z)
    assert fint[0, 2] == pytest.approx(-expected_f_top, rel=1e-4)
    assert fint[1, 2] == pytest.approx(-expected_f_top, rel=1e-4)


# ----------------------------------------------------------------------------
# 9. Flanagan-Belytschko hourglass mode orthogonalization and damping
# ----------------------------------------------------------------------------
def test_quad_flanagan_belytschko_hourglass_orthogonalization():
    model, group = _create_unit_quad(n2d=2, size=1.0)
    mat = _DummyMat(law=1, rho0=1000.0, E=2.0e7, nu=0.3)
    prop = _DummyProp(qa=0.0, qb=0.0, h=0.1)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_quad.init_group(group, model, _DummyLog())

    # 1) Rigid translation produces zero hourglass energy
    v_trans = np.tile([0.0, 2.0, -1.5], (4, 1))
    fint = np.zeros((4, 3))
    solid_quad.forces(group, model.x0, v_trans, model.vr, 0.01, fint, None)
    assert group.state["ehour"][0] == 0.0

    # 2) Pure hourglass deformation v_y = h = [1, -1, 1, -1]
    group.state["ehour"].fill(0.0)
    v_hg = np.zeros((4, 3))
    v_hg[:, 1] = [1.0, -1.0, 1.0, -1.0]
    fint.fill(0.0)
    solid_quad.forces(group, model.x0, v_hg, model.vr, 0.01, fint, None)

    # Strain rates are zero for pure hourglass mode
    # Hourglass force opposes motion and books positive energy
    assert group.state["ehour"][0] > 0.0
    # Hourglass forces in Y should alternate signs [-, +, -, +]
    assert fint[0, 1] < 0.0
    assert fint[1, 1] > 0.0
    assert fint[2, 1] < 0.0
    assert fint[3, 1] > 0.0


# ----------------------------------------------------------------------------
# 10. Jaumann stress rate rotational invariance under spinning
# ----------------------------------------------------------------------------
def test_quad_jaumann_rate_rotational_invariance():
    model, group = _create_unit_quad(n2d=2, size=1.0)
    mat = _DummyMat(law=1, rho0=1000.0, E=1.0e7, nu=0.0)
    prop = _DummyProp(qa=0.0, qb=0.0, h=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_quad.init_group(group, model, _DummyLog())

    # Initial state with pure shear stress: sig_yz = 1000 Pa
    group.state["sig"][0, 4] = 1000.0

    # Spin at angular rate omega = 2.0 rad/s in (Y, Z) plane:
    # v_y = -omega * (Z - 0.5), v_z = omega * (Y - 0.5)
    omega = 2.0
    dt = 0.05
    Y_c = model.x0[:, 1] - 0.5
    Z_c = model.x0[:, 2] - 0.5
    v = np.zeros((4, 3))
    v[:, 1] = -omega * Z_c
    v[:, 2] = omega * Y_c

    fint = np.zeros((4, 3))
    solid_quad.forces(group, model.x0, v, model.vr, dt, fint, None)

    # Jaumann rate: DZY - DYZ = omega - (-omega) = 2 omega
    # WYZ = 0.5 * dt * 2 omega = omega * dt = 2.0 * 0.05 = 0.1 rad
    # Q1 = 2 * sig_yz * WYZ = 2 * 1000 * 0.1 = 200 Pa
    # sig_yy = +200, sig_zz = -200
    sig = group.state["sig"][0]
    assert sig[1] == pytest.approx(200.0, rel=1e-5)
    assert sig[2] == pytest.approx(-200.0, rel=1e-5)


# ----------------------------------------------------------------------------
# 11. Rigid body translational and rotational velocity invariance
# ----------------------------------------------------------------------------
def test_quad_rigid_body_translation_and_rotation_invariance():
    model, group = _create_unit_quad(n2d=2, size=1.0)
    mat = _DummyMat(law=1, rho0=1000.0, E=2.0e7, nu=0.3)
    prop = _DummyProp(qa=0.0, qb=0.0, h=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_quad.init_group(group, model, _DummyLog())

    # 1) Translation
    v_trans = np.tile([0.0, 10.0, -8.0], (4, 1))
    fint = np.zeros((4, 3))
    solid_quad.forces(group, model.x0, v_trans, model.vr, 0.01, fint, None)
    assert np.allclose(fint, 0.0, atol=1e-10)
    assert np.allclose(group.state["sig"], 0.0, atol=1e-10)

    # 2) Rigid rotation around center
    omega = 1.5
    v_rot = np.zeros((4, 3))
    v_rot[:, 1] = -omega * (model.x0[:, 2] - 0.5)
    v_rot[:, 2] = omega * (model.x0[:, 1] - 0.5)
    fint.fill(0.0)
    solid_quad.forces(group, model.x0, v_rot, model.vr, 0.01, fint, None)
    assert np.allclose(fint, 0.0, atol=1e-10)
    assert np.allclose(group.state["sig"], 0.0, atol=1e-10)


# ----------------------------------------------------------------------------
# 12. Linear momentum conservation arbitrary in-plane motion
# ----------------------------------------------------------------------------
def test_quad_linear_momentum_conservation_arbitrary():
    model, group = _create_unit_quad(n2d=2, size=1.0)
    mat = _DummyMat(law=1, rho0=1500.0, E=1.0e8, nu=0.3)
    prop = _DummyProp(h=0.1)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_quad.init_group(group, model, _DummyLog())

    rng = np.random.default_rng(42)
    v = np.zeros((4, 3))
    v[:, 1:] = rng.uniform(-2.0, 2.0, (4, 2))
    fint = np.zeros((4, 3))
    solid_quad.forces(group, model.x0, v, model.vr, 0.01, fint, None)

    # Net force in Y and Z must be zero
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-8)


# ----------------------------------------------------------------------------
# 13. Angular momentum conservation arbitrary in-plane motion
# ----------------------------------------------------------------------------
def test_quad_angular_momentum_conservation_arbitrary():
    model, group = _create_unit_quad(n2d=2, size=1.0)
    mat = _DummyMat(law=1, rho0=1500.0, E=1.0e8, nu=0.3)
    prop = _DummyProp(h=0.1)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_quad.init_group(group, model, _DummyLog())

    rng = np.random.default_rng(123)
    v = np.zeros((4, 3))
    v[:, 1:] = rng.uniform(-1.5, 1.5, (4, 2))
    fint = np.zeros((4, 3))
    solid_quad.forces(group, model.x0, v, model.vr, 0.01, fint, None)

    # Torque in X: Y * FZ - Z * FY = 0
    torque_x = np.sum(model.x0[:, 1] * fint[:, 2] - model.x0[:, 2] * fint[:, 1])
    assert np.isclose(torque_x, 0.0, atol=1e-8)


# ----------------------------------------------------------------------------
# 14. Internal energy work balance
# ----------------------------------------------------------------------------
def test_quad_internal_energy_work_balance():
    model, group = _create_unit_quad(n2d=2, size=1.0)
    mat = _DummyMat(law=1, rho0=1000.0, E=1.0e7, nu=0.0)
    prop = _DummyProp(qa=0.0, qb=0.0, h=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_quad.init_group(group, model, _DummyLog())

    # Uniaxial tension in Z: v_z = 0.001 * Z
    v = np.zeros((4, 3))
    v[:, 2] = 0.001 * model.x0[:, 2]
    dt = 0.05
    fint = np.zeros((4, 3))
    solid_quad.forces(group, model.x0, v, model.vr, dt, fint, None)

    # External mechanical work done: W_ext = 0.5 * (-sum(fint * v * dt))
    w_ext = 0.5 * (-np.sum(fint * v) * dt)
    eint = group.state["eint"][0]
    assert eint == pytest.approx(w_ext, rel=1e-6)
    assert eint > 0.0


# ----------------------------------------------------------------------------
# 15. Bulk viscosity damping pressure
# ----------------------------------------------------------------------------
def test_quad_bulk_viscosity_damping_pressure():
    model, group = _create_unit_quad(n2d=2, size=1.0)
    mat = _DummyMat(law=1, rho0=1000.0, E=1.0e7, nu=0.0)
    prop = _DummyProp(qa=1.1, qb=0.05, h=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_quad.init_group(group, model, _DummyLog())

    # 1) Compression: trD < 0
    v_comp = np.zeros((4, 3))
    v_comp[:, 1:] = -0.1 * model.x0[:, 1:]
    fint = np.zeros((4, 3))
    solid_quad.forces(group, model.x0, v_comp, model.vr, 0.01, fint, None)
    w_visc = group.state["qvw_pend"][0]
    assert w_visc > 0.0  # Damping was booked

    # 2) Expansion: trD > 0
    group.state["qvw_pend"].fill(0.0)
    v_exp = np.zeros((4, 3))
    v_exp[:, 1:] = 0.1 * model.x0[:, 1:]
    fint.fill(0.0)
    solid_quad.forces(group, model.x0, v_exp, model.vr, 0.01, fint, None)
    assert group.state["qvw_pend"][0] == 0.0


# ----------------------------------------------------------------------------
# 16. Element deletion and failure handling (AUD-010 Fix)
# ----------------------------------------------------------------------------
def test_quad_element_deletion_and_failure():
    model, group = _create_unit_quad(n2d=2, size=1.0)
    mat = _DummyMat(law=1, rho0=1000.0, E=1.0e7, nu=0.3)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_quad.init_group(group, model, _DummyLog())

    # Delete element
    group.state["off"][0] = 0.0
    v = np.zeros((4, 3))
    v[:, 1:] = 1.0
    fint = np.zeros((4, 3))
    dt_crit = solid_quad.forces(group, model.x0, v, model.vr, 0.01, fint, None)

    assert np.allclose(fint, 0.0)
    assert dt_crit[0] == EP30


# ----------------------------------------------------------------------------
# 17. LAW0 void 2D quad material handling
# ----------------------------------------------------------------------------
def test_quad_law0_void_solid_material():
    model, group = _create_unit_quad(n2d=2, size=1.0)
    mat = _DummyMat(law=0, rho0=0.0, E=0.0, nu=0.0)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_quad.init_group(group, model, _DummyLog())

    v = np.zeros((4, 3))
    v[:, 1:] = 1.0
    fint = np.zeros((4, 3))
    dt_crit = solid_quad.forces(group, model.x0, v, model.vr, 0.01, fint, None)

    assert np.allclose(fint, 0.0)
    assert np.allclose(group.state["sig"], 0.0)
    assert dt_crit[0] == EP30

    ke, _ = solid_quad.tangent(group, model.x0)
    assert np.allclose(ke, 0.0)


# ----------------------------------------------------------------------------
# 18. Consistent mass matrix
# ----------------------------------------------------------------------------
def test_quad_consistent_mass_matrix():
    model, group = _create_unit_quad(n2d=2, size=1.0)
    mat = _DummyMat(law=1, rho0=36.0, E=100.0, nu=0.0)  # total mass = 36.0
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_quad.init_group(group, model, _DummyLog())

    me, edofs = solid_quad.consistent_mass(group)
    assert me.shape == (1, 12, 12)
    assert edofs.shape == (1, 12)
    M = me[0]

    # 1) Symmetry
    assert np.allclose(M, M.T, atol=1e-14)

    # 2) Positive definiteness
    eigvals = np.linalg.eigvalsh(M)
    assert (eigvals > 0.0).all()

    # 3) Kinetic energy for rigid translation
    v_rig = np.tile([1.0, 2.0, -3.0], 4)
    ke = 0.5 * v_rig @ M @ v_rig
    expected_ke = 0.5 * group.state["mass"][0] * (1.0**2 + 2.0**2 + (-3.0)**2)
    assert ke == pytest.approx(expected_ke, rel=1e-12)


# ----------------------------------------------------------------------------
# 19. Geometric stiffness matrix
# ----------------------------------------------------------------------------
def test_quad_geometric_stiffness_matrix():
    model, group = _create_unit_quad(n2d=2, size=1.0)
    mat = _DummyMat(law=1, rho0=1.0, E=100.0, nu=0.0)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_quad.init_group(group, model, _DummyLog())

    # Zero stress gives zero geometric stiffness
    k_geo, edofs = solid_quad.kgeo(group, model.x0)
    assert np.allclose(k_geo, 0.0)

    # Tensile and shear stress
    group.state["sig"][0] = [0.0, 100.0, 50.0, 0.0, 20.0, 0.0]
    k_geo, _ = solid_quad.kgeo(group, model.x0)
    Kg = k_geo[0]

    # Symmetry
    assert np.allclose(Kg, Kg.T, atol=1e-14)

    # Rigid translations in nullspace
    for d in range(3):
        v_trans = np.zeros(12)
        v_trans[d::3] = 1.0
        assert np.allclose(Kg @ v_trans, 0.0, atol=1e-12)


# ----------------------------------------------------------------------------
# 20. Material tangent stiffness nullspace and directional derivative
# ----------------------------------------------------------------------------
def test_quad_material_tangent_stiffness_and_directional_derivative():
    model, group = _create_unit_quad(n2d=2, size=1.0)
    mat = _DummyMat(law=1, rho0=1.0, E=2.1e7, nu=0.3)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_quad.init_group(group, model, _DummyLog())

    ke, edofs = solid_quad.tangent(group, model.x0)
    K = ke[0]

    # Symmetry
    assert np.allclose(K, K.T, atol=1e-6)

    # In-plane rigid translations in Y (d=1) and Z (d=2) in nullspace
    for d in (1, 2):
        v_trans = np.zeros(12)
        v_trans[d::3] = 1.0
        assert np.allclose(K @ v_trans, 0.0, atol=1e-7)

    # In-plane rigid rotation around center (0.5, 0.5)
    coords = model.x0
    v_rot = np.zeros((4, 3))
    v_rot[:, 1] = -(coords[:, 2] - 0.5)
    v_rot[:, 2] = coords[:, 1] - 0.5
    assert np.allclose(K @ v_rot.reshape(12), 0.0, atol=1e-7)

    # Directional derivative consistency with implicit internal forces
    rng = np.random.default_rng(99)
    du = 1e-6 * rng.uniform(-1.0, 1.0, (4, 3))
    f_lin = np.zeros((4, 3))
    solid_quad.implicit_internal_forces(group, model.x0, du, None, f_lin, None, nlgeom=False)

    expected_f = -(K @ du.reshape(12)).reshape(4, 3)
    assert np.allclose(f_lin, expected_f, atol=1e-10)
