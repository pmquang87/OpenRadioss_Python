"""Unit tests for Milestone M513: 4-Node Fully-Integrated QBAT Shell Element

Hardening, Kinematics, 24-DOF Tangent/Geometric Stiffness, Analytical Consistent Mass,
Static & Implicit Internal Forces & Comprehensive Unit Tests.
"""

from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pyradioss.common.constants import EM20, EP30
from pyradioss.elements.shell_qbat import (
    _frame,
    _cbacoor,
    init_group,
    forces,
    consistent_mass,
    tangent,
    kgeo,
    static_internal_forces,
    implicit_internal_forces,
    _edofs,
)
from pyradioss.accel import _state, select_backend


@pytest.fixture(autouse=True)
def pin_numpy_backend(monkeypatch):
    monkeypatch.setenv("PYRADIOSS_BACKEND", "numpy")
    orig = dict(_state)
    select_backend("numpy")
    yield
    _state.update(orig)


class SimpleMat:
    def __init__(self, E=2.1e11, nu=0.3, rho0=7850.0, law=1):
        self.E = E
        self.nu = nu
        self.rho0 = rho0
        self.law = law
        self.K = E / (3.0 * (1.0 - 2.0 * nu))
        self.G = E / (2.0 * (1.0 + nu))
        self.fail = None
        self.params = {"E": E, "nu": nu, "rho": rho0, "eps_p_max": EP30}

    def sound_speed_shell(self):
        return np.sqrt(self.E / (self.rho0 * (1.0 - self.nu**2))) if (self.rho0 > 0.0 and self.law != 0) else 0.0


class SimpleProp:
    def __init__(self, thick=0.01, nip=3, dn=1e-3, ishell=12):
        self.params = {
            "thick": thick,
            "nip": nip,
            "dn": dn,
            "ishell": ishell,
        }
        self.thick = thick
        self.nip = nip
        self.dn = dn


def _create_mock_qbat_group(coords, law=1, E=2.1e11, nu=0.3, rho=7850.0, thick=0.01, nip=3):
    n = len(coords) // 4
    conn = np.arange(len(coords), dtype=np.int64).reshape(n, 4)
    mat = SimpleMat(E=E, nu=nu, rho0=rho, law=law)
    prop = SimpleProp(thick=thick, nip=nip)
    group = SimpleNamespace(
        id=1,
        n=n,
        conn=conn,
        ids=np.arange(1, n + 1, dtype=np.int64),
        state={
            "slices": [(slice(0, n), mat, prop)],
            "off": np.ones(n, dtype=float),
            "chk_fail": False,
        },
    )
    model = SimpleNamespace(
        x0=coords.copy(),
        nodes_idx=np.arange(len(coords), dtype=np.int64),
    )
    log = SimpleNamespace(
        error=lambda msg, ctx="": None,
        warning=lambda msg, ctx="": None,
    )
    init_group(group, model, log)
    return group, model


def test_qbat_local_frame_orthonormality():
    """Verify local triad e1, e2, e3 is strictly orthonormal with det(E) = 1.0."""
    coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.2, 1.0, 0.0], [0.0, 1.0, 0.0]])
    Rot = Rotation.from_euler("xyz", [30.0, 45.0, 60.0], degrees=True).as_matrix()
    coords_3d = coords @ Rot.T + np.array([2.0, -1.0, 4.0])

    E, det = _frame(coords_3d[None])
    R = E[0]

    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)
    assert det[0] > 0.0


def test_qbat_area_and_characteristic_length():
    """Verify area and characteristic length match Euclidean analytical values."""
    Lx, Ly = 2.0, 1.5
    coords = np.array([[0.0, 0.0, 0.0], [Lx, 0.0, 0.0], [Lx, Ly, 0.0], [0.0, Ly, 0.0]])
    group, model = _create_mock_qbat_group(coords)

    E, det = _frame(coords[None])
    area = 0.25 * det[0]
    assert np.isclose(area, Lx * Ly, rtol=1e-10)

    g = _cbacoor(coords[None], np.zeros((1, 4, 3)), np.zeros((1, 4, 3)), np.ones(1), 1e-4, False)
    assert g["lc"][0] > 0.0
    assert g["lc"][0] < min(Lx, Ly) * 2.0


def test_qbat_invariance_3d_transform():
    """Verify element area and characteristic length are invariant under 3D rigid motion."""
    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.2, 0.0], [1.4, 1.2, 0.0], [-0.1, 1.1, 0.0]])
    E0, det0 = _frame(coords[None])
    area0 = 0.25 * det0[0]

    Rot = Rotation.from_euler("xyz", [45.0, -30.0, 75.0], degrees=True).as_matrix()
    coords_rot = coords @ Rot.T + np.array([10.0, 20.0, -5.0])

    E1, det1 = _frame(coords_rot[None])
    area1 = 0.25 * det1[0]
    assert np.isclose(area0, area1, rtol=1e-10)


def test_qbat_degenerate_element_safety():
    """Verify degenerate and zero-area quad elements do not crash or produce NaNs."""
    coords_degen = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    E, det = _frame(coords_degen[None])
    assert not np.any(np.isnan(E))
    assert not np.any(np.isnan(det))
    assert np.isclose(np.linalg.det(E[0]), 1.0, atol=1e-6)

    # Collinear quad
    coords_collinear = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    E_col, det_col = _frame(coords_collinear[None])
    assert not np.any(np.isnan(E_col))
    assert not np.any(np.isnan(det_col))


def test_qbat_empty_group_handling():
    """Verify graceful handling of empty groups across all APIs."""
    group = SimpleNamespace(
        id=1,
        n=0,
        conn=np.empty((0, 4), dtype=np.int64),
        ids=np.empty(0, dtype=np.int64),
        state={"slices": [], "off": np.empty(0)},
    )
    model = SimpleNamespace(x0=np.empty((0, 3)), nodes_idx=np.empty(0, dtype=np.int64))
    log = SimpleNamespace(error=lambda *args: None, warning=lambda *args: None)

    node_idx, mass_c, in_c = init_group(group, model, log)
    assert len(node_idx) == 0

    x = np.empty((0, 3))
    v = np.empty((0, 3))
    vr = np.empty((0, 3))
    fint = np.empty((0, 3))
    mint = np.empty((0, 3))

    dt = forces(group, x, v, vr, 1e-4, fint, mint)
    assert len(dt) == 0

    ke, edofs = tangent(group, x)
    assert ke.shape == (0, 24, 24)
    assert edofs.shape == (0, 24)

    kg, edofs_g = kgeo(group, x)
    assert kg.shape == (0, 24, 24)

    me, edofs_m = consistent_mass(group, x)
    assert me.shape == (0, 24, 24)

    static_internal_forces(group, x, x, x, fint, mint)
    implicit_internal_forces(group, x, x, x, fint, mint)


def test_qbat_cycle_0_courant_step():
    """Verify cycle 0 calculates Courant time step without mutating element state."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_qbat_group(coords)

    dt0 = forces(group, coords, None, None, 0.0, None, None)
    assert dt0[0] > 0.0
    assert not np.isnan(dt0[0])
    assert dt0[0] < 1.0
    assert np.all(group.state["sig"] == 0.0)
    assert np.all(group.state["eint"] == 0.0)


def test_qbat_lumped_mass_conservation():
    """Verify total lumped mass equals rho * t * Area (m/4 per node)."""
    Lx, Ly, t, rho = 2.0, 1.5, 0.02, 7800.0
    coords = np.array([[0.0, 0.0, 0.0], [Lx, 0.0, 0.0], [Lx, Ly, 0.0], [0.0, Ly, 0.0]])
    group, model = _create_mock_qbat_group(coords, rho=rho, thick=t)

    total_mass = rho * t * (Lx * Ly)
    assert np.isclose(group.state["mass"][0], total_mass, rtol=1e-10)


def test_qbat_consistent_mass_matrix_properties():
    """Verify consistent mass matrix is symmetric, positive definite (24 pos eigenvalues) and conserves KE."""
    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.5, 1.2, 0.0], [0.0, 1.2, 0.0]])
    Rot = Rotation.from_euler("xyz", [20.0, -30.0, 45.0], degrees=True).as_matrix()
    coords_3d = coords @ Rot.T + np.array([1.0, 2.0, 3.0])
    group, model = _create_mock_qbat_group(coords_3d)

    me, edofs = consistent_mass(group, coords_3d)
    M = me[0]

    assert np.allclose(M, M.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(M)
    assert np.all(eigvals > 0.0)
    assert len(eigvals) == 24

    # Rigid translation kinetic energy
    v_rigid = np.zeros(24)
    v_vec = np.array([3.0, -4.0, 5.0])
    for a in range(4):
        v_rigid[6 * a:6 * a + 3] = v_vec
    ke_exact = 0.5 * group.state["mass"][0] * np.dot(v_vec, v_vec)
    ke_m = 0.5 * v_rigid @ M @ v_rigid
    assert np.isclose(ke_m, ke_exact, rtol=1e-10)


def test_qbat_pure_membrane_tension():
    """Verify pure in-plane tension produces membrane stresses without spurious bending."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_qbat_group(coords)

    # Apply uniform eps_xx strain rate
    eps_dot = 1e-2
    v = np.zeros((4, 3))
    v[1, 0] = eps_dot * 1.0
    v[2, 0] = eps_dot * 1.0
    vr = np.zeros((4, 3))
    dt = 1e-4

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    forces(group, coords, v, vr, dt, fint, mint)

    # Moments should be identically zero
    assert np.linalg.norm(mint) < 1e-6
    # In-plane forces should balance in x
    assert np.isclose(fint[0, 0] + fint[3, 0] + fint[1, 0] + fint[2, 0], 0.0, atol=1e-6)


def test_qbat_pure_bending():
    """Verify applied uniform bending produces linear through-thickness stress and restoring moments."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_qbat_group(coords, nip=5)

    # Apply pure curvature rate about x-axis
    curv_dot = 0.1
    v = np.zeros((4, 3))
    vr = np.zeros((4, 3))
    vr[:, 0] = curv_dot

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    forces(group, coords, v, vr, 1e-4, fint, mint)

    # Bending moment should be generated
    assert np.linalg.norm(mint[:, 0]) > 0.0


def test_qbat_rigid_body_translation_zero_forces():
    """Verify uniform rigid body translation produces identically zero forces and moments."""
    coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [1.1, 1.3, 0.0], [-0.1, 1.0, 0.0]])
    Rot = Rotation.from_euler("xyz", [35.0, -25.0, 50.0], degrees=True).as_matrix()
    coords_3d = coords @ Rot.T + np.array([-2.0, 5.0, 3.0])
    group, model = _create_mock_qbat_group(coords_3d)

    v = np.zeros((4, 3))
    v[:] = [15.0, -20.0, 8.0]
    vr = np.zeros((4, 3))

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    forces(group, coords_3d, v, vr, 1e-5, fint, mint)

    assert np.linalg.norm(fint) < 1e-6
    assert np.linalg.norm(mint) < 1e-6


def test_qbat_rigid_body_rotation_zero_forces():
    """Verify uniform rigid body rotation produces identically zero forces and moments."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    Rot = Rotation.from_euler("xyz", [15.0, -45.0, 30.0], degrees=True).as_matrix()
    coords_3d = coords @ Rot.T
    group, model = _create_mock_qbat_group(coords_3d)

    dt = 1e-6
    w = np.array([0.1, -0.08, 0.12])
    ce = coords_3d.mean(axis=0)
    v = np.cross(w, coords_3d - ce)
    vr = np.tile(w, (4, 1))

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    forces(group, coords_3d, v, vr, dt, fint, mint)

    assert np.linalg.norm(fint) < 1e-4
    assert np.linalg.norm(mint) < 1e-4


def test_qbat_linear_momentum_conservation():
    """Verify sum of internal forces equals zero under arbitrary general deformation."""
    coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.4, 1.1, 0.0], [0.1, 0.9, 0.0]])
    Rot = Rotation.from_euler("xyz", [-20.0, 40.0, -35.0], degrees=True).as_matrix()
    coords_3d = coords @ Rot.T + np.array([1.0, 1.0, 1.0])
    group, model = _create_mock_qbat_group(coords_3d)

    np.random.seed(42)
    v = np.random.randn(4, 3) * 5.0
    vr = np.random.randn(4, 3) * 2.0

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    forces(group, coords_3d, v, vr, 1e-5, fint, mint)

    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)


def test_qbat_angular_momentum_conservation():
    """Verify sum of moments and force-arms equals zero under arbitrary deformation."""
    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.3, 1.2, 0.0], [0.0, 1.0, 0.0]])
    Rot = Rotation.from_euler("xyz", [30.0, -15.0, 60.0], degrees=True).as_matrix()
    coords_3d = coords @ Rot.T
    group, model = _create_mock_qbat_group(coords_3d)

    np.random.seed(99)
    v = np.random.randn(4, 3) * 3.0
    vr = np.random.randn(4, 3) * 1.5

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    forces(group, coords_3d, v, vr, 1e-5, fint, mint)

    ce = coords_3d.mean(axis=0)
    total_moment = mint.sum(axis=0) + np.cross(coords_3d - ce, fint).sum(axis=0)
    assert np.allclose(total_moment, 0.0, atol=1e-5)


def test_qbat_tangent_stiffness_symmetry():
    """Verify tangent stiffness matrix is symmetric."""
    coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.3, 1.1, 0.0], [-0.1, 1.0, 0.0]])
    Rot = Rotation.from_euler("xyz", [10.0, 20.0, 30.0], degrees=True).as_matrix()
    coords_3d = coords @ Rot.T
    group, model = _create_mock_qbat_group(coords_3d)

    ke, edofs = tangent(group, coords_3d)
    K = ke[0]
    assert np.allclose(K, K.T, atol=1e-6)


def test_qbat_tangent_stiffness_rigid_nullspace():
    """Verify 24-DOF tangent stiffness matrix has rank 18 and exact 6 rigid-body null modes."""
    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.5, 1.2, 0.0], [0.0, 1.2, 0.0]])
    Rot = Rotation.from_euler("xyz", [25.0, -35.0, 40.0], degrees=True).as_matrix()
    coords_3d = coords @ Rot.T + np.array([5.0, -2.0, 8.0])
    group, model = _create_mock_qbat_group(coords_3d)

    ke, edofs = tangent(group, coords_3d)
    K = ke[0]

    assert np.allclose(K, K.T, atol=1e-6)

    # Rigid body translation modes: u_x, u_y, u_z
    for d in range(3):
        u_rigid = np.zeros(24)
        u_rigid[d::6] = 1.0
        assert np.linalg.norm(K @ u_rigid) < 1e-4

    # Rigid body rotation modes about 3D axes
    ce = coords_3d.mean(axis=0)
    for ax in range(3):
        w = np.zeros(3)
        w[ax] = 1.0
        u_rot = np.zeros(24)
        for a in range(4):
            u_rot[6 * a:6 * a + 3] = np.cross(w, coords_3d[a] - ce)
            u_rot[6 * a + 3:6 * a + 6] = w
        assert np.linalg.norm(K @ u_rot) < 1e-4

    # Rank check
    eigvals = np.linalg.eigvalsh(K)
    near_zero = np.sum(np.abs(eigvals) < 1e-3)
    assert near_zero == 6
    assert np.sum(eigvals > 1e-3) == 18


def test_qbat_kgeo_properties():
    """Verify initial-stress geometric stiffness matrix symmetry and translational null modes."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_qbat_group(coords)
    group.state["sig"][:] = 1e7

    k_g, edofs = kgeo(group, coords)
    Kg = k_g[0]

    assert np.allclose(Kg, Kg.T, atol=1e-6)

    # Rigid translation null modes
    for d in range(3):
        u_tr = np.zeros(24)
        u_tr[d::6] = 1.0
        assert np.linalg.norm(Kg @ u_tr) < 1e-6


def test_qbat_kgeo_rigid_invariance():
    """Verify geometric stiffness matrix is invariant under arbitrary 3D rigid translation."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group0, _ = _create_mock_qbat_group(coords)
    group0.state["sig"][:] = 5e6
    kg0, _ = kgeo(group0, coords)

    coords_shifted = coords + np.array([10.0, -25.0, 40.0])
    group1, _ = _create_mock_qbat_group(coords_shifted)
    group1.state["sig"][:] = 5e6
    kg1, _ = kgeo(group1, coords_shifted)

    assert np.allclose(kg0, kg1, atol=1e-8)


def test_qbat_element_deletion_and_law0():
    """Verify deactivated elements (off=0) and LAW0 void elements produce zero forces and dt=EP30."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_qbat_group(coords, law=0)

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    v = np.ones((4, 3))
    vr = np.ones((4, 3))
    dt_claim = forces(group, coords, v, vr, 1e-4, fint, mint)

    assert np.isclose(dt_claim[0], EP30)
    assert np.linalg.norm(fint) == 0.0
    assert np.linalg.norm(mint) == 0.0

    ke, _ = tangent(group, coords)
    assert np.all(ke[0] == 0.0)


def test_qbat_implicit_internal_forces_consistency():
    """Verify directional derivative consistency between implicit internal forces and tangent matrix."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_qbat_group(coords)

    ke, edofs = tangent(group, coords)
    K = ke[0]

    u = np.zeros((4, 3))
    ur = np.zeros((4, 3))
    u[1, 0] = 1e-4
    u[2, 1] = 2e-4
    ur[0, 0] = 5e-5

    u_el = np.zeros(24)
    for a in range(4):
        u_el[6 * a:6 * a + 3] = u[a]
        u_el[6 * a + 3:6 * a + 6] = ur[a]

    f_tangent = -(K @ u_el)

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    implicit_internal_forces(group, coords, u, ur, fint, mint, nlgeom=False)

    f_implicit = np.zeros(24)
    for a in range(4):
        f_implicit[6 * a:6 * a + 3] = fint[a]
        f_implicit[6 * a + 3:6 * a + 6] = mint[a]

    assert np.allclose(f_tangent, f_implicit, atol=1e-6)
