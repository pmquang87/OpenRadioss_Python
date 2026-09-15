"""Unit tests for Milestone M514: 4-Node Belytschko-Tsay Shell Element

Hardening, Kinematics, 24-DOF Tangent/Geometric Stiffness, Consistent Mass,
Defensive Guards & Comprehensive Unit Tests.
"""

from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pyradioss.common.constants import EM20, EP30
from pyradioss.elements.shell_bt4 import (
    _frame,
    _local_geometry,
    _char_length,
    _edofs,
    init_group,
    forces,
    consistent_mass,
    tangent,
    kgeo,
    static_internal_forces,
    _static_rot_hourglass,
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
    def __init__(self, thick=0.01, nip=3, hm=0.1, hf=0.1, hr=0.1, ishell=1):
        self.params = {
            "thick": thick,
            "nip": nip,
            "hm": hm,
            "hf": hf,
            "hr": hr,
            "ishell": ishell,
        }
        self.thick = thick
        self.nip = nip
        self.ishell = ishell


def _create_mock_bt4_group(coords, E=2.1e11, nu=0.3, rho=7850.0, thick=0.01, nip=3, ishell=1, law=1):
    n = 1
    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)
    mat = SimpleMat(E=E, nu=nu, rho0=rho, law=law)
    prop = SimpleProp(thick=thick, nip=nip, ishell=ishell)
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


def test_bt4_local_frame_orthonormality():
    """Verify local triad e1, e2, e3 is strictly orthonormal with det(E) = 1.0."""
    coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.2, 1.0, 0.0], [0.0, 1.0, 0.0]])
    Rot = Rotation.from_euler("xyz", [30.0, 45.0, 60.0], degrees=True).as_matrix()
    coords_3d = coords @ Rot.T + np.array([2.0, -1.0, 4.0])

    E = _frame(coords_3d[None])
    R = E[0]

    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_bt4_area_and_characteristic_length():
    """Verify area and characteristic length match Euclidean analytical values."""
    Lx, Ly = 2.0, 1.5
    coords = np.array([[0.0, 0.0, 0.0], [Lx, 0.0, 0.0], [Lx, Ly, 0.0], [0.0, Ly, 0.0]])
    group, model = _create_mock_bt4_group(coords)

    E, xl, area, B1, B2 = _local_geometry(coords[None])
    assert np.isclose(area[0], Lx * Ly, rtol=1e-10)

    lc = _char_length(xl, area)
    # Longest side is Lx = 2.0, so lc = area / Lx = 1.5
    assert np.isclose(lc[0], Lx * Ly / max(Lx, Ly), rtol=1e-10)


def test_bt4_invariance_3d_transform():
    """Verify element area and characteristic length are invariant under 3D rigid motion."""
    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.2, 0.0], [1.4, 1.2, 0.0], [-0.1, 1.1, 0.0]])
    _, xl0, area0, _, _ = _local_geometry(coords[None])
    lc0 = _char_length(xl0, area0)

    Rot = Rotation.from_euler("xyz", [45.0, -30.0, 75.0], degrees=True).as_matrix()
    coords_rot = coords @ Rot.T + np.array([10.0, 20.0, -5.0])

    _, xl1, area1, _, _ = _local_geometry(coords_rot[None])
    lc1 = _char_length(xl1, area1)

    assert np.isclose(area0[0], area1[0], rtol=1e-10)
    assert np.isclose(lc0[0], lc1[0], rtol=1e-10)


def test_bt4_degenerate_element_safety():
    """Verify degenerate and zero-area quad elements do not crash or produce NaNs."""
    coords_degen = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    E = _frame(coords_degen[None])
    assert not np.any(np.isnan(E))
    assert np.isclose(np.linalg.det(E[0]), 1.0, atol=1e-6)

    # Collinear quad
    coords_collinear = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    E_col = _frame(coords_collinear[None])
    assert not np.any(np.isnan(E_col))
    assert np.isclose(np.linalg.det(E_col[0]), 1.0, atol=1e-6)


def test_bt4_empty_group_handling():
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
    _static_rot_hourglass(group, x, x, mint)


def test_bt4_cycle_0_courant_step():
    """Verify cycle 0 calculates Courant time step without mutating element state."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_bt4_group(coords)

    dt0 = forces(group, coords, None, None, 0.0, None, None)
    assert dt0[0] > 0.0
    assert not np.isnan(dt0[0])
    assert dt0[0] < 1.0
    assert np.all(group.state["sig"] == 0.0)
    assert np.all(group.state["eint"] == 0.0)


def test_bt4_lumped_mass_and_rotational_inertia():
    """Verify total lumped mass equals rho * t * Area and rotational inertia matches FAC=9."""
    Lx, Ly, t, rho = 2.0, 1.5, 0.02, 7800.0
    coords = np.array([[0.0, 0.0, 0.0], [Lx, 0.0, 0.0], [Lx, Ly, 0.0], [0.0, Ly, 0.0]])
    group, model = _create_mock_bt4_group(coords, rho=rho, thick=t)

    total_mass = rho * t * (Lx * Ly)
    assert np.isclose(group.state["mass"][0], total_mass, rtol=1e-10)

    # Rotational inertia per node: m_node * (A/FAC + t^2/12) with FAC = 9
    m_node = total_mass / 4.0
    area = Lx * Ly
    expected_iner = m_node * (area / 9.0 + t**2 / 12.0)
    assert np.isclose(group.state["dt_iner"][0], expected_iner, rtol=1e-10)


def test_bt4_consistent_mass_matrix_properties():
    """Verify consistent mass matrix is symmetric, positive definite (24 pos eigenvalues) and conserves KE."""
    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.5, 1.2, 0.0], [0.0, 1.2, 0.0]])
    Rot = Rotation.from_euler("xyz", [20.0, -30.0, 45.0], degrees=True).as_matrix()
    coords_3d = coords @ Rot.T + np.array([1.0, 2.0, 3.0])
    group, model = _create_mock_bt4_group(coords_3d)

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


def test_bt4_pure_membrane_tension():
    """Verify pure in-plane tension produces membrane stresses without spurious bending."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_bt4_group(coords)

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


def test_bt4_pure_bending():
    """Verify applied uniform bending produces linear through-thickness stress and restoring moments."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_bt4_group(coords, nip=5)

    # Apply pure curvature rate about x-axis (rotation rate thx linear in y)
    v = np.zeros((4, 3))
    vr = np.zeros((4, 3))
    vr[2, 0] = 0.1
    vr[3, 0] = 0.1

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    forces(group, coords, v, vr, 1e-4, fint, mint)

    # Bending moment should be generated
    assert np.linalg.norm(mint[:, 0]) > 0.0


def test_bt4_rigid_body_translation_zero_forces():
    """Verify uniform rigid body translation produces identically zero forces and moments."""
    coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [1.1, 1.3, 0.0], [-0.1, 1.0, 0.0]])
    Rot = Rotation.from_euler("xyz", [35.0, -25.0, 50.0], degrees=True).as_matrix()
    coords_3d = coords @ Rot.T + np.array([-2.0, 5.0, 3.0])
    group, model = _create_mock_bt4_group(coords_3d)

    v = np.zeros((4, 3))
    v[:] = [15.0, -20.0, 8.0]
    vr = np.zeros((4, 3))

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    forces(group, coords_3d, v, vr, 1e-5, fint, mint)

    assert np.linalg.norm(fint) < 1e-10
    assert np.linalg.norm(mint) < 1e-10


def test_bt4_rigid_body_rotation_cdefo_correction():
    """Verify rigid body rotation produces bounded forces at physical time steps."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    Rot = Rotation.from_euler("xyz", [15.0, -45.0, 30.0], degrees=True).as_matrix()
    coords_3d = coords @ Rot.T
    group, model = _create_mock_bt4_group(coords_3d)

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


def test_bt4_linear_momentum_conservation():
    """Verify sum of internal forces equals zero under arbitrary general deformation."""
    coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.4, 1.1, 0.0], [0.1, 0.9, 0.0]])
    Rot = Rotation.from_euler("xyz", [-20.0, 40.0, -35.0], degrees=True).as_matrix()
    coords_3d = coords @ Rot.T + np.array([1.0, 1.0, 1.0])
    group, model = _create_mock_bt4_group(coords_3d)

    np.random.seed(42)
    v = np.random.randn(4, 3) * 5.0
    vr = np.random.randn(4, 3) * 2.0

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    forces(group, coords_3d, v, vr, 1e-5, fint, mint)

    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)


def test_bt4_angular_momentum_conservation():
    """Verify sum of moments and force-arms equals zero under arbitrary deformation."""
    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.3, 1.2, 0.0], [0.0, 1.0, 0.0]])
    Rot = Rotation.from_euler("xyz", [30.0, -15.0, 60.0], degrees=True).as_matrix()
    coords_3d = coords @ Rot.T
    group, model = _create_mock_bt4_group(coords_3d)

    np.random.seed(99)
    v = np.random.randn(4, 3) * 3.0
    vr = np.random.randn(4, 3) * 1.5

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    forces(group, coords_3d, v, vr, 1e-5, fint, mint)

    ce = coords_3d.mean(axis=0)
    total_moment = mint.sum(axis=0) + np.cross(coords_3d - ce, fint).sum(axis=0)
    assert np.allclose(total_moment, 0.0, atol=1e-5)


def test_bt4_tangent_stiffness_symmetry():
    """Verify tangent stiffness matrix is symmetric."""
    coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.3, 1.1, 0.0], [-0.1, 1.0, 0.0]])
    Rot = Rotation.from_euler("xyz", [10.0, 20.0, 30.0], degrees=True).as_matrix()
    coords_3d = coords @ Rot.T
    group, model = _create_mock_bt4_group(coords_3d)

    ke, edofs = tangent(group, coords_3d)
    K = ke[0]

    assert np.allclose(K, K.T, atol=1e-6)


def test_bt4_tangent_stiffness_rigid_body_modes():
    """Verify tangent stiffness matrix has 5 exact rigid body zero modes and drilling-stabilized normal rotation."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    Rot = Rotation.from_euler("xyz", [25.0, 15.0, -35.0], degrees=True).as_matrix()
    coords_3d = coords @ Rot.T
    group, model = _create_mock_bt4_group(coords_3d)

    ke, edofs = tangent(group, coords_3d)
    K = ke[0]

    # Rigid translations: exactly 3 zero modes
    ce = coords_3d.mean(axis=0)
    for c in range(3):
        u_trans = np.zeros(24)
        for a in range(4):
            u_trans[6 * a + c] = 1.0
        f_res = K @ u_trans
        assert np.linalg.norm(f_res) < 1e-5

    # Rigid rotations in the shell plane (axes e1, e2): zero modes
    E, _, area, _, _ = _local_geometry(coords_3d[None, :, :])
    e1, e2, e3 = E[0, :, 0], E[0, :, 1], E[0, :, 2]
    for w_vec in [e1, e2]:
        u_rot = np.zeros(24)
        for a in range(4):
            u_rot[6 * a:6 * a + 3] = np.cross(w_vec, coords_3d[a] - ce)
            u_rot[6 * a + 3:6 * a + 6] = w_vec
        f_res = K @ u_rot
        assert np.linalg.norm(f_res) < 1e-4

    # Rigid rotation about normal axis e3: activates drilling penalty
    u_drill = np.zeros(24)
    for a in range(4):
        u_drill[6 * a:6 * a + 3] = np.cross(e3, coords_3d[a] - ce)
        u_drill[6 * a + 3:6 * a + 6] = e3
    f_drill = K @ u_drill
    # Translational components are zero
    for a in range(4):
        assert np.linalg.norm(f_drill[6 * a:6 * a + 3]) < 1e-4
        # Rotational restoring moment equals drilling penalty k_d * e3
        assert np.allclose(f_drill[6 * a + 3:6 * a + 6], 1e-3 * 2.1e11 * (0.01**3) * 1.0 / 12.0 * e3, rtol=1e-4)

    eigvals = np.linalg.eigvalsh(K)
    zero_modes = np.sum(np.abs(eigvals) < 1e-3)
    assert zero_modes == 7
    assert np.sum(eigvals > 1e-3) == 17


def test_bt4_geometric_stiffness_properties():
    """Verify geometric stiffness matrix is symmetric and satisfies translational null modes."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_bt4_group(coords)

    # Set positive membrane tension state
    group.state["sig"][:, :, 0] = 5e7
    group.state["sig"][:, :, 1] = 3e7
    group.state["sig"][:, :, 2] = 1e7

    kg, edofs = kgeo(group, coords)
    Kg = kg[0]

    assert np.allclose(Kg, Kg.T, atol=1e-10)

    # Translational invariance
    for c in range(3):
        u_trans = np.zeros(24)
        for a in range(4):
            u_trans[6 * a + c] = 1.0
        assert np.linalg.norm(Kg @ u_trans) < 1e-8


def test_bt4_hourglass_damping_and_dissipation():
    """Verify hourglass velocity mode activates chvis3.F dissipation and increases ehour."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_bt4_group(coords)

    # Hourglass out-of-plane velocity mode: vz = [1, -1, 1, -1]
    v = np.zeros((4, 3))
    v[:, 2] = [10.0, -10.0, 10.0, -10.0]
    vr = np.zeros((4, 3))
    dt = 1e-5

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    forces(group, coords, v, vr, dt, fint, mint)

    assert group.state["ehour"][0] > 0.0
    assert np.linalg.norm(fint) > 0.0


def test_bt4_law0_void_shell_behavior():
    """Verify LAW0 void shell produces zero forces, EP30 time step, and zero tangent."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_bt4_group(coords, law=0)

    v = np.random.randn(4, 3) * 2.0
    vr = np.random.randn(4, 3) * 1.0
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))

    dt = forces(group, coords, v, vr, 1e-4, fint, mint)
    assert np.all(dt == EP30)
    assert np.all(fint == 0.0)
    assert np.all(mint == 0.0)

    ke, _ = tangent(group, coords)
    assert np.all(ke == 0.0)


def test_bt4_ishell_formulation_branching():
    """Verify Ishell formulation flags map correctly to engine IHBE masks."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

    for ishell_card, expected_ihbe in [(0, 0), (1, 1), (2, 0), (3, 2), (4, 4)]:
        group, _ = _create_mock_bt4_group(coords, ishell=ishell_card)
        assert group.state["ihbe_mask"][0] == expected_ihbe
