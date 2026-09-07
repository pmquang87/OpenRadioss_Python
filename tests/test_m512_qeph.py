"""M512 — 4-Node QEPH Shell Element Tests.

Covers:
  - Local frame orthonormality and right-handedness (det=1)
  - Area and geometry calculations vs Euclidean analytical values
  - 3D rigid-body rotation and translation invariance
  - Hourglass orthogonalization (silent under constant strain & curvature)
  - Degenerate / zero area element safety
  - Empty group handling across all element APIs
  - Cycle 0 Courant step calculation without state modification
  - Lumped mass and rotational inertia conservation
  - Consistent mass matrix symmetry, positive definiteness (24 positive eigenvalues),
    and exact kinetic energy
  - Pure membrane tension
  - Pure bending moment and curvature
  - Rigid body translation producing zero forces and moments
  - Rigid body rotation producing zero forces and moments
  - Linear momentum conservation (sum F_i = 0)
  - Angular momentum conservation (sum (r_i x F_i + M_i) = 0)
  - Tangent stiffness rank 18 and exact 6 rigid-body null modes
  - Geometric stiffness symmetry and translational null modes
  - Element deletion (off=0) and LAW0 void handling
  - Directional derivative consistency of implicit internal forces vs tangent matrix
  - End-to-end multi-cycle integration
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pyradioss.common.constants import EM20, EP30
from pyradioss.common.messages import MessageLog
from pyradioss.elements import shell_qeph
from pyradioss.elements.shell_qeph import (
    _edofs, _geometry, consistent_mass, forces, init_group,
    kgeo, static_internal_forces, implicit_internal_forces, tangent
)
from pyradioss.model.model import ElementGroup, Model


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
    def __init__(self, thick=0.01, nip=3, dn=0.015, ishell=24):
        self.params = {
            "thick": thick,
            "nip": nip,
            "dn": dn,
            "ishell": ishell,
        }
        self.thick = thick
        self.nip = nip
        self.dn = dn


def _create_mock_qeph_group(coords, thick=0.01, nip=3, E=2.1e11, nu=0.3, rho0=7850.0, law=1):
    n = len(coords) // 4
    conn = np.arange(len(coords), dtype=np.int64).reshape(n, 4)
    mat = SimpleMat(E=E, nu=nu, rho0=rho0, law=law)
    prop = SimpleProp(thick=thick, nip=nip)

    model = Model()
    model.x0 = coords.copy()

    group = ElementGroup(
        ids=np.arange(1, n + 1, dtype=np.int64),
        conn=conn,
        part=np.zeros(n, dtype=np.int64),
        state={
            "slices": [(slice(0, n), mat, prop)],
            "part_ids": np.ones(n, dtype=np.int64),
        },
    )
    log = MessageLog()
    init_group(group, model, log)
    return group, model


# ============================================================================
# 1. Local frame orthonormality
# ============================================================================
def test_qeph_local_frame_orthonormality():
    """Verify local triads e1, e2, e3 are strictly orthonormal with det(E)=1."""
    xe = np.array([
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 2.0, 3.0], [4.0, 1.0, 0.0], [5.0, 3.0, 2.0], [2.0, 4.0, 5.0]],
        [[-1.0, 0.5, 2.0], [0.0, 1.5, -1.0], [1.2, 0.8, 3.0], [-0.5, -0.2, 1.5]]
    ])
    G = _geometry(xe)
    E = G["E"]  # (n, 3, 3) where columns are e1, e2, e3
    for i in range(len(xe)):
        Ei = E[i]
        e1, e2, e3 = Ei[:, 0], Ei[:, 1], Ei[:, 2]
        assert np.isclose(np.linalg.norm(e1), 1.0, atol=1e-12)
        assert np.isclose(np.linalg.norm(e2), 1.0, atol=1e-12)
        assert np.isclose(np.linalg.norm(e3), 1.0, atol=1e-12)
        assert np.isclose(np.dot(e1, e2), 0.0, atol=1e-12)
        assert np.isclose(np.dot(e2, e3), 0.0, atol=1e-12)
        assert np.isclose(np.dot(e1, e3), 0.0, atol=1e-12)
        assert np.isclose(np.linalg.det(Ei), 1.0, atol=1e-12)


# ============================================================================
# 2. Area and geometry calculations
# ============================================================================
def test_qeph_area_and_geometry():
    """Verify calculated area matches analytical Euclidean area."""
    xe = np.array([
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 3.0, 0.0], [0.0, 3.0, 0.0]]
    ])
    G = _geometry(xe)
    assert np.isclose(G["area"][0], 6.0, rtol=1e-12)
    assert np.isclose(G["z1"][0], 0.0, atol=1e-12)

    xe_par = np.array([
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.5, 1.5, 0.0], [0.5, 1.5, 0.0]]
    ])
    G_par = _geometry(xe_par)
    assert np.isclose(G_par["area"][0], 3.0, rtol=1e-12)


# ============================================================================
# 3. 3D rigid-body rotation and translation invariance
# ============================================================================
def test_qeph_3d_rigid_invariance():
    """Verify area and characteristic length are invariant under 3D rigid rotations and translations."""
    quad = np.array([[0.0, 0.0, 0.0], [1.5, 0.2, 0.0], [1.3, 1.4, 0.0], [-0.1, 1.1, 0.0]])
    R = Rotation.from_euler("xyz", [35.0, -45.0, 60.0], degrees=True).as_matrix()
    trans = np.array([12.5, -4.2, 7.8])
    quad_rot = (quad @ R.T) + trans

    G0 = _geometry(quad[None, :, :])
    G1 = _geometry(quad_rot[None, :, :])

    assert np.isclose(G0["area"][0], G1["area"][0], rtol=1e-12)
    assert np.isclose(G0["ll"][0], G1["ll"][0], rtol=1e-12)


# ============================================================================
# 4. Hourglass orthogonalization (silent under constant strain)
# ============================================================================
def test_qeph_hourglass_orthogonalization():
    """Verify Flanagan-Belytschko hourglass rates are silent under constant strain and curvature."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_qeph_group(coords)

    v = np.zeros((4, 3))
    v[:, 0] = 0.01 * coords[:, 0]
    v[:, 1] = 0.02 * coords[:, 1]
    vr = np.zeros((4, 3))

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    forces(group, coords, v, vr, 1e-4, fint, mint)

    assert np.allclose(group.state["hgstr"], 0.0, atol=1e-12)


# ============================================================================
# 5. Degenerate / zero area element safety
# ============================================================================
def test_qeph_degenerate_geometry_safety():
    """Verify collapsed / collinear elements do not produce NaNs or crashes."""
    coords_deg = np.array([
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],  # point collapsed
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],  # line collinear
    ])
    group, model = _create_mock_qeph_group(coords_deg)

    assert not np.any(np.isnan(group.state["area0"]))
    assert not np.any(np.isnan(group.state["mass"]))

    ke, edofs = tangent(group, coords_deg)
    assert not np.any(np.isnan(ke))

    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))
    v = np.zeros((8, 3))
    dt_crit = forces(group, coords_deg, v, None, 1e-5, fint, mint)
    assert not np.any(np.isnan(dt_crit))
    assert not np.any(np.isnan(fint))
    assert not np.any(np.isnan(mint))


# ============================================================================
# 6. Empty group handling across all APIs
# ============================================================================
def test_qeph_empty_group_handling():
    """Verify empty group (n=0) gracefully executes without error across all public APIs."""
    coords = np.empty((0, 3), dtype=float)
    group, model = _create_mock_qeph_group(coords)

    edofs = _edofs(group.conn)
    assert edofs.shape == (0, 24)

    Me, ed = consistent_mass(group, coords)
    assert Me.shape == (0, 24, 24)

    ke, ed = tangent(group, coords)
    assert ke.shape == (0, 24, 24)

    kg, ed = kgeo(group, coords)
    assert kg.shape == (0, 24, 24)

    fint = np.empty((0, 3))
    mint = np.empty((0, 3))
    dt = forces(group, coords, None, None, 0.0, fint, mint)
    assert dt.shape == (0,)

    static_internal_forces(group, coords, np.empty((0, 3)), None, fint, mint)
    implicit_internal_forces(group, coords, np.empty((0, 3)), None, fint, mint, nlgeom=False)
    implicit_internal_forces(group, coords, np.empty((0, 3)), None, fint, mint, nlgeom=True)


# ============================================================================
# 7. Cycle 0 Courant time step calculation
# ============================================================================
def test_qeph_cycle_0_courant_step():
    """Verify cycle 0 (dt<=0 or v is None) computes Courant time step without touching forces."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_qeph_group(coords)

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    dt_c = forces(group, coords, None, None, 0.0, fint, mint)

    assert np.allclose(fint, 0.0)
    assert np.allclose(mint, 0.0)

    c_s = group.state["slices"][0][1].sound_speed_shell()
    assert dt_c[0] > 0.0
    assert dt_c[0] < 1.0 / c_s * 2.0


# ============================================================================
# 8. Lumped mass and rotational inertia conservation
# ============================================================================
def test_qeph_lumped_mass_and_inertia():
    """Verify lumped mass (m/4 per node) and rotational inertia."""
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    thick = 0.02
    rho = 7850.0
    area = 2.0
    expected_mass = rho * thick * area

    group, model = _create_mock_qeph_group(coords, thick=thick, rho0=rho)
    assert np.isclose(group.state["mass"][0], expected_mass, rtol=1e-12)

    m_node = expected_mass / 4.0
    assert np.isclose(group.state["dt_iner"][0], m_node * (thick**2 + area) / 12.0, rtol=1e-12)


# ============================================================================
# 9. Consistent mass matrix symmetry, positive definiteness & kinetic energy
# ============================================================================
def test_qeph_consistent_mass_properties():
    """Verify consistent mass matrix is symmetric, strictly positive definite (24 positive eigenvalues),
    and exact kinetic energy."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_qeph_group(coords, thick=0.01, rho0=7850.0)

    Me, edofs = consistent_mass(group, coords)
    M = Me[0]

    assert np.allclose(M, M.T, atol=1e-12)

    evals = np.linalg.eigvalsh(M)
    assert np.all(evals > 0.0)
    assert len(evals) == 24

    V = np.array([2.5, -1.8, 3.2])
    v_full = np.zeros(24)
    for i in range(4):
        v_full[6 * i:6 * i + 3] = V

    ke_calc = 0.5 * v_full @ M @ v_full
    ke_exact = 0.5 * group.state["mass"][0] * np.sum(V**2)
    assert np.isclose(ke_calc, ke_exact, rtol=1e-12)


# ============================================================================
# 10. Pure membrane tension
# ============================================================================
def test_qeph_pure_membrane_tension():
    """Verify uniform in-plane stretching produces correct membrane resultant forces."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_qeph_group(coords, thick=0.01, E=2.1e11, nu=0.0)

    eps = 1e-4
    v = np.zeros((4, 3))
    v[:, 0] = eps * coords[:, 0] / 1e-3  # dt = 1e-3
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))

    forces(group, coords, v, None, 1e-3, fint, mint)

    # In-plane stress sigma_xx = E * eps = 2.1e7 Pa
    # Force on nodes 1 and 2 (right edge, x=1): Fx = 0.5 * sigma_xx * thick * length = 105,000 N
    # fint accumulates NEGATED internal force: fint = -F
    assert np.isclose(fint[1, 0], -105000.0, rtol=1e-4)
    assert np.isclose(fint[2, 0], -105000.0, rtol=1e-4)
    assert np.isclose(fint[0, 0], 105000.0, rtol=1e-4)
    assert np.isclose(fint[3, 0], 105000.0, rtol=1e-4)


# ============================================================================
# 11. Pure bending moment and curvature
# ============================================================================
def test_qeph_pure_bending_moment():
    """Verify pure bending rotation produces expected internal moments and torque balance."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    thick = 0.02
    E = 2.1e11
    group, model = _create_mock_qeph_group(coords, thick=thick, E=E, nu=0.0)

    vr = np.zeros((4, 3))
    vr[:, 1] = 0.01 * coords[:, 0] / 1e-3  # thy
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))

    forces(group, coords, np.zeros((4, 3)), vr, 1e-3, fint, mint)

    assert np.abs(mint[1, 1]) > 0.0
    net_torque = np.sum(np.cross(coords, fint) + mint, axis=0)
    assert np.allclose(net_torque, 0.0, atol=1e-4)
    assert np.allclose(np.sum(fint, axis=0), 0.0, atol=1e-4)


# ============================================================================
# 12. Rigid body translation producing zero forces
# ============================================================================
def test_qeph_rigid_translation_zero_forces():
    """Verify rigid body translation yields zero internal forces and moments."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_qeph_group(coords)

    V = np.array([12.0, -8.5, 4.2])
    v = np.repeat(V[None, :], 4, axis=0)
    vr = np.zeros((4, 3))

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    forces(group, coords, v, vr, 1e-3, fint, mint)

    assert np.allclose(fint, 0.0, atol=1e-10)
    assert np.allclose(mint, 0.0, atol=1e-10)


# ============================================================================
# 13. Rigid body rotation producing zero forces
# ============================================================================
def test_qeph_rigid_rotation_zero_forces():
    """Verify rigid body spin yields zero internal forces and moments."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_qeph_group(coords)

    ce = coords.mean(axis=0)
    w = np.array([0.0, 0.0, 0.01])
    v = np.cross(w, coords - ce)
    vr = np.tile(w, (4, 1))

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    forces(group, coords, v, vr, 1e-5, fint, mint)

    assert np.allclose(fint, 0.0, atol=1e-4)
    assert np.allclose(mint, 0.0, atol=1e-4)


# ============================================================================
# 14. Linear momentum conservation
# ============================================================================
def test_qeph_linear_momentum_conservation():
    """Verify sum of internal forces over all 4 nodes is exactly zero."""
    coords = np.array([[0.1, 0.2, 0.0], [1.1, 0.0, 0.0], [1.2, 1.3, 0.0], [0.0, 0.9, 0.0]])
    group, model = _create_mock_qeph_group(coords)

    rng = np.random.default_rng(42)
    v = rng.uniform(-5.0, 5.0, size=(4, 3))
    vr = rng.uniform(-2.0, 2.0, size=(4, 3))

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    forces(group, coords, v, vr, 1e-4, fint, mint)

    assert np.allclose(np.sum(fint, axis=0), 0.0, atol=1e-10)


# ============================================================================
# 15. Angular momentum conservation
# ============================================================================
def test_qeph_angular_momentum_conservation():
    """Verify total torque sum (r_i x F_i + M_i) = 0 to high precision."""
    coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [1.1, 1.3, 0.0], [-0.1, 1.0, 0.0]])
    group, model = _create_mock_qeph_group(coords)

    rng = np.random.default_rng(123)
    v = rng.uniform(-3.0, 3.0, size=(4, 3))
    vr = rng.uniform(-1.5, 1.5, size=(4, 3))

    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    forces(group, coords, v, vr, 1e-4, fint, mint)

    torque = np.zeros(3)
    for i in range(4):
        torque += np.cross(coords[i], fint[i]) + mint[i]

    assert np.allclose(torque, 0.0, atol=1e-8)


# ============================================================================
# 16. Tangent stiffness rank 18 & exact 6 rigid-body null modes
# ============================================================================
def test_qeph_tangent_stiffness_rigid_nullspace():
    """Verify 24-DOF tangent stiffness matrix has rank 18 and exact 6 rigid-body null modes."""
    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.5, 1.2, 0.0], [0.0, 1.2, 0.0]])
    Rot = Rotation.from_euler("xyz", [25.0, -35.0, 40.0], degrees=True).as_matrix()
    coords_3d = coords @ Rot.T + np.array([5.0, -2.0, 8.0])
    group, model = _create_mock_qeph_group(coords_3d)

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

    evals = np.linalg.eigvalsh(K)
    num_zeros = np.sum(np.abs(evals) < 1e-3)
    assert num_zeros == 6
    assert np.sum(evals >= 1e-3) == 18


# ============================================================================
# 17. Geometric stiffness symmetry and translational null modes
# ============================================================================
def test_qeph_geometric_stiffness_properties():
    """Verify initial-stress geometric stiffness is symmetric and possesses translational null modes."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_qeph_group(coords)

    group.state["sig"][:] = np.array([5.0e7, 3.0e7, 1.0e6])

    kg, edofs = kgeo(group, coords)
    Kg = kg[0]

    assert np.allclose(Kg, Kg.T, atol=1e-12)

    for d in range(3):
        u_trans = np.zeros(24)
        u_trans[d::6] = 1.0
        assert np.linalg.norm(Kg @ u_trans) < 1e-8


# ============================================================================
# 18. Element deletion (off=0) and LAW0 void handling
# ============================================================================
def test_qeph_element_deletion_and_void():
    """Verify element deletion and LAW0 void element zero out stiffness and forces."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],  # normal element
        [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [2.0, 1.0, 0.0],  # void element
    ])
    mat_norm = SimpleMat()
    mat_void = SimpleMat(E=0.0, nu=0.0, rho0=0.0, law=0)
    prop = SimpleProp()

    conn = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
    ids = np.array([1, 2], dtype=np.int64)
    model = Model()
    model.x0 = coords

    group = ElementGroup(
        ids=ids,
        conn=conn,
        part=np.zeros(2, dtype=np.int64),
        state={
            "slices": [
                (slice(0, 1), mat_norm, prop),
                (slice(1, 2), mat_void, prop),
            ],
            "part_ids": np.ones(2, dtype=np.int64),
        },
    )
    log = MessageLog()
    init_group(group, model, log)

    # Deactivate first element
    group.state["off"][0] = 0.0

    # 1. Tangent stiffness: both elements must be 0
    ke, edofs = tangent(group, coords)
    assert np.allclose(ke[0], 0.0)
    assert np.allclose(ke[1], 0.0)

    # 2. Forces and dt: element 0 dead, element 1 void -> both return EP30
    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))
    v = np.ones((8, 3))
    dte = forces(group, coords, v, None, 1e-4, fint, mint)
    assert np.isclose(dte[0], EP30)
    assert np.isclose(dte[1], EP30)
    assert np.allclose(fint, 0.0)
    assert np.allclose(mint, 0.0)


# ============================================================================
# 19. Directional derivative consistency of implicit internal forces vs tangent matrix
# ============================================================================
def test_qeph_directional_derivative_consistency():
    """Verify directional derivative of implicit internal forces matches tangent stiffness matrix."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_qeph_group(coords, thick=0.01, E=2.1e11, nu=0.3)

    ke, edofs = tangent(group, coords)
    K = ke[0]

    rng = np.random.default_rng(999)
    du_full = rng.uniform(-1e-4, 1e-4, size=24)

    df_tangent = K @ du_full

    u_disp = du_full.reshape(4, 6)[:, 0:3]
    ur_disp = du_full.reshape(4, 6)[:, 3:6]
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    implicit_internal_forces(group, coords, u_disp, ur_disp, fint, mint, nlgeom=False)

    df_implicit = np.zeros(24)
    for i in range(4):
        df_implicit[6 * i:6 * i + 3] = -fint[i]
        df_implicit[6 * i + 3:6 * i + 6] = -mint[i]

    assert np.allclose(df_tangent, df_implicit, rtol=1e-10)


# ============================================================================
# 20. End-to-end integration and consistency
# ============================================================================
def test_qeph_end_to_end_integration():
    """Verify multi-cycle dynamic simulation advances cleanly with energy accounting."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    group, model = _create_mock_qeph_group(coords, thick=0.01, E=2.1e11, nu=0.3, rho0=7850.0)

    x = coords.copy()
    v = np.zeros((4, 3))
    v[1:3, 0] = 5.0
    vr = np.zeros((4, 3))

    dt = 1e-6
    for cycle in range(5):
        fint = np.zeros((4, 3))
        mint = np.zeros((4, 3))
        dt_c = forces(group, x, v, vr, dt, fint, mint)
        assert dt_c[0] > dt
        x += v * dt

    assert group.state["eint"][0] > 0.0
