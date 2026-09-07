"""
Unit tests for Milestone M510: 16-Node Thick Shell / Solid Shell Element
(/SHEL16 + /PROP/TSHELL) Hardening, Serendipity 3D Kinematics, Virtual Midside
Node Redistribution, Failure Evaluation (AUD-010), 48-DOF Tangent Stiffness,
48-DOF Geometric Stiffness, Analytical Consistent Mass, and Implicit Internal Forces.

Fortran origin: ``engine/source/elements/thickshell/solide16/``
    s16forc3.F, s16deri3.F, s16rst.F, s16bilan.F, and starter s16mass3.F, s16init3.F.
"""

import numpy as np
import pytest
from types import SimpleNamespace

from pyradioss.common.constants import EP30
from pyradioss.elements import shell_thick16
from pyradioss.elements.shell_thick16 import (
    s16rst,
    s16deri3,
    _geometry,
    _reconstruct_xe,
    _reconstruct_ve,
    _edofs,
    init_group,
    forces,
    tangent,
    kgeo,
    consistent_mass,
    static_internal_forces,
    implicit_internal_forces,
    _IPERM1_16,
    _IPERM2_16,
    _M_SHEL16,
)


def _make_unit_prism_nodes():
    """Reference 16-node prismatic solid shell coordinates mapped to [0, 1]^3.
    Nodes 0..7: corners. Nodes 8..15: midsides."""
    ref_nodes = np.array([
        [-1, -1, -1],  # 0
        [-1, -1,  1],  # 1
        [ 1, -1,  1],  # 2
        [ 1, -1, -1],  # 3
        [-1,  1, -1],  # 4
        [-1,  1,  1],  # 5
        [ 1,  1,  1],  # 6
        [ 1,  1, -1],  # 7
        [-1, -1,  0],  # 8 (between 0 and 1)
        [ 0, -1,  1],  # 9 (between 1 and 2)
        [ 1, -1,  0],  # 10 (between 2 and 3)
        [ 0, -1, -1],  # 11 (between 3 and 0)
        [-1,  1,  0],  # 12 (between 4 and 5)
        [ 0,  1,  1],  # 13 (between 5 and 6)
        [ 1,  1,  0],  # 14 (between 6 and 7)
        [ 0,  1, -1],  # 15 (between 7 and 4)
    ], dtype=float)
    return 0.5 * (ref_nodes + 1.0)


def _make_mock_group(n=1, with_virt=False, E=210000.0, nu=0.3, rho0=7.8e-9):
    """Create a mock element group for testing."""
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    mat = SimpleNamespace(
        law=1,
        rho0=rho0,
        E=E,
        nu=nu,
        K=K,
        G=G,
        params={"E": E, "nu": nu, "rho": rho0, "eps_p_max": EP30},
        fail=None,
    )
    prop = SimpleNamespace(params={"npts_r": 2, "npts_s": 2, "npts_t": 2})

    conn = np.zeros((n, 16), dtype=int)
    for i in range(n):
        conn[i, :8] = np.arange(i * 16, i * 16 + 8)
        if with_virt:
            conn[i, 8:] = -1  # all midside nodes virtual
        else:
            conn[i, 8:] = np.arange(i * 16 + 8, (i + 1) * 16)

    # 2x2x2 Gauss integration points and weights
    g = 1.0 / np.sqrt(3.0)
    pts = []
    wts = []
    for r in [-g, g]:
        for s in [-g, g]:
            for t in [-g, g]:
                pts.append((r, s, t))
                wts.append(1.0)

    state = {
        "slices": [(slice(0, n), mat, prop)],
        "zw": [(pts, wts)],
        "sig": np.zeros((n, 8, 6)),
        "epsp": np.zeros((n, 8)),
        "eint": np.zeros(n),
        "ehour": np.zeros(n),
        "rho": np.full(n, rho0),
        "vol": np.ones(n),
        "mass": np.full(n, rho0 * 1.0),
        "lc": np.ones(n),
        "chk_fail": False,
        "off": np.ones(n),
        "dama": np.zeros((n, 8)),
    }
    group = SimpleNamespace(n=n, conn=conn, state=state)
    return group


# 1. Shape functions Kronecker delta
def test_shel16_shape_functions_kronecker():
    nodes = [
        (-1, -1, -1), (-1, -1,  1), ( 1, -1,  1), ( 1, -1, -1),
        (-1,  1, -1), (-1,  1,  1), ( 1,  1,  1), ( 1,  1, -1),
        (-1, -1,  0), ( 0, -1,  1), ( 1, -1,  0), ( 0, -1, -1),
        (-1,  1,  0), ( 0,  1,  1), ( 1,  1,  0), ( 0,  1, -1),
    ]
    for j, (rj, sj, tj) in enumerate(nodes):
        ni, _, _, _ = s16rst(rj, sj, tj)
        for i in range(16):
            expected = 1.0 if i == j else 0.0
            assert abs(ni[i] - expected) < 1e-12, f"Node {j}: N_{i} = {ni[i]}, expected {expected}"


# 2. Partition of unity and derivative sums
def test_shel16_partition_of_unity_and_derivatives():
    test_points = [
        (0.0, 0.0, 0.0),
        (0.3, -0.4, 0.5),
        (-0.7, 0.2, -0.1),
        (0.577, -0.577, 0.577),
    ]
    for r, s, t in test_points:
        ni, dr, ds, dt_ = s16rst(r, s, t)
        assert abs(sum(ni) - 1.0) < 1e-14
        assert abs(sum(dr)) < 1e-14
        assert abs(sum(ds)) < 1e-14
        assert abs(sum(dt_)) < 1e-14


# 3. Linear coordinate reproduction
def test_shel16_linear_coordinate_reproduction():
    phys_nodes = _make_unit_prism_nodes()
    for r, s, t in [(0.2, -0.3, 0.6), (-0.5, 0.4, -0.1), (0.1, 0.1, 0.1)]:
        ni, _, _, _ = s16rst(r, s, t)
        # Reconstruct coordinates from shape functions
        x_interp = np.sum([ni[i] * phys_nodes[i] for i in range(16)], axis=0)
        # Linear analytical mapping: x = (r+1)/2, y = (s+1)/2, z = (t+1)/2
        x_exact = 0.5 * (np.array([r, s, t]) + 1.0)
        np.testing.assert_allclose(x_interp, x_exact, atol=1e-14)


# 4. Volume of cuboid
def test_shel16_volume_cuboid():
    lx, ly, lz = 2.5, 3.0, 1.5
    phys_nodes = _make_unit_prism_nodes()
    phys_nodes[:, 0] *= lx
    phys_nodes[:, 1] *= ly
    phys_nodes[:, 2] *= lz
    xe = phys_nodes[None, :, :]
    _, vol_gp, vol_tot = _geometry(xe)
    np.testing.assert_allclose(vol_tot[0], lx * ly * lz, rtol=1e-12)


# 5. Volume invariance under 3D rotation and translation
def test_shel16_volume_invariance_rotation_translation():
    phys_nodes = _make_unit_prism_nodes()
    theta = 0.45
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    trans = np.array([10.0, -5.0, 3.0])
    phys_trans = (phys_nodes @ R.T) + trans
    _, _, vol1 = _geometry(phys_nodes[None, :, :])
    _, _, vol2 = _geometry(phys_trans[None, :, :])
    np.testing.assert_allclose(vol1[0], 1.0, rtol=1e-12)
    np.testing.assert_allclose(vol2[0], 1.0, rtol=1e-12)


# 6. Degenerate / inverted element safety
def test_shel16_degenerate_inverted_safety():
    # Collapsed/flat element: all z coordinates zero
    flat_nodes = _make_unit_prism_nodes()
    flat_nodes[:, 2] = 0.0
    xx = flat_nodes.T
    ni, dr, ds, dt_ = s16rst(0.0, 0.0, 0.0)
    px, py, pz, det = s16deri3(xx, dr, ds, dt_)
    assert not np.isnan(det)
    assert not np.isnan(px).any()
    assert det >= 1e-20


# 7. Empty group handling across all APIs
def test_shel16_empty_group_handling():
    group = SimpleNamespace(n=0, conn=np.zeros((0, 16), dtype=int), state={"slices": []})
    model = SimpleNamespace(numnod=0, x0=np.zeros((0, 3)))
    c, m, _ = init_group(group, model, None)
    assert len(c) == 0
    assert len(m) == 0

    x = np.zeros((0, 3))
    v = np.zeros((0, 3))
    dt_crit = forces(group, x, v, None, 1e-6, None, None)
    assert len(dt_crit) == 0

    ke, edofs = tangent(group, x)
    assert ke.shape == (0, 48, 48)
    assert edofs.shape == (0, 48)

    kg, edofs2 = kgeo(group, x)
    assert kg.shape == (0, 48, 48)
    assert edofs2.shape == (0, 48)

    me, edofs3 = consistent_mass(group)
    assert me.shape == (0, 48, 48)
    assert edofs3.shape == (0, 48)

    fint = np.zeros((0, 3))
    static_internal_forces(group, x, None, None, fint, None)
    implicit_internal_forces(group, x, None, None, fint, None)


# 8. Cycle 0 Courant step calculation
def test_shel16_cycle_0_courant_step():
    group = _make_mock_group(n=1)
    nodes = _make_unit_prism_nodes()
    # Call with dt <= 0.0 or v is None
    dt_crit1 = forces(group, nodes, None, None, 0.0, None, None)
    dt_crit2 = forces(group, nodes, np.zeros_like(nodes), None, -1.0, None, None)
    assert len(dt_crit1) == 1
    assert dt_crit1[0] > 0.0
    np.testing.assert_allclose(dt_crit1, dt_crit2)


# 9. Virtual midside node reconstruction
def test_shel16_virtual_midside_node_reconstruction():
    conn = np.zeros((1, 16), dtype=int)
    conn[0, :8] = np.arange(8)
    conn[0, 8:] = -1  # all virtual
    corners = _make_unit_prism_nodes()[:8]
    xe = _reconstruct_xe(conn, corners)
    # Check node 8: between corner 0 and corner 1
    np.testing.assert_allclose(xe[0, 8], 0.5 * (corners[0] + corners[1]))
    # Check node 9: between corner 1 and corner 2
    np.testing.assert_allclose(xe[0, 9], 0.5 * (corners[1] + corners[2]))


# 10. Virtual midside force redistribution
def test_shel16_virtual_midside_force_redistribution():
    group = _make_mock_group(n=1, with_virt=True)
    nodes = _make_unit_prism_nodes()[:8]  # only 8 corner nodes exist!
    group.conn[0, :8] = np.arange(8)
    group.conn[0, 8:] = -1

    # Give it tensile deformation along y
    v = np.zeros((8, 3))
    v[4:8, 1] = 1.0  # top corners moving in +y

    fint = np.zeros((8, 3))
    dt = 1e-6
    forces(group, nodes, v, None, dt, fint, None)

    # Nodal forces should only be on the 8 corners, and sum of forces should be zero
    np.testing.assert_allclose(np.sum(fint, axis=0), 0.0, atol=1e-10)


# 11. Lumped mass conservation matching s16mass3.F
def test_shel16_lumped_mass_conservation():
    group = _make_mock_group(n=1, with_virt=False)
    nodes = _make_unit_prism_nodes()
    model = SimpleNamespace(x0=nodes, numnod=16)
    conn_valid, mass_16, _ = init_group(group, model, None)

    rho = group.state["rho"][0]
    vol = group.state["vol"][0]
    total_mass = rho * vol

    # Total element mass
    np.testing.assert_allclose(np.sum(mass_16), total_mass, rtol=1e-12)
    # Corner mass: m/32 each
    for c in range(8):
        np.testing.assert_allclose(mass_16[c], total_mass / 32.0, rtol=1e-12)
    # Midside mass: 3m/32 each
    for m in range(8, 16):
        np.testing.assert_allclose(mass_16[m], 3.0 * total_mass / 32.0, rtol=1e-12)


# 12. Consistent mass matrix properties and kinetic energy
def test_shel16_consistent_mass_matrix_properties():
    group = _make_mock_group(n=1)
    nodes = _make_unit_prism_nodes()
    group.state["mass"] = np.array([2.5])
    me, edofs = consistent_mass(group)
    assert me.shape == (1, 48, 48)

    M = me[0]
    # Symmetry
    np.testing.assert_allclose(M, M.T, atol=1e-14)

    # Positive definiteness: 48 positive eigenvalues
    eigvals = np.linalg.eigvalsh(M)
    assert (eigvals > 0).all()

    # Exact kinetic energy for rigid body translation
    v_rigid = np.tile([3.0, -2.0, 1.5], 16)
    ke_mat = 0.5 * v_rigid @ M @ v_rigid
    ke_exact = 0.5 * 2.5 * (3.0**2 + (-2.0)**2 + 1.5**2)
    np.testing.assert_allclose(ke_mat, ke_exact, rtol=1e-12)


# 13. Rigid body translation produces zero strain and zero internal forces
def test_shel16_rigid_body_translation_zero_forces():
    group = _make_mock_group(n=1)
    nodes = _make_unit_prism_nodes()
    v_trans = np.tile([12.0, -8.0, 5.0], (16, 1))

    fint = np.zeros((16, 3))
    forces(group, nodes, v_trans, None, 1e-5, fint, None)

    # Stresses and internal forces should remain identically zero
    np.testing.assert_allclose(group.state["sig"], 0.0, atol=1e-12)
    np.testing.assert_allclose(fint, 0.0, atol=1e-12)


# 14. Rigid body rotation produces zero strain and zero internal forces
def test_shel16_rigid_body_rotation_zero_forces():
    group = _make_mock_group(n=1)
    nodes = _make_unit_prism_nodes()
    # Angular velocity omega = (0, 0, 1) -> v = omega x r = (-y, x, 0)
    v_rot = np.zeros((16, 3))
    v_rot[:, 0] = -nodes[:, 1]
    v_rot[:, 1] =  nodes[:, 0]

    fint = np.zeros((16, 3))
    forces(group, nodes, v_rot, None, 1e-6, fint, None)

    # Pure spin has zero symmetric strain rate
    np.testing.assert_allclose(group.state["sig"], 0.0, atol=1e-10)
    np.testing.assert_allclose(fint, 0.0, atol=1e-10)


# 15. Linear momentum conservation
def test_shel16_linear_momentum_conservation():
    group = _make_mock_group(n=1)
    nodes = _make_unit_prism_nodes()
    # Arbitrary non-rigid nodal velocities
    rng = np.random.default_rng(42)
    v = rng.standard_normal((16, 3))

    fint = np.zeros((16, 3))
    forces(group, nodes, v, None, 1e-6, fint, None)

    # Sum of internal forces is zero
    np.testing.assert_allclose(np.sum(fint, axis=0), 0.0, atol=1e-10)


# 16. Angular momentum conservation
def test_shel16_angular_momentum_conservation():
    group = _make_mock_group(n=1)
    nodes = _make_unit_prism_nodes()
    rng = np.random.default_rng(123)
    v = rng.standard_normal((16, 3))

    fint = np.zeros((16, 3))
    forces(group, nodes, v, None, 1e-6, fint, None)

    # Total internal torque sum r x F == 0
    torques = np.cross(nodes, fint)
    np.testing.assert_allclose(np.sum(torques, axis=0), 0.0, atol=1e-10)


# 17. Tangent stiffness nullspace (rank 42, 6 rigid body modes)
def test_shel16_tangent_stiffness_rigid_nullspace():
    group = _make_mock_group(n=1)
    nodes = _make_unit_prism_nodes()
    # Displace corner and midside nodes to test general quadratic serendipity geometry
    nodes[0] += [0.1, 0.05, -0.05]
    nodes[6] += [-0.05, 0.1, 0.05]
    nodes[10] += [0.02, -0.03, 0.01]

    ke, edofs = tangent(group, nodes)
    K = ke[0]
    assert K.shape == (48, 48)

    # 6 rigid body modes
    RBM = np.zeros((6, 48))
    for i in range(16):
        RBM[0, 3*i + 0] = 1.0  # Tx
        RBM[1, 3*i + 1] = 1.0  # Ty
        RBM[2, 3*i + 2] = 1.0  # Tz
        RBM[3, 3*i + 1] = -nodes[i, 2]  # Rx
        RBM[3, 3*i + 2] =  nodes[i, 1]
        RBM[4, 3*i + 0] =  nodes[i, 2]  # Ry
        RBM[4, 3*i + 2] = -nodes[i, 0]
        RBM[5, 3*i + 0] = -nodes[i, 1]  # Rz
        RBM[5, 3*i + 1] =  nodes[i, 0]

    for m in range(6):
        res = np.linalg.norm(K @ RBM[m])
        assert res < 1e-8, f"Mode {m} residual: {res}"

    eigvals = np.linalg.eigvalsh(K)
    num_zeros = np.sum(np.abs(eigvals) < 1e-6)
    assert num_zeros == 6


# 18. Geometric stiffness symmetry and translational nullspace
def test_shel16_geometric_stiffness_symmetry_and_nullspace():
    group = _make_mock_group(n=1)
    nodes = _make_unit_prism_nodes()
    # Assign hydrostatic + shear stress
    group.state["sig"][0, :, :] = [100.0, 50.0, 20.0, 15.0, 10.0, 5.0]

    kg, edofs = kgeo(group, nodes)
    K = kg[0]
    # Symmetry
    np.testing.assert_allclose(K, K.T, atol=1e-12)

    # Translational null modes: sum_a K_ab == 0
    for c in range(3):
        u_trans = np.zeros(48)
        u_trans[c::3] = 1.0
        np.testing.assert_allclose(K @ u_trans, 0.0, atol=1e-10)


# 19. Element deletion (off == 0) and LAW0 void handling
def test_shel16_element_deletion_and_law0():
    # Inactive element
    group = _make_mock_group(n=2)
    group.state["off"][1] = 0.0  # element 1 is dead
    nodes = np.tile(_make_unit_prism_nodes(), (2, 1))

    v = np.ones((32, 3))
    fint = np.zeros((32, 3))
    dt_crit = forces(group, nodes, v, None, 1e-6, fint, None)
    assert dt_crit[1] >= EP30 - 1.0

    ke, _ = tangent(group, nodes)
    np.testing.assert_allclose(ke[1], 0.0)

    # LAW0 void material
    group_law0 = _make_mock_group(n=1)
    group_law0.state["slices"][0][1].law = 0
    ke0, _ = tangent(group_law0, nodes[:16])
    np.testing.assert_allclose(ke0[0], 0.0)


# 20. Implicit internal forces consistency
def test_shel16_implicit_internal_forces_consistency():
    group = _make_mock_group(n=1)
    nodes = _make_unit_prism_nodes()
    ke, _ = tangent(group, nodes)
    K = ke[0]

    u = np.zeros((16, 3))
    u[1, 0] = 0.01
    u[5, 2] = -0.02
    u_flat = u.reshape(-1)

    fint = np.zeros((16, 3))
    implicit_internal_forces(group, nodes, u, None, fint, None, nlgeom=False)

    f_expected = -(K @ u_flat).reshape(-1, 3)
    np.testing.assert_allclose(fint, f_expected, atol=1e-10)
