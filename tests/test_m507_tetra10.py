"""Unit tests for M507: 10-Node Quadratic 3D Tetrahedral Solid Element
(/TETRA10 + /PROP/SOLID, pyradioss/elements/solid_tetra10.py).
"""

import numpy as np
import pytest

from pyradioss.elements import solid_tetra10


class _DummyModel:
    def __init__(self, x0):
        self.x0 = np.ascontiguousarray(x0, dtype=float)
        self.x = self.x0.copy()
        self.v = np.zeros_like(self.x0)
        self.vr = np.zeros_like(self.x0)
        self.numnod = len(x0)


class _DummyGroup:
    def __init__(self, conn, ids=None):
        self.conn = np.ascontiguousarray(conn, dtype=np.int64)
        self.n = len(self.conn)
        self.ids = np.arange(1, self.n + 1) if ids is None else np.ascontiguousarray(ids, dtype=np.int64)
        self.state = {}


class _DummyLog:
    def __init__(self):
        self.messages = []

    def error(self, msg, context=""):
        self.messages.append(("ERROR", context, msg))

    def warning(self, msg, context=""):
        self.messages.append(("WARN", context, msg))


class _DummyProp:
    def __init__(self, qa=1.1, qb=0.05):
        self.params = {"qa": qa, "qb": qb}
        self.qa = qa
        self.qb = qb


class _DummyMat:
    def __init__(self, law=1, rho0=7.8e-6, E=210.0, nu=0.3, params=None):
        self.law = law
        self.rho0 = rho0
        self.E = E
        self.nu = nu
        self.G = E / (2.0 * (1.0 + nu)) if (1.0 + nu) != 0 else 0.0
        self.K = E / (3.0 * (1.0 - 2.0 * nu)) if (1.0 - 2.0 * nu) != 0 else 0.0
        self.params = params or {}
        self.fail = None
        self.eos = None


def _create_unit_tetra10():
    """Returns a model and group for a standard right-angled unit 10-node tetrahedron
    consistent with _DN_DXI coordinate convention:
    Node 0: (1, 0, 0)
    Node 1: (0, 1, 0)
    Node 2: (0, 0, 1)
    Node 3: (0, 0, 0)
    Node 4: (0.5, 0.5, 0)   - mid (0, 1)
    Node 5: (0, 0.5, 0.5)   - mid (1, 2)
    Node 6: (0.5, 0, 0.5)   - mid (2, 0)
    Node 7: (0.5, 0, 0)     - mid (0, 3)
    Node 8: (0, 0.5, 0)     - mid (1, 3)
    Node 9: (0, 0, 0.5)     - mid (2, 3)
    Total volume = 1/6.
    """
    x0 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)
    model = _DummyModel(x0)
    group = _DummyGroup(conn)
    return model, group


# ----------------------------------------------------------------------------
# 1. Mass lumping and volume
# ----------------------------------------------------------------------------
def test_tetra10_mass_lumping_and_volume():
    model, group = _create_unit_tetra10()
    mat = _DummyMat(law=1, rho0=7.8e-6, E=210.0, nu=0.3)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    log = _DummyLog()

    node_idx, mass_c, _ = solid_tetra10.init_group(group, model, log)
    expected_vol = 1.0 / 6.0
    assert group.state["vol0"][0] == pytest.approx(expected_vol, rel=1e-6)

    total_mass = mat.rho0 * expected_vol
    assert group.state["mass"][0] == pytest.approx(total_mass, rel=1e-6)
    assert len(mass_c) == 10
    # Fully real 10-node element distributes mass equally to all 10 nodes (m/10)
    for m in mass_c:
        assert m == pytest.approx(total_mass / 10.0, rel=1e-6)
    assert np.array_equal(node_idx, np.arange(10))


# ----------------------------------------------------------------------------
# 2. Virtual midside nodes reconstruction
# ----------------------------------------------------------------------------
def test_tetra10_virtual_midside_nodes_reconstruction():
    # Only 4 corner nodes provided, midside nodes marked as -1
    x0 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, -1, -1, -1, -1, -1, -1]], dtype=np.int64)
    model = _DummyModel(x0)
    group = _DummyGroup(conn)
    mat = _DummyMat(law=1, rho0=1.0, E=100.0, nu=0.0)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    log = _DummyLog()

    node_idx, mass_c, _ = solid_tetra10.init_group(group, model, log)
    assert group.state["vol0"][0] == pytest.approx(1.0 / 6.0, rel=1e-6)

    total_mass = mat.rho0 * (1.0 / 6.0)
    assert group.state["mass"][0] == pytest.approx(total_mass, rel=1e-6)
    # When midsides are slaved (-1), mass is lumped equally onto the 4 corners (m/4 each)
    assert len(mass_c) == 4
    for m in mass_c:
        assert m == pytest.approx(total_mass / 4.0, rel=1e-6)
    assert np.array_equal(node_idx, np.arange(4))


# ----------------------------------------------------------------------------
# 3. Degenerate volume handling
# ----------------------------------------------------------------------------
def test_tetra10_degenerate_volume_handling():
    # Degenerate coplanar 10-node tet: all z coordinates zero
    x_deg = np.zeros((10, 3), dtype=float)
    x_deg[:4, :2] = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.0]]
    x_deg[4:, :2] = 0.5 * (x_deg[[0, 1, 2, 0, 1, 2], :2] + x_deg[[1, 2, 0, 3, 3, 3], :2])

    dndx, vol, vol_tot = solid_tetra10._geometry(x_deg[None, ...])
    assert not np.isnan(vol_tot).any()
    assert not np.isnan(dndx).any()
    assert np.allclose(dndx, 0.0)


# ----------------------------------------------------------------------------
# 4. Constant strain gradient operator
# ----------------------------------------------------------------------------
def test_tetra10_constant_strain_gradient_operator():
    model, group = _create_unit_tetra10()
    dndx, vol, vol_tot = solid_tetra10._geometry(model.x0[None, ...])

    # For each of the 4 Gauss points:
    for k in range(4):
        # 1) Partition of unity: sum_i dN_i/dx = 0
        grad_sum = dndx[0, k].sum(axis=0)  # (3,)
        assert np.allclose(grad_sum, 0.0, atol=1e-7)

        # 2) Linear coordinate reproduction: sum_i x_i * dN_i/dx_j = delta_ij
        grad_x = np.einsum("ia,ib->ab", model.x0, dndx[0, k])
        assert np.allclose(grad_x, np.eye(3), atol=1e-7)


# ----------------------------------------------------------------------------
# 5. Characteristic length and Courant step
# ----------------------------------------------------------------------------
def test_tetra10_characteristic_length_and_courant_step():
    model, group = _create_unit_tetra10()
    mat = _DummyMat(law=1, rho0=1000.0, E=1.0e7, nu=0.25)
    prop = _DummyProp(qa=0.0, qb=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra10.init_group(group, model, _DummyLog())

    fint = np.zeros((10, 3))
    # Cycle 0 Courant step (dt = 0.0)
    dt_crit = solid_tetra10.forces(group, model.x0, model.v, model.vr, 0.0, fint, None)
    assert dt_crit[0] > 0.0
    assert dt_crit[0] < 1.0

    # Negative dt returns EP30
    dt_neg = solid_tetra10.forces(group, model.x0, model.v, model.vr, -1e-5, fint, None)
    assert dt_neg[0] >= 1e29


# ----------------------------------------------------------------------------
# 6. Uniaxial strain analytical
# ----------------------------------------------------------------------------
def test_tetra10_uniaxial_strain_analytical():
    model, group = _create_unit_tetra10()
    E, nu, rho0 = 1000.0, 0.25, 1.0
    mat = _DummyMat(law=1, rho0=rho0, E=E, nu=nu)
    prop = _DummyProp(qa=0.0, qb=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra10.init_group(group, model, _DummyLog())

    # Prescribe 1D strain rate vx = eps_rate * x
    eps_rate = 0.01
    dt = 0.05
    v = np.zeros_like(model.x0)
    v[:, 0] = eps_rate * model.x0[:, 0]

    fint = np.zeros((10, 3))
    solid_tetra10.forces(group, model.x0, v, model.vr, dt, fint, None)

    sig = group.state["sig"][0]  # (4, 6)
    expected_sxx = (mat.K + 4.0 * mat.G / 3.0) * eps_rate * dt
    expected_lat = (mat.K - 2.0 * mat.G / 3.0) * eps_rate * dt

    for k in range(4):
        assert sig[k, 0] == pytest.approx(expected_sxx, rel=1e-6)
        assert sig[k, 1] == pytest.approx(expected_lat, rel=1e-6)
        assert sig[k, 2] == pytest.approx(expected_lat, rel=1e-6)
        assert np.allclose(sig[k, 3:], 0.0, atol=1e-12)

    # Force equilibrium: sum(fint) = 0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-12)


# ----------------------------------------------------------------------------
# 7. Pure shear analytical
# ----------------------------------------------------------------------------
def test_tetra10_pure_shear_analytical():
    model, group = _create_unit_tetra10()
    E, nu = 260.0, 0.3
    G = E / (2.0 * (1.0 + nu))
    mat = _DummyMat(law=1, rho0=1.0, E=E, nu=nu)
    prop = _DummyProp(qa=0.0, qb=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra10.init_group(group, model, _DummyLog())

    gamma_dot = 0.02
    dt = 0.05
    v = np.zeros_like(model.x0)
    v[:, 0] = gamma_dot * model.x0[:, 1]

    fint = np.zeros((10, 3))
    solid_tetra10.forces(group, model.x0, v, model.vr, dt, fint, None)

    sig = group.state["sig"][0]
    expected_sxy = G * gamma_dot * dt
    for k in range(4):
        assert sig[k, 3] == pytest.approx(expected_sxy, rel=1e-6)
        assert np.allclose(sig[k, :3], 0.0, atol=1e-12)

    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-12)


# ----------------------------------------------------------------------------
# 8. Hydrostatic compression analytical
# ----------------------------------------------------------------------------
def test_tetra10_hydrostatic_compression_analytical():
    model, group = _create_unit_tetra10()
    E, nu = 300.0, 0.2
    K = E / (3.0 * (1.0 - 2.0 * nu))
    mat = _DummyMat(law=1, rho0=1.0, E=E, nu=nu)
    prop = _DummyProp(qa=0.0, qb=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra10.init_group(group, model, _DummyLog())

    eps_dot = 0.005
    dt = 0.1
    v = -eps_dot * model.x0

    fint = np.zeros((10, 3))
    solid_tetra10.forces(group, model.x0, v, model.vr, dt, fint, None)

    sig = group.state["sig"][0]
    expected_p = 3.0 * K * eps_dot * dt
    for k in range(4):
        assert sig[k, 0] == pytest.approx(-expected_p, rel=1e-6)
        assert sig[k, 1] == pytest.approx(-expected_p, rel=1e-6)
        assert sig[k, 2] == pytest.approx(-expected_p, rel=1e-6)
        assert np.allclose(sig[k, 3:], 0.0, atol=1e-12)

    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-12)


# ----------------------------------------------------------------------------
# 9. Linear momentum conservation arbitrary 3D
# ----------------------------------------------------------------------------
def test_tetra10_linear_momentum_conservation_arbitrary_3d():
    model, group = _create_unit_tetra10()
    mat = _DummyMat(law=1, rho0=2.0, E=100.0, nu=0.25)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra10.init_group(group, model, _DummyLog())

    # Quadratic arbitrary velocity field
    rng = np.random.default_rng(42)
    v = rng.uniform(-1.0, 1.0, (10, 3))
    fint = np.zeros((10, 3))
    solid_tetra10.forces(group, model.x0, v, model.vr, 0.01, fint, None)

    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-7)


# ----------------------------------------------------------------------------
# 10. Angular momentum conservation arbitrary 3D
# ----------------------------------------------------------------------------
def test_tetra10_angular_momentum_conservation_arbitrary_3d():
    model, group = _create_unit_tetra10()
    mat = _DummyMat(law=1, rho0=2.0, E=150.0, nu=0.3)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra10.init_group(group, model, _DummyLog())

    rng = np.random.default_rng(123)
    v = rng.uniform(-1.0, 1.0, (10, 3))
    fint = np.zeros((10, 3))
    solid_tetra10.forces(group, model.x0, v, model.vr, 0.01, fint, None)

    moment = np.cross(model.x0, fint).sum(axis=0)
    assert np.allclose(moment, 0.0, atol=1e-12)


# ----------------------------------------------------------------------------
# 11. Energy accounting elastic cycle
# ----------------------------------------------------------------------------
def test_tetra10_energy_accounting_elastic_cycle():
    model, group = _create_unit_tetra10()
    mat = _DummyMat(law=1, rho0=1.0, E=200.0, nu=0.0)
    prop = _DummyProp(qa=0.0, qb=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra10.init_group(group, model, _DummyLog())

    vol = group.state["vol0"][0]
    dt = 0.01
    eps_rate = 0.05
    v = np.zeros_like(model.x0)
    v[:, 0] = eps_rate * model.x0[:, 0]

    fint = np.zeros((10, 3))
    solid_tetra10.forces(group, model.x0, v, model.vr, dt, fint, None)

    strain = eps_rate * dt
    expected_energy = 0.5 * mat.E * (strain ** 2) * vol
    assert group.state["eint"][0] == pytest.approx(expected_energy, rel=1e-6)


# ----------------------------------------------------------------------------
# 12. Rigid body rotation produces no stress
# ----------------------------------------------------------------------------
def test_tetra10_rigid_body_rotation_no_stress():
    model, group = _create_unit_tetra10()
    mat = _DummyMat(law=1, rho0=1.0, E=200.0, nu=0.3)
    prop = _DummyProp(qa=0.0, qb=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra10.init_group(group, model, _DummyLog())

    omega = np.array([0.0, 0.0, 1.5])
    v = np.cross(omega, model.x0)

    fint = np.zeros((10, 3))
    solid_tetra10.forces(group, model.x0, v, model.vr, 1e-4, fint, None)

    assert np.allclose(group.state["sig"][0], 0.0, atol=1e-12)
    assert np.allclose(fint, 0.0, atol=1e-12)


# ----------------------------------------------------------------------------
# 13. LAW2 Johnson-Cook plasticity
# ----------------------------------------------------------------------------
def test_tetra10_law2_johnson_cook_plasticity():
    model, group = _create_unit_tetra10()
    mat_params = {
        "A": 50.0, "B": 0.0, "n": 1.0, "sig_max": 1e30,
        "c": 0.0, "eps_dot_0": 1.0, "Chard": 0.0
    }
    mat = _DummyMat(law=2, rho0=1.0, E=1000.0, nu=0.0, params=mat_params)
    prop = _DummyProp(qa=0.0, qb=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra10.init_group(group, model, _DummyLog())

    # Prescribe tension past yield
    eps_rate = 1.0
    dt = 0.1  # strain = 0.1 > yield strain 0.05
    v = np.zeros_like(model.x0)
    v[:, 0] = eps_rate * model.x0[:, 0]

    fint = np.zeros((10, 3))
    solid_tetra10.forces(group, model.x0, v, model.vr, dt, fint, None)

    for k in range(4):
        sig = group.state["sig"][0, k]
        dev = sig[:3] - sig[:3].mean()
        vm = np.sqrt(1.5 * (dev ** 2).sum() + 3.0 * (sig[3:] ** 2).sum())
        assert vm == pytest.approx(50.0, rel=1e-3)
    assert (group.state["epsp"][0] > 0.0).all()


# ----------------------------------------------------------------------------
# 14. LAW0 void behavior
# ----------------------------------------------------------------------------
def test_tetra10_law0_void_behavior():
    model, group = _create_unit_tetra10()
    mat = _DummyMat(law=0, rho0=0.0, E=0.0, nu=0.0)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra10.init_group(group, model, _DummyLog())

    v = np.ones_like(model.x0)
    fint = np.zeros((10, 3))
    dt_crit = solid_tetra10.forces(group, model.x0, v, model.vr, 0.01, fint, None)

    assert np.allclose(group.state["sig"], 0.0)
    assert np.allclose(fint, 0.0)
    assert dt_crit[0] >= 1e29


# ----------------------------------------------------------------------------
# 15. Consistent mass matrix
# ----------------------------------------------------------------------------
def test_tetra10_consistent_mass_matrix():
    model, group = _create_unit_tetra10()
    mat = _DummyMat(law=1, rho0=6.0, E=100.0, nu=0.0)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra10.init_group(group, model, _DummyLog())

    me, edofs = solid_tetra10.consistent_mass(group)
    assert me.shape == (1, 30, 30)
    M = me[0]

    # Symmetry
    assert np.allclose(M, M.T, atol=1e-14)

    # Positive definiteness (all 30 eigenvalues strictly positive)
    eigvals = np.linalg.eigvalsh(M)
    assert (eigvals > 0.0).all()

    # Total mass kinetic energy: total mass = group mass
    v_rig = np.tile([1.0, 0.0, 0.0], 10)
    ke = 0.5 * v_rig @ M @ v_rig
    expected_ke = 0.5 * group.state["mass"][0] * 1.0**2
    assert ke == pytest.approx(expected_ke, rel=1e-12)
    assert group.state["mass"][0] == pytest.approx(1.0, rel=1e-6)


# ----------------------------------------------------------------------------
# 16. Geometric stiffness matrix
# ----------------------------------------------------------------------------
def test_tetra10_geometric_stiffness_matrix():
    model, group = _create_unit_tetra10()
    mat = _DummyMat(law=1, rho0=1.0, E=100.0, nu=0.0)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra10.init_group(group, model, _DummyLog())

    # Zero stress gives zero geometric stiffness
    k_geo, edofs = solid_tetra10.kgeo(group, model.x0)
    assert np.allclose(k_geo, 0.0)

    # Non-zero tensile stress
    group.state["sig"][0] = np.tile([10.0, 5.0, 0.0, 1.0, 0.0, 0.0], (4, 1))
    k_geo, _ = solid_tetra10.kgeo(group, model.x0)
    assert k_geo.shape == (1, 30, 30)
    Kg = k_geo[0]

    # Symmetry
    assert np.allclose(Kg, Kg.T, atol=1e-14)

    # Rigid body translations in null space
    for d in range(3):
        v_trans = np.zeros(30)
        v_trans[d::3] = 1.0
        assert np.allclose(Kg @ v_trans, 0.0, atol=1e-6)


# ----------------------------------------------------------------------------
# 17. Material tangent stiffness nullspace
# ----------------------------------------------------------------------------
def test_tetra10_material_tangent_stiffness_nullspace():
    model, group = _create_unit_tetra10()
    mat = _DummyMat(law=1, rho0=1.0, E=210.0, nu=0.3)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra10.init_group(group, model, _DummyLog())

    ke, edofs = solid_tetra10.tangent(group, model.x0)
    K = ke[0]

    # Symmetry
    assert np.allclose(K, K.T, atol=1e-12)

    # Eigenvalues: exactly 6 zero eigenvalues corresponding to rigid body modes (3 translations + 3 rotations)
    eigvals = np.linalg.eigvalsh(K)
    assert np.sum(np.abs(eigvals) < 1e-10) == 6
    assert np.all(eigvals[6:] > 0.0)


# ----------------------------------------------------------------------------
# 18. Implicit internal forces and tangent consistency
# ----------------------------------------------------------------------------
def test_tetra10_implicit_internal_forces_and_tangent_consistency():
    model, group = _create_unit_tetra10()
    mat = _DummyMat(law=1, rho0=1.0, E=100.0, nu=0.2)
    prop = _DummyProp(qa=0.0, qb=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra10.init_group(group, model, _DummyLog())

    ke, _ = solid_tetra10.tangent(group, model.x0)
    K = ke[0]

    h = 1e-6
    K_num = np.zeros((30, 30))
    for i in range(30):
        group.state["sig"][:] = 0.0
        u_plus = np.zeros((10, 3))
        u_plus.reshape(-1)[i] = h
        f_plus = np.zeros((10, 3))
        solid_tetra10.implicit_internal_forces(group, model.x0, u_plus, None, f_plus, None, nlgeom=False)

        group.state["sig"][:] = 0.0
        u_minus = np.zeros((10, 3))
        u_minus.reshape(-1)[i] = -h
        f_minus = np.zeros((10, 3))
        solid_tetra10.implicit_internal_forces(group, model.x0, u_minus, None, f_minus, None, nlgeom=False)

        K_num[:, i] = -(f_plus.reshape(-1) - f_minus.reshape(-1)) / (2.0 * h)

    assert np.allclose(K, K_num, rtol=1e-5, atol=1e-5)


# ----------------------------------------------------------------------------
# 19. Static internal forces assembly
# ----------------------------------------------------------------------------
def test_tetra10_static_internal_forces():
    model, group = _create_unit_tetra10()
    mat = _DummyMat(law=1, rho0=1.0, E=100.0, nu=0.0)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra10.init_group(group, model, _DummyLog())

    sig = np.tile([10.0, 20.0, 30.0, 5.0, 4.0, 3.0], (4, 1))
    group.state["sig"][0] = sig

    fint = np.zeros((10, 3))
    solid_tetra10.static_internal_forces(group, model.x0, None, None, fint, None)

    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-12)
    assert not np.allclose(fint, 0.0)


# ----------------------------------------------------------------------------
# 20. Starter parsing and defensive edge cases
# ----------------------------------------------------------------------------
def test_tetra10_starter_parsing_and_defensive_edge_cases():
    # Empty element group
    empty_group = _DummyGroup(np.zeros((0, 10), dtype=np.int64))
    empty_model = _DummyModel(np.zeros((0, 3)))
    log = _DummyLog()

    node_idx, mass_c, _ = solid_tetra10.init_group(empty_group, empty_model, log)
    assert len(node_idx) == 0
    assert len(mass_c) == 0

    dt_crit = solid_tetra10.forces(empty_group, empty_model.x, None, None, 0.01, None, None)
    assert len(dt_crit) == 0

    ke, edofs = solid_tetra10.tangent(empty_group, empty_model.x)
    assert ke.shape == (0, 30, 30)
    assert edofs.shape == (0, 30)

    kg, edofs_g = solid_tetra10.kgeo(empty_group, empty_model.x)
    assert kg.shape == (0, 30, 30)

    me, edofs_m = solid_tetra10.consistent_mass(empty_group)
    assert me.shape == (0, 30, 30)

    solid_tetra10.static_internal_forces(empty_group, empty_model.x, None, None, None, None)
    solid_tetra10.implicit_internal_forces(empty_group, empty_model.x, None, None, None, None, nlgeom=False)

    # Virtual node redistribution in forces
    x0 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, -1, -1, -1, -1, -1, -1]], dtype=np.int64)
    model = _DummyModel(x0)
    group = _DummyGroup(conn)
    mat = _DummyMat(law=1, rho0=1.0, E=100.0, nu=0.0)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra10.init_group(group, model, log)

    fint = np.zeros((4, 3))
    dt_slaved = solid_tetra10.forces(group, model.x0, None, None, 0.01, fint, None)
    assert dt_slaved[0] > 0.0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-12)
