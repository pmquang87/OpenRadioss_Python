"""Unit tests for M506: 4-Node 3D Constant-Strain Tetrahedral Solid Element
(/TETRA4 + /PROP/SOLID TYPE14/TYPE6, pyradioss/elements/solid_tetra4.py).
"""

import numpy as np
import pytest

from pyradioss.elements import solid_tetra4


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
    def __init__(self, itetra4=1, qa=1.1, qb=0.05):
        self.params = {"itetra4": itetra4, "qa": qa, "qb": qb}
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


def _create_unit_tetra():
    """Returns a model and group for a standard right-angled unit tetrahedron:
    Node 0: (0, 0, 0)
    Node 1: (1, 0, 0)
    Node 2: (0, 1, 0)
    Node 3: (0, 0, 1)
    Volume = 1/6.
    """
    x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)
    model = _DummyModel(x0)
    group = _DummyGroup(conn)
    return model, group


# ----------------------------------------------------------------------------
# 1. Mass lumping and volume
# ----------------------------------------------------------------------------
def test_tetra4_mass_lumping_and_volume():
    model, group = _create_unit_tetra()
    mat = _DummyMat(law=1, rho0=7.8e-6, E=210.0, nu=0.3)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    log = _DummyLog()

    node_idx, mass_c, _ = solid_tetra4.init_group(group, model, log)
    expected_vol = 1.0 / 6.0
    assert group.state["vol0"][0] == pytest.approx(expected_vol, rel=1e-12)

    total_mass = mat.rho0 * expected_vol
    assert group.state["mass"][0] == pytest.approx(total_mass, rel=1e-12)
    assert len(mass_c) == 4
    for m in mass_c:
        assert m == pytest.approx(total_mass / 4.0, rel=1e-12)
    assert np.array_equal(node_idx, np.array([0, 1, 2, 3]))


# ----------------------------------------------------------------------------
# 2. Degenerate and inverted volume handling
# ----------------------------------------------------------------------------
def test_tetra4_degenerate_and_inverted_volume_handling():
    # Inverted tetrahedron: swap nodes 1 and 2
    x_inv = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    model_inv = _DummyModel(x_inv)
    group_inv = _DummyGroup([[0, 1, 2, 3]])
    mat = _DummyMat(law=1, rho0=1.0, E=100.0, nu=0.0)
    prop = _DummyProp()
    group_inv.state["slices"] = [(slice(0, 1), mat, prop)]
    log = _DummyLog()

    # init_group should canonicalise winding (swap local nodes 2 and 4)
    solid_tetra4.init_group(group_inv, model_inv, log)
    assert group_inv.state["vol0"][0] > 0.0
    assert group_inv.state["vol0"][0] == pytest.approx(1.0 / 6.0, rel=1e-12)

    # Degenerate (coplanar) tetrahedron: all nodes in z=0 plane
    x_deg = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
    ], dtype=float)
    model_deg = _DummyModel(x_deg)
    group_deg = _DummyGroup([[0, 1, 2, 3]])
    group_deg.state["slices"] = [(slice(0, 1), mat, prop)]
    log_deg = _DummyLog()

    solid_tetra4.init_group(group_deg, model_deg, log_deg)
    assert len(log_deg.messages) == 1
    assert log_deg.messages[0][0] == "ERROR"
    # Ensure no NaN
    assert not np.isnan(group_deg.state["vol0"][0])
    assert not np.isnan(group_deg.state["dtfac"][0])


# ----------------------------------------------------------------------------
# 3. Constant strain gradient operator
# ----------------------------------------------------------------------------
def test_tetra4_constant_strain_gradient_operator():
    # Arbitrary non-degenerate tet
    xe = np.array([
        [[1.2, 0.5, -0.3],
         [3.4, 1.1, 0.2],
         [1.0, 4.2, 0.8],
         [2.1, 1.9, 3.5]]
    ], dtype=float)
    dndx, vol = solid_tetra4._geometry(xe)
    assert vol[0] > 0.0

    # 1) Partition of unity: sum of gradients of shape functions must be zero
    grad_sum = dndx[0].sum(axis=0)  # sum over 4 nodes, for each spatial coord x, y, z
    assert np.allclose(grad_sum, 0.0, atol=1e-14)

    # 2) Linear coordinate reproduction: sum_i x_{i,k} * dN_i/dx_j = delta_{kj}
    # (4, 3)^T @ (4, 3) -> (3, 3)
    grad_x = np.einsum("ia,ib->ab", xe[0], dndx[0])
    assert np.allclose(grad_x, np.eye(3), atol=1e-14)


# ----------------------------------------------------------------------------
# 4. Characteristic length and Courant step
# ----------------------------------------------------------------------------
def test_tetra4_characteristic_length_and_courant_step():
    model, group = _create_unit_tetra()
    mat = _DummyMat(law=1, rho0=1000.0, E=1.0e7, nu=0.25)
    prop = _DummyProp(qa=0.0, qb=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra4.init_group(group, model, _DummyLog())

    # Cycle 0 Courant step
    fint = np.zeros((4, 3))
    dt_crit = solid_tetra4.forces(group, model.x0, model.v, model.vr, 0.0, fint, None)
    assert dt_crit[0] > 0.0
    assert dt_crit[0] < 1.0

    # Negative dt returns EP30
    dt_neg = solid_tetra4.forces(group, model.x0, model.v, model.vr, -1e-5, fint, None)
    assert dt_neg[0] >= 1e29


# ----------------------------------------------------------------------------
# 5. Uniaxial tension analytical
# ----------------------------------------------------------------------------
def test_tetra4_uniaxial_tension_analytical():
    model, group = _create_unit_tetra()
    E, nu, rho0 = 1000.0, 0.0, 1.0  # nu = 0 gives simple 1D stress = E * eps
    mat = _DummyMat(law=1, rho0=rho0, E=E, nu=nu)
    prop = _DummyProp(qa=0.0, qb=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra4.init_group(group, model, _DummyLog())

    # Prescribe vx = eps_rate * x
    eps_rate = 0.01
    dt = 0.1
    v = np.zeros_like(model.x0)
    v[:, 0] = eps_rate * model.x0[:, 0]

    fint = np.zeros((4, 3))
    solid_tetra4.forces(group, model.x0, v, model.vr, dt, fint, None)

    sig = group.state["sig"][0]
    expected_sxx = E * eps_rate * dt
    assert sig[0] == pytest.approx(expected_sxx, rel=1e-12)
    assert np.allclose(sig[1:], 0.0, atol=1e-12)

    # Check equilibrium of internal forces: sum(fint) = 0
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-12)


# ----------------------------------------------------------------------------
# 6. Pure shear analytical
# ----------------------------------------------------------------------------
def test_tetra4_pure_shear_analytical():
    model, group = _create_unit_tetra()
    E, nu = 260.0, 0.3
    G = E / (2.0 * (1.0 + nu))
    mat = _DummyMat(law=1, rho0=1.0, E=E, nu=nu)
    prop = _DummyProp(qa=0.0, qb=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra4.init_group(group, model, _DummyLog())

    # Simple shear: vx = gamma_dot * y, vy = 0, vz = 0
    gamma_dot = 0.02
    dt = 0.05
    v = np.zeros_like(model.x0)
    v[:, 0] = gamma_dot * model.x0[:, 1]

    fint = np.zeros((4, 3))
    solid_tetra4.forces(group, model.x0, v, model.vr, dt, fint, None)

    sig = group.state["sig"][0]
    # sig[3] is xy engineering shear stress
    expected_sxy = G * gamma_dot * dt
    assert sig[3] == pytest.approx(expected_sxy, rel=1e-12)
    assert np.allclose(sig[:3], 0.0, atol=1e-12)
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-12)


# ----------------------------------------------------------------------------
# 7. Hydrostatic compression analytical
# ----------------------------------------------------------------------------
def test_tetra4_hydrostatic_compression_analytical():
    model, group = _create_unit_tetra()
    E, nu = 300.0, 0.2
    K = E / (3.0 * (1.0 - 2.0 * nu))
    mat = _DummyMat(law=1, rho0=1.0, E=E, nu=nu)
    prop = _DummyProp(qa=0.0, qb=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra4.init_group(group, model, _DummyLog())

    # Uniform volumetric compression
    eps_dot = 0.005
    dt = 0.1
    v = -eps_dot * model.x0

    fint = np.zeros((4, 3))
    solid_tetra4.forces(group, model.x0, v, model.vr, dt, fint, None)

    sig = group.state["sig"][0]
    expected_p = 3.0 * K * eps_dot * dt
    assert sig[0] == pytest.approx(-expected_p, rel=1e-12)
    assert sig[1] == pytest.approx(-expected_p, rel=1e-12)
    assert sig[2] == pytest.approx(-expected_p, rel=1e-12)
    assert np.allclose(sig[3:], 0.0, atol=1e-12)
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-12)


# ----------------------------------------------------------------------------
# 8. Linear momentum conservation arbitrary 3D
# ----------------------------------------------------------------------------
def test_tetra4_linear_momentum_conservation_arbitrary_3d():
    # Random node coordinates
    rng = np.random.default_rng(42)
    x0 = rng.uniform(0.0, 5.0, (4, 3))
    model = _DummyModel(x0)
    group = _DummyGroup([[0, 1, 2, 3]])
    mat = _DummyMat(law=1, rho0=2.0, E=100.0, nu=0.25)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra4.init_group(group, model, _DummyLog())

    # Arbitrary non-uniform velocity field
    v = rng.uniform(-1.0, 1.0, (4, 3))
    fint = np.zeros((4, 3))
    solid_tetra4.forces(group, model.x0, v, model.vr, 0.01, fint, None)

    # Net force must vanish identically
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-13)


# ----------------------------------------------------------------------------
# 9. Angular momentum conservation arbitrary 3D
# ----------------------------------------------------------------------------
def test_tetra4_angular_momentum_conservation_arbitrary_3d():
    rng = np.random.default_rng(123)
    x0 = rng.uniform(0.0, 5.0, (4, 3))
    model = _DummyModel(x0)
    group = _DummyGroup([[0, 1, 2, 3]])
    mat = _DummyMat(law=1, rho0=2.0, E=150.0, nu=0.3)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra4.init_group(group, model, _DummyLog())

    v = rng.uniform(-1.0, 1.0, (4, 3))
    fint = np.zeros((4, 3))
    solid_tetra4.forces(group, model.x0, v, model.vr, 0.01, fint, None)

    # Net moment about origin sum(x_i x f_i) must be zero for symmetric Cauchy stress
    moment = np.cross(model.x0, fint).sum(axis=0)
    assert np.allclose(moment, 0.0, atol=1e-12)


# ----------------------------------------------------------------------------
# 10. Energy accounting elastic cycle
# ----------------------------------------------------------------------------
def test_tetra4_energy_accounting_elastic_cycle():
    model, group = _create_unit_tetra()
    mat = _DummyMat(law=1, rho0=1.0, E=200.0, nu=0.0)
    prop = _DummyProp(qa=0.0, qb=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra4.init_group(group, model, _DummyLog())

    vol = group.state["vol0"][0]
    dt = 0.01
    eps_rate = 0.05
    v = np.zeros_like(model.x0)
    v[:, 0] = eps_rate * model.x0[:, 0]

    # Loading step
    fint = np.zeros((4, 3))
    solid_tetra4.forces(group, model.x0, v, model.vr, dt, fint, None)

    strain = eps_rate * dt
    expected_energy = 0.5 * mat.E * (strain ** 2) * vol
    assert group.state["eint"][0] == pytest.approx(expected_energy, rel=1e-10)


# ----------------------------------------------------------------------------
# 11. Rigid body rotation produces no stress
# ----------------------------------------------------------------------------
def test_tetra4_rigid_body_rotation_no_stress():
    model, group = _create_unit_tetra()
    mat = _DummyMat(law=1, rho0=1.0, E=200.0, nu=0.3)
    prop = _DummyProp(qa=0.0, qb=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra4.init_group(group, model, _DummyLog())

    # Rigid rotation with angular velocity omega = [0, 0, 1] -> v = omega x r
    omega = np.array([0.0, 0.0, 1.5])
    v = np.cross(omega, model.x0)

    fint = np.zeros((4, 3))
    solid_tetra4.forces(group, model.x0, v, model.vr, 1e-4, fint, None)

    assert np.allclose(group.state["sig"][0], 0.0, atol=1e-14)
    assert np.allclose(fint, 0.0, atol=1e-14)


# ----------------------------------------------------------------------------
# 12. LAW2 Johnson-Cook plasticity
# ----------------------------------------------------------------------------
def test_tetra4_law2_johnson_cook_plasticity():
    model, group = _create_unit_tetra()
    # Simple elastic-plastic: A = 50.0, B = 0, C = 0
    mat_params = {
        "A": 50.0, "B": 0.0, "n": 1.0, "sig_max": 1e30,
        "c": 0.0, "eps_dot_0": 1.0, "Chard": 0.0
    }
    mat = _DummyMat(law=2, rho0=1.0, E=1000.0, nu=0.0, params=mat_params)
    prop = _DummyProp(qa=0.0, qb=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra4.init_group(group, model, _DummyLog())

    # Pull past yield strain (eps_y = 50 / 1000 = 0.05)
    eps_rate = 1.0
    dt = 0.1  # eps = 0.1 > 0.05
    v = np.zeros_like(model.x0)
    v[:, 0] = eps_rate * model.x0[:, 0]

    fint = np.zeros((4, 3))
    solid_tetra4.forces(group, model.x0, v, model.vr, dt, fint, None)

    sig = group.state["sig"][0]
    dev = sig[:3] - sig[:3].mean()
    vm = np.sqrt(1.5 * (dev ** 2).sum() + 3.0 * (sig[3:] ** 2).sum())
    assert vm == pytest.approx(50.0, rel=1e-3)
    assert group.state["epsp"][0] > 0.0


# ----------------------------------------------------------------------------
# 13. LAW0 void behavior
# ----------------------------------------------------------------------------
def test_tetra4_law0_void_behavior():
    model, group = _create_unit_tetra()
    mat = _DummyMat(law=0, rho0=0.0, E=0.0, nu=0.0)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra4.init_group(group, model, _DummyLog())

    v = np.ones_like(model.x0)
    fint = np.zeros((4, 3))
    dt_crit = solid_tetra4.forces(group, model.x0, v, model.vr, 0.01, fint, None)

    assert np.allclose(group.state["sig"], 0.0)
    assert np.allclose(fint, 0.0)
    assert dt_crit[0] >= 1e29  # void element does not restrict time step


# ----------------------------------------------------------------------------
# 14. Consistent mass matrix
# ----------------------------------------------------------------------------
def test_tetra4_consistent_mass_matrix():
    model, group = _create_unit_tetra()
    mat = _DummyMat(law=1, rho0=6.0, E=100.0, nu=0.0)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra4.init_group(group, model, _DummyLog())

    me, edofs = solid_tetra4.consistent_mass(group)
    assert me.shape == (1, 12, 12)
    M = me[0]

    # Symmetry
    assert np.allclose(M, M.T, atol=1e-14)

    # Positive definite
    eigvals = np.linalg.eigvalsh(M)
    assert (eigvals > 0.0).all()

    # Total mass test: total mass is rho * vol = 6.0 * (1/6) = 1.0
    # Rigid unit translation along x: v = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    v_rig = np.tile([1.0, 0.0, 0.0], 4)
    ke = 0.5 * v_rig @ M @ v_rig
    assert ke == pytest.approx(0.5 * 1.0 * 1.0, rel=1e-12)

    # Row sum of mass blocks = m/4 * I
    for a in range(4):
        block_sum = sum(M[a*3:(a+1)*3, b*3:(b+1)*3] for b in range(4))
        assert np.allclose(block_sum, 0.25 * np.eye(3), atol=1e-12)


# ----------------------------------------------------------------------------
# 15. Geometric stiffness matrix
# ----------------------------------------------------------------------------
def test_tetra4_geometric_stiffness_matrix():
    model, group = _create_unit_tetra()
    mat = _DummyMat(law=1, rho0=1.0, E=100.0, nu=0.0)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra4.init_group(group, model, _DummyLog())

    # Zero stress gives zero geometric stiffness
    k_geo, edofs = solid_tetra4.kgeo(group, model.x0)
    assert np.allclose(k_geo, 0.0)

    # Non-zero tensile stress
    group.state["sig"][0] = [10.0, 5.0, 0.0, 1.0, 0.0, 0.0]
    k_geo, _ = solid_tetra4.kgeo(group, model.x0)
    assert k_geo.shape == (1, 12, 12)
    Kg = k_geo[0]

    # Symmetry
    assert np.allclose(Kg, Kg.T, atol=1e-14)

    # Rigid body translation modes in null space
    for d in range(3):
        v_trans = np.zeros(12)
        v_trans[d::3] = 1.0
        assert np.allclose(Kg @ v_trans, 0.0, atol=1e-12)


# ----------------------------------------------------------------------------
# 16. Material tangent stiffness nullspace
# ----------------------------------------------------------------------------
def test_tetra4_material_tangent_stiffness_nullspace():
    model, group = _create_unit_tetra()
    mat = _DummyMat(law=1, rho0=1.0, E=210.0, nu=0.3)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra4.init_group(group, model, _DummyLog())

    ke, edofs = solid_tetra4.tangent(group, model.x0)
    K = ke[0]

    # Symmetry
    assert np.allclose(K, K.T, atol=1e-12)

    # Eigenvalues: exactly 6 rigid body modes (3 translations + 3 rotations)
    eigvals = np.linalg.eigvalsh(K)
    assert np.sum(np.abs(eigvals) < 1e-10) == 6
    assert np.all(eigvals[6:] > 0.0)


# ----------------------------------------------------------------------------
# 17. Implicit internal forces and tangent consistency
# ----------------------------------------------------------------------------
def test_tetra4_implicit_internal_forces_and_tangent_consistency():
    model, group = _create_unit_tetra()
    mat = _DummyMat(law=1, rho0=1.0, E=100.0, nu=0.2)
    prop = _DummyProp(qa=0.0, qb=0.0)  # statics runs without bulk viscosity
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra4.init_group(group, model, _DummyLog())

    ke, _ = solid_tetra4.tangent(group, model.x0)
    K = ke[0]

    # Compute numerical stiffness via finite differences on implicit_internal_forces
    # Reset state before each evaluation to mirror implicit driver state snapshot/restore
    h = 1e-6
    K_num = np.zeros((12, 12))
    for i in range(12):
        group.state["sig"][:] = 0.0
        u_plus = np.zeros((4, 3))
        u_plus.reshape(-1)[i] = h
        f_plus = np.zeros((4, 3))
        solid_tetra4.implicit_internal_forces(group, model.x0, u_plus, None, f_plus, None, nlgeom=False)

        group.state["sig"][:] = 0.0
        u_minus = np.zeros((4, 3))
        u_minus.reshape(-1)[i] = -h
        f_minus = np.zeros((4, 3))
        solid_tetra4.implicit_internal_forces(group, model.x0, u_minus, None, f_minus, None, nlgeom=False)

        # Note: fint accumulates -int(B^T sig dV), so df = -K du -> K = -df / du
        K_num[:, i] = -(f_plus.reshape(-1) - f_minus.reshape(-1)) / (2.0 * h)

    assert np.allclose(K, K_num, rtol=1e-5, atol=1e-5)


# ----------------------------------------------------------------------------
# 18. Static internal forces assembly
# ----------------------------------------------------------------------------
def test_tetra4_static_internal_forces():
    model, group = _create_unit_tetra()
    mat = _DummyMat(law=1, rho0=1.0, E=100.0, nu=0.0)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra4.init_group(group, model, _DummyLog())

    # Prescribe a stress state
    sig = np.array([10.0, 20.0, 30.0, 5.0, 4.0, 3.0])
    group.state["sig"][0] = sig

    fint = np.zeros((4, 3))
    solid_tetra4.static_internal_forces(group, model.x0, None, None, fint, None)

    # Equilibrium check
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-12)
    assert not np.allclose(fint, 0.0)


# ----------------------------------------------------------------------------
# 19. Starter parsing and defensive edge cases
# ----------------------------------------------------------------------------
def test_tetra4_starter_parsing_and_defensive_edge_cases():
    # Empty element group
    empty_group = _DummyGroup(np.zeros((0, 4), dtype=np.int64))
    empty_model = _DummyModel(np.zeros((0, 3)))
    log = _DummyLog()

    node_idx, mass_c, _ = solid_tetra4.init_group(empty_group, empty_model, log)
    assert len(node_idx) == 0
    assert len(mass_c) == 0

    solid_tetra4.pre_forces(empty_group, empty_model, empty_model.x, 0.01)

    dt_crit = solid_tetra4.forces(empty_group, empty_model.x, None, None, 0.01, None, None)
    assert len(dt_crit) == 0

    ke, edofs = solid_tetra4.tangent(empty_group, empty_model.x)
    assert ke.shape == (0, 12, 12)
    assert edofs.shape == (0, 12)

    kg, edofs_g = solid_tetra4.kgeo(empty_group, empty_model.x)
    assert kg.shape == (0, 12, 12)

    me, edofs_m = solid_tetra4.consistent_mass(empty_group)
    assert me.shape == (0, 12, 12)

    solid_tetra4.static_internal_forces(empty_group, empty_model.x, None, None, None, None)
    solid_tetra4.implicit_internal_forces(empty_group, empty_model.x, None, None, None, None, nlgeom=False)

    # v is None handling with non-empty group
    model, group = _create_unit_tetra()
    mat = _DummyMat(law=1, rho0=1.0, E=100.0, nu=0.0)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_tetra4.init_group(group, model, log)

    fint = np.zeros((4, 3))
    dt_none = solid_tetra4.forces(group, model.x0, None, None, 0.01, fint, None)
    assert dt_none[0] > 0.0
    assert np.allclose(fint, 0.0)


# ----------------------------------------------------------------------------
# 20. Itetra3 smoothed FEM initialization
# ----------------------------------------------------------------------------
def test_tetra4_itetra3_smoothed_fem_initialization():
    # Two tetrahedra sharing a triangular face
    x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ], dtype=float)
    conn = np.array([
        [0, 1, 2, 3],
        [0, 2, 1, 4],  # Note: oriented so volume is positive
    ], dtype=np.int64)
    model = _DummyModel(x0)
    group = _DummyGroup(conn)
    mat = _DummyMat(law=1, rho0=1.0, E=100.0, nu=0.2)
    prop = _DummyProp(itetra4=3)
    group.state["slices"] = [(slice(0, 2), mat, prop)]
    log = _DummyLog()

    solid_tetra4.init_group(group, model, log)
    assert "sfem_isrot3" in group.state
    assert group.state["sfem_isrot3"].all()
    assert hasattr(model, "nodal_vol_0")
    # Shared nodes 0, 1, 2 should receive volume contributions from both tets
    assert model.nodal_vol_0[0] == pytest.approx(1.0 / 3.0, rel=1e-12)

    # Pre-forces updates nodal_vol_t
    solid_tetra4.pre_forces(group, model, model.x0, 0.01)
    assert hasattr(model, "nodal_vol_t")
    assert model.nodal_vol_t[0] == pytest.approx(1.0 / 3.0, rel=1e-12)

    # Forces pass runs smoothly with SFEM active
    fint = np.zeros((5, 3))
    dt_crit = solid_tetra4.forces(group, model.x0, model.v, model.vr, 0.01, fint, None)
    assert (dt_crit > 0.0).all()
