"""
Comprehensive unit test suite for Milestone M508:
20-Node Quadratic 3D Hexahedral Solid Element (/BRIC20, /HEXA20, /PROP/TYPE23)
Hardening, 3D Kinematics, Jaumann Stress Rate, 60-DOF Tangent/Geometric Stiffness,
Analytical/Numerical Consistent Mass, Bulk Viscosity & Energy Conservation.
"""

from __future__ import annotations

import pytest
import numpy as np

from pyradioss.elements import solid_bric20
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
    def __init__(self, qa=1.1, qb=0.05):
        self.qa = qa
        self.qb = qb
        self.params = {"qa": qa, "qb": qb}


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


def _create_unit_bric20():
    """Create a model with a single 20-node unit cube [0, 1]^3."""
    model = Model()
    # Map reference coordinates [-1, 1]^3 to unit cube [0, 1]^3: x = 0.5 * (xi + 1)
    x0 = 0.5 * (solid_bric20._XI_NODES + 1.0)
    model.x0 = x0.copy()
    model.x = x0.copy()
    model.v = np.zeros((20, 3), dtype=np.float64)
    model.vr = np.zeros((20, 3), dtype=np.float64)

    conn = np.arange(20, dtype=np.int64)[None, :]
    group = ElementGroup(ids=np.array([1], dtype=np.int64), conn=conn, part=np.array([0], dtype=np.int64))
    return model, group


# ----------------------------------------------------------------------------
# 1. Shape functions and reference nodes
# ----------------------------------------------------------------------------
def test_bric20_shape_functions_and_reference_nodes():
    assert solid_bric20._XI_CORNERS.shape == (8, 3)
    assert solid_bric20._XI_MIDSIDES.shape == (12, 3)
    assert solid_bric20._XI_NODES.shape == (20, 3)

    # Evaluate N at the 20 nodal positions: N_i(node_j) = delta_ij
    N_at_nodes = solid_bric20._eval_n(solid_bric20._XI_NODES)
    assert np.allclose(N_at_nodes, np.eye(20), atol=1e-14)

    # Partition of unity at arbitrary reference points
    rng = np.random.default_rng(42)
    pts_random = rng.uniform(-1.0, 1.0, (50, 3))
    N_rand = solid_bric20._eval_n(pts_random)
    assert np.allclose(N_rand.sum(axis=1), 1.0, atol=1e-14)


# ----------------------------------------------------------------------------
# 2. Unit cube geometry and volume
# ----------------------------------------------------------------------------
def test_bric20_unit_cube_geometry_and_volume():
    model, group = _create_unit_bric20()
    dndx, vol_gp, vol_tot = solid_bric20._geometry(model.x0[None, ...])

    assert dndx.shape == (1, 8, 20, 3)
    assert vol_gp.shape == (1, 8)
    assert vol_tot.shape == (1,)

    # Total volume of unit cube [0, 1]^3 must be 1.0
    assert vol_tot[0] == pytest.approx(1.0, rel=1e-14)
    # Each 2x2x2 Gauss point sub-volume must be 1/8 = 0.125
    assert np.allclose(vol_gp[0], 0.125, atol=1e-14)


# ----------------------------------------------------------------------------
# 3. Degenerate and collinear geometry error resilience
# ----------------------------------------------------------------------------
def test_bric20_degenerate_and_collinear_geometry_safety():
    # 20 identical points (collapsed element)
    xe_deg = np.zeros((1, 20, 3), dtype=np.float64)
    dndx, vol_gp, vol_tot = solid_bric20._geometry(xe_deg)

    assert vol_tot[0] == 0.0
    assert np.all(dndx == 0.0)
    assert not np.isnan(dndx).any()
    assert not np.isnan(vol_gp).any()

    # Coplanar / flat element (all z = 0)
    xe_flat = np.zeros((1, 20, 3), dtype=np.float64)
    xe_flat[0, :, :2] = solid_bric20._XI_NODES[:, :2]
    dndx, vol_gp, vol_tot = solid_bric20._geometry(xe_flat)
    assert vol_tot[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(dndx == 0.0)
    assert not np.isnan(dndx).any()


# ----------------------------------------------------------------------------
# 4. Constant strain gradient operator and partition of unity
# ----------------------------------------------------------------------------
def test_bric20_constant_strain_gradient_operator():
    model, group = _create_unit_bric20()
    dndx, vol_gp, vol_tot = solid_bric20._geometry(model.x0[None, ...])

    for k in range(8):
        # 1) Partition of unity: sum_i dN_i/dx = 0
        grad_sum = dndx[0, k].sum(axis=0)
        assert np.allclose(grad_sum, 0.0, atol=1e-14)

        # 2) Linear coordinate reproduction: sum_i x_i * dN_i/dx_j = delta_ij
        grad_x = np.einsum("ia,ib->ab", model.x0, dndx[0, k])
        assert np.allclose(grad_x, np.eye(3), atol=1e-14)


# ----------------------------------------------------------------------------
# 5. Characteristic length and Courant time step
# ----------------------------------------------------------------------------
def test_bric20_characteristic_length_and_courant_step():
    model, group = _create_unit_bric20()
    mat = _DummyMat(law=1, rho0=1000.0, E=1.0e7, nu=0.25)
    prop = _DummyProp(qa=0.0, qb=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_bric20.init_group(group, model, _DummyLog())

    # Unit cube has vol = 1.0, so lc = 1.0
    lc = solid_bric20._char_length(group.state["vol0"])
    assert lc[0] == pytest.approx(1.0, rel=1e-12)

    # Sound speed = sqrt(E / rho0) = 100 m/s
    c_sound = np.sqrt(1.0e7 / 1000.0)
    # dt_crit = dtfac * lc / c_sound = 0.5 * 1.0 / 100.0 = 0.005 s
    dt_crit = solid_bric20.forces(group, model.x0, model.v, model.vr, 0.0, None, None)
    assert dt_crit[0] == pytest.approx(0.005, rel=1e-6)


# ----------------------------------------------------------------------------
# 6. Empty group handling across all element APIs
# ----------------------------------------------------------------------------
def test_bric20_empty_group_handling():
    empty_group = ElementGroup(ids=np.zeros(0, dtype=np.int64),
                               conn=np.zeros((0, 20), dtype=np.int64),
                               part=np.zeros(0, dtype=np.int64))
    log = _DummyLog()
    model = Model()

    nodes, mass, inertia = solid_bric20.init_group(empty_group, model, log)
    assert len(nodes) == 0
    assert len(mass) == 0

    dt_crit = solid_bric20.forces(empty_group, model.x0, model.v, model.vr, 0.01, None, None)
    assert len(dt_crit) == 0

    ke, edofs = solid_bric20.tangent(empty_group, model.x0)
    assert ke.shape == (0, 60, 60)
    assert edofs.shape == (0, 60)

    kg, edofs_g = solid_bric20.kgeo(empty_group, model.x0)
    assert kg.shape == (0, 60, 60)

    me, edofs_m = solid_bric20.consistent_mass(empty_group)
    assert me.shape == (0, 60, 60)

    # Internal forces on empty group should execute without error
    solid_bric20.static_internal_forces(empty_group, model.x0, None, None, None, None)
    solid_bric20.implicit_internal_forces(empty_group, model.x0, None, None, None, None, nlgeom=False)
    solid_bric20.implicit_internal_forces(empty_group, model.x0, None, None, None, None, nlgeom=True)


# ----------------------------------------------------------------------------
# 7. Virtual midside node reconstruction and force redistribution
# ----------------------------------------------------------------------------
def test_bric20_virtual_midside_nodes_reconstruction():
    model = Model()
    # 8 corner nodes only (unit cube)
    corners = 0.5 * (solid_bric20._XI_CORNERS + 1.0)
    model.x0 = corners.copy()
    model.v = np.zeros((8, 3))
    model.vr = np.zeros((8, 3))

    # Connectivity with 8 real corners (0..7) and 12 virtual midsides (-1)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7] + [-1] * 12], dtype=np.int64)
    group = ElementGroup(ids=np.array([1], dtype=np.int64), conn=conn, part=np.array([0], dtype=np.int64))
    mat = _DummyMat(law=1, rho0=1000.0, E=1.0e7, nu=0.3)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]

    node_idx, mass_c, _ = solid_bric20.init_group(group, model, _DummyLog())
    assert len(node_idx) == 8  # Only 8 real nodes receive mass
    assert mass_c.sum() == pytest.approx(1000.0 * 1.0, rel=1e-12)

    # Force redistribution: fint on the 8 corners should sum to zero under arbitrary motion
    v = np.random.default_rng(42).uniform(-1.0, 1.0, (8, 3))
    fint = np.zeros((8, 3))
    solid_bric20.forces(group, model.x0, v, model.vr, 0.01, fint, None)
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-8)


# ----------------------------------------------------------------------------
# 8. Fortran s20mass3.F lumped mass distribution
# ----------------------------------------------------------------------------
def test_bric20_lumped_mass_fortran_distribution():
    model, group = _create_unit_bric20()
    # Unit cube, rho = 192.0 -> total mass = 192.0
    mat = _DummyMat(law=1, rho0=192.0, E=1.0e7, nu=0.3)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]

    node_idx, mass_c, _ = solid_bric20.init_group(group, model, _DummyLog())
    assert len(node_idx) == 20
    assert mass_c.sum() == pytest.approx(192.0, rel=1e-12)

    # Fortran s20mass3.F:
    # Corner nodes receive mass / 64 = 192 / 64 = 3.0
    # Midside nodes receive 7 * mass / 96 = 7 * 192 / 96 = 14.0
    assert np.allclose(mass_c[:8], 3.0, atol=1e-12)
    assert np.allclose(mass_c[8:], 14.0, atol=1e-12)


# ----------------------------------------------------------------------------
# 9. Uniform velocity constant strain rate and stress evolution
# ----------------------------------------------------------------------------
def test_bric20_uniform_velocity_constant_strain_rate():
    model, group = _create_unit_bric20()
    mat = _DummyMat(law=1, rho0=1000.0, E=2.0e7, nu=0.0)
    prop = _DummyProp(qa=0.0, qb=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_bric20.init_group(group, model, _DummyLog())

    # Pure uniaxial strain rate: v_x = 0.01 * x
    v = np.zeros((20, 3))
    v[:, 0] = 0.01 * model.x0[:, 0]
    dt = 0.1
    fint = np.zeros((20, 3))
    solid_bric20.forces(group, model.x0, v, model.vr, dt, fint, None)

    # Strain increment deps_xx = 0.01 * dt = 0.001
    # Stress increment ds_xx = E * deps_xx = 2.0e7 * 0.001 = 20000 Pa
    sig = group.state["sig"][0]
    assert np.allclose(sig[:, 0], 20000.0, rtol=1e-6)
    assert np.allclose(sig[:, 1:], 0.0, atol=1e-6)


# ----------------------------------------------------------------------------
# 10. Rigid body translational and rotational velocity invariance
# ----------------------------------------------------------------------------
def test_bric20_rigid_body_translation_and_rotation_invariance():
    model, group = _create_unit_bric20()
    mat = _DummyMat(law=1, rho0=1000.0, E=1.0e8, nu=0.3)
    prop = _DummyProp(qa=0.0, qb=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_bric20.init_group(group, model, _DummyLog())

    # 1) Rigid translation v = [10.0, -5.0, 3.0]
    v_trans = np.tile([10.0, -5.0, 3.0], (20, 1))
    fint = np.zeros((20, 3))
    solid_bric20.forces(group, model.x0, v_trans, model.vr, 0.01, fint, None)
    assert np.allclose(fint, 0.0, atol=1e-7)
    assert np.allclose(group.state["sig"], 0.0, atol=1e-7)

    # 2) Rigid rotation v = omega x (r - r0)
    omega = np.array([0.0, 0.0, 2.0])
    v_rot = np.cross(omega, model.x0)
    fint.fill(0.0)
    solid_bric20.forces(group, model.x0, v_rot, model.vr, 0.01, fint, None)
    assert np.allclose(fint, 0.0, atol=1e-7)
    assert np.allclose(group.state["sig"], 0.0, atol=1e-7)


# ----------------------------------------------------------------------------
# 11. Linear momentum conservation arbitrary 3D
# ----------------------------------------------------------------------------
def test_bric20_linear_momentum_conservation_arbitrary_3d():
    model, group = _create_unit_bric20()
    mat = _DummyMat(law=1, rho0=2000.0, E=1.0e8, nu=0.3)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_bric20.init_group(group, model, _DummyLog())

    rng = np.random.default_rng(42)
    v = rng.uniform(-2.0, 2.0, (20, 3))
    fint = np.zeros((20, 3))
    solid_bric20.forces(group, model.x0, v, model.vr, 0.01, fint, None)

    # Sum of internal forces over all 20 nodes must be zero (Newton's 3rd Law)
    assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-8)


# ----------------------------------------------------------------------------
# 12. Angular momentum conservation arbitrary 3D
# ----------------------------------------------------------------------------
def test_bric20_angular_momentum_conservation_arbitrary_3d():
    model, group = _create_unit_bric20()
    mat = _DummyMat(law=1, rho0=2000.0, E=1.5e8, nu=0.25)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_bric20.init_group(group, model, _DummyLog())

    rng = np.random.default_rng(123)
    v = rng.uniform(-1.5, 1.5, (20, 3))
    fint = np.zeros((20, 3))
    solid_bric20.forces(group, model.x0, v, model.vr, 0.01, fint, None)

    # Total torque sum(x_i x F_i) must be zero
    torque = np.cross(model.x0, fint).sum(axis=0)
    assert np.allclose(torque, 0.0, atol=1e-8)


# ----------------------------------------------------------------------------
# 13. Internal energy work balance
# ----------------------------------------------------------------------------
def test_bric20_internal_energy_work_balance():
    model, group = _create_unit_bric20()
    mat = _DummyMat(law=1, rho0=1000.0, E=1.0e7, nu=0.0)
    prop = _DummyProp(qa=0.0, qb=0.0)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_bric20.init_group(group, model, _DummyLog())

    # Uniaxial extension v_x = 0.001 * x
    v = np.zeros((20, 3))
    v[:, 0] = 0.001 * model.x0[:, 0]
    dt = 0.05
    fint = np.zeros((20, 3))
    solid_bric20.forces(group, model.x0, v, model.vr, dt, fint, None)

    # External mechanical work done: W_ext = 0.5 * (-sum(fint * v * dt)) for linear ramp from 0
    w_ext = 0.5 * (-np.sum(fint * v) * dt)
    eint = group.state["eint"][0]
    assert eint == pytest.approx(w_ext, rel=1e-6)
    assert eint > 0.0


# ----------------------------------------------------------------------------
# 14. Bulk viscosity damping pressure
# ----------------------------------------------------------------------------
def test_bric20_bulk_viscosity_damping_pressure():
    model, group = _create_unit_bric20()
    mat = _DummyMat(law=1, rho0=1000.0, E=1.0e7, nu=0.0)
    prop = _DummyProp(qa=1.1, qb=0.05)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_bric20.init_group(group, model, _DummyLog())

    # 1) Compression: tr(D) < 0
    v_comp = -0.1 * model.x0
    fint = np.zeros((20, 3))
    solid_bric20.forces(group, model.x0, v_comp, model.vr, 0.01, fint, None)
    w_visc = group.state["qvw_pend"][0]
    assert w_visc > 0.0  # Damping was booked

    # 2) Expansion: tr(D) > 0
    group.state["qvw_pend"].fill(0.0)
    v_exp = 0.1 * model.x0
    fint.fill(0.0)
    solid_bric20.forces(group, model.x0, v_exp, model.vr, 0.01, fint, None)
    assert group.state["qvw_pend"][0] == 0.0  # No viscosity in tension


# ----------------------------------------------------------------------------
# 15. Element deletion and deactivation
# ----------------------------------------------------------------------------
def test_bric20_element_deletion_and_deactivation():
    model, group = _create_unit_bric20()
    mat = _DummyMat(law=1, rho0=1000.0, E=1.0e7, nu=0.3)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_bric20.init_group(group, model, _DummyLog())

    # Kill the element
    group.state["off"][0] = 0.0

    v = np.random.default_rng(42).uniform(-1.0, 1.0, (20, 3))
    fint = np.zeros((20, 3))
    dt_crit = solid_bric20.forces(group, model.x0, v, model.vr, 0.01, fint, None)

    assert np.allclose(fint, 0.0)
    assert dt_crit[0] == EP30


# ----------------------------------------------------------------------------
# 16. LAW0 void solid material handling
# ----------------------------------------------------------------------------
def test_bric20_law0_void_solid_material():
    model, group = _create_unit_bric20()
    mat = _DummyMat(law=0, rho0=0.0, E=0.0, nu=0.0)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_bric20.init_group(group, model, _DummyLog())

    v = np.random.default_rng(42).uniform(-1.0, 1.0, (20, 3))
    fint = np.zeros((20, 3))
    dt_crit = solid_bric20.forces(group, model.x0, v, model.vr, 0.01, fint, None)

    assert np.allclose(fint, 0.0)
    assert np.allclose(group.state["sig"], 0.0)
    assert dt_crit[0] == EP30

    ke, _ = solid_bric20.tangent(group, model.x0)
    assert np.allclose(ke, 0.0)


# ----------------------------------------------------------------------------
# 17. Consistent mass matrix
# ----------------------------------------------------------------------------
def test_bric20_consistent_mass_matrix():
    model, group = _create_unit_bric20()
    # rho0 = 10.0, vol = 1.0 -> total mass = 10.0
    mat = _DummyMat(law=1, rho0=10.0, E=100.0, nu=0.0)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_bric20.init_group(group, model, _DummyLog())

    me, edofs = solid_bric20.consistent_mass(group)
    assert me.shape == (1, 60, 60)
    assert edofs.shape == (1, 60)
    M = me[0]

    # 1) Symmetry
    assert np.allclose(M, M.T, atol=1e-14)

    # 2) Positive definiteness: all 60 eigenvalues strictly positive
    eigvals = np.linalg.eigvalsh(M)
    assert (eigvals > 0.0).all()

    # 3) Total mass kinetic energy: 0.5 * v^T M v = 0.5 * m * |v|^2
    v_rig = np.tile([1.0, 0.0, 0.0], 20)
    ke = 0.5 * v_rig @ M @ v_rig
    expected_ke = 0.5 * group.state["mass"][0] * 1.0**2
    assert ke == pytest.approx(expected_ke, rel=1e-12)


# ----------------------------------------------------------------------------
# 18. Geometric stiffness matrix
# ----------------------------------------------------------------------------
def test_bric20_geometric_stiffness_matrix():
    model, group = _create_unit_bric20()
    mat = _DummyMat(law=1, rho0=1.0, E=100.0, nu=0.0)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_bric20.init_group(group, model, _DummyLog())

    # Zero stress gives zero geometric stiffness
    k_geo, edofs = solid_bric20.kgeo(group, model.x0)
    assert np.allclose(k_geo, 0.0)

    # Non-zero tensile stress
    group.state["sig"][0] = np.tile([100.0, 50.0, 20.0, 10.0, 5.0, 2.0], (8, 1))
    k_geo, _ = solid_bric20.kgeo(group, model.x0)
    assert k_geo.shape == (1, 60, 60)
    Kg = k_geo[0]

    # Symmetry
    assert np.allclose(Kg, Kg.T, atol=1e-14)

    # Rigid body translations in null space
    for d in range(3):
        v_trans = np.zeros(60)
        v_trans[d::3] = 1.0
        assert np.allclose(Kg @ v_trans, 0.0, atol=1e-12)


# ----------------------------------------------------------------------------
# 19. Material tangent stiffness nullspace
# ----------------------------------------------------------------------------
def test_bric20_material_tangent_stiffness_nullspace():
    model, group = _create_unit_bric20()
    mat = _DummyMat(law=1, rho0=1.0, E=2.1e7, nu=0.3)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_bric20.init_group(group, model, _DummyLog())

    ke, edofs = solid_bric20.tangent(group, model.x0)
    assert ke.shape == (1, 60, 60)
    K = ke[0]

    # Symmetry
    assert np.allclose(K, K.T, atol=1e-6)

    # 1) Rigid body translations in nullspace
    for d in range(3):
        v_trans = np.zeros(60)
        v_trans[d::3] = 1.0
        assert np.allclose(K @ v_trans, 0.0, atol=1e-7)

    # 2) Rigid body rotations in nullspace
    coords = model.x0
    for axis in range(3):
        omega = np.zeros(3)
        omega[axis] = 1.0
        v_rot = np.cross(omega, coords).reshape(60)
        assert np.allclose(K @ v_rot, 0.0, atol=1e-7)


# ----------------------------------------------------------------------------
# 20. Directional derivative consistency with implicit internal forces
# ----------------------------------------------------------------------------
def test_bric20_implicit_internal_forces_directional_derivative():
    model, group = _create_unit_bric20()
    mat = _DummyMat(law=1, rho0=1000.0, E=1.0e6, nu=0.0)
    prop = _DummyProp()
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    solid_bric20.init_group(group, model, _DummyLog())

    ke, _ = solid_bric20.tangent(group, model.x0)
    K = ke[0]

    # Small perturbation delta_u
    rng = np.random.default_rng(99)
    du = 1e-6 * rng.uniform(-1.0, 1.0, (20, 3))
    du_flat = du.reshape(60)

    # Linear implicit internal force: f_lin = -K * du
    f_lin = np.zeros((20, 3))
    solid_bric20.implicit_internal_forces(group, model.x0, du, None, f_lin, None, nlgeom=False)

    expected_f = -(K @ du_flat).reshape(20, 3)
    assert np.allclose(f_lin, expected_f, atol=1e-10)
