"""
tests/test_m505_sh3n.py — Milestone M505: 3-Node C0 Triangular Shell Element (/SH3N) Unit Tests.

Comprehensive direct unit test suite for pyradioss/elements/shell_tri3.py (670 lines)
and /SH3N + /PROP/SHELL parsing in pyradioss/input/starter_keywords.py:
1. Lumped translational mass (m/3) and Key's rotational inertia.
2. Zero-area degenerate element detection and error logging.
3. Corotational orthonormal triad construction, det(E)=1, and 3D frame invariance.
4. Collinear/degenerate nodes orthonormal fallback.
5. Constant-strain triangle (CST) membrane kinematics and force resultants.
6. Mindlin-Reissner plate bending kinematics, curvature rates, and moment resultants.
7. Centroid transverse shear rate and exact invariance under rigid out-of-plane tilt.
8. Exact 3D linear momentum conservation (sum F = 0) to machine precision.
9. Exact 3D angular momentum conservation (sum M + sum r x F = 0) to machine precision.
10. Energy accounting, reversible elastic loading cycle, and zero hourglass energy.
11. Characteristic length (smallest altitude) and Courant critical time step.
12. Through-thickness Gauss-Legendre multi-layer integration (NIP) and stress profiles.
13. LAW2 Johnson-Cook plane-stress elastoplasticity, radial return, and work hardening.
14. LAW0 VOID shell element zero-force and zero-stiffness behavior.
15. Consistent mass matrix symmetry, translational row-sum mass, and rigid kinetic energy.
16. Geometric stiffness matrix K_geo symmetry, translational nullspace, and restoring force.
17. Material tangent stiffness K_mat 6-DOF rigid body nullspace (rank 12).
18. Directional derivative consistency between material tangent and implicit internal forces.
19. Static internal forces evaluation at deformed configuration.
20. Defensive edge cases (empty groups, dt <= 0, None arrays) and starter keyword parsing.
"""

import io
import numpy as np
import pytest

from pyradioss.common.constants import EM20, EP30, SHEAR_FACTOR
from pyradioss.common.messages import MessageLog
from pyradioss.elements import shell_tri3
from pyradioss.model.entities import Material, Property, Part
from pyradioss.model.model import ElementGroup, Model
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.starter.initialization import build_element_groups


def _make_sh3n_model(coords, conn, thick=1.0, nip=3, E=210000.0, nu=0.3,
                     rho0=7.85e-9, law=1, jc_params=None):
    """Helper to construct a model with a 3-node triangular shell element group."""
    model = Model()
    model.x0 = np.array(coords, dtype=float)
    model.x = model.x0.copy()

    part = Part(id=1, prop_id=1, mat_id=1, title="SH3N_PART")
    model.parts[1] = part
    model.parts_list = [part]

    prop = Property(id=1, type=1, title="PROP_SHELL",
                    params={"thick": thick, "nip": nip, "hm": 0.01, "hf": 0.01, "hr": 0.01, "ishell": 0, "ish3n": 0})
    model.properties[1] = prop

    mat_params = {"E": E, "nu": nu}
    if law == 0:
        mat_params = {"E": 0.0, "nu": 0.0}
    elif law == 2 and jc_params is not None:
        mat_params.update(jc_params)
    mat = Material(id=1, law=law, rho0=rho0, params=mat_params)
    model.materials[1] = mat

    log = MessageLog()
    conn_arr = np.array(conn, dtype=int)
    n_elems = len(conn_arr)
    group = ElementGroup(
        ids=np.arange(1, n_elems + 1, dtype=int),
        conn=conn_arr,
        part=np.zeros(n_elems, dtype=int),
    )
    group.state = {
        "slices": [(slice(0, n_elems), mat, prop)],
        "off": np.ones(n_elems, dtype=float),
        "chk_fail": False,
    }
    if n_elems > 0:
        shell_tri3.init_group(group, model, log)
    return model, group, log


def test_sh3n_mass_lumping_and_rotary_inertia():
    """1. Verify exact lumped translational mass (m/3) and Key's rotational inertia."""
    # Right-angled triangle: base = 3.0, height = 4.0, area = 6.0
    coords = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
    conn = [[0, 1, 2]]
    thick = 0.5
    rho0 = 2.0e-6
    area = 0.5 * 3.0 * 4.0
    expected_total_mass = rho0 * thick * area
    expected_nodal_mass = expected_total_mass / 3.0
    expected_nodal_inertia = expected_nodal_mass * (area / 4.5 + thick ** 2 / 12.0)

    model, group, log = _make_sh3n_model(coords, conn, thick=thick, rho0=rho0)
    node_idx, mass_c, inertia_c = shell_tri3.init_group(group, model, log)

    assert len(mass_c) == 3
    assert np.allclose(mass_c, expected_nodal_mass)
    assert np.isclose(np.sum(mass_c), expected_total_mass)
    assert np.allclose(inertia_c, expected_nodal_inertia)
    assert np.isclose(group.state["mass"][0], expected_total_mass)
    assert np.isclose(group.state["area0"][0], area)


def test_sh3n_zero_area_element_logging():
    """2. Verify zero-area degenerate element detection and error logging."""
    coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]  # collinear
    conn = [[0, 1, 2]]
    model, group, log = _make_sh3n_model(coords, conn)
    assert any("zero area" in e for e in log.errors)


def test_sh3n_orthonormal_frame_construction():
    """3. Verify corotational orthonormal triad construction, det(E)=1, in 3D."""
    # Triangular element rotated in 3D
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    coords_local = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    coords_rot = coords_local @ R.T
    conn = [[0, 1, 2]]

    xe = coords_rot[conn]
    E, xl, area, B1, B2 = shell_tri3._local_geometry(xe)

    # Frame orthonormality: E.T @ E == I3, det(E) == 1.0
    assert np.allclose(E[0].T @ E[0], np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(E[0]), 1.0, atol=1e-12)
    # Area equals 2.0
    assert np.isclose(area[0], 2.0)
    # Gradient operators B1, B2 satisfy partition of unity and gradient identities
    # sum(B1) == 0, sum(B2) == 0
    assert np.isclose(np.sum(B1[0]), 0.0, atol=1e-12)
    assert np.isclose(np.sum(B2[0]), 0.0, atol=1e-12)
    # B1 . x == 1, B2 . y == 1, B1 . y == 0, B2 . x == 0
    xl0 = xl[0]
    assert np.isclose(np.dot(B1[0], xl0[:, 0]), 1.0, atol=1e-12)
    assert np.isclose(np.dot(B2[0], xl0[:, 1]), 1.0, atol=1e-12)
    assert np.isclose(np.dot(B1[0], xl0[:, 1]), 0.0, atol=1e-12)
    assert np.isclose(np.dot(B2[0], xl0[:, 0]), 0.0, atol=1e-12)


def test_sh3n_collinear_nodes_fallback():
    """4. Verify collinear/degenerate nodes fallback maintains orthonormal triad without NaN."""
    xe = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    E, xl, area, B1, B2 = shell_tri3._local_geometry(xe)
    assert not np.any(np.isnan(E))
    assert np.allclose(E[0].T @ E[0], np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(E[0]), 1.0, atol=1e-12)


def test_sh3n_cst_constant_strain_membrane():
    """5. Verify CST constant-strain membrane rates and force resultants."""
    coords = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    conn = [[0, 1, 2]]
    thick = 0.1
    E_mod = 200000.0
    nu = 0.0  # uncoupled for exact check
    model, group, log = _make_sh3n_model(coords, conn, thick=thick, E=E_mod, nu=nu, nip=1)

    # Apply pure uniform strain rate: exx = 10.0, eyy = 20.0
    v = np.zeros((3, 3))
    vr = np.zeros((3, 3))
    dt = 1e-3
    exx_rate = 10.0
    eyy_rate = 20.0
    v[1, 0] = exx_rate * 2.0
    v[2, 1] = eyy_rate * 2.0

    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    shell_tri3.forces(group, model.x, v, vr, dt, fint, mint)

    # Check stress after dt: sig_xx = E * exx_rate * dt, sig_yy = E * eyy_rate * dt
    expected_sig_xx = E_mod * exx_rate * dt
    expected_sig_yy = E_mod * eyy_rate * dt
    actual_sig = group.state["sig"][0, 0]
    assert np.isclose(actual_sig[0], expected_sig_xx, rtol=1e-5)
    assert np.isclose(actual_sig[1], expected_sig_yy, rtol=1e-5)
    assert np.isclose(actual_sig[2], 0.0, atol=1e-5)


def test_sh3n_mindlin_reissner_pure_bending():
    """6. Verify Mindlin-Reissner plate bending curvature rates and moment resultants."""
    coords = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    conn = [[0, 1, 2]]
    thick = 0.2
    E_mod = 210000.0
    nu = 0.3
    model, group, log = _make_sh3n_model(coords, conn, thick=thick, E=E_mod, nu=nu, nip=3)

    # Apply uniform curvature rate kappa_xx = 1.0
    # kappa_xx = B1 . thy -> thy = kappa_xx * x
    v = np.zeros((3, 3))
    vr = np.zeros((3, 3))
    vr[1, 1] = 1.0 * 2.0  # node 1 at x=2.0 has thy = 2.0
    dt = 1e-4

    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    shell_tri3.forces(group, model.x, v, vr, dt, fint, mint)

    # Bending moments should be non-zero and linear in curvature
    # In Mindlin plate, outer layers experience +z and -z strains
    sig_layers = group.state["sig"][0]  # (3, 3)
    assert sig_layers[0, 0] < 0.0       # bottom layer in compression
    assert sig_layers[2, 0] > 0.0       # top layer in tension
    assert np.isclose(sig_layers[1, 0], 0.0, atol=1e-4)  # mid-surface layer zero


def test_sh3n_transverse_shear_rigid_rotation_invariance():
    """7. Verify centroid transverse shear strain rate is identically zero under rigid body tilt."""
    coords = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 2.0, 0.0]]
    conn = [[0, 1, 2]]
    thick = 0.1
    model, group, log = _make_sh3n_model(coords, conn, thick=thick)

    # Rigid body rotation about Y axis with rate omega_y = 5.0 rad/s
    # In-plane rotation thy = 5.0 everywhere
    # Out-of-plane velocity vz = -omega_y * (x - xc)
    omega_y = 5.0
    xc = model.x[conn].mean(axis=1)[0, 0]
    v = np.zeros((3, 3))
    vr = np.zeros((3, 3))
    for i in range(3):
        v[i, 2] = -omega_y * (coords[i][0] - xc)
        vr[i, 1] = omega_y

    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    shell_tri3.forces(group, model.x, v, vr, 1e-3, fint, mint)

    # Transverse shear stress qshear must be zero to machine precision
    assert np.allclose(group.state["qshear"][0], 0.0, atol=1e-12)


def test_sh3n_linear_momentum_conservation_arbitrary_3d():
    """8. Verify exact linear momentum conservation sum F = 0 in arbitrary 3D orientations."""
    np.random.seed(123)
    # Random 3D oriented triangle
    coords = np.random.uniform(-5.0, 5.0, (3, 3))
    conn = [[0, 1, 2]]
    model, group, log = _make_sh3n_model(coords, conn, thick=0.2)

    # Random velocities and angular velocities
    v = np.random.uniform(-10.0, 10.0, (3, 3))
    vr = np.random.uniform(-10.0, 10.0, (3, 3))
    dt = 1e-4
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))

    shell_tri3.forces(group, model.x, v, vr, dt, fint, mint)

    # Sum of internal forces over all nodes must be exactly zero
    sum_f = np.sum(fint, axis=0)
    assert np.allclose(sum_f, 0.0, atol=1e-12)


def test_sh3n_angular_momentum_conservation_arbitrary_3d():
    """9. Verify exact angular momentum conservation sum M + sum r x F = 0 in 3D."""
    np.random.seed(456)
    coords = np.random.uniform(-3.0, 3.0, (3, 3))
    conn = [[0, 1, 2]]
    model, group, log = _make_sh3n_model(coords, conn, thick=0.15)

    v = np.random.uniform(-5.0, 5.0, (3, 3))
    vr = np.random.uniform(-5.0, 5.0, (3, 3))
    dt = 1e-4
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))

    shell_tri3.forces(group, model.x, v, vr, dt, fint, mint)

    # Moment arm r_i relative to origin
    total_moment = np.zeros(3)
    for i in range(3):
        total_moment += -mint[i] + np.cross(coords[i], -fint[i])

    assert np.allclose(total_moment, 0.0, atol=1e-12)


def test_sh3n_energy_accounting_reversible_elastic_cycle():
    """10. Verify internal energy accounting and reversibility over an elastic cycle."""
    coords = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    conn = [[0, 1, 2]]
    model, group, log = _make_sh3n_model(coords, conn, thick=0.1, E=200000.0, nu=0.3)

    v = np.zeros((3, 3))
    vr = np.zeros((3, 3))
    v[1, 0] = 10.0  # stretch side 1-2
    dt = 1e-4

    # Forward loading: 10 steps
    for _ in range(10):
        fint = np.zeros((3, 3))
        mint = np.zeros((3, 3))
        shell_tri3.forces(group, model.x, v, vr, dt, fint, mint)
        model.x += v * dt

    eint_peak = group.state["eint"][0]
    assert eint_peak > 0.0
    assert group.state["ehour"][0] == 0.0  # CST has zero hourglass energy

    # Reverse loading: 10 steps
    v_rev = -v
    for _ in range(10):
        fint = np.zeros((3, 3))
        mint = np.zeros((3, 3))
        shell_tri3.forces(group, model.x, v_rev, vr, dt, fint, mint)
        model.x += v_rev * dt

    eint_return = group.state["eint"][0]
    assert np.isclose(eint_return, 0.0, atol=1e-3 * eint_peak)


def test_sh3n_critical_timestep_courant():
    """11. Verify characteristic length (altitude) and Courant time step."""
    # Right-angled triangle with sides 3, 4, 5. Area = 6.
    # Longest side is 5. Smallest altitude = 2 * Area / 5 = 12 / 5 = 2.4.
    coords = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
    conn = [[0, 1, 2]]
    E_mod = 210000.0
    nu = 0.3
    rho0 = 7.85e-9
    thick = 0.05
    model, group, log = _make_sh3n_model(coords, conn, thick=thick, E=E_mod, nu=nu, rho0=rho0)

    c_sound = np.sqrt(E_mod / (rho0 * (1.0 - nu ** 2)))
    expected_lc = 2.0 * 6.0 / 5.0  # 2.4
    expected_dt_courant = expected_lc / c_sound

    v = np.zeros((3, 3))
    vr = np.zeros((3, 3))
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    dt_elem = shell_tri3.forces(group, model.x, v, vr, 1e-5, fint, mint)

    assert dt_elem[0] <= expected_dt_courant * 1.001
    assert dt_elem[0] >= expected_dt_courant * 0.5


def test_sh3n_through_thickness_nip_integration():
    """12. Verify multi-layer NIP Gauss-Legendre quadrature integration."""
    coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    conn = [[0, 1, 2]]
    thick = 0.2
    model, group, log = _make_sh3n_model(coords, conn, thick=thick, nip=5)

    assert group.state["sig"].shape == (1, 5, 3)
    zrel, wrel = group.state["zw"][0]
    assert len(zrel) == 5
    assert np.isclose(np.sum(wrel), 1.0)
    assert np.allclose(zrel, -zrel[::-1])


def test_sh3n_law2_johnson_cook_plasticity():
    """13. Verify LAW2 Johnson-Cook plane stress radial return plasticity."""
    coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    conn = [[0, 1, 2]]
    sig_y0 = 200.0
    jc_params = {
        "A": sig_y0,
        "B": 400.0,
        "n": 0.5,
        "sig_max": 1e9,
        "c": 0.0,
        "eps_dot_0": 1.0,
    }
    model, group, log = _make_sh3n_model(coords, conn, thick=0.1, E=200000.0, nu=0.0,
                                         law=2, jc_params=jc_params, nip=1)

    v = np.zeros((3, 3))
    vr = np.zeros((3, 3))
    v[1, 0] = 50.0  # high stretch
    dt = 1e-4

    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    shell_tri3.forces(group, model.x, v, vr, dt, fint, mint)

    sig = group.state["sig"][0, 0]
    epsp = group.state["epsp"][0, 0]

    assert epsp > 0.0
    expected_flow_stress = sig_y0 + 400.0 * (epsp ** 0.5)
    assert np.isclose(sig[0], expected_flow_stress, rtol=1e-3)


def test_sh3n_law0_void_behavior():
    """14. Verify LAW0 void material produces zero forces, zero moments, and dt=EP30."""
    coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    conn = [[0, 1, 2]]
    model, group, log = _make_sh3n_model(coords, conn, law=0, rho0=0.0)

    v = np.ones((3, 3)) * 5.0
    vr = np.ones((3, 3)) * 2.0
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    dt_res = shell_tri3.forces(group, model.x, v, vr, 1e-4, fint, mint)

    assert np.allclose(fint, 0.0)
    assert np.allclose(mint, 0.0)
    assert np.isclose(dt_res[0], EP30)

    ke, edofs = shell_tri3.tangent(group, model.x)
    assert np.allclose(ke, 0.0)


def test_sh3n_consistent_mass_matrix():
    """15. Verify (18x18) consistent mass matrix symmetry, translation row sum, and kinetic energy."""
    coords = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    conn = [[0, 1, 2]]
    thick = 0.2
    rho0 = 7.85e-9
    area = 2.0
    total_mass = rho0 * thick * area

    model, group, log = _make_sh3n_model(coords, conn, thick=thick, rho0=rho0)
    me, edofs = shell_tri3.consistent_mass(group)

    M = me[0]
    assert M.shape == (18, 18)
    assert np.allclose(M, M.T)

    for c in range(3):
        row_sum_node0 = np.sum(M[0 + c, [0 + c, 6 + c, 12 + c]])
        assert np.isclose(row_sum_node0, total_mass / 3.0)

    v_rigid = np.zeros(18)
    v_rigid[0] = v_rigid[6] = v_rigid[12] = 10.0
    ke_trans = 0.5 * v_rigid @ M @ v_rigid
    expected_ke = 0.5 * total_mass * 10.0 ** 2
    assert np.isclose(ke_trans, expected_ke)


def test_sh3n_geometric_stiffness_matrix():
    """16. Verify geometric stiffness K_geo symmetry, translational nullspace, and restoring force."""
    coords = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    conn = [[0, 1, 2]]
    model, group, log = _make_sh3n_model(coords, conn, thick=0.1)

    group.state["sig"][0, :, 0] = 10000.0
    group.state["sig"][0, :, 1] = 10000.0

    ke_geo, edofs = shell_tri3.kgeo(group, model.x)
    Kg = ke_geo[0]

    assert Kg.shape == (18, 18)
    assert np.allclose(Kg, Kg.T)

    v_trans = np.zeros(18)
    v_trans[0] = v_trans[6] = v_trans[12] = 1.0
    assert np.allclose(Kg @ v_trans, 0.0, atol=1e-12)

    w_pert = np.zeros(18)
    w_pert[8] = 0.1
    w_pert[14] = -0.1
    energy = w_pert @ Kg @ w_pert
    assert energy > 0.0


def test_sh3n_material_tangent_stiffness_nullspace():
    """17. Verify material tangent stiffness has exactly 6 zero eigenvalues (rigid modes) and rank 12."""
    coords = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 2.0, 0.0]]
    conn = [[0, 1, 2]]
    model, group, log = _make_sh3n_model(coords, conn, thick=0.1, E=210000.0, nu=0.3)

    ke, edofs = shell_tri3.tangent(group, model.x)
    K = ke[0]

    assert K.shape == (18, 18)
    assert np.allclose(K, K.T, atol=1e-8)

    eigvals = np.linalg.eigvalsh(K)
    zero_eigs = np.sum(np.abs(eigvals) < 1e-4)
    positive_eigs = np.sum(eigvals >= 1e-4)

    assert zero_eigs == 6
    assert positive_eigs == 12


def test_sh3n_implicit_tangent_directional_derivative_consistency():
    """18. Verify directional derivative consistency between material tangent and static internal forces."""
    coords = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    conn = [[0, 1, 2]]
    model, group, log = _make_sh3n_model(coords, conn, thick=0.1, E=10000.0, nu=0.25)

    ke, edofs = shell_tri3.tangent(group, model.x)
    K = ke[0]

    delta_u = np.zeros(18)
    delta_u[6] = 1e-6  # node 1 dx
    delta_u[13] = 2e-6 # node 2 dy

    df_tangent = K @ delta_u

    u_vec = np.zeros((3, 3))
    u_vec[1, 0] = delta_u[6]
    u_vec[2, 1] = delta_u[13]

    fint_0 = np.zeros((3, 3))
    mint_0 = np.zeros((3, 3))
    shell_tri3.forces(group, model.x, u_vec, np.zeros((3, 3)), 1.0, fint_0, mint_0)

    f_flat = np.zeros(18)
    for i in range(3):
        f_flat[i * 6:i * 6 + 3] = -fint_0[i]
        f_flat[i * 6 + 3:i * 6 + 6] = -mint_0[i]

    assert np.allclose(df_tangent[:6], f_flat[:6], rtol=1e-3, atol=1e-8)


def test_sh3n_static_internal_forces():
    """19. Verify static internal forces evaluation on deformed configuration."""
    coords = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    conn = [[0, 1, 2]]
    model, group, log = _make_sh3n_model(coords, conn, thick=0.1)

    group.state["sig"][0, :, 0] = 50.0
    group.state["sig"][0, :, 1] = 25.0

    fint_static = np.zeros((3, 3))
    mint_static = np.zeros((3, 3))
    shell_tri3.static_internal_forces(group, model.x, None, None, fint_static, mint_static)

    assert not np.allclose(fint_static, 0.0)
    assert np.allclose(np.sum(fint_static, axis=0), 0.0, atol=1e-12)


def test_sh3n_defensive_edge_cases_and_starter_parsing(tmp_path):
    """20. Verify defensive edge cases (empty groups, dt<=0, None arrays) and starter keyword parsing."""
    coords = np.zeros((0, 3))
    conn = np.zeros((0, 3), dtype=int)
    model, group, log = _make_sh3n_model(coords, conn)

    node_idx, mass_c, inertia_c = shell_tri3.init_group(group, model, log)
    assert len(node_idx) == 0 and len(mass_c) == 0

    dt_empty = shell_tri3.forces(group, model.x, None, None, 1e-4, None, None)
    assert len(dt_empty) == 0

    ke_empty, edofs_empty = shell_tri3.tangent(group, model.x)
    assert ke_empty.shape == (0, 18, 18)

    me_empty, _ = shell_tri3.consistent_mass(group)
    assert me_empty.shape == (0, 18, 18)

    kg_empty, _ = shell_tri3.kgeo(group, model.x)
    assert kg_empty.shape == (0, 18, 18)

    shell_tri3.static_internal_forces(group, model.x, None, None, None, None)
    shell_tri3.implicit_internal_forces(group, model.x, np.zeros((0, 3)), np.zeros((0, 3)), None, None, False)

    coords_1 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    conn_1 = [[0, 1, 2]]
    model_1, group_1, log_1 = _make_sh3n_model(coords_1, conn_1)
    dt_neg = shell_tri3.forces(group_1, model_1.x, None, None, -1.0, None, None)
    assert np.isclose(dt_neg[0], EP30)

    deck_text = """/BEGIN
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 1.0 0.0
/PART/1
ShellPart
1 1
/PROP/SHELL/1
ShellProp
1.5 3 0.01
/MAT/PLAS_JOHNS/1
Steel
7.85e-9 210000.0 0.3
200.0 400.0 0.5
/SH3N/1
1 1 2 3
/END
"""
    f_deck = tmp_path / "test_sh3n.rad"
    f_deck.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(f_deck))
    mod = Model()
    log_deck = MessageLog()
    parse_starter_deck(blocks, mod, log_deck)
    build_element_groups(mod, log_deck)
    assert mod.sh3n is not None
    assert mod.sh3n.n == 1
    assert np.isclose(mod.properties[1].params["thick"], 1.5)
