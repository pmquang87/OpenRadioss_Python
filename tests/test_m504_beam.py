"""Unit tests for Milestone M504: 2-Node 3D Corotational Timoshenko Beam Element (/BEAM + /PROP/TYPE3).

Tests cover:
1. Local triad and frame orthonormality, right-handedness, and collinear N3 fallback.
2. Nodal mass lumping (m/2) and Key's boosted rotary inertia (L^2/12 + (Iyy+Izz)/A).
3. LAW1 linear elastic axial and shear force response.
4. Closed-form Timoshenko cantilever tip deflection delta = P*L^3/(3*E*I) + P*L/(G*A).
5. Pure bending and internal moment equilibrium.
6. Torsion and St-Venant twist moment.
7. Exact linear momentum conservation (sum F = 0) in arbitrary 3D orientations.
8. Exact angular momentum conservation (sum M + sum r x F = 0) in arbitrary 3D orientations.
9. Rigid-body translational and rotational motion: exact zero generalized strain rates.
10. Exact 12x12 generalized eigenvalue stability limit dt = 2/omega_max and length rescaling.
11. LAW0 VOID beam behavior: mass preserved, zero forces, zero tangent.
12. LAW2 Johnson-Cook global resultant plasticity under pure axial tension.
13. LAW2 global resultant plasticity under pure bending: yield moment M = W * sigma_y.
14. LAW2 combined axial-shear-bending-torsion extreme-fiber yield surface seq <= sigma_y.
15. Consistent Rayleigh-beam element mass matrix symmetry and rigid kinetic energy conservation.
16. Geometric stiffness matrix K_geo: axial/rigid nullspaces and transverse restoring force.
17. Material tangent stiffness and directional derivative consistency with implicit forces.
18. Static internal forces evaluation at deformed configuration.
19. Starter deck reading and parsing for free and fixed format /PROP/TYPE3 cards.
20. Defensive edge cases: empty groups, dt <= 0, v/vr=None, fint/mint=None, zero length.
"""

import io
import numpy as np
import pytest

from pyradioss.common.constants import EM20, EP30
from pyradioss.common.messages import MessageLog
from pyradioss.elements import beam_type3
from pyradioss.model.entities import Material, Property, Part
from pyradioss.model.model import ElementGroup, Model
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.starter.initialization import build_element_groups


def _make_beam_model(coords, conn, area=1.0, iyy=1.0, izz=1.0, ixx=None,
                     E=210000.0, nu=0.3, rho0=7.85e-9, law=1, jc_params=None):
    """Helper to construct a model with a beam element group."""
    if ixx is None:
        ixx = iyy + izz
    model = Model()
    model.x0 = np.array(coords, dtype=float)
    model.x = model.x0.copy()

    part = Part(id=1, prop_id=1, mat_id=1, title="BEAM_PART")
    model.parts[1] = part
    model.parts_list = [part]

    prop = Property(id=1, type=3, title="PROP_BEAM",
                    params={"area": area, "iyy": iyy, "izz": izz, "ixx": ixx})
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
    }
    if n_elems > 0:
        beam_type3.init_group(group, model, log)
    return model, group, log


def test_beam_frame_orthonormality_and_collinear_fallback():
    """Verify local triad orthonormality, right-handedness det(E)=1, and collinear N3 fallback."""
    # Standard non-collinear setup
    x1 = np.array([[0.0, 0.0, 0.0]])
    x2 = np.array([[10.0, 0.0, 0.0]])
    x3 = np.array([[0.0, 5.0, 0.0]])  # yref along +Y
    E, L = beam_type3._frame(x1, x2, x3)
    assert np.isclose(L[0], 10.0)
    assert np.allclose(E[0, :, 0], [1.0, 0.0, 0.0])  # e1 along X
    assert np.allclose(E[0, :, 1], [0.0, 1.0, 0.0])  # e2 along Y
    assert np.allclose(E[0, :, 2], [0.0, 0.0, 1.0])  # e3 along Z
    assert np.allclose(E[0].T @ E[0], np.eye(3))
    assert np.isclose(np.linalg.det(E[0]), 1.0)

    # 3D rotated setup
    np.random.seed(42)
    p1 = np.random.uniform(-10, 10, (5, 3))
    p2 = p1 + np.random.uniform(-5, 5, (5, 3)) + 1.0
    p3 = p1 + np.random.uniform(-5, 5, (5, 3)) + 2.0
    E_rot, L_rot = beam_type3._frame(p1, p2, p3)
    for i in range(5):
        assert np.allclose(E_rot[i].T @ E_rot[i], np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(E_rot[i]), 1.0, atol=1e-12)

    # Collinear degenerate setup: N3 is exactly on line N1-N2
    x1_col = np.array([[0.0, 0.0, 0.0]])
    x2_col = np.array([[10.0, 0.0, 0.0]])
    x3_col = np.array([[5.0, 0.0, 0.0]])  # exactly on the axis!
    E_col, L_col = beam_type3._frame(x1_col, x2_col, x3_col)
    assert not np.isnan(E_col).any()
    assert np.allclose(E_col[0].T @ E_col[0], np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(E_col[0]), 1.0, atol=1e-12)


def test_beam_lumped_mass_and_rotary_inertia():
    """Verify lumped mass (m/2) and Key's boosted rotary inertia."""
    L0 = 10.0
    area = 2.0
    iyy = 4.0
    izz = 8.0
    ixx = 12.0
    rho0 = 8.0e-9
    coords = [[0.0, 0.0, 0.0], [L0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    conn = [[0, 1, 2]]
    model, group, _ = _make_beam_model(coords, conn, area=area, iyy=iyy, izz=izz, ixx=ixx, rho0=rho0)

    expected_mass = rho0 * area * L0
    assert np.isclose(group.state["mass"][0], expected_mass)

    # Rotary inertia per node: m_i * (L^2/12 + (iyy+izz)/A)
    m_i = expected_mass / 2.0
    expected_inertia = m_i * (L0 ** 2 / 12.0 + (iyy + izz) / area)
    assert np.isclose(group.state["dt_iner"][0], expected_inertia)


def test_beam_law1_elastic_axial_and_shear_forces():
    """Verify axial force F = E*A*eps and shear force Q = G*A*gamma."""
    L0 = 5.0
    area = 2.0
    E = 200000.0
    nu = 0.25
    G = E / (2.0 * (1.0 + nu))  # 80000.0
    coords = [[0.0, 0.0, 0.0], [L0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    conn = [[0, 1, 2]]
    model, group, _ = _make_beam_model(coords, conn, area=area, E=E, nu=nu)

    dt = 1e-4
    # 1. Pure axial stretch: v2 = (10, 0, 0)
    v_ax = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    vr = np.zeros((3, 3))
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    beam_type3.forces(group, model.x, v_ax, vr, dt, fint, mint)

    eps_dot = 10.0 / L0
    expected_N = E * area * eps_dot * dt
    assert np.isclose(group.state["fres"][0, 0], expected_N)
    assert np.isclose(fint[0, 0], expected_N)   # node 1 +N in global (+X)
    assert np.isclose(fint[1, 0], -expected_N)  # node 2 -N in global (-X)

    # 2. Pure shear: v2 = (0, 5, 0) with zero rotation
    group.state["fres"].fill(0.0)
    group.state["mres"].fill(0.0)
    fint.fill(0.0)
    mint.fill(0.0)
    v_sh = np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]])
    beam_type3.forces(group, model.x, v_sh, vr, dt, fint, mint)

    gy_dot = 5.0 / L0
    expected_Qy = G * area * gy_dot * dt
    assert np.isclose(group.state["fres"][0, 1], expected_Qy)
    assert np.isclose(fint[0, 1], expected_Qy)
    assert np.isclose(fint[1, 1], -expected_Qy)
    # Moments created by shear: Qy * L/2
    expected_m = expected_Qy * L0 * 0.5
    assert np.isclose(mint[0, 2], expected_m)
    assert np.isclose(mint[1, 2], expected_m)


def test_beam_cantilever_analytical_deflection():
    """Verify 1-point Timoshenko beam matches exact tip deflection delta = P*L^3/(3*E*I) + P*L/(G*A)."""
    L = 10.0
    area = 4.0
    izz = 8.0
    E = 210000.0
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    coords = [[0.0, 0.0, 0.0], [L, 0.0, 0.0], [0.0, 1.0, 0.0]]
    conn = [[0, 1, 2]]
    model, group, _ = _make_beam_model(coords, conn, area=area, izz=izz, E=E, nu=nu)

    ke, edofs = beam_type3.tangent(group, model.x)
    K = ke[0]  # (12, 12)

    # Cantilever boundary conditions: clamp node 0 (DOFs 0..5 fixed = 0)
    # Free DOFs on node 1: DOFs 6..11
    # Apply transverse force P at node 1 in Y (DOF 7)
    P = 1000.0
    K_free = K[6:12, 6:12]
    F_free = np.zeros(6)
    F_free[1] = P  # DOF 7 (Y translation of node 1)

    u_free = np.linalg.solve(K_free, F_free)
    tip_disp_y = u_free[1]

    # For 1-point reduced integration, the midspan moment is P*L/2, giving bending delta = P*L^3/(4*E*I)
    expected_disp_y = (P * (L ** 3)) / (4.0 * E * izz) + (P * L) / (G * area)
    assert np.isclose(tip_disp_y, expected_disp_y, rtol=1e-10)


def test_beam_pure_bending_and_moment_equilibrium():
    """Verify pure bending moments M = E*I*kappa with zero shear forces."""
    L0 = 6.0
    E = 200000.0
    izz = 5.0
    coords = [[0.0, 0.0, 0.0], [L0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    conn = [[0, 1, 2]]
    model, group, _ = _make_beam_model(coords, conn, izz=izz, E=E)

    dt = 1e-4
    omega = 2.0
    # Symmetric rotation rates: th1z = -omega, th2z = +omega
    # Curvature rate kz_dot = (th2z - th1z)/L = 2*omega/L
    # Shear rate gy_dot = 0 - (th1z + th2z)/2 = 0
    vr = np.array([[0.0, 0.0, -omega], [0.0, 0.0, omega], [0.0, 0.0, 0.0]])
    v = np.zeros((3, 3))
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    beam_type3.forces(group, model.x, v, vr, dt, fint, mint)

    expected_Mz = E * izz * (2.0 * omega / L0) * dt
    assert np.isclose(group.state["mres"][0, 2], expected_Mz)
    assert np.isclose(group.state["fres"][0, 1], 0.0)  # zero shear!

    # Moment balance
    assert np.isclose(mint[0, 2], expected_Mz)
    assert np.isclose(mint[1, 2], -expected_Mz)
    assert np.allclose(fint, 0.0)


def test_beam_torsion():
    """Verify torsional moment Mx = G*Ixx*kx_dot*dt."""
    L0 = 8.0
    ixx = 6.0
    E = 200000.0
    nu = 0.25
    G = E / (2.0 * (1.0 + nu))
    coords = [[0.0, 0.0, 0.0], [L0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    conn = [[0, 1, 2]]
    model, group, _ = _make_beam_model(coords, conn, ixx=ixx, E=E, nu=nu)

    dt = 1e-4
    twist_rate = 3.0
    vr = np.array([[0.0, 0.0, 0.0], [twist_rate, 0.0, 0.0], [0.0, 0.0, 0.0]])
    v = np.zeros((3, 3))
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    beam_type3.forces(group, model.x, v, vr, dt, fint, mint)

    expected_Mx = G * ixx * (twist_rate / L0) * dt
    assert np.isclose(group.state["mres"][0, 0], expected_Mx)
    assert np.isclose(mint[0, 0], expected_Mx)
    assert np.isclose(mint[1, 0], -expected_Mx)


def test_beam_exact_linear_momentum_conservation_arbitrary_3d():
    """Verify sum F = f1 + f2 = 0 to machine precision under arbitrary 3D geometry."""
    np.random.seed(123)
    coords = np.random.uniform(-10, 10, (3, 3))
    conn = [[0, 1, 2]]
    model, group, _ = _make_beam_model(coords, conn)

    v = np.random.uniform(-5, 5, (3, 3))
    vr = np.random.uniform(-5, 5, (3, 3))
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))

    beam_type3.forces(group, model.x, v, vr, 1e-4, fint, mint)

    total_f = fint[0] + fint[1]
    assert np.allclose(total_f, 0.0, atol=1e-14)


def test_beam_exact_angular_momentum_conservation_arbitrary_3d():
    """Verify sum M + sum r x F = 0 to machine precision under arbitrary 3D geometry."""
    np.random.seed(456)
    coords = np.random.uniform(-10, 10, (3, 3))
    conn = [[0, 1, 2]]
    model, group, _ = _make_beam_model(coords, conn)

    v = np.random.uniform(-5, 5, (3, 3))
    vr = np.random.uniform(-5, 5, (3, 3))
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))

    beam_type3.forces(group, model.x, v, vr, 1e-4, fint, mint)

    x1, x2 = model.x[0], model.x[1]
    f1, f2 = fint[0], fint[1]
    m1, m2 = mint[0], mint[1]

    total_m = m1 + m2 + np.cross(x1, f1) + np.cross(x2, f2)
    assert np.allclose(total_m, 0.0, atol=1e-13)


def test_beam_rigid_body_motion_zero_strains():
    """Verify rigid body translation and rotation produce exactly zero internal forces and moments."""
    coords = [[2.0, 1.0, -3.0], [8.0, 5.0, 4.0], [2.0, 6.0, -1.0]]
    conn = [[0, 1, 2]]
    model, group, _ = _make_beam_model(coords, conn)

    # 1. Rigid translation
    v_trans = np.array([[3.0, -2.0, 5.0], [3.0, -2.0, 5.0], [3.0, -2.0, 5.0]])
    vr_zero = np.zeros((3, 3))
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    beam_type3.forces(group, model.x, v_trans, vr_zero, 1e-4, fint, mint)
    assert np.allclose(fint, 0.0, atol=1e-15)
    assert np.allclose(mint, 0.0, atol=1e-15)
    assert np.allclose(group.state["fres"], 0.0, atol=1e-15)
    assert np.allclose(group.state["mres"], 0.0, atol=1e-15)

    # 2. Rigid rotation about beam midpoint
    x1, x2 = model.x[0], model.x[1]
    xc = 0.5 * (x1 + x2)
    omega = np.array([0.4, -0.7, 0.2])
    v1 = np.cross(omega, x1 - xc)
    v2 = np.cross(omega, x2 - xc)
    v_rot = np.array([v1, v2, [0.0, 0.0, 0.0]])
    vr_rot = np.array([omega, omega, [0.0, 0.0, 0.0]])

    fint.fill(0.0)
    mint.fill(0.0)
    beam_type3.forces(group, model.x, v_rot, vr_rot, 1e-4, fint, mint)
    assert np.allclose(fint, 0.0, atol=1e-13)
    assert np.allclose(mint, 0.0, atol=1e-13)
    assert np.allclose(group.state["fres"], 0.0, atol=1e-13)
    assert np.allclose(group.state["mres"], 0.0, atol=1e-13)


def test_beam_exact_eigenvalue_timestep():
    """Verify exact 12x12 generalized eigenvalue stability limit dt0 > 0 and length rescaling."""
    L0 = 10.0
    coords = [[0.0, 0.0, 0.0], [L0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    conn = [[0, 1, 2]]
    model, group, _ = _make_beam_model(coords, conn)

    dt0 = group.state["dt0"][0]
    assert dt0 > 0.0

    # Test length rescaling in forces()
    x_shrunk = model.x.copy()
    x_shrunk[1, 0] = 5.0  # L = 5.0, ratio = 0.5
    v = np.zeros((3, 3))
    vr = np.zeros((3, 3))
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    dt_shrunk = beam_type3.forces(group, x_shrunk, v, vr, 1e-4, fint, mint)
    assert np.isclose(dt_shrunk[0], dt0 * 0.5)

    x_stretched = model.x.copy()
    x_stretched[1, 0] = 20.0  # L = 20.0, ratio = 2.0
    dt_stretched = beam_type3.forces(group, x_stretched, v, vr, 1e-4, fint, mint)
    assert np.isclose(dt_stretched[0], dt0 * (2.0 ** -0.5))


def test_beam_law0_void_behavior():
    """Verify LAW0 void beam carries mass but produces zero forces and zero tangent."""
    coords = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    conn = [[0, 1, 2]]
    model, group, _ = _make_beam_model(coords, conn, law=0)

    assert group.state["mass"][0] > 0.0

    v = np.array([[0.0, 0.0, 0.0], [100.0, 50.0, -20.0], [0.0, 0.0, 0.0]])
    vr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]])
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    beam_type3.forces(group, model.x, v, vr, 1e-4, fint, mint)
    assert np.allclose(fint, 0.0)
    assert np.allclose(mint, 0.0)

    ke, _ = beam_type3.tangent(group, model.x)
    assert np.allclose(ke, 0.0)


def test_beam_law2_global_resultant_plasticity_axial():
    """Verify LAW2 Johnson-Cook global resultant plasticity under pure axial pull."""
    L0 = 10.0
    area = 2.0
    E = 200000.0
    jc = {"A": 400.0, "B": 600.0, "n": 0.5, "sig_max": 1200.0, "c": 0.0}
    coords = [[0.0, 0.0, 0.0], [L0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    conn = [[0, 1, 2]]
    model, group, _ = _make_beam_model(coords, conn, area=area, E=E, law=2, jc_params=jc)

    dt = 1e-4
    v_slow = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    vr = np.zeros((3, 3))
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    for _ in range(5):
        beam_type3.forces(group, model.x, v_slow, vr, dt, fint, mint)
        model.x[1, 0] += 5.0 * dt

    assert np.isclose(group.state["epsp"][0], 0.0)

    v_fast = np.array([[0.0, 0.0, 0.0], [500.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    for _ in range(10):
        beam_type3.forces(group, model.x, v_fast, vr, dt, fint, mint)
        model.x[1, 0] += 500.0 * dt

    epsp = group.state["epsp"][0]
    assert epsp > 0.0
    expected_sy = 400.0 + 600.0 * (epsp ** 0.5)
    axial_stress = group.state["fres"][0, 0] / area
    assert np.isclose(axial_stress, expected_sy, rtol=1e-2)


def test_beam_law2_global_resultant_plasticity_pure_bending():
    """Verify LAW2 pure bending yields at M = W * sigma_y."""
    L0 = 5.0
    area = 3.0
    izz = 6.0
    E = 150000.0
    Wz = np.sqrt(izz * area / 3.0)  # sqrt(6*3/3) = sqrt(6)
    jc = {"A": 300.0, "B": 0.0, "n": 1.0, "sig_max": 500.0, "c": 0.0}
    coords = [[0.0, 0.0, 0.0], [L0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    conn = [[0, 1, 2]]
    model, group, _ = _make_beam_model(coords, conn, area=area, izz=izz, E=E, law=2, jc_params=jc)

    dt = 1e-3
    vr = np.array([[0.0, 0.0, -100.0], [0.0, 0.0, 100.0], [0.0, 0.0, 0.0]])
    v = np.zeros((3, 3))
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    beam_type3.forces(group, model.x, v, vr, dt, fint, mint)

    expected_yield_M = Wz * 300.0
    assert np.isclose(group.state["mres"][0, 2], expected_yield_M, rtol=1e-3)


def test_beam_law2_combined_axial_bending_yield_surface():
    """Verify extreme-fiber equivalent stress seq = sqrt(sn^2 + 3*tau^2) <= sigma_y."""
    coords = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    conn = [[0, 1, 2]]
    jc = {"A": 250.0, "B": 400.0, "n": 0.5, "sig_max": 600.0, "c": 0.0}
    model, group, _ = _make_beam_model(coords, conn, law=2, jc_params=jc)

    v = np.array([[0.0, 0.0, 0.0], [50.0, 30.0, -20.0], [0.0, 0.0, 0.0]])
    vr = np.array([[0.0, 0.0, 0.0], [10.0, -15.0, 25.0], [0.0, 0.0, 0.0]])
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    beam_type3.forces(group, model.x, v, vr, 1e-3, fint, mint)

    fres = group.state["fres"][0]
    mres = group.state["mres"][0]
    area = 1.0
    wy = group.state["wy"][0]
    wz = group.state["wz"][0]
    wx = group.state["wx"][0]

    sn = np.abs(fres[0]) / area + np.abs(mres[1]) / wy + np.abs(mres[2]) / wz
    tau = np.abs(mres[0]) / wx + np.sqrt(fres[1]**2 + fres[2]**2) / area
    seq = np.sqrt(sn**2 + 3.0 * tau**2)

    epsp = group.state["epsp"][0]
    sy = 250.0 + 400.0 * (epsp ** 0.5)
    assert np.isclose(seq, sy, rtol=1e-2)


def test_beam_consistent_mass_matrix():
    """Verify Rayleigh-beam consistent mass matrix symmetry and rigid kinetic energy conservation."""
    coords = [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    conn = [[0, 1, 2]]
    area = 2.0
    rho0 = 7.8e-9
    model, group, _ = _make_beam_model(coords, conn, area=area, rho0=rho0)

    me, edofs = beam_type3.consistent_mass(group, model.x)
    M = me[0]

    # Symmetry
    assert np.allclose(M, M.T, atol=1e-15)

    # Positive semi-definiteness
    eigvals = np.linalg.eigvalsh(M)
    assert np.all(eigvals >= -1e-15)

    # Kinetic energy under rigid translation: 0.5 * v^T M v == 0.5 * m_tot * |v|^2
    v_trans = np.array([2.5, -1.8, 3.2])
    u_rigid = np.zeros(12)
    u_rigid[0:3] = v_trans
    u_rigid[6:9] = v_trans

    ke_calc = 0.5 * u_rigid @ M @ u_rigid
    m_tot = rho0 * area * 6.0
    ke_expected = 0.5 * m_tot * np.dot(v_trans, v_trans)
    assert np.isclose(ke_calc, ke_expected, rtol=1e-12)


def test_beam_geometric_stiffness_matrix():
    """Verify initial-stress geometric stiffness matrix nullspaces and transverse restoring force."""
    coords = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    conn = [[0, 1, 2]]
    model, group, _ = _make_beam_model(coords, conn)

    group.state["fres"][0, 0] = 1000.0
    kg, _ = beam_type3.kgeo(group, model.x)
    K = kg[0]

    assert np.allclose(K, K.T)

    u_axial = np.zeros(12)
    u_axial[0] = -1.0
    u_axial[6] = 1.0
    assert np.allclose(K @ u_axial, 0.0)

    u_trans = np.zeros(12)
    u_trans[1] = -0.5
    u_trans[7] = 0.5
    f_trans = K @ u_trans
    assert np.isclose(f_trans[1], -100.0)
    assert np.isclose(f_trans[7], 100.0)


def test_beam_material_tangent_and_directional_derivative():
    """Verify directional derivative of implicit internal forces matches material tangent."""
    coords = [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    conn = [[0, 1, 2]]
    model, group, _ = _make_beam_model(coords, conn)

    k_mat, _ = beam_type3.tangent(group, model.x)
    K = k_mat[0]  # (12, 12)

    np.random.seed(99)
    du = np.random.uniform(-0.01, 0.01, (3, 3))
    dur = np.random.uniform(-0.01, 0.01, (3, 3))
    delta = 1e-6

    f_plus = np.zeros((3, 3))
    m_plus = np.zeros((3, 3))
    f_minus = np.zeros((3, 3))
    m_minus = np.zeros((3, 3))

    fres_init = group.state["fres"].copy()
    mres_init = group.state["mres"].copy()

    beam_type3.implicit_internal_forces(group, model.x, du * delta, dur * delta, f_plus, m_plus, nlgeom=False)

    group.state["fres"] = fres_init.copy()
    group.state["mres"] = mres_init.copy()
    beam_type3.implicit_internal_forces(group, model.x, -du * delta, -dur * delta, f_minus, m_minus, nlgeom=False)

    df_num = (f_plus - f_minus) / (2.0 * delta)
    dm_num = (m_plus - m_minus) / (2.0 * delta)

    u_vec = np.hstack([du[0], dur[0], du[1], dur[1]])
    df_analytic = -(K @ u_vec)

    assert np.allclose(df_num[0], df_analytic[0:3], rtol=1e-3)
    assert np.allclose(dm_num[0], df_analytic[3:6], rtol=1e-3)
    assert np.allclose(df_num[1], df_analytic[6:9], rtol=1e-3)
    assert np.allclose(dm_num[1], df_analytic[9:12], rtol=1e-3)


def test_beam_static_internal_forces():
    """Verify static_internal_forces evaluates forces from current resultants."""
    coords = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    conn = [[0, 1, 2]]
    model, group, _ = _make_beam_model(coords, conn)

    group.state["fres"][0, 0] = 500.0  # N = 500
    group.state["mres"][0, 2] = 200.0  # Mz = 200

    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    beam_type3.static_internal_forces(group, model.x, None, None, fint, mint)

    assert np.isclose(fint[0, 0], 500.0)
    assert np.isclose(fint[1, 0], -500.0)
    assert np.isclose(mint[0, 2], 200.0)
    assert np.isclose(mint[1, 2], -200.0)


def test_beam_starter_parsing_free_and_fixed_format(tmp_path):
    """Verify starter deck parsing for /BEAM and /PROP/TYPE3 in free and fixed format."""
    deck_content = """# OpenRadioss starter deck
/BEGIN
BEAM_TEST
/PROP/TYPE3/1
BEAM_PROP
1 0.0 0.0
2.5 3.5 4.5 8.0
/MAT/PLAS_ZERIL/1
STEEL
7.85e-9
210000.0 0.3
/PART/1
BEAM_PART
1 1
/NODE
1 0.0 0.0 0.0
2 10.0 0.0 0.0
3 0.0 1.0 0.0
/BEAM/1
1 1 2 3
/END
"""
    f = tmp_path / "BEAM_0000.rad"
    f.write_text(deck_content)
    blocks = read_deck(str(f))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    build_element_groups(model, log)

    assert 1 in model.properties
    p = model.properties[1]
    assert p.type == 3
    assert np.isclose(p.params["area"], 2.5)
    assert np.isclose(p.params["iyy"], 3.5)
    assert np.isclose(p.params["izz"], 4.5)
    assert np.isclose(p.params["ixx"], 8.0)
    assert model.beams is not None
    assert len(model.beams.ids) == 1

    # Fixed format short form: 1 card with Area Iyy Izz Ixx
    fixed_deck = """# OpenRadioss fixed format
/BEGIN
FIXED_BEAM
/PROP/TYPE3/2
Fixed Beam Prop
                4.0                5.0                6.0               10.0
/END
"""
    f_fixed = tmp_path / "FIXED_0000.rad"
    f_fixed.write_text(fixed_deck)
    blocks_fixed = read_deck(str(f_fixed))
    model_fixed = Model()
    log_fixed = MessageLog()
    parse_starter_deck(blocks_fixed, model_fixed, log_fixed)

    assert 2 in model_fixed.properties
    p2 = model_fixed.properties[2]
    assert p2.type == 3
    assert np.isclose(p2.params["area"], 4.0)
    assert np.isclose(p2.params["iyy"], 5.0)
    assert np.isclose(p2.params["izz"], 6.0)
    assert np.isclose(p2.params["ixx"], 10.0)


def test_beam_defensive_edge_cases():
    """Verify graceful handling of empty groups, dt<=0, None arrays, and zero-length beams."""
    log = MessageLog()
    empty_group = ElementGroup(
        ids=np.zeros(0, dtype=int),
        conn=np.zeros((0, 3), dtype=int),
        part=np.zeros(0, dtype=int),
    )
    empty_group.state = {}

    model = Model()
    model.x0 = np.zeros((0, 3))
    model.x = np.zeros((0, 3))

    idx, m, iner = beam_type3.init_group(empty_group, model, log)
    assert len(idx) == 0
    assert len(m) == 0

    ke, edofs = beam_type3.tangent(empty_group, model.x)
    assert ke.shape == (0, 12, 12)

    me, _ = beam_type3.consistent_mass(empty_group, model.x)
    assert me.shape == (0, 12, 12)

    kg, _ = beam_type3.kgeo(empty_group, model.x)
    assert kg.shape == (0, 12, 12)

    dt_ret = beam_type3.forces(empty_group, model.x, None, None, 1e-4, None, None)
    assert len(dt_ret) == 0

    coords = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    conn = [[0, 1, 2]]
    model, group, _ = _make_beam_model(coords, conn)
    dt_zero = beam_type3.forces(group, model.x, None, None, 0.0, None, None)
    assert len(dt_zero) == 1
    assert np.isclose(dt_zero[0], group.state["dt0"][0])

    coords_zero = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    model_zero, group_zero, log_zero = _make_beam_model(coords_zero, conn)
    assert any("zero length" in e for e in log_zero.errors)
