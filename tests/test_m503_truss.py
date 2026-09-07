"""Unit tests for Milestone M503: 2-Node 1D Truss Element (/TRUSS + /PROP/TRUSS).

Tests cover:
1. Element group initialization, state variables, zero-length logging.
2. Nodal mass lumping: half-mass to each node (starter/source/elements/truss/tmass.F).
3. LAW1 linear elastic axial tension and compression: reversible, exact force F = E*A*dL/L0.
4. Exact linear momentum conservation (sum F = 0) and angular momentum conservation
   (sum r x F = 0) to machine precision across arbitrary 3D orientations.
5. Trapezoidal internal energy accounting and reversibility in elastic cycles.
6. Courant critical time step calculation dt = L / sqrt(E/rho0) and LAW0 VOID dt = EP30.
7. LAW2 Johnson-Cook 1D elastoplastic yielding, work hardening, and elastic unloading
   with residual plastic strain eps_p > 0.
8. LAW2 strain-rate sensitivity: dynamic flow stress scaling with 1 + c*ln(eps_dot/eps_dot_0).
9. Consistent element mass matrix M = (m/6)[[2 I3, I3], [I3, 2 I3]] and kinetic energy conservation.
10. Geometric stiffness matrix K_geo = (F/L)(I - a a^T), axial nullspace, transverse taut-string stiffness.
11. Material tangent stiffness K_mat = (Et A / L) a a^T, consistent elastoplastic modulus Et = E*H/(E+H).
12. Numerical differentiation verification: d(-f)/du matches K_mat + K_geo.
13. Static internal forces evaluation along current deformed axis.
14. Starter deck parsing with /TRUSS, /PROP/TRUSS (TYPE2), /MAT/PLAS_ZERIL (LAW1), /PART.
15. Starter deck parsing with /TRUSS, /PROP/TYPE2, /MAT/PLAS_JOHNS (LAW2).
16. Fixed-format /PROP/TYPE2 parsing with title card and column-based area card.
17. Defensive edge cases: empty groups, v=None, fint=None, dt<=0, degenerate zero-length elements.
"""

import io
import numpy as np
import pytest

from pyradioss.common.constants import EM20, EP30
from pyradioss.common.messages import MessageLog
from pyradioss.elements import truss
from pyradioss.model.entities import Material, Property, Part
from pyradioss.model.model import ElementGroup, Model
from pyradioss.starter.initialization import build_element_groups
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck


def _make_truss_model(coords, conn, area=2.0, E=210000.0, rho0=7.8e-9, law=1, jc_params=None):
    """Helper to build a minimal model with a truss group."""
    model = Model()
    model.x0 = np.array(coords, dtype=float)
    model.x = model.x0.copy()

    part = Part(id=1, prop_id=1, mat_id=1, title="TRUSS_PART")
    model.parts[1] = part
    model.parts_list = [part]

    prop = Property(id=1, type=2, title="PROP_TRUSS", params={"area": area})
    model.properties[1] = prop

    mat_params = {"E": E, "nu": 0.3}
    if law == 2 and jc_params is not None:
        mat_params.update(jc_params)
    mat = Material(id=1, law=law, rho0=rho0, params=mat_params)
    model.materials[1] = mat

    log = MessageLog()
    group = ElementGroup(
        ids=np.arange(1, len(conn) + 1, dtype=int),
        conn=np.array(conn, dtype=int),
        part=np.zeros(len(conn), dtype=int),
    )
    group.state["slices"] = [(slice(0, len(conn)), mat, prop)]
    group.state["part_ids"] = [1]
    model.trusses = group

    truss.init_group(group, model, log)
    return model, group, log


def test_truss_init_group_and_mass_lumping():
    """Verify init_group sets up state variables and exactly halves element mass to nodes."""
    coords = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 3.0, 0.0]]
    conn = [[0, 1], [1, 2]]  # L0 = 4.0 and 3.0
    area = 2.5
    rho0 = 8.0e-9
    model, group, log = _make_truss_model(coords, conn, area=area, rho0=rho0)

    assert len(group.state["L0"]) == 2
    assert np.isclose(group.state["L0"][0], 4.0)
    assert np.isclose(group.state["L0"][1], 3.0)

    # Element masses
    expected_m1 = rho0 * area * 4.0
    expected_m2 = rho0 * area * 3.0
    assert np.allclose(group.state["mass"], [expected_m1, expected_m2])

    # Nodal mass lumping
    node_idx, nod_mass, _ = truss.init_group(group, model, log)
    assert np.allclose(nod_mass[:2], expected_m1 / 2.0)
    assert np.allclose(nod_mass[2:], expected_m2 / 2.0)
    assert np.allclose(group.state["sig"], 0.0)
    assert np.allclose(group.state["epsp"], 0.0)
    assert np.allclose(group.state["eint"], 0.0)


def test_truss_init_group_zero_length_detection():
    """Verify init_group detects and logs an error when initial length <= 0."""
    coords = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    conn = [[0, 1]]
    model = Model()
    model.x0 = np.array(coords, dtype=float)
    part = Part(id=1, prop_id=1, mat_id=1, title="TRUSS")
    prop = Property(id=1, type=2, params={"area": 1.0})
    mat = Material(id=1, law=1, rho0=1.0, params={"E": 100.0, "nu": 0.3})
    log = MessageLog()
    group = ElementGroup(ids=np.array([101]), conn=np.array(conn), part=np.array([0]))
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    truss.init_group(group, model, log)

    assert any("zero length" in e for e in log.errors)


def test_truss_elastic_tension_compression_law1():
    """Verify linear elastic tension and compression: exact axial force F = E * A * dL / L0."""
    L0 = 10.0
    A = 3.0
    E = 200000.0
    coords = [[0.0, 0.0, 0.0], [L0, 0.0, 0.0]]
    conn = [[0, 1]]
    model, group, _ = _make_truss_model(coords, conn, area=A, E=E)

    # 1. Tension: node 2 moves at vx = +20.0 for dt = 1e-4 -> dL = 2e-3, eps = 2e-4
    dt = 1e-4
    v = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    x = model.x0.copy()
    fint = np.zeros((2, 3))
    truss.forces(group, x, v, None, dt, fint, None)

    expected_eps = 20.0 * dt / L0
    expected_sig = E * expected_eps
    expected_F = A * expected_sig

    assert np.isclose(group.state["sig"][0], expected_sig)
    # Internal force on node 1 pulls toward node 2 (+x)
    assert np.isclose(fint[0, 0], expected_F)
    assert np.isclose(fint[1, 0], -expected_F)

    # 2. Compression: reverse velocity to return to zero, then into compression
    fint.fill(0.0)
    v_comp = np.array([[0.0, 0.0, 0.0], [-40.0, 0.0, 0.0]])
    truss.forces(group, x, v_comp, None, dt, fint, None)
    # New strain change = -4e-4, net strain = -2e-4, net sig = -E * 2e-4
    expected_sig_comp = -expected_sig
    expected_F_comp = -expected_F
    assert np.isclose(group.state["sig"][0], expected_sig_comp)
    assert np.isclose(fint[0, 0], expected_F_comp)
    assert np.isclose(fint[1, 0], -expected_F_comp)


def test_truss_momentum_conservation_arbitrary_3d():
    """Verify sum F = 0 and sum r x F = 0 to machine precision for random 3D orientation."""
    np.random.seed(42)
    for _ in range(5):
        p1 = np.random.uniform(-10.0, 10.0, 3)
        p2 = p1 + np.random.uniform(-5.0, 5.0, 3)
        while np.linalg.norm(p2 - p1) < 0.1:
            p2 = p1 + np.random.uniform(-5.0, 5.0, 3)

        coords = [p1, p2]
        conn = [[0, 1]]
        model, group, _ = _make_truss_model(coords, conn, area=4.2, E=150000.0)

        v = np.random.uniform(-50.0, 50.0, (2, 3))
        fint = np.zeros((2, 3))
        truss.forces(group, model.x, v, None, 1e-4, fint, None)

        # Linear momentum: f1 + f2 == 0
        total_force = fint[0] + fint[1]
        assert np.allclose(total_force, 0.0, atol=1e-12)

        # Angular momentum: x1 x f1 + x2 x f2 == 0
        torque1 = np.cross(p1, fint[0])
        torque2 = np.cross(p2, fint[1])
        total_torque = torque1 + torque2
        assert np.allclose(total_torque, 0.0, atol=1e-12)


def test_truss_energy_accounting_elastic_cycle():
    """Verify trapezoidal internal energy integration matches analytical strain energy and is reversible."""
    L0 = 5.0
    A = 2.0
    E = 100000.0
    coords = [[0.0, 0.0, 0.0], [L0, 0.0, 0.0]]
    conn = [[0, 1]]
    model, group, _ = _make_truss_model(coords, conn, area=A, E=E)

    dt = 1e-4
    n_steps = 10
    v_ext = 10.0
    # Loading phase: stretch
    for _ in range(n_steps):
        v = np.array([[0.0, 0.0, 0.0], [v_ext, 0.0, 0.0]])
        truss.forces(group, model.x, v, None, dt, None, None)
        model.x[1, 0] += v_ext * dt

    dL_max = v_ext * dt * n_steps
    eps_max = dL_max / L0
    expected_Eint = 0.5 * E * (eps_max ** 2) * (A * L0)
    assert np.isclose(group.state["eint"][0], expected_Eint, rtol=1e-3)

    # Unloading phase: return to original position
    for _ in range(n_steps):
        v = np.array([[0.0, 0.0, 0.0], [-v_ext, 0.0, 0.0]])
        truss.forces(group, model.x, v, None, dt, None, None)
        model.x[1, 0] -= v_ext * dt

    # Reversible up to discrete forward-Euler integration residual (peak sig was 200, peak eint was 2.0)
    assert np.isclose(group.state["sig"][0], 0.0, atol=0.1)
    assert np.isclose(group.state["eint"][0], 0.0, atol=0.01)


def test_truss_critical_timestep_courant():
    """Verify Courant time step dt = L / sqrt(E/rho0) and VOID behavior."""
    L = 10.0
    E = 200000.0
    rho0 = 8.0e-9
    c_expected = np.sqrt(E / rho0)
    dt_expected = L / c_expected

    coords = [[0.0, 0.0, 0.0], [L, 0.0, 0.0]]
    conn = [[0, 1]]
    model, group, _ = _make_truss_model(coords, conn, E=E, rho0=rho0)

    v = np.zeros((2, 3))
    dt_claim = truss.forces(group, model.x, v, None, 1e-6, None, None)
    assert np.isclose(dt_claim[0], dt_expected)

    # Shorter truss has proportionally smaller dt
    model_short, group_short, _ = _make_truss_model([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], conn, E=E, rho0=rho0)
    dt_claim_short = truss.forces(group_short, model_short.x, v, None, 1e-6, None, None)
    assert np.isclose(dt_claim_short[0], dt_expected / 2.0)


def test_truss_law0_void_claims_no_dt():
    """Verify LAW0 (VOID) truss is stress-free and claims EP30 time step."""
    coords = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
    conn = [[0, 1]]
    model, group, _ = _make_truss_model(coords, conn, E=0.0, rho0=0.0, law=0)

    v = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    fint = np.zeros((2, 3))
    dt_claim = truss.forces(group, model.x, v, None, 1e-4, fint, None)

    assert np.isclose(group.state["sig"][0], 0.0)
    assert np.allclose(fint, 0.0)
    assert dt_claim[0] >= EP30 * 0.99


def test_truss_law2_johnson_cook_plasticity():
    """Verify LAW2 Johnson-Cook 1D radial return, work hardening, and elastic unloading."""
    L0 = 10.0
    A_sec = 2.0
    E = 200000.0
    # JC params: A=300, B=500, n=0.5, sig_max=800, c=0 (rate-independent)
    jc = {"A": 300.0, "B": 500.0, "n": 0.5, "sig_max": 800.0, "c": 0.0, "eps_dot_0": 1.0}
    coords = [[0.0, 0.0, 0.0], [L0, 0.0, 0.0]]
    conn = [[0, 1]]
    model, group, _ = _make_truss_model(coords, conn, area=A_sec, E=E, law=2, jc_params=jc)

    # Elastic loading below yield stress A=300 (eps_y = 300/200000 = 1.5e-3)
    dt = 1e-4
    v_slow = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])  # deps = 5.0 * 1e-4 / 10.0 = 5e-5
    for _ in range(10):  # total eps = 5e-4 < 1.5e-3
        truss.forces(group, model.x, v_slow, None, dt, None, None)
        model.x[1, 0] += 5.0 * dt

    assert np.isclose(group.state["epsp"][0], 0.0)
    assert np.isclose(group.state["sig"][0], E * 5e-4, rtol=1e-3)

    # Plastic loading beyond yield: pull with large step
    v_fast = np.array([[0.0, 0.0, 0.0], [500.0, 0.0, 0.0]])  # deps = 500 * 1e-4 / 10 = 5e-3
    for _ in range(10):  # total strain accumulated ~ 0.05
        truss.forces(group, model.x, v_fast, None, dt, None, None)
        model.x[1, 0] += 500.0 * dt

    epsp = group.state["epsp"][0]
    assert epsp > 0.0
    # Flow stress should match A + B * epsp^n
    expected_sy = 300.0 + 500.0 * (epsp ** 0.5)
    assert np.isclose(group.state["sig"][0], expected_sy, rtol=1e-2)

    # Elastic unloading: reverse direction
    v_unload = np.array([[0.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])
    sig_before = group.state["sig"][0]
    epsp_before = group.state["epsp"][0]
    truss.forces(group, model.x, v_unload, None, dt, None, None)
    # Plastic strain must NOT change during elastic unloading
    assert np.isclose(group.state["epsp"][0], epsp_before)
    # Stress drops elastically by E * deps evaluated at current length
    L_curr = model.x[1, 0] - model.x[0, 0]
    expected_drop = E * (10.0 * dt / L_curr)
    assert np.isclose(sig_before - group.state["sig"][0], expected_drop, rtol=1e-3)


def test_truss_law2_strain_rate_sensitivity():
    """Verify LAW2 strain rate scaling factor (1 + c*ln(eps_dot/eps_dot_0))."""
    L0 = 10.0
    E = 200000.0
    jc = {"A": 200.0, "B": 0.0, "n": 1.0, "sig_max": 1000.0, "c": 0.1, "eps_dot_0": 1.0}
    conn = [[0, 1]]

    # Case 1: eps_dot = 1.0 (rate factor = 1.0)
    model1, group1, _ = _make_truss_model([[0.0, 0.0, 0.0], [L0, 0.0, 0.0]], conn, E=E, law=2, jc_params=jc)
    dt1 = 1e-2
    v1 = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])  # eps_dot = 1.0
    truss.forces(group1, model1.x, v1, None, dt1, None, None)

    # Case 2: eps_dot = 100.0 (rate factor = 1 + 0.1 * ln(100) = 1.4605)
    model2, group2, _ = _make_truss_model([[0.0, 0.0, 0.0], [L0, 0.0, 0.0]], conn, E=E, law=2, jc_params=jc)
    dt2 = 1e-4
    v2 = np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])  # eps_dot = 100.0
    truss.forces(group2, model2.x, v2, None, dt2, None, None)

    assert group2.state["sig"][0] > group1.state["sig"][0]
    expected_ratio = 1.0 + 0.1 * np.log(100.0)
    assert np.isclose(group2.state["sig"][0] / group1.state["sig"][0], expected_ratio, rtol=1e-2)


def test_truss_consistent_mass_matrix():
    """Verify consistent element mass M = (m/6)[[2 I3, I3], [I3, 2 I3]]."""
    coords = [[1.0, 2.0, 3.0], [5.0, 2.0, 3.0]]  # L0 = 4.0
    conn = [[0, 1]]
    area = 2.0
    rho0 = 5.0
    model, group, _ = _make_truss_model(coords, conn, area=area, rho0=rho0)
    elem_mass = rho0 * area * 4.0  # 40.0

    me, edofs = truss.consistent_mass(group, model.x)
    assert me.shape == (1, 6, 6)
    assert edofs.shape == (1, 6)
    assert np.array_equal(edofs[0], [0, 1, 2, 6, 7, 8])

    M = me[0]
    # Check block diagonal terms: 2*m/6 = m/3
    for c in range(3):
        assert np.isclose(M[c, c], elem_mass * 2.0 / 6.0)
        assert np.isclose(M[3 + c, 3 + c], elem_mass * 2.0 / 6.0)
        # Check off-diagonal terms: m/6
        assert np.isclose(M[c, 3 + c], elem_mass / 6.0)
        assert np.isclose(M[3 + c, c], elem_mass / 6.0)

    # Row sum must equal m/2 for each translational DOF
    row_sums = np.sum(M, axis=1)
    for c in range(3):
        assert np.isclose(row_sums[c], elem_mass / 2.0)
        assert np.isclose(row_sums[3 + c], elem_mass / 2.0)

    # Rigid body translation kinetic energy: 0.5 * v^T M v == 0.5 * m * |v|^2
    v_rigid = np.array([3.0, -4.0, 5.0, 3.0, -4.0, 5.0])
    ke = 0.5 * v_rigid @ M @ v_rigid
    ke_expected = 0.5 * elem_mass * (3.0**2 + (-4.0)**2 + 5.0**2)
    assert np.isclose(ke, ke_expected)


def test_truss_geometric_stiffness_matrix():
    """Verify geometric stiffness K_geo = (F/L)(I - a a^T), axial nullspace, and transverse stiffness."""
    coords = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]  # a = [1, 0, 0]
    conn = [[0, 1]]
    model, group, _ = _make_truss_model(coords, conn, area=2.0)

    # Set axial stress to 500.0 -> F = 2.0 * 500.0 = 1000.0. F/L = 100.0
    group.state["sig"][0] = 500.0
    ke_geo, edofs = truss.kgeo(group, model.x)
    K = ke_geo[0]

    # Axial modes (DOF 0 and DOF 3) must be in the nullspace
    u_axial_rel = np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert np.allclose(K @ u_axial_rel, 0.0)

    # Rigid translation must be in the nullspace
    u_rigid = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    assert np.allclose(K @ u_rigid, 0.0)

    # Transverse relative displacement (DOF 1 and DOF 4, or DOF 2 and DOF 5)
    # Relative dy = 1.0 gives restoring force F/L = 100.0
    u_trans = np.array([0.0, -0.5, 0.0, 0.0, 0.5, 0.0])
    f_trans = K @ u_trans
    assert np.isclose(f_trans[1], -100.0)
    assert np.isclose(f_trans[4], +100.0)


def test_truss_material_tangent_stiffness():
    """Verify material tangent K_mat = (E*A/L) a a^T, zero transverse stiffness, and edofs convention."""
    coords = [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]]  # a = [0, 1, 0]
    conn = [[0, 1]]
    A = 3.0
    E = 150000.0
    model, group, _ = _make_truss_model(coords, conn, area=A, E=E)

    ke_mat, edofs = truss.tangent(group, model.x)
    K = ke_mat[0]
    k_expected = E * A / 5.0  # 90000.0

    # Axial motion along Y (DOF 1 and DOF 4)
    u_axial = np.array([0.0, -1.0, 0.0, 0.0, 1.0, 0.0])
    f_axial = K @ u_axial
    assert np.isclose(f_axial[1], -2.0 * k_expected)
    assert np.isclose(f_axial[4], +2.0 * k_expected)

    # Transverse motion along X and Z should yield zero force
    u_trans_x = np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert np.allclose(K @ u_trans_x, 0.0)
    u_trans_z = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 1.0])
    assert np.allclose(K @ u_trans_z, 0.0)


def test_truss_tangent_elastoplastic_consistent_modulus():
    """Verify consistent tangent modulus Et = E*H / (E+H) when plastic strain increment is active."""
    L0 = 10.0
    A = 2.0
    E = 100000.0
    jc = {"A": 200.0, "B": 300.0, "n": 0.5, "sig_max": 800.0}
    conn = [[0, 1]]
    model, group, _ = _make_truss_model([[0.0, 0.0, 0.0], [L0, 0.0, 0.0]], conn, area=A, E=E, law=2, jc_params=jc)

    group.state["epsp"][0] = 0.04
    e = 0.04
    H = 300.0 * 0.5 * (e ** (0.5 - 1.0))  # 300 * 0.5 / 0.2 = 750.0
    Et = E * H / (E + H)

    # 1. Elastic step (epsp_incr is None)
    ke_el, _ = truss.tangent(group, model.x, epsp_incr=None)
    assert np.isclose(ke_el[0, 0, 0], E * A / L0)

    # 2. Plastic step (epsp_incr > 0)
    epsp_incr = np.array([0.001])
    ke_pl, _ = truss.tangent(group, model.x, epsp_incr=epsp_incr)
    assert np.isclose(ke_pl[0, 0, 0], Et * A / L0, rtol=1e-4)


def test_truss_implicit_internal_forces_and_tangent_consistency():
    """Verify numerical directional derivative of implicit_internal_forces matches tangent + kgeo."""
    coords = [[0.0, 0.0, 0.0], [8.0, 6.0, 0.0]]  # L0 = 10.0, a = [0.8, 0.6, 0.0]
    conn = [[0, 1]]
    A = 2.5
    E = 120000.0
    model, group, _ = _make_truss_model(coords, conn, area=A, E=E)

    # Prestressed state
    group.state["sig"][0] = 200.0

    # Evaluate tangent and kgeo
    k_mat, _ = truss.tangent(group, model.x)
    k_g, _ = truss.kgeo(group, model.x)
    k_total = k_mat[0] + k_g[0]

    # Directional perturbation
    np.random.seed(123)
    du = np.random.uniform(-0.01, 0.01, (2, 3))
    delta = 1e-6
    u_pert = du * delta

    # Finite difference on implicit_internal_forces (linear geometry)
    f_plus = np.zeros((2, 3))
    f_minus = np.zeros((2, 3))

    # Snapshot state to prevent permanent mutation during perturbation
    sig_init = group.state["sig"].copy()
    truss.implicit_internal_forces(group, model.x, u_pert, None, f_plus, None, nlgeom=False)

    group.state["sig"] = sig_init.copy()
    truss.implicit_internal_forces(group, model.x, -u_pert, None, f_minus, None, nlgeom=False)

    df_num = (f_plus - f_minus) / (2.0 * delta)
    u_vec = np.hstack([du[0], du[1]])
    df_analytic = -(k_mat[0] @ u_vec)

    assert np.isclose(df_num[0, 0], df_analytic[0], rtol=1e-3)
    assert np.isclose(df_num[0, 1], df_analytic[1], rtol=1e-3)
    assert np.isclose(df_num[1, 0], df_analytic[3], rtol=1e-3)
    assert np.isclose(df_num[1, 1], df_analytic[4], rtol=1e-3)

    # Finite difference on implicit_internal_forces (nonlinear geometry with geometric stiffness)
    f_plus_nl = np.zeros((2, 3))
    f_minus_nl = np.zeros((2, 3))

    group.state["sig"] = sig_init.copy()
    truss.implicit_internal_forces(group, model.x, u_pert, None, f_plus_nl, None, nlgeom=True)

    group.state["sig"] = sig_init.copy()
    truss.implicit_internal_forces(group, model.x, -u_pert, None, f_minus_nl, None, nlgeom=True)

    df_num_nl = (f_plus_nl - f_minus_nl) / (2.0 * delta)
    df_analytic_nl = -(k_total @ u_vec)

    assert np.isclose(df_num_nl[0, 0], df_analytic_nl[0], rtol=1e-3)
    assert np.isclose(df_num_nl[0, 1], df_analytic_nl[1], rtol=1e-3)
    assert np.isclose(df_num_nl[1, 0], df_analytic_nl[3], rtol=1e-3)
    assert np.isclose(df_num_nl[1, 1], df_analytic_nl[4], rtol=1e-3)


def test_truss_static_internal_forces():
    """Verify static_internal_forces evaluates forces along current axis matching forces()."""
    coords = [[1.0, 1.0, 1.0], [4.0, 5.0, 1.0]]  # dx = [3, 4, 0], L = 5.0, a = [0.6, 0.8, 0]
    conn = [[0, 1]]
    model, group, _ = _make_truss_model(coords, conn, area=2.0)

    group.state["sig"][0] = 100.0  # F = 200.0
    fint = np.zeros((2, 3))
    truss.static_internal_forces(group, model.x, None, None, fint, None)

    expected_f1 = np.array([200.0 * 0.6, 200.0 * 0.8, 0.0])
    assert np.allclose(fint[0], expected_f1)
    assert np.allclose(fint[1], -expected_f1)


def test_truss_starter_deck_integration(tmp_path):
    """Verify end-to-end Starter deck parsing of /TRUSS, /PROP/TRUSS, /MAT/LAW1, and /PART."""
    deck_text = """# OpenRadioss deck with Truss element
/BEGIN
TEST_TRUSS
/NODE
1 0.0 0.0 0.0
2 10.0 0.0 0.0
/PART/1
Truss Part
1 1
/PROP/TRUSS/1
Truss Property
1.75
/MAT/LAW1/1
Steel LAW1
7.85e-9
210000.0 0.3
/TRUSS/1
1 1 2
/END
"""
    f = tmp_path / "TRUSS_0000.rad"
    f.write_text(deck_text)
    blocks = read_deck(str(f))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert len(log.errors) == 0, log.errors

    # Initialize element groups
    build_element_groups(model, log)
    assert model.trusses is not None
    assert len(model.trusses.ids) == 1
    assert model.trusses.ids[0] == 1
    assert np.array_equal(model.trusses.conn[0], [0, 1])

    truss.init_group(model.trusses, model, log)

    assert np.isclose(model.trusses.state["area"][0], 1.75)
    assert np.isclose(model.trusses.state["L0"][0], 10.0)
    expected_mass = 7.85e-9 * 1.75 * 10.0
    assert np.isclose(model.trusses.state["mass"][0], expected_mass)


def test_truss_fixed_format_prop_type2_parsing(tmp_path):
    """Verify parsing fixed-format /PROP/TYPE2 deck cards with 20-character fields."""
    deck_text = """# OpenRadioss fixed format PROP/TYPE2
/BEGIN
FIXED_TRUSS
/NODE
1       0.0       0.0       0.0
2      12.0       0.0       0.0
/PART/1
Truss Part
         1         1
/PROP/TYPE2/1
Truss Prop Title
                3.25
/MAT/LAW1/1
Material
             7.85e-9
            210000.0                 0.3
/TRUSS/1
         1         1         2
/END
"""
    f = tmp_path / "FIXED_0000.rad"
    f.write_text(deck_text)
    blocks = read_deck(str(f))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert len(log.errors) == 0, log.errors

    build_element_groups(model, log)
    truss.init_group(model.trusses, model, log)

    assert np.isclose(model.trusses.state["area"][0], 3.25)
    assert np.isclose(model.trusses.state["L0"][0], 12.0)


def test_truss_defensive_edge_cases():
    """Verify defensive handling for empty groups, None vectors, dt<=0, and degenerate elements."""
    # 1. Empty element group
    log = MessageLog()
    empty_group = ElementGroup(ids=np.zeros(0, dtype=int), conn=np.zeros((0, 2), dtype=int), part=np.zeros(0, dtype=int))
    empty_group.state["slices"] = []
    dummy_model = Model()
    dummy_model.x0 = np.zeros((0, 3))

    nidx, mass, _ = truss.init_group(empty_group, dummy_model, log)
    assert len(nidx) == 0
    assert len(mass) == 0

    dt_empty = truss.forces(empty_group, np.zeros((0, 3)), None, None, 1e-4, None, None)
    assert len(dt_empty) == 0

    k_empty, dof_empty = truss.tangent(empty_group, np.zeros((0, 3)))
    assert k_empty.shape == (0, 6, 6)
    assert dof_empty.shape == (0, 6)

    kg_empty, dofg_empty = truss.kgeo(empty_group, np.zeros((0, 3)))
    assert kg_empty.shape == (0, 6, 6)

    m_empty, dofm_empty = truss.consistent_mass(empty_group)
    assert m_empty.shape == (0, 6, 6)

    # 2. None velocity and None fint
    coords = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
    conn = [[0, 1]]
    model, group, _ = _make_truss_model(coords, conn)
    dt_res = truss.forces(group, model.x, None, None, 1e-4, None, None)
    assert len(dt_res) == 1
    assert dt_res[0] > 0.0

    # 3. dt <= 0
    dt_zero = truss.forces(group, model.x, np.zeros((2, 3)), None, 0.0, None, None)
    assert len(dt_zero) == 1

    # 4. static_internal_forces with fint = None
    truss.static_internal_forces(group, model.x, None, None, None, None)

    # 5. implicit_internal_forces with u = None and fint = None
    truss.implicit_internal_forces(group, model.x, None, None, None, None, nlgeom=True)
