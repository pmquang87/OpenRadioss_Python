"""
Unit tests for Milestone M518:
2-Node 1D Axial Spring Element (/SPRING + /PROP/SPRING TYPE4, /PROP/SPR_PRE TYPE32)
Hardening, Kinematics, Energy Balance, Critical Time Step with Damping Reduction,
6-DOF Material Tangent Stiffness, 6-DOF Taut-String Initial-Stress Geometric Stiffness,
Diagonal Consistent Mass, Assembled Viscous Damping Matrix, Total-Form Implicit Residual
with Exact Directional Derivative Consistency, and Degenerate Fallback.

Fortran origin:
  engine/source/elements/spring/rforc3.F
  starter/source/elements/spring/rmass3.F
  starter/source/properties/spring/hm_read_prop04.F
  starter/source/properties/spring/hm_read_prop32.F
  engine/source/implicit/imp_glob_k.F
"""

import numpy as np
import pytest

from pyradioss.common.constants import EM20, EP30
from pyradioss.common.messages import MessageLog
from pyradioss.elements import spring, spring_general
from pyradioss.model.entities import Property
from pyradioss.model.model import ElementGroup, Model


def _make_spring_model(prop, x0_nodes=None):
    if x0_nodes is None:
        x0_nodes = np.array([[0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0]])
    m = Model()
    m.add_nodes(np.array([1, 2]), x0_nodes)
    g = ElementGroup(ids=np.array([1]), conn=np.array([[0, 1]]),
                     part=np.array([0]))
    g.state["slices"] = [(slice(0, 1), None, prop)]
    m.springs = g
    return m, g


# ============================================================================
# 1. Empty Group Initialization Guard
# ============================================================================
def test_spring_empty_group_init():
    """Empty group init_group returns empty node arrays and allocates empty state."""
    m = Model()
    g = ElementGroup(ids=np.empty(0, dtype=np.int64),
                     conn=np.empty((0, 2), dtype=np.int64),
                     part=np.empty(0, dtype=np.int64))
    g.state["slices"] = []
    log = MessageLog()
    node_idx, massn, inertn = spring.init_group(g, m, log)
    assert len(node_idx) == 0
    assert len(massn) == 0
    assert inertn is None
    assert len(g.state["L0"]) == 0
    assert len(g.state["force"]) == 0
    assert len(g.state["mass"]) == 0
    assert len(g.state["k"]) == 0
    assert len(g.state["cdamp"]) == 0
    assert len(g.state["idx4"]) == 0
    assert len(g.state["idx6"]) == 0
    assert len(g.state["idx32"]) == 0


# ============================================================================
# 2. Empty Group Forces, Tangents, Residuals & Matrices Guard
# ============================================================================
def test_spring_empty_group_forces_and_matrices():
    """Empty group methods return cleanly shaped empty arrays without errors."""
    m = Model()
    g = ElementGroup(ids=np.empty(0, dtype=np.int64),
                     conn=np.empty((0, 2), dtype=np.int64),
                     part=np.empty(0, dtype=np.int64))
    g.state["slices"] = []
    spring.init_group(g, m, None)

    x = np.empty((0, 3))
    v = np.empty((0, 3))
    fint = np.empty((0, 3))

    dtc = spring.forces(g, x, v, None, 0.001, fint, None)
    assert len(dtc) == 0

    ke, edofs = spring.tangent(g, x)
    assert ke.shape == (0, 6, 6)
    assert edofs.shape == (0, 6)

    kge, edofs2 = spring.kgeo(g, x)
    assert kge.shape == (0, 6, 6)
    assert edofs2.shape == (0, 6)

    me, edofs3 = spring.consistent_mass(g, x)
    assert me.shape == (0, 6, 6)
    assert edofs3.shape == (0, 6)

    ce, edofs4 = spring.damping_matrix(g, x)
    assert ce.shape == (0, 6, 6)
    assert edofs4.shape == (0, 6)

    # implicit residuals run without error on empty group
    spring.implicit_internal_forces(g, x, np.empty((0, 3)), None, fint, None, nlgeom=True)
    spring.implicit_internal_forces(g, x, np.empty((0, 3)), None, fint, None, nlgeom=False)


# ============================================================================
# 3. Coincident Nodes Degenerate Fallback (L <= EM20)
# ============================================================================
def test_spring_coincident_nodes_fallback():
    """When nodes coincide (norm < EM20), axis falls back to [1, 0, 0] without NaNs."""
    prop = Property(id=1, type=4, params={"mass": 1.0, "k": 100.0, "c": 0.0})
    x0 = np.array([[2.5, 3.5, -1.0], [2.5, 3.5, -1.0]])
    m, g = _make_spring_model(prop, x0)
    spring.init_group(g, m, None)

    conn, L, a = spring._spring_axis(g, x0)
    assert np.isclose(L[0], EM20)
    assert np.allclose(a[0], [1.0, 0.0, 0.0])

    fint = np.zeros((2, 3))
    dtc = spring.forces(g, x0, None, None, 0.001, fint, None)
    assert np.all(np.isfinite(fint))
    assert np.all(np.isfinite(dtc))

    ke, _ = spring.tangent(g, x0)
    assert np.all(np.isfinite(ke))
    kge, _ = spring.kgeo(g, x0)
    assert np.all(np.isfinite(kge))


# ============================================================================
# 4. Axial Force under Tension
# ============================================================================
def test_spring_forces_axial_tension():
    """Under extension Delta L > 0, spring develops tension F = K * Delta L pulling nodes."""
    prop = Property(id=1, type=4, params={"mass": 1.0, "k": 500.0, "c": 0.0})
    x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    m, g = _make_spring_model(prop, x0)
    spring.init_group(g, m, None)

    x = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    fint = np.zeros((2, 3))
    dtc = spring.forces(g, x, None, None, 0.001, fint, None)

    # F = 500 * (1.5 - 1.0) = 250.0
    assert np.isclose(g.state["force"][0], 250.0)
    # Node 0 is pulled toward Node 1 (+X): +250
    # Node 1 is pulled toward Node 0 (-X): -250
    assert np.allclose(fint[0], [250.0, 0.0, 0.0])
    assert np.allclose(fint[1], [-250.0, 0.0, 0.0])


# ============================================================================
# 5. Axial Force under Compression
# ============================================================================
def test_spring_forces_axial_compression():
    """Under compression Delta L < 0, spring develops compression pushing nodes apart."""
    prop = Property(id=1, type=4, params={"mass": 2.0, "k": 1000.0, "c": 0.0})
    x0 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    m, g = _make_spring_model(prop, x0)
    spring.init_group(g, m, None)

    x = np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]])
    fint = np.zeros((2, 3))
    dtc = spring.forces(g, x, None, None, 0.001, fint, None)

    # F = 1000 * (1.8 - 2.0) = -200.0
    assert np.isclose(g.state["force"][0], -200.0)
    # Node 0 pushed away from Node 1 (-X): -200
    # Node 1 pushed away from Node 0 (+X): +200
    assert np.allclose(fint[0], [-200.0, 0.0, 0.0])
    assert np.allclose(fint[1], [200.0, 0.0, 0.0])


# ============================================================================
# 6. Viscous Damping Force
# ============================================================================
def test_spring_forces_viscous_damping():
    """At rest length (L=L0), relative velocity Ldot produces pure damping force F = C * Ldot."""
    prop = Property(id=1, type=4, params={"mass": 1.0, "k": 100.0, "c": 20.0})
    x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    m, g = _make_spring_model(prop, x0)
    spring.init_group(g, m, None)

    v = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    fint = np.zeros((2, 3))
    dtc = spring.forces(g, x0, v, None, 0.001, fint, None)

    # F = 100 * (0) + 20 * (5.0) = 100.0
    assert np.isclose(g.state["force"][0], 100.0)
    assert np.allclose(fint[0], [100.0, 0.0, 0.0])
    assert np.allclose(fint[1], [-100.0, 0.0, 0.0])


# ============================================================================
# 7. Internal Energy Conservation & Damping Dissipation
# ============================================================================
def test_spring_internal_energy_conservation():
    """Work done equals elastic strain energy for undamped, plus dissipation for damped."""
    prop = Property(id=1, type=4, params={"mass": 1.0, "k": 200.0, "c": 0.0})
    x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    m, g = _make_spring_model(prop, x0)
    spring.init_group(g, m, None)

    dt = 0.01
    v_rel = 2.0
    v = np.array([[0.0, 0.0, 0.0], [v_rel, 0.0, 0.0]])
    cur_x = x0.copy()
    for _ in range(10):
        cur_x[1, 0] += v_rel * dt
        fint = np.zeros((2, 3))
        spring.forces(g, cur_x, v, None, dt, fint, None)

    # Displacement = 10 * 2.0 * 0.01 = 0.2
    # Elastic strain energy = 0.5 * 200.0 * (0.2)^2 = 4.0
    expected_elastic = 0.5 * 200.0 * (0.2 ** 2)
    assert np.isclose(g.state["eint"][0], expected_elastic, rtol=1e-4)

    # Test with damping C = 10.0
    prop_d = Property(id=2, type=4, params={"mass": 1.0, "k": 200.0, "c": 10.0})
    m_d, g_d = _make_spring_model(prop_d, x0)
    spring.init_group(g_d, m_d, None)
    cur_x = x0.copy()
    for _ in range(10):
        cur_x[1, 0] += v_rel * dt
        fint = np.zeros((2, 3))
        spring.forces(g_d, cur_x, v, None, dt, fint, None)

    # For damping, initial force was 0, jumping to C*v_rel on step 1.
    # The trapezoidal rule over 10 steps yields:
    # 0.5 * [ (0 + 20) + 9 * (20 + 20) ] * v_rel * dt = 0.5 * 380 * 0.02 = 3.8
    # Total eint = 4.0 (elastic) + 3.8 (damping) = 7.8
    fd = 10.0 * v_rel
    expected_damping = 0.5 * (fd + 9.0 * (2.0 * fd)) * (v_rel * dt)
    expected_total = expected_elastic + expected_damping
    assert np.isclose(g_d.state["eint"][0], expected_total, rtol=1e-5)


# ============================================================================
# 8. Linear Momentum Conservation in Arbitrary 3D Orientation
# ============================================================================
def test_spring_linear_momentum_conservation():
    """Nodal internal forces sum to zero to machine precision in arbitrary 3D orientation."""
    prop = Property(id=1, type=4, params={"mass": 3.0, "k": 350.0, "c": 15.0})
    x0 = np.array([[1.2, -0.5, 3.1], [4.5, 2.8, -1.9]])
    m, g = _make_spring_model(prop, x0)
    spring.init_group(g, m, None)

    x = np.array([[1.1, -0.7, 3.3], [4.9, 3.0, -1.5]])
    v = np.array([[-1.0, 2.5, 0.4], [3.2, -1.1, 4.0]])
    fint = np.zeros((2, 3))
    spring.forces(g, x, v, None, 0.005, fint, None)

    f_total = fint[0] + fint[1]
    assert np.linalg.norm(f_total) < 1e-14


# ============================================================================
# 9. Angular Momentum Conservation in Arbitrary 3D Orientation
# ============================================================================
def test_spring_angular_momentum_conservation():
    """Collinear equal-and-opposite forces produce zero net torque about origin."""
    prop = Property(id=1, type=4, params={"mass": 2.5, "k": 400.0, "c": 10.0})
    x0 = np.array([[2.0, -1.0, 0.5], [5.0, 3.0, 4.0]])
    m, g = _make_spring_model(prop, x0)
    spring.init_group(g, m, None)

    x = np.array([[2.2, -0.8, 0.7], [5.5, 3.4, 4.2]])
    v = np.array([[0.5, 1.0, -0.5], [-0.5, 2.0, 1.5]])
    fint = np.zeros((2, 3))
    spring.forces(g, x, v, None, 0.002, fint, None)

    tau = np.cross(x[0], fint[0]) + np.cross(x[1], fint[1])
    assert np.linalg.norm(tau) < 1e-13


# ============================================================================
# 10. Undamped Critical Time Step (Two-Mass Oscillator)
# ============================================================================
def test_spring_critical_dt_undamped():
    """Undamped critical dt matches omega = 2 sqrt(K/M) -> dt = 2/omega = sqrt(M/K)."""
    M = 2.0
    K = 50.0
    prop = Property(id=1, type=4, params={"mass": M, "k": K, "c": 0.0})
    x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    m, g = _make_spring_model(prop, x0)
    spring.init_group(g, m, None)

    dtc = spring.forces(g, x0, None, None, 0.0, None, None)
    # omega = 2 * sqrt(50 / 2) = 10.0 -> dt = 2 / 10 = 0.2
    assert np.isclose(dtc[0], 0.2)


# ============================================================================
# 11. Damped Critical Time Step Reduction
# ============================================================================
def test_spring_critical_dt_damped():
    """Damped critical dt includes damping reduction (2/omega)*(sqrt(1+xi^2) - xi)."""
    M = 2.0
    K = 50.0
    C = 2.5
    prop = Property(id=1, type=4, params={"mass": M, "k": K, "c": C})
    x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    m, g = _make_spring_model(prop, x0)
    spring.init_group(g, m, None)

    dtc = spring.forces(g, x0, None, None, 0.0, None, None)
    omega = 2.0 * np.sqrt(K / M)
    xi = C / np.sqrt(K * M)
    expected_dt = (2.0 / omega) * (np.sqrt(1.0 + xi ** 2) - xi)
    assert np.isclose(dtc[0], expected_dt)


# ============================================================================
# 12. Zero Stiffness Time Step Guard
# ============================================================================
def test_spring_zero_stiffness_dt():
    """When spring stiffness is zero, dt_crit returns EP30 without divide-by-zero."""
    prop = Property(id=1, type=4, params={"mass": 2.0, "k": 0.0, "c": 0.0})
    x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    m, g = _make_spring_model(prop, x0)
    spring.init_group(g, m, None)

    dtc = spring.forces(g, x0, None, None, 0.0, None, None)
    assert dtc[0] == EP30


# ============================================================================
# 13. Material Tangent Stiffness Symmetry & Spectral Rank
# ============================================================================
def test_spring_tangent_stiffness_symmetry_and_rank():
    """Material tangent Ke = K [[a a^T, -a a^T], [-a a^T, a a^T]] is symmetric with rank 1."""
    K = 300.0
    prop = Property(id=1, type=4, params={"mass": 1.0, "k": K, "c": 0.0})
    x0 = np.array([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]])
    m, g = _make_spring_model(prop, x0)
    spring.init_group(g, m, None)

    ke, edofs = spring.tangent(g, x0)
    Km = ke[0]
    assert np.allclose(Km, Km.T, atol=1e-14)

    eigvals = np.linalg.eigvalsh(Km)
    zeros = eigvals[:5]
    nonzero = eigvals[5]
    # exactly 5 zero modes (3 rigid translations + 2 rotations)
    assert np.all(np.abs(zeros) < 1e-12)
    # exactly 1 nonzero mode equal to 2 * K
    assert np.isclose(nonzero, 2.0 * K)


# ============================================================================
# 14. Initial-Stress Geometric Stiffness Symmetry & Spectral Rank
# ============================================================================
def test_spring_geometric_stiffness_symmetry_and_rank():
    """Geometric stiffness Kgeo = (F/L) [[I-a a^T, -(I-a a^T)], [-(I-a a^T), I-a a^T]] has rank 2."""
    K = 300.0
    prop = Property(id=1, type=4, params={"mass": 1.0, "k": K, "c": 0.0})
    x0 = np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]])  # L0 = 4.0
    m, g = _make_spring_model(prop, x0)
    spring.init_group(g, m, None)

    # Stretch to L = 5.0 -> F = 300 * 1.0 = 300.0
    x = np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    fint = np.zeros((2, 3))
    spring.forces(g, x, None, None, 0.0, fint, None)

    kge, _ = spring.kgeo(g, x)
    Kg = kge[0]
    assert np.allclose(Kg, Kg.T, atol=1e-14)

    eigvals = np.linalg.eigvalsh(Kg)
    # exactly 4 zero modes (3 translations + 1 axial elongation mode)
    # exactly 2 nonzero modes with eigenvalue 2 * F / L = 2 * 300 / 5 = 120.0
    zeros = eigvals[:4]
    nonzeros = eigvals[4:]
    assert np.all(np.abs(zeros) < 1e-12)
    assert np.allclose(nonzeros, [120.0, 120.0])


# ============================================================================
# 15. Consistent Mass Matrix Properties
# ============================================================================
def test_spring_consistent_mass_properties():
    """Exact element mass is diagonal M/2 on all 6 translational DOFs (trace 3M)."""
    M = 4.0
    prop = Property(id=1, type=4, params={"mass": M, "k": 100.0, "c": 0.0})
    m, g = _make_spring_model(prop)
    spring.init_group(g, m, None)

    me, edofs = spring.consistent_mass(g)
    Me = me[0]
    assert np.allclose(Me, np.diag(np.diag(Me)))
    assert np.allclose(np.diag(Me), M / 2.0)
    assert np.isclose(np.trace(Me), 3.0 * M)


# ============================================================================
# 16. Viscous Damping Matrix Properties
# ============================================================================
def test_spring_damping_matrix_properties():
    """Viscous damping matrix Ce = C [[a a^T, -a a^T], [-a a^T, a a^T]] has rank 1."""
    C = 45.0
    prop = Property(id=1, type=4, params={"mass": 1.0, "k": 100.0, "c": C})
    x0 = np.array([[1.0, 1.0, 0.0], [3.0, 4.0, 0.0]])
    m, g = _make_spring_model(prop, x0)
    spring.init_group(g, m, None)

    ce, edofs = spring.damping_matrix(g, x0)
    Ce = ce[0]
    assert np.allclose(Ce, Ce.T, atol=1e-14)

    eigvals = np.linalg.eigvalsh(Ce)
    zeros = eigvals[:5]
    nonzero = eigvals[5]
    assert np.all(np.abs(zeros) < 1e-12)
    assert np.isclose(nonzero, 2.0 * C)


# ============================================================================
# 17. Total-Form Directional Derivative under Linear Geometry
# ============================================================================
def test_spring_directional_derivative_linear_geom():
    """Finite difference of implicit_internal_forces matches -K_mat * Delta u under nlgeom=False."""
    K = 400.0
    prop = Property(id=1, type=4, params={"mass": 1.0, "k": K, "c": 0.0})
    x0 = np.array([[1.0, 2.0, 3.0], [4.0, 6.0, 5.0]])
    m, g = _make_spring_model(prop, x0)
    spring.init_group(g, m, None)

    u_base = np.array([[0.05, -0.02, 0.01], [-0.03, 0.04, 0.02]])
    delta_u = np.array([[0.02, 0.01, -0.03], [0.01, -0.02, 0.04]])

    # Evaluate material tangent at x0
    ke, _ = spring.tangent(g, x0)
    Km = ke[0]
    du_flat = delta_u.reshape(-1)
    dF_expected = -Km @ du_flat  # in FEM convention, -K du = dFint

    # Central difference of implicit_internal_forces
    eps = 1e-6
    f_plus = np.zeros_like(x0)
    f_minus = np.zeros_like(x0)

    # g.state["force"] committed value
    g.state["force"][:] = 50.0

    spring.implicit_internal_forces(g, x0, u_base + eps * delta_u, None, f_plus, None, nlgeom=False)
    g.state["force"][:] = 50.0
    spring.implicit_internal_forces(g, x0, u_base - eps * delta_u, None, f_minus, None, nlgeom=False)

    dF_numeric = (f_plus - f_minus).reshape(-1) / (2.0 * eps)
    assert np.allclose(dF_numeric, dF_expected, rtol=1e-6)


# ============================================================================
# 18. Total-Form Directional Derivative under Nonlinear Geometry
# ============================================================================
def test_spring_directional_derivative_nonlinear_geom():
    """Finite difference of implicit_internal_forces matches -(K_mat + K_geo) * Delta u."""
    K = 500.0
    prop = Property(id=1, type=4, params={"mass": 1.0, "k": K, "c": 0.0})
    x0 = np.array([[1.0, 1.0, 1.0], [3.0, 4.0, 2.0]])
    m, g = _make_spring_model(prop, x0)
    spring.init_group(g, m, None)

    u_base = np.array([[0.1, -0.05, 0.08], [-0.04, 0.12, -0.06]])
    delta_u = np.array([[-0.03, 0.04, 0.02], [0.05, -0.01, -0.04]])

    x_cur = x0 + u_base
    ke, _ = spring.tangent(g, x_cur)
    # Establish force state at x_cur for kgeo
    f_dummy = np.zeros_like(x0)
    spring.implicit_internal_forces(g, x0, u_base, None, f_dummy, None, nlgeom=True)
    kge, _ = spring.kgeo(g, x_cur)

    K_tot = ke[0] + kge[0]
    du_flat = delta_u.reshape(-1)
    dF_expected = -K_tot @ du_flat

    eps = 1e-6
    f_plus = np.zeros_like(x0)
    f_minus = np.zeros_like(x0)

    spring.implicit_internal_forces(g, x0, u_base + eps * delta_u, None, f_plus, None, nlgeom=True)
    spring.implicit_internal_forces(g, x0, u_base - eps * delta_u, None, f_minus, None, nlgeom=True)

    dF_numeric = (f_plus - f_minus).reshape(-1) / (2.0 * eps)
    assert np.allclose(dF_numeric, dF_expected, rtol=1e-5)


# ============================================================================
# 19. TYPE32 Pretensioner Spring Cycle
# ============================================================================
def test_spring_type32_pretensioner_cycle():
    """TYPE32 pretensioner spring evaluates pretension law and accumulates force."""
    p32 = Property(id=1, type=32, params={
        "mass": 0.05,
        "stif0": 100.0,
        "stif1": 250.0,
        "ityp": 1,
        "f1": 80.0,
        "d1": 0.0,
        "ilock": 0,
        "scale_t": 1.0,
        "scale_d": 1.0,
        "scale_f": 1.0,
        "sens_id": 0,
    })
    x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    m, g = _make_spring_model(p32, x0)
    spring.init_group(g, m, None)

    assert len(g.state["idx32"]) == 1
    assert g.state["stif0"][0] == 100.0
    assert g.state["stif1"][0] == 250.0

    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    fint = np.zeros((2, 3))
    dtc = spring.forces(g, x0, v, None, 0.01, fint, None)
    assert np.all(np.isfinite(fint))
    assert dtc[0] > 0.0
    assert g.state["force"][0] > 0.0


# ============================================================================
# 20. Mixed Group Dispatch (TYPE4, TYPE8, TYPE32)
# ============================================================================
def test_spring_mixed_group_dispatch():
    """Group containing TYPE4, TYPE8, and TYPE32 dispatches all branches correctly."""
    p4 = Property(id=1, type=4, params={"mass": 1.0, "k": 200.0, "c": 0.0})
    p8 = Property(id=2, type=8, params={
        "mass": 1.5, "inertia": 0.5, "skew_id": 0,
        "k1": 300.0, "k2": 300.0, "k3": 300.0,
        "k4": 100.0, "k5": 100.0, "k6": 100.0,
    })
    p32 = Property(id=3, type=32, params={
        "mass": 0.1, "stif0": 150.0, "stif1": 200.0, "ityp": 1, "f1": 50.0
    })

    m = Model()
    nodes_x = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],  # elem 0 (type 4)
        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],  # elem 1 (type 8)
        [0.0, 2.0, 0.0], [1.0, 2.0, 0.0],  # elem 2 (type 32)
    ])
    m.add_nodes(np.arange(1, 7), nodes_x)
    conn = np.array([[0, 1], [2, 3], [4, 5]])
    g = ElementGroup(ids=np.array([1, 2, 3]), conn=conn, part=np.array([0, 0, 0]))
    g.state["slices"] = [
        (slice(0, 1), None, p4),
        (slice(1, 2), None, p8),
        (slice(2, 3), None, p32),
    ]
    m.springs = g

    node_idx, massn, inertn = spring.init_group(g, m, None)
    assert len(g.state["idx4"]) == 1
    assert len(g.state["idx6"]) == 1
    assert len(g.state["idx32"]) == 1

    fint = np.zeros((6, 3))
    mint = np.zeros((6, 3))
    v = np.zeros((6, 3))
    dtc = spring.forces(g, nodes_x, v, None, 0.001, fint, mint)

    assert len(dtc) == 3
    assert np.all(dtc > 0.0)
    assert np.all(np.isfinite(dtc))
