"""
Unit tests for Flexible Body Modal Solver (/FXBODY).

Verifies:
1. Modal superposition: single mode cantilever beam analytical solution
2. Generalized force projection: Φᵀ · F matches expected
3. Modal integration: free vibration frequency matches input ω
4. Energy conservation: KE + PE = const for undamped system
5. Critical dt: 2 / ω_max (undamped and damped)
6. Physical recovery: u = Φ · q and v = Φ · q̇ for known q, q̇
7. Physical node update in model
8. Initialization from model definitions (init_flex_bodies)
9. Engine cycle hook (flex_body_forces)
10. Damped modal vibration & energy dissipation
"""

import numpy as np
import pytest

from pyradioss.engine.flex_body import (
    FlexBody,
    ModalEnergy,
    init_flex_bodies,
    flex_body_forces,
)
from pyradioss.model.entities import FxBody
from pyradioss.model.model import Model


class MockModel:
    """Mock OpenRadioss Model for testing node updates and cycle hook."""

    def __init__(self, node_ids, coords):
        self.node_ids = np.asarray(node_ids, dtype=np.int64)
        self.x0 = np.asarray(coords, dtype=float).copy()
        self.x = np.asarray(coords, dtype=float).copy()
        self.v = np.zeros_like(self.x)
        self.vr = np.zeros_like(self.x)
        self.u = np.zeros_like(self.x)
        self.fext = np.zeros_like(self.x)
        self.fint = np.zeros_like(self.x)
        self.fcont = np.zeros_like(self.x)
        self._id2idx = {nid: i for i, nid in enumerate(self.node_ids)}
        self.fxbodies = {}
        self.flex_bodies = {}
        self.dt = 0.001


def test_modal_superposition_cantilever_beam():
    """Test modal superposition against analytical cantilever beam solution.

    Analytical Euler-Bernoulli beam theory:
      Tip static deflection under tip load P:
        u_tip = P * L^3 / (3 * E * I) = P / K_stat
    """
    L = 2.0
    E = 2.0e11
    b = 0.05
    h = 0.1
    I = b * h**3 / 12.0  # 4.166667e-6 m^4
    K_stat = 3.0 * E * I / (L**3)  # 312500.0 N/m
    M_modal = 10.0
    omega = np.sqrt(K_stat / M_modal)

    # Tip node: node ID 10 with 3 DOFs (UX, UY, UZ)
    boundary_nodes = [10]
    # Mode shape: tip normalized to 1.0 in Y direction
    phi = np.array([[0.0], [1.0], [0.0]])

    body = FlexBody(
        modes=[1],
        frequencies=[omega],
        modal_mass=[M_modal],
        modal_stiffness=[K_stat],
        boundary_nodes=boundary_nodes,
        phi=phi,
        body_id=1,
    )

    # Apply tip static load P = 5000 N in Y direction
    P = 5000.0
    F_ext = np.array([0.0, P, 0.0])
    Q = body.compute_generalized_forces(F_ext)
    assert np.isclose(Q[0], P)

    # Static equilibrium in modal coordinates: K * q = Q => q = Q / K
    q_stat = float(Q[0] / K_stat)
    body.q[0] = q_stat
    u = body.recover_displacements()

    # Analytical tip deflection
    u_analytical = P * (L**3) / (3.0 * E * I)
    assert np.isclose(u[1], u_analytical, rtol=1e-5)
    assert np.isclose(u[1], 0.016, rtol=1e-5)

    # Dynamic step response under sudden load:
    # Analytical: q(t) = (P / K) * (1 - cos(omega * t))
    # Peak deflection is exactly 2 * u_analytical at t = pi / omega
    body.reset()
    body.compute_generalized_forces(F_ext)

    dt = (2.0 * np.pi / omega) / 400.0
    half_period = np.pi / omega
    t = 0.0
    max_u = 0.0
    while t <= half_period * 1.05:
        body.modal_time_step(dt)
        disp = body.recover_displacements()
        if disp[1] > max_u:
            max_u = disp[1]
        t += dt

    # Peak dynamic deflection = 2 * u_static
    assert np.isclose(max_u, 2.0 * u_analytical, rtol=1e-3)


def test_generalized_force_projection():
    """Test generalized force projection: Q = Φᵀ · F matches expected matrix product."""
    np.random.seed(42)
    n_nodes = 4
    n_dof = n_nodes * 3  # 12 DOFs
    n_modes = 3

    phi = np.random.randn(n_dof, n_modes)
    frequencies = np.array([10.0, 25.0, 60.0])
    boundary_nodes = [101, 102, 103, 104]

    body = FlexBody(
        modes=[1, 2, 3],
        frequencies=frequencies,
        boundary_nodes=boundary_nodes,
        phi=phi,
    )

    # 1D physical force vector
    F = np.random.randn(n_dof)
    expected_Q = phi.T @ F
    Q = body.compute_generalized_forces(F)

    assert np.allclose(Q, expected_Q)
    assert np.allclose(body.q_force, expected_Q)

    # 2D physical force array (n_nodes, 3)
    F_2d = F.reshape(n_nodes, 3)
    Q_2d = body.compute_generalized_forces(F_2d)
    assert np.allclose(Q_2d, expected_Q)


def test_modal_integration_free_vibration_frequency():
    """Test modal integration: free vibration frequency matches input ω."""
    omega_input = 25.0  # rad/s
    period = 2.0 * np.pi / omega_input  # ~0.2513 s
    modal_mass = 2.0
    modal_stiffness = (omega_input**2) * modal_mass

    body = FlexBody(
        modes=[1],
        frequencies=[omega_input],
        modal_mass=[modal_mass],
        modal_stiffness=[modal_stiffness],
    )

    # Initial condition: q(0) = 0.05, q_dot(0) = 0.0
    q0 = 0.05
    body.q[0] = q0
    body.q_dot[0] = 0.0

    dt = 0.0005  # small enough for high accuracy
    n_steps = int(4.0 * period / dt)

    t_hist = []
    q_hist = []
    for step in range(n_steps):
        t = (step + 1) * dt
        body.modal_time_step(dt)
        t_hist.append(t)
        q_hist.append(float(body.q[0]))

    t_arr = np.array(t_hist)
    q_arr = np.array(q_hist)
    q_exact = q0 * np.cos(omega_input * t_arr)

    # Max difference compared to exact analytical solution
    max_err = np.max(np.abs(q_arr - q_exact))
    assert max_err < 5e-4

    # Measure numerical frequency from zero crossings
    zero_crossings = np.where(np.diff(np.sign(q_arr)))[0]
    crossing_times = t_arr[zero_crossings]
    # Each half period has one zero crossing
    half_periods = np.diff(crossing_times)
    measured_half_period = np.mean(half_periods)
    measured_period = 2.0 * measured_half_period
    measured_omega = 2.0 * np.pi / measured_period

    assert np.isclose(measured_omega, omega_input, rtol=1e-3)


def test_energy_conservation_undamped():
    """Test energy conservation: KE + PE = const for undamped multi-mode system."""
    frequencies = np.array([8.0, 20.0, 50.0])
    modal_mass = np.array([1.0, 1.5, 0.8])
    modal_stiffness = (frequencies**2) * modal_mass

    body = FlexBody(
        modes=[1, 2, 3],
        frequencies=frequencies,
        modal_mass=modal_mass,
        modal_stiffness=modal_stiffness,
    )

    # Initial non-zero displacements and velocities
    body.q = np.array([0.04, -0.02, 0.01])
    body.q_dot = np.array([0.15, -0.10, 0.25])

    e0 = body.compute_modal_energy()
    initial_total_energy = float(e0)

    # ModalEnergy unpacking check
    ke0, pe0 = e0
    assert np.isclose(ke0 + pe0, initial_total_energy)
    assert np.isclose(e0.kinetic, ke0)
    assert np.isclose(e0.potential, pe0)

    dt = 0.001
    energies = []
    for _ in range(1500):
        body.modal_time_step(dt)
        e = body.compute_modal_energy()
        energies.append(float(e))

    energies = np.array(energies)
    max_rel_fluctuation = np.max(np.abs(energies - initial_total_energy)) / initial_total_energy

    # Symplectic integration preserves Hamiltonian without energy drift
    assert max_rel_fluctuation < 1e-3
    assert np.isclose(energies[-1], initial_total_energy, rtol=1e-3)


def test_critical_dt():
    """Test critical dt: dt = 2 / ω_max."""
    # Undamped case
    frequencies = [12.0, 40.0, 200.0]
    omega_max = 200.0
    body = FlexBody(modes=[1, 2, 3], frequencies=frequencies)

    dt_crit = body.critical_dt()
    expected_dt = 2.0 / omega_max
    assert np.isclose(dt_crit, expected_dt)
    assert np.isclose(dt_crit, 0.01)

    # Damped case with beta > 0
    beta = 0.0005
    body_damped = FlexBody(
        modes=[1, 2, 3],
        frequencies=frequencies,
        beta=beta,
    )
    dt_damped = body_damped.critical_dt()

    # OpenRadioss formula: min(DTC1, DTC2)
    w = omega_max
    dtc1 = (-beta * w + np.sqrt(beta**2 * w**2 + 4.0)) / w
    dtc2 = 2.0 / (beta * w**2)
    expected_damped = min(dtc1, dtc2)

    assert np.isclose(dt_damped, expected_damped)
    assert dt_damped < dt_crit


def test_physical_recovery():
    """Test physical recovery: u = Φ·q and v = Φ·q̇ for known q, q̇."""
    n_dof = 6
    n_modes = 2
    phi = np.array([
        [1.0, 0.5],
        [0.0, 1.2],
        [-0.5, 0.8],
        [2.0, -1.0],
        [0.4, 0.2],
        [-1.1, 0.0],
    ])

    body = FlexBody(
        modes=[1, 2],
        frequencies=[10.0, 30.0],
        boundary_nodes=[1, 2],
        phi=phi,
    )

    q_known = np.array([0.03, -0.015])
    q_dot_known = np.array([1.2, -0.4])

    body.q = q_known.copy()
    body.q_dot = q_dot_known.copy()

    u = body.recover_displacements()
    v = body.recover_velocities()

    assert np.allclose(u, phi @ q_known)
    assert np.allclose(v, phi @ q_dot_known)
    assert np.allclose(body.u, u)
    assert np.allclose(body.v, v)


def test_update_physical_nodes():
    """Test updating physical model node arrays with recovered u and v."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    model = MockModel(node_ids=[10, 20], coords=coords)

    phi = np.array([
        [0.01, 0.0],
        [0.02, 0.0],
        [0.00, 0.0],
        [0.03, 0.0],
        [0.04, 0.0],
        [0.00, 0.0],
    ])

    body = FlexBody(
        modes=[1, 2],
        frequencies=[10.0, 20.0],
        boundary_nodes=[10, 20],
        phi=phi,
    )

    body.q = np.array([1.0, 0.0])
    body.q_dot = np.array([10.0, 0.0])

    body.update_physical_nodes(model)

    # Check displacements: node 10 gets [0.01, 0.02, 0.0], node 20 gets [0.03, 0.04, 0.0]
    expected_x = coords + np.array([
        [0.01, 0.02, 0.0],
        [0.03, 0.04, 0.0],
    ])
    expected_v = np.array([
        [0.10, 0.20, 0.0],
        [0.30, 0.40, 0.0],
    ])

    assert np.allclose(model.x, expected_x)
    assert np.allclose(model.v, expected_v)


def test_init_flex_bodies_from_model():
    """Test init_flex_bodies initializes FlexBody instances from model.fxbodies."""
    model = Model()
    model.fxbodies[1] = FxBody(
        id=1,
        title="Test Flex Body 1",
        node_id=101,
        imin=1,
        imax=3,
    )
    model.fxbodies[2] = FxBody(
        id=2,
        title="Test Flex Body 2",
        node_id=201,
        imin=1,
        imax=2,
    )

    bodies = init_flex_bodies(model)
    assert len(bodies) == 2
    assert bodies[0].body_id == 1
    assert bodies[0].n_modes == 3
    assert bodies[0].boundary_nodes[0] == 101
    assert bodies[1].body_id == 2
    assert bodies[1].n_modes == 2

    # Second call returns existing instances
    bodies2 = init_flex_bodies(model)
    assert bodies2 == bodies


def test_flex_body_forces_engine_hook():
    """Test the flex_body_forces cycle hook gathers forces and updates state."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    model = MockModel(node_ids=[1, 2], coords=coords)
    model.fext[1] = np.array([0.0, 100.0, 0.0])  # Force on node 2

    phi = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],  # Node 2 Y direction
        [0.0, 0.0],
        [0.0, 0.0],
    ])

    body = FlexBody(
        modes=[1, 2],
        frequencies=[50.0, 100.0],
        boundary_nodes=[1, 2],
        phi=phi,
        body_id=1,
    )

    dt_crit = flex_body_forces(model, [body], dt=0.001)

    assert np.isclose(dt_crit, 2.0 / 100.0)
    assert np.isclose(body.q_force[0], 100.0)
    assert np.any(body.q != 0.0)
    assert np.any(model.x != coords)


def test_damped_free_vibration():
    """Test damped free vibration decays energy and dissipates work."""
    omega = 20.0
    zeta = 0.05  # 5% modal damping
    body = FlexBody(
        modes=[1],
        frequencies=[omega],
        modal_mass=[1.0],
        damping_ratio=zeta,
    )

    body.q[0] = 0.1
    body.q_dot[0] = 0.0
    initial_energy = float(body.compute_modal_energy())

    dt = 0.001
    for _ in range(1000):  # 1.0 second, ~3.18 cycles
        body.modal_time_step(dt)

    final_energy = float(body.compute_modal_energy())
    # Energy must decrease
    assert final_energy < initial_energy
    # Dissipated energy must be positive
    assert body.dissipated_energy > 0.0
    # Total energy + dissipated work should balance initial energy
    assert np.isclose(final_energy + body.dissipated_energy, initial_energy, rtol=0.02)


def test_state_serialization_restart():
    """Test serializing and restoring flexible body state for restart."""
    body = FlexBody(
        modes=[1, 2],
        frequencies=[10.0, 30.0],
        body_id=5,
    )
    body.q = np.array([0.02, -0.01])
    body.q_dot = np.array([0.5, -0.2])
    body.external_work = 12.5
    body.dissipated_energy = 1.3

    saved = body.state_dict()
    assert saved["body_id"] == 5

    body2 = FlexBody(
        modes=[1, 2],
        frequencies=[10.0, 30.0],
        body_id=5,
    )
    body2.load_state_dict(saved)

    assert np.allclose(body2.q, body.q)
    assert np.allclose(body2.q_dot, body.q_dot)
    assert np.isclose(body2.external_work, body.external_work)
    assert np.isclose(body2.dissipated_energy, body.dissipated_energy)
