"""Milestone M613: Dynamic Relaxation Complete Parity Test Suite.

Tests 100% Fortran parity for:
1. /KEREL:
   - Peak kinetic energy detection and velocity zeroing (v[idx] = 0.0, vr[ir] = 0.0)
   - Exact annihilated kinetic energy booking (de == ke) matching static.F:116
   - Global energy conservation on oscillating system
2. /DYREL:
   - Acceleration coupling matching static.F:83-109 / 135-165:
     A <- -2*beta*V + (1 - beta*dt12)*A
   - Exponential velocity and amplitude decay to static equilibrium
3. /ADYREL:
   - Butterworth 2nd-order filter and period/frequency detection
   - Adaptive beta evolution (ENER_W0 from static.F:312)
   - Timestep reduction stability factor matching dtnodarayl.F:168-188:
     Dampa3 = 2*beta / (1 + beta*dt12)
     BB = 0.5 * Dampa3 * dt^2
     dt_reduced = sqrt(BB^2 + dt^2) - BB
4. Static Relaxation Convergence Monitor:
   - check_convergence(e_kin, e_int, e_kin_max, tol=1e-3)
   - Automatic termination when static equilibrium is achieved on 1-DOF oscillator
"""
from __future__ import annotations

import numpy as np
import pytest

from pyradioss.engine.damping import DynamicRelaxation, butterworth_filter, check_convergence
from pyradioss.model.model import EngineControls, NodeGroup


class MockModel:
    """Minimal model for unit testing DynamicRelaxation."""

    def __init__(self, n_nodes: int = 1):
        self.numnod = n_nodes
        self.mass = np.ones(n_nodes, dtype=np.float64) * 2.0
        self.inertia = np.ones(n_nodes, dtype=np.float64) * 0.5
        self.node_groups = {}


# ============================================================================
# 1. /KEREL Tests: Peak Detection, Velocity Zeroing, Exact Energy Booking
# ============================================================================

def test_kerel_peak_detection_and_exact_dissipation():
    """Verify /KEREL peak detection, velocity zeroing, and exact energy dissipation booking de == ke."""
    model = MockModel(2)
    controls = EngineControls()
    controls.kerel_active = True
    controls.kerel_tstart = 0.0
    controls.kerel_tstop = 1.0

    relax = DynamicRelaxation(model, controls)
    assert relax.active is True
    assert relax.ke_prev == 0.0

    v = np.zeros((2, 3), dtype=np.float64)
    vr = np.zeros((2, 3), dtype=np.float64)

    # Step 1: KE = 10.0 (increasing)
    # mass = [2.0, 2.0] -> 0.5 * 2.0 * (3.16227766)^2 = 10.0
    v[0, 0] = np.sqrt(10.0)
    de1 = relax.apply(0.01, 0.01, v, vr, model.mass, model.inertia)
    assert de1 == 0.0
    assert relax.ke_prev == pytest.approx(10.0)
    assert v[0, 0] == pytest.approx(np.sqrt(10.0))

    # Step 2: KE = 16.0 (increasing further)
    v[0, 0] = 4.0  # 0.5 * 2.0 * 16 = 16.0
    de2 = relax.apply(0.02, 0.01, v, vr, model.mass, model.inertia)
    assert de2 == 0.0
    assert relax.ke_prev == pytest.approx(16.0)

    # Step 3: KE = 9.0 (peak passed: ke < ke_prev)
    v[0, 0] = 3.0  # 0.5 * 2.0 * 9 = 9.0
    # Current annihilated kinetic energy at this step is exactly 9.0 (static.F:116)
    de3 = relax.apply(0.03, 0.01, v, vr, model.mass, model.inertia)
    # Peak detected: velocities must be zeroed in-place!
    assert np.allclose(v, 0.0)
    assert np.allclose(vr, 0.0)
    assert de3 == pytest.approx(9.0)  # exact dissipated energy = current KE annihilated
    assert relax.ke_prev == 0.0

    # Step 4: Next step with zero velocity
    de4 = relax.apply(0.04, 0.01, v, vr, model.mass, model.inertia)
    assert de4 == 0.0
    assert np.allclose(v, 0.0)


def test_kerel_rotational_dof_zeroing_and_energy_booking():
    """Verify that /KEREL zeros rotational velocities vr and includes rotational KE in de."""
    model = MockModel(1)
    controls = EngineControls()
    controls.kerel_active = True
    controls.kerel_tstart = 0.0
    controls.kerel_tstop = 1.0

    relax = DynamicRelaxation(model, controls)
    v = np.zeros((1, 3), dtype=np.float64)
    vr = np.zeros((1, 3), dtype=np.float64)

    # Mass = 2.0, inertia = 0.5
    # Step 1: v = 2.0 (KE_trans = 0.5 * 2.0 * 4 = 4.0), vr = 4.0 (KE_rot = 0.5 * 0.5 * 16 = 4.0) -> KE = 8.0
    v[0, 0] = 2.0
    vr[0, 1] = 4.0
    de1 = relax.apply(0.01, 0.01, v, vr, model.mass, model.inertia)
    assert de1 == 0.0
    assert relax.ke_prev == pytest.approx(8.0)

    # Step 2: v = 3.0 (KE_trans = 9.0), vr = 4.0 (KE_rot = 4.0) -> KE = 13.0 (peak)
    v[0, 0] = 3.0
    vr[0, 1] = 4.0
    de2 = relax.apply(0.02, 0.01, v, vr, model.mass, model.inertia)
    assert de2 == 0.0
    assert relax.ke_prev == pytest.approx(13.0)

    # Step 3: v = 2.0 (KE_trans = 4.0), vr = 2.0 (KE_rot = 1.0) -> KE = 5.0 (dropped)
    v[0, 0] = 2.0
    vr[0, 1] = 2.0
    de3 = relax.apply(0.03, 0.01, v, vr, model.mass, model.inertia)
    assert de3 == pytest.approx(5.0)  # exact total annihilated KE
    assert np.allclose(v, 0.0)
    assert np.allclose(vr, 0.0)
    assert relax.ke_prev == 0.0


def test_kerel_1dof_oscillator_energy_conservation_and_decay():
    """Verify that a 1-DOF spring-mass system with /KEREL maintains exact energy balance and settles."""
    m = 2.0
    k = 200.0  # omega_n = 10 rad/s, T_n = 0.6283 s
    dt = 0.002
    t_end = 1.5

    model = MockModel(1)
    model.mass[0] = m
    controls = EngineControls()
    controls.kerel_active = True
    controls.kerel_tstart = 0.0
    controls.kerel_tstop = t_end

    relax = DynamicRelaxation(model, controls)

    x = 1.0
    v = np.zeros((1, 3))
    vr = np.zeros((1, 3))
    e_damp_total = 0.0
    e_initial = 0.5 * k * (x ** 2)

    for step in range(int(t_end / dt)):
        t = step * dt
        a = (-k * x) / m
        v[0, 0] += a * dt
        de = relax.apply(t, dt, v, vr, model.mass, model.inertia)
        e_damp_total += de
        x += v[0, 0] * dt

        # Current total energy in system + dissipated energy
        e_kin = float(0.5 * m * (v[0, 0] ** 2))
        e_int = 0.5 * k * (x ** 2)
        e_total = e_kin + e_int + e_damp_total

        # Energy balance check: work dissipated is exact, total energy strictly conserved
        assert e_total == pytest.approx(e_initial, rel=1e-2)

    # At the end, kinetic energy has been nearly eliminated
    assert abs(v[0, 0]) < 0.1
    assert e_damp_total > 0.0


# ============================================================================
# 2. /DYREL Tests: Acceleration Coupling and Exponential Decay
# ============================================================================

def test_dyrel_acceleration_damping_coupling():
    """Verify apply_acceleration_damping matches static.F:83-96.

    Formula:
    A <- -2*beta*V + (1 - beta*dt12)*A
    """
    model = MockModel(2)
    controls = EngineControls()
    controls.dyrel_active = True
    controls.dyrel_beta = 1.5
    controls.dyrel_period = 0.0  # betate = 1.5

    relax = DynamicRelaxation(model, controls)
    assert relax.betate == 1.5

    dt = 0.001
    dt12 = 0.001

    a = np.array([[10.0, -5.0, 2.0], [4.0, 0.0, -8.0]], dtype=np.float64)
    v = np.array([[2.0, 1.0, -1.0], [-3.0, 0.5, 2.0]], dtype=np.float64)

    # Expected analytical calculation
    beta = 1.5
    omega = beta * dt12
    uomega = 1.0 - omega
    domega = 2.0 * beta
    a_expected = -domega * v + uomega * a

    relax.apply_acceleration_damping(a, v, dt, dt12)

    assert np.allclose(a, a_expected, rtol=1e-14, atol=1e-14)


def test_dyrel_acceleration_damping_rotational():
    """Verify apply_acceleration_damping works on rotational acceleration AR as well."""
    model = MockModel(1)
    controls = EngineControls()
    controls.dyrel_active = True
    controls.dyrel_beta = 2.0
    controls.dyrel_period = 0.1  # betate = 2.0 / 0.1 = 20.0

    relax = DynamicRelaxation(model, controls)
    assert relax.betate == 20.0

    dt12 = 0.005
    beta = 20.0
    domega = 2.0 * beta
    uomega = 1.0 - beta * dt12

    a = np.array([[10.0, 0.0, 0.0]])
    v = np.array([[1.0, 0.0, 0.0]])
    ar = np.array([[0.0, 5.0, -2.0]])
    vr = np.array([[0.0, 0.5, -0.2]])

    ar_expected = -domega * vr + uomega * ar

    relax.apply_acceleration_damping(a, v, 0.005, dt12, ar=ar, vr=vr)
    assert np.allclose(ar, ar_expected)


def test_dyrel_exponential_decay_to_static_equilibrium():
    """Verify that a 1-DOF spring-mass system with constant load settles to static equilibrium with /DYREL."""
    # System: m*x'' + c*x' + k*x = F_ext
    # Static equilibrium position: x_eq = F_ext / k
    m = 1.0
    k = 100.0
    f_ext = 50.0
    x_eq = f_ext / k  # 0.5 m
    t_end = 3.0
    dt = 0.001

    model = MockModel(1)
    model.mass[0] = m
    controls = EngineControls()
    controls.dyrel_active = True
    controls.dyrel_beta = 1.0
    controls.dyrel_period = 2.0 * np.pi / np.sqrt(k / m)  # 2*pi/10 = 0.6283 s

    relax = DynamicRelaxation(model, controls)

    x = 0.0
    v = np.zeros((1, 3))
    vr = np.zeros((1, 3))

    for step in range(int(t_end / dt)):
        t = step * dt
        # Net acceleration from external force and spring internal force
        a = (f_ext - k * x) / m
        v[0, 0] += a * dt
        relax.apply(t, dt, v, vr, model.mass, model.inertia)
        x += v[0, 0] * dt

    # System must have converged close to static equilibrium x_eq = 0.5
    assert x == pytest.approx(x_eq, abs=0.02)
    assert abs(v[0, 0]) < 0.05


# ============================================================================
# 3. /ADYREL Tests: Adaptive Beta and Timestep Reduction Factor
# ============================================================================

def test_adyrel_timestep_reduction_stability_factor():
    """Verify compute_timestep_reduction against dtnodarayl.F:168-188.

    Dampa3 = 2*beta / (1 + beta*dt12)
    BB = 0.5 * Dampa3 * dt^2
    dt_reduced = sqrt(BB^2 + dt^2) - BB
    """
    model = MockModel(1)
    controls = EngineControls()
    controls.adyrel_active = True
    controls.adyrel_freq_c = 0.0

    relax = DynamicRelaxation(model, controls)

    # 1. Zero beta: no reduction
    dt = 1e-4
    dt12 = 1e-4
    assert relax.compute_timestep_reduction(dt, dt12, beta=0.0) == pytest.approx(dt)

    # 2. Non-zero beta
    beta = 50.0
    dampa3 = 2.0 * beta / (1.0 + beta * dt12)
    bb = 0.5 * dampa3 * (dt ** 2)
    dt_red_expected = np.sqrt(bb ** 2 + dt ** 2) - bb

    dt_red_actual = relax.compute_timestep_reduction(dt, dt12, beta=beta)
    assert dt_red_actual == pytest.approx(dt_red_expected, rel=1e-15)
    assert dt_red_actual < dt  # must be reduced for stability

    # 3. Monotonic reduction: higher beta -> more reduction (smaller dt_reduced)
    dt_red_high_beta = relax.compute_timestep_reduction(dt, dt12, beta=100.0)
    assert dt_red_high_beta < dt_red_actual

    # 4. Zero dt input
    assert relax.compute_timestep_reduction(0.0, 0.0) == 0.0


def test_adyrel_adaptive_frequency_and_beta():
    """Verify that ADYREL adapts betate based on oscillating kinetic and internal energy."""
    model = MockModel(1)
    controls = EngineControls()
    controls.adyrel_active = True
    controls.adyrel_freq_c = 0.0

    relax = DynamicRelaxation(model, controls)
    dt = 0.001
    # Harmonic motion with T = 0.5 s -> f = 2.0 Hz
    T = 0.5
    omega = 2.0 * np.pi / T

    for cycle in range(1, 350):
        t = cycle * dt
        ke = float(10.0 * np.sin(omega * t) ** 2)
        ie = float(10.0 * np.cos(omega * t) ** 2)
        relax.update_adaptive_frequency(t, dt, cycle, e_int=ie, e_kin=ke)

    # Frequency adaptation must be triggered after cycle 200
    assert relax.adyrel_betate > 0.0
    # Stability timestep reduction with the adapted beta
    dt_red = relax.compute_timestep_reduction(dt, dt)
    assert 0.0 < dt_red <= dt


# ============================================================================
# 4. Static Relaxation Convergence Monitor Tests
# ============================================================================

def test_static_relaxation_convergence_monitor():
    """Verify check_convergence logic for static equilibrium."""
    # 1. High KE, Low IE -> Not converged
    assert check_convergence(e_kin=10.0, e_int=1.0, e_kin_max=10.0, tol=1e-3) is False

    # 2. Ratio E_kin / E_int < tol (1e-3) -> Converged
    assert check_convergence(e_kin=1e-4, e_int=10.0, e_kin_max=10.0, tol=1e-3) is True

    # 3. Ratio E_kin / E_kin_max < tol (1e-3) -> Converged
    assert check_convergence(e_kin=1e-4, e_int=0.0, e_kin_max=1.0, tol=1e-3) is True

    # 4. Method on DynamicRelaxation instance
    model = MockModel(1)
    controls = EngineControls()
    relax = DynamicRelaxation(model, controls)
    assert relax.check_convergence(e_kin=1e-4, e_int=10.0, e_kin_max=10.0) is True
    assert relax.check_convergence(e_kin=5.0, e_int=10.0, e_kin_max=10.0) is False


def test_vibrating_oscillator_reaches_convergence():
    """Verify that check_convergence accurately signals termination on a decaying oscillator."""
    m = 1.0
    k = 250.0
    f_ext = 50.0
    x_eq = f_ext / k  # 0.2 m
    dt = 0.001
    max_steps = 3000

    model = MockModel(1)
    model.mass[0] = m
    controls = EngineControls()
    controls.dyrel_active = True
    controls.dyrel_beta = 1.0
    controls.dyrel_period = 2.0 * np.pi / np.sqrt(k / m)

    relax = DynamicRelaxation(model, controls)

    x = 0.0
    v = np.zeros((1, 3))
    vr = np.zeros((1, 3))
    e_kin_max = 0.0
    converged_count = 0

    for step in range(max_steps):
        t = step * dt
        a = (f_ext - k * x) / m
        v[0, 0] += a * dt
        relax.apply(t, dt, v, vr, model.mass, model.inertia)
        x += v[0, 0] * dt

        e_kin = float(0.5 * m * (v[0, 0] ** 2))
        e_int = 0.5 * k * (x ** 2)
        e_kin_max = max(e_kin_max, e_kin)

        if relax.check_convergence(e_kin, e_int, e_kin_max, tol=1e-3):
            converged_count += 1
        else:
            converged_count = 0

        if converged_count >= 50:
            converged = True
            converged_step = step
            break

    assert converged is True
    assert converged_step < max_steps
    assert x == pytest.approx(x_eq, abs=0.02)
    assert abs(v[0, 0]) < 0.1
