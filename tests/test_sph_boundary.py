"""Unit tests for /SPH/INFLOW and /SPH/OUTFLOW boundary conditions.

Upstream Fortran references:
- ``starter/source/loads/sph/hm_read_sphio.F`` (Subroutine HM_READ_SPHIO)
- ``engine/source/elements/sph/sponof1.F`` (Subroutine SPONOF1)
- ``engine/source/elements/sph/sponof2.F`` (Subroutine SPONOF2)

Verifies:
1. SPH inflow geometry setup, plane orientation, and particle lattice spacing.
2. Continuous mass injection rate m_dot = rho * A * v_in.
3. Periodic particle injection at intervals dt_inject = dx / v_in.
4. Correct particle initial state (position, velocity vector, density, energy, mass).
5. SPH outflow plane signed distance and crossing detection.
6. Outflow particle deactivation and mass/momentum removal.
7. Coupled SphBoundaryManager simulation with simultaneous inflow and outflow.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.engine.sph_boundary import (
    SphBoundaryManager,
    SphInflow,
    SphInflowParams,
    SphOutflow,
    SphOutflowParams,
    build_sph_inflow,
    build_sph_outflow,
)


def test_inflow_params_and_geometry():
    """Verify geometry, area, and continuous mass injection rate calculations."""
    # 1. Rectangular planar inflow
    inflow_rect = SphInflow(
        origin=(0.0, 0.0, 0.0),
        normal=(1.0, 0.0, 0.0),
        width=0.4,
        height=0.2,
        density=1000.0,
        velocity=5.0,
        particle_spacing=0.05,
    )
    assert inflow_rect.area == pytest.approx(0.4 * 0.2, rel=1e-6)
    assert inflow_rect.speed == pytest.approx(5.0, rel=1e-6)
    # Theoretical continuous mass flow rate m_dot = rho * A * v
    expected_mdot = 1000.0 * (0.4 * 0.2) * 5.0  # 400.0 kg/s
    assert inflow_rect.mass_flow_rate == pytest.approx(expected_mdot, rel=1e-6)
    # Injection interval dt_inject = dx / v = 0.05 / 5.0 = 0.01 s
    assert inflow_rect.dt_inject == pytest.approx(0.01, rel=1e-6)

    # 2. Circular nozzle inflow
    radius = 0.1
    inflow_circ = SphInflow(
        origin=(1.0, 2.0, 3.0),
        normal=(0.0, 0.0, 1.0),
        radius=radius,
        density=800.0,
        velocity=10.0,
        particle_spacing=0.02,
    )
    expected_circ_area = math.pi * (radius ** 2)
    assert inflow_circ.area == pytest.approx(expected_circ_area, rel=1e-6)
    expected_circ_mdot = 800.0 * expected_circ_area * 10.0
    assert inflow_circ.mass_flow_rate == pytest.approx(expected_circ_mdot, rel=1e-6)


def test_inflow_particle_creation_spacing_and_properties():
    """Verify generated SPH particles have correct coordinates, velocity, mass, and energy."""
    dx = 0.05
    rho0 = 1000.0
    e0 = 250.0
    v_mag = 4.0
    normal = (1.0, 0.0, 0.0)

    inflow = SphInflow(
        origin=(2.0, 0.0, 0.0),
        normal=normal,
        width=0.2,
        height=0.2,
        density=rho0,
        internal_energy=e0,
        velocity=v_mag,
        particle_spacing=dx,
    )

    layer = inflow.generate_single_layer(time=0.0, offset_distance=0.025)
    pos = layer["pos"]
    vel = layer["vel"]
    mass = layer["mass"]
    rho = layer["rho"]
    energy = layer["energy"]
    h = layer["h"]

    n_pts = len(pos)
    assert n_pts == 16  # 4 x 4 grid (0.2 / 0.05 = 4)

    # Position check: x should be origin.x + offset = 2.0 + 0.025 = 2.025
    assert np.allclose(pos[:, 0], 2.025, atol=1e-6)

    # Velocity check: aligned with flow normal
    expected_vel = np.array([v_mag, 0.0, 0.0])
    np.testing.assert_allclose(vel, np.broadcast_to(expected_vel, (n_pts, 3)), atol=1e-6)

    # Mass check: m_p = rho * dx^3
    expected_m = rho0 * (dx ** 3)
    assert np.allclose(mass, expected_m, rtol=1e-6)
    assert np.allclose(rho, rho0, rtol=1e-6)
    assert np.allclose(energy, e0, rtol=1e-6)
    assert np.allclose(h, 1.2 * dx, rtol=1e-6)

    # Verify spacing between nearest neighbor particles in the plane
    dy = np.diff(np.unique(np.round(pos[:, 1], 6)))
    assert np.allclose(dy, dx, atol=1e-6)
    dz = np.diff(np.unique(np.round(pos[:, 2], 6)))
    assert np.allclose(dz, dx, atol=1e-6)


def test_inflow_periodic_injection_and_mass_flow_rate():
    """Verify discrete periodic injection matches continuous mass flow rate over time."""
    dx = 0.02
    v_in = 10.0
    rho = 1000.0
    width = 0.1
    height = 0.1

    inflow = SphInflow(
        origin=(0.0, 0.0, 0.0),
        normal=(0.0, 1.0, 0.0),
        width=width,
        height=height,
        density=rho,
        velocity=v_in,
        particle_spacing=dx,
    )

    # dt_inject = dx / v = 0.02 / 10.0 = 0.002 s
    dt_inj = inflow.dt_inject
    assert dt_inj == pytest.approx(0.002, rel=1e-6)

    # Step simulation with smaller dt = 0.0005 s (4 steps per layer)
    dt_sim = 0.0005
    total_time = 0.05  # 50 ms -> 25 injection cycles
    n_steps = int(total_time / dt_sim)

    injected_mass = 0.0
    injected_count = 0

    for step in range(n_steps):
        t_cur = (step + 1) * dt_sim
        batch = inflow.update(dt=dt_sim, time=t_cur)
        if batch is not None:
            injected_count += len(batch["pos"])
            injected_mass += float(np.sum(batch["mass"]))

    # Expected continuous mass: M = m_dot * total_time = (rho * A * v) * T
    theoretical_mass = inflow.mass_flow_rate * total_time

    # Discrete mass injection must equal theoretical mass within 1 layer tolerance
    single_layer_mass = inflow.particles_per_layer * inflow.mass_particle
    assert abs(injected_mass - theoretical_mass) <= single_layer_mass
    assert injected_mass == pytest.approx(theoretical_mass, rel=0.05)
    assert injected_count > 0


def test_inflow_time_dependent_velocity():
    """Verify time-dependent velocity function v_in(t) is evaluated correctly."""
    # Velocity ramping linearly: v(t) = 2.0 + 100.0 * t
    def vel_func(t: float) -> np.ndarray:
        speed = 2.0 + 100.0 * t
        return np.array([0.0, speed, 0.0])

    inflow = SphInflow(
        origin=(0.0, 0.0, 0.0),
        normal=(0.0, 1.0, 0.0),
        width=0.1,
        height=0.1,
        velocity=vel_func,
        particle_spacing=0.05,
    )

    v0 = inflow.get_velocity_vector(time=0.0)
    assert np.allclose(v0, [0.0, 2.0, 0.0])

    v1 = inflow.get_velocity_vector(time=0.02)
    assert np.allclose(v1, [0.0, 4.0, 0.0])


def test_outflow_plane_signed_distance_and_filtering():
    """Verify outflow plane detects particles crossing the boundary and deactivates them."""
    # Outflow plane at x = 5.0 with normal (1, 0, 0)
    outflow = SphOutflow(
        point=(5.0, 0.0, 0.0),
        normal=(1.0, 0.0, 0.0),
        distance_buffer=0.0,
    )

    pos = np.array([
        [2.0, 0.0, 0.0],   # upstream (d = -3.0) -> inside domain
        [4.9, 1.0, 0.0],   # upstream (d = -0.1) -> inside domain
        [5.0, 0.0, 0.0],   # exactly on plane (d = 0.0) -> not crossed
        [5.1, 0.0, 0.0],   # crossed (d = +0.1) -> outside domain
        [6.5, -1.0, 2.0],  # crossed (d = +1.5) -> outside domain
    ])
    mass = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    dists = outflow.compute_signed_distance(pos)
    np.testing.assert_allclose(dists, [-3.0, -0.1, 0.0, 0.1, 1.5], atol=1e-6)

    crossed, active = outflow.check_particles(pos, mass=mass)
    np.testing.assert_array_equal(crossed, [False, False, False, True, True])
    np.testing.assert_array_equal(active, [True, True, True, False, False])
    assert outflow.total_deactivated_particles == 2
    assert outflow.total_deactivated_mass == 2.0

    # Test filtering dictionary
    particles = {
        "pos": pos,
        "vel": np.ones((5, 3)),
        "mass": mass,
    }
    filtered = outflow.filter_particles(particles)
    assert len(filtered["pos"]) == 3
    np.testing.assert_allclose(filtered["pos"], pos[:3])


def test_outflow_with_distance_buffer():
    """Verify distance_buffer parameter creates a buffer zone before deactivation."""
    # Outflow plane at z = 10.0 with buffer = 0.5
    outflow = SphOutflow(
        point=(0.0, 0.0, 10.0),
        normal=(0.0, 0.0, 1.0),
        distance_buffer=0.5,
    )

    pos = np.array([
        [0.0, 0.0, 9.5],   # d = -0.5 <= 0.5 -> keep
        [0.0, 0.0, 10.2],  # d = 0.2 <= 0.5 -> keep in buffer
        [0.0, 0.0, 10.6],  # d = 0.6 > 0.5 -> deactive
    ])
    crossed, active = outflow.check_particles(pos)
    assert crossed[0] is False or crossed[0] == 0
    assert crossed[1] is False or crossed[1] == 0
    assert crossed[2] is True or crossed[2] == 1


def test_sph_boundary_manager_coupled_simulation():
    """Verify composite SphBoundaryManager with both inflow and outflow simultaneously."""
    manager = SphBoundaryManager()

    # Inflow at x = 0.0 pointing +x, speed 10.0 m/s, dx = 0.05 m
    inflow = manager.add_inflow({
        "origin": (0.0, 0.0, 0.0),
        "normal": (1.0, 0.0, 0.0),
        "width": 0.1,
        "height": 0.1,
        "velocity": 10.0,
        "density": 1000.0,
        "particle_spacing": 0.05,
    })

    # Outflow at x = 0.5 m pointing +x
    outflow = manager.add_outflow({
        "point": (0.5, 0.0, 0.0),
        "normal": (1.0, 0.0, 0.0),
    })

    particles = {
        "pos": np.empty((0, 3)),
        "vel": np.empty((0, 3)),
        "mass": np.empty(0),
    }

    dt = 0.0025  # half injection cycle (dt_inj = 0.05 / 10 = 0.005 s)
    total_steps = 100  # 0.25 seconds total

    for step in range(total_steps):
        t_now = step * dt
        # 1. Apply boundary step (inflow injection + outflow removal)
        particles, stats = manager.step(particles, dt=dt, time=t_now)

        # 2. Advect active particles: x_new = x + v * dt
        if len(particles["pos"]) > 0:
            particles["pos"] += particles["vel"] * dt

    # After particles traverse from x=0 to x=0.5 (time = 0.5 / 10 = 0.05 s),
    # steady state is reached where inflow rate equals outflow rate!
    assert inflow.total_injected_particles > 0
    assert outflow.total_deactivated_particles > 0
    # Injected particles must be greater than remaining in channel
    assert inflow.total_injected_particles > len(particles["pos"])
    # All remaining particles must be inside the channel [0.0, 0.5]
    assert np.all(particles["pos"][:, 0] <= 0.5 + 1e-5)


def test_builder_factories():
    """Verify build_sph_inflow and build_sph_outflow factory functions."""
    inf = build_sph_inflow(width=0.5, height=0.5, velocity=2.0)
    assert isinstance(inf, SphInflow)
    assert inf.speed == pytest.approx(2.0, rel=1e-6)

    outf = build_sph_outflow(point=(1.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))
    assert isinstance(outf, SphOutflow)
    assert np.allclose(outf.point, [1.0, 0.0, 0.0])
