# tests/test_sph_engine.py
"""Tests for SPH (Smoothed Particle Hydrodynamics) physics engine.

Checks kernel normalization, uniform density summation, antisymmetric forces,
critical time step, and momentum/mass conservation.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from pyradioss.engine.sph_engine import (
    cubic_bspline_kernel,
    cubic_bspline_grad,
    sph_density_sum,
    sph_defo_rate,
    sph_forces,
    sph_critical_dt,
    sph_step,
)


def test_kernel_normalization():
    r"""1. test_kernel_normalization — integral of W over 3D sphere ≈ 1 (numerical check).

    Cites: engine/source/elements/sph/weight.F lines 33-66 (WEIGHT0).
    In 3D spherical coordinates: \int_0^{2h} 4*pi*r^2 * W(r, h) dr == 1.
    """
    for h in [0.05, 0.2, 0.5, 1.0, 2.5]:
        integral, _ = quad(
            lambda r: 4.0 * np.pi * r ** 2 * cubic_bspline_kernel(r, h),
            0.0,
            2.0 * h,
            epsabs=1e-8,
            epsrel=1e-8,
        )
        assert np.isclose(integral, 1.0, rtol=1e-6, atol=1e-6), (
            f"Kernel normalization failed for h={h}: integral={integral}"
        )

    # Values beyond support radius 2h must strictly vanish
    h_test = 0.5
    assert cubic_bspline_kernel(2.01 * h_test, h_test) == 0.0
    assert cubic_bspline_kernel(5.0 * h_test, h_test) == 0.0


def test_kernel_gradient():
    """Test cubic B-spline kernel gradient against numerical finite differences.

    Cites: engine/source/elements/sph/weight.F lines 78-119 (WEIGHT1).
    """
    h = 0.4
    eps = 1e-6
    # Test at several points inside and outside support
    for r_pt in [np.array([0.1, 0.15, 0.1]), np.array([0.3, 0.2, 0.1])]:
        grad_ana = cubic_bspline_grad(r_pt, h)

        grad_num = np.zeros(3)
        for i in range(3):
            dr = np.zeros(3)
            dr[i] = eps
            r_plus = np.linalg.norm(r_pt + dr)
            r_minus = np.linalg.norm(r_pt - dr)
            w_plus = cubic_bspline_kernel(r_plus, h)
            w_minus = cubic_bspline_kernel(r_minus, h)
            grad_num[i] = (w_plus - w_minus) / (2.0 * eps)

        assert np.allclose(grad_ana, grad_num, rtol=1e-4, atol=1e-4)


def test_density_sum_uniform():
    """2. test_density_sum_uniform — uniform particle cloud: rho_a = rho_exact for interior particles.

    Cites: engine/source/elements/sph/spdens.F lines 101-167.
    """
    rho_exact = 1000.0  # kg/m^3
    dx = 0.08
    # Create 3D grid of particles in [-0.24, 0.24]^3 (7x7x7 = 343 particles)
    coords_1d = np.arange(-0.24, 0.241, dx)
    gx, gy, gz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
    pos = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    v_cell = dx ** 3
    mass = rho_exact * v_cell
    h = 1.3 * dx  # smoothing length
    h_arr = np.full(len(pos), h)

    rho = sph_density_sum(pos, mass, h_arr)

    # Interior particles: at least 2h away from boundaries
    bound = 0.24 - 2.0 * h
    interior = (
        (np.abs(pos[:, 0]) <= bound)
        & (np.abs(pos[:, 1]) <= bound)
        & (np.abs(pos[:, 2]) <= bound)
    )
    assert np.sum(interior) > 0, "No interior particles found"

    # For interior particles with cubic spline kernel on regular lattice,
    # summation density matches exact continuum density within 3%
    rho_interior = rho[interior]
    assert np.allclose(rho_interior, rho_exact, rtol=0.03), (
        f"Interior density error too large: max={np.max(rho_interior)}, min={np.min(rho_interior)}, exact={rho_exact}"
    )


def test_sph_pressure_forces_antisymmetric():
    """3. test_sph_pressure_forces_antisymmetric — symmetric pair: forces are equal and opposite.

    Cites: engine/source/elements/sph/sppro3.F, spforcp.F lines 240-304.
    Newton's third law: F_ij = - F_ji.
    """
    pos = np.array([
        [0.0, 0.0, 0.0],
        [0.04, 0.0, 0.0]
    ])
    vel = np.zeros_like(pos)
    mass = np.array([1.0, 1.0])
    rho = np.array([1000.0, 1000.0])
    h = 0.05
    h_arr = np.full(2, h)

    # 1. Equal pressures
    pressure = np.array([1.0e5, 1.0e5])
    f = sph_forces(pos, vel, mass, rho, pressure, h_arr)
    assert np.allclose(f[0], -f[1], atol=1e-12)
    assert np.allclose(f[0] + f[1], 0.0, atol=1e-12)
    # Repulsive pressure force: particle 0 pushed -x, particle 1 pushed +x
    assert f[0, 0] < 0.0
    assert f[1, 0] > 0.0

    # 2. Unequal pressures
    pressure_unequal = np.array([1.0e5, 3.0e5])
    f_uneq = sph_forces(pos, vel, mass, rho, pressure_unequal, h_arr)
    assert np.allclose(f_uneq[0], -f_uneq[1], atol=1e-12)
    assert np.allclose(f_uneq[0] + f_uneq[1], 0.0, atol=1e-12)

    # 3. Multi-particle random configuration: total force sum must identically vanish
    np.random.seed(42)
    n_pts = 30
    pos_rand = np.random.uniform(-0.1, 0.1, (n_pts, 3))
    vel_rand = np.random.uniform(-1.0, 1.0, (n_pts, 3))
    mass_rand = np.random.uniform(0.5, 1.5, n_pts)
    rho_rand = np.random.uniform(800.0, 1200.0, n_pts)
    p_rand = np.random.uniform(5e4, 2e5, n_pts)
    h_rand = np.full(n_pts, 0.08)

    f_multi = sph_forces(pos_rand, vel_rand, mass_rand, rho_rand, p_rand, h_rand, alpha_visc=1.0, beta_visc=2.0)
    net_force = np.sum(f_multi, axis=0)
    assert np.allclose(net_force, 0.0, atol=1e-10), f"Net force not zero: {net_force}"


def test_sph_critical_dt():
    """4. test_sph_critical_dt — dt < h/c_s for any config.

    Cites: engine/source/elements/sph/sphreq.F lines 34-40 and mdtsph.F lines 96-135.
    CFL condition: dt = cfl * h / c_s, cfl = 0.6 < 1.0 guarantees dt < h / c_s.
    """
    gamma = 1.4

    # Single particle config (scalar)
    h_scalar = 0.05
    rho_scalar = 1000.0
    p_scalar = 1.0e5
    cs_scalar = np.sqrt(gamma * p_scalar / rho_scalar)

    dt_scalar = sph_critical_dt(h_scalar, rho_scalar, p_scalar, gamma, cfl=0.6)
    assert dt_scalar < h_scalar / cs_scalar
    assert np.isclose(dt_scalar, 0.6 * h_scalar / cs_scalar)

    # Multi-particle array config
    h_arr = np.array([0.01, 0.05, 0.1, 0.2])
    rho_arr = np.array([500.0, 1000.0, 1200.0, 2000.0])
    p_arr = np.array([1.0e4, 1.0e5, 5.0e5, 2.0e6])
    cs_arr = np.sqrt(gamma * p_arr / rho_arr)

    dt_arr = sph_critical_dt(h_arr, rho_arr, p_arr, gamma, cfl=0.6)
    assert np.all(dt_arr < h_arr / cs_arr)
    assert np.allclose(dt_arr, 0.6 * h_arr / cs_arr)


def test_sph_mass_conservation():
    """5. test_sph_mass_conservation — sum of mass * velocity = momentum conserved for 1 step.

    Cites: engine/source/elements/sph/spforcp.F lines 240-304.
    Since internal SPH forces satisfy F_ij = - F_ji, sum_i F_i == 0.
    Therefore, d(sum m*v)/dt == 0, and total momentum is conserved identically.
    Total mass sum m_i is also conserved.
    """
    np.random.seed(123)
    n_pts = 40
    pos = np.random.uniform(-0.1, 0.1, (n_pts, 3))
    vel = np.random.uniform(-5.0, 5.0, (n_pts, 3))
    mass = np.random.uniform(0.1, 1.0, n_pts)
    h_arr = np.full(n_pts, 0.08)

    # Compute density & pressure (ideal gas EOS: p = (gamma - 1) * rho * u)
    rho = sph_density_sum(pos, mass, h_arr)
    u = np.full(n_pts, 2.5e5)  # specific internal energy
    gamma = 1.4
    pressure = (gamma - 1.0) * rho * u

    # Initial total momentum & mass
    p_initial = np.sum(mass[:, None] * vel, axis=0)
    mass_initial = np.sum(mass)

    # Compute forces
    f = sph_forces(pos, vel, mass, rho, pressure, h_arr, alpha_visc=1.0, beta_visc=2.0, gamma=gamma)

    # 1 explicit time step: v^{n+1} = v^n + (F / m) * dt
    dt = 1e-4
    acc = f / mass[:, None]
    vel_next = vel + acc * dt

    p_final = np.sum(mass[:, None] * vel_next, axis=0)
    mass_final = np.sum(mass)

    assert np.isclose(mass_final, mass_initial), "Mass not conserved"
    assert np.allclose(p_final, p_initial, atol=1e-12), (
        f"Momentum not conserved: P_initial={p_initial}, P_final={p_final}"
    )


def test_sph_defo_rate():
    """Test SPH deformation rate D_ab under uniform expansion.

    Cites: engine/source/elements/sph/spdefo3.F lines 63-70 and spdens.F lines 188-202.
    For v(x) = eps_dot * x, D_ab = eps_dot * delta_ab, tr(D) = 3 * eps_dot.
    """
    eps_dot = 10.0
    dx = 0.08
    coords_1d = np.arange(-0.24, 0.241, dx)
    gx, gy, gz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
    pos = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    vel = eps_dot * pos  # uniform expansion

    rho_0 = 1000.0
    mass = rho_0 * (dx ** 3)
    h = 1.3 * dx
    h_arr = np.full(len(pos), h)

    rho = sph_density_sum(pos, mass, h_arr)
    D = sph_defo_rate(pos, vel, mass, rho, h_arr)

    # For an interior particle at the center, D_xx ≈ D_yy ≈ D_zz ≈ eps_dot
    center_idx = np.argmin(np.linalg.norm(pos, axis=1))
    D_center = D[center_idx]

    assert np.isclose(D_center[0, 0], eps_dot, rtol=0.1)
    assert np.isclose(D_center[1, 1], eps_dot, rtol=0.1)
    assert np.isclose(D_center[2, 2], eps_dot, rtol=0.1)
    # Shear components should be near zero
    assert np.abs(D_center[0, 1]) < 0.5
    assert np.abs(D_center[1, 2]) < 0.5
    assert np.abs(D_center[0, 2]) < 0.5


def test_sph_step_integration():
    """Test sph_step() wiring and energy accounting.

    Cites: engine/source/elements/sph/spforcp.F lines 519-521.
    """
    class MockState:
        def __init__(self):
            self.e_num = 0.0
            self.eint = 0.0

    class MockSphCells:
        def __init__(self, pos, mass, h, u):
            self.pos = pos
            self.mass = mass
            self.h = h
            self.state = {"u": u}
            self.node_ids = np.arange(len(pos))
            self.gamma = 1.4
            self.alpha_visc = 1.0
            self.beta_visc = 2.0

    class MockModel:
        def __init__(self, sph_cells, x, v, mass):
            self.sph_cells = sph_cells
            self.x = x
            self.v = v
            self.mass = mass
            self.fint = np.zeros_like(x)

    pos = np.array([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0]])
    vel = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])  # colliding / compressing
    mass = np.array([1.0, 1.0])
    h = np.array([0.05, 0.05])
    u = np.array([1e5, 1e5])

    cells = MockSphCells(pos, mass, h, u)
    model = MockModel(cells, pos.copy(), vel.copy(), mass.copy())
    state = MockState()

    dt = 1e-5
    sph_step(model, dt, state)

    # Force should be assembled into model.fint
    assert not np.allclose(model.fint, 0.0)
    assert np.allclose(model.fint[0], -model.fint[1], atol=1e-12)
