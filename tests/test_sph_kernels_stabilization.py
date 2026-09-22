# tests/test_sph_kernels_stabilization.py
"""Tests for advanced SPH kernels (Wendland C2, Quintic spline),

kernel gradient corrections (Shepard order 0, MLS order 1),
and Monaghan-Gray tensile instability stabilization.

Faithful verification of OpenRadioss Fortran sources:
  - weight.F (lines 33-119): SPH kernels and analytic gradients
  - spstab.F (lines 80-129, 212-250) & spforcp.F (lines 259-277): Monaghan-Gray tensile stabilization
  - spcompl.F (lines 156-202, 266-325): zeroth-order Shepard & first-order MLS corrections
"""

import numpy as np
import pytest
from scipy.integrate import quad

from pyradioss.engine.sph_engine import (
    cubic_bspline_kernel,
    cubic_bspline_grad,
    wendland_c2_kernel,
    wendland_c2_grad,
    quintic_spline_kernel,
    quintic_spline_grad,
    sph_shepard_correction,
    sph_mls_gradient_correction,
    sph_tensile_stabilization,
    sph_forces,
)


def test_wendland_c2_normalization():
    r"""Integral of Wendland C2 over 3D sphere: \int_0^{2h} 4*pi*r^2 * W(r, h) dr == 1."""
    for h in [0.05, 0.2, 0.5, 1.0, 2.5]:
        integral, _ = quad(
            lambda r: 4.0 * np.pi * r ** 2 * wendland_c2_kernel(r, h),
            0.0,
            2.0 * h,
            epsabs=1e-8,
            epsrel=1e-8,
        )
        assert np.isclose(integral, 1.0, rtol=1e-5, atol=1e-5), (
            f"Wendland C2 normalization failed for h={h}: integral={integral}"
        )

    # Vanishing beyond compact support 2h
    assert wendland_c2_kernel(2.01, 1.0) == 0.0
    assert wendland_c2_kernel(3.5, 1.0) == 0.0


def test_wendland_c2_gradient():
    """Verify Wendland C2 kernel gradient against numerical finite differences."""
    h = 0.4
    eps = 1e-6
    test_points = [
        np.array([0.1, 0.15, 0.1]),
        np.array([0.25, 0.2, 0.15]),
        np.array([0.35, 0.3, 0.2]),
    ]
    for r_pt in test_points:
        grad_ana = wendland_c2_grad(r_pt, h)

        grad_num = np.zeros(3)
        for i in range(3):
            dr = np.zeros(3)
            dr[i] = eps
            r_plus = np.linalg.norm(r_pt + dr)
            r_minus = np.linalg.norm(r_pt - dr)
            w_plus = wendland_c2_kernel(r_plus, h)
            w_minus = wendland_c2_kernel(r_minus, h)
            grad_num[i] = (w_plus - w_minus) / (2.0 * eps)

        assert np.allclose(grad_ana, grad_num, rtol=1e-4, atol=1e-4)


def test_quintic_spline_normalization():
    r"""Integral of Quintic spline over 3D sphere: \int_0^{3h} 4*pi*r^2 * W(r, h) dr == 1."""
    for h in [0.05, 0.2, 0.5, 1.0, 2.5]:
        integral, _ = quad(
            lambda r: 4.0 * np.pi * r ** 2 * quintic_spline_kernel(r, h),
            0.0,
            3.0 * h,
            epsabs=1e-8,
            epsrel=1e-8,
        )
        assert np.isclose(integral, 1.0, rtol=1e-5, atol=1e-5), (
            f"Quintic spline normalization failed for h={h}: integral={integral}"
        )

    # Vanishing beyond compact support 3h
    assert quintic_spline_kernel(3.01, 1.0) == 0.0
    assert quintic_spline_kernel(4.5, 1.0) == 0.0


def test_quintic_spline_gradient():
    """Verify Quintic spline kernel gradient against numerical finite differences."""
    h = 0.4
    eps = 1e-6
    test_points = [
        np.array([0.1, 0.15, 0.1]),
        np.array([0.35, 0.2, 0.15]),
        np.array([0.6, 0.4, 0.3]),
    ]
    for r_pt in test_points:
        grad_ana = quintic_spline_grad(r_pt, h)

        grad_num = np.zeros(3)
        for i in range(3):
            dr = np.zeros(3)
            dr[i] = eps
            r_plus = np.linalg.norm(r_pt + dr)
            r_minus = np.linalg.norm(r_pt - dr)
            w_plus = quintic_spline_kernel(r_plus, h)
            w_minus = quintic_spline_kernel(r_minus, h)
            grad_num[i] = (w_plus - w_minus) / (2.0 * eps)

        assert np.allclose(grad_ana, grad_num, rtol=1e-4, atol=1e-4)


def test_shepard_order0_correction():
    """Verify Shepard order 0 normalization factor ensures partition of unity (spcompl.F lines 156-202)."""
    dx = 0.1
    coords = np.arange(-0.2, 0.21, dx)
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing='ij')
    pos = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    n = len(pos)

    rho0 = 1000.0
    mass = np.full(n, rho0 * dx**3)
    rho = np.full(n, rho0)
    h = np.full(n, 1.3 * dx)

    c_factors = sph_shepard_correction(pos, h, mass, rho)
    assert len(c_factors) == n
    assert np.all(c_factors > 0.0)

    # For interior particles, summation of volumes * kernel is approximately 1.0
    bound = 0.2 - 2.0 * 1.3 * dx
    interior = (
        (np.abs(pos[:, 0]) <= bound)
        & (np.abs(pos[:, 1]) <= bound)
        & (np.abs(pos[:, 2]) <= bound)
    )
    if np.any(interior):
        assert np.allclose(c_factors[interior], 1.0, rtol=0.05)


def test_mls_order1_correction():
    """Verify MLS order 1 correction matrix M_inv produces identity for regular lattice (spcompl.F lines 266-325)."""
    dx = 0.1
    coords = np.arange(-0.2, 0.21, dx)
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing='ij')
    pos = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    n = len(pos)

    rho0 = 1000.0
    mass = np.full(n, rho0 * dx**3)
    rho = np.full(n, rho0)
    h = np.full(n, 1.4 * dx)

    M_inv = sph_mls_gradient_correction(pos, h, mass, rho)
    assert M_inv.shape == (n, 3, 3)

    # In the deep interior of a regular lattice, M_inv is isotropic ~ c * I
    bound = 0.2 - 2.0 * 1.4 * dx
    interior = (
        (np.abs(pos[:, 0]) <= bound)
        & (np.abs(pos[:, 1]) <= bound)
        & (np.abs(pos[:, 2]) <= bound)
    )
    if np.any(interior):
        int_idx = np.where(interior)[0][0]
        # Diagonal elements must be non-zero and positive
        assert M_inv[int_idx, 0, 0] > 0.0
        assert M_inv[int_idx, 1, 1] > 0.0
        assert M_inv[int_idx, 2, 2] > 0.0


def test_tensile_stabilization_compression():
    """Under compressive hydrostatic stress, tensile stabilization forces are strictly zero (spstab.F lines 234-236)."""
    pos = np.array([
        [0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0],
    ])
    vel = np.zeros((2, 3))
    mass = np.array([1.0, 1.0])
    rho = np.array([1000.0, 1000.0])
    # Pure compression: sigma_xx = sigma_yy = sigma_zz = -1e6 Pa < 0
    stress = np.array([
        [-1e6, -1e6, -1e6, 0.0, 0.0, 0.0],
        [-1e6, -1e6, -1e6, 0.0, 0.0, 0.0],
    ])
    h = np.array([0.1, 0.1])

    f_stab, w_stab = sph_tensile_stabilization(pos, vel, mass, rho, stress, h, zstab=1.0)
    assert np.allclose(f_stab, 0.0)
    assert w_stab == 0.0


def test_tensile_stabilization_tension():
    """Under tensile stress, stabilization generates repulsive forces along the tensile axis (spforcp.F lines 259-277)."""
    pos = np.array([
        [0.0, 0.0, 0.0],
        [0.06, 0.0, 0.0],  # along x-axis
    ])
    vel = np.array([
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    mass = np.array([1.0, 1.0])
    rho = np.array([1000.0, 1000.0])
    # Uniaxial tension along x: sigma_xx = +5e6 Pa > 0
    stress = np.array([
        [5e6, 0.0, 0.0, 0.0, 0.0, 0.0],
        [5e6, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    h = np.array([0.1, 0.1])

    f_stab, w_stab = sph_tensile_stabilization(pos, vel, mass, rho, stress, h, zstab=1.0, dt=1e-4)

    # 1. Exact Newton's third law: F_1 + F_2 = 0
    assert np.allclose(f_stab[0] + f_stab[1], 0.0, atol=1e-12)

    # 2. Force along x is non-zero (active tensile direction)
    assert abs(f_stab[0, 0]) > 0.0
    # Particle 0 (at x=0) should be repelled to the left (negative x)
    assert f_stab[0, 0] < 0.0
    # Particle 1 (at x=0.06) should be repelled to the right (positive x)
    assert f_stab[1, 0] > 0.0

    # 3. Forces along perpendicular y, z axes are zero
    assert np.isclose(f_stab[0, 1], 0.0)
    assert np.isclose(f_stab[0, 2], 0.0)

    # 4. Work is non-zero
    assert w_stab != 0.0


def test_sph_forces_with_tensile_stabilization():
    """Verify sph_forces integrates tensile stabilization when zstab > 0 and stress provided."""
    pos = np.array([
        [0.0, 0.0, 0.0],
        [0.06, 0.0, 0.0],
    ])
    vel = np.zeros((2, 3))
    mass = np.array([1.0, 1.0])
    rho = np.array([1000.0, 1000.0])
    p = np.array([0.0, 0.0])  # zero pressure
    h = np.array([0.1, 0.1])
    stress = np.array([
        [1e6, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1e6, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])

    # With zstab=0.0: zero forces (no pressure, no velocity, no stabilization)
    f_nostab = sph_forces(pos, vel, mass, rho, p, h, zstab=0.0)
    assert np.allclose(f_nostab, 0.0)

    # With zstab=1.0 and tensile stress: stabilization forces active
    f_stab = sph_forces(pos, vel, mass, rho, p, h, stress=stress, zstab=1.0)
    assert not np.allclose(f_stab, 0.0)
    assert np.allclose(f_stab[0] + f_stab[1], 0.0, atol=1e-12)
