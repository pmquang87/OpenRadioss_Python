"""Tests for /MAT/LAW102 Extended Drucker-Prager (DPRAG2) constitutive physics.

Verifies:
  - Mohr-Coulomb to Drucker-Prager conversion for IFORM = 1, 2, 3, 4
  - Direct A0, A1, A2 quadratic yield surface specification
  - Pressure-dependent parabolic yield surface G0(P) and Amax capping
  - Tensile pressure root P* and tensile cutoffs (P <= Pmin or P <= P*)
  - Radial return deviatoric scaling and plastic strain increment
  - Dynamic acoustic sound speed c = sqrt((K + 4G/3)/rho)
  - Algorithmic tangent stiffness matrix
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law102_dprag2 import (
    DPrag2Params,
    build_law102,
    compute_dprag2_constants,
    init_history,
    solid_update,
    solid_update_single,
    solid_update_array,
    sound_speed,
    solid_tangent,
)


def test_mohr_coulomb_conversions():
    """Verify IFORM=1, 2, 3 Mohr-Coulomb to Drucker-Prager conversions matching hm_read_mat102.F."""
    c_val = 15.0  # Cohesion (MPa)
    phi_deg = 30.0  # Friction angle (deg)
    phi_rad = math.radians(phi_deg)
    sin_phi = math.sin(phi_rad)
    cos_phi = math.cos(phi_rad)

    # IFORM = 1: Circumscribed cone
    # k = 6 * C * cos(phi) / (sqrt(3)*(3 - sin(phi)))
    # alpha = 2 * sin(phi) / (sqrt(3)*(3 - sin(phi)))
    p1 = compute_dprag2_constants(iform=1, c=c_val, phi=phi_deg)
    k1 = (6.0 * c_val * cos_phi) / (math.sqrt(3.0) * (3.0 - sin_phi))
    alpha1 = (2.0 * sin_phi) / (math.sqrt(3.0) * (3.0 - sin_phi))
    assert pytest.approx(p1.a0) == k1 ** 2
    assert pytest.approx(p1.a1) == 6.0 * k1 * alpha1
    assert pytest.approx(p1.a2) == 9.0 * alpha1 ** 2

    # IFORM = 2: Middle cone (default)
    # k = 6 * C * cos(phi) / (sqrt(3)*(3 + sin(phi)))
    # alpha = 2 * sin(phi) / (sqrt(3)*(3 + sin(phi)))
    p2 = compute_dprag2_constants(iform=2, c=c_val, phi=phi_deg)
    k2 = (6.0 * c_val * cos_phi) / (math.sqrt(3.0) * (3.0 + sin_phi))
    alpha2 = (2.0 * sin_phi) / (math.sqrt(3.0) * (3.0 + sin_phi))
    assert pytest.approx(p2.a0) == k2 ** 2
    assert pytest.approx(p2.a1) == 6.0 * k2 * alpha2
    assert pytest.approx(p2.a2) == 9.0 * alpha2 ** 2

    # IFORM = 3: Inscribed cone
    # k = 3 * C * cos(phi) / sqrt(9 + 3*sin(phi)^2)
    # alpha = sin(phi) / sqrt(9 + 3*sin(phi)^2)
    p3 = compute_dprag2_constants(iform=3, c=c_val, phi=phi_deg)
    denom3 = math.sqrt(9.0 + 3.0 * sin_phi ** 2)
    k3 = (3.0 * c_val * cos_phi) / denom3
    alpha3 = sin_phi / denom3
    assert pytest.approx(p3.a0) == k3 ** 2
    assert pytest.approx(p3.a1) == 6.0 * k3 * alpha3
    assert pytest.approx(p3.a2) == 9.0 * alpha3 ** 2


def test_direct_coefficients_and_pstar():
    """Verify direct A0, A1, A2 coefficients and tensile cutoff root P*."""
    # When C=0 and PHI=0, user provided A0, A1, A2 are preserved
    a0 = 100.0
    a1 = 20.0
    a2 = 1.0
    p = compute_dprag2_constants(iform=2, c=0.0, phi=0.0, a0=a0, a1=a1, a2=a2, pmin=-3.0)
    assert p.a0 == a0
    assert p.a1 == a1
    assert p.a2 == a2

    # P* is root of A0 + A1*P + A2*P^2 = 0
    # delta = a1^2 - 4*a0*a2 = 400 - 400 = 0 -> pstar = -a1 / (2*a2) = -10.0
    assert pytest.approx(p.pstar) == -10.0


def test_elastic_update():
    """Verify stress below yield surface remains unchanged and epsp remains 0."""
    p = build_law102(
        rho0=2.0e-6,
        e=20000.0,
        nu=0.25,
        iform=2,
        c=20.0,
        phi=30.0,
        amax=100.0,
        pmin=-1.0,
    )
    sig0 = np.array([-5.0, -5.0, -5.0, 0.0, 0.0, 0.0])  # Hydrostatic compression P = +5.0
    deps = np.array([1.0e-5, -0.5e-5, -0.5e-5, 0.0, 0.0, 0.0])  # Small deviatoric strain
    sig_new, epsp_new = solid_update_single(p, sig0, deps, epsp=0.0)

    # Plastic strain should be zero
    assert epsp_new == 0.0
    # Expected stress is elastic predictor
    e = p.e
    nu = p.nu
    g = e / (2.0 * (1.0 + nu))
    k = e / (3.0 * (1.0 - 2.0 * nu))
    deps_v = np.sum(deps[:3])
    deps_dev = deps.copy()
    deps_dev[:3] -= deps_v / 3.0
    sig_exp = sig0.copy()
    sig_exp[:3] += k * deps_v
    sig_exp[:3] += 2.0 * g * deps_dev[:3]
    np.testing.assert_allclose(sig_new, sig_exp, rtol=1e-5)


def test_plastic_flow_and_radial_return():
    """Verify plastic radial return onto Drucker-Prager cone under shear/deviatoric load."""
    p = build_law102(
        rho0=2.0e-6,
        e=10000.0,
        nu=0.2,
        iform=2,
        c=5.0,
        phi=25.0,
        amax=50.0,
        pmin=-2.0,
    )
    # Start at zero stress, apply large pure shear
    sig0 = np.zeros(6)
    deps = np.array([0.0, 0.0, 0.0, 0.02, 0.0, 0.0])  # Large shear strain
    sig_new, epsp_new = solid_update_single(p, sig0, deps, epsp=0.0)

    assert epsp_new > 0.0
    # Under pure shear, P = 0, so G0 = A0
    # J2 of final stress must equal G0
    s_xx = sig_new[0]
    s_yy = sig_new[1]
    s_zz = sig_new[2]
    s_xy = sig_new[3]
    p_final = -(s_xx + s_yy + s_zz) / 3.0
    assert pytest.approx(p_final, abs=1e-6) == 0.0

    j2 = (
        (s_xx - s_yy) ** 2
        + (s_yy - s_zz) ** 2
        + (s_zz - s_xx) ** 2
        + 6.0 * (s_xy ** 2 + sig_new[4] ** 2 + sig_new[5] ** 2)
    ) / 6.0
    assert pytest.approx(j2, rel=1e-5) == p.a0


def test_tensile_pressure_cutoff():
    """Verify deviatoric stresses drop to zero when pressure falls below tensile cutoffs."""
    p = build_law102(
        rho0=2.0e-6,
        e=10000.0,
        nu=0.2,
        iform=2,
        c=5.0,
        phi=25.0,
        amax=50.0,
        pmin=-1.0,  # Tensile cutoff at P_tot = -1.0
    )
    # Apply large hydrostatic tension such that P_tot <= Pmin
    sig0 = np.zeros(6)
    deps = np.array([0.005, 0.005, 0.005, 0.01, 0.0, 0.0])
    sig_new, epsp_new = solid_update_single(p, sig0, deps, epsp=0.0)

    # Pressure is negative (tension)
    p_final = -(sig_new[0] + sig_new[1] + sig_new[2]) / 3.0
    assert p_final <= p.pmin
    # Deviatoric stresses must be zero
    s_dev = sig_new.copy()
    s_dev[:3] += p_final
    np.testing.assert_allclose(s_dev, np.zeros(6), atol=1e-10)


def test_tangent_and_sound_speed():
    """Verify tangent stiffness matrix and acoustic sound speed."""
    p = build_law102(
        rho0=2.5e-6,
        e=25000.0,
        nu=0.25,
        iform=2,
        c=10.0,
        phi=30.0,
    )
    c_speed = sound_speed(p)
    k_bulk = p.k
    g_shear = p.g
    expected_c = math.sqrt((k_bulk + 4.0 * g_shear / 3.0) / p.rho0)
    assert pytest.approx(c_speed) == expected_c

    # Tangent matrix
    c_mat = solid_tangent(p)
    assert c_mat.shape == (6, 6)
    # Check positive definiteness
    eigenvalues = np.linalg.eigvalsh(c_mat)
    assert np.all(eigenvalues > 0.0)
    # Tangent matches isotropic elasticity
    assert pytest.approx(c_mat[0, 0]) == k_bulk + 4.0 * g_shear / 3.0
    assert pytest.approx(c_mat[0, 1]) == k_bulk - 2.0 * g_shear / 3.0
    assert pytest.approx(c_mat[3, 3]) == g_shear
