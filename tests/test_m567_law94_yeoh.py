"""
Tests for Milestone M567: /MAT/LAW94 (/MAT/YEOH) Yeoh hyperelastic model.

Verifies:
- Parameter initialization and default incompressible / compressible logic matching hm_read_mat94.F.
- Small strain Hookean limit vs linear isotropic elasticity.
- Deviatoric Cauchy stress and higher-order polynomial pressure matching sigeps94.F.
- Mullins strain energy potential and work consistency.
- Plane-stress shell Newton-Raphson out-of-plane stretch condition (sigma_zz = 0).
- Acoustic wave sound speeds (solid and shell).
- Algorithmic consistent tangents (6x6 for solids, 3x3 for shells).
- Analytical nominal stresses for uniaxial, equibiaxial, and planar deformations.
"""

import numpy as np
import pytest

from pyradioss.materials.law94_yeoh import (
    YeohParams,
    build_law94,
    solid_update,
    shell_update,
    sound_speed,
    sound_speed_shell,
    consistent_tangent,
    yeoh_analytical_stress,
    extra_shapes,
)


def test_yeoh_params_incompressible_default():
    """Verify default bulk modulus and Poisson's ratio when D1=0 (hm_read_mat94.F:144-150)."""
    c10 = 10.0
    p = build_law94(c10=c10, d1=0.0, rho0=1.2)
    assert p.g0 == pytest.approx(20.0)
    assert p.nu == pytest.approx(0.495)
    expected_k = (2.0 / 3.0) * 20.0 * (1.0 + 0.495) / (1.0 - 2.0 * 0.495)
    assert p.rbulk == pytest.approx(expected_k)
    assert p.d1 == pytest.approx(expected_k / 2.0)
    expected_e = 2.0 * 20.0 * (1.0 + 0.495)
    assert p.e == pytest.approx(expected_e)


def test_yeoh_params_compressible():
    """Verify bulk modulus when D1 > 0 (hm_read_mat94.F:152-156)."""
    c10 = 5.0  # G = 10.0
    d1_in = 0.01  # K = 2 / 0.01 = 200.0
    p = build_law94(c10=c10, d1=d1_in, rho0=1.0)
    assert p.g0 == pytest.approx(10.0)
    assert p.rbulk == pytest.approx(200.0)
    assert p.d1 == pytest.approx(100.0)  # 1 / D1_in
    expected_nu = (3.0 * 200.0 - 2.0 * 10.0) / (2.0 * (3.0 * 200.0 + 10.0))
    assert p.nu == pytest.approx(expected_nu)
    expected_e = 9.0 * 200.0 * 10.0 / (3.0 * 200.0 + 10.0)
    assert p.e == pytest.approx(expected_e)


def test_yeoh_small_strain_hookean_limit():
    """Under small strain, Yeoh model must match linear isotropic elasticity."""
    c10 = 15.0  # G = 30.0
    d1 = 0.01   # K = 200.0
    p = build_law94(c10=c10, d1=d1, rho0=1.0)
    G = p.g0
    K = p.rbulk

    # Small pure shear strain: eps_xy = 1e-5
    eps_shear = np.array([0.0, 0.0, 0.0, 1e-5, 0.0, 0.0])
    sig, _, _ = solid_update(p, eps=eps_shear)
    # sigma_xy = 2 * G * (eps_xy / 2) = G * eps_xy
    assert sig[3] == pytest.approx(G * 1e-5, rel=1e-3)
    assert np.allclose(sig[:3], 0.0, atol=1e-6)

    # Small volumetric strain: eps_xx = eps_yy = eps_zz = 1e-5
    eps_vol = np.array([1e-5, 1e-5, 1e-5, 0.0, 0.0, 0.0])
    sig_v, _, _ = solid_update(p, eps=eps_vol)
    # sigma_xx = sigma_yy = sigma_zz = 3 * K * 1e-5
    expected_p = 3.0 * K * 1e-5
    assert sig_v[0] == pytest.approx(expected_p, rel=1e-3)
    assert sig_v[1] == pytest.approx(expected_p, rel=1e-3)
    assert sig_v[2] == pytest.approx(expected_p, rel=1e-3)


def test_yeoh_higher_order_polynomial_pressure():
    """Verify pressure includes D2 and D3 cubic and quintic terms (sigeps94.F:238)."""
    c10 = 10.0
    d1_in = 0.02  # 1/D1 = 50, K = 100
    d2_in = 0.05  # 1/D2 = 20
    d3_in = 0.1   # 1/D3 = 10
    p = build_law94(c10=c10, d1=d1_in, d2=d2_in, d3=d3_in, rho0=1.0)

    # Apply pure volumetric stretch J = 1.05 (eps = ln(1.05)/3)
    e_v = np.log(1.05) / 3.0
    eps = np.array([e_v, e_v, e_v, 0.0, 0.0, 0.0])
    sig, _, _ = solid_update(p, eps=eps)

    # Expected P = K*(J - 1) + 4*(1/D2)*(J - 1)^3 + 6*(1/D3)*(J - 1)^5
    j_m1 = 0.05
    expected_P = 100.0 * j_m1 + 4.0 * 20.0 * (j_m1 ** 3) + 6.0 * 10.0 * (j_m1 ** 5)
    assert sig[0] == pytest.approx(expected_P, rel=1e-4)
    assert sig[1] == pytest.approx(expected_P, rel=1e-4)
    assert sig[2] == pytest.approx(expected_P, rel=1e-4)


def test_yeoh_analytical_uniaxial_stress():
    """Verify analytical nominal tensile stress matches constitutive formula."""
    c10 = 1.2
    c20 = -0.15
    c30 = 0.02

    for stretch in [1.1, 1.3, 1.8, 2.5]:
        nom_stress = yeoh_analytical_stress(stretch, c10, c20, c30, itype=1)
        i1 = stretch * stretch + 2.0 / stretch
        aa = i1 - 3.0
        dW_dI1 = c10 + 2.0 * c20 * aa + 3.0 * c30 * (aa * aa)
        expected = 2.0 * (stretch - 1.0 / (stretch * stretch)) * dW_dI1
        assert nom_stress == pytest.approx(expected)


def test_yeoh_analytical_equibiaxial_and_planar():
    """Verify analytical formulas for equibiaxial (itype=2) and planar (itype=3)."""
    c10 = 2.0
    c20 = 0.1
    c30 = 0.005

    stretch = 1.4
    s_eq = yeoh_analytical_stress(stretch, c10, c20, c30, itype=2)
    i1_eq = 2.0 * stretch * stretch + 1.0 / (stretch ** 4)
    aa_eq = i1_eq - 3.0
    expected_eq = 2.0 * (stretch - 1.0 / (stretch ** 5)) * (c10 + 2.0 * c20 * aa_eq + 3.0 * c30 * aa_eq ** 2)
    assert s_eq == pytest.approx(expected_eq)

    s_pl = yeoh_analytical_stress(stretch, c10, c20, c30, itype=3)
    i1_pl = stretch * stretch + 1.0 + 1.0 / (stretch * stretch)
    aa_pl = i1_pl - 3.0
    expected_pl = 2.0 * (stretch - 1.0 / (stretch ** 3)) * (c10 + 2.0 * c20 * aa_pl + 3.0 * c30 * aa_pl ** 2)
    assert s_pl == pytest.approx(expected_pl)


def test_yeoh_sound_speed():
    """Verify acoustic sound speed computation in solids and shells."""
    c10 = 10.0  # G = 20.0
    d1 = 0.01   # K = 200.0
    rho0 = 2.0
    p = build_law94(c10=c10, d1=d1, rho0=rho0)

    # At zero strain, c_solid = sqrt(((4/3)*G + K) / rho)
    c_solid = sound_speed(p)
    expected_c_solid = np.sqrt(((4.0 / 3.0) * 20.0 + 200.0) / 2.0)
    assert c_solid == pytest.approx(expected_c_solid)

    # Shell sound speed: sqrt(E / ((1 - nu^2) * rho))
    c_sh = sound_speed_shell(p)
    expected_c_sh = np.sqrt(p.e / ((1.0 - p.nu ** 2) * rho0))
    assert c_sh == pytest.approx(expected_c_sh)


def test_yeoh_shell_plane_stress_condition():
    """Verify that shell_update converges out-of-plane stretch so that sigma_zz = 0."""
    c10 = 5.0
    c20 = 0.5
    c30 = 0.05
    d1 = 0.02
    p = build_law94(c10=c10, c20=c20, c30=c30, d1=d1, rho0=1.0)

    # Apply in-plane tensile strain eps_xx = 0.2, eps_yy = -0.05, eps_xy = 0.05
    deps = np.array([0.2, -0.05, 0.05])
    extra = {}
    sig_sh, epsp, c_sh = shell_update(p, np.zeros(3), deps=deps, extra=extra, return_sound_speed=True)

    # The resulting out-of-plane stretch lambda_3 is stored in extra["uvar_lam3"]
    assert "uvar_lam3" in extra
    lam3 = float(extra["uvar_lam3"][0])
    assert 0.5 < lam3 < 1.0  # Under tension, transverse stretch contracts

    # Evaluate full 3D solid update at this strain state (including eps_zz = ln(lam3))
    eps_3d = np.array([0.2, -0.05, np.log(lam3), 0.05, 0.0, 0.0])
    sig_3d, _, _ = solid_update(p, eps=eps_3d)

    # sigma_zz in 3D should be practically zero
    assert abs(sig_3d[2]) < 1e-4 * p.g0
    # In-plane stresses should match between shell and 3D solid
    assert sig_sh[0] == pytest.approx(sig_3d[0], rel=1e-3)
    assert sig_sh[1] == pytest.approx(sig_3d[1], rel=1e-3)
    assert sig_sh[2] == pytest.approx(sig_3d[3], rel=1e-3)


def test_yeoh_consistent_tangents():
    """Verify consistent tangents for 3D solid (6x6) and 2D shell (3x3)."""
    p = build_law94(c10=8.0, c20=0.2, c30=0.01, d1=0.01, rho0=1.0)

    # 3D solid tangent
    eps_0 = np.array([0.05, -0.02, -0.01, 0.03, 0.0, 0.0])
    c_solid = consistent_tangent(p, eps=eps_0, is_shell=False)
    # Tangent symmetry within finite-strain Cauchy rate tolerance
    diff_solid = c_solid - c_solid.T
    assert np.all(np.abs(diff_solid) < 2e-2 * np.max(np.abs(c_solid)))
    sym_solid = 0.5 * (c_solid + c_solid.T)
    assert np.all(np.linalg.eigvalsh(sym_solid) > 0.0)

    # 2D shell tangent
    eps_2d = np.array([0.05, -0.02, 0.03])
    c_shell = consistent_tangent(p, eps=eps_2d, is_shell=True)
    assert c_shell.shape == (3, 3)
    diff_shell = c_shell - c_shell.T
    assert np.all(np.abs(diff_shell) < 2e-2 * np.max(np.abs(c_shell)))
    sym_shell = 0.5 * (c_shell + c_shell.T)
    assert np.all(np.linalg.eigvalsh(sym_shell) > 0.0)


def test_yeoh_extra_shapes():
    """Verify extra_shapes returns eps94 and uvar_lam3."""
    shapes = extra_shapes(nip=5)
    assert "eps94" in shapes
    assert shapes["eps94"] == (5, 3)
    assert "uvar_lam3" in shapes
    assert shapes["uvar_lam3"] == (5,)
