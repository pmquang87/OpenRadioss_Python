"""
Unit tests for LAW57 (Barlat 1991 6-parameter anisotropic plasticity for 3D solids).

Upstream Fortran reference:
  - C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\materials\\mat\\mat057\\sigeps57.F

Theory
------
Barlat 1991 anisotropic yield criterion:
    Phi = |s1 - s2|^m + |s2 - s3|^m + |s3 - s1|^m = 2 * sigma_y^m
    sigma_eq = (0.5 * Phi)**(1 / m)
where s1, s2, s3 are the principal values of the modified stress tensor:
    s_xx = [c*(sigma_xx - sigma_yy) - b*(sigma_zz - sigma_xx)] / 3
    s_yy = [a*(sigma_yy - sigma_zz) - c*(sigma_xx - sigma_yy)] / 3
    s_zz = [b*(sigma_zz - sigma_xx) - a*(sigma_yy - sigma_zz)] / 3
    s_yz = f * sigma_yz
    s_zx = g * sigma_zx
    s_xy = h * sigma_xy

Verifies:
1. Isotropic recovery (m=2, a=b=c=f=g=h=1 => von Mises)
2. Analytical gradient vs central finite difference perturbation
3. Uniaxial tension yield: initiates at sigma_y, plastic strain increases
4. Lankford r-values match input: r0, r45, r90 reproduced by plastic flow rule
5. Radial return Newton-Raphson convergence and volume preservation (tr(deps_p)=0)
6. Consistent algorithmic tangent for solids vs directional perturbation
7. Vectorized batch execution equivalence (1D vs 2D)
8. Acoustic sound speed for 3D solids: sqrt((K + 4/3*G)/rho0)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials.law57_barlat import (
    Law57Params,
    Barlat1991Params,
    build_law57,
    barlat1991_yield_function,
    barlat1991_equivalent_stress,
    barlat1991_gradient,
    barlat1991_r_values,
    calibrate_barlat1991,
    solid_update,
    solid_tangent,
    sound_speed_solid_law57,
    sound_speed,
    LAW_DISPATCH_METADATA,
)


# ============================================================================
# 1. Isotropic Recovery (von Mises Equivalence for m=2, a=b=c=f=g=h=1)
# ============================================================================

def test_barlat1991_isotropic_recovery():
    """Verify that Barlat 1991 reduces identically to von Mises when m=2 and a=b=c=f=g=h=1."""
    test_stresses = [
        np.array([200.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 150.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 180.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 100.0, 0.0, 0.0]),
        np.array([120.0, 80.0, -50.0, 40.0, -30.0, 25.0]),
        np.array([-200.0, -100.0, -50.0, 15.0, 20.0, -35.0]),
    ]

    for sig in test_stresses:
        phi, seq = barlat1991_yield_function(sig, a=1.0, b=1.0, c=1.0, f=1.0, g=1.0, h=1.0, m=2.0)

        # Classical von Mises equivalent stress: sqrt(3 * J2)
        p = np.sum(sig[:3]) / 3.0
        s = sig.copy()
        s[:3] -= p
        j2 = 0.5 * np.sum(s[:3] ** 2) + np.sum(s[3:] ** 2)
        vm = math.sqrt(3.0 * j2)

        assert seq == pytest.approx(vm, rel=1.0e-9)
        assert phi == pytest.approx(2.0 * (vm ** 2), rel=1.0e-9)


# ============================================================================
# 2. Analytical Gradient vs Finite Difference & Euler Identity
# ============================================================================

def test_barlat1991_gradient_finite_difference():
    """Verify analytical yield surface normal gradient against central finite difference."""
    a, b, c, f, g, h, m = 1.2, 0.85, 1.05, 1.1, 0.95, 1.15, 6.0
    test_stresses = [
        np.array([250.0, 50.0, -30.0, 40.0, -20.0, 15.0]),
        np.array([180.0, -60.0, 20.0, -35.0, 25.0, -10.0]),
        np.array([50.0, 50.0, 0.0, 50.0, 0.0, 0.0]),
    ]

    h_step = 1.0e-6
    for sig in test_stresses:
        seq, grad = barlat1991_gradient(sig, a=a, b=b, c=c, f=f, g=g, h=h, m=m)

        fd_grad = np.zeros(6, dtype=float)
        for j in range(6):
            sp = sig.copy()
            sm = sig.copy()
            sp[j] += h_step
            sm[j] -= h_step
            _, seq_p = barlat1991_yield_function(sp, a=a, b=b, c=c, f=f, g=g, h=h, m=m)
            _, seq_m = barlat1991_yield_function(sm, a=a, b=b, c=c, f=f, g=g, h=h, m=m)
            fd_grad[j] = (seq_p - seq_m) / (2.0 * h_step)

        max_err = np.max(np.abs(grad - fd_grad))
        assert max_err < 1.0e-5

        # Euler's theorem for homogeneous degree 1 function: sig . grad == seq
        euler_val = np.dot(sig, grad)
        assert euler_val == pytest.approx(seq, rel=1.0e-7)

        # Isochoric plastic normality: tr(N) == N_xx + N_yy + N_zz == 0
        assert (grad[0] + grad[1] + grad[2]) == pytest.approx(0.0, abs=1.0e-9)


# ============================================================================
# 3. Uniaxial Tension Yield
# ============================================================================

def test_yield_in_uniaxial_tension():
    """Verify that yield starts at sigma_y in uniaxial tension and plastic strain evolves."""
    e0 = 200_000.0
    nu = 0.3
    sigy0 = 250.0
    h_hard = 1000.0

    mat = build_law57(
        E=e0,
        nu=nu,
        sigy0=sigy0,
        m=6.0,
        a_barlat=1.0,
        b_barlat=1.0,
        c_barlat=1.0,
        f_barlat=1.0,
        g_barlat=1.0,
        h_barlat=1.0,
        curves=[([0.0, 0.1], [sigy0, sigy0 + h_hard * 0.1], 0.0)],
    )

    # Elastic loading: strain below yield
    # For uniaxial stress: deps_xx = sig / E, deps_yy = deps_zz = -nu * deps_xx
    sig0 = np.zeros(6, dtype=float)
    eps_y = sigy0 / e0
    deps_el = np.array([0.8 * eps_y, -nu * 0.8 * eps_y, -nu * 0.8 * eps_y, 0.0, 0.0, 0.0])

    sig_out, epsp_out, _ = solid_update(mat, sig0.copy(), deps_el)
    assert epsp_out == 0.0
    assert sig_out[0] == pytest.approx(0.8 * sigy0, rel=1.0e-4)
    assert sig_out[1] == pytest.approx(0.0, abs=1.0e-6)
    assert sig_out[2] == pytest.approx(0.0, abs=1.0e-6)

    # Plastic loading: strain exceeding yield
    deps_pl = np.array([1.5 * eps_y, -nu * 1.5 * eps_y, -nu * 1.5 * eps_y, 0.0, 0.0, 0.0])
    sig_pl, epsp_pl, _ = solid_update(mat, sig0.copy(), deps_pl)

    assert epsp_pl > 0.0
    # Equivalent stress must lie on the hardening curve
    seq_pl = barlat1991_equivalent_stress(sig_pl, m=6.0)
    expected_yield = sigy0 + h_hard * epsp_pl
    assert seq_pl == pytest.approx(expected_yield, rel=1.0e-5)


# ============================================================================
# 4. Lankford R-Values Match Input
# ============================================================================

def test_r_values_match_input():
    """Verify that Barlat 1991 anisotropy coefficients reproduce input Lankford r-values."""
    # Test case 1: Isotropic sheet (r00=1.0, r45=1.0, r90=1.0)
    r0, r45, r90 = barlat1991_r_values(a=1.0, b=1.0, c=1.0, f=1.0, g=1.0, h=1.0, m=6.0)
    assert r0 == pytest.approx(1.0, abs=1.0e-9)
    assert r45 == pytest.approx(1.0, abs=1.0e-9)
    assert r90 == pytest.approx(1.0, abs=1.0e-9)

    # Test case 2: Typical automotive aluminum sheet (AA6016)
    r00_tgt = 1.5
    r45_tgt = 1.2
    r90_tgt = 1.8
    m_exp = 6.0

    a, b, c, f, g, h = calibrate_barlat1991(r00=r00_tgt, r45=r45_tgt, r90=r90_tgt, m=m_exp)
    r0_achieved, r45_achieved, r90_achieved = barlat1991_r_values(a=a, b=b, c=c, f=f, g=g, h=h, m=m_exp)

    assert r0_achieved == pytest.approx(r00_tgt, rel=1.0e-4)
    assert r45_achieved == pytest.approx(r45_tgt, rel=1.0e-4)
    assert r90_achieved == pytest.approx(r90_tgt, rel=1.0e-4)

    # Verify plastic strain increments under uniaxial tension match r-values directly
    ref_s = 200.0
    _, n0 = barlat1991_gradient(np.array([ref_s, 0.0, 0.0, 0.0, 0.0, 0.0]), a, b, c, f, g, h, m_exp)
    r0_flow = n0[1] / n0[2]
    assert r0_flow == pytest.approx(r00_tgt, rel=1.0e-4)

    _, n90 = barlat1991_gradient(np.array([0.0, ref_s, 0.0, 0.0, 0.0, 0.0]), a, b, c, f, g, h, m_exp)
    r90_flow = n90[0] / n90[2]
    assert r90_flow == pytest.approx(r90_tgt, rel=1.0e-4)

    _, n45 = barlat1991_gradient(np.array([0.5 * ref_s, 0.5 * ref_s, 0.0, 0.5 * ref_s, 0.0, 0.0]), a, b, c, f, g, h, m_exp)
    eps_w = 0.5 * (n45[0] + n45[1]) - 0.5 * n45[3]
    r45_flow = eps_w / n45[2]
    assert r45_flow == pytest.approx(r45_tgt, rel=1.0e-4)


# ============================================================================
# 5. Radial Return Newton Iterations & Incompressible Plasticity
# ============================================================================

def test_radial_return_newton_convergence():
    """Verify Newton return mapping converges to yield surface and maintains tr(deps_p)=0."""
    e0 = 210_000.0
    nu = 0.3
    sigy0 = 300.0
    mat = Law57Params(
        E=e0,
        nu=nu,
        sigy0=sigy0,
        m=6.0,
        a_barlat=1.1,
        b_barlat=0.9,
        c_barlat=1.0,
        h_barlat=1.05,
    )

    sig_in = np.array([100.0, 50.0, -20.0, 20.0, 10.0, -15.0])
    deps = np.array([3.0e-3, 1.0e-3, -1.5e-3, 2.0e-3, -1.0e-3, 5.0e-4])

    sig_out, epsp_out, c_sound = solid_update(mat, sig_in, deps, epsp=0.0)

    # Converged stress must lie on the yield surface
    bp = mat.barlat1991
    seq = barlat1991_equivalent_stress(sig_out, a=bp.a, b=bp.b, c=bp.c, f=bp.f, g=bp.g, h=bp.h, m=bp.m)
    assert seq == pytest.approx(sigy0, rel=1.0e-6)
    assert epsp_out > 0.0

    # Volumetric plastic response is zero (tr(deps_p) = 0), so pressure matches trial pressure
    g = mat.G
    k = mat.bulk
    lam = k - (2.0 / 3.0) * g
    tr_deps = deps[0] + deps[1] + deps[2]
    p_trial = (sig_in[0] + sig_in[1] + sig_in[2]) / 3.0 + k * tr_deps
    p_final = (sig_out[0] + sig_out[1] + sig_out[2]) / 3.0
    assert p_final == pytest.approx(p_trial, rel=1.0e-6)


# ============================================================================
# 6. Consistent Algorithmic Tangent for Solids
# ============================================================================

def test_solid_tangent_elastic_and_plastic():
    """Verify solid consistent tangent matrix matches elastic C and finite differences."""
    mat = Law57Params(E=200_000.0, nu=0.3, sigy0=250.0)

    # 1. Pure elastic tangent (deps is None)
    c_el = solid_tangent(mat)
    assert c_el.shape == (6, 6)
    lam = mat.bulk - (2.0 / 3.0) * mat.G
    g = mat.G
    assert c_el[0, 0] == pytest.approx(lam + 2.0 * g)
    assert c_el[0, 1] == pytest.approx(lam)
    assert c_el[3, 3] == pytest.approx(g)

    # 2. Plastic tangent verified against directional finite difference perturbation
    sig_in = np.array([200.0, 50.0, 0.0, 20.0, 0.0, 0.0])
    deps = np.array([2.0e-3, -5.0e-4, -5.0e-4, 1.0e-3, 0.0, 0.0])

    d_algo = solid_tangent(mat, sig=sig_in, deps=deps, epsp=0.0)
    assert d_algo.shape == (6, 6)

    h_step = 1.0e-6
    d_fd = np.zeros((6, 6), dtype=float)
    for j in range(6):
        deps_p = deps.copy()
        deps_m = deps.copy()
        deps_p[j] += h_step
        deps_m[j] -= h_step

        res_p = solid_update(mat, sig_in.copy(), deps_p, epsp=0.0, return_sound_speed=False)
        res_m = solid_update(mat, sig_in.copy(), deps_m, epsp=0.0, return_sound_speed=False)

        sp = res_p[0]
        sm = res_m[0]
        d_fd[:, j] = (sp - sm) / (2.0 * h_step)

    max_err = np.max(np.abs(d_algo - d_fd))
    scale = np.max(np.abs(d_algo))
    assert max_err / scale < 1.0e-3


# ============================================================================
# 7. Batched Execution & Sound Speed
# ============================================================================

def test_batched_solid_update_equivalence():
    """Verify batched 2D solid_update matches 1D single-element evaluation."""
    mat = Law57Params(
        E=200_000.0,
        nu=0.3,
        sigy0=250.0,
        a_barlat=1.2,
        b_barlat=0.85,
        c_barlat=1.0,
        h_barlat=1.1,
    )

    sig_batch = np.array([
        [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [200.0, 50.0, -10.0, 30.0, 10.0, -5.0],
    ])
    deps_batch = np.array([
        [1.0e-4, -3.0e-5, -3.0e-5, 0.0, 0.0, 0.0],
        [2.5e-3, -8.0e-4, -8.0e-4, 1.2e-3, 5.0e-4, -3.0e-4],
    ])
    epsp_batch = np.array([0.0, 0.01])

    sig_out_batch, epsp_out_batch, c_batch = solid_update(
        mat, sig_batch, deps_batch, epsp=epsp_batch
    )

    for i in range(2):
        sig_i, epsp_i, c_i = solid_update(
            mat, sig_batch[i], deps_batch[i], epsp=epsp_batch[i]
        )
        assert np.allclose(sig_out_batch[i], sig_i, atol=1.0e-6)
        assert epsp_out_batch[i] == pytest.approx(epsp_i, rel=1.0e-6)
        assert c_batch[i] == pytest.approx(c_i, rel=1.0e-6)


def test_sound_speed_solid():
    """Verify solid acoustic sound speed matches analytical longitudinal wave speed."""
    e0 = 210_000.0
    nu = 0.3
    rho = 7.85e-9
    mat = Law57Params(E=e0, nu=nu, rho0=rho)

    k = e0 / (3.0 * (1.0 - 2.0 * nu))
    g = e0 / (2.0 * (1.0 + nu))
    expected_c = math.sqrt((k + (4.0 / 3.0) * g) / rho)

    c_solid = sound_speed_solid_law57(mat)
    assert c_solid == pytest.approx(expected_c, rel=1.0e-7)

    # Dispatcher
    c_disp = sound_speed(mat, is_shell=False)
    assert c_disp == pytest.approx(expected_c, rel=1.0e-7)
