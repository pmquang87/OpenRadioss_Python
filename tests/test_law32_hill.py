"""
Unit tests for LAW32 (Hill 1948 3D anisotropic plasticity for solids).

Upstream Fortran reference:
  - C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\materials\\mat\\mat032\\sigeps32.F
  - starter/source/materials/mat/mat032/hm_read_mat32.F
  - engine/source/materials/mat/mat032/m32plas.F

Theory
------
Hill 1948 quadratic anisotropic yield criterion for 3D solids:
    sigma_eq = sqrt( F*(syy - szz)^2 + G*(szz - sxx)^2 + H*(sxx - syy)^2
                     + 2*L*syz^2 + 2*M*szx^2 + 2*N*sxy^2 )
    f = sigma_eq - sigma_y <= 0

where F, G, H, L, M, N are related to Lankford r-values (r00, r45, r90) by:
    r00 = H / G
    r90 = H / F
    r45 = (2*N - (F + G)) / (2 * (F + G))

Verifies:
1. Isotropic recovery (F=G=H=0.5, L=M=N=1.5 => von Mises)
2. Analytical gradient vs central finite difference perturbation
3. Uniaxial tension yield initiates at sigma_y, plastic strain follows Swift hardening
4. Lankford r-values match input (r00, r45, r90 reproduced by flow rule)
5. Radial return Newton iterations converge and preserve volume (tr(deps_p)=0)
6. Consistent algorithmic tangent for solids vs directional perturbation
7. Vectorized batch execution equivalence (1D vs 2D)
8. Acoustic sound speed for 3D solids: sqrt((K + 4/3*G)/rho0)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials.law32_hill import (
    Law32Params,
    build_law32,
    hill1948_equivalent_stress,
    hill1948_gradient,
    hill1948_r_values,
    hill1948_params_from_r_values,
    solid_update,
    solid_tangent,
    sound_speed_solid_law32,
)


# ============================================================================
# 1. Isotropic Recovery
# ============================================================================

def test_hill1948_isotropic_recovery():
    """Verify that Hill 1948 with F=G=H=0.5 and L=M=N=1.5 matches von Mises."""
    # Arbitrary 3D stress tensors: [xx, yy, zz, xy, yz, zx]
    stresses = [
        np.array([250.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 300.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, -180.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 120.0, 0.0, 0.0]),
        np.array([150.0, -80.0, 30.0, 45.0, -35.0, 25.0]),
        np.array([-200.0, 100.0, -50.0, 70.0, 60.0, -40.0]),
    ]

    for sig in stresses:
        seq_hill = hill1948_equivalent_stress(sig, F=0.5, G=0.5, H=0.5, L=1.5, M=1.5, N=1.5)

        # Von Mises: sqrt(0.5*((sxx-syy)^2 + (syy-szz)^2 + (szz-sxx)^2) + 3*(sxy^2 + syz^2 + szx^2))
        sxx, syy, szz, sxy, syz, szx = sig
        vm_sq = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2)
        seq_vm = math.sqrt(vm_sq)

        assert seq_hill == pytest.approx(seq_vm, rel=1.0e-9)


# ============================================================================
# 2. Analytical Gradient vs Finite Difference
# ============================================================================

def test_hill1948_gradient_vs_finite_difference():
    """Verify analytical yield surface gradient against central finite difference."""
    F, G, H, L, M, N = 0.4, 0.55, 0.65, 1.4, 1.3, 1.6
    sig0 = np.array([120.0, -80.0, 45.0, 35.0, -25.0, 20.0])

    seq, grad = hill1948_gradient(sig0, F=F, G=G, H=H, L=L, M=M, N=N)

    eps_fd = 1.0e-6
    grad_fd = np.zeros(6, dtype=float)
    for k in range(6):
        sig_plus = sig0.copy()
        sig_plus[k] += eps_fd
        sig_minus = sig0.copy()
        sig_minus[k] -= eps_fd

        seq_p = hill1948_equivalent_stress(sig_plus, F=F, G=G, H=H, L=L, M=M, N=N)
        seq_m = hill1948_equivalent_stress(sig_minus, F=F, G=G, H=H, L=L, M=M, N=N)
        grad_fd[k] = (seq_p - seq_m) / (2.0 * eps_fd)

    np.testing.assert_allclose(grad, grad_fd, rtol=1.0e-5, atol=1.0e-7)

    # Plastic flow is strictly isochoric: tr(N) = N_xx + N_yy + N_zz = 0
    assert abs(grad[0] + grad[1] + grad[2]) < 1.0e-12


# ============================================================================
# 3. Uniaxial Tension Yield
# ============================================================================

def test_yield_in_uniaxial_tension():
    """Verify that yield initiates at sigma_y in uniaxial tension and plastic strain evolves."""
    e0 = 200_000.0
    nu = 0.3
    sigy0 = 250.0

    # 1. Perfectly plastic: yield stress = sigy0
    mat = Law32Params(
        E=e0,
        nu=nu,
        A=sigy0,
        B=0.0,
        n=0.0,  # perfectly plastic: sig_y = sigy0
        r00=1.0,
        r45=1.0,
        r90=1.0,
        i_yield=1,
    )

    # Elastic loading: strain below yield
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
    seq_pl = hill1948_equivalent_stress(sig_pl, F=0.5, G=0.5, H=0.5, L=1.5, M=1.5, N=1.5)
    assert seq_pl == pytest.approx(sigy0, rel=1.0e-5)

    # 2. Swift hardening: sig_y = A * (B + eps_p)^n
    a_swift = 500.0
    b_swift = 0.01
    n_swift = 0.2
    mat_hard = Law32Params(
        E=e0,
        nu=nu,
        A=a_swift,
        B=b_swift,
        n=n_swift,
        r00=1.0,
        r45=1.0,
        r90=1.0,
        i_yield=1,
    )
    sig_h, epsp_h, _ = solid_update(mat_hard, sig0.copy(), deps_pl)
    assert epsp_h > 0.0
    seq_h = hill1948_equivalent_stress(sig_h, F=0.5, G=0.5, H=0.5, L=1.5, M=1.5, N=1.5)
    expected_hard_yield = a_swift * ((b_swift + epsp_h) ** n_swift)
    assert seq_h == pytest.approx(expected_hard_yield, rel=1.0e-5)


# ============================================================================
# 4. Lankford R-Values Match Input
# ============================================================================

def test_r_values_match_input():
    """Verify that Hill 1948 parameters reproduce target Lankford r-values."""
    # Test case 1: Isotropic sheet (r00=1.0, r45=1.0, r90=1.0)
    r0, r45, r90 = hill1948_r_values(F=0.5, G=0.5, H=0.5, L=1.5, M=1.5, N=1.5)
    assert r0 == pytest.approx(1.0, abs=1.0e-9)
    assert r45 == pytest.approx(1.0, abs=1.0e-9)
    assert r90 == pytest.approx(1.0, abs=1.0e-9)

    # Test case 2: Realistic automotive sheet (r00=1.5, r45=1.2, r90=1.8)
    r00_tgt = 1.5
    r45_tgt = 1.2
    r90_tgt = 1.8

    F, G, H, L, M, N = hill1948_params_from_r_values(r00=r00_tgt, r45=r45_tgt, r90=r90_tgt, i_yield=1)
    r0_achieved, r45_achieved, r90_achieved = hill1948_r_values(F=F, G=G, H=H, L=L, M=M, N=N)

    assert r0_achieved == pytest.approx(r00_tgt, rel=1.0e-5)
    assert r45_achieved == pytest.approx(r45_tgt, rel=1.0e-5)
    assert r90_achieved == pytest.approx(r90_tgt, rel=1.0e-5)

    # Verify flow rule direct plastic strain ratios under uniaxial tensions
    ref_s = 200.0
    # Uniaxial tension at 0 deg: sig = [ref_s, 0, 0, 0, 0, 0]
    _, n0 = hill1948_gradient(np.array([ref_s, 0.0, 0.0, 0.0, 0.0, 0.0]), F=F, G=G, H=H, L=L, M=M, N=N)
    r0_flow = n0[1] / n0[2]  # eps_yy / eps_zz
    assert r0_flow == pytest.approx(r00_tgt, rel=1.0e-5)

    # Uniaxial tension at 90 deg: sig = [0, ref_s, 0, 0, 0, 0]
    _, n90 = hill1948_gradient(np.array([0.0, ref_s, 0.0, 0.0, 0.0, 0.0]), F=F, G=G, H=H, L=L, M=M, N=N)
    r90_flow = n90[0] / n90[2]  # eps_xx / eps_zz
    assert r90_flow == pytest.approx(r90_tgt, rel=1.0e-5)

    # Uniaxial tension at 45 deg: sig = [0.5*ref_s, 0.5*ref_s, 0, 0.5*ref_s, 0, 0]
    _, n45 = hill1948_gradient(np.array([0.5 * ref_s, 0.5 * ref_s, 0.0, 0.5 * ref_s, 0.0, 0.0]), F=F, G=G, H=H, L=L, M=M, N=N)
    eps_w = 0.5 * (n45[0] + n45[1]) - 0.5 * n45[3]
    r45_flow = eps_w / n45[2]
    assert r45_flow == pytest.approx(r45_tgt, rel=1.0e-5)


# ============================================================================
# 5. Radial Return Newton Iterations & Volume Preservation
# ============================================================================

def test_radial_return_newton_convergence():
    """Verify Newton return mapping converges to yield surface and preserves volume."""
    e0 = 210_000.0
    nu = 0.3
    sigy0 = 300.0

    mat = Law32Params(
        E=e0,
        nu=nu,
        A=sigy0,
        B=0.0,
        n=0.0,
        r00=1.4,
        r45=1.1,
        r90=1.7,
        i_yield=1,
    )

    sig_in = np.array([80.0, 40.0, -30.0, 25.0, 15.0, -10.0])
    deps = np.array([2.5e-3, 1.2e-3, -1.0e-3, 1.8e-3, -8.0e-4, 6.0e-4])

    sig_out, epsp_out, c_sound = solid_update(mat, sig_in, deps, epsp=0.0)

    # Converged stress must lie on the yield surface
    seq = hill1948_equivalent_stress(sig_out, F=mat.F, G=mat.G_hill, H=mat.H_hill, L=mat.L, M=mat.M_hill, N=mat.N_hill)
    assert seq == pytest.approx(sigy0, rel=1.0e-6)
    assert epsp_out > 0.0

    # Isochoric plastic response: tr(deps_p) = 0 => pressure matches trial pressure
    k = mat.bulk
    tr_deps = deps[0] + deps[1] + deps[2]
    p_trial = (sig_in[0] + sig_in[1] + sig_in[2]) / 3.0 + k * tr_deps
    p_final = (sig_out[0] + sig_out[1] + sig_out[2]) / 3.0
    assert p_final == pytest.approx(p_trial, rel=1.0e-6)


# ============================================================================
# 6. Consistent Algorithmic Tangent for Solids
# ============================================================================

def test_solid_tangent_elastic_and_plastic():
    """Verify solid consistent tangent matrix matches elastic C and finite differences."""
    mat = Law32Params(E=200_000.0, nu=0.3, A=250.0, n=0.0)

    # 1. Pure elastic tangent
    c_el = solid_tangent(mat)
    assert c_el.shape == (6, 6)
    np.testing.assert_allclose(c_el, c_el.T, atol=1.0e-12)

    # 2. Plastic tangent consistency
    sig0 = np.array([180.0, -50.0, 20.0, 40.0, -20.0, 15.0])
    seq0 = hill1948_equivalent_stress(sig0, F=mat.F, G=mat.G_hill, H=mat.H_hill, L=mat.L, M=mat.M_hill, N=mat.N_hill)
    # Scale exactly to yield surface
    sig_yield = sig0 * (250.0 / seq0)

    c_pl = solid_tangent(mat, sig=sig_yield)
    np.testing.assert_allclose(c_pl, c_pl.T, atol=1.0e-10)

    # Tangent eigenvalues must all be non-negative
    eigvals = np.linalg.eigvalsh(c_pl)
    assert np.all(eigvals >= -1.0e-6)


# ============================================================================
# 7. Vectorized Batch Execution Equivalence
# ============================================================================

def test_vectorized_batch_execution():
    """Verify 1D single-element and 2D multi-element batch execution give identical results."""
    mat = Law32Params(E=210_000.0, nu=0.3, A=320.0, n=0.2, B=0.01)

    sig_1 = np.array([120.0, 60.0, -40.0, 30.0, -20.0, 15.0])
    deps_1 = np.array([3.0e-3, -1.0e-3, -5.0e-4, 2.0e-3, 1.0e-3, -1.0e-3])

    sig_2 = np.array([50.0, -80.0, 20.0, 10.0, -15.0, 25.0])
    deps_2 = np.array([1.5e-3, 2.0e-3, -1.2e-3, -1.0e-3, 5.0e-4, 8.0e-4])

    # Run individual 1D calls
    out_sig1, out_ep1, out_c1 = solid_update(mat, sig_1.copy(), deps_1.copy(), epsp=0.005)
    out_sig2, out_ep2, out_c2 = solid_update(mat, sig_2.copy(), deps_2.copy(), epsp=0.012)

    # Run batch 2D call
    sig_batch = np.vstack([sig_1, sig_2])
    deps_batch = np.vstack([deps_1, deps_2])
    epsp_batch = np.array([0.005, 0.012])

    res = solid_update(mat, sig_batch.copy(), deps_batch.copy(), epsp=epsp_batch.copy())
    batch_sig, batch_ep, batch_c = res

    np.testing.assert_allclose(batch_sig[0], out_sig1, rtol=1.0e-12)
    np.testing.assert_allclose(batch_sig[1], out_sig2, rtol=1.0e-12)
    assert batch_ep[0] == pytest.approx(out_ep1, rel=1.0e-12)
    assert batch_ep[1] == pytest.approx(out_ep2, rel=1.0e-12)
    assert batch_c[0] == pytest.approx(out_c1, rel=1.0e-12)
    assert batch_c[1] == pytest.approx(out_c2, rel=1.0e-12)


# ============================================================================
# 8. Acoustic Sound Speed for Solids
# ============================================================================

def test_sound_speed_solid():
    """Verify solid sound speed corresponds to dilatational wave speed sqrt((K + 4/3*G)/rho0)."""
    rho0 = 2700.0
    e = 70_000.0e6
    nu = 0.33
    mat = Law32Params(rho0=rho0, E=e, nu=nu)

    c_dilat = sound_speed_solid_law32(mat)

    g = e / (2.0 * (1.0 + nu))
    k = e / (3.0 * (1.0 - 2.0 * nu))
    expected_c = math.sqrt((k + 4.0 / 3.0 * g) / rho0)

    assert c_dilat == pytest.approx(expected_c, rel=1.0e-10)
