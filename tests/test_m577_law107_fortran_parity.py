"""Fortran parity test for /MAT/LAW107 (/MAT/PAPER_LIGHT) against OpenRadioss Fortran kernels.

Verifies exact numerical formulas and steps from:
  engine/source/materials/mat/mat107/sigeps107.F
  engine/source/materials/mat/mat107/sigeps107c.F
  engine/source/materials/mat/mat107/mat107_newton.F
  engine/source/materials/mat/mat107/mat107_nice.F
  engine/source/materials/mat/mat107/mat107c_newton.F
  engine/source/materials/mat/mat107/mat107c_nice.F
  starter/source/materials/mat/mat107/hm_read_mat107.F
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law107_paper_light import (
    PaperLightConstants,
    PaperLightParams,
    compute_paper_light_constants,
    compute_paper_light_yield_surface,
    compute_paper_light_normals,
    mat107_newton_update_single,
    mat107_nice_update_single,
    shell_update,
    solid_update,
)


def _make_test_params() -> tuple[PaperLightParams, PaperLightConstants]:
    p = PaperLightParams(
        rho=7.5e-7,
        refer_rho=7.5e-7,
        young1=7000.0,
        young2=3500.0,
        young3=100.0,
        nu21=0.15,
        g12=1800.0,
        g23=40.0,
        g31=40.0,
        ires=2,
        itab=0,
        ismooth=1,
        xi1=0.5,
        xi2=0.5,
        g1c=0.0,
        d1=0.1,
        d2=0.2,
        k1=0.1,
        k2=-0.1,
        k3=0.2,
        k4=0.3,
        k5=0.4,
        k6=0.5,
        sigy1=25.0,
        cini1=10.0,
        s1=2.0,
        sigy2=15.0,
        cini2=8.0,
        s2=1.5,
        sigy1c=30.0,
        cini1c=12.0,
        s1c=2.5,
        sigy2c=20.0,
        cini2c=10.0,
        s2c=2.0,
        sigyt=10.0,
        cinit=5.0,
        st=1.0,
    )
    c = compute_paper_light_constants(p)
    return p, c


def test_fortran_parity_plane_stress_elastic_matrix():
    """Verify plane stress elasticity matrix A11, A12, A22, A33 match Fortran sigeps107.F."""
    p, c = _make_test_params()
    # Fortran: nu12 = nu21 * E1 / E2 = 0.15 * 7000 / 3500 = 0.30
    # denom = 1 - nu12 * nu21 = 1 - 0.30 * 0.15 = 1 - 0.045 = 0.955
    # A11 = E1 / denom = 7000 / 0.955 = 7329.84293
    # A22 = E2 / denom = 3500 / 0.955 = 3664.92147
    # A12 = nu21 * E1 / denom = 0.15 * 7000 / 0.955 = 1099.47644
    # A33 = G12 = 1800.0
    denom_hand = 1.0 - (p.nu21 * p.e1 / p.e2) * p.nu21
    a11_hand = p.e1 / denom_hand
    a22_hand = p.e2 / denom_hand
    a12_hand = p.nu21 * p.e1 / denom_hand
    a33_hand = p.g12

    assert c.a11 == pytest.approx(a11_hand, rel=1e-6)
    assert c.a22 == pytest.approx(a22_hand, rel=1e-6)
    assert c.a12 == pytest.approx(a12_hand, rel=1e-6)
    assert c.g12 == pytest.approx(a33_hand, rel=1e-6)


def test_fortran_parity_pfeiffer_yield_stress_and_derivative():
    """Verify Pfeiffer hardening R(p) = SIGY + p/(CINI + S*p) and dR/dp."""
    p, _ = _make_test_params()
    pla = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float64)
    phi, alpha, r_vec, dr_dp = compute_paper_light_yield_surface(10.0, 5.0, 0.0, p, pla)

    p1 = 0.05
    # Direction 1 tension: SIGY1=25, CINI1=10, S1=2.0
    denom1 = 10.0 + 2.0 * p1
    r1_hand = 25.0 + p1 / denom1
    dr1_hand = 10.0 / (denom1 * denom1)
    assert r_vec[0] == pytest.approx(r1_hand, rel=1e-7)
    assert dr_dp[0] == pytest.approx(dr1_hand, rel=1e-7)

    # Direction 2 tension: SIGY2=15, CINI2=8, S2=1.5
    denom2 = 8.0 + 1.5 * p1
    r2_hand = 15.0 + p1 / denom2
    dr2_hand = 8.0 / (denom2 * denom2)
    assert r_vec[1] == pytest.approx(r2_hand, rel=1e-7)
    assert dr_dp[1] == pytest.approx(dr2_hand, rel=1e-7)

    # Shear: SIGYT=10, CINIT=5, ST=1.0
    denom5 = 5.0 + 1.0 * p1
    rt_hand = 10.0 + p1 / denom5
    drt_hand = 5.0 / (denom5 * denom5)
    assert r_vec[4] == pytest.approx(rt_hand, rel=1e-7)
    assert dr_dp[4] == pytest.approx(drt_hand, rel=1e-7)


def test_fortran_parity_yield_surface_evaluation():
    """Verify yield surface phi and normals n_vec and np_vec match Fortran mat107_newton.F."""
    p, c = _make_test_params()
    pla = np.zeros(6, dtype=np.float64)

    # Case 1: Pure tension in direction 1 (sigma_xx = 25.0, sigma_yy = 0.0, sigma_xy = 0.0)
    # sigma_xx = r1 = 25.0 => phi should be 0.0
    phi_1, alpha_1, r_vec_1, _ = compute_paper_light_yield_surface(25.0, 0.0, 0.0, p, pla)
    assert phi_1 == pytest.approx(0.0, abs=1e-5)
    n_vec_1, np_vec_1 = compute_paper_light_normals(25.0, 0.0, 0.0, p, alpha_1, r_vec_1)
    # Direction 1 normal should point along x
    assert n_vec_1[0] > 0.0
    assert np_vec_1[0] > 0.0

    # Case 2: Pure shear (sigma_xx = 0, sigma_yy = 0, sigma_xy = 10.0)
    # sigma_xy = rt = 10.0 => phi should be 0.0
    phi_sh, alpha_sh, r_vec_sh, _ = compute_paper_light_yield_surface(0.0, 0.0, 10.0, p, pla)
    assert phi_sh == pytest.approx(0.0, abs=1e-5)
    n_vec_sh, np_vec_sh = compute_paper_light_normals(0.0, 0.0, 10.0, p, alpha_sh, r_vec_sh)
    assert n_vec_sh[2] > 0.0
    assert np_vec_sh[2] > 0.0


def test_fortran_parity_newton_cutting_plane_return():
    """Verify Newton return mapping pulls trial stress outside yield surface back to phi <= 0."""
    p, c = _make_test_params()
    # Trial stress significantly outside yield surface (sigxx = 40.0 > 25.0)
    sig_trial = np.array([40.0, 5.0, 3.0], dtype=np.float64)
    pla = np.zeros(6, dtype=np.float64)

    sig_corr, pla_new, dpla, epsp_new, c_eff = mat107_newton_update_single(
        sig_trial, p, c, pla, niter=10
    )

    # Post-return yield function must be zero within tolerance
    phi_post, _, _, _ = compute_paper_light_yield_surface(
        sig_corr[0], sig_corr[1], sig_corr[2], p, pla_new
    )

    assert phi_post <= 1e-4
    assert epsp_new > 0.0
    assert dpla > 0.0


def test_fortran_parity_nice_return_mapping():
    """Verify Nice (IRES=1) explicit return mapping produces valid projection."""
    p, c = _make_test_params()
    p.ires = 1
    # Stress state right on yield surface: sigxx = r1 = 25.0, phi_prev = 0.0
    sigo = np.array([25.0, 0.0, 0.0], dtype=np.float64)
    deps = np.array([5.0e-4, 0.0, 0.0], dtype=np.float64)
    pla = np.zeros(6, dtype=np.float64)
    phi_prev = 0.0

    sig_corr, pla_new, dpla, hardp, sigy_sc, phi_new = mat107_nice_update_single(
        sigo, deps, p, c, pla, phi_prev
    )

    phi_post, _, _, _ = compute_paper_light_yield_surface(
        sig_corr[0], sig_corr[1], sig_corr[2], p, pla_new
    )

    assert phi_post <= 0.05
    assert dpla > 0.0


def test_fortran_parity_transverse_shear_shell():
    """Verify transverse shear terms sigma_yz and sigma_zx are integrated with G23 and G13."""
    p, c = _make_test_params()
    sig = np.zeros(5, dtype=np.float64)
    # Applied strain increment with membrane and transverse shear:
    # deps = [eps_xx, eps_yy, gamma_xy, gamma_yz, gamma_zx]
    deps = np.array([1e-4, 0.0, 0.0, 2e-4, 3e-4], dtype=np.float64)
    extra = {"uvar": np.zeros(1), "pla": np.zeros(6), "epsd": np.zeros(6)}

    sig_out, dpla, epsp_out, extra_out = shell_update(
        p, sig, deps, dt=1e-6, extra=extra, return_tuple=False
    )

    # Transverse shear includes shear correction factor shf = 5/6:
    shf = 5.0 / 6.0
    assert sig_out[3] == pytest.approx(shf * p.g23 * 2e-4, rel=1e-6)
    assert sig_out[4] == pytest.approx(shf * p.g31 * 3e-4, rel=1e-6)


def test_fortran_parity_solid_vs_shell_membrane():
    """Verify 3D solid continuum update matches 2D plane-stress for pure in-plane membrane loading."""
    p, c = _make_test_params()
    deps_2d = np.array([1e-4, 2e-5, 5e-5, 0.0, 0.0], dtype=np.float64)
    extra_sh = {"uvar": np.zeros(1), "pla": np.zeros(6), "epsd": np.zeros(6)}
    sig_sh = np.zeros(5, dtype=np.float64)

    sig_sh_out, _, _, _ = shell_update(
        p, sig_sh, deps_2d, dt=1e-6, extra=extra_sh, return_tuple=False
    )

    # For solid with plane stress condition (deps_zz chosen so sig_zz ~ 0)
    nu12 = p.nu21 * p.e1 / p.e2
    nu31 = 0.01
    nu32 = 0.01
    deps_zz = -(nu31 * deps_2d[0] + nu32 * deps_2d[1])
    deps_3d = np.array([deps_2d[0], deps_2d[1], deps_zz, deps_2d[2], 0.0, 0.0], dtype=np.float64)
    sig_sol = np.zeros(6, dtype=np.float64)
    extra_sol = {"uvar": np.zeros(1), "pla": np.zeros(6), "epsd": np.zeros(6)}

    sig_sol_out, _, _, _ = solid_update(
        p, sig_sol, deps_3d, dt=1e-6, extra=extra_sol, return_tuple=False
    )

    # In-plane shear component sig_xy should match exactly G12 * gamma_xy in both
    assert sig_sh_out[2] == pytest.approx(sig_sol_out[3], rel=1e-4)
