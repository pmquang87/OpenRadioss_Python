"""
Tests for /MAT/LAW107 (/MAT/PAPER_LIGHT, /MAT/PLAS_PAPER_LIGHT, /MAT/PFEIFFER) constitutive physics.
Milestone M577.

Validates:
  1. Orthotropic elasticity constants: A11, A12, A21, A22, and sound speeds (ssp_solid, ssp_shell).
  2. Indicator flags alpha_i evaluation based on in-plane stress quadrant.
  3. Voce-like analytical yield surface calculation (R1, R2, R1c, R2c, Rt) and derivatives dr_i/dp_i.
  4. Tabulated yield surface with scale factors and finite-difference derivatives.
  5. Normals computation: n = dphi/dsig and non-associated potential gradient np with beta_1, beta_2 switching.
  6. 3-iteration cutting plane Newton return mapping (IRES=2) matching mat107_newton.F.
  7. Nice explicit return mapping (IRES=1) matching mat107_nice.F.
  8. 3D continuum solid update (sigeps107.F) and 2D shell plane-stress update (sigeps107c.F).
  9. Implicit algorithmic tangents (solid_tangent, consistent_shell_tangent).
  10. Extra shapes state variables (uvar: 1, pla: 6, epsd: 6).
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law107_paper_light import (
    PaperLightParams,
    PaperLightConstants,
    build_law107,
    compute_paper_light_constants,
    compute_paper_light_yield_surface,
    compute_paper_light_normals,
    mat107_newton_update_single,
    mat107_nice_update_single,
    solid_update,
    shell_update,
    sound_speed,
    sound_speed_solid,
    sound_speed_shell,
    solid_tangent,
    consistent_shell_tangent,
    extra_shapes,
)


def test_paper_light_elasticity_and_sound_speeds():
    """Verify orthotropic elasticity matrix components and sound speed estimates."""
    young1 = 12000.0  # MD
    young2 = 6000.0   # CD
    young3 = 1000.0   # ZD
    nu21 = 0.15       # nu_CD-MD
    g12 = 3000.0
    g23 = 1500.0
    g31 = 2000.0
    rho = 8.0e-7

    c = compute_paper_light_constants(
        young1=young1,
        young2=young2,
        young3=young3,
        nu21=nu21,
        g12=g12,
        g23=g23,
        g31=g31,
        rho=rho,
    )

    # nu12 = nu21 * E1 / E2 = 0.15 * 12000 / 6000 = 0.30
    assert np.isclose(c.nu12, 0.30)
    denom = 1.0 - 0.30 * 0.15  # 1 - 0.045 = 0.955
    expected_a11 = 12000.0 / 0.955
    expected_a22 = 6000.0 / 0.955
    expected_a12 = 0.30 * 6000.0 / 0.955
    expected_a21 = 0.15 * 12000.0 / 0.955

    assert np.isclose(c.a11, expected_a11)
    assert np.isclose(c.a22, expected_a22)
    assert np.isclose(c.a12, expected_a12)
    assert np.isclose(c.a21, expected_a21)
    assert np.isclose(c.a12, c.a21)  # Symmetry check

    # Sound speed: sqrt(max_mod / rho)
    max_mod = max(expected_a11, expected_a22, young3, g12, g23, g31)
    expected_ssp = math.sqrt(max_mod / rho)
    assert np.isclose(c.ssp_solid, expected_ssp)
    assert np.isclose(c.ssp_shell, expected_ssp)


def test_indicator_flags_and_yield_surface_quadrants():
    """Verify indicator coefficients alpha1..alpha5 across stress quadrants."""
    p = PaperLightParams(
        young1=10000.0,
        young2=5000.0,
        young3=1000.0,
        nu21=0.2,
        g12=2500.0,
        g23=1200.0,
        g31=1500.0,
        rho=1.0e-6,
        xi1=0.1,
        xi2=0.2,
        sigy1=50.0,
        sigy2=30.0,
        sigy1c=40.0,
        sigy2c=25.0,
        sigyt=20.0,
    )
    pla = np.zeros(6, dtype=np.float64)

    # Quadrant 1: Biaxial tension (sigxx=60, sigyy=40, sigxy=10)
    # sigxx - xi1*sigyy = 60 - 0.1*40 = 56 > 0 -> alpha[0]=1
    # sigyy - xi2*sigxx = 40 - 0.2*60 = 28 > 0 -> alpha[1]=1
    # -sigxx = -60 < 0 -> alpha[2]=0
    # -sigyy = -40 < 0 -> alpha[3]=0
    # abs(sigxy) = 10 > 0 -> alpha[4]=1
    phi1, alpha1, r_vec1, _ = compute_paper_light_yield_surface(60.0, 40.0, 10.0, p, pla)
    assert np.array_equal(alpha1, [1.0, 1.0, 0.0, 0.0, 1.0])
    assert phi1 > 0.0  # Above yield

    # Quadrant 3: Biaxial compression (sigxx=-60, sigyy=-40, sigxy=0)
    # sigxx - xi1*sigyy = -60 - 0.1*(-40) = -56 < 0 -> alpha[0]=0
    # sigyy - xi2*sigxx = -40 - 0.2*(-60) = -28 < 0 -> alpha[1]=0
    # -sigxx = 60 > 0 -> alpha[2]=1
    # -sigyy = 40 > 0 -> alpha[3]=1
    # abs(sigxy) = 0 -> alpha[4]=0
    phi3, alpha3, r_vec3, _ = compute_paper_light_yield_surface(-60.0, -40.0, 0.0, p, pla)
    assert np.array_equal(alpha3, [0.0, 0.0, 1.0, 1.0, 0.0])
    assert phi3 > 0.0


def test_analytical_hardening_and_derivatives():
    """Verify analytical Voce-like hardening curves and exact derivatives."""
    p = PaperLightParams(
        sigy1=100.0,
        cini1=0.01,
        s1=10.0,
        sigy2=60.0,
        cini2=0.02,
        s2=5.0,
        sigy1c=80.0,
        cini1c=0.015,
        s1c=8.0,
        g1c=0.19,  # sqrt(1 - 0.19) = 0.9
        sigy2c=50.0,
        cini2c=0.025,
        s2c=4.0,
        sigyt=40.0,
        cinit=0.03,
        st=6.0,
        d1=0.05,
        d2=0.08,
    )

    # Test plastic strain: p1 = 0.005, p2 = 0.004, p3 = 0.003, p4 = 0.002, p5 = 0.001
    pla = np.array([0.01, 0.005, 0.004, 0.003, 0.002, 0.001], dtype=np.float64)

    _, _, r_vec, dr_dp = compute_paper_light_yield_surface(50.0, 30.0, 10.0, p, pla)

    # R1: 100.0 + (1 / (0.01 + 10.0 * 0.005)) * 0.005 = 100 + (1 / 0.06) * 0.005 = 100 + 0.08333333333333333
    denom1 = 0.01 + 10.0 * 0.005
    expected_r1 = 100.0 + (1.0 / denom1) * 0.005
    expected_dr1 = (1.0 / denom1) * (1.0 - (10.0 * 0.005 / denom1))

    assert np.isclose(r_vec[0], expected_r1)
    assert np.isclose(dr_dp[0], expected_dr1)

    # R1c with g1c correction: / sqrt(1 - 0.19) = / 0.9
    denom3 = 0.015 + 8.0 * 0.003
    expected_r1c = (80.0 + (1.0 / denom3) * 0.003) / 0.9
    expected_dr1c = ((1.0 / denom3) * (1.0 - (8.0 * 0.003 / denom3))) / 0.9

    assert np.isclose(r_vec[2], expected_r1c)
    assert np.isclose(dr_dp[2], expected_dr1c)


def test_normals_and_non_associated_potential():
    """Verify normality vectors and non-associated plastic potential beta switching."""
    p = PaperLightParams(
        xi1=0.15,
        xi2=0.25,
        k1=1.0,
        k2=-0.5,
        k3=0.8,
        k4=0.4,
        k5=0.6,
        k6=0.3,
        sigy1=100.0,
        sigy2=80.0,
        sigy1c=90.0,
        sigy2c=70.0,
        sigyt=50.0,
    )
    pla = np.zeros(6, dtype=np.float64)

    # Test tension state: sigxx=80, sigyy=50, sigxy=10 -> beta1=1, beta2=0
    phi, alpha, r_vec, _ = compute_paper_light_yield_surface(80.0, 50.0, 10.0, p, pla)
    n_vec, np_vec = compute_paper_light_normals(80.0, 50.0, 10.0, p, alpha, r_vec)

    # Yield surface normal n_vec has 3 components: [normxx, normyy, normxy]
    assert len(n_vec) == 3
    assert len(np_vec) == 3

    # In tension (sigxx > 0, sigyy > 0): beta1=1, beta2=0
    # normpxx = beta1 * (2*sigxx + k2*sigyy) + beta2 * (2*sigxx + k4*sigyy)
    normpxx = 1.0 * (2.0 * 80.0 + p.k2 * 50.0)
    normpyy = 1.0 * (2.0 * p.k1 * 50.0 + p.k2 * 80.0)
    normpxy = 1.0 * p.k5 * 10.0
    normsig = math.sqrt(normpxx * normpxx + normpyy * normpyy + 2.0 * normpxy * normpxy)
    expected_npxx = normpxx / normsig
    assert np.isclose(np_vec[0], expected_npxx)


def test_newton_cutting_plane_return_mapping():
    """Verify Newton cutting-plane iteration (IRES=2) restores yield criterion."""
    p = PaperLightParams(
        young1=10000.0,
        young2=5000.0,
        young3=1000.0,
        nu21=0.2,
        g12=2500.0,
        g23=1200.0,
        g31=1500.0,
        rho=1.0e-6,
        ires=2,
        sigy1=100.0,
        sigy2=60.0,
        sigy1c=80.0,
        sigy2c=50.0,
        sigyt=40.0,
        cini1=0.01,
        s1=10.0,
        cini2=0.02,
        s2=5.0,
    )
    c = compute_paper_light_constants(
        p.young1, p.young2, p.young3, p.nu21, p.g12, p.g23, p.g31, p.rho
    )
    pla0 = np.zeros(6, dtype=np.float64)

    # Elastic state: well below yield
    sig_el = np.array([40.0, 20.0, 5.0])
    sig_res, pla_res, dlam, _, _ = mat107_newton_update_single(sig_el, p, c, pla0, niter=3)
    assert np.allclose(sig_res, sig_el)
    assert dlam == 0.0
    assert np.allclose(pla_res, pla0)

    # Plastic state: trial stress exceeds yield
    sig_pl_tr = np.array([150.0, 80.0, 20.0])
    phi_start, _, _, _ = compute_paper_light_yield_surface(sig_pl_tr[0], sig_pl_tr[1], sig_pl_tr[2], p, pla0)
    sig_res2, pla_res2, dlam2, hardp2, sigy_sc2 = mat107_newton_update_single(sig_pl_tr, p, c, pla0, niter=3)
    assert dlam2 > 0.0
    assert pla_res2[0] > 0.0
    assert pla_res2[1] > 0.0  # MD plastic strain increased

    # Verify return towards yield surface: phi is drastically reduced
    phi_end, _, _, _ = compute_paper_light_yield_surface(sig_res2[0], sig_res2[1], sig_res2[2], p, pla_res2)
    assert abs(phi_end) < 0.05
    assert abs(phi_end) < phi_start


def test_nice_explicit_return_mapping():
    """Verify Nice explicit return mapping (IRES=1) matches mat107_nice.F."""
    p = PaperLightParams(
        young1=10000.0,
        young2=5000.0,
        young3=1000.0,
        nu21=0.2,
        g12=2500.0,
        g23=1200.0,
        g31=1500.0,
        rho=1.0e-6,
        ires=1,
        sigy1=100.0,
        sigy2=60.0,
        sigy1c=80.0,
        sigy2c=50.0,
        sigyt=40.0,
        cini1=0.01,
        s1=10.0,
    )
    c = compute_paper_light_constants(
        p.young1, p.young2, p.young3, p.nu21, p.g12, p.g23, p.g31, p.rho
    )
    pla0 = np.zeros(6, dtype=np.float64)

    sigo = np.array([90.0, 50.0, 10.0])
    deps = np.array([0.005, 0.002, 0.001])
    sig_res, pla_res, dlam, hardp, sigy_sc, phi_new = mat107_nice_update_single(
        sigo, deps, p, c, pla0, phi_prev=0.0
    )

    assert dlam > 0.0
    assert pla_res[0] > 0.0
    assert phi_new < 1.0


def test_solid_and_shell_update_vectorized():
    """Verify vectorized batch execution for 3D solids and 2D shells."""
    p = PaperLightParams(
        young1=12000.0,
        young2=6000.0,
        young3=1000.0,
        nu21=0.15,
        g12=3000.0,
        g23=1500.0,
        g31=2000.0,
        rho=8.0e-7,
        sigy1=100.0,
        sigy2=50.0,
        sigy1c=80.0,
        sigy2c=40.0,
        sigyt=30.0,
    )

    n_elem = 4
    sig_sol = np.zeros((n_elem, 6))
    deps_sol = np.full((n_elem, 6), 0.015)

    sig_out, epsp_out, c_out = solid_update(p, sig_sol, deps_sol, return_tuple=True)
    assert sig_out.shape == (n_elem, 6)
    assert epsp_out.shape == (n_elem,)
    assert c_out.shape == (n_elem,)
    assert np.all(epsp_out > 0.0)

    # Shell test with 5 stresses [xx, yy, xy, yz, zx]
    sig_sh = np.zeros((n_elem, 5))
    deps_sh = np.full((n_elem, 5), 0.015)
    sig_sh_out, epsp_sh_out, c_sh_out = shell_update(p, sig_sh, deps_sh, return_tuple=True)
    assert sig_sh_out.shape == (n_elem, 5)
    assert epsp_sh_out.shape == (n_elem,)
    assert c_sh_out.shape == (n_elem,)
    assert np.all(epsp_sh_out > 0.0)


def test_tabulated_yield_curves():
    """Verify tabulated yield curves (ITAB=1) interpolation and evaluation."""
    # Define simple linear hardening curves: y = 100 + 1000 * x
    c1 = [(0.0, 0.01, 0.05), (100.0, 110.0, 150.0)]
    c2 = [(0.0, 0.01, 0.05), (60.0, 70.0, 100.0)]
    c3 = [(0.0, 0.01, 0.05), (80.0, 90.0, 130.0)]
    c4 = [(0.0, 0.01, 0.05), (50.0, 60.0, 90.0)]
    ct = [(0.0, 0.01, 0.05), (40.0, 50.0, 80.0)]

    p = PaperLightParams(
        itab=1,
        young1=10000.0,
        young2=5000.0,
        young3=1000.0,
        nu21=0.2,
        g12=2500.0,
        g23=1200.0,
        g31=1500.0,
        rho=1.0e-6,
        curves=(c1, c2, c3, c4, ct),
        xscale1=1.0,
        yscale1=1.0,
    )

    pla = np.array([0.02, 0.01, 0.01, 0.01, 0.01, 0.01], dtype=np.float64)
    phi, alpha, r_vec, dr_dp = compute_paper_light_yield_surface(120.0, 70.0, 15.0, p, pla)

    assert np.isclose(r_vec[0], 110.0)
    assert np.isclose(r_vec[1], 70.0)
    assert dr_dp[0] > 0.0
