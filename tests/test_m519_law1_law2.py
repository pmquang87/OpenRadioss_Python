"""
Unit tests for Milestone M519:
Foundational Constitutive Material Laws (/MAT/LAW1 /MAT/ELAST & /MAT/LAW2 /MAT/PLAS_JOHNS)
Hardening, 3D & Plane-Stress Radial Return, Adiabatic Thermal Softening,
Algorithmic Consistent Tangents, Rate Sensitivity, and Comprehensive Unit Tests.

Fortran origin:
  engine/source/materials/mat/mat001/sigeps01.F
  engine/source/materials/mat/mat001/sigeps01c.F
  engine/source/materials/mat/mat002/sigeps02.F
  engine/source/materials/mat/mat002/sigeps02c.F
"""

import numpy as np
import pytest

from pyradioss.materials import law01_elastic, law02_johnson_cook
from pyradioss.model.entities import Material


def _make_elastic_mat(E=210000.0, nu=0.3, rho0=7800.0):
    return Material(id=1, law=1, rho0=rho0, params={"E": E, "nu": nu})


def _make_jc_mat(E=210000.0, nu=0.3, rho0=7800.0, A=300.0, B=400.0, n=0.5,
                 sig_max=800.0, c=0.0, eps_dot_0=1.0, T_melt=1800.0,
                 T_i=293.0, rho_cp=3.5e6, mT=1.0):
    params = {
        "E": E,
        "nu": nu,
        "A": A,
        "B": B,
        "n": n,
        "sig_max": sig_max,
        "c": c,
        "eps_dot_0": eps_dot_0,
        "T_melt": T_melt,
        "T_i": T_i,
        "rho_cp": rho_cp,
        "mT": mT,
    }
    return Material(id=2, law=2, rho0=rho0, params=params)


# ============================================================================
# 1. LAW1 Empty Arrays Guard
# ============================================================================
def test_law1_empty_arrays():
    """Empty inputs to solid_update and shell_update return cleanly."""
    mat = _make_elastic_mat()
    sig6 = np.empty((0, 6))
    deps6 = np.empty((0, 6))
    out6 = law01_elastic.solid_update(mat, sig6, deps6)
    assert out6.shape == (0, 6)

    sig3 = np.empty((0, 3))
    deps3 = np.empty((0, 3))
    out3 = law01_elastic.shell_update(mat, sig3, deps3)
    assert out3.shape == (0, 3)


# ============================================================================
# 2. LAW1 3D Solid Uniaxial Strain Response
# ============================================================================
def test_law1_solid_uniaxial_stress():
    """1D strain increment deps11 produces exact (lambda + 2G)*deps11 and lateral lambda*deps11."""
    E = 210000.0
    nu = 0.3
    mat = _make_elastic_mat(E=E, nu=nu)
    G = E / (2.0 * (1.0 + nu))
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

    sig = np.zeros((1, 6))
    deps = np.array([[0.001, 0.0, 0.0, 0.0, 0.0, 0.0]])
    law01_elastic.solid_update(mat, sig, deps)

    expected_11 = (lam + 2.0 * G) * 0.001
    expected_22 = lam * 0.001
    expected_33 = lam * 0.001

    assert np.isclose(sig[0, 0], expected_11)
    assert np.isclose(sig[0, 1], expected_22)
    assert np.isclose(sig[0, 2], expected_33)
    assert np.allclose(sig[0, 3:], 0.0)


# ============================================================================
# 3. LAW1 3D Solid Pure Shear Response
# ============================================================================
def test_law1_solid_pure_shear():
    """Engineering shear strains produce tau = G * gamma on all shear components."""
    mat = _make_elastic_mat()
    G = mat.G
    sig = np.zeros((1, 6))
    deps = np.array([[0.0, 0.0, 0.0, 0.002, -0.0015, 0.003]])
    law01_elastic.solid_update(mat, sig, deps)

    assert np.allclose(sig[0, :3], 0.0)
    assert np.isclose(sig[0, 3], G * 0.002)
    assert np.isclose(sig[0, 4], G * (-0.0015))
    assert np.isclose(sig[0, 5], G * 0.003)


# ============================================================================
# 4. LAW1 Shell Plane Stress Uniaxial & Shear Response
# ============================================================================
def test_law1_shell_uniaxial_stress():
    """Plane stress Hooke update produces E/(1-nu^2) * (dxx + nu*dyy) and G*dgamma."""
    E = 210000.0
    nu = 0.3
    mat = _make_elastic_mat(E=E, nu=nu)
    c = E / (1.0 - nu ** 2)
    G = mat.G

    sig = np.zeros((1, 3))
    deps = np.array([[0.001, -0.0002, 0.0015]])
    law01_elastic.shell_update(mat, sig, deps)

    expected_xx = c * (0.001 + nu * (-0.0002))
    expected_yy = c * (-0.0002 + nu * 0.001)
    expected_xy = G * 0.0015

    assert np.isclose(sig[0, 0], expected_xx)
    assert np.isclose(sig[0, 1], expected_yy)
    assert np.isclose(sig[0, 2], expected_xy)


# ============================================================================
# 5. LAW1 Tangent Spectral Properties & Compliance Inverse
# ============================================================================
def test_law1_tangent_spectral_properties():
    """solid_tangent and shell_membrane_tangent are symmetric positive-definite."""
    mat = _make_elastic_mat()
    C_solid = law01_elastic.solid_tangent(mat)
    assert np.allclose(C_solid, C_solid.T)
    eig_solid = np.linalg.eigvalsh(C_solid)
    assert np.all(eig_solid > 0.0)

    # Compliance inverse check for solid
    E, nu, G = mat.E, mat.nu, mat.G
    S_solid = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            S_solid[i, j] = -nu / E if i != j else 1.0 / E
    for k in range(3, 6):
        S_solid[k, k] = 1.0 / G
    assert np.allclose(C_solid @ S_solid, np.eye(6), atol=1e-12)

    # Shell membrane tangent
    C_shell = law01_elastic.shell_membrane_tangent(mat)
    assert np.allclose(C_shell, C_shell.T)
    eig_shell = np.linalg.eigvalsh(C_shell)
    assert np.all(eig_shell > 0.0)


# ============================================================================
# 6. LAW2 Empty Arrays Guard
# ============================================================================
def test_law2_empty_arrays():
    """Empty inputs to solid_update, shell_update, and tangents return empty arrays."""
    mat = _make_jc_mat()
    sig6 = np.empty((0, 6))
    deps6 = np.empty((0, 6))
    epsp = np.empty(0)
    out_s, out_ep = law02_johnson_cook.solid_update(mat, sig6, deps6, epsp, 0.001)
    assert out_s.shape == (0, 6)
    assert len(out_ep) == 0

    sig3 = np.empty((0, 3))
    deps3 = np.empty((0, 3))
    out_sh, out_sh_ep = law02_johnson_cook.shell_update(mat, sig3, deps3, epsp, 0.001)
    assert out_sh.shape == (0, 3)
    assert len(out_sh_ep) == 0

    D_solid = law02_johnson_cook.consistent_solid_tangent(mat, sig6, epsp, epsp)
    assert D_solid.shape == (0, 6, 6)

    D_shell = law02_johnson_cook.consistent_shell_tangent(mat, sig3, epsp, epsp)
    assert D_shell.shape == (0, 3, 3)


# ============================================================================
# 7. LAW2 Solid Elastic Trial (Below Yield)
# ============================================================================
def test_law2_solid_elastic_trial():
    """Strain increment within yield surface remains purely elastic with zero plastic strain."""
    mat = _make_jc_mat(A=300.0)
    sig = np.zeros((1, 6))
    deps = np.array([[0.0005, 0.0, 0.0, 0.0, 0.0, 0.0]])
    epsp = np.zeros(1)

    law02_johnson_cook.solid_update(mat, sig, deps, epsp, 0.001)

    # Plastic strain must remain zero
    assert epsp[0] == 0.0
    # Stress must equal Hooke's law prediction
    sig_elastic = np.zeros((1, 6))
    law01_elastic.solid_update(mat, sig_elastic, deps)
    assert np.allclose(sig, sig_elastic)


# ============================================================================
# 8. LAW2 Solid Plastic Yielding & Consistency
# ============================================================================
def test_law2_solid_plastic_yielding():
    """Large strain exceeding yield surface converges von Mises stress exactly onto yield curve."""
    mat = _make_jc_mat(A=300.0, B=400.0, n=0.5)
    sig = np.zeros((1, 6))
    deps = np.array([[0.005, 0.0, 0.0, 0.0, 0.0, 0.0]])
    epsp = np.zeros(1)

    law02_johnson_cook.solid_update(mat, sig, deps, epsp, 0.001)

    assert epsp[0] > 0.0
    p = np.mean(sig[0, :3])
    s = sig[0].copy()
    s[:3] -= p
    j2 = 0.5 * (s[0] ** 2 + s[1] ** 2 + s[2] ** 2) + s[3] ** 2 + s[4] ** 2 + s[5] ** 2
    q_conv = np.sqrt(3.0 * j2)
    q_expected = 300.0 + 400.0 * (epsp[0] ** 0.5)

    assert np.isclose(q_conv, q_expected, rtol=1e-6)


# ============================================================================
# 9. LAW2 Solid Radial Collinearity & Pressure Conservation
# ============================================================================
def test_law2_solid_radial_direction():
    """Plastic deviatoric stress is collinear with trial deviator, and pressure is untouched."""
    mat = _make_jc_mat(A=300.0, B=400.0, n=0.5)
    # 3D mixed shear and normal strain
    deps = np.array([[0.004, -0.002, 0.001, 0.003, 0.001, -0.002]])
    sig_trial = np.zeros((1, 6))
    law01_elastic.solid_update(mat, sig_trial, deps)
    p_trial = np.mean(sig_trial[0, :3])
    s_trial = sig_trial[0].copy()
    s_trial[:3] -= p_trial

    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    law02_johnson_cook.solid_update(mat, sig, deps, epsp, 0.001)
    p_conv = np.mean(sig[0, :3])
    s_conv = sig[0].copy()
    s_conv[:3] -= p_conv

    # Check pressure untouched
    assert np.isclose(p_conv, p_trial, atol=1e-12)
    # Check collinearity: s_conv / ||s_conv|| == s_trial / ||s_trial||
    norm_trial = np.linalg.norm(s_trial)
    norm_conv = np.linalg.norm(s_conv)
    assert np.allclose(s_conv / norm_conv, s_trial / norm_trial, atol=1e-12)


# ============================================================================
# 10. LAW2 Solid Hydrostatic Invariance
# ============================================================================
def test_law2_solid_hydrostatic_invariance():
    """Pure hydrostatic strain increment produces zero deviatoric stress and zero plastic strain."""
    mat = _make_jc_mat(A=300.0)
    eps_v = 0.003
    deps = np.array([[eps_v / 3.0, eps_v / 3.0, eps_v / 3.0, 0.0, 0.0, 0.0]])
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)

    law02_johnson_cook.solid_update(mat, sig, deps, epsp, 0.001)

    assert epsp[0] == 0.0
    p = np.mean(sig[0, :3])
    expected_p = mat.K * eps_v
    assert np.isclose(p, expected_p)
    assert np.allclose(sig[0, 3:], 0.0)


# ============================================================================
# 11. LAW2 Solid Power-Law Work Hardening Accumulation
# ============================================================================
def test_law2_solid_power_law_hardening():
    """Successive plastic increments accumulate plastic strain with exact yield surface growth."""
    mat = _make_jc_mat(A=250.0, B=350.0, n=0.4)
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    deps = np.array([[0.003, 0.0, 0.0, 0.0, 0.0, 0.0]])

    for _ in range(8):
        law02_johnson_cook.solid_update(mat, sig, deps, epsp, 0.001)
        p = np.mean(sig[0, :3])
        s = sig[0].copy()
        s[:3] -= p
        j2 = 0.5 * np.sum(s[:3] ** 2) + np.sum(s[3:] ** 2)
        q = np.sqrt(3.0 * j2)
        expected_sy = 250.0 + 350.0 * (epsp[0] ** 0.4)
        assert np.isclose(q, expected_sy, rtol=1e-5)


# ============================================================================
# 12. LAW2 Solid sig_max Capping & Zero Hardening Slope
# ============================================================================
def test_law2_solid_sig_max_cap():
    """When flow stress reaches sig_max, yield stress is capped and hardening slope becomes 0."""
    sig_cap = 320.0
    mat = _make_jc_mat(A=300.0, B=200.0, n=0.5, sig_max=sig_cap)
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    # Large strain to exceed cap
    deps = np.array([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]])
    law02_johnson_cook.solid_update(mat, sig, deps, epsp, 0.001)

    p = np.mean(sig[0, :3])
    s = sig[0].copy()
    s[:3] -= p
    j2 = 0.5 * np.sum(s[:3] ** 2) + np.sum(s[3:] ** 2)
    q = np.sqrt(3.0 * j2)
    assert np.isclose(q, sig_cap, rtol=1e-6)

    # Check hardening slope returned by helper is 0
    sy, H = law02_johnson_cook._yield_stress(mat, epsp, 1.0)
    assert np.isclose(sy[0], sig_cap)
    assert H[0] == 0.0


# ============================================================================
# 13. LAW2 Strain-Rate Sensitivity
# ============================================================================
def test_law2_strain_rate_sensitivity():
    """Strain rate exceeding eps_dot_0 scales yield stress by 1 + c*ln(eps_dot / eps_dot_0)."""
    c_rate = 0.04
    eps0 = 1.0
    mat = _make_jc_mat(A=300.0, B=0.0, c=c_rate, eps_dot_0=eps0)
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)

    # deps = 0.002 over dt = 1e-4 -> strain rate = 20.0
    dt = 1e-4
    deps = np.array([[0.002, -0.001, -0.001, 0.0, 0.0, 0.0]])
    law02_johnson_cook.solid_update(mat, sig, deps, epsp, dt)

    p = np.mean(sig[0, :3])
    s = sig[0].copy()
    s[:3] -= p
    q = np.sqrt(3.0 * (0.5 * np.sum(s[:3] ** 2) + np.sum(s[3:] ** 2)))

    # Compute expected equivalent strain rate
    tr3 = np.sum(deps[0, :3]) / 3.0
    exx, eyy, ezz = deps[0, 0] - tr3, deps[0, 1] - tr3, deps[0, 2] - tr3
    ee = exx ** 2 + eyy ** 2 + ezz ** 2
    rate = np.sqrt((2.0 / 3.0) * ee) / dt
    expected_factor = 1.0 + c_rate * np.log(rate / eps0)
    expected_sy = 300.0 * expected_factor

    assert np.isclose(q, expected_sy, rtol=1e-5)


# ============================================================================
# 14. LAW2 Zero/Negative dt Safe Guard
# ============================================================================
def test_law2_zero_dt_guard():
    """dt <= 0.0 treats strain rate as 0 without divide-by-zero or explosive hardening."""
    mat = _make_jc_mat(A=300.0, B=200.0, c=0.05, eps_dot_0=1.0)
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    deps = np.array([[0.003, 0.0, 0.0, 0.0, 0.0, 0.0]])

    # dt = 0.0 should not error or produce inf
    law02_johnson_cook.solid_update(mat, sig, deps, epsp, 0.0)
    assert np.all(np.isfinite(sig))
    assert np.all(np.isfinite(epsp))

    # dt = -1.0 safe
    sig_neg = np.zeros((1, 6))
    epsp_neg = np.zeros(1)
    law02_johnson_cook.solid_update(mat, sig_neg, deps, epsp_neg, -1.0)
    assert np.allclose(sig, sig_neg)
    assert np.allclose(epsp, epsp_neg)


# ============================================================================
# 15. LAW2 Adiabatic Deformation Heating
# ============================================================================
def test_law2_adiabatic_temperature_rise():
    """Plastic work heats material with dT = sigma_y * d(epsp) / rho_Cp."""
    rho_cp = 3.5e6
    mat = _make_jc_mat(A=300.0, B=0.0, rho_cp=rho_cp, mT=1.0)
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    temp = np.zeros(1)
    extra = {"temp": temp}

    deps = np.array([[0.005, 0.0, 0.0, 0.0, 0.0, 0.0]])
    law02_johnson_cook.solid_update(mat, sig, deps, epsp, 0.001, extra=extra)

    dl = epsp[0]
    sy = 300.0
    expected_dT = sy * dl / rho_cp
    assert temp[0] > 0.0
    assert np.isclose(temp[0], expected_dT, rtol=1e-5)


# ============================================================================
# 16. LAW2 Homologous Thermal Softening
# ============================================================================
def test_law2_thermal_softening():
    """Prescribed temperature rise reduces yield stress by exact (1 - T*^mT) factor."""
    T_i = 293.0
    T_melt = 1800.0
    mT = 1.0
    mat = _make_jc_mat(A=400.0, B=0.0, T_i=T_i, T_melt=T_melt, mT=mT)

    temp_rise = 450.0  # T = 743.0
    extra = {"temp": np.array([temp_rise])}
    tstar = temp_rise / (T_melt - T_i)
    softening = 1.0 - (tstar ** mT)
    expected_sy = 400.0 * softening

    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    deps = np.array([[0.004, 0.0, 0.0, 0.0, 0.0, 0.0]])
    law02_johnson_cook.solid_update(mat, sig, deps, epsp, 0.001, extra=extra)

    p = np.mean(sig[0, :3])
    s = sig[0].copy()
    s[:3] -= p
    q = np.sqrt(3.0 * (0.5 * np.sum(s[:3] ** 2) + np.sum(s[3:] ** 2)))
    assert np.isclose(q, expected_sy, rtol=1e-5)


# ============================================================================
# 17. LAW2 Shell Plane Stress Radial Projection (Iplas=2)
# ============================================================================
def test_law2_shell_plane_stress_return():
    """In-plane stress radial projection converges plane-stress von Mises onto yield surface."""
    mat = _make_jc_mat(A=280.0, B=320.0, n=0.5)
    sig = np.zeros((1, 3))
    deps = np.array([[0.004, -0.001, 0.003]])
    epsp = np.zeros(1)

    law02_johnson_cook.shell_update(mat, sig, deps, epsp, 0.001)

    assert epsp[0] > 0.0
    sxx, syy, sxy = sig[0, 0], sig[0, 1], sig[0, 2]
    q_conv = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2)
    expected_sy = 280.0 + 320.0 * (epsp[0] ** 0.5)

    assert np.isclose(q_conv, expected_sy, rtol=1e-5)


# ============================================================================
# 18. Consistent Solid Algorithmic Tangent Directional Derivative
# ============================================================================
def test_law2_consistent_solid_tangent_derivative():
    """Finite difference of solid_update matches consistent_solid_tangent to high accuracy."""
    mat = _make_jc_mat(A=300.0, B=400.0, n=0.5)
    # Prior state: already in plastic flow
    sig_init = np.array([[200.0, -100.0, -100.0, 50.0, 0.0, 0.0]])
    epsp_init = np.array([0.002])

    deps_base = np.array([[0.002, -0.0005, -0.0005, 0.001, 0.0, 0.0]])
    sig = sig_init.copy()
    epsp = epsp_init.copy()
    law02_johnson_cook.solid_update(mat, sig, deps_base, epsp, 0.001)
    dep_incr = epsp - epsp_init

    D_alg = law02_johnson_cook.consistent_solid_tangent(mat, sig, epsp, dep_incr)[0]

    # Check finite difference derivative in an arbitrary direction delta_eps
    delta_eps = np.array([[0.0005, -0.0002, 0.0001, 0.0004, -0.0002, 0.0003]])
    eps = 1e-7

    sig_p = sig_init.copy()
    epsp_p = epsp_init.copy()
    law02_johnson_cook.solid_update(mat, sig_p, deps_base + eps * delta_eps, epsp_p, 0.001)

    sig_m = sig_init.copy()
    epsp_m = epsp_init.copy()
    law02_johnson_cook.solid_update(mat, sig_m, deps_base - eps * delta_eps, epsp_m, 0.001)

    d_sig_num = (sig_p[0] - sig_m[0]) / (2.0 * eps)
    d_sig_tan = D_alg @ delta_eps[0]

    rel_err = np.linalg.norm(d_sig_num - d_sig_tan) / np.linalg.norm(d_sig_tan)
    assert rel_err < 1e-5


# ============================================================================
# 19. Consistent Shell Algorithmic Tangent Directional Derivative
# ============================================================================
def test_law2_consistent_shell_tangent_derivative():
    """Finite difference of shell_update matches consistent_shell_tangent to high accuracy."""
    mat = _make_jc_mat(A=280.0, B=350.0, n=0.5)
    sig_init = np.array([[180.0, -80.0, 40.0]])
    epsp_init = np.array([0.0015])

    deps_base = np.array([[0.002, -0.0008, 0.0012]])
    sig = sig_init.copy()
    epsp = epsp_init.copy()
    law02_johnson_cook.shell_update(mat, sig, deps_base, epsp, 0.001)
    dep_incr = epsp - epsp_init

    D_shell = law02_johnson_cook.consistent_shell_tangent(mat, sig, epsp, dep_incr)[0]

    delta_eps = np.array([[0.0004, -0.0003, 0.0005]])
    eps = 1e-7

    sig_p = sig_init.copy()
    epsp_p = epsp_init.copy()
    law02_johnson_cook.shell_update(mat, sig_p, deps_base + eps * delta_eps, epsp_p, 0.001)

    sig_m = sig_init.copy()
    epsp_m = epsp_init.copy()
    law02_johnson_cook.shell_update(mat, sig_m, deps_base - eps * delta_eps, epsp_m, 0.001)

    d_sig_num = (sig_p[0] - sig_m[0]) / (2.0 * eps)
    d_sig_tan = D_shell @ delta_eps[0]

    rel_err = np.linalg.norm(d_sig_num - d_sig_tan) / np.linalg.norm(d_sig_tan)
    assert rel_err < 1e-5


# ============================================================================
# 20. LAW2 Elastic Unloading & Kuhn-Tucker Irreversibility
# ============================================================================
def test_law2_elastic_unloading_kuhn_tucker():
    """Reversing strain after plastic loading unloads elastically with frozen plastic strain."""
    mat = _make_jc_mat(A=300.0, B=400.0, n=0.5)
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)

    # Step 1: forward plastic load
    deps_fwd = np.array([[0.005, 0.0, 0.0, 0.0, 0.0, 0.0]])
    law02_johnson_cook.solid_update(mat, sig, deps_fwd, epsp, 0.001)
    epsp_peak = epsp[0]
    assert epsp_peak > 0.0

    # Step 2: reverse strain (unloading)
    deps_rev = np.array([[-0.001, 0.0, 0.0, 0.0, 0.0, 0.0]])
    law02_johnson_cook.solid_update(mat, sig, deps_rev, epsp, 0.001)

    # Plastic strain must remain exactly frozen
    assert epsp[0] == epsp_peak

    # Stress must now sit strictly inside the yield surface
    p = np.mean(sig[0, :3])
    s = sig[0].copy()
    s[:3] -= p
    j2 = 0.5 * np.sum(s[:3] ** 2) + np.sum(s[3:] ** 2)
    q = np.sqrt(3.0 * j2)
    current_yield = 300.0 + 400.0 * (epsp_peak ** 0.5)
    assert q < current_yield
