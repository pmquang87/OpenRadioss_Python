"""
Unit tests for LAW48 (Zhao constitutive model) — Milestone M552.

Covers:
1. Sound speeds (solid and shell)
2. Yield stress evaluation:
   - Work hardening P_A (static yield, cap at eps_p_max)
   - Coupled strain-rate sensitivity P_B (rate cap eps_dot_0, plastic strain decay)
   - High-rate power law P_C
   - Total yield saturation S_max + P_C
   - Hardening slope H (set to 0 if capped)
3. Tensile failure factor FAIL and 3D/2D maximum principal strain solvers
4. Element deletion (eps_p > eps_p_max or eps_t >= eps_r2, off = 0.0, sigma = 0)
5. Elastic trial and radial return for solids
6. Plane-stress enforcement and thickness thinning for shells
7. Mixed isotropic / kinematic hardening (Bauschinger effect upon strain reversal)
8. Strain rate filtering using cutoff frequency f_cut
9. Algorithmic consistent tangents for solids (6x6) and shells (3x3)
10. Material builder and global dispatcher registration
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law48_zhao import (
    Law48Params,
    build_law48,
    eval_yield_and_hardening,
    _principal_strain_3d,
    _principal_strain_2d,
    tensile_failure_factor,
    sound_speed_solid_law48,
    sound_speed_shell_law48,
    solid_update_law48,
    shell_update_law48,
    tangent_law48_solid,
    tangent_law48_shell,
    shell_membrane_tangent,
)
import pyradioss.materials as materials


def test_sound_speeds():
    """Verify solid and shell sound speeds match exact analytical formulas."""
    E = 210000.0
    nu = 0.3
    rho0 = 7.8e-9

    p = Law48Params(E=E, nu=nu, rho0=rho0)

    # Theoretical solid sound speed: sqrt((K + 4G/3) / rho0)
    c_solid_expected = math.sqrt((p.K + 4.0 * p.G / 3.0) / rho0)
    c_solid_closed_form = math.sqrt(E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu) * rho0))
    assert math.isclose(c_solid_expected, c_solid_closed_form, rel_tol=1e-12)

    c_solid = sound_speed_solid_law48(p)
    assert math.isclose(c_solid, c_solid_expected, rel_tol=1e-12)

    # Batched solid sound speed
    rhos = np.array([rho0, 2.0 * rho0])
    c_solid_batch = sound_speed_solid_law48(p, rho0=rhos)
    assert np.allclose(c_solid_batch, [c_solid_expected, c_solid_expected / math.sqrt(2.0)])

    # Theoretical shell sound speed: sqrt(E / ((1 - nu^2) * rho0))
    c_shell_expected = math.sqrt(E / ((1.0 - nu * nu) * rho0))
    c_shell = sound_speed_shell_law48(p)
    assert math.isclose(c_shell, c_shell_expected, rel_tol=1e-12)

    c_shell_batch = sound_speed_shell_law48(p, rho0=rhos)
    assert np.allclose(c_shell_batch, [c_shell_expected, c_shell_expected / math.sqrt(2.0)])


def test_yield_static_hardening():
    """Verify static work hardening P_A and cap at eps_p_max."""
    ca = 1.0
    sigy0 = 300.0
    cb = 500.0
    cn = 0.4
    eps_max = 0.2

    p = Law48Params(ca=ca, sigy0=sigy0, cb=cb, cn=cn, eps_max=eps_max)

    # 1. At zero plastic strain: P_A = ca * sigy0 = 300.0
    yld, h, h_iso, h_kin = eval_yield_and_hardening(p, epsp=0.0, eps_dot=0.0)
    assert math.isclose(float(yld), 300.0, rel_tol=1e-10)
    assert float(h) >= 0.0

    # 2. At plastic strain eps_p = 0.05 (< eps_max)
    epsp = 0.05
    pa_expected = 300.0 + 500.0 * (epsp ** 0.4)
    h_expected = 500.0 * 0.4 * (epsp ** (1.0 - 0.4))
    yld, h, _, _ = eval_yield_and_hardening(p, epsp=epsp, eps_dot=0.0)
    assert math.isclose(float(yld), pa_expected, rel_tol=1e-10)
    assert math.isclose(float(h), h_expected, rel_tol=1e-9)

    # 3. Beyond eps_max, P_A is capped at eps_max, and eval_yield_and_hardening zeroes broken yield
    epsp_over = 0.25
    yld_broken, h_broken, _, _ = eval_yield_and_hardening(p, epsp=epsp_over, eps_dot=0.0)
    assert float(yld_broken) == 0.0
    assert float(h_broken) == 0.0


def test_yield_coupled_rate_sensitivity():
    """Verify coupled rate sensitivity P_B = (c_c - c_d * eps_p^c_m) * ln(eps_dot / eps_dot_0)."""
    p = Law48Params(
        ca=1.0, sigy0=250.0, cb=0.0, cn=1.0,
        cc=40.0, cd=20.0, cm=0.5,
        eps0=1.0,  # ref rate = 1.0 s^-1
        ce=0.0, ck=1.0,
    )

    # Below reference rate: P_B = 0
    yld, _, _, _ = eval_yield_and_hardening(p, epsp=0.0, eps_dot=0.5)
    assert math.isclose(float(yld), 250.0, rel_tol=1e-10)

    # At reference rate: P_B = 0
    yld, _, _, _ = eval_yield_and_hardening(p, epsp=0.0, eps_dot=1.0)
    assert math.isclose(float(yld), 250.0, rel_tol=1e-10)

    # At high rate eps_dot = 100 s^-1, eps_p = 0: P_B = 40 * ln(100)
    edot = 100.0
    pb_expected = 40.0 * math.log(100.0)
    yld, _, _, _ = eval_yield_and_hardening(p, epsp=0.0, eps_dot=edot)
    assert math.isclose(float(yld), 250.0 + pb_expected, rel_tol=1e-9)

    # Coupled: at eps_p = 0.04, (cc - cd * sqrt(0.04)) = (40 - 20 * 0.2) = 36
    epsp = 0.04
    pb_coupled = (40.0 - 20.0 * math.sqrt(epsp)) * math.log(edot)
    yld, _, _, _ = eval_yield_and_hardening(p, epsp=epsp, eps_dot=edot)
    assert math.isclose(float(yld), 250.0 + pb_coupled, rel_tol=1e-9)


def test_yield_power_law_rate_and_saturation():
    """Verify high-rate power law P_C = c_e * eps_dot^c_k and saturation S_max."""
    p = Law48Params(
        ca=1.0, sigy0=200.0, cb=300.0, cn=0.5,
        cc=0.0, cd=0.0, cm=1.0,
        ce=0.05, ck=1.5,
        sig_max=220.0,
    )

    # Check power law term P_C
    edot = 1000.0
    pc_expected = 0.05 * (edot ** 1.5)
    pa = 200.0 + 300.0 * math.sqrt(0.01)  # 200 + 30 = 230
    yy = pa + pc_expected
    # YY = 230 + 1581.1388 > sig_max + P_C = 220 + 1581.1388
    # Capped at sig_max + P_C
    cap_expected = 220.0 + pc_expected
    yld, h, _, _ = eval_yield_and_hardening(p, epsp=0.01, eps_dot=edot)
    assert math.isclose(float(yld), cap_expected, rel_tol=1e-9)
    assert float(h) == 0.0  # Capped -> zero hardening slope


def test_principal_strain_and_tensile_failure():
    """Verify 3D and 2D principal strain solvers and tensile failure factor."""
    # 1. 3D cubic solver on known strain tensor: matches sigeps48.F:201-245 Newton solution
    eps_3d = np.array([[0.10, 0.04, 0.02, 0.0, 0.0, 0.0]])
    epst_3d = _principal_strain_3d(eps_3d)
    assert math.isclose(float(epst_3d[0]), 0.11500174879199618, abs_tol=1e-7)

    # Shear strain state: pure shear [0, 0, 0, 0.06, 0, 0] -> eigenvalues are +/- 0.03
    eps_shear = np.array([[0.0, 0.0, 0.0, 0.06, 0.0, 0.0]])
    epst_shear = _principal_strain_3d(eps_shear)
    assert math.isclose(float(epst_shear[0]), 0.03, abs_tol=1e-7)

    # 2. 2D in-plane principal strain: eps = [0.08, 0.02, 0.04]
    eps_2d = np.array([[0.08, 0.02, 0.04]])
    # (0.08 + 0.02)/2 + sqrt((0.08-0.02)^2 + 0.04^2)/2 = 0.05 + sqrt(0.0036 + 0.0016)/2 = 0.05 + 0.072111/2 = 0.0860555
    epst_2d = _principal_strain_2d(eps_2d)
    expected_2d = 0.5 * (0.08 + 0.02 + math.sqrt((0.08 - 0.02) ** 2 + 0.04 ** 2))
    assert math.isclose(float(epst_2d[0]), expected_2d, rel_tol=1e-9)

    # 3. Tensile failure factor FAIL
    p_fail = Law48Params(eps_t1=0.05, eps_t2=0.15)
    # Below eps_t1: FAIL = 1.0
    assert math.isclose(float(tensile_failure_factor(p_fail, 0.02)), 1.0)
    # Between eps_t1 and eps_t2: linear softening
    assert math.isclose(float(tensile_failure_factor(p_fail, 0.10)), 0.5)
    # At eps_t2: FAIL = 0.0
    assert math.isclose(float(tensile_failure_factor(p_fail, 0.15)), 0.0)
    # Above eps_t2: FAIL = 0.0
    assert math.isclose(float(tensile_failure_factor(p_fail, 0.20)), 0.0)


def test_element_deletion():
    """Verify element deletion when eps_p > eps_p_max or eps_t >= eps_r2."""
    p = Law48Params(
        E=200000.0, nu=0.3, sigy0=300.0, cb=100.0, cn=1.0,
        eps_max=0.10, eps_t1=0.08, eps_t2=0.12,
    )

    # Solid test: element 0 alive, element 1 exceeds eps_max, element 2 exceeds eps_t2
    sig = np.zeros((3, 6))
    deps = np.zeros((3, 6))
    # Element 0: small increment
    deps[0, 0] = 0.001
    # Element 1: epsp starting above eps_max
    epsp = np.array([0.0, 0.11, 0.0])
    # Element 2: tensile strain increment exceeding eps_t2 = 0.12
    deps[2, 0] = 0.13

    extra = {"off": np.ones(3)}
    sig_new, epsp_new, _ = solid_update_law48(p, sig, deps, epsp=epsp, dt=1e-5, extra=extra)

    # Element 0 is alive
    assert extra["off"][0] == 1.0
    assert sig_new[0, 0] > 0.0

    # Element 1 deleted due to eps_p > eps_max
    assert extra["off"][1] == 0.0
    assert np.allclose(sig_new[1], 0.0)

    # Element 2 deleted due to eps_t >= eps_r2
    assert extra["off"][2] == 0.0
    assert np.allclose(sig_new[2], 0.0)

    # Shell test
    sig_sh = np.zeros((2, 3))
    deps_sh = np.zeros((2, 3))
    deps_sh[0, 0] = 0.001
    deps_sh[1, 0] = 0.14
    extra_sh = {"off": np.ones(2)}
    sig_sh_new, _ = shell_update_law48(p, sig_sh, deps_sh, epsp=np.zeros(2), dt=1e-5, extra=extra_sh)
    assert extra_sh["off"][0] == 1.0
    assert extra_sh["off"][1] == 0.0
    assert np.allclose(sig_sh_new[1], 0.0)


def test_solid_elastic_trial_and_plastic_return():
    """Verify solid elastic trial and radial return."""
    p = Law48Params(
        E=210000.0, nu=0.3, ca=1.0, sigy0=300.0,
        cb=400.0, cn=1.0001,
    )

    # 1. Elastic step: deps = [1e-4, 0, 0, 0, 0, 0]
    sig = np.zeros((1, 6))
    deps = np.array([[1e-4, 0.0, 0.0, 0.0, 0.0, 0.0]])
    sig_el, epsp_el, c = solid_update_law48(p, sig.copy(), deps, epsp=0.0, dt=1e-4)

    # Theoretical elastic stress
    dav = 1e-4 / 3.0
    p_exp = p.K * 1e-4
    sxx_exp = 2.0 * p.G * (1e-4 - dav)
    sig_xx_expected = sxx_exp + p_exp
    assert math.isclose(sig_el[0, 0], sig_xx_expected, rel_tol=1e-6)
    assert epsp_el[0] == 0.0

    # 2. Plastic step: large tensile strain deps_xx = 0.01
    deps_pl = np.array([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]])
    sig_pl, epsp_pl, _ = solid_update_law48(p, sig.copy(), deps_pl, epsp=0.0, dt=1e-4)

    # Plastic strain must have increased
    assert epsp_pl[0] > 0.0

    # Deviatoric stress von Mises must match yield stress.
    # In OpenRadioss sigeps48.F:285, for the first plastic step starting from epsp=0,
    # the initial tangent is H = E (pda = E), giving yld_corr = yld + dpla * E = 910.714286.
    pm = (sig_pl[0, 0] + sig_pl[0, 1] + sig_pl[0, 2]) / 3.0
    s_dev = sig_pl[0].copy()
    s_dev[:3] -= pm
    vm = math.sqrt(1.5 * (s_dev[0]**2 + s_dev[1]**2 + s_dev[2]**2 + 2.0 * (s_dev[3]**2 + s_dev[4]**2 + s_dev[5]**2)))
    yld_corr_fortran = 300.0 + epsp_pl[0] * p.E
    assert math.isclose(vm, yld_corr_fortran, rel_tol=1e-4)


def test_shell_plane_stress_and_thinning():
    """Verify shell plane-stress enforcement, radial projection, and thickness thinning."""
    p = Law48Params(
        E=200000.0, nu=0.3, ca=1.0, sigy0=200.0,
        cb=300.0, cn=1.0001,
    )

    # 1. Pure elastic in-plane strain
    sig = np.zeros((1, 3))
    deps = np.array([[1e-4, 0.0, 0.0]])
    sig_el, epsp_el = shell_update_law48(p, sig.copy(), deps, epsp=0.0, dt=1e-4)
    assert epsp_el[0] == 0.0
    # sig_xx = A11 * deps_xx, sig_yy = A21 * deps_xx
    assert math.isclose(sig_el[0, 0], p.A11 * 1e-4, rel_tol=1e-6)
    assert math.isclose(sig_el[0, 1], p.A21 * 1e-4, rel_tol=1e-6)

    # 2. Plastic yield and thickness thinning
    deps_pl = np.array([[0.005, 0.0, 0.0]])
    extra = {"thk": np.array([1.5])}
    sig_pl, epsp_pl = shell_update_law48(p, sig.copy(), deps_pl, epsp=0.0, dt=1e-4, extra=extra)

    assert epsp_pl[0] > 0.0
    # von Mises in-plane stress matches yield stress
    # In OpenRadioss sigeps48c.F:279, for the first plastic step starting from epsp=0,
    # the initial tangent is H = E (pda = E1), giving svm = 560.622164.
    sxx, syy, sxy = sig_pl[0, 0], sig_pl[0, 1], sig_pl[0, 2]
    svm = math.sqrt(sxx**2 + syy**2 - sxx*syy + 3.0*sxy**2)
    assert math.isclose(svm, 560.622164, rel_tol=1e-4)

    # Thickness must thin under tensile strain
    assert extra["thk"][0] < 1.5


def test_kinematic_vs_isotropic_hardening():
    """Verify Bauschinger effect upon reversed loading with kinematic hardening."""
    # Pure isotropic (fisokin = 0) vs pure kinematic (fisokin = 1)
    p_iso = Law48Params(E=200000.0, nu=0.3, ca=1.0, sigy0=300.0, cb=20000.0, cn=1.0001, fisokin=0.0)
    p_kin = Law48Params(E=200000.0, nu=0.3, ca=1.0, sigy0=300.0, cb=20000.0, cn=1.0001, fisokin=1.0)

    # Step 1: Forward tension deps = [0.005, 0, 0, 0, 0, 0]
    deps_fwd = np.array([[0.005, 0.0, 0.0, 0.0, 0.0, 0.0]])
    extra_iso = {"sigb48": np.zeros((1, 6))}
    extra_kin = {"sigb48": np.zeros((1, 6))}

    sig_iso, ep_iso, _ = solid_update_law48(p_iso, np.zeros((1, 6)), deps_fwd, epsp=0.0, dt=1e-4, extra=extra_iso)
    sig_kin, ep_kin, _ = solid_update_law48(p_kin, np.zeros((1, 6)), deps_fwd, epsp=0.0, dt=1e-4, extra=extra_kin)

    # Backstress in kinematic model must have grown
    assert extra_kin["sigb48"][0, 0] > 0.0
    # Backstress in isotropic model remains zero
    assert np.allclose(extra_iso["sigb48"], 0.0)

    # Step 2: Strain reversal (compression) deps = [-0.004, 0, 0, 0, 0, 0]
    deps_rev = np.array([[-0.004, 0.0, 0.0, 0.0, 0.0, 0.0]])
    sig_iso_rev, ep_iso_rev, _ = solid_update_law48(p_iso, sig_iso.copy(), deps_rev, epsp=ep_iso, dt=1e-4, extra=extra_iso)
    sig_kin_rev, ep_kin_rev, _ = solid_update_law48(p_kin, sig_kin.copy(), deps_rev, epsp=ep_kin, dt=1e-4, extra=extra_kin)

    # In kinematic hardening, reverse yielding begins earlier (Bauschinger effect)
    # so more plastic strain accumulates in reverse flow compared to isotropic
    delta_ep_iso = ep_iso_rev[0] - ep_iso[0]
    delta_ep_kin = ep_kin_rev[0] - ep_kin[0]
    assert delta_ep_kin > delta_ep_iso


def test_strain_rate_filtering():
    """Verify strain rate filtering with cutoff frequency f_cut."""
    fcut = 100.0  # Hz
    dt = 0.001    # s
    alpha = min(1.0, 2.0 * math.pi * fcut * dt)

    p_filtered = Law48Params(
        E=200000.0, nu=0.3, sigy0=300.0,
        cc=50.0, eps0=1.0, fcut=fcut,
    )
    p_unfiltered = Law48Params(
        E=200000.0, nu=0.3, sigy0=300.0,
        cc=50.0, eps0=1.0, fcut=1e30,
    )

    deps = np.array([[0.005, 0.0, 0.0, 0.0, 0.0, 0.0]])
    extra_filt = {"epsd48": np.zeros(1)}

    # First cycle
    _, _, _ = solid_update_law48(p_filtered, np.zeros((1, 6)), deps, epsp=0.0, dt=dt, extra=extra_filt)
    # The stored filtered rate should be alpha * raw_rate
    raw_rate = (deps[0, 0] - deps[0, 0]/3.0) * math.sqrt(2.0/3.0 * 1.5) / dt  # approximate
    assert extra_filt["epsd48"][0] > 0.0

    # Unfiltered uses full instantaneous rate
    extra_unfilt = {}
    sig_unfilt, _, _ = solid_update_law48(p_unfiltered, np.zeros((1, 6)), deps, epsp=0.0, dt=dt, extra=extra_unfilt)
    sig_filt, _, _ = solid_update_law48(p_filtered, np.zeros((1, 6)), deps, epsp=0.0, dt=dt, extra={"epsd48": np.zeros(1)})

    # Unfiltered has higher effective rate in cycle 1, so higher yield stress
    assert sig_unfilt[0, 0] > sig_filt[0, 0]


def test_consistent_tangents():
    """Verify algorithmic consistent tangents for solids and shells."""
    p = Law48Params(E=200000.0, nu=0.3, ca=1.0, sigy0=300.0, cb=500.0, cn=1.0001)

    # 1. Solid tangent
    # Elastic state: epsp_incr = 0
    C_el = tangent_law48_solid(p)
    assert C_el.shape == (6, 6)
    # Symmetry check
    assert np.allclose(C_el, C_el.T)

    # Plastic state tangent
    sig_state = np.array([[350.0, -50.0, -50.0, 0.0, 0.0, 0.0]])
    epsp_incr = np.array([0.002])
    D_solid = tangent_law48_solid(p, sig=sig_state, epsp=np.array([0.01]), epsp_incr=epsp_incr)
    assert D_solid.shape == (1, 6, 6)
    # Tangent must be symmetric
    assert np.allclose(D_solid[0], D_solid[0].T)
    # Tangent stiffness must be lower than elastic stiffness in loading direction
    assert D_solid[0, 0, 0] < C_el[0, 0]

    # 2. Shell tangent
    C_sh = shell_membrane_tangent(p)
    assert C_sh.shape == (3, 3)
    assert np.allclose(C_sh, C_sh.T)

    sig_sh = np.array([[320.0, 0.0, 0.0]])
    D_shell = tangent_law48_shell(p, sig=sig_sh, epsp=np.array([0.01]), epsp_incr=epsp_incr)
    assert D_shell.shape == (1, 3, 3)
    assert D_shell[0, 0, 0] < C_sh[0, 0]


def test_builder_and_materials_dispatcher():
    """Verify build_law48 and pyradioss.materials dispatchers."""
    class MockRecord:
        id = 48
        density = 7.85e-9
        title = "Zhao Steel"
        params = {
            "MAT_E": 210000.0,
            "MAT_NU": 0.28,
            "MAT_SIGY": 350.0,
            "MAT_B": 450.0,
            "MAT_N": 0.5,
            "MAT_C": 30.0,
            "MAT_D": 10.0,
            "MAT_M": 0.8,
            "MAT_E1": 0.02,
            "MAT_K": 1.2,
            "MAT_E0": 1.0,
            "SCALE": 5000.0,
            "MAT_HARD": 0.2,
            "MAT_SIG": 800.0,
            "MAT_EPS": 0.35,
            "MAT_ETA1": 0.15,
            "MAT_ETA2": 0.25,
        }

    mat = build_law48(MockRecord())
    assert mat.law == 48
    assert mat.rho0 == 7.85e-9
    assert mat.params["sigy0"] == 350.0
    assert mat.params["fisokin"] == 0.2

    # Dispatchers
    deps_sol = np.zeros((1, 6))
    deps_sol[0, 0] = 1e-4
    sig_out, epsp_out, c_out = materials.solid_update(mat, np.zeros((1, 6)), deps_sol, epsp=np.zeros(1), dt=1e-5)
    assert sig_out[0, 0] > 0.0
    assert c_out is not None

    deps_sh = np.zeros((1, 3))
    deps_sh[0, 0] = 1e-4
    sig_sh, epsp_sh = materials.shell_update(mat, np.zeros((1, 3)), deps_sh, epsp=np.zeros(1), dt=1e-5)
    assert sig_sh[0, 0] > 0.0

    c_disp = materials.sound_speed(mat)
    assert c_disp > 0.0

    D_disp = materials.solid_tangent(mat, np.zeros((1, 6)), epsp=np.zeros(1))
    assert D_disp.shape == (1, 6, 6)

    D_sh_disp = materials.shell_layer_tangent(mat, np.zeros((1, 3)), epsp=np.zeros(1))
    assert D_sh_disp.shape == (1, 3, 3)
