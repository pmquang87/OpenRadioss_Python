import numpy as np
import pytest

from pyradioss.materials import law27_brittle
from pyradioss.model.entities import Material


def _make_law27(E=200000.0, nu=0.3, rho0=2.5e-9, eps_t1=0.01, eps_m1=0.03,
                dmax1=0.9, eps_f1=0.05, eps_t2=0.01, eps_m2=0.03,
                dmax2=0.9, eps_f2=0.05, **kwargs):
    params = {
        "E": E,
        "nu": nu,
        "eps_t1": eps_t1,
        "eps_m1": eps_m1,
        "dmax1": dmax1,
        "eps_f1": eps_f1,
        "eps_t2": eps_t2,
        "eps_m2": eps_m2,
        "dmax2": dmax2,
        "eps_f2": eps_f2,
    }
    params.update(kwargs)
    mat = Material(1, law=27, rho0=rho0, params=params)
    return mat


def _init_extra(n=1):
    return {
        "eps27": np.zeros((n, 3)),
        "crk27": np.zeros(n),
        "ang27": np.zeros(n),
        "dmg27": np.zeros((n, 2)),
        "layfail": np.ones(n),
    }


def test_law27_empty_arrays():
    mat = _make_law27()
    sig = np.empty((0, 3))
    deps = np.empty((0, 3))
    epsp = np.empty(0)
    extra = _init_extra(0)

    s_out, p_out = law27_brittle.shell_update(mat, sig, deps, epsp, 1e-4, extra)
    assert s_out.shape == (0, 3)
    assert p_out.shape == (0,)

    C = law27_brittle.consistent_shell_tangent(mat, extra)
    assert C.shape == (0, 3, 3)

    C_none = law27_brittle.consistent_shell_tangent(mat, None)
    assert C_none.shape == (0, 3, 3)


def test_law27_solid_update_raises_not_implemented():
    mat = _make_law27()
    with pytest.raises(NotImplementedError, match="shell elements only"):
        law27_brittle.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), None, 1e-4)


def test_law27_linear_elastic_uncracked_response():
    E = 210000.0
    nu = 0.3
    mat = _make_law27(E=E, nu=nu, eps_t1=0.01)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))
    # Strain well below eps_t1 = 0.01
    deps = np.array([[0.002, 0.001, 0.0005]])
    s, _ = law27_brittle.shell_update(mat, sig, deps, np.zeros(1), 1e-4, extra)

    cps = E / (1.0 - nu ** 2)
    G = E / (2.0 * (1.0 + nu))
    expected_sxx = cps * (0.002 + nu * 0.001)
    expected_syy = cps * (0.001 + nu * 0.002)
    expected_sxy = G * 0.0005

    assert s[0, 0] == pytest.approx(expected_sxx, rel=1e-10)
    assert s[0, 1] == pytest.approx(expected_syy, rel=1e-10)
    assert s[0, 2] == pytest.approx(expected_sxy, rel=1e-10)
    assert extra["crk27"][0] == 0.0
    assert np.allclose(extra["dmg27"][0], 0.0)


def test_law27_crack_initiation_and_frozen_angle():
    mat = _make_law27(eps_t1=0.01)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))

    # Step 1: cause cracking at angle theta = pi/6 (30 deg)
    theta_target = np.pi / 6.0
    exx = 0.015 + 0.01 * np.cos(2.0 * theta_target)
    eyy = 0.015 - 0.01 * np.cos(2.0 * theta_target)
    gxy = 0.02 * np.sin(2.0 * theta_target)
    deps1 = np.array([[exx, eyy, gxy]])

    law27_brittle.shell_update(mat, sig, deps1, np.zeros(1), 1e-4, extra)
    assert extra["crk27"][0] == 1.0
    assert extra["ang27"][0] == pytest.approx(theta_target, rel=1e-6)

    # Step 2: apply additional deformation in a different orientation
    # The crack frame must remain frozen!
    deps2 = np.array([[0.001, -0.002, 0.005]])
    law27_brittle.shell_update(mat, sig, deps2, np.zeros(1), 1e-4, extra)
    assert extra["ang27"][0] == pytest.approx(theta_target, rel=1e-6)


def test_law27_tensile_damage_evolution_curve():
    E = 200000.0
    nu = 0.25
    eps_t1 = 0.005
    eps_m1 = 0.025
    dmax1 = 0.8
    mat = _make_law27(E=E, nu=nu, eps_t1=eps_t1, eps_m1=eps_m1, dmax1=dmax1)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))

    # Uniaxial tension along x-axis to en1 = 0.015 (halfway between eps_t1 and eps_m1)
    en1 = 0.015
    deps = np.array([[en1, 0.0, 0.0]])
    s, _ = law27_brittle.shell_update(mat, sig, deps, np.zeros(1), 1e-4, extra)

    expected_d1 = dmax1 * (en1 - eps_t1) / (eps_m1 - eps_t1)  # 0.8 * 0.010 / 0.020 = 0.4
    assert extra["dmg27"][0, 0] == pytest.approx(expected_d1, rel=1e-10)

    cps = E / (1.0 - nu ** 2)
    expected_sxx = (1.0 - expected_d1) * cps * en1
    assert s[0, 0] == pytest.approx(expected_sxx, rel=1e-10)


def test_law27_damage_irreversibility_memory():
    mat = _make_law27(eps_t1=0.005, eps_m1=0.025, dmax1=0.8)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))

    # Step 1: advance to d1 = 0.4
    law27_brittle.shell_update(mat, sig, np.array([[0.015, 0.0, 0.0]]), np.zeros(1), 1e-4, extra)
    d1_peak = extra["dmg27"][0, 0]
    assert d1_peak == pytest.approx(0.4, rel=1e-10)

    # Step 2: unload to strain = 0.008 (below peak 0.015, but above eps_t1 0.005)
    law27_brittle.shell_update(mat, sig, np.array([[-0.007, 0.0, 0.0]]), np.zeros(1), 1e-4, extra)
    # Damage must not heal
    assert extra["dmg27"][0, 0] == pytest.approx(d1_peak, rel=1e-10)

    # Step 3: further unload to zero strain
    law27_brittle.shell_update(mat, sig, np.array([[-0.008, 0.0, 0.0]]), np.zeros(1), 1e-4, extra)
    assert extra["dmg27"][0, 0] == pytest.approx(d1_peak, rel=1e-10)
    assert np.allclose(sig[0], 0.0, atol=1e-12)


def test_law27_unilateral_crack_closure_stiffness_recovery():
    E = 200000.0
    nu = 0.3
    mat = _make_law27(E=E, nu=nu, eps_t1=0.005, eps_m1=0.025, dmax1=0.8)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))

    # Step 1: Open crack with tension
    law27_brittle.shell_update(mat, sig, np.array([[0.015, 0.0, 0.0]]), np.zeros(1), 1e-4, extra)
    assert extra["dmg27"][0, 0] > 0.0

    # Step 2: Apply compression across the crack (total strain exx = -0.005)
    deps_comp = np.array([[-0.020, 0.0, 0.0]])
    s, _ = law27_brittle.shell_update(mat, sig, deps_comp, np.zeros(1), 1e-4, extra)

    # In compression, the crack closes and transmits full undamaged elastic stiffness!
    cps = E / (1.0 - nu ** 2)
    expected_sxx = cps * (-0.005)
    expected_syy = cps * nu * (-0.005)
    assert s[0, 0] == pytest.approx(expected_sxx, rel=1e-10)
    assert s[0, 1] == pytest.approx(expected_syy, rel=1e-10)


def test_law27_biaxial_cracking_two_directions():
    mat = _make_law27(eps_t1=0.005, eps_m1=0.025, dmax1=0.8,
                      eps_t2=0.008, eps_m2=0.028, dmax2=0.7)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))

    # Biaxial tension with major strain along axis 1: exx = 0.018 > eyy = 0.015
    deps = np.array([[0.018, 0.015, 0.0]])
    law27_brittle.shell_update(mat, sig, deps, np.zeros(1), 1e-4, extra)

    assert extra["crk27"][0] == 1.0
    assert extra["ang27"][0] == pytest.approx(0.0, abs=1e-6)
    d1 = extra["dmg27"][0, 0]
    d2 = extra["dmg27"][0, 1]
    assert d1 > 0.0
    assert d2 > 0.0
    # Both crack directions damaged independently
    expected_d1 = 0.8 * (0.018 - 0.005) / (0.025 - 0.005)  # 0.52
    expected_d2 = 0.7 * (0.015 - 0.008) / (0.028 - 0.008)  # 0.245
    assert d1 == pytest.approx(expected_d1, rel=1e-10)
    assert d2 == pytest.approx(expected_d2, rel=1e-10)


def test_law27_shear_degradation_worst_crack():
    E = 200000.0
    nu = 0.25
    G = E / (2.0 * (1.0 + nu))
    mat = _make_law27(E=E, nu=nu, eps_t1=0.005, eps_m1=0.025, dmax1=0.8,
                      eps_t2=0.005, eps_m2=0.025, dmax2=0.5)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))

    # Step 1: Open crack along axis 1 (exx = 0.015 -> d1 = 0.4, d2 = 0)
    law27_brittle.shell_update(mat, sig, np.array([[0.015, 0.0, 0.0]]), np.zeros(1), 1e-4, extra)
    d1 = extra["dmg27"][0, 0]
    d2 = extra["dmg27"][0, 1]
    assert d1 == pytest.approx(0.4, rel=1e-10)
    assert d2 == 0.0

    # Step 2: Apply shear increment gxy = 0.004
    s, _ = law27_brittle.shell_update(mat, sig, np.array([[0.0, 0.0, 0.004]]), np.zeros(1), 1e-4, extra)
    # Shear stress is degraded by (1 - max(d1, d2)) * G * gxy
    expected_sxy = (1.0 - 0.4) * G * 0.004
    assert s[0, 2] == pytest.approx(expected_sxy, rel=1e-10)


def test_law27_layer_rupture_at_failure_strain():
    mat = _make_law27(eps_t1=0.005, eps_m1=0.020, eps_f1=0.040)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))

    # Step 1: Below failure strain (exx = 0.035 < 0.040)
    s, _ = law27_brittle.shell_update(mat, sig, np.array([[0.035, 0.0, 0.0]]), np.zeros(1), 1e-4, extra)
    assert extra["layfail"][0] == 1.0
    assert s[0, 0] > 0.0

    # Step 2: Exceed failure strain (total exx = 0.045 > 0.040)
    s, _ = law27_brittle.shell_update(mat, sig, np.array([[0.010, 0.0, 0.0]]), np.zeros(1), 1e-4, extra)
    assert extra["layfail"][0] == 0.0
    # Broken layer carries identically zero stress
    assert np.allclose(s[0], 0.0, atol=1e-12)


def test_law27_cracking_under_arbitrary_shear_orientation():
    mat = _make_law27(eps_t1=0.01)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))

    # Pure engineering shear gxy = 0.04
    # Major principal strain e1 = 0.5 * gxy = 0.02 > eps_t1 (0.01)
    # Principal direction is exactly 45 deg (pi/4)
    deps = np.array([[0.0, 0.0, 0.04]])
    law27_brittle.shell_update(mat, sig, deps, np.zeros(1), 1e-4, extra)

    assert extra["crk27"][0] == 1.0
    assert extra["ang27"][0] == pytest.approx(np.pi / 4.0, rel=1e-6)


def test_law27_plasticity_yields_before_cracking():
    # Large eps_t1 (0.1) so it does not crack
    # Plasticity yield stress A = 200 MPa, B = 300 MPa
    mat = _make_law27(E=210000.0, nu=0.3, eps_t1=0.1, A=200.0, B=300.0, n=0.5, sig_max=1000.0)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))
    epsp = np.zeros(1)

    # Elastic prediction for exx = 0.001: 210000 / 0.91 * 0.001 = 230.77 MPa > 200 MPa
    deps = np.array([[0.001, 0.0, 0.0]])
    s, p = law27_brittle.shell_update(mat, sig, deps, epsp, 1e-5, extra)

    assert extra["crk27"][0] == 0.0
    assert p[0] > 0.0
    vm = np.sqrt(s[0, 0]**2 - s[0, 0]*s[0, 1] + s[0, 1]**2 + 3.0*s[0, 2]**2)
    expected_sy = 200.0 + 300.0 * (p[0] ** 0.5)
    assert vm == pytest.approx(expected_sy, rel=5e-3)


def test_law27_plasticity_coupled_with_damage():
    # Modest eps_t1 (0.001) and plasticity A = 250 MPa, B = 300 MPa
    mat = _make_law27(E=200000.0, nu=0.3, eps_t1=0.001, eps_m1=0.005, dmax1=0.5,
                      A=250.0, B=300.0, n=0.5, sig_max=1000.0)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))
    epsp = np.zeros(1)

    # exx = 0.0015 -> cracks, d1 > 0, and damaged trial stress enters plasticity
    deps = np.array([[0.0015, 0.0, 0.0]])
    s, p = law27_brittle.shell_update(mat, sig, deps, epsp, 1e-5, extra)

    assert extra["crk27"][0] == 1.0
    assert extra["dmg27"][0, 0] > 0.0
    assert p[0] > 0.0
    vm = np.sqrt(s[0, 0]**2 - s[0, 0]*s[0, 1] + s[0, 1]**2 + 3.0*s[0, 2]**2)
    expected_sy = 250.0 + 300.0 * (p[0] ** 0.5)
    assert vm == pytest.approx(expected_sy, rel=0.025)


def test_law27_zero_negative_dt_guard():
    mat = _make_law27(eps_t1=0.1, A=150.0, B=200.0, n=0.5, sig_max=1000.0)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))

    # Call with dt = 0.0
    s1, p1 = law27_brittle.shell_update(mat, sig.copy(), np.array([[0.001, 0.0, 0.0]]), np.zeros(1), 0.0, extra)
    assert not np.isnan(s1).any()
    assert not np.isnan(p1).any()

    # Call with dt < 0.0
    extra2 = _init_extra(1)
    s2, p2 = law27_brittle.shell_update(mat, sig.copy(), np.array([[0.001, 0.0, 0.0]]), np.zeros(1), -1e-4, extra2)
    assert not np.isnan(s2).any()
    assert not np.isnan(p2).any()


def test_law27_consistent_tangent_uncracked_branch():
    E = 210000.0
    nu = 0.3
    mat = _make_law27(E=E, nu=nu, eps_t1=0.01)
    extra = _init_extra(1)

    # Uncracked (crk27 = 0)
    C = law27_brittle.consistent_shell_tangent(mat, extra)
    cps = E / (1.0 - nu ** 2)
    G = E / (2.0 * (1.0 + nu))
    expected_C = np.array([[cps, cps * nu, 0.0],
                           [cps * nu, cps, 0.0],
                           [0.0, 0.0, G]])
    assert np.allclose(C[0], expected_C, atol=1e-10)


def test_law27_consistent_tangent_frozen_damage_secant():
    E = 200000.0
    nu = 0.3
    mat = _make_law27(E=E, nu=nu, eps_t1=0.005, eps_m1=0.025, dmax1=0.8)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))

    # Step 1: Advance to d1 = 0.4
    law27_brittle.shell_update(mat, sig, np.array([[0.015, 0.0, 0.0]]), np.zeros(1), 1e-4, extra)

    # Step 2: Unload inside the damage surface (strain = 0.010, driving value = 0.2 < stored d1 = 0.4)
    law27_brittle.shell_update(mat, sig, np.array([[-0.005, 0.0, 0.0]]), np.zeros(1), 1e-4, extra)

    # Tangent must be the secant stiffness (1 - d1) on row 0
    C = law27_brittle.consistent_shell_tangent(mat, extra)
    cps = E / (1.0 - nu ** 2)
    G = E / (2.0 * (1.0 + nu))
    # In crack frame (aligned with x since ang = 0)
    expected_C11 = (1.0 - 0.4) * cps
    expected_C12 = (1.0 - 0.4) * cps * nu
    expected_C21 = cps * nu  # direction 2 is undamaged
    expected_C22 = cps
    expected_C33 = (1.0 - 0.4) * G

    assert C[0, 0, 0] == pytest.approx(expected_C11, rel=1e-10)
    assert C[0, 0, 1] == pytest.approx(expected_C12, rel=1e-10)
    assert C[0, 1, 0] == pytest.approx(expected_C21, rel=1e-10)
    assert C[0, 1, 1] == pytest.approx(expected_C22, rel=1e-10)
    assert C[0, 2, 2] == pytest.approx(expected_C33, rel=1e-10)


def test_law27_consistent_tangent_growing_damage_softening():
    E = 200000.0
    nu = 0.2
    eps_t1 = 0.005
    eps_m1 = 0.025
    dmax1 = 0.8
    mat = _make_law27(E=E, nu=nu, eps_t1=eps_t1, eps_m1=eps_m1, dmax1=dmax1)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))

    # At en1 = 0.012, damage is growing and tangent is non-zero
    en1 = 0.012
    deps = np.array([[en1, 0.0, 0.0]])
    law27_brittle.shell_update(mat, sig, deps, np.zeros(1), 1e-4, extra)

    C = law27_brittle.consistent_shell_tangent(mat, extra)
    cps = E / (1.0 - nu ** 2)
    d1 = 0.8 * (en1 - eps_t1) / (eps_m1 - eps_t1)  # 0.28
    dd1 = dmax1 / (eps_m1 - eps_t1)                 # 0.8 / 0.020 = 40.0
    s1el = cps * en1

    expected_C11 = (1.0 - d1) * cps - dd1 * s1el  # (1 - 0.28 - 0.48) * cps = 0.24 * cps
    assert C[0, 0, 0] == pytest.approx(expected_C11, rel=1e-10)

    # Numerical directional derivative check
    h = 1e-7
    # Finite difference in direction 1
    ex_plus = {
        "eps27": np.array([[0.0, 0.0, 0.0]]),
        "crk27": np.array([0.0]),
        "ang27": np.array([0.0]),
        "dmg27": np.array([[0.0, 0.0]]),
        "layfail": np.array([1.0]),
    }
    s_plus, _ = law27_brittle.shell_update(mat, np.zeros((1, 3)), np.array([[en1 + h, 0.0, 0.0]]), np.zeros(1), 1e-4, ex_plus)

    ex_minus = {
        "eps27": np.array([[0.0, 0.0, 0.0]]),
        "crk27": np.array([0.0]),
        "ang27": np.array([0.0]),
        "dmg27": np.array([[0.0, 0.0]]),
        "layfail": np.array([1.0]),
    }
    s_minus, _ = law27_brittle.shell_update(mat, np.zeros((1, 3)), np.array([[en1 - h, 0.0, 0.0]]), np.zeros(1), 1e-4, ex_minus)

    fd_ds11 = (s_plus[0, 0] - s_minus[0, 0]) / (2.0 * h)
    assert fd_ds11 == pytest.approx(C[0, 0, 0], rel=1e-5)


def test_law27_consistent_tangent_closed_crack_recovery():
    E = 200000.0
    nu = 0.3
    mat = _make_law27(E=E, nu=nu, eps_t1=0.005, eps_m1=0.025, dmax1=0.8)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))

    # Step 1: open crack
    law27_brittle.shell_update(mat, sig, np.array([[0.015, 0.0, 0.0]]), np.zeros(1), 1e-4, extra)
    # Step 2: compress
    law27_brittle.shell_update(mat, sig, np.array([[-0.020, 0.0, 0.0]]), np.zeros(1), 1e-4, extra)

    # Tangent must recover full elastic normal rows
    C = law27_brittle.consistent_shell_tangent(mat, extra)
    cps = E / (1.0 - nu ** 2)
    assert C[0, 0, 0] == pytest.approx(cps, rel=1e-10)
    assert C[0, 0, 1] == pytest.approx(cps * nu, rel=1e-10)


def test_law27_consistent_tangent_broken_layer_zero():
    mat = _make_law27(eps_t1=0.005, eps_m1=0.020, eps_f1=0.040)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))

    # Break layer
    law27_brittle.shell_update(mat, sig, np.array([[0.050, 0.0, 0.0]]), np.zeros(1), 1e-4, extra)
    assert extra["layfail"][0] == 0.0

    C = law27_brittle.consistent_shell_tangent(mat, extra)
    assert np.allclose(C[0], 0.0, atol=1e-12)


def test_law27_multi_element_batch_vectorization():
    mat = _make_law27(E=200000.0, nu=0.3, eps_t1=0.005, eps_m1=0.025, dmax1=0.8,
                      eps_f1=0.05)
    N = 5
    extra_batch = _init_extra(N)

    # 5 different strains:
    # 0: uncracked
    # 1: growing damage
    # 2: open crack with shear
    # 3: closed crack (compression)
    # 4: broken layer
    deps_batch = np.array([
        [0.002, 0.001, 0.0005],
        [0.015, 0.0, 0.0],
        [0.015, 0.0, 0.004],
        [-0.005, 0.0, 0.0],
        [0.060, 0.0, 0.0],
    ])

    sig_batch = np.zeros((N, 3))
    s_bat, _ = law27_brittle.shell_update(mat, sig_batch, deps_batch, np.zeros(N), 1e-4, extra_batch)
    C_bat = law27_brittle.consistent_shell_tangent(mat, extra_batch)

    # Evaluate element by element
    for i in range(N):
        extra_ind = _init_extra(1)
        sig_ind = np.zeros((1, 3))
        s_ind, _ = law27_brittle.shell_update(mat, sig_ind, deps_batch[i:i+1], np.zeros(1), 1e-4, extra_ind)
        C_ind = law27_brittle.consistent_shell_tangent(mat, extra_ind)

        assert np.allclose(s_bat[i], s_ind[0], atol=1e-12)
        assert np.allclose(C_bat[i], C_ind[0], atol=1e-12)
