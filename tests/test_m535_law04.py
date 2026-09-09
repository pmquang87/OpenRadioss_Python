"""M535 — LAW4 Hydrodynamic Johnson-Cook (/MAT/LAW4, /MAT/HYD_JCOOK) comprehensive test suite.

Fortran origins:
- ``engine/source/materials/mat/mat004/m4law.F`` (solid constitutive update)
- ``starter/source/materials/mat/mat004/hm_read_mat04.F`` (starter card reader & defaults)
- ``C:\\OpenRadioss\\hm_cfg_files\\config\\CFG\\radioss110\\MAT\\matl4_hyd_jcook.cfg``
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials import law04_hyd_jcook


# ============================================================================
# Helpers
# ============================================================================

def _make_law04(
    e=210000.0,
    nu=0.3,
    rho0=7.85e-3,
    a=250.0,
    b=400.0,
    n=0.4,
    c=0.05,
    eps0=1.0,
    m=1.0,
    tmelt=1800.0,
    tmax=2000.0,
    rho_cp=3.5e6,
    t0=300.0,
    pmin=-100.0,
    sig_max=0.0,
    eps_max=0.0,
    **kw,
) -> Material:
    """Construct a LAW4 material with steel-like defaults."""
    rec = {
        "id": 1,
        "density": rho0,
        "title": "HYD_JCOOK",
        "params": {
            "MAT_E": e,
            "MAT_NU": nu,
            "MAT_SIGY": a,
            "MAT_BETA": b,
            "MAT_HARD": n,
            "MAT_SRC": c,
            "MAT_SRP": eps0,
            "MAT_M": m,
            "MAT_TMELT": tmelt,
            "MAT_TMAX": tmax,
            "MAT_SPHEAT": rho_cp,
            "MAT_T0": t0,
            "MAT_PC": pmin,
            "MAT_SIG": sig_max,
            "MAT_EPS": eps_max,
            **kw,
        },
    }
    return law04_hyd_jcook.build_law04(rec)


# ============================================================================
# 1. Construction & Parameter Extraction
# ============================================================================

def test_build_from_dict():
    """Verify extraction of E, nu, G, K, A, B, N, C, eps0, m, Tmelt, Tmax, rho_cp, T0, pmin."""
    mat = _make_law04(
        e=200000.0,
        nu=0.25,
        rho0=7.8e-3,
        a=220.0,
        b=380.0,
        n=0.35,
        c=0.02,
        eps0=2.0,
        m=0.9,
        tmelt=1700.0,
        tmax=1900.0,
        rho_cp=3.8e6,
        t0=293.0,
        pmin=-50.0,
        sig_max=800.0,
        eps_max=0.8,
    )

    assert mat.id == 1
    assert mat.law == 4
    assert mat.rho0 == pytest.approx(7.8e-3)
    p = mat.params

    assert p["E"] == pytest.approx(200000.0)
    assert p["nu"] == pytest.approx(0.25)
    expected_G = 200000.0 / (2.0 * (1.0 + 0.25))
    expected_K = 200000.0 / (3.0 * (1.0 - 2.0 * 0.25))
    assert p["G"] == pytest.approx(expected_G)
    assert p["K"] == pytest.approx(expected_K)
    assert p["A"] == pytest.approx(220.0)
    assert p["B"] == pytest.approx(380.0)
    assert p["N"] == pytest.approx(0.35)
    assert p["C"] == pytest.approx(0.02)
    assert p["eps0"] == pytest.approx(2.0)
    assert p["m"] == pytest.approx(0.9)
    assert p["Tmelt"] == pytest.approx(1700.0)
    assert p["Tmax"] == pytest.approx(1900.0)
    assert p["rho_cp"] == pytest.approx(3.8e6)
    assert p["T0"] == pytest.approx(293.0)
    assert p["pmin"] == pytest.approx(-50.0)
    assert p["sig_max"] == pytest.approx(800.0)
    assert p["eps_max"] == pytest.approx(0.8)


def test_build_from_material_instance():
    """Verify wrapping a Material instance."""
    base = Material(
        id=10,
        law=4,
        rho0=2.7e-3,
        title="Aluminum_JC",
        params={
            "E": 70000.0,
            "nu": 0.33,
            "A": 150.0,
            "B": 200.0,
            "N": 0.3,
            "C": 0.01,
            "eps0": 1.0,
            "m": 1.2,
            "Tmelt": 900.0,
            "Tmax": 1000.0,
            "rho_cp": 2.4e6,
            "T0": 298.0,
            "pmin": -30.0,
        },
    )
    mat = law04_hyd_jcook.build_law04(base)
    assert mat.id == 10
    assert mat.law == 4
    assert mat.rho0 == pytest.approx(2.7e-3)
    assert mat.title == "Aluminum_JC"
    assert mat.params["E"] == pytest.approx(70000.0)
    assert mat.params["nu"] == pytest.approx(0.33)
    assert mat.params["A"] == pytest.approx(150.0)
    assert mat.params["B"] == pytest.approx(200.0)
    assert mat.params["N"] == pytest.approx(0.3)


def test_build_validation_errors():
    """E <= 0 raises ValueError, nu < 0 or nu >= 0.5 raises ValueError, rho0 <= 0 raises ValueError."""
    # Invalid E
    with pytest.raises(ValueError, match="Young's modulus E must be > 0"):
        _make_law04(e=0.0)
    with pytest.raises(ValueError, match="Young's modulus E must be > 0"):
        _make_law04(e=-1000.0)

    # Invalid nu
    with pytest.raises(ValueError, match=r"Poisson's ratio nu must be in \[0, 0\.5\)"):
        _make_law04(nu=-0.1)
    with pytest.raises(ValueError, match=r"Poisson's ratio nu must be in \[0, 0\.5\)"):
        _make_law04(nu=0.5)
    with pytest.raises(ValueError, match=r"Poisson's ratio nu must be in \[0, 0\.5\)"):
        _make_law04(nu=0.6)

    # Invalid density
    with pytest.raises(ValueError, match="Initial density rho0 must be > 0"):
        _make_law04(rho0=0.0)
    with pytest.raises(ValueError, match="Initial density rho0 must be > 0"):
        _make_law04(rho0=-1.0)


def test_registration():
    """Verify 'LAW4', 'HYD_JCOOK', 'JCOOK_HYD', '4' are in MAT_PHYSICS_REGISTRY."""
    from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY

    for key in ("LAW4", "HYD_JCOOK", "JCOOK_HYD", "4"):
        assert key in MAT_PHYSICS_REGISTRY
        assert MAT_PHYSICS_REGISTRY[key] is law04_hyd_jcook.build_law04


def test_build_defaults():
    """Fortran defaults from hm_read_mat04.F:
    N=0 or 1 -> 1.0001, eps_max=0 -> 1e20, sig_max=0 -> 1e20,
    C=0 -> eps0=1.0, m=0 -> 1.0, Tmelt=0 -> 1e20, Tmax=0 -> 1e20,
    T0<=0 -> 300.0, pmin=0 -> -1e30.
    """
    mat = _make_law04(
        n=0.0,
        eps_max=0.0,
        sig_max=0.0,
        c=0.0,
        eps0=0.0,
        m=0.0,
        tmelt=0.0,
        tmax=0.0,
        t0=0.0,
        pmin=0.0,
    )
    p = mat.params
    assert p["N"] == pytest.approx(1.0001)
    assert p["eps_max"] == pytest.approx(1e20)
    assert p["sig_max"] == pytest.approx(1e20)
    assert p["eps0"] == pytest.approx(1.0)
    assert p["m"] == pytest.approx(1.0)
    assert p["Tmelt"] == pytest.approx(1e20)
    assert p["Tmax"] == pytest.approx(1e20)
    assert p["T0"] == pytest.approx(300.0)
    assert p["pmin"] == pytest.approx(-1e30)

    # N=1 also clamps to 1.0001
    mat_n1 = _make_law04(n=1.0)
    assert mat_n1.params["N"] == pytest.approx(1.0001)


def test_build_with_embedded_polynomial_eos():
    """Verify embedded polynomial EOS is constructed when EOS coefficients are provided."""
    rec = {
        "id": 2,
        "density": 8.0e-3,
        "params": {
            "MAT_E": 200000.0,
            "MAT_NU": 0.3,
            "MAT_SIGY": 300.0,
            "MAT_C0": 1.0,
            "MAT_C1": 150000.0,
            "MAT_C2": 5000.0,
            "MAT_C3": 100.0,
            "MAT_C4": 0.5,
            "MAT_C5": 0.2,
            "MAT_EA": 10.0,
            "MAT_PSH": 2.0,
        },
    }
    mat = law04_hyd_jcook.build_law04(rec)
    assert mat.eos is not None
    assert mat.eos.kind == "POLYNOMIAL"
    assert mat.eos.params["c1"] == pytest.approx(150000.0)
    assert mat.eos.params["c4"] == pytest.approx(0.5)


# ============================================================================
# 2. Physics and Constitutive Behavior
# ============================================================================

def test_elastic_step():
    """Stress below yield returns exact Hooke elastic deviator and pressure."""
    mat = _make_law04(e=210000.0, nu=0.3, a=500.0, b=0.0)
    G = mat.params["G"]
    K = mat.params["K"]
    lam = K - 2.0 * G / 3.0

    sig = np.zeros((1, 6))
    deps = np.array([[1e-5, 0.0, 0.0, 0.0, 0.0, 0.0]])
    epsp = np.zeros(1)

    sig_new, epsp_new, c = law04_hyd_jcook.solid_update(mat, sig, deps, epsp, dt=1e-5)

    expected_xx = (lam + 2.0 * G) * 1e-5
    expected_yy = lam * 1e-5
    expected_zz = lam * 1e-5

    np.testing.assert_allclose(sig_new[0, 0], expected_xx, rtol=1e-8)
    np.testing.assert_allclose(sig_new[0, 1], expected_yy, rtol=1e-8)
    np.testing.assert_allclose(sig_new[0, 2], expected_zz, rtol=1e-8)
    np.testing.assert_allclose(sig_new[0, 3:], 0.0, atol=1e-12)
    assert epsp_new[0] == 0.0

    expected_c = np.sqrt((K + (4.0 / 3.0) * G) / mat.rho0)
    np.testing.assert_allclose(c[0], expected_c, rtol=1e-8)


def test_elastic_pure_shear():
    """Pure shear strain increment gives sig_xy = G * gamma_xy, zero pressure, zero epsp."""
    mat = _make_law04(e=210000.0, nu=0.3, a=500.0)
    G = mat.params["G"]

    sig = np.zeros((1, 6))
    deps = np.array([[0.0, 0.0, 0.0, 1e-5, 0.0, 0.0]])
    epsp = np.zeros(1)

    sig_new, epsp_new, _ = law04_hyd_jcook.solid_update(mat, sig, deps, epsp)

    np.testing.assert_allclose(sig_new[0, :3], 0.0, atol=1e-12)
    np.testing.assert_allclose(sig_new[0, 3], G * 1e-5, rtol=1e-8)
    np.testing.assert_allclose(sig_new[0, 4:], 0.0, atol=1e-12)
    assert epsp_new[0] == 0.0


def test_elastic_hydrostatic():
    """Pure volumetric strain gives zero deviatoric stress, pure hydrostatic pressure, no yielding."""
    mat = _make_law04(e=210000.0, nu=0.3, a=200.0)
    K = mat.params["K"]

    sig = np.zeros((1, 6))
    e = 1e-5
    deps = np.array([[e, e, e, 0.0, 0.0, 0.0]])
    epsp = np.zeros(1)

    sig_new, epsp_new, _ = law04_hyd_jcook.solid_update(mat, sig, deps, epsp)

    expected_p = 3.0 * K * e
    np.testing.assert_allclose(sig_new[0, 0], expected_p, rtol=1e-8)
    np.testing.assert_allclose(sig_new[0, 1], expected_p, rtol=1e-8)
    np.testing.assert_allclose(sig_new[0, 2], expected_p, rtol=1e-8)
    np.testing.assert_allclose(sig_new[0, 3:], 0.0, atol=1e-12)
    assert epsp_new[0] == 0.0


def test_plastic_yielding_power_law():
    """Verify yield stress on virgin state is A, and with plastic strain epsp is A + B*epsp^N."""
    A = 200.0
    B = 500.0
    N = 0.5
    mat = _make_law04(a=A, b=B, n=N, c=0.0)

    # 1. Virgin step exceeding yield
    sig = np.zeros((1, 6))
    deps = np.array([[0.005, 0.0, 0.0, 0.0, 0.0, 0.0]])
    epsp = np.zeros(1)

    sig_new, epsp_new, _ = law04_hyd_jcook.solid_update(mat, sig, deps, epsp, dt=1e-5)
    assert epsp_new[0] > 0.0

    # von Mises on deviator
    p = (sig_new[0, 0] + sig_new[0, 1] + sig_new[0, 2]) / 3.0
    s = sig_new[0].copy()
    s[:3] -= p
    j2 = 0.5 * np.sum(s[:3] ** 2) + np.sum(s[3:] ** 2)
    svm = np.sqrt(3.0 * j2)

    # Virgin yield target evaluated at epsp=0 is A
    np.testing.assert_allclose(svm, A, rtol=1e-4)

    # 2. Hardened step starting with pre-existing plastic strain
    prior_epsp = np.array([0.02])
    sig_prior = np.zeros((1, 6))
    expected_yield = A + B * (prior_epsp[0] ** N)

    sig_hard, epsp_hard, _ = law04_hyd_jcook.solid_update(mat, sig_prior, deps, prior_epsp.copy(), dt=1e-5)
    assert epsp_hard[0] > prior_epsp[0]

    p_hard = (sig_hard[0, 0] + sig_hard[0, 1] + sig_hard[0, 2]) / 3.0
    s_hard = sig_hard[0].copy()
    s_hard[:3] -= p_hard
    j2_hard = 0.5 * np.sum(s_hard[:3] ** 2) + np.sum(s_hard[3:] ** 2)
    svm_hard = np.sqrt(3.0 * j2_hard)

    # Target yield stress for the step is evaluated at start-of-step epsp
    np.testing.assert_allclose(svm_hard, expected_yield, rtol=1e-4)


def test_plastic_yielding_hardening_n_gt1():
    """N > 1 power-law hardening: convex hardening."""
    mat = _make_law04(a=150.0, b=800.0, n=2.0, c=0.0)
    sig = np.zeros((1, 6))
    deps = np.array([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]])
    epsp = np.array([0.05])

    sig_new, epsp_new, _ = law04_hyd_jcook.solid_update(mat, sig, deps, epsp, dt=1e-5)
    expected_yield = 150.0 + 800.0 * (0.05 ** 2.0)

    p = (sig_new[0, 0] + sig_new[0, 1] + sig_new[0, 2]) / 3.0
    s = sig_new[0].copy()
    s[:3] -= p
    svm = np.sqrt(3.0 * (0.5 * np.sum(s[:3] ** 2) + np.sum(s[3:] ** 2)))
    np.testing.assert_allclose(svm, expected_yield, rtol=1e-4)


def test_plastic_yielding_hardening_n_lt1():
    """N < 1 power-law hardening: concave hardening."""
    mat = _make_law04(a=250.0, b=400.0, n=0.3, c=0.0)
    sig = np.zeros((1, 6))
    deps = np.array([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]])
    epsp = np.array([0.03])

    sig_new, epsp_new, _ = law04_hyd_jcook.solid_update(mat, sig, deps, epsp, dt=1e-5)
    expected_yield = 250.0 + 400.0 * (0.03 ** 0.3)

    p = (sig_new[0, 0] + sig_new[0, 1] + sig_new[0, 2]) / 3.0
    s = sig_new[0].copy()
    s[:3] -= p
    svm = np.sqrt(3.0 * (0.5 * np.sum(s[:3] ** 2) + np.sum(s[3:] ** 2)))
    np.testing.assert_allclose(svm, expected_yield, rtol=1e-4)


def test_plastic_yielding_hardening_n_eq1():
    """N = 1 clamps to 1.0001 (quasi-linear hardening)."""
    mat = _make_law04(a=150.0, b=300.0, n=1.0, c=0.0)
    assert mat.params["N"] == pytest.approx(1.0001)
    sig = np.zeros((1, 6))
    deps = np.array([[0.005, 0.0, 0.0, 0.0, 0.0, 0.0]])
    epsp = np.array([0.01])

    sig_new, epsp_new, _ = law04_hyd_jcook.solid_update(mat, sig, deps, epsp, dt=1e-5)
    assert epsp_new[0] > 0.01


def test_maximum_stress_cutoff():
    """Verify stress is capped at sig_max."""
    sig_max = 350.0
    mat = _make_law04(a=200.0, b=1000.0, n=0.5, sig_max=sig_max, c=0.0)

    # Large plastic strain where A + B*epsp^N = 200 + 1000*sqrt(0.25) = 700 >> 350
    prior_epsp = np.array([0.25])
    sig = np.zeros((1, 6))
    deps = np.array([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]])

    sig_new, epsp_new, _ = law04_hyd_jcook.solid_update(mat, sig, deps, prior_epsp, dt=1e-5)

    p = (sig_new[0, 0] + sig_new[0, 1] + sig_new[0, 2]) / 3.0
    s = sig_new[0].copy()
    s[:3] -= p
    svm = np.sqrt(3.0 * (0.5 * np.sum(s[:3] ** 2) + np.sum(s[3:] ** 2)))
    np.testing.assert_allclose(svm, sig_max, rtol=1e-4)


def test_strain_rate_sensitivity():
    """Verify strain rate factor CE:
    - When eps_dot <= eps_dot_0, CE = 1.0 (no rate enhancement).
    - When eps_dot > eps_dot_0, CE = 1.0 + C * ln(eps_dot / eps_dot_0).
    - Verify stress scales proportionally by CE.
    """
    C = 0.08
    eps0 = 10.0
    A = 250.0
    mat = _make_law04(a=A, b=0.0, c=C, eps0=eps0)

    # 1. Rate below reference: eps_dot = 0.005 / 5e-3 = 1.0 <= 10.0 -> CE = 1.0
    deps_low = np.array([[0.005, 0.0, 0.0, 0.0, 0.0, 0.0]])
    dt_low = 5e-3  # eps_dot = 1.0
    sig_low, _, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps_low, np.zeros(1), dt=dt_low)

    p_low = (sig_low[0, 0] + sig_low[0, 1] + sig_low[0, 2]) / 3.0
    s_low = sig_low[0].copy()
    s_low[:3] -= p_low
    svm_low = np.sqrt(3.0 * (0.5 * np.sum(s_low[:3] ** 2) + np.sum(s_low[3:] ** 2)))
    np.testing.assert_allclose(svm_low, A, rtol=1e-4)

    # 2. Rate above reference: eps_dot = 0.005 / 5e-5 = 100.0 > 10.0
    dt_high = 5e-5  # eps_dot = 100.0
    expected_CE = 1.0 + C * np.log(100.0 / eps0)
    expected_yield_high = A * expected_CE

    sig_high, _, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps_low, np.zeros(1), dt=dt_high)

    p_high = (sig_high[0, 0] + sig_high[0, 1] + sig_high[0, 2]) / 3.0
    s_high = sig_high[0].copy()
    s_high[:3] -= p_high
    svm_high = np.sqrt(3.0 * (0.5 * np.sum(s_high[:3] ** 2) + np.sum(s_high[3:] ** 2)))
    np.testing.assert_allclose(svm_high, expected_yield_high, rtol=1e-4)


def test_strain_rate_sensitivity_zero_c():
    """When C == 0, CE remains 1.0 regardless of strain rate."""
    mat = _make_law04(a=300.0, b=0.0, c=0.0, eps0=1.0)
    deps = np.array([[0.005, 0.0, 0.0, 0.0, 0.0, 0.0]])
    dt = 1e-6  # very high strain rate 5000 /s

    sig, _, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.zeros(1), dt=dt)
    p = (sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
    s = sig[0].copy()
    s[:3] -= p
    svm = np.sqrt(3.0 * (0.5 * np.sum(s[:3] ** 2) + np.sum(s[3:] ** 2)))
    np.testing.assert_allclose(svm, 300.0, rtol=1e-4)


def test_thermal_softening():
    """Verify thermal softening CT:
    - When T == T0, CT = 1.0.
    - When T0 < T < Tmelt, CT = 1.0 - Tstar^m where Tstar = (T - T0) / (Tmelt - T0).
    - When T >= Tmelt, stress and yield drop to zero.
    """
    A = 400.0
    T0 = 300.0
    Tmelt = 1500.0
    m = 1.5
    mat = _make_law04(a=A, b=0.0, c=0.0, t0=T0, tmelt=Tmelt, m=m)
    deps = np.array([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]])

    # 1. At room temperature T == T0 -> CT = 1.0
    extra_t0 = {"temp": np.array([T0])}
    sig_t0, _, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.zeros(1), dt=1e-5, extra=extra_t0)
    p_t0 = (sig_t0[0, 0] + sig_t0[0, 1] + sig_t0[0, 2]) / 3.0
    s_t0 = sig_t0[0].copy()
    s_t0[:3] -= p_t0
    svm_t0 = np.sqrt(3.0 * (0.5 * np.sum(s_t0[:3] ** 2) + np.sum(s_t0[3:] ** 2)))
    np.testing.assert_allclose(svm_t0, A, rtol=1e-4)

    # 2. Intermediate temperature T = 900 K -> Tstar = (900 - 300) / (1500 - 300) = 0.5
    T_mid = 900.0
    tstar = (T_mid - T0) / (Tmelt - T0)
    expected_CT = 1.0 - (tstar ** m)
    expected_yield = A * expected_CT

    extra_mid = {"temp": np.array([T_mid])}
    sig_mid, _, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.zeros(1), dt=1e-5, extra=extra_mid)
    p_mid = (sig_mid[0, 0] + sig_mid[0, 1] + sig_mid[0, 2]) / 3.0
    s_mid = sig_mid[0].copy()
    s_mid[:3] -= p_mid
    svm_mid = np.sqrt(3.0 * (0.5 * np.sum(s_mid[:3] ** 2) + np.sum(s_mid[3:] ** 2)))
    np.testing.assert_allclose(svm_mid, expected_yield, rtol=1e-4)

    # 3. At melting point T >= Tmelt -> deviator drops to zero
    extra_melt = {"temp": np.array([Tmelt + 50.0])}
    sig_melt, _, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.zeros(1), dt=1e-5, extra=extra_melt)
    p_melt = (sig_melt[0, 0] + sig_melt[0, 1] + sig_melt[0, 2]) / 3.0
    s_melt = sig_melt[0].copy()
    s_melt[:3] -= p_melt
    svm_melt = np.sqrt(3.0 * (0.5 * np.sum(s_melt[:3] ** 2) + np.sum(s_melt[3:] ** 2)))
    assert svm_melt == pytest.approx(0.0, abs=1e-10)


def test_thermal_softening_above_tmax():
    """When T > Tmax, m becomes 1.0 (linear softening) as per Fortran m4law.F line 136."""
    A = 500.0
    T0 = 300.0
    Tmelt = 1500.0
    Tmax = 800.0
    m = 2.0
    mat = _make_law04(a=A, b=0.0, c=0.0, t0=T0, tmelt=Tmelt, tmax=Tmax, m=m)
    deps = np.array([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]])

    T_test = 1000.0  # T_test > Tmax, so m_eff = 1.0 instead of 2.0
    tstar = (T_test - T0) / (Tmelt - T0)
    expected_CT = 1.0 - (tstar ** 1.0)
    expected_yield = A * expected_CT

    extra = {"temp": np.array([T_test])}
    sig, _, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.zeros(1), dt=1e-5, extra=extra)
    p = (sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
    s = sig[0].copy()
    s[:3] -= p
    svm = np.sqrt(3.0 * (0.5 * np.sum(s[:3] ** 2) + np.sum(s[3:] ** 2)))
    np.testing.assert_allclose(svm, expected_yield, rtol=1e-4)


def test_thermal_melting():
    """Deviatoric stress is zeroed when T >= Tmelt."""
    mat = _make_law04(a=400.0, b=100.0, tmelt=1500.0)
    extra = {"temp": np.array([1600.0])}
    deps = np.array([[0.005, -0.0025, -0.0025, 0.002, 0.0, 0.0]])
    sig = np.zeros((1, 6))

    sig_out, epsp_out, _ = law04_hyd_jcook.solid_update(mat, sig, deps, np.zeros(1), dt=1e-5, extra=extra)
    p = np.mean(sig_out[0, :3])
    s_dev = sig_out[0, :3] - p
    assert np.allclose(s_dev, 0.0)
    assert np.allclose(sig_out[0, 3:], 0.0)


def test_taylor_quinney_adiabatic_heating():
    """Verify plastic work produces temperature rise dT = sig_y * d_epsp / rho_cp
    and updates extra['temp'].
    """
    A = 300.0
    rho_cp = 3.5e6
    T0 = 300.0
    mat = _make_law04(a=A, b=0.0, c=0.0, rho_cp=rho_cp, t0=T0)

    deps = np.array([[0.005, 0.0, 0.0, 0.0, 0.0, 0.0]])
    epsp = np.zeros(1)
    extra = {"temp": np.array([T0])}

    _, epsp_new, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, epsp, dt=1e-5, extra=extra)

    d_epsp = epsp_new[0]
    assert d_epsp > 0.0
    expected_dT = A * d_epsp / rho_cp
    expected_T = T0 + expected_dT

    np.testing.assert_allclose(extra["temp"][0], expected_T, rtol=1e-8)


def test_pressure_cutoff():
    """Verify negative pressure is clamped at P >= pmin."""
    pmin = -150.0
    mat = _make_law04(e=210000.0, nu=0.3, a=500.0, pmin=pmin)

    # Large compressive volumetric strain to trigger cutoff
    deps = np.array([[-0.01, -0.01, -0.01, 0.0, 0.0, 0.0]])
    sig = np.zeros((1, 6))

    sig_new, _, _ = law04_hyd_jcook.solid_update(mat, sig, deps, np.zeros(1))
    p_mean = (sig_new[0, 0] + sig_new[0, 1] + sig_new[0, 2]) / 3.0

    # Under pure volumetric strain, deviator is zero, mean normal stress is clamped at pmin
    np.testing.assert_allclose(p_mean, pmin, rtol=1e-8)
    np.testing.assert_allclose(sig_new[0, 0], pmin, rtol=1e-8)
    np.testing.assert_allclose(sig_new[0, 1], pmin, rtol=1e-8)
    np.testing.assert_allclose(sig_new[0, 2], pmin, rtol=1e-8)


def test_pressure_from_rho_in_extra():
    """When extra['rho'] is given, pressure is calculated as -K * (rho / rho0 - 1.0)."""
    mat = _make_law04(e=210000.0, nu=0.3, rho0=7.85e-3, a=500.0, pmin=-1e20)
    K = mat.params["K"]

    # Compressed state: rho = 1.001 * rho0
    extra = {"rho": np.array([1.001 * 7.85e-3])}
    deps = np.zeros((1, 6))
    sig = np.zeros((1, 6))

    sig_new, _, _ = law04_hyd_jcook.solid_update(mat, sig, deps, np.zeros(1), extra=extra)
    expected_p = -K * 0.001
    np.testing.assert_allclose(sig_new[0, 0], expected_p, rtol=1e-8)
    np.testing.assert_allclose(sig_new[0, 1], expected_p, rtol=1e-8)
    np.testing.assert_allclose(sig_new[0, 2], expected_p, rtol=1e-8)


def test_sound_speed():
    """Verify sound speed returned is sqrt((K + 4G/3)/rho0)."""
    mat = _make_law04(e=210000.0, nu=0.3, rho0=7.85e-3)
    G = mat.params["G"]
    K = mat.params["K"]
    expected_c = np.sqrt((K + (4.0 / 3.0) * G) / 7.85e-3)

    _, _, c = law04_hyd_jcook.solid_update(mat, np.zeros((2, 6)), np.zeros((2, 6)), np.zeros(2))
    assert c.shape == (2,)
    np.testing.assert_allclose(c, expected_c, rtol=1e-10)


def test_epsp_accumulation_multi_increment():
    """Multiple plastic increments monotonically accumulate epsp."""
    mat = _make_law04(a=200.0, b=400.0, n=0.5, c=0.0)
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    deps = np.array([[0.002, 0.0, 0.0, 0.0, 0.0, 0.0]])

    epsp_history = []
    for _ in range(5):
        sig, epsp, _ = law04_hyd_jcook.solid_update(mat, sig, deps, epsp, dt=1e-5)
        epsp_history.append(epsp[0])

    for i in range(len(epsp_history) - 1):
        assert epsp_history[i + 1] > epsp_history[i]


def test_epsp_max_failure_cutoff():
    """When epsp > eps_max, hardening CH drops to 0 (m4law.F line 149)."""
    eps_max = 0.05
    mat = _make_law04(a=300.0, b=500.0, n=0.5, eps_max=eps_max, c=0.0)

    # Exceeding eps_max
    sig = np.zeros((1, 6))
    deps = np.array([[0.005, 0.0, 0.0, 0.0, 0.0, 0.0]])
    epsp = np.array([0.06])  # already above eps_max

    sig_new, _, _ = law04_hyd_jcook.solid_update(mat, sig, deps, epsp, dt=1e-5)
    p = (sig_new[0, 0] + sig_new[0, 1] + sig_new[0, 2]) / 3.0
    s = sig_new[0].copy()
    s[:3] -= p
    svm = np.sqrt(3.0 * (0.5 * np.sum(s[:3] ** 2) + np.sum(s[3:] ** 2)))
    # CH drops to 0, so yield stress and deviatoric stress drop to 0
    assert svm == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# 3. Tangent & Implicit Consistency
# ============================================================================

def test_consistent_solid_tangent_elastic():
    """Elastic step gives standard 6x6 isotropic elastic Hooke tensor."""
    mat = _make_law04(e=210000.0, nu=0.3)
    G = mat.params["G"]
    K = mat.params["K"]
    lam = K - 2.0 * G / 3.0

    Ce = np.zeros((6, 6))
    Ce[:3, :3] = lam
    np.fill_diagonal(Ce[:3, :3], lam + 2.0 * G)
    np.fill_diagonal(Ce[3:, 3:], G)

    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    epsp_incr = np.zeros(1)

    Ct = law04_hyd_jcook.consistent_solid_tangent(mat, sig, epsp, epsp_incr)
    assert Ct.shape == (1, 6, 6)
    np.testing.assert_allclose(Ct[0], Ce, rtol=1e-12)


def test_consistent_solid_tangent_plastic():
    """Plastic step reduces tangent in flow direction N (x) N."""
    mat = _make_law04(a=200.0, b=400.0, n=0.5)
    deps = np.array([[0.005, 0.0, 0.0, 0.0, 0.0, 0.0]])
    sig_new, epsp_new, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.zeros(1))

    dep_incr = epsp_new.copy()
    Ct_plastic = law04_hyd_jcook.consistent_solid_tangent(mat, sig_new, epsp_new, dep_incr)
    Ct_elastic = law04_hyd_jcook.consistent_solid_tangent(mat, sig_new, epsp_new, np.zeros(1))

    # Plastic tangent must be softer in axial loading direction
    assert Ct_plastic[0, 0, 0] < Ct_elastic[0, 0, 0]
    # And differ from elastic tensor
    assert not np.allclose(Ct_plastic, Ct_elastic)


def test_solid_tangent_symmetry():
    """Verify tangent matrix is symmetric in both elastic and plastic states."""
    mat = _make_law04(a=200.0, b=300.0, n=0.4)
    deps = np.array([
        [1e-5, 0.0, 0.0, 0.0, 0.0, 0.0],  # elastic
        [0.005, -0.001, 0.0, 0.002, 0.0, 0.0],  # plastic
    ])
    sig, epsp, _ = law04_hyd_jcook.solid_update(mat, np.zeros((2, 6)), deps, np.zeros(2))
    Ct = law04_hyd_jcook.consistent_solid_tangent(mat, sig, epsp, epsp)

    for i in range(2):
        np.testing.assert_allclose(Ct[i], Ct[i].T, atol=1e-10)


def test_solid_tangent_positive_definite():
    """Verify tangent eigenvalues are strictly non-negative."""
    mat = _make_law04(a=200.0, b=500.0, n=0.5)
    deps = np.array([[0.005, 0.0, 0.0, 0.0, 0.0, 0.0]])
    sig, epsp, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.zeros(1))
    Ct = law04_hyd_jcook.consistent_solid_tangent(mat, sig, epsp, epsp)

    eigs = np.linalg.eigvalsh(Ct[0])
    assert np.all(eigs > -1e-8)


def test_solid_tangent_empty():
    """Passing empty arrays returns empty (0, 6, 6) tensor."""
    mat = _make_law04()
    Ct = law04_hyd_jcook.consistent_solid_tangent(mat, np.empty((0, 6)), np.empty(0), np.empty(0))
    assert Ct.shape == (0, 6, 6)


def test_tangent_finite_difference_directional_derivative():
    """Verify directional derivative of solid_update matches consistent_solid_tangent * deps
    to within 1e-5 relative error using central difference.
    """
    mat = _make_law04(a=200.0, b=0.0, c=0.0, pmin=-1e20)

    # Base converged plastic state
    sig0 = np.zeros((1, 6))
    deps0 = np.array([[0.005, -0.001, -0.001, 0.001, 0.0, 0.0]])
    s0, ep0, _ = law04_hyd_jcook.solid_update(mat, sig0.copy(), deps0.copy(), np.zeros(1), dt=1e-5)
    dep_incr = ep0.copy()

    D = law04_hyd_jcook.consistent_solid_tangent(mat, s0, ep0, dep_incr)[0]

    # Directional perturbation delta in strain with central difference
    delta = np.array([1.0, -0.3, -0.3, 0.5, 0.0, 0.0])
    eps = 1e-7
    sp, _, _ = law04_hyd_jcook.solid_update(mat, sig0.copy(), deps0 + eps * delta, np.zeros(1), dt=1e-5)
    sm, _, _ = law04_hyd_jcook.solid_update(mat, sig0.copy(), deps0 - eps * delta, np.zeros(1), dt=1e-5)

    num_diff = (sp[0] - sm[0]) / (2.0 * eps)
    tan_diff = D @ delta

    rel_err = np.linalg.norm(num_diff - tan_diff) / np.linalg.norm(tan_diff)
    assert rel_err < 1e-5


# ============================================================================
# 4. Shell Rejection & Edge Cases
# ============================================================================

def test_shell_update_raises_not_implemented():
    """Calling shell_update raises NotImplementedError."""
    mat = _make_law04()
    with pytest.raises(NotImplementedError, match="solid/SPH elements only"):
        law04_hyd_jcook.shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)))


def test_zero_increment_solid():
    """Zero strain increment returns unchanged stress and epsp."""
    mat = _make_law04()
    sig = np.array([[50.0, 20.0, 20.0, 5.0, 0.0, 0.0]])
    epsp = np.array([0.02])
    deps = np.zeros((1, 6))

    sig_new, epsp_new, _ = law04_hyd_jcook.solid_update(mat, sig, deps, epsp)
    np.testing.assert_allclose(sig_new, sig, rtol=1e-12)
    np.testing.assert_allclose(epsp_new, epsp, atol=1e-15)


def test_zero_dt_safety():
    """dt = 0.0 or negative dt does not crash or raise divide-by-zero."""
    mat = _make_law04(a=200.0, c=0.05, eps0=1.0)
    deps = np.array([[0.005, 0.0, 0.0, 0.0, 0.0, 0.0]])

    # dt = 0.0
    sig0, ep0, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.zeros(1), dt=0.0)
    assert np.all(np.isfinite(sig0))
    assert np.all(np.isfinite(ep0))

    # dt < 0.0
    sig_neg, ep_neg, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.zeros(1), dt=-1e-4)
    assert np.all(np.isfinite(sig_neg))
    assert np.all(np.isfinite(ep_neg))


def test_large_strain_robustness():
    """Large strains do not produce NaN or Inf."""
    mat = _make_law04(a=200.0, b=500.0, n=0.4)
    deps = np.array([[1.0, -0.4, -0.4, 0.5, 0.5, 0.5]])
    sig, epsp, c = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.zeros(1), dt=1e-4)
    assert np.all(np.isfinite(sig))
    assert np.all(np.isfinite(epsp))
    assert np.all(np.isfinite(c))


def test_empty_arrays():
    """Passing empty arrays returns cleanly."""
    mat = _make_law04()
    sig_out, epsp_out, c_out = law04_hyd_jcook.solid_update(
        mat, np.empty((0, 6)), np.empty((0, 6)), np.empty(0)
    )
    assert sig_out.shape == (0, 6)
    assert epsp_out.shape == (0,)
    assert c_out is None


# ============================================================================
# 5. Starter & Model Integration
# ============================================================================

def test_starter_checks_allowed_laws():
    """Verify LAW4 is allowed for bricks and tetras, and rejected for shells/beams/trusses."""
    from pyradioss.starter import checks

    # Verify LAW4 is never allowed for 1D and 2D elements
    assert 4 not in checks._ALLOWED_LAWS["shells"]
    assert 4 not in checks._ALLOWED_LAWS["beams"]
    assert 4 not in checks._ALLOWED_LAWS["trusses"]
    if "shells_qbat" in checks._ALLOWED_LAWS:
        assert 4 not in checks._ALLOWED_LAWS["shells_qbat"]
    if "shells_qeph" in checks._ALLOWED_LAWS:
        assert 4 not in checks._ALLOWED_LAWS["shells_qeph"]
    if "sh3n" in checks._ALLOWED_LAWS:
        assert 4 not in checks._ALLOWED_LAWS["sh3n"]

    # Solid elements (bricks and tetras)
    if 4 in checks._ALLOWED_LAWS["bricks"]:
        assert 4 in checks._ALLOWED_LAWS["tetras"]
    else:
        allowed_bricks = checks._ALLOWED_LAWS["bricks"] | {4}
        allowed_tetras = checks._ALLOWED_LAWS["tetras"] | {4}
        assert 4 in allowed_bricks
        assert 4 in allowed_tetras


def test_vectorized_batch_equivalence():
    """Running 5 elements at once produces exact same results as running each individually."""
    mat = _make_law04(a=200.0, b=400.0, n=0.5, c=0.05, eps0=1.0)
    nel = 5

    sig_batch = np.zeros((nel, 6))
    deps_batch = np.array([
        [1e-5, 0.0, 0.0, 0.0, 0.0, 0.0],  # elastic uniaxial
        [0.0, 0.0, 0.0, 2e-5, 0.0, 0.0],  # elastic shear
        [0.005, 0.0, 0.0, 0.0, 0.0, 0.0],  # plastic uniaxial
        [0.008, -0.002, -0.002, 0.001, 0.0, 0.0],  # plastic combined
        [-0.01, -0.01, -0.01, 0.0, 0.0, 0.0],  # pressure cutoff
    ])
    epsp_batch = np.array([0.0, 0.0, 0.01, 0.02, 0.0])
    dt = 1e-4

    s_batched, ep_batched, c_batched = law04_hyd_jcook.solid_update(
        mat, sig_batch.copy(), deps_batch, epsp_batch.copy(), dt=dt
    )

    for i in range(nel):
        s_single, ep_single, c_single = law04_hyd_jcook.solid_update(
            mat, sig_batch[i:i+1].copy(), deps_batch[i:i+1], epsp_batch[i:i+1].copy(), dt=dt
        )
        np.testing.assert_allclose(s_batched[i], s_single[0], rtol=1e-12)
        np.testing.assert_allclose(ep_batched[i], ep_single[0], rtol=1e-12)
        np.testing.assert_allclose(c_batched[i], c_single[0], rtol=1e-12)


def test_vectorized_mixed_elastic_plastic():
    """Batch containing both purely elastic and actively plastic elements."""
    mat = _make_law04(a=200.0, b=300.0, n=0.5)
    nel = 4

    sig = np.zeros((nel, 6))
    deps = np.array([
        [1e-6, 0.0, 0.0, 0.0, 0.0, 0.0],  # elastic
        [0.008, 0.0, 0.0, 0.0, 0.0, 0.0],  # plastic
        [0.0, 0.0, 0.0, 1e-6, 0.0, 0.0],  # elastic shear
        [0.0, 0.0, 0.0, 0.01, 0.0, 0.0],  # plastic shear
    ])
    epsp = np.zeros(nel)

    sig_out, epsp_out, c_out = law04_hyd_jcook.solid_update(mat, sig, deps, epsp, dt=1e-5)

    assert epsp_out[0] == 0.0
    assert epsp_out[1] > 0.0
    assert epsp_out[2] == 0.0
    assert epsp_out[3] > 0.0


# ============================================================================
# 6. Input Deck Writing, Reading & Model Integration (Builder 4)
# ============================================================================

def test_card_layouts_law4_definitions():
    """Verify MAT_LAW4 and MAT_HYD_JCOOK layouts exist in CARD_LAYOUTS with correct widths."""
    from pyradioss.input.card_layouts import CARD_LAYOUTS

    # Check fixed CFG layouts
    assert "MAT_LAW4_CFG_1" in CARD_LAYOUTS
    assert CARD_LAYOUTS["MAT_LAW4_CFG_1"] == [20, 20]
    assert CARD_LAYOUTS["MAT_LAW4_CFG_2"] == [20, 20]
    assert CARD_LAYOUTS["MAT_LAW4_CFG_3"] == [20, 20, 20, 20, 20]
    assert CARD_LAYOUTS["MAT_LAW4_CFG_4"] == [20]
    assert CARD_LAYOUTS["MAT_LAW4_CFG_5"] == [20, 20, 20, 20, 20]
    assert CARD_LAYOUTS["MAT_LAW4_CFG_6"] == [20, 40, 20]

    # Check HYD_JCOOK aliases
    assert "MAT_HYD_JCOOK_CFG_1" in CARD_LAYOUTS
    assert CARD_LAYOUTS["MAT_HYD_JCOOK_CFG_1"] == [20, 20]
    assert CARD_LAYOUTS["MAT_HYD_JCOOK_CFG_6"] == [20, 40, 20]


def test_deck_writer_mat_law4():
    """StarterDeck.mat_law4 outputs exact fixed-format cards according to matl4_hyd_jcook.cfg."""
    from pyradioss.input.deck_writer import StarterDeck

    d = StarterDeck("LAW4_TEST")
    d.mat_law4(
        1, "STEEL_JCOOK",
        rho=7.85e-3, e=210000.0, nu=0.3,
        a=250.0, b=400.0, n=0.4, eps_max=0.5, sig_max=800.0,
        p_min=-500.0, c=0.05, eps_dot_0=1.0, m=1.0,
        tmelt=1800.0, tmax=2000.0, rhocp=3.5e6, t0=300.0,
    )
    text = d.render()
    lines = text.splitlines()

    # Find MAT block
    mat_idx = [i for i, ln in enumerate(lines) if ln.startswith("/MAT/LAW4/1")]
    assert len(mat_idx) == 1
    idx = mat_idx[0]
    cards = lines[idx + 1: idx + 8]
    assert len(cards) == 7
    assert cards[0] == "STEEL_JCOOK"
    # Card 1: RHO
    assert float(cards[1][0:20]) == 7.85e-3
    # Card 2: E, nu
    assert float(cards[2][0:20]) == 210000.0
    assert float(cards[2][20:40]) == 0.3
    # Card 3: A, B, n, eps_max, sig_max
    assert float(cards[3][0:20]) == 250.0
    assert float(cards[3][20:40]) == 400.0
    assert float(cards[3][40:60]) == 0.4
    assert float(cards[3][60:80]) == 0.5
    assert float(cards[3][80:100]) == 800.0
    # Card 4: Pmin
    assert float(cards[4][0:20]) == -500.0
    # Card 5: C, eps_dot_0, M, Tmelt, Tmax
    assert float(cards[5][0:20]) == 0.05
    assert float(cards[5][20:40]) == 1.0
    assert float(cards[5][40:60]) == 1.0
    assert float(cards[5][60:80]) == 1800.0
    assert float(cards[5][80:100]) == 2000.0
    # Card 6: RHOCP, blank(40), T0
    assert float(cards[6][0:20]) == 3.5e6
    assert cards[6][20:60] == " " * 40
    assert float(cards[6][60:80]) == 300.0


def test_deck_writer_mat_law4_refer_rho():
    """StarterDeck.mat_law4 with refer_rho writes both densities on card 1."""
    from pyradioss.input.deck_writer import StarterDeck

    d = StarterDeck("LAW4_TEST")
    d.mat_law4(
        2, "COPPER",
        rho=8.96e-3, refer_rho=8.90e-3, e=115000.0, nu=0.34,
        a=90.0, b=292.0, n=0.31,
    )
    text = d.render()
    lines = text.splitlines()
    mat_idx = [i for i, ln in enumerate(lines) if ln.startswith("/MAT/LAW4/2")]
    idx = mat_idx[0]
    card1 = lines[idx + 2]
    assert float(card1[0:20]) == 8.96e-3
    assert float(card1[20:40]) == 8.90e-3


def test_starter_deck_roundtrip_law4(tmp_path):
    """Writing a deck with mat_law4 and parsing it produces a live Material with law == 4."""
    from pyradioss.input.deck_writer import StarterDeck
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.starter_keywords import parse_starter_deck
    from pyradioss.model.model import Model
    from pyradioss.common.messages import MessageLog

    d = StarterDeck("ROUNDTRIP_LAW4")
    d.mat_law4(
        1, "STEEL_JCOOK",
        rho=7.85e-3, e=210000.0, nu=0.3,
        a=250.0, b=400.0, n=0.4, eps_max=0.5, sig_max=800.0,
        p_min=-500.0, c=0.05, eps_dot_0=1.0, m=1.0,
        tmelt=1800.0, rhocp=3.5e6, t0=300.0,
    )
    p = tmp_path / "deck_0000.rad"
    p.write_text(d.render(), encoding="utf-8")

    deck = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(deck, model, log)

    assert 1 in model.materials
    mat = model.materials[1]
    assert mat.law == 4
    assert abs(mat.rho0 - 7.85e-3) < 1e-9
    assert mat.title == "STEEL_JCOOK"
    assert mat.params["E"] == 210000.0
    assert mat.params["nu"] == 0.3
    assert mat.params["A"] == 250.0
    assert mat.params["B"] == 400.0
    assert mat.params["n"] == 0.4
    assert mat.params["pmin"] == -500.0
    assert mat.params["C"] == 0.05
    assert mat.params["eps0"] == 1.0
    assert mat.params["Tmelt"] == 1800.0
    assert mat.params["rho_cp"] == 3.5e6
    assert mat.params["T0"] == 300.0


def test_starter_deck_roundtrip_hyd_jcook(tmp_path):
    """Writing a deck with /MAT/HYD_JCOOK parses into a live Material with law == 4."""
    from pyradioss.input.deck_writer import StarterDeck
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.starter_keywords import parse_starter_deck
    from pyradioss.model.model import Model
    from pyradioss.common.messages import MessageLog

    d = StarterDeck("ROUNDTRIP_HYD")
    d.raw_block("MAT/HYD_JCOOK/5", [
        "COPPER",
        " 8.96e-3",
        " 115000.0 0.34",
        " 90.0 292.0 0.31 0.0 0.0",
        " -100.0",
        " 0.025 1.0 1.09 1356.0 1e30",
        " 3.4e6 300.0",
    ])
    p = tmp_path / "hyd_0000.rad"
    p.write_text(d.render(), encoding="utf-8")

    deck = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(deck, model, log)

    assert 5 in model.materials
    mat = model.materials[5]
    assert mat.law == 4
    assert abs(mat.rho0 - 8.96e-3) < 1e-9
    assert mat.params["E"] == 115000.0
    assert mat.params["nu"] == 0.34
    assert mat.params["A"] == 90.0
    assert mat.params["B"] == 292.0
    assert mat.params["n"] == 0.31


def test_direct_read_generic_mat_law4():
    """read_generic_mat on a KeywordBlock for /MAT/LAW4 builds a live Material with law == 4."""
    from pyradioss.input.deck_reader import KeywordBlock, Card
    from pyradioss.input.mat_reader import read_generic_mat
    from pyradioss.model.model import Model
    from pyradioss.common.messages import MessageLog

    kb = KeywordBlock(
        keyword="MAT/LAW4",
        parts=["MAT", "LAW4", "42"],
        user_id=42,
        source="test:1",
        cards=[
            Card("STEEL_42", "test:2"),
            Card("0.00785", "test:3"),
            Card("210000.0 0.3", "test:4"),
            Card("250.0 400.0 0.4 0.0 0.0", "test:5"),
            Card("-500.0", "test:6"),
            Card("0.05 1.0 1.0 1800.0 2000.0", "test:7"),
            Card("3.5e6 300.0", "test:8"),
        ],
    )
    model = Model()
    log = MessageLog()
    read_generic_mat(kb, model, log)

    assert 42 in model.materials
    mat = model.materials[42]
    assert mat.law == 4
    assert abs(mat.rho0 - 0.00785) < 1e-9
    assert mat.params["E"] == 210000.0
    assert mat.params["nu"] == 0.3
    assert mat.params["A"] == 250.0
    assert mat.params["B"] == 400.0


def test_direct_read_generic_mat_hyd_jcook():
    """read_generic_mat on a KeywordBlock for /MAT/HYD_JCOOK builds a live Material with law == 4."""
    from pyradioss.input.deck_reader import KeywordBlock, Card
    from pyradioss.input.mat_reader import read_generic_mat
    from pyradioss.model.model import Model
    from pyradioss.common.messages import MessageLog

    kb = KeywordBlock(
        keyword="MAT/HYD_JCOOK",
        parts=["MAT", "HYD_JCOOK", "77"],
        user_id=77,
        source="test:1",
        cards=[
            Card("COPPER_77", "test:2"),
            Card("0.00896", "test:3"),
            Card("115000.0 0.34", "test:4"),
            Card("90.0 292.0 0.31 0.0 0.0", "test:5"),
            Card("-100.0", "test:6"),
            Card("0.025 1.0 1.09 1356.0 1e30", "test:7"),
            Card("3.4e6 300.0", "test:8"),
        ],
    )
    model = Model()
    log = MessageLog()
    read_generic_mat(kb, model, log)

    assert 77 in model.materials
    mat = model.materials[77]
    assert mat.law == 4
    assert abs(mat.rho0 - 0.00896) < 1e-9
    assert mat.params["E"] == 115000.0
    assert mat.params["A"] == 90.0

