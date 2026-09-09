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

    # Large tensile volumetric strain to trigger negative pressure cutoff
    deps = np.array([[0.01, 0.01, 0.01, 0.0, 0.0, 0.0]])
    sig = np.zeros((1, 6))

    sig_new, _, _ = law04_hyd_jcook.solid_update(mat, sig, deps, np.zeros(1))
    p_hydro = -(sig_new[0, 0] + sig_new[0, 1] + sig_new[0, 2]) / 3.0

    # Under pure volumetric strain, deviator is zero, pressure is clamped at pmin
    np.testing.assert_allclose(p_hydro, pmin, rtol=1e-8)
    np.testing.assert_allclose(sig_new[0, 0], -pmin, rtol=1e-8)
    np.testing.assert_allclose(sig_new[0, 1], -pmin, rtol=1e-8)
    np.testing.assert_allclose(sig_new[0, 2], -pmin, rtol=1e-8)


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
    assert 4 in checks._ALLOWED_LAWS["bricks"]
    assert 4 in checks._ALLOWED_LAWS["tetras"]



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


# ============================================================================
# 7. Wave 2 Comprehensive Edge Cases & Verification
# ============================================================================

def test_build_validation_missing_keys_and_valid_nu_boundaries():
    """Verify ValueError on missing required parameters and valid nu boundaries."""
    # Missing E
    with pytest.raises(ValueError, match="Young's modulus E must be > 0"):
        law04_hyd_jcook.build_law04({"id": 1, "density": 7.8e-3, "params": {"MAT_NU": 0.3}})

    # Missing nu
    with pytest.raises(ValueError, match=r"Poisson's ratio nu must be in \[0, 0\.5\)"):
        law04_hyd_jcook.build_law04({"id": 1, "density": 7.8e-3, "params": {"MAT_E": 210000.0}})

    # Missing density (rho0)
    with pytest.raises(ValueError, match="Initial density rho0 must be > 0"):
        law04_hyd_jcook.build_law04({"id": 1, "params": {"MAT_E": 210000.0, "MAT_NU": 0.3}})

    # Valid boundary nu = 0.0
    mat_nu0 = _make_law04(nu=0.0)
    assert mat_nu0.params["nu"] == 0.0
    assert mat_nu0.params["G"] == pytest.approx(mat_nu0.params["E"] / 2.0)
    assert mat_nu0.params["K"] == pytest.approx(mat_nu0.params["E"] / 3.0)

    # Valid boundary nu = 0.49999
    mat_nu_high = _make_law04(nu=0.49999)
    assert mat_nu_high.params["nu"] == pytest.approx(0.49999)


def test_hardening_clamping_n_variations():
    """Verify n=0 or n=1 via various keys ('N', 'n', 'MAT_HARD') clamps to 1.0001 and steps cleanly."""
    for key in ("N", "n", "MAT_HARD"):
        for val in (0, 1, 0.0, 1.0):
            rec = {
                "id": 1,
                "density": 7.85e-3,
                "params": {"MAT_E": 210000.0, "MAT_NU": 0.3, "MAT_SIGY": 250.0, "MAT_BETA": 400.0, key: val},
            }
            mat = law04_hyd_jcook.build_law04(rec)
            assert mat.params["N"] == pytest.approx(1.0001)
            assert mat.params["n"] == pytest.approx(1.0001)

            # Check plastic step execution with clamped n=1.0001
            sig, epsp, _ = law04_hyd_jcook.solid_update(
                mat, np.zeros((1, 6)), np.array([[0.005, 0.0, 0.0, 0.0, 0.0, 0.0]]), np.zeros(1), dt=1e-5
            )
            assert epsp[0] > 0.0
            assert np.all(np.isfinite(sig))


def test_zero_strain_increment_with_active_plasticity():
    """Zero strain increment on rate-independent plastic element preserves stress and plastic strain."""
    # 1. Rate-independent: stress and plastic strain remain exactly constant
    mat = _make_law04(a=250.0, b=400.0, n=0.5, c=0.0, rho_cp=0.0, t0=300.0)
    deps_init = np.array([[0.005, -0.001, -0.001, 0.001, 0.0, 0.0]])
    s0, ep0, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps_init, np.zeros(1), dt=1e-5)

    deps_zero = np.zeros((1, 6))
    s1, ep1, _ = law04_hyd_jcook.solid_update(mat, s0.copy(), deps_zero, ep0.copy(), dt=1e-5)

    np.testing.assert_allclose(s1, s0, rtol=1e-12)
    np.testing.assert_allclose(ep1, ep0, atol=1e-15)

    # 2. Rate-dependent: dropping strain rate to 0 relaxes dynamic overstress to static yield surface
    mat_rate = _make_law04(a=250.0, b=0.0, c=0.1, eps0=1.0)
    s_dyn, ep_dyn, _ = law04_hyd_jcook.solid_update(mat_rate, np.zeros((1, 6)), deps_init, np.zeros(1), dt=1e-5)
    s_stat, ep_stat, _ = law04_hyd_jcook.solid_update(mat_rate, s_dyn.copy(), deps_zero, ep_dyn.copy(), dt=1e-5)
    p_stat = np.mean(s_stat[0, :3])
    dev_stat = s_stat[0].copy()
    dev_stat[:3] -= p_stat
    svm_stat = np.sqrt(3.0 * (0.5 * np.sum(dev_stat[:3] ** 2) + np.sum(dev_stat[3:] ** 2)))
    np.testing.assert_allclose(svm_stat, 250.0, rtol=1e-4)


def test_zero_and_negative_dt_unenhanced_rate():
    """dt <= 0 forces eps_dot = 0, C_E = 1.0, and unenhanced yield stress."""
    A = 300.0
    C = 0.1
    eps0 = 1.0
    mat = _make_law04(a=A, b=0.0, c=C, eps0=eps0)

    deps = np.array([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]])  # large increment

    # With dt = 0.0
    s_dt0, ep_dt0, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.zeros(1), dt=0.0)
    p_dt0 = (s_dt0[0, 0] + s_dt0[0, 1] + s_dt0[0, 2]) / 3.0
    dev0 = s_dt0[0].copy()
    dev0[:3] -= p_dt0
    svm_dt0 = np.sqrt(3.0 * (0.5 * np.sum(dev0[:3] ** 2) + np.sum(dev0[3:] ** 2)))
    np.testing.assert_allclose(svm_dt0, A, rtol=1e-4)

    # With dt < 0.0
    s_dtneg, _, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.zeros(1), dt=-1e-3)
    p_dtneg = (s_dtneg[0, 0] + s_dtneg[0, 1] + s_dtneg[0, 2]) / 3.0
    devneg = s_dtneg[0].copy()
    devneg[:3] -= p_dtneg
    svm_dtneg = np.sqrt(3.0 * (0.5 * np.sum(devneg[:3] ** 2) + np.sum(devneg[3:] ** 2)))
    np.testing.assert_allclose(svm_dtneg, A, rtol=1e-4)

    # In contrast, with dt > 0, rate enhancement occurs
    s_dtpos, _, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.zeros(1), dt=1e-5)
    p_dtpos = (s_dtpos[0, 0] + s_dtpos[0, 1] + s_dtpos[0, 2]) / 3.0
    devpos = s_dtpos[0].copy()
    devpos[:3] -= p_dtpos
    svm_dtpos = np.sqrt(3.0 * (0.5 * np.sum(devpos[:3] ** 2) + np.sum(devpos[3:] ** 2)))
    assert svm_dtpos > A * 1.5


def test_strain_rate_threshold_exact_and_shear_components():
    """Exact rate threshold eps_dot == eps0 gives CE=1.0; shear tensor rate is 0.5 * gamma_dot."""
    A = 200.0
    C = 0.05
    eps0 = 10.0
    mat = _make_law04(a=A, b=0.0, c=C, eps0=eps0)

    # 1. Exact threshold: eps_dot = 10.0 == eps0 -> CE = 1.0
    dt = 1e-3
    deps_thresh = np.array([[10.0 * dt, 0.0, 0.0, 0.0, 0.0, 0.0]])
    s_thresh, _, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps_thresh, np.zeros(1), dt=dt)
    p_th = np.mean(s_thresh[0, :3])
    dev_th = s_thresh[0].copy()
    dev_th[:3] -= p_th
    svm_th = np.sqrt(3.0 * (0.5 * np.sum(dev_th[:3] ** 2) + np.sum(dev_th[3:] ** 2)))
    np.testing.assert_allclose(svm_th, A, rtol=1e-4)

    # 2. Shear strain rate: engineering shear increment gamma_12 = 40.0 * dt -> rate measure = 0.5 * 40 = 20 > 10
    deps_shear = np.array([[0.0, 0.0, 0.0, 40.0 * dt, 0.0, 0.0]])
    expected_ce = 1.0 + C * np.log(20.0 / eps0)
    expected_yield = A * expected_ce

    s_sh, _, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps_shear, np.zeros(1), dt=dt)
    p_sh = np.mean(s_sh[0, :3])
    dev_sh = s_sh[0].copy()
    dev_sh[:3] -= p_sh
    svm_sh = np.sqrt(3.0 * (0.5 * np.sum(dev_sh[:3] ** 2) + np.sum(dev_sh[3:] ** 2)))
    np.testing.assert_allclose(svm_sh, expected_yield, rtol=1e-4)


def test_temperature_sub_t0_and_clamping():
    """Verify sub-T0 temperatures yield CT = 1.0 and T >= Tmelt relaxes all deviatoric stresses."""
    A = 350.0
    T0 = 300.0
    Tmelt = 1500.0
    mat = _make_law04(a=A, b=0.0, c=0.0, t0=T0, tmelt=Tmelt)
    deps = np.array([[0.005, 0.0, 0.0, 0.0, 0.0, 0.0]])

    # Sub-T0 temperature (e.g. 150 K)
    extra_sub = {"temp": np.array([150.0])}
    s_sub, _, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.zeros(1), dt=1e-5, extra=extra_sub)
    p_sub = np.mean(s_sub[0, :3])
    dev_sub = s_sub[0].copy()
    dev_sub[:3] -= p_sub
    svm_sub = np.sqrt(3.0 * (0.5 * np.sum(dev_sub[:3] ** 2) + np.sum(dev_sub[3:] ** 2)))
    np.testing.assert_allclose(svm_sub, A, rtol=1e-4)

    # Melted temperature (T >= Tmelt) retains pressure while deviator is zero
    extra_melt = {"temp": np.array([Tmelt + 100.0])}
    s_melt, _, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.zeros(1), dt=1e-5, extra=extra_melt)
    p_melt = np.mean(s_melt[0, :3])
    assert p_melt > 0.0
    np.testing.assert_allclose(s_melt[0, :3], p_melt, rtol=1e-8)
    np.testing.assert_allclose(s_melt[0, 3:], 0.0, atol=1e-12)


def test_taylor_quinney_zero_specific_heat_and_multi_increment():
    """rho_cp == 0 prevents heating; rho_cp > 0 accumulates temperature and induces progressive softening."""
    # 1. rho_cp == 0
    mat_no_heat = _make_law04(a=300.0, b=0.0, rho_cp=0.0, t0=300.0)
    extra_no = {"temp": np.array([300.0])}
    deps = np.array([[0.005, 0.0, 0.0, 0.0, 0.0, 0.0]])
    law04_hyd_jcook.solid_update(mat_no_heat, np.zeros((1, 6)), deps, np.zeros(1), dt=1e-5, extra=extra_no)
    assert extra_no["temp"][0] == 300.0

    # 2. Multi-increment progressive heating and softening
    mat_heat = _make_law04(a=400.0, b=0.0, rho_cp=1.0e6, t0=300.0, tmelt=1000.0, m=1.0)
    extra_h = {"temp": np.array([300.0])}
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)

    flow_stresses = []
    for _ in range(4):
        sig, epsp, _ = law04_hyd_jcook.solid_update(mat_heat, sig, deps, epsp, dt=1e-5, extra=extra_h)
        p = np.mean(sig[0, :3])
        dev = sig[0].copy()
        dev[:3] -= p
        svm = np.sqrt(3.0 * (0.5 * np.sum(dev[:3] ** 2) + np.sum(dev[3:] ** 2)))
        flow_stresses.append(svm)

    # Temperature increases monotonically
    assert extra_h["temp"][0] > 300.0
    # Flow stress progressively softens due to heating
    for i in range(len(flow_stresses) - 1):
        assert flow_stresses[i + 1] < flow_stresses[i]


def test_pressure_cutoff_tensile_vs_compressive_and_mixed_batch():
    """Compressive pressure is unclamped, severe tensile pressure is clamped to pmin, deviator untouched."""
    pmin = -100.0
    mat = _make_law04(e=210000.0, nu=0.3, a=500.0, pmin=pmin)
    K = mat.params["K"]

    # Batch of 3 elements:
    # 0: Compressive volumetric strain (e_vol = -0.003 < 0 -> P = +K * 0.003)
    # 1: Moderate tensile volumetric strain (e_vol = +1e-4 -> P = -K * 1e-4 > pmin)
    # 2: Severe tensile volumetric strain (e_vol = +0.015 -> P = -K * 0.015 << pmin)
    deps = np.array([
        [-0.001, -0.001, -0.001, 1e-5, 0.0, 0.0],
        [3.33333333e-5, 3.33333333e-5, 3.33333333e-5, 1e-5, 0.0, 0.0],
        [0.005, 0.005, 0.005, 1e-5, 0.0, 0.0],
    ])
    sig = np.zeros((3, 6))
    epsp = np.zeros(3)

    sig_new, _, _ = law04_hyd_jcook.solid_update(mat, sig, deps, epsp)

    # Element 0: compressive pressure P = -mean(sig[:3]) = +K * 0.003
    p0 = -np.mean(sig_new[0, :3])
    expected_p0 = K * 0.003
    np.testing.assert_allclose(p0, expected_p0, rtol=1e-6)

    # Element 1: moderate tension, not clamped: P = -K * 1e-4 > pmin
    p1 = -np.mean(sig_new[1, :3])
    expected_p1 = -K * 1e-4
    assert expected_p1 > pmin
    np.testing.assert_allclose(p1, expected_p1, rtol=1e-6)

    # Element 2: severe tension, clamped to pmin: P = pmin
    p2 = -np.mean(sig_new[2, :3])
    np.testing.assert_allclose(p2, pmin, rtol=1e-8)

    # Deviatoric shear stress xy = G * deps_xy is identical across all three
    G = mat.params["G"]
    for i in range(3):
        np.testing.assert_allclose(sig_new[i, 3], G * 1e-5, rtol=1e-8)


def test_embedded_polynomial_eos_integration():
    """Embedded polynomial EOS builds EquationOfState and integrates via eos module."""
    from pyradioss.materials import eos

    mat = _make_law04(
        c0=1.5,
        c1=160000.0,
        c2=4000.0,
        c3=200.0,
        c4=0.4,
        c5=0.1,
        e0=12.0,
        psh=3.0,
        pmin=-200.0,
    )
    assert mat.eos is not None
    assert mat.eos.kind == "POLYNOMIAL"
    assert mat.eos.params["c0"] == 1.5
    assert mat.eos.params["c1"] == 160000.0
    assert mat.eos.params["c2"] == 4000.0
    assert mat.eos.params["c3"] == 200.0
    assert mat.eos.params["c4"] == 0.4
    assert mat.eos.params["c5"] == 0.1
    assert mat.eos.params["e0"] == 12.0
    assert mat.eos.params["psh"] == 3.0
    assert mat.eos.params["pmin"] == -200.0

    # Test eos.initial_state
    e0_init, p0_init = eos.initial_state(mat.eos)
    assert e0_init == 12.0
    assert p0_init == pytest.approx(1.5 + 0.4 * 12.0)

    # Test eos.update
    mu = np.array([0.02])
    dv = np.array([-0.01])
    e_old = np.array([12.0])
    p_old = np.array([p0_init])
    de_other = np.array([0.5])
    p_new, e_new, c2 = eos.update(mat.eos, mu, dv, e_old, p_old, de_other)
    assert p_new[0] > p_old[0]
    assert e_new[0] > 0.0
    assert c2[0] > 0.0

    # Material without EOS parameters has mat.eos is None
    mat_plain = _make_law04()
    assert mat_plain.eos is None


def test_tangent_directional_derivative_shear_and_multiaxial():
    """Consistent solid tangent matches directional derivative of solid_update for shear and multiaxial loading."""
    from pyradioss import materials

    # 1. Verification of exact directional derivative with consistent tangent (b=0)
    mat = _make_law04(a=250.0, b=0.0, c=0.0, pmin=-1e20)
    sig0 = np.zeros((1, 6))
    deps0 = np.array([[0.006, -0.002, -0.002, 0.003, 0.001, -0.001]])
    s0, ep0, _ = law04_hyd_jcook.solid_update(mat, sig0.copy(), deps0.copy(), np.zeros(1), dt=1e-5)
    dep_incr = ep0.copy()

    D = materials.solid_tangent(mat, s0, ep0, dep_incr)[0]
    D_direct = law04_hyd_jcook.consistent_solid_tangent(mat, s0, ep0, dep_incr)[0]
    np.testing.assert_allclose(D, D_direct, rtol=1e-12)

    # Direction 1: Pure shear perturbation
    delta_shear = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    eps = 1e-7
    sp, _, _ = law04_hyd_jcook.solid_update(mat, sig0.copy(), deps0 + eps * delta_shear, np.zeros(1), dt=1e-5)
    sm, _, _ = law04_hyd_jcook.solid_update(mat, sig0.copy(), deps0 - eps * delta_shear, np.zeros(1), dt=1e-5)
    num_diff_sh = (sp[0] - sm[0]) / (2.0 * eps)
    tan_diff_sh = D @ delta_shear
    rel_err_sh = np.linalg.norm(num_diff_sh - tan_diff_sh) / np.linalg.norm(tan_diff_sh)
    assert rel_err_sh < 1e-5

    # Direction 2: General multiaxial perturbation
    delta_multi = np.array([1.0, -0.5, -0.2, 0.4, -0.3, 0.2])
    sp_m, _, _ = law04_hyd_jcook.solid_update(mat, sig0.copy(), deps0 + eps * delta_multi, np.zeros(1), dt=1e-5)
    sm_m, _, _ = law04_hyd_jcook.solid_update(mat, sig0.copy(), deps0 - eps * delta_multi, np.zeros(1), dt=1e-5)
    num_diff_m = (sp_m[0] - sm_m[0]) / (2.0 * eps)
    tan_diff_m = D @ delta_multi
    rel_err_m = np.linalg.norm(num_diff_m - tan_diff_m) / np.linalg.norm(tan_diff_m)
    assert rel_err_m < 1e-5

    # 2. Hardening material (b > 0): tangent is symmetric, positive semi-definite, and softer than elastic
    mat_hard = _make_law04(a=250.0, b=500.0, n=0.4, c=0.0)
    s_h, ep_h, _ = law04_hyd_jcook.solid_update(mat_hard, sig0.copy(), deps0.copy(), np.zeros(1), dt=1e-5)
    D_hard = materials.solid_tangent(mat_hard, s_h, ep_h, ep_h)[0]
    D_el = materials.solid_tangent(mat_hard, s_h, ep_h, np.zeros(1))[0]
    np.testing.assert_allclose(D_hard, D_hard.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(D_hard) > -1e-8)
    assert D_hard[0, 0] < D_el[0, 0]


def test_vectorized_batch_eight_diverse_states():
    """Vectorized solid_update on 8 heterogeneous states exactly matches individual element updates."""
    mat = _make_law04(a=220.0, b=450.0, n=0.45, c=0.04, eps0=2.0, t0=300.0, tmelt=1600.0, tmax=1800.0, pmin=-120.0)
    nel = 8

    sig_batch = np.zeros((nel, 6))
    deps_batch = np.array([
        [1e-5, 0.0, 0.0, 0.0, 0.0, 0.0],  # 0: elastic tension
        [0.0, 0.0, 0.0, 2e-5, 0.0, 0.0],  # 1: elastic shear
        [0.005, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2: virgin plastic
        [0.008, -0.002, -0.002, 0.001, 0.0, 0.0],  # 3: hardened plastic
        [0.015, -0.005, -0.005, 0.002, 0.0, 0.0],  # 4: high strain rate
        [0.006, -0.002, -0.002, 0.0, 0.0, 0.0],  # 5: thermally softened
        [0.004, -0.001, -0.001, 0.001, 0.0, 0.0],  # 6: melted (T >= Tmelt)
        [-0.01, -0.01, -0.01, 0.0, 0.0, 0.0],  # 7: tensile pressure cutoff
    ])
    epsp_batch = np.array([0.0, 0.0, 0.0, 0.03, 0.01, 0.005, 0.02, 0.0])
    temp_batch = np.array([300.0, 300.0, 300.0, 300.0, 300.0, 800.0, 1700.0, 300.0])
    extra_batch = {"temp": temp_batch.copy()}
    dt = 1e-4

    s_batched, ep_batched, c_batched = law04_hyd_jcook.solid_update(
        mat, sig_batch.copy(), deps_batch, epsp_batch.copy(), dt=dt, extra=extra_batch
    )

    for i in range(nel):
        extra_single = {"temp": np.array([temp_batch[i]])}
        s_single, ep_single, c_single = law04_hyd_jcook.solid_update(
            mat, sig_batch[i:i+1].copy(), deps_batch[i:i+1], epsp_batch[i:i+1].copy(), dt=dt, extra=extra_single
        )
        np.testing.assert_allclose(s_batched[i], s_single[0], rtol=1e-12)
        np.testing.assert_allclose(ep_batched[i], ep_single[0], rtol=1e-12)
        np.testing.assert_allclose(c_batched[i], c_single[0], rtol=1e-12)
        np.testing.assert_allclose(extra_batch["temp"][i], extra_single["temp"][0], rtol=1e-12)


def test_starter_check_model_solids_accepted_shells_rejected(tmp_path):
    """check_model accepts LAW4 for solid elements and logs error for shell elements."""
    from pyradioss.starter.checks import check_model
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.starter_keywords import parse_starter_deck
    from pyradioss.starter.initialization import (
        resolve_materials, build_element_groups, initialize_elements_and_mass,
    )
    from pyradioss.model.model import Model
    from pyradioss.common.messages import MessageLog

    # 1. Solid deck with LAW4 (Hexa8)
    deck_solid = (
        "/NODE\n"
        "1 0.0 0.0 0.0\n"
        "2 1.0 0.0 0.0\n"
        "3 1.0 1.0 0.0\n"
        "4 0.0 1.0 0.0\n"
        "5 0.0 0.0 1.0\n"
        "6 1.0 0.0 1.0\n"
        "7 1.0 1.0 1.0\n"
        "8 0.0 1.0 1.0\n"
        "/BRICK/1/1\n"
        "1 1 2 3 4 5 6 7 8\n"
        "/PART/1\n"
        "Part_Solid\n"
        "1 1 1\n"
        "/PROP/SOLID/1\n"
        "Solid_Prop\n"
        "/MAT/LAW4/1\n"
        "LAW4_MAT\n"
        " 0.00785\n"
        " 210000.0 0.3\n"
        " 250.0 400.0 0.4 0.0 0.0\n"
        " -500.0\n"
        " 0.05 1.0 1.0 1800.0 2000.0\n"
        " 3.5e6 300.0\n"
        "/END\n"
    )
    p_solid = tmp_path / "solid_0000.rad"
    p_solid.write_text(deck_solid, encoding="utf-8")
    model_s = Model()
    log_s = MessageLog()
    parse_starter_deck(read_deck(str(p_solid)), model_s, log_s)
    resolve_materials(model_s, log_s)
    build_element_groups(model_s, log_s)
    initialize_elements_and_mass(model_s, log_s)
    check_model(model_s, log_s)
    assert not log_s.errors, log_s.errors

    # 2. Shell deck with LAW4
    deck_shell = (
        "/NODE\n"
        "1 0.0 0.0 0.0\n"
        "2 1.0 0.0 0.0\n"
        "3 1.0 1.0 0.0\n"
        "4 0.0 1.0 0.0\n"
        "/SHELL/1\n"
        "1 1 2 3 4\n"
        "/PART/1\n"
        "Part_Shell\n"
        "1 1 1\n"
        "/PROP/SHELL/1\n"
        "Shell_Prop\n"
        " 1.0 5\n"
        "/MAT/LAW4/1\n"
        "LAW4_MAT\n"
        " 0.00785\n"
        " 210000.0 0.3\n"
        " 250.0 400.0 0.4 0.0 0.0\n"
        " -500.0\n"
        " 0.05 1.0 1.0 1800.0 2000.0\n"
        " 3.5e6 300.0\n"
        "/END\n"
    )
    p_shell = tmp_path / "shell_0000.rad"
    p_shell.write_text(deck_shell, encoding="utf-8")
    model_sh = Model()
    log_sh = MessageLog()
    parse_starter_deck(read_deck(str(p_shell)), model_sh, log_sh)
    resolve_materials(model_sh, log_sh)
    build_element_groups(model_sh, log_sh)
    initialize_elements_and_mass(model_sh, log_sh)
    check_model(model_sh, log_sh)
    assert log_sh.errors
    assert any("not ported for shells elements" in err for err in log_sh.errors)


def test_materials_dispatcher_integration():
    """Verify materials module dispatches solid_update, solid_tangent, shell_update, needs_env, extra_shapes."""
    from pyradioss import materials

    mat = _make_law04(a=200.0)
    sig = np.zeros((1, 6))
    deps = np.array([[1e-5, 0.0, 0.0, 0.0, 0.0, 0.0]])
    sig_out, epsp_out, c_out = materials.solid_update(mat, sig, deps, np.zeros(1), dt=1e-5)
    assert sig_out.shape == (1, 6)
    assert epsp_out.shape == (1,)
    assert c_out.shape == (1,)

    Ct = materials.solid_tangent(mat, sig_out, epsp_out, np.zeros(1))
    assert Ct.shape == (1, 6, 6)

    assert materials.needs_env(mat) is True
    assert "temp" in materials.extra_shapes(mat)

    with pytest.raises(NotImplementedError, match="solid/SPH elements only"):
        materials.shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)), np.zeros(1), dt=1e-5)


# ============================================================================
# 7. Wave 2 Verifier 1 Audit Checks (m4law.F / hm_read_mat04.F Fidelity)
# ============================================================================

def test_audit_check1_hydrostatic_deviatoric_split():
    """Audit check 1:
    In m4law.F: P = -THIRD*(SIG1+SIG2+SIG3), DAV = -THIRD*(D1+D2+D3).
    Check how pressure and deviatoric stresses are calculated and recombined in solid_update.
    """
    mat = _make_law04(e=210000.0, nu=0.3, a=500.0, pmin=-1e30)
    G = mat.params["G"]
    K = mat.params["K"]

    sig_old = np.array([[100.0, 50.0, -30.0, 20.0, -10.0, 15.0]])
    deps = np.array([[0.001, -0.0005, 0.0002, 0.0004, -0.0002, 0.0006]])

    # Fortran reference calculations
    sig1, sig2, sig3, sig4, sig5, sig6 = sig_old[0]
    d1, d2, d3, d4, d5, d6 = deps[0]

    p_old_expected = -(sig1 + sig2 + sig3) / 3.0
    dav_expected = -(d1 + d2 + d3) / 3.0

    s_trial_xx = sig1 + p_old_expected + 2.0 * G * (d1 + dav_expected)
    s_trial_yy = sig2 + p_old_expected + 2.0 * G * (d2 + dav_expected)
    s_trial_zz = sig3 + p_old_expected + 2.0 * G * (d3 + dav_expected)
    s_trial_xy = sig4 + G * d4
    s_trial_yz = sig5 + G * d5
    s_trial_zx = sig6 + G * d6

    p_new_expected = p_old_expected + 3.0 * K * dav_expected
    sig_new_expected = np.array([
        s_trial_xx - p_new_expected,
        s_trial_yy - p_new_expected,
        s_trial_zz - p_new_expected,
        s_trial_xy,
        s_trial_yz,
        s_trial_zx,
    ])

    sig_new, epsp_new, _ = law04_hyd_jcook.solid_update(mat, sig_old.copy(), deps.copy())

    np.testing.assert_allclose(sig_new[0], sig_new_expected, rtol=1e-12)

    # Verify that deviator trace is identically zero
    p_result = -(sig_new[0, 0] + sig_new[0, 1] + sig_new[0, 2]) / 3.0
    s_result = sig_new[0, :3] + p_result
    np.testing.assert_allclose(np.sum(s_result), 0.0, atol=1e-12)
    np.testing.assert_allclose(p_result, p_new_expected, rtol=1e-12)


def test_audit_check2_strain_rate_epd_formula():
    """Audit check 2:
    EPD = MAX(ABS(D1), ABS(D2), ABS(D3), HALF*ABS(D4), HALF*ABS(D5), HALF*ABS(D6)).
    Verify law04_hyd_jcook.py reproduces this exact formula for all 6 components.
    """
    dt = 1e-4
    c_rate = 0.1
    eps0 = 1.0
    A = 200.0
    mat = _make_law04(a=A, b=0.0, c=c_rate, eps0=eps0)

    # Test 6 cases where each component in turn dominates the strain rate
    components = [
        [0.005, 0.0, 0.0, 0.0, 0.0, 0.0],          # D1 dominates: EPD = 0.005 / dt = 50
        [0.0, -0.006, 0.0, 0.0, 0.0, 0.0],         # D2 dominates: EPD = 0.006 / dt = 60
        [0.0, 0.0, 0.007, 0.0, 0.0, 0.0],          # D3 dominates: EPD = 0.007 / dt = 70
        [0.0, 0.0, 0.0, 0.016, 0.0, 0.0],          # D4 dominates: EPD = 0.5 * 0.016 / dt = 80
        [0.0, 0.0, 0.0, 0.0, -0.018, 0.0],         # D5 dominates: EPD = 0.5 * 0.018 / dt = 90
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.020],          # D6 dominates: EPD = 0.5 * 0.020 / dt = 100
    ]
    expected_epds = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    for comp, exp_epd in zip(components, expected_epds):
        deps = np.array([comp])
        sig = np.zeros((1, 6))
        expected_ce = 1.0 + c_rate * math.log(exp_epd / eps0)
        expected_sigy = A * expected_ce

        # Under plastic loading, von Mises equals yield stress:
        sig_out, _, _ = law04_hyd_jcook.solid_update(mat, sig, deps, np.zeros(1), dt=dt)
        p = -(sig_out[0, 0] + sig_out[0, 1] + sig_out[0, 2]) / 3.0
        s = sig_out[0].copy()
        s[:3] += p
        svm = math.sqrt(3.0 * (0.5 * (s[0]**2 + s[1]**2 + s[2]**2) + s[3]**2 + s[4]**2 + s[5]**2))
        np.testing.assert_allclose(svm, expected_sigy, rtol=1e-4)


def test_audit_check3_rate_enhancement_ce_branches():
    """Audit check 3:
    Rate enhancement factor C_E:
    When EPD <= EPDR, C_E = 1.0.
    When EPD > EPDR, C_E = 1.0 + C * ln(EPD / EPDR).
    Verify handling of C = 0, EPD <= EPDR, and EPD > EPDR.
    """
    A = 300.0

    # 1. C = 0 -> C_E is strictly 1.0 regardless of extreme strain rates
    mat_c0 = _make_law04(a=A, b=0.0, c=0.0, eps0=1.0)
    deps = np.array([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]])
    sig_c0, _, _ = law04_hyd_jcook.solid_update(mat_c0, np.zeros((1, 6)), deps, np.zeros(1), dt=1e-8)  # 1e6 s^-1
    p_c0 = -(sig_c0[0, 0] + sig_c0[0, 1] + sig_c0[0, 2]) / 3.0
    s_c0 = sig_c0[0].copy()
    s_c0[:3] += p_c0
    svm_c0 = math.sqrt(3.0 * (0.5 * np.sum(s_c0[:3]**2) + np.sum(s_c0[3:]**2)))
    np.testing.assert_allclose(svm_c0, A, rtol=1e-4)

    # 2. EPD <= EPDR -> C_E = 1.0
    C = 0.05
    EPDR = 100.0
    mat_rate = _make_law04(a=A, b=0.0, c=C, eps0=EPDR)
    dt_slow = 1e-3  # EPD = 0.01 / 1e-3 = 10.0 <= 100.0
    sig_slow, _, _ = law04_hyd_jcook.solid_update(mat_rate, np.zeros((1, 6)), deps, np.zeros(1), dt=dt_slow)
    p_slow = -(sig_slow[0, 0] + sig_slow[0, 1] + sig_slow[0, 2]) / 3.0
    s_slow = sig_slow[0].copy()
    s_slow[:3] += p_slow
    svm_slow = math.sqrt(3.0 * (0.5 * np.sum(s_slow[:3]**2) + np.sum(s_slow[3:]**2)))
    np.testing.assert_allclose(svm_slow, A, rtol=1e-4)

    # 3. EPD > EPDR -> C_E = 1.0 + C * ln(EPD / EPDR)
    dt_fast = 1e-5  # EPD = 0.01 / 1e-5 = 1000.0 > 100.0
    ce_expected = 1.0 + C * math.log(1000.0 / EPDR)
    sig_fast, _, _ = law04_hyd_jcook.solid_update(mat_rate, np.zeros((1, 6)), deps, np.zeros(1), dt=dt_fast)
    p_fast = -(sig_fast[0, 0] + sig_fast[0, 1] + sig_fast[0, 2]) / 3.0
    s_fast = sig_fast[0].copy()
    s_fast[:3] += p_fast
    svm_fast = math.sqrt(3.0 * (0.5 * np.sum(s_fast[:3]**2) + np.sum(s_fast[3:]**2)))
    np.testing.assert_allclose(svm_fast, A * ce_expected, rtol=1e-4)


def test_audit_check4_thermal_softening_melting_tmax():
    """Audit check 4:
    TSTAR = (T - T0) / (TMELT - T0).
    If T >= TMELT, does deviatoric stress completely vanish (sigma_dev = 0, QH = 0)?
    If T > TMAX, is CMX = 1 used?
    """
    A = 400.0
    T0 = 300.0
    Tmelt = 1500.0
    Tmax = 900.0
    m = 1.8
    mat = _make_law04(a=A, b=0.0, c=0.0, t0=T0, tmelt=Tmelt, tmax=Tmax, m=m)
    deps = np.array([[0.01, -0.005, -0.005, 0.005, 0.0, 0.0]])

    # Case A: T < T0 -> C_T = 1.0
    extra_cold = {"temp": np.array([250.0])}
    sig_cold, _, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.zeros(1), dt=1e-5, extra=extra_cold)
    p_cold = -(sig_cold[0, 0] + sig_cold[0, 1] + sig_cold[0, 2]) / 3.0
    s_cold = sig_cold[0].copy()
    s_cold[:3] += p_cold
    svm_cold = math.sqrt(3.0 * (0.5 * np.sum(s_cold[:3]**2) + np.sum(s_cold[3:]**2)))
    np.testing.assert_allclose(svm_cold, A, rtol=1e-4)

    # Case B: T0 < T <= Tmax -> C_T = 1.0 - Tstar^m
    T_mid = 600.0
    tstar_mid = (T_mid - T0) / (Tmelt - T0)
    ct_mid = 1.0 - (tstar_mid ** m)
    extra_mid = {"temp": np.array([T_mid])}
    sig_mid, _, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.zeros(1), dt=1e-5, extra=extra_mid)
    p_mid = -(sig_mid[0, 0] + sig_mid[0, 1] + sig_mid[0, 2]) / 3.0
    s_mid = sig_mid[0].copy()
    s_mid[:3] += p_mid
    svm_mid = math.sqrt(3.0 * (0.5 * np.sum(s_mid[:3]**2) + np.sum(s_mid[3:]**2)))
    np.testing.assert_allclose(svm_mid, A * ct_mid, rtol=1e-4)

    # Case C: T > Tmax -> CMX = 1.0 used (m4law.F line 136)
    T_high = 1200.0
    tstar_high = (T_high - T0) / (Tmelt - T0)
    ct_high = 1.0 - (tstar_high ** 1.0)  # CMX forced to 1.0
    extra_high = {"temp": np.array([T_high])}
    sig_high, _, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.zeros(1), dt=1e-5, extra=extra_high)
    p_high = -(sig_high[0, 0] + sig_high[0, 1] + sig_high[0, 2]) / 3.0
    s_high = sig_high[0].copy()
    s_high[:3] += p_high
    svm_high = math.sqrt(3.0 * (0.5 * np.sum(s_high[:3]**2) + np.sum(s_high[3:]**2)))
    np.testing.assert_allclose(svm_high, A * ct_high, rtol=1e-4)

    # Case D: T >= Tmelt -> Deviatoric stresses completely vanish (sigma_dev = 0, QH = 0)
    extra_melt = {"temp": np.array([1600.0])}
    sig_melt, epsp_melt, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.zeros(1), dt=1e-5, extra=extra_melt)
    p_melt = -(sig_melt[0, 0] + sig_melt[0, 1] + sig_melt[0, 2]) / 3.0
    s_melt_dev = sig_melt[0, :3] + p_melt
    np.testing.assert_allclose(s_melt_dev, 0.0, atol=1e-12)
    np.testing.assert_allclose(sig_melt[0, 3:], 0.0, atol=1e-12)
    assert epsp_melt[0] == 0.0  # DPLA = 0 when melted


def test_audit_check5_taylor_quinney_heating():
    """Audit check 5:
    Plastic work heating (Taylor-Quinney):
    If rho * C_p > 0, is Delta T = sigma_y * Delta eps_p / (rho * C_p) added and tracked in extra["temp"]?
    """
    A = 250.0
    rho_cp = 4.0e6
    T0 = 295.0
    mat = _make_law04(a=A, b=0.0, c=0.0, rho_cp=rho_cp, t0=T0)

    deps = np.array([[0.008, 0.0, 0.0, 0.0, 0.0, 0.0]])
    extra = {"temp": np.array([T0])}

    _, epsp_new, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.zeros(1), dt=1e-5, extra=extra)

    dpla = epsp_new[0]
    assert dpla > 0.0
    expected_delta_T = A * dpla / rho_cp
    expected_T = T0 + expected_delta_T

    np.testing.assert_allclose(extra["temp"][0], expected_T, rtol=1e-8)


def test_audit_check6_pressure_cutoff():
    """Audit check 6:
    Pressure cutoff: Is P >= Pmin enforced?
    Verify tensile pressure is clamped at Pmin (a negative cutoff), while compressive pressure is unclamped.
    """
    pmin = -200.0
    mat = _make_law04(e=210000.0, nu=0.3, a=500.0, pmin=pmin)
    K = mat.params["K"]

    # 1. Large tensile volumetric strain: P = -K * tr(deps) << pmin -> clamped to pmin
    deps_tensile = np.array([[0.01, 0.01, 0.01, 0.0, 0.0, 0.0]])
    sig_tensile, _, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps_tensile, np.zeros(1))
    p_tensile = -(sig_tensile[0, 0] + sig_tensile[0, 1] + sig_tensile[0, 2]) / 3.0
    np.testing.assert_allclose(p_tensile, pmin, rtol=1e-8)
    np.testing.assert_allclose(sig_tensile[0, :3], -pmin, rtol=1e-8)  # sigma = -P = +200

    # 2. Large compressive volumetric strain: P = -K * tr(deps) >> 0 -> UNCLAMPED
    deps_compressive = np.array([[-0.01, -0.01, -0.01, 0.0, 0.0, 0.0]])
    sig_compressive, _, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps_compressive, np.zeros(1))
    p_compressive = -(sig_compressive[0, 0] + sig_compressive[0, 1] + sig_compressive[0, 2]) / 3.0
    expected_p_comp = 3.0 * K * 0.01
    np.testing.assert_allclose(p_compressive, expected_p_comp, rtol=1e-8)
    np.testing.assert_allclose(sig_compressive[0, :3], -expected_p_comp, rtol=1e-8)


def test_audit_check7_sound_speed():
    """Audit check 7:
    Sound speed: Is c = sqrt((K + 4/3*G)/rho0) correct?
    """
    mat = _make_law04(e=205000.0, nu=0.28, rho0=7.8e-3)
    G = mat.params["G"]
    K = mat.params["K"]
    expected_c = math.sqrt((K + (4.0 / 3.0) * G) / 7.8e-3)

    _, _, c = law04_hyd_jcook.solid_update(mat, np.zeros((3, 6)), np.zeros((3, 6)), np.zeros(3))
    assert c.shape == (3,)
    np.testing.assert_allclose(c, expected_c, rtol=1e-12)


def test_audit_check8_ipla_radial_return_branches():
    """Audit check 8:
    Verify IPLA=0, 1, 2 radial return formulations from m4law.F lines 177-219.
    """
    mat0 = _make_law04(a=200.0, b=400.0, n=0.5, ipla=0)
    mat1 = _make_law04(a=200.0, b=400.0, n=0.5, ipla=1)
    mat2 = _make_law04(a=200.0, b=400.0, n=0.5, ipla=2)

    deps = np.array([[0.005, 0.0, 0.0, 0.0, 0.0, 0.0]])
    epsp_init = np.array([0.01])

    sig0, ep0, _ = law04_hyd_jcook.solid_update(mat0, np.zeros((1, 6)), deps, epsp_init.copy(), dt=1e-5)
    sig1, ep1, _ = law04_hyd_jcook.solid_update(mat1, np.zeros((1, 6)), deps, epsp_init.copy(), dt=1e-5)
    sig2, ep2, _ = law04_hyd_jcook.solid_update(mat2, np.zeros((1, 6)), deps, epsp_init.copy(), dt=1e-5)

    # In IPLA=2, denominator is 3G instead of 3G + QH, so plastic increment is slightly larger
    assert ep2[0] > ep0[0]
    # In IPLA=1, actual yield stress updates by DPLA*QH and rescales
    assert ep1[0] > 0.0


# ============================================================================
# Wave 2 Verifier 2: Mathematical Audit of Algorithmic Consistent Tangent
# ============================================================================

def test_audit_wave2_verifier2_tangent_box73_exact_algebra():
    """Audit check 1:
    Verify tangent matches Simo & Hughes Box 7.3 J2 radial return:
    D_alg = C_el - a * (C_el - K 1 (x) 1) + b * (N (x) N)
    with a = 3G * dep / q_tr, b = 6G^2 * (dep / q_tr - 1 / (3G + H)).
    """
    mat = _make_law04(a=200.0, b=400.0, n=0.5, sig_max=500.0, eps_max=0.5, c=0.0, pmin=-1e20)
    G = mat.params["G"]
    K = mat.params["K"]
    deps = np.array([[0.005, -0.001, -0.001, 0.001, 0.0, 0.0]])
    epsp_init = np.array([0.01])
    s0, ep0, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, epsp_init.copy(), dt=1e-5, extra={"ipla": 1})
    dep = ep0[0] - epsp_init[0]

    D_calc = law04_hyd_jcook.consistent_solid_tangent(mat, s0, epsp_init, np.array([dep]), extra={"ipla": 1})[0]

    # Manual Box 7.3 derivation
    lam = K - 2.0 * G / 3.0
    Cel = np.zeros((6, 6))
    Cel[:3, :3] = lam
    np.fill_diagonal(Cel[:3, :3], lam + 2.0 * G)
    np.fill_diagonal(Cel[3:, 3:], G)

    ee = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    KeeT = K * np.outer(ee, ee)
    C_minus_vol = Cel - KeeT

    p = (s0[0, 0] + s0[0, 1] + s0[0, 2]) / 3.0
    s_dev = s0[0].copy()
    s_dev[:3] -= p
    snorm = math.sqrt(s_dev[0]**2 + s_dev[1]**2 + s_dev[2]**2 + 2.0 * (s_dev[3]**2 + s_dev[4]**2 + s_dev[5]**2))
    Nv = s_dev / snorm
    q = math.sqrt(1.5) * snorm
    q_tr = q + 3.0 * G * dep
    H = 400.0 * 0.5 * (0.01 ** (0.5 - 1.0))
    a = 3.0 * G * dep / q_tr
    b = 6.0 * G * G * (dep / q_tr - 1.0 / (3.0 * G + H))

    D_box73 = Cel - a * C_minus_vol + b * np.outer(Nv, Nv)

    np.testing.assert_allclose(D_calc, D_box73, rtol=1e-12)


def test_audit_wave2_verifier2_melted_state():
    """Audit check 2:
    In melted state (T >= Tmelt), shear strength is 0, so deviatoric tangent vanishes,
    leaving only the bulk volumetric modulus K 1 (x) 1.
    """
    mat = _make_law04(a=200.0, b=400.0, tmelt=1500.0)
    K = mat.params["K"]
    deps = np.array([[0.005, -0.001, -0.001, 0.001, 0.0, 0.0]])
    extra_melt = {"temp": np.array([1600.0])}

    s_melt, ep_melt, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.zeros(1), dt=1e-5, extra=extra_melt)
    # Check deviator is 0
    p = np.mean(s_melt[0, :3])
    np.testing.assert_allclose(s_melt[0, :3] - p, 0.0, atol=1e-12)
    np.testing.assert_allclose(s_melt[0, 3:], 0.0, atol=1e-12)

    D_melt = law04_hyd_jcook.consistent_solid_tangent(mat, s_melt, ep_melt, ep_melt, extra=extra_melt)[0]
    ee = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    KeeT = K * np.outer(ee, ee)

    np.testing.assert_allclose(D_melt, KeeT, atol=1e-10)


def test_audit_wave2_verifier2_symmetry_and_positive_semidefinite():
    """Audit check 3:
    Check symmetry and positive-semidefiniteness / positive eigenvalues across states.
    Eigenvalues are:
    lambda_1 = 3K (volumetric)
    lambda_2 = 2G * H / (3G + H) >= 0 (flow direction N)
    lambda_{3..6} = 2G * sigma_y / q_tr > 0 (deviatoric orthogonal to N)
    """
    mat = _make_law04(a=200.0, b=400.0, n=0.5, tmelt=1800.0)
    K = mat.params["K"]
    deps = np.array([[0.005, -0.001, -0.001, 0.001, 0.0, 0.0]])

    # 1. Plastic state
    s0, ep0, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps, np.array([0.01]), dt=1e-5, extra={"ipla": 1})
    D_pl = law04_hyd_jcook.consistent_solid_tangent(mat, s0, np.array([0.01]), ep0 - 0.01, extra={"ipla": 1})[0]

    # Symmetry
    np.testing.assert_allclose(D_pl, D_pl.T, atol=1e-12)
    # Positive eigenvalues
    eigs_pl = np.linalg.eigvalsh(D_pl)
    assert np.all(eigs_pl >= -1e-8)
    # Analytical eigenvalue check
    assert np.isclose(eigs_pl[-1], 3.0 * K, rtol=1e-6)

    # 2. Melted state
    extra_melt = {"temp": np.array([1900.0])}
    D_melt = law04_hyd_jcook.consistent_solid_tangent(mat, s0, ep0, ep0, extra=extra_melt)[0]
    np.testing.assert_allclose(D_melt, D_melt.T, atol=1e-12)
    eigs_melt = np.linalg.eigvalsh(D_melt)
    assert np.all(eigs_melt >= -1e-8)
    assert np.isclose(eigs_melt[-1], 3.0 * K, rtol=1e-6)
    assert np.allclose(eigs_melt[:5], 0.0, atol=1e-8)


def test_audit_wave2_verifier2_directional_derivative_consistency_all_states():
    """Audit check 4:
    Verify finite-difference directional derivative consistency across multiple states:
    - Elastic state
    - Yielding with power-law hardening (IPLA=1)
    - Rate sensitivity active (frozen CE)
    - Thermal softening active (frozen CT)
    - Saturated hardening (sig >= sig_max -> H = 0)
    - Broken state (epsp > eps_max -> H = 0)
    - Melted state (T >= Tmelt)
    - Unhardened return (IPLA=0, b=0)
    """
    delta = np.array([1.0, -0.3, -0.3, 0.5, 0.0, 0.0])
    eps = 1e-7

    def _verify_state(mat, deps0, epsp0, extra=None):
        s0, ep0, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps0.copy(), epsp0.copy(), dt=1e-5, extra=extra)
        dep_incr = ep0 - epsp0
        D = law04_hyd_jcook.consistent_solid_tangent(mat, s0, epsp0, dep_incr, extra=extra)[0]

        sp, _, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps0 + eps * delta, epsp0.copy(), dt=1e-5, extra=extra)
        sm, _, _ = law04_hyd_jcook.solid_update(mat, np.zeros((1, 6)), deps0 - eps * delta, epsp0.copy(), dt=1e-5, extra=extra)
        num_diff = (sp[0] - sm[0]) / (2.0 * eps)
        tan_diff = D @ delta
        rel_err = np.linalg.norm(num_diff - tan_diff) / np.linalg.norm(tan_diff)
        assert rel_err < 1e-5, f"rel_err={rel_err:.4e} exceeds 1e-5"

    deps_plas = np.array([[0.005, -0.001, -0.001, 0.001, 0.0, 0.0]])
    deps_elas = np.array([[1e-5, 0.0, 0.0, 0.0, 0.0, 0.0]])

    # 1. Elastic state
    mat_el = _make_law04(a=500.0, b=0.0, pmin=-1e20)
    _verify_state(mat_el, deps_elas, np.zeros(1))

    # 2. Yielding with power-law hardening (IPLA=1)
    mat_pl = _make_law04(a=200.0, b=400.0, n=0.5, c=0.0, pmin=-1e20)
    _verify_state(mat_pl, deps_plas, np.array([0.01]), extra={"ipla": 1})

    # 3. Rate sensitivity active (frozen CE)
    mat_rate = _make_law04(a=200.0, b=400.0, n=0.5, c=0.05, eps0=1.0, pmin=-1e20)
    _verify_state(mat_rate, deps_plas, np.array([0.01]), extra={"ipla": 1, "ce": 1.25})

    # 4. Thermal softening active (T = 1000K, T0=300, Tmelt=1800)
    mat_therm = _make_law04(a=200.0, b=400.0, n=0.5, c=0.0, t0=300.0, tmelt=1800.0, pmin=-1e20)
    _verify_state(mat_therm, deps_plas, np.array([0.01]), extra={"ipla": 1, "temp": np.array([1000.0])})

    # 5. Saturated hardening (sig >= sig_max -> H = 0)
    mat_sat = _make_law04(a=200.0, b=400.0, n=0.5, sig_max=200.0, c=0.0, pmin=-1e20)
    _verify_state(mat_sat, deps_plas, np.array([0.01]), extra={"ipla": 1})

    # 6. Broken state (epsp > eps_max -> H = 0)
    mat_brk = _make_law04(a=200.0, b=400.0, n=0.5, eps_max=0.01, c=0.0, pmin=-1e20)
    _verify_state(mat_brk, deps_plas, np.array([0.02]), extra={"ipla": 1})

    # 7. Melted state (T >= Tmelt)
    mat_melt = _make_law04(a=200.0, b=400.0, n=0.5, tmelt=1800.0, pmin=-1e20)
    _verify_state(mat_melt, deps_plas, np.zeros(1), extra={"temp": np.array([1900.0])})

    # 8. Unhardened return (IPLA=0, b=0)
    mat_b0 = _make_law04(a=200.0, b=0.0, c=0.0, pmin=-1e20)
    _verify_state(mat_b0, deps_plas, np.zeros(1), extra={"ipla": 0})



