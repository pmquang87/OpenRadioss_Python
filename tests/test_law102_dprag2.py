"""Comprehensive unit tests for /MAT/LAW102 Extended Drucker-Prager with Cap Plasticity.

Verifies:
  - Shear yield surface: F_s = q - p * tan(phi) - c
  - Cap yield surface: F_c = (p - P_a)^2 / R^2 + (q / (k * P_a))^2 - 1 = 0
  - Elastic behavior under the cap (p > P_a, F_c <= 0)
  - Pure hydrostatic cap compaction (q = 0, p -> P_a + R)
  - Hardening cap evolution of P_a with plastic volumetric strain epspv:
      * Exponential hardening: P_a = P_a0 * exp(W * epspv)
      * Power-law hardening: P_a = P_a0 * (1 + W * epspv^D)
  - Backward compatibility with standard non-cap formulation (r_cap = 0)
  - Solver interface, history extra dictionary (pa, epspv), and tangent aliases
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law102_dprag2 import (
    DPrag2Params,
    build_law102,
    init_history,
    solid_update,
    solid_update_single,
    solid_update_array,
    sound_speed,
    solid_tangent,
    tangent,
    consistent_solid_tangent,
    extra_shapes,
)


def test_shear_yield_surface():
    """Verify shear yield surface F_s = q - p*tan(phi) - c for p <= P_a."""
    e = 20000.0
    nu = 0.2
    c_cohesion = 10.0
    phi_deg = 30.0
    phi_rad = math.radians(phi_deg)
    tan_phi = math.tan(phi_rad)
    pa = 60.0
    r_cap = 20.0

    p = build_law102(
        rho0=2.0e-6,
        e=e,
        nu=nu,
        c=c_cohesion,
        phi=phi_deg,
        iform=2,
        r_cap=r_cap,
        pa=pa,
        w_cap=0.0,
    )

    # Initial hydrostatic pressure p0 = 20.0 < pa (compressive)
    sig0 = np.array([-20.0, -20.0, -20.0, 0.0, 0.0, 0.0])
    # Apply pure shear strain to trigger yielding on the shear envelope
    deps = np.array([0.0, 0.0, 0.0, 0.015, 0.0, 0.0])

    hist0 = np.array([0.0, 20.0, pa, 0.0])
    sig_new, hist_new, ssp = _solid_update_single_core_run(p, deps, sig0, hist0)

    # Pressure should remain p0 = 20.0 (deviatoric strain has zero trace)
    p_final = -(sig_new[0] + sig_new[1] + sig_new[2]) / 3.0
    assert pytest.approx(p_final, abs=1e-5) == 20.0

    # Expected yield stress on shear envelope: q = p * tan(phi) + c
    q_expected = p_final * tan_phi + c_cohesion

    # Compute q from sig_new: q = sqrt(3 * J2)
    s = sig_new.copy()
    s[:3] += p_final
    j2 = 0.5 * (s[0] ** 2 + s[1] ** 2 + s[2] ** 2) + s[3] ** 2 + s[4] ** 2 + s[5] ** 2
    q_final = math.sqrt(3.0 * j2)

    assert pytest.approx(q_final, rel=1e-5) == q_expected
    # Plastic shear strain accumulated
    assert hist_new[0] > 0.0
    # Volumetric plastic strain must remain 0 on the shear surface
    assert hist_new[3] == 0.0
    # Cap position must be unchanged
    assert hist_new[2] == pa


def test_cap_yield_surface():
    """Verify return projection onto elliptic cap: (p - P_a)^2 / R^2 + (q / (k*P_a))^2 = 1."""
    e = 30000.0
    nu = 0.25
    pa = 40.0
    r_cap = 15.0
    k_cap = 0.5
    b = k_cap * pa  # 20.0

    p = build_law102(
        rho0=2.5e-6,
        e=e,
        nu=nu,
        c=5.0,
        phi=25.0,
        r_cap=r_cap,
        pa=pa,
        k_cap=k_cap,
        w_cap=0.0,  # No hardening for geometric surface check
    )

    # Start at p0 = 40.0 (at cap threshold)
    sig0 = np.array([-40.0, -40.0, -40.0, 0.0, 0.0, 0.0])
    # Apply combined compressive volumetric and shear strain
    # Volumetric strain rate pushes p_trial well past pa:
    deps = np.array([-0.001, -0.001, -0.001, 0.003, 0.0, 0.0])

    hist0 = np.array([0.0, 40.0, pa, 0.0])
    sig_new, hist_new, ssp = _solid_update_single_core_run(p, deps, sig0, hist0)

    p_final = -(sig_new[0] + sig_new[1] + sig_new[2]) / 3.0
    s = sig_new.copy()
    s[:3] += p_final
    j2 = 0.5 * (s[0] ** 2 + s[1] ** 2 + s[2] ** 2) + s[3] ** 2 + s[4] ** 2 + s[5] ** 2
    q_final = math.sqrt(3.0 * j2)

    # Must satisfy cap equation: ((p - Pa) / R)^2 + (q / b)^2 == 1.0
    cap_val = ((p_final - pa) / r_cap) ** 2 + (q_final / b) ** 2
    assert pytest.approx(cap_val, abs=1e-5) == 1.0

    # Must have produced plastic volumetric strain
    assert hist_new[3] > 0.0
    # Volumetric plastic strain equals (p_trial - p_final) / bulk
    bulk = p.bulk
    scrt = sum(deps[:3]) / 3.0
    p_trial = 40.0 - 3.0 * bulk * scrt
    depspv_expected = (p_trial - p_final) / bulk
    assert pytest.approx(hist_new[3], rel=1e-5) == depspv_expected


def test_elastic_under_cap():
    """Verify trial state with p > P_a but inside the cap remains purely elastic."""
    e = 30000.0
    nu = 0.25
    pa = 40.0
    r_cap = 20.0
    k_cap = 0.6

    p = build_law102(
        rho0=2.5e-6,
        e=e,
        nu=nu,
        c=5.0,
        phi=25.0,
        r_cap=r_cap,
        pa=pa,
        k_cap=k_cap,
    )

    # Initial stress inside cap: p = 45.0 (pa < p < pa + R = 60.0)
    sig0 = np.array([-45.0, -45.0, -45.0, 0.0, 0.0, 0.0])
    # Very small strain step, keeping state inside the ellipse
    deps = np.array([1.0e-5, -0.5e-5, -0.5e-5, 0.0, 0.0, 0.0])

    hist0 = np.array([0.0, 45.0, pa, 0.0])
    sig_new, hist_new, _ = _solid_update_single_core_run(p, deps, sig0, hist0)

    # Elastic: no plastic strain accumulation
    assert hist_new[0] == 0.0
    assert hist_new[3] == 0.0
    assert hist_new[2] == pa


def test_pure_hydrostatic_cap_compaction():
    """Verify that pure hydrostatic compaction is capped at P = P_a + R."""
    e = 30000.0
    nu = 0.2
    pa = 50.0
    r_cap = 20.0
    k_cap = 0.5

    p = build_law102(
        rho0=2.0e-6,
        e=e,
        nu=nu,
        c=10.0,
        phi=30.0,
        r_cap=r_cap,
        pa=pa,
        k_cap=k_cap,
        w_cap=0.0,
    )

    # Start at zero stress, apply heavy hydrostatic compression
    # Trial pressure = -3 * bulk * (-0.005) = bulk * 0.015 >> pa + r_cap = 70.0
    sig0 = np.zeros(6)
    deps = np.array([-0.005, -0.005, -0.005, 0.0, 0.0, 0.0])

    hist0 = np.array([0.0, 0.0, pa, 0.0])
    sig_new, hist_new, _ = _solid_update_single_core_run(p, deps, sig0, hist0)

    p_final = -(sig_new[0] + sig_new[1] + sig_new[2]) / 3.0
    # Under pure hydrostatic loading (q = 0), the cap ellipse apex is at p = pa + r_cap
    assert pytest.approx(p_final, rel=1e-5) == pa + r_cap
    # All shear stresses and deviatoric components must be zero
    s = sig_new.copy()
    s[:3] += p_final
    np.testing.assert_allclose(s, np.zeros(6), atol=1e-10)

    # Volumetric plastic strain
    bulk = p.bulk
    p_trial = -bulk * sum(deps[:3])
    expected_depspv = (p_trial - (pa + r_cap)) / bulk
    assert pytest.approx(hist_new[3], rel=1e-5) == expected_depspv


def test_cap_hardening_exponential():
    """Verify exponential cap evolution: P_a = P_a0 * exp(W * epspv)."""
    e = 25000.0
    nu = 0.2
    pa0 = 30.0
    r_cap = 15.0
    k_cap = 0.5
    w_cap = 40.0
    d_cap = 0.0  # Exponential hardening

    p = build_law102(
        rho0=2.0e-6,
        e=e,
        nu=nu,
        c=8.0,
        phi=28.0,
        r_cap=r_cap,
        pa=pa0,
        k_cap=k_cap,
        w_cap=w_cap,
        d_cap=d_cap,
    )

    sig0 = np.array([-pa0, -pa0, -pa0, 0.0, 0.0, 0.0])
    deps = np.array([-0.002, -0.002, -0.002, 0.0, 0.0, 0.0])

    hist0 = np.array([0.0, pa0, pa0, 0.0])
    sig_new, hist_new, _ = _solid_update_single_core_run(p, deps, sig0, hist0)

    epspv_new = hist_new[3]
    assert epspv_new > 0.0

    pa_expected = pa0 * math.exp(w_cap * epspv_new)
    assert pytest.approx(hist_new[2], rel=1e-6) == pa_expected
    assert hist_new[2] > pa0


def test_cap_hardening_power_law():
    """Verify power-law cap evolution: P_a = P_a0 * (1 + W * epspv^D)."""
    e = 25000.0
    nu = 0.2
    pa0 = 25.0
    r_cap = 12.0
    k_cap = 0.45
    w_cap = 50.0
    d_cap = 1.5

    p = build_law102(
        rho0=2.0e-6,
        e=e,
        nu=nu,
        c=6.0,
        phi=25.0,
        r_cap=r_cap,
        pa=pa0,
        k_cap=k_cap,
        w_cap=w_cap,
        d_cap=d_cap,
    )

    sig0 = np.array([-pa0, -pa0, -pa0, 0.0, 0.0, 0.0])
    deps = np.array([-0.002, -0.002, -0.002, 0.0, 0.0, 0.0])

    hist0 = np.array([0.0, pa0, pa0, 0.0])
    sig_new, hist_new, _ = _solid_update_single_core_run(p, deps, sig0, hist0)

    epspv_new = hist_new[3]
    assert epspv_new > 0.0

    pa_expected = pa0 * (1.0 + w_cap * (epspv_new ** d_cap))
    assert pytest.approx(hist_new[2], rel=1e-6) == pa_expected
    assert hist_new[2] > pa0


def test_non_cap_backward_compatibility():
    """Verify that when r_cap = 0 or pa = 0, standard Fortran sigeps102.F formulation runs."""
    p_std = build_law102(
        rho0=2.0e-6,
        e=10000.0,
        nu=0.2,
        iform=2,
        c=5.0,
        phi=25.0,
        amax=50.0,
        pmin=-2.0,
    )
    assert p_std.r_cap == 0.0
    assert p_std.pa == 0.0

    sig0 = np.zeros(6)
    deps = np.array([0.0, 0.0, 0.0, 0.02, 0.0, 0.0])

    # Using standard 2-variable history
    hist = init_history(1, params=p_std)
    assert hist.shape == (1, 2)

    sig_new, epsp_new = solid_update_single(p_std, sig0, deps, epsp=0.0)
    assert epsp_new > 0.0

    # Verify tangent aliases and shape
    c_tan = tangent(p_std)
    assert c_tan.shape == (6, 6)
    c_cons = consistent_solid_tangent(p_std)
    np.testing.assert_allclose(c_tan, c_cons)


def test_solid_update_interface_with_extra():
    """Verify solid_update with extra dictionary tracking cap history variables pa and epspv."""
    p = build_law102(
        rho0=2.0e-6,
        e=20000.0,
        nu=0.25,
        c=10.0,
        phi=30.0,
        r_cap=15.0,
        pa=50.0,
        k_cap=0.5,
        w_cap=30.0,
    )

    extra = {}
    shapes = extra_shapes(p)
    assert shapes["uvar102"] == (4,)

    sig = np.array([-50.0, -50.0, -50.0, 0.0, 0.0, 0.0])
    deps = np.array([-0.002, -0.002, -0.002, 0.005, 0.0, 0.0])

    sig_out, epsp_out, c_out = solid_update(p, sig, deps, extra=extra)

    assert "uvar102" in extra
    assert extra["uvar102"].shape == (1, 4)
    assert "pa" in extra
    assert "epspv" in extra
    assert extra["pa"] > 50.0
    assert extra["epspv"] > 0.0
    assert c_out > 0.0


def _solid_update_single_core_run(params, deps, sig_old, history):
    from pyradioss.materials.law102_dprag2 import _solid_update_single_core
    return _solid_update_single_core(params, deps, sig_old, history)
