"""Tests for LAW10 (/MAT/LAW10, /MAT/SOIL, /MAT/DPRAG).

Validates:
1. Parameter extraction, defaults, auto-estimation (hm_read_mat10.F).
2. Vectorized 3D solid update (m10law.F, compaction.F90).
3. Compaction equation of state loading and unloading (iform=1, iform=2).
4. Drucker-Prager yield surface, von Mises cap, tension cutoff, pressure root closure.
5. Radial return plastic strain and deviatoric projection.
6. Longitudinal sound speed calculation.
7. Consistent algorithmic elastoplastic solid tangent and finite difference check.
8. Shell update raises NotImplementedError.
9. Physics registry mapping for all LAW10 aliases.
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from pyradioss.materials import law10_soil
from pyradioss.materials.law10_soil import (
    build_law10,
    _ensure_params,
    solid_update,
    shell_update,
    sound_speed,
    consistent_solid_tangent,
    solid_tangent,
    tangent,
    needs_defgrad,
    extra_shapes,
)
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY


# =============================================================================
# 1. Parameter Extraction & Validation
# =============================================================================

def test_law10_param_extraction_and_defaults():
    rec = {
        "id": 1,
        "MAT_RHO": 2000.0,
        "MAT_E": 3.0e7,
        "MAT_NU": 0.25,
        "MAT_A0": 1.0e6,
        "MAT_A1": 0.5,
        "MAT_A2": 0.0,
    }
    mat = build_law10(rec)
    p = mat.params
    assert mat.law == 10
    assert mat.rho0 == 2000.0

    # Moduli
    assert p["E"] == 3.0e7
    assert p["nu"] == 0.25
    expected_g = 3.0e7 / (2.0 * 1.25)
    expected_k = 3.0e7 / (3.0 * (1.0 - 2.0 * 0.25))
    assert math.isclose(p["G"], expected_g)
    assert math.isclose(p["K"], expected_k)

    # Defaults
    assert p["Amax"] == 1e20
    assert p["pmin"] == -1e30
    assert p["c1"] == expected_k
    assert p["bunl"] == expected_k
    assert p["mue_max"] == 1e20
    assert p["iform"] == 1
    # Linear root: pstar = -A0/A1
    assert math.isclose(p["pstar"], -1.0e6 / 0.5)


def test_law10_validation_errors():
    # Negative density
    with pytest.raises(ValueError, match="Density rho0 must be > 0"):
        build_law10({"MAT_RHO": -100.0, "MAT_E": 1e7, "MAT_NU": 0.2})

    # Zero E
    with pytest.raises(ValueError, match="Young's modulus E must be > 0"):
        build_law10({"MAT_RHO": 2000.0, "MAT_E": 0.0, "MAT_NU": 0.2})

    # Invalid nu >= 0.5
    with pytest.raises(ValueError, match="Poisson's ratio nu must be in"):
        build_law10({"MAT_RHO": 2000.0, "MAT_E": 1e7, "MAT_NU": 0.5})

    # Invalid nu < 0
    with pytest.raises(ValueError, match="Poisson's ratio nu must be in"):
        build_law10({"MAT_RHO": 2000.0, "MAT_E": 1e7, "MAT_NU": -0.1})


def test_law10_pstar_computation():
    # Linear: a2 == 0, a1 != 0
    p1 = _ensure_params({"rho0": 1000.0, "E": 1e6, "nu": 0.2, "A0": 500.0, "A1": 2.0, "A2": 0.0})
    assert math.isclose(p1["pstar"], -250.0)

    # Quadratic with real root: a2 != 0, delta >= 0
    # a0 = 100, a1 = 20, a2 = 0.5 -> delta = 400 - 4*0.5*100 = 200 >= 0
    p2 = _ensure_params({"rho0": 1000.0, "E": 1e6, "nu": 0.2, "A0": 100.0, "A1": 20.0, "A2": 0.5})
    expected_pstar = (-20.0 + math.sqrt(200.0)) / (2.0 * 0.5)
    assert math.isclose(p2["pstar"], expected_pstar)

    # Quadratic with no real intersection: delta < 0
    # a0 = 1000, a1 = 10, a2 = 1.0 -> delta = 100 - 4000 < 0
    p3 = _ensure_params({"rho0": 1000.0, "E": 1e6, "nu": 0.2, "A0": 1000.0, "A1": 10.0, "A2": 1.0})
    expected_pstar_ext = -10.0 / (2.0 * 1.0)
    assert math.isclose(p3["pstar"], expected_pstar_ext)

    # Both 0: a1 == 0, a2 == 0
    p4 = _ensure_params({"rho0": 1000.0, "E": 1e6, "nu": 0.2, "A0": 100.0, "A1": 0.0, "A2": 0.0})
    assert p4["pstar"] == -1e30


def test_law10_auto_estimation():
    # 1. Missing mue_max estimated from bunl (c3 == 0, c2 != 0)
    # XMUMX = (BUNL - C1)/(2*C2)
    p1 = _ensure_params({
        "rho0": 1000.0, "E": 1e6, "nu": 0.0,
        "c1": 1.0e6, "c2": 5.0e5, "c3": 0.0,
        "bunl": 2.0e6, "mue_max": 0.0,
    })
    expected_mumx = (2.0e6 - 1.0e6) / (2.0 * 5.0e5)
    assert math.isclose(p1["mue_max"], expected_mumx)

    # 2. Missing mue_max estimated from bunl (c3 != 0)
    # DET = sqrt(c2^2 + 3*c3*(bunl - c1)), XMUMX = (DET - c2)/(3*c3)
    p2 = _ensure_params({
        "rho0": 1000.0, "E": 1e6, "nu": 0.0,
        "c1": 1.0e6, "c2": 2.0e5, "c3": 1.0e5,
        "bunl": 3.0e6, "mue_max": 0.0,
    })
    det = math.sqrt(2.0e5**2 + 3.0 * 1.0e5 * (3.0e6 - 1.0e6))
    expected_mumx_2 = (det - 2.0e5) / (3.0 * 1.0e5)
    assert math.isclose(p2["mue_max"], expected_mumx_2)

    # 3. Missing bunl estimated from mue_max (c3 == 0, c2 != 0)
    # BUNL = C1 + 2*C2*XMUMX
    p3 = _ensure_params({
        "rho0": 1000.0, "E": 1e6, "nu": 0.0,
        "c1": 1.0e6, "c2": 5.0e5, "c3": 0.0,
        "bunl": 0.0, "mue_max": 0.5,
    })
    expected_bunl = 1.0e6 + 2.0 * 5.0e5 * 0.5
    assert math.isclose(p3["bunl"], expected_bunl)

    # 4. Missing bunl estimated from mue_max (c3 != 0)
    # BUNL = C1 + 2*C2*XMUMX + 3*C3^2*XMUMX^2 (hm_read_mat10.F line 256)
    p4 = _ensure_params({
        "rho0": 1000.0, "E": 1e6, "nu": 0.0,
        "c1": 1.0e6, "c2": 2.0e5, "c3": 10.0,
        "bunl": 0.0, "mue_max": 0.2,
    })
    expected_bunl_2 = 1.0e6 + 2.0 * 2.0e5 * 0.2 + 3.0 * (10.0**2) * (0.2**2)
    assert math.isclose(p4["bunl"], expected_bunl_2)


def test_law10_registration():
    for key in ("LAW10", "SOIL", "DPRAG", "DPRAG1", "10"):
        assert key in MAT_PHYSICS_REGISTRY
        fn = MAT_PHYSICS_REGISTRY[key]
        assert callable(fn)


# =============================================================================
# 2. Constitutive Stress Update (solid_update)
# =============================================================================

def test_law10_empty_and_zero_dt():
    mat = build_law10({"MAT_RHO": 1800.0, "MAT_E": 1.0e7, "MAT_NU": 0.3})
    # Empty
    sig_empty = np.empty((0, 6), dtype=float)
    out = solid_update(mat, sig_empty, dt=1e-3)
    assert out.shape == (0, 6)

    # Zero dt
    sig = np.array([[100.0, 50.0, 25.0, 10.0, 5.0, 2.0]], dtype=float)
    out_zero = solid_update(mat, sig, dt=0.0)
    np.testing.assert_allclose(out_zero, sig)


def test_law10_hydrostatic_compaction():
    # Pure volumetric compression: deps = [-0.01, -0.01, -0.01, 0, 0, 0]
    # Compaction strain mu = -tr(deps) = 0.03
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 3.0e7,
        "MAT_NU": 0.25,
        "c0": 1.0e4,
        "c1": 2.0e6,
        "c2": 5.0e6,
        "c3": 1.0e6,
        "A0": 1.0e10,  # Huge yield envelope -> strictly elastic deviator
    })
    sig = np.zeros((1, 6), dtype=float)
    deps = np.array([[-0.01, -0.01, -0.01, 0.0, 0.0, 0.0]], dtype=float)
    extra = {}

    sig_new = solid_update(mat, sig, d_eps=deps, dt=1e-4, extra=extra)

    mu = 0.03
    mu2 = mu * mu
    expected_p = 1.0e4 + 2.0e6 * mu + (5.0e6 + 1.0e6 * mu) * mu2

    # Cauchy stress sigma_ii = -P
    np.testing.assert_allclose(sig_new[0, 0], -expected_p, rtol=1e-5)
    np.testing.assert_allclose(sig_new[0, 1], -expected_p, rtol=1e-5)
    np.testing.assert_allclose(sig_new[0, 2], -expected_p, rtol=1e-5)
    np.testing.assert_allclose(sig_new[0, 3:], 0.0, atol=1e-12)
    assert math.isclose(extra["mu_bak"][0], 0.03)


def test_law10_compaction_unloading_iform1_and_iform2():
    mat1 = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 3.0e7,
        "MAT_NU": 0.25,
        "c0": 0.0,
        "c1": 1.0e6,
        "c2": 2.0e6,
        "c3": 0.0,
        "bunl": 5.0e6,
        "iform": 1,
        "A0": 1.0e12,
    })
    sig = np.zeros((1, 6), dtype=float)
    # Load to mu = 0.05
    extra1 = {}
    deps_load = np.array([[-0.05/3.0, -0.05/3.0, -0.05/3.0, 0.0, 0.0, 0.0]])
    sig_1 = solid_update(mat1, sig, d_eps=deps_load, dt=1e-4, extra=extra1)
    p_bak = 1.0e6 * 0.05 + 2.0e6 * (0.05**2)
    np.testing.assert_allclose(sig_1[0, 0], -p_bak, rtol=1e-5)
    assert math.isclose(extra1["mu_bak"][0], 0.05)

    # Unload to mu = 0.04 (d_mu = -0.01, so deps = +0.01/3 in tension)
    deps_unload = np.array([[+0.01/3.0, +0.01/3.0, +0.01/3.0, 0.0, 0.0, 0.0]])
    sig_2 = solid_update(mat1, sig_1, d_eps=deps_unload, dt=1e-4, extra=extra1)

    # iform = 1: constant unload modulus b_eff = bunl = 5e6
    # p_unl = p_bak - (mu_bak - mu) * bunl = p_bak - (0.05 - 0.04) * 5e6
    expected_p_unl = p_bak - 0.01 * 5.0e6
    np.testing.assert_allclose(sig_2[0, 0], -expected_p_unl, rtol=1e-5)
    # mu_bak remains 0.05
    assert math.isclose(extra1["mu_bak"][0], 0.05)

    # Now test iform = 2: continuous unload modulus
    mat2 = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 3.0e7,
        "MAT_NU": 0.25,
        "c0": 0.0,
        "c1": 1.0e6,
        "c2": 2.0e6,
        "c3": 0.0,
        "bunl": 5.0e6,
        "mue_max": 0.1,
        "iform": 2,
        "A0": 1.0e12,
    })
    extra2 = {}
    sig_3 = solid_update(mat2, sig, d_eps=deps_load, dt=1e-4, extra=extra2)
    sig_4 = solid_update(mat2, sig_3, d_eps=deps_unload, dt=1e-4, extra=extra2)
    # alpha = mu_bak / mue_max = 0.05 / 0.1 = 0.5
    # b_eff = 0.5 * 5e6 + 0.5 * 1e6 = 3e6
    expected_p_unl_2 = p_bak - 0.01 * 3.0e6
    np.testing.assert_allclose(sig_4[0, 0], -expected_p_unl_2, rtol=1e-5)


def test_law10_drucker_prager_elastic_and_plastic():
    # Material with A0 = 1e6 (J2 yield = 1e6, von Mises yield = sqrt(3*J2) = 1732.05)
    # A1 = 0, A2 = 0 (von Mises behavior)
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 1.0e7,
        "MAT_NU": 0.25,
        "A0": 1.0e6,
        "A1": 0.0,
        "A2": 0.0,
        "c0": 0.0,
        "c1": 1.0e7,
    })
    g = 1.0e7 / (2.0 * 1.25)  # 4e6
    sig = np.zeros((1, 6), dtype=float)

    # 1. Pure shear strain within elastic limit
    # dgamma = 1e-4 -> s_xy = G * dgamma = 400.0
    # J2 = s_xy^2 = 1.6e5 <= A0 (1e6) -> strictly elastic
    extra_el = {}
    deps_el = np.array([[0.0, 0.0, 0.0, 1.0e-4, 0.0, 0.0]])
    sig_el = solid_update(mat, sig, d_eps=deps_el, dt=1e-4, extra=extra_el)
    np.testing.assert_allclose(sig_el[0, 3], 400.0)
    assert extra_el["ratio"][0] == 1.0
    assert extra_el["epxe"][0] == 0.0

    # 2. Pure shear strain exceeding yield
    # dgamma = 1e-3 -> s_tr_xy = 4000.0
    # J2_tr = 4000^2 = 1.6e7 > A0 (1e6)
    # ratio = sqrt(1e6 / 1.6e7) = 1000 / 4000 = 0.25
    # s_new_xy = 0.25 * 4000 = 1000.0
    extra_pl = {}
    deps_pl = np.array([[0.0, 0.0, 0.0, 1.0e-3, 0.0, 0.0]])
    sig_pl = solid_update(mat, sig, d_eps=deps_pl, dt=1e-4, extra=extra_pl)
    np.testing.assert_allclose(sig_pl[0, 3], 1000.0, rtol=1e-5)
    np.testing.assert_allclose(extra_pl["ratio"][0], 0.25, rtol=1e-5)

    # Check plastic strain dpla = (1 - ratio)*sqrt(3*J2)/(3*G)
    # = (1 - 0.25) * sqrt(3 * 1.6e7) / (3 * 4e6) = 0.75 * 6928.2 / 1.2e7 = 4.330127e-4
    expected_dpla = (1.0 - 0.25) * math.sqrt(3.0 * 1.6e7) / (3.0 * 4.0e6)
    np.testing.assert_allclose(extra_pl["epxe"][0], expected_dpla, rtol=1e-5)

    # Check J2 on updated stress equals A0
    j2_new = sig_pl[0, 3] ** 2
    np.testing.assert_allclose(j2_new, 1.0e6, rtol=1e-5)


def test_law10_pressure_dependent_yield_envelope():
    # A0 = 1e6, A1 = 2.0, A2 = 0
    # G0(P) = A0 + A1*P
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 1.0e7,
        "MAT_NU": 0.25,
        "A0": 1.0e6,
        "A1": 2.0,
        "A2": 0.0,
        "c0": 0.0,
        "c1": 1.0e7,
    })
    # Apply hydrostatic pressure + shear
    # P = 5.0e5 -> G0 = 1e6 + 2.0 * 5e5 = 2.0e6
    sig = np.zeros((1, 6), dtype=float)
    # Volumetric strain mu = 5e5 / 1e7 = 0.05 -> deps_vol = -0.05
    # Shear strain gamma_xy = 2e-3 -> s_tr_xy = G * 2e-3 = 8000.0
    # J2_tr = 8000^2 = 6.4e7
    # ratio = sqrt(2.0e6 / 6.4e7) = 1414.213 / 8000 = 0.1767767
    deps = np.array([[-0.05/3.0, -0.05/3.0, -0.05/3.0, 2.0e-3, 0.0, 0.0]])
    extra = {}
    sig_new = solid_update(mat, sig, d_eps=deps, dt=1e-4, extra=extra)

    expected_g0 = 2.0e6
    expected_ratio = math.sqrt(expected_g0 / (8000.0**2))
    expected_sxy = expected_ratio * 8000.0

    np.testing.assert_allclose(extra["g0"][0], expected_g0, rtol=1e-5)
    np.testing.assert_allclose(sig_new[0, 3], expected_sxy, rtol=1e-5)
    np.testing.assert_allclose(sig_new[0, 0], -5.0e5, rtol=1e-5)


def test_law10_von_mises_cap_and_tension_cutoff():
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 1.0e7,
        "MAT_NU": 0.25,
        "A0": 1.0e6,
        "A1": 5.0,
        "Amax": 2.5e6,     # Cap at 2.5e6
        "pmin": -1.0e4,    # Tension cutoff at -1e4
        "c0": 0.0,
        "c1": 1.0e7,
    })
    sig = np.zeros((1, 6), dtype=float)

    # 1. High compression: P = 1e6 -> A0 + A1*P = 1e6 + 5e6 = 6e6 > Amax (2.5e6)
    # G0 must be clamped to Amax = 2.5e6
    deps_comp = np.array([[-1e6/1e7/3.0, -1e6/1e7/3.0, -1e6/1e7/3.0, 1.0e-3, 0.0, 0.0]])
    extra_cap = {}
    sig_cap = solid_update(mat, sig, d_eps=deps_comp, dt=1e-4, extra=extra_cap)
    np.testing.assert_allclose(extra_cap["g0"][0], 2.5e6, rtol=1e-5)

    # 2. Tension below pmin: P <= pmin (-1e4) -> G0 = 0, ratio = 0
    # Volumetric expansion: mu = -0.002 -> P = 1e7 * (-0.002) = -2e4 < -1e4
    deps_tens = np.array([[+0.002/3.0, +0.002/3.0, +0.002/3.0, 1.0e-3, 0.0, 0.0]])
    extra_tens = {}
    sig_tens = solid_update(mat, sig, d_eps=deps_tens, dt=1e-4, extra=extra_tens)
    assert extra_tens["g0"][0] == 0.0
    assert extra_tens["ratio"][0] == 0.0
    # Deviatoric stress is completely relaxed
    np.testing.assert_allclose(sig_tens[0, 3:], 0.0, atol=1e-12)
    # Pressure is floored at pmin = -1e4, so sig_ii = -(-1e4) = +1e4
    np.testing.assert_allclose(sig_tens[0, :3], 1.0e4, rtol=1e-5)


def test_law10_element_inactive_off():
    mat = build_law10({"MAT_RHO": 2000.0, "MAT_E": 1.0e7, "MAT_NU": 0.25, "A0": 1.0e6})
    sig = np.zeros((1, 6), dtype=float)
    deps = np.array([[0.0, 0.0, 0.0, 1.0e-3, 0.0, 0.0]])
    extra = {"off": np.array([0.0])}
    sig_out = solid_update(mat, sig, d_eps=deps, dt=1e-4, extra=extra)
    # Deviatoric stress must be 0 when off=0
    np.testing.assert_allclose(sig_out[0, 3], 0.0)


def test_law10_calling_conventions():
    mat = build_law10({"MAT_RHO": 2000.0, "MAT_E": 1.0e7, "MAT_NU": 0.25, "A0": 1.0e6})
    sig = np.zeros((1, 6), dtype=float)
    dt = 1e-4

    # Convention 1: eps_dot and dt
    eps_dot = np.array([[0.0, 0.0, 0.0, 10.0, 0.0, 0.0]])
    res1 = solid_update(mat, sig, eps_dot, dt)

    # Convention 2: d_eps keyword
    deps = eps_dot * dt
    res2 = solid_update(mat, sig, d_eps=deps, dt=dt)
    np.testing.assert_allclose(res1, res2)

    # Convention 3: element kernel signature (mat, sig, deps, epsp, dt, extra)
    epsp = np.zeros(1)
    extra = {}
    sig_k, epsp_k, c_k = solid_update(mat, sig, deps, epsp, dt, extra, return_tuple=True)
    np.testing.assert_allclose(sig_k, res1)
    assert c_k is not None


# =============================================================================
# 3. Shell Update
# =============================================================================

def test_law10_shell_update_not_implemented():
    mat = build_law10({"MAT_RHO": 2000.0, "MAT_E": 1.0e7, "MAT_NU": 0.25})
    sig = np.zeros((1, 3), dtype=float)
    with pytest.raises(NotImplementedError, match="3D solid elements only"):
        shell_update(mat, sig)


# =============================================================================
# 4. Sound Speed
# =============================================================================

def test_law10_sound_speed():
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 3.0e7,
        "MAT_NU": 0.25,
        "c1": 2.0e7,
        "bunl": 5.0e7,
    })
    g = 3.0e7 / (2.0 * 1.25)
    k_eff = 5.0e7
    expected_c = math.sqrt((k_eff + (4.0 / 3.0) * g) / 2000.0)

    c_calc = sound_speed(mat)
    assert math.isclose(c_calc, expected_c)

    # Array density support
    c_arr = sound_speed(mat, rho=np.array([2000.0, 1800.0]))
    assert c_arr.shape == (2,)
    assert math.isclose(c_arr[0], expected_c)


# =============================================================================
# 5. Consistent Algorithmic Tangent
# =============================================================================

def test_law10_consistent_tangent_elastic():
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 1.2e7,
        "MAT_NU": 0.2,
        "A0": 1.0e12,  # Elastic
    })
    p = mat.params
    g = p["G"]
    k = p["K"]
    sig = np.zeros((1, 6), dtype=float)
    extra = {"ratio": np.array([1.0]), "j2": np.array([0.0])}

    d_tangent = consistent_solid_tangent(mat, sig, extra=extra)
    assert d_tangent.shape == (1, 6, 6)

    # Check normal and shear stiffness
    c11 = k + (4.0 / 3.0) * g
    c12 = k - (2.0 / 3.0) * g
    np.testing.assert_allclose(d_tangent[0, 0, 0], c11)
    np.testing.assert_allclose(d_tangent[0, 1, 1], c11)
    np.testing.assert_allclose(d_tangent[0, 2, 2], c11)
    np.testing.assert_allclose(d_tangent[0, 0, 1], c12)
    np.testing.assert_allclose(d_tangent[0, 3, 3], g)
    np.testing.assert_allclose(d_tangent[0, 4, 4], g)
    np.testing.assert_allclose(d_tangent[0, 5, 5], g)


def test_law10_consistent_tangent_finite_difference():
    # Test that D_tangent : d_eps matches the directional derivative of solid_update
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 1.0e7,
        "MAT_NU": 0.25,
        "A0": 1.0e6,
        "A1": 0.0,
        "c0": 0.0,
        "c1": 1.0e7,
    })
    # Yielding state under pure shear
    sig_base = np.zeros((1, 6), dtype=float)
    deps_base = np.array([[0.0, 0.0, 0.0, 2.0e-3, 0.0, 0.0]])
    extra = {}
    sig_cur = solid_update(mat, sig_base, d_eps=deps_base, dt=1e-4, extra=extra)

    # Tangent at yielding state
    d_tang = consistent_solid_tangent(mat, sig_cur, extra=extra)[0]

    # Perturb strain increment by delta_eps
    delta_eps = np.array([1e-6, -5e-7, -5e-7, 2e-6, 0.0, 0.0])
    deps_perturbed = deps_base + delta_eps[None, :]

    extra_p = {}
    sig_perturbed = solid_update(mat, sig_base, d_eps=deps_perturbed, dt=1e-4, extra=extra_p)

    d_sig_fd = sig_perturbed[0] - sig_cur[0]
    d_sig_tangent = d_tang @ delta_eps

    # Verify finite-difference consistency
    np.testing.assert_allclose(d_sig_tangent, d_sig_fd, rtol=1e-3, atol=1e-2)


# =============================================================================
# 6. Additional Parity & API Verification (M10LAW / sigeps10.F)
# =============================================================================

def test_law10_public_api_and_metadata():
    """Verify standard pyradioss material API exports."""
    assert callable(law10_soil.solid_update)
    assert callable(law10_soil.shell_update)
    assert callable(law10_soil.sound_speed)
    assert callable(law10_soil.consistent_solid_tangent)
    assert callable(law10_soil.solid_tangent)
    assert callable(law10_soil.tangent)
    assert callable(law10_soil.needs_defgrad)
    assert callable(law10_soil.extra_shapes)
    assert law10_soil.tangent is law10_soil.consistent_solid_tangent
    assert law10_soil.solid_tangent is law10_soil.consistent_solid_tangent

    # Hypoelastic law does not need deformation gradient F
    assert not law10_soil.needs_defgrad()

    # Extra shapes for history variables
    shapes_none = law10_soil.extra_shapes()
    assert shapes_none["mu_bak"] == ()
    assert shapes_none["epxe"] == ()
    assert shapes_none["mu"] == ()

    shapes_nip = law10_soil.extra_shapes(nip=8)
    assert shapes_nip["mu_bak"] == (8,)
    assert shapes_nip["epxe"] == (8,)
    assert shapes_nip["mu"] == (8,)


def test_law10_1d_array_support():
    """Verify solid_update and consistent_solid_tangent accept 1D (6,) arrays."""
    mat = build_law10({"MAT_RHO": 2000.0, "MAT_E": 1.0e7, "MAT_NU": 0.25, "A0": 1.0e6})
    sig_1d = np.zeros(6, dtype=float)
    deps_1d = np.array([0.0, 0.0, 0.0, 1.0e-3, 0.0, 0.0])

    extra = {}
    sig_out = solid_update(mat, sig_1d, d_eps=deps_1d, dt=1e-4, extra=extra)
    assert sig_out.shape == (6,)
    np.testing.assert_allclose(sig_out[3], 1000.0, rtol=1e-5)

    # Tangent with 1D stress
    d_tang = consistent_solid_tangent(mat, sig_out, extra=extra)
    assert d_tang.shape == (6, 6)


def test_law10_shock_bulk_viscosity():
    """Verify shock bulk viscosity (q / qvis) increases hydrostatic pressure and modifies yield surface."""
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 1.0e7,
        "MAT_NU": 0.25,
        "A0": 1.0e6,
        "A1": 2.0,
        "c0": 0.0,
        "c1": 1.0e7,
    })
    sig = np.zeros((1, 6), dtype=float)
    # Hydrostatic compression mu = 0.02 -> P_eos = 2e5
    deps = np.array([[-0.02 / 3.0, -0.02 / 3.0, -0.02 / 3.0, 1.0e-3, 0.0, 0.0]])

    # Without bulk viscosity
    extra_no_q = {}
    sig_no_q = solid_update(mat, sig, d_eps=deps, dt=1e-4, extra=extra_no_q)
    # P_tot = 2e5, G0 = 1e6 + 2.0 * 2e5 = 1.4e6
    assert math.isclose(extra_no_q["ptot"][0], 2.0e5)
    assert math.isclose(extra_no_q["g0"][0], 1.4e6)
    np.testing.assert_allclose(sig_no_q[0, 0], -2.0e5, rtol=1e-5)

    # With bulk viscosity q = 5.0e4
    extra_q = {"q": np.array([5.0e4])}
    sig_q = solid_update(mat, sig, d_eps=deps, dt=1e-4, extra=extra_q)
    # P_tot = 2e5 + 5e4 = 2.5e5, G0 = 1e6 + 2.0 * 2.5e5 = 1.5e6
    assert math.isclose(extra_q["ptot"][0], 2.5e5)
    assert math.isclose(extra_q["g0"][0], 1.5e6)
    # Total stress includes q: -(P_new + q) = -2.5e5
    np.testing.assert_allclose(sig_q[0, 0], -2.5e5, rtol=1e-5)


def test_law10_extended_aliases():
    """Verify MAT_G, MAT_K, MAT_PMIN, PFRAC, MAT_BUNL, MAT_XMUMX aliases in _ensure_params."""
    p = _ensure_params({
        "MAT_RHO": 2500.0,
        "MAT_E": 2.0e7,
        "MAT_NU": 0.2,
        "MAT_G": 8.5e6,     # overrides default E/(2(1+nu))
        "MAT_K": 1.2e7,     # overrides default E/(3(1-2nu))
        "MAT_PMIN": -5.0e4, # overrides pmin
        "MAT_BUNL": 3.0e7,
        "MAT_XMUMX": 0.4,
    })
    assert p["G"] == 8.5e6
    assert p["K"] == 1.2e7
    assert p["pmin"] == -5.0e4
    assert p["bunl"] == 3.0e7
    assert p["mue_max"] == 0.4
    assert p["MAT_G"] == 8.5e6
    assert p["MAT_K"] == 1.2e7
    assert p["MAT_PMIN"] == -5.0e4
    assert p["PFRAC"] == -5.0e4
    assert p["MAT_BUNL"] == 3.0e7
    assert p["MAT_XMUMX"] == 0.4
