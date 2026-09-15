"""Comprehensive unit test suite for Milestone M556: /MAT/LAW21 (/MAT/DPRAG).

Constitutive physics for the Drucker-Prager parabolic/linear yield surface material
model with compaction equation of state in `pyradioss/materials/law21_dprag.py`.

Fortran references:
- ``engine/source/materials/mat/mat021/m21law.F``
- ``starter/source/materials/mat/mat021/hm_read_mat21.F``
- ``hm_cfg_files/config/CFG/radioss110/MAT/matl21_dprag.cfg``
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials.law21_dprag import (
    build_law21,
    _ensure_params,
    solid_update_law21,
    solid_update,
    sound_speed_solid_law21,
    sound_speed_solid,
    tangent_law21_solid,
    consistent_solid_tangent,
    shell_update_law21,
    shell_update,
)
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
from pyradioss.model.entities import Material


# =============================================================================
# 1. Parameter Validation and Defaults
# =============================================================================

def test_law21_parameter_validation_valid():
    """Valid parameters produce expected derived values (G, K, c1, bunl, Amax, pmin, pstar)."""
    rec = {
        "id": 1,
        "MAT_RHO": 2400.0,
        "MAT_E": 3.0e10,
        "MAT_NU": 0.20,
        "MAT_A0": 1.0e6,
        "MAT_A1": 0.6,
        "MAT_A2": 0.0,
    }
    mat = build_law21(rec)
    p = mat.params
    assert mat.law == 21
    assert mat.rho0 == 2400.0

    expected_g = 3.0e10 / (2.0 * (1.0 + 0.20))
    expected_k = 3.0e10 / (3.0 * (1.0 - 2.0 * 0.20))
    assert math.isclose(p["G"], expected_g)
    assert math.isclose(p["K"], expected_k)

    # Defaults
    assert p["c1"] == expected_k
    assert p["bunl"] == p["c1"]
    assert p["amax"] == 1e20
    assert p["pmin"] == -1e30
    assert p["pext"] == 0.0
    assert p["mumax"] == 1e20
    assert p["pfscale"] == 1.0
    # Linear root: pstar = -A0 / A1
    assert math.isclose(p["pstar"], -1.0e6 / 0.6)


def test_law21_parameter_validation_invalid_density():
    """Density rho0 <= 0 raises ValueError."""
    rec = {"id": 1, "MAT_RHO": 0.0, "MAT_E": 1.0e10, "MAT_NU": 0.2}
    with pytest.raises(ValueError, match="Density rho0 must be > 0"):
        build_law21(rec)

    rec_neg = {"id": 1, "MAT_RHO": -100.0, "MAT_E": 1.0e10, "MAT_NU": 0.2}
    with pytest.raises(ValueError, match="Density rho0 must be > 0"):
        build_law21(rec_neg)


def test_law21_parameter_validation_invalid_young():
    """Young's modulus E <= 0 raises ValueError."""
    rec = {"id": 1, "MAT_RHO": 2000.0, "MAT_E": 0.0, "MAT_NU": 0.2}
    with pytest.raises(ValueError, match="Young's modulus E must be > 0"):
        build_law21(rec)


def test_law21_parameter_validation_invalid_nu():
    """Poisson's ratio nu < 0 or >= 0.5 raises ValueError."""
    rec_neg = {"id": 1, "MAT_RHO": 2000.0, "MAT_E": 1.0e10, "MAT_NU": -0.1}
    with pytest.raises(ValueError, match="Poisson's ratio nu must be in"):
        build_law21(rec_neg)

    rec_incomp = {"id": 1, "MAT_RHO": 2000.0, "MAT_E": 1.0e10, "MAT_NU": 0.5}
    with pytest.raises(ValueError, match="Poisson's ratio nu must be in"):
        build_law21(rec_incomp)


def test_law21_parameter_validation_invalid_tensile_bulk():
    """Explicit tensile bulk modulus c1 <= 0 raises ValueError."""
    rec = {"id": 1, "MAT_RHO": 2000.0, "MAT_E": 1.0e10, "MAT_NU": 0.2, "MAT_BULK": -50.0}
    with pytest.raises(ValueError, match="Tensile bulk modulus C1 must be > 0"):
        build_law21(rec)


# =============================================================================
# 2. Pressure Root P* (Closure Pressure)
# =============================================================================

def test_law21_pstar_linear():
    """Linear yield surface A2 = 0, A1 != 0 -> P* = -A0 / A1."""
    p = {"rho0": 2000.0, "E": 1e9, "nu": 0.25, "a0": 2.5e6, "a1": 0.5, "a2": 0.0}
    params = _ensure_params(p)
    assert math.isclose(params["pstar"], -2.5e6 / 0.5)


def test_law21_pstar_quadratic_real_roots():
    """Parabolic yield surface with real roots -> P* = (-A1 + sqrt(Delta)) / (2*A2)."""
    # G0(P) = 1.0e6 + 2.0e3 * P + 1.0 * P^2 = (P + 1000)^2
    # Delta = (2000)^2 - 4 * 1.0e6 * 1 = 0 -> P* = -1000.0
    p = {"rho0": 2000.0, "E": 1e9, "nu": 0.25, "a0": 1.0e6, "a1": 2.0e3, "a2": 1.0}
    params = _ensure_params(p)
    assert math.isclose(params["pstar"], -1000.0, abs_tol=1e-5)

    # Distinct real roots: A0 = 6, A1 = 5, A2 = 1 -> roots -3, -2
    # (-A1 + sqrt(Delta))/(2*A2) = (-5 + 1)/2 = -2.0
    p2 = {"rho0": 2000.0, "E": 1e9, "nu": 0.25, "a0": 6.0, "a1": 5.0, "a2": 1.0}
    params2 = _ensure_params(p2)
    assert math.isclose(params2["pstar"], -2.0)


def test_law21_pstar_quadratic_no_real_roots():
    """Parabolic yield surface with Delta < 0 -> P* = -infinity (-1e30)."""
    # A0 = 1e6, A1 = 0, A2 = 1 -> Delta = -4e6 < 0
    p = {"rho0": 2000.0, "E": 1e9, "nu": 0.25, "a0": 1.0e6, "a1": 0.0, "a2": 1.0}
    params = _ensure_params(p)
    assert params["pstar"] == -1e30


def test_law21_pstar_von_mises():
    """Constant von Mises yield surface A1 = 0, A2 = 0 -> P* = -infinity."""
    p = {"rho0": 2000.0, "E": 1e9, "nu": 0.25, "a0": 1.0e6, "a1": 0.0, "a2": 0.0}
    params = _ensure_params(p)
    assert params["pstar"] == -1e30


# =============================================================================
# 3. Compaction Equation of State Lookup, Scale, and Slope
# =============================================================================

def test_law21_compaction_eos_linear_default():
    """Without a user curve, pressure is linear in volumetric strain P = C1 * mu."""
    mat = {
        "rho0": 2000.0,
        "E": 1.0e9,
        "nu": 0.25,
        "c1": 8.0e8,
        "a0": 1.0e12,  # Elastic
    }
    sig = np.zeros((1, 6), dtype=float)
    # Volumetric compression deps_xx = deps_yy = deps_zz = -0.01 -> tr(deps) = -0.03 -> mu = +0.03
    deps = np.array([[-0.01, -0.01, -0.01, 0.0, 0.0, 0.0]], dtype=float)
    extra = {}
    sig_new = solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

    expected_p = 8.0e8 * 0.03  # 2.4e7
    assert math.isclose(extra["p_new"][0], expected_p)
    # Normal stresses are -P (compression positive pressure)
    assert math.isclose(sig_new[0, 0], -expected_p)
    assert math.isclose(sig_new[0, 1], -expected_p)
    assert math.isclose(sig_new[0, 2], -expected_p)


def test_law21_compaction_eos_table_lookup_and_slope():
    """Piecewise-linear table lookup returns exact interpolated pressure and slope."""
    xs = np.array([0.0, 0.05, 0.10, 0.20])
    ys = np.array([0.0, 1.0e7, 3.0e7, 8.0e7])
    curve = (xs, ys)

    mat = {
        "rho0": 2000.0,
        "E": 1.0e10,
        "nu": 0.25,
        "c1": 5.0e8,
        "a0": 1.0e14,
        "curve": curve,
    }

    # Test point inside interval [0.05, 0.10]: mu = 0.075 -> y = 2.0e7, slope = 2.0e7 / 0.05 = 4.0e8
    sig = np.zeros((1, 6), dtype=float)
    deps = np.array([[-0.025, -0.025, -0.025, 0.0, 0.0, 0.0]], dtype=float)
    extra = {}
    solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

    assert math.isclose(extra["p_new"][0], 2.0e7, rel_tol=1e-6)
    assert math.isclose(extra["dpdm"][0], 4.0e8, rel_tol=1e-6)


def test_law21_compaction_eos_pfscale():
    """PFscale scales pressure and slope proportionally."""
    xs = np.array([0.0, 0.10])
    ys = np.array([0.0, 2.0e7])
    curve = (xs, ys)

    mat = {
        "rho0": 2000.0,
        "E": 1.0e10,
        "nu": 0.25,
        "pfscale": 2.5,
        "a0": 1.0e14,
        "curve": curve,
    }

    sig = np.zeros((1, 6), dtype=float)
    deps = np.array([[-0.02, -0.02, -0.02, 0.0, 0.0, 0.0]], dtype=float)  # mu = 0.06
    extra = {}
    solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

    # Unscaled y = 0.06 * 2.0e8 = 1.2e7, scaled by 2.5 = 3.0e7
    assert math.isclose(extra["p_new"][0], 3.0e7, rel_tol=1e-6)
    # Slope = 2.5 * 2.0e8 = 5.0e8
    assert math.isclose(extra["dpdm"][0], 5.0e8, rel_tol=1e-6)


def test_law21_compaction_eos_extrapolation():
    """Extrapolating outside table domain adheres to end slopes."""
    xs = np.array([0.0, 0.10])
    ys = np.array([0.0, 2.0e7])  # slope = 2.0e8
    curve = (xs, ys)

    mat = {
        "rho0": 2000.0,
        "E": 1.0e10,
        "nu": 0.25,
        "a0": 1.0e14,
        "curve": curve,
    }

    # mu = 0.15 > xs[-1]
    sig = np.zeros((1, 6), dtype=float)
    deps = np.array([[-0.05, -0.05, -0.05, 0.0, 0.0, 0.0]], dtype=float)
    extra = {}
    solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

    # 2.0e7 + 2.0e8 * 0.05 = 3.0e7
    assert math.isclose(extra["p_new"][0], 3.0e7, rel_tol=1e-6)
    assert math.isclose(extra["dpdm"][0], 2.0e8, rel_tol=1e-6)


# =============================================================================
# 4. Hysteretic Unloading & Evolving Bulk Modulus
# =============================================================================

def test_law21_compaction_hysteretic_unloading():
    """Unloading from compacted state follows the unloading bulk modulus."""
    xs = np.array([0.0, 0.10])
    ys = np.array([0.0, 1.0e7])  # loading slope = 1.0e8
    curve = (xs, ys)

    mat = {
        "rho0": 2000.0,
        "E": 1.0e10,
        "nu": 0.25,
        "c1": 5.0e8,
        "bunl": 1.0e9,
        "mumax": 1e20,  # alpha ~ 0 -> bulk = c1 = 5.0e8
        "a0": 1.0e14,
        "curve": curve,
    }

    # Step 1: load to mu = 0.10 -> P = 1.0e7, mu_bak becomes 0.10
    sig = np.zeros((1, 6), dtype=float)
    deps1 = np.array([[-1.0/30.0, -1.0/30.0, -1.0/30.0, 0.0, 0.0, 0.0]], dtype=float)  # tr = -0.10
    extra = {}
    solid_update_law21(mat, sig, deps=deps1, dt=1.0, extra=extra)
    assert math.isclose(extra["mu_bak"][0], 0.10, rel_tol=1e-6)
    assert math.isclose(extra["p_new"][0], 1.0e7, rel_tol=1e-6)

    # Step 2: partial volumetric expansion (unloading) to mu = 0.09 (delta mu = -0.01)
    deps2 = np.array([[1.0/300.0, 1.0/300.0, 1.0/300.0, 0.0, 0.0, 0.0]], dtype=float)  # tr = +0.01 -> mu = 0.09
    solid_update_law21(mat, sig, deps=deps2, dt=1.0, extra=extra)

    # Unloading formula: P_unl = P_bak - (mu_bak - mu) * bulk
    # bulk = 5.0e8, (0.10 - 0.09) * 5.0e8 = 5.0e6
    # P_unl = 1.0e7 - 5.0e6 = 5.0e6
    # Loading P(0.09) = 9.0e6 > P_unl, so min(P_unl, P_load) selects P_unl
    assert math.isclose(extra["p_new"][0], 5.0e6, rel_tol=1e-6)
    # mu_bak is preserved at historical peak 0.10
    assert math.isclose(extra["mu_bak"][0], 0.10, rel_tol=1e-6)


def test_law21_evolving_unloading_modulus():
    """Evolving unloading bulk modulus alpha = mu_bak / mumax interpolates bmin to bmax."""
    xs = np.array([0.0, 0.20])
    ys = np.array([0.0, 2.0e7])
    curve = (xs, ys)

    bmin = 4.0e8
    bmax = 1.2e9
    mumax = 0.20

    mat = {
        "rho0": 2000.0,
        "E": 1.0e10,
        "nu": 0.25,
        "c1": bmin,
        "bunl": bmax,
        "mumax": mumax,
        "a0": 1.0e14,
        "curve": curve,
    }

    # Compact to mu = 0.10 -> alpha = 0.10 / 0.20 = 0.5
    # Expected bulk = 0.5 * 1.2e9 + 0.5 * 4.0e8 = 8.0e8
    sig = np.zeros((1, 6), dtype=float)
    deps1 = np.array([[-1.0/30.0, -1.0/30.0, -1.0/30.0, 0.0, 0.0, 0.0]], dtype=float)
    extra = {}
    solid_update_law21(mat, sig, deps=deps1, dt=1.0, extra=extra)

    # Unload slightly by 0.005
    deps2 = np.array([[0.005/3.0, 0.005/3.0, 0.005/3.0, 0.0, 0.0, 0.0]], dtype=float)
    solid_update_law21(mat, sig, deps=deps2, dt=1.0, extra=extra)

    assert math.isclose(extra["bulk"][0], 8.0e8, rel_tol=1e-6)


def test_law21_compaction_reloading_transition():
    """Reloading past historical mu_bak resumes the primary loading curve."""
    xs = np.array([0.0, 0.20])
    ys = np.array([0.0, 2.0e7])
    curve = (xs, ys)

    mat = {
        "rho0": 2000.0,
        "E": 1.0e10,
        "nu": 0.25,
        "c1": 5.0e8,
        "bunl": 1.0e9,
        "a0": 1.0e14,
        "curve": curve,
    }

    # Step 1: compact to mu = 0.05
    sig = np.zeros((1, 6), dtype=float)
    extra = {}
    solid_update_law21(mat, sig, deps=np.array([[-0.05/3, -0.05/3, -0.05/3, 0, 0, 0]]), dt=1.0, extra=extra)
    assert math.isclose(extra["mu_bak"][0], 0.05, rel_tol=1e-6)

    # Step 2: unload to mu = 0.03
    solid_update_law21(mat, sig, deps=np.array([[0.02/3, 0.02/3, 0.02/3, 0, 0, 0]]), dt=1.0, extra=extra)
    assert math.isclose(extra["mu_bak"][0], 0.05, rel_tol=1e-6)

    # Step 3: reload all the way to mu = 0.10 > 0.05
    solid_update_law21(mat, sig, deps=np.array([[-0.07/3, -0.07/3, -0.07/3, 0, 0, 0]]), dt=1.0, extra=extra)
    assert math.isclose(extra["mu_bak"][0], 0.10, rel_tol=1e-6)
    assert math.isclose(extra["p_new"][0], 1.0e7, rel_tol=1e-6)


# =============================================================================
# 5. Yield Surface & Radial Return (Linear & Parabolic)
# =============================================================================

def test_law21_elastic_trial_within_yield_surface():
    """Deviatoric trial state within yield surface gives ratio = 1.0, purely elastic response."""
    mat = {
        "rho0": 2000.0,
        "E": 2.0e10,
        "nu": 0.25,  # G = 8.0e9
        "a0": 1.0e14,  # High yield envelope
        "a1": 0.0,
        "a2": 0.0,
    }
    sig = np.zeros((1, 6), dtype=float)
    # Pure shear gamma_xy = 1.0e-5 -> tau_xy = G * gamma = 8.0e4 -> J2 = (8.0e4)^2 = 6.4e9 < 1.0e14
    deps = np.array([[0.0, 0.0, 0.0, 1.0e-5, 0.0, 0.0]], dtype=float)
    extra = {}
    sig_new = solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

    assert math.isclose(extra["ratio"][0], 1.0)
    assert math.isclose(extra["dpla"][0], 0.0)
    assert math.isclose(sig_new[0, 3], 8.0e4)


def test_law21_plastic_radial_return_linear():
    """Plastic yielding on linear Drucker-Prager surface projects deviatoric stress to envelope."""
    # A0 = 1.0e10, A1 = 0.5, A2 = 0.0
    # Pressure P = 2.0e7 -> Ptot = 2.0e7
    # G0 = 1.0e10 + 0.5 * 2.0e7 = 2.0e10
    mat = {
        "rho0": 2000.0,
        "E": 2.0e10,
        "nu": 0.25,
        "c1": 1.0e9,
        "a0": 1.0e10,
        "a1": 500.0,
        "a2": 0.0,
    }
    g = 2.0e10 / (2.0 * 1.25)  # 8.0e9

    sig = np.zeros((1, 6), dtype=float)
    # Apply hydrostatic compression (mu = 0.02 -> P = 2.0e7) and large shear (gamma_xy = 1.0e-4)
    # Trial tau_xy = 8.0e9 * 1.0e-4 = 8.0e5 -> J2 = 6.4e11 >> 2.0e10
    deps = np.array([[-0.02/3, -0.02/3, -0.02/3, 1.0e-4, 0.0, 0.0]], dtype=float)
    extra = {}
    sig_new = solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

    expected_g0 = 2.0e10
    expected_ratio = math.sqrt(expected_g0 / (8.0e5)**2)
    assert math.isclose(extra["g0"][0], expected_g0)
    assert math.isclose(extra["ratio"][0], expected_ratio, rel_tol=1e-6)
    assert extra["ratio"][0] < 1.0

    # New J2 must equal G0
    j2_new = (
        0.5 * (sig_new[0, 0] - (-2.0e7))**2 * 3
        + sig_new[0, 3]**2
    )
    assert math.isclose(sig_new[0, 3]**2, expected_g0, rel_tol=1e-6)
    # Plastic strain accumulated
    assert extra["dpla"][0] > 0.0


def test_law21_plastic_radial_return_parabolic():
    """Parabolic yield surface A2 != 0 correctly evaluates G0(P) and performs return mapping."""
    # G0(P) = A0 + A1 * P + A2 * P^2
    # A0 = 1.0e10, A1 = 0.2, A2 = 1.0e-8
    # P = 1.0e7 -> A1*P = 2.0e6, A2*P^2 = 1.0e-8 * 1.0e14 = 1.0e6 -> G0 = 1.0e10 + 3.0e6
    mat = {
        "rho0": 2000.0,
        "E": 2.0e10,
        "nu": 0.25,
        "c1": 1.0e9,
        "a0": 1.0e10,
        "a1": 0.2,
        "a2": 1.0e-8,
    }
    sig = np.zeros((1, 6), dtype=float)
    deps = np.array([[-0.01/3, -0.01/3, -0.01/3, 5.0e-5, 0.0, 0.0]], dtype=float)  # P = 1.0e7
    extra = {}
    sig_new = solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

    expected_g0 = 1.0e10 + 0.2 * 1.0e7 + 1.0e-8 * (1.0e7)**2
    assert math.isclose(extra["g0"][0], expected_g0, rel_tol=1e-6)
    assert math.isclose(sig_new[0, 3]**2, expected_g0, rel_tol=1e-6)


def test_law21_von_mises_cap_cutoff():
    """Yield surface envelope G0 is capped by Amax."""
    mat = {
        "rho0": 2000.0,
        "E": 2.0e10,
        "nu": 0.25,
        "c1": 1.0e9,
        "a0": 1.0e10,
        "a1": 10.0,
        "amax": 5.0e10,  # Cap at 5.0e10
    }
    # P = 1.0e10 -> A0 + A1*P = 1.0e10 + 1.0e11 = 1.1e11 > Amax
    sig = np.zeros((1, 6), dtype=float)
    deps = np.array([[-10.0/3, -10.0/3, -10.0/3, 1.0e-4, 0.0, 0.0]], dtype=float)
    extra = {}
    solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

    assert math.isclose(extra["g0"][0], 5.0e10)


def test_law21_root_cutoff_pstar():
    """When Ptot <= P*, yield surface collapses to G0 = 0 and deviatoric stress is zero."""
    # A0 = 1.0e8, A1 = 1.0 -> P* = -1.0e8
    mat = {
        "rho0": 2000.0,
        "E": 2.0e10,
        "nu": 0.25,
        "c1": 1.0e9,
        "a0": 1.0e8,
        "a1": 1.0,
        "a2": 0.0,
    }
    # Tensile strain: mu = -0.15 -> P = -1.5e8 < P* (-1.0e8)
    sig = np.zeros((1, 6), dtype=float)
    deps = np.array([[0.05, 0.05, 0.05, 1.0e-5, 0.0, 0.0]], dtype=float)
    extra = {}
    sig_new = solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

    assert extra["g0"][0] == 0.0
    assert extra["ratio"][0] == 0.0
    assert sig_new[0, 3] == 0.0


def test_law21_tensile_cutoff_pmin():
    """When pressure reaches Pmin, yield surface collapses and pressure is clamped."""
    mat = {
        "rho0": 2000.0,
        "E": 2.0e10,
        "nu": 0.25,
        "c1": 1.0e9,
        "pmin": -5.0e7,  # Tensile cutoff
        "a0": 1.0e12,
        "a1": 0.0,
    }
    # Tensile strain: mu = -0.10 -> unconstrained P would be -1.0e8 < -5.0e7
    sig = np.zeros((1, 6), dtype=float)
    deps = np.array([[1.0/30.0, 1.0/30.0, 1.0/30.0, 1.0e-5, 0.0, 0.0]], dtype=float)
    extra = {}
    sig_new = solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

    assert math.isclose(extra["p_new"][0], -5.0e7)
    assert extra["g0"][0] == 0.0
    assert extra["ratio"][0] == 0.0
    assert sig_new[0, 3] == 0.0


def test_law21_external_pressure_shift():
    """External pressure shift pext increases Ptot = P + pext, expanding yield envelope."""
    mat = {
        "rho0": 2000.0,
        "E": 2.0e10,
        "nu": 0.25,
        "c1": 1.0e9,
        "pext": 5.0e7,  # Shift
        "a0": 1.0e10,
        "a1": 1.0,
        "a2": 0.0,
    }
    # P = 0 (zero volumetric strain) -> Ptot = 5.0e7 -> G0 = 1.0e10 + 5.0e7
    sig = np.zeros((1, 6), dtype=float)
    deps = np.array([[0.0, 0.0, 0.0, 1.0e-5, 0.0, 0.0]], dtype=float)
    extra = {}
    solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

    expected_g0 = 1.0e10 + 5.0e7
    assert math.isclose(extra["g0"][0], expected_g0)


# =============================================================================
# 6. Sound Speed Calculation
# =============================================================================

def test_law21_sound_speed_elastic_and_compaction():
    """Sound speed matches c = sqrt(|4/3 * G + K_eff| / rho0)."""
    e = 3.0e10
    nu = 0.20
    rho0 = 2500.0
    g = e / (2.0 * (1.0 + nu))  # 1.25e10
    k = e / (3.0 * (1.0 - 2.0 * nu))  # 1.6667e10
    mat = {
        "rho0": rho0,
        "E": e,
        "nu": nu,
        "c1": k,
        "bunl": k,
        "a0": 1.0e14,
    }
    expected_c = math.sqrt(((4.0 / 3.0) * g + k) / rho0)
    c_val = sound_speed_solid_law21(mat)
    assert math.isclose(c_val, expected_c, rel_tol=1e-6)

    # In compaction regime with higher tangent bulk modulus
    xs = np.array([0.0, 0.10])
    ys = np.array([0.0, 5.0e9])  # slope = 5.0e10
    mat_compact = dict(mat)
    mat_compact["curve"] = (xs, ys)

    extra = {"mu": np.array([0.05]), "mu_bak": np.array([0.05])}
    c_compact = sound_speed_solid_law21(mat_compact, extra=extra)
    expected_c_compact = math.sqrt(((4.0 / 3.0) * g + 5.0e10) / rho0)
    assert math.isclose(c_compact, expected_c_compact, rel_tol=1e-6)


# =============================================================================
# 7. Algorithmic Consistent Tangent vs Central Finite Differences
# =============================================================================

def test_law21_consistent_tangent_elastic():
    """Elastic consistent tangent matches Hookean tensor Keet + Cdev."""
    e = 1.2e10
    nu = 0.20
    g = e / (2.0 * (1.0 + nu))
    k = e / (3.0 * (1.0 - 2.0 * nu))
    mat = {
        "rho0": 2000.0,
        "E": e,
        "nu": nu,
        "c1": k,
        "bunl": k,
        "a0": 1.0e14,
    }

    sig = np.zeros((1, 6), dtype=float)
    extra = {}
    # Small elastic strain increment
    solid_update_law21(mat, sig, deps=np.array([[1e-5, 0, 0, 0, 0, 0]]), dt=1.0, extra=extra)

    d_ana = tangent_law21_solid(mat, sig, extra=extra)[0]

    # Expected Hookean tensor
    ee = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    c_dev = np.zeros((6, 6), dtype=float)
    c_dev[0, 0] = c_dev[1, 1] = c_dev[2, 2] = (4.0 / 3.0) * g
    c_dev[0, 1] = c_dev[0, 2] = c_dev[1, 0] = c_dev[1, 2] = c_dev[2, 0] = c_dev[2, 1] = -(2.0 / 3.0) * g
    c_dev[3, 3] = c_dev[4, 4] = c_dev[5, 5] = g
    d_expected = k * np.outer(ee, ee) + c_dev

    np.testing.assert_allclose(d_ana, d_expected, rtol=1e-6, atol=1e-3)


@pytest.mark.parametrize("a2_val", [0.0, 1.0e-8])
def test_law21_consistent_tangent_vs_finite_difference(a2_val):
    """Algorithmic consistent tangent matches central numerical finite differences for both linear and parabolic yield surfaces."""
    e = 2.0e10
    nu = 0.25
    mat = {
        "rho0": 2000.0,
        "E": e,
        "nu": nu,
        "c1": 1.0e9,
        "bunl": 1.0e9,
        "a0": 1.0e10,
        "a1": 0.5,
        "a2": a2_val,
    }

    # Plastic state with both normal and shear components
    sig_init = np.zeros((1, 6), dtype=float)
    deps0 = np.array([[-0.015, -0.010, -0.005, 8.0e-5, 4.0e-5, 2.0e-5]], dtype=float)

    extra = {}
    sig_base = solid_update_law21(mat, sig_init, deps=deps0, dt=1.0, extra=extra)

    # Analytical tangent
    d_ana = tangent_law21_solid(mat, sig_base, extra=extra)[0]

    # Central finite differences: dsigma_i / ddeps_j
    h = 1.0e-8
    d_num = np.zeros((6, 6), dtype=float)

    for j in range(6):
        deps_plus = deps0.copy()
        deps_plus[0, j] += h
        extra_p = {}
        sig_p = solid_update_law21(mat, sig_init, deps=deps_plus, dt=1.0, extra=extra_p)

        deps_minus = deps0.copy()
        deps_minus[0, j] -= h
        extra_m = {}
        sig_m = solid_update_law21(mat, sig_init, deps=deps_minus, dt=1.0, extra=extra_m)

        d_num[:, j] = (sig_p[0] - sig_m[0]) / (2.0 * h)

    # Verify all 36 entries match to high precision
    np.testing.assert_allclose(d_ana, d_num, rtol=1e-4, atol=1e2)


def test_law21_consistent_tangent_symmetric_flag():
    """symmetric=True returns symmetrized matrix 0.5 * (D + D^T)."""
    mat = {
        "rho0": 2000.0,
        "E": 2.0e10,
        "nu": 0.25,
        "c1": 1.0e9,
        "a0": 1.0e10,
        "a1": 0.5,
        "a2": 0.0,
    }
    sig = np.zeros((1, 6), dtype=float)
    deps = np.array([[-0.015, -0.010, -0.005, 8.0e-5, 0, 0]], dtype=float)
    extra = {}
    sig_up = solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

    d_exact = tangent_law21_solid(mat, sig_up, extra=extra, symmetric=False)[0]
    d_sym = tangent_law21_solid(mat, sig_up, extra=extra, symmetric=True)[0]

    np.testing.assert_allclose(d_sym, 0.5 * (d_exact + d_exact.T), rtol=1e-9)


def test_law21_consistent_tangent_fracture_cutoff():
    """When pressure is below or at pmin, tangent is zero."""
    mat = {
        "rho0": 2000.0,
        "E": 2.0e10,
        "nu": 0.25,
        "c1": 1.0e9,
        "pmin": -1.0e7,
        "a0": 1.0e10,
    }
    sig = np.zeros((1, 6), dtype=float)
    # Tensile strain past pmin
    deps = np.array([[0.02, 0.02, 0.02, 0, 0, 0]], dtype=float)
    extra = {}
    sig_up = solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

    d_tangent = tangent_law21_solid(mat, sig_up, extra=extra)[0]
    np.testing.assert_allclose(d_tangent, np.zeros((6, 6)), atol=1e-12)


# =============================================================================
# 8. Shell Update Rejection & Interface Conventions
# =============================================================================

def test_law21_shell_update_raises():
    """shell_update_law21 raises NotImplementedError for plane-stress shells."""
    with pytest.raises(NotImplementedError, match="LAW21 .* 3D solid elements only"):
        shell_update_law21({}, np.zeros(3))

    with pytest.raises(NotImplementedError, match="LAW21 .* 3D solid elements only"):
        shell_update({}, np.zeros(3))


def test_law21_element_deactivation_off():
    """off = 0 deactivates the element, zeroing Cauchy stress and tangent."""
    mat = {"rho0": 2000.0, "E": 1.0e10, "nu": 0.25, "a0": 1.0e10}
    sig = np.zeros((1, 6), dtype=float)
    deps = np.array([[-0.01, -0.01, -0.01, 1e-4, 0, 0]], dtype=float)
    extra = {"off": np.array([0.0])}

    sig_new = solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)
    np.testing.assert_allclose(sig_new, 0.0)

    d_tangent = tangent_law21_solid(mat, sig_new, extra=extra)[0]
    np.testing.assert_allclose(d_tangent, 0.0)


def test_law21_1d_and_2d_batch_equivalence():
    """Single 1D array (6,) and batched 2D array (1, 6) give identical results."""
    mat = {"rho0": 2000.0, "E": 1.0e10, "nu": 0.25, "a0": 1.0e10, "a1": 0.5}
    sig_1d = np.zeros(6, dtype=float)
    deps_1d = np.array([-0.01, -0.01, -0.01, 5e-5, 0, 0], dtype=float)

    sig_2d = sig_1d.reshape(1, 6)
    deps_2d = deps_1d.reshape(1, 6)

    res_1d = solid_update_law21(mat, sig_1d, deps=deps_1d, dt=1.0)
    res_2d = solid_update_law21(mat, sig_2d, deps=deps_2d, dt=1.0)

    assert res_1d.ndim == 1
    assert res_2d.ndim == 2
    np.testing.assert_allclose(res_1d, res_2d[0])

    d_1d = tangent_law21_solid(mat, res_1d)
    d_2d = tangent_law21_solid(mat, res_2d)
    assert d_1d.shape == (6, 6)
    assert d_2d.shape == (1, 6, 6)
    np.testing.assert_allclose(d_1d, d_2d[0])


def test_law21_zero_dt_no_op():
    """dt <= 0 returns a copy of input stress unchanged."""
    mat = {"rho0": 2000.0, "E": 1.0e10, "nu": 0.25, "a0": 1.0e10}
    sig = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=float)
    sig_out = solid_update_law21(mat, sig, dt=0.0)
    np.testing.assert_allclose(sig_out, sig)


def test_law21_empty_array_safety():
    """Empty (0, 6) array returns cleanly without error."""
    mat = {"rho0": 2000.0, "E": 1.0e10, "nu": 0.25, "a0": 1.0e10}
    sig = np.empty((0, 6), dtype=float)
    sig_out = solid_update_law21(mat, sig, dt=1.0)
    assert sig_out.shape == (0, 6)

    d_out = tangent_law21_solid(mat, sig)
    assert d_out.shape == (0, 6, 6)


def test_law21_mat_physics_registry():
    """LAW21 and DPRAG are registered in MAT_PHYSICS_REGISTRY."""
    assert "LAW21" in MAT_PHYSICS_REGISTRY
    assert "DPRAG" in MAT_PHYSICS_REGISTRY
    builder = MAT_PHYSICS_REGISTRY["LAW21"]
    mat = builder({"id": 42, "MAT_RHO": 2000.0, "MAT_E": 1.0e10, "MAT_NU": 0.25})
    assert mat.id == 42
    assert mat.law == 21
