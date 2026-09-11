"""
Unit tests for Milestone M550: LAW69 hyperelastic material law
(/MAT/LAW69, /MAT/HYP_ELAS, /MAT/HYPERELASTIC).

Tests cover:
1. Law69Params dataclass attributes, defaults, and properties.
2. Curve fitting fit_law69_curve:
   - Mooney-Rivlin (law_id=2) analytical linear least-squares.
   - Ogden (law_id=1) nonlinear least-squares.
   - Automatic fitting (law_id=-1).
   - Initial elastic constants Gs, G0, K, E.
3. build_law69 factory function:
   - Direct argument configuration.
   - Parameter extraction from dictionary / GenericMaterialRecord.
   - Automatic fitting from supplied test curve.
4. 3D continuum solid kernel sigeps69_solid:
   - Zero strain gives zero stress.
   - Uniaxial tension response.
   - Vectorization across multiple elements (NEL > 1).
   - Nonlinear sound speed calculation.
   - Anti-buckling pressure factor.
   - Tensile cut-off behavior.
5. 2D shell plane-stress kernel sigeps69c_shell:
   - Zero strain gives zero stress.
   - Plane stress out-of-plane stretch solve (Newton-Raphson).
   - Thickness update.
   - Transverse shear stress update.
   - Shell sound speed.
   - Tensile cut-off behavior for shells.
6. Exact algorithmic consistent tangents:
   - tangent_law69_solid accuracy against independent central finite differences (< 1e-4).
   - tangent_law69_shell accuracy against independent central finite differences (< 1e-4).
7. Dispatch and registry:
   - Registration in MAT_PHYSICS_REGISTRY.
   - Dispatch through pyradioss.materials.solid_update and shell_update.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
import pyradioss.materials as materials
from pyradioss.materials.law69_hyperelastic import (
    Law69Params,
    build_law69,
    fit_law69_curve,
    sigeps69_solid,
    sigeps69c_shell,
    solid_sound_speed,
    shell_sound_speed,
    tangent_law69_solid,
    tangent_law69_shell,
)


# ============================================================================
# 1. Parameter Dataclass & Initialization Tests
# ============================================================================


def test_law69_params_defaults():
    """Verify default initialization of Law69Params."""
    p = Law69Params()
    assert p.id == 1
    assert p.rho0 == 1.0
    assert p.rhor == 1.0
    assert p.law_id == 1
    assert p.fct_id == 0
    assert p.nu == 0.495
    assert p.fscale == 1.0
    assert p.nip == 2
    assert p.fct_id1 == 0
    assert len(p.mu) == 2
    assert len(p.alpha) == 2
    assert p.tenscut == 1e20
    assert p.icheck == -3
    assert p.gmax > 0.0
    assert np.isclose(p.g0, p.gmax / 2.0)
    assert np.isclose(p.E, p.gmax * (1.0 + p.nu))
    assert np.isclose(p.K, p.rbulk)
    assert np.isclose(p.G, p.g0)


def test_law69_params_properties_and_dict():
    """Verify dictionary representation and aliases."""
    p = Law69Params(id=42, rho0=1.1, nu=0.48, mu=np.array([20.0, -5.0]), alpha=np.array([2.0, -2.0]))
    d = p.params
    assert d["id"] == 42
    assert d["rho0"] == 1.1
    assert d["nu"] == 0.48
    assert d["K"] == p.rbulk
    assert d["G"] == p.g0
    assert np.allclose(d["mu"], [20.0, -5.0])
    assert np.allclose(d["alpha"], [2.0, -2.0])


# ============================================================================
# 2. Curve Fitting Tests (fit_law69_curve)
# ============================================================================


def test_fit_law69_curve_mooney_rivlin():
    """Test linear least squares fitting for Mooney-Rivlin (law_id=2)."""
    # Ground truth: mu_1 = 15.0, mu_2 = -5.0, alpha_1 = 2.0, alpha_2 = -2.0
    # C10 = 7.5, C01 = 2.5
    mu_true = np.array([15.0, -5.0])
    al_true = np.array([2.0, -2.0])

    eps = np.linspace(0.01, 0.50, 50)
    lam = 1.0 + eps
    # Analytical Mooney-Rivlin engineering stress
    sig_eng = mu_true[0] * (lam - lam ** (-2.0)) + mu_true[1] * (lam ** (-3.0) - 1.0)

    mu_fit, al_fit, gmax, g0, rbulk, E = fit_law69_curve(eps, sig_eng, law_id=2, nu=0.495)

    assert np.allclose(mu_fit, mu_true, rtol=1e-5, atol=1e-5)
    assert np.allclose(al_fit, al_true)
    expected_gs = float(np.sum(mu_true * al_true))
    assert np.isclose(gmax, expected_gs)
    assert np.isclose(g0, expected_gs / 2.0)
    expected_k = expected_gs * (1.0 + 0.495) / (3.0 * (1.0 - 2.0 * 0.495))
    assert np.isclose(rbulk, expected_k)
    assert np.isclose(E, expected_gs * (1.0 + 0.495))


def test_fit_law69_curve_ogden():
    """Test nonlinear least squares fitting for Ogden (law_id=1)."""
    mu_true = np.array([8.0, 1.5])
    al_true = np.array([1.8, -2.2])

    eps = np.linspace(0.02, 0.40, 40)
    lam = 1.0 + eps
    sig_eng = np.zeros_like(lam)
    for m, a in zip(mu_true, al_true):
        sig_eng += m * (lam ** (a - 1.0) - lam ** (-0.5 * a - 1.0))

    mu_fit, al_fit, gmax, g0, rbulk, E = fit_law69_curve(eps, sig_eng, law_id=1, n_pair=2, nu=0.495)

    # Verify fitted curve reproduces the stress data closely
    pred_sig = np.zeros_like(lam)
    for m, a in zip(mu_fit, al_fit):
        pred_sig += m * (lam ** (a - 1.0) - lam ** (-0.5 * a - 1.0))

    rel_err = np.linalg.norm(pred_sig - sig_eng) / np.linalg.norm(sig_eng)
    assert rel_err < 1e-3
    assert gmax > 0.0
    assert g0 > 0.0
    assert rbulk > 0.0


def test_fit_law69_curve_auto():
    """Test automatic fitting (law_id=-1) selecting suitable hyperelastic model."""
    mu_true = np.array([12.0, -3.0])
    eps = np.linspace(0.05, 0.35, 30)
    lam = 1.0 + eps
    sig_eng = mu_true[0] * (lam - lam ** (-2.0)) + mu_true[1] * (lam ** (-3.0) - 1.0)

    mu_fit, al_fit, gmax, g0, rbulk, E = fit_law69_curve(eps, sig_eng, law_id=-1, n_pair=2)

    pred_sig = np.zeros_like(lam)
    for m, a in zip(mu_fit, al_fit):
        pred_sig += m * (lam ** (a - 1.0) - lam ** (-0.5 * a - 1.0))

    rel_err = np.linalg.norm(pred_sig - sig_eng) / np.linalg.norm(sig_eng)
    assert rel_err < 1e-3


# ============================================================================
# 3. Factory Function Tests (build_law69)
# ============================================================================


def test_build_law69_direct():
    """Verify build_law69 with explicit arguments."""
    mat = build_law69(
        id=10,
        rho0=2.0,
        law_id=1,
        nu=0.49,
        mu=[15.0, -2.0],
        alpha=[2.5, -2.0],
    )
    assert mat.id == 10
    assert mat.rho0 == 2.0
    assert mat.rhor == 2.0
    assert mat.nu == 0.49
    assert mat.nordre == 2
    assert np.allclose(mat.mu, [15.0, -2.0])
    assert np.allclose(mat.alpha, [2.5, -2.0])
    assert mat.gmax == 15.0 * 2.5 + (-2.0) * (-2.0)
    assert mat.g0 == mat.gmax / 2.0


def test_build_law69_from_curve_data():
    """Verify build_law69 triggers fit_law69_curve when curve arrays provided."""
    eps = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    lam = 1.0 + eps
    sig = 10.0 * (lam - lam ** (-2.0)) - 2.0 * (lam ** (-3.0) - 1.0)

    mat = build_law69(
        id=5,
        rho0=1.2,
        law_id=2,
        curve_strain=eps,
        curve_stress=sig,
    )
    assert mat.id == 5
    assert mat.nordre == 2
    assert np.isclose(mat.mu[0], 10.0, rtol=1e-4)
    assert np.isclose(mat.mu[1], -2.0, rtol=1e-4)


def test_build_law69_from_dict():
    """Verify build_law69 unpacking from CFG attribute dictionary."""
    cfg_record = type("GenericRecord", (), {})()
    cfg_record.id = 7
    cfg_record.density = 1.05
    cfg_record.title = "RUBBER_LAW69"
    cfg_record.params = {
        "MAT_Iflag": 2,
        "MAT_NU": 0.495,
        "NIP": 2,
        "mu": np.array([12.0, -4.0]),
        "alpha": np.array([2.0, -2.0]),
    }
    mat = build_law69(cfg_record)
    assert mat.id == 7
    assert mat.rho0 == 1.05
    assert mat.law_id == 2
    assert mat.nu == 0.495
    assert np.allclose(mat.mu, [12.0, -4.0])


# ============================================================================
# 4. 3D Continuum Solid Kernel Tests (sigeps69_solid)
# ============================================================================


def test_solid_zero_strain_zero_stress():
    """Verify zero strain produces zero Cauchy stress in solids."""
    mat = build_law69(mu=[20.0, 5.0], alpha=[2.0, -2.0], nu=0.495)
    sig = np.zeros(6, dtype=np.float64)
    eps = np.zeros(6, dtype=np.float64)

    sig_new, epsp_new, c = sigeps69_solid(mat, sig, eps=eps)
    assert np.allclose(sig_new, 0.0, atol=1e-12)
    assert np.isclose(epsp_new, 0.0)
    assert c > 0.0


def test_solid_uniaxial_tension_response():
    """Verify 3D solid kernel under uniaxial tension."""
    mat = build_law69(mu=[25.0, 5.0], alpha=[2.0, -2.0], nu=0.495)
    eps_xx = 0.05
    # Nearly incompressible lateral contraction
    eps_yy = -0.495 * eps_xx
    eps_zz = -0.495 * eps_xx
    eps = np.array([eps_xx, eps_yy, eps_zz, 0.0, 0.0, 0.0])
    sig = np.zeros(6, dtype=np.float64)

    sig_new, _, c = sigeps69_solid(mat, sig, eps=eps)
    assert sig_new[0] > 0.0  # Positive tensile stress along x
    assert abs(sig_new[1]) < 0.1 * sig_new[0]  # Lateral stresses are close to zero
    assert abs(sig_new[2]) < 0.1 * sig_new[0]
    assert np.allclose(sig_new[3:], 0.0, atol=1e-12)  # No shears
    assert c > 0.0


def test_solid_vectorization_multi_element():
    """Verify solid kernel vectorizes cleanly across multiple elements (NEL > 1)."""
    mat = build_law69(mu=[18.0, 4.0], alpha=[2.0, -2.0], nu=0.49)
    nel = 8
    sig = np.zeros((nel, 6), dtype=np.float64)
    eps = np.zeros((nel, 6), dtype=np.float64)

    strains = np.linspace(0.01, 0.15, nel)
    for i in range(nel):
        eps[i, 0] = strains[i]
        eps[i, 1] = -0.49 * strains[i]
        eps[i, 2] = -0.49 * strains[i]

    sig_new, epsp_new, c = sigeps69_solid(mat, sig, eps=eps)
    assert sig_new.shape == (nel, 6)
    assert epsp_new.shape == (nel,)
    assert c.shape == (nel,)
    # Stress and sound speed should increase monotonically with strain
    assert np.all(np.diff(sig_new[:, 0]) > 0.0)
    assert np.all(c > 0.0)


def test_solid_sound_speed_stiffening():
    """Verify sound speed increases at large stretch due to material stiffening."""
    mat = build_law69(mu=[10.0, 2.0], alpha=[3.0, -2.0], nu=0.49)
    c_rest = solid_sound_speed(mat)

    eps_large = np.array([0.40, -0.20, -0.20, 0.0, 0.0, 0.0])
    c_stretched = solid_sound_speed(mat, eps=eps_large)

    assert c_stretched > c_rest


def test_solid_tenscut():
    """Verify tensile cut-off zeroes stress and flags element deletion."""
    mat = build_law69(mu=[50.0], alpha=[2.0], tenscut=10.0)
    eps = np.array([0.30, -0.15, -0.15, 0.0, 0.0, 0.0])
    sig = np.zeros(6, dtype=np.float64)
    off = np.array([1.0])

    sig_new, _, _ = sigeps69_solid(mat, sig, eps=eps, off=off)
    assert np.allclose(sig_new, 0.0)
    assert off[0] == 0.0


def test_solid_anti_buckling():
    """Verify anti-buckling check executes when RBULK > 24*GMAX under compression."""
    # Choose rbulk >> 24 * gmax
    mat = build_law69(mu=[1.0], alpha=[2.0], rbulk=500.0, nu=0.499)
    assert mat.rbulk > 24.0 * mat.gmax

    # Large compression amin < 0.2
    # lambda = 0.15 in engineering strain
    eps_comp = np.array([-0.85, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig = np.zeros(6, dtype=np.float64)

    sig_new, _, c = sigeps69_solid(mat, sig, eps=eps_comp, ismstr=1)
    assert sig_new[0] < 0.0
    assert c > 0.0


# ============================================================================
# 5. 2D Shell Plane-Stress Kernel Tests (sigeps69c_shell)
# ============================================================================


def test_shell_zero_strain_zero_stress():
    """Verify zero strain produces zero stress in shells."""
    mat = build_law69(mu=[20.0, 5.0], alpha=[2.0, -2.0], nu=0.495)
    sig = np.zeros(3, dtype=np.float64)
    eps = np.zeros(3, dtype=np.float64)

    sig_new, epsp_new, c = sigeps69c_shell(mat, sig, eps=eps, return_sound_speed=True)
    assert np.allclose(sig_new, 0.0, atol=1e-12)
    assert np.isclose(epsp_new, 0.0)
    assert c > 0.0


def test_shell_plane_stress_uniaxial():
    """Verify shell kernel plane-stress solve and lambda_3 update."""
    mat = build_law69(mu=[15.0, 3.0], alpha=[2.0, -2.0], nu=0.495)
    eps_xx = 0.10
    eps = np.array([eps_xx, -0.048, 0.0])  # in-plane [xx, yy, xy]
    sig = np.zeros(3, dtype=np.float64)
    uvar = np.zeros((1, 9), dtype=np.float64)
    uvar[0, 2] = 1.0  # Initial out-of-plane stretch lambda_3

    sig_new, _, c = sigeps69c_shell(mat, sig, eps=eps, uvar=uvar, return_sound_speed=True)
    assert sig_new[0] > 0.0  # Positive in-plane tension
    assert uvar[0, 2] < 1.0  # Thickness should shrink under in-plane stretch
    assert c > 0.0


def test_shell_thickness_and_shear_update():
    """Verify shell thickness and transverse shear updates."""
    mat = build_law69(mu=[12.0, 2.0], alpha=[2.0, -2.0], nu=0.495)
    sig = np.zeros(5, dtype=np.float64)
    deps = np.array([0.05, -0.02, 0.0, 0.01, 0.02])  # [xx, yy, xy, yz, zx]
    uvar = np.zeros((1, 9), dtype=np.float64)
    uvar[0, 2] = 1.0
    thkn = np.array([2.0])
    thklyl = np.array([2.0])
    off = np.array([1.0])

    sig_new, _ = sigeps69c_shell(mat, sig, deps=deps, uvar=uvar, thkn=thkn, thklyl=thklyl, off=off)
    # Transverse shear: sig_new = sig_0 + G0 * deps
    assert np.isclose(sig_new[3], mat.g0 * deps[3])
    assert np.isclose(sig_new[4], mat.g0 * deps[4])
    # Thickness update should have been applied
    assert thkn[0] != 2.0


def test_shell_tenscut():
    """Verify tensile cut-off in shell elements."""
    mat = build_law69(mu=[40.0], alpha=[2.0], tenscut=5.0)
    eps = np.array([0.25, 0.0, 0.0])
    sig = np.zeros(3, dtype=np.float64)
    off = np.array([1.0])

    sig_new, _ = sigeps69c_shell(mat, sig, eps=eps, off=off)
    assert np.allclose(sig_new, 0.0)
    assert off[0] == 0.8  # FOUR_OVER_5 in sigeps69c.F


# ============================================================================
# 6. Algorithmic Consistent Tangent Tests (< 1e-4 accuracy vs FD)
# ============================================================================


def test_tangent_law69_solid_accuracy():
    """Verify tangent_law69_solid is accurate against central finite differences to < 1e-4."""
    mat = build_law69(mu=[16.0, 4.0], alpha=[2.0, -2.0], nu=0.49)
    eps = np.array([0.04, -0.015, -0.018, 0.01, 0.005, -0.008])

    # Consistent algorithmic tangent from module
    C_algo = tangent_law69_solid(mat, eps, h=1e-7)
    assert C_algo.shape == (6, 6)

    # Independent central finite differences with h = 1e-5
    h_test = 1e-5
    C_fd = np.zeros((6, 6), dtype=np.float64)
    dummy_sig = np.zeros(6, dtype=np.float64)

    for j in range(6):
        eps_p = eps.copy()
        eps_p[j] += h_test
        eps_m = eps.copy()
        eps_m[j] -= h_test

        sp, _, _ = sigeps69_solid(mat, dummy_sig, eps=eps_p)
        sm, _, _ = sigeps69_solid(mat, dummy_sig, eps=eps_m)

        C_fd[:, j] = (sp - sm) / (2.0 * h_test)

    max_diff = np.max(np.abs(C_algo - C_fd))
    rel_diff = max_diff / np.max(np.abs(C_algo))

    assert max_diff < 1e-4 or rel_diff < 1e-4


def test_tangent_law69_shell_accuracy():
    """Verify tangent_law69_shell is accurate against central finite differences to < 1e-4."""
    mat = build_law69(mu=[14.0, 3.0], alpha=[2.0, -2.0], nu=0.49)
    eps = np.array([0.03, -0.012, 0.008])

    C_algo = tangent_law69_shell(mat, eps, h=1e-7)
    assert C_algo.shape == (3, 3)

    # Independent central finite differences with h = 1e-5
    h_test = 1e-5
    C_fd = np.zeros((3, 3), dtype=np.float64)
    dummy_sig = np.zeros(3, dtype=np.float64)

    for j in range(3):
        eps_p = eps.copy()
        eps_p[j] += h_test
        eps_m = eps.copy()
        eps_m[j] -= h_test

        sp, _ = sigeps69c_shell(mat, dummy_sig, eps=eps_p, return_sound_speed=False)
        sm, _ = sigeps69c_shell(mat, dummy_sig, eps=eps_m, return_sound_speed=False)

        C_fd[:, j] = (sp - sm) / (2.0 * h_test)

    max_diff = np.max(np.abs(C_algo - C_fd))
    rel_diff = max_diff / np.max(np.abs(C_algo))

    assert max_diff < 1e-4 or rel_diff < 1e-4


# ============================================================================
# 7. Package Integration & Registry Tests
# ============================================================================


def test_registry_law69():
    """Verify LAW69 is registered under expected keys in MAT_PHYSICS_REGISTRY."""
    expected_keys = [69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC"]
    for key in expected_keys:
        assert key in MAT_PHYSICS_REGISTRY, f"Key {key} not found in MAT_PHYSICS_REGISTRY"
        builder = MAT_PHYSICS_REGISTRY[key]
        assert callable(builder)
        mat = builder(id=1, rho0=1.0)
        assert isinstance(mat, Law69Params)


def test_materials_package_dispatch():
    """Verify pyradioss.materials dispatch routines for law 69."""
    mat = build_law69(id=1, rho0=1.0, mu=[10.0, 2.0], alpha=[2.0, -2.0])
    # Attach law identifier for generic Material dispatch
    mat.law = 69
    mat.law_name = "LAW69"

    # 1. Solid update
    sig = np.zeros(6, dtype=np.float64)
    deps = np.array([0.02, -0.01, -0.01, 0.0, 0.0, 0.0])
    sig_out, epsp_out, c_out = materials.solid_update(mat, sig, deps=deps)
    assert sig_out[0] > 0.0
    assert c_out is not None and c_out > 0.0

    # 2. Shell update
    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.02, -0.01, 0.0])
    sig_sh_out, epsp_sh_out = materials.shell_update(mat, sig_sh, deps=deps_sh)
    assert sig_sh_out[0] > 0.0

    # 3. Sound speed
    c = materials.sound_speed(mat)
    assert c > 0.0

    # 4. Consistent solid tangent
    C = materials.consistent_solid_tangent(mat, sig=sig)
    assert C.shape == (1, 6, 6)
