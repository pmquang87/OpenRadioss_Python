"""
Unit tests for LAW40 (generalized Kelvin-Maxwell visco-elasticity, /MAT/KELVINMAX).

Tests:
  - 1D analytical Prony shear relaxation modulus G(t)
  - 1D step-strain relaxation stress in shear, deviatoric, and uniaxial modes
  - 1D tabulated Kelvin stress helper
  - Tabulated curve evaluation and loading/unloading resolution
  - 3D solid_update step-strain relaxation matching analytical ODE solution
  - 3D solid_update tabulated pressure loading and unloading branch selection
  - Consistent solid tangent properties (symmetry, dt=0 instantaneous, dt->inf relaxed, FD check)
  - Material physics registry registration
"""

from __future__ import annotations

import math
from types import SimpleNamespace
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials import law40_kelvinmax
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY


# ============================================================================
# 1. 1D Analytical Prony Modulus & Stress Relations
# ============================================================================

def test_prony_shear_modulus_analytical():
    G_inf = 10.0
    G_branches = [5.0, 3.0, 2.0]
    beta_branches = [1.0, 10.0, 50.0]

    # At t = 0: G(0) = G_inf + sum(G_branches)
    g0 = law40_kelvinmax.prony_shear_modulus(0.0, G_inf, G_branches, beta_branches)
    assert pytest.approx(G_inf + sum(G_branches)) == g0

    # Intermediate time t = 0.5 s
    t_val = 0.5
    expected = G_inf + sum(gj * math.exp(-bj * t_val) for gj, bj in zip(G_branches, beta_branches))
    g_mid = law40_kelvinmax.prony_shear_modulus(t_val, G_inf, G_branches, beta_branches)
    assert pytest.approx(expected, rel=1e-6) == g_mid

    # Asymptotic t -> inf: G(inf) = G_inf
    g_inf_res = law40_kelvinmax.prony_shear_modulus(100.0, G_inf, G_branches, beta_branches)
    assert pytest.approx(G_inf, rel=1e-5) == g_inf_res

    # Vectorized evaluation
    t_arr = np.array([0.0, 0.1, 1.0, 10.0])
    g_arr = law40_kelvinmax.prony_shear_modulus(t_arr, G_inf, G_branches, beta_branches)
    assert g_arr.shape == (4,)
    assert pytest.approx(G_inf + sum(G_branches)) == g_arr[0]
    assert pytest.approx(G_inf, rel=1e-4) == g_arr[3]


def test_kelvin_maxwell_prony_stress_modes():
    G_inf = 8.0
    G_branches = [4.0, 2.0]
    beta_branches = [2.0, 20.0]
    eps0 = 0.01

    # Shear mode: s(t) = 2 * G(t) * eps0
    s_shear_0 = law40_kelvinmax.kelvin_maxwell_prony_stress(
        0.0, eps0, G_inf, G_branches, beta_branches, mode="shear"
    )
    assert pytest.approx(2.0 * (G_inf + sum(G_branches)) * eps0) == s_shear_0

    s_shear_inf = law40_kelvinmax.kelvin_maxwell_prony_stress(
        100.0, eps0, G_inf, G_branches, beta_branches, mode="shear"
    )
    assert pytest.approx(2.0 * G_inf * eps0, rel=1e-4) == s_shear_inf

    # Deviatoric mode: s_11(t) = (4/3) * G(t) * eps0
    s_dev_0 = law40_kelvinmax.kelvin_maxwell_prony_stress(
        0.0, eps0, G_inf, G_branches, beta_branches, mode="deviatoric"
    )
    assert pytest.approx((4.0 / 3.0) * (G_inf + sum(G_branches)) * eps0) == s_dev_0

    # Uniaxial mode with specified bulk modulus K
    K_bulk = 50.0
    sig_uni_0 = law40_kelvinmax.kelvin_maxwell_prony_stress(
        0.0, eps0, G_inf, G_branches, beta_branches, mode="uniaxial", K=K_bulk
    )
    expected_uni_0 = (K_bulk + (4.0 / 3.0) * (G_inf + sum(G_branches))) * eps0
    assert pytest.approx(expected_uni_0) == sig_uni_0

    # Invalid mode check
    with pytest.raises(ValueError, match="Unknown mode"):
        law40_kelvinmax.kelvin_maxwell_prony_stress(
            0.0, eps0, G_inf, G_branches, beta_branches, mode="invalid_mode"
        )


def test_tabulated_kelvin_stress():
    xs = np.array([0.0, 0.05, 0.10, 0.20])
    ys = np.array([0.0, 10.0, 25.0, 60.0])
    curve = (xs, ys)

    # Pure curve response
    sig_c = law40_kelvinmax.tabulated_kelvin_stress(0.05, curve=curve, fscale=2.0)
    assert pytest.approx(20.0) == sig_c

    # Combined with viscoelastic instantaneous stiffness
    G_inf = 5.0
    G_branches = [10.0]
    beta_branches = [1.0]
    sig_comb = law40_kelvinmax.tabulated_kelvin_stress(
        0.05, G_inf=G_inf, G_branches=G_branches, beta_branches=beta_branches, curve=curve, fscale=1.0
    )
    # y(0.05) = 10.0, 2 * G_sum * 0.05 = 2 * 15 * 0.05 = 1.5 -> 11.5
    assert pytest.approx(11.5) == sig_comb


# ============================================================================
# 2. Curve Interpolation, Extrapolation & Resolution
# ============================================================================

def test_curve_interpolation_and_extrapolation():
    p = {
        "pc_x": np.array([0.0, 0.1, 0.2]),
        "pc_y": np.array([0.0, 10.0, 30.0]),
        "pc_s": np.array([100.0, 200.0]),
        "pcu_x": np.array([0.0, 0.1, 0.2]),
        "pcu_y": np.array([0.0, 5.0, 15.0]),
        "pcu_s": np.array([50.0, 100.0]),
    }

    # Internal loading point
    y_load, s_load = law40_kelvinmax._curve(p, 0.05, unload=False)
    assert pytest.approx(5.0) == y_load
    assert pytest.approx(100.0) == s_load

    # Extrapolated point beyond upper bound: 30.0 + 200.0 * (0.25 - 0.2) = 40.0
    y_ext, s_ext = law40_kelvinmax._curve(p, 0.25, unload=False)
    assert pytest.approx(40.0) == y_ext
    assert pytest.approx(200.0) == s_ext

    # Unloading curve selection
    y_unl, s_unl = law40_kelvinmax._curve(p, 0.05, unload=True)
    assert pytest.approx(2.5) == y_unl
    assert pytest.approx(50.0) == s_unl


def test_resolve_curve_functions():
    mat = law40_kelvinmax.build_law40({
        "MAT_BULK": 60.0,
        "MAT_GI": 10.0,
        "fct_id": 101,
        "fct_unload_id": 102,
    })

    fct1 = SimpleNamespace(
        x=np.array([0.0, 0.1, 0.2]),
        y=np.array([0.0, 5.0, 12.0]),
        slope=np.array([50.0, 70.0]),
    )
    fct2 = SimpleNamespace(
        x=np.array([0.0, 0.1, 0.2]),
        y=np.array([0.0, 2.0, 6.0]),
        slope=np.array([20.0, 40.0]),
    )
    model = SimpleNamespace(functions={101: fct1, 102: fct2})

    law40_kelvinmax.resolve(mat, model)
    assert "pc_x" in mat.params
    assert "pcu_x" in mat.params
    np.testing.assert_allclose(mat.params["pc_x"], fct1.x)
    np.testing.assert_allclose(mat.params["pcu_x"], fct2.x)


# ============================================================================
# 3. 3D Solid Stress Relaxation vs Analytical Solution
# ============================================================================

def test_solid_update_matches_prony_relaxation():
    G_inf = 5.0
    G_branches = [10.0, 4.0]
    beta_branches = [2.0, 20.0]
    K_bulk = 50.0

    mat = law40_kelvinmax.build_law40({
        "MAT_BULK": K_bulk,
        "MAT_GI": G_inf,
        "MAT_G0": G_branches[0],
        "MAT_DECAY": beta_branches[0],
        "MAT_G2": G_branches[1],
        "MAT_DECAY2": beta_branches[1],
        "rho": 1.0,
    })

    n = 1
    sig = np.zeros((n, 6))
    gamma_xy = 0.02
    extra = {
        "eps40": np.zeros((n, 6)),
        "uv40": np.zeros((n, 40)),
        "rho": np.full(n, mat.rho0),
    }

    # Step-strain state at t = 0+: eps40 set and internal branch stresses v_j0 = G_j * gamma_xy
    extra["eps40"][0, 3] = gamma_xy
    extra["uv40"][0, 13] = G_branches[0] * gamma_xy
    extra["uv40"][0, 19] = G_branches[1] * gamma_xy

    # Hold constant strain and relax for total time t_total
    t_total = 0.8
    n_steps = 400
    dt_step = t_total / n_steps
    for _ in range(n_steps):
        sig, _ = law40_kelvinmax.solid_update(
            mat, sig, np.zeros((n, 6)), dt=dt_step, extra=extra
        )

    # Analytical deviatoric stress for shear: s_xy(t) = 2 * G(t) * (gamma_xy / 2) = G(t) * gamma_xy
    g_expected = law40_kelvinmax.prony_shear_modulus(
        t_total, G_inf, G_branches, beta_branches
    )
    s_expected = g_expected * gamma_xy
    assert pytest.approx(s_expected, rel=1e-3) == sig[0, 3]


# ============================================================================
# 4. Tabulated Pressure Loading and Unloading
# ============================================================================

def test_solid_update_tabulated_pressure_loading_unloading():
    K_bulk = 50.0
    mat = law40_kelvinmax.build_law40({
        "MAT_BULK": K_bulk,
        "MAT_GI": 10.0,
        "fct_id": 1,
        "fct_unload_id": 2,
        "fscale": 1.0,
        "fscale_unload": 1.0,
        "itype": 0,
        "rho": 1.0,
    })

    mat.params["pc_x"] = np.array([0.0, 0.05, 0.10])
    mat.params["pc_y"] = np.array([0.0, 20.0, 50.0])
    mat.params["pc_s"] = np.array([400.0, 600.0])

    mat.params["pcu_x"] = np.array([0.0, 0.05, 0.10])
    mat.params["pcu_y"] = np.array([0.0, 10.0, 25.0])
    mat.params["pcu_s"] = np.array([200.0, 300.0])

    n = 1
    sig = np.zeros((n, 6))
    extra = {
        "eps40": np.zeros((n, 6)),
        "uv40": np.zeros((n, 40)),
        "rho": np.full(n, 1.0),
        "eps_max": np.zeros(n),
    }

    # Step 1: Volumetric compression (loading) to mu_v = 0.05 (deps = -0.05 / 3 on diagonal)
    deps_load = np.zeros((n, 6))
    deps_load[0, 0] = deps_load[0, 1] = deps_load[0, 2] = -0.05 / 3.0
    sig, c = law40_kelvinmax.solid_update(mat, sig, deps_load, dt=1e-4, extra=extra)

    # In compression, pressure P = 20.0, mean stress sig_v = -P = -20.0
    sig_v_load = (sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
    assert pytest.approx(-20.0, rel=1e-2) == sig_v_load
    assert pytest.approx(0.05, rel=1e-3) == extra["eps_max"][0]

    # Step 2: Unloading to mu_v = 0.025 (partial expansion)
    deps_unload = np.zeros((n, 6))
    deps_unload[0, 0] = deps_unload[0, 1] = deps_unload[0, 2] = +0.025 / 3.0
    sig, c = law40_kelvinmax.solid_update(mat, sig, deps_unload, dt=1e-4, extra=extra)

    # On unloading curve at mu_v = 0.025: y = 0.0 + 200.0 * 0.025 = 5.0 -> sig_v = -5.0
    sig_v_unload = (sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
    assert pytest.approx(-5.0, rel=1e-2) == sig_v_unload
    # Peak strain tracked remains 0.05
    assert pytest.approx(0.05, rel=1e-3) == extra["eps_max"][0]


# ============================================================================
# 5. Consistent Solid Tangent
# ============================================================================

def test_consistent_solid_tangent_instantaneous_and_relaxed():
    K_bulk = 80.0
    G_inf = 10.0
    G_branches = [6.0, 4.0]
    beta_branches = [5.0, 50.0]

    mat = law40_kelvinmax.build_law40({
        "MAT_BULK": K_bulk,
        "MAT_GI": G_inf,
        "MAT_G0": G_branches[0],
        "MAT_DECAY": beta_branches[0],
        "MAT_G2": G_branches[1],
        "MAT_DECAY2": beta_branches[1],
    })

    sig = np.zeros((1, 6))

    # Instantaneous limit: dt = 0 -> G_eff = G_inf + sum(G_branches) = 20.0
    C_inst = law40_kelvinmax.consistent_solid_tangent(mat, sig, dt=0.0)
    G_sum = G_inf + sum(G_branches)
    c11_inst = K_bulk + (4.0 / 3.0) * G_sum
    c12_inst = K_bulk - (2.0 / 3.0) * G_sum
    c44_inst = G_sum

    assert pytest.approx(c11_inst) == C_inst[0, 0, 0]
    assert pytest.approx(c12_inst) == C_inst[0, 0, 1]
    assert pytest.approx(c44_inst) == C_inst[0, 3, 3]

    # Relaxed limit: dt -> infinity -> G_eff = G_inf = 10.0
    C_relax = law40_kelvinmax.consistent_solid_tangent(mat, sig, dt=1e5)
    c11_relax = K_bulk + (4.0 / 3.0) * G_inf
    c12_relax = K_bulk - (2.0 / 3.0) * G_inf
    c44_relax = G_inf

    assert pytest.approx(c11_relax, rel=1e-4) == C_relax[0, 0, 0]
    assert pytest.approx(c12_relax, rel=1e-4) == C_relax[0, 0, 1]
    assert pytest.approx(c44_relax, rel=1e-4) == C_relax[0, 3, 3]

    # Major and minor symmetries: C_ijkl = C_jikl = C_ijlk = C_klij
    np.testing.assert_allclose(C_inst[0], C_inst[0].T, atol=1e-12)


def test_consistent_solid_tangent_finite_difference():
    mat = law40_kelvinmax.build_law40({
        "MAT_BULK": 50.0,
        "MAT_GI": 8.0,
        "MAT_G0": 4.0,
        "MAT_DECAY": 2.0,
    })

    sig0 = np.zeros((1, 6))
    dt = 1e-3
    C = law40_kelvinmax.consistent_solid_tangent(mat, sig0, dt=dt)

    # Test perturbation along each component k
    d_eps_mag = 1e-6
    for k in range(6):
        deps = np.zeros((1, 6))
        deps[0, k] = d_eps_mag
        extra = {
            "eps40": np.zeros((1, 6)),
            "uv40": np.zeros((1, 40)),
            "rho": np.full(1, mat.rho0),
        }
        sig_perturbed, _ = law40_kelvinmax.solid_update(
            mat, sig0.copy(), deps, dt=dt, extra=extra
        )
        delta_sig = sig_perturbed[0] - sig0[0]

        # For engineering shears (k=3,4,5), deps stores gamma = 2*eps
        # Tangent relates d_sig to d_gamma: d_sig_i = C_{ik} * d_gamma_k
        expected_delta = C[0, :, k] * d_eps_mag
        np.testing.assert_allclose(delta_sig, expected_delta, rtol=1e-3, atol=1e-9)


# ============================================================================
# 6. Material Physics Registry
# ============================================================================

def test_law40_registry_entry():
    assert "KELVINMAX" in MAT_PHYSICS_REGISTRY
    assert "LAW40" in MAT_PHYSICS_REGISTRY
    fn = MAT_PHYSICS_REGISTRY["LAW40"]
    mat = fn({"MAT_BULK": 40.0, "MAT_GI": 5.0})
    assert isinstance(mat, Material)
    assert mat.law == 40
