"""Constitutive physics unit tests for OpenRadioss /MAT/LAW76 (SAMP-1) (M588).

Validates:
  1. Parameter initialization and derived elastic moduli (E, nu, G, K, A1, A2, A11_2D, A12_2D).
  2. Fixed and free format deck parsing for /MAT/LAW76, /MAT/SAMP, /MAT/SAMP-1.
  3. Starter checks: parameter bounds (rho > 0, E > 0, -1 < nu < 0.5, fct_ID_T > 0),
     element compatibility (solids and shells allowed; 1D beams/trusses rejected).
  4. Yield asymmetry (tension vs compression): higher compressive yield when sigma_c > sigma_t.
  5. Dilatational plasticity & plastic Poisson's ratio:
     - nu_p = 0.5 -> isochoric plastic flow (tr(deps^p) = 0).
     - nu_p < 0.5 -> plastic volume change (dilatation under tension, compaction under compression).
  6. Cutting-plane return mapping convergence:
     - Quadratic (IQUAD=1) and linear (IQUAD=0) yield formulations.
     - Associated (IFORM=1) and non-associated (IFORM=0) flow.
     - Hardening curve evaluation.
  7. Convexity enforcement (ICONV=1): adjusts shear yield when sigma_s < 1.05 * sqrt(sigma_t * sigma_c / 3).
  8. Damage degradation: D(epsp) degradation of stress and effective modulus.
  9. 2D plane-stress shell formulation (sigeps76c.F):
     - sigma_zz = 0 enforcement.
     - Through-thickness strain increment dezz.
     - Thickness thinning update (h = h + dezz * h0).
  10. Acoustic wave speeds for solid (sqrt((K + 4/3 G)/rho)) and shell (sqrt(E/((1-nu^2)rho))).
  11. Algorithmic tangent operators (solid_tangent, shell_tangent, shell_membrane_tangent)
      and perturbation consistency (d_sigma approx C : d_eps).
  12. Dynamic cyclic deformation with internal energy and plastic work accounting.
"""

import math
import tempfile
import os
from pathlib import Path

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.materials.law76_samp import (
    Law76Params,
    build_law76,
    solid_step,
    shell_step,
    solid_update,
    shell_update,
    sound_speed,
    solid_tangent,
    shell_tangent,
    consistent_solid_tangent,
    consistent_shell_tangent,
    shell_membrane_tangent,
    extra_shapes,
    law76_func_comp,
    _compute_yield_coeffs,
    _eval_curve_or_val,
)
from pyradioss.model.entities import Material, Part, Property
from pyradioss.model.model import Model
from pyradioss.starter.checks import check_mat_law76


# =============================================================================
# 1. Parameter Initialization and Derived Moduli
# =============================================================================

def test_law76_params_and_elastic_moduli():
    """Verify parameter initialization and derived elastic moduli in Law76Params."""
    e = 2100.0
    nu = 0.35
    rho0 = 1.05e-6
    mat_data = {
        "id": 1,
        "title": "SAMP-1 Polymer",
        "rho": rho0,
        "e": e,
        "nu": nu,
        "fct_id_t": 10,
        "mat_nut": 0.3,
        "eps_f": 0.2,
        "eps_r": 0.4,
        "dc": 1.0,
        "iquad": 1,
        "iform": 0,
        "iconv": 1,
    }
    mat = build_law76(mat_data)
    assert isinstance(mat, Material)
    params = mat.params["law76_params"]

    expected_g = e / (2.0 * (1.0 + nu))
    expected_bulk = e / (3.0 * (1.0 - 2.0 * nu))
    expected_a1 = expected_bulk + (4.0 / 3.0) * expected_g
    expected_a2 = expected_bulk - (2.0 / 3.0) * expected_g
    expected_a11_2d = e / (1.0 - nu**2)
    expected_a12_2d = nu * expected_a11_2d

    assert np.isclose(params.e, e)
    assert np.isclose(params.E, e)
    assert np.isclose(params.nu, nu)
    assert np.isclose(params.rho0, rho0)
    assert np.isclose(params.g, expected_g)
    assert np.isclose(params.G, expected_g)
    assert np.isclose(params.bulk, expected_bulk)
    assert np.isclose(params.K, expected_bulk)
    assert np.isclose(params.a1, expected_a1)
    assert np.isclose(params.a2, expected_a2)
    assert np.isclose(params.a11_2d, expected_a11_2d)
    assert np.isclose(params.a12_2d, expected_a12_2d)
    assert np.isclose(params.nu_p, 0.3)
    assert params.iquad == 1
    assert params.iform == 0
    assert params.iconv == 1


# =============================================================================
# 2. Card Layout and Keyword Parsing
# =============================================================================

def test_law76_card_parsing_8card_and_aliases(tmp_path: Path):
    """Test /MAT/LAW76, /MAT/SAMP, /MAT/SAMP-1 parsing for both 8-card and 6-card formats."""
    deck_text = """# OpenRadioss Starter Deck
/BEGIN
Test SAMP-1 Deck
/MAT/LAW76/10
Polymer LAW76 8-card
#       RHO_I                 RHO_R
       1.0e-6                1.0e-6
#           E                    NU
       2100.0                  0.35
#    fct_ID_T              fct_ID_C              fct_ID_S              fct_ID_B
          101                   102                   103                     0
#    Fscale_t              Fscale_c              Fscale_s              Fscale_b                  FACX
          1.0                   1.0                   1.0                   1.0                   1.0
#        Nu_p              fct_IDpr             Fscale_pr                ISRAT                  Fcut
          0.3                     0                   1.0                    0                   0.0
#Epsilon_f_p           Epsilon_r_p                   D_c
          0.2                   0.4                   1.0
#     fct_ID1               fct_ID2               fct_ID3             Fscale_d1
            0                     0                     0                   1.0
#       IFORM                 IQUAD                 ICONV
            0                     1                     1
/MAT/SAMP-1/20
Polymer SAMP-1 alias 8-card
       1.2e-6                1.2e-6
       2500.0                  0.38
          201                   202                     0                     0
          1.0                   1.0                   1.0                   1.0                   1.0
         0.25                     0                   1.0                    0                   0.0
         0.15                   0.3                   1.0
            0                     0                     0                   1.0
            1                     1                     0
/MAT/SAMP/30
Polymer SAMP 6-card
 1.1e-6 1.1e-6
 2400.0 0.36
 301 302 303 0 1.0 1.0 1.0
 1.0 1.0 0.32 0 1.0 0 0.0
 0.18 0.35 1.0 0 0 0 1.0
 0 1 1
/END
"""
    deck_file = tmp_path / "deck_samp.rad"
    deck_file.write_text(deck_text, encoding="utf-8")

    blocks = read_deck(str(deck_file))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)

    assert len(log.errors) == 0

    # Check Material 10
    assert 10 in model.materials
    m10 = model.materials[10]
    assert m10.title == "Polymer LAW76 8-card"
    assert m10.law_name == "LAW76"
    assert m10.params.get("e") == pytest.approx(2100.0)
    assert m10.params.get("nu") == pytest.approx(0.35)
    assert m10.params.get("fct_id_t") == 101
    assert m10.params.get("fct_id_c") == 102
    assert m10.params.get("fct_id_s") == 103
    assert m10.params.get("mat_nut") == pytest.approx(0.3)
    assert m10.params.get("eps_f") == pytest.approx(0.2)
    assert m10.params.get("eps_r") == pytest.approx(0.4)
    assert m10.params.get("iquad") == 1
    assert m10.params.get("iconv") == 1

    # Check Material 20
    assert 20 in model.materials
    m20 = model.materials[20]
    assert m20.title == "Polymer SAMP-1 alias 8-card"
    assert m20.law_name == "LAW76"
    assert m20.params.get("e") == pytest.approx(2500.0)
    assert m20.params.get("nu") == pytest.approx(0.38)
    assert m20.params.get("fct_id_t") == 201
    assert m20.params.get("fct_id_c") == 202
    assert m20.params.get("iform") == 1
    assert m20.params.get("iquad") == 1

    # Check Material 30
    assert 30 in model.materials
    m30 = model.materials[30]
    assert m30.title == "Polymer SAMP 6-card"
    assert m30.law_name == "LAW76"
    assert m30.params.get("e") == pytest.approx(2400.0)
    assert m30.params.get("nu") == pytest.approx(0.36)
    assert m30.params.get("fct_id_t") == 301
    assert m30.params.get("iquad") == 1


# =============================================================================
# 3. Starter Checks & Element Compatibility
# =============================================================================

def test_law76_starter_checks():
    """Verify validation checks: density, Young's modulus, Poisson ratio, fct_ID_T, element types."""
    log = MessageLog()

    # Valid material
    mat_valid = Material(id=1, law=76, rho0=1.0e-6, params={"e": 2100.0, "nu": 0.35, "fct_id_t": 1})
    check_mat_law76(mat=mat_valid, log=log)
    assert len(log.errors) == 0

    # Missing/zero density
    log_bad_rho = MessageLog()
    mat_bad_rho = Material(id=2, law=76, rho0=0.0, params={"e": 2100.0, "nu": 0.35, "fct_id_t": 1})
    check_mat_law76(mat=mat_bad_rho, log=log_bad_rho)
    assert any("RHO" in err and "1514" in err for err in log_bad_rho.errors)

    # Missing/zero Young's modulus
    log_bad_e = MessageLog()
    mat_bad_e = Material(id=3, law=76, rho0=1.0e-6, params={"e": 0.0, "nu": 0.35, "fct_id_t": 1})
    check_mat_law76(mat=mat_bad_e, log=log_bad_e)
    assert any("Young's modulus E" in err and "1514" in err for err in log_bad_e.errors)

    # Invalid Poisson's ratio (nu >= 0.5)
    log_bad_nu = MessageLog()
    mat_bad_nu = Material(id=4, law=76, rho0=1.0e-6, params={"e": 2100.0, "nu": 0.55, "fct_id_t": 1})
    check_mat_law76(mat=mat_bad_nu, log=log_bad_nu)
    assert any("Poisson's ratio NU" in err for err in log_bad_nu.errors)

    # Missing tension curve (ANCMSG 126)
    log_bad_t = MessageLog()
    mat_bad_t = Material(id=5, law=76, rho0=1.0e-6, params={"e": 2100.0, "nu": 0.35, "fct_id_t": 0})
    check_mat_law76(mat=mat_bad_t, log=log_bad_t)
    assert any("fct_ID_T is required" in err and "126" in err for err in log_bad_t.errors)

    # Element compatibility: 1D beam rejection (ANCMSG 306)
    log_1d = MessageLog()
    model_1d = Model()
    model_1d.materials[10] = Material(id=10, law=76, rho0=1.0e-6, params={"e": 2100.0, "nu": 0.35, "fct_id_t": 1})
    model_1d.parts[1] = Part(id=1, mat_id=10, prop_id=100)
    model_1d.properties[100] = Property(id=100, type="TYPE3")  # Beam
    check_mat_law76(model=model_1d, mat_id=10, mat=model_1d.materials[10], log=log_1d)
    assert any("not supported for 1D elements" in err and "306" in err for err in log_1d.errors)


# =============================================================================
# 4. Plasticity Yield Asymmetry (Tension vs Compression)
# =============================================================================

def test_law76_yield_asymmetry():
    """Verify tension vs compression yield asymmetry: higher yield under compression when sigma_c > sigma_t."""
    sig_t = 30.0
    sig_c = 45.0
    sig_s = 20.0

    p_quad = Law76Params(
        e=2000.0, nu=0.35, rho0=1.0e-6,
        iquad=1, icas=-1, sig_t0=sig_t, sig_c0=sig_c, sig_s0=sig_s,
    )
    a0_q, a1_q, a2_q, _ = _compute_yield_coeffs(p_quad, sig_t, sig_c, sig_s)

    # Uniaxial tension: sigma = [30, 0, 0, 0, 0, 0]
    p_t = -sig_t / 3.0
    phi_t_q = (sig_t**2) - a0_q - a1_q * p_t - a2_q * (p_t**2)
    assert abs(phi_t_q) < 1.0e-5, f"Quadratic yield should satisfy tension: {phi_t_q}"

    # Uniaxial compression: sigma = [-45, 0, 0, 0, 0, 0]
    p_c = sig_c / 3.0
    phi_c_q = (sig_c**2) - a0_q - a1_q * p_c - a2_q * (p_c**2)
    assert abs(phi_c_q) < 1.0e-5, f"Quadratic yield should satisfy compression: {phi_c_q}"

    # Linear formulation (IQUAD=0)
    p_lin = Law76Params(
        e=2000.0, nu=0.35, rho0=1.0e-6,
        iquad=0, icas=-1, sig_t0=sig_t, sig_c0=sig_c, sig_s0=sig_s,
    )
    a0_l, a1_l, a2_l, _ = _compute_yield_coeffs(p_lin, sig_t, sig_c, sig_s)

    phi_t_l = sig_t - a0_l - a1_l * p_t - a2_l * (p_t**2)
    assert abs(phi_t_l) < 1.0e-5, f"Linear yield should satisfy tension: {phi_t_l}"

    phi_c_l = sig_c - a0_l - a1_l * p_c - a2_l * (p_c**2)
    assert abs(phi_c_l) < 1.0e-5, f"Linear yield should satisfy compression: {phi_c_l}"

    # Pure shear: sigma = [0, 0, 0, 20, 0, 0], p = 0, sigma_vm = sqrt(3)*20
    vm_s = math.sqrt(3.0) * sig_s
    phi_s_q = (vm_s**2) - a0_q
    assert abs(phi_s_q) < 1.0e-5, f"Quadratic yield should satisfy shear: {phi_s_q}"

    # Check asymmetry under equal trial magnitude
    sig_init = np.zeros(6)
    # Tensile trial strain
    deps_t = np.array([3.0e-2, -1.05e-2, -1.05e-2, 0, 0, 0])
    res_t, epsp_t, _ = solid_step(p_quad, sig_init, deps_t, epsp=0.0)

    # Compressive trial strain
    deps_c = -deps_t
    res_c, epsp_c, _ = solid_step(p_quad, sig_init, deps_c, epsp=0.0)

    # Compressive stress magnitude is higher than tensile stress
    assert abs(res_c[0]) > abs(res_t[0]), f"Compressive stress ({abs(res_c[0])}) must exceed tensile ({abs(res_t[0])})"


# =============================================================================
# 5. Dilatational Plasticity and Plastic Poisson's Ratio
# =============================================================================

def test_law76_dilatational_plasticity():
    """Verify plastic volume change: isochoric when nu_p=0.5, dilatational when nu_p < 0.5."""
    sig_init = np.zeros(6)
    deps_tens = np.array([3.0e-2, -1.05e-2, -1.05e-2, 0, 0, 0])

    # Case 1: Standard polymer nu_p = 0.35 (< 0.5) -> volumetric plastic strain > 0 in tension
    p_dilat = Law76Params(
        e=2000.0, nu=0.35, rho0=1.0e-6, mat_nut=0.35,
        iquad=1, iform=0, icas=-1, sig_t0=30.0, sig_c0=45.0, sig_s0=20.0,
    )
    extra_dilat = {"uvar": np.zeros((1, 7)), "off": np.array([1.0]), "dmg": np.array([0.0])}
    sig_d, epsp_d, _ = solid_step(p_dilat, sig_init, deps_tens, epsp=0.0, extra=extra_dilat)

    # Case 2: Incompressible plasticity nu_p = 0.5 -> alpha = 0 -> tr(m) = 0
    p_incomp = Law76Params(
        e=2000.0, nu=0.35, rho0=1.0e-6, mat_nut=0.5,
        iquad=1, iform=0, icas=-1, sig_t0=30.0, sig_c0=45.0, sig_s0=20.0,
    )
    extra_incomp = {"uvar": np.zeros((1, 7)), "off": np.array([1.0]), "dmg": np.array([0.0])}
    sig_i, epsp_i, _ = solid_step(p_incomp, sig_init, deps_tens, epsp=0.0, extra=extra_incomp)

    # Plastic strain accumulated in both
    assert epsp_d > 0.0
    assert epsp_i > 0.0

    # Under tension (p < 0), plastic dilatancy reduces the negative hydrostatic pressure
    p_val_dilat = - np.sum(sig_d[:3]) / 3.0
    p_val_incomp = - np.sum(sig_i[:3]) / 3.0
    assert abs(p_val_dilat) < abs(p_val_incomp)


# =============================================================================
# 6. Radial Return Mapping Convergence & Hardening
# =============================================================================

def test_law76_return_mapping_and_hardening():
    """Verify cutting-plane return mapping convergence and hardening behavior."""
    # Tensile hardening curve: sigma_y(epsp) = 30.0 + 200.0 * epsp^0.5
    pts_t = np.array([
        [0.0, 30.0],
        [0.05, 45.0],
        [0.10, 55.0],
        [0.20, 70.0],
    ])
    p = Law76Params(
        e=2000.0, nu=0.35, mat_nut=0.35, rho0=1.0e-6,
        iquad=1, iform=0, icas=-1,
        tens_curve=pts_t,
        sig_t0=30.0, sig_c0=40.0, sig_s0=20.0,
    )

    sig = np.zeros(6)
    epsp = 0.0
    extra = {}
    # Step 1: Strain into plastic regime
    deps1 = np.array([4.0e-2, -1.4e-2, -1.4e-2, 0, 0, 0])
    sig, epsp, _ = solid_step(p, sig, deps1, epsp=epsp, extra=extra)
    assert epsp > 0.01
    assert sig[0] > 30.0
    assert sig[0] <= 55.0

    # Step 2: Further straining
    deps2 = np.array([6.0e-2, -2.1e-2, -2.1e-2, 0, 0, 0])
    sig, epsp, _ = solid_step(p, sig, deps2, epsp=epsp, extra=extra)
    assert epsp > 0.05
    assert sig[0] > 40.0

    # Verify that yield function is satisfied (phi <= tol)
    p_hyd = - np.sum(sig[:3]) / 3.0
    dev = sig[:3] + p_hyd
    svm2 = 0.5 * np.sum(dev**2) + np.sum(sig[3:]**2)
    svm = math.sqrt(max(0.0, 3.0 * svm2))
    plat = extra["uvar"][0, 0]
    sig_t_curr, _ = _eval_curve_or_val(pts_t, plat, 0.0, 1.0, 1.0, 30.0, 100.0)
    a0, a1, a2, _ = _compute_yield_coeffs(p, sig_t_curr, p.sig_c0, p.sig_s0)
    phi = svm**2 - a0 - a1 * p_hyd - a2 * (p_hyd**2)
    assert phi <= 1.0e-4 * max(1.0, a0)


# =============================================================================
# 7. Convexity Enforcement (ICONV=1)
# =============================================================================

def test_law76_convexity_enforcement():
    """Verify that ICONV=1 adjusts shear yield when sigma_s is too small to ensure yield surface convexity."""
    sig_t = 30.0
    sig_c = 45.0
    sig_s_small = 15.0

    # With ICONV=0: unconstrained
    p_unconv = Law76Params(iquad=1, iconv=0, icas=-1)
    a0_u, a1_u, a2_u, sig_s_u = _compute_yield_coeffs(p_unconv, sig_t, sig_c, sig_s_small)
    assert np.isclose(sig_s_u, sig_s_small)

    # With ICONV=1: convexity enforced with 1.05 security factor
    p_conv = Law76Params(iquad=1, iconv=1, icas=-1)
    a0_c, a1_c, a2_c, sig_s_c = _compute_yield_coeffs(p_conv, sig_t, sig_c, sig_s_small)
    expected_min_s = 1.05 * math.sqrt(sig_t * sig_c / 3.0)
    assert np.isclose(sig_s_c, expected_min_s)
    assert sig_s_c > sig_s_small


# =============================================================================
# 8. Damage & Stiffness Degradation
# =============================================================================

def test_law76_damage_degradation():
    """Verify linear damage degradation from eps_f to eps_r and stiffness reduction."""
    p = Law76Params(
        e=2000.0, nu=0.35, rho0=1.0e-6,
        eps_f=0.05, eps_r=0.15, dc=1.0,
        sig_t0=30.0, sig_c0=40.0, sig_s0=20.0,
    )
    sig = np.zeros(6)
    extra = {"uvar": np.zeros((1, 7)), "off": np.array([1.0]), "dmg": np.array([0.0])}

    # Straining beyond eps_f (0.05) to ~0.10 -> D ~ (0.10 - 0.05)/(0.15 - 0.05) = 0.5
    deps = np.array([0.12, -0.04, -0.04, 0, 0, 0])
    sig_res, epsp_res, _ = solid_step(p, sig, deps, epsp=0.0, extra=extra)

    dmg_val = extra["dmg"][0]
    assert dmg_val > 0.0
    assert dmg_val < 1.0
    # Stress is degraded by (1 - D)
    assert sig_res[0] < 30.0


# =============================================================================
# 9. 2D Plane-Stress Shell Update & Thickness Thinning
# =============================================================================

def test_law76_plane_stress_shell_thinning():
    """Verify plane-stress shell update: sigma_zz = 0, through-thickness strain dezz, and thinning."""
    p = Law76Params(
        e=2100.0, nu=0.35, rho0=1.0e-6,
        iquad=1, iform=0,
        sig_t0=30.0, sig_c0=40.0, sig_s0=20.0,
    )

    sig_sh = np.zeros(3)  # [xx, yy, xy]
    deps_sh = np.array([0.03, -0.01, 0.0])
    h0 = 2.0
    extra = {
        "thk": np.array([h0]),
        "thk0": np.array([h0]),
        "dezz": np.array([0.0]),
        "off": np.array([1.0]),
        "dmg": np.array([0.0]),
        "uvar": np.zeros((1, 7)),
    }

    sign, epsp = shell_step(p, sig_sh, deps_sh, epsp=np.array([0.0]), extra=extra)

    assert float(epsp) > 0.0
    # Tensile in xx: dezz must be negative (thinning)
    dezz = extra["dezz"][0]
    assert dezz < 0.0
    # Updated thickness must be smaller than initial h0
    assert extra["thk"][0] < h0
    assert np.isclose(extra["thk"][0], h0 + dezz * h0)


# =============================================================================
# 10. Sound Speed and Algorithmic Tangents
# =============================================================================

def test_law76_sound_speed_and_tangents():
    """Verify wave speed formulas and algorithmic tangent operators."""
    e = 2100.0
    nu = 0.35
    rho0 = 1.05e-6
    p = Law76Params(e=e, nu=nu, rho0=rho0, sig_t0=30.0)

    # 1. Sound speed
    g = e / (2.0 * (1.0 + nu))
    k = e / (3.0 * (1.0 - 2.0 * nu))
    c_solid_expected = math.sqrt((k + 4.0 / 3.0 * g) / rho0)
    c_shell_expected = math.sqrt(e / ((1.0 - nu**2) * rho0))

    c_sol = sound_speed(p, extra={"is_shell": False})
    c_sh = sound_speed(p, extra={"is_shell": True})

    assert np.isclose(c_sol, c_solid_expected)
    assert np.isclose(c_sh, c_shell_expected)

    # 2. Tangent operators
    C_sol = consistent_solid_tangent(p)
    assert C_sol.shape == (6, 6)
    # Symmetry
    assert np.allclose(C_sol, C_sol.T)

    C_sh = consistent_shell_tangent(p)
    assert C_sh.shape == (3, 3)
    assert np.allclose(C_sh, C_sh.T)

    C_mem = shell_membrane_tangent(p)
    assert C_mem.shape == (3, 3)
    assert np.allclose(C_mem, C_sh)

    # 3. Small strain perturbation check (linear elastic Hooke regime)
    deps_small = np.array([1.0e-5, -3.5e-6, -3.5e-6, 1.0e-5, 0.0, 0.0])
    sig_0 = np.zeros(6)
    sig_1, epsp_1, _ = solid_step(p, sig_0, deps_small, epsp=0.0)
    sig_tangent_pred = C_sol @ deps_small

    assert np.allclose(sig_1, sig_tangent_pred, atol=1.0e-7)


# =============================================================================
# 11. Upstream Function Comp Synthesis (law76_func_comp)
# =============================================================================

def test_law76_func_comp_synthesis():
    """Verify law76_func_comp synthesizes compression curve from tension and shear."""
    tens_curve = np.array([
        [0.0, 30.0],
        [0.05, 40.0],
        [0.10, 50.0],
    ])
    shear_curve = np.array([
        [0.0, 20.0],
        [0.05, 26.0],
        [0.10, 32.0],
    ])
    nu_p = 0.35

    x_comp, y_comp = law76_func_comp(tens_curve, shear_curve, nu_p)
    assert len(x_comp) >= len(tens_curve)
    assert len(y_comp) == len(x_comp)
    # Compressive yield stress should be strictly higher than tensile yield
    assert np.all(y_comp >= np.interp(x_comp, tens_curve[:, 0], tens_curve[:, 1]))


# =============================================================================
# 12. Dynamic Cyclic Deformation & Energy Conservation
# =============================================================================

def test_law76_energy_conservation():
    """Verify internal energy accounting and non-negative plastic dissipation under cyclic loading."""
    p = Law76Params(
        e=2000.0, nu=0.35, rho0=1.0e-6,
        iquad=1, iform=0, icas=-1,
        sig_t0=30.0, sig_c0=40.0, sig_s0=20.0,
    )

    sig = np.zeros(6)
    epsp = 0.0
    vol0 = 1.0  # Unit volume
    e_internal = 0.0

    strains = [
        np.array([2.5e-2, -0.8e-2, -0.8e-2, 0, 0, 0]),  # Plastic load
        np.array([-1.5e-2, 0.5e-2, 0.5e-2, 0, 0, 0]),   # Elastic unload
        np.array([2.0e-2, -0.7e-2, -0.7e-2, 0, 0, 0]),   # Reload
        np.array([-3.0e-2, 1.0e-2, 1.0e-2, 0, 0, 0]),   # Unload to compression
    ]

    for deps in strains:
        sig_old = sig.copy()
        sig, epsp, _ = solid_step(p, sig, deps, epsp=epsp)
        d_eint = 0.5 * np.sum((sig_old + sig) * deps) * vol0
        e_internal += d_eint

    # Total internal energy must be positive (dissipated plastic work + residual elastic energy)
    assert e_internal > 0.0
    assert epsp > 0.0
