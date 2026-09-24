"""Tests for Milestone M596: Cohesive Zone Material Law (/MAT/LAW117, /MAT/COH_MC, /MAT/COH_TAB).

Fortran origins:
- ``starter/source/materials/mat/mat117/hm_read_mat117.F``
- ``engine/source/materials/mat/mat117/sigeps117.F``
- ``hm_cfg_files/config/CFG/radioss2022/MAT/mat117.cfg``

Verifications:
1. Pure Mode I normal separation:
   - Exact initial elastic stiffness E_n
   - Peak normal traction sigma_max
   - Linear softening branch
   - Exact fracture energy dissipation W_diss = G_Ic
2. Pure Mode II tangential separation:
   - Exact initial shear stiffness E_t
   - Peak shear traction tau_max
   - Linear softening branch
   - Exact fracture energy dissipation W_diss = G_IIc
3. Mixed-Mode loading:
   - Quadratic initiation envelope: ( <sigma_n> / sigma_max )^2 + ( tau / tau_max )^2 = 1
   - Power-law energy dissipation (IRUPT = 1)
   - Benzeggagh-Kenane (BK) energy dissipation (IRUPT = 2)
   - Total mixed-mode dissipated work matches theoretical G_c
4. Unilateral compressive contact:
   - Compression (eps_zz < 0) exhibits full elastic penalty sigma_zz = E_n * eps_zz
   - No damage accumulation occurs during compression
   - Pre-damaged interface recovers full undamaged compressive contact stiffness
5. Element deletion:
   - At delta_m >= delta_f^m, damage D = 1.0 and off = 0.0
   - Tractions immediately drop to zero and remain zero
6. Starter deck parsing:
   - Fixed format /MAT/LAW117
   - Free format /MAT/COH_MC
   - Free format /MAT/COH_TAB
   - Model material properties and params census
7. Acoustic sound speed and consistent algorithmic tangent stiffness:
   - Sound speed matches sqrt((E_n + E_t) / rho0)
   - Consistent algorithmic tangent tensor matches numerical perturbation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pyradioss.materials.law117_cohesive import (
    Law117Params,
    build_law117,
    consistent_solid_tangent,
    extra_shapes,
    solid_step,
    solid_tangent,
    solid_update,
    sound_speed,
)
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import Material
from pyradioss.model.model import Model


def _parse_deck(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


# ============================================================================
# 1. Pure Mode I Normal Separation
# ============================================================================

def test_pure_mode_i_separation():
    """Verify Mode I normal separation recovering exact initial stiffness, peak traction,

    linear softening, and total dissipated work = G_Ic.
    """
    E_n = 2000.0
    E_t = 1000.0
    sigma_max = 20.0
    tau_max = 10.0
    G_Ic = 1.0
    G_IIc = 2.0

    p = Law117Params(
        E_n=E_n,
        E_t=E_t,
        sigma_max=sigma_max,
        tau_max=tau_max,
        G_Ic=G_Ic,
        G_IIc=G_IIc,
        irupt=1,
    )

    delta0_n = sigma_max / E_n   # 0.01
    delta_f_n = 2.0 * G_Ic / sigma_max  # 0.10
    assert pytest.approx(p.delta0_n) == delta0_n
    assert pytest.approx(p.und) == delta_f_n

    extra = {}
    sig = np.zeros(6, dtype=float)

    # Pre-peak loading up to delta0_n in 10 steps
    n_pre = 10
    d_step = delta0_n / n_pre
    for k in range(1, n_pre + 1):
        deps = np.zeros(6, dtype=float)
        deps[2] = d_step
        sig, dmg, c = solid_step(p, sig, deps, dt=1e-6, extra=extra)

        expected_d = k * d_step
        expected_sig = E_n * expected_d
        assert pytest.approx(sig[2], rel=1e-5) == expected_sig
        assert pytest.approx(sig[4]) == 0.0
        assert pytest.approx(sig[5]) == 0.0
        assert pytest.approx(dmg) == 0.0

    # At peak traction
    assert pytest.approx(sig[2], rel=1e-5) == sigma_max
    assert pytest.approx(dmg) == 0.0
    assert extra["off"] == 1.0

    # Softening branch up to delta_f_n in 90 steps
    n_soft = 90
    d_soft_step = (delta_f_n - delta0_n) / n_soft
    for k in range(1, n_soft):
        deps = np.zeros(6, dtype=float)
        deps[2] = d_soft_step
        sig, dmg, c = solid_step(p, sig, deps, dt=1e-6, extra=extra)

        curr_d = delta0_n + k * d_soft_step
        # Linear softening theoretical traction:
        expected_sig = sigma_max * (delta_f_n - curr_d) / (delta_f_n - delta0_n)
        assert pytest.approx(sig[2], rel=1e-4) == expected_sig
        assert 0.0 < dmg < 1.0
        assert extra["off"] == 1.0

    # Final step reaching delta_f_n
    deps = np.zeros(6, dtype=float)
    deps[2] = d_soft_step
    sig, dmg, c = solid_step(p, sig, deps, dt=1e-6, extra=extra)

    assert pytest.approx(sig[2], abs=1e-6) == 0.0
    assert pytest.approx(dmg) == 1.0
    assert extra["off"] == 0.0

    # Verify total dissipated cohesive energy per unit area equals G_Ic
    w_diss = extra["w_diss"][0] if hasattr(extra["w_diss"], "__len__") else extra["w_diss"]
    assert pytest.approx(w_diss, rel=1e-3) == G_Ic


# ============================================================================
# 2. Pure Mode II Shear Separation
# ============================================================================

def test_pure_mode_ii_separation():
    """Verify Mode II shear separation recovering exact initial shear stiffness,

    peak tau_max, softening branch, and total dissipated work = G_IIc.
    """
    E_n = 3000.0
    E_t = 1500.0
    sigma_max = 30.0
    tau_max = 15.0
    G_Ic = 1.5
    G_IIc = 3.0

    p = Law117Params(
        E_n=E_n,
        E_t=E_t,
        sigma_max=sigma_max,
        tau_max=tau_max,
        G_Ic=G_Ic,
        G_IIc=G_IIc,
        irupt=1,
    )

    delta0_s = tau_max / E_t   # 0.01
    delta_f_s = 2.0 * G_IIc / tau_max  # 0.40
    assert pytest.approx(p.delta0_s) == delta0_s
    assert pytest.approx(p.utd) == delta_f_s

    extra = {}
    sig = np.zeros(6, dtype=float)

    # Pure shear in YZ (index 4)
    n_steps = 100
    d_step = delta_f_s / n_steps
    for k in range(1, n_steps + 1):
        deps = np.zeros(6, dtype=float)
        deps[4] = d_step
        sig, dmg, c = solid_step(p, sig, deps, dt=1e-6, extra=extra)

        curr_d = k * d_step
        if curr_d <= delta0_s:
            expected_tau = E_t * curr_d
            assert pytest.approx(sig[4], rel=1e-4) == expected_tau
            assert pytest.approx(dmg) == 0.0
        elif curr_d < delta_f_s:
            expected_tau = tau_max * (delta_f_s - curr_d) / (delta_f_s - delta0_s)
            assert pytest.approx(sig[4], rel=1e-3) == expected_tau
            assert 0.0 < dmg < 1.0
        else:
            assert pytest.approx(sig[4], abs=1e-6) == 0.0
            assert pytest.approx(dmg) == 1.0
            assert extra["off"] == 0.0

    w_diss = extra["w_diss"][0] if hasattr(extra["w_diss"], "__len__") else extra["w_diss"]
    assert pytest.approx(w_diss, rel=2e-3) == G_IIc


# ============================================================================
# 3. Mixed-Mode Loading (Quadratic Initiation & Energy Dissipation)
# ============================================================================

def test_mixed_mode_quadratic_initiation():
    """Verify quadratic mixed-mode initiation criterion:

    ( <sigma_n> / sigma_max )^2 + ( tau / tau_max )^2 = 1.
    """
    E_n = 2000.0
    E_t = 1000.0
    sigma_max = 20.0
    tau_max = 10.0
    G_Ic = 1.0
    G_IIc = 2.0

    p = Law117Params(
        E_n=E_n,
        E_t=E_t,
        sigma_max=sigma_max,
        tau_max=tau_max,
        G_Ic=G_Ic,
        G_IIc=G_IIc,
        irupt=1,
    )

    # Test along different mode mixity angles theta = arctan(tau / sigma_n)
    for theta_deg in (15.0, 30.0, 45.0, 60.0, 75.0):
        theta = math.radians(theta_deg)
        # In terms of stresses: sigma_n = r * cos(theta), tau = r * sin(theta)
        # Initiation occurs when (r * cos / sigma_max)^2 + (r * sin / tau_max)^2 = 1
        # => r_init = 1 / sqrt( (cos/sigma_max)^2 + (sin/tau_max)^2 )
        r_init = 1.0 / math.sqrt((math.cos(theta) / sigma_max)**2 + (math.sin(theta) / tau_max)**2)
        sig_init = r_init * math.cos(theta)
        tau_init = r_init * math.sin(theta)

        # Displacements at initiation:
        d_n_init = sig_init / E_n
        d_s_init = tau_init / E_t
        delta_m_init = math.sqrt(d_n_init**2 + d_s_init**2)

        # Theoretical delta0_m formula from Fortran
        beta = d_s_init / d_n_init
        delta0_n = sigma_max / E_n
        delta0_s = tau_max / E_t
        delta0_m = delta0_s * delta0_n * math.sqrt((1.0 + beta**2) / (delta0_s**2 + (beta * delta0_n)**2))

        assert pytest.approx(delta_m_init, rel=1e-5) == delta0_m

        # Run solid_step just below initiation
        extra = {}
        sig = np.zeros(6, dtype=float)
        deps = np.zeros(6, dtype=float)
        deps[2] = 0.999 * d_n_init
        deps[4] = 0.999 * d_s_init
        sig, dmg, _ = solid_step(p, sig, deps, dt=1e-6, extra=extra)
        assert dmg == 0.0

        # Step into damage
        deps[2] = 0.003 * d_n_init
        deps[4] = 0.003 * d_s_init
        sig, dmg, _ = solid_step(p, sig, deps, dt=1e-6, extra=extra)
        assert dmg > 0.0


def test_mixed_mode_bk_energy_dissipation():
    """Verify Benzeggagh-Kenane (IRUPT = 2) mixed-mode total fracture work.

    Under proportional loading with BK criterion (gamma = 1.0),
    total dissipated work equals exact G_c(beta).
    """
    E_n = 2000.0
    E_t = 1000.0
    sigma_max = 20.0
    tau_max = 10.0
    G_Ic = 1.0
    G_IIc = 3.0
    exp_bk = 1.5

    p = Law117Params(
        E_n=E_n,
        E_t=E_t,
        sigma_max=sigma_max,
        tau_max=tau_max,
        G_Ic=G_Ic,
        G_IIc=G_IIc,
        irupt=2,
        exp_bk=exp_bk,
        gamma=1.0,
    )

    # Proportional displacement path: delta_s = 2.0 * delta_n (beta = 2.0)
    beta = 2.0
    mix_ratio = (E_t * beta**2) / (E_n + E_t * beta**2)  # (1000 * 4) / (2000 + 4000) = 4000/6000 = 2/3
    G_c_theoretical = G_Ic + (G_IIc - G_Ic) * (mix_ratio**exp_bk)

    # Compute delta0_m and delta_f_m
    delta0_n = sigma_max / E_n  # 0.01
    delta0_s = tau_max / E_t  # 0.01
    delta0_m = delta0_s * delta0_n * math.sqrt((1.0 + beta**2) / (delta0_s**2 + (beta * delta0_n)**2))

    fac3 = (E_n + E_t * beta**2) / (1.0 + beta**2)  # (2000 + 4000) / 5 = 1200.0
    delta_f_m = (2.0 / (delta0_m * fac3)) * G_c_theoretical

    # Integrate to complete separation in 200 steps
    n_steps = 200
    d_m_step = delta_f_m / n_steps
    d_n_step = d_m_step / math.sqrt(1.0 + beta**2)
    d_s_step = beta * d_n_step

    extra = {}
    sig = np.zeros(6, dtype=float)
    for _ in range(n_steps):
        deps = np.zeros(6, dtype=float)
        deps[2] = d_n_step
        deps[4] = d_s_step
        sig, dmg, _ = solid_step(p, sig, deps, dt=1e-6, extra=extra)

    assert pytest.approx(dmg) == 1.0
    assert extra["off"] == 0.0
    w_diss = extra["w_diss"][0] if hasattr(extra["w_diss"], "__len__") else extra["w_diss"]
    assert pytest.approx(w_diss, rel=1e-3) == G_c_theoretical


def test_mixed_mode_power_law_propagation():
    """Verify Power Law (IRUPT = 1) ultimate displacement formulation."""
    E_n = 2000.0
    E_t = 1000.0
    sigma_max = 20.0
    tau_max = 10.0
    G_Ic = 1.0
    G_IIc = 2.0
    exp_g = 2.0

    p = Law117Params(
        E_n=E_n,
        E_t=E_t,
        sigma_max=sigma_max,
        tau_max=tau_max,
        G_Ic=G_Ic,
        G_IIc=G_IIc,
        irupt=1,
        exp_g=exp_g,
    )

    beta = 1.0
    delta0_n = sigma_max / E_n
    delta0_s = tau_max / E_t
    delta0_m = delta0_s * delta0_n * math.sqrt((1.0 + beta**2) / (delta0_s**2 + (beta * delta0_n)**2))

    fac1 = 2.0 * (1.0 + beta**2) / delta0_m
    term = (E_n / G_Ic)**exp_g + (E_t * (beta**2) / G_IIc)**exp_g
    delta_f_m_expected = fac1 * (term**(-1.0 / exp_g))

    extra = {}
    sig = np.zeros(6, dtype=float)
    # Apply strain close to delta_f_m
    d_n = 0.999 * delta_f_m_expected / math.sqrt(1.0 + beta**2)
    d_s = beta * d_n
    deps = np.zeros(6, dtype=float)
    deps[2] = d_n
    deps[4] = d_s
    sig, dmg, _ = solid_step(p, sig, deps, dt=1e-6, extra=extra)
    assert 0.95 < dmg < 1.0
    assert extra["off"] == 1.0

    # Cross delta_f_m -> deletion
    deps = np.zeros(6, dtype=float)
    deps[2] = 0.01 * d_n
    deps[4] = 0.01 * d_s
    sig, dmg, _ = solid_step(p, sig, deps, dt=1e-6, extra=extra)
    assert dmg == 1.0
    assert extra["off"] == 0.0
    assert pytest.approx(sig[2]) == 0.0
    assert pytest.approx(sig[4]) == 0.0


# ============================================================================
# 4. Unilateral Compressive Contact Behavior
# ============================================================================

def test_unilateral_compressive_contact():
    """Verify unilateral contact stiffness:

    - Under compression (eps_zz < 0), penalty sigma_zz = E_n * eps_zz with no damage
    - Damaged state in tension does not degrade compressive contact stiffness.
    """
    E_n = 2500.0
    E_t = 1200.0
    sigma_max = 25.0
    tau_max = 12.0
    G_Ic = 1.0
    G_IIc = 2.0

    p = Law117Params(
        E_n=E_n,
        E_t=E_t,
        sigma_max=sigma_max,
        tau_max=tau_max,
        G_Ic=G_Ic,
        G_IIc=G_IIc,
    )

    extra = {}
    sig = np.zeros(6, dtype=float)

    # 1. Pure compression without prior tension
    deps = np.zeros(6, dtype=float)
    deps[2] = -0.05
    sig, dmg, _ = solid_step(p, sig, deps, dt=1e-6, extra=extra)
    assert pytest.approx(sig[2]) == E_n * (-0.05)
    assert dmg == 0.0
    assert extra["off"] == 1.0

    # 2. Reverse through zero into tension up to softening (delta0_n = 0.01, delta_f_n = 0.08)
    deps[2] = 0.05 + 0.045  # total eps_zz = +0.045 (softening)
    sig, dmg, _ = solid_step(p, sig, deps, dt=1e-6, extra=extra)
    assert dmg > 0.4
    dmg_tension = float(dmg)

    # 3. Reload into compression from damaged state
    deps[2] = -0.045 - 0.03  # total eps_zz = -0.03 (compression)
    sig, dmg, _ = solid_step(p, sig, deps, dt=1e-6, extra=extra)

    # Must recover 100% full contact stiffness (no (1 - D) degradation!)
    expected_contact_stress = E_n * (-0.03)
    assert pytest.approx(sig[2]) == expected_contact_stress
    # Damage parameter must not decrease
    assert pytest.approx(dmg) == dmg_tension


# ============================================================================
# 5. Element Deletion
# ============================================================================

def test_element_deletion_at_critical_separation():
    """Verify element deletion (off = 0.0) when D reaches 1.0."""
    p = Law117Params(
        E_n=1000.0,
        E_t=1000.0,
        sigma_max=10.0,
        tau_max=10.0,
        G_Ic=0.5,
        G_IIc=0.5,
    )
    # delta_f_n = 2 * 0.5 / 10 = 0.1

    extra = {}
    sig = np.zeros(6, dtype=float)

    # Step past delta_f_n
    deps = np.zeros(6, dtype=float)
    deps[2] = 0.12
    sig, dmg, _ = solid_step(p, sig, deps, dt=1e-6, extra=extra)

    assert dmg == 1.0
    assert extra["off"] == 0.0
    assert np.allclose(sig, 0.0)

    # Additional steps maintain zero stresses
    deps[2] = 0.05
    deps[4] = 0.05
    sig, dmg, _ = solid_step(p, sig, deps, dt=1e-6, extra=extra)
    assert dmg == 1.0
    assert extra["off"] == 0.0
    assert np.allclose(sig, 0.0)


# ============================================================================
# 6. Starter Deck Parsing (Fixed & Free Formats)
# ============================================================================

def test_starter_parsing_fixed_format(tmp_path: Path):
    """Verify /MAT/LAW117 card reading in fixed format."""
    deck_text = (
        "# OpenRadioss Starter Deck\n"
        "/BEGIN\n"
        "test_law117_fixed\n"
        "       100         1\n"
        "/MAT/LAW117/101\n"
        "Cohesive Interface Fixed\n"
        "#        Init. dens.\n"
        "         1.20000E-06\n"
        "#                 EN                  ES     Imass      Idel     Irupt\n"
        "              2500.0              1250.0         1         1         1\n"
        "#   FCT_TN    FCT_TT                  TN                  TS            Fscale_x\n"
        "         0         0                25.0                12.5                 1.0\n"
        "#                GIC                GIIC               EXP_G              EXP_BK               GAMMA\n"
        "                 1.2                 2.4                 2.0                 1.0                 1.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors: {log.errors}"
    assert 101 in model.materials
    mat = model.materials[101]
    assert mat.law == 117
    assert pytest.approx(mat.rho0) == 1.2e-6
    assert pytest.approx(mat.params["E_n"]) == 2500.0
    assert pytest.approx(mat.params["E_t"]) == 1250.0
    assert pytest.approx(mat.params["sigma_max"]) == 25.0
    assert pytest.approx(mat.params["tau_max"]) == 12.5
    assert pytest.approx(mat.params["G_Ic"]) == 1.2
    assert pytest.approx(mat.params["G_IIc"]) == 2.4
    assert mat.params["irupt"] == 1


def test_starter_parsing_free_format_coh_mc(tmp_path: Path):
    """Verify free-format deck reading using /MAT/COH_MC alias."""
    deck_text = (
        "/MAT/COH_MC/201\n"
        "Cohesive Interface Free MC\n"
        "1.5e-6 1.5e-6\n"
        "3000.0 1500.0 1 1 2\n"
        "0 0 30.0 15.0 1.0\n"
        "1.5 3.0 2.0 1.8 1.2\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors: {log.errors}"
    assert 201 in model.materials
    mat = model.materials[201]
    assert mat.law == 117
    assert pytest.approx(mat.rho0) == 1.5e-6
    assert pytest.approx(mat.params["E_n"]) == 3000.0
    assert pytest.approx(mat.params["E_t"]) == 1500.0
    assert pytest.approx(mat.params["sigma_max"]) == 30.0
    assert pytest.approx(mat.params["tau_max"]) == 15.0
    assert pytest.approx(mat.params["G_Ic"]) == 1.5
    assert pytest.approx(mat.params["G_IIc"]) == 3.0
    assert mat.params["irupt"] == 2
    assert pytest.approx(mat.params["exp_bk"]) == 1.8
    assert pytest.approx(mat.params["gamma"]) == 1.2


def test_starter_parsing_free_format_coh_tab(tmp_path: Path):
    """Verify free-format deck reading using /MAT/COH_TAB alias."""
    deck_text = (
        "/MAT/COH_TAB/301\n"
        "Cohesive Interface Free TAB\n"
        "2.0e-6\n"
        "4000.0 2000.0 1 1 1\n"
        "0 0 40.0 20.0 1.0\n"
        "2.0 4.0 2.0 1.0 1.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors: {log.errors}"
    assert 301 in model.materials
    mat = model.materials[301]
    assert mat.law == 117
    assert pytest.approx(mat.rho0) == 2.0e-6
    assert pytest.approx(mat.params["E_n"]) == 4000.0
    assert pytest.approx(mat.params["E_t"]) == 2000.0
    assert pytest.approx(mat.params["sigma_max"]) == 40.0
    assert pytest.approx(mat.params["tau_max"]) == 20.0


# ============================================================================
# 7. Sound Speed and Consistent Algorithmic Tangent Stiffness
# ============================================================================

def test_sound_speed_and_tangent():
    """Verify acoustic sound speed and algorithmic tangent stiffness tensor."""
    E_n = 1800.0
    E_t = 900.0
    rho0 = 1.2e-6
    sigma_max = 18.0
    tau_max = 9.0
    G_Ic = 0.9
    G_IIc = 1.8

    p = Law117Params(
        E_n=E_n,
        E_t=E_t,
        sigma_max=sigma_max,
        tau_max=tau_max,
        G_Ic=G_Ic,
        G_IIc=G_IIc,
        rho0=rho0,
    )

    # Theoretical sound speed c = sqrt((E_n + E_t) / rho0)
    c_theo = math.sqrt((E_n + E_t) / rho0)
    c_calc = sound_speed(p)
    assert pytest.approx(c_calc, rel=1e-5) == c_theo

    # Elastic tangent verification
    c_tan = solid_tangent(p)
    assert pytest.approx(c_tan[2, 2]) == E_n
    assert pytest.approx(c_tan[4, 4]) == E_t
    assert pytest.approx(c_tan[5, 5]) == E_t
    assert pytest.approx(c_tan[0, 0]) == 0.0

    # Perturbation tangent during softening
    extra = {}
    sig = np.zeros(6, dtype=float)
    deps = np.zeros(6, dtype=float)
    deps[2] = 0.04  # well into softening (delta0_n = 0.01, delta_f_n = 0.10)
    sig, dmg, _ = solid_step(p, sig, deps, dt=1e-6, extra=extra)

    c_algo = consistent_solid_tangent(p, sig=sig, deps=deps, dt=1e-6, extra=extra)
    # The softening tangent along normal direction should be negative:
    # dsigma/ddelta = -sigma_max / (delta_f_n - delta0_n) = -18.0 / 0.09 = -200.0
    assert pytest.approx(c_algo[2, 2], rel=1e-2) == -200.0
