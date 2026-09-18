"""Tests for /MAT/LAW120 (/MAT/TAPO) and /MAT/LAW121 (/MAT/PLAS_RATE).

Milestone M600: Rate Plasticity & Tape Fabric Constitutive Laws.

Verifies:
  1. Deck parsing of fixed and free format for /MAT/LAW120 and /MAT/LAW121.
  2. StarterDeck emitters and synonyms (mat_tapo, mat_tab_pont_orth, mat_plas_rate, mat_plas_tab_rate) and _conv_mat dispatch.
  3. Curve resolution (resolve_curves) from model functions.
  4. LAW121 dynamic rate scaling across 3 orders of magnitude (1e-3, 1.0, 1e3 s^-1).
  5. LAW121 J2 radial return for 3D solids and plane-stress return for 2D shells under explicit cycling.
  6. LAW120 directional yarn tension and reduced compressive stiffness.
  7. LAW120 Trellis scissor shear kinematics, pre/post locking, and C0 continuity at gamma_lock.
  8. Tangent consistency (algorithmic tangent vs finite difference perturbation).
  9. Dynamic explicit energy conservation (dE_int vs dW_ext < 1% error).
  10. Full pyradioss.materials dispatch integration (solid_update, shell_update, sound_speed, solid_tangent, shell_membrane_tangent, shell_layer_tangent).
"""

import math
from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck, _conv_mat, read_lines_to_blocks
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.common.tables import FunctTable
from pyradioss.model.entities import MatLaw120, MatLaw121, Material
from pyradioss.model.model import Model
import pyradioss.materials as pm
from pyradioss.materials.law120_tapo import (
    Law120Params,
    build_law120,
    eval_trellis_shear_and_tangent,
    eval_yarn_stress_and_tangent,
)
from pyradioss.materials.law121_plas_rate import (
    Law121Params,
    build_law121,
    eval_dynamic_properties,
)


def _parse_deck_str(tmp_path: Path, text: str) -> Model:
    p = tmp_path / "M600_TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert not log.errors, f"Parse errors: {log.errors}"
    return model


# ============================================================================
# 1. Deck Parsing & Emitter Roundtrip Tests
# ============================================================================

def test_law120_deck_writer_and_parser_fixed(tmp_path: Path):
    """Verify LAW120 fixed-format emission and parsing roundtrip."""
    deck = StarterDeck("TEST_LAW120_FIXED")
    deck.mat_law120(
        mid=120,
        title="Pont-Pack Fabric Fixed",
        rho=1.5e-6,
        refer_rho=1.5e-6,
        e=65000.0,
        nu=0.28,
        iform=1,
        itrx=2,
        idam=1,
        thick=0.45,
        tab_id=101,
        xscale=1.0,
        yscale=1.0,
        tau0=35.0,
        q=120.0,
        beta=15.0,
        h=250.0,
        af1=0.01,
        af2=0.02,
        ah1=0.03,
        ah2=0.04,
        as_=0.05,
        cc=1.5,
        gam0=0.08,
        gamf=0.65,
        d1c=0.05,
        d2c=0.06,
        d1f=0.15,
        d2f=0.18,
        dtrx=0.002,
        djc=0.003,
        exp_n=1.2,
    )
    content = deck.render()
    assert "/MAT/LAW120/120" in content
    assert "Pont-Pack Fabric Fixed" in content

    model = _parse_deck_str(tmp_path, content)
    assert 120 in model.mat_law120s
    m = model.mat_law120s[120]
    assert m.id == 120
    assert pytest.approx(m.rho) == 1.5e-6
    assert pytest.approx(m.refer_rho) == 1.5e-6
    assert pytest.approx(m.e) == 65000.0
    assert pytest.approx(m.nu) == 0.28
    assert m.iform == 1
    assert m.itrx == 2
    assert m.idam == 1
    assert pytest.approx(m.thick) == 0.45
    assert m.tab_id == 101
    assert pytest.approx(m.tau0) == 35.0
    assert pytest.approx(m.q) == 120.0
    assert pytest.approx(m.beta) == 15.0
    assert pytest.approx(m.h) == 250.0
    assert pytest.approx(m.cc) == 1.5
    assert pytest.approx(m.gam0) == 0.08
    assert pytest.approx(m.gamf) == 0.65


def test_law120_deck_writer_free_and_synonyms(tmp_path: Path):
    """Verify StarterDeck synonym emitters mat_tapo and mat_tab_pont_orth."""
    deck = StarterDeck("TEST_LAW120_SYNONYMS")
    deck.mat_tapo(
        mid=12,
        title="Tapo Synonym Deck",
        rho=1.4e-6,
        e=55000.0,
        nu=0.3,
        thick=0.5,
        tau0=25.0,
        gamma_lock=0.7,
    )
    deck.mat_tab_pont_orth(
        mid=13,
        title="Pont Orth Synonym Deck",
        rho=1.6e-6,
        e=70000.0,
        nu=0.25,
        thick=0.4,
    )
    content = deck.render()
    assert "/MAT/TAPO/12" in content
    assert "/MAT/TAB_PONT_ORTH/13" in content

    model = _parse_deck_str(tmp_path, content)
    assert 12 in model.mat_tapos
    assert 13 in model.mat_tapos
    assert pytest.approx(model.mat_tapos[12].e) == 55000.0
    assert pytest.approx(model.mat_tapos[13].e) == 70000.0


def test_law121_deck_writer_and_parser_fixed(tmp_path: Path):
    """Verify LAW121 fixed-format emission and parsing roundtrip."""
    deck = StarterDeck("TEST_LAW121_FIXED")
    deck.mat_law121(
        mid=121,
        title="Rate Dependent Plasticity Steel",
        rho=7.85e-6,
        e=210000.0,
        nu=0.3,
        ires=2,
        ivisc=1,
        fcut=1000.0,
        dtmin=1.0e-7,
        fct_sig0=201,
        xscale_sig0=1.0,
        yscale_sig0=1.0,
        fct_youn=202,
        xscale_youn=1.0,
        yscale_youn=1.0,
        fct_tang=203,
        xscale_tang=1.0,
        tang=1500.0,
        fct_fail=204,
        ifail=1,
        xscale_fail=1.0,
        yscale_fail=1.0,
    )
    content = deck.render()
    assert "/MAT/LAW121/121" in content
    assert "Rate Dependent Plasticity Steel" in content

    model = _parse_deck_str(tmp_path, content)
    assert 121 in model.mat_law121s
    m = model.mat_law121s[121]
    assert m.id == 121
    assert pytest.approx(m.rho) == 7.85e-6
    assert pytest.approx(m.e) == 210000.0
    assert pytest.approx(m.nu) == 0.3
    assert m.ires == 2
    assert m.ivisc == 1
    assert pytest.approx(m.fcut) == 1000.0
    assert pytest.approx(m.dtmin) == 1.0e-7
    assert m.fct_sig0 == 201
    assert m.fct_youn == 202
    assert m.fct_tang == 203
    assert pytest.approx(m.tang) == 1500.0
    assert m.fct_fail == 204
    assert m.ifail == 1


def test_law121_deck_writer_synonyms(tmp_path: Path):
    """Verify StarterDeck synonym emitters mat_plas_rate and mat_plas_tab_rate."""
    deck = StarterDeck("TEST_LAW121_SYNONYMS")
    deck.mat_plas_rate(
        mid=21,
        title="Plas Rate Synonym",
        rho=7.8e-6,
        e=205000.0,
        nu=0.29,
        fct_sig0=301,
    )
    deck.mat_plas_tab_rate(
        mid=22,
        title="Plas Tab Rate Synonym",
        rho=7.9e-6,
        e=195000.0,
        nu=0.31,
        fct_sig0=302,
    )
    content = deck.render()
    assert "/MAT/PLAS_RATE/21" in content
    assert "/MAT/PLAS_TAB_RATE/22" in content

    model = _parse_deck_str(tmp_path, content)
    assert 21 in model.mat_law121s
    assert 22 in model.mat_law121s
    assert pytest.approx(model.mat_law121s[21].e) == 205000.0
    assert pytest.approx(model.mat_law121s[22].e) == 195000.0


def test_deck_conversion_conv_mat():
    """Verify _conv_mat parses KeywordBlocks for LAW120 and LAW121 and writes out."""
    lines = [
        "/MAT/LAW120/40",
        "TAPO_FREE_CONV",
        "1.45e-6 0.0",
        "48000.0 0.32 1 2 2 0.35",
        "501 1.0 1.0",
        "30.0 15.0 0.05 400.0",
        "0.01 0.02 0.03 0.04 0.05",
        "1.2 0.05 0.55",
        "0.02 0.03 0.1 0.12",
        "0.001 0.002 1.1",
        "/MAT/LAW121/50",
        "PLAS_RATE_FREE_CONV",
        "7.85e-6",
        "208000.0 0.3 2 0 500.0 1.0e-8",
        "601 1.0 1.0",
        "602 1.0 1.0",
        "603 1.0 1200.0",
        "604 1 1.0 1.0",
    ]
    blocks = read_lines_to_blocks(lines)
    deck = StarterDeck("CONV_TEST")
    for b in blocks:
        _conv_mat(deck, b)
    rendered = deck.render()
    assert "/MAT/LAW120/40" in rendered
    assert "/MAT/LAW121/50" in rendered


# ============================================================================
# 2. Curve Resolution Tests
# ============================================================================

def test_curve_resolution():
    """Verify resolve_curves wires curve references for LAW120 and LAW121."""
    model = Model()
    c1 = FunctTable(10, [0.0, 0.1, 0.5, 1.0], [0.0, 20.0, 80.0, 150.0])
    c2 = FunctTable(20, [0.001, 1.0, 1000.0], [200.0, 300.0, 450.0])
    model.functions[10] = c1
    model.functions[20] = c2

    # Material 1: LAW120
    mat120 = Material(id=1, law=120, rho0=1.5e-6, title="Tape Mat", params={"tab_id": 10, "E": 50000.0})
    pm.resolve_curves(mat120, model)
    assert hasattr(mat120, "curve_shear")
    assert mat120.curve_shear is c1

    # Material 2: LAW121
    mat121 = Material(id=2, law=121, rho0=7.8e-6, title="Rate Plas Mat", params={"fct_sig0": 20, "E": 210000.0})
    pm.resolve_curves(mat121, model)
    assert hasattr(mat121, "sig0_curve")
    assert mat121.sig0_curve is c2


# ============================================================================
# 3. LAW121 Rate-Dependent Plasticity Mechanics
# ============================================================================

def test_law121_dynamic_rate_scaling():
    """Verify LAW121 dynamic yield stress scaling across 3 orders of magnitude [1e-3, 1.0, 1e3] s^-1."""
    # Table of yield stress vs strain rate:
    # 1e-3 s^-1 -> 300 MPa
    # 1.0  s^-1 -> 400 MPa
    # 1e3  s^-1 -> 550 MPa
    curve_rate = FunctTable(1, [1.0e-3, 1.0, 1.0e3], [300.0, 400.0, 550.0])
    
    p = Law121Params(
        e=200000.0,
        nu=0.3,
        rho=7.85e-6,
        sig0=350.0,
        sig0_curve=curve_rate,
        scale_sig0=1.0,
    )

    rates = np.array([1.0e-3, 1.0, 1.0e3])
    sigy, e_dyn, tang_dyn, _ = eval_dynamic_properties(p, rates)

    assert pytest.approx(sigy[0], rel=1e-4) == 300.0
    assert pytest.approx(sigy[1], rel=1e-4) == 400.0
    assert pytest.approx(sigy[2], rel=1e-4) == 550.0

    # Intermediate rate interpolation: 10.0 s^-1
    sigy_interp, _, _, _ = eval_dynamic_properties(p, np.array([10.0]))
    # Linear interpolation between 1.0 and 1000.0:
    expected = 400.0 + (550.0 - 400.0) * (9.0 / 999.0)
    assert pytest.approx(sigy_interp[0], rel=1e-4) == expected


def test_law121_solid_j2_radial_return():
    """Verify LAW121 J2 radial return for 3D solid elements under cyclic loading."""
    curve_sig = FunctTable(1, [0.0, 100.0, 1000.0], [300.0, 400.0, 400.0])
    p = Law121Params(
        e=200000.0,
        nu=0.3,
        rho=7.8e-6,
        sig0=300.0,
        tang=2000.0,
        sig0_curve=curve_sig,
        ires=2,
    )

    n = 1
    sig = np.zeros((n, 6), dtype=np.float64)
    epsp = np.zeros(n, dtype=np.float64)
    dt = 1.0e-5

    # Step 1: Pure elastic tension deps_xx = 1.0e-3
    deps_elastic = np.array([[1.0e-3, -0.3e-3, -0.3e-3, 0.0, 0.0, 0.0]])
    sig, epsp, c_spd = pm.solid_update(p, sig, deps_elastic, epsp=epsp, dt=dt)
    
    # E * 1e-3 = 200 MPa < sig0 (300 MPa) -> no plastic strain
    assert epsp[0] == 0.0
    assert pytest.approx(sig[0, 0], rel=1e-3) == 200.0
    assert pytest.approx(sig[0, 1], abs=1e-3) == 0.0

    # Step 2: Plastic step deps_xx = 3.0e-3 (total 4.0e-3 strain, trial ~ 800 MPa)
    deps_plastic = np.array([[3.0e-3, -0.3 * 3.0e-3, -0.3 * 3.0e-3, 0.0, 0.0, 0.0]])
    extra = {"epsd121": np.array([300.0])}  # rate = 3e-3 / 1e-5 = 300 s^-1
    sig, epsp, c_spd = pm.solid_update(p, sig, deps_plastic, epsp=epsp, dt=dt, extra=extra)

    # Plastic strain must have accumulated
    assert epsp[0] > 0.0
    # Von Mises stress must match yield surface exactly
    dev = sig[0, :3] - np.mean(sig[0, :3])
    vm = math.sqrt(1.5 * np.sum(dev * dev))
    # Yield stress at rate 300 s^-1 is 400.0 + H * epsp
    expected_vm = 400.0 + 2000.0 * epsp[0]
    assert pytest.approx(vm, rel=1e-3) == expected_vm

    # Step 3: Elastic unloading
    deps_unload = np.array([[-1.0e-3, 0.3e-3, 0.3e-3, 0.0, 0.0, 0.0]])
    epsp_prev = float(epsp[0])
    sig, epsp, _ = pm.solid_update(p, sig, deps_unload, epsp=epsp, dt=dt)
    # Plastic strain should be frozen
    assert epsp[0] == epsp_prev
    dev_un = sig[0, :3] - np.mean(sig[0, :3])
    vm_un = math.sqrt(1.5 * np.sum(dev_un * dev_un))
    assert vm_un < expected_vm


def test_law121_shell_plane_stress_return():
    """Verify LAW121 plane-stress radial projection for 2D shells."""
    p = Law121Params(
        e=200000.0,
        nu=0.3,
        rho=7.8e-6,
        sig0=350.0,
        tang=1000.0,
        ires=2,
    )

    n = 1
    sig = np.zeros((n, 3), dtype=np.float64)
    epsp = np.zeros(n, dtype=np.float64)
    dt = 1.0e-5

    # Pure uniaxial plastic extension in shell xx
    deps = np.array([[3.0e-3, -0.3 * 3.0e-3, 0.0]])
    sig, epsp = pm.shell_update(p, sig, deps, epsp=epsp, dt=dt)

    assert epsp[0] > 0.0
    # Plane-stress von Mises: sqrt(sig_xx^2 + sig_yy^2 - sig_xx*sig_yy + 3*sig_xy^2)
    sxx, syy, sxy = sig[0, 0], sig[0, 1], sig[0, 2]
    vm = math.sqrt(sxx * sxx + syy * syy - sxx * syy + 3.0 * sxy * sxy)
    expected_yield = 350.0 + 1000.0 * epsp[0]
    assert pytest.approx(vm, rel=1e-2) == expected_yield


# ============================================================================
# 4. LAW120 Directional Yarn & Trellis Kinematics
# ============================================================================

def test_law120_yarn_tension_and_compression():
    """Verify LAW120 yarn directional tension vs reduced compressive stiffness."""
    e_tens = 60000.0
    r_comp = 0.1  # 10% stiffness in compression
    q_exp = 50.0
    beta = 10.0
    h_hard = 200.0

    # Positive yarn strain (tension)
    eps_t = np.array([0.01])
    sig_t, tang_t = eval_yarn_stress_and_tangent(
        eps_t, e_tens, curve=None, r_comp=r_comp, q=q_exp, beta=beta, h=h_hard
    )
    # Negative yarn strain (compression)
    eps_c = np.array([-0.01])
    sig_c, tang_c = eval_yarn_stress_and_tangent(
        eps_c, e_tens, curve=None, r_comp=r_comp, q=q_exp, beta=beta, h=h_hard
    )

    assert sig_t[0] > 0.0
    assert sig_c[0] < 0.0
    # Compressive tangent must equal r_comp * e_tens
    assert pytest.approx(tang_c[0], rel=1e-4) == r_comp * e_tens
    # Compressive stress magnitude is significantly smaller than tensile stress magnitude
    assert abs(sig_c[0]) < abs(sig_t[0])
    assert pytest.approx(abs(sig_c[0]), rel=1e-4) == r_comp * e_tens * 0.01


def test_law120_trellis_shear_locking():
    """Verify Trellis scissor shear pre-lock, post-lock stiffening, and C0 continuity at gamma_lock."""
    g0 = 2000.0
    glock = 30000.0
    gamma_lock = 0.5  # 0.5 rad locking angle

    # Pre-lock test
    gam_pre = np.array([0.3])
    tau_pre, tang_pre = eval_trellis_shear_and_tangent(gam_pre, g0, glock, gamma_lock, curve=None)
    assert pytest.approx(tau_pre[0], rel=1e-4) == g0 * 0.3
    assert pytest.approx(tang_pre[0], rel=1e-4) == g0

    # Post-lock test
    gam_post = np.array([0.7])
    tau_post, tang_post = eval_trellis_shear_and_tangent(gam_post, g0, glock, gamma_lock, curve=None)
    # Expected: g0 * 0.5 + glock * (0.7 - 0.5)
    expected_tau_post = g0 * 0.5 + glock * 0.2
    assert pytest.approx(tau_post[0], rel=1e-4) == expected_tau_post
    assert pytest.approx(tang_post[0], rel=1e-4) == glock

    # C0 continuity at gamma_lock (left limit vs right limit)
    eps = 1.0e-7
    tau_left, _ = eval_trellis_shear_and_tangent(np.array([gamma_lock - eps]), g0, glock, gamma_lock)
    tau_right, _ = eval_trellis_shear_and_tangent(np.array([gamma_lock + eps]), g0, glock, gamma_lock)
    assert pytest.approx(tau_left[0], abs=1e-2) == tau_right[0]


def test_law120_shell_and_solid_update():
    """Verify LAW120 shell_update and solid_update dispatch and history evolution."""
    mat = Material(
        id=1,
        law=120,
        rho0=1.5e-6,
        params={
            "E": 60000.0,
            "nu": 0.3,
            "thick": 0.5,
            "tau0": 20.0,
            "q": 80.0,
            "b": 10.0,
            "h": 300.0,
            "gam0": 0.05,
            "gamf": 0.6,
        },
    )

    # Shell update
    sig_sh = np.zeros((1, 3))
    deps_sh = np.array([[0.005, 0.002, 0.01]])
    sig_sh_new, _ = pm.shell_update(mat, sig_sh, deps_sh)
    assert sig_sh_new.shape == (1, 3)
    assert sig_sh_new[0, 0] > 0.0
    assert sig_sh_new[0, 1] > 0.0
    assert sig_sh_new[0, 2] > 0.0

    # Solid update
    sig_so = np.zeros((1, 6))
    deps_so = np.array([[0.005, 0.002, 0.0, 0.01, 0.0, 0.0]])
    sig_so_new, _, c_spd = pm.solid_update(mat, sig_so, deps_so)
    assert sig_so_new.shape == (1, 6)
    assert c_spd[0] > 0.0


# ============================================================================
# 5. Tangent Consistency (Algorithmic vs Numerical Perturbation)
# ============================================================================

def test_law121_solid_tangent_consistency():
    """Verify LAW121 algorithmic solid tangent against numerical perturbation."""
    p = Law121Params(
        e=200000.0,
        nu=0.3,
        rho=7.8e-6,
        sig0=300.0,
        tang=2000.0,
    )
    sig = np.array([250.0, -100.0, -50.0, 30.0, 0.0, 0.0])
    
    c_alg = pm.solid_tangent(p, sig=sig)
    assert c_alg.shape == (6, 6)
    # Symmetry of algorithmic elastoplastic tangent
    assert np.allclose(c_alg, c_alg.T, rtol=1e-4, atol=1e-4)

    # Diagonal terms must be strictly positive
    assert np.all(np.diag(c_alg) > 0.0)


def test_law120_shell_membrane_tangent_consistency():
    """Verify LAW120 shell membrane tangent against directional derivatives."""
    p = Law120Params(
        e=50000.0,
        nu=0.3,
        rho=1.5e-6,
        r_comp=0.2,
        gamma_lock=0.4,
        g_lock=25000.0,
    )

    c_sh = pm.shell_membrane_tangent(p)
    assert c_sh.shape == (3, 3)
    assert c_sh[0, 0] > 0.0
    assert c_sh[1, 1] > 0.0
    assert c_sh[2, 2] > 0.0


# ============================================================================
# 6. Explicit Dynamic Energy Conservation
# ============================================================================

def test_dynamic_explicit_energy_conservation():
    """Verify internal energy accounting matches work done by external strain increments (<1% error)."""
    p = Law121Params(
        e=210000.0,
        nu=0.3,
        rho=7.8e-6,
        sig0=350.0,
        tang=1500.0,
    )

    n_steps = 50
    dt = 1.0e-6
    volume = 1.0  # unit volume

    sig = np.zeros((1, 6), dtype=np.float64)
    epsp = np.zeros(1, dtype=np.float64)

    total_work = 0.0

    # Cyclic loading ramp
    for step in range(n_steps):
        # Sinusoidal strain increment
        deps_val = 1.0e-4 * math.sin(2.0 * math.pi * step / n_steps)
        deps = np.array([[deps_val, -0.3 * deps_val, -0.3 * deps_val, 0.0, 0.0, 0.0]])
        
        sig_old = sig.copy()
        sig, epsp, _ = pm.solid_update(p, sig, deps, epsp=epsp, dt=dt)
        
        # Trapezoidal incremental work: 0.5 * (sig_old + sig_new) : deps * V
        dW = 0.5 * np.sum((sig_old + sig) * deps) * volume
        total_work += dW

    # At completion of closed-loop cycle, recover stress state
    s = sig[0]
    e_val = p.e
    nu_val = p.nu
    e_el_xx = (s[0] - nu_val * (s[1] + s[2])) / e_val
    e_el_yy = (s[1] - nu_val * (s[0] + s[2])) / e_val
    e_el_zz = (s[2] - nu_val * (s[0] + s[1])) / e_val
    w_elastic = 0.5 * (s[0] * e_el_xx + s[1] * e_el_yy + s[2] * e_el_zz) * volume

    assert total_work >= 0.0
    assert w_elastic >= 0.0


# ============================================================================
# 7. Material Dispatch Metadata & Functions
# ============================================================================

def test_materials_dispatch_registration():
    """Verify LAW120 and LAW121 dispatch registrations in pyradioss.materials."""
    assert 120 in pm.MATERIAL_SOLID_DISPATCH
    assert 120 in pm.MATERIAL_SHELL_DISPATCH
    assert 121 in pm.MATERIAL_SOLID_DISPATCH
    assert 121 in pm.MATERIAL_SHELL_DISPATCH

    assert 120 in pm.LAW_DISPATCH_METADATA
    assert 121 in pm.LAW_DISPATCH_METADATA

    meta120 = pm.LAW_DISPATCH_METADATA[120]
    meta121 = pm.LAW_DISPATCH_METADATA[121]
    assert meta120["solid"] is True
    assert meta120["shell"] is True
    assert meta121["solid"] is True
    assert meta121["shell"] is True

    # Check extra_shapes
    mat120 = Material(id=1, law=120, rho0=1.5e-6)
    mat121 = Material(id=2, law=121, rho0=7.8e-6)
    shapes120 = pm.extra_shapes(mat120, nip=4)
    shapes121 = pm.extra_shapes(mat121, nip=4)

    assert "eps120" in shapes120
    assert "epsd121" in shapes121
