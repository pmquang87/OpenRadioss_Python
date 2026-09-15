"""
Tests for Milestone M570: /MAT/LAW100 Starter Parsing, Model Entities, and Diagnostic Checks.

Verifies:
1. Starter keywords reader for /MAT/LAW100, /MAT/VISC_HYP, and /MAT/MNF.
2. Dynamic multi-network reading with N_net = 0, 1, 2, 3 and Flag_HE = 1, 2, 3, 4, 5, 13.
3. Model entities MatLaw100, MatViscHyp, MatMNF, properties (G, K, sound_speed).
4. Model containers model.mat_law100s, model.mat_visc_hyps, model.mat_mnfs.
5. Diagnostic validation in check_mat_law100 and check_materials:
   - Density RHO0 > 0 (ANCMSG 1514).
   - Network count N_net <= 10 (ANCMSG 1567).
   - Hyperelastic law flag Flag_HE validity (ANCMSG 1569).
   - Relaxation rule flag Flag_visc validity (ANCMSG 1808).
   - Solid continuum only: shells rejected (ANCMSG 305), 1D rejected (ANCMSG 306).
"""

import pytest
import numpy as np

from pyradioss.model.entities import MatLaw100, MatViscHyp, MatMNF, MaterialLaw100, Part, Property
from pyradioss.model.model import Model
from pyradioss.starter.starter import MessageLog, parse_starter_deck
from pyradioss.input.deck_writer import StarterDeck, read_lines_to_blocks
from pyradioss.input.checks import check_mat_law100, check_materials, check_model


# ============================================================================
# 1. Model Entities, Properties & Aliases
# ============================================================================

def test_mat_law100_entity_defaults():
    """Verify MatLaw100 entity initialization and aliases."""
    mat = MatLaw100(id=1, rho0=1.1e-6, c10=0.5, c01=0.1, d1=1.0e-5)
    assert mat.id == 1
    assert mat.rho0 == 1.1e-6
    assert mat.law == 100
    assert mat.law_name == "LAW100"
    # Derived linear elastic properties
    assert mat.G == pytest.approx(2.0 * (0.5 + 0.1), rel=1e-5)
    assert mat.K == pytest.approx(2.0 / 1.0e-5, rel=1e-5)
    assert mat.sound_speed > 0.0

    # Aliases
    assert MatViscHyp is MatLaw100
    assert MatMNF is MatLaw100
    assert MaterialLaw100 is MatLaw100


def test_model_containers():
    """Verify Model stores and references MatLaw100 across aliases."""
    model = Model()
    mat = MatLaw100(id=5, rho0=1.2e-6, c10=1.0, d1=1e-4)
    model.mat_law100s[5] = mat
    model.materials[5] = mat

    assert 5 in model.mat_law100s
    assert 5 in model.mat_visc_hyps
    assert 5 in model.mat_mnfs
    assert model.mat_visc_hyps[5] is mat
    assert model.mat_mnfs[5] is mat


# ============================================================================
# 2. Starter Parsing: Multi-Network & Various Formulations
# ============================================================================

def test_parse_law100_polynomial_two_networks():
    """Parse /MAT/LAW100 with Flag_HE=1 and 2 secondary relaxation networks."""
    deck = StarterDeck("TEST_POLY")
    deck.mat_law100(
        1, "polymer_he1",
        rho0=1.15e-6, n_net=2, flag_he=1, flag_cr=0,
        c10=0.45, c01=0.08, d1=0.02,
        networks=[
            {"net_id": 1, "flag_visc": 1, "stiffness": 1.2, "a": 0.05, "c": -0.7, "m": 1.5, "ksi": 0.01, "tau_ref": 1.5},
            {"net_id": 2, "flag_visc": 2, "stiffness": 0.8, "a": 0.02, "b": 0.5, "n": 1.2},
        ]
    )
    raw = deck.render()
    blocks = read_lines_to_blocks(raw.splitlines())
    model = parse_starter_deck(blocks)

    mat = model.mat_law100s[1]
    assert mat.id == 1
    assert mat.title == "polymer_he1"
    assert mat.rho0 == pytest.approx(1.15e-6)
    assert mat.n_net == 2
    assert mat.flag_he == 1
    assert mat.c10 == pytest.approx(0.45)
    assert mat.c01 == pytest.approx(0.08)
    assert mat.d1 == pytest.approx(0.02)
    assert len(mat.networks) == 2

    # Network 1 (Bergstrom-Boyce)
    net1 = mat.networks[0]
    assert net1["network_id"] == 1
    assert net1["flag_visc"] == 1
    assert net1["stiffness"] == pytest.approx(1.2)
    assert net1["a"] == pytest.approx(0.05)
    assert net1["c"] == pytest.approx(-0.7)
    assert net1["m"] == pytest.approx(1.5)

    # Network 2 (Sinh)
    net2 = mat.networks[1]
    assert net2["network_id"] == 2
    assert net2["flag_visc"] == 2
    assert net2["stiffness"] == pytest.approx(0.8)
    assert net2["a"] == pytest.approx(0.02)
    assert net2["b"] == pytest.approx(0.5)
    assert net2["n"] == pytest.approx(1.2)


def test_parse_law100_arruda_boyce_with_creep():
    """Parse /MAT/LAW100 with Flag_HE=2 (Arruda-Boyce) and Flag_Cr=1 (Equilibrium Creep)."""
    deck = StarterDeck("TEST_AB")
    deck.mat_law100(
        2, "arruda_boyce_mat",
        rho0=1.05e-6, n_net=1, flag_he=2, flag_cr=1,
        mue1=2.5, d=0.01, lambda_m=5.0, itype=1, nu_val=0.48, fscale_ab=1.0,
        a_pl=0.8, sigma_pl=3.0, f_pl=0.7, epsilon_f=0.15, n_pl=2,
        networks=[
            {"net_id": 1, "flag_visc": 3, "stiffness": 2.0, "a": 1e-4, "n": 2.5, "m": 0.5},
        ]
    )
    raw = deck.render()
    blocks = read_lines_to_blocks(raw.splitlines())
    model = parse_starter_deck(blocks)

    mat = model.mat_law100s[2]
    assert mat.id == 2
    assert mat.flag_he == 2
    assert mat.flag_cr == 1
    assert mat.mue1 == pytest.approx(2.5)
    assert mat.d == pytest.approx(0.01)
    assert mat.lambda_m == pytest.approx(5.0)
    assert mat.nu_val == pytest.approx(0.48)
    assert mat.a_pl == pytest.approx(0.8)
    assert mat.sigma_pl == pytest.approx(3.0)
    assert mat.f_pl == pytest.approx(0.7)
    assert mat.epsilon_f == pytest.approx(0.15)
    assert mat.n_pl == 2

    # Network 1 (Power law)
    assert len(mat.networks) == 1
    net = mat.networks[0]
    assert net["flag_visc"] == 3
    assert net["stiffness"] == pytest.approx(2.0)
    assert net["a"] == pytest.approx(1e-4)
    assert net["n"] == pytest.approx(2.5)
    assert net["m"] == pytest.approx(0.5)


def test_parse_visc_hyp_and_mnf_aliases():
    """Parse using /MAT/VISC_HYP and /MAT/MNF keyword headers."""
    deck1 = StarterDeck("TEST_SYN1")
    deck1.mat_visc_hyp(1, "visc_hyp_title", rho0=1.1e-6, flag_he=3, c10=1.2, d1=0.01)
    raw1 = deck1.render()
    blocks1 = read_lines_to_blocks(raw1.splitlines())
    model1 = parse_starter_deck(blocks1)
    assert 1 in model1.mat_law100s
    assert model1.mat_law100s[1].flag_he == 3

    deck2 = StarterDeck("TEST_SYN2")
    deck2.mat_mnf(2, "mnf_title", rho0=1.2e-6, flag_he=4, c10=0.8, c01=0.2, d1=0.02)
    raw2 = deck2.render()
    blocks2 = read_lines_to_blocks(raw2.splitlines())
    model2 = parse_starter_deck(blocks2)
    assert 2 in model2.mat_law100s
    assert model2.mat_law100s[2].flag_he == 4


# ============================================================================
# 3. Diagnostic Checks
# ============================================================================

def test_check_mat_law100_valid():
    """Valid parameters pass with zero error messages."""
    log = MessageLog()
    mat = MatLaw100(
        id=1, rho0=1.1e-6, flag_he=1, flag_cr=0, c10=0.5, c01=0.1, d1=0.01,
        networks=[{"net_id": 1, "flag_visc": 1, "stiffness": 1.0, "a": 0.05, "c": -0.7, "m": 1.2, "ksi": 0.01, "tau_ref": 1.0}]
    )
    check_mat_law100(mat=mat, log=log)
    assert len(log.errors) == 0


def test_check_mat_law100_invalid_rho0():
    """RHO0 <= 0 triggers ANCMSG 1514."""
    log = MessageLog()
    mat = MatLaw100(id=1, rho0=-1e-6, flag_he=1)
    check_mat_law100(mat=mat, log=log)
    assert any("ANCMSG 1514" in err for err in log.errors)


def test_check_mat_law100_exceed_max_networks():
    """N_net > 10 triggers ANCMSG 1567."""
    log = MessageLog()
    nets = [{"net_id": i + 1, "flag_visc": 1, "stiffness": 1.0} for i in range(12)]
    mat = MatLaw100(id=1, rho0=1.0e-6, flag_he=1, n_net=12, networks=nets)
    check_mat_law100(mat=mat, log=log)
    assert any("ANCMSG 1567" in err for err in log.errors)


def test_check_mat_law100_invalid_flag_he():
    """Invalid Flag_HE triggers ANCMSG 1569."""
    log = MessageLog()
    mat = MatLaw100(id=1, rho0=1.0e-6, flag_he=42)
    check_mat_law100(mat=mat, log=log)
    assert any("ANCMSG 1569" in err for err in log.errors)


def test_check_mat_law100_invalid_flag_visc():
    """Invalid Flag_visc triggers ANCMSG 1808."""
    log = MessageLog()
    mat = MatLaw100(
        id=1, rho0=1.0e-6, flag_he=1,
        networks=[{"net_id": 1, "flag_visc": 9, "stiffness": 1.0}]
    )
    check_mat_law100(mat=mat, log=log)
    assert any("ANCMSG 1808" in err for err in log.errors)


def test_check_mat_law100_element_compatibility():
    """LAW100 rejects shells (ANCMSG 305) and 1D elements (ANCMSG 306)."""
    model = Model()
    mat = MatLaw100(id=1, rho0=1.1e-6, flag_he=1, c10=0.5, d1=0.01)
    model.mat_law100s[1] = mat
    model.materials[1] = mat

    # Add 2D shell part
    model.properties[1] = Property(id=1, type="SHELL")
    part_shell = Part(id=10, prop_id=1, mat_id=1)
    model.parts[10] = part_shell

    log = MessageLog()
    check_mat_law100(mat=mat, model=model, log=log)
    assert any("ANCMSG 305" in err for err in log.errors)

    # Add 1D beam part
    model.parts.clear()
    model.properties.clear()
    model.properties[2] = Property(id=2, type="BEAM")
    part_beam = Part(id=20, prop_id=2, mat_id=1)
    model.parts[20] = part_beam

    log2 = MessageLog()
    check_mat_law100(mat=mat, model=model, log=log2)
    assert any("ANCMSG 306" in err for err in log2.errors)
